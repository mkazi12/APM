#!/usr/bin/env python3
"""Fit a candidate Hey Gemma classifier using local, session-separated recordings.

Keeps the pretrained audio backbone, mixes in the original synthetic TRAIN split,
and chooses a threshold on real validation sessions. Never trains on test clips or
overwrites the installed detector. Run record_wakeword.py first.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import onnxruntime as ort
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

try:
    from . import train_wakeword as bootstrap
    from .speech_alignment import speech_bounds
    from .wake_recordings import (WARMUP_FRAMES, choose_threshold, load_recordings, make_model,
                                  read_audio, score_recordings, summarize)
except ImportError:
    import train_wakeword as bootstrap
    from speech_alignment import speech_bounds
    from wake_recordings import (WARMUP_FRAMES, choose_threshold, load_recordings, make_model,
                                 read_audio, score_recordings, summarize)

ROOT = Path(__file__).resolve().parents[1]
RATE, BLOCK, SEED = 16000, 1280, 427


def streamed_features(model, audio: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    model.reset()
    silence = np.zeros(BLOCK, dtype=np.int16)
    for _ in range(WARMUP_FRAMES):
        model.predict(silence)
    features, scores = [], []
    for start in range(0, len(audio), BLOCK):
        frame = audio[start:start + BLOCK]
        if len(frame) < BLOCK:
            frame = np.pad(frame, (0, BLOCK - len(frame)))
        prediction = model.predict(frame)
        scores.append(float(prediction[name]))
        feature = model.preprocessor.get_features(16)[0].copy()
        if feature.shape != (16, 96):
            raise ValueError(f"Expected (16, 96) backbone features, got {feature.shape}")
        features.append(feature)
    return np.asarray(features, dtype=np.float32), np.asarray(scores)


def positive_indices(frame_count: int, speech_end: int) -> np.ndarray:
    # Align to the end of the complete phrase, with room for acoustic uncertainty.
    frame_ends = (np.arange(frame_count) + 1) * BLOCK
    return np.flatnonzero((frame_ends >= speech_end)
                          & (frame_ends <= speech_end + 0.32 * RATE))


def augment(audio, background, rng):
    """Mild gain and actual training-room noise; preserve phrase timing."""
    wave = audio.astype(np.float32) / 32768
    wave *= rng.uniform(0.65, 1.15)
    if background is not None:
        noise = np.resize(background.astype(np.float32) / 32768, len(wave))
        noise = np.roll(noise, int(rng.integers(len(noise))))
        noise_rms = np.sqrt(np.mean(noise ** 2))
        wave_rms = np.sqrt(np.mean(wave ** 2))
        if noise_rms > 1e-6:
            wave += noise * (wave_rms / noise_rms) * 10 ** (-rng.uniform(15, 25) / 20)
    return (np.clip(wave, -0.999, 0.999) * 32767).astype(np.int16)


def personal_features(model_path, records, augmentations=2):
    if not records or any(r["split"] != "train" for r in records):
        raise ValueError("Feature fitting accepts training sessions only")
    rng = np.random.default_rng(SEED)
    model = make_model(model_path)
    backgrounds = [read_audio(r) for r in records if r["label"] == "background"]
    xs, ys, weights, alignment = [], [], [], []
    for record in records:
        audio = read_audio(record)
        positive = record["label"] == "positive"
        if positive:
            try:
                speech_start, speech_end = speech_bounds(audio)
            except ValueError as exc:
                raise ValueError(f"{record['path']}: {exc}") from exc
            alignment.append({"path": str(record["path"]),
                              "speech_start_seconds": speech_start / RATE,
                              "speech_end_seconds": speech_end / RATE})
        variants = [audio]
        if positive:
            for _ in range(augmentations):
                background = backgrounds[int(rng.integers(len(backgrounds)))] if backgrounds else None
                variants.append(augment(audio, background, rng))
        for wave in variants:
            features, scores = streamed_features(model, wave, model_path.stem)
            if positive:
                selected = positive_indices(len(features), speech_end)
            else:
                # Include both representative frames and the baseline's hardest errors.
                uniform = np.linspace(0, len(features) - 1, min(64, len(features)), dtype=int)
                hardest = np.argsort(scores)[-min(32, len(features)):]
                selected = np.unique(np.concatenate([uniform, hardest]))
            if not len(selected):
                raise ValueError(f"No usable feature windows in {record['path']}")
            xs.append(features[selected])
            ys.extend([int(positive)] * len(selected))
            # Each original recording has equal mass within its label, regardless
            # of its length or how many augmented windows it contributes.
            weights.extend([1 / (len(selected) * len(variants))] * len(selected))
        print(f"Extracted {record['label']}: {record['path'].name}", flush=True)
    return np.concatenate(xs), np.asarray(ys), np.asarray(weights), alignment


def load_synthetic(path):
    with np.load(path, allow_pickle=False) as data:
        x, y = data["x"].copy(), data["y"].copy()
    if x.ndim != 3 or x.shape[1:] != (16, 96) or y.shape != (len(x),):
        raise ValueError("Synthetic training cache has an incompatible feature shape")
    if not np.isfinite(x).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("Synthetic training cache must have finite features and both binary labels")
    return x.astype(np.float32), y.astype(np.int64)


def training_weights(synthetic_y, personal_y, personal_weights, personal_share=0.65):
    """Balance positive/negative classes and synthetic/personal sources explicitly."""
    if not 0 < personal_share < 1:
        raise ValueError("Personal weight share must be between 0 and 1")
    sw = np.zeros(len(synthetic_y))
    pw = np.zeros(len(personal_y))
    for label in (0, 1):
        s, p = synthetic_y == label, personal_y == label
        if not s.any() or not p.any() or personal_weights[p].sum() <= 0:
            raise ValueError("Both training sources must contain positive and negative examples")
        sw[s] = (1 - personal_share) / s.sum()
        pw[p] = personal_share * personal_weights[p] / personal_weights[p].sum()
    result = np.concatenate([sw, pw])
    return result * len(result) / result.sum()


def recording_holdout(records):
    """Reserve original takes for a provisional calibration, without copying files.

    Entire speech recordings stay together before any augmentation. Background
    goes only to validation, so it cannot leak into training noise augmentation.
    This is explicitly weaker evidence than a separate recording session.
    """
    if any(r["split"] == "validation" for r in records):
        raise ValueError("Separate validation recordings exist; omit --holdout-from-training")
    rng = np.random.default_rng(SEED)
    train, validation = [], []
    for label in ("positive", "negative"):
        group = [r for r in records if r["split"] == "train" and r["label"] == label]
        if len(group) < 5:
            raise ValueError(f"Need at least 5 {label} takes for a recording holdout")
        selected = set(rng.permutation(len(group))[:max(1, round(len(group) * 0.2))])
        for i, record in enumerate(group):
            if i in selected:
                validation.append(dict(record, split="validation", source_split="train"))
            else:
                train.append(record)
    validation += [dict(r, split="validation", source_split="train") for r in records
                   if r["split"] == "train" and r["label"] == "background"]
    if not any(r["label"] == "background" for r in validation):
        raise ValueError("Record background audio before calibrating a candidate")
    return train, validation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "work/wake-personal")
    parser.add_argument("--base-model", type=Path, default=ROOT / "models/hey_gemma.onnx")
    parser.add_argument("--synthetic-features", type=Path,
                        default=ROOT / "work/wake-training/train-features-v1.npz",
                        help="Original TRAIN split only; never provide validation or test features")
    parser.add_argument("--output-dir", type=Path, help="New directory; existing directories are refused")
    parser.add_argument("--augmentations", type=int, default=2)
    parser.add_argument("--head", choices=("linear", "mlp"), default="linear",
                        help="Classifier over the frozen backbone; MLP learns nonlinear phrase distinctions")
    parser.add_argument("--holdout-from-training", action="store_true",
                        help="Provisional: reserve 20%% of original speech takes and all background for calibration")
    args = parser.parse_args(argv)
    if not 0 <= args.augmentations <= 10:
        parser.error("--augmentations must be between 0 and 10")
    try:
        records = load_recordings(args.dataset)
        train = [r for r in records if r["split"] == "train"]
        validation = [r for r in records if r["split"] == "validation"]
        if args.holdout_from_training:
            train, validation = recording_holdout(records)
            print("Provisional recording holdout: no independent session test yet.", flush=True)
        for name, rows in (("train", train), ("validation", validation)):
            required = {"positive", "negative"} if name == "train" and args.holdout_from_training else {"positive", "negative", "background"}
            if not required <= {r["label"] for r in rows}:
                raise ValueError(f"Record {name} positives, negatives, and background before training")
        sx, sy = load_synthetic(args.synthetic_features)
        px, py, pw, alignment = personal_features(args.base_model, train, args.augmentations)
        x, y = np.concatenate([sx, px]), np.concatenate([sy, py])
        weights = training_weights(sy, py, pw)
        scaler = StandardScaler()
        features = scaler.fit_transform(x.reshape(len(x), -1), sample_weight=weights)
        classifier = (LogisticRegression(C=0.025, max_iter=1000, random_state=SEED)
                      if args.head == "linear" else
                      MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.5, max_iter=150,
                                    batch_size=128, early_stopping=False, random_state=SEED))
        print(f"Fitting {len(sx)} synthetic and {len(px)} personal training windows", flush=True)
        with threadpool_limits(limits=2):
            classifier.fit(features, y, sample_weight=weights)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
        output_dir = args.output_dir or args.dataset / "candidates" / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        for name in ("melspectrogram.onnx", "embedding_model.onnx"):
            shutil.copy2(args.base_model.parent / name, output_dir / name)
        candidate = output_dir / "hey_gemma.onnx"
        bootstrap.export_head(classifier, scaler, candidate)
        probe = np.concatenate([sx[:8], px[:8]])
        expected = classifier.predict_proba(scaler.transform(probe.reshape(len(probe), -1)))[:, 1]
        exported = ort.InferenceSession(str(candidate), providers=["CPUExecutionProvider"])
        np.testing.assert_allclose(exported.run(None, {"input": probe})[0].reshape(-1), expected, atol=1e-5)
        candidate_rows = score_recordings(candidate, validation)
        threshold, target_met = choose_threshold(candidate_rows)
        base_rows = score_recordings(args.base_model, validation)
        base_tuned, base_target_met = choose_threshold(base_rows)
        base_metadata = args.base_model.with_suffix(".metadata.json")
        base_threshold = float(json.loads(base_metadata.read_text()).get("recommended_threshold", 0.7)) if base_metadata.exists() else 0.7
        if not np.isfinite(base_threshold) or not 0 < base_threshold < 1:
            raise ValueError("Invalid baseline metadata threshold")
        report = {
            "phrase": "Hey Gemma", "status": "experimental-personal-candidate",
            "model": candidate.name, "model_sha256": bootstrap.sha256(candidate),
            "recommended_threshold": threshold, "validation_target_met": bool(target_met),
            "threshold_selected_using": ("Reserved original takes from training session; preliminary calibration only"
                                         if args.holdout_from_training else "Real validation sessions only; two consecutive 80 ms frames"),
            "validation_policy": "recording_holdout" if args.holdout_from_training else "separate_sessions",
            "independent_session_evaluated": not args.holdout_from_training,
            "validation_targets": {"recall": 0.9, "negative_clip_false_trigger_fraction": 0.0,
                                   "background_false_activations_per_hour": 2.0},
            "validation": summarize(candidate_rows, threshold),
            "baseline_validation": summarize(base_rows, base_threshold),
            "baseline_recalibrated_validation": summarize(base_rows, base_tuned),
            "baseline_recalibrated_target_met": bool(base_target_met),
            "validation_clip_scores": [{"path": r["path"], "label": r["label"],
                                        "candidate_sustained_max": r["sustained_max"],
                                        "baseline_sustained_max": b["sustained_max"]}
                                       for r, b in zip(candidate_rows, base_rows)],
            "input": ["batch", 16, 96], "output": ["batch", 1],
            "audio": "16 kHz mono int16 PCM", "seed": SEED,
            "head": args.head,
            "classifier": ("Weighted standardized logistic regression over frozen openWakeWord backbone"
                           if args.head == "linear" else
                           "Weighted standardized MLP (64,32) over frozen openWakeWord backbone"),
            "synthetic_train_windows": len(sx), "personal_train_windows": len(px),
            "training_sessions": sorted({r["session_id"] for r in train}),
            "validation_sessions": sorted({r["session_id"] for r in validation}),
            "base_model_sha256": bootstrap.sha256(args.base_model),
            "synthetic_training_cache_sha256": bootstrap.sha256(args.synthetic_features),
            "recording_sha256": [{"path": str(r["path"]), "split": r["split"],
                                  "sha256": bootstrap.sha256(r["path"])} for r in train + validation],
            "positive_alignment": alignment, "test_evaluated": False,
            "limitations": [
                "Small personal dataset is an experiment, not a guarantee of reliable detection.",
                "Positive alignment assumes a single isolated Hey Gemma with silence on either side.",
                "A short background recording cannot establish a reliable long-term false activation rate.",
                "Test on a separate recording session after choosing the model and threshold.",
                "No speaker authentication; other voices may still activate this detector.",
            ],
        }
        if args.holdout_from_training:
            report["limitations"].insert(0, "Calibration uses reserved takes from the same session; fresh-session performance is unknown.")
        candidate.with_suffix(".metadata.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("validation_target_met", "validation",
              "baseline_validation", "baseline_recalibrated_validation")}, indent=2))
        print(f"Candidate: {candidate.resolve()}")
        print("Installed model unchanged. Compare on a separate test session before selecting this candidate.")
        if not target_met:
            print("Validation targets NOT met. Inspect missed wakes and false triggers before use.")
        return 0
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Training failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
