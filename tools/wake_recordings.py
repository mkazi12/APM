"""Validated personal recordings and reproducible 80 ms wake-word evaluation.

Background events require two consecutive above-threshold frames. After an
event, rearm only after two continuous seconds below threshold. This measures
the detector alone, without the assistant's command/playback pauses.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 16000
BLOCK = 1280
FRAME_SECONDS = BLOCK / RATE
# Flush openWakeWord's randomized reset embeddings before scoring or extracting
# training features: 26 full frames = 2.08 seconds of deterministic silence.
WARMUP_FRAMES = 26
REARM_SECONDS = 2.0
LABELS = {"positive", "negative", "background"}
SPLITS = {"train", "validation", "test"}


def read_audio(record: dict) -> np.ndarray:
    """Read a complete, nonempty 16 kHz mono PCM16 WAV; never resample silently."""
    path = Path(record["path"])
    with sf.SoundFile(path) as source:
        if (source.samplerate != RATE or source.channels != 1
                or source.subtype != "PCM_16" or source.format not in {"WAV", "WAVEX"}):
            raise ValueError(f"Expected 16 kHz mono PCM16 WAV: {path}")
        audio = source.read(dtype="int16")
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    return audio


def load_recordings(dataset: Path) -> list[dict]:
    """Load manifests, enforcing session boundaries and detecting copied PCM."""
    dataset = Path(dataset).resolve()
    records, sessions, audio_splits = [], set(), {}
    for manifest_path in sorted(dataset.glob("*/*/session.json")):
        if not manifest_path.resolve().is_relative_to(dataset):
            raise ValueError(f"Manifest escapes dataset: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        split, session = manifest.get("split"), manifest.get("session_id")
        if (manifest.get("schema_version") != 1 or split not in SPLITS
                or not isinstance(session, str) or not session
                or split != manifest_path.parent.parent.name
                or session != manifest_path.parent.name):
            raise ValueError(f"Invalid manifest identity or schema: {manifest_path}")
        if session in sessions:
            raise ValueError(f"Duplicate session ID across dataset: {session}")
        sessions.add(session)
        entries = manifest.get("recordings")
        if not isinstance(entries, list):
            raise ValueError(f"Manifest recordings must be a list: {manifest_path}")
        seen_files = set()
        for item in entries:
            if not isinstance(item, dict):
                raise ValueError(f"Invalid recording entry: {manifest_path}")
            filename = item.get("file")
            if (not isinstance(filename, str) or Path(filename).name != filename
                    or "\\" in filename or Path(filename).suffix.lower() != ".wav"):
                raise ValueError(f"Unsafe WAV filename in {manifest_path}: {filename!r}")
            path = (manifest_path.parent / filename).resolve()
            if not path.is_relative_to(manifest_path.parent.resolve()) or not path.is_relative_to(dataset):
                raise ValueError(f"Recording escapes session: {path}")
            if path in seen_files:
                raise ValueError(f"Duplicate recording in session: {path}")
            seen_files.add(path)
            duration = item.get("duration_seconds")
            if (item.get("label") not in LABELS or not isinstance(item.get("text"), str)
                    or isinstance(duration, bool) or not isinstance(duration, (int, float))
                    or not math.isfinite(duration) or duration <= 0):
                raise ValueError(f"Invalid recording metadata: {path}")
            record = dict(item, path=path, split=split, session_id=session)
            audio = read_audio(record)
            measured_duration = len(audio) / RATE
            if abs(duration - measured_duration) > 0.01:
                raise ValueError(f"Duration disagrees with WAV samples: {path}")
            record["duration_seconds"] = measured_duration
            digest = hashlib.sha256(audio.astype("<i2", copy=False).tobytes()).hexdigest()
            if digest in audio_splits and audio_splits[digest] != split:
                raise ValueError(f"Identical audio across splits ({audio_splits[digest]}, {split}): {path}")
            audio_splits[digest] = split
            records.append(record)
    return records


def make_model(model_path: Path):
    """Use official openWakeWord with local, shared ONNX feature backbones."""
    from openwakeword.model import Model

    model_path = Path(model_path).resolve()
    backbone_dir = model_path.parent
    if not model_path.is_file() or not all((backbone_dir / name).is_file()
            for name in ("melspectrogram.onnx", "embedding_model.onnx")):
        raise ValueError(f"Missing model or ONNX backbones beside {model_path}")
    return Model(wakeword_models=[str(model_path)], inference_framework="onnx", ncpu=1,
                 melspec_model_path=str(backbone_dir / "melspectrogram.onnx"),
                 embedding_model_path=str(backbone_dir / "embedding_model.onnx"))


def stream_scores(model, audio: np.ndarray) -> list[float]:
    """Reset, warm up, then score every sample using exactly 80 ms per call."""
    if audio.ndim != 1 or audio.dtype != np.int16 or not len(audio):
        raise ValueError("Streaming input must be nonempty mono int16 audio")
    model.reset()
    for _ in range(WARMUP_FRAMES):
        model.predict(np.zeros(BLOCK, dtype=np.int16))
    scores = []
    for offset in range(0, len(audio), BLOCK):
        frame = audio[offset:offset + BLOCK]
        if len(frame) < BLOCK:
            frame = np.pad(frame, (0, BLOCK - len(frame)))
        prediction = model.predict(frame)
        if len(prediction) != 1:
            raise ValueError("Evaluation expects exactly one wake-word output")
        score = float(next(iter(prediction.values())))
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid model probability: {score}")
        scores.append(score)
    return scores


def sustained_score(scores: list[float]) -> float:
    return max((min(a, b) for a, b in zip(scores, scores[1:])), default=0.0)


def score_recordings(model_path: Path, records: list[dict]) -> list[dict]:
    model = make_model(model_path)
    rows = []
    for record in records:
        audio = read_audio(record)
        scores = stream_scores(model, audio)
        rows.append({**{key: record[key] for key in ("label", "split", "session_id", "text")},
                     "path": str(record["path"]), "duration_seconds": len(audio) / RATE,
                     "scores": scores, "sustained_max": sustained_score(scores)})
    return rows


def validate_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("Threshold must be a finite number strictly between 0 and 1")
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Threshold must be a finite number strictly between 0 and 1")
    return float(threshold)


def background_events(scores: list[float], threshold: float) -> int:
    events, consecutive, below = 0, 0, 0
    armed = True
    rearm_frames = math.ceil(REARM_SECONDS / FRAME_SECONDS)
    for score in scores:
        if score < threshold:
            consecutive = 0
            below += 1
            if below >= rearm_frames:
                armed = True
        else:
            below = 0
            consecutive += 1
            if armed and consecutive >= 2:
                events += 1
                armed = False
    return events


def summarize(rows: list[dict], threshold: float) -> dict:
    threshold = validate_threshold(threshold)
    positives = [r for r in rows if r["label"] == "positive"]
    negatives = [r for r in rows if r["label"] == "negative"]
    background = [r for r in rows if r["label"] == "background"]
    detected = sum(r["sustained_max"] >= threshold for r in positives)
    false_clips = sum(r["sustained_max"] >= threshold for r in negatives)
    hours = sum(r["duration_seconds"] for r in background) / 3600
    events = sum(background_events(r["scores"], threshold) for r in background)
    return {"threshold": threshold, "positive_clips": len(positives),
            "detected_positive_clips": detected, "recall": detected / len(positives) if positives else None,
            "negative_clips": len(negatives), "false_trigger_clips": false_clips,
            "false_trigger_fraction": false_clips / len(negatives) if negatives else None,
            "background_clips": len(background), "background_hours": hours,
            "background_activation_events": events,
            "background_false_activations_per_hour": events / hours if hours else None,
            "consecutive_frames": 2, "rearm_seconds_below_threshold": REARM_SECONDS}


def calibration_missing(rows: list[dict]) -> list[str]:
    return sorted(LABELS - {r["label"] for r in rows})


def threshold_candidates(rows: list[dict]) -> list[float]:
    """Include every speech boundary and intervening range, even near 0 or 1.

    Background event counts can change at individual frame scores. Use all
    distinct background boundaries up to 256, then representative quantiles to
    bound replay cost for long recordings. Speech boundaries remain exact.
    """
    speech = [r["sustained_max"] for r in rows if r["label"] != "background"]
    background = np.unique([score for r in rows if r["label"] == "background"
                            for score in r["scores"]])
    if len(background) > 256:
        background = np.quantile(background, np.linspace(0, 1, 256), method="nearest")
    boundaries = sorted({0.0, 1.0, *speech, *background.tolist()})
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in boundaries):
        raise ValueError("Calibration scores must be finite probabilities")
    midpoints = [(left + right) / 2 for left, right in zip(boundaries, boundaries[1:])]
    return sorted({score for score in boundaries + midpoints if 0 < score < 1})


def choose_threshold(validation_rows: list[dict], min_recall: float = 0.9,
                     max_false_trigger_fraction: float = 0.0,
                     max_background_faph: float = 2.0) -> tuple[float, bool]:
    """Tune on validation only; missing categories produce an unusable fallback.

    Among feasible thresholds, maximize recall, then choose the tested threshold
    nearest the midpoint of that feasible range to leave margin. If infeasible,
    minimize the sum of target violations, then prefer higher recall. The caller
    must report missing categories and must not deploy a fallback as calibrated.
    """
    if any(r["split"] != "validation" for r in validation_rows):
        raise ValueError("Threshold selection accepts validation recordings only")
    if not (0 < min_recall <= 1 and 0 <= max_false_trigger_fraction <= 1
            and math.isfinite(max_background_faph) and max_background_faph >= 0):
        raise ValueError("Invalid threshold calibration targets")
    if calibration_missing(validation_rows):
        return 0.5, False
    candidates = [summarize(validation_rows, value) for value in threshold_candidates(validation_rows)]
    def feasible(result):
        return (result["recall"] >= min_recall
                and result["false_trigger_fraction"] <= max_false_trigger_fraction
                and result["background_false_activations_per_hour"] <= max_background_faph)
    successful = [r for r in candidates if feasible(r)]
    if successful:
        best_recall = max(r["recall"] for r in successful)
        best = [r for r in successful if r["recall"] == best_recall]
        midpoint = (best[0]["threshold"] + best[-1]["threshold"]) / 2
        chosen = min(best, key=lambda r: abs(r["threshold"] - midpoint))
        return chosen["threshold"], True
    def failure_key(result):
        loss = (max(0, min_recall - result["recall"]) / min_recall
                + max(0, result["false_trigger_fraction"] - max_false_trigger_fraction)
                + max(0, result["background_false_activations_per_hour"] - max_background_faph)
                / max(1, max_background_faph))
        return loss, -result["recall"]
    best_key = min(failure_key(r) for r in candidates)
    best = [r for r in candidates if failure_key(r) == best_key]
    midpoint = (best[0]["threshold"] + best[-1]["threshold"]) / 2
    return min(best, key=lambda r: abs(r["threshold"] - midpoint))["threshold"], False
