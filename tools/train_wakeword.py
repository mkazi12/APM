#!/usr/bin/env python3
"""Bootstrap an experimental Hey Gemma openWakeWord head on a Mac.

Uses local macOS `say`, the unchanged official openWakeWord audio backbone,
and a tiny CPU classifier. Synthetic-only validation is *not* a measure of
real microphone accuracy. Does not record a microphone or upload audio.

Run from the repository root with:
    .venv/bin/python tools/train_wakeword.py
    .venv/bin/python tools/evaluate_wakeword.py --threshold 0.9

Training-only dependencies: openwakeword, onnxruntime, onnx, scikit-learn,
numpy, scipy, soundfile. Generated audio/features are cached in work/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
from onnx import TensorProto, helper, numpy_helper
from scipy.signal import resample_poly
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

RATE = 16000
SAMPLES = 32000
SEED = 427
TRAIN_VOICES = ["Samantha (English (US))", "Daniel", "Karen", "Rishi", "Tessa", "Fred"]
VALIDATION_VOICES = ["Moira"]
TEST_VOICES = ["Eddy (English (US))", "Flo (English (US))"]
POSITIVE = ["Hey Gemma.", "Hey, Gemma!", "Hey Gemma?", "Hey Jemma."]
NEGATIVE = [
    "Gemma", "Hey", "Hey Emma", "Hey Jenna", "Hey Gamma", "Hey Hannah", "Hey Gem", "Hey Jim",
    "Hey James", "Hey Janet", "Hey Gina", "Hey Jennifer", "Hey grandma", "Hey mama", "Hey Kendra",
    "Okay Gemma", "Gemma is here", "I met Gemma yesterday", "Gemma told me", "What is Gemma?",
    "Hey Siri", "Hey Google", "Hey Jarvis", "Alexa", "Hello there", "Good morning", "Good evening",
    "Thank you", "You're welcome", "How are you?", "What is your name?", "What time is it?",
    "Turn off the kitchen lights", "Turn on the kitchen lights", "Close the garage", "Open the garage",
    "Turn them back on", "Don't close the garage", "Stop", "Cancel that", "Never mind", "Please wait",
    "The lights are off", "The garage is closed", "Can you help me?", "Where are my keys?",
    "Have a good day", "The weather is nice", "I am making dinner", "Let's watch television",
    "Turn up the music", "Would you like some coffee?", "We should leave soon", "I need to go shopping",
    "Pick up the phone", "Call me later", "See you tomorrow", "A very good idea", "It is raining outside",
    "The dog is sleeping", "We live in California", "That is my friend", "The train is coming",
    "What's for dinner?", "Can you hear me?", "Testing one two three", "Go to the gym", "Get the camera",
    "A game of chess", "Hey, give me a minute", "Hey, get me some water", "Hey gentleman", "Hey Germany",
    "Hey Gemini", "Hey gems", "Hey, jump up", "Play jazz music", "Happy birthday", "I remember now",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthesize(job: tuple[str, str, int, Path]) -> Path:
    voice, text, speed, directory = job
    key = hashlib.sha256(f"{voice}|{speed}|{text}".encode()).hexdigest()[:20]
    path = directory / f"{key}.wav"
    if not path.exists():
        subprocess.run(["/usr/bin/say", "-v", voice, "-r", str(speed),
                        "--data-format=LEI16@16000", "--file-format=WAVE", "-o", str(path), text],
                       check=True, capture_output=True)
    return path


def synthesize_group(jobs: list[tuple[str, str, int, Path]]) -> list[Path]:
    """Reuse voice initialization across phrases, split only on deliberate long gaps."""
    def path_for(job):
        voice, text, speed, directory = job
        key = hashlib.sha256(f"{voice}|{speed}|{text}".encode()).hexdigest()[:20]
        return directory / f"{key}.wav"
    missing = [job for job in jobs if not path_for(job).exists()]
    if not missing:
        return [path_for(job) for job in jobs]
    voice, _, speed, directory = missing[0]
    batch_key = hashlib.sha256(repr(missing).encode()).hexdigest()[:16]
    batch_path = directory.parent / f"batch-{batch_key}.wav"
    text = " [[slnc 750]] ".join(job[1] for job in missing)
    subprocess.run(["/usr/bin/say", "-v", voice, "-r", str(speed),
                    "--data-format=LEI16@16000", "--file-format=WAVE", "-o", str(batch_path), text],
                   check=True, capture_output=True)
    audio, sr = sf.read(batch_path, dtype="float32")
    active = np.flatnonzero(np.abs(audio) > 0.003)
    breaks = np.flatnonzero(np.diff(active) > sr * 0.5)
    starts = np.r_[active[0], active[breaks + 1]]
    ends = np.r_[active[breaks], active[-1]]
    if len(starts) != len(missing):
        print(f"{voice}: batch split mismatch ({len(starts)} vs {len(missing)}); retrying individual clips", flush=True)
        return [synthesize(job) for job in jobs]
    for job, start, end in zip(missing, starts, ends):
        clip = audio[max(0, start-320):min(len(audio), end+320)]
        sf.write(path_for(job), clip, RATE, subtype="PCM_16")
    print(f"Synthesized {len(missing)} clips: {voice}, {speed} wpm", flush=True)
    return [path_for(job) for job in jobs]


def prepare_wave(path: Path, rng: np.random.Generator, augmented: bool) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if sr != RATE:
        raise ValueError(f"Unexpected sample rate in {path}: {sr}")
    active = np.flatnonzero(np.abs(audio) > max(0.003, float(np.max(np.abs(audio))) * 0.025))
    if not len(active):
        raise ValueError(f"Empty synthetic voice: {path}")
    audio = audio[max(0, active[0] - 160):min(len(audio), active[-1] + 240)]
    if augmented:
        speed = int(rng.integers(91, 111))
        audio = resample_poly(audio, 100, speed).astype(np.float32)
        # Mild echo and varying level; this is not a substitute for real room data.
        if rng.random() < 0.5:
            delay = int(rng.integers(160, 1600))
            echo = np.pad(audio, (delay, 0))[:len(audio)]
            audio = audio + echo * rng.uniform(0.05, 0.25)
    post = int(rng.uniform(0.0, 0.32) * RATE)
    output = np.zeros(SAMPLES, dtype=np.float32)
    end = SAMPLES - post
    count = min(len(audio), end)
    output[end-count:end] = audio[-count:]
    output *= rng.uniform(0.25, 1.0) if augmented else 0.7
    if augmented:
        noise = rng.normal(0, rng.uniform(0.0002, 0.006), SAMPLES)
        output += noise.astype(np.float32)
    return (np.clip(output, -1, 1) * 32767).astype(np.int16)


def make_split(name: str, voices: list[str], work: Path, extractor) -> tuple[np.ndarray, np.ndarray]:
    cache = work / f"{name}-features-v1.npz"
    if cache.exists():
        loaded = np.load(cache)
        return loaded["x"], loaded["y"]
    rng = np.random.default_rng(SEED + {"train": 0, "validation": 1, "test": 2}[name])
    specs = [(voice, phrase, speed, work / "clips", label)
             for voice in voices for label, phrases in [(1, POSITIVE), (0, NEGATIVE)]
             for phrase in phrases for speed in (145, 190)]
    print(f"{name}: synthesizing {len(specs)} clips across {len(voices)} voices", flush=True)
    groups = [[s[:4] for s in specs if s[0] == voice and s[2] == speed]
              for voice in voices for speed in (145, 190)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(synthesize_group, groups))
    paths = [synthesize(s[:4]) for s in specs]  # all cached by grouped synthesis
    waves, labels = [], []
    for spec, path in zip(specs, paths):
        label = spec[-1]
        repeats = 12 if label else 1
        if name != "train":
            repeats = 3 if label else 1
        for i in range(repeats):
            waves.append(prepare_wave(path, rng, augmented=(name == "train" or i > 0)))
            labels.append(label)
    # Teach background-only negatives without pretending they are realistic noise recordings.
    for i in range(80 if name == "train" else 20):
        amplitude = rng.uniform(0, 0.02)
        noise = rng.normal(0, amplitude, SAMPLES)
        waves.append((np.clip(noise, -1, 1) * 32767).astype(np.int16))
        labels.append(0)
    print(f"{name}: extracting {len(waves)} fixed-backbone feature windows", flush=True)
    features = []
    for start in range(0, len(waves), 64):
        batch = np.stack(waves[start:start+64])
        embedded = extractor.embed_clips(batch, batch_size=64, ncpu=2)
        if embedded.shape[1:] != (16, 96):
            raise ValueError(f"Unexpected backbone features {embedded.shape}")
        features.append(embedded)
    x = np.concatenate(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    np.savez_compressed(cache, x=x, y=y)
    return x, y


def make_runtime_features(work: Path, models: Path) -> tuple[np.ndarray, np.ndarray]:
    """Mine training-only features through exactly the deployed streaming path."""
    cache = work / "train-runtime-features-v1.npz"
    if cache.exists():
        data = np.load(cache)
        return data["x"], data["y"]
    from openwakeword.model import Model
    model = Model(wakeword_models=[str(models / "hey_gemma.onnx")], inference_framework="onnx",
                  melspec_model_path=str(models / "melspectrogram.onnx"),
                  embedding_model_path=str(models / "embedding_model.onnx"), ncpu=1)
    x, y = [], []
    for voice in TRAIN_VOICES:
        for label, phrases in [(1, POSITIVE), (0, NEGATIVE)]:
            for text in phrases:
                for speed in (145, 190):
                    audio, _ = sf.read(synthesize((voice, text, speed, work / "clips")), dtype="int16")
                    active = np.flatnonzero(np.abs(audio.astype(np.int32)) > 100)
                    wake_end = 16000 + active[-1]
                    stream = np.pad(audio, (16000, 16000))
                    model.reset()
                    candidates = []
                    for offset in range(0, len(stream)-1280, 1280):
                        score = float(model.predict(stream[offset:offset+1280])["hey_gemma"])
                        feature = model.preprocessor.get_features(16)[0].copy()
                        if label and wake_end <= offset+1280 <= wake_end+5120:
                            x.append(feature); y.append(1)
                        elif not label and offset >= 12800:
                            candidates.append((score, feature))
                    if not label:
                        for _, feature in sorted(candidates, key=lambda item: item[0], reverse=True)[:12]:
                            x.append(feature); y.append(0)
        print(f"Streaming training features: {voice} complete", flush=True)
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64)
    np.savez_compressed(cache, x=x, y=y)
    return x, y


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    return {"positive_windows": int((y == 1).sum()), "negative_windows": int((y == 0).sum()),
            "true_positive_rate": float(np.mean(p[y == 1] >= threshold)),
            "false_positive_rate_per_window": float(np.mean(p[y == 0] >= threshold)),
            "roc_auc": float(roc_auc_score(y, p)), "threshold": threshold}


def export_head(classifier, scaler, output: Path) -> None:
    if isinstance(classifier, LogisticRegression):
        weights = (classifier.coef_.reshape(-1) / scaler.scale_).astype(np.float32)
        bias = np.asarray([classifier.intercept_[0] - np.dot(weights, scaler.mean_)], dtype=np.float32)
        graph = helper.make_graph([
            helper.make_node("Flatten", ["input"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "weights"], ["logits"]),
            helper.make_node("Add", ["logits", "bias"], ["shifted"]),
            helper.make_node("Sigmoid", ["shifted"], ["output"]),
        ], "experimental_hey_gemma", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 16, 96])],
           [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])],
           [numpy_helper.from_array(weights[:, None], "weights"), numpy_helper.from_array(bias, "bias")])
        model = helper.make_model(graph, producer_name="APM experimental wake trainer",
                                  opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        onnx.save(model, output)
        return
    nodes = [helper.make_node("Flatten", ["input"], ["flat"], axis=1),
             helper.make_node("Sub", ["flat", "mean"], ["centered"]),
             helper.make_node("Div", ["centered", "scale"], ["normalized"])]
    initializers = [numpy_helper.from_array(scaler.mean_.astype(np.float32), "mean"),
                    numpy_helper.from_array(scaler.scale_.astype(np.float32), "scale")]
    previous = "normalized"
    for i, (weights, bias) in enumerate(zip(classifier.coefs_, classifier.intercepts_)):
        w, b, logits, activation = f"weights_{i}", f"bias_{i}", f"logits_{i}", f"activation_{i}"
        initializers.extend([numpy_helper.from_array(weights.astype(np.float32), w),
                             numpy_helper.from_array(bias.astype(np.float32), b)])
        nodes.append(helper.make_node("Gemm", [previous, w, b], [logits]))
        last = i == len(classifier.coefs_)-1
        nodes.append(helper.make_node("Sigmoid" if last else "Relu", [logits], ["output" if last else activation]))
        previous = activation
    graph = helper.make_graph(nodes, "experimental_hey_gemma",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 16, 96])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])], initializers)
    model = helper.make_model(graph, producer_name="APM experimental wake trainer",
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--head", choices=("linear", "mlp"), default="linear",
                        help="Linear is the tested prototype; MLP remains experimental.")
    args = parser.parse_args()
    np.random.seed(SEED)
    if platform.system() != "Darwin":
        parser.error("Training synthesis uses macOS say. Exported ONNX runs on Mac and Linux.")
    work, models = args.root / "work/wake-training", args.root / "models"
    (work / "clips").mkdir(parents=True, exist_ok=True)
    models.mkdir(exist_ok=True)
    sources = {}
    for name in ("melspectrogram.onnx", "embedding_model.onnx"):
        url = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/" + name
        if not (models / name).exists():
            urlretrieve(url, models / name)
        sources[name] = {"url": url, "sha256": sha256(models / name)}
    from openwakeword.utils import AudioFeatures
    extractor = AudioFeatures(melspec_model_path=str(models / "melspectrogram.onnx"),
                              embedding_model_path=str(models / "embedding_model.onnx"), ncpu=2)
    started = time.perf_counter()
    train_x, train_y = make_split("train", TRAIN_VOICES, work, extractor)
    val_x, val_y = make_split("validation", VALIDATION_VOICES, work, extractor)
    test_x, test_y = make_split("test", TEST_VOICES, work, extractor)
    bootstrap_only = args.head == "mlp" and not (models / "hey_gemma.onnx").exists()
    if bootstrap_only:
        # First pass creates a bootstrap only for hard-negative mining.
        # Automatically perform a second pass once that seed head is exported.
        print("First pass: bootstrap head for streaming hard-negative mining.", flush=True)
    elif args.head == "mlp":
        runtime_x, runtime_y = make_runtime_features(work, models)
        train_x = np.concatenate([train_x, runtime_x])
        train_y = np.concatenate([train_y, runtime_y])
    scaler = StandardScaler()
    features = scaler.fit_transform(train_x.reshape(len(train_x), -1))
    if args.head == "linear":
        classifier = LogisticRegression(C=0.025, class_weight="balanced", max_iter=600, random_state=SEED)
        print(f"Training linear prototype on {len(train_y)} windows", flush=True)
        with threadpool_limits(limits=2):
            classifier.fit(features, train_y)
    else:
        classifier = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.05, learning_rate_init=0.001,
                                   batch_size=256, max_iter=100, early_stopping=True, n_iter_no_change=10,
                                   random_state=SEED)
        weight = np.where(train_y == 1, (train_y == 0).sum()/(train_y == 1).sum(), 1.0)
        print(f"Training nonlinear experiment on {len(train_y)} windows", flush=True)
        with threadpool_limits(limits=2):
            classifier.fit(features, train_y, sample_weight=weight)
    val_p = classifier.predict_proba(scaler.transform(val_x.reshape(len(val_x), -1)))[:, 1]
    # Tune only on validation voice; held-out test speakers remain untouched.
    candidates = np.linspace(0.5, 0.95, 46)
    threshold = 0.5
    for candidate in candidates:
        if np.mean(val_p[val_y == 0] >= candidate) <= 0.01:
            threshold = float(round(candidate, 2))
            break
    else:
        threshold = 0.95
    test_p = classifier.predict_proba(scaler.transform(test_x.reshape(len(test_x), -1)))[:, 1]
    output = models / "hey_gemma.onnx"
    export_head(classifier, scaler, output)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    exported_p = session.run(None, {"input": test_x[:32]})[0].reshape(-1)
    np.testing.assert_allclose(exported_p, test_p[:32], atol=1e-5)
    report = {"phrase": "Hey Gemma", "status": "experimental-synthetic-bootstrap",
              "model": output.name, "model_sha256": sha256(output), "recommended_threshold": threshold,
              "input": ["batch", 16, 96], "output": ["batch", 1], "audio": "16 kHz mono int16 PCM",
              "engine": "openWakeWord 0.6.0 ONNX", "seed": SEED,
              "classifier": ("standardized logistic regression; normalization folded into ONNX weights"
                             if args.head == "linear" else "standardized MLP (64,32)"),
              "head": args.head,
              "train_voices": TRAIN_VOICES, "validation_voices": VALIDATION_VOICES, "test_voices": TEST_VOICES,
              "train_windows": len(train_y), "validation": metrics(val_y, val_p, threshold),
              "held_out_synthetic_test": metrics(test_y, test_p, threshold),
              "backbone_sources": sources, "synthetic_audio_source": "locally installed macOS say voices",
              "duration_seconds": round(time.perf_counter()-started, 2),
              "limitations": ["No real microphone recordings evaluated.",
                              "Small synthetic dataset; unrelated natural speech, TV, music and room noise are not adequately represented.",
                              "Per-window synthetic false positives do not establish false activations per hour.",
                              "Validate on real speakers, distances and room noise before relying on detection.",
                              "Bootstrap trained with Apple system voices; redistribution/commercial licensing has not been established."],
              "recipe": "tools/train_wakeword.py", "positive_phrases": POSITIVE, "negative_phrases": NEGATIVE}
    (models / "hey_gemma.metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"model": str(output), "size_bytes": output.stat().st_size,
                      "validation": report["validation"], "held_out_synthetic_test": report["held_out_synthetic_test"]}, indent=2))
    if bootstrap_only:
        main()


if __name__ == "__main__":
    main()
