#!/usr/bin/env python3
"""Replay synthetic clips through openWakeWord's actual 80ms streaming API.

Run after tools/train_wakeword.py. Threshold selection uses validation voice
Moira only; Eddy/Flo remain excluded from fitting. Test results are synthetic,
small, and diagnostic, not a real microphone or false-activations/hour claim.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import soundfile as sf
from openwakeword.model import Model
import train_wakeword as trainer


def stream_scores(model, audio: np.ndarray) -> list[float]:
    model.reset()
    stream = np.pad(audio, (16000, 16000))
    return [float(model.predict(stream[i:i+1280])["hey_gemma"])
            for i in range(0, len(stream)-1280, 1280)]


def sustained_score(scores: list[float]) -> float:
    return max(min(scores[i-1], scores[i]) for i in range(1, len(scores)))


def summarize(rows: list[dict], threshold: float) -> dict:
    positives = [r for r in rows if r["label"]]
    negatives = [r for r in rows if not r["label"]]
    tp = sum(r["sustained_max"] >= threshold for r in positives)
    fp = sum(r["sustained_max"] >= threshold for r in negatives)
    return {"positive_clips": len(positives), "detected_positive_clips": tp,
            "negative_clips": len(negatives), "false_trigger_clips": fp,
            "recall": tp / len(positives), "false_trigger_fraction": fp / len(negatives),
            "threshold": threshold, "consecutive_frames": 2,
            "false_trigger_examples": [r["text"] for r in negatives if r["sustained_max"] >= threshold]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--threshold", type=float, help="Fixed threshold; otherwise tune on Moira validation only")
    args = parser.parse_args()
    np.random.seed(trainer.SEED + 4)
    work, models = args.root / "work/wake-training", args.root / "models"
    model = Model(wakeword_models=[str(models / "hey_gemma.onnx")], inference_framework="onnx",
                  melspec_model_path=str(models / "melspectrogram.onnx"),
                  embedding_model_path=str(models / "embedding_model.onnx"), ncpu=1)
    rows = []
    started = time.perf_counter()
    for split, voices in [("validation", trainer.VALIDATION_VOICES), ("test", trainer.TEST_VOICES)]:
        for voice in voices:
            for label, phrases in [(1, trainer.POSITIVE), (0, trainer.NEGATIVE)]:
                for text in phrases:
                    for speed in (145, 190):
                        path = trainer.synthesize((voice, text, speed, work / "clips"))
                        audio, _ = sf.read(path, dtype="int16")
                        scores = stream_scores(model, audio)
                        rows.append({"split": split, "voice": voice, "label": label, "text": text,
                                     "speed": speed, "sustained_max": sustained_score(scores)})
        print(f"{split}: streaming replay complete", flush=True)
    validation = [r for r in rows if r["split"] == "validation"]
    threshold = args.threshold
    if threshold is None:
        threshold = 0.99
        for candidate in np.linspace(0.5, 0.99, 50):
            result = summarize(validation, float(candidate))
            if result["false_trigger_fraction"] <= 0.01 and result["recall"] >= 0.75:
                threshold = round(float(candidate), 2)
                break
    connected = []
    for voice in trainer.TEST_VOICES:
        for speed in (145, 190):
            for command in ("Turn off the kitchen lights", "Close the garage", "What is your name?"):
                paths = trainer.synthesize_group([
                    (voice, "Hey Gemma.", speed, work / "clips"),
                    (voice, command, speed, work / "clips"),
                    (voice, "Hey Gemma, " + command.lower(), speed, work / "clips")])
                wake, _ = sf.read(paths[0], dtype="float32")
                cmd, _ = sf.read(paths[1], dtype="float32")
                natural, _ = sf.read(paths[2], dtype="float32")
                def trim(x):
                    active = np.flatnonzero(np.abs(x) > 0.003)
                    return x[max(0, active[0]-160):active[-1]+161]
                wake, cmd = trim(wake), trim(cmd)
                for kind, audio in [("joined-no-pause", np.concatenate([wake, cmd])),
                                    ("natural-connected", natural)]:
                    scores = stream_scores(model, (audio*32767).astype(np.int16))
                    triggers = [(j+1)*0.08-1 for j in range(1, len(scores))
                                if min(scores[j-1], scores[j]) >= threshold]
                    delay = (triggers[0]-len(wake)/16000
                             if triggers and kind == "joined-no-pause" else None)
                    connected.append({"voice": voice, "speed": speed, "command": command, "kind": kind,
                                      "trigger_seconds_after_audio_start": triggers[0] if triggers else None,
                                      "delay_after_exact_joined_wake_end": delay,
                                      "sustained_max": sustained_score(scores)})
    report = {"model_sha256": trainer.sha256(models / "hey_gemma.onnx"),
              "recommended_threshold": threshold, "frame_ms": 80, "consecutive_frames": 2,
              "threshold_selected_using": ("Explicit prototype threshold; favors wake recall, not a claim of meeting calibration targets"
                  if args.threshold is not None else "Moira validation voice only; targets <=1% synthetic negative-clip triggers and >=75% wake recall"),
              "validation_target_met": (summarize(validation, threshold)["false_trigger_fraction"] <= 0.01
                                        and summarize(validation, threshold)["recall"] >= 0.75),
              "validation": summarize(validation, threshold),
              "test": summarize([r for r in rows if r["split"] == "test"], threshold),
              "connected_commands": connected,
              "evaluation_seconds": round(time.perf_counter()-started, 2),
              "limitations": "Synthetic voices only, test set used diagnostically during development; no real room/microphone evaluation. No false-activations/hour estimate."}
    (work / "streaming-evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    (work / "streaming-clip-scores.json").write_text(json.dumps(rows, indent=2) + "\n")
    metadata_path = models / "hey_gemma.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["recommended_threshold"] = threshold
    metadata["streaming_evaluation"] = report
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "connected_commands"}, indent=2))
    print(f"Connected-command triggers: {sum(r['trigger_seconds_after_audio_start'] is not None for r in connected)}/{len(connected)}")


if __name__ == "__main__":
    main()
