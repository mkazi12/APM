#!/usr/bin/env python3
"""Compare Hey Gemma models on complete, held-out personal recordings.

Validation reports both fixed thresholds and independently calibrated thresholds.
Test evaluation uses only fixed thresholds and never tunes. An explicit baseline
threshold can replay a value selected previously on validation data.
Reports do not update models, model metadata, or the assistant configuration.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

try:
    from .wake_recordings import (FRAME_SECONDS, WARMUP_FRAMES, calibration_missing, choose_threshold, load_recordings,
                                  score_recordings, summarize, validate_threshold)
except ImportError:
    from wake_recordings import (FRAME_SECONDS, WARMUP_FRAMES, calibration_missing, choose_threshold, load_recordings,
                                score_recordings, summarize, validate_threshold)


def metadata_threshold(model_path: Path) -> float:
    metadata_path = model_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text())
    return validate_threshold(metadata.get("recommended_threshold"))


def evaluate(dataset: Path, model_path: Path, candidate: Path | None, split: str,
             baseline_threshold: float | None = None) -> dict:
    if split not in {"validation", "test"}:
        raise ValueError("Evaluate validation or test recordings only")
    # Validate the entire dataset so copied audio in other splits cannot leak in.
    records = [r for r in load_recordings(dataset) if r["split"] == split]
    if not records:
        raise ValueError(f"No {split} recordings in {dataset}; record a separate session first")
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "dataset": str(dataset.resolve()), "split": split,
              "counts": {label: sum(r["label"] == label for r in records)
                         for label in ("positive", "negative", "background")},
              "sessions": sorted({r["session_id"] for r in records}), "models": {},
              "method": {"frame_ms": int(FRAME_SECONDS * 1000), "warmup_seconds": WARMUP_FRAMES * FRAME_SECONDS,
                         "warmup_scores_excluded": True, "consecutive_frames": 2,
                         "background_rearm": "Two continuous seconds below threshold; reset between clips",
                         "calibration_candidates": "Speech score boundaries and adjacent midpoints; background boundaries capped at 256 quantiles",
                         "threshold_selection": "Validation only" if split == "validation" else "None; fixed thresholds only"},
              "limitations": ["Clip recall and negative speech clip fraction are separate from background events/hour.",
                              "Background rate is descriptive of the measured duration; short recordings cannot establish a reliable hourly rate.",
                              "Evaluation excludes assistant command-capture and playback pauses; microphone capture failures require a live test."]}
    for name, path in [("baseline", model_path)] + ([("candidate", candidate)] if candidate else []):
        threshold = (validate_threshold(baseline_threshold)
                     if name == "baseline" and baseline_threshold is not None else metadata_threshold(path))
        rows = score_recordings(path, records)
        result = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "fixed": {**summarize(rows, threshold),
                            "threshold_source": ("explicit baseline threshold; expected to be selected on validation"
                                                 if name == "baseline" and baseline_threshold is not None
                                                 else "model metadata recommended_threshold")}, "scores": rows}
        if split == "validation":
            missing = calibration_missing(rows)
            if missing:
                result["calibration"] = {"available": False, "missing_labels": missing,
                                         "target_met": False, "threshold": None}
            else:
                selected, met = choose_threshold(rows)
                result["calibration"] = {"available": True, "target_met": met,
                                         "targets": {"min_recall": 0.9, "max_false_trigger_fraction": 0.0,
                                                     "max_background_false_activations_per_hour": 2.0},
                                         **summarize(rows, selected)}
        report["models"][name] = result
    return report


def print_report(report: dict) -> None:
    print(f"Split: {report['split']}; sessions: {len(report['sessions'])}; clips: {report['counts']}")
    print(f"{'Model / threshold source':30} {'Threshold':>9} {'Recall':>9} {'Neg. frac':>10} {'BG events/h':>12} {'BG hours':>10}")
    def number(value, digits=3):
        return "n/a" if value is None else f"{value:.{digits}f}"
    for name, result in report["models"].items():
        for source in ("fixed", "calibration"):
            if source not in result:
                continue
            metric = result[source]
            if metric.get("available") is False:
                print(f"{name}: calibration unavailable; missing {', '.join(metric['missing_labels'])}")
                continue
            print(f"{name + ' / ' + source:30} {number(metric['threshold']):>9} {number(metric['recall']):>9} "
                  f"{number(metric['false_trigger_fraction']):>10} "
                  f"{number(metric['background_false_activations_per_hour']):>12} {number(metric['background_hours']):>10}")
            if source == "calibration":
                print(f"  Validation targets met: {metric['target_met']}" +
                      ("; threshold is best effort only" if not metric["target_met"] else ""))
    print("Background events require two consecutive high frames, then two seconds below threshold to rearm.")
    print("Short background recordings provide a preliminary measured rate, not a reliability guarantee.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("work/wake-personal"))
    parser.add_argument("--model", type=Path, default=Path("models/hey_gemma.onnx"))
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline-threshold", type=float,
                        help="Fixed baseline override, selected previously using validation; never fitted on test")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    args = parser.parse_args()
    try:
        report = evaluate(args.dataset, args.model, args.candidate, args.split, args.baseline_threshold)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    destination = args.dataset / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = destination / f"{args.split}-{stamp}-{uuid4().hex[:8]}.json"
    with output.open("x") as target:
        json.dump(report, target, indent=2, allow_nan=False)
        target.write("\n")
    print_report(report)
    print(f"Report: {output.resolve()}")


if __name__ == "__main__":
    main()
