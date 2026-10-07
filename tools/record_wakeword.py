#!/usr/bin/env python3
"""Record local, session-separated wake-word data; audio never leaves this Mac.

    .venv/bin/python tools/record_wakeword.py --list-mics
    .venv/bin/python tools/record_wakeword.py --split train --mic 0

Run validation and test in separate sittings, using the intended microphone.
Generated data defaults to the git-ignored work/wake-personal directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import uuid

import numpy as np
import sounddevice as sd
import soundfile as sf

RATE = 16000
CLIP_SECONDS = 4.0
DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "work" / "wake-personal"
NEGATIVE_PROMPTS = (
    "Hey Emma", "Good morning", "Hey Jenna", "What time is it?",
    "Hey Gemini", "Turn on the kitchen lights", "Gemma", "Where are my keys?",
    "Hey", "I am making dinner", "Hey Gamma", "Please close the door",
    "Hey Hannah", "The weather is nice today", "Hey Jim", "We should leave soon",
    "Hey grandma", "Thank you", "Hey gentleman", "Would you like some coffee?",
)


def record_audio(seconds: float, microphone: int | None, speech: bool) -> tuple[np.ndarray, str]:
    """Capture a complete take and return any PortAudio callback status."""
    for count in (3, 2, 1):
        print(f"{count}…", flush=True)
        time.sleep(0.5)
    audio = sd.rec(round(seconds * RATE), samplerate=RATE, channels=1,
                   dtype="float32", device=microphone)
    try:
        time.sleep(min(0.5, seconds))
        print("SAY IT NOW — then leave trailing silence." if speech else
              "Recording background — do not say the wake phrase.", flush=True)
        status = sd.wait()
    finally:
        sd.stop()
    print("Recording finished.", flush=True)
    return audio[:, 0].copy(), str(status) if status else ""


def audio_quality(audio: np.ndarray, label: str, status: str = "") -> tuple[str, str | None]:
    """Reject damaged/empty audio; let the user review faint speech and whispers."""
    if audio.ndim != 1 or not len(audio) or not np.all(np.isfinite(audio)):
        return "Invalid audio", "Empty, non-mono, or non-finite recording."
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    clipped = int(np.count_nonzero(np.abs(audio) >= 0.9999))
    details = f"Peak {peak:.4f} · RMS {rms:.4f} · clipped samples {clipped}"
    if status:
        return details, f"Dropped or invalid audio buffers: {status}"
    if clipped:
        return details, "Clipped audio. Reduce microphone gain or move farther away."
    if label != "background" and peak < 2 / 32768:
        return details, "Signal is too close to silence at PCM16 precision. Check the microphone or move closer."
    if label != "background" and (peak < 0.001 or rms < 0.0001):
        details += " · Very quiet: play back and check the words before keeping this take."
    return details, None


def review_take(label: str, prompt: str, seconds: float, microphone: int | None) -> np.ndarray | None:
    """Return a kept take, or None when the user quits; retries never save audio."""
    while True:
        print(f"\n{label.upper()}: {prompt}\nCapture length: {seconds:g} seconds.")
        if input("Press Enter to begin, or q to finish: ").strip().lower() in {"q", "quit"}:
            return None
        try:
            audio, status = record_audio(seconds, microphone, label != "background")
        except sd.PortAudioError as exc:
            print(f"Microphone error: {exc}. Check the device before retrying.")
            continue
        details, rejection = audio_quality(audio, label, status)
        print(details)
        if rejection:
            print(f"Take cannot be kept: {rejection}")
        while True:
            choices = "[r]etry / [p]lay / [q]uit" if rejection else "[k]eep / [r]etry / [p]lay / [q]uit"
            choice = input(f"{choices}: ").strip().lower()
            if choice in {"q", "quit"}:
                return None
            if choice in {"r", "retry"}:
                break
            if choice in {"k", "keep"} and not rejection:
                return audio
            if choice in {"p", "play"}:
                try:
                    sd.play(audio, RATE, blocking=True)
                except sd.PortAudioError as exc:
                    print(f"Playback unavailable: {exc}")
                finally:
                    sd.stop()
            else:
                print("Choose one of the listed actions.")


def write_manifest(directory: Path, manifest: dict) -> None:
    temporary = directory / "session.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(directory / "session.json")


def save_take(directory: Path, manifest: dict, audio: np.ndarray, label: str, prompt: str) -> Path:
    number = 1 + sum(item["label"] == label for item in manifest["recordings"])
    path = directory / f"{label}-{number:03d}.wav"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary = path.with_suffix(".wav.tmp")
    sf.write(temporary, audio, RATE, format="WAV", subtype="PCM_16")
    temporary.replace(path)
    manifest["recordings"].append({
        "file": path.name, "label": label, "text": prompt,
        "duration_seconds": len(audio) / RATE,
    })
    write_manifest(directory, manifest)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--session", help="New session name; existing sessions cannot be overwritten")
    parser.add_argument("--mic", type=int, help="Input device index (see --list-mics)")
    parser.add_argument("--list-mics", action="store_true")
    parser.add_argument("--positives", type=int)
    parser.add_argument("--negatives", type=int)
    parser.add_argument("--background-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.list_mics:
        return args
    if not args.split:
        parser.error("--split is required when recording")
    args.positives = args.positives if args.positives is not None else (30 if args.split == "train" else 15)
    args.negatives = args.negatives if args.negatives is not None else (20 if args.split == "train" else 10)
    if args.positives < 0 or args.negatives < 0:
        parser.error("recording counts must be nonnegative")
    if not np.isfinite(args.background_seconds) or args.background_seconds < 0:
        parser.error("--background-seconds must be finite and nonnegative (0 skips background)")
    if args.positives + args.negatives == 0 and args.background_seconds == 0:
        parser.error("request at least one recording")
    if args.session and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", args.session):
        parser.error("--session must be 1–100 letters, digits, dots, underscores or hyphens, starting with a letter or digit")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_mics:
        default = sd.default.device[0]
        for index, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0:
                print(f"{index}: {device['name']}" + (" (default)" if index == default else ""))
        return 0
    microphone = args.mic if args.mic is not None else int(sd.default.device[0])
    device = sd.query_devices(microphone, "input")
    sd.check_input_settings(device=microphone, channels=1, dtype="float32", samplerate=RATE)
    created_at = datetime.now(timezone.utc)
    session_id = args.session or f"{created_at:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    directory = args.dataset.expanduser().resolve() / args.split / session_id
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1, "session_id": session_id, "split": args.split,
        "created_at": created_at.isoformat(),
        "microphone": {"index": microphone, "name": str(device["name"])},
        "recordings": [],
    }
    write_manifest(directory, manifest)
    print(f"Local recordings only. Microphone: {device['name']}\nSession: {directory}")
    print("Use the microphone and room where you use Gemma. Vary normal speaking speed,")
    print("loudness, and distance across takes. Say only the displayed phrase, after the cue.")
    print("Collect validation and test in separate sittings; never copy takes between splits.")
    prompts = [("positive", "Hey Gemma", CLIP_SECONDS)] * args.positives
    prompts += [("negative", NEGATIVE_PROMPTS[i % len(NEGATIVE_PROMPTS)], CLIP_SECONDS)
                for i in range(args.negatives)]
    if args.background_seconds > 0:
        prompts.append(("background", "Normal room noise; no 'Hey Gemma'", args.background_seconds))
    result = 0
    try:
        for index, (label, prompt, seconds) in enumerate(prompts, 1):
            print(f"\nTake {index}/{len(prompts)}")
            audio = review_take(label, prompt, seconds, microphone)
            if audio is None:
                break
            path = save_take(directory, manifest, audio, label, prompt)
            print(f"Saved {path.name}")
    except (KeyboardInterrupt, EOFError):
        print("\nRecording interrupted; previously kept takes remain saved.")
        result = 130
    finally:
        sd.stop()
    print(f"Kept {len(manifest['recordings'])} recordings. Manifest: {directory / 'session.json'}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
