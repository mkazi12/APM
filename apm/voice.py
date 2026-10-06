"""Wake-word gated, local audio conversation. Detection runs on CPU."""
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import json
import queue
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
from .terminal import Status

RATE = 16000
BLOCK = 1280  # 80 ms: openWakeWord's streaming step.
DEFAULT_WAKE_MODEL = Path(__file__).resolve().parent.parent / "models" / "hey_gemma.onnx"


@dataclass
class VoiceConfig:
    wake_model: Path = DEFAULT_WAKE_MODEL
    threshold: float | None = None
    silence_seconds: float = 0.8
    start_timeout: float = 5.0
    max_seconds: float = 25.0
    pre_roll_seconds: float = 1.6
    microphone: int | None = None
    speak: bool = True

    def validate(self):
        if self.threshold is not None and not 0 < self.threshold < 1:
            raise ValueError("Wake threshold must be between 0 and 1")
        if not 0.3 <= self.silence_seconds <= 3:
            raise ValueError("End-of-speech silence must be between 0.3 and 3 seconds")
        if min(self.start_timeout, self.max_seconds, self.pre_roll_seconds) <= 0:
            raise ValueError("Capture durations must be positive")
        if self.max_seconds + self.pre_roll_seconds > 30:
            raise ValueError("Capture including pre-roll must fit within 30 seconds")


class WakeDetector:
    def __init__(self, config):
        from openwakeword.model import Model
        self.config = config
        model = Path(config.wake_model).expanduser().resolve()
        required = [model, model.parent / "melspectrogram.onnx", model.parent / "embedding_model.onnx"]
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"Wake-word asset missing: {path}. See README voice setup.")
        self.threshold = config.threshold
        if self.threshold is None:
            metadata = model.with_suffix(".metadata.json")
            self.threshold = float(json.loads(metadata.read_text()).get("recommended_threshold", 0.7)) if metadata.exists() else 0.7
        if not 0 < self.threshold < 1:
            raise ValueError("Wake model threshold must be between 0 and 1")
        self.model = Model(wakeword_models=[str(model)], inference_framework="onnx",
                           melspec_model_path=str(required[1]), embedding_model_path=str(required[2]))
        self.name = model.stem
        self.hits = 0
        self.last_score = 0.0

    def reset(self):
        self.model.reset()
        self.hits = 0
        self.last_score = 0.0

    def detect(self, pcm):
        result = self.model.predict(pcm)
        self.last_score = float(result[self.name])
        self.hits = self.hits + 1 if self.last_score >= self.threshold else 0
        return self.hits >= 2


class SpeechDetector:
    def __init__(self):
        import webrtcvad
        self.vad = webrtcvad.Vad(2)

    def is_speech(self, pcm):
        # Four 20 ms frames; reject isolated clicks.
        votes = sum(self.vad.is_speech(pcm[i:i+320].astype("<i2").tobytes(), RATE)
                    for i in range(0, BLOCK, 320))
        return votes >= 2


class Capture:
    """Pure endpoint state: returns audio only after post-wake speech + silence."""
    def __init__(self, pre_roll, config):
        self.config = config
        self.frames = list(pre_roll)
        self.elapsed = 0.0
        self.speech_seconds = 0.0
        self.silence = 0.0
        self.has_speech = False

    def feed(self, pcm, speech):
        self.frames.append(pcm.copy())
        self.elapsed += BLOCK / RATE
        if speech:
            self.speech_seconds += BLOCK / RATE
            self.silence = 0.0
        else:
            self.silence += BLOCK / RATE
        self.has_speech = self.speech_seconds >= 0.24
        # Never execute a cut-off command: the user may be about to negate it.
        if self.elapsed >= self.config.max_seconds:
            return "too_long", None
        if not self.has_speech and self.elapsed >= self.config.start_timeout:
            return "no_speech", None
        if self.has_speech and self.silence >= self.config.silence_seconds:
            return "complete", np.concatenate(self.frames).astype(np.float32) / 32768.0
        return "listening", None


class Microphone:
    def __init__(self, device=None):
        self.device = device
        self.queue = queue.Queue(maxsize=25)
        self.enabled = threading.Event()
        self.problem = threading.Event()
        self.stream = None

    def callback(self, data, frames, info, status):
        if not self.enabled.is_set():
            return
        if status:
            self.problem.set()
        try:
            self.queue.put_nowait(data[:, 0].copy())
        except queue.Full:
            self.problem.set()

    def __enter__(self):
        import sounddevice as sd
        self.stream = sd.InputStream(samplerate=RATE, blocksize=BLOCK, device=self.device,
                                     channels=1, dtype="int16", callback=self.callback)
        try:
            self.stream.start()
        except BaseException:
            self.stream.close()
            raise
        self.resume()
        return self

    def pause(self):
        self.enabled.clear()

    def resume(self):
        self.enabled.clear()
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.problem.clear()
        self.enabled.set()

    def read(self):
        if self.problem.is_set():
            raise RuntimeError("Microphone audio dropped; current command discarded")
        try:
            data = self.queue.get(timeout=2)
        except queue.Empty as exc:
            raise RuntimeError("No microphone audio received. Check input selection and microphone permission.") from exc
        if self.problem.is_set() or len(data) != BLOCK:
            raise RuntimeError("Microphone audio interrupted; current command discarded")
        return data

    def __exit__(self, *_):
        self.pause()
        if self.stream:
            try:
                self.stream.stop()
            finally:
                self.stream.close()


class Speaker:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.command = None
        if enabled:
            if sys.platform == "darwin":
                self.command = ["/usr/bin/say", "-f", "-"]
            elif shutil.which("espeak-ng"):
                self.command = [shutil.which("espeak-ng"), "--stdin"]
            else:
                raise RuntimeError("Install espeak-ng for spoken replies on Linux, or use --no-speak")

    def speak(self, text):
        if not self.enabled or not text.strip():
            return
        process = subprocess.Popen(self.command, stdin=subprocess.PIPE, text=True)
        try:
            process.communicate(text)
            if process.returncode:
                raise RuntimeError("Speech playback failed")
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=3)
            raise


def wait_for_command(microphone, detector, speech, config):
    ring = deque(maxlen=round(config.pre_roll_seconds * RATE / BLOCK))
    detector.reset()
    with Status('Waiting for “Hey Gemma” · mic on'):
        while True:
            frame = microphone.read()
            ring.append(frame)
            if detector.detect(frame):
                break
    capture = Capture(ring, config)
    if sys.stdout.isatty():
        print("\a", end="", flush=True)
    with Status("Listening · speak now"):
        while True:
            frame = microphone.read()
            state, audio = capture.feed(frame, speech.is_speech(frame))
            if state != "listening":
                return state, audio


def voice_session(model, home, request, config=None, debug=False):
    config = config or VoiceConfig()
    config.validate()
    if hasattr(model, "capabilities") and "audio" not in model.capabilities:
        raise RuntimeError("The selected backend does not support native audio")
    with Status('Loading “Hey Gemma” detector'):
        detector = WakeDetector(config)
        speech = SpeechDetector()
        speaker = Speaker(config.speak)
    print('Voice mode · say “Hey Gemma”, then your request. Ctrl-C returns to text.')
    print('Experimental wake detector: false triggers and missed wakes are possible.')
    try:
        with Microphone(config.microphone) as microphone:
            while True:
                state, audio = wait_for_command(microphone, detector, speech, config)
                microphone.pause()
                if state == "complete":
                    try:
                        reply = request(model, home, audio=audio, debug=debug)
                    except Exception as exc:
                        print(f"Voice request failed: {exc}")
                    else:
                        if config.speak and reply:
                            try:
                                with Status("Speaking · mic paused"):
                                    speaker.speak(reply)
                            except Exception as exc:
                                print(f"Reply was completed, but speech playback failed: {exc}")
                elif state == "too_long":
                    print("Command was too long and was discarded. Please try a shorter request.")
                else:
                    print("No command heard. Waiting for the wake phrase again.")
                # Discard playback tail and any audio received during inference.
                time.sleep(0.35)
                detector.reset()
                microphone.resume()
    except KeyboardInterrupt:
        print("\nMicrophone stopped. Returning to text.")
