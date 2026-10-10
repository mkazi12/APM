"""Wake-word gated, local audio conversation. Detection runs on CPU."""
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
import json
import math
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
DEFAULT_VOICE_SETTINGS = DEFAULT_WAKE_MODEL.parent.parent / "work" / "voice.json"


def resolve_wake_model(override=None, *, settings_path=None):
    """Use an explicit model, a saved local selection, or the bundled baseline."""
    if override is not None:
        return Path(override).expanduser().resolve()
    settings = Path(settings_path) if settings_path is not None else DEFAULT_VOICE_SETTINGS
    try:
        saved = json.loads(settings.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return DEFAULT_WAKE_MODEL.resolve()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read voice settings {settings}; fix the file or use --wake-model.") from exc
    if (not isinstance(saved, dict) or set(saved) != {"wake_model"}
            or not isinstance(saved["wake_model"], str) or not saved["wake_model"].strip()):
        raise ValueError(f"Invalid voice settings {settings}; expected a wake_model path or use --wake-model.")
    model = Path(saved["wake_model"]).expanduser()
    if not model.is_absolute():
        model = settings.parent / model
    return model.resolve()


@dataclass
class VoiceConfig:
    wake_model: Path | None = None
    threshold: float | None = None
    silence_seconds: float = 0.8
    start_timeout: float = 5.0
    max_seconds: float = 25.0
    pre_roll_seconds: float = 1.6
    follow_up_timeout: float = 8.0
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
        if (isinstance(self.follow_up_timeout, bool)
                or not math.isfinite(self.follow_up_timeout)
                or not 0 <= self.follow_up_timeout <= 15):
            raise ValueError("Follow-up timeout must be between 0 and 15 seconds; 0 disables it")


class WakeDetector:
    def __init__(self, config):
        from openwakeword.model import Model
        self.config = config
        model = resolve_wake_model(config.wake_model)
        self.model_path = model
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
        self.reset()

    def reset(self):
        import webrtcvad
        # Mode 2 discards quiet commands that the wake-word model can still hear.
        # Capture requires a sustained onset to reject isolated noise votes.
        self.vad = webrtcvad.Vad(1)

    def is_speech(self, pcm):
        # Four 20 ms frames; reject isolated clicks.
        votes = sum(self.vad.is_speech(pcm[i:i+320].astype("<i2").tobytes(), RATE)
                    for i in range(0, BLOCK, 320))
        return votes >= 2


class Capture:
    """Pure endpoint state: returns audio only after post-wake speech + silence."""
    def __init__(self, pre_roll, config, *, onset_seconds=0.24):
        self.config = config
        self.onset_seconds = onset_seconds
        self.frames = list(pre_roll)
        self.pre_roll_frames = len(self.frames)
        self.elapsed = 0.0
        self.speech_seconds = 0.0
        self.speech_run = 0.0
        self.longest_speech_run = 0.0
        self.silence = 0.0
        self.has_speech = False

    def feed(self, pcm, speech):
        self.frames.append(pcm.copy())
        self.elapsed += BLOCK / RATE
        if speech:
            self.speech_seconds += BLOCK / RATE
            self.speech_run += BLOCK / RATE
            self.longest_speech_run = max(self.longest_speech_run, self.speech_run)
            self.silence = 0.0
        else:
            self.speech_run = 0.0
            self.silence += BLOCK / RATE
        # Separate clicks/noise bursts must not add up to a spoken command.
        # Once speech has started, allow pauses inside the command as before.
        self.has_speech = self.has_speech or self.speech_run >= self.onset_seconds
        # Never execute a cut-off command: the user may be about to negate it.
        if self.elapsed >= self.config.max_seconds:
            return "too_long", None
        if not self.has_speech and self.elapsed >= self.config.start_timeout:
            return "no_speech", None
        if self.has_speech and self.silence >= self.config.silence_seconds:
            return "complete", np.concatenate(self.frames).astype(np.float32) / 32768.0
        return "listening", None

    def diagnostics(self):
        """Summarize capture levels without saving or exposing microphone audio."""
        frames = self.frames[self.pre_roll_frames:]
        samples = np.concatenate(frames).astype(np.float64) / 32768 if frames else np.zeros(1)
        return {"capture_seconds": round(self.elapsed, 2),
                "speech_seconds": round(self.speech_seconds, 2),
                "longest_speech_run_seconds": round(self.longest_speech_run, 2),
                "input_peak": round(float(np.max(np.abs(samples))), 5),
                "input_rms": round(float(np.sqrt(np.mean(samples ** 2))), 5)}


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


def _wait_for_quiet(microphone, speech):
    # A missing prefix could contain "don't". Discard speech overlapping a
    # transition, then require a completely new onset after the listening cue.
    quiet_frames = 0
    for _ in range(round(2 * RATE / BLOCK)):
        frame = microphone.read()
        quiet_frames = 0 if speech.is_speech(frame) else quiet_frames + 1
        if quiet_frames >= 3:
            return
    raise RuntimeError("No quiet boundary; wait for the listening cue before speaking")


_PAUSE_FAILURE_MESSAGES = {
    "disconnected": (
        "The music player disconnected, so APM cannot confirm it is quiet. "
        "Pause any old player tabs and reconnect the player you want to use."),
    "unsupported": (
        "The music provider cannot confirm pause. "
        "Use a player that supports pause before starting voice mode."),
    "authorization_lost": (
        "Apple Music authorization ended and the player could not confirm it stopped. "
        "Open its player tab and reconnect Apple Music."),
    "timeout": (
        "The music player did not confirm pause in time. Bring its browser tab forward "
        "and check its playback diagnostics."),
    "player_update_required": (
        "The Apple Music player tab is outdated. Refresh the browser page (Cmd-R on Mac), "
        "then reconnect Apple Music. The 'Reload setup' button does not refresh the page; "
        "restarting APM alone does not update it."),
    "bridge_authorization_failed": (
        "APM's connection to the local music server was rejected. Restart APM to load the current "
        "music connection, then reconnect the player."),
    "bridge_update_required": (
        "The music server does not support pause. Restart apm-music with the updated code, "
        "then refresh the player tab and reconnect Apple Music."),
    "bridge_unreachable": (
        "The local music server could not be reached. Start or check apm-music, "
        "then open its player page and reconnect Apple Music."),
}


class MusicPauseFailure(RuntimeError):
    """A pause failure with a fixed, token-free recovery message for voice."""

    def __init__(self, reason):
        self.reason = reason if isinstance(reason, str) and reason in _PAUSE_FAILURE_MESSAGES else None
        message = _PAUSE_FAILURE_MESSAGES.get(self.reason, "Music pause could not be confirmed.")
        super().__init__(message + " Microphone stopped; returning to text. "
                         "Fix the player connection, then type /voice.")


def _pause_for_capture(microphone, speech, pause_music):
    """Return whether playback audio must be dropped, or report a pause failure."""
    if pause_music is None:
        return False
    try:
        outcome = pause_music()
    except Exception:
        outcome = {"status": "unknown"}
    if not isinstance(outcome, dict):
        return None
    if outcome.get("status") == "unavailable" and outcome.get("reason") == "not_configured":
        # No configured provider is safe. A lost connection or unsupported
        # pause says nothing about audio still playing in a browser tab.
        return False
    if outcome.get("status") != "paused" or outcome.get("playing") is not False:
        reason = outcome.get("reason")
        if isinstance(reason, str) and reason in _PAUSE_FAILURE_MESSAGES:
            # Remote messages may include credentials or arbitrary response
            # text. Only a known reason selects our own recovery instructions.
            raise MusicPauseFailure(reason)
        return None
    if outcome.get("was_playing") is False:
        return False
    # Keep lyrics, the room's playback tail, and input queued during pause out
    # of Gemma's command. The listening cue follows this transition.
    microphone.pause()
    time.sleep(0.15)
    speech.reset()
    microphone.resume()
    _wait_for_quiet(microphone, speech)
    return True


def _capture_command(microphone, speech, config, pre_roll=(), *, follow_up=False, debug=False):
    # Follow-ups commonly consist of a short "yes" or "no". Two consecutive
    # speech frames admit those while still rejecting an isolated noise frame.
    capture = Capture(pre_roll, config, onset_seconds=0.16 if follow_up else 0.24)
    if sys.stdout.isatty():
        print("\a", end="", flush=True)
    label = "Listening for your reply · no wake phrase needed" if follow_up else "Listening · speak now"
    with Status(label):
        while True:
            frame = microphone.read()
            state, audio = capture.feed(frame, speech.is_speech(frame))
            if state != "listening":
                break
    if debug:
        print("Voice capture: " + json.dumps({"state": state, "follow_up": follow_up, **capture.diagnostics()}))
    return state, audio


def wait_for_command(microphone, detector, speech, config, debug=False, *, pause_music=None):
    ring = deque(maxlen=round(config.pre_roll_seconds * RATE / BLOCK))
    detector.reset()
    with Status('Waiting for “Hey Gemma” · mic on'):
        while True:
            frame = microphone.read()
            # Let the speech detector adapt to the current room before the wake
            # triggers. A cold VAD can mistake startup noise for command speech.
            speech.is_speech(frame)
            ring.append(frame)
            if detector.detect(frame):
                break
    quieted = _pause_for_capture(microphone, speech, pause_music)
    if quieted is None:
        return "pause_failed", None
    if quieted:
        ring.clear()
    return _capture_command(microphone, speech, config, ring, debug=debug)


def wait_for_reply(microphone, speech, config, debug=False, *, pause_music=None):
    if config.follow_up_timeout <= 0:
        return "no_speech", None
    quieted = _pause_for_capture(microphone, speech, pause_music)
    if quieted is None:
        return "pause_failed", None
    if not quieted:
        # Mic input was disabled during inference/TTS. Do not execute the
        # suffix of an answer begun before it resumed, even without music.
        speech.reset()
        _wait_for_quiet(microphone, speech)
    reply_config = replace(config, start_timeout=config.follow_up_timeout)
    return _capture_command(microphone, speech, reply_config, follow_up=True, debug=debug)


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
    print('During music, wait for it to pause and the listening cue before speaking. Questions allow a reply without another wake phrase.')
    print(f"Wake model · {detector.model_path} · threshold {detector.threshold}")
    print('Experimental wake detector: false triggers and missed wakes are possible.')
    from .conversation import reply_needs_answer
    pause_music = getattr(getattr(home, "music", None), "pause", None)
    pause_music = pause_music if callable(pause_music) else None
    follow_up = False
    capture_failures = 0
    try:
        with Microphone(config.microphone) as microphone:
            while True:
                try:
                    if follow_up:
                        state, audio = wait_for_reply(microphone, speech, config, debug=debug, pause_music=pause_music)
                    else:
                        state, audio = wait_for_command(microphone, detector, speech, config,
                                                        debug=debug, pause_music=pause_music)
                except MusicPauseFailure as exc:
                    microphone.pause()
                    print(str(exc))
                    return
                except (RuntimeError, ValueError, OSError) as exc:
                    microphone.pause()
                    follow_up = False
                    capture_failures += 1
                    if capture_failures >= 3:
                        raise RuntimeError("Microphone capture failed repeatedly; check the input device and restart /voice") from exc
                    print(f"Voice capture interrupted; this turn was discarded: {exc}. Say Hey Gemma again.")
                    time.sleep(0.35)
                    detector.reset()
                    speech.reset()
                    microphone.resume()
                    continue
                capture_failures = 0
                was_follow_up, follow_up = follow_up, False
                microphone.pause()
                if state == "complete":
                    try:
                        reply = request(model, home, audio=audio, debug=debug)
                    except Exception as exc:
                        print(f"Voice request failed: {exc}")
                    else:
                        follow_up = bool(config.follow_up_timeout and
                                         (reply.expects_reply if hasattr(reply, "expects_reply")
                                          else reply_needs_answer(reply)))
                        if config.speak and reply:
                            try:
                                with Status("Speaking · mic paused"):
                                    speaker.speak(reply)
                            except Exception as exc:
                                print(f"Reply was completed, but speech playback failed: {exc}")
                elif state == "too_long":
                    print("Command was too long and was discarded. Please try a shorter request.")
                elif state == "pause_failed":
                    print("Music pause could not be confirmed. Microphone stopped; returning to text. "
                          "Pause the player, reload its page and reconnect Apple Music, then type /voice.")
                    # Do not re-arm on lyrics or another wake while the same
                    # broken player still cannot confirm silence. Exiting the
                    # microphone context closes capture and preserves chat.
                    return
                elif was_follow_up:
                    print("No reply heard. Waiting for the wake phrase again.")
                else:
                    print("No command heard. Waiting for the wake phrase again.")
                # Discard playback tail and any audio received during inference.
                time.sleep(0.35)
                detector.reset()
                microphone.resume()
    except KeyboardInterrupt:
        print("\nVoice mode interrupted (Ctrl-C/SIGINT). Microphone stopped; type /voice to listen again.")
