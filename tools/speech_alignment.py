"""Find one isolated wake phrase in a prompted 16 kHz recording.

Assumes silence around a single short phrase, as collected by record_wakeword.py.
This estimates acoustic boundaries, not the words spoken. A bandpass and adaptive
energy gates avoid WebRTC VAD startup detections and low-frequency room noise.
Filtering is for alignment only: the supplied PCM is never modified.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


def speech_bounds(audio: np.ndarray, rate: int = 16000) -> tuple[int, int]:
    """Return start/end sample offsets for an isolated phrase with quiet margins.

    Reject silence, short transients, multiple substantial phrases, overlong
    speech, and speech cut off at a recording boundary. The recorder's cue occurs
    after 0.5 seconds; noise estimation uses its leading and trailing quiet areas.
    """
    if (rate != 16000 or not isinstance(audio, np.ndarray) or audio.ndim != 1
            or audio.dtype != np.int16 or len(audio) < int(0.6 * rate)):
        raise ValueError("Alignment requires at least 0.6 seconds of mono 16 kHz int16 audio")
    frame_size = rate // 50
    filtered = sosfilt(butter(3, (300, 4000), btype="band", fs=rate, output="sos"),
                       audio.astype(np.float64))
    # Padding only the final analysis frame preserves alignment to original PCM.
    filtered = np.pad(filtered, (0, (-len(filtered)) % frame_size))
    rms = np.sqrt(np.mean(filtered.reshape(-1, frame_size) ** 2, axis=1))
    smooth = np.convolve(rms, np.ones(3) / 3, mode="same")
    noise = max(1.0, float(np.median(np.concatenate((smooth[10:25], smooth[-25:-5])))))
    high = max(2.5 * noise, 0.08 * float(smooth.max()), 3.0)
    low = max(1.5 * noise, 0.025 * float(smooth.max()), 2.0)
    active = np.flatnonzero(smooth > high)
    if not len(active):
        raise ValueError("No clear speech above the recording's background noise")
    # Bridge pauses within 'Hey Gemma', but keep separate utterances apart.
    groups = np.split(active, np.flatnonzero(np.diff(active) > 16) + 1)
    groups = [group for group in groups if len(group) >= 3]
    if not groups:
        raise ValueError("Only short transients found; no isolated speech phrase")

    def energy(group: np.ndarray) -> float:
        return float(np.maximum(smooth[group[0]:group[-1] + 1] ** 2 - noise ** 2, 0).sum())

    groups.sort(key=energy, reverse=True)
    strongest = groups[0]
    core_duration = (strongest[-1] - strongest[0] + 1) * frame_size / rate
    if core_duration < 0.2:
        raise ValueError("Only a short transient found; speech must last at least 0.2 seconds")
    for other in groups[1:]:
        duration = (other[-1] - other[0] + 1) * frame_size / rate
        if duration >= 0.2 and energy(other) > 0.25 * energy(strongest):
            raise ValueError("Multiple substantial sound groups; expected one isolated 'Hey Gemma'")
    start, end = int(strongest[0]), int(strongest[-1])
    # Use a lower gate for softer consonants; permit gaps up to 60 milliseconds.
    while start > 0 and np.any(smooth[max(0, start - 3):start] > low):
        start -= 1
    while end < len(smooth) - 1 and np.any(smooth[end + 1:end + 4] > low):
        end += 1
    start = max(0, start - 2) * frame_size
    end = min(len(audio), (end + 3) * frame_size)
    if start < int(0.2 * rate) or end > len(audio) - int(0.2 * rate):
        raise ValueError("Speech reaches a recording boundary; leave silence before and after the phrase")
    if (end - start) / rate > 2.5:
        raise ValueError("Speech is longer than 2.5 seconds; expected one isolated 'Hey Gemma'")
    return start, end
