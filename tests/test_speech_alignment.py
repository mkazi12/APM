import unittest

import numpy as np

from tools.speech_alignment import speech_bounds

RATE = 16000


def phrase(start=1.0, end=1.8, amplitude=1000, background=0):
    """Speech-like syllable envelopes above a low-frequency room-noise floor."""
    rng = np.random.default_rng(427)
    time = np.arange(RATE * 4) / RATE
    noise = background * (np.sin(2 * np.pi * 75 * time) + rng.normal(0, 0.12, len(time)))
    inside = (time >= start) & (time < end)
    phase = (time[inside] - start) / (end - start)
    envelope = np.sin(np.pi * phase) ** 0.5 * (0.65 + 0.35 * np.cos(6 * np.pi * phase))
    noise[inside] += amplitude * envelope * (
        np.sin(2 * np.pi * 500 * time[inside]) + 0.4 * np.sin(2 * np.pi * 1300 * time[inside]))
    return np.clip(noise, -32768, 32767).astype(np.int16)


class SpeechAlignmentTests(unittest.TestCase):
    def test_quiet_and_loud_phrases_survive_low_frequency_noise(self):
        for amplitude, background in ((200, 65), (1400, 65), (500, 4)):
            with self.subTest(amplitude=amplitude, background=background):
                audio = phrase(amplitude=amplitude, background=background)
                original = audio.copy()
                start, end = speech_bounds(audio)
                self.assertLessEqual(abs(start / RATE - 1.0), 0.18)
                self.assertLessEqual(abs(end / RATE - 1.8), 0.18)
                np.testing.assert_array_equal(audio, original)

    def test_delayed_phrase_is_located_without_fixed_time_window(self):
        start, end = speech_bounds(phrase(start=2.1, end=2.7, background=60))
        self.assertLess(abs(start / RATE - 2.1), 0.18)
        self.assertLess(abs(end / RATE - 2.7), 0.18)

    def test_silence_and_transient_only_are_rejected(self):
        silence = np.zeros(RATE * 4, dtype=np.int16)
        click = silence.copy()
        click[RATE] = 16000
        for audio in (silence, click):
            with self.subTest(kind="silence" if audio is silence else "click"), self.assertRaises(ValueError):
                speech_bounds(audio)

    def test_startup_transient_does_not_override_phrase(self):
        audio = phrase(background=60)
        audio[10] = 10000
        start, end = speech_bounds(audio)
        self.assertGreater(start / RATE, 0.8)
        self.assertLess(end / RATE, 2.0)

    def test_overlong_and_boundary_cut_speech_are_rejected(self):
        for start, end in ((0.0, 0.9), (3.2, 4.0), (0.5, 3.5)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                speech_bounds(phrase(start=start, end=end))

    def test_separate_substantial_phrases_are_rejected(self):
        audio = phrase(start=0.8, end=1.4) + phrase(start=2.2, end=2.8)
        with self.assertRaisesRegex(ValueError, "Multiple"):
            speech_bounds(audio)


if __name__ == "__main__":
    unittest.main()
