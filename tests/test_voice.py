import contextlib
import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from apm.voice import BLOCK, RATE, Capture, Microphone, VoiceConfig, voice_session

FRAME = np.full(BLOCK, 1000, dtype=np.int16)
SILENCE = np.zeros(BLOCK, dtype=np.int16)


class DummyMicrophone(Microphone):
    def __enter__(self):
        self.closed = False
        self.resume()
        return self

    def __exit__(self, *_):
        self.pause()
        self.closed = True


class VoiceTests(unittest.TestCase):
    def test_complete_clip_keeps_pre_roll_and_waits_for_silence(self):
        capture = Capture([FRAME], VoiceConfig(silence_seconds=0.32))
        for _ in range(4):
            state, audio = capture.feed(FRAME, True)
            self.assertEqual(state, 'listening')
        for _ in range(3):
            state, _ = capture.feed(SILENCE, False)
            self.assertEqual(state, 'listening')
        state, audio = capture.feed(SILENCE, False)
        self.assertEqual(state, 'complete')
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(len(audio), 9 * BLOCK)
        np.testing.assert_allclose(audio[:BLOCK], FRAME / 32768.0)

    def test_wake_without_command_is_not_submitted(self):
        capture = Capture([FRAME] * 20, VoiceConfig())
        for _ in range(70):
            state, audio = capture.feed(SILENCE, False)
            if state != 'listening':
                break
        self.assertEqual(state, 'no_speech')
        self.assertIsNone(audio)

    def test_overlong_command_is_discarded_not_truncated(self):
        capture = Capture([FRAME], VoiceConfig(max_seconds=0.4))
        for _ in range(5):
            state, audio = capture.feed(FRAME, True)
        self.assertEqual(state, 'too_long')
        self.assertIsNone(audio)

    def test_audio_received_while_paused_is_discarded(self):
        mic = Microphone()
        data = FRAME[:, None]
        mic.resume()
        mic.callback(data, BLOCK, None, False)
        mic.pause()
        mic.callback(data, BLOCK, None, False)
        self.assertEqual(mic.queue.qsize(), 1)
        mic.resume()
        self.assertTrue(mic.queue.empty())
        mic.callback(data, BLOCK, None, False)
        np.testing.assert_array_equal(mic.read(), FRAME)

    def test_dropped_audio_is_not_used_as_a_command(self):
        for overflow in (False, True):
            with self.subTest(overflow=overflow):
                mic = Microphone()
                mic.resume()
                for _ in range(26 if overflow else 1):
                    mic.callback(FRAME[:, None], BLOCK, None, not overflow)
                with self.assertRaisesRegex(RuntimeError, 'dropped'):
                    mic.read()

    def run_session(self, states, request):
        mic = DummyMicrophone()
        speaker = MagicMock()
        def speak(text):
            self.assertFalse(mic.enabled.is_set())
        speaker.speak.side_effect = speak
        def invoke(*args, **kwargs):
            self.assertFalse(mic.enabled.is_set())
            return request(*args, **kwargs)
        with patch('apm.voice.Microphone', return_value=mic), \
             patch('apm.voice.WakeDetector'), patch('apm.voice.SpeechDetector'), \
             patch('apm.voice.Speaker', return_value=speaker), \
             patch('apm.voice.wait_for_command', side_effect=states), \
             patch('apm.voice.time.sleep'), contextlib.redirect_stdout(io.StringIO()):
            voice_session(object(), object(), invoke)
        self.assertTrue(mic.closed)
        self.assertFalse(mic.enabled.is_set())
        return speaker

    def test_inference_and_speech_playback_do_not_listen(self):
        request = MagicMock(return_value='Kitchen lights: off (simulated).')
        speaker = self.run_session([('complete', FRAME / 32768.0), KeyboardInterrupt()], request)
        request.assert_called_once()
        speaker.speak.assert_called_once_with('Kitchen lights: off (simulated).')

    def test_empty_or_cut_off_commands_do_not_reach_model(self):
        request = MagicMock()
        speaker = self.run_session([('no_speech', None), ('too_long', None), KeyboardInterrupt()], request)
        request.assert_not_called()
        speaker.speak.assert_not_called()

    def test_failed_request_returns_to_wake_listening(self):
        request = MagicMock(side_effect=[RuntimeError('offline'), 'Hello'])
        speaker = self.run_session([('complete', FRAME), ('complete', FRAME), KeyboardInterrupt()], request)
        self.assertEqual(request.call_count, 2)
        speaker.speak.assert_called_once_with('Hello')
