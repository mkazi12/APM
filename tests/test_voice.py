import contextlib
import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from apm.conversation import AssistantReply
from apm.voice import BLOCK, RATE, Capture, Microphone, MusicPauseFailure, SpeechDetector, VoiceConfig, voice_session, wait_for_command, wait_for_reply

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

    def test_isolated_noise_votes_do_not_accumulate_into_a_command(self):
        capture = Capture([], VoiceConfig(start_timeout=1))
        for i in range(13):
            state, audio = capture.feed(FRAME, i in (0, 1, 5, 6))
        self.assertGreater(capture.speech_seconds, 0.24)
        self.assertEqual(state, 'no_speech')
        self.assertIsNone(audio)

    def test_quiet_voiced_signal_is_captured_without_changing_audio_level(self):
        # A quiet harmonic syllable, around 100 PCM units peak. The old mode 2
        # rejects this whole signal; it models the quiet-command regression.
        t = np.arange(RATE) / RATE
        envelope = np.where(t < 0.5, np.sin(np.pi * np.minimum(2 * t, 1)) ** 2, 0)
        voiced = (80 * (np.sin(2 * np.pi * 130 * t) + 0.4 * np.sin(2 * np.pi * 260 * t)
                       + 0.25 * np.sin(2 * np.pi * 520 * t)) * envelope).astype(np.int16)
        stream = np.concatenate([np.zeros(RATE, dtype=np.int16), voiced,
                                 np.zeros(2 * RATE, dtype=np.int16)])
        speech = SpeechDetector()
        capture = Capture([], VoiceConfig())
        for start in range(0, len(stream), BLOCK):
            frame = stream[start:start + BLOCK]
            state, audio = capture.feed(frame, speech.is_speech(frame))
            if state != 'listening':
                break
        self.assertEqual(state, 'complete')
        np.testing.assert_array_equal(audio * 32768, stream[:len(audio)])

    def test_waiting_audio_primes_speech_detector_and_debug_excludes_pre_roll(self):
        microphone = MagicMock()
        microphone.read.side_effect = [FRAME * 10] * 2 + [SILENCE] * 13
        detector = MagicMock()
        detector.detect.side_effect = [False, True]
        speech = MagicMock()
        speech.is_speech.return_value = False
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            state, audio = wait_for_command(microphone, detector, speech,
                                           VoiceConfig(start_timeout=1), debug=True)
        self.assertEqual(state, 'no_speech')
        self.assertIsNone(audio)
        self.assertEqual(speech.is_speech.call_count, 15)
        self.assertIn('"input_peak": 0.0', out.getvalue())
        self.assertIn('"speech_seconds": 0.0', out.getvalue())

    def test_overlong_command_is_discarded_not_truncated(self):
        capture = Capture([FRAME], VoiceConfig(max_seconds=0.4))
        for _ in range(5):
            state, audio = capture.feed(FRAME, True)
        self.assertEqual(state, 'too_long')
        self.assertIsNone(audio)

    def test_wake_pauses_music_before_capture_and_discards_music_preroll(self):
        microphone, detector, speech = MagicMock(), MagicMock(), MagicMock()
        microphone.read.side_effect = [FRAME * 10] * 2 + [SILENCE] * 3 + [FRAME] * 4 + [SILENCE] * 4
        detector.detect.side_effect = [False, True]
        speech.is_speech.side_effect = [True] * 2 + [False] * 3 + [True] * 4 + [False] * 4
        def pause():
            self.assertEqual(microphone.read.call_count, 2)
            return {"status": "paused", "playing": False, "was_playing": True}
        with patch('apm.voice.time.sleep') as settle:
            state, audio = wait_for_command(microphone, detector, speech,
                                           VoiceConfig(silence_seconds=0.32), pause_music=pause)
        self.assertEqual(state, 'complete')
        self.assertEqual(len(audio), 8 * BLOCK)
        self.assertLessEqual(np.max(audio), 1000 / 32768)
        microphone.pause.assert_called_once()
        microphone.resume.assert_called_once()
        speech.reset.assert_called_once()
        settle.assert_called_once_with(0.15)

    def test_speech_straddling_music_pause_is_not_submitted_as_a_partial_command(self):
        microphone, detector, speech = MagicMock(), MagicMock(), MagicMock()
        # A wake, the tail of an overlapping utterance, a quiet boundary, then
        # a fresh complete request. Only the fresh request reaches the model.
        microphone.read.side_effect = [FRAME * 10] + [FRAME * 2] * 4 + [SILENCE] * 3 + [FRAME] * 4 + [SILENCE] * 4
        detector.detect.return_value = True
        speech.is_speech.side_effect = [True] * 5 + [False] * 3 + [True] * 4 + [False] * 4
        pause = lambda: {"status": "paused", "playing": False, "was_playing": True}
        with patch('apm.voice.time.sleep'):
            state, audio = wait_for_command(microphone, detector, speech,
                                           VoiceConfig(silence_seconds=0.32), pause_music=pause)
        self.assertEqual(state, 'complete')
        self.assertEqual(len(audio), 8 * BLOCK)
        self.assertLessEqual(np.max(audio), 1000 / 32768)

    def test_already_quiet_player_preserves_immediate_command_preroll(self):
        microphone, detector, speech = MagicMock(), MagicMock(), MagicMock()
        microphone.read.side_effect = [FRAME] * 5 + [SILENCE] * 4
        detector.detect.return_value = True
        speech.is_speech.side_effect = [True] * 5 + [False] * 4
        pause = MagicMock(return_value={"status": "paused", "playing": False, "was_playing": False})
        state, audio = wait_for_command(microphone, detector, speech,
                                       VoiceConfig(silence_seconds=0.32), pause_music=pause)
        self.assertEqual(state, 'complete')
        self.assertEqual(len(audio), 9 * BLOCK)
        microphone.pause.assert_not_called()
        microphone.resume.assert_not_called()
        speech.reset.assert_not_called()

    def test_unconfirmed_pause_does_not_submit_music_as_a_command(self):
        for outcome in ({"status": "unknown"}, {"status": "paused", "playing": True}, None):
            with self.subTest(outcome=outcome):
                microphone, detector, speech = MagicMock(), MagicMock(), MagicMock()
                microphone.read.return_value = FRAME
                detector.detect.return_value = True
                state, audio = wait_for_command(microphone, detector, speech, VoiceConfig(),
                                                pause_music=lambda: outcome)
                self.assertEqual((state, audio), ('pause_failed', None))
                self.assertEqual(microphone.read.call_count, 1)

    def test_known_pause_failures_propagate_fixed_recovery_without_capturing(self):
        recovery = {
            'player_update_required': 'player tab is outdated',
            'bridge_authorization_failed': 'connection to the local music server was rejected',
            'bridge_update_required': 'music server does not support pause',
            'bridge_unreachable': 'local music server could not be reached',
        }
        for follow_up in (False, True):
            for reason, expected in recovery.items():
                with self.subTest(follow_up=follow_up, reason=reason):
                    microphone, detector, speech = MagicMock(), MagicMock(), MagicMock()
                    microphone.read.return_value = FRAME
                    detector.detect.return_value = True
                    pause = MagicMock(return_value={'status':'unknown', 'reason':reason,
                        'message':'Bearer secret-response-token http://example.invalid/private'})
                    with self.assertRaises(MusicPauseFailure) as caught:
                        if follow_up:
                            wait_for_reply(microphone, speech, VoiceConfig(), pause_music=pause)
                        else:
                            wait_for_command(microphone, detector, speech, VoiceConfig(), pause_music=pause)
                    self.assertIn(expected, str(caught.exception))
                    self.assertNotIn('secret-response-token', str(caught.exception))
                    self.assertNotIn('example.invalid', str(caught.exception))
                    self.assertEqual(microphone.read.call_count, 0 if follow_up else 1)
                    pause.assert_called_once()

    def test_unrecognized_pause_details_and_exceptions_remain_generic(self):
        for result in ({'status':'unknown', 'reason':'secret-response-token', 'message':'secret-message'},
                       {'status':'unknown', 'reason':['secret-response-token']},
                       RuntimeError('secret-exception-token')):
            with self.subTest(result=type(result).__name__):
                microphone, detector, speech = MagicMock(), MagicMock(), MagicMock()
                microphone.read.return_value = FRAME
                detector.detect.return_value = True
                pause = MagicMock(side_effect=result) if isinstance(result, Exception) else MagicMock(return_value=result)
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    response = wait_for_command(microphone, detector, speech, VoiceConfig(), pause_music=pause)
                self.assertEqual(response, ('pause_failed', None))
                self.assertNotIn('secret', output.getvalue())

    def test_followup_captures_speech_without_a_wake_and_times_out_on_silence(self):
        microphone, speech = MagicMock(), MagicMock()
        microphone.read.side_effect = [SILENCE] * 3 + [FRAME] * 4 + [SILENCE] * 4
        speech.is_speech.side_effect = [False] * 3 + [True] * 4 + [False] * 4
        config = VoiceConfig(silence_seconds=0.32, follow_up_timeout=0.32)
        state, audio = wait_for_reply(microphone, speech, config)
        self.assertEqual(state, 'complete')
        self.assertEqual(len(audio), 8 * BLOCK)
        microphone.read.side_effect = [SILENCE] * 7
        speech.is_speech.side_effect = [False] * 7
        self.assertEqual(wait_for_reply(microphone, speech, config), ('no_speech', None))

    def test_followup_configuration_is_bounded(self):
        for value in (-1, 16, float('nan'), float('inf'), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                VoiceConfig(follow_up_timeout=value).validate()
        VoiceConfig(follow_up_timeout=0).validate()

    def test_short_followup_yes_is_accepted_but_one_noise_frame_is_not(self):
        for speech_frames, expected in ((2, 'complete'), (1, 'no_speech')):
            with self.subTest(speech_frames=speech_frames):
                microphone, speech = MagicMock(), MagicMock()
                microphone.read.side_effect = [SILENCE] * 3 + [FRAME] * speech_frames + [SILENCE] * 6
                speech.is_speech.side_effect = [False] * 3 + [True] * speech_frames + [False] * 6
                state, audio = wait_for_reply(microphone, speech,
                    VoiceConfig(silence_seconds=0.32, follow_up_timeout=0.4))
                self.assertEqual(state, expected)
                self.assertEqual(audio is None, expected == 'no_speech')

    def test_followup_discards_speech_begun_before_microphone_reopened(self):
        for pause_music in (None, lambda: {"status": "paused", "playing": False, "was_playing": False}):
            with self.subTest(pause_music=pause_music):
                microphone, speech = MagicMock(), MagicMock()
                microphone.read.side_effect = [FRAME * 2] * 4 + [SILENCE] * 3 + [FRAME] * 2 + [SILENCE] * 4
                speech.is_speech.side_effect = [True] * 4 + [False] * 3 + [True] * 2 + [False] * 4
                state, audio = wait_for_reply(microphone, speech,
                    VoiceConfig(silence_seconds=0.32), pause_music=pause_music)
                self.assertEqual(state, 'complete')
                self.assertEqual(len(audio), 6 * BLOCK)
                self.assertLessEqual(np.max(audio), 1000 / 32768)

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

    def run_session(self, states, request, *, followups=(), config=None):
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
             patch('apm.voice.wait_for_command', side_effect=states) as wakes, \
             patch('apm.voice.wait_for_reply', side_effect=followups) as replies, \
             patch('apm.voice.time.sleep'), contextlib.redirect_stdout(io.StringIO()):
            voice_session(object(), object(), invoke, config=config)
        self.assertTrue(mic.closed)
        self.assertFalse(mic.enabled.is_set())
        return speaker, wakes, replies

    def test_inference_and_speech_playback_do_not_listen(self):
        request = MagicMock(return_value='Kitchen lights: off (simulated).')
        speaker, _, replies = self.run_session([('complete', FRAME / 32768.0), KeyboardInterrupt()], request)
        request.assert_called_once()
        speaker.speak.assert_called_once_with('Kitchen lights: off (simulated).')
        replies.assert_not_called()

    def test_empty_or_cut_off_commands_do_not_reach_model(self):
        request = MagicMock()
        speaker, _, _ = self.run_session([('no_speech', None), ('too_long', None), KeyboardInterrupt()], request)
        request.assert_not_called()
        speaker.speak.assert_not_called()

    def test_failed_request_returns_to_wake_listening(self):
        request = MagicMock(side_effect=[RuntimeError('offline'), 'Hello'])
        speaker, _, _ = self.run_session([('complete', FRAME), ('complete', FRAME), KeyboardInterrupt()], request)
        self.assertEqual(request.call_count, 2)
        speaker.speak.assert_called_once_with('Hello')

    def test_questions_accept_consecutive_answers_then_return_to_wake(self):
        request = MagicMock(side_effect=['How can I help for you today', 'Which one exactly?', 'Done.'])
        speaker, wakes, replies = self.run_session(
            [('complete', FRAME), KeyboardInterrupt()], request,
            followups=[('complete', FRAME), ('complete', FRAME)])
        self.assertEqual(request.call_count, 3)
        self.assertEqual(speaker.speak.call_count, 3)
        self.assertEqual(wakes.call_count, 2)
        self.assertEqual(replies.call_count, 2)

    def test_followup_timeout_returns_to_wake_without_inference(self):
        request = MagicMock(return_value='Which room?')
        _, wakes, replies = self.run_session([('complete', FRAME), KeyboardInterrupt()], request,
                                             followups=[('no_speech', None)])
        request.assert_called_once()
        self.assertEqual(wakes.call_count, 2)
        replies.assert_called_once()

    def test_unconfirmed_music_pause_stops_voice_once_without_rearming(self):
        request = MagicMock()
        with patch('builtins.print') as printed:
            speaker, wakes, replies = self.run_session(
                [('pause_failed', None), ('complete', FRAME)], request)
        request.assert_not_called()
        speaker.speak.assert_not_called()
        wakes.assert_called_once()
        replies.assert_not_called()
        notices = [args[0] for args, _ in printed.call_args_list
                   if args and isinstance(args[0], str) and 'Music pause could not be confirmed' in args[0]]
        self.assertEqual(len(notices), 1)
        self.assertIn('Microphone stopped; returning to text', notices[0])
        self.assertIn('then type /voice', notices[0])

    def test_followup_pause_failure_closes_capture_without_another_request(self):
        request = MagicMock(return_value='Which room?')
        speaker, wakes, replies = self.run_session([('complete', FRAME), ('complete', FRAME)], request,
                                                 followups=[('pause_failed', None), ('complete', FRAME)])
        request.assert_called_once()
        speaker.speak.assert_called_once_with('Which room?')
        wakes.assert_called_once()
        replies.assert_called_once()

    def test_specific_pause_failure_stops_once_and_preserves_completed_question(self):
        for follow_up in (False, True):
            with self.subTest(follow_up=follow_up), patch('builtins.print') as printed:
                request = MagicMock(return_value='Which room?')
                failure = MusicPauseFailure('player_update_required')
                speaker, wakes, replies = self.run_session(
                    [('complete', FRAME), ('complete', FRAME)] if follow_up else [failure, ('complete', FRAME)],
                    request, followups=[failure, ('complete', FRAME)] if follow_up else [])
                self.assertEqual(request.call_count, 1 if follow_up else 0)
                self.assertEqual(speaker.speak.call_count, 1 if follow_up else 0)
                wakes.assert_called_once()
                self.assertEqual(replies.call_count, 1 if follow_up else 0)
                notices = [args[0] for args, _ in printed.call_args_list
                           if args and isinstance(args[0], str) and 'Microphone stopped' in args[0]]
                self.assertEqual(len(notices), 1)
                self.assertIn('Refresh the browser page (Cmd-R on Mac)', notices[0])
                self.assertIn("'Reload setup' button does not refresh the page", notices[0])
                self.assertIn('restarting APM alone does not update it', notices[0])
                self.assertIn('then type /voice', notices[0])

    def test_playback_acknowledgement_cannot_open_a_followup_from_song_title(self):
        request = MagicMock(return_value=AssistantReply('Playing Who Are You? by The Who.', expects_reply=False))
        _, _, replies = self.run_session([('complete', FRAME), KeyboardInterrupt()], request)
        replies.assert_not_called()

    def test_disabled_followups_require_a_new_wake(self):
        request = MagicMock(return_value='Which room?')
        _, _, replies = self.run_session([('complete', FRAME), KeyboardInterrupt()], request,
                                         config=VoiceConfig(follow_up_timeout=0))
        replies.assert_not_called()

    def test_dropped_followup_is_discarded_and_returns_to_wake(self):
        request = MagicMock(return_value='Which room?')
        _, wakes, replies = self.run_session([('complete', FRAME), KeyboardInterrupt()], request,
                                             followups=[RuntimeError('Microphone audio dropped')])
        request.assert_called_once()
        self.assertEqual(wakes.call_count, 2)
        replies.assert_called_once()

    def test_permanently_failed_microphone_does_not_retry_forever(self):
        request = MagicMock()
        with self.assertRaisesRegex(RuntimeError, 'failed repeatedly'):
            self.run_session([RuntimeError('No audio')] * 3, request)
        request.assert_not_called()
