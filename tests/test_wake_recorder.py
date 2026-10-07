import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from tools import record_wakeword as recorder


def speech():
    audio = np.zeros(recorder.RATE * 4, dtype=np.float32)
    audio[8000:24000] = 0.2 * np.sin(np.arange(16000) * 0.2)
    return audio


class WakeRecorderTests(unittest.TestCase):
    def test_rejects_silence_clipping_and_dropped_buffers(self):
        self.assertIsNotNone(recorder.audio_quality(np.zeros(100), "positive")[1])
        self.assertIsNone(recorder.audio_quality(np.zeros(100), "background")[1])
        clipped = speech()
        clipped[12000] = 1
        self.assertIn("Clipped", recorder.audio_quality(clipped, "negative")[1])
        self.assertIn("Dropped", recorder.audio_quality(speech(), "positive", "input overflow")[1])
        self.assertIsNone(recorder.audio_quality(speech(), "positive")[1])

    def test_invalid_take_cannot_be_kept_and_retry_returns_new_audio(self):
        good = speech()
        with patch.object(recorder, "record_audio", side_effect=[(np.zeros(64000), ""), (good, "")]) as capture, \
             patch("builtins.input", side_effect=["", "k", "r", "", "k"]), \
             contextlib.redirect_stdout(io.StringIO()):
            kept = recorder.review_take("positive", "Hey Gemma", 4, 2)
        self.assertIs(kept, good)
        self.assertEqual(capture.call_count, 2)

    def test_faint_speech_can_be_reviewed_and_kept_at_original_level(self):
        quiet = speech() * 0.001
        details, rejection = recorder.audio_quality(quiet, "positive")
        self.assertIsNone(rejection)
        self.assertIn("Very quiet", details)
        with patch.object(recorder, "record_audio", return_value=(quiet, "")), \
             patch.object(recorder.sd, "play") as play, patch.object(recorder.sd, "stop"), \
             patch("builtins.input", side_effect=["", "p", "k"]), \
             contextlib.redirect_stdout(io.StringIO()):
            kept = recorder.review_take("positive", "Hey Gemma", 4, 2)
        self.assertIs(kept, quiet)
        np.testing.assert_array_equal(play.call_args.args[0], quiet)
        self.assertIsNotNone(recorder.audio_quality(quiet * 0.01, "positive")[1])
        self.assertIsNotNone(recorder.audio_quality(quiet, "positive", "input overflow")[1])

    def test_recording_status_is_retained_and_stream_stopped(self):
        with patch.object(recorder.sd, "rec", return_value=speech()[:, None]) as record, \
             patch.object(recorder.sd, "wait", return_value="input overflow"), \
             patch.object(recorder.sd, "stop") as stop, \
             patch.object(recorder.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            audio, status = recorder.record_audio(4, 2, True)
        record.assert_called_once_with(64000, samplerate=16000, channels=1, dtype="float32", device=2)
        self.assertEqual(audio.shape, (64000,))
        self.assertEqual(status, "input overflow")
        stop.assert_called_once()

    def test_saved_audio_and_manifest_are_consistent_pcm16(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = {"recordings": []}
            path = recorder.save_take(directory, manifest, speech(), "positive", "Hey Gemma")
            disk_manifest = json.loads((directory / "session.json").read_text())
            self.assertEqual(disk_manifest["recordings"], [{
                "file": "positive-001.wav", "label": "positive", "text": "Hey Gemma",
                "duration_seconds": 4.0,
            }])
            info = sf.info(path)
            self.assertEqual((info.samplerate, info.channels, info.subtype), (16000, 1, "PCM_16"))
            self.assertFalse(list(directory.glob("*.tmp")))
            with self.assertRaises(FileExistsError):
                recorder.save_take(directory, {"recordings": []}, speech(), "positive", "Hey Gemma")

    def test_completed_take_survives_interruption_and_session_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(recorder.sd, "query_devices", return_value={"name": "Test mic"}), \
             patch.object(recorder.sd, "check_input_settings"), \
             patch.object(recorder.sd, "stop"), \
             patch.object(recorder, "review_take", side_effect=[speech(), KeyboardInterrupt()]), \
             contextlib.redirect_stdout(io.StringIO()):
            args = ["--dataset", temporary, "--split", "train", "--session", "sitting-one", "--mic", "2",
                    "--positives", "2", "--negatives", "0", "--background-seconds", "0"]
            self.assertEqual(recorder.main(args), 130)
            path = Path(temporary) / "train" / "sitting-one" / "session.json"
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["split"], "train")
            self.assertEqual(manifest["session_id"], "sitting-one")
            self.assertEqual(len(manifest["recordings"]), 1)
            with self.assertRaises(FileExistsError):
                recorder.main(args)
            self.assertEqual(manifest, json.loads(path.read_text()))

    def test_split_defaults_and_invalid_arguments(self):
        self.assertEqual(recorder.parse_args(["--split", "train"]).positives, 30)
        self.assertEqual(recorder.parse_args(["--split", "validation"]).negatives, 10)
        self.assertTrue(recorder.parse_args(["--list-mics"]).list_mics)
        for args in ([], ["--split", "test", "--session", "../escape"],
                     ["--split", "train", "--background-seconds", "nan"],
                     ["--split", "test", "--positives", "-1"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                recorder.parse_args(args)


if __name__ == "__main__":
    unittest.main()
