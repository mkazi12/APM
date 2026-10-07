import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from tools import evaluate_personal_wakeword as evaluator
from tools import wake_recordings as wr


def row(label, scores, split="validation", duration=1):
    return {"label": label, "scores": scores, "split": split,
            "sustained_max": wr.sustained_score(scores), "duration_seconds": duration}


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dataset = Path(self.temp.name)

    def session(self, split, name, audio=None, **overrides):
        directory = self.dataset / split / name
        directory.mkdir(parents=True)
        audio = np.arange(1600, dtype=np.int16) if audio is None else audio
        sf.write(directory / "clip.wav", audio, wr.RATE, subtype="PCM_16")
        entry = {"file": "clip.wav", "label": "positive", "text": "Hey Gemma",
                 "duration_seconds": len(audio) / wr.RATE, **overrides}
        manifest = {"schema_version": 1, "session_id": name, "split": split, "recordings": [entry]}
        path = directory / "session.json"
        path.write_text(json.dumps(manifest))
        return path

    def test_copied_pcm_across_splits_is_rejected(self):
        self.session("train", "first")
        self.session("validation", "second")
        with self.assertRaisesRegex(ValueError, "Identical audio across splits"):
            wr.load_recordings(self.dataset)

    def test_duplicate_session_ids_are_rejected(self):
        self.session("train", "same")
        self.session("validation", "same", np.ones(1600, dtype=np.int16))
        with self.assertRaisesRegex(ValueError, "Duplicate session ID"):
            wr.load_recordings(self.dataset)

    def test_path_escape_and_wrong_audio_format_are_rejected(self):
        manifest = self.session("train", "first", file="../clip.wav")
        with self.assertRaisesRegex(ValueError, "Unsafe WAV filename"):
            wr.load_recordings(self.dataset)
        content = json.loads(manifest.read_text())
        content["recordings"][0]["file"] = "clip.wav"
        manifest.write_text(json.dumps(content))
        sf.write(manifest.parent / "clip.wav", np.ones(1600), 8000, subtype="FLOAT")
        with self.assertRaisesRegex(ValueError, "16 kHz mono PCM16"):
            wr.load_recordings(self.dataset)

    def test_empty_manifests_are_allowed_and_durations_are_measured(self):
        manifest = self.session("train", "first")
        loaded = wr.load_recordings(self.dataset)
        self.assertEqual(loaded[0]["duration_seconds"], 0.1)
        self.assertIsInstance(loaded[0]["path"], Path)
        content = json.loads(manifest.read_text())
        content["recordings"] = []
        manifest.write_text(json.dumps(content))
        self.assertEqual(wr.load_recordings(self.dataset), [])

    def test_whole_clip_is_streamed_with_padding_and_warmup_discarded(self):
        class FakeModel:
            def reset(self):
                self.frames = []

            def predict(self, frame):
                self.frames.append(frame.copy())
                return {"candidate_name": 0.9 if len(self.frames) <= wr.WARMUP_FRAMES else 0.2}

        audio = np.arange(100 * wr.BLOCK + 17, dtype=np.int16)
        model = FakeModel()
        scores = wr.stream_scores(model, audio)
        self.assertEqual(len(scores), 101)
        self.assertEqual(set(scores), {0.2})
        self.assertTrue(all(len(f) == wr.BLOCK and f.dtype == np.int16 for f in model.frames))
        streamed = np.concatenate(model.frames[wr.WARMUP_FRAMES:])
        np.testing.assert_array_equal(streamed[:len(audio)], audio)
        self.assertTrue(np.all(streamed[len(audio):] == 0))
        self.assertTrue(np.all(np.concatenate(model.frames[:wr.WARMUP_FRAMES]) == 0))
        wr.stream_scores(model, np.ones(wr.BLOCK, dtype=np.int16))
        self.assertEqual(len(model.frames), wr.WARMUP_FRAMES + 1)

    def test_hourly_rate_uses_only_measured_background_duration_and_rearms(self):
        rows = [row("positive", [0.9, 0.9], duration=20),
                row("negative", [0.9, 0.9], duration=9000),
                row("background", [0.9] * 40 + [0.0] * 25 + [0.9] * 40, duration=1800)]
        result = wr.summarize(rows, 0.5)
        self.assertEqual(result["recall"], 1)
        self.assertEqual(result["false_trigger_fraction"], 1)
        self.assertEqual(result["background_hours"], 0.5)
        self.assertEqual(result["background_activation_events"], 2)
        self.assertEqual(result["background_false_activations_per_hour"], 4)
        result = wr.summarize([], 0.5)
        for name in ("recall", "false_trigger_fraction", "background_false_activations_per_hour"):
            self.assertIsNone(result[name])

    def test_calibration_rejects_test_leakage_and_reports_missing_or_impossible_targets(self):
        rows = [row("positive", [0.8, 0.8]), row("negative", [0.2, 0.2]),
                row("background", [0.1, 0.1], duration=1800)]
        threshold, met = wr.choose_threshold(rows)
        self.assertTrue(met)
        self.assertGreater(threshold, 0.2)
        self.assertLessEqual(threshold, 0.8)
        self.assertFalse(wr.choose_threshold(rows[:2])[1])
        rows[1] = row("negative", [0.9, 0.9])
        self.assertFalse(wr.choose_threshold(rows)[1])
        rows[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation recordings only"):
            wr.choose_threshold(rows)

    def test_calibration_finds_low_and_narrow_score_ranges(self):
        for positive, negative in ((0.03, 0.01), (0.80001, 0.8)):
            with self.subTest(positive=positive, negative=negative):
                rows = [row("positive", [positive, positive]),
                        row("negative", [negative, negative]),
                        row("background", [0, 0], duration=1800)]
                threshold, met = wr.choose_threshold(rows)
                self.assertTrue(met)
                self.assertGreater(threshold, negative)
                self.assertLessEqual(threshold, positive)

    def test_long_background_candidate_count_is_bounded(self):
        rows = [row("positive", [0.8, 0.8]), row("negative", [0.2, 0.2]),
                row("background", np.linspace(0, 1, 10000).tolist(), duration=800)]
        self.assertLessEqual(len(wr.threshold_candidates(rows)), 2 * (256 + 4))

    def test_test_comparison_never_calibrates_and_metadata_threshold_is_validated(self):
        model = self.dataset / "head.onnx"
        model.write_bytes(b"test model")
        metadata = model.with_suffix(".metadata.json")
        metadata.write_text(json.dumps({"recommended_threshold": 0.9}))
        self.session("test", "held-out")
        rows = [dict(row("positive", [0.8, 0.8], split="test"), session_id="held-out", text="Hey Gemma", path="clip.wav")]
        with patch.object(evaluator, "score_recordings", return_value=rows), \
             patch.object(evaluator, "choose_threshold", side_effect=AssertionError("Test data must never tune")):
            report = evaluator.evaluate(self.dataset, model, None, "test")
            overridden = evaluator.evaluate(self.dataset, model, None, "test", baseline_threshold=0.7)
        self.assertEqual(report["models"]["baseline"]["fixed"]["recall"], 0)
        self.assertNotIn("calibration", report["models"]["baseline"])
        self.assertEqual(overridden["models"]["baseline"]["fixed"]["recall"], 1)
        self.assertIn("explicit baseline threshold", overridden["models"]["baseline"]["fixed"]["threshold_source"])
        self.assertEqual(evaluator.metadata_threshold(model), 0.9)
        for invalid in (None, 0, 1, float("nan"), float("inf"), True, "0.9"):
            metadata.write_text(json.dumps({"recommended_threshold": invalid}))
            with self.assertRaises(ValueError):
                evaluator.metadata_threshold(model)

    def test_best_effort_threshold_keeps_margin_from_accepted_positives(self):
        rows = [row("positive", [0.7, 0.7]) for _ in range(5)]
        rows += [row("positive", [0.1, 0.1]), row("negative", [0.4, 0.4]),
                 row("background", [0.01, 0.01], duration=30)]
        threshold, met = wr.choose_threshold(rows)
        self.assertFalse(met)
        self.assertGreater(threshold, 0.4)
        self.assertLess(threshold, 0.7)


if __name__ == "__main__":
    unittest.main()
