import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from tools import train_personal_wakeword as trainer


class PersonalTrainingTests(unittest.TestCase):
    def test_training_rejects_held_out_clips_before_reading_them(self):
        for split in ("validation", "test"):
            with patch.object(trainer, "read_audio", side_effect=AssertionError("Must not read")):
                with self.assertRaisesRegex(ValueError, "training sessions only"):
                    trainer.personal_features(Path("model.onnx"), [{"split": split}])

    def test_class_and_source_weights_do_not_depend_on_augmented_window_counts(self):
        sy = np.array([0] * 100 + [1] * 20)
        py = np.array([0] * 30 + [1] * 10)
        # Three negatives, one positive; equal total mass for each recording.
        pw = np.full(40, 0.1)
        weights = trainer.training_weights(sy, py, pw)
        sw, pw = weights[:len(sy)], weights[len(sy):]
        for label in (0, 1):
            total = sw[sy == label].sum() + pw[py == label].sum()
            self.assertAlmostEqual(total, weights.sum() / 2)
            self.assertAlmostEqual(pw[py == label].sum() / total, 0.65)
        with self.assertRaisesRegex(ValueError, "positive and negative"):
            trainer.training_weights(sy, np.ones(3), np.ones(3))

    def test_missed_wake_examples_still_enter_training(self):
        # The existing detector outputs zero for this entire positive. Selection
        # must use the speech boundary, not the model's mistaken prediction.
        record = {"split": "train", "label": "positive", "path": Path("missed.wav")}
        features = np.arange(50 * 16 * 96, dtype=np.float32).reshape(50, 16, 96)
        with patch.object(trainer, "make_model"), \
             patch.object(trainer, "read_audio", return_value=np.zeros(64000, dtype=np.int16)), \
             patch.object(trainer, "speech_bounds", return_value=(8000, 24000)), \
             patch.object(trainer, "streamed_features", return_value=(features, np.zeros(50))):
            x, y, weights, alignment = trainer.personal_features(Path("base.onnx"), [record], 0)
        self.assertGreater(len(x), 1)
        self.assertTrue(np.all(y == 1))
        self.assertAlmostEqual(weights.sum(), 1)
        self.assertEqual(alignment[0]["speech_end_seconds"], 1.5)

    def test_streaming_feature_extraction_includes_partial_last_frame(self):
        class Preprocessor:
            def get_features(self, count):
                return np.zeros((1, count, 96), dtype=np.float32)

        class Model:
            preprocessor = Preprocessor()

            def reset(self):
                self.frames = []

            def predict(self, frame):
                self.frames.append(frame.copy())
                return {"base": 0}

        model = Model()
        audio = np.arange(2561, dtype=np.int16)
        features, _ = trainer.streamed_features(model, audio, "base")
        self.assertEqual(len(features), 3)
        restored = np.concatenate(model.frames[trainer.WARMUP_FRAMES:])
        np.testing.assert_array_equal(restored[:len(audio)], audio)
        self.assertTrue(np.all(restored[len(audio):] == 0))

    def test_invalid_synthetic_cache_cannot_train(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.npz"
            np.savez(path, x=np.zeros((2, 16, 96)), y=np.ones(2))
            with self.assertRaisesRegex(ValueError, "both binary labels"):
                trainer.load_synthetic(path)

    def test_recording_holdout_reserves_whole_takes_and_all_background(self):
        records = [{"path": Path(f"{label}-{i}.wav"), "split": "train", "label": label}
                   for label, count in (("positive", 30), ("negative", 20), ("background", 1))
                   for i in range(count)]
        train, validation = trainer.recording_holdout(records)
        self.assertEqual(len(train), 40)
        self.assertEqual(len(validation), 11)
        self.assertFalse({r["path"] for r in train} & {r["path"] for r in validation})
        self.assertFalse(any(r["label"] == "background" for r in train))
        self.assertTrue(all(r["split"] == "validation" for r in validation))
        self.assertTrue(all(r["split"] == "train" for r in records))
        self.assertEqual(trainer.recording_holdout(records), (train, validation))


if __name__ == "__main__":
    unittest.main()
