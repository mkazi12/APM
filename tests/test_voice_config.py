"""Saved wake-model selection without opening microphones or loading ONNX."""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import MagicMock, patch

from apm import voice


class VoiceConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.settings = self.root / "settings" / "voice.json"
        self.settings.parent.mkdir()
        self.model_factory = MagicMock()
        package = ModuleType("openwakeword")
        module = ModuleType("openwakeword.model")
        module.Model = self.model_factory
        package.model = module
        self.module_patch = patch.dict(sys.modules, {"openwakeword": package, "openwakeword.model": module})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def save(self, model):
        self.settings.write_text(json.dumps({"wake_model": str(model)}))

    def assets(self, name="personal", threshold=0.998):
        directory = self.root / name
        directory.mkdir()
        model = directory / "hey_gemma.onnx"
        for filename in (model.name, "melspectrogram.onnx", "embedding_model.onnx"):
            (directory / filename).write_bytes(b"synthetic test placeholder; ONNX loader is mocked")
        model.with_suffix(".metadata.json").write_text(json.dumps({"recommended_threshold": threshold}))
        return model

    def test_unset_config_defers_to_saved_selection(self):
        self.assertIsNone(voice.VoiceConfig().wake_model)
        self.save("../personal/hey_gemma.onnx")
        with patch.object(voice, "DEFAULT_VOICE_SETTINGS", self.settings):
            self.assertEqual(voice.resolve_wake_model(), self.root / "personal" / "hey_gemma.onnx")

    def test_absent_settings_falls_back_to_original_default(self):
        fallback = self.root / "bootstrap" / "hey_gemma.onnx"
        with patch.object(voice, "DEFAULT_WAKE_MODEL", fallback):
            self.assertEqual(voice.resolve_wake_model(settings_path=self.settings), fallback)
        self.assertFalse(self.settings.exists())

    def test_relative_saved_selection_is_relative_to_settings_file(self):
        self.save("../personal/hey_gemma.onnx")
        self.assertEqual(voice.resolve_wake_model(settings_path=self.settings),
                         self.root / "personal" / "hey_gemma.onnx")

    def test_absolute_saved_selection_is_preserved(self):
        model = self.root / "elsewhere" / "hey_gemma.onnx"
        self.save(model)
        self.assertEqual(voice.resolve_wake_model(settings_path=self.settings), model)

    def test_explicit_override_wins_without_reading_saved_config(self):
        self.settings.write_text("malformed saved JSON")
        override = self.root / "explicit" / "hey_gemma.onnx"
        with patch.object(Path, "read_text", side_effect=AssertionError("Saved configuration should not be read")):
            self.assertEqual(voice.resolve_wake_model(override, settings_path=self.settings), override)

    def test_malformed_saved_config_names_the_file(self):
        invalid = ["not JSON", "[]", "null", "{}", '{"wake_model": null}',
                   '{"wake_model": 123}', '{"wake_model": ""}',
                   '{"wake_model": "   "}', '{"wake_model": "model.onnx", "unknown": true}']
        for content in invalid:
            with self.subTest(content=content):
                self.settings.write_text(content)
                with self.assertRaises(ValueError) as caught:
                    voice.resolve_wake_model(settings_path=self.settings)
                self.assertIn(str(self.settings), str(caught.exception))

    def test_missing_saved_model_does_not_fall_back(self):
        fallback = self.assets("bootstrap")
        missing = self.root / "missing-personal" / "hey_gemma.onnx"
        self.save(missing)
        with patch.object(voice, "DEFAULT_VOICE_SETTINGS", self.settings), \
             patch.object(voice, "DEFAULT_WAKE_MODEL", fallback):
            with self.assertRaises(FileNotFoundError) as caught:
                voice.WakeDetector(voice.VoiceConfig())
        self.assertIn(str(missing), str(caught.exception))
        self.model_factory.assert_not_called()

    def test_missing_selected_backbone_does_not_fall_back(self):
        selected = self.assets()
        missing = selected.parent / "embedding_model.onnx"
        missing.unlink()
        self.save(selected)
        with patch.object(voice, "DEFAULT_VOICE_SETTINGS", self.settings):
            with self.assertRaises(FileNotFoundError) as caught:
                voice.WakeDetector(voice.VoiceConfig())
        self.assertIn(str(missing), str(caught.exception))
        self.model_factory.assert_not_called()

    def test_detector_loads_selected_metadata_and_backbones(self):
        selected = self.assets(threshold=0.9982648491859436)
        self.save(selected)
        with patch.object(voice, "DEFAULT_VOICE_SETTINGS", self.settings):
            detector = voice.WakeDetector(voice.VoiceConfig())
        self.assertEqual(detector.model_path, selected)
        self.assertEqual(detector.threshold, 0.9982648491859436)
        self.model_factory.assert_called_once_with(
            wakeword_models=[str(selected)], inference_framework="onnx",
            melspec_model_path=str(selected.parent / "melspectrogram.onnx"),
            embedding_model_path=str(selected.parent / "embedding_model.onnx"))

    def test_explicit_model_and_threshold_override_saved_selection(self):
        explicit = self.assets("explicit", threshold=0.98)
        self.settings.write_text("malformed saved JSON")
        with patch.object(voice, "DEFAULT_VOICE_SETTINGS", self.settings):
            detector = voice.WakeDetector(voice.VoiceConfig(wake_model=explicit, threshold=0.82))
        self.assertEqual(detector.model_path, explicit)
        self.assertEqual(detector.threshold, 0.82)
        self.assertEqual(self.model_factory.call_args.kwargs["wakeword_models"], [str(explicit)])


if __name__ == "__main__":
    unittest.main()
