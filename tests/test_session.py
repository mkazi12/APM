import contextlib
import io
import sys
import unittest
from unittest.mock import MagicMock, patch
from apm.cli import main

class SessionTests(unittest.TestCase):
    def test_multiple_commands_load_once_and_keep_device_state(self):
        factory = MagicMock()
        factory.return_value.predict.side_effect = [
            ({"tool_calls": [{"function": {"name": "set_lights", "arguments": {"device": "kitchen_lights", "on": False}}}]}, {}),
            ({"tool_calls": [{"function": {"name": "get_device_state", "arguments": {"device": "kitchen_lights"}}}]}, {})]
        output = io.StringIO()
        with patch("apm.cli.create_backend", factory), patch.object(sys, "argv", ["apm"]), patch("builtins.input", side_effect=["turn off kitchen lights", "check kitchen lights", "/quit"]), contextlib.redirect_stdout(output):
            main()
        factory.assert_called_once()
        self.assertEqual(factory.return_value.predict.call_count, 2)
        self.assertEqual(output.getvalue().count('Kitchen lights: off (simulated).'), 2)

    def test_failed_request_does_not_end_session(self):
        factory = MagicMock()
        factory.return_value.predict.side_effect = [ValueError("bad response"), ({"content": "Which device?"}, {})]
        output = io.StringIO()
        with patch("apm.cli.create_backend", factory), patch.object(sys, "argv", ["apm"]), patch("builtins.input", side_effect=["first", "second", "/quit"]), contextlib.redirect_stdout(output):
            main()
        self.assertIn("Request failed", output.getvalue())
        self.assertIn("Which device?", output.getvalue())
        factory.assert_called_once()

    def test_startup_voice_failure_preserves_loaded_text_session(self):
        output = io.StringIO()
        with patch("apm.cli.create_backend") as factory, \
             patch("apm.voice.voice_session", side_effect=RuntimeError("microphone unavailable")), \
             patch.object(sys, "argv", ["apm", "--voice"]), \
             patch("builtins.input", side_effect=["/quit"]), contextlib.redirect_stdout(output):
            main()
        factory.assert_called_once()
        self.assertIn("Voice unavailable: microphone unavailable", output.getvalue())
        self.assertIn("Ready.", output.getvalue())

    def test_record_uses_explicit_microphone(self):
        import numpy as np
        with patch("apm.cli.create_backend"), patch("apm.cli.run_request"), \
             patch("sounddevice.rec", return_value=np.zeros((16000, 1))) as record, \
             patch("sounddevice.wait"), patch("builtins.input", return_value=""), \
             patch.object(sys, "argv", ["apm", "--record", "1", "--mic", "7"]):
            main()
        self.assertEqual(record.call_args.kwargs['device'], 7)
