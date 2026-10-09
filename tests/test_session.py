import contextlib
import io
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from apm.cli import main, run_request

class SessionTests(unittest.TestCase):
    def setUp(self):
        from apm.music import MusicService
        music = patch("apm.music_connection.load_music_service", side_effect=MusicService)
        music.start()
        self.addCleanup(music.stop)
        from apm.tasks import TaskService
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "assistant.sqlite3"
        service = patch("apm.tasks.TaskService", side_effect=lambda *_args, **_kwargs: TaskService(path, timezone="UTC"))
        service.start()
        self.addCleanup(service.stop)

    def test_model_question_opens_followup_and_commits_context(self):
        model = MagicMock()
        home = MagicMock()
        home.context.return_value = {"revision": 1}
        home.execute.return_value = []
        response = {"content": "Which room exactly?"}
        model.predict.return_value = (response, {})
        with contextlib.redirect_stdout(io.StringIO()):
            reply = run_request(model, home, text="Turn on the lights")
        self.assertEqual(reply, response["content"])
        self.assertTrue(reply.expects_reply)
        model.commit.assert_called_once_with(response, [])

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
