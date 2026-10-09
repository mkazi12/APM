"""Connection reuse must neither rotate credentials nor silently change ports."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from apm.apple_music_server import main, save_connection


class MusicStartupTests(unittest.TestCase):
    def test_reuse_keeps_token_and_does_not_publish_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "connection.json"
            token = "test-token-" + "x" * 32
            save_connection(path, "http://127.0.0.1:8766", token)
            before = path.read_bytes()
            output = io.StringIO()
            with patch("apm.apple_music_server.create_music_app") as app, \
                    patch("socket.socket"), patch("uvicorn.Server") as server, \
                    patch("apm.apple_music_server.save_connection") as save, \
                    contextlib.redirect_stdout(output):
                main(["--reuse-connection", "--connection", str(path)])
            self.assertEqual(app.call_args.args, (token,))
            server.return_value.run.assert_called_once()
            save.assert_not_called()
            self.assertEqual(path.read_bytes(), before)
            self.assertNotIn(token, output.getvalue())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_reuse_missing_malformed_or_wrong_port_fails_before_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "connection.json"
            for data in (None, {}, {"url": "http://127.0.0.1:8767", "token": "x" * 40},
                         {"url": "https://example.com", "token": "x" * 40},
                         {"url": "http://127.0.0.1:8766", "token": "short"}):
                if data is not None:
                    path.write_text(json.dumps(data))
                with self.subTest(data=data), patch("socket.socket") as sock, \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(["--reuse-connection", "--connection", str(path)])
                sock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
