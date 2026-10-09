import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from apm.music_connection import RemoteMusicService


class PauseDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.music = RemoteMusicService("http://127.0.0.1:8766", "x" * 40)

    def test_old_player_requires_update_and_does_not_claim_silence(self):
        with patch.object(self.music, "_call", return_value={
            "status": "unknown", "reason": "player_update_required",
            "message": "private upstream details",
        }) as request:
            result = self.music.pause()
        self.assertEqual(result["reason"], "player_update_required")
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["playing"])
        self.assertNotIn("private", str(result))
        request.assert_called_once()

    def test_transport_diagnostics_are_sanitized_and_never_retried(self):
        for error, reason in (
            (HTTPError("private-url", 401, "private-token", {}, None), "bridge_authorization_failed"),
            (HTTPError("private-url", 404, "private-token", {}, None), "bridge_update_required"),
            (URLError("private-token"), "bridge_unreachable"),
            (TimeoutError("private-token"), "bridge_unreachable"),
        ):
            with self.subTest(reason=reason):
                with patch.object(self.music, "_call", side_effect=error) as request:
                    result = self.music.pause()
                self.assertEqual(result["reason"], reason)
                self.assertEqual(result["status"], "unknown")
                self.assertIsNone(result["playing"])
                self.assertNotIn("private", str(result))
                request.assert_called_once()

    def test_unknown_provider_details_are_not_forwarded(self):
        with patch.object(self.music, "_call", return_value={
            "status": "unknown", "reason": "private-token", "message": "private-url",
        }):
            result = self.music.pause()
        self.assertEqual(result["reason"], "pause_unconfirmed")
        self.assertNotIn("private", str(result))


class ResumeConnectionTests(unittest.TestCase):
    def setUp(self):
        self.music = RemoteMusicService("http://127.0.0.1:8766", "x" * 40)

    def test_browser_reason_survives_resume_without_forwarding_raw_messages(self):
        with patch.object(self.music, "_call", return_value={
            "status": "unknown", "reason": "autoplay_blocked", "message": "private upstream token",
        }) as request:
            result = self.music.resume()
        self.assertEqual(result["reason"], "autoplay_blocked")
        self.assertIn("Start playback", result["message"])
        self.assertNotIn("private", str(result))
        request.assert_called_once()

    def test_resume_success_and_empty_send_only_empty_body_and_sanitize_metadata(self):
        for status, accepted, playing in (("resumed", True, True), ("empty", False, False)):
            with self.subTest(status=status), patch.object(self.music, "_call", return_value={
                "status":status, "accepted":accepted, "playing":playing,
                "track_id":"private-id", "message":"private-provider-message",
            }) as request:
                result = self.music.resume()
                self.assertEqual(result["status"], status)
                self.assertIs(result["playing"], playing)
                self.assertNotIn("private", str(result))
                request.assert_called_once_with("/v1/music/resume", {}, timeout=22)

    def test_resume_unavailable_and_outdated_player_have_fixed_helpful_messages(self):
        for reason in ("not_configured", "unsupported", "disconnected", "player_update_required"):
            with self.subTest(reason=reason), patch.object(self.music, "_call", return_value={
                "status":"unknown" if reason == "player_update_required" else "unavailable",
                "reason":reason, "accepted":False, "playing":None, "message":"private-token",
            }) as request:
                result = self.music.resume()
                self.assertEqual(result["reason"], reason)
                self.assertIsNone(result["playing"])
                self.assertTrue(result["message"])
                self.assertNotIn("private", str(result))
                request.assert_called_once()

    def test_resume_rejects_invalid_confirmation_and_never_retries_errors(self):
        for value in ({"status":"resumed", "accepted":True, "playing":None},
                      {"status":"resumed", "accepted":1, "playing":True},
                      {"status":"empty", "accepted":True, "playing":False},
                      {"status":"unknown", "reason":"private-token", "message":"private-url"},
                      HTTPError("private-url", 401, "private-token", {}, None),
                      HTTPError("private-url", 404, "private-token", {}, None),
                      URLError("private-token"), TimeoutError("private-token")):
            with self.subTest(value=type(value).__name__):
                options = {"side_effect":value} if isinstance(value, Exception) else {"return_value":value}
                with patch.object(self.music, "_call", **options) as request:
                    result = self.music.resume()
                self.assertEqual(result["status"], "unknown")
                self.assertIsNone(result["playing"])
                self.assertNotIn("private", str(result))
                request.assert_called_once()
