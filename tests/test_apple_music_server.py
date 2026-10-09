import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import uuid

from fastapi.testclient import TestClient

from apm.apple_music_server import create_music_app, save_connection
from apm.music_connection import RemoteMusicService, load_music_service
from apm.musickit import MusicKitProvider


TOKEN = "a-local-test-token-with-at-least-32-characters"


class Signer:
    def __init__(self, configured=True):
        self.configured = configured

    def status(self):
        return {"configured": self.configured, "message": "Ready" if self.configured else "Set up the developer key"}

    def __call__(self):
        return "test-developer-token"


class AppleMusicServerTests(unittest.TestCase):
    def client(self, configured=True):
        signer = Signer(configured)
        provider = MusicKitProvider(signer, request=lambda *a, **kw: {"results": {"songs": {"data": []}}})
        client = TestClient(create_music_app(TOKEN, provider=provider, signer=signer))
        self.addCleanup(provider.close)
        self.addCleanup(client.close)
        return client, provider

    def headers(self):
        return {"Authorization": "Bearer " + TOKEN}

    def test_config_requires_bearer_and_never_exposes_signing_key(self):
        client, _ = self.client()
        self.assertEqual(client.get("/v1/config").status_code, 401)
        response = client.get("/v1/config", headers=self.headers())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"configured": True, "developer_token": "test-developer-token", "app_name": "APM Assistant"})
        self.assertEqual(response.headers["cache-control"], "no-store")
        status = client.get("/v1/music/status", headers=self.headers()).json()
        self.assertNotIn("test-developer-token", json.dumps(status))
        self.assertFalse(status["player"]["connected"])

    def test_player_allows_origin_referrer_for_apple_callback_only(self):
        from html.parser import HTMLParser

        class ReferrerMetaParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.overrides = []

            def handle_starttag(self, tag, attrs):
                if tag == "meta":
                    attributes = dict(attrs)
                    if (attributes.get("name") or "").strip().lower() == "referrer":
                        self.overrides.append(attributes)

        client, _ = self.client()
        page = client.get("/?diagnostics=1")
        self.assertEqual(page.status_code, 200)
        # Apple needs the origin to construct its popup callback. strict-origin
        # excludes path/query data even for same-origin requests and suppresses
        # referrers on a security downgrade.
        self.assertEqual(page.headers["referrer-policy"], "strict-origin")
        # A document-level meta policy overrides the response header and can
        # silently prevent Apple's popup from finding its callback origin.
        parser = ReferrerMetaParser()
        parser.feed(page.text)
        parser.close()
        self.assertEqual(parser.overrides, [], "The served player must inherit its HTTP referrer policy")
        self.assertEqual(page.headers["cache-control"], "no-store")
        self.assertIn("frame-ancestors 'none'", page.headers["content-security-policy"])
        for path in ("/static/music.js", "/v1/config", "/v1/music/status"):
            response = client.get(path, headers=self.headers())
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["referrer-policy"], "no-referrer")

    def test_bad_origins_and_hosts_rejected_even_with_bearer(self):
        client, _ = self.client()
        for origin in ("https://attacker.invalid", "http://localhost:9999", "null"):
            response = client.get("/v1/config", headers={**self.headers(), "Origin": origin})
            self.assertEqual(response.status_code, 403)
        self.assertEqual(client.get("/v1/config", headers={**self.headers(), "Host": "attacker.invalid"}).status_code, 400)
        self.assertEqual(client.get("/v1/config", headers={**self.headers(), "Origin": "http://testserver"}).status_code, 200)

    def test_setup_page_and_session_have_honest_connection_state(self):
        client, provider = self.client(False)
        state = client.get("/v1/config", headers=self.headers()).json()
        self.assertFalse(state["configured"])
        self.assertNotIn("developer_token", state)
        response = client.post("/v1/player/session", headers=self.headers(), json={"session_id": str(uuid.uuid4()), "storefront": "us"})
        self.assertEqual(response.status_code, 400)
        self.assertFalse(provider.status()["connected"])

    def test_player_session_validation_disconnect_and_search_without_playback(self):
        client, provider = self.client()
        session = str(uuid.uuid4())
        response = client.post("/v1/player/session", headers=self.headers(), json={"session_id": session, "storefront": "us"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(provider.status()["connected"])
        searched = client.post("/v1/music/resolve", headers=self.headers(), json={"title": "I'm On Fire"})
        self.assertEqual(searched.json()["status"], "not_found")
        self.assertIsNone(provider.poll(session, wait_seconds=0))
        response = client.post("/v1/player/disconnect", headers=self.headers(), json={"session_id": session})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(provider.status()["connected"])

    def test_body_errors_do_not_echo_submitted_tokens(self):
        client, _ = self.client()
        response = client.post("/v1/music/play", headers=self.headers(), json={"title": "Song", "private_key": "do-not-echo"})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("do-not-echo", response.text)

    def test_completion_diagnostics_are_strict_private_and_do_not_consume_command_on_error(self):
        client, provider = self.client()
        session = str(uuid.uuid4())
        provider.activate(session, "us")
        replies = []
        request = threading.Thread(target=lambda: replies.append(client.post(
            "/v1/music/resume", headers=self.headers(), json={})))
        request.start()
        command = provider.poll(session, wait_seconds=1)
        self.assertIsNotNone(command)
        detail = {"phase":"play", "reason":"autoplay_blocked", "elapsed_ms":100, "phase_ms":80,
                  "sdk_reason":"USER_INTERACTION_REQUIRED", "queue_check":"matched", "state":"paused",
                  "track_matches":True}
        route = f"/v1/player/commands/{command['id']}/result"
        body = {"session_id":session, "result":{"error":"playback_unconfirmed"}}
        for invalid in ({**detail, "url":"private-token"}, {**detail, "reason":"private-token"},
                        {**detail, "elapsed_ms":"100"}, {**detail, "elapsed_ms":True},
                        {**detail, "phase_ms":101}, {**detail, "track_matches":1},
                        {**detail, "sdk_reason":None}, {}):
            response = client.post(route, headers=self.headers(), json={**body, "diagnostics":invalid})
            self.assertEqual(response.status_code, 422)
            self.assertNotIn("private-token", response.text)
            self.assertTrue(request.is_alive())
            self.assertNotIn("last_command", provider.status())
        self.assertEqual(client.post(route, json={**body, "diagnostics":detail}).status_code, 401)
        response = client.post(route, headers=self.headers(), json={**body, "diagnostics":detail})
        self.assertEqual(response.status_code, 200)
        request.join(2)
        self.assertFalse(request.is_alive())
        self.assertEqual(replies[0].json()["status"], "unknown")
        self.assertEqual(replies[0].json()["reason"], "autoplay_blocked")
        state = client.get("/v1/music/status", headers=self.headers()).json()["player"]
        self.assertEqual(state["last_command"], {"operation":"resume", **detail})
        self.assertNotIn(command["id"], json.dumps(state))
        self.assertNotIn(session, json.dumps(state))
        repeated = client.post(route, headers=self.headers(), json={**body, "diagnostics":detail})
        self.assertEqual(repeated.status_code, 400)
        self.assertEqual(provider.status()["last_command"], {"operation":"resume", **detail})

    def test_song_request_round_trip_waits_for_browser_readback(self):
        client, provider = self.client()
        provider._request = lambda *_args, **_kwargs: {"results": {"songs": {"data": [{
            "id": "123456", "type": "songs", "attributes": {
                "name": "I'm On Fire", "artistName": "Bruce Springsteen", "albumName": "Born in the U.S.A.",
                "playParams": {"id": "123456", "kind": "song"}}}]}}}
        session = str(uuid.uuid4())
        client.post("/v1/player/session", headers=self.headers(), json={"session_id": session, "storefront": "us"})
        replies = []
        request = threading.Thread(target=lambda: replies.append(client.post(
            "/v1/music/play", headers=self.headers(), json={"title": "I'm On Fire", "artist": "Bruce Springstein"})))
        request.start()
        command = provider.poll(session, wait_seconds=1)
        self.assertIsNotNone(command)
        self.assertTrue(request.is_alive())
        observed = client.post(f"/v1/player/commands/{command['id']}/result", headers=self.headers(),
                               json={"session_id": session, "result": {"accepted": True, "playing": True, "track_id": "123456"}})
        self.assertEqual(observed.status_code, 200)
        request.join(timeout=2)
        self.assertFalse(request.is_alive())
        self.assertEqual(replies[0].json()["status"], "playing")

    def test_pause_round_trip_is_authenticated_strict_and_confirmed(self):
        client, provider = self.client()
        self.assertEqual(client.post("/v1/music/pause", json={}).status_code, 401)
        self.assertEqual(client.post("/v1/music/pause", headers=self.headers(), json={"private": "secret"}).status_code, 422)
        self.assertEqual(client.post("/v1/music/pause", headers=self.headers(), json={}).json()["status"], "unavailable")
        session = str(uuid.uuid4())
        provider.activate(session, "us")
        replies = []
        request = threading.Thread(target=lambda: replies.append(client.post("/v1/music/pause", headers=self.headers(), json={})))
        request.start()
        command = provider.poll(session, wait_seconds=1)
        self.assertEqual(command["operation"], "pause")
        self.assertNotIn("track_id", command)
        self.assertTrue(request.is_alive())
        response = client.post(f"/v1/player/commands/{command['id']}/result", headers=self.headers(),
                               json={"session_id": session, "result": {"accepted": True, "playing": False, "was_playing": False}})
        self.assertEqual(response.status_code, 200)
        request.join(1)
        self.assertFalse(request.is_alive())
        self.assertEqual(replies[0].json()["status"], "paused")
        self.assertIs(replies[0].json()["was_playing"], False)

    def test_resume_round_trip_requires_auth_and_never_requests_a_catalog_song(self):
        client, provider = self.client()
        self.assertEqual(client.post("/v1/music/resume", json={}).status_code, 401)
        self.assertEqual(client.post("/v1/music/resume", headers={**self.headers(), "Origin":"https://evil.invalid"}, json={}).status_code, 403)
        for body in ({"track_id":"private-id"}, {"title":"A Song"}, {"selection_id":"private-id"}, []):
            response = client.post("/v1/music/resume", headers=self.headers(), json=body)
            self.assertEqual(response.status_code, 422)
            self.assertNotIn("private-id", response.text)
        self.assertEqual(client.post("/v1/music/resume", headers=self.headers(), json={}).json()["status"], "unavailable")
        session = str(uuid.uuid4())
        activated = client.post("/v1/player/session", headers=self.headers(),
                                json={"session_id":session, "storefront":"us", "protocol_version":3})
        self.assertEqual(activated.status_code, 200)
        with patch.object(provider, "_request", side_effect=AssertionError("Resume must not search")) as catalog:
            for observed, expected in (({"accepted":True, "playing":True, "track_id":"123456"}, "resumed"),
                                       ({"accepted":False, "playing":False, "track_id":None}, "empty")):
                replies = []
                request = threading.Thread(target=lambda: replies.append(client.post(
                    "/v1/music/resume", headers=self.headers(), json={})))
                request.start()
                command = provider.poll(session, wait_seconds=1)
                self.assertIsNotNone(command)
                self.assertEqual(command["operation"], "resume")
                self.assertNotIn("track_id", command)
                self.assertTrue(request.is_alive())
                response = client.post(f"/v1/player/commands/{command['id']}/result", headers=self.headers(),
                                       json={"session_id":session, "result":observed})
                self.assertEqual(response.status_code, 200)
                request.join(2)
                self.assertFalse(request.is_alive())
                self.assertEqual(replies[0].json()["status"], expected)
            catalog.assert_not_called()

    def test_player_protocol_handshake_identifies_old_page_and_requires_reload_for_pause(self):
        client, provider = self.client()
        session = str(uuid.uuid4())
        legacy = {"session_id": session, "storefront": "us"}
        self.assertEqual(client.post("/v1/player/session", headers=self.headers(), json=legacy).status_code, 200)
        player = client.get("/v1/music/status", headers=self.headers()).json()["player"]
        self.assertEqual(player["protocol_version"], 1)
        self.assertFalse(player["supports_pause"])
        pause = client.post("/v1/music/pause", headers=self.headers(), json={}).json()
        self.assertEqual((pause["status"], pause["reason"]), ("unknown", "player_update_required"))
        self.assertIsNone(pause["playing"])
        self.assertIsNone(provider.poll(session, wait_seconds=0))
        self.assertTrue(provider.status()["connected"])
        for version in (True, "2", None, 0, 4):
            response = client.post("/v1/player/session", headers=self.headers(), json={**legacy, "protocol_version": version})
            self.assertEqual(response.status_code, 422)
        response = client.post("/v1/player/session", headers=self.headers(), json={**legacy, "protocol_version": 2})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(provider.status()["supports_pause"])
        resume = client.post("/v1/music/resume", headers=self.headers(), json={}).json()
        self.assertEqual((resume["status"], resume["reason"]), ("unknown", "player_update_required"))
        self.assertIn("enable resume", resume["message"])
        self.assertIsNone(provider.poll(session, wait_seconds=0))
        response = client.post("/v1/player/session", headers=self.headers(), json={**legacy, "protocol_version":3})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(provider.status()["supports_pause"])
        self.assertTrue(provider.status()["supports_resume"])

    def test_remote_pause_is_bounded_sanitized_and_never_retried(self):
        service = RemoteMusicService("http://127.0.0.1:8766", TOKEN)
        confirmed = {"status": "paused", "accepted": True, "playing": False, "was_playing": True}
        with patch.object(service, "_call", return_value=confirmed) as call:
            self.assertEqual(service.pause()["status"], "paused")
            call.assert_called_once_with("/v1/music/pause", {}, timeout=2)
        for value in ({"status": "paused", "accepted": True, "playing": True},
                      TimeoutError("private-token")):
            with patch.object(service, "_call", **({"side_effect": value} if isinstance(value, Exception) else {"return_value": value})) as call:
                result = service.pause()
                self.assertEqual(result["status"], "unknown")
                self.assertNotIn("private-token", str(result))
                call.assert_called_once()

    def test_connection_file_is_private_and_loads_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "connection.json"
            self.assertFalse(load_music_service(path).status()["configured"])
            save_connection(path, "http://127.0.0.1:8766", TOKEN)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            service = load_music_service(path)
            self.assertIsInstance(service, RemoteMusicService)
            with patch.object(service, "_call", return_value={"status": "matched"}) as call:
                service.resolve("Song", "Artist")
                self.assertEqual(call.call_args.args, ("/v1/music/resolve", {"title": "Song", "artist": "Artist", "version": None}))

    def test_remote_music_validates_before_dispatch_and_does_not_retry_uncertain_write(self):
        service = RemoteMusicService("http://127.0.0.1:8766", TOKEN)
        with patch.object(service, "_call", side_effect=TimeoutError("secret-service-details")) as call:
            with self.assertRaises(ValueError):
                service.play("Song", version="invented")
            call.assert_not_called()
            result = service.play("Song")
            self.assertEqual(result["status"], "unknown")
            self.assertNotIn("secret-service-details", str(result))
            call.assert_called_once()

    def test_remote_connection_rejects_network_hosts_and_embedded_credentials(self):
        for url in ("https://example.com", "http://localhost:8766", "http://127.0.0.1:8766/path", "http://name:secret@127.0.0.1:8766"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                RemoteMusicService(url, TOKEN)


if __name__ == "__main__":
    unittest.main()
