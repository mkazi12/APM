import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from apm.server import create_app


HAS_API_DEPS = all(importlib.util.find_spec(name) for name in ("fastapi", "httpx"))
if HAS_API_DEPS:
    from fastapi.testclient import TestClient


class FakeProvider:
    name = "Test Music"

    def __init__(self):
        self.tracks = []
        self.searches = []
        self.plays = []
        self.closed = 0
        self.search_error = None
        self.play_error = None
        self.result = None

    def search(self, title, artist=None):
        self.searches.append((title, artist))
        if self.search_error is not None:
            raise self.search_error
        return list(self.tracks)

    def play(self, track_id):
        from apm.music import PlaybackResult

        self.plays.append(track_id)
        if self.play_error is not None:
            raise self.play_error
        return self.result if self.result is not None else PlaybackResult(True, playing=True, track_id=track_id)

    def close(self):
        self.closed += 1


@unittest.skipUnless(HAS_API_DEPS, "Install the home extra and httpx to test the music API")
class MusicServerTests(unittest.TestCase):
    def setUp(self):
        from apm.home import HomeController
        from apm.music import MusicService

        self.environment = patch.dict(os.environ, {}, clear=False)
        self.environment.start()
        os.environ.pop("APM_API_TOKEN", None)
        self.addCleanup(self.environment.stop)
        self.provider = FakeProvider()
        self.music = MusicService(self.provider)
        self.home = HomeController()
        self.client = TestClient(create_app(self.home, music=self.music))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def track(self, identifier="track-one", title="A Song", artist="A Singer", **kwargs):
        from apm.music import Track

        return Track(identifier, title, (artist,), **kwargs)

    def test_unconfigured_default_is_available_without_tasks_and_never_claims_playback(self):
        from apm.home import HomeController

        with TestClient(create_app(HomeController())) as client:
            self.assertEqual(client.get("/v1/music/status").json(), {"configured": False, "provider": None})
            for path in ("/v1/music/resolve", "/v1/music/play"):
                result = client.post(path, json={"title": "A Song"})
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json()["status"], "not_configured")
                self.assertFalse(result.json().get("playing", False))
            self.assertEqual(client.get("/v1/clock").status_code, 404)
        self.assertEqual(self.provider.plays, [])

    def test_status_and_context_report_music_without_mutating_home_tools(self):
        from apm.toolsets.music import MUSIC_TOOLS

        before = self.home.context()
        self.assertEqual(self.client.get("/v1/music/status").json(), {"configured": True, "provider": "Test Music"})
        result = self.client.get("/v1/context")
        self.assertEqual(result.status_code, 200, result.text)
        context = result.json()
        self.assertEqual(context["music"], {"configured": True, "provider": "Test Music"})
        self.assertEqual(context["tools"], before["tools"] + MUSIC_TOOLS)
        self.assertEqual(self.home.context(), before)
        self.assertEqual(self.provider.searches, [])
        self.assertEqual(self.provider.plays, [])

    def test_resolve_only_searches_and_play_uses_resolved_track(self):
        self.provider.tracks = [self.track()]
        query = {"title": "A Song", "artist": "A Singer", "version": "studio"}
        result = self.client.post("/v1/music/resolve", json=query)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["status"], "matched")
        self.assertEqual(result.json()["track"]["title"], "A Song")
        self.assertIsInstance(result.json()["track"]["selection_id"], str)
        self.assertEqual(self.provider.plays, [])
        played = self.client.post("/v1/music/play", json=query)
        self.assertEqual(played.status_code, 200, played.text)
        self.assertEqual(played.json()["status"], "playing")
        self.assertTrue(played.json()["accepted"])
        self.assertTrue(played.json()["playing"])
        self.assertEqual(self.provider.plays, ["track-one"])
        self.assertEqual(played.json()["track"]["title"], "A Song")

    def test_pause_is_strict_and_reports_confirmed_state_without_search(self):
        from apm.music import PlaybackResult
        with patch.object(self.provider, "pause", create=True, return_value=PlaybackResult(True, False, was_playing=True)) as pause:
            self.assertEqual(self.client.post("/v1/music/pause", json={"track_id": "private-id"}).status_code, 422)
            pause.assert_not_called()
            response = self.client.post("/v1/music/pause", json={})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "paused")
            self.assertIs(response.json()["was_playing"], True)
            pause.assert_called_once_with()
        self.assertEqual(self.provider.searches, [])
        self.assertEqual(self.provider.plays, [])

    def test_resume_is_strict_and_only_resumes_existing_queue(self):
        from apm.music import PlaybackResult
        with patch.object(self.provider, "resume", create=True,
                          return_value=PlaybackResult(True, True, "existing-queue-track")) as resume:
            for body in ({"track_id":"private-id"}, {"title":"Song"}, {"selection_id":"private-id"}, []):
                response = self.client.post("/v1/music/resume", json=body)
                self.assertEqual(response.status_code, 422)
                self.assertNotIn("private-id", response.text)
            resume.assert_not_called()
            response = self.client.post("/v1/music/resume", json={})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "resumed")
            self.assertIs(response.json()["playing"], True)
            self.assertNotIn("existing-queue-track", response.text)
            resume.assert_called_once_with()
        self.assertEqual(self.provider.searches, [])
        self.assertEqual(self.provider.plays, [])

    def test_resume_empty_and_uncertain_results_are_honest(self):
        from apm.music import PlaybackResult
        for value, status in ((PlaybackResult(False, False, None), "empty"),
                              (PlaybackResult(True), "unknown"),
                              (RuntimeError("private-token"), "unknown")):
            with self.subTest(status=status):
                options = {"side_effect":value} if isinstance(value, Exception) else {"return_value":value}
                with patch.object(self.provider, "resume", create=True, **options) as resume:
                    response = self.client.post("/v1/music/resume", json={})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], status)
                self.assertIsNot(response.json()["playing"], True)
                self.assertNotIn("private-token", response.text)
                resume.assert_called_once_with()

    def test_playback_acknowledgement_is_not_reported_as_observed_playing(self):
        from apm.music import PlaybackResult

        self.provider.tracks = [self.track()]
        self.provider.result = PlaybackResult(True)
        result = self.client.post("/v1/music/play", json={"title": "A Song", "artist": "A Singer"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["status"], "accepted")
        self.assertTrue(result.json()["accepted"])
        self.assertIsNone(result.json()["playing"])

    def test_resolve_offers_explicit_selection_and_rejects_replay(self):
        self.provider.tracks = [self.track("one", "Stay", "Singer One"), self.track("two", "Stay", "Singer Two")]
        result = self.client.post("/v1/music/resolve", json={"title": "Stay"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["status"], "ambiguous")
        self.assertEqual(self.provider.plays, [])
        candidates = result.json()["candidates"]
        chosen = next(candidate for candidate in candidates if candidate["artists"] == ["Singer Two"])
        path = f"/v1/music/selections/{chosen['selection_id']}/play"
        invalid = self.client.post(path, json={"track_id": "injected-provider-id"})
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(self.provider.plays, [])
        selected = self.client.post(path, json={})
        self.assertEqual(selected.status_code, 200, selected.text)
        self.assertEqual(self.provider.plays, ["two"])
        self.assertEqual(self.client.post(path).status_code, 400)
        self.assertEqual(self.provider.plays, ["two"])

    def test_play_many_compilation_editions_uses_provider_first_once_without_choices(self):
        self.provider.tracks = [self.track(f"edition-{index}", album=f"Compilation {index}")
                                for index in range(24, -1, -1)]
        result = self.client.post("/v1/music/play", json={"title": "A Song", "artist": "A Singer"})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["status"], "playing")
        self.assertEqual(result.json()["track"]["album"], "Compilation 24")
        self.assertNotIn("candidates", result.json())
        self.assertEqual(self.provider.plays, ["edition-24"])

    def test_play_without_artist_uses_first_eligible_match(self):
        self.provider.tracks = [self.track("one", "Stay", "Singer One"), self.track("two", "Stay", "Singer Two")]
        result = self.client.post("/v1/music/play", json={"title": "Stay"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["status"], "playing")
        self.assertNotIn("candidates", result.json())
        self.assertEqual(self.provider.plays, ["one"])

    def test_live_only_play_still_requires_a_choice(self):
        self.provider.tracks = [self.track("live", "A Song (Live)", version="live")]
        result = self.client.post("/v1/music/play", json={"title": "A Song", "artist": "A Singer"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["status"], "ambiguous")
        self.assertEqual(result.json()["candidates"][0]["version"], "live")
        self.assertEqual(self.provider.plays, [])

    def test_play_filters_wrong_artist_and_wrong_version_before_first_result(self):
        self.provider.tracks = [self.track("cover", "A Song", "Another Artist"),
                                self.track("live", "A Song (Live)", version="live"),
                                self.track("studio")]
        result = self.client.post("/v1/music/play", json={"title": "A Song", "artist": "A Singer", "version": "studio"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["status"], "playing")
        self.assertEqual(self.provider.plays, ["studio"])

    def test_not_found_does_not_play_a_different_song(self):
        self.provider.tracks = [self.track(title="Different Song")]
        result = self.client.post("/v1/music/play", json={"title": "A Song", "artist": "A Singer"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["status"], "not_found")
        self.assertEqual(self.provider.plays, [])

    def test_strict_bounded_query_and_selection_bodies_reject_invalid_inputs(self):
        invalid_queries = [
            {}, {"title": 42}, {"title": True}, {"title": ""}, {"title": "x" * 201},
            {"title": "A Song", "artist": 42}, {"title": "A Song", "artist": "x" * 201},
            {"title": "A Song", "version": "unknown"}, {"title": "A Song", "version": "bootleg"},
            {"title": "A Song", "private": "private-input"},
        ]
        for path in ("/v1/music/resolve", "/v1/music/play"):
            for query in invalid_queries:
                with self.subTest(path=path, query=query):
                    result = self.client.post(path, json=query)
                    self.assertEqual(result.status_code, 422, result.text)
                    self.assertNotIn("private-input", result.text)
            for query in ({"title": "   "}, {"title": "private\ninput"}, {"title": "A Song", "artist": "private\x00input"}):
                result = self.client.post(path, json=query)
                self.assertEqual(result.status_code, 400, result.text)
                self.assertNotIn("private", result.text)
        self.assertEqual(self.client.post("/v1/music/selections/not-a-selection/play").status_code, 400)
        self.assertEqual(self.provider.searches, [])
        self.assertEqual(self.provider.plays, [])

    def test_search_and_playback_failures_remain_sanitized_domain_results(self):
        self.provider.search_error = RuntimeError("private-provider-token")
        result = self.client.post("/v1/music/resolve", json={"title": "A Song"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["status"], "failed")
        self.assertNotIn("private-provider-token", result.text)
        self.provider.search_error = None
        self.provider.tracks = [self.track()]
        self.provider.play_error = RuntimeError("private-provider-token")
        result = self.client.post("/v1/music/play", json={"title": "A Song", "artist": "A Singer"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["status"], "unknown")
        self.assertNotIn("private-provider-token", result.text)
        self.assertEqual(self.provider.plays, ["track-one"])

    def test_existing_auth_and_origin_checks_protect_music_operations(self):
        client = TestClient(create_app(self.home, token="api-secret", music=self.music))
        query = {"title": "A Song"}
        self.assertEqual(client.get("/v1/music/status").status_code, 401)
        for path in ("/v1/music/resolve", "/v1/music/play", "/v1/music/pause", "/v1/music/resume", "/v1/music/selections/invalid/play"):
            self.assertEqual(client.post(path, json=query).status_code, 401)
        result = client.post("/v1/music/play", json=query,
                             headers={"Authorization": "Bearer api-secret", "Origin": "https://evil.example"})
        self.assertEqual(result.status_code, 403)
        self.assertEqual(self.provider.searches, [])
        self.assertEqual(self.provider.plays, [])
        self.assertEqual(client.get("/v1/music/status", headers={"Authorization": "Bearer api-secret"}).status_code, 200)
        schema = client.get("/openapi.json").json()
        self.assertEqual(schema["paths"]["/v1/music/play"]["post"]["security"], [{"HTTPBearer": []}])

    def test_bodyless_controls_reject_other_local_origins_even_with_valid_token(self):
        for token in ("", "api-secret"):
            with self.subTest(authenticated=bool(token)):
                client = TestClient(create_app(self.home, token=token, music=self.music),
                                    base_url="http://127.0.0.1:8765")
                authorization = {"Authorization": "Bearer " + token} if token else {}
                with patch.object(self.music, "pause", return_value={"status": "paused"}) as pause, \
                        patch.object(self.music, "resume", return_value={"status": "resumed"}) as resume:
                    for path in ("/v1/music/pause", "/v1/music/resume"):
                        for origin in ("http://127.0.0.1:9999", "http://localhost:8765",
                                       "https://127.0.0.1:8765"):
                            response = client.post(path, headers={**authorization, "Origin": origin})
                            self.assertEqual(response.status_code, 403)
                            self.assertNotIn("access-control-allow-origin", response.headers)
                    pause.assert_not_called()
                    resume.assert_not_called()
                    for path, action in (("/v1/music/pause", pause), ("/v1/music/resume", resume)):
                        response = client.post(path, headers={**authorization, "Origin": "http://127.0.0.1:8765"})
                        self.assertEqual(response.status_code, 200)
                        action.assert_called_once_with()
                        response = client.post(path, headers=authorization)
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(action.call_count, 2)

    def test_music_is_injected_into_combined_task_context_and_closed_once(self):
        from apm.home import HomeController
        from apm.music import MusicService
        from apm.tasks import TaskService
        from apm.toolsets.music import MUSIC_TOOLS

        with tempfile.TemporaryDirectory(prefix=".test-music-api-", dir=Path(__file__).resolve().parents[1]) as directory:
            tasks = TaskService(Path(directory) / "tasks.sqlite3", timezone="UTC")
            provider = FakeProvider()
            music = MusicService(provider)
            with patch.object(music, "close", wraps=music.close) as close_music:
                with TestClient(create_app(HomeController(), tasks=tasks, music=music)) as client:
                    result = client.get("/v1/context")
                    self.assertEqual(result.status_code, 200, result.text)
                    context = result.json()
                    self.assertEqual(context["music"]["provider"], "Test Music")
                    names = [item["function"]["name"] for item in context["tools"]]
                    for tool in MUSIC_TOOLS:
                        self.assertEqual(names.count(tool["function"]["name"]), 1)
                    self.assertIn("create_timer", names)
                close_music.assert_called_once()
            self.assertEqual(provider.closed, 1)

    def test_home_only_lifespan_closes_music_exactly_once(self):
        from apm.home import HomeController
        from apm.music import MusicService

        provider = FakeProvider()
        music = MusicService(provider)
        with patch.object(music, "close", wraps=music.close) as close_music:
            with TestClient(create_app(HomeController(), music=music)):
                self.assertEqual(provider.closed, 0)
            close_music.assert_called_once()
        self.assertEqual(provider.closed, 1)


if __name__ == "__main__":
    unittest.main()
