"""Assistant/music integration with fake catalog, playback, and inference only."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import io
import json
import unittest
from unittest.mock import Mock

from jsonschema import ValidationError, validate

from apm.assistant import AssistantController
from apm.cli import describe_results, run_request
from apm.home import HomeController
from apm.music import MusicService, PlaybackResult, Track
from apm.ollama_backend import OllamaBackend
from apm.toolsets.music import MUSIC_TOOLS


# Metadata verified against the artist's official catalog, not a playable ID:
# https://brucespringsteen.net/track/im-on-fire/
# https://brucespringsteen.net/albums/born-in-the-u-s-a/
TRACK = Track("fixture-bruce-studio", "I'm On Fire", ("Bruce Springsteen",),
              album="Born in the U.S.A.")


def call(name, **arguments):
    return {"function": {"name": name, "arguments": arguments}}


class FakeTasks:
    def list_tasks(self):
        return []

    def clock(self):
        return {"utc": datetime(2026, 10, 7, 12, tzinfo=timezone.utc).isoformat(), "timezone": "UTC"}


class FakeProvider:
    name = "fixture-catalog"

    def __init__(self, tracks=(TRACK,), playing=True):
        self.tracks = list(tracks)
        self.playing = playing
        self.searches, self.plays = [], []
        self.pauses = self.resumes = 0

    def search(self, title, artist=None):
        self.searches.append((title, artist))
        return list(self.tracks)

    def play(self, track_id):
        self.plays.append(track_id)
        return PlaybackResult(accepted=True, playing=self.playing,
                              track_id=track_id if self.playing else None)

    def pause(self):
        self.pauses += 1
        was_playing = self.playing
        self.playing = False
        return PlaybackResult(True, False, was_playing=was_playing)

    def resume(self):
        self.resumes += 1
        self.playing = True
        return PlaybackResult(True, True, track_id=TRACK.id)

    def close(self):
        pass


class FakeTransport:
    def __init__(self, calls):
        self.calls, self.payloads = calls, []

    def stream(self, _path, payload):
        self.payloads.append(deepcopy(payload))
        yield {"message": {"tool_calls": deepcopy(self.calls)}, "done": True}


class MusicContextTests(unittest.TestCase):
    def assistant(self, provider=None):
        home = HomeController()
        music = MusicService(provider=provider)
        assistant = AssistantController(home, FakeTasks(), music=music)
        self.addCleanup(assistant.close)
        return assistant

    def test_tools_reject_missing_titles_blank_queries_and_invented_provider_fields(self):
        schemas = {item["function"]["name"]: item["function"]["parameters"] for item in MUSIC_TOOLS}
        for name in ("resolve_music", "play_music"):
            validate({"title": "I'm on fire"}, schemas[name])
            validate({"title": "I'm on fire", "artist": "Bruce Springsteen", "version": "live"}, schemas[name])
            for invalid in ({}, {"title": " "}, {"title": "x" * 1000},
                            {"title": "Song", "artist": ""}, {"title": "Song", "version": "whatever"},
                            {"title": "Song", "track_id": "invented"}):
                with self.subTest(name=name, invalid=invalid), self.assertRaises(ValidationError):
                    validate(invalid, schemas[name])
        with self.assertRaises(ValidationError):
            validate({"track_id": TRACK.id}, schemas["play_music_selection"])

    def test_music_controls_take_no_ids_and_never_enter_scheduling(self):
        provider = FakeProvider()
        assistant = self.assistant(provider)
        assistant.tasks.validate_request = Mock(side_effect=AssertionError('Music must not use a task UUID'))
        model = OllamaBackend(transport=FakeTransport([call('pause_music')]))
        with redirect_stdout(io.StringIO()):
            pause = run_request(model, assistant, text='Hey Gemma pause the music')
        self.assertEqual(pause, 'Music paused.')
        self.assertFalse(pause.expects_reply)
        model.transport = FakeTransport([call('resume_music')])
        with redirect_stdout(io.StringIO()):
            resume = run_request(model, assistant, text='Continue playing the music')
        self.assertEqual(resume, 'Music resumed.')
        self.assertFalse(resume.expects_reply)
        self.assertEqual((provider.pauses, provider.resumes), (1, 1))
        self.assertEqual(provider.searches, [])
        self.assertEqual(provider.plays, [])
        assistant.tasks.validate_request.assert_not_called()

    def test_music_controls_reject_arguments_before_any_batch_action(self):
        provider = FakeProvider()
        assistant = self.assistant(provider)
        for name in ('pause_music', 'resume_music'):
            for extra in ({'task_id': 'music'}, {'title': 'Doppler'}, {'track_id': TRACK.id}):
                with self.subTest(name=name, extra=extra), self.assertRaises(ValueError):
                    assistant.execute([call('set_lights', device='kitchen_lights', on=False), call(name, **extra)])
        self.assertEqual(assistant.home.get_state('kitchen_lights')['state'], 'on')
        self.assertEqual((provider.pauses, provider.resumes), (0, 0))

    def test_confirmed_controls_allow_later_batch_actions(self):
        provider = FakeProvider()
        assistant = self.assistant(provider)
        results = assistant.execute([call('pause_music'), call('resume_music'),
                                     call('set_lights', device='kitchen_lights', on=False)])
        self.assertTrue(all(result['ok'] for result in results))
        self.assertEqual(assistant.home.get_state('kitchen_lights')['state'], 'off')

    def test_unavailable_controls_do_not_claim_success_or_request_an_answer(self):
        assistant = self.assistant()
        for name in ('pause_music', 'resume_music'):
            with self.subTest(name=name):
                model = OllamaBackend(transport=FakeTransport([call(name)]))
                with redirect_stdout(io.StringIO()):
                    reply = run_request(model, assistant, text='Pause or resume the music')
                self.assertFalse(reply.expects_reply)
                result = json.loads(model.turns[-1][-1]['content'])
                self.assertFalse(result['ok'])
                self.assertEqual(result['data']['status'], 'unavailable')
                self.assertNotIn(reply, ('Music paused.', 'Music resumed.'))

    def test_unconfigured_music_is_exposed_to_model_and_reports_no_playback(self):
        assistant = self.assistant()
        context = assistant.context()
        self.assertFalse(context["music"]["configured"])
        self.assertTrue({item["function"]["name"] for item in MUSIC_TOOLS}.issubset(
            {item["function"]["name"] for item in context["tools"]}))
        transport = FakeTransport([call("play_music", title="I'm on fire", artist="Bruce Springsteen")])
        model = OllamaBackend(transport=transport)
        with redirect_stdout(io.StringIO()):
            reply = run_request(model, assistant, text="Play I'm on fire by Bruce Springsteen")
        self.assertEqual(transport.payloads[0]["tools"], context["tools"])
        self.assertIn("music", transport.payloads[0]["messages"][0]["content"].lower())
        result = json.loads(model.turns[-1][-1]["content"])
        self.assertFalse(result["ok"])
        self.assertEqual(result["data"]["status"], "not_configured")
        self.assertIn("no music provider", reply.lower())
        self.assertNotIn("playing", reply.lower())

    def test_misspelled_artist_resolves_catalog_metadata_without_playing(self):
        provider = FakeProvider()
        assistant = self.assistant(provider)
        result = assistant.execute([call("resolve_music", title="i'm on fire", artist="bruce springstein")])[0]
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["status"], "matched")
        track = result["data"]["track"]
        self.assertEqual(track["title"], TRACK.title)
        self.assertEqual(track["artists"], ["Bruce Springsteen"])
        self.assertNotEqual(track["selection_id"], TRACK.id)
        self.assertNotIn(TRACK.id, json.dumps(result))
        self.assertEqual(provider.plays, [])

    def test_model_tool_dispatch_plays_first_compilation_once_without_album_choices(self):
        editions = [replace(TRACK, id=f"fixture-edition-{index}", album=f"Compilation {index}")
                    for index in range(24, -1, -1)]
        provider = FakeProvider(editions)
        assistant = self.assistant(provider)
        transport = FakeTransport([call("play_music", title="i'm on fire", artist="bruce springstein")])
        model = OllamaBackend(transport=transport)
        with redirect_stdout(io.StringIO()):
            reply = run_request(model, assistant, text="play i'm on fire by bruce springstein")
        self.assertEqual(provider.plays, [editions[0].id])
        self.assertEqual(len(transport.payloads), 1)
        result = json.loads(model.turns[-1][-1]["content"])
        self.assertTrue(result["data"]["playing"])
        self.assertNotIn("candidates", result["data"])
        self.assertEqual(result["data"]["track"]["album"], "Compilation 24")
        self.assertEqual(result["data"]["track"]["artists"], ["Bruce Springsteen"])
        self.assertIn("Bruce Springsteen", reply)
        self.assertFalse(reply.expects_reply)
        self.assertNotIn("choose", reply.lower())
        self.assertNotIn(editions[0].id, json.dumps(model.turns))

    def test_delivered_music_clarification_requests_a_voice_followup(self):
        live = replace(TRACK, id="fixture-live", title="I'm On Fire (Live)", version="live")
        provider = FakeProvider([live])
        assistant = self.assistant(provider)
        model = OllamaBackend(transport=FakeTransport([call("play_music", title=TRACK.title)]))
        with redirect_stdout(io.StringIO()):
            reply = run_request(model, assistant, text="Play I'm On Fire")
        self.assertTrue(reply.expects_reply)
        self.assertEqual(provider.plays, [])

    def test_song_title_question_does_not_request_a_voice_followup(self):
        song = replace(TRACK, title="Who Are You?", artists=("The Who",))
        provider = FakeProvider([song])
        assistant = self.assistant(provider)
        model = OllamaBackend(transport=FakeTransport([call("play_music", title=song.title)]))
        with redirect_stdout(io.StringIO()):
            reply = run_request(model, assistant, text="Play Who Are You")
        self.assertIn(song.title, reply)
        self.assertFalse(reply.expects_reply)
        self.assertEqual(provider.plays, [song.id])

    def test_ambiguous_versions_or_missing_catalog_results_never_play(self):
        live = replace(TRACK, id="fixture-live", title="I'm On Fire (Live)", version="live")
        unknown = replace(TRACK, id="fixture-unknown", version="unknown")
        for tracks in ((live,), (unknown,), ()):
            with self.subTest(tracks=tracks):
                provider = FakeProvider(tracks)
                assistant = self.assistant(provider)
                result = assistant.execute([call("play_music", title=TRACK.title)])[0]
                self.assertTrue(result["ok"])
                self.assertIn(result["data"]["status"], {"ambiguous", "not_found"})
                self.assertEqual(provider.plays, [])
                self.assertFalse(result["data"].get("playing", False))

    def test_user_selection_uses_returned_choice_and_cannot_be_replayed(self):
        another = Track("fixture-other-artist", TRACK.title, ("Fixture Artist",))
        provider = FakeProvider((TRACK, another))
        assistant = self.assistant(provider)
        resolved = assistant.execute([call("resolve_music", title=TRACK.title)])[0]["data"]
        chosen = next(track for track in resolved["candidates"] if track["artists"] == ["Bruce Springsteen"])
        selection_id = chosen["selection_id"]
        played = assistant.execute([call("play_music_selection", selection_id=selection_id)])[0]
        self.assertTrue(played["data"]["playing"])
        self.assertEqual(provider.plays, [TRACK.id])
        try:
            repeated = assistant.execute([call("play_music_selection", selection_id=selection_id)])[0]
        except ValueError:
            pass  # Preflight may reject a consumed selection before dispatch.
        else:
            self.assertFalse(repeated["ok"])
        self.assertEqual(provider.plays, [TRACK.id])

    def test_invalid_mixed_batch_cannot_play_music_or_change_home(self):
        provider = FakeProvider()
        assistant = self.assistant(provider)
        with self.assertRaises(ValueError):
            assistant.execute([call("set_lights", device="kitchen_lights", on=False),
                               call("play_music", title=TRACK.title),
                               call("play_music", title=TRACK.title, version="invented")])
        self.assertEqual(assistant.home.get_state("kitchen_lights")["state"], "on")
        self.assertEqual(provider.searches, [])
        self.assertEqual(provider.plays, [])

    def test_accepted_command_is_not_described_as_verified_playback(self):
        provider = FakeProvider(playing=None)
        assistant = self.assistant(provider)
        result = assistant.execute([call("play_music", title=TRACK.title, artist="Bruce Springsteen")])[0]
        self.assertTrue(result["data"]["accepted"])
        self.assertIsNone(result["data"]["playing"])
        self.assertFalse(describe_results([result]).lower().startswith("playing"))
        self.assertEqual(provider.plays, [TRACK.id])


if __name__ == "__main__":
    unittest.main()
