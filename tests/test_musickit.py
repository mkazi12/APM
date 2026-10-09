"""Fake Apple catalog and browser transports; no API calls or audio playback."""
from copy import deepcopy
import threading
import time
import unittest
from urllib.parse import parse_qs, urlsplit
import uuid

from apm.music import MusicPauseError, MusicPlaybackError, MusicService, MusicUnavailable, PlaybackResult
from apm.musickit import MusicKitProvider, validate_completion_diagnostics


def song(identifier="123456789", *, title="A Ballad", artist="A Singer"):
    return {"id": identifier, "type": "songs", "attributes": {
        "name": title, "artistName": artist, "albumName": "A Studio Album",
        "playParams": {"id": identifier, "kind": "song"},
        "previews": [{"url": "https://example.invalid/preview.m4a"}]}}


def catalog(*songs):
    return {"results": {"songs": {"data": list(songs)}}}


def diagnostics(**fields):
    return {"phase":"confirm", "reason":"confirmed", "elapsed_ms":200, "phase_ms":100, **fields}


class Transport:
    def __init__(self, pages=None):
        self.pages = [catalog(song())] if pages is None else pages
        self.calls = []

    def __call__(self, url, *, headers, timeout):
        self.calls.append((url, dict(headers), timeout))
        page = self.pages[min(len(self.calls) - 1, len(self.pages) - 1)]
        if isinstance(page, Exception):
            raise page
        return deepcopy(page)


class MusicKitTests(unittest.TestCase):
    def provider(self, pages=None, *, clock=None, timeout=1):
        transport = Transport(pages)
        provider = MusicKitProvider(lambda: "synthetic-developer-token", request=transport,
                                    now=time.monotonic if clock is None else lambda: clock[0],
                                    command_timeout=timeout)
        self.addCleanup(provider.close)
        session = str(uuid.uuid4())
        provider.activate(session, "US")
        return provider, session, transport

    def start_play(self, provider, identifier="123456789"):
        return self.start_operation(lambda: provider.play(identifier), provider)

    def start_operation(self, operation, provider):
        result, errors = [], []

        def play():
            try:
                result.append(operation())
            except Exception as error:
                errors.append(error)

        worker = threading.Thread(target=play, daemon=True)
        worker.start()
        self.addCleanup(lambda: (provider.close(), worker.join(1)))
        return worker, result, errors

    def command(self, provider, session):
        command = provider.poll(session, wait_seconds=0.5)
        self.assertIsNotNone(command)
        return command

    def test_constructor_and_status_make_no_catalog_requests(self):
        transport = Transport()
        provider = MusicKitProvider(lambda: "token", request=transport)
        self.addCleanup(provider.close)
        self.assertEqual(provider.status(), {"connected": False, "storefront": None, "protocol_version": None, "supports_pause": False, "supports_resume": False})
        self.assertEqual(transport.calls, [])
        with self.assertRaises(RuntimeError):
            provider.search("A Ballad")

    def test_pause_disconnected_is_noop_without_catalog_or_command(self):
        transport = Transport()
        provider = MusicKitProvider(lambda: "token", request=transport)
        self.addCleanup(provider.close)
        result = MusicService(provider).pause()
        self.assertEqual((result["status"], result["reason"]), ("unavailable", "disconnected"))
        self.assertEqual(transport.calls, [])

    def test_legacy_player_requires_update_without_sending_pause_or_revoking_session(self):
        provider, session, transport = self.provider()
        provider.activate(session, "us", protocol_version=1)
        result = MusicService(provider).pause()
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["reason"], "player_update_required")
        self.assertIsNone(result["playing"])
        self.assertEqual(provider.status(), {"connected": True, "storefront": "us", "protocol_version": 1, "supports_pause": False, "supports_resume": False})
        self.assertIsNone(provider.poll(session, wait_seconds=0))
        self.assertEqual(transport.calls, [])
        provider.activate(session, "us", protocol_version=2)
        self.assertTrue(provider.status()["supports_pause"])
        for version in (True, "2", None, 0, 4):
            with self.subTest(version=version), self.assertRaises(ValueError):
                provider.activate(session, "us", protocol_version=version)

    def test_pause_preempts_dispatched_play_and_requires_browser_confirmation(self):
        provider, session, transport = self.provider(timeout=20)
        provider.search("A Ballad")
        playing, _, play_errors = self.start_play(provider)
        old = self.command(provider, session)
        result = []
        pausing = threading.Thread(target=lambda: result.append(MusicService(provider).pause()), daemon=True)
        pausing.start()
        command = self.command(provider, session)
        self.assertEqual(command["operation"], "pause")
        self.assertNotIn("track_id", command)
        self.assertLessEqual(command["expires_in_ms"], 1500)
        playing.join(0.5)
        self.assertFalse(playing.is_alive())
        self.assertEqual(len(play_errors), 1)
        self.assertTrue(pausing.is_alive())
        with self.assertRaises(ValueError):
            provider.complete(session, old["id"], {"accepted": True, "playing": True, "track_id": old["track_id"]})
        provider.complete(session, command["id"], {"accepted": True, "playing": False, "was_playing": True})
        pausing.join(0.5)
        self.assertFalse(pausing.is_alive())
        self.assertEqual(result[0]["status"], "paused")
        self.assertIs(result[0]["was_playing"], True)
        self.assertEqual(len(transport.calls), 1)

    def test_pause_timeout_is_bounded_with_frozen_clock_and_never_retried(self):
        provider = MusicKitProvider(lambda: "token", request=Transport(), now=lambda: 100, pause_timeout=0.02)
        self.addCleanup(provider.close)
        session = str(uuid.uuid4())
        provider.activate(session, "us")
        started = time.monotonic()
        result = MusicService(provider).pause()
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["reason"], "timeout")
        self.assertIsNone(provider.poll(session, wait_seconds=0))
        detail = provider.status()["last_command"]
        self.assertEqual({key: detail[key] for key in ("operation", "source", "phase", "reason", "dispatched", "late_completion")},
                         {"operation":"pause", "source":"bridge", "phase":"delivery", "reason":"timeout",
                          "dispatched":False, "late_completion":False})
        self.assertGreaterEqual(detail["elapsed_ms"], 20)

    def test_dispatched_pause_timeout_records_late_delivery_without_reviving_result(self):
        clock = [100.0]
        provider, session, _ = self.provider(clock=clock)
        worker, result, errors = self.start_operation(lambda: MusicService(provider).pause(), provider)
        command = self.command(provider, session)
        clock[0] += 2
        detail = provider.status()["last_command"]
        worker.join(1)
        self.assertEqual(errors, [])
        self.assertEqual(result[0]["reason"], "timeout")
        self.assertIsNone(result[0]["playing"])
        self.assertEqual((detail["phase"], detail["dispatched"], detail["late_completion"]), ("completion", True, False))
        with self.assertRaises(ValueError):
            provider.complete(session, str(uuid.uuid4()), {"accepted":True,"playing":False})
        self.assertEqual(provider.status()["last_command"], detail)
        with self.assertRaises(ValueError):
            provider.complete(session, command["id"], {"accepted":True,"playing":False},
                              diagnostics=diagnostics(state="paused"))
        self.assertEqual(provider.status()["last_command"], {**detail, "late_completion":True})
        self.assertNotIn(command["id"], str(provider.status()))
        self.assertNotIn(session, str(provider.status()))
        self.assertEqual(result[0]["reason"], "timeout")
        provider.activate(str(uuid.uuid4()), "us")
        self.assertNotIn("last_command", provider.status())

    def test_previous_expired_command_cannot_modify_newer_completion_diagnostics(self):
        clock = [100.0]
        provider, session, _ = self.provider(clock=clock)
        worker, _, _ = self.start_operation(provider.pause, provider)
        expired = self.command(provider, session)
        clock[0] += 2
        self.assertEqual(provider.status()["last_command"]["reason"], "timeout")
        worker.join(1)
        worker, _, errors = self.start_operation(provider.pause, provider)
        current = self.command(provider, session)
        detail = diagnostics(state="paused")
        provider.complete(session, current["id"], {"accepted":True,"playing":False}, diagnostics=detail)
        worker.join(1)
        self.assertEqual(errors, [])
        with self.assertRaises(ValueError):
            provider.complete(session, expired["id"], {"accepted":True,"playing":False})
        self.assertEqual(provider.status()["last_command"], {"operation":"pause", **detail})

    def test_resume_uses_current_browser_queue_without_catalog_search(self):
        for response, expected in (
            ({"accepted": True, "playing": True, "track_id": "999"}, PlaybackResult(True, True, "999")),
            ({"accepted": False, "playing": False, "track_id": None}, PlaybackResult(False, False, None)),
        ):
            with self.subTest(response=response):
                provider, session, transport = self.provider()
                worker, result, errors = self.start_operation(provider.resume, provider)
                command = self.command(provider, session)
                self.assertEqual(command["operation"], "resume")
                self.assertNotIn("track_id", command)
                self.assertIsNone(provider.poll(session, wait_seconds=0))
                self.assertTrue(worker.is_alive(), "Resume must wait for browser observation")
                with self.assertRaises(ValueError):
                    provider.complete(session, command["id"], {**response, "was_playing": False})
                provider.complete(session, command["id"], response)
                worker.join(1)
                self.assertFalse(worker.is_alive())
                self.assertEqual((result, errors), ([expected], []))
                self.assertEqual(transport.calls, [])

    def test_resume_legacy_clients_require_update_without_revoking_pause_support(self):
        provider, session, transport = self.provider()
        for version in (1, 2):
            with self.subTest(version=version):
                provider.activate(session, "us", protocol_version=version)
                with self.assertRaises(MusicPauseError) as error:
                    provider.resume()
                self.assertEqual(error.exception.reason, "player_update_required")
                self.assertEqual(provider.status()["supports_pause"], version == 2)
                self.assertFalse(provider.status()["supports_resume"])
                self.assertTrue(provider.status()["connected"])
                self.assertIsNone(provider.poll(session, wait_seconds=0))
        self.assertEqual(transport.calls, [])

    def test_pause_preempts_resume_and_rejects_its_late_completion(self):
        provider, session, transport = self.provider()
        resuming, resume_result, resume_errors = self.start_operation(provider.resume, provider)
        old = self.command(provider, session)
        pausing, result, errors = self.start_operation(provider.pause, provider)
        pause = self.command(provider, session)
        self.assertEqual(pause["operation"], "pause")
        resuming.join(0.5)
        self.assertFalse(resuming.is_alive())
        self.assertEqual(resume_result, [])
        self.assertEqual(len(resume_errors), 1)
        with self.assertRaises(ValueError):
            provider.complete(session, old["id"], {"accepted": True, "playing": True, "track_id": "999"})
        provider.complete(session, pause["id"], {"accepted": True, "playing": False, "was_playing": True})
        pausing.join(0.5)
        self.assertEqual((result, errors), ([PlaybackResult(True, False, None, True)], []))
        self.assertEqual(transport.calls, [])

    def test_resume_timeout_disconnect_and_errors_do_not_retry(self):
        provider, session, transport = self.provider(clock=[100], timeout=0.02)
        with self.assertRaisesRegex(MusicPlaybackError, "timeout"):
            provider.resume()
        self.assertIsNone(provider.poll(session, wait_seconds=0))
        self.assertEqual(transport.calls, [])
        for action in ("disconnect", "error"):
            with self.subTest(action=action):
                provider, session, transport = self.provider()
                worker, result, errors = self.start_operation(provider.resume, provider)
                command = self.command(provider, session)
                if action == "disconnect":
                    provider.disconnect(session)
                    with self.assertRaises(MusicUnavailable):
                        provider.resume()
                else:
                    provider.complete(session, command["id"], {"error": "playback_unconfirmed"})
                    self.assertIsNone(provider.poll(session, wait_seconds=0))
                worker.join(0.5)
                self.assertFalse(worker.is_alive())
                self.assertEqual(result, [])
                self.assertEqual(len(errors), 1)
                self.assertEqual(transport.calls, [])

    def test_catalog_uses_fixed_https_endpoint_and_only_developer_token(self):
        provider, _, transport = self.provider()
        tracks = provider.search("A Ballad", "A Singer")
        self.assertEqual([(track.id, track.title, track.artists, track.album) for track in tracks],
                         [("123456789", "A Ballad", ("A Singer",), "A Studio Album")])
        self.assertTrue(tracks[0].playable)
        self.assertEqual(len(transport.calls), 1)
        url, headers, timeout = transport.calls[0]
        parsed = urlsplit(url)
        self.assertEqual((parsed.scheme, parsed.netloc, parsed.path),
                         ("https", "api.music.apple.com", "/v1/catalog/us/search"))
        self.assertEqual(parse_qs(parsed.query), {"term": ["A Ballad A Singer"], "types": ["songs"], "limit": ["25"]})
        self.assertEqual(headers, {"Authorization": "Bearer synthetic-developer-token", "Accept": "application/json"})
        self.assertEqual(timeout, 10)

    def test_title_only_fallback_recovers_from_artist_typo_or_wrong_cover(self):
        cover = song("222", artist="Another Singer")
        original = song("333", title="I'm On Fire", artist="Bruce Springsteen")
        provider, _, transport = self.provider([catalog(cover), catalog(original)])
        tracks = provider.search("I'm on Fire", "Bruce Springstein")
        self.assertEqual([track.id for track in tracks], ["222", "333"])
        self.assertEqual([parse_qs(urlsplit(call[0]).query)["term"][0] for call in transport.calls],
                         ["I'm on Fire Bruce Springstein", "I'm on Fire"])

    def test_empty_fallback_is_bounded_to_two_requests(self):
        provider, _, transport = self.provider([catalog()])
        self.assertEqual(provider.search("Missing song", "Missing artist"), [])
        self.assertEqual(len(transport.calls), 2)
        provider.search("Missing song")
        self.assertEqual(len(transport.calls), 3)

    def test_fallback_also_respects_specific_edition_requests(self):
        wrong = song("111", title="A Ballad (2020 Remaster)")
        correct = song("222", title="A Ballad (2010 Remaster)")
        provider, _, transport = self.provider([catalog(wrong), catalog(correct)])
        result = MusicService(provider).resolve("A Ballad (2010 Remaster)", "A Singer")
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["track"]["title"], correct["attributes"]["name"])
        self.assertEqual(len(transport.calls), 2)

    def test_preview_only_library_unplayable_and_malformed_songs_are_excluded(self):
        preview = song("101")
        del preview["attributes"]["playParams"]
        library = song("102")
        library["attributes"]["playParams"]["isLibrary"] = True
        unavailable = song("103")
        unavailable["attributes"]["isPlayable"] = False
        mismatched = song("104")
        mismatched["attributes"]["playParams"]["id"] = "999"
        video = song("105")
        video["type"] = "music-videos"
        bad_id = song("https://example.invalid/track")
        bad_title = song("106", title="")
        provider, _, _ = self.provider([catalog(preview, library, unavailable, mismatched, video, bad_id, bad_title, song())])
        self.assertEqual([track.id for track in provider.search("A Ballad")], ["123456789"])

    def test_search_input_cannot_change_endpoint(self):
        provider, _, transport = self.provider([catalog()])
        title = "https://example.invalid/?secret=a&types=albums"
        provider.search(title)
        parsed = urlsplit(transport.calls[0][0])
        self.assertEqual(parsed.netloc, "api.music.apple.com")
        self.assertEqual(parse_qs(parsed.query)["term"], [title])

    def test_invalid_catalog_and_auth_errors_are_sanitized_by_service(self):
        for page in (None, {}, {"results": []}, {"results": {"songs": {"data": None}}},
                     catalog(*[song(str(1000 + index)) for index in range(51)]),
                     PermissionError("token=synthetic-secret-token")):
            with self.subTest(page=page):
                provider, _, _ = self.provider([page])
                result = MusicService(provider).resolve("A Ballad")
                self.assertEqual(result["status"], "failed")
                self.assertNotIn("synthetic-secret-token", str(result))

    def test_search_deduplicates_ids_and_rejects_conflicting_metadata(self):
        provider, _, _ = self.provider([catalog(song(), song())])
        self.assertEqual(len(provider.search("A Ballad")), 1)
        provider, _, _ = self.provider([catalog(song(), song(artist="Different Singer"))])
        with self.assertRaises(ValueError):
            provider.search("A Ballad")

    def test_play_requires_id_returned_by_this_session(self):
        provider, _, _ = self.provider()
        for identifier in ("123456789", "https://example.invalid/song", "i.Library", "0", 123):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                provider.play(identifier)

    def test_browser_command_dispatches_once_and_observation_roundtrips(self):
        provider, session, _ = self.provider()
        provider.search("A Ballad")
        worker, result, errors = self.start_play(provider)
        command = self.command(provider, session)
        self.assertEqual(command["operation"], "play")
        self.assertEqual(command["track_id"], "123456789")
        self.assertGreater(command["expires_in_ms"], 0)
        self.assertLessEqual(command["expires_in_ms"], 1000)
        self.assertEqual(str(uuid.UUID(command["id"])), command["id"])
        self.assertIsNone(provider.poll(session, wait_seconds=0))
        provider.complete(session, command["id"], {"accepted": True, "playing": True, "track_id": command["track_id"]})
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(result, [PlaybackResult(True, True, "123456789")])
        with self.assertRaises(ValueError):
            provider.complete(session, command["id"], {"accepted": True})

    def test_concurrent_browser_polls_cannot_duplicate_command(self):
        provider, session, _ = self.provider()
        provider.search("A Ballad")
        worker, _, _ = self.start_play(provider)
        barrier = threading.Barrier(3)
        commands = []

        def poll():
            barrier.wait()
            commands.append(provider.poll(session, wait_seconds=0.1))

        readers = [threading.Thread(target=poll) for _ in range(2)]
        for reader in readers:
            reader.start()
        barrier.wait()
        for reader in readers:
            reader.join(1)
            self.assertFalse(reader.is_alive())
        dispatched = [command for command in commands if command is not None]
        self.assertEqual(len(dispatched), 1)
        provider.complete(session, dispatched[0]["id"], {"accepted": True})
        worker.join(1)

    def test_browser_acknowledgment_does_not_invent_observed_playback(self):
        provider, session, _ = self.provider()
        provider.search("A Ballad")
        worker, result, errors = self.start_play(provider)
        command = self.command(provider, session)
        provider.complete(session, command["id"], {"accepted": True})
        worker.join(1)
        self.assertEqual(errors, [])
        self.assertEqual(result, [PlaybackResult(True, None, None)])

    def test_browser_uncertain_error_unblocks_without_retry(self):
        provider, session, _ = self.provider()
        provider.search("A Ballad")
        worker, result, errors = self.start_play(provider)
        command = self.command(provider, session)
        provider.complete(session, command["id"], {"error": "playback_unconfirmed"})
        worker.join(1)
        self.assertEqual(result, [])
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertIsNone(provider.poll(session, wait_seconds=0))

    def test_safe_browser_failure_diagnostics_propagate_once_without_identifiers(self):
        provider, session, _ = self.provider()
        worker, result, errors = self.start_operation(provider.resume, provider)
        command = self.command(provider, session)
        detail = diagnostics(phase="play", reason="autoplay_blocked", sdk_reason="USER_INTERACTION_REQUIRED",
                             queue_check="matched", state="paused", track_matches=True)
        provider.complete(session, command["id"], {"error":"playback_unconfirmed"}, diagnostics=detail)
        worker.join(1)
        self.assertEqual(result, [])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], MusicPlaybackError)
        self.assertEqual(errors[0].reason, "autoplay_blocked")
        self.assertIsNone(provider.poll(session, wait_seconds=0))
        expected = {"operation":"resume", **detail}
        self.assertEqual(provider.status()["last_command"], expected)
        self.assertNotIn(command["id"], str(provider.status()))
        self.assertNotIn(session, str(provider.status()))
        detail["reason"] = "private-token"
        snapshot = provider.status()
        snapshot["last_command"]["reason"] = "private-token"
        self.assertEqual(provider.status()["last_command"], expected)

    def test_diagnostics_cannot_turn_acknowledgment_into_observed_playback(self):
        for outcome, detail, expected in (
            ({"accepted":True}, diagnostics(state="playing", track_matches=True), PlaybackResult(True, None, None)),
            ({"accepted":False,"playing":False,"track_id":None}, diagnostics(reason="queue_unavailable"),
             PlaybackResult(False, False, None)),
        ):
            with self.subTest(outcome=outcome):
                provider, session, _ = self.provider()
                worker, result, errors = self.start_operation(provider.resume, provider)
                command = self.command(provider, session)
                provider.complete(session, command["id"], outcome, diagnostics=detail)
                worker.join(1)
                self.assertEqual(errors, [])
                self.assertEqual(result, [expected])

    def test_invalid_diagnostics_are_rejected_without_consuming_pending_command(self):
        provider, session, _ = self.provider()
        worker, result, errors = self.start_operation(provider.resume, provider)
        command = self.command(provider, session)
        invalid = [[], {}, diagnostics(url="private-token"), diagnostics(phase="private-token"),
                   diagnostics(reason="private-token"), diagnostics(sdk_reason="private-token"),
                   diagnostics(sdk_reason=None), diagnostics(queue_check="private-token"),
                   diagnostics(state="private-token"), diagnostics(track_matches=1),
                   diagnostics(elapsed_ms=True), diagnostics(elapsed_ms=200.0),
                   diagnostics(elapsed_ms="200"), diagnostics(elapsed_ms=-1),
                   diagnostics(elapsed_ms=60001), diagnostics(phase_ms=True),
                   diagnostics(phase_ms=201)]
        for detail in invalid:
            with self.subTest(detail=detail), self.assertRaisesRegex(ValueError, "Invalid music command diagnostics"):
                provider.complete(session, command["id"], {"accepted":True}, diagnostics=detail)
        self.assertTrue(worker.is_alive())
        self.assertNotIn("last_command", provider.status())
        provider.complete(session, command["id"], {"accepted":False,"playing":False},
                          diagnostics=diagnostics(track_matches=None))
        worker.join(1)
        self.assertEqual((result, errors), ([PlaybackResult(False, False, None)], []))

    def test_latest_completion_without_diagnostics_and_session_revocation_clear_summary(self):
        for action in ("no_diagnostics", "replace", "disconnect", "expire", "close"):
            with self.subTest(action=action):
                clock = [100.0]
                provider, session, _ = self.provider(clock=clock)
                worker, _, errors = self.start_operation(provider.resume, provider)
                command = self.command(provider, session)
                provider.complete(session, command["id"], {"accepted":True}, diagnostics=diagnostics())
                worker.join(1)
                self.assertEqual(errors, [])
                self.assertIn("last_command", provider.status())
                if action == "no_diagnostics":
                    worker, _, _ = self.start_operation(provider.resume, provider)
                    command = self.command(provider, session)
                    provider.complete(session, command["id"], {"accepted":True})
                    worker.join(1)
                elif action == "replace":
                    provider.activate(str(uuid.uuid4()), "us")
                elif action == "disconnect":
                    provider.disconnect(session)
                elif action == "expire":
                    clock[0] += 30
                else:
                    provider.close()
                self.assertNotIn("last_command", provider.status())

    def test_late_unknown_or_undispatched_completion_cannot_publish_diagnostics(self):
        clock = [100.0]
        provider, session, _ = self.provider(clock=clock)
        worker, result, errors = self.start_operation(provider.resume, provider)
        with provider._condition:
            self.assertTrue(provider._condition.wait_for(lambda: bool(provider._pending), timeout=0.5))
            identifier = next(iter(provider._pending))
        with self.assertRaises(ValueError):
            provider.complete(session, identifier, {"accepted":True}, diagnostics=diagnostics())
        with self.assertRaises(ValueError):
            provider.complete(session, str(uuid.uuid4()), {"accepted":True}, diagnostics=diagnostics())
        command = self.command(provider, session)
        clock[0] += 2
        with self.assertRaises(ValueError):
            provider.complete(session, command["id"], {"accepted":True}, diagnostics=diagnostics())
        worker.join(1)
        self.assertEqual(result, [])
        self.assertEqual(len(errors), 1)
        self.assertEqual(provider.status()["last_command"]["source"], "bridge")
        self.assertEqual(provider.status()["last_command"]["phase"], "completion")
        self.assertTrue(provider.status()["last_command"]["late_completion"])

    def test_diagnostic_validator_accepts_boundaries_without_retaining_extra_data(self):
        for detail in (diagnostics(elapsed_ms=0, phase_ms=0),
                       diagnostics(elapsed_ms=60000, phase_ms=60000, track_matches=False)):
            self.assertEqual(validate_completion_diagnostics(detail), detail)

    def test_timeout_is_bounded_even_with_frozen_clock(self):
        provider, session, _ = self.provider(clock=[100.0], timeout=0.02)
        provider.search("A Ballad")
        started = time.monotonic()
        with self.assertRaisesRegex(MusicPlaybackError, "timeout"):
            provider.play("123456789")
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertIsNone(provider.poll(session, wait_seconds=0))

    def test_late_complete_cannot_revive_timed_out_command(self):
        clock = [100.0]
        provider, session, _ = self.provider(clock=clock)
        provider.search("A Ballad")
        worker, result, errors = self.start_play(provider)
        command = self.command(provider, session)
        clock[0] += 2
        with self.assertRaises(ValueError):
            provider.complete(session, command["id"], {"accepted": True, "playing": True, "track_id": command["track_id"]})
        worker.join(1)
        self.assertEqual(result, [])
        self.assertIsInstance(errors[0], RuntimeError)

    def test_player_expires_and_poll_renews_connection(self):
        clock = [100.0]
        provider, session, _ = self.provider(clock=clock)
        clock[0] += 29
        self.assertIsNone(provider.poll(session, wait_seconds=0))
        clock[0] += 29
        self.assertEqual(provider.status(), {"connected": True, "storefront": "us", "protocol_version": 3, "supports_pause": True, "supports_resume": True})
        clock[0] += 1
        self.assertEqual(provider.status(), {"connected": False, "storefront": None, "protocol_version": None, "supports_pause": False, "supports_resume": False})
        with self.assertRaises(ValueError):
            provider.poll(session, wait_seconds=0)
        provider.activate(session, "gb")
        self.assertEqual(provider.status(), {"connected": True, "storefront": "gb", "protocol_version": 3, "supports_pause": True, "supports_resume": True})

    def test_expired_player_cancels_pending_waiter(self):
        clock = [100.0]
        provider, session, _ = self.provider(clock=clock, timeout=60)
        provider.search("A Ballad")
        worker, result, errors = self.start_play(provider)
        self.command(provider, session)
        clock[0] += 30
        self.assertFalse(provider.status()["connected"])
        worker.join(1)
        self.assertEqual(result, [])
        self.assertRegex(str(errors[0]), "expired")

    def test_replacement_disconnect_and_close_cancel_waiters(self):
        for action in ("replace", "disconnect", "close"):
            with self.subTest(action=action):
                provider, session, _ = self.provider()
                provider.search("A Ballad")
                worker, result, errors = self.start_play(provider)
                command = self.command(provider, session)
                if action == "replace":
                    provider.activate(str(uuid.uuid4()), "gb")
                elif action == "disconnect":
                    provider.disconnect(session)
                else:
                    provider.close()
                    provider.close()
                worker.join(1)
                self.assertFalse(worker.is_alive())
                self.assertEqual(result, [])
                self.assertIsInstance(errors[0], RuntimeError)
                with self.assertRaises(ValueError):
                    provider.complete(session, command["id"], {"accepted": True})

    def test_superseded_search_cannot_authorize_tracks_for_new_player(self):
        provider, _, _ = self.provider()

        def replace_session(url, *, headers, timeout):
            provider.activate(str(uuid.uuid4()), "gb")
            return catalog(song())

        provider._request = replace_session
        with self.assertRaises(ValueError):
            provider.search("A Ballad")
        with self.assertRaises(ValueError):
            provider.play("123456789")

    def test_queue_is_bounded_to_eight_pending_commands(self):
        provider, session, _ = self.provider(timeout=5)
        provider.search("A Ballad")
        workers = []
        for _ in range(8):
            workers.append(self.start_play(provider)[0])
            self.command(provider, session)
        with self.assertRaisesRegex(RuntimeError, "queue is busy"):
            provider.play("123456789")
        provider.disconnect(session)
        for worker in workers:
            worker.join(1)
            self.assertFalse(worker.is_alive())

    def test_result_and_session_validation_cannot_complete_unknown_command(self):
        provider, session, _ = self.provider()
        for storefront in ("", "usa", "../", "u1", "üS"):
            with self.subTest(storefront=storefront), self.assertRaises(ValueError):
                provider.activate(session, storefront)
        for result in ({"accepted": 1}, {"accepted": True, "playing": "yes"},
                       {"accepted": True, "track_id": "https://example.invalid"},
                       {"error": "secret-token"}, {"error": "playback_unconfirmed", "token": "secret"}, []):
            with self.subTest(result=result), self.assertRaises(ValueError):
                provider.complete(session, str(uuid.uuid4()), result)
        with self.assertRaises(ValueError):
            provider.complete(session, str(uuid.uuid4()), {"accepted": True})
        with self.assertRaises(ValueError):
            provider.poll(str(uuid.uuid4()), wait_seconds=0)
        with self.assertRaises(ValueError):
            provider.disconnect(str(uuid.uuid4()))
        self.assertTrue(provider.status()["connected"])

    def test_timeouts_reject_nonfinite_boolean_and_unbounded_values(self):
        provider, session, _ = self.provider()
        for value in (True, -1, float("nan"), float("inf"), 10 ** 1000, "15", 16):
            with self.subTest(value=str(value)[:20]), self.assertRaises(ValueError):
                provider.poll(session, wait_seconds=value)
        for value in (0, True, -1, float("nan"), 61, 10 ** 1000):
            with self.subTest(value=str(value)[:20]), self.assertRaises(ValueError):
                MusicKitProvider(lambda: "token", request=Transport(), command_timeout=value)


if __name__ == "__main__":
    unittest.main()
