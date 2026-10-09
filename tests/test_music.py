"""Catalog fixtures use synthetic IDs; no provider or playback is contacted."""
from dataclasses import replace
import unittest
from unittest.mock import Mock
import uuid

from apm.music import MAX_SELECTIONS, MusicPauseError, MusicPlaybackError, MusicService, MusicUnavailable, PlaybackResult, Track


SONG = Track("synthetic-studio", "I'm On Fire", ("Bruce Springsteen",), "Born in the U.S.A.")


class Provider:
    name = "Test catalog"

    def __init__(self, tracks=None, result=None):
        self.tracks = [SONG] if tracks is None else tracks
        self.result = result
        self.searches = []
        self.plays = []
        self.closed = 0

    def search(self, title, artist=None):
        self.searches.append((title, artist))
        if isinstance(self.tracks, Exception):
            raise self.tracks
        return self.tracks

    def play(self, track_id):
        self.plays.append(track_id)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result if self.result is not None else PlaybackResult(True, True, track_id)

    def close(self):
        self.closed += 1


class MusicTests(unittest.TestCase):
    def test_pause_preserves_safe_browser_failure_reason_without_start_playback_advice(self):
        for reason in MusicPlaybackError.REASONS:
            with self.subTest(reason=reason):
                provider = Provider()
                provider.pause = Mock(side_effect=MusicPlaybackError(reason))
                result = MusicService(provider).pause()
                self.assertEqual((result["status"], result["reason"]), ("unknown", reason))
                self.assertIsNone(result["accepted"])
                self.assertIsNone(result["playing"])
                self.assertIn("pause", result["message"])
                self.assertNotIn("Start playback", result["message"])
                provider.pause.assert_called_once_with()
                self.assertEqual(provider.plays, [])

    def test_browser_failure_reason_is_preserved_without_claiming_success_or_retrying(self):
        for reason in MusicPlaybackError.REASONS:
            provider = Provider(result=MusicPlaybackError(reason))
            provider.resume = Mock(side_effect=MusicPlaybackError(reason))
            service = MusicService(provider)
            for result in (service.play(SONG.title, SONG.artists[0]), service.resume()):
                self.assertEqual((result["status"], result["reason"]), ("unknown", reason))
                self.assertIsNone(result["accepted"])
                self.assertIsNone(result["playing"])
                self.assertIn("not retried", result["message"])
            self.assertEqual(provider.plays, [SONG.id])
            provider.resume.assert_called_once_with()
        for reason in (None, {}, "token-secret", "confirmed"):
            with self.assertRaises(ValueError):
                MusicPlaybackError(reason)

    def test_playback_controls_accept_only_empty_arguments_without_provider_work(self):
        provider = Provider()
        provider.pause, provider.resume = Mock(), Mock()
        service = MusicService(provider)
        for operation in ("pause", "resume"):
            service.validate_request(operation, {})
            for arguments in (None, [], {"track_id": "123"}, {"title": "A Song"},
                              {"selection_id": str(uuid.uuid4())}):
                with self.subTest(operation=operation, arguments=arguments), self.assertRaises(ValueError):
                    service.validate_request(operation, arguments)
        provider.pause.assert_not_called()
        provider.resume.assert_not_called()
        self.assertEqual(provider.searches, [])

    def test_resume_uses_existing_queue_and_observed_identity_without_catalog_lookup(self):
        provider = Provider()
        provider.resume = Mock(return_value=PlaybackResult(True, True, "existing-queue-recording"))
        result = MusicService(provider).resume()
        self.assertEqual(result["status"], "resumed")
        self.assertIs(result["accepted"], True)
        self.assertIs(result["playing"], True)
        self.assertNotIn("track", result)
        self.assertNotIn("existing-queue-recording", str(result))
        provider.resume.assert_called_once_with()
        self.assertEqual(provider.searches, [])
        self.assertEqual(provider.plays, [])

    def test_resume_empty_queue_does_not_select_a_replacement_song(self):
        provider = Provider()
        provider.resume = Mock(return_value=PlaybackResult(False, False, None))
        result = MusicService(provider).resume()
        self.assertEqual(result["status"], "empty")
        self.assertIs(result["accepted"], False)
        self.assertIs(result["playing"], False)
        self.assertIn("no song", result["message"])
        provider.resume.assert_called_once_with()
        self.assertEqual(provider.searches, [])
        self.assertEqual(provider.plays, [])

    def test_resume_unavailable_reasons_are_explicit_and_helpful(self):
        disconnected = Provider()
        disconnected.resume = Mock(side_effect=MusicUnavailable("private-token"))
        for service, reason in ((MusicService(), "not_configured"),
                                (MusicService(Provider()), "unsupported"),
                                (MusicService(disconnected), "disconnected")):
            with self.subTest(reason=reason):
                result = service.resume()
                self.assertEqual((result["status"], result["reason"]), ("unavailable", reason))
                self.assertIs(result["accepted"], False)
                self.assertIsNone(result["playing"])
                self.assertTrue(result["message"])
                self.assertNotIn("private-token", str(result))

    def test_resume_unknown_or_malformed_observations_never_claim_playing_or_retry(self):
        outcomes = (PlaybackResult(True), PlaybackResult(True, False, "queue-item"),
                    PlaybackResult(True, True, None), PlaybackResult(False, True, "queue-item"),
                    PlaybackResult(True, True, ""), PlaybackResult(True, True, "x" * 257),
                    PlaybackResult(1, True, "queue-item"), PlaybackResult(True, 1, "queue-item"),
                    PlaybackResult(True, True, 123), PlaybackResult(True, True, "queue-item", False),
                    {"accepted":True,"playing":True}, TimeoutError("private-provider-token"))
        for outcome in outcomes:
            with self.subTest(outcome=type(outcome).__name__):
                provider = Provider()
                provider.resume = Mock(side_effect=outcome) if isinstance(outcome, Exception) else Mock(return_value=outcome)
                result = MusicService(provider).resume()
                self.assertEqual(result["status"], "unknown")
                self.assertIsNone(result["playing"])
                self.assertNotIn("private-provider-token", str(result))
                provider.resume.assert_called_once_with()
                self.assertEqual(provider.searches, [])
                self.assertEqual(provider.plays, [])

    def test_resume_old_player_requires_browser_refresh_and_closed_service_does_no_work(self):
        provider = Provider()
        provider.resume = Mock(side_effect=MusicPauseError("player_update_required"))
        service = MusicService(provider)
        result = service.resume()
        self.assertEqual((result["status"], result["reason"]), ("unknown", "player_update_required"))
        self.assertIn("Refresh the browser page", result["message"])
        self.assertIn("enable resume", result["message"])
        service.close()
        self.assertEqual(service.resume()["reason"], "not_configured")
        provider.resume.assert_called_once_with()

    def test_pause_without_provider_or_supported_player_is_a_safe_noop(self):
        for service, reason in ((MusicService(), "not_configured"), (MusicService(Provider()), "unsupported")):
            result = service.pause()
            self.assertEqual((result["status"], result["reason"]), ("unavailable", reason))
            self.assertFalse(result["accepted"])
            self.assertIsNone(result["playing"])

    def test_pause_requires_observation_and_preserves_previous_playback_state(self):
        for was_playing in (True, False, None):
            provider = Provider()
            provider.pause = Mock(return_value=PlaybackResult(True, False, was_playing=was_playing))
            result = MusicService(provider).pause()
            self.assertEqual(result["status"], "paused")
            self.assertIs(result["was_playing"], was_playing)
            self.assertIs(result["playing"], False)
            provider.pause.assert_called_once_with()
            self.assertEqual(provider.searches, [])
            self.assertEqual(provider.plays, [])

    def test_pause_failure_unconfirmed_and_disconnected_are_sanitized(self):
        for response in (PlaybackResult(True), PlaybackResult(True, True), PlaybackResult(True, False, was_playing=1),
                         {"playing": False}, TimeoutError("private-token"), MusicUnavailable("private-token")):
            provider = Provider()
            provider.pause = Mock(side_effect=response) if isinstance(response, Exception) else Mock(return_value=response)
            result = MusicService(provider).pause()
            self.assertEqual(result["status"], "unavailable" if isinstance(response, MusicUnavailable) else "unknown")
            self.assertNotIn("private-token", str(result))
            self.assertIsNone(result["playing"])
            provider.pause.assert_called_once_with()

    def test_missing_provider_is_explicit_and_offline(self):
        service = MusicService()
        self.assertEqual(service.status(), {"configured": False, "provider": None})
        self.assertEqual(service.resolve("I'm on fire", "Bruce Springsteen")["status"], "not_configured")
        self.assertEqual(service.play("I'm on fire")["status"], "not_configured")
        self.assertEqual(service.select(str(uuid.uuid4()))["status"], "not_configured")
        service.close()

    def test_requested_artist_typo_resolves_to_catalog_metadata(self):
        provider = Provider()
        service = MusicService(provider)
        resolved = service.resolve("i'm on fire", "bruce springstein")
        self.assertEqual(resolved["status"], "matched")
        self.assertEqual(resolved["track"]["title"], "I'm On Fire")
        self.assertEqual(resolved["track"]["artists"], ["Bruce Springsteen"])
        self.assertEqual(provider.plays, [])
        played = service.select(resolved["track"]["selection_id"])
        self.assertEqual(played["status"], "playing")
        self.assertTrue(played["playing"])
        self.assertEqual(provider.plays, [SONG.id])
        self.assertNotIn(SONG.id, str(resolved))
        self.assertNotIn(SONG.id, str(played))
        self.assertNotIn("selection_id", played["track"])

    def test_unicode_case_apostrophe_and_accents_normalize(self):
        service = MusicService(Provider())
        for title in ("I’m On Fire", "IM ON FIRE!", "I'm on fire"):
            with self.subTest(title=title):
                self.assertEqual(service.resolve(title, "BRÜCE SPRINGSTEEN")["status"], "matched")

    def test_title_containing_artist_recovers_after_literal_search_misses(self):
        track = Track("synthetic-threads", "Threads", ("ear",), "Rumspringa")
        for operation in ("resolve", "play"):
            with self.subTest(operation=operation):
                provider = Provider([track])
                result = getattr(MusicService(provider), operation)("Thread by Ear")
                self.assertEqual(provider.searches, [("Thread by Ear", None), ("Thread", "Ear")])
                self.assertEqual(result["status"], "matched" if operation == "resolve" else "playing")
                self.assertEqual(result["track"]["title"], "Threads")
                self.assertEqual(result["track"]["artists"], ["ear"])
                self.assertEqual(provider.plays, [] if operation == "resolve" else [track.id])

    def test_by_in_a_matching_literal_title_is_not_split(self):
        track = Track("synthetic-stand-by-me", "Stand by Me", ("A Singer",))
        provider = Provider([track])
        result = MusicService(provider).play("Stand by Me")
        self.assertEqual(result["status"], "playing")
        self.assertEqual(provider.searches, [("Stand by Me", None)])
        self.assertEqual(provider.plays, [track.id])

    def test_explicit_artist_prevents_reinterpreting_by_in_title(self):
        provider = Provider([Track("synthetic-threads", "Threads", ("ear",))])
        result = MusicService(provider).play("Thread by Ear", "ear")
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(provider.searches, [("Thread by Ear", "ear")])
        self.assertEqual(provider.plays, [])

    def test_artist_split_fallback_is_bounded_to_one_search_and_never_plays_no_match(self):
        provider = Provider([])
        result = MusicService(provider).play("A Song by Another Name by An Artist")
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(provider.searches, [("A Song by Another Name by An Artist", None),
                                            ("A Song by Another Name", "An Artist")])
        self.assertEqual(provider.plays, [])

    def test_cover_by_wrong_artist_never_auto_plays(self):
        provider = Provider([replace(SONG, id="synthetic-cover", artists=("Another Singer",))])
        self.assertEqual(MusicService(provider).play("I'm on Fire", "Bruce Springsteen")["status"], "not_found")
        self.assertEqual(provider.plays, [])

    def test_weak_title_match_never_auto_plays(self):
        provider = Provider([replace(SONG, title="Fire")])
        self.assertEqual(MusicService(provider).play("I'm on Fire", "Bruce Springsteen")["status"], "not_found")
        self.assertEqual(provider.plays, [])

    def test_unspecified_artist_browses_choices_but_play_uses_first_eligible_result(self):
        provider = Provider([SONG, replace(SONG, id="synthetic-cover", artists=("Another Singer",))])
        service = MusicService(provider)
        result = service.resolve("I'm on fire")
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(provider.plays, [])
        result = service.play("I'm on fire")
        self.assertEqual(result["status"], "playing")
        self.assertNotIn("candidates", result)
        self.assertEqual(provider.plays, [SONG.id])

    def test_close_competing_titles_browse_as_choices_but_play_uses_best_score(self):
        provider = Provider([replace(SONG, id="synthetic-other", title="I'm On Fires"), SONG])
        service = MusicService(provider)
        result = service.resolve("I'm on fire", "Bruce Springsteen")
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(provider.plays, [])
        result = service.play("I'm on fire", "Bruce Springsteen")
        self.assertEqual(result["status"], "playing")
        self.assertEqual(provider.plays, [SONG.id])

    def test_many_compilation_editions_play_provider_first_result_once(self):
        editions = [replace(SONG, id=f"synthetic-edition-{index}", album=f"Compilation {index}")
                    for index in range(24, -1, -1)]
        provider = Provider(editions)
        service = MusicService(provider)
        browsed = service.resolve("I'm on fire", "Bruce Springsteen")
        self.assertEqual(browsed["status"], "ambiguous")
        self.assertEqual(len(browsed["candidates"]), 10)
        self.assertEqual(provider.plays, [])
        played = service.play("I'm on fire", "Bruce Springsteen")
        self.assertEqual(played["status"], "playing")
        self.assertEqual(played["track"]["album"], "Compilation 24")
        self.assertNotIn("candidates", played)
        self.assertNotIn("selection_id", played["track"])
        self.assertEqual(provider.plays, [editions[0].id])

    def test_first_result_policy_still_filters_wrong_artist_and_unplayable_tracks(self):
        wrong_artist = replace(SONG, id="synthetic-cover", artists=("Another Singer",))
        unplayable = replace(SONG, id="synthetic-unplayable", playable=False)
        provider = Provider([wrong_artist, unplayable, SONG])
        played = MusicService(provider).play("I'm on fire", "Bruce Springsteen")
        self.assertEqual(played["status"], "playing")
        self.assertEqual(provider.plays, [SONG.id])

    def test_studio_preferred_but_lone_alternate_version_requires_choice(self):
        live = replace(SONG, id="synthetic-live", title="I'm On Fire (Live)", version="live")
        provider = Provider([live, SONG])
        self.assertEqual(MusicService(provider).play("I'm on fire", "Bruce Springsteen")["status"], "playing")
        self.assertEqual(provider.plays, [SONG.id])
        provider = Provider([live])
        service = MusicService(provider)
        result = service.play("I'm on fire", "Bruce Springsteen")
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(result["candidates"][0]["version"], "live")
        self.assertEqual(provider.plays, [])
        service.select(result["candidates"][0]["selection_id"])
        self.assertEqual(provider.plays, [live.id])

    def test_explicit_version_must_match(self):
        for version in ("live", "remix", "acoustic", "karaoke", "instrumental", "remaster"):
            with self.subTest(version=version):
                alternate = replace(SONG, id="synthetic-" + version, version=version,
                                    title=f"I'm On Fire ({version.title()})")
                provider = Provider([SONG, alternate])
                result = MusicService(provider).play("I'm on fire", "Bruce Springsteen", version)
                self.assertEqual(result["status"], "playing")
                self.assertEqual(provider.plays, [alternate.id])
        provider = Provider([SONG])
        self.assertEqual(MusicService(provider).play("I'm on fire", version="live")["status"], "not_found")
        self.assertEqual(provider.plays, [])

    def test_title_version_marker_cannot_masquerade_as_studio(self):
        provider = Provider([replace(SONG, title="I'm On Fire - Live at a Venue")])
        result = MusicService(provider).play("I'm on fire", "Bruce Springsteen")
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(result["candidates"][0]["version"], "live")
        self.assertEqual(provider.plays, [])
        other = Track("synthetic-live-forever", "Live Forever", ("A Band",))
        self.assertEqual(MusicService(Provider([other])).resolve("Live Forever")["status"], "matched")

    def test_query_version_suffix_selects_requested_recording(self):
        for label, version in (("Live", "live"), ("2010 Remaster", "remaster"), ("Acoustic Version", "acoustic")):
            with self.subTest(label=label):
                studio = Track("synthetic-ballad-studio", "A Ballad", ("A Singer",))
                alternate = replace(studio, id="synthetic-ballad-alternate", title=f"A Ballad ({label})", version=version)
                provider = Provider([studio, alternate])
                result = MusicService(provider).play(f"A Ballad ({label})", "A Singer")
                self.assertEqual(result["status"], "playing")
                self.assertEqual(result["query"]["version"], version)
                self.assertEqual(provider.plays, [alternate.id])

    def test_conflicting_explicit_version_fails_preflight(self):
        provider = Provider()
        service = MusicService(provider)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            service.validate_request("play", {"title": "A Ballad (Live)", "version": "studio"})
        self.assertEqual(provider.searches, [])
        self.assertEqual(provider.plays, [])

    def test_real_title_parenthetical_is_not_stripped(self):
        provider = Provider([Track("synthetic-ballad", "A Song (Where I Live)", ("A Singer",))])
        service = MusicService(provider)
        self.assertEqual(service.resolve("A Song (Where I Live)")["status"], "matched")
        self.assertEqual(service.resolve("A Song")["status"], "not_found")

    def test_mixed_version_qualifiers_require_choice(self):
        studio = Track("synthetic-ballad-studio", "A Ballad", ("A Singer",))
        mixed = replace(studio, id="synthetic-ballad-mixed", title="A Ballad (Live Remix)")
        provider = Provider([studio, mixed])
        service = MusicService(provider)
        result = service.play("A Ballad (Live Remix)", "A Singer")
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(provider.plays, [])
        mixed_offer = next(track for track in result["candidates"] if "Live Remix" in track["title"])
        self.assertEqual(mixed_offer["version"], "unknown")

    def test_specific_edition_cannot_silently_match_another_edition(self):
        cases = [("2010 Remaster", "2020 Remaster", "remaster"),
                 ("Live at Rome 2013", "Live at Wembley 1985", "live")]
        for requested, available, version in cases:
            with self.subTest(requested=requested):
                other = Track("synthetic-other-edition", f"A Ballad ({available})", ("A Singer",), version=version)
                provider = Provider([other])
                service = MusicService(provider)
                result = service.play(f"A Ballad ({requested})", "A Singer")
                self.assertEqual(result["status"], "not_found")
                self.assertEqual(provider.plays, [])
                # Generic version requests still permit a specific recording.
                self.assertEqual(service.play("A Ballad", "A Singer", version)["status"], "playing")

    def test_specific_edition_matches_correct_candidate_and_normalized_label(self):
        requested = Track("synthetic-2010", "A Ballad (2010 Remaster)", ("A Singer",), version="remaster")
        other = replace(requested, id="synthetic-2020", title="A Ballad (2020 Remaster)")
        provider = Provider([other, requested])
        result = MusicService(provider).play("A Ballad (Remastered 2010)", "A Singer")
        self.assertEqual(result["status"], "playing")
        self.assertEqual(provider.plays, [requested.id])

    def test_recording_identity_cannot_collapse_conflicting_editions(self):
        one = Track("synthetic-2010", "A Ballad (2010 Remaster)", ("A Singer",),
                    version="remaster", recording_id="synthetic-shared-recording")
        two = replace(one, id="synthetic-2020", title="A Ballad (2020 Remaster)")
        provider = Provider([one, two])
        result = MusicService(provider).play("A Ballad (2020 Remaster)", "A Singer")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(provider.plays, [])

    def test_conflicting_nonstudio_metadata_cannot_auto_play(self):
        provider = Provider([Track("synthetic-conflict", "A Ballad (Live)", ("A Singer",), version="acoustic")])
        service = MusicService(provider)
        self.assertEqual(service.play("A Ballad", "A Singer", "acoustic")["status"], "not_found")
        result = service.play("A Ballad", "A Singer")
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(result["candidates"][0]["version"], "unknown")
        self.assertEqual(provider.plays, [])

    def test_unknown_version_requires_choice(self):
        provider = Provider([replace(SONG, version="unknown")])
        self.assertEqual(MusicService(provider).play("I'm on fire")["status"], "ambiguous")
        self.assertEqual(provider.plays, [])

    def test_playing_requires_observation_of_exact_selected_id(self):
        cases = [(PlaybackResult(True), "accepted", None),
                 (PlaybackResult(True, True), "accepted", None),
                 (PlaybackResult(True, True, "synthetic-other"), "unknown", None),
                 (PlaybackResult(True, False, SONG.id), "accepted", False),
                 (PlaybackResult(False), "failed", None),
                 (PlaybackResult(True, True, SONG.id), "playing", True)]
        for response, status, playing in cases:
            with self.subTest(response=response):
                result = MusicService(Provider(result=response)).play("I'm on fire", "Bruce Springsteen")
                self.assertEqual((result["status"], result["playing"]), (status, playing))

    def test_selection_is_single_use_even_when_playback_uncertain(self):
        provider = Provider(result=TimeoutError("Bearer secret-token in service response"))
        service = MusicService(provider)
        selection = service.resolve("I'm on fire")["track"]["selection_id"]
        result = service.select(selection)
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["accepted"])
        self.assertNotIn("secret-token", str(result))
        with self.assertRaises(ValueError):
            service.select(selection)
        self.assertEqual(provider.plays, [SONG.id])

    def test_forged_expired_and_raw_provider_ids_cannot_play(self):
        current = [100.0]
        provider = Provider()
        service = MusicService(provider, now=lambda: current[0])
        selection = service.resolve("I'm on fire")["track"]["selection_id"]
        for identifier in (SONG.id, "https://example.invalid/music", str(uuid.uuid4())):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                service.select(identifier)
        current[0] += 600
        with self.assertRaises(ValueError):
            service.select(selection)
        self.assertEqual(provider.plays, [])

    def test_selection_cache_is_bounded_and_evicts_oldest(self):
        provider = Provider()
        service = MusicService(provider, now=lambda: 100.0)
        first = service.resolve("I'm on fire")["track"]["selection_id"]
        latest = None
        for _ in range(MAX_SELECTIONS):
            latest = service.resolve("I'm on fire")["track"]["selection_id"]
        with self.assertRaises(ValueError):
            service.select(first)
        self.assertEqual(service.select(latest)["status"], "playing")

    def test_identity_deduplication_and_unproven_editions(self):
        self.assertEqual(MusicService(Provider([SONG, SONG])).resolve("I'm on fire")["status"], "matched")
        album = replace(SONG, recording_id="synthetic-recording")
        compilation = replace(album, id="synthetic-compilation", album="Compilation")
        self.assertEqual(MusicService(Provider([album, compilation])).resolve("I'm on fire")["status"], "matched")
        result = MusicService(Provider([SONG, replace(SONG, id="synthetic-other-edition")])).resolve("I'm on fire")
        self.assertEqual(result["status"], "ambiguous")

    def test_unplayable_tracks_excluded(self):
        provider = Provider([replace(SONG, playable=False)])
        self.assertEqual(MusicService(provider).play("I'm on fire")["status"], "not_found")
        self.assertEqual(provider.plays, [])

    def test_malformed_catalog_and_auth_errors_are_sanitized(self):
        malformed = [None, [SONG] * 51, [object()], [replace(SONG, id="")],
                     [replace(SONG, artists=["Bruce Springsteen"])],
                     [replace(SONG, playable=1)], [replace(SONG, version="secret-token")],
                     [SONG, replace(SONG, artists=("Different Artist",))],
                     PermissionError("Bearer secret-token authentication failed")]
        for catalog in malformed:
            with self.subTest(catalog=catalog):
                provider = Provider()
                provider.tracks = catalog
                result = MusicService(provider).play("I'm on fire")
                self.assertEqual(result["status"], "failed")
                self.assertNotIn("secret-token", str(result))
                self.assertEqual(provider.plays, [])

    def test_malformed_playback_is_uncertain_and_never_retried(self):
        for response in (True, {"accepted": True}, PlaybackResult(1), PlaybackResult(True, "true")):
            with self.subTest(response=response):
                provider = Provider(result=response)
                result = MusicService(provider).play("I'm on fire")
                self.assertEqual(result["status"], "unknown")
                self.assertEqual(provider.plays, [SONG.id])

    def test_preflight_validates_without_searching_or_playing(self):
        provider = Provider()
        service = MusicService(provider)
        invalid = [("play", {"title": ""}), ("resolve", {"title": "\nhello"}),
                   ("resolve", {"title": "x" * 201}), ("resolve", {"title": "..."}),
                   ("play", {"title": "Hello", "artist": " "}),
                   ("play", {"title": "Hello", "artist": []}),
                   ("play", {"title": "Hello", "version": "unknown"}),
                   ("play", {"title": "Hello", "version": []}),
                   ("play", {"title": "Hello", "track_id": SONG.id}),
                   ("select", {"selection_id": SONG.id}), ("other", {})]
        for operation, args in invalid:
            with self.subTest(operation=operation, args=args), self.assertRaises(ValueError):
                service.validate_request(operation, args)
        service.validate_request("select", {"selection_id": str(uuid.uuid4())})
        service.validate_request("play", {"title": "Hello", "artist": "An Artist", "version": "studio"})
        self.assertEqual(provider.searches, [])
        self.assertEqual(provider.plays, [])

    def test_close_is_idempotent_and_revokes_selections(self):
        provider = Provider()
        service = MusicService(provider)
        selection = service.resolve("I'm on fire")["track"]["selection_id"]
        service.close()
        service.close()
        self.assertEqual(provider.closed, 1)
        self.assertFalse(service.status()["configured"])
        self.assertEqual(service.select(selection)["status"], "not_configured")
        self.assertEqual(provider.plays, [])


if __name__ == "__main__":
    unittest.main()
