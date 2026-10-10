"""Apple catalog search plus a bounded bridge to an authorized MusicKit browser.

Only developer credentials are used for catalog requests. User authorization
and playback stay in the browser. Commands are handed out once and never
retried: a timeout cannot establish whether browser playback already started.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import math
import re
from threading import Condition
import time
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid

from .music import MusicPauseError, MusicPlaybackError, MusicUnavailable, PlaybackResult, Track, _normalized, _similarity, _text, _title_key, _title_parts, _version

PLAYER_TTL = 30
CATALOG_TTL = 600
MAX_KNOWN_TRACKS = 1024
MAX_PENDING = 8

DIAGNOSTIC_PHASES = frozenset({"preflight", "waiting", "queue", "play", "confirm"})
DIAGNOSTIC_REASONS = frozenset({
    "confirmed", "expired", "cancelled", "authorization_lost", "queue_rejected", "queue_unavailable",
    "queue_mismatch", "autoplay_blocked", "play_rejected", "media_error", "timeout", "unknown",
})
DIAGNOSTIC_SDK_REASONS = frozenset({
    "ACCESS_DENIED", "AUTHORIZATION_ERROR", "CONTENT_EQUIVALENT", "CONTENT_RESTRICTED",
    "CONTENT_UNAVAILABLE", "CONTENT_UNSUPPORTED", "DEVICE_LIMIT", "GEO_BLOCK", "MEDIA_CERTIFICATE",
    "MEDIA_DESCRIPTOR", "MEDIA_LICENSE", "MEDIA_KEY", "MEDIA_PLAYBACK", "MEDIA_SESSION", "NETWORK_ERROR",
    "NOT_FOUND", "OUTPUT_RESTRICTED", "SERVER_ERROR", "SERVICE_UNAVAILABLE", "STREAM_UPSELL",
    "SUBSCRIPTION_ERROR", "TOKEN_EXPIRED", "UNAUTHORIZED_ERROR", "UNSUPPORTED_ERROR",
    "USER_INTERACTION_REQUIRED", "WIDEVINE_CDM_EXPIRED",
})
DIAGNOSTIC_QUEUE_CHECKS = frozenset({"absent", "different_queue", "wrong_length", "wrong_track", "matched"})
DIAGNOSTIC_STATES = frozenset({
    "none", "loading", "playing", "paused", "stopped", "ended", "seeking", "waiting", "stalled",
    "completed", "unknown",
})


def validate_completion_diagnostics(value):
    """Copy only bounded, allowlisted browser diagnostics; never raw errors."""
    required = {"phase", "reason", "elapsed_ms", "phase_ms"}
    allowed = required | {"sdk_reason", "queue_check", "state", "track_matches"}
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - allowed:
        raise ValueError("Invalid music command diagnostics")
    for field, choices in (("phase", DIAGNOSTIC_PHASES), ("reason", DIAGNOSTIC_REASONS),
                           ("sdk_reason", DIAGNOSTIC_SDK_REASONS), ("queue_check", DIAGNOSTIC_QUEUE_CHECKS),
                           ("state", DIAGNOSTIC_STATES)):
        if field in value and (not isinstance(value[field], str) or value[field] not in choices):
            raise ValueError("Invalid music command diagnostics")
    for field in ("elapsed_ms", "phase_ms"):
        if type(value[field]) is not int or not 0 <= value[field] <= 60000:
            raise ValueError("Invalid music command diagnostics")
    if value["phase_ms"] > value["elapsed_ms"]:
        raise ValueError("Invalid music command diagnostics")
    if "track_matches" in value and value["track_matches"] is not None and type(value["track_matches"]) is not bool:
        raise ValueError("Invalid music command diagnostics")
    return dict(value)


def _uuid(value):
    try:
        if not isinstance(value, str) or str(uuid.UUID(value)) != value:
            raise ValueError
    except (ValueError, AttributeError):
        raise ValueError("A canonical MusicKit session or command UUID is required") from None
    return value


def _track_id(value):
    if not isinstance(value, str) or re.fullmatch(r"[1-9][0-9]{0,19}", value) is None:
        raise ValueError("A numeric Apple Music catalog song ID is required")
    return value


def _seconds(value, maximum, *, allow_zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or value > maximum or value < 0 or not math.isfinite(value)
            or (value == 0 and not allow_zero)):
        raise ValueError("Invalid MusicKit timeout")
    return float(value)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def _request(url, *, headers, timeout):
    # The URL is built exclusively from a validated storefront and query. Never
    # follow a redirect carrying the developer's Authorization header elsewhere.
    with build_opener(_NoRedirect()).open(Request(url, headers=headers), timeout=timeout) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Apple Music catalog response is too large")
    return json.loads(raw)


@dataclass
class _Command:
    id: str
    track_id: str | None
    deadline: float
    real_deadline: float
    started_at: float
    operation: str = "play"
    dispatched: bool = False
    result: PlaybackResult | None = None
    error: str | None = None
    error_reason: str | None = None


class MusicKitProvider:
    name = "Apple Music"

    def __init__(self, developer_token, *, request=None, now=time.monotonic, command_timeout=20, pause_timeout=1.5):
        if not callable(developer_token) or not callable(now):
            raise ValueError("MusicKit requires developer-token and clock callables")
        if request is not None and not callable(request):
            raise ValueError("MusicKit request transport must be callable")
        self._developer_token = developer_token
        self._request = _request if request is None else request
        self._now_source = now
        self._command_timeout = _seconds(command_timeout, 60)
        self._pause_timeout = _seconds(pause_timeout, 2)
        self._condition = Condition()
        self._session = None
        self._storefront = None
        self._protocol_version = None
        self._last_poll = 0.0
        self._closed = False
        self._pending = OrderedDict()
        self._known_tracks = OrderedDict()
        self._last_command = None
        self._last_expired_id = None

    def _now(self):
        value = self._now_source()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("MusicKit clock must return finite monotonic seconds")
        return value

    def _revoke(self, reason):
        for command in self._pending.values():
            command.error = reason
        self._pending.clear()
        self._known_tracks.clear()
        self._last_command = None
        self._last_expired_id = None
        self._session = self._storefront = self._protocol_version = None
        self._condition.notify_all()

    def _expire(self, now):
        if self._session is not None and now - self._last_poll >= PLAYER_TTL:
            self._revoke("Music player connection expired; reconnect before trying again")
        real_now = time.monotonic()
        for identifier, command in list(self._pending.items()):
            if now >= command.deadline or real_now >= command.real_deadline:
                command.error = "Music playback timed out and was not retried"
                command.error_reason = "timeout"
                self._last_command = {
                    "operation": command.operation, "source": "bridge",
                    "phase": "completion" if command.dispatched else "delivery",
                    "reason": "timeout", "elapsed_ms": min(60000, max(0, int((real_now - command.started_at) * 1000))),
                    "dispatched": command.dispatched, "late_completion": False,
                }
                self._last_expired_id = command.id
                self._pending.pop(identifier)
                self._condition.notify_all()
        for identifier, expires in list(self._known_tracks.items()):
            if expires <= now:
                self._known_tracks.pop(identifier)

    def _active(self, session_id, now):
        self._expire(now)
        if self._closed or self._session is None or session_id != self._session:
            raise ValueError("Music player session is not active; reconnect")

    def activate(self, session_id, storefront, protocol_version=3):
        session_id = _uuid(session_id)
        if not isinstance(storefront, str) or re.fullmatch(r"[A-Za-z]{2}", storefront) is None:
            raise ValueError("MusicKit storefront must be a two-letter country code")
        storefront = storefront.lower()
        if type(protocol_version) is not int or protocol_version not in {1, 2, 3}:
            raise ValueError("Unsupported music player protocol")
        with self._condition:
            if self._closed:
                raise ValueError("MusicKit provider is closed")
            now = self._now()
            self._expire(now)
            if (self._session != session_id or self._storefront != storefront
                    or self._protocol_version != protocol_version):
                self._revoke("Music player session was replaced; playback was not retried")
            self._session, self._storefront, self._last_poll = session_id, storefront, now
            self._protocol_version = protocol_version
            self._condition.notify_all()

    def disconnect(self, session_id):
        session_id = _uuid(session_id)
        with self._condition:
            self._active(session_id, self._now())
            self._revoke("Music player disconnected; playback was not retried")

    def status(self):
        with self._condition:
            self._expire(self._now())
            connected = not self._closed and self._session is not None
            result = {"connected": connected, "storefront": self._storefront,
                      "protocol_version": self._protocol_version,
                      "supports_pause": connected and self._protocol_version >= 2,
                      "supports_resume": connected and self._protocol_version >= 3}
            if self._last_command is not None:
                result["last_command"] = dict(self._last_command)
            return result

    @staticmethod
    def _tracks(payload):
        if not isinstance(payload, dict) or payload.get("errors") or not isinstance(payload.get("results"), dict):
            raise ValueError("Invalid Apple Music catalog response")
        songs = payload["results"].get("songs", {"data": []})
        if not isinstance(songs, dict) or not isinstance(songs.get("data"), list) or len(songs["data"]) > 50:
            raise ValueError("Invalid Apple Music catalog song collection")
        tracks = []
        for item in songs["data"]:
            if not isinstance(item, dict) or item.get("type") != "songs":
                continue
            attributes = item.get("attributes")
            if not isinstance(attributes, dict):
                continue
            play = attributes.get("playParams")
            # Preview URLs are deliberately ignored. Full song play parameters
            # must refer to this catalog ID, never a personal library identifier.
            if (not isinstance(play, dict) or play.get("kind") != "song"
                    or ("isLibrary" in play and play["isLibrary"] is not False) or play.get("id") != item.get("id")
                    or ("isPlayable" in play and play["isPlayable"] is not True)
                    or ("isPlayable" in attributes and attributes["isPlayable"] is not True)):
                continue
            try:
                identifier = _track_id(item.get("id"))
                title = _text(attributes.get("name"), "Catalog title")
                artist = _text(attributes.get("artistName"), "Catalog artist")
                album = _text(attributes.get("albumName", ""), "Catalog album", empty=True)
            except ValueError:
                continue
            tracks.append(Track(identifier, title, (artist,), album))
        return tracks

    def _search_page(self, storefront, term):
        token = self._developer_token()
        if not isinstance(token, str) or not token or len(token) > 8192 or any(character.isspace() for character in token):
            raise ValueError("Apple Music developer token is unavailable")
        url = f"https://api.music.apple.com/v1/catalog/{storefront}/search?" + urlencode(
            {"term": term, "types": "songs", "limit": 25})
        payload = self._request(url, headers={"Authorization": "Bearer " + token, "Accept": "application/json"}, timeout=10)
        return self._tracks(payload)

    @staticmethod
    def _viable(tracks, title, artist):
        title_key, versions, details = _title_parts(title)
        for track in tracks:
            if len(versions) == 1 and _version(track) not in versions:
                continue
            if details and _title_parts(track.title)[2] != details:
                continue
            if (_similarity(title_key, _title_key(track.title)) >= 0.88
                    and (artist is None or max(_similarity(_normalized(artist), _normalized(name))
                                               for name in track.artists) >= 0.86)):
                return True
        return False

    def search(self, title, artist=None):
        title = _text(title, "Title")
        artist = _text(artist, "Artist") if artist is not None else None
        with self._condition:
            self._expire(self._now())
            if self._closed or self._session is None:
                raise RuntimeError("Connect an authorized Apple Music browser before searching")
            session, storefront = self._session, self._storefront
        tracks = self._search_page(storefront, f"{title} {artist}" if artist else title)
        if artist and not self._viable(tracks, title, artist):
            tracks += self._search_page(storefront, title)
        unique = {}
        for track in tracks:
            if track.id in unique and unique[track.id] != track:
                raise ValueError("Conflicting Apple Music catalog song identity")
            unique[track.id] = track
        tracks = list(unique.values())[:50]
        with self._condition:
            now = self._now()
            self._active(session, now)
            if storefront != self._storefront:
                raise RuntimeError("Apple Music storefront changed during search")
            for track in tracks:
                self._known_tracks[track.id] = now + CATALOG_TTL
                self._known_tracks.move_to_end(track.id)
            while len(self._known_tracks) > MAX_KNOWN_TRACKS:
                self._known_tracks.popitem(last=False)
        return tracks

    def play(self, track_id):
        track_id = _track_id(track_id)
        return self._command("play", track_id, self._command_timeout)

    def play_guarded(self, track_id, cancelled):
        """Cancel a service intent atomically with enqueuing its playback."""
        track_id = _track_id(track_id)
        return self._command("play", track_id, self._command_timeout, cancelled=cancelled)

    def pause(self):
        return self._command("pause", None, self._pause_timeout)

    def resume(self):
        # The browser owns the current queue and position. Resuming needs no
        # catalog request or remembered search ID, and must never replace it.
        return self._command("resume", None, self._command_timeout)

    def resume_guarded(self, cancelled):
        return self._command("resume", None, self._command_timeout, cancelled=cancelled)

    def _command(self, operation, track_id, timeout, *, cancelled=None):
        with self._condition:
            if cancelled is not None and cancelled.is_set():
                raise MusicPlaybackError("cancelled")
            now = self._now()
            self._expire(now)
            if self._closed or self._session is None:
                if operation in {"pause", "resume"}:
                    raise MusicUnavailable("Music player is disconnected")
                raise RuntimeError("No connected Apple Music player; playback was not attempted")
            if operation == "play" and track_id not in self._known_tracks:
                raise ValueError("Apple Music song was not returned by this session's catalog search")
            if ((operation == "pause" and self._protocol_version < 2)
                    or (operation == "resume" and self._protocol_version < 3)):
                # An older browser may still be playing. Do not call this a
                # disconnected no-op or send a command it cannot understand.
                raise MusicPauseError("player_update_required")
            if operation == "pause":
                # Wake-time pause supersedes queued or dispatched play/resume.
                # The browser also cancels their observers and guards late starts.
                for identifier, pending in list(self._pending.items()):
                    if pending.operation in {"play", "resume"}:
                        pending.error = "Music playback was interrupted by pause"
                        self._pending.pop(identifier)
                self._condition.notify_all()
            if len(self._pending) >= MAX_PENDING:
                raise RuntimeError("Apple Music playback queue is busy")
            started_at = time.monotonic()
            command = _Command(str(uuid.uuid4()), track_id, now + timeout,
                               started_at + timeout, started_at, operation=operation)
            self._pending[command.id] = command
            self._condition.notify_all()
            while command.result is None and command.error is None:
                now = self._now()
                self._expire(now)
                if command.error is not None:
                    break
                remaining = min(command.deadline - now, command.real_deadline - time.monotonic(),
                                PLAYER_TTL - (now - self._last_poll))
                self._condition.wait(max(0, remaining))
            if command.error is not None:
                if command.error_reason is not None:
                    raise MusicPlaybackError(command.error_reason)
                raise RuntimeError(command.error)
            return command.result

    def poll(self, session_id, wait_seconds=15):
        session_id = _uuid(session_id)
        wait_seconds = _seconds(wait_seconds, 15, allow_zero=True)
        with self._condition:
            now = self._now()
            self._active(session_id, now)
            self._last_poll = now
            deadline, real_deadline = now + wait_seconds, time.monotonic() + wait_seconds
            while True:
                now = self._now()
                self._active(session_id, now)
                for command in self._pending.values():
                    if not command.dispatched:
                        remaining = min(command.deadline - now, command.real_deadline - time.monotonic())
                        if remaining <= 0:
                            continue
                        command.dispatched = True
                        self._last_poll = now
                        return {"id": command.id, "operation": command.operation,
                                **({"track_id": command.track_id} if command.operation == "play" else {}),
                                "expires_in_ms": max(1, int(remaining * 1000))}
                remaining = min(deadline - now, real_deadline - time.monotonic())
                if remaining <= 0:
                    self._last_poll = now
                    return None
                self._condition.wait(remaining)

    def complete(self, session_id, command_id, result, diagnostics=None):
        session_id, command_id = _uuid(session_id), _uuid(command_id)
        diagnostics = validate_completion_diagnostics(diagnostics) if diagnostics is not None else None
        error = result == {"error": "playback_unconfirmed"}
        if not error:
            if (not isinstance(result, dict) or set(result) - {"accepted", "playing", "track_id", "was_playing"}
                    or type(result.get("accepted")) is not bool
                    or (result.get("playing") is not None and type(result["playing"]) is not bool)
                    or (result.get("was_playing") is not None and type(result["was_playing"]) is not bool)):
                raise ValueError("Invalid MusicKit playback result")
            identifier = result.get("track_id")
            if identifier is not None:
                _track_id(identifier)
            playback = PlaybackResult(result["accepted"], result.get("playing"), identifier, result.get("was_playing"))
        with self._condition:
            self._active(session_id, self._now())
            command = self._pending.get(command_id)
            if command is None or not command.dispatched:
                if (command_id == self._last_expired_id and self._last_command is not None
                        and self._last_command.get("dispatched") is True):
                    # A late response is evidence of delivery only. Its result
                    # and browser diagnostics cannot revive an expired write.
                    self._last_command["late_completion"] = True
                raise ValueError("MusicKit command is unknown, expired, or was not dispatched")
            if not error and command.operation in {"play", "resume"} and "was_playing" in result:
                raise ValueError("Invalid play result")
            self._pending.pop(command_id)
            # Keep only the newest accepted completion, never identifiers,
            # catalog metadata, arbitrary SDK messages, or stale-session data.
            self._last_command = ({"operation": command.operation, **diagnostics}
                                  if diagnostics is not None else None)
            self._last_expired_id = None
            if error:
                command.error = "Music browser could not confirm playback; the request was not retried"
                if diagnostics is not None:
                    command.error_reason = diagnostics["reason"] if diagnostics["reason"] != "confirmed" else "unknown"
            else:
                command.result = playback
            self._condition.notify_all()

    def close(self):
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._revoke("Apple Music provider closed; playback was not retried")
