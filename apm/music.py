"""Provider-neutral catalog resolution and honest playback acknowledgments.

Similarity scores are conservative string heuristics, not probabilities. A
provider must return typo-tolerant search candidates; this service never opens
URLs or accepts a provider track ID from a caller. Selection tokens are local,
short lived, and consumed before attempting playback, including failed calls.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from difflib import SequenceMatcher
import math
import re
from threading import RLock
import time
from typing import Protocol
import unicodedata
import uuid

VERSIONS = frozenset({"studio", "live", "remix", "acoustic", "karaoke", "instrumental", "remaster"})
CATALOG_VERSIONS = VERSIONS | {"unknown"}
SELECTION_TTL = 600
MAX_SELECTIONS = 256
MAX_CANDIDATES = 10


@dataclass(frozen=True)
class Track:
    id: str
    title: str
    artists: tuple[str, ...]
    album: str = ""
    version: str = "studio"
    playable: bool = True
    recording_id: str | None = None


@dataclass(frozen=True)
class PlaybackResult:
    accepted: bool
    playing: bool | None = None
    track_id: str | None = None
    was_playing: bool | None = None


class MusicUnavailable(RuntimeError):
    """A playback control was not sent because no player is connected."""


class MusicPlaybackError(RuntimeError):
    """A browser failure with a fixed, credential-free reason code."""

    REASONS = frozenset({"expired", "cancelled", "authorization_lost", "queue_rejected",
                         "queue_unavailable", "queue_mismatch", "autoplay_blocked",
                         "play_rejected", "media_error", "timeout", "unknown"})

    def __init__(self, reason):
        if not isinstance(reason, str) or reason not in self.REASONS:
            raise ValueError("Invalid music playback reason")
        self.reason = reason
        super().__init__(reason)

    def user_message(self):
        detail = {
            "autoplay_blocked": "The browser requires a click. Click Start playback in the Apple Music player.",
            "authorization_lost": "Apple Music authorization was lost. Reconnect its player page.",
            "queue_rejected": "Apple Music could not load the requested song.",
            "queue_unavailable": "Apple Music did not provide a playable queue.",
            "queue_mismatch": "Apple Music loaded a different queue; requested playback is unconfirmed.",
            "play_rejected": "Apple Music rejected the playback start.",
            "media_error": "Apple Music reported a media playback error.",
            "timeout": "Apple Music did not confirm playback before the request expired.",
            "expired": "The playback request expired before it could finish.",
            "cancelled": "The playback request was interrupted.",
            "unknown": "Playback could not be confirmed.",
        }[self.reason]
        return detail + " The request was not retried."


class MusicPauseError(RuntimeError):
    """Playback control needs user intervention; only safe reason codes are exposed."""

    def __init__(self, reason):
        if reason != "player_update_required":
            raise ValueError("Invalid music pause reason")
        self.reason = reason
        super().__init__(reason)


class MusicProvider(Protocol):
    name: str

    def search(self, title: str, artist: str | None = None) -> list[Track]: ...
    def play(self, track_id: str) -> PlaybackResult: ...
    def pause(self) -> PlaybackResult: ...
    def resume(self) -> PlaybackResult: ...
    def close(self) -> None: ...


def _text(value, field, *, empty=False, maximum=200):
    if (not isinstance(value, str) or len(value) > maximum
            or not all(character.isprintable() for character in value)
            or (not empty and not value.strip())):
        raise ValueError(f"{field} must contain 1–{maximum} printable characters")
    return value.strip()


def _normalized(value):
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(character for character in value if not unicodedata.combining(character))
    # Apostrophes connect words: I'm, I’m, and Im should compare identically.
    value = value.replace("'", "").replace("’", "").replace("ʼ", "")
    return " ".join("".join(character if character.isalnum() else " " for character in value).split())


_VERSION_WORD = re.compile(r"\b(live|remix(?:ed)?|acoustic|karaoke|instrumental|remaster(?:ed)?)\b", re.I)
_VERSION_LABEL = re.compile(r"^(?:(?:\d{4})\s+remaster(?:ed)?\b|(?:live|remix(?:ed)?|acoustic|karaoke|instrumental|remaster(?:ed)?)\b)", re.I)
_VERSION_SUFFIX = re.compile(r"\s*(?:\([^()]*\)|\[[^\[\]]*\]|\s[-–—]\s.+)$")


def _title_parts(title):
    versions = set()
    details = []
    while suffix := _VERSION_SUFFIX.search(title):
        label = suffix.group().strip().strip("()[]").lstrip("-–— ")
        if not _VERSION_LABEL.match(label):
            break
        for match in _VERSION_WORD.finditer(label):
            word = match.group().lower()
            versions.add("remaster" if word.startswith("remaster") else "remix" if word.startswith("remix") else word)
        detail = _normalized(_VERSION_WORD.sub(" ", label))
        detail = " ".join(word for word in detail.split() if word not in {"version", "recording", "edition"})
        if detail:
            details.append(detail)
        title = title[:suffix.start()]
    # Preserve years, venues, and named mixes. Generic labels can match a more
    # specific edition; an explicitly requested edition requires these details.
    return _normalized(title), versions, tuple(sorted(details))


def _title_key(title):
    # Keep real title text such as A Song (Where I Live) or Live Forever.
    return _title_parts(title)[0]


def _version(track):
    # An explicit live/remix title must not accidentally become a studio choice
    # merely because an adapter omitted Track.version's optional argument.
    _, versions, _ = _title_parts(track.title)
    if len(versions) > 1:
        return "unknown"
    if track.version == "studio" and versions:
        return next(iter(versions))
    if versions and track.version not in versions:
        return "unknown"
    return track.version


def _similarity(left, right):
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def _selection_id(value):
    try:
        if not isinstance(value, str) or len(value) != 36 or str(uuid.UUID(value)) != value:
            raise ValueError
    except (ValueError, AttributeError):
        raise ValueError("A valid music selection ID is required") from None
    return value


class MusicService:
    def __init__(self, provider: MusicProvider | None = None, *, now=None):
        self._provider = provider
        self._provider_name = None if provider is None else _text(provider.name, "Provider name", maximum=100)
        self._now_source = time.monotonic if now is None else now
        self._lock = RLock()
        self._selections = OrderedDict()
        self._closed = False

    def status(self):
        return {"configured": self._provider is not None and not self._closed,
                "provider": self._provider_name}

    def validate_request(self, operation, args):
        if not isinstance(args, dict):
            raise ValueError("Music arguments must be an object")
        if operation in {"pause", "resume"}:
            if args:
                raise ValueError("Music playback control does not accept arguments")
        elif operation == "select":
            if set(args) != {"selection_id"}:
                raise ValueError("Music selection requires only a selection_id")
            _selection_id(args["selection_id"])
        elif operation in {"resolve", "play"}:
            if "title" not in args or set(args) - {"title", "artist", "version"}:
                raise ValueError("Music search requires title, with optional artist and version")
            title = _text(args["title"], "Title")
            if not _normalized(title):
                raise ValueError("Title must contain letters or numbers")
            if args.get("artist") is not None:
                artist = _text(args["artist"], "Artist")
                if not _normalized(artist):
                    raise ValueError("Artist must contain letters or numbers")
            if args.get("version") is not None and (not isinstance(args["version"], str) or args["version"] not in VERSIONS):
                raise ValueError("Unknown music version")
            _, suffix_versions, _ = _title_parts(title)
            if args.get("version") is not None and suffix_versions and args["version"] not in suffix_versions:
                raise ValueError("Requested version conflicts with the version in the title")
        else:
            raise ValueError("Unknown music operation")

    def _now(self):
        value = self._now_source()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Music selection clock must return finite monotonic seconds")
        return value

    def _prune(self, now):
        for identifier, (expires, _, _) in list(self._selections.items()):
            if expires <= now:
                self._selections.pop(identifier)

    @staticmethod
    def _catalog(raw):
        if not isinstance(raw, list) or len(raw) > 50:
            raise ValueError("Invalid music catalog")
        by_id, by_recording = {}, {}
        for track in raw:
            if not isinstance(track, Track):
                raise ValueError("Invalid music catalog")
            _text(track.id, "Catalog ID", maximum=256)
            _text(track.title, "Catalog title")
            _text(track.album, "Album", empty=True)
            if not isinstance(track.artists, tuple) or not 1 <= len(track.artists) <= 10:
                raise ValueError("Invalid music catalog")
            for artist in track.artists:
                _text(artist, "Catalog artist")
            if (not isinstance(track.version, str) or track.version not in CATALOG_VERSIONS
                    or type(track.playable) is not bool):
                raise ValueError("Invalid music catalog")
            if track.recording_id is not None:
                _text(track.recording_id, "Recording ID", maximum=256)
            previous = by_id.get(track.id)
            if previous is not None and previous != track:
                raise ValueError("Conflicting catalog IDs")
            by_id[track.id] = track
        result = []
        for track in by_id.values():
            if not track.playable:
                continue
            if track.recording_id is not None:
                previous = by_recording.get(track.recording_id)
                if previous is not None:
                    # Recording identity is trusted only with consistent core
                    # metadata. Different editions without identity stay apart.
                    if (_title_key(previous.title) != _title_key(track.title)
                            or {_normalized(a) for a in previous.artists} != {_normalized(a) for a in track.artists}
                            or _version(previous) != _version(track)
                            or _title_parts(previous.title)[2] != _title_parts(track.title)[2]):
                        raise ValueError("Conflicting recording identity")
                    continue
                by_recording[track.recording_id] = track
            result.append(track)
        return result

    @staticmethod
    def _public(track):
        return {"title": track.title, "artists": list(track.artists),
                "album": track.album, "version": _version(track)}

    def _offer(self, tracks, query):
        with self._lock:
            now = self._now()
            self._prune(now)
            offers = []
            for track in tracks[:MAX_CANDIDATES]:
                identifier = str(uuid.uuid4())
                self._selections[identifier] = (now + SELECTION_TTL, track, dict(query))
                offers.append({**self._public(track), "selection_id": identifier})
            while len(self._selections) > MAX_SELECTIONS:
                self._selections.popitem(last=False)
            return offers

    def _result(self, status, query=None, **fields):
        result = {"status": status, "provider": self._provider_name, **fields}
        if query is not None:
            result["query"] = query
        return result

    def resolve(self, title, artist=None, version=None):
        return self._resolve(title, artist, version, choose_first=False)

    def _resolve(self, title, artist, version, *, choose_first):
        self.validate_request("resolve", {"title": title, "artist": artist, "version": version})
        title_key, suffix_versions, suffix_details = _title_parts(title)
        if version is None and len(suffix_versions) == 1:
            version = next(iter(suffix_versions))
        query = {"title": title.strip(), "artist": artist.strip() if artist is not None else None, "version": version}
        if not self.status()["configured"]:
            return self._result("not_configured", query, message="No music provider is connected.")
        try:
            tracks = self._catalog(self._provider.search(query["title"], query["artist"]))
        except Exception:
            return self._result("failed", query, message="Music search is unavailable. No playback was requested.")
        artist_key = _normalized(artist) if artist is not None else None
        ranked = []
        for track in tracks:
            title_score = _similarity(title_key, _title_key(track.title))
            artist_score = max(_similarity(artist_key, _normalized(a)) for a in track.artists) if artist_key else 1.0
            if title_score < 0.88 or artist_score < 0.86:
                continue
            if version is not None and _version(track) != version:
                continue
            if suffix_details and _title_parts(track.title)[2] != suffix_details:
                continue
            ranked.append((0.6 * title_score + 0.4 * artist_score, track))
        if not ranked:
            # Small models sometimes put "Song by Artist" entirely in title.
            # Try the literal title first (e.g. "Stand by Me"), then recover
            # the two fields only if no literal catalog match was found.
            if artist is None:
                parts = re.split(r"\s+by\s+", title.strip(), flags=re.I)
                if len(parts) > 1:
                    song, performer = " by ".join(parts[:-1]), parts[-1]
                    if _normalized(song) and _normalized(performer):
                        return self._resolve(song, performer, version, choose_first=choose_first)
            return self._result("not_found", query, message="No sufficiently close playable title and artist match was found.")
        ranked.sort(key=lambda item: item[0], reverse=True)
        # Search-only resolution keeps different artists available for browsing,
        # even if one artist's recording is studio and another is live.
        artist_groups = {tuple(sorted(_normalized(a) for a in track.artists)) for _, track in ranked}
        artist_ambiguous = artist is None and len(artist_groups) > 1
        if version is None and (choose_first or not artist_ambiguous) and not suffix_versions:
            studio = [item for item in ranked if _version(item[1]) == "studio"]
            if studio:
                ranked = studio
        best_score, best = ranked[0]
        clear_margin = len(ranked) == 1 or best_score - ranked[1][0] >= 0.07
        version_clear = version is not None or _version(best) == "studio"
        # Play requests use the highest scoring eligible recording. Python's
        # stable sort preserves the provider's relevance order for ties, so
        # compilation releases do not force a spoken album-selection loop.
        # Search-only requests can still offer alternatives for browsing.
        matched = ((choose_first or (clear_margin and not artist_ambiguous))
                   and version_clear and len(suffix_versions) <= 1)
        offers = self._offer([track for _, track in ranked] if not matched else [best], query)
        if matched:
            return self._result("matched", query, track=offers[0])
        return self._result("ambiguous", query, candidates=offers,
                            message="Choose a recording or specify its artist and version before playback.")

    def play(self, title, artist=None, version=None):
        result = self._resolve(title, artist, version, choose_first=True)
        if result["status"] != "matched":
            return result
        return self.select(result["track"]["selection_id"])

    def pause(self):
        """Pause without a catalog request; absence is a safe, explicit no-op."""
        self.validate_request("pause", {})
        fields = {"accepted": False, "playing": None, "was_playing": None}
        if self._closed or self._provider is None:
            return self._result("unavailable", reason="not_configured", **fields,
                                message="No music provider is connected.")
        pause = getattr(self._provider, "pause", None)
        if not callable(pause):
            return self._result("unavailable", reason="unsupported", **fields,
                                message="The music provider does not support pause.")
        try:
            result = pause()
            if (not isinstance(result, PlaybackResult) or type(result.accepted) is not bool
                    or (result.playing is not None and type(result.playing) is not bool)
                    or (result.was_playing is not None and type(result.was_playing) is not bool)):
                raise ValueError("Invalid pause result")
        except MusicPauseError:
            return self._result("unknown", reason="player_update_required", accepted=None, playing=None, was_playing=None,
                                message="Reload and reconnect the Apple Music player to enable pause.")
        except MusicUnavailable:
            return self._result("unavailable", reason="disconnected", **fields,
                                message="The music player is disconnected. Reconnect its browser tab.")
        except Exception:
            return self._result("unknown", accepted=None, playing=None, was_playing=None,
                                message="Music pause could not be confirmed.")
        if result.accepted and result.playing is False:
            return self._result("paused", accepted=True, playing=False, was_playing=result.was_playing)
        return self._result("unknown", accepted=result.accepted, playing=None,
                            was_playing=result.was_playing, message="Music pause could not be confirmed.")

    def resume(self):
        """Resume the provider's existing queue without searching or replacing it."""
        self.validate_request("resume", {})
        fields = {"accepted": False, "playing": None}
        if self._closed or self._provider is None:
            return self._result("unavailable", reason="not_configured", **fields,
                                message="No music provider is connected.")
        resume = getattr(self._provider, "resume", None)
        if not callable(resume):
            return self._result("unavailable", reason="unsupported", **fields,
                                message="The music provider does not support resume.")
        try:
            result = resume()
            if (not isinstance(result, PlaybackResult) or type(result.accepted) is not bool
                    or (result.playing is not None and type(result.playing) is not bool)
                    or result.was_playing is not None):
                raise ValueError("Invalid resume result")
            if result.track_id is not None:
                _text(result.track_id, "Current recording ID", maximum=256)
        except MusicPauseError:
            return self._result("unknown", reason="player_update_required", accepted=None, playing=None,
                                message="The player tab is outdated. Refresh the browser page (Cmd-R on Mac), "
                                        "then reconnect Apple Music to enable resume.")
        except MusicUnavailable:
            return self._result("unavailable", reason="disconnected", **fields,
                                message="The music player is disconnected. Reconnect its browser tab.")
        except MusicPlaybackError as exc:
            return self._result("unknown", reason=exc.reason, accepted=None, playing=None,
                                message=exc.user_message())
        except Exception:
            return self._result("unknown", accepted=None, playing=None,
                                message="Music resume could not be confirmed. The request was not retried.")
        if result.accepted and result.playing is True and result.track_id is not None:
            return self._result("resumed", accepted=True, playing=True)
        if not result.accepted and result.playing is False and result.track_id is None:
            return self._result("empty", accepted=False, playing=False,
                                message="There is no song in the player's queue to resume. Ask to play a song first.")
        return self._result("unknown", accepted=result.accepted, playing=None,
                            message="Music resume could not be confirmed. The request was not retried.")

    def select(self, selection_id):
        self.validate_request("select", {"selection_id": selection_id})
        with self._lock:
            if not self.status()["configured"]:
                return self._result("not_configured", message="No music provider is connected.")
            self._prune(self._now())
            selection = self._selections.pop(selection_id, None)
            if selection is None:
                raise ValueError("Music selection is invalid, expired, or already used; search again")
            _, track, query = selection
        metadata = self._public(track)
        try:
            result = self._provider.play(track.id)
            if (not isinstance(result, PlaybackResult) or type(result.accepted) is not bool
                    or (result.playing is not None and type(result.playing) is not bool)
                    or (result.track_id is not None and not isinstance(result.track_id, str))):
                raise ValueError("Invalid playback result")
        except MusicPlaybackError as exc:
            return self._result("unknown", query, track=metadata, accepted=None, playing=None,
                                reason=exc.reason, message=exc.user_message())
        except Exception:
            return self._result("unknown", query, track=metadata, accepted=None, playing=None,
                                message="Playback could not be confirmed. The request was not retried.")
        if result.playing is True and result.track_id == track.id:
            return self._result("playing", query, track=metadata, accepted=result.accepted, playing=True)
        if result.playing is True and result.track_id is not None and result.track_id != track.id:
            return self._result("unknown", query, track=metadata, accepted=result.accepted, playing=None,
                                message="The provider reported a different recording; requested playback is unconfirmed.")
        status = "accepted" if result.accepted else "failed"
        return self._result(status, query, track=metadata, accepted=result.accepted,
                            playing=False if result.playing is False and result.track_id == track.id else None,
                            message="Playback request accepted; playing is not confirmed." if result.accepted else "Playback request was not accepted.")

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._selections.clear()
        if self._provider is not None:
            try:
                self._provider.close()
            except Exception:
                pass
