"""Connect voice/text clients to the single local MusicKit service."""
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, HTTPRedirectHandler, ProxyHandler

from .music import MusicPlaybackError, MusicService

DEFAULT_CONNECTION = Path(__file__).resolve().parent.parent / "work" / "music-connection.json"


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


class RemoteMusicService(MusicService):
    def __init__(self, url, token):
        super().__init__()
        parsed = urlsplit(url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}
                or parsed.username or parsed.password or parsed.path not in {"", "/"}
                or parsed.query or parsed.fragment or parsed.port is None):
            raise ValueError("Music connection must point to a loopback HTTP port")
        if not isinstance(token, str) or not 32 <= len(token) <= 256 or not all(33 <= ord(c) <= 126 for c in token):
            raise ValueError("Invalid local music connection token")
        self._url, self._token = url.rstrip("/"), token
        self._provider_name = "Apple Music"
        self._http = build_opener(ProxyHandler({}), _NoRedirect())

    def _call(self, path, body=None, *, timeout=45):
        request = Request(self._url + path, data=None if body is None else json.dumps(body).encode(),
                          headers={"Authorization": "Bearer " + self._token,
                                   "Content-Type": "application/json"})
        with self._http.open(request, timeout=timeout) as response:
            raw = response.read(256_001)
        if len(raw) > 256_000:
            raise ValueError("Music response exceeds size limit")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Invalid music response")
        return data

    def status(self):
        try:
            return self._call("/v1/music/status", timeout=2)
        except Exception:
            return {"configured": False, "provider": self._provider_name,
                    "message": "Apple Music bridge is offline. Start apm-music and open its player page."}

    def _request(self, operation, args, path):
        self.validate_request(operation, args)
        try:
            result = self._call(path, args if operation != "select" else {})
            if not isinstance(result.get("status"), str):
                raise ValueError("Invalid music response")
            return result
        except Exception:
            # A lost response after a write says nothing about its outcome.
            return self._result("failed" if operation == "resolve" else "unknown",
                                message="Apple Music bridge could not confirm the request. It was not retried.")

    def resolve(self, title, artist=None, version=None):
        return self._request("resolve", {"title": title, "artist": artist, "version": version}, "/v1/music/resolve")

    def play(self, title, artist=None, version=None):
        return self._request("play", {"title": title, "artist": artist, "version": version}, "/v1/music/play")

    def pause(self):
        self.validate_request("pause", {})
        reason = "pause_unconfirmed"
        message = "Music pause could not be confirmed."
        try:
            result = self._call("/v1/music/pause", {}, timeout=2)
            status, accepted, playing = result.get("status"), result.get("accepted"), result.get("playing")
            was_playing = result.get("was_playing")
            if was_playing is not None and type(was_playing) is not bool:
                raise ValueError("Invalid pause result")
            if status == "paused" and accepted is True and playing is False:
                return self._result("paused", accepted=True, playing=False, was_playing=was_playing)
            if (status == "unavailable" and accepted is False and playing is None
                    and result.get("reason") in {"not_configured", "unsupported", "disconnected"}):
                return self._result("unavailable", reason=result["reason"], accepted=False,
                                    playing=None, was_playing=None,
                                    message=self._unavailable_message("pause", result["reason"]))
            if status == "unknown" and result.get("reason") == "player_update_required":
                reason = "player_update_required"
                message = "Reload and reconnect the Apple Music player to enable pause."
            elif (status == "unknown" and isinstance(result.get("reason"), str)
                  and result["reason"] in MusicPlaybackError.REASONS):
                reason = result["reason"]
                message = MusicPlaybackError(reason).pause_message()
        except HTTPError as exc:
            if exc.code in {401, 403}:
                reason = "bridge_authorization_failed"
                message = "Restart APM using the current music connection, then reconnect the player."
            elif exc.code == 404:
                reason = "bridge_update_required"
                message = "Restart the music server, then reload and reconnect its player page."
        except (URLError, TimeoutError, ConnectionError):
            reason = "bridge_unreachable"
            message = "Check the music server and reconnect its player; pause could not be confirmed."
        except Exception:
            pass
        return self._result("unknown", accepted=None, playing=None, was_playing=None,
                            reason=reason, message=message)

    @staticmethod
    def _unavailable_message(operation, reason):
        return {"not_configured": "No music provider is connected.",
                "unsupported": f"The music provider does not support {operation}.",
                "disconnected": "The music player is disconnected. Reconnect its browser tab."}[reason]

    def resume(self):
        self.validate_request("resume", {})
        reason = "resume_unconfirmed"
        message = "Music resume could not be confirmed. The request was not retried."
        try:
            result = self._call("/v1/music/resume", {}, timeout=22)
            status, accepted, playing = result.get("status"), result.get("accepted"), result.get("playing")
            if status == "resumed" and accepted is True and playing is True:
                return self._result("resumed", accepted=True, playing=True)
            if status == "empty" and accepted is False and playing is False:
                return self._result("empty", accepted=False, playing=False,
                                    message="There is no song in the player's queue to resume. Ask to play a song first.")
            if (status == "unavailable" and accepted is False and playing is None
                    and result.get("reason") in {"not_configured", "unsupported", "disconnected"}):
                return self._result("unavailable", reason=result["reason"], accepted=False, playing=None,
                                    message=self._unavailable_message("resume", result["reason"]))
            if status == "unknown" and result.get("reason") == "player_update_required":
                reason = "player_update_required"
                message = ("The player tab is outdated. Refresh the browser page (Cmd-R on Mac), "
                           "then reconnect Apple Music to enable resume.")
            elif (status == "unknown" and isinstance(result.get("reason"), str)
                  and result["reason"] in MusicPlaybackError.REASONS):
                reason = result["reason"]
                message = MusicPlaybackError(reason).user_message()
        except HTTPError as exc:
            if exc.code in {401, 403}:
                reason = "bridge_authorization_failed"
                message = "Restart APM using the current music connection, then reconnect the player."
            elif exc.code == 404:
                reason = "bridge_update_required"
                message = "Restart the music server, then refresh the browser player page and reconnect."
        except (URLError, TimeoutError, ConnectionError):
            reason = "bridge_unreachable"
            message = "Check the music server and reconnect its player; resume could not be confirmed."
        except Exception:
            pass
        return self._result("unknown", accepted=None, playing=None, reason=reason, message=message)

    def select(self, selection_id):
        self.validate_request("select", {"selection_id": selection_id})
        return self._request("select", {"selection_id": selection_id},
                             f"/v1/music/selections/{selection_id}/play")


def load_music_service(path=None):
    path = Path(path) if path is not None else DEFAULT_CONNECTION
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return MusicService()
    except (OSError, ValueError) as exc:
        raise ValueError("Cannot read the saved local music connection; restart apm-music") from exc
    if not isinstance(data, dict) or set(data) != {"url", "token"}:
        raise ValueError("Invalid local music connection; restart apm-music")
    return RemoteMusicService(data["url"], data["token"])
