"""Loopback-only REST interface for local devices, tasks, and music playback.

This is an initial local API, not a remotely accessible mobile service. LAN
exposure, secure pairing, and a chat endpoint are future work. Discovery only
returns proposals; devices must be explicitly registered before use.
"""

import argparse
from contextlib import asynccontextmanager
from copy import deepcopy
from http import HTTPStatus
import os
from pathlib import Path
import secrets
import sqlite3
from typing import Any, Literal
from urllib.parse import urlsplit

from jsonschema import ValidationError

from .backends.base import BackendError


_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "testserver"})
_PUBLIC_DOC_PATHS = frozenset({"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"})


def _local_authority(value: str, *, origin: bool = False) -> bool:
    """Accept explicit loopback authorities, rejecting URL parser ambiguities."""
    if not value or any(char.isspace() or ord(char) < 32 for char in value):
        return False
    try:
        parsed = urlsplit(value if origin else "//" + value)
        if origin and parsed.scheme not in {"http", "https"}:
            return False
        if parsed.username is not None or parsed.password is not None:
            return False
        if parsed.path or parsed.query or parsed.fragment or "\\" in value:
            return False
        if parsed.hostname not in _LOCAL_HOSTS:
            return False
        port = parsed.port
        return port is None or 1 <= port <= 65535
    except ValueError:
        return False


def _api_token(token: str | None) -> str | None:
    token = os.environ.get("APM_API_TOKEN") if token is None else token
    if token is None or token == "":
        return None
    if not isinstance(token, str) or any(not 33 <= ord(char) <= 126 for char in token):
        raise ValueError("APM_API_TOKEN must contain printable ASCII without spaces.")
    return token


def create_app(home: Any, token: str | None = None, *, tasks=None, scheduler=None, music=None):
    """Create a local API around a validated HomeController-compatible object.

    Credentials stay within the controller: integration views and context must
    already be redacted. A configured token protects controller routes and
    health; public schemas let the local docs UI load before authorization.
    No cross-origin browser access is enabled.
    """
    try:
        from fastapi import Depends, FastAPI, Request, Response
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse
        from fastapi.security import HTTPBearer
        from pydantic import BaseModel, ConfigDict, Field, StrictStr
        from starlette.concurrency import run_in_threadpool
        from starlette.exceptions import HTTPException
    except ImportError as exc:
        raise RuntimeError("Install the home extra to run the API: pip install -e '.[home]'") from exc

    configured_token = _api_token(token)
    if music is None:
        from .music import MusicService
        music = MusicService()
    assistant = home
    if tasks is not None:
        from .assistant import AssistantController
        assistant = AssistantController(home, tasks, music=music)

    @asynccontextmanager
    async def lifespan(app):
        try:
            if scheduler is not None:
                await run_in_threadpool(scheduler.start)
            yield
        finally:
            try:
                if scheduler is not None:
                    await run_in_threadpool(scheduler.stop)
                    if getattr(scheduler, "running", False):
                        raise RuntimeError("Scheduler is still running; task storage remains open.")
                if tasks is not None:
                    await run_in_threadpool(tasks.close)
            finally:
                try:
                    await run_in_threadpool(music.close)
                finally:
                    await run_in_threadpool(home.close)

    security = [Depends(HTTPBearer(auto_error=False))] if configured_token is not None else []
    app = FastAPI(title="APM Local Assistant API", version="0.1.0", lifespan=lifespan,
                  dependencies=security)

    def error(status: int, detail: str, headers: dict | None = None):
        return JSONResponse(status_code=status, content={"detail": detail}, headers=headers)

    @app.middleware("http")
    async def local_requests_only(request: Request, call_next):
        hosts = request.headers.getlist("host")
        if len(hosts) != 1 or not _local_authority(hosts[0]):
            return error(400, "Untrusted Host header.")
        origins = request.headers.getlist("origin")
        if origins and (len(origins) != 1 or not _local_authority(origins[0], origin=True)):
            return error(403, "Untrusted request origin.")
        if configured_token is not None and request.url.path not in _PUBLIC_DOC_PATHS:
            authorization = request.headers.getlist("authorization")
            supplied = authorization[0].split(" ", 1) if len(authorization) == 1 else []
            valid = (
                len(supplied) == 2
                and supplied[0].lower() == "bearer"
                and secrets.compare_digest(supplied[1].encode("utf-8"), configured_token.encode("ascii"))
            )
            if not valid:
                return error(401, "Bearer authentication required.", {"WWW-Authenticate": "Bearer"})
        try:
            response = await call_next(request)
        except Exception:
            # Never echo adapter errors, request bodies, or configuration secrets.
            return error(500, "Internal server error.")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(KeyError)
    async def missing_resource(request: Request, exc: KeyError):
        return error(404, "Resource not found.")

    @app.exception_handler(ValueError)
    @app.exception_handler(ValidationError)
    async def invalid_configuration(request: Request, exc: Exception):
        return error(400, "Invalid configuration or command.")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # Pydantic errors can include the submitted value, including credentials.
        return error(422, "Invalid request body or parameters.")

    @app.exception_handler(BackendError)
    async def backend_failure(request: Request, exc: BackendError):
        return error(502, "Device backend operation failed; command outcome may be unknown.")

    @app.exception_handler(HTTPException)
    async def http_failure(request: Request, exc: HTTPException):
        try:
            detail = HTTPStatus(exc.status_code).phrase
        except ValueError:
            detail = "Request failed."
        headers = {key: value for key, value in (exc.headers or {}).items()
                   if key.lower() in {"allow", "www-authenticate"}}
        return error(exc.status_code, detail, headers)

    class CommandRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        state: StrictStr

    class MusicRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        title: StrictStr = Field(min_length=1, max_length=200)
        artist: StrictStr | None = Field(default=None, min_length=1, max_length=200)
        version: Literal["studio", "live", "remix", "acoustic", "karaoke", "instrumental", "remaster"] | None = None

    class EmptyRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/integrations")
    def integrations():
        return home.integrations()

    @app.put("/v1/integrations/{integration_id}")
    def put_integration(integration_id: str, config: dict[str, Any]):
        return home.put_integration(integration_id, config)

    @app.get("/v1/integrations/{integration_id}/discover")
    def discover(integration_id: str):
        return home.discover(integration_id)

    @app.get("/v1/devices")
    def devices():
        return home.devices()

    @app.put("/v1/devices/{device_id}")
    def put_device(device_id: str, config: dict[str, Any]):
        return home.put_device(device_id, config)

    @app.delete("/v1/devices/{device_id}", status_code=204)
    def remove_device(device_id: str):
        home.remove_device(device_id)
        return Response(status_code=204)

    @app.get("/v1/devices/{device_id}/state")
    def get_state(device_id: str):
        return home.get_state(device_id)

    @app.post("/v1/devices/{device_id}/commands")
    def command(device_id: str, request: CommandRequest):
        return home.command(device_id, request.state)

    @app.get("/v1/context")
    def context():
        context = assistant.context()
        if tasks is None:
            from .toolsets.music import MUSIC_TOOLS
            context = {**context, "music": music.status(),
                       "tools": context["tools"] + deepcopy(MUSIC_TOOLS)}
        return context

    @app.get("/v1/music/status")
    def music_status():
        return music.status()

    @app.post("/v1/music/resolve")
    def resolve_music(request: MusicRequest):
        return music.resolve(request.title, artist=request.artist, version=request.version)

    @app.post("/v1/music/play")
    def play_music(request: MusicRequest):
        return music.play(request.title, artist=request.artist, version=request.version)

    @app.post("/v1/music/pause")
    def pause_music(request: EmptyRequest | None = None):
        return music.pause()

    @app.post("/v1/music/resume")
    def resume_music(request: EmptyRequest | None = None):
        return music.resume()

    @app.post("/v1/music/selections/{selection_id}/play")
    def play_music_selection(selection_id: str, request: EmptyRequest | None = None):
        return music.select(selection_id)

    if tasks is not None:
        class TaskRequest(BaseModel):
            model_config = ConfigDict(extra="forbid")

        class TimerRequest(TaskRequest):
            name: StrictStr = Field(min_length=1, max_length=120)
            duration_seconds: float = Field(strict=True, gt=0, le=31_536_000, allow_inf_nan=False)

        class ReminderRequest(TaskRequest):
            name: StrictStr = Field(min_length=1, max_length=120)
            due_at: StrictStr
            timezone: StrictStr | None = None
            repeat: Literal["daily", "weekly"] | None = None

        class TimerAction(TaskRequest):
            action: Literal["pause", "resume", "extend", "cancel", "snooze", "complete"]
            seconds: float | None = Field(default=None, strict=True, gt=0, le=31_536_000, allow_inf_nan=False)

        class ReminderAction(TaskRequest):
            action: Literal["cancel", "snooze", "complete"]
            seconds: float | None = Field(default=None, strict=True, gt=0, le=31_536_000, allow_inf_nan=False)

        def require_kind(task_id: str, kind: str):
            task = tasks.get_task(task_id)
            if task["kind"] != kind:
                raise KeyError(task_id)
            return task

        def public_notification(event):
            if not isinstance(event, dict):
                return event
            return {key: value for key, value in event.items() if key not in {"claim_token", "leased_until"}}

        @app.get("/v1/clock")
        def clock(timezone: str | None = None):
            return tasks.clock(timezone=timezone)

        @app.get("/v1/timers")
        def timers(status: Literal["scheduled", "paused", "completed", "cancelled", "due"] | None = None):
            return tasks.list_tasks(kind="timer", status=status)

        @app.post("/v1/timers", status_code=201)
        def create_timer(request: TimerRequest):
            return tasks.create_timer(request.name, request.duration_seconds)

        @app.get("/v1/timers/{task_id}")
        def get_timer(task_id: str):
            return require_kind(task_id, "timer")

        @app.post("/v1/timers/{task_id}/actions")
        def timer_action(task_id: str, request: TimerAction):
            require_kind(task_id, "timer")
            if (request.action in {"extend", "snooze"}) != (request.seconds is not None):
                raise ValueError("Only extend and snooze accept and require seconds.")
            if request.action == "snooze":
                return tasks.snooze_task(task_id, request.seconds)
            if request.action == "complete":
                return tasks.complete_task(task_id)
            return tasks.update_timer(task_id, request.action, seconds=request.seconds)

        @app.get("/v1/reminders")
        def reminders(status: Literal["scheduled", "paused", "completed", "cancelled", "due"] | None = None):
            return tasks.list_tasks(kind="reminder", status=status)

        @app.post("/v1/reminders", status_code=201)
        def create_reminder(request: ReminderRequest):
            return tasks.create_reminder(request.name, request.due_at,
                                         timezone=request.timezone, repeat=request.repeat)

        @app.get("/v1/reminders/{task_id}")
        def get_reminder(task_id: str):
            return require_kind(task_id, "reminder")

        @app.post("/v1/reminders/{task_id}/actions")
        def reminder_action(task_id: str, request: ReminderAction):
            require_kind(task_id, "reminder")
            if request.action == "snooze":
                if request.seconds is None:
                    raise ValueError("Snooze requires seconds.")
                return tasks.snooze_task(task_id, request.seconds)
            if request.seconds is not None:
                raise ValueError("Only snooze accepts seconds.")
            if request.action == "cancel":
                return tasks.cancel_task(task_id)
            return tasks.complete_task(task_id)

        @app.get("/v1/notifications")
        def notifications(unread_only: bool = False):
            return [public_notification(event) for event in tasks.notifications(unread_only=unread_only)]

        @app.post("/v1/notifications/{notification_id}/acknowledge")
        def acknowledge_notification(notification_id: str, request: TaskRequest | None = None):
            return public_notification(tasks.acknowledge_notification(notification_id))

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local APM assistant API on 127.0.0.1.")
    parser.add_argument("--registry", type=Path, help="Path to a home registry JSON file.")
    parser.add_argument("--init", action="store_true", help="Create a simulation registry before serving; requires --registry.")
    parser.add_argument("--port", type=int, default=8765, help="Loopback port (default: 8765).")
    parser.add_argument("--database", type=Path, default=Path("work/assistant.sqlite3"),
                        help="Persistent timers and reminders database (default: work/assistant.sqlite3).")
    parser.add_argument("--timezone", help="Default IANA timezone, for example America/Los_Angeles; detected when omitted.")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.init and args.registry is None:
        parser.error("--init requires --registry")
    try:
        import uvicorn
    except ImportError:
        parser.exit(2, "Install the home extra to run the API: pip install -e '.[home]'\n")

    from .home import initialize_home, load_home
    from .scheduler import Scheduler
    from .tasks import TaskService

    home = None
    tasks = None
    music = None
    try:
        # Check token syntax before creating a file or loading device adapters.
        token = _api_token(None)
        if args.init:
            initialize_home(args.registry)
        home = load_home(args.registry)
        tasks = TaskService(args.database, timezone=args.timezone)
        from .music_connection import load_music_service
        music = load_music_service()
        app = create_app(home, token=token, tasks=tasks, scheduler=Scheduler(tasks), music=music)
    except (OSError, ValueError, ValidationError, BackendError, RuntimeError, sqlite3.Error):
        if tasks is not None:
            tasks.close()
        if home is not None:
            home.close()
        if music is not None:
            music.close()
        parser.exit(2, "Unable to initialize local API. Check the registry, database, timezone, token, and home dependencies.\n")
    uvicorn.run(app, host="127.0.0.1", port=args.port, proxy_headers=False, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
