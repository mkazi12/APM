"""A loopback MusicKit player, catalog adapter, and shared assistant endpoint."""
import argparse
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import secrets
import tempfile
from typing import Literal
from urllib.parse import urlsplit

from .music import MusicService
from .music_connection import DEFAULT_CONNECTION

DEFAULT_SETTINGS = Path(__file__).resolve().parent.parent / "work" / "apple-music.json"
STATIC = Path(__file__).resolve().parent / "static"


def create_music_app(token, *, settings_path=DEFAULT_SETTINGS, provider=None, signer=None):
    from fastapi import FastAPI, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator
    from .apple_music_tokens import DeveloperToken
    from .musickit import MusicKitProvider, validate_completion_diagnostics
    from .server import _local_authority

    signer = signer if signer is not None else DeveloperToken(settings_path)
    provider = provider if provider is not None else MusicKitProvider(signer)
    music = MusicService(provider)

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            music.close()

    app = FastAPI(title="APM Apple Music", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        hosts = request.headers.getlist("host")
        if len(hosts) != 1 or not _local_authority(hosts[0]):
            return JSONResponse({"error": "Untrusted host"}, status_code=400)
        origins = request.headers.getlist("origin")
        if origins and (len(origins) != 1 or urlsplit(origins[0]).netloc != hosts[0]
                        or not _local_authority(origins[0], origin=True)):
            return JSONResponse({"error": "Untrusted origin"}, status_code=403)
        if request.url.path not in {"/", "/static/music.js"}:
            auth = request.headers.getlist("authorization")
            if len(auth) != 1 or not secrets.compare_digest(auth[0].encode(), ("Bearer " + token).encode()):
                return JSONResponse({"error": "Open the connection link printed by apm-music"}, status_code=401)
        try:
            response = await call_next(request)
        except Exception:
            return JSONResponse({"error": "Music operation failed; playback may be unconfirmed"}, status_code=503)
        response.headers["Cache-Control"] = "no-store"
        # Apple's authorization page uses document.referrer to establish the
        # opener's origin, even when MusicKit supplies a referrer query argument.
        # Suppressing it prevents Apple's popup from creating its callback.
        # Share only the document's origin, never its path/query/fragment.
        response.headers["Referrer-Policy"] = "strict-origin" if request.url.path == "/" else "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "frame-ancestors 'none'; base-uri 'none'; object-src 'none'"
        return response

    @app.exception_handler(ValueError)
    async def bad_value(_request, _exc):
        return JSONResponse({"error": "Invalid or expired music request"}, status_code=400)

    @app.exception_handler(RequestValidationError)
    async def bad_body(_request, _exc):
        return JSONResponse({"error": "Invalid request body"}, status_code=422)

    class Body(BaseModel):
        model_config = ConfigDict(extra="forbid")

    class Session(Body):
        session_id: StrictStr = Field(min_length=36, max_length=36)

    class Activate(Session):
        storefront: StrictStr = Field(min_length=2, max_length=2)
        protocol_version: StrictInt = Field(default=1, ge=1, le=3)

    class Completion(Session):
        result: dict
        diagnostics: dict | None = None

        @field_validator("diagnostics", mode="before")
        @classmethod
        def safe_diagnostics(cls, value):
            return validate_completion_diagnostics(value) if value is not None else None

    class Search(Body):
        title: StrictStr = Field(min_length=1, max_length=200)
        artist: StrictStr | None = Field(default=None, min_length=1, max_length=200)
        version: Literal["studio", "live", "remix", "acoustic", "karaoke", "instrumental", "remaster"] | None = None

    @app.get("/")
    def index():
        return FileResponse(STATIC / "music.html")

    @app.get("/static/music.js")
    def script():
        return FileResponse(STATIC / "music.js", media_type="text/javascript")

    @app.get("/v1/config")
    def config():
        state = signer.status()
        if not state["configured"]:
            return state
        return {"configured": True, "developer_token": signer(), "app_name": "APM Assistant"}

    @app.get("/v1/music/status")
    def status():
        return {"configured": signer.status()["configured"], "provider": provider.name,
                "player": provider.status()}

    @app.post("/v1/player/session")
    def activate(body: Activate):
        if not signer.status()["configured"]:
            raise ValueError("Developer setup incomplete")
        provider.activate(body.session_id, body.storefront, body.protocol_version)
        return {"ok": True}

    @app.post("/v1/player/disconnect")
    def disconnect(body: Session):
        provider.disconnect(body.session_id)
        return {"ok": True}

    @app.get("/v1/player/commands")
    def commands(session_id: str):
        return {"command": provider.poll(session_id)}

    @app.post("/v1/player/commands/{command_id}/result")
    def complete(command_id: str, body: Completion):
        provider.complete(body.session_id, command_id, body.result, diagnostics=body.diagnostics)
        return {"ok": True}

    @app.post("/v1/music/resolve")
    def resolve(body: Search):
        return music.resolve(body.title, body.artist, body.version)

    @app.post("/v1/music/play")
    def play(body: Search):
        return music.play(body.title, body.artist, body.version)

    @app.post("/v1/music/pause")
    def pause(body: Body | None = None):
        return music.pause()

    @app.post("/v1/music/resume")
    def resume(body: Body | None = None):
        return music.resume()

    @app.post("/v1/music/selections/{selection_id}/play")
    def select(selection_id: str, body: Body | None = None):
        return music.select(selection_id)

    return app


def save_connection(path, url, token):
    """Atomically publish a local token for voice clients, owner-readable only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".music-connection-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"url": url, "token": token}, stream)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Connect APM to Apple Music through a local MusicKit browser player")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--connection", type=Path, default=DEFAULT_CONNECTION)
    parser.add_argument("--reuse-connection", action="store_true",
                        help="Reuse the saved connection token so existing APM clients can reconnect")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    import socket
    import uvicorn
    url = f"http://127.0.0.1:{args.port}"
    if args.reuse_connection:
        from .music_connection import RemoteMusicService, load_music_service
        try:
            saved = load_music_service(args.connection)
            if not isinstance(saved, RemoteMusicService) or saved._url != url:
                raise ValueError
            token = saved._token
        except (OSError, ValueError, TypeError, AttributeError):
            parser.error("Cannot reuse connection: a valid saved connection for this port is required")
    else:
        token = secrets.token_urlsafe(32)
    app = create_music_app(token, settings_path=args.settings)
    # Bind before publishing credentials, so a second launch cannot replace a
    # working connection file and then fail because the port is already taken.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", args.port))
        sock.listen(128)
        if not args.reuse_connection:
            save_connection(args.connection, url, token)
            print(f"Apple Music connection: {url}/#token={token}", flush=True)
            print("Keep this server and its player page open. Restart APM to load the connection.", flush=True)
        else:
            print(f"Apple Music connection restored: {url}/", flush=True)
            print("Keep this server open. Reconnect the player in its existing browser tab.", flush=True)
        server = uvicorn.Server(uvicorn.Config(app, proxy_headers=False, access_log=False))
        server.run(sockets=[sock])


if __name__ == "__main__":
    main()
