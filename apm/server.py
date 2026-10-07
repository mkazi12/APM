"""Loopback-only REST interface for the local home registry and device adapters.

This is an initial local API, not a remotely accessible mobile service. LAN
exposure, secure pairing, and a chat endpoint are future work. Discovery only
returns proposals; devices must be explicitly registered before use.
"""

import argparse
from contextlib import asynccontextmanager
from http import HTTPStatus
import os
from pathlib import Path
import secrets
from typing import Any
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


def create_app(home: Any, token: str | None = None):
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
        from pydantic import BaseModel, ConfigDict, StrictStr
        from starlette.concurrency import run_in_threadpool
        from starlette.exceptions import HTTPException
    except ImportError as exc:
        raise RuntimeError("Install the home extra to run the API: pip install -e '.[home]'") from exc

    configured_token = _api_token(token)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await run_in_threadpool(home.close)

    security = [Depends(HTTPBearer(auto_error=False))] if configured_token is not None else []
    app = FastAPI(title="APM Local Home API", version="0.1.0", lifespan=lifespan,
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
        return home.context()

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local APM home API on 127.0.0.1.")
    parser.add_argument("--registry", type=Path, help="Path to a home registry JSON file.")
    parser.add_argument("--init", action="store_true", help="Create a simulation registry before serving; requires --registry.")
    parser.add_argument("--port", type=int, default=8765, help="Loopback port (default: 8765).")
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

    home = None
    try:
        # Check token syntax before creating a file or loading device adapters.
        token = _api_token(None)
        if args.init:
            initialize_home(args.registry)
        home = load_home(args.registry)
        app = create_app(home, token=token)
    except (OSError, ValueError, ValidationError, BackendError, RuntimeError):
        if home is not None:
            home.close()
        parser.exit(2, "Unable to initialize local API. Check the registry, token, and home dependencies.\n")
    uvicorn.run(app, host="127.0.0.1", port=args.port, proxy_headers=False, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
