"""Homebridge UI REST adapter; no connection occurs until an operation is called.

Verified against official sources (2026-10-07):
https://github.com/homebridge/homebridge-config-ui-x/blob/latest/src/modules/accessories/accessories.controller.ts
https://github.com/homebridge/homebridge-config-ui-x/blob/latest/src/modules/accessories/accessories.service.ts
https://github.com/homebridge/homebridge-config-ui-x/blob/latest/src/core/auth/auth.controller.ts
https://github.com/homebridge/hap-client/blob/latest/src/interfaces.ts
https://github.com/homebridge/HAP-NodeJS/blob/latest/src/lib/definitions/CharacteristicDefinitions.ts

The UI's HAP accessory API requires Homebridge accessory control/insecure mode
to be enabled by its owner. This adapter never changes that setting. Discovery
returns service IDs, not raw HAP aid/iid pairs. Each PUT sets one characteristic.
Username/password login cannot handle interactive 2FA; supply a valid UI token.
The UI may return an error after a physical write if its own refresh fails, so
failed writes are never retried and cannot establish the physical outcome.

Injected transports implement request(method, url, *, headers, body, timeout)
and return (HTTP status, decoded JSON). They must not follow redirects.
"""
from __future__ import annotations

import json
import math
import re
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .base import BackendError, DeviceState


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _HttpTransport:
    def __init__(self):
        # Ignore ambient proxies: these requests carry credentials to an explicitly
        # configured home server. All redirects, even same-origin, are rejected.
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def request(self, method, url, *, headers, body, timeout):
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        request = Request(url, data=data, headers=headers, method=method)
        try:
            response = self._opener.open(request, timeout=timeout)
        except HTTPError as exc:
            status = exc.code
            exc.close()  # Never expose or parse server error bodies.
            return status, None
        with response:
            status = response.status
            data = response.read(4 * 1024 * 1024 + 1)
            if len(data) > 4 * 1024 * 1024:
                raise BackendError("Homebridge response exceeded the size limit")
            try:
                return status, json.loads(data)
            except (ValueError, UnicodeDecodeError):
                # A 2xx write is still acknowledged if its response body is bad.
                # Reads and login validate the response shape before using it.
                return status, None


class HomebridgeBackend:
    _SERVICE_TYPES = {"light": {"Lightbulb", "Switch", "Outlet"}, "garage": {"GarageDoorOpener"}}

    def __init__(self, url: str, *, token: str | None = None,
                 username: str | None = None, password: str | None = None,
                 timeout: float = 5, transport=None):
        try:
            parsed = urlsplit(url)
            valid_url = (isinstance(url, str) and parsed.scheme in {"http", "https"}
                         and parsed.hostname and parsed.port != 0 and parsed.username is None
                         and parsed.password is None and not parsed.query and not parsed.fragment
                         and not any(ch.isspace() or ord(ch) < 32 for ch in url))
        except (ValueError, TypeError, AttributeError):
            valid_url = False
        if not valid_url:
            raise BackendError("Homebridge URL must be an HTTP(S) UI base URL without credentials or query parameters")
        if (isinstance(timeout, bool) or not isinstance(timeout, (float, int))
                or not math.isfinite(timeout) or not 0 < timeout <= 60):
            raise BackendError("Homebridge timeout must be between 0 and 60 seconds")
        if token is not None:
            if not self._valid_token(token) or username is not None or password is not None:
                raise BackendError("Configure either a Homebridge token or username/password")
        elif not (isinstance(username, str) and username.strip()
                  and isinstance(password, str) and password):
            raise BackendError("Homebridge requires a token or username/password")
        self.url = url.rstrip("/")
        self.timeout = float(timeout)
        self._token = token
        self._username = username
        self._password = password
        self._expires_at = None
        self._transport = transport if transport is not None else _HttpTransport()

    @staticmethod
    def _valid_token(token):
        return (isinstance(token, str) and 0 < len(token) <= 16384
                and token.isascii() and not any(ch.isspace() or ord(ch) < 33 for ch in token))

    @classmethod
    def validate_target(cls, kind: str, target: dict) -> None:
        if not isinstance(kind, str) or kind not in cls._SERVICE_TYPES:
            raise BackendError("Homebridge supports light and garage device kinds")
        if (not isinstance(target, dict) or set(target) != {"unique_id"}
                or not isinstance(target["unique_id"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", target["unique_id"])):
            raise BackendError("Homebridge target requires a 64-character lowercase hex unique_id from discovery")

    def _raw(self, method, path, body=None, *, authenticated=True):
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if authenticated:
            headers["Authorization"] = "Bearer " + self._token
        try:
            status, data = self._transport.request(method, self.url + path,
                                                 headers=headers, body=body, timeout=self.timeout)
            if type(status) is not int or not 100 <= status <= 599:
                raise ValueError("Invalid status")
            return status, data
        except Exception:
            suffix = "; write outcome is unknown and was not retried" if method == "PUT" else ""
            raise BackendError("Homebridge request failed or timed out" + suffix) from None

    @staticmethod
    def _check_status(status, method):
        if 200 <= status < 300:
            return
        if status in {401, 403}:
            reason = "Homebridge authentication was rejected"
        elif status == 412:
            reason = "Homebridge login requires two-factor authentication; provide a valid UI token"
        elif 300 <= status < 400:
            reason = "Homebridge redirect refused; configure the final UI URL"
        elif status == 400:
            reason = "Homebridge rejected the request; check the target and that accessory control is enabled"
        else:
            reason = f"Homebridge returned HTTP {status}"
        if method == "PUT":
            reason += "; write outcome is unknown and was not retried"
        raise BackendError(reason)

    def _login(self):
        status, data = self._raw("POST", "/api/auth/login", {
            "username": self._username, "password": self._password}, authenticated=False)
        self._check_status(status, "POST")
        if not isinstance(data, dict) or not self._valid_token(data.get("access_token")):
            raise BackendError("Homebridge returned an invalid authentication response")
        self._token = data["access_token"]
        ttl = data.get("expires_in")
        self._expires_at = (time.monotonic() + max(0, ttl - min(5, ttl / 10))
                            if type(ttl) in (float, int) and math.isfinite(ttl) and ttl > 0 else None)

    def _api(self, method, path, body=None):
        if self._token is None or (self._expires_at is not None and time.monotonic() >= self._expires_at):
            self._login()
        status, data = self._raw(method, path, body)
        # Only a read can be repeated after renewing expired login credentials.
        if status == 401 and method == "GET" and self._username is not None:
            self._login()
            status, data = self._raw(method, path, body)
        self._check_status(status, method)
        return data

    @staticmethod
    def _characteristic(service, name):
        characteristics = service.get("serviceCharacteristics")
        if not isinstance(characteristics, list):
            raise BackendError("Homebridge returned invalid service characteristics")
        matches = [item for item in characteristics if isinstance(item, dict) and item.get("type") == name]
        if len(matches) != 1:
            raise BackendError("Homebridge required characteristic is missing or ambiguous")
        return matches[0]

    def _service(self, kind, target):
        self.validate_target(kind, target)
        service = self._api("GET", "/api/accessories/" + target["unique_id"])
        if (not isinstance(service, dict) or service.get("uniqueId") != target["unique_id"]
                or not isinstance(service.get("type"), str)
                or service.get("type") not in self._SERVICE_TYPES[kind]):
            raise BackendError("Homebridge service does not match the configured target and kind")
        return service

    def _state(self, kind, service):
        characteristic = self._characteristic(service, "On" if kind == "light" else "CurrentDoorState")
        if characteristic.get("canRead") is not True:
            raise BackendError("Homebridge state characteristic is not readable")
        value = characteristic.get("value")
        if kind == "light":
            # Avoid Python's truthiness: the string 'false' is not true/on.
            if type(value) is bool:
                return "on" if value else "off"
            return {0: "off", 1: "on"}.get(value, "unknown") if type(value) is int else "unknown"
        return {0: "open", 1: "closed", 2: "opening", 3: "closing", 4: "stopped"}.get(value, "unknown") if type(value) is int else "unknown"

    def discover(self) -> list[dict]:
        services = self._api("GET", "/api/accessories")
        if not isinstance(services, list):
            raise BackendError("Homebridge returned an invalid accessory list")
        result, seen = [], set()
        for service in services:
            if not isinstance(service, dict) or not isinstance(service.get("type"), str):
                continue
            kind = next((kind for kind, types in self._SERVICE_TYPES.items() if service.get("type") in types), None)
            if kind is None:
                continue
            target = {"unique_id": service.get("uniqueId")}
            self.validate_target(kind, target)
            if target["unique_id"] in seen:
                raise BackendError("Homebridge returned duplicate accessory IDs")
            seen.add(target["unique_id"])
            name = service.get("serviceName")
            name = name if isinstance(name, str) and name.strip() else service["type"]
            try:
                readable = self._characteristic(service, "On" if kind == "light" else "CurrentDoorState").get("canRead") is True
                writable = self._characteristic(service, "On" if kind == "light" else "TargetDoorState").get("canWrite") is True
                available = readable and writable
            except BackendError:
                available = False
            result.append({"kind": kind, "name": name, "available": available, "target": target})
        return result

    def get_state(self, kind: str, target: dict) -> DeviceState:
        return DeviceState(self._state(kind, self._service(kind, target)))

    def set_state(self, kind: str, target: dict, state: str) -> DeviceState:
        self.validate_target(kind, target)
        valid = {"light": {"on", "off"}, "garage": {"open", "closed"}}
        if not isinstance(state, str) or state not in valid[kind]:
            raise BackendError("Unsupported Homebridge requested state")
        # Read first to authenticate and check binding/permissions before mutation.
        service = self._service(kind, target)
        characteristic = "On" if kind == "light" else "TargetDoorState"
        if self._characteristic(service, characteristic).get("canWrite") is not True:
            raise BackendError("Homebridge target characteristic is not writable")
        value = state == "on" if kind == "light" else (0 if state == "open" else 1)
        self._api("PUT", "/api/accessories/" + target["unique_id"],
                  {"characteristicType": characteristic, "value": value})
        try:
            observed = self.get_state(kind, target).state
        except BackendError:
            observed = "unknown"
        return DeviceState(observed, accepted=True)

    def close(self) -> None:
        pass
