"""Persistent device registry and the shared app/LLM execution boundary.

The API is the single registry writer. Voice clients reload its atomic JSON
snapshots before each request. Creating/loading a home performs no network I/O.
"""
from __future__ import annotations

from copy import deepcopy
import json
import hashlib
import math
import os
from pathlib import Path
import re
import tempfile
from threading import RLock
from urllib.parse import urlsplit

from jsonschema import ValidationError, validate

from .backends.base import BackendError, DeviceState
from .tools import tool

ID = r"^[a-z][a-z0-9_]{0,63}$"
TEXT = {"type": "string", "minLength": 1, "maxLength": 120}
ENV = {"type": "string", "pattern": r"^[A-Z][A-Z0-9_]{0,127}$"}
INTEGRATION_SCHEMA = {
    "type": "object", "required": ["type"], "additionalProperties": False,
    "properties": {
        "type": {"enum": ["simulated", "homebridge", "matter"]},
        "url": {"type": "string", "maxLength": 2048},
        "timeout": {"type": "number", "exclusiveMinimum": 0, "maximum": 60},
        "token_env": ENV, "username_env": ENV, "password_env": ENV,
    },
}
DEVICE_SCHEMA = {
    "type": "object", "required": ["name", "kind", "integration", "target"],
    "additionalProperties": False,
    "properties": {
        "name": TEXT, "room": TEXT,
        "aliases": {"type": "array", "items": TEXT, "uniqueItems": True, "maxItems": 12},
        "kind": {"enum": ["light", "garage"]},
        "integration": {"type": "string", "pattern": ID},
        "target": {"type": "object"},
    },
}
DEFAULT_HOME = {
    "version": 1, "integrations": {"demo": {"type": "simulated"}},
    "devices": {
        "kitchen_lights": {"name": "Kitchen lights", "room": "Kitchen", "aliases": ["kitchen light"],
                           "kind": "light", "integration": "demo", "target": {"id": "kitchen_lights"}},
        "garage": {"name": "Garage door", "room": "Garage", "aliases": ["garage"],
                   "kind": "garage", "integration": "demo", "target": {"id": "garage"}},
    },
}


def _validate(value, schema, description):
    try:
        validate(value, schema)
    except ValidationError:
        # Never include submitted config (possibly including misplaced secrets).
        raise ValueError(f"Invalid {description}; check its fields and types") from None


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(ID, value):
        raise ValueError("IDs must start with a lowercase letter and use at most 64 letters, digits or underscores")


def _integration(config):
    _validate(config, INTEGRATION_SCHEMA, "integration")
    if "timeout" in config and not math.isfinite(config["timeout"]):
        raise ValueError("Integration timeout must be finite")
    kind = config["type"]
    if kind == "simulated":
        if set(config) != {"type"}:
            raise ValueError("Simulated integrations only accept type")
        return
    try:
        url = config["url"]
        parsed = urlsplit(url)
        schemes = {"http", "https"} if kind == "homebridge" else {"ws", "wss"}
        if (parsed.scheme not in schemes or not parsed.hostname or parsed.port == 0
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or any(c.isspace() or ord(c) < 32 for c in url)):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ValueError("Integration requires a valid server URL without embedded credentials or query parameters") from None
    credentials = set(config) & {"token_env", "username_env", "password_env"}
    if kind == "homebridge" and credentials not in ({"token_env"}, {"username_env", "password_env"}):
        raise ValueError("Homebridge requires token_env or both username_env and password_env")
    if kind == "matter" and credentials:
        raise ValueError("Matter credentials are managed by the local Matter controller")


class _SimulatedBackend:
    def __init__(self):
        self.states = {}

    @staticmethod
    def validate_target(kind, target):
        if not isinstance(target, dict) or set(target) != {"id"}:
            raise ValueError("Simulated target requires an id")
        _id(target["id"])

    def discover(self):
        return []

    def get_state(self, kind, target):
        return DeviceState(self.states.get(target["id"], "on" if kind == "light" else "open"))

    def set_state(self, kind, target, state):
        self.states[target["id"]] = state
        return DeviceState(state, accepted=True)

    def close(self):
        pass


def _adapter_class(kind):
    if kind == "simulated":
        return _SimulatedBackend
    if kind == "homebridge":
        from .backends.homebridge import HomebridgeBackend
        return HomebridgeBackend
    from .backends.matter import MatterBackend
    return MatterBackend


def _check_document(document):
    if (not isinstance(document, dict) or set(document) != {"version", "integrations", "devices"}
            or type(document["version"]) is not int or document["version"] != 1
            or not isinstance(document["integrations"], dict) or not isinstance(document["devices"], dict)
            or len(document["integrations"]) > 16 or len(document["devices"]) > 64):
        raise ValueError("Expected a version 1 home registry with up to 16 integrations and 64 devices")
    for key, config in document["integrations"].items():
        _id(key)
        _integration(config)
    bindings = set()
    for key, config in document["devices"].items():
        _id(key)
        _validate(config, DEVICE_SCHEMA, "device")
        integration = document["integrations"].get(config["integration"])
        if integration is None:
            raise ValueError("Device references an unregistered integration")
        _adapter_class(integration["type"]).validate_target(config["kind"], config["target"])
        identity = (config["integration"] if integration["type"] == "simulated"
                    else integration["url"].rstrip("/"))
        binding = (integration["type"], identity, json.dumps(config["target"], sort_keys=True))
        if binding in bindings:
            raise ValueError("This integration target is already registered")
        bindings.add(binding)


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".home-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as output:
            json.dump(document, output, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def initialize_home(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents an --init typo from overwriting a real home.
    with path.open("x") as output:
        os.chmod(path, 0o600)
        json.dump(DEFAULT_HOME, output, indent=2)
        output.write("\n")


def load_home(path=None):
    return HomeController(path)


class HomeController:
    def __init__(self, path=None, adapters=None):
        self.path = Path(path) if path is not None else None
        self._lock = RLock()
        self._adapters = {}
        self._document = deepcopy(DEFAULT_HOME)
        self._reload()
        self._adapters.update(adapters or {})

    def _reload(self):
        if self.path is None:
            return
        try:
            with self.path.open() as source:
                document = json.load(source)
        except (OSError, ValueError):
            raise ValueError("Cannot read the home registry; initialize it first and check its JSON") from None
        _check_document(document)
        for key in list(self._adapters):
            if document["integrations"].get(key) != self._document["integrations"].get(key):
                self._adapters.pop(key).close()
        self._document = document

    def _save(self, document):
        _check_document(document)
        if self.path is not None:
            _write(self.path, document)
        for key in list(self._adapters):
            if document["integrations"].get(key) != self._document["integrations"].get(key):
                self._adapters.pop(key).close()
        self._document = document

    def _adapter(self, key):
        if key not in self._adapters:
            config = self._document["integrations"][key]
            cls = _adapter_class(config["type"])
            if config["type"] == "simulated":
                adapter = cls()
            else:
                options = {"timeout": config.get("timeout", 5 if config["type"] == "homebridge" else 10)}
                for name in ("token", "username", "password"):
                    env = config.get(name + "_env")
                    if env:
                        value = os.environ.get(env)
                        if not value:
                            raise BackendError("Integration credential environment variable is not set")
                        options[name] = value
                adapter = cls(config["url"], **options)
            self._adapters[key] = adapter
        return self._adapters[key]

    def integrations(self):
        with self._lock:
            self._reload()
            return [{"id": key, "type": config["type"],
                     **({"url": config["url"]} if "url" in config else {}),
                     "configured": all(bool(os.environ.get(config[name])) for name in
                                       ("token_env", "username_env", "password_env") if name in config)}
                    for key, config in self._document["integrations"].items()]

    def put_integration(self, key, config):
        with self._lock:
            self._reload()
            document = deepcopy(self._document)
            document["integrations"][key] = deepcopy(config)
            self._save(document)
            return next(item for item in self.integrations() if item["id"] == key)

    def discover(self, key):
        with self._lock:
            self._reload()
            return self._adapter(key).discover()

    def _device(self, key, config):
        return {"id": key, **deepcopy(config),
                "simulated": self._document["integrations"][config["integration"]]["type"] == "simulated",
                "capabilities": ["on", "off"] if config["kind"] == "light" else ["open", "closed"]}

    def devices(self):
        with self._lock:
            self._reload()
            return [self._device(key, value) for key, value in self._document["devices"].items()]

    def put_device(self, key, config):
        with self._lock:
            self._reload()
            document = deepcopy(self._document)
            document["devices"][key] = deepcopy(config)
            self._save(document)
            return self._device(key, config)

    def remove_device(self, key):
        with self._lock:
            self._reload()
            document = deepcopy(self._document)
            del document["devices"][key]
            self._save(document)

    def context(self):
        with self._lock:
            self._reload()
            # Model-visible metadata intentionally omits URLs, credentials, and protocol targets.
            devices = [{field: value for field, value in self._device(key, config).items()
                        if field not in {"integration", "target"}}
                       for key, config in self._document["devices"].items()]
            lights = [item["id"] for item in devices if item["kind"] == "light"]
            garages = [item["id"] for item in devices if item["kind"] == "garage"]
            tools = []
            if lights:
                tools.append(tool("set_lights", "Set a registered light or switch on or off.", {
                    "device": {"type": "string", "enum": lights}, "on": {"type": "boolean"}}))
            if garages:
                tools.append(tool("set_garage", "Set a registered garage door target position.", {
                    "device": {"type": "string", "enum": garages},
                    "target": {"type": "string", "enum": ["open", "closed"]}}))
            if devices:
                tools.append(tool("get_device_state", "Read the current device state.", {
                    "device": {"type": "string", "enum": [item["id"] for item in devices]}}))
            revision = hashlib.sha256(json.dumps(self._document, sort_keys=True).encode()).hexdigest()
            return {"devices": devices, "tools": tools, "revision": revision}

    def _operate(self, device, state=None):
        config = self._document["devices"][device]
        adapter = self._adapter(config["integration"])
        observed = (adapter.get_state(config["kind"], config["target"]) if state is None else
                    adapter.set_state(config["kind"], config["target"], state))
        return {"device": device, "label": config["name"], "state": observed.state,
                "simulated": self._document["integrations"][config["integration"]]["type"] == "simulated",
                "ok": True, "accepted": observed.accepted,
                **({"requested_state": state} if state is not None else {})}

    def get_state(self, key):
        with self._lock:
            self._reload()
            return self._operate(key)

    def command(self, key, state):
        with self._lock:
            self._reload()
            kind = self._document["devices"][key]["kind"]
            allowed = ("on", "off") if kind == "light" else ("open", "closed")
            if state not in allowed:
                raise ValueError("Unsupported state for this device")
            return self._operate(key, state)

    def snapshot(self):
        calls = [{"function": {"name": "get_device_state", "arguments": {"device": device["id"]}}}
                 for device in self.devices()]
        # Inventory still reads reachable devices when a different device is offline.
        return [result for call in calls for result in self.execute([call])]

    def execute(self, calls, expected_revision=None):
        with self._lock:
            context = self.context()
            if expected_revision is not None and context["revision"] != expected_revision:
                raise ValueError("Device registry changed during the request; please try again")
            schemas = {item["function"]["name"]: item["function"]["parameters"] for item in context["tools"]}
            if not isinstance(calls, list) or len(calls) > 8:
                raise ValueError("Expected at most eight tool calls")
            checked = []
            for call in calls:
                try:
                    name, args = call["function"]["name"], call["function"]["arguments"]
                    schema = schemas[name]
                except (KeyError, TypeError):
                    raise ValueError("Unknown or malformed tool call") from None
                _validate(args, schema, "tool arguments")
                checked.append((name, args))
            results, failed = [], False
            for name, args in checked:
                device = args["device"]
                config = self._document["devices"][device]
                base = {"tool": name, "device": device, "label": config["name"],
                        "simulated": self._document["integrations"][config["integration"]]["type"] == "simulated"}
                if failed:
                    results.append({**base, "ok": False, "state": "unknown", "error": "Skipped after an earlier device error"})
                    continue
                state = ("on" if args["on"] else "off") if name == "set_lights" else args.get("target")
                try:
                    result = self._operate(device, state)
                    results.append({**base, **result})
                    # An acknowledged but unconfirmed write is not a confirmed physical success.
                    failed = state is not None and result["state"] == "unknown"
                except BackendError as exc:
                    results.append({**base, "ok": False, "state": "unknown", "error": str(exc)})
                    failed = True
            return results

    def close(self):
        with self._lock:
            for adapter in self._adapters.values():
                adapter.close()
            self._adapters.clear()
