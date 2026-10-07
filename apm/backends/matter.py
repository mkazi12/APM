"""On/Off lights through a configured, already commissioned Matter Server.

Uses the schema-11 Python Matter Server WebSocket API, also supported by the
Matter.js server. No commissioning or network discovery is performed.
Protocol: https://github.com/matter-js/python-matter-server/blob/main/docs/websockets_api.md
Wire models: https://github.com/matter-js/python-matter-server/blob/main/matter_server/common/models.py

``transport`` can be a connected WebSocket-like object implementing send(str),
recv(), settimeout(seconds), close(), or a factory(url, timeout=seconds) returning
one. Factories and the optional websocket-client dependency are used lazily.
Injected transports must honor settimeout for send/recv and close promptly.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import threading
import time
from urllib.parse import urlsplit
import uuid

from .base import BackendError, DeviceState

SCHEMA_VERSION = 11


class MatterBackend:
    def __init__(self, url: str, *, timeout: float = 10, transport=None):
        try:
            if not isinstance(url, str):
                raise ValueError("URL must be a string")
            parsed = urlsplit(url)
            valid = (parsed.scheme in {"ws", "wss"} and parsed.hostname
                     and not parsed.username and not parsed.password
                     and not parsed.query and not parsed.fragment)
            _ = parsed.port
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise BackendError("Matter URL must be ws:// or wss:// with a host and no embedded credentials, query, or fragment")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise BackendError("Matter timeout must be a positive finite number")
        self.url = url
        self.timeout = float(timeout)
        self._transport = transport
        self._transport_used = False
        self._socket = None
        self._lock = threading.RLock()

    @staticmethod
    def validate_target(kind: str, target: dict) -> None:
        if kind != "light":
            raise BackendError("Matter currently supports On/Off lights only; garage control is unsupported")
        if not isinstance(target, dict) or set(target) != {"node_id", "endpoint"}:
            raise BackendError("Matter light target requires only node_id and endpoint")
        if type(target["node_id"]) is not int or not 0 < target["node_id"] < 2**64:
            raise BackendError("Matter node_id must be a positive 64-bit integer")
        if type(target["endpoint"]) is not int or not 0 <= target["endpoint"] < 65535:
            raise BackendError("Matter endpoint must be an integer from 0 through 65534")

    @contextmanager
    def _operation(self):
        deadline = time.monotonic() + self.timeout
        if not self._lock.acquire(timeout=self.timeout):
            raise BackendError("Matter operation timed out waiting for the connection")
        try:
            self._connect(deadline)
            yield deadline
        finally:
            self._lock.release()

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BackendError("Matter operation timed out; no automatic retry was sent")
        return remaining

    def _connect(self, deadline: float) -> None:
        if self._socket is not None:
            return
        factory = self._transport
        if factory is None:
            try:
                from websocket import create_connection
            except ImportError:
                raise BackendError("Matter support requires the optional websocket-client package; install the home extra") from None
            factory = create_connection
        try:
            if callable(factory):
                self._socket = factory(self.url, timeout=self._remaining(deadline))
            else:
                if self._transport_used:
                    raise BackendError("Injected Matter transport is closed; use a factory to create a new connection")
                self._socket = factory
                self._transport_used = True
            # The server speaks first. There is no client 'hello' request.
            hello = self._receive(deadline)
            version, minimum = hello.get("schema_version"), hello.get("min_supported_schema_version")
            if (not isinstance(hello.get("sdk_version"), str)
                    or type(version) is not int or type(minimum) is not int
                    or not minimum <= SCHEMA_VERSION <= version):
                raise BackendError("Matter Server does not advertise compatible schema 11")
            self._validate_nodes(self._rpc("start_listening", {}, deadline))
        except BackendError:
            self._disconnect()
            raise
        except Exception:
            self._disconnect()
            raise BackendError("Could not connect to Matter Server") from None

    def _receive(self, deadline: float) -> dict:
        try:
            self._socket.settimeout(self._remaining(deadline))
            raw = self._socket.recv()
            if not raw:
                raise BackendError("Matter Server closed the connection")
            message = json.loads(raw)
            if not isinstance(message, dict):
                raise BackendError("Matter Server sent an invalid message")
            self._remaining(deadline)
            return message
        except BackendError:
            self._disconnect()
            raise
        except Exception:
            self._disconnect()
            raise BackendError("Matter connection failed or timed out; no automatic retry was sent") from None

    def _rpc(self, command: str, args: dict, deadline: float):
        message_id = uuid.uuid4().hex
        try:
            self._socket.settimeout(self._remaining(deadline))
            self._socket.send(json.dumps({"message_id": message_id, "command": command, "args": args}))
        except BackendError:
            self._disconnect()
            raise
        except Exception:
            self._disconnect()
            raise BackendError("Matter request could not be acknowledged; no automatic retry was sent") from None
        while True:
            message = self._receive(deadline)
            if message.get("event") == "server_shutdown":
                self._disconnect()
                raise BackendError("Matter Server is shutting down")
            if "event" in message or message.get("message_id") != message_id:
                continue
            if "error_code" in message:
                # Server details and low-level exception text can contain secrets.
                code = message["error_code"] if type(message["error_code"]) is int else "unknown"
                raise BackendError(f"Matter Server rejected {command} (error {code})")
            if "result" not in message:
                self._disconnect()
                raise BackendError("Matter Server response has no result")
            return message["result"]

    @staticmethod
    def _validate_nodes(nodes) -> list[dict]:
        if not isinstance(nodes, list):
            raise BackendError("Matter Server returned an invalid node list")
        seen = set()
        for node in nodes:
            if (not isinstance(node, dict) or type(node.get("node_id")) is not int
                    or not 0 < node["node_id"] < 2**64 or node["node_id"] in seen
                    or type(node.get("available")) is not bool
                    or not isinstance(node.get("attributes"), dict)):
                raise BackendError("Matter Server returned invalid or duplicate node data")
            seen.add(node["node_id"])
        return nodes

    @staticmethod
    def _endpoints(node: dict) -> list[int]:
        attributes = node["attributes"]
        endpoints = set()
        for path in attributes:
            if not isinstance(path, str):
                continue
            parts = path.split("/")
            if len(parts) == 3 and parts[0].isascii() and parts[0].isdigit():
                endpoint = int(parts[0])
                if 0 <= endpoint < 65535:
                    endpoints.add(endpoint)
        supported = []
        for endpoint in sorted(endpoints):
            descriptor = f"{endpoint}/29/1"  # Descriptor.ServerList
            if descriptor in attributes:
                servers = attributes[descriptor]
                if isinstance(servers, list) and any(type(item) is int and item == 6 for item in servers):
                    supported.append(endpoint)
            elif type(attributes.get(f"{endpoint}/6/0")) is bool:
                # Older/incomplete interviews can still expose a real OnOff attribute.
                supported.append(endpoint)
        return supported

    def _node(self, target: dict, deadline: float) -> dict:
        nodes = self._validate_nodes(self._rpc("get_nodes", {"only_available": False}, deadline))
        node = next((node for node in nodes if node["node_id"] == target["node_id"]), None)
        if node is None:
            raise BackendError("Configured Matter node was not found on this controller")
        if not node["available"]:
            raise BackendError("Configured Matter node is unavailable")
        if target["endpoint"] not in self._endpoints(node):
            raise BackendError("Configured Matter endpoint does not advertise an On/Off server")
        return node

    @staticmethod
    def _name(node: dict, endpoint: int) -> str:
        attributes = node["attributes"]
        for path in (f"{endpoint}/57/5", f"{endpoint}/40/5", "0/40/5", "0/40/3"):
            label = attributes.get(path)
            if isinstance(label, str):
                label = " ".join("".join(char for char in label if char.isprintable()).split())[:120]
                if label:
                    return label
        return f"Matter node {node['node_id']} endpoint {endpoint}"

    def discover(self) -> list[dict]:
        """List the controller's commissioned On/Off endpoints; never scan/pair."""
        with self._operation() as deadline:
            nodes = self._validate_nodes(self._rpc("get_nodes", {"only_available": False}, deadline))
            return [{"kind": "light", "name": self._name(node, endpoint),
                     "available": node["available"],
                     "target": {"node_id": node["node_id"], "endpoint": endpoint}}
                    for node in nodes for endpoint in self._endpoints(node)]

    def _read_state(self, target: dict, deadline: float) -> str:
        path = f"{target['endpoint']}/6/0"
        result = self._rpc("read_attribute", {"node_id": target["node_id"], "attribute_path": path}, deadline)
        if not isinstance(result, dict) or type(result.get(path)) is not bool:
            raise BackendError("Matter On/Off attribute was missing or invalid")
        return "on" if result[path] else "off"

    def get_state(self, kind: str, target: dict) -> DeviceState:
        self.validate_target(kind, target)
        with self._operation() as deadline:
            self._node(target, deadline)
            return DeviceState(self._read_state(target, deadline))

    def set_state(self, kind: str, target: dict, state: str) -> DeviceState:
        self.validate_target(kind, target)
        if not isinstance(state, str) or state not in {"on", "off"}:
            raise BackendError("Matter light state must be on or off")
        with self._operation() as deadline:
            self._node(target, deadline)
            self._rpc("device_command", {"node_id": target["node_id"], "endpoint_id": target["endpoint"],
                      "cluster_id": 6, "command_name": "On" if state == "on" else "Off", "payload": {}}, deadline)
            try:
                observed = self._read_state(target, deadline)
            except BackendError:
                return DeviceState("unknown", accepted=True)
            return DeviceState(observed, accepted=True)

    def _disconnect(self) -> None:
        socket, self._socket = self._socket, None
        if socket is not None:
            try:
                # websocket-client.shutdown() closes immediately, without waiting
                # for the peer's close handshake after an operation deadline.
                shutdown = getattr(socket, "shutdown", None)
                shutdown() if callable(shutdown) else socket.close()
            except Exception:
                pass

    def close(self) -> None:
        with self._lock:
            self._disconnect()
