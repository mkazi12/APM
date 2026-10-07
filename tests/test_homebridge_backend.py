import io
import json
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

from apm.backends.base import BackendError, DeviceState
from apm.backends.homebridge import HomebridgeBackend, _HttpTransport, _NoRedirect

LIGHT = "a" * 64
GARAGE = "b" * 64


def characteristic(name, value, *, writable=True, readable=True):
    return {"type": name, "value": value, "canRead": readable, "canWrite": writable}


def light(value=False):
    return {"uniqueId": LIGHT, "type": "Lightbulb", "serviceName": "Kitchen lights",
            "serviceCharacteristics": [characteristic("Brightness", 70), characteristic("On", value)]}


def garage(current=1, target=1):
    return {"uniqueId": GARAGE, "type": "GarageDoorOpener", "serviceName": "Garage",
            "serviceCharacteristics": [characteristic("TargetDoorState", target),
                                       characteristic("ObstructionDetected", False, writable=False),
                                       characteristic("CurrentDoorState", current, writable=False)]}


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class HomebridgeTests(unittest.TestCase):
    def backend(self, *responses, **kwargs):
        transport = FakeTransport(*responses)
        if not kwargs:
            kwargs = {"token": "private-token"}
        backend = HomebridgeBackend("http://homebridge.local:8581", transport=transport, **kwargs)
        return backend, transport

    def test_construct_and_validate_are_offline_and_reject_unsafe_bindings(self):
        backend, transport = self.backend()
        backend.validate_target("light", {"unique_id": LIGHT})
        backend.validate_target("garage", {"unique_id": GARAGE})
        HomebridgeBackend.validate_target("garage", {"unique_id": GARAGE})
        backend.close()
        self.assertEqual(transport.calls, [])
        for kind, target in [("lock", {"unique_id": LIGHT}), ("light", {}),
                             ("garage", {"unique_id": "../auth/login"}),
                             ("light", {"unique_id": LIGHT, "url": "http://other"})]:
            with self.subTest(kind=kind, target=target), self.assertRaises(BackendError):
                backend.validate_target(kind, target)
        self.assertEqual(transport.calls, [])

    def test_invalid_endpoint_auth_and_timeout_fail_before_connecting(self):
        for url in ("ftp://host", "http://user:secret@host", "http://host?secret=abc", "http://host:99999", "http://host\n"):
            with self.subTest(url=url), self.assertRaises(BackendError):
                HomebridgeBackend(url, token="secret")
        for kwargs in ({}, {"username": "admin"}, {"token": "secret\nheader"},
                       {"token": "secret", "timeout": float("nan")},
                       {"token": "secret", "username": "admin", "password": "secret"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(BackendError):
                HomebridgeBackend("http://homebridge.local", **kwargs)

    def test_login_payload_and_light_put_use_verified_ui_schema(self):
        backend, transport = self.backend((201, {"access_token": "jwt-token", "expires_in": 3600}),
                                          (200, light(False)), (200, light(True)), (200, light(True)),
                                          username="admin", password="secret-password")
        self.assertEqual(backend.set_state("light", {"unique_id": LIGHT}, "on"), DeviceState("on", True))
        login, before, write, after = transport.calls
        self.assertEqual(login["url"], "http://homebridge.local:8581/api/auth/login")
        self.assertEqual(login["body"], {"username": "admin", "password": "secret-password"})
        self.assertNotIn("Authorization", login["headers"])
        self.assertEqual([c["method"] for c in transport.calls], ["POST", "GET", "PUT", "GET"])
        self.assertEqual(write["url"], f"http://homebridge.local:8581/api/accessories/{LIGHT}")
        self.assertEqual(write["body"], {"characteristicType": "On", "value": True})
        self.assertEqual(before["headers"]["Authorization"], "Bearer jwt-token")
        self.assertEqual(after["timeout"], 5)

    def test_garage_write_targets_target_state_but_returns_observed_current_state(self):
        backend, transport = self.backend((200, garage()), (200, garage(current=2, target=0)),
                                          (200, garage(current=2, target=0)))
        self.assertEqual(backend.set_state("garage", {"unique_id": GARAGE}, "open"), DeviceState("opening", True))
        self.assertEqual(transport.calls[1]["body"], {"characteristicType": "TargetDoorState", "value": 0})

    def test_all_current_door_states_and_strict_light_booleans(self):
        for value, expected in [(0, "open"), (1, "closed"), (2, "opening"), (3, "closing"),
                                (4, "stopped"), (5, "unknown"), (True, "unknown"), ("0", "unknown")]:
            backend, _ = self.backend((200, garage(current=value, target=0)))
            self.assertEqual(backend.get_state("garage", {"unique_id": GARAGE}), DeviceState(expected))
        backend, _ = self.backend((200, light("false")))
        self.assertEqual(backend.get_state("light", {"unique_id": LIGHT}), DeviceState("unknown"))

    def test_discovery_parses_multiple_services_and_characteristics_without_exposing_secrets(self):
        first = light()
        first["instance"] = {"pin": "secret-pin", "username": "secret-network-id"}
        first["accessoryInformation"] = {"Serial Number": "private-serial"}
        offline = garage()
        offline["serviceCharacteristics"][0]["canWrite"] = False
        backend, transport = self.backend((200, [first, offline, {"type": "TemperatureSensor"}]))
        self.assertEqual(backend.discover(), [
            {"kind": "light", "name": "Kitchen lights", "available": True, "target": {"unique_id": LIGHT}},
            {"kind": "garage", "name": "Garage", "available": False, "target": {"unique_id": GARAGE}}])
        self.assertEqual(len(transport.calls), 1)

    def test_binding_mismatch_or_unwritable_characteristic_blocks_write(self):
        invalid_id = light()
        invalid_id["uniqueId"] = GARAGE
        invalid_kind = garage()
        invalid_kind["uniqueId"] = LIGHT
        unwritable = light()
        unwritable["serviceCharacteristics"][1]["canWrite"] = False
        duplicate = light()
        duplicate["serviceCharacteristics"].append(characteristic("On", False))
        for response in (invalid_id, invalid_kind, unwritable, duplicate):
            backend, transport = self.backend((200, response))
            with self.assertRaises(BackendError):
                backend.set_state("light", {"unique_id": LIGHT}, "off")
            self.assertEqual([c["method"] for c in transport.calls], ["GET"])

    def test_readback_failure_preserves_acknowledgement_without_claiming_physical_success(self):
        for response in ((500, {"message": "secret-password"}), URLError("secret-token"), (200, None)):
            backend, transport = self.backend((200, garage()), (200, None), response)
            self.assertEqual(backend.set_state("garage", {"unique_id": GARAGE}, "open"), DeviceState("unknown", True))
            self.assertEqual(sum(c["method"] == "PUT" for c in transport.calls), 1)

    def test_failed_write_never_retries_and_redacts_unknown_outcome(self):
        for response in ((401, {"secret": "private-token"}), (400, {"message": "private-token"}),
                         (500, "private-token"), TimeoutError("private-token")):
            backend, transport = self.backend((200, light()), response)
            with self.assertRaisesRegex(BackendError, "unknown.*not retried") as caught:
                backend.set_state("light", {"unique_id": LIGHT}, "off")
            self.assertNotIn("private-token", str(caught.exception))
            self.assertEqual([c["method"] for c in transport.calls], ["GET", "PUT"])

    def test_expired_login_token_can_retry_read_once_before_any_write(self):
        backend, transport = self.backend((200, {"access_token": "first"}), (401, {}),
                                          (200, {"access_token": "renewed"}), (200, light()),
                                          username="admin", password="secret")
        self.assertEqual(backend.get_state("light", {"unique_id": LIGHT}), DeviceState("off"))
        self.assertEqual([c["method"] for c in transport.calls], ["POST", "GET", "POST", "GET"])
        self.assertEqual(transport.calls[-1]["headers"]["Authorization"], "Bearer renewed")

    def test_auth_and_reachability_errors_are_redacted(self):
        for response in ((401, {"password": "do-not-print"}), (403, "do-not-print"),
                         (302, {"location": "http://other/?do-not-print"}),
                         URLError("http://user:do-not-print@host"), (200, {"secret": "do-not-print"})):
            backend, _ = self.backend(response)
            with self.assertRaises(BackendError) as caught:
                backend.discover()
            self.assertNotIn("do-not-print", str(caught.exception))
        backend, _ = self.backend((412, "do-not-print"), username="admin", password="do-not-print")
        with self.assertRaisesRegex(BackendError, "two-factor"):
            backend.discover()

    def test_http_transport_serialization_timeout_and_cross_host_redirect_refusal(self):
        transport = _HttpTransport()
        response = MagicMock()
        response.status = 200
        response.read.return_value = b'{"ok":true}'
        response.__enter__.return_value = response
        with patch.object(transport._opener, "open", return_value=response) as opened:
            status, data = transport.request("PUT", "https://homebridge.local/api/accessories/id",
                headers={"Authorization": "Bearer private-token", "Content-Type": "application/json"},
                body={"characteristicType": "On", "value": False}, timeout=2.5)
        self.assertEqual((status, data), (200, {"ok": True}))
        request = opened.call_args.args[0]
        self.assertEqual(json.loads(request.data), {"characteristicType": "On", "value": False})
        self.assertEqual(opened.call_args.kwargs["timeout"], 2.5)
        handler = next(h for h in transport._opener.handlers if isinstance(h, _NoRedirect))
        self.assertIsNone(handler.redirect_request(request, None, 302, "Found", {}, "https://other.example"))
        secret_body = io.BytesIO(b'{"secret":"private-token"}')
        with patch.object(transport._opener, "open", side_effect=HTTPError(request.full_url, 401, "Unauthorized", {}, secret_body)):
            self.assertEqual(transport.request("GET", request.full_url, headers={}, body=None, timeout=2), (401, None))
        self.assertTrue(secret_body.closed)


if __name__ == "__main__":
    unittest.main()
