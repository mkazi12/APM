from collections import deque
import json
import unittest
from unittest.mock import patch

from apm.backends.base import BackendError, DeviceState
from apm.backends.matter import MatterBackend

TARGET = {"node_id": 7, "endpoint": 1}
HELLO = {"schema_version": 13, "min_supported_schema_version": 11, "sdk_version": "matter-server/test"}


def node(**changes):
    value = {"node_id": 7, "available": True,
             "attributes": {"0/40/5": "Desk lamp", "1/29/1": [29, 6], "1/6/0": False,
                            "2/29/1": [29, 257], "2/257/0": 1}}
    value.update(changes)
    return value


class FakeTransport:
    """In-memory protocol server, without network connections or actual devices."""
    def __init__(self, nodes=None):
        self.nodes = [node()] if nodes is None else nodes
        self.responses = deque([json.dumps(HELLO)])
        self.sent = []
        self.timeouts = []
        self.closed = False
        self.on_send = None
        self.on_recv = None
        self.read_value = False

    def settimeout(self, timeout):
        self.timeouts.append(timeout)

    def send(self, raw):
        message = json.loads(raw)
        self.sent.append(message)
        if self.on_send and self.on_send(message):
            return
        command = message["command"]
        if command in {"start_listening", "get_nodes"}:
            result = self.nodes
        elif command == "read_attribute":
            result = {message["args"]["attribute_path"]: self.read_value}
        elif command == "device_command":
            self.read_value = message["args"]["command_name"] == "On"
            result = None
        else:
            raise AssertionError(f"Unexpected command: {command}")
        self.respond(message, result)

    def respond(self, message, result):
        self.responses.append(json.dumps({"message_id": message["message_id"], "result": result}))

    def recv(self):
        if self.on_recv:
            self.on_recv()
        if not self.responses:
            raise TimeoutError("SECRET server address and credentials")
        return self.responses.popleft()

    def close(self):
        self.closed = True


class MatterBackendTests(unittest.TestCase):
    def backend(self, transport=None, **kwargs):
        return MatterBackend("ws://localhost:5580/ws", transport=transport or FakeTransport(), **kwargs)

    def test_constructor_and_validation_do_not_connect(self):
        calls = []
        backend = self.backend(lambda *args, **kwargs: calls.append((args, kwargs)))
        backend.validate_target("light", TARGET)
        MatterBackend.validate_target("light", TARGET)
        self.assertEqual(calls, [])
        for kind, target in (("garage", TARGET), ("light", {"node_id": True, "endpoint": 1}),
                             ("light", {"node_id": 0, "endpoint": 1}),
                             ("light", {"node_id": 7, "endpoint": -1}),
                             ("light", {**TARGET, "command": "Toggle"})):
            with self.subTest(kind=kind, target=target), self.assertRaises(BackendError):
                backend.validate_target(kind, target)
        self.assertEqual(calls, [])

    def test_discovery_lists_only_onoff_servers_and_keeps_offline_nodes(self):
        second = node(node_id=8, available=False, attributes={"3/57/5": "Bridge lamp", "3/29/1": [6]})
        fake = FakeTransport([node(), second])
        found = self.backend(fake).discover()
        self.assertEqual(found, [
            {"kind": "light", "name": "Desk lamp", "available": True, "target": TARGET},
            {"kind": "light", "name": "Bridge lamp", "available": False,
             "target": {"node_id": 8, "endpoint": 3}},
        ])
        self.assertEqual([m["command"] for m in fake.sent], ["start_listening", "get_nodes"])

    def test_descriptor_serverlist_overrides_stale_onoff_attribute(self):
        fake = FakeTransport([node(attributes={"1/29/1": [29], "1/6/0": True,
                                             "2/6/0": False, "3/29/2": [6]})])
        self.assertEqual([row["target"]["endpoint"] for row in self.backend(fake).discover()], [2])

    def test_live_read_ignores_cached_state_events_and_other_response_ids(self):
        fake = FakeTransport()
        fake.read_value = True
        def interleave(message):
            if message["command"] == "read_attribute":
                fake.responses.extend([json.dumps({"event": "attribute_updated", "data": [7, "1/6/0", False]}),
                                       json.dumps({"message_id": "some-other-request", "result": {"1/6/0": False}})])
        fake.on_send = interleave
        self.assertEqual(self.backend(fake).get_state("light", TARGET), DeviceState("on"))
        self.assertEqual(fake.sent[-1]["args"], {"node_id": 7, "attribute_path": "1/6/0"})

    def test_set_sends_exact_onoff_command_and_reports_observed_state(self):
        for desired, command in (("on", "On"), ("off", "Off")):
            with self.subTest(desired=desired):
                fake = FakeTransport()
                result = self.backend(fake).set_state("light", TARGET, desired)
                self.assertEqual(result, DeviceState(desired, accepted=True))
                write = next(m for m in fake.sent if m["command"] == "device_command")
                self.assertEqual(write["args"], {"node_id": 7, "endpoint_id": 1, "cluster_id": 6,
                                                 "command_name": command, "payload": {}})
                self.assertEqual(fake.sent[-1]["command"], "read_attribute")

    def test_acknowledgement_does_not_imply_requested_state(self):
        for read_response, expected in (({"1/6/0": False}, "off"), ({"1/6/0": None}, "unknown")):
            with self.subTest(read_response=read_response):
                fake = FakeTransport()
                def respond(message):
                    if message["command"] == "read_attribute":
                        fake.respond(message, read_response)
                        return True
                fake.on_send = respond
                self.assertEqual(self.backend(fake).set_state("light", TARGET, "on"),
                                 DeviceState(expected, accepted=True))

    def test_acknowledged_write_read_timeout_returns_unknown_without_retry(self):
        fake = FakeTransport()
        fake.on_send = lambda message: message["command"] == "read_attribute"
        result = self.backend(fake).set_state("light", TARGET, "on")
        self.assertEqual(result, DeviceState("unknown", accepted=True))
        self.assertEqual(sum(m["command"] == "device_command" for m in fake.sent), 1)
        self.assertTrue(fake.closed)

    def test_unacknowledged_write_is_not_retried_or_reported_accepted(self):
        fake = FakeTransport()
        fake.on_send = lambda message: message["command"] == "device_command"
        with self.assertRaises(BackendError) as error:
            self.backend(fake).set_state("light", TARGET, "on")
        self.assertNotIn("SECRET", str(error.exception))
        self.assertEqual(sum(m["command"] == "device_command" for m in fake.sent), 1)
        self.assertNotIn("read_attribute", [m["command"] for m in fake.sent])

    def test_unavailable_or_unsupported_nodes_do_not_receive_commands(self):
        for nodes in ([], [node(available=False)], [node(attributes={"1/29/1": [257]})]):
            with self.subTest(nodes=nodes):
                fake = FakeTransport(nodes)
                with self.assertRaises(BackendError):
                    self.backend(fake).set_state("light", TARGET, "on")
                self.assertNotIn("device_command", [m["command"] for m in fake.sent])

    def test_invalid_attributes_are_not_coerced_to_on(self):
        for value in (None, "false", 1, {}, []):
            with self.subTest(value=value):
                fake = FakeTransport()
                fake.read_value = value
                with self.assertRaisesRegex(BackendError, "attribute"):
                    self.backend(fake).get_state("light", TARGET)

    def test_error_zero_is_failure_and_server_details_are_not_exposed(self):
        fake = FakeTransport()
        def reject(message):
            if message["command"] == "device_command":
                fake.responses.append(json.dumps({"message_id": message["message_id"], "error_code": 0,
                                                   "details": "SECRET credential in upstream exception"}))
                return True
        fake.on_send = reject
        with self.assertRaises(BackendError) as error:
            self.backend(fake).set_state("light", TARGET, "on")
        self.assertIn("error 0", str(error.exception))
        self.assertNotIn("SECRET", str(error.exception))

    def test_events_cannot_extend_whole_operation_deadline(self):
        clock = [0.0]
        fake = FakeTransport()
        def events(message):
            if message["command"] == "get_nodes":
                fake.responses.extend([json.dumps({"event": "attribute_updated", "data": []})] * 50)
                return True
        fake.on_send = events
        fake.on_recv = lambda: clock.__setitem__(0, clock[0] + 0.2)
        with patch("apm.backends.matter.time.monotonic", side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(BackendError, "timed out"):
                self.backend(fake, timeout=1).discover()
        self.assertLess(clock[0], 1.5)
        self.assertTrue(all(0 < timeout <= 1 for timeout in fake.timeouts))
        self.assertTrue(fake.closed)

    def test_incompatible_hello_and_malformed_json_fail_closed(self):
        for hello in ({**HELLO, "min_supported_schema_version": 12}, {"result": []}, "not JSON"):
            with self.subTest(hello=hello):
                fake = FakeTransport()
                fake.responses = deque([hello if isinstance(hello, str) else json.dumps(hello)])
                with self.assertRaises(BackendError):
                    self.backend(fake).discover()
                self.assertEqual(fake.sent, [])
                self.assertTrue(fake.closed)

    def test_factory_is_lazy_and_close_is_idempotent(self):
        fake = FakeTransport()
        calls = []
        def factory(url, timeout):
            calls.append((url, timeout))
            return fake
        backend = self.backend(factory)
        self.assertEqual(calls, [])
        backend.get_state("light", TARGET)
        self.assertEqual(len(calls), 1)
        backend.close()
        backend.close()
        self.assertTrue(fake.closed)

    def test_invalid_configuration_does_not_expose_url(self):
        for url in ("https://localhost", "ws://user:SECRET@localhost/ws", "ws://localhost:bad/ws"):
            with self.subTest(url=url), self.assertRaises(BackendError) as error:
                MatterBackend(url)
            self.assertNotIn("SECRET", str(error.exception))
        for timeout in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(timeout=timeout), self.assertRaises(BackendError):
                self.backend(timeout=timeout)


if __name__ == "__main__":
    unittest.main()
