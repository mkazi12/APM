from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jsonschema import ValidationError, validate

from apm.backends.base import BackendError, DeviceState
from apm.home import DEFAULT_HOME, HomeController, initialize_home, load_home


def call(name, **arguments):
    return {"function": {"name": name, "arguments": arguments}}


def real_home():
    return {
        "version": 1,
        "integrations": {
            "matter": {"type": "matter", "url": "ws://matter.local:5580/ws"},
            "bridge": {"type": "homebridge", "url": "http://bridge.local:8581",
                       "token_env": "APM_HOME_TEST_TOKEN"},
        },
        "devices": {
            "desk": {"name": "Desk lamp", "room": "Office", "aliases": ["reading light"],
                     "kind": "light", "integration": "matter", "target": {"node_id": 11, "endpoint": 1}},
            "porch": {"name": "Porch light", "kind": "light", "integration": "matter",
                      "target": {"node_id": 11, "endpoint": 2}},
            "east_garage": {"name": "East garage", "kind": "garage", "integration": "bridge",
                            "target": {"unique_id": "a" * 64}},
            "west_garage": {"name": "West garage", "kind": "garage", "integration": "bridge",
                            "target": {"unique_id": "b" * 64}},
        },
    }


class FakeAdapter:
    def __init__(self, outcomes=()):
        self.calls = []
        self.outcomes = list(outcomes)
        self.closed = False

    def get_state(self, kind, target):
        self.calls.append(("get", kind, deepcopy(target)))
        return DeviceState("off" if kind == "light" else "closed")

    def set_state(self, kind, target, state):
        self.calls.append(("set", kind, deepcopy(target), state))
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return DeviceState(state, accepted=True)

    def discover(self):
        self.calls.append(("discover",))
        return [{"kind": "light", "name": "Discovered light", "available": True,
                 "target": {"node_id": 12, "endpoint": 1}}]

    def close(self):
        self.closed = True


class HomeRegistryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "home.json"
        # These tests must never instantiate a real connection-capable adapter,
        # even when a regression accidentally evicts an injected fake.
        for name in ("matter.MatterBackend", "homebridge.HomebridgeBackend"):
            guard = patch(f"apm.backends.{name}.__init__",
                          side_effect=AssertionError("Unexpected real adapter construction"))
            guard.start()
            self.addCleanup(guard.stop)

    def write(self, document):
        self.path.write_text(json.dumps(document), encoding="utf-8")
        return self.path

    def load_real(self, *, matter=None, bridge=None):
        self.write(real_home())
        matter = matter or FakeAdapter()
        bridge = bridge or FakeAdapter()
        return HomeController(self.path, adapters={"matter": matter, "bridge": bridge}), matter, bridge

    def test_offline_load_and_context_need_no_credentials_or_adapters(self):
        self.write(real_home())
        with patch.dict("os.environ", {}, clear=True):
            home = load_home(self.path)
            self.assertEqual(len(home.context()["devices"]), 4)
            integrations = {item["id"]: item for item in home.integrations()}
            self.assertFalse(integrations["bridge"]["configured"])
            self.assertTrue(integrations["matter"]["configured"])

    def test_missing_or_invalid_json_registry_has_clear_sanitized_error(self):
        with self.assertRaisesRegex(ValueError, "registry.*initialize"):
            load_home(self.path)
        self.path.write_text('{"SECRET-invalid-json":')
        with self.assertRaises(ValueError) as error:
            load_home(self.path)
        self.assertNotIn("SECRET", str(error.exception))

    def test_invalid_registry_is_rejected_before_any_adapter_construction(self):
        documents = []
        for mutate in (
            lambda d: d.update(version=2),
            lambda d: d["devices"]["desk"].update(integration="missing"),
            lambda d: d["devices"]["desk"].update(target={"node_id": True, "endpoint": 1}),
            lambda d: d["devices"]["desk"].update(kind="garage"),
            lambda d: d["integrations"]["bridge"].update(token="SECRET-direct-token"),
            lambda d: d["integrations"]["bridge"].pop("token_env"),
            lambda d: d["integrations"]["matter"].update(url="ws://user:SECRET@matter.local/ws"),
            lambda d: d["integrations"]["matter"].update(timeout=float("nan")),
            lambda d: d["integrations"]["matter"].update(timeout=float("inf")),
        ):
            document = real_home()
            mutate(document)
            documents.append(document)
        for index, document in enumerate(documents):
            with self.subTest(index=index):
                self.write(document)
                with self.assertRaises((ValueError, BackendError)) as error:
                    load_home(self.path)
                self.assertNotIn("SECRET", str(error.exception))

    def test_context_omits_server_details_credentials_and_protocol_targets(self):
        self.write(real_home())
        with patch.dict("os.environ", {"APM_HOME_TEST_TOKEN": "SECRET-test-token"}):
            context = load_home(self.path).context()
        encoded = json.dumps(context)
        for hidden in ("SECRET", "APM_HOME_TEST_TOKEN", "matter.local", "bridge.local", "unique_id", "node_id"):
            self.assertNotIn(hidden, encoded)
        desk = next(item for item in context["devices"] if item["id"] == "desk")
        self.assertEqual(desk["room"], "Office")
        self.assertEqual(desk["aliases"], ["reading light"])
        self.assertNotIn("integration", desk)
        self.assertNotIn("target", desk)

    def test_dynamic_schemas_address_multiple_lights_and_garages(self):
        self.write(real_home())
        context = load_home(self.path).context()
        schemas = {item["function"]["name"]: item["function"]["parameters"] for item in context["tools"]}
        self.assertEqual(set(schemas["set_lights"]["properties"]["device"]["enum"]), {"desk", "porch"})
        self.assertEqual(set(schemas["set_garage"]["properties"]["device"]["enum"]),
                         {"east_garage", "west_garage"})
        validate({"device": "west_garage", "target": "closed"}, schemas["set_garage"])
        with self.assertRaises(ValidationError):
            validate({"target": "closed"}, schemas["set_garage"])
        empty = real_home()
        empty["devices"] = {}
        self.write(empty)
        context = load_home(self.path).context()
        self.assertEqual(context["devices"], [])
        self.assertEqual(context["tools"], [])

    def test_entire_batch_is_validated_before_first_read_or_write(self):
        home, matter, bridge = self.load_real()
        valid = call("set_lights", device="desk", on=False)
        for invalid in (call("shell", command="anything"), call("set_lights", device="porch", on="false"),
                        call("set_garage", target="closed"), call("get_device_state", device="missing"),
                        call("set_lights", device="east_garage", on=True)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                home.execute([valid, invalid])
        with self.assertRaises(ValueError):
            home.execute([valid] * 9)
        self.assertEqual(matter.calls, [])
        self.assertEqual(bridge.calls, [])

    def test_persisted_registry_keeps_injected_adapters_and_routes_exact_targets(self):
        home, matter, bridge = self.load_real()
        results = home.execute([call("set_lights", device="porch", on=False),
                                call("set_garage", device="west_garage", target="closed"),
                                call("get_device_state", device="desk")])
        self.assertFalse(matter.closed)
        self.assertFalse(bridge.closed)
        self.assertEqual(matter.calls, [("set", "light", {"node_id": 11, "endpoint": 2}, "off"),
                                       ("get", "light", {"node_id": 11, "endpoint": 1})])
        self.assertEqual(bridge.calls, [("set", "garage", {"unique_id": "b" * 64}, "closed")])
        self.assertTrue(all(result["ok"] and not result["simulated"] for result in results))
        self.assertEqual([result["state"] for result in results], ["off", "closed", "off"])

    def test_partial_device_failure_stops_later_calls_without_repeating_success(self):
        matter = FakeAdapter([DeviceState("off", accepted=True), BackendError("Device unavailable")])
        home, matter, bridge = self.load_real(matter=matter)
        result = home.execute([call("set_lights", device="desk", on=False),
                               call("set_lights", device="porch", on=False),
                               call("set_garage", device="east_garage", target="closed")])
        self.assertTrue(result[0]["ok"])
        self.assertFalse(result[1]["ok"])
        self.assertIn("unavailable", result[1]["error"])
        self.assertFalse(result[2]["ok"])
        self.assertIn("Skipped", result[2]["error"])
        self.assertEqual(len(matter.calls), 2)
        self.assertEqual(bridge.calls, [])

    def test_acknowledged_unknown_write_skips_following_actions(self):
        home, matter, bridge = self.load_real(matter=FakeAdapter([DeviceState("unknown", accepted=True)]))
        result = home.execute([call("set_lights", device="desk", on=False),
                               call("set_garage", device="west_garage", target="closed")])
        self.assertTrue(result[0]["accepted"])
        self.assertEqual(result[0]["state"], "unknown")
        self.assertEqual(result[0]["requested_state"], "off")
        self.assertFalse(result[1]["ok"])
        self.assertEqual(bridge.calls, [])
        self.assertEqual(len(matter.calls), 1)

    def test_failed_atomic_replacement_preserves_file_memory_and_adapter(self):
        home, matter, _ = self.load_real()
        before_file = self.path.read_bytes()
        before_memory = deepcopy(home._document)
        with patch("apm.home.os.replace", side_effect=OSError("Disk unavailable")):
            with self.assertRaises(OSError):
                home.put_integration("matter", {"type": "matter", "url": "ws://new-server.local/ws"})
        self.assertEqual(self.path.read_bytes(), before_file)
        self.assertEqual(home._document, before_memory)
        self.assertFalse(matter.closed)
        self.assertEqual(list(self.path.parent.glob(".home-*")), [])

    def test_failed_device_write_does_not_change_in_memory_metadata(self):
        home, _, _ = self.load_real()
        before = deepcopy(home._document)
        changed = {**before["devices"]["desk"], "name": "Changed name"}
        with patch("apm.home._write", side_effect=PermissionError("Read only")):
            with self.assertRaises(PermissionError):
                home.put_device("desk", changed)
        self.assertEqual(home._document, before)
        self.assertEqual(json.loads(self.path.read_text()), before)

    def test_app_written_metadata_is_visible_to_next_voice_context(self):
        self.write(real_home())
        voice = load_home(self.path)
        app = load_home(self.path)
        old_context = voice.context()
        changed = {**real_home()["devices"]["desk"], "name": "Study lamp", "aliases": ["task light"]}
        app.put_device("desk", changed)
        app.remove_device("porch")
        context = voice.context()
        self.assertEqual(next(d for d in context["devices"] if d["id"] == "desk")["name"], "Study lamp")
        lights = next(t for t in context["tools"] if t["function"]["name"] == "set_lights")
        self.assertEqual(lights["function"]["parameters"]["properties"]["device"]["enum"], ["desk"])
        self.assertEqual(next(d for d in old_context["devices"] if d["id"] == "desk")["name"], "Desk lamp")

    def test_external_integration_change_closes_previous_adapter(self):
        home, matter, bridge = self.load_real()
        document = real_home()
        document["integrations"]["matter"]["url"] = "ws://replacement.local/ws"
        self.write(document)
        home.context()
        self.assertTrue(matter.closed)
        self.assertFalse(bridge.closed)

    def test_stale_model_revision_cannot_control_a_rebound_device(self):
        home, matter, bridge = self.load_real()
        original = home.context()["revision"]
        document = real_home()
        document["devices"]["desk"]["target"] = {"node_id": 99, "endpoint": 1}
        self.write(document)
        calls = [call("set_lights", device="desk", on=True)]
        with self.assertRaisesRegex(ValueError, "registry changed"):
            home.execute(calls, expected_revision=original)
        self.assertEqual(matter.calls, [])
        self.assertEqual(bridge.calls, [])
        current = home.context()["revision"]
        self.assertNotEqual(current, original)
        self.assertTrue(home.execute(calls, expected_revision=current)[0]["ok"])
        self.assertEqual(matter.calls, [("set", "light", {"node_id": 99, "endpoint": 1}, "on")])

    def test_snapshot_reads_other_devices_after_one_is_unavailable(self):
        home, matter, bridge = self.load_real()
        with patch.object(matter, "get_state", side_effect=[BackendError("Offline"), DeviceState("on")]) as reads:
            result = home.snapshot()
        by_device = {row["device"]: row for row in result}
        self.assertEqual(len(by_device), 4)
        self.assertFalse(by_device["desk"]["ok"])
        self.assertTrue(by_device["porch"]["ok"])
        self.assertEqual(by_device["porch"]["state"], "on")
        self.assertTrue(by_device["east_garage"]["ok"])
        self.assertTrue(by_device["west_garage"]["ok"])
        self.assertEqual(reads.call_count, 2)
        self.assertEqual(len(bridge.calls), 2)

    def test_missing_credentials_fail_at_operation_without_network(self):
        self.write(real_home())
        home = load_home(self.path)
        with patch.dict("os.environ", {}, clear=True):
            result = home.execute([call("set_garage", device="east_garage", target="closed")])
        self.assertFalse(result[0]["ok"])
        self.assertIn("credential environment variable", result[0]["error"])

    def test_duplicate_physical_target_is_rejected_including_integration_alias(self):
        for separate_integration in (False, True):
            with self.subTest(separate_integration=separate_integration):
                document = real_home()
                duplicate = deepcopy(document["devices"]["desk"])
                if separate_integration:
                    document["integrations"]["matter_alias"] = deepcopy(document["integrations"]["matter"])
                    duplicate["integration"] = "matter_alias"
                document["devices"]["duplicate"] = duplicate
                self.write(document)
                with self.assertRaisesRegex(ValueError, "already registered|Duplicate|duplicate"):
                    load_home(self.path)

    def test_initialization_is_exclusive_and_default_home_is_simulated(self):
        initialize_home(self.path)
        self.assertEqual(json.loads(self.path.read_text()), DEFAULT_HOME)
        with self.assertRaises(FileExistsError):
            initialize_home(self.path)
        home = load_home(self.path)
        result = home.execute([call("set_lights", device="kitchen_lights", on=False)])
        self.assertEqual(result[0]["state"], "off")
        self.assertTrue(result[0]["simulated"])


if __name__ == "__main__":
    unittest.main()
