"""Model integration checks using only fake inference and device transports."""
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from apm.backends.base import DeviceState
from apm.cli import describe_results, run_request
from apm.home import HomeController
from apm.model import Gemma
from apm.ollama_backend import OllamaBackend


CALL = {"function": {"name": "set_lights", "arguments": {"device": "hall_lamp", "on": False}}}


class FakeOllamaTransport:
    def __init__(self, response=None, before_reply=None):
        self.response = response or {"content": "Hello."}
        self.before_reply = before_reply
        self.sent = []

    def stream(self, path, payload):
        self.sent.append((path, deepcopy(payload)))
        if self.before_reply:
            self.before_reply()
        yield {"message": deepcopy(self.response), "done": True, "done_reason": "stop"}


class FakeDevices:
    def __init__(self):
        self.writes = []

    def set_state(self, kind, target, state):
        self.writes.append((kind, deepcopy(target), state))
        return DeviceState(state, accepted=True)

    def close(self):
        pass


class HomeContextTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "home.json"
        self.document = {
            "version": 1,
            "integrations": {"bridge": {"type": "homebridge", "url": "http://homebridge.local:8581",
                                         "token_env": "HOME_CONTEXT_TEST_TOKEN"}},
            "devices": {
                "hall_lamp": {"name": "Hall lamp", "room": "Entry", "aliases": ["welcome light"],
                              "kind": "light", "integration": "bridge", "target": {"unique_id": "a" * 64}},
                "bay_door": {"name": "Bay door", "room": "Garage", "aliases": ["car door"],
                             "kind": "garage", "integration": "bridge", "target": {"unique_id": "b" * 64}},
            },
        }
        self.path.write_text(json.dumps(self.document))
        self.devices = FakeDevices()
        self.home = HomeController(self.path, adapters={"bridge": self.devices})
        self.addCleanup(self.home.close)

    @staticmethod
    def catalog(system):
        return json.loads(system.split("Device catalog (JSON):\n", 1)[1])

    @classmethod
    def ollama_catalog(cls, payload):
        return cls.catalog(next(message["content"] for message in payload["messages"]
                                if message["role"] == "system"
                                and "Device catalog (JSON):\n" in message["content"]))

    def test_ollama_transmits_registered_context_and_copies_dynamic_tools(self):
        context = self.home.context()
        transport = FakeOllamaTransport()
        model = OllamaBackend(transport=transport)
        expected_tools = deepcopy(context["tools"])
        model.configure_home(context)
        context["tools"][0]["function"]["parameters"]["properties"]["device"]["enum"].append("unregistered")
        model.predict(text="What can you control in the entry?")
        payload = transport.sent[-1][1]
        self.assertEqual(payload["tools"], expected_tools)
        system = "\n".join(message["content"] for message in payload["messages"]
                           if message["role"] == "system")
        catalog = self.ollama_catalog(payload)
        self.assertEqual([item["id"] for item in catalog], ["hall_lamp", "bay_door"])
        self.assertEqual(catalog[0]["room"], "Entry")
        self.assertEqual(catalog[0]["aliases"], ["welcome light"])
        self.assertFalse(catalog[0]["simulated"])
        self.assertNotIn("All devices in this prototype are simulated", system)
        self.assertNotIn("kitchen_lights", system)
        for private_value in ("homebridge.local", "HOME_CONTEXT_TEST_TOKEN", "a" * 64):
            self.assertNotIn(private_value, system)
        self.assertTrue(all("target" not in item and "integration" not in item for item in catalog))

    def test_cli_passes_fresh_context_revision_and_commits_real_tool_result(self):
        transport = FakeOllamaTransport({"tool_calls": [CALL]})
        model = OllamaBackend(transport=transport)
        context = self.home.context()
        with patch.object(self.home, "execute", wraps=self.home.execute) as execute, redirect_stdout(io.StringIO()):
            reply = run_request(model, self.home, text="Switch off the welcome light", debug=True)
        self.assertEqual(execute.call_args.kwargs, {"expected_revision": context["revision"]})
        self.assertEqual(self.devices.writes, [("light", {"unique_id": "a" * 64}, "off")])
        self.assertEqual(reply, "Hall lamp: off.")
        self.assertEqual(transport.sent[-1][1]["tools"], context["tools"])
        committed = json.loads(model.turns[-1][-1]["content"])
        self.assertFalse(committed["simulated"])
        self.assertTrue(committed["accepted"])
        self.assertEqual(committed["state"], "off")

    def test_registry_metadata_changes_refresh_the_next_request_and_tool_enums(self):
        transport = FakeOllamaTransport()
        model = OllamaBackend(transport=transport)
        before = self.home.context()["revision"]
        with redirect_stdout(io.StringIO()):
            run_request(model, self.home, text="Hello", debug=True)
        updated = deepcopy(self.document)
        updated["devices"]["hall_lamp"].update(name="Desk lamp", room="Office", aliases=["work light"])
        del updated["devices"]["bay_door"]
        # Emulate a separate app process replacing the on-disk registry.
        self.path.write_text(json.dumps(updated))
        with redirect_stdout(io.StringIO()):
            run_request(model, self.home, text="Which light is in my office?", debug=True)
        old_payload, new_payload = transport.sent[0][1], transport.sent[1][1]
        self.assertEqual(self.ollama_catalog(old_payload)[0]["name"], "Hall lamp")
        self.assertEqual(self.ollama_catalog(new_payload)[0]["aliases"], ["work light"])
        self.assertEqual(self.ollama_catalog(new_payload)[0]["room"], "Office")
        self.assertEqual({tool["function"]["name"] for tool in new_payload["tools"]}, {"set_lights", "get_device_state"})
        for tool in new_payload["tools"]:
            self.assertEqual(tool["function"]["parameters"]["properties"]["device"]["enum"], ["hall_lamp"])
        self.assertNotEqual(before, self.home.context()["revision"])

    def test_registry_rebinding_during_inference_prevents_stale_action(self):
        def rebind():
            changed = deepcopy(self.document["devices"]["hall_lamp"])
            changed["target"]["unique_id"] = "c" * 64
            self.home.put_device("hall_lamp", changed)
        model = OllamaBackend(transport=FakeOllamaTransport({"tool_calls": [CALL]}, before_reply=rebind))
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "registry changed"):
            run_request(model, self.home, text="Turn off the hall lamp", debug=True)
        self.assertEqual(self.devices.writes, [])
        self.assertEqual(model.turns, [])

    def test_transformers_generation_and_parsing_receive_the_dynamic_context(self):
        class Inputs(dict):
            def to(self, device):
                return self

        processor = MagicMock()
        processor.apply_chat_template.return_value = Inputs(input_ids=np.array([[10, 11]]))
        processor.decode.return_value = "Hello"
        processor.parse_response.return_value = {"content": "Hello"}
        model = Gemma.__new__(Gemma)
        model.device, model.history, model.pending = "cpu", [], None
        model.processor = processor
        model.model = SimpleNamespace(generate=MagicMock(return_value=np.array([[10, 11, 12]])),
                                      generation_config=SimpleNamespace(eos_token_id=12), dtype="unused")
        context = self.home.context()
        expected = deepcopy(context["tools"])
        model.configure_home(context)
        context["tools"].clear()
        fake_torch = SimpleNamespace(is_floating_point=lambda value: False, inference_mode=nullcontext)
        with patch.dict("sys.modules", {"torch": fake_torch}):
            result, _ = model.predict(text="Hi")
        self.assertEqual(result, {"content": "Hello"})
        arguments = processor.apply_chat_template.call_args
        self.assertEqual(arguments.kwargs["tools"], expected)
        self.assertEqual(self.catalog(arguments.args[0][0]["content"])[0]["id"], "hall_lamp")
        self.assertNotIn("All devices in this prototype are simulated", arguments.args[0][0]["content"])
        self.assertEqual(processor.parse_response.call_args.kwargs["tools"], expected)

    def test_descriptions_distinguish_acknowledgement_observation_and_simulation(self):
        base = {"device": "bay_door", "label": "Bay door", "ok": True,
                "accepted": True, "requested_state": "closed", "simulated": False}
        self.assertEqual(describe_results([{**base, "state": "unknown"}]),
                         "Bay door: requested closed; observed unknown.")
        self.assertEqual(describe_results([{**base, "state": "closing"}]),
                         "Bay door: requested closed; observed closing.")
        self.assertEqual(describe_results([{**base, "state": "closed"}]), "Bay door: closed.")
        self.assertEqual(describe_results([{**base, "state": "closed", "simulated": True}]),
                         "Bay door: closed (simulated).")
        self.assertEqual(describe_results([{**base, "ok": False, "state": "unknown", "error": "Device unavailable"}]),
                         "Bay door: Device unavailable.")


if __name__ == "__main__":
    unittest.main()
