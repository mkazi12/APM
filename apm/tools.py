"""An explicit execution boundary. No generated Python is executed."""
from copy import deepcopy

DEVICES = {"kitchen_lights": {"kind": "light", "state": "on"},
           "garage": {"kind": "garage", "state": "open"}}

def tool(name, description, properties):
    return {"type": "function", "function": {"name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": list(properties), "additionalProperties": False}}}

TOOLS = [
    tool("set_lights", "Set the kitchen lights explicitly on or off.", {
        "device": {"type": "string", "enum": ["kitchen_lights"]},
        "on": {"type": "boolean"}}),
    tool("set_garage", "Set the garage door target position.", {
        "target": {"type": "string", "enum": ["open", "closed"]}}),
    tool("get_device_state", "Read the current device state.", {
        "device": {"type": "string", "enum": list(DEVICES)}}),
]

class SimulatedHome:
    def __init__(self):
        self.devices = deepcopy(DEVICES)

    def context(self):
        return {"tools": deepcopy(TOOLS), "devices": [
            {"id": key, "name": key.replace("_", " "), "kind": value["kind"], "simulated": True}
            for key, value in self.devices.items()]}

    def snapshot(self):
        return [{"device": key, "state": value["state"], "simulated": True}
                for key, value in self.devices.items()]

    def close(self):
        pass

    def execute(self, calls, expected_revision=None):
        from jsonschema import validate
        schemas = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
        # Validate the ENTIRE batch before any state changes.
        if not isinstance(calls, list) or len(calls) > 8:
            raise ValueError("Expected at most eight tool calls")
        checked = []
        for call in calls:
            fn = call["function"]
            name, args = fn["name"], fn["arguments"]
            if name not in schemas:
                raise ValueError(f"Unknown tool: {name}")
            validate(args, schemas[name])
            checked.append((name, args))
        results = []
        for name, args in checked:
            if name == "set_lights":
                device = args["device"]
                self.devices[device]["state"] = "on" if args["on"] else "off"
            elif name == "set_garage":
                device = "garage"
                self.devices[device]["state"] = args["target"]
            else:
                device = args["device"]
            results.append({"tool": name, "device": device, "simulated": True,
                            "state": self.devices[device]["state"]})
        return results
