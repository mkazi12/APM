import unittest
from apm.tools import SimulatedHome

def call(name, **args):
    return {"function": {"name": name, "arguments": args}}

class ExecutionTests(unittest.TestCase):
    def test_explicit_set_is_idempotent(self):
        home = SimulatedHome()
        calls = [call("set_lights", device="kitchen_lights", on=False)]
        self.assertEqual(home.execute(calls), home.execute(calls))
        self.assertEqual(home.devices["kitchen_lights"]["state"], "off")

    def test_invalid_batch_does_not_partially_execute(self):
        home = SimulatedHome()
        with self.assertRaises(Exception):
            home.execute([call("set_garage", target="closed"), call("shell", command="anything")])
        self.assertEqual(home.devices["garage"]["state"], "open")

    def test_invalid_arguments_are_rejected(self):
        for args in [{"device": "bedroom", "on": False}, {"device": "kitchen_lights", "on": "false"},
                     {"device": "kitchen_lights", "on": False, "extra": 1}]:
            with self.subTest(args=args), self.assertRaises(Exception):
                SimulatedHome().execute([call("set_lights", **args)])

    def test_read_does_not_mutate(self):
        self.assertEqual(SimulatedHome().execute([call("get_device_state", device="garage")])[0]["state"], "open")

if __name__ == "__main__":
    unittest.main()
