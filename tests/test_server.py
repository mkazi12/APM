import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jsonschema import ValidationError

from apm.backends.base import BackendError
from apm.server import _local_authority, create_app, main


HAS_SERVER_DEPS = all(importlib.util.find_spec(name) for name in ("fastapi", "httpx", "uvicorn"))
if HAS_SERVER_DEPS:
    from fastapi.testclient import TestClient


class FakeHome:
    def __init__(self):
        self.calls = []
        self.fail = None
        self.closed = False

    def call(self, name, *args):
        self.calls.append((name, *args))
        if self.fail is not None:
            raise self.fail

    def integrations(self):
        self.call("integrations")
        return [{"id": "bridge", "type": "homebridge", "configured": True}]

    def put_integration(self, identifier, config):
        self.call("put_integration", identifier, config)
        return {"id": identifier, "type": config["type"], "configured": True}

    def discover(self, identifier):
        self.call("discover", identifier)
        return [{"kind": "light", "name": "Kitchen", "available": True, "target": {"unique_id": "abc"}}]

    def devices(self):
        self.call("devices")
        return [{"id": "kitchen", "kind": "light", "name": "Kitchen", "integration": "bridge"}]

    def put_device(self, identifier, config):
        self.call("put_device", identifier, config)
        return {"id": identifier, **config}

    def remove_device(self, identifier):
        self.call("remove_device", identifier)

    def get_state(self, identifier):
        self.call("get_state", identifier)
        return {"device": identifier, "state": "off"}

    def command(self, identifier, state):
        self.call("command", identifier, state)
        return {"device": identifier, "state": state, "accepted": True}

    def context(self):
        self.call("context")
        return {"devices": [], "tools": []}

    def close(self):
        self.closed = True


@unittest.skipUnless(HAS_SERVER_DEPS, "Install the home extra and httpx to test the local API")
class ServerTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=False)
        self.environment.start()
        os.environ.pop("APM_API_TOKEN", None)
        self.addCleanup(self.environment.stop)
        self.home = FakeHome()
        self.client = TestClient(create_app(self.home))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_routes_forward_only_explicit_operations_and_redacted_views(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        self.assertEqual(self.client.get("/v1/integrations").json()[0]["id"], "bridge")
        config = {"type": "homebridge", "url": "http://bridge.local:8581", "token": "private-secret"}
        result = self.client.put("/v1/integrations/bridge", json=config)
        self.assertEqual(result.status_code, 200)
        self.assertNotIn("private-secret", result.text)
        self.assertIn(("put_integration", "bridge", config), self.home.calls)
        proposed = self.client.get("/v1/integrations/bridge/discover").json()
        self.assertEqual(proposed[0]["kind"], "light")
        self.assertFalse(any(call[0] == "put_device" for call in self.home.calls))
        self.assertEqual(self.client.get("/v1/devices").json()[0]["id"], "kitchen")
        device = {"kind": "light", "name": "Kitchen", "integration": "bridge", "target": {"unique_id": "abc"}}
        self.assertEqual(self.client.put("/v1/devices/kitchen", json=device).json(), {"id": "kitchen", **device})
        self.assertEqual(self.client.get("/v1/devices/kitchen/state").json()["state"], "off")
        result = self.client.post("/v1/devices/kitchen/commands", json={"state": "on"})
        self.assertEqual(result.json(), {"device": "kitchen", "state": "on", "accepted": True})
        self.assertEqual(self.home.calls[-1], ("command", "kitchen", "on"))
        context = self.client.get("/v1/context").json()
        self.assertEqual(context["devices"], [])
        self.assertFalse(context["music"]["configured"])
        self.assertIn("play_music", [item["function"]["name"] for item in context["tools"]])
        result = self.client.delete("/v1/devices/kitchen")
        self.assertEqual((result.status_code, result.content), (204, b""))
        self.assertEqual(self.home.calls[-1], ("remove_device", "kitchen"))
        self.assertEqual(self.client.post("/v1/chat", json={}).status_code, 404)

    def test_token_checks_controller_and_health_routes_before_controller_call(self):
        client = TestClient(create_app(self.home, token="test-secret"))
        for path in ("/health", "/v1/devices"):
            for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic test-secret"},
                            {"Authorization": "Bearer test-secret "}):
                with self.subTest(path=path, headers=headers):
                    response = client.get(path, headers=headers)
                    self.assertEqual(response.status_code, 401)
                    self.assertEqual(response.headers["www-authenticate"], "Bearer")
                    self.assertNotIn("test-secret", response.text)
        self.assertEqual(self.home.calls, [])
        self.assertEqual(client.get("/v1/devices", headers={"Authorization": "Bearer test-secret"}).status_code, 200)

    def test_public_docs_support_bearer_authorization_without_exposing_secrets(self):
        client = TestClient(create_app(self.home, token="private-api-token"))
        for path in ("/docs", "/redoc", "/openapi.json"):
            result = client.get(path, headers={"Origin": "http://testserver"})
            self.assertEqual(result.status_code, 200)
            self.assertNotIn("private-api-token", result.text)
        schema = client.get("/openapi.json").json()
        self.assertEqual(schema["components"]["securitySchemes"]["HTTPBearer"], {"type": "http", "scheme": "bearer"})
        self.assertEqual(schema["paths"]["/v1/devices"]["get"]["security"], [{"HTTPBearer": []}])
        self.assertEqual(client.get("/docs", headers={"Host": "evil.example"}).status_code, 400)
        self.assertEqual(client.get("/openapi.json", headers={"Origin": "https://evil.example"}).status_code, 403)
        self.assertEqual(client.get("/v1/devices").status_code, 401)
        self.assertEqual(client.get("/v1/devices", headers={"Authorization": "Bearer private-api-token",
                                                          "Origin": "http://testserver"}).status_code, 200)

    def test_environment_token_and_explicit_override(self):
        with patch.dict(os.environ, {"APM_API_TOKEN": "environment-secret"}):
            client = TestClient(create_app(self.home))
            self.assertEqual(client.get("/health").status_code, 401)
            self.assertEqual(client.get("/health", headers={"Authorization": "Bearer environment-secret"}).status_code, 200)
            override = TestClient(create_app(self.home, token="explicit-secret"))
            self.assertEqual(override.get("/health", headers={"Authorization": "Bearer environment-secret"}).status_code, 401)
            self.assertEqual(override.get("/health", headers={"Authorization": "Bearer explicit-secret"}).status_code, 200)
        for token in ("secret\n", "two tokens", "sëcret"):
            with self.subTest(token=token), self.assertRaises(ValueError):
                create_app(self.home, token=token)

    def test_untrusted_host_and_origin_are_rejected_without_cors(self):
        for host in ("evil.example", "127.0.0.1.evil.example", "localhost@evil.example", "localhost:99999", "localhost/path"):
            with self.subTest(host=host):
                self.assertEqual(self.client.get("/v1/devices", headers={"Host": host}).status_code, 400)
        for origin in ("https://evil.example", "null", "http://localhost.evil.example", "http://user:secret@localhost",
                       "http://localhost/path", "http://[", "http://127.0.0.1:8765", "http://localhost:8765"):
            with self.subTest(origin=origin):
                result = self.client.post("/v1/devices/kitchen/commands", json={"state": "on"}, headers={"Origin": origin})
                self.assertEqual(result.status_code, 403)
                self.assertNotIn("access-control-allow-origin", result.headers)
        self.assertEqual(self.home.calls, [])
        self.assertEqual(self.client.get("/health", headers={"Origin": "http://testserver"}).status_code, 200)
        self.assertEqual(self.client.get("/health").headers["cache-control"], "no-store")

    def test_browser_origin_matches_serving_scheme_hostname_and_effective_port(self):
        cases = (
            ("http://127.0.0.1:8765", "http://127.0.0.1:8765", 200),
            ("http://127.0.0.1:8765", "http://127.0.0.1:8766", 403),
            ("http://127.0.0.1:8765", "http://localhost:8765", 403),
            ("http://127.0.0.1:8765", "https://127.0.0.1:8765", 403),
            ("http://localhost", "http://LOCALHOST:80", 200),
            ("http://localhost:80", "http://localhost", 200),
            ("https://localhost", "https://localhost:443", 200),
            ("https://localhost:443", "https://localhost", 200),
            ("https://localhost", "https://localhost:80", 403),
            ("http://[::1]:8765", "http://[::1]:8765", 200),
            ("http://[::1]:8765", "http://localhost:8765", 403),
        )
        for serving, origin, status in cases:
            with self.subTest(serving=serving, origin=origin):
                client = TestClient(create_app(self.home, token=""), base_url=serving)
                response = client.get("/health", headers={"Origin": origin})
                self.assertEqual(response.status_code, status)
                self.assertNotIn("access-control-allow-origin", response.headers)
                self.assertEqual(client.get("/health").status_code, 200)

    def test_duplicate_security_headers_are_rejected(self):
        self.assertEqual(self.client.get("/health", headers=[("Host", "localhost"), ("Host", "evil.example")]).status_code, 400)
        self.assertEqual(self.client.get("/health", headers=[("Origin", "http://localhost"), ("Origin", "https://evil.example")]).status_code, 403)
        client = TestClient(create_app(self.home, token="secret"))
        self.assertEqual(client.get("/health", headers=[("Authorization", "Bearer secret"), ("Authorization", "Bearer wrong")]).status_code, 401)

    def test_invalid_bodies_never_reach_controller_or_echo_inputs(self):
        for body in ({}, {"state": 1}, {"state": None}, {"state": "on", "token": "private-secret"}, ["private-secret"]):
            with self.subTest(body=body):
                result = self.client.post("/v1/devices/kitchen/commands", json=body)
                self.assertEqual(result.status_code, 422)
                self.assertNotIn("private-secret", result.text)
        for path in ("/v1/integrations/bridge", "/v1/devices/kitchen"):
            self.assertEqual(self.client.put(path, json=["private-secret"]).status_code, 422)
            result = self.client.put(path, content='{"token":"private-secret",', headers={"Content-Type": "application/json"})
            self.assertEqual(result.status_code, 422)
            self.assertNotIn("private-secret", result.text)
        self.assertEqual(self.home.calls, [])

    def test_controller_failures_have_sanitized_json_statuses(self):
        for exception, code in ((KeyError("private-secret"), 404), (ValueError("private-secret"), 400),
                                (ValidationError("private-secret"), 400), (BackendError("private-secret"), 502),
                                (RuntimeError("private-secret"), 500)):
            with self.subTest(exception=type(exception).__name__):
                self.home.fail = exception
                result = self.client.get("/v1/devices")
                self.assertEqual(result.status_code, code)
                self.assertNotIn("private-secret", result.text)
                self.assertIsInstance(result.json()["detail"], str)

    def test_lifespan_closes_controller(self):
        home = FakeHome()
        with TestClient(create_app(home)) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertFalse(home.closed)
        self.assertTrue(home.closed)

    def test_real_registry_persists_registration_but_not_observed_state_or_credentials(self):
        from apm.home import HomeController, initialize_home

        workspace = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix=".test-server-", dir=workspace) as directory:
            path = Path(directory) / "home.json"
            initialize_home(path)
            with TestClient(create_app(HomeController(path))) as client:
                device = {"name": "Desk lamp", "kind": "light", "integration": "demo", "target": {"id": "desk"}}
                result = client.put("/v1/devices/desk", json=device)
                self.assertEqual(result.status_code, 200)
                self.assertTrue(result.json()["simulated"])
                self.assertEqual(json.loads(path.read_text())["devices"]["desk"], device)
                before_command = path.read_text()
                result = client.post("/v1/devices/desk/commands", json={"state": "off"})
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json()["state"], "off")
                self.assertTrue(result.json()["accepted"])
                self.assertEqual(client.get("/v1/devices/desk/state").json()["state"], "off")
                self.assertEqual(path.read_text(), before_command)
                self.assertEqual(client.post("/v1/devices/desk/commands", json={"state": "open"}).status_code, 400)
                self.assertEqual(client.get("/v1/devices/missing/state").status_code, 404)

                with patch.dict(os.environ, {"TEST_HOME_TOKEN": "private-credential"}):
                    integration = {"type": "homebridge", "url": "http://bridge.local:8581", "token_env": "TEST_HOME_TOKEN"}
                    result = client.put("/v1/integrations/bridge", json=integration)
                    self.assertEqual(result.status_code, 200)
                    self.assertTrue(result.json()["configured"])
                    for response in (result, client.get("/v1/integrations"), client.get("/v1/context")):
                        self.assertNotIn("private-credential", response.text)
                        self.assertNotIn("TEST_HOME_TOKEN", response.text)
                    self.assertNotIn("private-credential", path.read_text())
                    invalid = {"type": "homebridge", "url": "http://bridge.local:8581", "token": "private-credential"}
                    response = client.put("/v1/integrations/invalid", json=invalid)
                    self.assertEqual(response.status_code, 400)
                    self.assertNotIn("private-credential", response.text)
                    self.assertNotIn("invalid", json.loads(path.read_text())["integrations"])

            with TestClient(create_app(HomeController(path))) as client:
                self.assertIn("desk", [item["id"] for item in client.get("/v1/devices").json()])
                self.assertEqual(client.delete("/v1/devices/desk").status_code, 204)
                self.assertNotIn("desk", json.loads(path.read_text())["devices"])

    def test_cli_only_binds_loopback_and_disables_proxy_headers(self):
        with patch("apm.home.load_home", return_value=self.home) as load, patch("uvicorn.run") as run, \
                patch("apm.tasks.TaskService"), patch("apm.scheduler.Scheduler"):
            self.assertEqual(main(["--registry", "config/test-home.json", "--port", "9000"]), 0)
        load.assert_called_once_with(Path("config/test-home.json"))
        self.assertEqual(run.call_args.kwargs, {"host": "127.0.0.1", "port": 9000, "proxy_headers": False, "access_log": False})

    def test_cli_init_is_explicit_and_does_not_overwrite_registry(self):
        with patch("apm.home.initialize_home") as initialize, patch("apm.home.load_home", return_value=self.home), patch("uvicorn.run"), \
                patch("apm.tasks.TaskService"), patch("apm.scheduler.Scheduler"):
            self.assertEqual(main(["--init", "--registry", "config/test-home.json"]), 0)
        initialize.assert_called_once_with(Path("config/test-home.json"))
        with patch("apm.home.initialize_home", side_effect=FileExistsError("private-secret")), patch("uvicorn.run") as run, patch("sys.stderr"):
            with self.assertRaises(SystemExit) as error:
                main(["--init", "--registry", "config/test-home.json"])
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()


class ServerArgumentTests(unittest.TestCase):
    def test_cli_rejects_remote_bind_and_invalid_arguments(self):
        for args in (["--host", "0.0.0.0"], ["--init"], ["--port", "0"], ["--port", "65536"]):
            with self.subTest(args=args), patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                main(args)
            self.assertEqual(error.exception.code, 2)

    def test_local_authorities_reject_url_ambiguities(self):
        for host in ("localhost", "127.0.0.1:8765", "[::1]:8765", "testserver"):
            self.assertTrue(_local_authority(host), host)
        for host in ("", "localhost\n", "localhost\\@evil.example", "localhost:0", "http://localhost", "evil.example"):
            self.assertFalse(_local_authority(host), host)


if __name__ == "__main__":
    unittest.main()
