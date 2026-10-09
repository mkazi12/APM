from datetime import datetime, timedelta, timezone
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from apm.server import create_app, main


HAS_API_DEPS = all(importlib.util.find_spec(name) for name in ("fastapi", "httpx", "uvicorn"))
if HAS_API_DEPS:
    from fastapi.testclient import TestClient


class FakeScheduler:
    def __init__(self, tasks):
        self.tasks = tasks
        self.starts = 0
        self.stops = 0

    def start(self):
        self.tasks.clock()
        self.starts += 1

    def stop(self):
        # The scheduler must stop while the database is still usable.
        self.tasks.clock()
        self.stops += 1


@unittest.skipUnless(HAS_API_DEPS, "Install the home extra and httpx to test the task API")
class TaskServerTests(unittest.TestCase):
    def setUp(self):
        from apm.home import HomeController
        from apm.tasks import TaskService

        self.environment = patch.dict(os.environ, {}, clear=False)
        self.environment.start()
        os.environ.pop("APM_API_TOKEN", None)
        self.addCleanup(self.environment.stop)
        self.directory = tempfile.TemporaryDirectory(prefix=".test-task-api-", dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "assistant.sqlite3"
        self.now = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
        self.tasks = TaskService(self.path, timezone="UTC", now=lambda: self.now)
        self.home = HomeController()
        self.scheduler = FakeScheduler(self.tasks)
        self.client = TestClient(create_app(self.home, tasks=self.tasks, scheduler=self.scheduler))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def timer(self, name="Tea", duration=60):
        result = self.client.post("/v1/timers", json={"name": name, "duration_seconds": duration})
        self.assertEqual(result.status_code, 201, result.text)
        return result.json()

    def reminder(self, name="Laundry", **extra):
        body = {"name": name, "due_at": (self.now + timedelta(hours=1)).isoformat(), "timezone": "UTC", **extra}
        result = self.client.post("/v1/reminders", json=body)
        self.assertEqual(result.status_code, 201, result.text)
        return result.json()

    def test_clock_and_context_include_tasks_without_changing_devices(self):
        before = self.client.get("/v1/devices").json()
        result = self.client.get("/v1/clock", params={"timezone": "America/Los_Angeles"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["timezone"], "America/Los_Angeles")
        self.assertEqual(result.json()["date"], "2026-10-07")
        self.assertEqual(self.client.get("/v1/clock", params={"timezone": "Not/AZone"}).status_code, 400)
        context = self.client.get("/v1/context")
        self.assertEqual(context.status_code, 200, context.text)
        functions = [item["function"]["name"] for item in context.json()["tools"]]
        self.assertTrue(any("timer" in name for name in functions))
        self.assertTrue(any("reminder" in name for name in functions))
        self.assertIn("set_lights", functions)
        self.assertEqual(self.client.get("/v1/devices").json(), before)
        self.assertEqual(self.scheduler.starts, 1)

    def test_timer_create_read_filter_pause_extend_resume_and_cancel(self):
        timer = self.timer()
        task_id = timer["id"]
        path = f"/v1/timers/{task_id}"
        self.assertEqual(timer["kind"], "timer")
        self.assertEqual(timer["status"], "scheduled")
        self.assertEqual(self.client.get(path).json()["name"], "Tea")
        self.assertEqual([item["id"] for item in self.client.get("/v1/timers").json()], [task_id])
        result = self.client.post(path + "/actions", json={"action": "pause"})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["status"], "paused")
        result = self.client.post(path + "/actions", json={"action": "extend", "seconds": 30})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["remaining_seconds"], 90)
        self.assertEqual(self.client.get("/v1/timers", params={"status": "scheduled"}).json(), [])
        self.assertEqual(self.client.post(path + "/actions", json={"action": "resume"}).json()["status"], "scheduled")
        self.assertEqual(self.client.post(path + "/actions", json={"action": "cancel"}).json()["status"], "cancelled")
        self.assertEqual(self.client.get("/v1/timers", params={"status": "cancelled"}).json()[0]["id"], task_id)

    def test_reminder_create_repeat_read_snooze_complete_and_cancel(self):
        reminder = self.reminder(repeat="daily")
        task_id = reminder["id"]
        path = f"/v1/reminders/{task_id}"
        self.assertEqual(reminder["kind"], "reminder")
        self.assertEqual(reminder["repeat"], "daily")
        self.assertEqual(self.client.get(path).json()["name"], "Laundry")
        self.assertEqual(self.client.get("/v1/reminders").json()[0]["id"], task_id)
        result = self.client.post(path + "/actions", json={"action": "snooze", "seconds": 300})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["status"], "scheduled")
        one_time = self.reminder("Take package")
        result = self.client.post(f"/v1/reminders/{one_time['id']}/actions", json={"action": "complete"})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["status"], "completed")
        result = self.client.post(path + "/actions", json={"action": "cancel"})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["status"], "cancelled")

    def test_timer_snooze_and_complete_use_shared_task_operations(self):
        timer = self.timer()
        path = f"/v1/timers/{timer['id']}/actions"
        result = self.client.post(path, json={"action": "snooze", "seconds": 120})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["status"], "scheduled")
        self.assertEqual(result.json()["remaining_seconds"], 120)
        result = self.client.post(path, json={"action": "complete"})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["status"], "completed")

    def test_routes_reject_other_task_kind_before_mutation(self):
        timer = self.timer()
        reminder = self.reminder()
        for prefix, task, action in (("timers", reminder, "cancel"), ("reminders", timer, "cancel")):
            path = f"/v1/{prefix}/{task['id']}"
            self.assertEqual(self.client.get(path).status_code, 404)
            self.assertEqual(self.client.post(path + "/actions", json={"action": action}).status_code, 404)
            self.assertEqual(self.tasks.get_task(task["id"])["status"], "scheduled")

    def test_strict_request_shapes_reject_boolean_string_and_nonfinite_durations(self):
        for duration in (True, False, "60", 0, -1, 31_536_001, None):
            with self.subTest(duration=duration):
                result = self.client.post("/v1/timers", json={"name": "Tea", "duration_seconds": duration})
                self.assertEqual(result.status_code, 422)
        for literal in ("NaN", "Infinity", "-Infinity"):
            result = self.client.post("/v1/timers", content='{"name":"Tea","duration_seconds":' + literal + '}',
                                      headers={"Content-Type": "application/json"})
            self.assertEqual(result.status_code, 422)
        for body in ({"name": 8, "duration_seconds": 30}, {"name": "Tea", "duration_seconds": 30, "secret": "private-input"}):
            result = self.client.post("/v1/timers", json=body)
            self.assertEqual(result.status_code, 422)
            self.assertNotIn("private-input", result.text)
        self.assertEqual(self.client.get("/v1/timers").json(), [])

    def test_reminder_validation_and_action_argument_errors_do_not_change_tasks(self):
        for extra, status in (({"repeat": "monthly"}, 422), ({"due_at": 42}, 422),
                              ({"due_at": "2026-10-08T12:00:00"}, 400), ({"timezone": "Invalid/Zone"}, 400),
                              ({"due_at": "2026-01-01T12:00:00+00:00"}, 400), ({"extra": "private-input"}, 422)):
            body = {"name": "Laundry", "due_at": (self.now + timedelta(hours=1)).isoformat(), "timezone": "UTC", **extra}
            result = self.client.post("/v1/reminders", json=body)
            self.assertEqual(result.status_code, status, result.text)
            self.assertNotIn("private-input", result.text)
        timer = self.timer()
        reminder = self.reminder()
        for prefix, task, body, status in (
            ("timers", timer, {"action": "extend"}, 400),
            ("timers", timer, {"action": "pause", "seconds": 10}, 400),
            ("timers", timer, {"action": "extend", "seconds": True}, 422),
            ("timers", timer, {"action": "pause", "unknown": "private-input"}, 422),
            ("reminders", reminder, {"action": "snooze"}, 400),
            ("reminders", reminder, {"action": "complete", "seconds": 10}, 400),
            ("reminders", reminder, {"action": "snooze", "seconds": "10"}, 422),
            ("reminders", reminder, {"action": "pause"}, 422),
        ):
            result = self.client.post(f"/v1/{prefix}/{task['id']}/actions", json=body)
            self.assertEqual(result.status_code, status, result.text)
            self.assertNotIn("private-input", result.text)
            self.assertEqual(self.tasks.get_task(task["id"])["status"], "scheduled")

    def test_due_notifications_are_durable_filterable_acknowledgeable_and_hide_leases(self):
        timer = self.timer(duration=1)
        self.now += timedelta(seconds=2)
        events = self.tasks.process_due()
        self.assertEqual(len(events), 1)
        notification_id = events[0]["id"]
        self.tasks.claim_notification()
        result = self.client.get("/v1/notifications", params={"unread_only": "true"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()[0]["task_id"], timer["id"])
        self.assertNotIn("claim_token", result.json()[0])
        self.assertNotIn("leased_until", result.json()[0])
        invalid = self.client.post(f"/v1/notifications/{notification_id}/acknowledge", json={"extra": "private-input"})
        self.assertEqual(invalid.status_code, 422)
        self.assertNotIn("private-input", invalid.text)
        response = self.client.post(f"/v1/notifications/{notification_id}/acknowledge")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn("claim_token", response.text)
        self.assertEqual(self.client.get("/v1/notifications", params={"unread_only": "true"}).json(), [])
        self.assertEqual(len(self.client.get("/v1/notifications").json()), 1)

    def test_task_endpoints_share_existing_auth_and_origin_checks(self):
        client = TestClient(create_app(self.home, token="api-secret", tasks=self.tasks))
        for path in ("/v1/clock", "/v1/timers", "/v1/reminders", "/v1/notifications"):
            self.assertEqual(client.get(path).status_code, 401)
            self.assertEqual(client.get(path, headers={"Authorization": "Bearer api-secret"}).status_code, 200)
        body = {"name": "Tea", "duration_seconds": 60}
        self.assertEqual(client.post("/v1/timers", json=body).status_code, 401)
        self.assertEqual(client.post("/v1/timers", json=body,
                                    headers={"Authorization": "Bearer api-secret", "Origin": "https://evil.example"}).status_code, 403)
        self.assertEqual(self.tasks.list_tasks(), [])
        schema = client.get("/openapi.json").json()
        self.assertEqual(schema["paths"]["/v1/timers"]["post"]["security"], [{"HTTPBearer": []}])

    def test_database_survives_a_separate_service_connection(self):
        from apm.tasks import TaskService

        timer = self.timer()
        other = TaskService(self.path, timezone="UTC", now=lambda: self.now)
        try:
            self.assertEqual(other.get_task(timer["id"])["name"], "Tea")
        finally:
            other.close()

    def test_cli_passes_explicit_database_and_timezone(self):
        path = Path(self.directory.name) / "cli.sqlite3"
        with patch("apm.home.load_home", return_value=self.home), \
                patch("apm.tasks.TaskService", return_value=self.tasks) as task_type, \
                patch("apm.scheduler.Scheduler", return_value=self.scheduler) as scheduler_type, \
                patch("uvicorn.run") as run:
            self.assertEqual(main(["--database", str(path), "--timezone", "UTC"]), 0)
        task_type.assert_called_once_with(path, timezone="UTC")
        scheduler_type.assert_called_once_with(self.tasks)
        self.assertEqual(run.call_args.kwargs["host"], "127.0.0.1")
        self.assertFalse(path.exists())

    def test_lifespan_stops_scheduler_before_closing_database_and_home(self):
        from apm.home import HomeController
        from apm.tasks import TaskService

        tasks = TaskService(Path(self.directory.name) / "lifecycle.sqlite3", timezone="UTC")
        home = HomeController()
        scheduler = FakeScheduler(tasks)
        with patch.object(home, "close", wraps=home.close) as close_home, patch.object(tasks, "close", wraps=tasks.close) as close_tasks:
            with TestClient(create_app(home, tasks=tasks, scheduler=scheduler)):
                self.assertEqual(scheduler.starts, 1)
                self.assertEqual(scheduler.stops, 0)
            self.assertEqual(scheduler.stops, 1)
            close_tasks.assert_called_once()
            close_home.assert_called_once()

    def test_failed_scheduler_shutdown_leaves_its_database_open(self):
        from apm.home import HomeController
        from apm.tasks import TaskService

        tasks = TaskService(Path(self.directory.name) / "shutdown.sqlite3", timezone="UTC")
        self.addCleanup(tasks.close)
        home = HomeController()
        scheduler = FakeScheduler(tasks)
        with patch.object(scheduler, "stop", side_effect=RuntimeError("Scheduler is still stopping")), \
                patch.object(tasks, "close", wraps=tasks.close) as close_tasks, \
                patch.object(home, "close", wraps=home.close) as close_home:
            with self.assertRaises(RuntimeError):
                with TestClient(create_app(home, tasks=tasks, scheduler=scheduler)):
                    pass
            close_tasks.assert_not_called()
            close_home.assert_called_once()
            self.assertEqual(tasks.clock()["timezone"], "UTC")


if __name__ == "__main__":
    unittest.main()
