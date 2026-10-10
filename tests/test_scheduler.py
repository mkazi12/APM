from copy import deepcopy
from datetime import datetime, timedelta, timezone
import io
import threading
import time
import unittest
from unittest.mock import patch

from apm.scheduler import Scheduler, terminal_sink
from apm.tasks import TaskService


class FakeService:
    """Small atomic lease store with a manually advanced clock."""
    def __init__(self, count=1, due=0):
        self.now = 0
        self.due = due
        self.pending = [self.event(index) for index in range(count)]
        self.events = []
        self.tokens = 0
        self.lock = threading.RLock()
        self.process_calls = 0
        self.release_calls = []
        self.delivered = []

    @staticmethod
    def event(index):
        return {"id": str(index), "task_id": "task-" + str(index), "kind": "timer", "name": "Test timer",
                "due_at": "2026-10-07T10:00:00+00:00", "claim_token": None,
                "leased_until": None, "delivered_at": None}

    def process_due(self):
        with self.lock:
            self.process_calls += 1
            if self.now < self.due:
                return []
            created, self.pending = self.pending, []
            self.events.extend(created)
            return deepcopy(created)

    def claim_notification(self, lease_seconds=30):
        with self.lock:
            for event in self.events:
                if event["delivered_at"] is None and (event["leased_until"] is None or event["leased_until"] <= self.now):
                    self.tokens += 1
                    event.update(claim_token=str(self.tokens), leased_until=self.now + lease_seconds)
                    return deepcopy(event)
        return None

    def notification_is_claimed(self, identifier, token):
        with self.lock:
            return any(e["id"] == identifier and e["claim_token"] == token
                       and e["delivered_at"] is None and e["leased_until"] > self.now for e in self.events)

    def mark_delivered(self, identifier, token):
        with self.lock:
            if not self.notification_is_claimed(identifier, token):
                raise ValueError("Stale claim")
            event = next(e for e in self.events if e["id"] == identifier)
            event["delivered_at"] = self.now
            self.delivered.append(identifier)

    def release_notification(self, identifier, token):
        with self.lock:
            if not self.notification_is_claimed(identifier, token):
                raise ValueError("Stale claim")
            event = next(e for e in self.events if e["id"] == identifier)
            event.update(claim_token=None, leased_until=None)
            self.release_calls.append(identifier)


class SchedulerTests(unittest.TestCase):
    def test_rearmed_old_deadline_does_not_block_unrelated_alarm_delivery(self):
        now = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)
        service = TaskService(":memory:", timezone="UTC", now=lambda: now)
        self.addCleanup(service.close)
        rearmed = service.create_timer("Rearmed timer", 30)
        now += timedelta(seconds=30)
        old = service.process_due()[0]
        now -= timedelta(seconds=30)  # Host wall-clock correction.
        service.snooze_task(rearmed["id"], 30)
        healthy = service.create_timer("Independent timer", 40)
        now += timedelta(seconds=41)
        received = []
        scheduler = Scheduler(service, sink=lambda event: received.append(event))
        self.assertEqual(scheduler.run_once(), 2)
        self.assertEqual({event["task_id"] for event in received}, {rearmed["id"],healthy["id"]})
        self.assertNotIn(old["id"], [event["id"] for event in received])
        self.assertIsNone(scheduler.last_error)
        self.assertEqual(scheduler.run_once(), 0)

    def test_sqlite_worker_delivers_other_alarm_after_per_event_sink_failure(self):
        now = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)
        service = TaskService(":memory:", timezone="UTC", now=lambda: now)
        self.addCleanup(service.close)
        failed = service.create_timer("Failing sink", 1)
        healthy = service.create_timer("Working sink", 2)
        now += timedelta(seconds=3)
        received = []
        def sink(event):
            if event["task_id"] == failed["id"]:
                raise OSError("Cannot deliver this event")
            received.append(event["task_id"])
        scheduler = Scheduler(service, sink)
        with self.assertLogs("apm.scheduler", level="WARNING"):
            self.assertEqual(scheduler.run_once(), 0)
        with self.assertLogs("apm.scheduler", level="WARNING"):
            self.assertEqual(scheduler.run_once(), 1)
        self.assertEqual(received, [healthy["id"]])
        events = {event["task_id"]: event for event in service.notifications()}
        self.assertIsNotNone(events[healthy["id"]]["delivered_at"])
        self.assertIsNone(events[failed["id"]]["delivered_at"])
        self.assertIsNone(events[failed["id"]]["leased_until"])

    def test_due_clock_is_independent_of_model_or_request_activity(self):
        service = FakeService(due=10)
        received, delivered = [], threading.Event()
        def sink(event):
            received.append((event["id"], threading.get_ident()))
            delivered.set()
        scheduler = Scheduler(service, sink, poll_interval=0.01)
        self.addCleanup(scheduler.stop)
        scheduler.start()
        scheduler.start()  # Idempotent, does not start a second worker.
        self.assertFalse(delivered.wait(0.03))
        service.now = 10
        # No inference call or user request is needed to trigger the notification.
        self.assertTrue(delivered.wait(1))
        scheduler.stop()
        self.assertEqual(received, [("0", received[0][1])])
        self.assertNotEqual(received[0][1], threading.get_ident())
        self.assertEqual(service.delivered, ["0"])
        self.assertFalse(scheduler.running)

    def test_failed_sink_releases_lease_and_retries_only_next_cycle(self):
        service = FakeService(count=2)
        attempts = []
        def sink(event):
            attempts.append(event["id"])
            if len(attempts) == 1:
                raise OSError("private reminder contents")
        scheduler = Scheduler(service, sink)
        with self.assertLogs("apm.scheduler", level="WARNING") as logged:
            self.assertEqual(scheduler.run_once(), 0)
        self.assertEqual(attempts, ["0"])
        self.assertEqual(service.release_calls, ["0"])
        self.assertNotIn("private reminder", "".join(logged.output))
        self.assertIsNotNone(scheduler.last_error)
        self.assertEqual(scheduler.run_once(), 2)
        self.assertEqual(attempts, ["0", "0", "1"])
        self.assertIsNone(scheduler.last_error)

    def test_atomic_claim_prevents_two_workers_delivering_same_event(self):
        service = FakeService()
        entered, finish = threading.Event(), threading.Event()
        calls = []
        def sink(event):
            calls.append(event["id"])
            entered.set()
            finish.wait(1)
        first, second = Scheduler(service, sink), Scheduler(service, sink)
        thread = threading.Thread(target=first.run_once)
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertEqual(second.run_once(), 0)
            self.assertEqual(first.run_once(), 0)  # No reentrant cycle on one instance.
        finally:
            finish.set()
            thread.join(1)
        self.assertEqual(calls, ["0"])
        self.assertEqual(service.delivered, ["0"])

    def test_delivery_ack_failure_keeps_lease_until_expiry_and_can_repeat(self):
        service, calls = FakeService(), []
        scheduler = Scheduler(service, lambda event: calls.append(event["id"]))
        with patch.object(service, "mark_delivered", side_effect=OSError("private text")), \
             self.assertLogs("apm.scheduler", level="WARNING"):
            self.assertEqual(scheduler.run_once(), 0)
        self.assertEqual(service.release_calls, [])
        self.assertEqual(scheduler.run_once(), 0)
        service.now = 31
        self.assertEqual(scheduler.run_once(), 1)
        self.assertEqual(calls, ["0", "0"])

    def test_cancelled_claim_is_suppressed_before_output(self):
        service, calls = FakeService(), []
        scheduler = Scheduler(service, lambda event: calls.append(event["id"]))
        original_claim = service.claim_notification
        def claim_then_cancel(**kwargs):
            event = original_claim(**kwargs)
            if event:
                service.events.clear()
            return event
        with patch.object(service, "claim_notification", side_effect=claim_then_cancel):
            self.assertEqual(scheduler.run_once(), 0)
        self.assertEqual(calls, [])

    def test_cycle_is_bounded_and_remaining_work_waits_for_next_tick(self):
        service, calls = FakeService(count=105), []
        scheduler = Scheduler(service, lambda event: calls.append(event["id"]))
        self.assertEqual(scheduler.run_once(), 100)
        self.assertEqual(scheduler.run_once(), 5)
        self.assertEqual(len(set(calls)), 105)

    def test_stop_wakes_long_poll_immediately_and_restart_works(self):
        service, invoked = FakeService(count=0), threading.Event()
        original_process = service.process_due
        def process():
            result = original_process()
            invoked.set()
            return result
        service.process_due = process
        scheduler = Scheduler(service, lambda event: None, poll_interval=60)
        self.addCleanup(scheduler.stop)
        scheduler.start()
        self.assertTrue(invoked.wait(1))
        started = time.monotonic()
        scheduler.stop()
        scheduler.stop()
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertFalse(scheduler.running)
        invoked.clear()
        scheduler.start()
        self.assertTrue(invoked.wait(1))

    def test_sink_can_stop_worker_without_delivering_rest_of_batch(self):
        service = FakeService(count=3)
        requested_stop = threading.Event()
        def sink(event):
            scheduler.stop()
            requested_stop.set()
        scheduler = Scheduler(service, sink)
        self.addCleanup(scheduler.stop)
        scheduler.start()
        self.assertTrue(requested_stop.wait(1))
        scheduler.stop()
        self.assertEqual(len(service.delivered), 1)
        self.assertFalse(scheduler.running)

    def test_blocked_sink_shutdown_raises_before_service_can_be_closed(self):
        entered, finish = threading.Event(), threading.Event()
        def sink(event):
            entered.set()
            finish.wait(1)
        scheduler = Scheduler(FakeService(), sink)
        scheduler.start()
        try:
            self.assertTrue(entered.wait(1))
            with patch("apm.scheduler._STOP_TIMEOUT_SECONDS", 0.01), \
                 self.assertLogs("apm.scheduler", level="WARNING"), \
                 self.assertRaisesRegex(RuntimeError, "keep its task service open"):
                scheduler.stop()
            self.assertTrue(scheduler.running)
            with self.assertRaisesRegex(RuntimeError, "still stopping"):
                scheduler.start()
        finally:
            finish.set()
            scheduler.stop()
        self.assertFalse(scheduler.running)

    def test_terminal_sink_bells_reports_lateness_and_strips_control_sequences(self):
        class FrozenDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 10, 7, 10, 1, 15, tzinfo=timezone.utc)
        stream = io.StringIO()
        event = {**FakeService.event(0), "kind": "reminder", "name": "Water\nplants\x1b"}
        with patch("apm.scheduler.datetime", FrozenDateTime):
            terminal_sink(event, stream)
        self.assertEqual(stream.getvalue(), "\aReminder: Water plants (1m 15s late).\n")
        stream.close()
        scheduler = Scheduler(FakeService(), lambda event: terminal_sink(event, stream))
        with self.assertLogs("apm.scheduler", level="WARNING"):
            self.assertEqual(scheduler.run_once(), 0)

    def test_process_failure_is_sanitized_and_does_not_kill_subsequent_cycles(self):
        service = FakeService()
        scheduler = Scheduler(service, lambda event: None)
        with patch.object(service, "process_due", side_effect=RuntimeError("private task name")), \
             self.assertLogs("apm.scheduler", level="WARNING") as logs:
            self.assertEqual(scheduler.run_once(), 0)
        self.assertNotIn("private task", "".join(logs.output))
        self.assertEqual(scheduler.run_once(), 1)


if __name__ == "__main__":
    unittest.main()
