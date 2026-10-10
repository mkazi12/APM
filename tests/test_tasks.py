from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
import unittest
from unittest.mock import patch
import uuid

from apm.tasks import MAX_SECONDS, TaskService

UTC = timezone.utc


class Clock:
    def __init__(self, value="2026-10-07T12:00:00+00:00"):
        self.value = datetime.fromisoformat(value)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += timedelta(seconds=seconds)


class TaskServiceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "tasks.sqlite3"
        self.clock = Clock()
        self.service = self.open()

    def open(self, zone="UTC"):
        service = TaskService(self.path, timezone=zone, now=self.clock)
        self.addCleanup(service.close)
        return service

    def due_timer(self):
        task = self.service.create_timer("Tea", 30)
        self.clock.advance(30)
        events = self.service.process_due()
        return task, events[0]

    def legacy_database(self):
        """A real v1 fixture, independent of the current schema initializer."""
        path = self.path.with_name("legacy.sqlite3")
        identifier, first, second, token = (str(uuid.uuid4()) for _ in range(4))
        with sqlite3.connect(path) as db:
            db.executescript("""
                CREATE TABLE tasks (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('timer','reminder')),
                    name TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('scheduled','paused','completed','cancelled','due')),
                    due_at TEXT, timezone TEXT NOT NULL, repeat TEXT, remaining_seconds REAL, duration_seconds REAL,
                    anchor_local TEXT, anchor_fold INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, completed_at TEXT, cancelled_at TEXT);
                CREATE TABLE notifications (
                    id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(id), kind TEXT NOT NULL,
                    name TEXT NOT NULL, due_at TEXT NOT NULL, created_at TEXT NOT NULL,
                    delivered_at TEXT, acknowledged_at TEXT, leased_until TEXT, claim_token TEXT, invalidated_at TEXT,
                    delivery_attempts INTEGER NOT NULL DEFAULT 0, last_attempt_at TEXT, UNIQUE(task_id,due_at));
                CREATE INDEX tasks_due ON tasks(status,due_at);
                CREATE INDEX notifications_pending ON notifications(delivered_at,acknowledged_at,leased_until);
                PRAGMA user_version=1;
            """)
            db.execute("""INSERT INTO tasks(id,kind,name,status,due_at,timezone,repeat,anchor_local,created_at,updated_at)
                VALUES (?,'reminder','Daily','scheduled','2026-10-08T09:00:00.000000Z','UTC','daily',
                        '2026-10-06T09:00:00','2026-10-05T12:00:00.000000Z','2026-10-07T12:00:00.000000Z')""", (identifier,))
            db.execute("""INSERT INTO notifications(id,task_id,kind,name,due_at,created_at,delivered_at,acknowledged_at,delivery_attempts,last_attempt_at)
                VALUES (?,?,'reminder','Daily','2026-10-06T09:00:00.000000Z','2026-10-06T09:00:00.000000Z',
                        '2026-10-06T09:00:01.000000Z','2026-10-06T09:00:02.000000Z',1,'2026-10-06T09:00:00.000000Z')""", (first,identifier))
            db.execute("""INSERT INTO notifications(id,task_id,kind,name,due_at,created_at,leased_until,claim_token,delivery_attempts,last_attempt_at)
                VALUES (?,?,'reminder','Daily','2026-10-07T09:00:00.000000Z','2026-10-07T12:00:00.000000Z',
                        '2026-10-07T12:00:30.000000Z',?,2,'2026-10-07T12:00:00.000000Z')""", (second,identifier,token))
        return path, identifier, second, token

    def test_clock_uses_correct_local_date_and_offset(self):
        result = self.service.clock("America/Los_Angeles")
        self.assertEqual(result["date"], "2026-10-07")
        self.assertEqual(result["weekday"], "Wednesday")
        self.assertTrue(result["local"].endswith("-07:00"))
        self.assertEqual(result["timezone"], "America/Los_Angeles")

    def test_timezone_precedence_and_invalid_zone(self):
        with patch.dict(os.environ, {"APM_TIMEZONE": "Europe/London", "TZ": "Asia/Tokyo"}):
            service = TaskService(":memory:", now=self.clock)
            self.addCleanup(service.close)
            self.assertEqual(service.default_timezone, "Europe/London")
            explicit = TaskService(":memory:", timezone="UTC", now=self.clock)
            self.addCleanup(explicit.close)
            self.assertEqual(explicit.default_timezone, "UTC")
        with self.assertRaises(ValueError):
            self.service.clock("Not/A_Zone")

    def test_timer_persistence_pause_extend_resume_and_other_instance_visibility(self):
        timer = self.service.create_timer("Tea", 60)
        other = self.open()
        self.clock.advance(10)
        self.assertEqual(other.get_task(timer["id"])["remaining_seconds"], 50)
        paused = other.update_timer(timer["id"], "pause")
        self.assertEqual(paused["remaining_seconds"], 50)
        self.assertIsNone(paused["due_at"])
        self.clock.advance(100)
        self.assertEqual(self.service.get_task(timer["id"])["remaining_seconds"], 50)
        self.assertEqual(self.service.update_timer(timer["id"], "extend", 20)["remaining_seconds"], 70)
        resumed = other.update_timer(timer["id"], "resume")
        self.assertEqual(resumed["status"], "scheduled")
        self.assertEqual(resumed["remaining_seconds"], 70)
        self.service.close()
        reopened = self.open()
        self.assertEqual(reopened.get_task(timer["id"])["remaining_seconds"], 70)

    def test_invalid_numbers_names_states_and_preflight_make_no_writes(self):
        for value in (True, False, 0, -1, float("nan"), float("inf"), MAX_SECONDS + 1, 10**1000, "30"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.service.create_timer("Tea", value)
        for name in ("", "   ", "x" * 121, "line\nbreak", None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.service.create_timer(name, 30)
        self.assertEqual(self.service.list_tasks(), [])
        with self.assertRaises(ValueError):
            self.service.validate_request("create_reminder", {"name": "Future", "due_at": "2026-10-08T09:00:00"})
        self.assertEqual(self.service.list_tasks(), [])
        timer = self.service.create_timer("Tea", 30)
        with self.assertRaises(ValueError):
            self.service.update_timer(timer["id"], "resume")
        with self.assertRaises(ValueError):
            self.service.update_timer(timer["id"], "pause", seconds=5)
        self.service.complete_task(timer["id"])
        with self.assertRaises(ValueError):
            self.service.cancel_task(timer["id"])

    def test_reminder_requires_future_offset_and_valid_explicit_zone_offset(self):
        for due in ("2026-10-08", "2026-10-08T09:00:00", "2026-10-06T09:00:00Z",
                    "2028-10-08T09:00:00Z"):
            with self.subTest(due=due), self.assertRaises(ValueError):
                self.service.create_reminder("Call", due)
        with self.assertRaisesRegex(ValueError, "offset"):
            self.service.create_reminder("Call", "2026-10-08T09:00:00-08:00", timezone="America/Los_Angeles")
        reminder = self.service.create_reminder("Call", "2026-10-08T09:00:00-07:00", timezone="America/Los_Angeles")
        self.assertEqual(reminder["due_at"], "2026-10-08T16:00:00.000000Z")
        west = self.open("America/Los_Angeles")
        self.assertEqual(west.create_reminder("UTC input", "2026-10-08T16:00:00Z")["timezone"], "America/Los_Angeles")

    def test_due_processing_is_atomic_across_concurrent_instances(self):
        timer = self.service.create_timer("Tea", 1)
        other = self.open()
        self.clock.advance(2)
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda service: service.process_due(), [self.service, other]))
        self.assertEqual(sum(len(rows) for rows in results), 1)
        self.assertEqual(self.service.get_task(timer["id"])["status"], "due")
        self.assertEqual(len(other.notifications()), 1)
        self.assertEqual(other.process_due(), [])

    def test_claims_have_one_owner_and_expired_tokens_cannot_ack_new_lease(self):
        _, event = self.due_timer()
        other = self.open()
        with ThreadPoolExecutor(max_workers=2) as executor:
            claims = list(executor.map(lambda service: service.claim_notification(10), [self.service, other]))
        first = next(item for item in claims if item)
        self.assertEqual(sum(item is not None for item in claims), 1)
        self.assertTrue(self.service.notification_is_claimed(event["id"], first["claim_token"]))
        self.clock.advance(11)
        second = other.claim_notification(10)
        self.assertNotEqual(first["claim_token"], second["claim_token"])
        with self.assertRaises(ValueError):
            self.service.mark_delivered(event["id"], first["claim_token"])
        with self.assertRaises(ValueError):
            self.service.release_notification(event["id"], first["claim_token"])
        delivered = other.mark_delivered(event["id"], second["claim_token"])
        self.assertIsNotNone(delivered["delivered_at"])
        self.assertIsNone(self.service.claim_notification())

    def test_failed_sink_can_release_and_reclaim_notification(self):
        _, event = self.due_timer()
        first = self.service.claim_notification()
        self.service.release_notification(event["id"], first["claim_token"])
        second = self.service.claim_notification()
        self.assertNotEqual(first["claim_token"], second["claim_token"])
        self.assertIsNone(self.service.notifications()[0]["claim_token"])

    def test_failing_old_notification_does_not_starve_newer_alarm(self):
        _, first = self.due_timer()
        claimed = self.service.claim_notification()
        self.service.release_notification(first["id"], claimed["claim_token"])
        _, second = self.due_timer()
        next_claim = self.service.claim_notification()
        self.assertEqual(next_claim["id"], second["id"])
        self.service.mark_delivered(second["id"], next_claim["claim_token"])
        self.assertEqual(self.service.claim_notification()["id"], first["id"])

    def test_cancel_snooze_and_complete_invalidate_claimed_old_alarms(self):
        for action in ("cancel", "snooze", "complete"):
            with self.subTest(action=action):
                task, event = self.due_timer()
                claimed = self.service.claim_notification()
                if action == "cancel":
                    self.service.cancel_task(task["id"])
                elif action == "snooze":
                    self.service.snooze_task(task["id"], 60)
                else:
                    self.service.complete_task(task["id"])
                self.assertFalse(self.service.notification_is_claimed(event["id"], claimed["claim_token"]))
                with self.assertRaises(ValueError):
                    self.service.mark_delivered(event["id"], claimed["claim_token"])
                self.assertEqual(self.service.notifications(unread_only=True), [])
                self.assertIsNone(self.service.claim_notification())
                if action == "snooze":
                    self.clock.advance(60)
                    self.assertEqual(len(self.service.process_due()), 1)
                    self.service.acknowledge_notification(self.service.notifications(unread_only=True)[0]["id"])

    def test_acknowledgement_prevents_pending_delivery(self):
        _, event = self.due_timer()
        acknowledged = self.service.acknowledge_notification(event["id"])
        self.assertIsNotNone(acknowledged["acknowledged_at"])
        self.assertIsNone(self.service.claim_notification())
        self.assertEqual(self.service.notifications(unread_only=True), [])

    def test_restart_emits_one_coalesced_daily_notification_and_keeps_wall_time(self):
        reminder = self.service.create_reminder("Breakfast", "2026-10-08T09:00:00-07:00",
                                                timezone="America/Los_Angeles", repeat="daily")
        self.service.close()
        self.clock.value = datetime.fromisoformat("2026-11-05T20:00:00+00:00")
        restarted = self.open()
        events = restarted.process_due()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["due_at"], reminder["due_at"])
        self.assertEqual(restarted.get_task(reminder["id"])["due_at"], "2026-11-06T17:00:00.000000Z")
        self.assertEqual(restarted.process_due(), [])

    def test_nonexistent_dst_times_rejected_and_recurring_gap_is_skipped(self):
        self.clock.value = datetime.fromisoformat("2026-03-06T12:00:00+00:00")
        with self.assertRaisesRegex(ValueError, "daylight"):
            self.service.create_reminder("Gap", "2026-03-08T02:30:00-08:00", timezone="America/Los_Angeles")
        reminder = self.service.create_reminder("Early", "2026-03-07T02:30:00-08:00",
                                                timezone="America/Los_Angeles", repeat="daily")
        self.clock.value = datetime.fromisoformat("2026-03-07T11:00:00+00:00")
        self.service.process_due()
        self.assertEqual(self.service.get_task(reminder["id"])["due_at"], "2026-03-09T09:30:00.000000Z")

    def test_weekly_reminder_preserves_local_weekday_across_dst(self):
        reminder = self.service.create_reminder("Weekly", "2026-10-25T09:00:00-07:00",
                                                timezone="America/Los_Angeles", repeat="weekly")
        self.clock.value = datetime.fromisoformat("2026-10-25T17:00:00+00:00")
        self.service.process_due()
        self.assertEqual(self.service.get_task(reminder["id"])["due_at"], "2026-11-01T17:00:00.000000Z")

    def test_snoozing_before_first_recurrence_keeps_original_future_anchor(self):
        reminder = self.service.create_reminder("Daily", "2026-10-08T09:00:00Z", repeat="daily")
        self.service.snooze_task(reminder["id"], 60)
        self.clock.advance(60)
        self.service.process_due()
        self.assertEqual(self.service.get_task(reminder["id"])["due_at"], reminder["due_at"])

    def test_unknown_valid_ids_are_not_found_errors(self):
        unknown = str(uuid.uuid4())
        for operation in (self.service.get_task, self.service.cancel_task,
                          self.service.complete_task, self.service.acknowledge_notification):
            with self.subTest(operation=operation.__name__), self.assertRaises(KeyError):
                operation(unknown)

    def test_lists_are_bounded_and_active_tasks_precede_completed_history(self):
        for index in range(101):
            task = self.service.create_timer(f"Finished {index}", 10)
            self.service.complete_task(task["id"])
        active = self.service.create_timer("Still running", 60)
        listed = self.service.list_tasks()
        self.assertEqual(len(listed), 100)
        self.assertEqual(listed[0]["id"], active["id"])
        self.assertEqual(len(self.service.list_tasks(status="scheduled")), 1)

    def test_name_search_filters_before_limit_with_unicode_and_literal_wildcards(self):
        for index in range(101):
            self.service.create_timer(f"Earlier timer {index}", 10)
        matching = self.service.create_timer("Straße 100%_ Tea", 600)
        self.service.validate_request("list_tasks", {"name":"STRASSE 100%_", "kind":"timer", "status":"scheduled"})
        found = self.service.list_tasks(name="STRASSE 100%_", kind="timer", status="scheduled")
        self.assertEqual([task["id"] for task in found], [matching["id"]])
        self.assertEqual(self.service.list_tasks(name="100%_", status="paused"), [])
        self.assertEqual(self.service.list_tasks(name="100X_"), [])
        for invalid in ("", "   ", True, "x" * 121):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.service.list_tasks(name=invalid)

    def test_rearming_same_deadline_creates_new_occurrence_and_invalidates_old_claim(self):
        for action in ("snooze", "extend"):
            with self.subTest(action=action):
                task, old = self.due_timer()
                claimed = self.service.claim_notification()
                self.clock.advance(-30)
                if action == "snooze":
                    updated = self.service.snooze_task(task["id"], 30)
                else:
                    # A due timer's old deadline may be future after a clock
                    # correction; tiny additions round back to that deadline.
                    updated = self.service.update_timer(task["id"], "extend", 1e-9)
                self.assertEqual(updated["due_at"], old["due_at"])
                self.assertFalse(self.service.notification_is_claimed(old["id"], claimed["claim_token"]))
                self.clock.advance(31)
                other = self.open()
                with ThreadPoolExecutor(max_workers=2) as executor:
                    produced = list(executor.map(lambda service: service.process_due(), [self.service,other]))
                fresh = [event for events in produced for event in events]
                self.assertEqual(len(fresh), 1)
                self.assertNotEqual(fresh[0]["id"], old["id"])
                self.assertEqual(fresh[0]["due_at"], old["due_at"])
                self.assertNotIn("occurrence", fresh[0])
                self.assertNotIn("occurrence", self.service.get_task(task["id"]))
                self.service.acknowledge_notification(fresh[0]["id"])

    def test_v1_migration_preserves_history_leases_and_recurring_schedule(self):
        path, identifier, event, token = self.legacy_database()
        with sqlite3.connect(path) as db:
            db.row_factory = sqlite3.Row
            original = {row["id"]:dict(row) for row in db.execute("SELECT * FROM notifications")}
        migrated = TaskService(path, timezone="UTC", now=self.clock)
        self.addCleanup(migrated.close)
        self.assertEqual(migrated.get_task(identifier)["due_at"], "2026-10-08T09:00:00.000000Z")
        self.assertTrue(migrated.notification_is_claimed(event, token))
        with sqlite3.connect(path) as db:
            db.row_factory = sqlite3.Row
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 2)
            after = {row["id"]:{key:row[key] for key in row.keys() if key != "occurrence"}
                     for row in db.execute("SELECT * FROM notifications")}
            self.assertEqual(after, original)
            self.assertEqual(db.execute("PRAGMA foreign_key_check").fetchall(), [])
        migrated.snooze_task(identifier, 60)
        self.assertFalse(migrated.notification_is_claimed(event, token))
        self.clock.advance(60)
        self.assertEqual(len(migrated.process_due()), 1)
        self.clock.value = datetime.fromisoformat("2026-10-08T09:00:00+00:00")
        self.assertEqual(len(migrated.process_due()), 1)
        self.assertEqual(migrated.get_task(identifier)["due_at"], "2026-10-09T09:00:00.000000Z")
        migrated.close()
        reopened = TaskService(path, timezone="UTC", now=self.clock)
        self.addCleanup(reopened.close)
        self.assertEqual(len(reopened.notifications()), 4)
        self.assertEqual(reopened.process_due(), [])

    def test_failed_v1_migration_rolls_back_schema_and_history(self):
        path, _, _, _ = self.legacy_database()
        migrate = TaskService._migrate_occurrences
        def fail_after_migration(db):
            migrate(db)
            raise OSError("Synthetic migration failure")
        with patch.object(TaskService, "_migrate_occurrences", side_effect=fail_after_migration), self.assertRaises(OSError):
            TaskService(path, timezone="UTC", now=self.clock)
        with sqlite3.connect(path) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM notifications").fetchone()[0], 2)
            self.assertNotIn("occurrence", [row[1] for row in db.execute("PRAGMA table_info(tasks)")])
            self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='notifications_v2'").fetchone())

    def test_database_and_sidecars_are_private_and_newer_schema_is_refused(self):
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(self.path) + suffix)
            if sidecar.exists():
                self.assertEqual(stat.S_IMODE(sidecar.stat().st_mode), 0o600)
        self.service.close()
        with sqlite3.connect(self.path) as db:
            db.execute("PRAGMA user_version = 3")
        with self.assertRaisesRegex(ValueError, "newer"):
            TaskService(self.path, timezone="UTC")

    def test_close_is_idempotent_and_naive_clock_is_rejected(self):
        self.service.close()
        self.service.close()
        with self.assertRaises(RuntimeError):
            self.service.list_tasks()
        service = TaskService(":memory:", timezone="UTC", now=lambda: datetime(2026, 1, 1))
        self.addCleanup(service.close)
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            service.create_timer("Tea", 10)


if __name__ == "__main__":
    unittest.main()
