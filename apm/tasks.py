"""Persistent local timers, reminders, and leased notification delivery.

SQLite transactions serialize API and scheduler instances sharing this file.
Recurring reminders keep local wall time across DST. A nonexistent local time
is skipped; ambiguous times retain the original occurrence's fold preference.
Overdue recurrences coalesce into one notification, then resume in the future.
Notifications are at-least-once: a process can fail after output but before its
delivery acknowledgement. Claim tokens prevent an expired owner updating a
new owner's lease; cancellation invalidates pending notifications atomically.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone as datetime_timezone
import math
import os
from pathlib import Path
import sqlite3
from threading import RLock
import uuid
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

UTC = datetime_timezone.utc
MAX_SECONDS = 365 * 24 * 60 * 60
KINDS = {"timer", "reminder"}
STATUSES = {"scheduled", "paused", "completed", "cancelled", "due"}
SCHEMA_VERSION = 2


def _zone(value):
    if not isinstance(value, str) or not value or len(value) > 100:
        raise ValueError("Timezone must be an IANA name such as America/Los_Angeles or UTC")
    try:
        return ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError("Unknown timezone; use an IANA name such as America/Los_Angeles or UTC") from None


def _default_zone(explicit):
    if explicit is not None:
        return _zone(explicit).key
    for variable in ("APM_TIMEZONE", "TZ"):
        if value := os.environ.get(variable):
            return _zone(value.removeprefix(":")).key
    localtime = Path("/etc/localtime")
    try:
        if localtime.is_symlink():
            resolved = str(localtime.resolve())
            if "/zoneinfo/" in resolved:
                return _zone(resolved.split("/zoneinfo/", 1)[1]).key
    except (OSError, ValueError):
        pass
    return "UTC"


def _iso(value):
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _date(value):
    return datetime.fromisoformat(value)


def _name(value):
    if (not isinstance(value, str) or not value.strip() or len(value.strip()) > 120
            or not all(character.isprintable() for character in value)):
        raise ValueError("Name must contain 1–120 printable characters")
    return value.strip()


def _seconds(value):
    try:
        number = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else float("nan")
    except OverflowError:
        number = float("nan")
    if not math.isfinite(number) or not 0 < number <= MAX_SECONDS:
        raise ValueError("Seconds must be a finite positive number no greater than 365 days")
    return number


def _identifier(value):
    try:
        if not isinstance(value, str) or len(value) != 36:
            raise ValueError
        parsed = str(uuid.UUID(value))
        if parsed != value.lower():
            raise ValueError
        return parsed
    except (ValueError, AttributeError):
        raise ValueError("A valid task or notification UUID is required") from None


class TaskService:
    def __init__(self, path, *, timezone=None, now=None):
        self.default_timezone = _default_zone(timezone)
        self._now_source = now or (lambda: datetime.now(UTC))
        self._lock = RLock()
        self._db = None
        self.path = None if os.fspath(path) == ":memory:" else Path(path).expanduser().resolve()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(descriptor)
        connection = sqlite3.connect(str(self.path) if self.path else ":memory:", timeout=5,
                                     isolation_level=None, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.create_function("task_casefold", 1, str.casefold, deterministic=True)
        self._db = connection
        try:
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA foreign_keys = ON")
            if connection.execute("PRAGMA user_version").fetchone()[0] > SCHEMA_VERSION:
                raise ValueError("Task database was created by a newer application version")
            connection.execute("PRAGMA journal_mode = WAL")
            with self._transaction() as db:
                version = db.execute("PRAGMA user_version").fetchone()[0]
                if version > SCHEMA_VERSION:
                    raise ValueError("Task database was created by a newer application version")
                if version == 0:
                    existing = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchone()
                    if existing:
                        raise ValueError("Unrecognized task database schema")
                    db.execute("""CREATE TABLE tasks (
                        id TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('timer','reminder')),
                        name TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('scheduled','paused','completed','cancelled','due')),
                        due_at TEXT, timezone TEXT NOT NULL, repeat TEXT,
                        remaining_seconds REAL, duration_seconds REAL,
                        anchor_local TEXT, anchor_fold INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                        completed_at TEXT, cancelled_at TEXT)""")
                    db.execute("""CREATE TABLE notifications (
                        id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(id),
                        kind TEXT NOT NULL, name TEXT NOT NULL, due_at TEXT NOT NULL,
                        created_at TEXT NOT NULL, delivered_at TEXT, acknowledged_at TEXT,
                        leased_until TEXT, claim_token TEXT, invalidated_at TEXT,
                        delivery_attempts INTEGER NOT NULL DEFAULT 0, last_attempt_at TEXT,
                        UNIQUE(task_id, due_at))""")
                    db.execute("CREATE INDEX tasks_due ON tasks(status, due_at)")
                    db.execute("CREATE INDEX notifications_pending ON notifications(delivered_at, acknowledged_at, leased_until)")
                    db.execute("PRAGMA user_version = 1")
                    version = 1
                if version == 1:
                    self._migrate_occurrences(db)
            self._secure_sidecars()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _migrate_occurrences(db):
        """Preserve v1 history while separating occurrence identity from time."""
        db.execute("ALTER TABLE tasks ADD COLUMN occurrence INTEGER NOT NULL DEFAULT 0")
        db.execute("""CREATE TABLE notifications_v2 (
            id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(id),
            kind TEXT NOT NULL, name TEXT NOT NULL, due_at TEXT NOT NULL,
            created_at TEXT NOT NULL, delivered_at TEXT, acknowledged_at TEXT,
            leased_until TEXT, claim_token TEXT, invalidated_at TEXT,
            delivery_attempts INTEGER NOT NULL DEFAULT 0, last_attempt_at TEXT,
            occurrence INTEGER NOT NULL, UNIQUE(task_id, occurrence))""")
        db.execute("""INSERT INTO notifications_v2 (
            id,task_id,kind,name,due_at,created_at,delivered_at,acknowledged_at,
            leased_until,claim_token,invalidated_at,delivery_attempts,last_attempt_at,occurrence)
            SELECT id,task_id,kind,name,due_at,created_at,delivered_at,acknowledged_at,
            leased_until,claim_token,invalidated_at,delivery_attempts,last_attempt_at,
            ROW_NUMBER() OVER (PARTITION BY task_id ORDER BY created_at,due_at,id)-1
            FROM notifications""")
        db.execute("""UPDATE tasks SET occurrence=(
            SELECT COUNT(*) FROM notifications_v2 WHERE task_id=tasks.id)""")
        db.execute("DROP TABLE notifications")
        db.execute("ALTER TABLE notifications_v2 RENAME TO notifications")
        db.execute("CREATE INDEX notifications_pending ON notifications(delivered_at, acknowledged_at, leased_until)")
        if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("Task database contains invalid notification references")
        db.execute("PRAGMA user_version = 2")

    def _secure_sidecars(self):
        if self.path is not None:
            for suffix in ("-wal", "-shm"):
                try:
                    os.chmod(str(self.path) + suffix, 0o600)
                except FileNotFoundError:
                    pass

    def _now(self):
        value = self._now_source()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Task clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    def _open(self):
        if self._db is None:
            raise RuntimeError("Task service is closed")
        return self._db

    @contextmanager
    def _transaction(self):
        with self._lock:
            db = self._open()
            try:
                db.execute("BEGIN IMMEDIATE")
                yield db
                db.commit()
            except BaseException:
                if db.in_transaction:
                    db.rollback()
                raise

    def clock(self, timezone=None):
        zone = _zone(self.default_timezone if timezone is None else timezone)
        now = self._now()
        local = now.astimezone(zone)
        return {"utc": _iso(now), "local": local.isoformat(), "timezone": zone.key,
                "date": local.date().isoformat(), "weekday": local.strftime("%A")}

    def _reminder_values(self, name, due_at, timezone, repeat, now):
        name = _name(name)
        if repeat is not None and (not isinstance(repeat, str) or repeat not in {"daily", "weekly"}):
            raise ValueError("Repeat must be daily, weekly, or null")
        zone = _zone(self.default_timezone if timezone is None else timezone)
        try:
            if not isinstance(due_at, str) or len(due_at) > 100:
                raise ValueError
            parsed = datetime.fromisoformat(due_at)
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError
            due = parsed.astimezone(UTC)
            local = due.astimezone(zone)
        except (ValueError, OverflowError):
            raise ValueError("due_at must be an ISO datetime with an explicit UTC offset") from None
        if timezone is not None and (parsed.replace(tzinfo=None) != local.replace(tzinfo=None)
                                     or parsed.utcoffset() != local.utcoffset()):
            raise ValueError("Reminder time or UTC offset does not match the supplied timezone (check daylight saving time)")
        if not 0 < (due - now).total_seconds() <= MAX_SECONDS:
            raise ValueError("Reminder must be in the future and no more than 365 days away")
        return name, due, zone.key, local.replace(tzinfo=None).isoformat(), local.fold

    def validate_request(self, operation: str, kwargs: dict) -> None:
        """Preflight arguments without writes or task/state lookup for batch tools."""
        options = {
            "clock": (set(), {"timezone"}),
            "create_timer": ({"name", "duration_seconds"}, set()),
            "create_reminder": ({"name", "due_at"}, {"timezone", "repeat"}),
            "list_tasks": (set(), {"kind", "status", "name"}),
            "get_task": ({"id"}, set()),
            "update_timer": ({"id", "action"}, {"seconds"}),
            "snooze_task": ({"id", "duration_seconds"}, set()),
            "cancel_task": ({"id"}, set()),
            "complete_task": ({"id"}, set()),
        }
        if not isinstance(operation, str) or operation not in options or not isinstance(kwargs, dict):
            raise ValueError("Unknown or malformed task operation")
        required, optional = options[operation]
        if not required <= kwargs.keys() or not kwargs.keys() <= required | optional:
            raise ValueError("Invalid task operation arguments")
        if "id" in kwargs:
            _identifier(kwargs["id"])
        if operation == "clock":
            _zone(self.default_timezone if kwargs.get("timezone") is None else kwargs["timezone"])
        elif operation == "create_timer":
            _name(kwargs["name"])
            _seconds(kwargs["duration_seconds"])
        elif operation == "create_reminder":
            self._reminder_values(kwargs["name"], kwargs["due_at"], kwargs.get("timezone"), kwargs.get("repeat"), self._now())
        elif operation == "list_tasks":
            if kwargs.get("name") is not None:
                _name(kwargs["name"])
            for field, allowed in (("kind", KINDS), ("status", STATUSES)):
                value = kwargs.get(field)
                if value is not None and (not isinstance(value, str) or value not in allowed):
                    raise ValueError(f"Unknown task {field}")
        elif operation == "update_timer":
            action = kwargs["action"]
            if not isinstance(action, str) or action not in {"pause", "resume", "extend", "cancel"}:
                raise ValueError("Timer action must be pause, resume, extend, or cancel")
            if action == "extend":
                _seconds(kwargs.get("seconds"))
            elif kwargs.get("seconds") is not None:
                raise ValueError("Only extend accepts seconds")
        elif operation == "snooze_task":
            _seconds(kwargs["duration_seconds"])

    def _task(self, row, now):
        result = dict(row)
        result.pop("anchor_local")
        result.pop("anchor_fold")
        result.pop("occurrence")
        if result["kind"] == "timer":
            if result["status"] == "scheduled":
                result["remaining_seconds"] = max(0.0, (_date(result["due_at"]) - now).total_seconds())
            elif result["status"] != "paused":
                result["remaining_seconds"] = 0.0
        return result

    @staticmethod
    def _task_row(db, identifier):
        row = db.execute("SELECT * FROM tasks WHERE id = ?", (identifier,)).fetchone()
        if row is None:
            raise KeyError("Task was not found")
        return row

    def create_timer(self, name, duration_seconds):
        name, seconds = _name(name), _seconds(duration_seconds)
        now = self._now()
        identifier = str(uuid.uuid4())
        with self._transaction() as db:
            db.execute("""INSERT INTO tasks(id,kind,name,status,due_at,timezone,duration_seconds,created_at,updated_at)
                          VALUES (?,'timer',?,'scheduled',?,?,?,?,?)""",
                       (identifier, name, _iso(now + timedelta(seconds=seconds)), self.default_timezone, seconds, _iso(now), _iso(now)))
            return self._task(self._task_row(db, identifier), now)

    def create_reminder(self, name, due_at, timezone=None, repeat=None):
        now = self._now()
        name, due, zone, anchor, fold = self._reminder_values(name, due_at, timezone, repeat, now)
        identifier = str(uuid.uuid4())
        with self._transaction() as db:
            db.execute("""INSERT INTO tasks(id,kind,name,status,due_at,timezone,repeat,anchor_local,anchor_fold,created_at,updated_at)
                          VALUES (?,'reminder',?,'scheduled',?,?,?,?,?,?,?)""",
                       (identifier, name, _iso(due), zone, repeat, anchor, fold, _iso(now), _iso(now)))
            return self._task(self._task_row(db, identifier), now)

    def list_tasks(self, kind=None, status=None, name=None):
        self.validate_request("list_tasks", {"kind": kind, "status": status, "name": name})
        query = _name(name).casefold() if name is not None else None
        with self._lock:
            now = self._now()
            rows = self._open().execute("""SELECT * FROM tasks WHERE (? IS NULL OR kind=?) AND (? IS NULL OR status=?)
                AND (? IS NULL OR instr(task_casefold(name),?)>0)
                ORDER BY CASE status WHEN 'due' THEN 0 WHEN 'scheduled' THEN 1 WHEN 'paused' THEN 2 ELSE 3 END,
                CASE WHEN status IN ('due','scheduled') THEN due_at END, created_at DESC, id LIMIT 100""",
                                       (kind, kind, status, status, query, query)).fetchall()
            return [self._task(row, now) for row in rows]

    def get_task(self, id):
        identifier = _identifier(id)
        with self._lock:
            return self._task(self._task_row(self._open(), identifier), self._now())

    @staticmethod
    def _invalidate(db, identifier, now):
        stamp = _iso(now)
        db.execute("""UPDATE notifications SET invalidated_at=?,
            acknowledged_at=COALESCE(acknowledged_at,?), delivered_at=COALESCE(delivered_at,?),
            leased_until=NULL,claim_token=NULL WHERE task_id=? AND acknowledged_at IS NULL""",
                   (stamp, stamp, stamp, identifier))

    def _terminal(self, db, row, status, now):
        if row["status"] == status:
            return self._task(row, now)
        if row["status"] in {"completed", "cancelled"}:
            raise ValueError(f"Cannot change a {row['status']} task")
        self._invalidate(db, row["id"], now)
        db.execute("""UPDATE tasks SET status=?,remaining_seconds=NULL,updated_at=?,
            completed_at=CASE WHEN ?='completed' THEN ? ELSE completed_at END,
            cancelled_at=CASE WHEN ?='cancelled' THEN ? ELSE cancelled_at END WHERE id=?""",
                   (status, _iso(now), status, _iso(now), status, _iso(now), row["id"]))
        return self._task(self._task_row(db, row["id"]), now)

    def cancel_task(self, id):
        identifier, now = _identifier(id), self._now()
        with self._transaction() as db:
            return self._terminal(db, self._task_row(db, identifier), "cancelled", now)

    def complete_task(self, id):
        identifier, now = _identifier(id), self._now()
        with self._transaction() as db:
            return self._terminal(db, self._task_row(db, identifier), "completed", now)

    def update_timer(self, id, action, seconds=None):
        self.validate_request("update_timer", {"id": id, "action": action, "seconds": seconds})
        identifier, now = _identifier(id), self._now()
        with self._transaction() as db:
            row = self._task_row(db, identifier)
            if row["kind"] != "timer":
                raise ValueError("This operation requires a timer")
            if action == "cancel":
                return self._terminal(db, row, "cancelled", now)
            remaining = row["remaining_seconds"] if row["status"] == "paused" else (
                max(0.0, (_date(row["due_at"]) - now).total_seconds()) if row["due_at"] else 0)
            duration = row["duration_seconds"]
            if action == "pause":
                if row["status"] != "scheduled" or remaining <= 0:
                    raise ValueError("Only a running timer that is not yet due can be paused")
                status, due = "paused", None
            elif action == "resume":
                if row["status"] != "paused":
                    raise ValueError("Only a paused timer can be resumed")
                status, due = "scheduled", _iso(now + timedelta(seconds=remaining))
                remaining = None
            else:
                if row["status"] not in {"scheduled", "paused", "due"}:
                    raise ValueError("Only an active timer can be extended")
                remaining += _seconds(seconds)
                if remaining > MAX_SECONDS:
                    raise ValueError("Extended timer cannot have more than 365 days remaining")
                duration += seconds
                self._invalidate(db, identifier, now)
                status = "paused" if row["status"] == "paused" else "scheduled"
                due = None if status == "paused" else _iso(now + timedelta(seconds=remaining))
                if status == "scheduled":
                    remaining = None
            db.execute("""UPDATE tasks SET status=?,due_at=?,remaining_seconds=?,duration_seconds=?,updated_at=?,
                          occurrence=occurrence+? WHERE id=?""",
                       (status, due, remaining, duration, _iso(now), int(action == "extend"), identifier))
            return self._task(self._task_row(db, identifier), now)

    def snooze_task(self, id, duration_seconds):
        identifier, seconds, now = _identifier(id), _seconds(duration_seconds), self._now()
        with self._transaction() as db:
            row = self._task_row(db, identifier)
            if row["status"] not in {"due", "scheduled"}:
                raise ValueError("Only a scheduled or due task can be snoozed")
            self._invalidate(db, identifier, now)
            db.execute("""UPDATE tasks SET status='scheduled',due_at=?,remaining_seconds=NULL,updated_at=?,
                          occurrence=occurrence+1 WHERE id=?""",
                       (_iso(now + timedelta(seconds=seconds)), _iso(now), identifier))
            return self._task(self._task_row(db, identifier), now)

    @staticmethod
    def _next_occurrence(row, now):
        zone = _zone(row["timezone"])
        anchor = _date(row["anchor_local"])
        step = 1 if row["repeat"] == "daily" else 7
        elapsed_days = (now.astimezone(zone).date() - anchor.date()).days
        index = max(0, elapsed_days // step)
        for _ in range(32):
            naive = anchor + timedelta(days=step * index)
            local = naive.replace(tzinfo=zone, fold=row["anchor_fold"])
            candidate = local.astimezone(UTC)
            roundtrip = candidate.astimezone(zone)
            if (roundtrip.replace(tzinfo=None) == naive and roundtrip.utcoffset() == local.utcoffset()
                    and candidate > now):
                return candidate
            index += 1
        raise ValueError("Could not find the next valid local reminder occurrence")

    @staticmethod
    def _event(row, *, include_token=False):
        result = dict(row)
        result.pop("occurrence")
        if not include_token:
            result["claim_token"] = None
        return result

    def process_due(self):
        now = self._now()
        stamp = _iso(now)
        events = []
        with self._transaction() as db:
            rows = db.execute("SELECT * FROM tasks WHERE status='scheduled' AND due_at<=? ORDER BY due_at,id LIMIT 100",
                              (stamp,)).fetchall()
            for row in rows:
                identifier = str(uuid.uuid4())
                db.execute("""INSERT INTO notifications(id,task_id,kind,name,due_at,created_at,occurrence)
                              VALUES (?,?,?,?,?,?,?)""", (identifier, row["id"], row["kind"], row["name"], row["due_at"], stamp, row["occurrence"]))
                if row["repeat"]:
                    due = _iso(self._next_occurrence(row, now))
                    db.execute("UPDATE tasks SET due_at=?,updated_at=?,occurrence=occurrence+1 WHERE id=?", (due, stamp, row["id"]))
                else:
                    db.execute("UPDATE tasks SET status='due',remaining_seconds=NULL,updated_at=? WHERE id=?", (stamp, row["id"]))
                events.append(self._event(db.execute("SELECT * FROM notifications WHERE id=?", (identifier,)).fetchone()))
        return events

    def notifications(self, unread_only=False):
        if type(unread_only) is not bool:
            raise ValueError("unread_only must be a boolean")
        with self._lock:
            rows = self._open().execute("""SELECT * FROM notifications
                WHERE (?=0 OR (acknowledged_at IS NULL AND invalidated_at IS NULL))
                ORDER BY created_at DESC,due_at DESC,id LIMIT 100""", (int(unread_only),)).fetchall()
            return [self._event(row) for row in rows]

    @staticmethod
    def _event_row(db, identifier):
        row = db.execute("SELECT * FROM notifications WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise KeyError("Notification was not found")
        return row

    def acknowledge_notification(self, id):
        identifier, now = _identifier(id), self._now()
        with self._transaction() as db:
            self._event_row(db, identifier)
            db.execute("""UPDATE notifications SET acknowledged_at=COALESCE(acknowledged_at,?),
                          leased_until=NULL,claim_token=NULL WHERE id=?""", (_iso(now), identifier))
            return self._event(self._event_row(db, identifier))

    def claim_notification(self, lease_seconds=30):
        seconds, now = _seconds(lease_seconds), self._now()
        with self._transaction() as db:
            row = db.execute("""SELECT * FROM notifications WHERE delivered_at IS NULL
                AND acknowledged_at IS NULL AND invalidated_at IS NULL
                AND (leased_until IS NULL OR leased_until<=?)
                ORDER BY delivery_attempts,COALESCE(last_attempt_at,created_at),created_at,due_at,id LIMIT 1""", (_iso(now),)).fetchone()
            if row is None:
                return None
            token = str(uuid.uuid4())
            # New alarms outrank retries; repeated sink failures rotate among
            # pending alarms instead of one poison event starving the queue.
            db.execute("""UPDATE notifications SET claim_token=?,leased_until=?,
                          delivery_attempts=delivery_attempts+1,last_attempt_at=? WHERE id=?""",
                       (token, _iso(now + timedelta(seconds=seconds)), _iso(now), row["id"]))
            return self._event(self._event_row(db, row["id"]), include_token=True)

    @staticmethod
    def _owned(row, token, now):
        return bool(row is not None and row["claim_token"] == token and row["leased_until"]
                    and row["leased_until"] > _iso(now) and row["delivered_at"] is None
                    and row["acknowledged_at"] is None and row["invalidated_at"] is None)

    def notification_is_claimed(self, id, claim_token):
        identifier, token = _identifier(id), _identifier(claim_token)
        with self._lock:
            row = self._open().execute("SELECT * FROM notifications WHERE id=?", (identifier,)).fetchone()
            return self._owned(row, token, self._now())

    def mark_delivered(self, id, claim_token):
        identifier, token, now = _identifier(id), _identifier(claim_token), self._now()
        with self._transaction() as db:
            row = self._event_row(db, identifier)
            if not self._owned(row, token, now):
                raise ValueError("Notification claim is stale, expired, or no longer active")
            db.execute("UPDATE notifications SET delivered_at=?,leased_until=NULL,claim_token=NULL WHERE id=?", (_iso(now), identifier))
            return self._event(self._event_row(db, identifier))

    def release_notification(self, id, claim_token):
        identifier, token, now = _identifier(id), _identifier(claim_token), self._now()
        with self._transaction() as db:
            row = self._event_row(db, identifier)
            if not self._owned(row, token, now):
                raise ValueError("Notification claim is stale, expired, or no longer active")
            db.execute("UPDATE notifications SET leased_until=NULL,claim_token=NULL WHERE id=?", (identifier,))

    def close(self):
        with self._lock:
            connection, self._db = self._db, None
            if connection is not None:
                connection.close()
