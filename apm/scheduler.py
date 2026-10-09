"""Durable task notifications delivered independently of model inference.

TaskService atomically creates due events and leases delivery claims. Multiple
workers therefore do not normally deliver the same event concurrently. Delivery
is at least once: a crash after output but before acknowledgement can repeat it.
Sinks must finish within the 30-second lease and should be quick and idempotent.
This worker cannot wake a sleeping computer or provide native OS notifications.
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
import math
import sys
import threading


_LOGGER = logging.getLogger(__name__)
_MAX_PER_CYCLE = 100
_LEASE_SECONDS = 30
_STOP_TIMEOUT_SECONDS = 6  # Allows a five-second SQLite busy timeout to unwind.


def terminal_sink(event: dict, stream=None) -> None:
    """Print one local notification; never invoke speech or an inference model."""
    stream = sys.stdout if stream is None else stream
    kind = {"timer": "Timer", "reminder": "Reminder", "todo": "Task"}.get(event["kind"], "Notification")
    name = " ".join("".join(char for char in event["name"] if char.isprintable() or char.isspace()).split())[:200]
    due = datetime.fromisoformat(event["due_at"])
    if due.tzinfo is None:
        raise ValueError("Notification deadline must include a timezone")
    late = max(0, int((datetime.now(timezone.utc) - due).total_seconds()))
    suffix = ""
    if late:
        duration = (f"{late}s" if late < 60 else f"{late // 60}m {late % 60}s" if late < 3600
                    else f"{late // 3600}h {(late % 3600) // 60}m")
        suffix = f" ({duration} late)"
    print(f"\a{kind}: {name or 'Unnamed task'}{suffix}.", file=stream, flush=True)


class Scheduler:
    def __init__(self, service, sink=None, *, poll_interval=0.5):
        if (isinstance(poll_interval, bool) or not isinstance(poll_interval, (float, int))
                or not math.isfinite(poll_interval) or poll_interval <= 0):
            raise ValueError("Scheduler poll interval must be a positive finite number")
        if sink is not None and not callable(sink):
            raise ValueError("Scheduler sink must be callable")
        self.service = service
        self.sink = terminal_sink if sink is None else sink
        self.poll_interval = float(poll_interval)
        self.last_error = None
        self._stop = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._cycle_lock = threading.Lock()
        self._thread = None

    @property
    def running(self) -> bool:
        with self._lifecycle_lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                if self._stop.is_set():
                    raise RuntimeError("Scheduler is still stopping")
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="apm-scheduler", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._stop.set()
            thread = self._thread
        # A sink may request shutdown itself. Never join the current thread.
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=_STOP_TIMEOUT_SECONDS)
            if thread.is_alive():
                self._error("Scheduler did not stop; keep its task service open until the worker exits")
                raise RuntimeError(self.last_error)

    def _error(self, message):
        # Do not log exception details: they can contain private task names.
        self.last_error = message
        _LOGGER.warning(message)

    def _release(self, event):
        try:
            self.service.release_notification(event["id"], event["claim_token"])
        except Exception:
            # Cancellation/snooze may already have invalidated this lease.
            pass

    def run_once(self) -> int:
        """Process due tasks and attempt up to 100 leased notifications once."""
        if self._stop.is_set() or not self._cycle_lock.acquire(blocking=False):
            return 0
        delivered = 0
        try:
            try:
                self.service.process_due()
            except Exception:
                self._error("Scheduler could not process due tasks; it will retry on the next tick")
                return 0
            self.last_error = None
            for _ in range(_MAX_PER_CYCLE):
                if self._stop.is_set():
                    break
                try:
                    event = self.service.claim_notification(lease_seconds=_LEASE_SECONDS)
                except Exception:
                    self._error("Scheduler could not claim a notification; it will retry on the next tick")
                    break
                if event is None:
                    break
                if self._stop.is_set():
                    self._release(event)
                    break
                try:
                    # Suppress an event cancelled/snoozed after the lease was issued.
                    # Cancellation during output itself cannot be made atomic.
                    if not self.service.notification_is_claimed(event["id"], event["claim_token"]):
                        continue
                    self.sink(event)
                except Exception:
                    self._release(event)
                    self._error("Scheduler notification delivery failed; it will retry on the next tick")
                    break  # Prevent tight retries of a failing sink in this cycle.
                try:
                    self.service.mark_delivered(event["id"], event["claim_token"])
                except Exception:
                    # Output already happened. Keep the lease until it expires to
                    # avoid immediately repeating an unacknowledged notification.
                    self._error("Scheduler could not acknowledge delivery; the notification may repeat")
                    break
                delivered += 1
            return delivered
        finally:
            self._cycle_lock.release()

    def _run(self):
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self.poll_interval)
