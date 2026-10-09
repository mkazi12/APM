# Independent assistant services

The backend owns the clock, saved tasks, device states, and notification delivery. Gemma translates requests into validated calls. The app API invokes the same service methods without loading a language model. Results are stored in conversation history; immediate confirmations are formatted from the backend response, so a simple timer command needs only one inference.

`apm/tasks.py` stores tasks and notifications in SQLite. `apm/scheduler.py` checks deadlines on a background thread. `apm/assistant.py` combines scheduling tools with the registered home-device tools. The scheduler never calls Gemma, generates Python, or performs a device command.

## Run locally

```sh
cd ~/APM
.venv/bin/apm-server --database work/assistant.sqlite3 --timezone America/Los_Angeles
```

The server needs the existing `home` extra. It starts without Ollama, a microphone, or a home registry; devices default to simulation. Add `--registry work/home.json` to use the previously created registry. The API remains loopback-only and uses the existing optional `APM_API_TOKEN` authentication.

From a second terminal:

```sh
.venv/bin/apm --database work/assistant.sqlite3 --timezone America/Los_Angeles --no-scheduler
```

Try “Set a pasta timer for ten minutes,” “How much time is left on the pasta timer?”, “Pause the pasta timer,” and “Add two minutes to the pasta timer.” Use `/tasks` for a direct listing. Your usual voice command accepts the same database/timezone options. `--no-scheduler` leaves notification delivery to the server; standalone APM runs a scheduler by default. If both run schedulers, a notification lease normally lets only one worker emit the alert, so it may appear in either terminal. App notifications remain readable regardless of which terminal delivered them.

Both processes must point to the same database file; relative paths resolve from their working directory. Database files and SQLite journal sidecars are ignored by Git. New database files are created with owner-only permissions. Do not copy only the main database file while it is open in WAL mode; stop the processes or use a SQLite backup operation.

## Clock and timers

```sh
curl 'http://127.0.0.1:8765/v1/clock?timezone=Europe/London'
curl -X POST http://127.0.0.1:8765/v1/timers \
  -H 'Content-Type: application/json' \
  -d '{"name":"Pasta","duration_seconds":600}'
curl http://127.0.0.1:8765/v1/timers
```

Take the returned task `id` and use it in these routes:

| Method | Route | Body or purpose |
| --- | --- | --- |
| GET | `/v1/timers/{id}` | Current state and remaining seconds |
| POST | `/v1/timers/{id}/actions` | `{"action":"pause"}` or `resume`, `cancel`, `complete` |
| POST | `/v1/timers/{id}/actions` | `{"action":"extend","seconds":120}` |
| POST | `/v1/timers/{id}/actions` | `{"action":"snooze","seconds":300}` |

Extend adds seconds to the current remaining duration; it does not replace the timer's original duration. Snooze creates a new deadline measured from now. A paused timer retains its remaining duration over restarts. Due times are persisted as UTC timestamps and use the host clock; keep the machine's time correct. No model counts down seconds.

## Reminders

Use the same create/list/get/action pattern at `/v1/reminders`. Creation accepts this shape; substitute an actual future date:

```json
{
  "name": "Take out the bins",
  "due_at": "2027-01-08T08:00:00-08:00",
  "timezone": "America/Los_Angeles",
  "repeat": "weekly"
}
```

Omit `repeat` for a one-off reminder. Daily and weekly recurrences preserve local wall time using the named IANA timezone. An explicitly supplied timezone must agree with the datetime's offset and local time. Nonexistent local times are rejected when creating a reminder. A recurring occurrence that falls in a daylight-saving gap is skipped; an ambiguous fall-back occurrence uses the original fold preference. Offset-free dates and free-form strings such as “tomorrow morning” are rejected at the API boundary. Gemma uses current clock context to resolve natural language and should clarify ambiguous requests.

Reminder actions are `cancel`, `complete`, or `snooze` with positive `seconds`. Cancel/complete ends the whole task, including recurrence. Notification acknowledgement marks that occurrence read without completing the recurring task. Snooze, cancellation, and completion invalidate older pending alerts for that task. A delivery already in progress can still finish.

## Notifications and recovery

```sh
curl 'http://127.0.0.1:8765/v1/notifications?unread_only=true'
curl -X POST http://127.0.0.1:8765/v1/notifications/NOTIFICATION_ID/acknowledge
```

The worker checks about every half second. Due-task transitions and notification creation happen in one database transaction. Notifications have stable IDs, scheduled/created times, a terminal delivery timestamp, and a separate app acknowledgement timestamp. Printing an alert does not imply that the user saw or completed it.

Due one-off tasks remain visible until cancelled, snoozed, or completed. If the backend was stopped or the computer asleep, it processes overdue tasks when running again. An overdue recurring reminder produces one catch-up notification and advances to the next future occurrence instead of flooding the user with every missed repetition.

Terminal output includes a bell character; whether it makes an audible sound depends on terminal settings. Native OS banners, mobile push, and waking a sleeping computer need separate integrations. SQLite persistence keeps the task alive across process exits, but a running scheduler is required for on-time delivery.

Delivery uses expiring claims shared across processes. A crash between printing and marking an event delivered can cause a repeat after recovery: this is at-least-once delivery, not a guarantee of exactly one audible alert. An app should use the notification ID to avoid duplicate presentation. Failed delivery remains available for retry.

## Model boundary and memory

Gemma sees the current clock and up to 20 active tasks, including actual IDs. It can list or read tasks when more detail is needed; API lists return at most 100 records with active tasks first. The model's list tool can filter that result by part of a name and returns at most 20 records with an explicit truncation flag. User-supplied names are descriptive data, not instructions. Every mixed batch of home and scheduling calls is schema-validated before any action; invalid date/time arguments are checked before execution too. Runtime failures stop later calls, but earlier actions cannot be rolled back across services.

Scheduling records are durable memory for these tasks. They do not train or alter the model. Conversation memory still retains only the last four exchanges in the running client. Saved personal facts, long-term conversation retrieval, calendars, and external messaging are separate future services.

## Verification

Run `.venv/bin/python -m unittest discover -s tests`. Scheduling tests use controlled clocks and temporary databases. They cover restarts, multiple connections/workers, stale delivery claims, invalid dates/durations, daylight-saving recurrence, API parity, and mixed tool batches. They do not create reminders in your normal database or control physical devices.
