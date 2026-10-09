"""One validated tool boundary for the conversational assistant's services.

Scheduling and device state belong to the services. This layer exposes the same
operations to Gemma that the app API exposes to a person using buttons.
"""
from copy import deepcopy
import sqlite3

from jsonschema import validate, ValidationError

from .backends.base import BackendError
from .music import MusicService
from .toolsets.music import MUSIC_TOOLS


def _tool(name, description, properties, required=(), **constraints):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": list(required),
                           "additionalProperties": False, **constraints}}}


_NAME = {"type": "string", "minLength": 1, "maxLength": 120}
_SECONDS = {"type": "number", "exclusiveMinimum": 0, "maximum": 365 * 86400}
_ZONE = {"type": "string", "minLength": 1, "maxLength": 120}
_TASK_ID = {"type": "string", "minLength": 1, "maxLength": 64}
TASK_TOOLS = [
    _tool("get_clock", "Read current date and time; optionally choose an IANA timezone.",
          {"timezone": _ZONE}),
    _tool("create_timer", "Start a named countdown. Convert the requested duration to seconds.",
          {"name": _NAME, "duration_seconds": _SECONDS}, ["name", "duration_seconds"]),
    _tool("create_reminder", "Save a reminder at an exact future ISO datetime with UTC offset. Use the user's IANA timezone.",
          {"name": _NAME, "due_at": {"type": "string", "minLength": 10, "maxLength": 64},
           "timezone": _ZONE, "repeat": {"enum": ["daily", "weekly"]}}, ["name", "due_at"]),
    _tool("list_scheduled_tasks", "Read up to 20 saved tasks; optionally match part of a name to find IDs or disambiguate. A truncated result is not a complete inventory.",
          {"kind": {"enum": ["timer", "reminder"]},
           "name": _NAME,
           "status": {"enum": ["scheduled", "paused", "due", "completed", "cancelled"]}}),
    _tool("get_scheduled_task", "Read current status and remaining time for a saved task by its actual ID.",
          {"task_id": _TASK_ID}, ["task_id"]),
    _tool("cancel_scheduled_task", "Cancel or delete a timer/reminder at the user's request and record it as cancelled.",
          {"task_id": _TASK_ID}, ["task_id"]),
    _tool("complete_scheduled_task", "Mark a task completed only when the user reports doing or finishing it. Ends its recurrence.",
          {"task_id": _TASK_ID}, ["task_id"]),
    _tool("manage_scheduled_task", "Modify an existing timer/reminder by ID. Pause/resume/extend apply to timers, never music playback. For music use pause_music or resume_music. Extend adds seconds; snooze restarts from now.",
          {"task_id": _TASK_ID, "action": {"enum": ["pause", "resume", "extend", "snooze"]},
           "seconds": _SECONDS}, ["task_id", "action"],
          allOf=[{"if": {"properties": {"action": {"enum": ["extend", "snooze"]}}},
                  "then": {"required": ["seconds"]}, "else": {"not": {"required": ["seconds"]}}}]),
]


class AssistantController:
    def __init__(self, home, tasks, music=None):
        self.home = home
        self.tasks = tasks
        self._owns_music = music is None
        self.music = MusicService() if music is None else music

    def context(self):
        context = self.home.context()
        active = [task for task in self.tasks.list_tasks()
                  if task["status"] not in {"completed", "cancelled"}]
        fields = {"id", "kind", "name", "status", "due_at", "timezone", "repeat", "remaining_seconds"}
        return {**context, "tools": context["tools"] + deepcopy(TASK_TOOLS + MUSIC_TOOLS),
                "music": self.music.status(),
                "clock": self.tasks.clock(),
                "scheduled_tasks": [{key: value for key, value in task.items() if key in fields}
                                    for task in active[:20]],
                "task_context_truncated": len(active) > 20}

    @staticmethod
    def _task_request(name, args):
        if name == "get_clock":
            return "clock", args
        if name in {"create_timer", "create_reminder"}:
            return name, args
        if name == "list_scheduled_tasks":
            return "list_tasks", {key: value for key, value in args.items() if key != "name"}
        if name == "get_scheduled_task":
            return "get_task", {"id": args["task_id"]}
        if name in {"cancel_scheduled_task", "complete_scheduled_task"}:
            return name.split("_", 1)[0] + "_task", {"id": args["task_id"]}
        action, task_id = args["action"], args["task_id"]
        if action in {"pause", "resume", "extend"}:
            return "update_timer", {"id": task_id, "action": action,
                                    **({"seconds": args["seconds"]} if "seconds" in args else {})}
        if action == "snooze":
            return "snooze_task", {"id": task_id, "duration_seconds": args["seconds"]}
        raise ValueError("Unknown task action")

    def execute(self, calls, expected_revision=None):
        context = self.home.context()
        if expected_revision is not None and expected_revision != context.get("revision"):
            raise ValueError("Device registry changed during the request; please try again")
        schemas = {item["function"]["name"]: item["function"]["parameters"]
                   for item in context["tools"] + TASK_TOOLS + MUSIC_TOOLS}
        task_names = {item["function"]["name"] for item in TASK_TOOLS}
        music_operations = {"resolve_music": "resolve", "play_music": "play", "play_music_selection": "select",
                            "pause_music": "pause", "resume_music": "resume"}
        if not isinstance(calls, list) or len(calls) > 8:
            raise ValueError("Expected at most eight tool calls")
        checked = []
        # Validate all schemas and date/time arguments before *any* side effect,
        # including a batch that mixes physical devices and scheduled tasks.
        for call in calls:
            try:
                fn = call["function"]
                name, args = fn["name"], fn["arguments"]
                schema = schemas[name]
            except (KeyError, TypeError):
                raise ValueError("Unknown or malformed tool call") from None
            try:
                validate(args, schema)
            except ValidationError:
                raise ValueError("Invalid tool arguments; no action executed") from None
            request = None
            if name in task_names:
                request = self._task_request(name, args)
                self.tasks.validate_request(*request)
            elif name in music_operations:
                request = (music_operations[name], args)
                self.music.validate_request(*request)
            checked.append((call, name, args, request))
        results, failed = [], False
        for call, name, args, request in checked:
            if failed:
                results.append({"tool": name, "ok": False,
                                "error": "Skipped after an earlier operation failed or could not be confirmed"})
                continue
            try:
                if request is None:
                    result = self.home.execute([call], expected_revision=context.get("revision"))[0]
                elif name in music_operations:
                    operation, kwargs = request
                    data = getattr(self.music, operation)(**kwargs)
                    result = {"tool": name, "ok": data["status"] not in {"failed", "unknown", "not_configured", "unavailable"}, "data": data}
                    if not result["ok"]:
                        result["error"] = data.get("message", "The music request could not be completed")
                else:
                    operation, kwargs = request
                    kwargs = dict(kwargs)
                    # Use positional IDs so the service's internal parameter name
                    # isn't part of the app/tool protocol.
                    method = getattr(self.tasks, operation)
                    data = method(kwargs.pop("id"), **kwargs) if "id" in kwargs else method(**kwargs)
                    result = {"tool": name, "ok": True, "data": data}
                    if name == "list_scheduled_tasks":
                        source_limited = len(data) >= 100
                        if "name" in args:
                            data = [task for task in data if args["name"].casefold() in task["name"].casefold()]
                        result.update(data=data[:20], truncated=source_limited or len(data) > 20)
                failed = result.get("ok") is False or (result.get("accepted") and result.get("state") == "unknown")
                if name in music_operations and result["data"]["status"] not in {"matched", "playing", "paused", "resumed"}:
                    failed = True
            except KeyError:
                result = {"tool": name, "ok": False, "error": "Task or device not found; refresh the list before retrying"}
                failed = True
            except (ValueError, BackendError) as exc:
                result = {"tool": name, "ok": False, "error": str(exc)}
                failed = True
            except sqlite3.Error:
                result = {"tool": name, "ok": False, "error": "Scheduling storage is unavailable; check the task list before retrying"}
                failed = True
            results.append(result)
        return results

    def snapshot(self):
        return self.home.snapshot()

    def close(self):
        # The process owning the scheduler owns TaskService lifecycle too.
        try:
            self.home.close()
        finally:
            if self._owns_music:
                self.music.close()
