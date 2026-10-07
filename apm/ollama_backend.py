"""Local Ollama inference. Python retains control of validation and execution."""
import base64
from copy import deepcopy
import io
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .prompts import SYSTEM
from .tools import TOOLS


class OllamaError(RuntimeError):
    pass


class Transport:
    def __init__(self, host="http://127.0.0.1:11434", timeout=180):
        if not host.startswith(("http://", "https://")):
            raise ValueError("Ollama host must start with http:// or https://")
        self.host = host.rstrip("/")
        self.timeout = timeout

    def stream(self, path, payload):
        request = Request(self.host + path, data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                for line in response:
                    if line.strip():
                        item = json.loads(line)
                        if item.get("error"):
                            raise OllamaError(item["error"])
                        yield item
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise OllamaError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, ConnectionError) as exc:
            raise OllamaError(f"Cannot reach Ollama at {self.host}: {exc}. Open Ollama or run ollama serve.") from exc
        except (ValueError, UnicodeError) as exc:
            raise OllamaError("Ollama returned an invalid response") from exc

    def request(self, path, payload):
        items = list(self.stream(path, payload))
        if len(items) != 1:
            raise OllamaError("Expected a single Ollama response")
        return items[0]


def encode_audio(audio):
    """Pack the existing mono 16 kHz waveform as a native audio attachment."""
    import numpy as np
    import soundfile as sf
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or not 0 < len(samples) <= 30 * 16000:
        raise ValueError("Audio must be mono, 16 kHz, and at most 30 seconds")
    if not np.isfinite(samples).all():
        raise ValueError("Audio samples must be finite")
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(samples, -1, 1), 16000, format="WAV", subtype="PCM_16")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class OllamaBackend:
    def __init__(self, model_id="gemma4:e2b", host="http://127.0.0.1:11434",
                 keep_alive="10m", context=4096, transport=None):
        self.model_id = model_id
        self.transport = transport or Transport(host)
        self.keep_alive = keep_alive
        self.context = context
        self.turns = []
        self.pending = None
        self.capabilities = set()
        self.tools = deepcopy(TOOLS)
        self.system = SYSTEM

    def configure_home(self, context):
        from .prompts import home_system
        self.tools = deepcopy(context["tools"])
        self.system = home_system(context)

    def prepare(self):
        info = self.transport.request("/api/show", {"model": self.model_id})
        self.capabilities = set(info.get("capabilities", []))
        if "tools" not in self.capabilities:
            raise OllamaError(f"{self.model_id} does not advertise tool calling. Use gemma4:e2b.")
        # Execute one token so 'Ready' follows actual inference, not just loading.
        reply = self.transport.request("/api/chat", {
            "model": self.model_id, "messages": [{"role": "user", "content": "Hello"}],
            "stream": False, "think": False, "keep_alive": self.keep_alive,
            "options": {"num_ctx": self.context, "num_predict": 1, "temperature": 0}})
        if not reply.get("done"):
            raise OllamaError("Ollama warm-up did not finish")

    def reset(self):
        self.turns.clear()
        self.pending = None

    def predict(self, text=None, audio=None, on_text=None):
        started = time.perf_counter()
        self.pending = None
        if audio is None and (not isinstance(text, str) or not text.strip()):
            raise ValueError("Provide non-empty text or an audio clip")
        user = {"role": "user", "content": text or "Follow the spoken request in this audio."}
        if audio is not None:
            if "audio" not in self.capabilities:
                raise OllamaError("This Ollama model/server does not advertise native audio support. Update Ollama or use --backend transformers; no separate transcription fallback is used.")
            # Ollama 0.34.1 overloads 'images' for native multimedia attachments,
            # including WAV. This passes the audio to Gemma's audio encoder.
            user["images"] = [encode_audio(audio)]
        self.pending = user
        messages = [{"role": "system", "content": self.system}]
        messages.extend(message for turn in self.turns for message in turn)
        messages.append(user)
        response = {"role": "assistant", "content": ""}
        calls = []
        final = None
        first_text = None
        try:
            for item in self.transport.stream("/api/chat", {
                "model": self.model_id, "messages": messages, "tools": self.tools,
                "stream": True, "think": False, "keep_alive": self.keep_alive,
                "options": {"num_ctx": self.context, "num_predict": 256, "temperature": 0}}):
                message = item.get("message", {})
                content = message.get("content", "")
                if not isinstance(content, str) or not isinstance(message.get("tool_calls", []), list):
                    raise OllamaError("Invalid assistant message from Ollama")
                if content:
                    response["content"] += content
                    if first_text is None:
                        first_text = time.perf_counter() - started
                    if on_text:
                        on_text(content)
                calls.extend(message.get("tool_calls", []))
                if item.get("done"):
                    final = item
                    break
            if final is None:
                raise OllamaError("Ollama stream ended before completion; no action executed")
            if final.get("done_reason") == "length":
                raise OllamaError("Model reached its output limit; no action executed")
            if calls:
                # The home controller validates ALL
                # function names and arguments before dispatching any action.
                response["tool_calls"] = calls
            if not calls and not response["content"].strip():
                raise OllamaError("Ollama returned no answer or tool call")
        except BaseException:
            self.pending = None
            raise
        timing = {"seconds": round(time.perf_counter() - started, 3),
                  "first_text_seconds": round(first_text, 3) if first_text is not None else None,
                  "generated_tokens": final.get("eval_count"), "backend": "ollama",
                  "model": self.model_id}
        for name in ("load_duration", "prompt_eval_duration", "eval_duration"):
            if isinstance(final.get(name), (int, float)):
                timing[name.replace("duration", "seconds")] = round(final[name] / 1e9, 3)
        return response, timing

    def commit(self, response, results):
        if self.pending is None:
            return
        turn = [self.pending, deepcopy(response)]
        for result in results:
            turn.append({"role": "tool", "tool_name": result["tool"],
                         "content": json.dumps(result)})
        self.turns.append(turn)
        self.turns = self.turns[-4:]
        self.pending = None
