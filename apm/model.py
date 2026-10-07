"""Native Gemma audio + tool calling through Transformers."""
import time
import sys
import re
from copy import deepcopy
from .terminal import Status

def progress(message):
    print(f"[APM] {message}", file=sys.stderr, flush=True)
from .tools import TOOLS

from .prompts import SYSTEM

class Gemma:
    def __init__(self, model_id, device="auto"):
        import torch
        from transformers import AutoProcessor, Gemma4ForConditionalGeneration
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.history = []
        self.pending = None
        self.tools = deepcopy(TOOLS)
        self.system = SYSTEM
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        with Status(f"Loading processor ({device})"):
            self.processor = AutoProcessor.from_pretrained(model_id)
        options = {}
        if device == "mps":
            from transformers import MetalConfig
            options["quantization_config"] = MetalConfig(bits=4, group_size=64)
        with Status("Loading and quantizing E2B" if device == "mps" else "Loading E2B"):
            self.model = Gemma4ForConditionalGeneration.from_pretrained(
                model_id, dtype=dtype, device_map=device,
                attn_implementation="sdpa", **options).eval()
        progress(f"Model ready; footprint={self.model.get_memory_footprint()/2**30:.2f} GiB")
        if not hasattr(self.processor, "parse_response"):
            raise RuntimeError("This Transformers processor lacks parse_response; upgrade Transformers.")

    def configure_home(self, context):
        from .prompts import home_system
        self.tools = deepcopy(context["tools"])
        self.system = home_system(context)

    def reset(self):
        self.history.clear()
        self.pending = None

    def commit(self, response, results):
        if self.pending is None:
            return
        assistant = {"role": "assistant"}
        if response.get("tool_calls"):
            assistant["tool_calls"] = response["tool_calls"]
            assistant["tool_responses"] = [
                {"name": result["tool"], "response": result} for result in results]
        else:
            assistant["content"] = response.get("content") or ""
        self.history.extend([self.pending, assistant])
        self.history = self.history[-8:]  # Four complete exchanges, bounded memory.
        self.pending = None

    def predict(self, text=None, audio=None, on_text=None):
        import torch
        content = [{"type": "text", "text": text}] if text else [
            {"type": "audio", "audio": audio}]
        self.pending = {"role": "user", "content": content}
        messages = [{"role": "system", "content": self.system}] + self.history + [self.pending]
        started = time.perf_counter()

        inputs = self.processor.apply_chat_template(
            messages, tools=self.tools, tokenize=True, return_dict=True,
            return_tensors="pt", add_generation_prompt=True,
            enable_thinking=False).to(self.device)
        # Keep token IDs integral while matching audio features to model dtype.
        for key, value in inputs.items():
            if torch.is_floating_point(value):
                inputs[key] = value.to(self.model.dtype)
        streamer = ReplyStreamer(self.processor, on_text) if on_text else None
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=192, do_sample=False, streamer=streamer)

        generated = output[0][inputs["input_ids"].shape[-1]:]
        raw = self.processor.decode(generated, skip_special_tokens=False)
        # Never dispatch a potentially truncated tool request.
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else (eos or [])
        if len(generated) >= 192 and int(generated[-1]) not in eos:
            raise ValueError("Model output reached token limit; no action executed")
        parsed = self.processor.parse_response(raw, prefix=self.processor.decode(inputs["input_ids"][0], skip_special_tokens=False), tools=self.tools)
        return parsed, {"seconds": round(time.perf_counter()-started, 3),
                        "generated_tokens": len(generated), "device": self.device}


class ReplyStreamer:
    """Stream visible prose only; never expose native tool syntax as a reply."""
    def __init__(self, processor, callback):
        self.processor, self.callback = processor, callback
        self.prompt = True
        self.ids = []
        self.sent = ""

    def put(self, value):
        if self.prompt:
            self.prompt = False
            return
        self.ids.extend(value.reshape(-1).tolist())
        raw = self.processor.decode(self.ids, skip_special_tokens=False)
        visible = visible_reply(raw)
        if visible.startswith(self.sent):
            new = visible[len(self.sent):]
            if new:
                self.callback(new)
                self.sent = visible

    def end(self):
        pass


def visible_reply(raw):
    # Stop before any tool request, including incomplete special-token prefixes.
    raw = raw.split("<|tool_call>", 1)[0]
    if "<|channel>" in raw and "analysis" in raw.split("<|channel>", 1)[1].split("<", 1)[0]:
        return ""
    raw = re.sub(r"<[^>]*>", "", raw)
    raw = re.sub(r"<[^>]*$", "", raw)
    return raw.replace("\ufffd", "")
