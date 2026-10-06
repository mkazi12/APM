import argparse
import json
import time
from .terminal import Status
from pathlib import Path
from .tools import SimulatedHome


def describe_results(results):
    labels = {"kitchen_lights": "Kitchen lights", "garage": "Garage door"}
    return " ".join(f"{labels.get(r['device'], r['device'])}: {r['state']} (simulated)." for r in results)


def run_request(model, home, text=None, audio=None, debug=False):
    streamed = []
    started = time.perf_counter()
    with Status("Preparing reply", enabled=not debug) as status:
        def emit(chunk):
            if not chunk:
                return
            if not streamed:
                status.stop()
                print("Assistant: ", end="", flush=True)
            streamed.append(chunk)
            print(chunk, end="", flush=True)
        response, timing = model.predict(text, audio, on_text=None if debug else emit)
        results = home.execute(response.get("tool_calls") or [])
        model.commit(response, results)
    elapsed = time.perf_counter() - started
    if debug:
        timing = dict(timing, response_seconds=round(elapsed, 3))
        print(json.dumps({"response": response, "results": results, "timing": timing}, indent=2))
    else:
        if not streamed:
            print("Assistant: ", end="", flush=True)
        if results:
            if streamed:
                print()
            print(describe_results(results), end="")
        elif not streamed:
            print(response.get("content") or "I couldn't form a response. Please try again.", end="")
        print(f"\n  Responded in {elapsed:.1f}s\n", flush=True)
    return describe_results(results) if results else (response.get("content") or "")


def interactive_session(model, home, debug=False, voice_config=None):
    print("Ready. /voice starts hands-free mode; /clear resets chat; /state shows devices; /quit exits.")
    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if not text:
            continue
        if text in ("/quit", "/bye"):
            return
        if text == "/voice":
            from .voice import voice_session
            try:
                voice_session(model, home, run_request, config=voice_config, debug=debug)
            except Exception as exc:
                print(f"Voice unavailable: {exc}")
            continue
        if text == "/clear":
            model.reset()
            print("Conversation cleared. Model remains loaded.")
            continue
        if text == "/state":
            print(describe_results([{"device": key, "state": value["state"]} for key, value in home.devices.items()]))
            continue
        if text.startswith("/"):
            print("Available commands: /voice, /clear, /state, /quit")
            continue
        try:
            run_request(model, home, text=text, debug=debug)
        except KeyboardInterrupt:
            print("\nExiting interrupted inference.")
            return
        except Exception as exc:
            print(f"\nRequest failed: {type(exc).__name__}: {exc}")


def create_backend(args):
    if args.backend == "ollama":
        from .ollama_backend import OllamaBackend
        model = OllamaBackend(args.model or "gemma4:e2b", host=args.ollama_host,
                              keep_alive=args.keep_alive)
        with Status("Connecting to Ollama and warming E2B"):
            model.prepare()
        print(f"Model ready · {model.model_id} via Ollama · native audio: "
              f"{'yes' if 'audio' in model.capabilities else 'unavailable'}")
        return model
    if not args.debug:
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_HUB_DISABLE_XET_PROGRESS_BARS"] = "1"
        from huggingface_hub.utils import disable_progress_bars
        from transformers.utils import logging
        disable_progress_bars()
        logging.disable_progress_bar()
        logging.set_verbosity_error()
    from .model import Gemma
    return Gemma(args.model or "google/gemma-4-E2B-it", args.device)


def main():
    parser = argparse.ArgumentParser(description="Local native-audio Gemma agent; simulated devices only")
    parser.add_argument("--backend", choices=["ollama", "transformers"], default="ollama")
    parser.add_argument("--model", help="Defaults to gemma4:e2b (Ollama) or google/gemma-4-E2B-it (Transformers)")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--keep-alive", default="10m", help="How long Ollama retains an idle model")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--interactive", action="store_true", help="Keep the model loaded after an initial request")
    parser.add_argument("--debug", action="store_true", help="Show raw JSON, timing, and library progress")
    parser.add_argument("--wake-model", type=Path, help="Custom openWakeWord ONNX model; default models/hey_gemma.onnx")
    parser.add_argument("--wake-threshold", type=float, help="Override the threshold recommended by the wake model")
    parser.add_argument("--silence", type=float, default=0.8, help="Seconds of silence ending a voice command")
    parser.add_argument("--mic", type=int, help="Input device index from --list-mics")
    parser.add_argument("--no-speak", action="store_true", help="Display voice answers without spoken playback")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--voice", action="store_true", help="Listen continuously for Hey Gemma")
    group.add_argument("--list-mics", action="store_true", help="List audio devices and exit")
    group.add_argument("--text")
    group.add_argument("--audio", type=Path)
    group.add_argument("--record", type=float, metavar="SECONDS")
    group.add_argument("--demo", action="store_true", help="Exercise tool execution without loading a model")
    args = parser.parse_args()
    from .voice import VoiceConfig, DEFAULT_WAKE_MODEL
    voice_config = VoiceConfig(wake_model=args.wake_model or DEFAULT_WAKE_MODEL,
                               threshold=args.wake_threshold, silence_seconds=args.silence,
                               microphone=args.mic, speak=not args.no_speak)
    home = SimulatedHome()
    try:
        voice_config.validate()
        if args.list_mics:
            import sounddevice as sd
            print(sd.query_devices())
            return
        if args.demo:
            calls = [{"function": {"name": "set_lights", "arguments": {"device": "kitchen_lights", "on": False}}}]
            print(json.dumps({"mode": "scripted demo; no LLM", "results": home.execute(calls)}, indent=2))
            return
        audio = None
        if args.audio:
            import librosa
            audio, _ = librosa.load(args.audio, sr=16000, mono=True)
        if args.record is not None:
            if not 0 < args.record <= 30:
                parser.error("Recording duration must be between 0 and 30 seconds")
            import sounddevice as sd
            input("Press Enter to record, then speak: ")
            audio = sd.rec(round(args.record*16000), samplerate=16000, channels=1, dtype="float32", device=args.mic)
            sd.wait()
            audio = audio[:, 0]
        if audio is not None and not 0 < len(audio) <= 30*16000:
            parser.error("Audio must contain between 0 and 30 seconds")
        model = create_backend(args)
        if args.voice:
            from .voice import voice_session
            try:
                voice_session(model, home, run_request, config=voice_config, debug=args.debug)
            except Exception as exc:
                print(f"Voice unavailable: {exc}")
        if args.text is not None or audio is not None:
            run_request(model, home, args.text, audio, args.debug)
        if args.interactive or (args.text is None and audio is None):
            interactive_session(model, home, args.debug, voice_config)
    except KeyboardInterrupt:
        parser.exit(130, "\nAPM interrupted.\n")
    except Exception as exc:
        parser.exit(1, f"APM failed: {type(exc).__name__}: {exc}\n")

if __name__ == "__main__":
    main()
