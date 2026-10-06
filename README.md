# APM — local home voice agent

APM runs **Gemma 4 E2B through a local Ollama server** by default. Python owns the chat interface, bounded conversation history, native audio capture, tool validation, and simulated home controls. No MCP or separate speech-recognition model is required. Real Homebridge devices are not connected yet.

## Start on this Mac

Open the Ollama app. The `gemma4:e2b` model is already installed here. Then:

```sh
cd ~/APM
.venv/bin/apm
```

Type normally at `>>>`. `/voice` starts wake-word listening, `/state` shows simulated devices, `/clear` resets conversation without unloading the model, and `/quit` exits. The spinner shows startup/response progress; answers stream and display total response time.

Each process starts with kitchen lights on and garage open. The model retains the last four complete exchanges, including tool results. The Python device state persists for that process. Ollama manages inference and may reuse cached prompt computations. `--keep-alive 10m` keeps the model available for ten minutes after the last request, including after APM exits; restarting APM need not reload an already resident model. After the timeout, it loads from disk again.

```sh
.venv/bin/apm --text "Turn off the kitchen lights"
.venv/bin/apm --audio /absolute/path/command.wav
.venv/bin/apm --record 5
.venv/bin/apm --text "Hello" --interactive
.venv/bin/apm --text "Hello" --debug
.venv/bin/apm --demo
```

`--debug` prints model responses, validated execution results, and timing fields. `--demo` is scripted and does not invoke a model. Startup performs a one-token warm-up before saying ready. Startup time is separate from response time.

## Install on another machine

Install a compatible local Ollama server, then pull the exact model:

```sh
ollama pull gemma4:e2b
uv venv .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/apm
```

The default endpoint is `http://127.0.0.1:11434`. `--ollama-host` can override it explicitly. There is no automatic cloud fallback, model download, or server installation. Connection and missing-model errors are reported to the terminal. APM talks to the server API rather than the shell's Ollama executable.

The same Python/Ollama API design applies to Jetson Orin Nano Super, but install a compatible JetPack/Ollama CUDA build and verify GPU use, memory, audio, and latency on that hardware. Mac timings are not Jetson benchmarks. Ollama chooses the inference device; `--device` only affects the optional Transformers backend.

## Native audio

Verified locally with Ollama **server 0.34.1** and `gemma4:e2b` (GGUF Q4_K_M), which advertises both `audio` and `tools`. APM sends a mono 16 kHz PCM16 WAV attachment through the native `/api/chat` multimedia field (`images`, which also carries audio in this server version). The audio goes to E2B's audio encoder, alongside the tool definitions. It is not converted to a transcript by another model.

Clips must be at most 30 seconds; files are downmixed/resampled. Recording may require macOS microphone permission. Unsupported audio capability and overlong clips fail explicitly. Synthetic recorded speech is covered by a live check; real microphone accuracy remains to be evaluated. Gemma produces text; voice mode speaks its answer using local operating-system speech synthesis.

## Hands-free “Hey Gemma” mode

```sh
.venv/bin/apm --voice
```

Say **“Hey Gemma”**, wait for **Listening · speak now** (and the terminal bell, if enabled), then say **“Turn off the kitchen lights.”** APM ends the recording after 0.8 seconds of silence, sends the audio directly to E2B, executes validated functions, and reads the result aloud. Waiting for the listening cue is recommended for this first version; a very short command spoken before wake detection finishes can be missed. A 1.6-second pre-roll preserves the beginning of captured speech.

`Ctrl-C` stops microphone capture and returns to text chat. `/voice` starts it again with the same model and conversation. `/quit` exits from text mode. Status indicators show waiting, listening, processing, and speaking; the displayed response time measures model processing and tool execution, excluding capture and spoken playback.

```sh
.venv/bin/apm --list-mics
.venv/bin/apm --voice --mic 1
.venv/bin/apm --voice --no-speak
.venv/bin/apm --voice --wake-threshold 0.9 --silence 1.2
```

Choose the input index from your own `--list-mics` output. Allow microphone access for your terminal when macOS prompts. `--mic` also selects the input for `--record`. The default threshold comes from the model metadata. Lower wake thresholds may catch more utterances but increase accidental triggers. A longer silence setting allows longer pauses inside a request.

The CPU runs openWakeWord with a custom, experimental **Hey Gemma** classifier; it does not run the language model continuously. It needs `models/hey_gemma.onnx`, `models/melspectrogram.onnx`, and `models/embedding_model.onnx`. Copy the models directory together with the project when moving machines. `--wake-model /path/model.onnx` can select another classifier, with the two backbone files beside it.

The wake classifier is bootstrapped locally from synthetic macOS voices, with held-out synthetic voices for evaluation. The current classifier still confuses some similar names and did not meet our synthetic false-trigger target. It is a prototype for microphone trials. Synthetic evaluation does **not** establish accuracy with your microphone, accent, TV audio, or room noise. Model provenance and synthetic metrics are stored alongside the model. Training scratch files are under the ignored `work/wake-training/` directory. No microphone recordings were used in training. To reproduce the Mac-only synthetic training:

```sh
uv pip install --python .venv/bin/python -e '.[wake-training]'
.venv/bin/python tools/train_wakeword.py
.venv/bin/python tools/evaluate_wakeword.py --threshold 0.9
```

Training writes a new classifier and metadata to `models/`. The command above reproduces the prototype threshold of 0.9; omitting `--threshold` asks the streaming evaluator to seek a threshold using the validation voice. Always inspect `validation_target_met` and both recall and false-trigger counts; the supplied prototype does not meet the desired calibration target. The exported detector runs without the training tools.

APM holds captured microphone audio in memory and does not write it to disk. Wake detection and audio collection pause during model inference and speech playback, then queued audio is cleared before listening resumes. Interrupting a spoken answer returns to text; voice interruption/barge-in is not implemented. Empty, overlong, or dropped recordings are not sent for tool execution. Each command has a 25-second recording cap; an overlong command is discarded rather than cut off.

Mac spoken replies use `/usr/bin/say`; Linux uses `espeak-ng` (install it on the Jetson or use `--no-speak`). The Python voice loop and ONNX CPU detector are intended to transfer to Jetson, but microphone drivers, native audio inference, ARM64 dependency installation, and latency still require testing on that machine. Linux may also need PortAudio from its package manager for `sounddevice`.

## Optional Transformers backend

The previous implementation remains available for experiments:

```sh
uv pip install --python .venv/bin/python -e '.[transformers]'
.venv/bin/apm --backend transformers
```

It uses `google/gemma-4-E2B-it`, MPS/Metal 4-bit linear layers on Mac, BF16 on CUDA, or FP32 on CPU. That CUDA path is not a memory-optimized Jetson deployment. Its libraries and downloaded weights were not deleted during migration. Default Ollama execution does not import or load PyTorch/Transformers.

## Verification and next work

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover tool validation, incomplete/truncated streams, native audio encoding, history, reusable sessions, audio endpoint behavior, discarded audio, microphone pausing during replies, and error reporting. All calls are validated before simulation changes; model-generated code is never executed. Tool confirmations come from execution results.

Next: real microphone evaluation, then an authenticated Homebridge adapter matched to the actual installation. Device integrations must be local for fully offline home control.

References: [Ollama chat API](https://docs.ollama.com/api/chat), [tool calling](https://docs.ollama.com/capabilities/tool-calling), [v0.34.1 multimedia transport](https://github.com/ollama/ollama/blob/v0.34.1/llm/llama_server.go), [Gemma audio preparation](https://ai.google.dev/gemma/docs/capabilities/audio).
