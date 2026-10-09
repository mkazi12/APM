# APM — local home voice agent

APM runs **Gemma 4 E2B through a local Ollama server** by default. Python owns the chat interface, bounded conversation history, native audio capture, tool validation, and home controls. Simulation is the default; an optional device registry routes commands to Homebridge and Matter Server adapters. No MCP or separate speech-recognition model is required.

## Start on this Mac

Open the Ollama app. The `gemma4:e2b` model is already installed here. Then:

```sh
cd ~/APM
.venv/bin/apm
```

Type normally at `>>>`. `/voice` starts wake-word listening, `/state` reads device states, `/tasks` lists saved timers and reminders, `/clear` resets conversation without unloading the model, and `/quit` exits. The spinner shows startup/response progress; answers stream and display total response time, with model and action timings.

In simulation, each process starts with kitchen lights on and garage open; simulated state lasts for that process. The model retains the last four complete exchanges, including tool results. Ollama manages inference and may reuse cached prompt computations. `--keep-alive 10m` keeps the model available for ten minutes after the last request, including after APM exits; restarting APM need not reload an already resident model. After the timeout, it loads from disk again.

```sh
.venv/bin/apm --text "Turn off the kitchen lights"
.venv/bin/apm --audio /absolute/path/command.wav
.venv/bin/apm --record 5
.venv/bin/apm --text "Hello" --interactive
.venv/bin/apm --text "Hello" --debug
.venv/bin/apm --demo
```

`--debug` prints model responses, validated execution results, and timing fields. `--demo` is scripted and does not invoke a model. Startup warms the configured instructions and tools before saying ready; audio modes also initialize the audio encoder with generated silence. Warm-up responses are discarded without executing tools. Startup time is separate from response time.

For Ollama, stable instructions and tool definitions precede conversation history, while the fresh clock, tasks, music status, and device catalog are supplied just before each new request. Changing the clock therefore does not invalidate the entire cached prompt. Current state is refreshed every request and is not saved as stale conversation history; original voice attachments and tool results remain in the bounded history.

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

## Homebridge, Matter, and the app backend

The optional local API stores device names, rooms, aliases, capabilities, and adapter bindings. Gemma receives this catalog and tools restricted to registered devices on each request. Homebridge supports On/Off lights, switches, outlets, and garage doors; Matter Server supports commissioned On/Off endpoints. Pairing and a mobile interface are future work.

```sh
uv pip install --python .venv/bin/python -e '.[home]'
.venv/bin/apm-server --registry work/home.json --init
```

`--init` creates a simulation registry and refuses to overwrite an existing file. Omit it on subsequent starts. Open `http://127.0.0.1:8765/docs` to try the API. To use the same registry in the voice assistant, add `--home-config work/home.json` to your usual APM command. The server owns registry writes; voice clients reload them before each request. Simulated states are per process, while physical device states are read from their controllers.

See [the backend setup guide](docs/home-backend.md) for registration examples, real connection prerequisites, and API endpoints. `apm --home-config PATH --check-home` validates a registry and prints the model's catalog without connecting to devices or loading Gemma. The adapters have mocked protocol tests; physical hardware still needs an integration test in your home.

## Clocks, timers, and reminders

The assistant and API share a persistent SQLite task service. Gemma interprets a request and calls a validated function; the backend stores the task, calculates its state, and delivers due notifications independently of model inference. The model receives the current clock, timezone, and a bounded catalog of active task IDs on every request. Ollama uses an 8,192-token context budget to accommodate the additional tools and task catalog.

```sh
.venv/bin/apm --text 'Set a pasta timer for ten minutes' --interactive
.venv/bin/apm --text 'Remind me tomorrow at 8 AM to take out the bins' --interactive
```

Timers support pause, resume, adding time, snooze, cancellation, and completion. Reminders support one-off, daily, and weekly schedules. The default database is `work/assistant.sqlite3`; `--database` overrides it in both programs. `--timezone America/Los_Angeles` explicitly sets the local IANA timezone; otherwise it is detected. `/clear` clears chat history, not saved tasks.

Keep the independent backend running if notifications should continue after the voice client exits:

```sh
.venv/bin/apm-server --registry work/home.json --database work/assistant.sqlite3
```

Then use your normal voice command with `--no-scheduler` to let that server handle alerts. Both programs must use the same database path. Without a separate server, APM runs its own scheduler while the process is open. Alerts currently print a terminal message and bell and remain available through `/v1/notifications`; the computer must be awake and at least one scheduler running. Missed deadlines are recovered on restart. Native mobile push, OS wakeups, general personal-memory tools, and calendar account integration are not included yet.

See [the scheduling guide](docs/scheduling.md) for API examples, recurrence behavior, delivery semantics, and local testing.

## Music requests

Music has a provider-neutral resolver and playback interface. Gemma extracts the title and artist, the resolver checks actual catalog candidates, and the player receives a verified catalog ID. Small spelling differences such as “Bruce Springstein” are handled by title/artist matching. Play requests use the top eligible match, with ties following the provider's result order, so compilation editions do not trigger an album-selection prompt. Requested artists and versions still constrain the match; live-only, unknown, or mixed versions can require a choice. Search-only requests preserve alternatives for browsing. An accepted command is reported separately from confirmed playback.

Without a saved connection, the default has **no music provider connected**. A full-catalog Apple Music adapter is available through a local MusicKit browser player; it needs Apple developer configuration and user authorization. See [connect Apple Music](docs/apple-music.md). Other providers are not implemented. Automated tests use synthetic IDs and fake playback; real playback must be checked with your authorized account.

Say “pause the music” or “continue playing the music” to control the loaded song. These use dedicated music tools without timer/task IDs. Resume keeps the current queue and playback position; if nothing is loaded, APM asks you to play a song first.

Tool definitions live in `apm/toolsets/music.py`, with resolution and provider contracts in `apm/music.py`. The app API exposes `/v1/music/status`, `/v1/music/resolve`, `/v1/music/play`, `/v1/music/pause`, `/v1/music/resume`, and selection playback. See [music resolution and adapter setup](docs/music.md).

## Native audio

Verified locally with Ollama **server 0.34.1** and `gemma4:e2b` (GGUF Q4_K_M), which advertises both `audio` and `tools`. APM sends a mono 16 kHz PCM16 WAV attachment through the native `/api/chat` multimedia field (`images`, which also carries audio in this server version). The audio goes to E2B's audio encoder, alongside the tool definitions. It is not converted to a transcript by another model.

Clips must be at most 30 seconds; files are downmixed/resampled. Recording may require macOS microphone permission. Unsupported audio capability and overlong clips fail explicitly. Synthetic recorded speech is covered by a live check; real microphone accuracy remains to be evaluated. Gemma produces text; voice mode speaks its answer using local operating-system speech synthesis.

## Hands-free “Hey Gemma” mode

```sh
.venv/bin/apm --voice
```

Say **“Hey Gemma”**, wait for **Listening · speak now** (and the terminal bell, if enabled), then say **“Turn off the kitchen lights.”** APM ends the recording after 0.8 seconds of silence, sends the audio directly to E2B, executes validated functions, and reads the result aloud. Waiting for the listening cue is recommended for this first version; a very short command spoken before wake detection finishes can be missed. A 1.6-second pre-roll preserves the beginning of captured speech.

When connected music is playing, detecting the wake phrase pauses it before command capture. Wait for the music to stop and the listening cue before speaking: APM clears the music pre-roll and waits for a quiet boundary so lyrics or a partial command are not submitted. An unconfirmed pause discards the turn, stops the microphone, and returns to text after one message. Pause the player, reload and reconnect its page, then type `/voice` to resume; chat history stays intact. Music stays paused until another playback request. This does not add speaker identification or acoustic echo cancellation; loud music can still mask the wake phrase or cause false wakes.

After Gemma asks a question or requests a music selection, **Listening for your reply · no wake phrase needed** opens an eight-second window to begin answering. Questions can continue the conversation; a completed action or silence returns to wake listening. Recognition uses question wording and tool results without another model call. Use `--follow-up-timeout SECONDS` to adjust the window (up to 15), or `--follow-up-timeout 0` to require the wake phrase every time. The microphone remains paused while Gemma speaks; answer after the listening cue.

`Ctrl-C` stops microphone capture and returns to text chat. `/voice` starts it again with the same model and conversation. `/quit` exits from text mode. Status indicators show waiting, listening, processing, and speaking; the displayed response time measures model processing and tool execution, excluding capture and spoken playback.

```sh
.venv/bin/apm --list-mics
.venv/bin/apm --voice --mic 1
.venv/bin/apm --voice --no-speak
.venv/bin/apm --voice --wake-threshold 0.9 --silence 1.2
```

Choose the input index from your own `--list-mics` output. Allow microphone access for your terminal when macOS prompts. `--mic` also selects the input for `--record`. The default threshold comes from the model metadata. Lower wake thresholds may catch more utterances but increase accidental triggers. A longer silence setting allows longer pauses inside a request.

Command capture uses a speech detector separate from wake recognition. It adapts while waiting and requires 0.24 seconds of consecutive speech to begin a command, or 0.16 seconds for short follow-up answers such as “yes” and “no,” with a less aggressive setting for quiet voices. If the wake triggers but you see `No command heard`, add `--debug`: the `Voice capture` line reports recognized speech duration and microphone levels without saving any audio. This message means command capture timed out before sending audio to Gemma.

The CPU runs openWakeWord with a custom, experimental **Hey Gemma** classifier; it does not run the language model continuously. Voice startup prints the selected model and threshold. Selection uses `--wake-model` first, then an explicitly saved `work/voice.json` selection, then the bundled `models/hey_gemma.onnx` baseline. The selected classifier needs `melspectrogram.onnx` and `embedding_model.onnx` beside it. Copy that directory together with the project when moving machines. A missing saved model produces an error rather than silently reverting to the baseline.

The wake classifier is bootstrapped locally from synthetic macOS voices, with held-out synthetic voices for evaluation. The current classifier still confuses some similar names and did not meet our synthetic false-trigger target. It is a prototype for microphone trials. Synthetic evaluation does **not** establish accuracy with your microphone, accent, TV audio, or room noise. Model provenance and synthetic metrics are stored alongside the model. Training scratch files are under the ignored `work/wake-training/` directory. No microphone recordings were used in training. To reproduce the Mac-only synthetic training:

```sh
uv pip install --python .venv/bin/python -e '.[wake-training]'
.venv/bin/python tools/train_wakeword.py
.venv/bin/python tools/evaluate_wakeword.py --threshold 0.9
```

Training writes a new classifier and metadata to `models/`. The command above reproduces the prototype threshold of 0.9; omitting `--threshold` asks the streaming evaluator to seek a threshold using the validation voice. Always inspect `validation_target_met` and both recall and false-trigger counts; the supplied prototype does not meet the desired calibration target. The exported detector runs without the training tools.

Normal APM voice mode holds captured microphone audio in memory and does not write it to disk. Wake detection and audio collection pause during model inference and speech playback, then queued audio is cleared before listening resumes. Interrupting a spoken answer returns to text; voice interruption/barge-in is not implemented. Empty, overlong, or dropped recordings are not sent for tool execution. Each command has a 25-second recording cap; an overlong command is discarded rather than cut off.

Mac spoken replies use `/usr/bin/say`; Linux uses `espeak-ng` (install it on the Jetson or use `--no-speak`). The Python voice loop and ONNX CPU detector are intended to transfer to Jetson, but microphone drivers, native audio inference, ARM64 dependency installation, and latency still require testing on that machine. Linux may also need PortAudio from its package manager for `sounddevice`.

## Train “Hey Gemma” with your voice

When wake detection misses your voice, first measure the existing classifier on your microphone. A voice-specific verifier only filters detections and cannot recover phrases missed by the base model. The local workflow below compares threshold calibration with retraining the classifier over the same frozen audio backbone. It never replaces the installed model automatically.

The recorder explicitly saves the takes you keep as 16 kHz mono PCM16 WAV files in the ignored `work/wake-personal/` directory. Nothing is uploaded. Run it in a terminal; it waits for Enter before each take, then gives a visual speaking cue. Playback, retry, and quit are available after every take. Use the same microphone as voice mode and say **only “Hey Gemma”** for positive takes, with silence before and after. Vary your usual speed, volume, and distance. Stop APM voice mode while recording.

```sh
.venv/bin/python tools/record_wakeword.py --list-mics
.venv/bin/python tools/record_wakeword.py --split validation
.venv/bin/python tools/evaluate_personal_wakeword.py
```

Add `--mic INDEX` to the recorder to select an input from the list; omitting it uses the default. Validation defaults to 15 wake phrases, 10 other phrases, and 30 seconds of background. The report compares the current `0.9` threshold with one selected from your validation recordings. If calibration alone catches your voice without accidental activations, try its threshold using `apm --voice --wake-threshold VALUE`, then evaluate on fresh recordings.

If the baseline still misses phrases or confuses other speech, collect a separate training session and fit a candidate:

```sh
.venv/bin/python tools/record_wakeword.py --split train
.venv/bin/python tools/train_personal_wakeword.py
```

Training defaults to 30 positive takes, 20 confusing/everyday negative phrases, and 30 seconds of background. These counts are a small personal experiment, **not an established requirement or a reliability guarantee**. Change them with `--positives`, `--negatives`, and `--background-seconds`, or collect additional sessions. Keep each session in a single split, and collect validation and test in different sittings. Do not copy recordings between splits. Train on normal, isolated wake phrases; the trainer rejects positive takes without a usable speech boundary and asks you to re-record them.

For a first candidate when you have recorded only the training batch, run `.venv/bin/python tools/train_personal_wakeword.py --holdout-from-training`. This reserves 20% of the original positive and negative takes, plus all background audio, for provisional threshold calibration. Originals stay in place; no take or augmented copy appears on both sides. These same-session results are weaker evidence than a separate validation session, and fresh speech is still needed afterward. Once separate validation recordings exist, omit this flag.

If normal speech works but a whispered wake phrase fails, add actual whispered recordings using the same recorder: `.venv/bin/python tools/record_wakeword.py --split train --session whisper-01 --positives 20 --negatives 10`. Whisper every speech prompt, including the other phrases, from your usual position. This adds a new session and keeps the existing normal-voice recordings. Choose a new session name for another batch. The recorder flags very quiet takes for playback and review instead of rejecting them solely for low volume; empty, damaged, clipped, or effectively silent recordings are still rejected. Raw recording levels are preserved. Reducing the volume of normal speech does not recreate the acoustic characteristics of a whisper. These example counts are a starting experiment, not a guarantee; compare the next candidate on fresh whispered and normal speech.

The trainer needs the `wake-training` extra and the original `work/wake-training/train-features-v1.npz` cache, which already exists on this Mac. On another installation, reproduce the synthetic bootstrap **before personal training** using the earlier instructions (that bootstrap command overwrites `models/`). Personal training mixes only synthetic training features and your training sessions, modestly augments positives with training-room noise, and selects a threshold using validation sessions. It prints both baseline and candidate validation results. Test sessions are excluded from fitting and threshold selection.

Each run writes a new directory under `work/wake-personal/candidates/`, containing the candidate `hey_gemma.onnx`, matching backbones, and metadata. A reported `validation_target_met: false` means that even the best-effort threshold did not achieve the requested tradeoff; do not treat it as calibrated. Automatic speech boundaries and small datasets remain experimental.

After choosing a candidate and threshold, record a fresh test session and compare it with the installed detector. Replace `/absolute/path/to/candidate/hey_gemma.onnx` with the path printed by training:

```sh
.venv/bin/python tools/record_wakeword.py --split test
.venv/bin/python tools/evaluate_personal_wakeword.py --split test --candidate /absolute/path/to/candidate/hey_gemma.onnx
.venv/bin/apm --voice --wake-model /absolute/path/to/candidate/hey_gemma.onnx
```

To keep a chosen candidate as the default for plain `apm --voice` and `/voice`, save this local, ignored `work/voice.json` file (relative model paths resolve from `work/`):

```json
{"wake_model": "/absolute/path/to/candidate/hey_gemma.onnx"}
```

Training does not change this selection. Remove `work/voice.json` to restore the bundled baseline, or override it for one run with `--wake-model`. The model's metadata supplies its threshold unless `--wake-threshold` is passed.

The test evaluator uses fixed thresholds and never tunes on test recordings. To compare a baseline threshold already chosen on validation, add `--baseline-threshold VALUE`; do not repeatedly adjust it against the same test session. Evaluation reports wake recall, false triggers on negative speech clips, and background activation events per measured hour separately. Background events require two high frames, then two seconds below threshold before counting another. Thirty seconds of quiet background is only a quick check: add much longer, representative background recordings before drawing conclusions about daily reliability. Connected wake-and-command speech still needs separate live testing; this recording workflow trains isolated wake phrases.

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
