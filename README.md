# Listenbury

Listenbury is a single-binary, low-latency PETE implementation focused on real-time embodied conversation: hearing, turn-taking, local inference, speech planning, and speaking.

Part of [Project PETE](https://dancxjo.github.io/project-pete.html): the Pseudoconscious Experiment in Thought and Emotions. Listenbury explores low-latency local conversational flow where listening, inference, planning, and speaking can overlap.

## Current status

Listenbury is an active prototype with working pipeline components and CLI demos:

- Live listening loop (`listen`, feature-gated)
- Local LLM prompt (`ask`, feature-gated)
- Whisper ASR WAV transcription (`transcribe`, feature-gated)
- Piper TTS demo (`say`, feature-gated)
- End-to-end WAV reply (ASR -> LLM -> TTS) (`reply`, feature-gated)
- Model asset inspection/fetch utilities (`models`, feature-gated)

The repository currently emphasizes local backend integration and CLI-driven validation.

## Low-latency reflex planning (design)

Listenbury now includes a controller/filler-planner skeleton for low-latency social reflexes while the main LLM is still generating.

- `FillerPlanner` produces `FillerDecision` from `FillerContext`.
- Initial path is intentionally conservative and silence-first.
- Cached backchannels are selected through `BackchannelId` and converted into safe `SpeechUnit::Backchannel` values for `SpeechPlanner` / mouth playback.
- Repetition guards are included by default:
  - avoid repeating the same filler within 60 seconds,
  - avoid more than one filler per user turn unless explicitly configured.

### Runtime context updates: append-only at safe boundaries

The controller uses append-only `RuntimePacket` events and applies updates at safe boundaries (especially between committed `SpeechUnit`s), instead of pretending earlier prompt text was mutated in place.

Initial stance:

- append new runtime facts to ongoing context,
- decide continue/cancel/restart at safe boundaries,
- avoid token-level prompt mutation/KV-cache surgery for v1.

## Requirements

- Rust toolchain (edition 2024; stable toolchain recommended)
- Cargo
- [`just`](https://github.com/casey/just), optional but recommended for project shortcuts

On Debian/Ubuntu systems, install the native build dependencies for the
default feature set before running `cargo build` or `cargo run`:

```bash
sudo apt update
sudo apt install -y build-essential cmake clang libclang-dev pkg-config libasound2-dev
```

Install `just` with Cargo:

```bash
cargo install just
```

Or use your system package manager if it has a recent package, for example:

```bash
sudo apt install -y just
```

Depending on enabled features, additional runtime/system dependencies are needed:

- `audio-cpal` (enabled by default): Linux builds require ALSA development files (`alsa.pc`, from `libasound2-dev` on Debian/Ubuntu) and `pkg-config`
- `llm-llama-cpp`: local GGUF model file(s)
- `llm-llama-cpp-cuda`: NVIDIA CUDA toolkit for llama.cpp GPU offload
- `asr-whisper`: local Whisper `.bin` model file(s)
- `asr-whisper-cuda`: NVIDIA CUDA toolkit for whisper.cpp GPU execution
- `tts-piper`: Piper executable and Piper `.onnx` voice model (and optional `.onnx.json` config)
- `model-download`: network access for model downloads

## Build and install dependencies

### 1) Build with defaults (full local stack)

```bash
just build
```

If this fails with `Package 'alsa' not found` or `alsa.pc` missing, install:

```bash
sudo apt install -y pkg-config libasound2-dev
```

This enables default features:

- `audio-cpal`
- `resample-rubato`
- `vad-silero`
- `vad-webrtc`
- `llm-llama-cpp`
- `asr-whisper`
- `tts-piper`
- `vision-webcam`

The default feature set is CPU-only for Whisper and llama.cpp. If Whisper logs
`whisper_backend_init_gpu: no GPU found` while `nvidia-smi` can see your GPU,
rebuild with `asr-whisper-cuda`; the plain `asr-whisper` feature does not compile
the CUDA backend.

### 2) Build a minimal profile (avoids audio backend/system ALSA dependency)

```bash
cargo build --no-default-features --features tts-piper
```

### 3) Build selected local AI pipeline features without defaults

```bash
cargo build --no-default-features --features "asr-whisper llm-llama-cpp tts-piper model-download"
```

For NVIDIA GPU builds, use the CUDA feature variants:

```bash
just build-cuda
```

The CUDA recipes default `CUDA_LIBRARY_PATH` to `/usr/lib/x86_64-linux-gnu`,
which is where Debian/Ubuntu CUDA packages commonly install `libcudart_static.a`,
`libcublas_static.a`, and related libraries. The recipes also pass that path to
Rust via `RUSTFLAGS` so native CUDA archives can be found during the final link.
If your CUDA toolkit is elsewhere, set it explicitly before running Cargo or
`just`:

```bash
CUDA_LIBRARY_PATH=/path/to/cuda/lib64 just build-cuda
```

When switching an existing checkout from CPU-only to CUDA, a clean rebuild avoids
reusing a previously compiled CPU-only native backend:

```bash
just clean
just build-cuda
```

Useful `just` recipes:

```bash
just              # list available recipes
just run -- --help   # cargo run -- --help
just cuda -- --help  # cargo run with asr-whisper-cuda and llm-llama-cpp-cuda
just check
just check-cuda
just test
```

## Usage

Run commands with `just`:

```bash
just run <command> [args...]
just cuda <command> [args...]
```

CLI commands:

```text
listenbury transcribe [input.wav] [--seconds 30] [--until-ctrl-c] [--whisper-model <model.bin>]
listenbury say [--piper-bin <piper>] [--piper-voice <voice.onnx>] [--output-wav <out.wav>] "text"
listenbury listen [--seconds <n>] [--model-profile tiny] [--no-backchannels] [--whisper-model <model.bin>] [--llm-model <model.gguf>] [--piper-bin <piper>] [--piper-voice <voice.onnx>]
listenbury ask [--llm-model <model.gguf>] [--mode <spoken|chat|inner|raw>] [--max-tokens <n>] "prompt"
listenbury complete [--llm-model <model.gguf>] [--max-tokens <n>] "prompt"
listenbury reply <input.wav> [--whisper-model <model.bin>] [--llm-model <model.gguf>] [--piper-bin <piper>] [--piper-voice <voice.onnx>]
listenbury models <list|use|fetch|status|path>
```

### Command notes

- `transcribe`: transcribes microphone audio by default; pass a WAV path to transcribe a file with Whisper
- `say`: plays synthesized speech through the default speaker; `--output-wav` writes a WAV instead
- `listen`:
  - half-duplex loop: listen for a completed breath group, transcribe, generate, synthesize, play, return to listening
  - no interruption while Pete is speaking (capture pauses during playback)
  - cached filler/backchannel speech can start after about 800ms of Pete thinking with no safe speech ready
  - listens until Ctrl-C by default; pass `--seconds <n>` for a bounded run
  - `--no-backchannels` skips cached filler and short backchannel-only planner units in spoken responses
- `ask`: wraps your prompt as a short Pete Listenbury spoken/chat turn, with a default 48-token budget
- `complete`: streams raw text completion from a local llama.cpp model
- `reply`:
  - input WAV is mixed/resampled to mono 16kHz before transcription
  - writes output WAV to `out/listenbury-round-trip.wav`
- Diagnostics and milestone experiments are hidden under `listenbury dev ...`.

## Models and paths

If built with `model-download`, use:

```bash
cargo run -- models path
cargo run -- models list
cargo run -- models status
cargo run -- models use voice Amy
cargo run -- models use llm gpt-oss
cargo run -- models fetch
```

`models fetch` downloads the currently selected Whisper, LLM, and Piper voice assets. Use `models fetch <name>` for one selectable bundle, or `models fetch --all` to download every registered asset.

Runtime commands also download the selected registered model on demand if it is missing locally, so the first `listen`, `ask`, `say`, or `reply` after a model switch may wait while the model arrives.

Quick voice and LLM overrides:

```bash
export PETE_VOICE=Amy
export PETE_LLM=gpt-oss
just listen
```

Registered model assets include:

- `ggml-tiny.en.bin` (Whisper)
- `llama-3.2-3b-instruct-q4_k_m.gguf` (Llama 3.2 3B Instruct via llama.cpp)
- `gpt-oss-20b-mxfp4.gguf` (gpt-oss 20B via llama.cpp)
- Piper voices: Ryan, Amy, HFC Male, HFC Female, John, LJSpeech, and LibriTTS

### Environment variables

- `LISTENBURY_HOME`: base directory for model asset management
- `LISTENBURY_WHISPER_MODEL`: path override for round-trip Whisper model
- `LISTENBURY_LLM_MODEL`: path override for round-trip llama.cpp model
- `LISTENBURY_PIPER_BIN`: path override for round-trip Piper executable
- `LISTENBURY_PIPER_VOICE`: path override for round-trip Piper voice model
- `PETE_LLM`: selected registered LLM alias, for example `gpt-oss`
- `PETE_VOICE`: selected registered Piper voice alias, for example `Amy`

`transcribe`, `listen`, `ask`, `say`, and `reply` model resolution order is: explicit CLI flag -> path environment variable -> selected registered model under `LISTENBURY_HOME` -> first matching file discovered under `./models`.

## Validation

Useful local checks:

```bash
cargo test --no-default-features --features tts-piper
```

If your environment has all system dependencies installed, you can also run broader checks/tests with default features.
