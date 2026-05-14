# Listenbury

Listenbury is a single-binary, low-latency PETE implementation focused on real-time embodied conversation: hearing, turn-taking, local inference, speech planning, and speaking.

Part of [Project PETE](https://dancxjo.github.io/project-pete.html): the Pseudoconscious Experiment in Thought and Emotions. Listenbury explores low-latency local conversational flow where listening, inference, planning, and speaking can overlap.

## Current status

Listenbury is an active prototype with working pipeline components and CLI demos:

- Hearing/turn-taking demos (`demo-vad`, `fake-turn`)
- Local LLM turn demo (`llama-turn`, feature-gated)
- Whisper ASR synthetic transcription demo (`transcribe-synthetic`, feature-gated)
- Piper TTS demo (`piper-say`, feature-gated)
- End-to-end WAV round trip (ASR -> LLM -> TTS) (`round-trip-wav`, feature-gated)
- Model asset inspection/fetch utilities (`models`, feature-gated)

The repository currently emphasizes local backend integration and CLI-driven validation.

## Requirements

- Rust toolchain (edition 2024; stable toolchain recommended)
- Cargo

On Debian/Ubuntu systems, install the native build dependencies for the
default feature set before running `cargo build` or `cargo run`:

```bash
sudo apt update
sudo apt install -y build-essential cmake clang libclang-dev pkg-config libasound2-dev
```

Depending on enabled features, additional runtime/system dependencies are needed:

- `audio-cpal` (enabled by default): Linux builds require ALSA development files (`alsa.pc`, from `libasound2-dev` on Debian/Ubuntu) and `pkg-config`
- `llm-llama-cpp`: local GGUF model file(s)
- `asr-whisper`: local Whisper `.bin` model file(s)
- `tts-piper`: Piper executable and Piper `.onnx` voice model (and optional `.onnx.json` config)
- `model-download`: network access for model downloads

## Build and install dependencies

### 1) Build with defaults (full local stack)

```bash
cargo build
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

### 2) Build a minimal profile (avoids audio backend/system ALSA dependency)

```bash
cargo build --no-default-features --features tts-piper
```

### 3) Build selected local AI pipeline features without defaults

```bash
cargo build --no-default-features --features "asr-whisper llm-llama-cpp tts-piper model-download"
```

## Usage

Run commands with Cargo:

```bash
cargo run -- <command> [args...]
```

CLI commands:

```text
listenbury fake-turn "hello there"
listenbury demo-vad
listenbury llama-turn <model.gguf> "prompt"
listenbury transcribe-synthetic <model.bin>
listenbury piper-say <piper-bin> <voice.onnx> "text"
listenbury round-trip-wav <input.wav> [--whisper-model <model.bin>] [--llm-model <model.gguf>] [--piper-bin <piper>] [--piper-voice <voice.onnx>]
listenbury models <fetch|status|path>
```

### Command notes

- `fake-turn`: uses mock token streaming and speech planning
- `demo-vad`: emits VAD/breath-grouping events from synthetic amplitudes
- `llama-turn`: streams text tokens from a local llama.cpp model
- `transcribe-synthetic`: pushes synthetic audio into Whisper recognizer
- `piper-say`: writes synthesized WAV to `out/listenbury-piper-test.wav`
- `round-trip-wav`:
  - input must be mono, 16kHz PCM WAV
  - writes output WAV to `out/listenbury-round-trip.wav`

## Models and paths

If built with `model-download`, use:

```bash
cargo run -- models path
cargo run -- models status
cargo run -- models fetch
```

Default model assets fetched by `models fetch`:

- `ggml-tiny.en.bin` (Whisper)
- `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` (Llama)
- `en_US-lessac-medium.onnx` + `en_US-lessac-medium.onnx.json` (Piper)

### Environment variables

- `LISTENBURY_HOME`: base directory for model asset management
- `LISTENBURY_WHISPER_MODEL`: path override for round-trip Whisper model
- `LISTENBURY_LLM_MODEL`: path override for round-trip llama.cpp model
- `LISTENBURY_PIPER_BIN`: path override for round-trip Piper executable
- `LISTENBURY_PIPER_VOICE`: path override for round-trip Piper voice model

`round-trip-wav` model resolution order is: explicit CLI flag -> environment variable -> first matching file discovered under `./models`.

## Validation

Useful local checks:

```bash
cargo test --no-default-features --features tts-piper
```

If your environment has all system dependencies installed, you can also run broader checks/tests with default features.
