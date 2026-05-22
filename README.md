# PETE Listenbury

**PETE Listenbury** is an experimental Rust system for low-latency, local, spoken interaction with an embodied conversational agent.

It is part of **Project PETE**: the *Pseudoconscious Experiment in Thought and Emotions*. Earlier PETE prototypes explored whether relatively simple language-agent systems could produce coherent emotional and narrative behavior. Listenbury focuses on the main practical failure mode of those systems: latency.

The central research question is:

> How can a conversational agent listen, infer, plan, speak, and monitor itself in overlapping time, rather than waiting for each stage to finish before beginning the next?

Listenbury begins with an operational question: what timing, memory, perception, and expressive coordination are required for an artificial interlocutor to remain socially acceptable in an ongoing spoken exchange?

In this sense, PETE uses “pseudoconscious” seriously but cautiously. The project studies the behaviors adjacent to lived interaction: continuity, emotional tone, turn-taking, self-monitoring, and repair. It treats these as engineering and research problems before treating them as philosophical claims.

## Research Motivation

Most spoken LLM interfaces are architecturally sequential:

1. record user speech,
2. transcribe it,
3. send the transcript to a language model,
4. wait for the full response,
5. synthesize speech,
6. play the audio.

This produces more than technical delay. It produces a social failure.

A conversational agent that pauses too long, interrupts at the wrong time, speaks in full paragraphs when a nod would do, misses opportunities for backchannels, or cannot recover when the user starts talking again is not merely slow. It becomes socially awkward. The user begins to reject it as an interlocutor. Even capable systems can become impractical or annoying when their timing violates ordinary conversational expectations.

Listenbury treats latency as a social and interactional problem, not merely a throughput problem.

The project explores a more overlapping model:

- audio capture and voice activity detection happen continuously,
- speech is segmented into breath groups or conversational units,
- ASR produces text for completed segments,
- local LLM generation can begin quickly,
- speech planning can operate on partial language-model output,
- TTS can begin before a full conversational response is complete,
- playback and expressive output can be scheduled together,
- self-hearing can compare intended speech with actual emitted audio,
- runtime context can be appended at safe boundaries during generation.

The goal is not simply to make an assistant faster. The goal is to study the timing structure of spoken cognition-like behavior: how an agent can become socially tolerable, interruptible, responsive, and expressive in real time.

## Current Status

Listenbury is an active prototype. It currently emphasizes local backend integration, CLI-driven validation, and incremental development of the real-time speech loop.

Working or partially working components include:

- local Whisper-based ASR for WAV transcription,
- live microphone capture through CPAL,
- voice activity detection experiments,
- local llama.cpp-backed LLM prompting and text completion,
- Piper-based speech synthesis,
- WAV input/output utilities,
- an end-to-end WAV reply path: ASR -> LLM -> TTS,
- a live half-duplex listening loop,
- model selection and fetch utilities,
- CUDA feature variants for Whisper and llama.cpp,
- early filler/backchannel planning,
- runtime packet/context update design using append-only updates at safe boundaries,
- optional Qdrant and Neo4j services for cold memory experiments.

The repository is still a research prototype, not a polished application. Some code paths are diagnostic or experimental, and some components are intentionally conservative while the timing model is being tested.

## What “Low Latency” Means Here

Listenbury treats latency as a systems problem, not just a model-speed problem.

There are several distinct latencies:

- **auditory latency**: how quickly speech is detected,
- **segmentation latency**: how quickly a breath group is judged complete,
- **ASR latency**: how quickly speech becomes text,
- **LLM first-token latency**: how quickly the agent begins forming a response,
- **planning latency**: how quickly partial text becomes speakable units,
- **TTS latency**: how quickly text becomes audio,
- **playback latency**: how quickly audio reaches the speaker,
- **social latency**: how long the system feels silent or unresponsive.

A usable embodied agent may need to reduce all of these. In some cases, it may be better to produce a safe backchannel or filler than to wait silently for a complete response.

Listenbury therefore distinguishes between:

- the **hot path**, which must remain fast enough for live interaction,
- and the **cold path**, where slower memory, graph storage, reflection, or summarization can occur asynchronously.

## Architecture

At a high level:

```text
microphone
   ↓
audio capture / resampling
   ↓
VAD and breath-group segmentation
   ↓
ASR
   ↓
conversation/runtime context
   ↓
local LLM
   ↓
speech planner
   ↓
TTS
   ↓
speaker playback
   ↓
self-hearing / monitoring
```

The current implementation is organized around a single Rust binary with feature-gated subsystems.

Major subsystems include:

- **Hearing**: audio capture, WAV handling, resampling, VAD, ASR integration.
- **Inference**: local llama.cpp-backed language model execution.
- **Speech planning**: conversion from LLM output into speakable units.
- **Mouth / playback**: TTS generation and audio output.
- **Runtime context**: append-only events that can update the ongoing conversational state.
- **Model management**: local model selection, paths, and optional downloading.
- **Cold memory**: optional Qdrant/Neo4j support for slower memory experiments.

## Runtime Context and Safe Boundaries

Listenbury does not currently attempt arbitrary prompt mutation or KV-cache surgery in the middle of token generation.

Instead, runtime updates are modeled as append-only packets. The system can apply those updates at safe boundaries, especially between committed speech units.

The initial rule is:

- append new facts rather than pretending old prompt text changed,
- continue, cancel, or restart generation at explicit boundaries,
- avoid token-level mutation in the first version,
- preserve intelligibility and timing over theoretical cleverness.

## Reflexes, Fillers, and Backchannels

Human conversation often includes short responses that are not full semantic turns: acknowledgments, hesitation markers, backchannels, and floor-holding sounds.

Listenbury includes an early conservative planner for this kind of behavior, and it is enabled by default.

When enabled, the current design prefers silence unless a short, cached response is useful and safe. Repetition guards prevent the same filler from being repeated too often.

This area is intentionally cautious. Poorly timed filler speech is worse than silence.

## Syllable-Level Prosody Planning

A longer-term goal is to move below sentence-level or clause-level TTS planning into syllable-level expressive timing.

Human speech is not emitted as plain text. It is shaped continuously by breath, emphasis, affect, uncertainty, interruption, and social intention. A system that waits for complete text and then synthesizes a neutral waveform loses access to much of this control.

Listenbury therefore aims to explore a pipeline where speech is represented as a stream of small expressive units:

```text
LLM text stream
   ↓
phonemicization
   ↓
syllable / stress / phrase planning
   ↓
prosody planner
   ↓
TTS acoustic generation
   ↓
playback with self-monitoring
```

In this model, emotional state would not merely choose an emoji or a voice preset. It would influence timing, pitch contour, intensity, pause length, and emphasis dynamically as speech is being prepared.

A future version should be able to represent a planned utterance as something like:

```text
word -> phonemes -> syllables -> stress groups -> breath groups -> expressive audio
```

This would allow the system to:

- begin speaking before a whole paragraph is complete,
- adjust prosody as emotional state changes,
- align facial expression or gesture to syllable timing,
- soften or abandon speech when interrupted,
- repair an utterance when later LLM output changes the intended phrase,
- compare intended syllable timing with actual emitted audio.

This requires taking ownership of phonemicization and TTS model execution rather than treating TTS as a black-box command that consumes finished text. Piper can serve as an initial model source, but deeper control likely requires running the ONNX model directly and building Listenbury’s own text-to-phoneme, syllabification, duration, and prosody layers around it.

## Self-Hearing

One of the long-term goals is for Listenbury to distinguish between:

- intended speech,
- synthesized target PCM,
- audio actually emitted by the speaker,
- and audio perceived through the microphone.

This matters because an embodied system’s speech is not merely text output. It is an event in the world. Playback may fail, clip, crackle, overlap with user speech, or be interrupted.

A future version of Listenbury should be able to say, in effect:

> I intended to say this, but what reached the room was different.

That comparison is useful for interruption handling, self-suppression, conversational repair, and embodied accountability.

## Project Scope

Listenbury is currently focused on local, real-time spoken interaction. It is not currently trying to be:

- a general chatbot framework,
- a cloud assistant,
- a production speech API,
- a complete cognitive architecture,
- or a polished end-user application.

It is a research workbench for the timing problem underneath believable spoken agents.

## Relation to Project PETE

PETE stands for **Pseudoconscious Experiment in Thought and Emotions**.

Earlier PETE systems explored narrative continuity, emotional state, memory, and social presence. They demonstrated that emotionally coherent behavior can emerge from relatively simple language-agent loops when supported by memory and narrative framing.

Their weakness was latency.

Listenbury is the next step: preserving the expressive and narrative ambitions of PETE while rebuilding the interaction loop around immediacy.

In that sense:

- **Daringsby** explored emotional/narrative coherence.
- **psycheOS** explored memory, distillation, and cognitive layering.
- **Listenbury** explores real-time embodied conversation.

## Requirements

- Rust toolchain, edition 2024
- Cargo
- [`just`](https://github.com/casey/just), recommended
- Linux audio dependencies for CPAL/ALSA
- local model files for enabled ASR, LLM, and TTS backends

On Debian/Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential cmake clang libclang-dev pkg-config libasound2-dev
cargo install just
```

Depending on features, you may also need:

- Whisper model files,
- GGUF LLM model files,
- Piper executable and voice models,
- NVIDIA CUDA toolkit for CUDA builds.

## Building

Default local build:

```bash
just build
```

The default build enables the portable local stack, including audio capture,
resampling, WebRTC VAD, ASR, local LLM, Piper TTS, Riper scaffolding,
webcam support, and model download support. Riper is Listenbury's own custom
translation of Piper: it runs Piper-compatible voice models through our local
phoneme, prosody, and ONNX execution path instead of shelling out to the Piper
process. CUDA and Metal accelerator variants remain explicit opt-ins because
they depend on the target machine and platform.

CUDA build:

```bash
just build-cuda
```

Minimal build example:

```bash
cargo build --no-default-features --features tts-piper
```

Selected local AI stack:

```bash
cargo build --no-default-features --features "asr-whisper llm-llama-cpp tts-piper model-download"
```

When switching between CPU and CUDA builds, a clean rebuild may avoid stale native artifacts:

```bash
just clean
just build-cuda
```

## Usage

List available commands:

```bash
just run -- --help
```

Common commands:

```bash
just run ask "Can you hear me?"
just run complete "The system listens because"
just run transcribe input.wav
just run say "Hello from Pete Listenbury."
just run reply input.wav
just run listen
```

CUDA variants:

```bash
just cuda ask "Can you hear me?"
just cuda listen
```

## WaveDeck live browser timeline

The repository now includes WaveDeck, a live timeline viewer for `listenbury transcribe --web`
and `listenbury listen --web`:

```text
web/browser-transcript-player/
```

Host it directly from Listenbury:

```bash
cargo run -- web
```

Then open:

```text
http://127.0.0.1:8787/
```

Useful options:

```bash
cargo run -- web --host 127.0.0.1 --port 8787
cargo run -- web --open
```

The hosted server exposes stable routes for the viewer and event APIs:

```text
/                     live WaveDeck viewer for the active --web session
/wavedeck             same live WaveDeck viewer alias as /
/replay               trace replay / fixture tooling (offline, deterministic)
/screenplay           narrative screenplay view
/assets/...           bundled static assets
/fixtures/...         bundled fixture files (demo payload, sample traces)
/api/demo-payload     bundled demo JSON fixture (alias for /fixtures/demo.json)
/api/payload          JSON from --payload (when provided)
/api/trace            JSONL from --trace (when provided)
/api/trace-session    structured recorded session JSON from --trace (when provided)
/api/trace-viewer-payload  converted viewer payload from --trace (when provided)
/api/live-session-audio.wav  live microphone session audio from transcribe --web
/api/session-audio/{artifact_id}  full-session audio artifact from structured --trace
/api/live-events      SSE stream of live LiveTraceEvent JSON (requires a --web session)
/healthz              simple health check
```

The main WaveDeck page (`/`) is the live session view. `/wavedeck` is the same
view under an explicit name. Both routes connect directly to `/api/live-events`
on load and use `/api/live-session-audio.wav` as the real microphone audio
source when running `listenbury transcribe --web`; they do not switch to fixture
payloads automatically.

The **replay page** (`/replay`) is dedicated to offline trace exploration.
It loads JSONL trace files, structured recorded sessions, or the trace attached
to `listenbury web --trace ...`, and replays events deterministically with full
playback controls:

- pause/resume
- per-event stepping
- seek to arbitrary position
- configurable playback speed (0.1× – 16×)

Open the hosted WaveDeck page and it will immediately connect to `/api/live-events`
for the active `--web` session. With `listenbury transcribe --web`, the viewer
also draws the real microphone session waveform from `/api/live-session-audio.wav`.
The viewer renders multiple streams vertically with a shared ruler, highlights
the active word during playback, and lets you click chips/ruler positions to
seek the shared audio timeline. Drag across the ruler or an empty lane region to
mark a time range; releasing the mouse zooms directly into that range.

When `listenbury web` is started with `--trace <path>`, `/replay` and
`/screenplay` load the recorded trace/session instead of waiting for live SSE
events. Event/marker selections can also expose and play saved clip references
through `audio_ref` when present in payload data.

Words without `timing` are displayed with fallback layout timing so they are
visibly distinct from measured/aligned timings.

Structured trace-session directories can include a durable full-session audio
artifact recorded in `metadata.json` under `audio_artifacts`. WaveDeck serves
that artifact through `/api/session-audio/{artifact_id}` and uses it as the
shared waveform ruler, with optional event/marker `audio_ref` clip playback.

### Fixture files

Sample fixtures are organised under:

```text
examples/browser-transcript-player/fixtures/
```

| File | Description |
|------|-------------|
| `demo.json` | Static viewer payload with three word lanes and example events |
| `live-trace.sample.jsonl` | Sample live trace JSONL (replay-able) |
| `live-trace.sample.viewer.json` | Pre-converted viewer payload from the sample trace |

### Export runtime live trace JSONL into viewer payload JSON

`listenbury listen --jsonl ...` and `listenbury dev continue --jsonl ...`
write runtime traces either as a plain JSONL file (`*.jsonl`) or as a
structured session directory when the output path is not a `.jsonl` file:

```text
session/
  metadata.json
  events.jsonl
  audio/
    session.wav
```

`metadata.json` preserves the session ID, start time, event stream path, and
runtime configuration. For structured sessions, `listenbury listen --jsonl
<session-dir>` also records the full session WAV as a durable audio artifact
with duration, sample rate, channel count, creation time, and session linkage in
`metadata.json`. `dev continue` also emits `asr_timed_word_stream` artifacts
carrying serialized live ASR `TimedWordStream` objects. Convert a JSONL file or a
structured session directory into the browser viewer payload with:

```bash
cargo run -- dev trace-viewer-export out/live-session out/live-trace.viewer.json
```

Then load `out/live-trace.viewer.json` in the replay page (`/replay`).

The bundled sample trace can be regenerated with:

```bash
cargo run -- dev trace-viewer-export \
  examples/browser-transcript-player/fixtures/live-trace.sample.jsonl \
  examples/browser-transcript-player/fixtures/live-trace.sample.viewer.json
```

### Browser transcript player JSON format

The viewer accepts any of the following:

- a single `TimedWordStream`
- an array of `TimedWordStream`s
- an object with a `streams` array

The bundled demo uses the richer object form:

```json
{
  "title": "Listenbury WaveDeck Demo",
  "audio": {
    "url": "../../welcome.wav",
    "duration_ms": 2081
  },
  "streams": [
    {
      "label": "User transcript",
      "stream": {
        "id": 1,
        "source": "RecordedAudio",
        "words": [
          {
            "id": 1,
            "text": "welcome",
            "timing": { "start_ms": 0, "end_ms": 720 },
            "timing_confidence": 0.97,
            "commitment": "Final",
            "boundary_source": "Whisper",
            "lexical_span": { "start": 0, "end": 7 },
            "audio_ref": null
          }
        ]
      }
    }
  ]
}
```

Each `stream` payload is the shared Rust `TimedWordStream` model serialized
through `serde`, so the viewer consumes the same substrate used by ASR and TTS
export paths.

The bundled demo includes a provisional auditory-scene lane to illustrate
multi-lane/event-like debugging context. This is a temporary shape; future
non-word span/event lanes should evolve toward dedicated schemas.

Model utilities:

```bash
cargo run -- models list
cargo run -- models status
cargo run -- models path
cargo run -- models use voice Amy
cargo run -- models use llm gpt-oss
cargo run -- models fetch
```

## Important Commands

### `ask`

Runs a local LLM prompt using a Pete Listenbury interaction wrapper.

```bash
listenbury ask "Can you hear me?"
```

### `complete`

Runs raw local text completion.

```bash
listenbury complete "The next thing to say is"
```

### `transcribe`

Transcribes microphone audio or a WAV file using Whisper.

```bash
listenbury transcribe input.wav
```

### `say`

Synthesizes and plays speech using Piper.

```bash
listenbury say "Testing one two three."
```

### `riper-compare`

Compares external Piper process synthesis with Riper.

```bash
listenbury riper-compare "Testing one two three."
```

### `reply`

Runs a WAV through ASR, generates a reply, and synthesizes the response.

```bash
listenbury reply input.wav
```

### `listen`

Runs the current live half-duplex loop.

```bash
listenbury listen
```

The current loop listens for speech, transcribes a completed segment, generates a response, synthesizes it, plays it, and returns to listening. It is intentionally simple and serves as the baseline for future duplex and interruption-aware behavior.

Pass `--duplex` to run the continuous duplex development pipeline through the
public listen command. It uses the same mic, ASR, LLM, TTS, VAD, and JSONL model
flags as `listen`, but routes execution through `listenbury dev continue`.

```bash
listenbury listen --duplex
```

#### Live browser viewer

Pass `--web` to start the embedded WaveDeck browser viewer alongside the listen loop. Events (mic activity, ASR results, LLM turns, speaker activity) are streamed live to the browser via Server-Sent Events.

```bash
listenbury listen --web
listenbury listen --web --duplex
# or via justfile:
just run listen --web
```

The server prints the viewer URL on startup:

```text
Listenbury web viewer available at http://127.0.0.1:8787/
```

Open that URL in a browser to see a live event timeline that updates in real time as events happen on the mic and speakers.

Note: live SSE currently streams future events only. If you connect late, earlier events are not replayed yet.

Optional web flags:

```bash
listenbury listen --web --web-host 0.0.0.0 --web-port 9000
```

WaveDeck is live-only and shows a pulsing **Live** indicator. Playback and timeline zoom controls stay available so you can drag-zoom into a time range as the DAW-style timeline updates. A new **Graph** mode (Cytoscape.js) provides span/alignment inspection with revision lineage chains, node-driven timeline focus, and filters for modality, turn, time window, commitment, and revision presence.

Event kinds are grouped into timeline lanes:

| Lane | Events |
|---|---|
| **Mic** | `capture_started`, `speech_started`, `breath_group_opened/closed` |
| **ASR** | `asr_started/finished`, `transcript` |
| **LLM** | `llm_generation_started`, `first_llm_token`, `first_safe_speech_unit_emitted` |
| **Speaker** | `playback_started/finished`, `first_tts_audio_frame_available` |

## Configuration

Common environment variables:

```bash
LISTENBURY_HOME=/path/to/listenbury/models
LISTENBURY_WHISPER_MODEL=/path/to/model.bin
LISTENBURY_LLM_MODEL=/path/to/model.gguf
LISTENBURY_PIPER_BIN=/path/to/piper
LISTENBURY_PIPER_VOICE=/path/to/voice.onnx
LISTENBURY_PIPER_BACKEND=process # process|riper|riper-fallback
PETE_LLM=gpt-oss
PETE_VOICE=Amy
```

Model resolution order is generally:

1. explicit CLI flag,
2. environment variable,
3. selected registered model under `LISTENBURY_HOME`,
4. discovered local file under `./models`.

## Optional Cold Memory

Listenbury’s hot path should not depend on Qdrant, Neo4j, Docker, or network services.

Optional cold-memory services can be started for slower memory experiments:

```bash
cp .env.example .env
docker compose up -d qdrant neo4j
```

Stop them with:

```bash
docker compose stop qdrant neo4j
docker compose rm -f qdrant neo4j
```

Cold memory is intended for asynchronous traces, summaries, recall experiments, and analysis. It should not block live speech.

## Development Principles

1. **Keep the hot path hot.**  
   Anything needed for immediate conversational timing should avoid avoidable network calls, heavyweight databases, or blocking reflection.

2. **Prefer measurable timing over vague intelligence.**  
   Latency, first-token time, breath-group completion time, TTS startup time, and playback timing are first-class research data.

3. **Commit speech at boundaries.**  
   The system should not pretend it can freely revise already-spoken audio. Revision must happen before commitment, or through explicit conversational repair.

4. **Separate planning from playback.**  
   Speech text, expressive units, TTS audio, and actual speaker output are related but distinct.

5. **Treat self-hearing as part of embodiment.**  
   The agent should eventually know not only what it intended to say, but what was actually heard.

6. **Keep claims precise.**  
   Listenbury studies pseudoconscious behavior, timing, and embodied interaction through working systems and measurable constraints.

7. **Prosody is part of thought-in-action.**  
   Speech timing, stress, intonation, hesitation, and repair are not decorative output effects. They are part of how a spoken agent participates in conversation.

8. **Avoid socially invalid timing.**  
   A system that is technically correct but conversationally mistimed will be rejected by users. Responsiveness, interruptibility, and graceful silence are core requirements.

## Research Directions

Near-term:

- improve live VAD and breath-group segmentation,
- reduce ASR and TTS startup latency,
- add stronger diagnostics for timing and audio routing,
- separate orchestration code into clearer modules,
- stabilize the speech planner and playback executor,
- improve interruption and fadeout behavior,
- add self-hearing suppression.

Medium-term:

- support overlapping listening and speaking,
- plan TTS from partial LLM output,
- coordinate speech, face, and prosody commands,
- add target-versus-actual PCM comparison,
- support vision as another live input stream,
- persist timing traces for analysis.

Long-term:

- model turn-taking as a continuous control problem,
- support conversational repair when speech diverges from intent,
- build memory systems that operate asynchronously from the hot path,
- develop syllable-level expressive speech planning,
- evaluate whether low-latency embodied timing improves perceived presence, trust, or usability.

## Academic Framing

Listenbury may be relevant to work in:

- human-computer interaction,
- spoken dialogue systems,
- embodied conversational agents,
- social robotics,
- assistive communication,
- low-latency speech interfaces,
- local-first AI systems,
- cognitive architectures,
- and educational technology.

A reasonable academic description would be:

> PETE Listenbury investigates how a local conversational agent can coordinate listening, inference, speech planning, synthesis, playback, prosody, and self-monitoring in overlapping time to reduce the latency, social awkwardness, and practical brittleness of spoken LLM interaction.

## Limitations

Current limitations include:

- live interaction is mostly half-duplex,
- interruption handling is early,
- filler/backchannel behavior is conservative,
- prompt/context updates are append-only and boundary-based,
- model behavior depends heavily on local model choice,
- audio device configuration can be fragile on Linux,
- CUDA support depends on local driver/toolkit compatibility,
- cold memory is optional and not yet central to the real-time loop.

## Citation / Attribution

This is an independent research prototype by Travis D. Reed as part of Project PETE.

If this work informs academic or applied research, please cite the repository and describe it as a prototype for low-latency embodied spoken interaction.

## License

See `LICENSE`.
