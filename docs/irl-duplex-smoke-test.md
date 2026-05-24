# IRL full-duplex smoke test (mouth ↔ self-hearing ↔ overlap routing ↔ duplex control ↔ ASR)

This is an operational, repeatable smoke test for validating real-world duplex behavior (not just synthetic fixtures).

## 1) Recommended setup

- **Room:** quiet room with low HVAC/fan noise.
- **Audio path:** one mic + one speaker (or laptop mic/speaker) in the same room.
- **Placement:** keep speaker close enough for self-hearing, but not so loud it hard-clips the mic.
- **Models/env:** ensure model paths resolve (`LISTENBURY_WHISPER_MODEL`, `LISTENBURY_LLM_MODEL`, `LISTENBURY_PIPER_BIN`, `LISTENBURY_PIPER_VOICE`).

## 2) Run the live IRL smoke test

From repo root:

```bash
mkdir -p out
cargo run --no-default-features --features "audio-cpal asr-whisper llm-llama-cpp tts-piper" -- \
  dev continue \
  --jsonl out/irl-duplex-live.jsonl \
  --tts-vad-pause-ms 250 \
  --tts-vad-listen-ms 700
```

## 3) How to provoke overlap

1. Let Pete start speaking.
2. During Pete speech, do both:
   - **brief interruption** (~100–200 ms) to test non-yield continuation.
   - **sustained interruption** (>250 ms, then keep speaking past ~700 ms) to force pause + clear.
3. Repeat 2–3 times.
4. Stop with `Ctrl-C`.

## 4) Expected behavior (overlap/yield)

- Brief overlap should usually be treated as transient overlap (continue speaking).
- Sustained overlap should trigger floor yielding:
  - playback pause first,
  - then queued synthetic audio clear if overlap persists through listen window.
- ASR transcripts should continue appearing for externally-routed speech.

Useful stderr indicators in `dev continue`:

- `[ear] routing=MixedSelfAndExternal ...`
- `[ear] routing=ExternalSpeechCandidate ...`
- `[dev continue] heard: ...`
- `[dev continue] speaking: ...`

## 5) Inspect traces

### A. Live trace JSONL

Quick checks:

```bash
jq -r '.kind' out/irl-duplex-live.jsonl | sort | uniq -c
jq -c 'select(.kind=="transcript") | {turn, text}' out/irl-duplex-live.jsonl
jq -c 'select(.kind=="asr_timed_word_stream") | {turn, has_artifact:(.artifact != null)}' out/irl-duplex-live.jsonl
```

Convert for viewer:

```bash
cargo run -- dev trace-viewer-export out/irl-duplex-live.jsonl out/irl-duplex-live.viewer.json
cargo run -- web --trace out/irl-duplex-live.jsonl
```

Open `http://127.0.0.1:8787/` and inspect `/api/trace-viewer-payload`.

### B. Deterministic overlap control reference trace

Run synthetic overlap control trace (for expected controller decisions reference):

```bash
cargo run --no-default-features --features "audio-cpal asr-whisper llm-llama-cpp tts-piper" -- \
  dev continue \
  --duplex-trace-scenario overlap-yield \
  --jsonl out/irl-duplex-overlap-reference.jsonl \
  --tts-vad-pause-ms 250 \
  --tts-vad-listen-ms 700

jq -c 'select(.kind=="controller_decision") | .details' out/irl-duplex-overlap-reference.jsonl
```

Expected reference decisions include:

- `short_overlap_blip` with decision `continue`
- `sustained_overlap` with `yield_pause`
- `sustained_overlap` with `yield_clear_queue`

## 6) Common troubleshooting

- `no default input device available`: select/enable a system default mic.
- model load errors (`failed to load Whisper model...`, llama/piper init failures): verify paths and files.
- little/no overlap detection:
  - increase speaker volume slightly,
  - move mic closer to speaker + talker,
  - reduce room noise.
- trace file empty/missing: confirm `--jsonl` path and that process ran long enough to emit events.

---

If you want one-command execution, use `scripts/irl-duplex-smoke-test.sh`.
