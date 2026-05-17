#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${1:-out/irl-duplex-smoke}"
PAUSE_MS="${TTS_VAD_PAUSE_MS:-250}"
LISTEN_MS="${TTS_VAD_LISTEN_MS:-700}"

LIVE_JSONL="$OUT_DIR/irl-duplex-live.jsonl"
VIEWER_JSON="$OUT_DIR/irl-duplex-live.viewer.json"
REFERENCE_JSONL="$OUT_DIR/irl-duplex-overlap-reference.jsonl"

mkdir -p "$OUT_DIR"

echo "[1/3] Running IRL live duplex smoke test (Ctrl-C to stop)..."
cargo run --no-default-features --features "audio-cpal asr-whisper llm-llama-cpp tts-piper" -- \
  dev continue \
  --jsonl "$LIVE_JSONL" \
  --tts-vad-pause-ms "$PAUSE_MS" \
  --tts-vad-listen-ms "$LISTEN_MS"

echo "[2/3] Exporting viewer payload..."
cargo run -- dev trace-viewer-export "$LIVE_JSONL" "$VIEWER_JSON"

echo "[3/3] Writing deterministic overlap reference trace..."
cargo run --no-default-features --features "audio-cpal asr-whisper llm-llama-cpp tts-piper" -- \
  dev continue \
  --duplex-trace-scenario overlap-yield \
  --jsonl "$REFERENCE_JSONL" \
  --tts-vad-pause-ms "$PAUSE_MS" \
  --tts-vad-listen-ms "$LISTEN_MS"

echo "Done."
echo "  live trace:      $LIVE_JSONL"
echo "  viewer payload:  $VIEWER_JSON"
echo "  overlap ref:     $REFERENCE_JSONL"
echo "Next:"
echo "  cargo run -- web --trace \"$LIVE_JSONL\""
