import test from "node:test";
import assert from "node:assert/strict";

import {
  AlignmentKind,
  SpanModality,
  alignWordSpansToParentSpan,
  buildSharedSpanModel,
  createSpan,
  projectTimedWordsToSpans,
} from "./shared-span-model.mjs";

test("shared span model serializes and deserializes", () => {
  const model = buildSharedSpanModel({
    spans: [
      createSpan({
        id: "asr-word:1",
        start_ms: 100,
        end_ms: 180,
        modality: SpanModality.Word,
        metadata: { text: "hello" },
      }),
    ],
    alignments: [
      {
        source: "asr-word:1",
        target: "transcript:1",
        kind: AlignmentKind.AlignedTo,
        confidence: 0.92,
      },
    ],
  });
  const roundTrip = JSON.parse(JSON.stringify(model));
  assert.deepEqual(roundTrip, model);
});

test("asr and tts timed words project into alignments", () => {
  const asrWords = projectTimedWordsToSpans({
    words: [{ id: 1, text: "hello", timing: { start_ms: 100, end_ms: 180 } }],
    idPrefix: "asr-word",
    stream: "asr_timed_word_stream",
    turn: 1,
  });
  const ttsWords = projectTimedWordsToSpans({
    words: [{ id: 5, text: "hello", timing: { start_ms: 420, end_ms: 520 } }],
    idPrefix: "tts-word",
    stream: "tts_timed_word_stream_revision",
    turn: 1,
  });
  const transcriptSpan = createSpan({
    id: "transcript:1",
    start_ms: 90,
    end_ms: 210,
    modality: SpanModality.Transcript,
  });
  const playbackSpan = createSpan({
    id: "playback:1",
    start_ms: 400,
    end_ms: 560,
    modality: SpanModality.Playback,
  });

  const asrAlignments = alignWordSpansToParentSpan(asrWords, transcriptSpan, AlignmentKind.AlignedTo, 0.9);
  const ttsAlignments = alignWordSpansToParentSpan(ttsWords, playbackSpan, AlignmentKind.PlayedAs, 0.9);

  assert.equal(asrAlignments.length, 1);
  assert.equal(asrAlignments[0].kind, AlignmentKind.AlignedTo);
  assert.equal(ttsAlignments.length, 1);
  assert.equal(ttsAlignments[0].kind, AlignmentKind.PlayedAs);
});
