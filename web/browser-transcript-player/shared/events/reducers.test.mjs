/**
 * web/browser-transcript-player/shared/events/reducers.test.mjs
 *
 * Unit tests for the shared live-event reducer layer.
 *
 * Run with:
 *   node --test web/browser-transcript-player/shared/events/reducers.test.mjs
 *
 * Covers the acceptance criteria from the issue:
 *   - speculative → committed transitions
 *   - cancellation handling
 *   - transcript revisions
 *   - playback lifecycle transitions
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
  textContent,
  normalizeSemanticText,
  joinWords,
  joinSemanticText,
  normalizedId,
  speechUnitIdFromEvent,
  transcriptCandidateText,
  wordStreamText,
  tokenize,
  deletedTextBetween,
  recordDeletedText,
  MAX_DELETED_SNIPPETS,
  applyTranscriptCandidate,
  applyAsrWordStream,
  applyTtsWordStreamRevision,
  applyLlmTextEvent,
  commitLlmText,
  setLlmProspective,
  finalizeLlmText,
  speechUnitText,
} from "./reducers.mjs";

import {
  isLlmTextEvent,
  isGeneratedSpeechEventKind,
  isProspectiveCommitment,
  isPlayedCommitment,
  LIVE_EVENT_LANE,
  SPAN_PAIRS,
  END_TO_START,
} from "./schema.mjs";

import {
  currentUserText,
  prospectiveTail,
  turnHasUserDialogue,
  turnHasLlmDialogue,
} from "./selectors.mjs";

// ── Test helpers ──────────────────────────────────────────────────────────

function mkTurn(overrides = {}) {
  return {
    flags: { cancelled: false, revised: false, prospective: false },
    userFinal: "",
    userStable: "",
    userUnstable: "",
    userCandidateText: "",
    userWords: [],
    userDeleted: [],
    llmFinal: "",
    llmProspective: "",
    llmFragments: [],
    llmWords: [],
    llmDeleted: [],
    speechUnitsById: new Map(),
    ...overrides,
  };
}

function mkEvent(kind, elapsed_ms, extra = {}) {
  return { kind, elapsed_ms, ...extra };
}

// ── Text normalisation ────────────────────────────────────────────────────

test("textContent normalises whitespace", () => {
  assert.equal(textContent("  hello   world  "), "hello world");
  assert.equal(textContent(null), "");
  assert.equal(textContent(undefined), "");
  assert.equal(textContent(42), "42");
});

test("textContent removes space before punctuation", () => {
  assert.equal(textContent("hello , world"), "hello, world");
  assert.equal(textContent("end ."), "end.");
});

test("normalizeSemanticText returns null for empty/non-string", () => {
  assert.equal(normalizeSemanticText(""), null);
  assert.equal(normalizeSemanticText("   "), null);
  assert.equal(normalizeSemanticText(null), null);
  assert.equal(normalizeSemanticText(42), null);
  assert.equal(normalizeSemanticText("hello"), "hello");
});

test("joinWords joins non-empty words with spaces", () => {
  assert.equal(joinWords(["hello", "world"]), "hello world");
  assert.equal(joinWords(["end", "."]), "end.");
  assert.equal(joinWords([]), "");
  assert.equal(joinWords([null, "", "foo"]), "foo");
});

test("joinSemanticText joins parts normalising whitespace", () => {
  assert.equal(joinSemanticText("hello", "world"), "hello world");
  assert.equal(joinSemanticText("  foo  ", "  bar  "), "foo bar");
  assert.equal(joinSemanticText("", null, "baz"), "baz");
  assert.equal(joinSemanticText(), "");
});

test("normalizedId handles various id types", () => {
  assert.equal(normalizedId("abc"), "abc");
  assert.equal(normalizedId(42), "42");
  assert.equal(normalizedId(null), null);
  assert.equal(normalizedId(undefined), null);
  assert.deepEqual(normalizedId({ 0: "first" }), "first");
});

test("speechUnitIdFromEvent extracts id from direct field or artifact", () => {
  assert.equal(speechUnitIdFromEvent({ speech_unit_id: "u1" }), "u1");
  assert.equal(speechUnitIdFromEvent({ artifact: { speech_unit_id: "u2" } }), "u2");
  assert.equal(speechUnitIdFromEvent({}), null);
});

test("transcriptCandidateText joins stable and unstable text", () => {
  assert.equal(transcriptCandidateText({ stable_text: "hello", unstable_text: "world" }), "hello world");
  assert.equal(transcriptCandidateText({ stable_text: "only stable" }), "only stable");
  assert.equal(transcriptCandidateText(null), "");
  assert.equal(transcriptCandidateText("not an object"), "");
});

test("wordStreamText joins word objects", () => {
  assert.equal(wordStreamText([{ text: "foo" }, { text: "bar" }]), "foo bar");
  assert.equal(wordStreamText([]), "");
  assert.equal(wordStreamText(null), "");
});

// ── Deletion tracking ─────────────────────────────────────────────────────

test("tokenize splits text into tokens", () => {
  assert.deepEqual(tokenize("hello world"), ["hello", "world"]);
  assert.deepEqual(tokenize(""), []);
  assert.deepEqual(tokenize("  foo  bar  "), ["foo", "bar"]);
});

test("deletedTextBetween detects removed words", () => {
  assert.deepEqual(deletedTextBetween("hello world", "world"), ["hello"]);
  assert.deepEqual(deletedTextBetween("the cat sat", "the sat"), ["cat"]);
  assert.deepEqual(deletedTextBetween("foo", "foo bar"), []);
  assert.deepEqual(deletedTextBetween("", "foo"), []);
});

test("recordDeletedText appends unique snippets up to MAX_DELETED_SNIPPETS", () => {
  const target = [];
  recordDeletedText(target, ["alpha", "beta"], 100);
  assert.equal(target.length, 2);
  assert.equal(target[0].text, "alpha");
  assert.equal(target[0].elapsedMs, 100);

  // Duplicates are ignored.
  recordDeletedText(target, ["alpha"], 200);
  assert.equal(target.length, 2);

  // Cap at MAX_DELETED_SNIPPETS.
  for (let i = 0; i < MAX_DELETED_SNIPPETS + 2; i++) {
    recordDeletedText(target, [`unique-${i}`], i * 10);
  }
  assert.equal(target.length, MAX_DELETED_SNIPPETS);
});

// ── Schema classifiers ────────────────────────────────────────────────────

test("isLlmTextEvent classifies LLM text event kinds", () => {
  assert.equal(isLlmTextEvent("speech_unit_committed"), true);
  assert.equal(isLlmTextEvent("speech_unit_cancelled"), true);
  assert.equal(isLlmTextEvent("speculative_speech_updated"), true);
  assert.equal(isLlmTextEvent("tts_enqueue_started"), true);
  assert.equal(isLlmTextEvent("asr_started"), false);
});

test("isGeneratedSpeechEventKind is an alias for isLlmTextEvent", () => {
  assert.equal(isGeneratedSpeechEventKind, isLlmTextEvent);
});

test("isProspectiveCommitment returns true for non-final states", () => {
  assert.equal(isProspectiveCommitment("Hypothetical"), true);
  assert.equal(isProspectiveCommitment("Prepared"), true);
  assert.equal(isProspectiveCommitment("StableText"), false);
  assert.equal(isProspectiveCommitment("Final"), false);
  assert.equal(isProspectiveCommitment("Confirmed"), false);
  assert.equal(isProspectiveCommitment("Played"), false);
  assert.equal(isProspectiveCommitment(null), true);
});

test("isPlayedCommitment returns true only for committed states", () => {
  assert.equal(isPlayedCommitment("Final"), true);
  assert.equal(isPlayedCommitment("Confirmed"), true);
  assert.equal(isPlayedCommitment("Played"), true);
  assert.equal(isPlayedCommitment("Hypothetical"), false);
  assert.equal(isPlayedCommitment(null), false);
});

test("LIVE_EVENT_LANE maps event kinds to lanes", () => {
  assert.equal(LIVE_EVENT_LANE.asr_started, "ASR");
  assert.equal(LIVE_EVENT_LANE.transcript_confirmed, "ASR");
  assert.equal(LIVE_EVENT_LANE.playback_started, "Speaker");
  assert.equal(LIVE_EVENT_LANE.speech_started, "Mic");
  assert.equal(LIVE_EVENT_LANE.llm_generation_started, "LLM");
  assert.equal(LIVE_EVENT_LANE["prosody.frame"], "Prosody");
  assert.equal(LIVE_EVENT_LANE["prosody.contour"], "Prosody");
  assert.equal(LIVE_EVENT_LANE["prosody.pause"], "Prosody");
  assert.equal(LIVE_EVENT_LANE["prosody.phrase_candidate"], "Prosody");
  assert.equal(LIVE_EVENT_LANE["prosody.accent_candidate"], "Prosody");
  assert.equal(LIVE_EVENT_LANE.echo_planning_started, "Speaker");
});

test("SPAN_PAIRS and END_TO_START are consistent", () => {
  for (const [startKind, info] of Object.entries(SPAN_PAIRS)) {
    const reverse = END_TO_START[info.end];
    assert.ok(reverse, `END_TO_START missing entry for end-kind "${info.end}"`);
    assert.equal(reverse.startKind, startKind);
    assert.equal(reverse.lane, info.lane);
  }
});

// ── Transcript candidate reduction ───────────────────────────────────────

test("transcript revision: stable prefix grows while unstable tail changes", () => {
  const turn = mkTurn();

  applyTranscriptCandidate(
    turn,
    mkEvent("transcript_candidate", 100, {
      artifact: { stable_text: "I", unstable_text: "think" },
    }),
  );
  assert.equal(turn.userStable, "I");
  assert.equal(turn.userUnstable, "think");
  assert.equal(turn.userCandidateText, "I think");

  applyTranscriptCandidate(
    turn,
    mkEvent("transcript_candidate", 200, {
      artifact: { stable_text: "I think", unstable_text: "that" },
    }),
  );
  assert.equal(turn.userStable, "I think");
  assert.equal(turn.userUnstable, "that");
  assert.equal(turn.userCandidateText, "I think that");
});

test("transcript revision: deleted tokens are recorded", () => {
  const turn = mkTurn();

  applyTranscriptCandidate(
    turn,
    mkEvent("transcript_candidate", 100, {
      artifact: { stable_text: "hello world" },
    }),
  );

  // Next candidate drops "world".
  applyTranscriptCandidate(
    turn,
    mkEvent("transcript_candidate", 200, {
      artifact: { stable_text: "hello", unstable_text: "" },
    }),
  );

  assert.equal(turn.flags.revised, true);
  assert.ok(
    turn.userDeleted.some((entry) => entry.text === "world"),
    "deleted 'world' should be recorded",
  );
});

test("transcript candidate cancellation clears candidate text", () => {
  const turn = mkTurn({ userCandidateText: "some text", userStable: "some", userUnstable: "text" });

  applyTranscriptCandidate(
    turn,
    mkEvent("transcript_candidate", 300, { text: "candidate_cancelled" }),
  );

  assert.equal(turn.flags.cancelled, true);
  assert.equal(turn.userStable, "");
  assert.equal(turn.userUnstable, "");
  assert.equal(turn.userCandidateText, "");
  assert.ok(
    turn.userDeleted.some((entry) => entry.text === "some text"),
    "previous candidate text should be recorded as deleted",
  );
});

// ── ASR word stream reduction ─────────────────────────────────────────────

test("ASR word stream updates userWords and marks revised", () => {
  const turn = mkTurn();

  applyAsrWordStream(
    turn,
    mkEvent("asr_timed_word_stream", 100, {
      artifact: {
        words: [
          { text: "I", commitment: "StableText" },
          { text: "that", commitment: "Hypothetical" },
        ],
      },
    }),
  );
  assert.equal(turn.userWords.length, 2);
  assert.equal(turn.userWords[0].text, "I");
  assert.equal(turn.userWords[1].text, "that");

  // Retroactive correction: "that" → "what".
  applyAsrWordStream(
    turn,
    mkEvent("asr_timed_word_stream", 200, {
      artifact: {
        words: [
          { text: "I", commitment: "StableText" },
          { text: "what", commitment: "Hypothetical" },
        ],
      },
    }),
  );
  assert.equal(turn.userWords[1].text, "what");
  assert.equal(turn.flags.revised, true);
  assert.ok(
    turn.userDeleted.some((entry) => entry.text === "that"),
    "replaced word 'that' should be in deleted list",
  );
});

// ── LLM speculative → committed transitions ───────────────────────────────

test("speculative speech updates llmProspective and marks prospective", () => {
  const turn = mkTurn();

  applyLlmTextEvent(
    turn,
    mkEvent("speculative_speech_updated", 100, { text: "I think so." }),
  );
  assert.equal(turn.llmProspective, "I think so.");
  assert.equal(turn.llmFinal, "");
  assert.equal(turn.flags.prospective, true);
});

test("speech_unit_committed transitions speculative → committed", () => {
  const turn = mkTurn();

  applyLlmTextEvent(
    turn,
    mkEvent("speculative_speech_updated", 100, { text: "Maybe later." }),
  );

  applyLlmTextEvent(
    turn,
    mkEvent("speech_unit_committed", 200, { text: "Maybe later." }),
  );

  assert.equal(turn.llmFinal, "Maybe later.");
  assert.equal(turn.llmProspective, "Maybe later.");
});

test("tts_enqueue_started commits LLM text", () => {
  const turn = mkTurn();

  applyLlmTextEvent(
    turn,
    mkEvent("tts_enqueue_started", 100, { text: "Hello!" }),
  );
  assert.equal(turn.llmFinal, "Hello!");
});

test("commitLlmText appends new text cleanly", () => {
  const turn = mkTurn();

  commitLlmText(turn, "First sentence.");
  assert.equal(turn.llmFinal, "First sentence.");

  commitLlmText(turn, "Second sentence.");
  assert.equal(turn.llmFinal, "First sentence. Second sentence.");
});

test("commitLlmText does not duplicate already-committed prefix", () => {
  const turn = mkTurn({ llmFinal: "Hello there." });

  commitLlmText(turn, "Hello there.");
  assert.equal(turn.llmFinal, "Hello there.");
});

test("setLlmProspective sets prospective text and single fragment", () => {
  const turn = mkTurn();
  setLlmProspective(turn, "Prospective text.");
  assert.equal(turn.llmProspective, "Prospective text.");
  assert.deepEqual(turn.llmFragments, ["Prospective text."]);
});

test("finalizeLlmText commits and clears llmWords", () => {
  const turn = mkTurn({ llmWords: [{ text: "word", commitment: "Played" }] });

  finalizeLlmText(turn, "Finalised text.");
  assert.equal(turn.llmFinal, "Finalised text.");
  assert.deepEqual(turn.llmWords, []);
});

// ── Cancellation handling ─────────────────────────────────────────────────

test("speech_unit_cancelled marks turn cancelled and records deleted text", () => {
  const turn = mkTurn();

  applyLlmTextEvent(
    turn,
    mkEvent("speech_unit_committed", 100, { text: "This will be cancelled." }),
  );

  applyLlmTextEvent(
    turn,
    mkEvent("speech_unit_cancelled", 150, { text: "This will be cancelled." }),
  );

  assert.equal(turn.flags.cancelled, true);
  assert.ok(
    turn.llmDeleted.some((entry) => entry.text === "This will be cancelled."),
    "cancelled text should be in llmDeleted",
  );
});

test("speech_unit_cancelled removes speech unit from registry", () => {
  const turn = mkTurn();

  applyLlmTextEvent(
    turn,
    mkEvent("speech_unit_committed", 100, { speech_unit_id: "u1", text: "Unit one." }),
  );
  assert.equal(turn.speechUnitsById.get("u1"), "Unit one.");

  applyLlmTextEvent(
    turn,
    mkEvent("speech_unit_cancelled", 120, { speech_unit_id: "u1", text: "Unit one." }),
  );
  assert.equal(turn.speechUnitsById.has("u1"), false);
});

test("TTS word stream revision with all-cancelled words marks turn cancelled", () => {
  const turn = mkTurn({ llmFragments: ["goodbye"] });

  applyTtsWordStreamRevision(
    turn,
    mkEvent("tts_timed_word_stream_revision", 200, {
      artifact: {
        words: [
          { text: "goodbye", commitment: "Cancelled" },
        ],
      },
    }),
  );
  assert.equal(turn.flags.cancelled, true);
  assert.equal(turn.llmWords.length, 0);
});

// ── Playback lifecycle transitions ────────────────────────────────────────

test("playback lifecycle: speech committed before playback resolves label", () => {
  const turn = mkTurn();

  // Speech unit committed first.
  applyLlmTextEvent(
    turn,
    mkEvent("speech_unit_committed", 100, { speech_unit_id: "su1", text: "My name is Pete." }),
  );
  assert.equal(turn.speechUnitsById.get("su1"), "My name is Pete.");
  assert.equal(turn.llmFinal, "My name is Pete.");
});

test("speechUnitText resolves text from turn registry when event has no direct text", () => {
  const turn = mkTurn();
  turn.speechUnitsById.set("su2", "Indirect text.");

  const text = speechUnitText(turn, mkEvent("playback_started", 120, { speech_unit_id: "su2" }));
  assert.equal(text, "Indirect text.");
});

test("speechUnitText prefers direct event text over registry lookup", () => {
  const turn = mkTurn();
  turn.speechUnitsById.set("su3", "Registry text.");

  const text = speechUnitText(turn, mkEvent("playback_started", 130, {
    speech_unit_id: "su3",
    text: "Direct text.",
  }));
  assert.equal(text, "Direct text.");
});

// ── Selectors ─────────────────────────────────────────────────────────────

test("currentUserText prefers final over word stream over candidate", () => {
  const t1 = mkTurn({ userFinal: "Final text." });
  assert.equal(currentUserText(t1), "Final text.");

  const t2 = mkTurn({ userWords: [{ text: "stream" }] });
  assert.equal(currentUserText(t2), "stream");

  const t3 = mkTurn({ userCandidateText: "candidate" });
  assert.equal(currentUserText(t3), "candidate");
});

test("prospectiveTail extracts the tail beyond committed text", () => {
  // Prospective extends beyond the final: return the extension.
  assert.equal(prospectiveTail("Hello", "Hello there."), "there.");
  // Prospective equals final: no tail.
  assert.equal(prospectiveTail("Hello there.", "Hello there."), "");
  // No final text yet: return the whole prospective.
  assert.equal(prospectiveTail("", "incoming"), "incoming");
  // Prospective is shorter/different from final: return the whole prospective as-is.
  assert.equal(prospectiveTail("Hello there.", "Hello"), "Hello");
});

test("turnHasUserDialogue detects any user content", () => {
  assert.equal(turnHasUserDialogue(mkTurn()), false);
  assert.equal(turnHasUserDialogue(mkTurn({ userFinal: "hi" })), true);
  assert.equal(turnHasUserDialogue(mkTurn({ userWords: [{ text: "w" }] })), true);
  assert.equal(turnHasUserDialogue(mkTurn({ userDeleted: [{ text: "x", elapsedMs: 0 }] })), true);
});

test("turnHasLlmDialogue detects any LLM content", () => {
  assert.equal(turnHasLlmDialogue(mkTurn()), false);
  assert.equal(turnHasLlmDialogue(mkTurn({ llmFinal: "answer" })), true);
  assert.equal(turnHasLlmDialogue(mkTurn({ llmProspective: "maybe" })), true);
  assert.equal(turnHasLlmDialogue(mkTurn({ llmDeleted: [{ text: "gone", elapsedMs: 0 }] })), true);
});
