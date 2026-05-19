/**
 * web/browser-transcript-player/shared/events/schema.mjs
 *
 * Shared live event schema: kind-to-lane routing, span pairing rules, and
 * commitment/event-kind classifiers.  Consumed by WaveDeck (app.js),
 * screenplay (screenplay-model.mjs), and replay tooling.
 */

// Lane assignment for live trace event kinds.
export const LIVE_EVENT_LANE = Object.freeze({
  capture_started: "Mic",
  listening_started: "Mic",
  speech_started: "Mic",
  speech_stopped: "Mic",
  breath_group_opened: "Mic",
  breath_group_closed: "Mic",
  auditory_observation: "Mic",
  environmental_sound: "Mic",
  self_voice_heard: "Speaker",
  overlap_detected: "Mic",
  asr_started: "ASR",
  asr_finished: "ASR",
  transcript: "ASR",
  transcript_candidate: "ASR",
  transcript_proposition: "ASR",
  transcript_confirmed: "ASR",
  transcription_refinement_error: "ASR",
  asr_timed_word_stream: "ASR",
  "prosody.frame": "Prosody",
  "prosody.contour": "Prosody",
  "prosody.pause": "Prosody",
  "prosody.phrase_candidate": "Prosody",
  "prosody.accent_candidate": "Prosody",
  echo_planning_started: "Speaker",
  llm_generation_started: "LLM",
  first_llm_token: "LLM",
  llm_token: "LLM",
  llm_token_delta: "LLM",
  token_emitted: "LLM",
  first_safe_speech_unit_emitted: "LLM",
  speech_unit_committed: "LLM",
  speech_unit_cancelled: "LLM",
  speculative_speech_updated: "LLM",
  first_tts_audio_frame_available: "Speaker",
  playback_started: "Speaker",
  playback_finished: "Speaker",
  self_hearing_suppression_started: "Speaker",
  self_hearing_suppression_ended: "Speaker",
  face_expression: "Emotion",
  face_state: "Emotion",
  facial_expression: "Emotion",
  expression_changed: "Emotion",
  emotion_state: "Emotion",
  affect_state: "Emotion",
});

// Span pairing rules: maps start-event kind → { end-event kind, lane }.
// Used by the live-session reducer and its projection function.
export const SPAN_PAIRS = Object.freeze({
  speech_started: { end: "speech_stopped", lane: "Mic" },
  asr_started: { end: "asr_finished", lane: "ASR" },
  playback_started: { end: "playback_finished", lane: "Speaker" },
  llm_generation_started: { end: "playback_started", lane: "LLM" },
  self_hearing_suppression_started: { end: "self_hearing_suppression_ended", lane: "Speaker" },
});

// Reverse mapping: end-event kind → { startKind, lane }.
export const END_TO_START = Object.freeze(
  Object.fromEntries(
    Object.entries(SPAN_PAIRS).map(([startKind, info]) => [
      info.end,
      { startKind, lane: info.lane },
    ]),
  ),
);

/**
 * Returns true for event kinds that carry LLM-generated speech text.
 * (Also known as `isGeneratedSpeechEventKind` in WaveDeck.)
 */
export function isLlmTextEvent(kind) {
  return [
    "first_safe_speech_unit_emitted",
    "speech_unit_committed",
    "speech_unit_cancelled",
    "speculative_speech_updated",
    "tts_enqueue_started",
  ].includes(kind);
}

/** Alias kept for backward compatibility with WaveDeck usage. */
export const isGeneratedSpeechEventKind = isLlmTextEvent;

/**
 * Returns true when a word commitment level is not yet finalised
 * (i.e. the word is still speculative or provisional).
 */
export function isProspectiveCommitment(commitment) {
  return !["Final", "Confirmed", "StableText", "Played"].includes(String(commitment ?? ""));
}

/**
 * Returns true when a word commitment level indicates the word has been
 * played back (i.e. it is committed or finalised).
 */
export function isPlayedCommitment(commitment) {
  return ["Final", "Confirmed", "Played"].includes(String(commitment ?? ""));
}
