/**
 * web/browser-transcript-player/shared/events/reducers.mjs
 *
 * Shared live event reduction layer: text normalisation utilities, turn-level
 * state reducers, and text-deletion helpers.
 *
 * Consumed by WaveDeck (app.js), screenplay (screenplay-model.mjs), and
 * replay/inspector tooling so that all consumers share the same semantic
 * event model.
 *
 * Turn state shape expected by the reducers in this module:
 *   {
 *     flags:           { cancelled, revised, prospective }
 *     userStable:      string
 *     userUnstable:    string
 *     userCandidateText: string
 *     userWords:       Word[]
 *     userDeleted:     DeletedEntry[]
 *     userFinal:       string
 *     llmFinal:        string
 *     llmProspective:  string
 *     llmFragments:    string[]
 *     llmWords:        Word[]
 *     llmDeleted:      DeletedEntry[]
 *     speechUnitsById: Map<id, string>
 *   }
 */

import { isLlmTextEvent, isProspectiveCommitment, isPlayedCommitment } from "./schema.mjs";

export { isLlmTextEvent, isProspectiveCommitment, isPlayedCommitment };

// Maximum number of deleted-text snippets retained per turn.
export const MAX_DELETED_SNIPPETS = 8;

// ── Text normalisation ────────────────────────────────────────────────────

/**
 * Normalise arbitrary value to a clean string (never null/undefined).
 * Collapses whitespace and removes space before punctuation.
 * Returns an empty string for null/undefined/non-string input.
 */
export function textContent(value) {
  return String(value ?? "")
    .replace(/\s+/g, " ")
    .replace(/\s+([,.;:!?])/g, "$1")
    .trim();
}

/**
 * Like textContent but returns null instead of an empty string.
 * Useful in `??` chains where an absent value must short-circuit to the
 * next candidate.
 */
export function normalizeSemanticText(value) {
  if (typeof value !== "string") {
    return null;
  }
  const text = value.replace(/\s+/g, " ").replace(/\s+([,.;:!?])/g, "$1").trim();
  return text.length > 0 ? text : null;
}

/**
 * Join an array of word strings into a natural-language sentence, respecting
 * punctuation attachment (no space before , . ; : ! ?).
 */
export function joinWords(words) {
  return words
    .map(textContent)
    .filter(Boolean)
    .reduce((acc, word) => acc + (/^[,.;:!?)]/.test(word) ? word : `${acc ? " " : ""}${word}`), "");
}

/**
 * Join an arbitrary number of string parts with a single space, normalising
 * the result via textContent.
 */
export function joinSemanticText(...parts) {
  return textContent(parts.map(textContent).filter(Boolean).join(" "));
}

/** Normalise an opaque ID value to a string or null. */
export function normalizedId(value) {
  if (value == null) {
    return null;
  }
  if (typeof value === "string" || typeof value === "number") {
    return String(value);
  }
  if (typeof value === "object" && value !== null && "0" in value) {
    return String(value[0]);
  }
  return JSON.stringify(value);
}

/** Extract the speech-unit ID from an event (direct field or inside artifact). */
export function speechUnitIdFromEvent(event) {
  return normalizedId(event?.speech_unit_id ?? event?.artifact?.speech_unit_id);
}

/**
 * Extract combined text from a transcript candidate artifact
 * (stable_text + unstable_text).
 */
export function transcriptCandidateText(candidate) {
  if (!candidate || typeof candidate !== "object") return "";
  return joinSemanticText(candidate.stable_text, candidate.unstable_text);
}

/**
 * Join an ASR/TTS word stream array into a single text string.
 */
export function wordStreamText(words) {
  return joinWords((words ?? []).map((word) => word?.text));
}

// ── Deletion tracking ─────────────────────────────────────────────────────

/** Split normalised text into non-whitespace tokens. */
export function tokenize(text) {
  return textContent(text).match(/\S+/g) ?? [];
}

/**
 * Return an array of word-group strings that exist in `previous` but not
 * in `next`.  Contiguous runs of dropped tokens are grouped together.
 */
export function deletedTextBetween(previous, next) {
  const prevTokens = tokenize(previous);
  const nextTokens = new Set(tokenize(next));
  if (!prevTokens.length) {
    return [];
  }

  const deleted = [];
  let current = [];
  for (const token of prevTokens) {
    if (nextTokens.has(token)) {
      if (current.length) {
        deleted.push(joinWords(current));
        current = [];
      }
    } else {
      current.push(token);
    }
  }
  if (current.length) {
    deleted.push(joinWords(current));
  }
  return deleted;
}

/**
 * Append non-empty, non-duplicate text snippets to `target`, capping the
 * list at MAX_DELETED_SNIPPETS.
 */
export function recordDeletedText(target, snippets, elapsedMs) {
  for (const snippet of snippets.map(textContent).filter(Boolean)) {
    if (target.some((entry) => entry.text === snippet)) {
      continue;
    }
    target.push({ text: snippet, elapsedMs: elapsedMs ?? 0 });
  }
  if (target.length > MAX_DELETED_SNIPPETS) {
    target.splice(0, target.length - MAX_DELETED_SNIPPETS);
  }
}

// ── Turn state reducers ───────────────────────────────────────────────────
//
// These mutate turn objects that conform to the shape described at the top of
// this file.  Both screenplay-model.mjs and future tooling should use the
// same functions rather than maintaining separate implementations.

/**
 * Resolve the text for a speech unit: use direct event text if present,
 * otherwise look up the speech-unit ID in turn.speechUnitsById.
 */
export function speechUnitText(turn, event) {
  const direct = textContent(event?.text);
  if (direct) {
    return direct;
  }
  const speechUnitId = speechUnitIdFromEvent(event);
  if (!speechUnitId) {
    return "";
  }
  return textContent(turn.speechUnitsById.get(speechUnitId));
}

/**
 * Apply a `transcript_candidate` event to the narrative turn.
 * Updates userStable, userUnstable, userCandidateText, and userDeleted.
 */
export function applyTranscriptCandidate(turn, event) {
  const artifact = event.artifact && typeof event.artifact === "object" ? event.artifact : null;
  if (!artifact) {
    if (String(event.text ?? "").includes("candidate_cancelled")) {
      turn.flags.cancelled = true;
      recordDeletedText(turn.userDeleted, [turn.userCandidateText], event.elapsed_ms);
      turn.userStable = "";
      turn.userUnstable = "";
      turn.userCandidateText = "";
    }
    return;
  }

  const stable = textContent(artifact.stable_text);
  const unstable = textContent(artifact.unstable_text);
  const next = joinSemanticText(stable, unstable);
  const deleted = deletedTextBetween(turn.userCandidateText, next);
  if (deleted.length) {
    turn.flags.revised = true;
  }
  recordDeletedText(turn.userDeleted, deleted, event.elapsed_ms);
  turn.userStable = stable;
  turn.userUnstable = unstable;
  turn.userCandidateText = next;
  turn.flags.prospective ||= Boolean(unstable);
}

/**
 * Apply an `asr_timed_word_stream` event to the narrative turn.
 * Updates userWords, userCandidateText, and userDeleted.
 */
export function applyAsrWordStream(turn, event) {
  const words = Array.isArray(event.artifact?.words) ? event.artifact.words : [];
  if (!words.length) {
    return;
  }

  const next = joinWords(words.map((word) => word?.text));
  const deleted = deletedTextBetween(
    joinWords(turn.userWords.map((word) => word.text)),
    next,
  );
  if (deleted.length) {
    turn.flags.revised = true;
  }
  recordDeletedText(turn.userDeleted, deleted, event.elapsed_ms);
  turn.userWords = words
    .filter((word) => textContent(word?.text))
    .map((word) => ({
      text: textContent(word.text),
      commitment: String(word.commitment ?? ""),
      id: word.id ?? null,
      span_id: word.span_id ?? null,
      timing: word.timing ?? null,
    }));
  turn.userCandidateText = turn.userFinal || next;
  turn.flags.prospective ||= turn.userWords.some((word) => isProspectiveCommitment(word.commitment));
}

/**
 * Apply a `tts_timed_word_stream_revision` event to the narrative turn.
 * Updates llmWords, llmFinal, llmFragments, and llmDeleted.
 */
export function applyTtsWordStreamRevision(turn, event) {
  const words = Array.isArray(event.artifact?.words) ? event.artifact.words : [];
  const text = joinWords(words.map((word) => word?.text));
  if (!text) {
    return;
  }

  const reason = String(event.reason ?? "").toLowerCase();
  const cancelled =
    reason.includes("cancel") ||
    words.every((word) => String(word?.commitment ?? "") === "Cancelled");
  if (cancelled) {
    turn.flags.cancelled = true;
    recordDeletedText(turn.llmDeleted, [text], event.elapsed_ms);
    _removeLlmFragment(turn, text);
    turn.llmWords = [];
    return;
  }

  turn.llmWords = words
    .filter((word) => textContent(word?.text))
    .map((word) => ({
      text: textContent(word.text),
      commitment: String(word.commitment ?? ""),
      id: word.id ?? null,
      span_id: word.span_id ?? null,
      timing: word.timing ?? null,
    }));

  const playedPrefix = [];
  for (const word of turn.llmWords) {
    if (!isPlayedCommitment(word.commitment)) {
      break;
    }
    playedPrefix.push(word.text);
  }
  let committedRevision = false;
  if (playedPrefix.length) {
    commitLlmText(turn, joinWords(playedPrefix));
    committedRevision = true;
  } else if (reason.includes("committed")) {
    commitLlmText(turn, text);
    committedRevision = true;
  }
  if (!committedRevision) {
    _addLlmFragment(turn, text);
  }
}

/**
 * Apply an LLM text event (speech_unit_committed, speech_unit_cancelled,
 * speculative_speech_updated, tts_enqueue_started, etc.) to the turn.
 */
export function applyLlmTextEvent(turn, event) {
  const speechUnitId = speechUnitIdFromEvent(event);
  const text = speechUnitText(turn, event);
  if (!text) {
    return;
  }
  if (speechUnitId) {
    turn.speechUnitsById.set(speechUnitId, text);
  }

  if (event.kind === "speech_unit_cancelled") {
    turn.flags.cancelled = true;
    recordDeletedText(turn.llmDeleted, [text], event.elapsed_ms);
    _removeLlmFragment(turn, text);
    if (speechUnitId) {
      turn.speechUnitsById.delete(speechUnitId);
    }
    return;
  }

  if (event.kind === "speculative_speech_updated") {
    turn.flags.prospective = true;
    setLlmProspective(turn, text);
    return;
  }

  if (event.kind === "tts_enqueue_started" || event.kind === "speech_unit_committed") {
    commitLlmText(turn, text);
    return;
  }

  _addLlmFragment(turn, text);
}

/**
 * Commit a final LLM text chunk to the turn, merging it cleanly with any
 * previously committed text.
 */
export function commitLlmText(turn, text) {
  const next = textContent(text);
  if (!next) {
    return;
  }

  const current = textContent(turn.llmFinal);
  if (!current) {
    turn.llmFinal = next;
  } else if (current === next || current.endsWith(next)) {
    turn.llmFinal = current;
  } else if (next.startsWith(current)) {
    turn.llmFinal = next;
  } else {
    turn.llmFinal = joinSemanticText(current, next);
  }

  if (!turn.llmProspective || turn.llmFinal.startsWith(turn.llmProspective)) {
    turn.llmProspective = turn.llmFinal;
  } else if (!turn.llmProspective.startsWith(turn.llmFinal)) {
    turn.llmProspective = joinSemanticText(turn.llmFinal, turn.llmProspective);
  }
  turn.llmFragments = turn.llmProspective ? [turn.llmProspective] : [];
}

/**
 * Set the speculative/prospective LLM text, replacing the current fragment
 * list with a single entry.
 */
export function setLlmProspective(turn, text) {
  const cleaned = textContent(text);
  if (!cleaned) {
    return;
  }
  turn.llmProspective = cleaned;
  if (!turn.llmFragments.includes(cleaned)) {
    turn.llmFragments = [cleaned];
  }
}

/**
 * Finalise an LLM turn with committed text, clearing the word stream.
 */
export function finalizeLlmText(turn, text) {
  commitLlmText(turn, text);
  turn.llmWords = [];
}

// ── Private helpers ───────────────────────────────────────────────────────

function _addLlmFragment(turn, text) {
  const cleaned = textContent(text);
  if (!cleaned) {
    return;
  }
  const last = turn.llmFragments[turn.llmFragments.length - 1];
  if (last !== cleaned && !turn.llmFragments.includes(cleaned)) {
    turn.llmFragments.push(cleaned);
  }
  turn.llmProspective = joinSemanticText(...turn.llmFragments);
}

function _removeLlmFragment(turn, text) {
  const cleaned = textContent(text);
  turn.llmFragments = turn.llmFragments.filter((fragment) => fragment !== cleaned);
  turn.llmProspective = joinSemanticText(...turn.llmFragments);
}
