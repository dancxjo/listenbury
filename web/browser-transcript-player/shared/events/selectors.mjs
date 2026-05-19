/**
 * web/shared/events/selectors.mjs
 *
 * Read-only derived-state selectors for the shared narrative turn model.
 * These functions never mutate their arguments and are safe to call during
 * projection/rendering passes.
 */

import { textContent, joinWords, joinSemanticText } from "./reducers.mjs";

// ── Turn text selectors ───────────────────────────────────────────────────

/**
 * Return the best available committed or in-progress user (ASR) text for the
 * turn, in order of preference: final > word stream > candidate text.
 */
export function currentUserText(turn) {
  return (
    turn.userFinal ||
    joinWords(turn.userWords.map((word) => word.text)) ||
    turn.userCandidateText
  );
}

/**
 * Return the prospective tail of LLM text: the portion of the speculative
 * text that extends beyond the committed final text.
 */
export function prospectiveTail(finalText, prospectiveText) {
  const finalClean = textContent(finalText);
  const prospectiveClean = textContent(prospectiveText);
  if (!prospectiveClean || prospectiveClean === finalClean) {
    return "";
  }
  if (finalClean.endsWith(prospectiveClean)) {
    return "";
  }
  if (prospectiveClean.startsWith(finalClean)) {
    return prospectiveClean.slice(finalClean.length).trim();
  }
  return prospectiveClean;
}

// ── Turn presence checks ──────────────────────────────────────────────────

/**
 * Returns true if the turn has any user (ASR) dialogue content.
 */
export function turnHasUserDialogue(turn) {
  return Boolean(
    turn.userFinal ||
      turn.userStable ||
      turn.userUnstable ||
      turn.userWords.length ||
      turn.userDeleted.length,
  );
}

/**
 * Returns true if the turn has any LLM-generated dialogue content.
 */
export function turnHasLlmDialogue(turn) {
  return Boolean(turn.llmFinal || turn.llmProspective || turn.llmDeleted.length);
}
