/**
 * phoneme-projection.mjs
 *
 * Utilities for projecting CMUdict phoneme sequences into word timing
 * intervals on the WaveDeck timeline.
 *
 * The first pass uses proportional timing with optional vowel weighting.
 * Phoneme spans produced here are clearly labelled as "projected" — they
 * represent pronunciation-derived estimates, not measured acoustic timings.
 *
 * Future implementations may refine these spans using energy landmarks or
 * forced alignment.
 */

// ---------------------------------------------------------------------------
// ARPAbet vowel set
// ---------------------------------------------------------------------------

/**
 * The complete set of ARPAbet vowel base symbols.
 *
 * Stress digits (0/1/2) are stripped before checking, so both `"AH"` and
 * `"AH1"` match.
 */
const VOWEL_BASES = new Set([
  "AA", "AE", "AH", "AO", "AW", "AY",
  "EH", "ER", "EY",
  "IH", "IY",
  "OW", "OY",
  "UH", "UW",
]);

/**
 * Return `true` when `symbol` is an ARPAbet vowel (stress digit ignored).
 *
 * @param {string} symbol
 * @returns {boolean}
 */
export function isVowel(symbol) {
  return VOWEL_BASES.has(symbol.replace(/[012]$/, ""));
}

// ---------------------------------------------------------------------------
// Proportional phoneme projection
// ---------------------------------------------------------------------------

/** Weight applied to vowel phonemes when computing proportional duration. */
const VOWEL_WEIGHT = 2.0;
/** Weight applied to consonant phonemes when computing proportional duration. */
const CONSONANT_WEIGHT = 1.0;

/**
 * Project an array of ARPAbet phoneme symbols proportionally into a timing
 * interval, optionally weighting vowels longer than consonants.
 *
 * The result preserves monotonic ordering, keeps every span within
 * [startMs, endMs], and never produces overlapping spans.  The final phoneme
 * always ends exactly at `endMs` so no duration is lost to rounding.
 *
 * These spans carry provenance metadata so callers know the timings are
 * *projected* estimates, not acoustically measured.
 *
 * @param {string[]} phonemes   ARPAbet symbols (e.g. `["TH", "R", "IY1"]`)
 * @param {number}   startMs   Word interval start in milliseconds (inclusive)
 * @param {number}   endMs     Word interval end in milliseconds (inclusive end boundary; the last phoneme spans up to this value)
 * @param {string}   [source]  Provenance label (default: `"cmudict.proportional"`)
 * @returns {Array<{symbol: string, start_ms: number, end_ms: number, source: string}>}
 */
export function projectPhonemesIntoWordInterval(
  phonemes,
  startMs,
  endMs,
  source = "cmudict.proportional",
) {
  if (!Array.isArray(phonemes) || phonemes.length === 0) return [];
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs <= startMs) return [];

  const duration = endMs - startMs;
  const weights = phonemes.map((p) => (isVowel(p) ? VOWEL_WEIGHT : CONSONANT_WEIGHT));
  const totalWeight = weights.reduce((sum, w) => sum + w, 0);

  const spans = [];
  let curMs = startMs;

  for (let i = 0; i < phonemes.length; i++) {
    const proportionMs = (weights[i] / totalWeight) * duration;
    const spanStart = Math.round(curMs);
    // The last phoneme always snaps to endMs to avoid rounding drift.
    const spanEnd = i === phonemes.length - 1 ? endMs : Math.round(curMs + proportionMs);

    spans.push({
      symbol: phonemes[i],
      start_ms: spanStart,
      end_ms: spanEnd,
      source,
    });

    curMs += proportionMs;
  }

  return spans;
}

// ---------------------------------------------------------------------------
// Stress pattern helper
// ---------------------------------------------------------------------------

/**
 * Extract the stress pattern from a phoneme list as a compact string of
 * stress digits (`"0"`, `"1"`, `"2"`) for vowels only.
 *
 * For example `["TH", "R", "IY1"]` → `"1"`.
 *
 * @param {string[]} phonemes
 * @returns {string}
 */
export function stressPattern(phonemes) {
  return phonemes
    .filter((p) => isVowel(p))
    .map((p) => {
      const last = p[p.length - 1];
      return "012".includes(last) ? last : "0";
    })
    .join("");
}
