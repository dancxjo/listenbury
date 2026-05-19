/**
 * phoneme-projection.mjs
 *
 * Utilities for projecting CMUdict/ARPAbet phoneme sequences into timed
 * WaveDeck spans with phonological metadata:
 * - source symbol (ARPAbet token)
 * - default underlying IPA phone string
 * - optional environment-conditioned realized IPA
 * - explicit realization provenance
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

const ARPABET_TO_IPA_DEFAULTS = {
  AA: "ɑ",
  AE: "æ",
  AH: "ʌ",
  AO: "ɔ",
  AW: "aʊ",
  AY: "aɪ",
  B: "b",
  CH: "tʃ",
  D: "d",
  DH: "ð",
  EH: "ɛ",
  ER: "ɝ",
  EY: "eɪ",
  F: "f",
  G: "ɡ",
  HH: "h",
  IH: "ɪ",
  IY: "iː",
  JH: "dʒ",
  K: "k",
  L: "l",
  M: "m",
  N: "n",
  NG: "ŋ",
  OW: "oʊ",
  OY: "ɔɪ",
  P: "p",
  R: "ɹ",
  S: "s",
  SH: "ʃ",
  T: "t",
  TH: "θ",
  UH: "ʊ",
  UW: "uː",
  V: "v",
  W: "w",
  Y: "j",
  Z: "z",
  ZH: "ʒ",
};

const STRESS_BY_DIGIT = {
  0: "unstressed",
  1: "primary",
  2: "secondary",
};
const STRESS_DIGITS = new Set(["0", "1", "2"]);

/**
 * @typedef {"word_initial"|"word_medial"|"word_final"|"singleton"} WordPosition
 *
 * @typedef {Object} Environment
 * @property {string|null} left_phone
 * @property {string|null} right_phone
 * @property {string|null} left_class
 * @property {string|null} right_class
 * @property {WordPosition|null} word_position
 * @property {string|null} syllable_position
 * @property {string|null} stress_context
 * @property {string|null} phrase_position
 * @property {string|null} language
 * @property {string|null} dialect
 *
 * @typedef {Object} Phone
 * @property {string} ipa
 * @property {string|null} sourceSymbol
 * @property {string} status
 *
 * @typedef {Object} PhoneString
 * @property {Phone[]} phones
 *
 * @typedef {Object} Realization
 * @property {string} ipa
 * @property {"default"|"allophone_rule"} method
 * @property {string|null} rule
 * @property {Environment|null} environment
 *
 * @typedef {Object} Phoneme
 * @property {string} symbol
 * @property {string} sourceSymbol
 * @property {string} source
 * @property {"primary"|"secondary"|"unstressed"|null} stress
 * @property {Phone[]} defaultPhoneString
 * @property {Realization} realization
 */

/**
 * Return `true` when `symbol` is an ARPAbet vowel (stress digit ignored).
 *
 * @param {string} symbol
 * @returns {boolean}
 */
export function isVowel(symbol) {
  return VOWEL_BASES.has(symbol.replace(/[012]$/, ""));
}

function isStressDigit(char) {
  return STRESS_DIGITS.has(char);
}

function parseArpabetToken(token) {
  if (token == null) {
    return {
      sourceSymbol: "",
      symbol: "",
      stressDigit: null,
      stress: null,
    };
  }
  const raw = String(token ?? "").trim();
  const last = raw[raw.length - 1];
  const hasStress = isStressDigit(last);
  const base = hasStress ? raw.slice(0, -1) : raw;
  return {
    sourceSymbol: raw,
    symbol: base,
    stressDigit: hasStress ? Number(last) : null,
    stress: hasStress ? STRESS_BY_DIGIT[last] : null,
  };
}

function defaultPhoneFromArpabet(base, sourceSymbol) {
  const ipa = ARPABET_TO_IPA_DEFAULTS[base];
  if (!ipa) {
    return { ipa: `?${base}`, sourceSymbol, status: "unknown_symbol" };
  }
  return { ipa, sourceSymbol, status: "mapped" };
}

function wordPosition(index, length) {
  if (length <= 1) return "singleton";
  if (index === 0) return "word_initial";
  if (index === length - 1) return "word_final";
  return "word_medial";
}

function defaultEnvironment(index, sequenceLength, sequence) {
  const phoneIpa = (entry) => entry?.realization?.ipa ?? entry?.defaultPhoneString?.[0]?.ipa ?? null;
  const left = sequence[index - 1] ?? null;
  const right = sequence[index + 1] ?? null;
  return {
    left_phone: phoneIpa(left),
    right_phone: phoneIpa(right),
    left_class: left ? (isVowel(left.sourceSymbol) ? "vowel" : "consonant") : null,
    right_class: right ? (isVowel(right.sourceSymbol) ? "vowel" : "consonant") : null,
    word_position: wordPosition(index, sequenceLength),
    syllable_position: null,
    stress_context: null,
    phrase_position: null,
    language: "en",
    dialect: "american_english",
  };
}

/**
 * Build a phonological phoneme object from a CMUdict/ARPAbet symbol.
 *
 * @param {string} token
 * @param {string} [source]
 * @returns {Phoneme}
 */
export function phonemeFromArpabet(token, source = "cmudict") {
  const parsed = parseArpabetToken(token);
  const phone = defaultPhoneFromArpabet(parsed.symbol, parsed.sourceSymbol);
  return {
    symbol: parsed.symbol,
    sourceSymbol: parsed.sourceSymbol,
    source,
    stress: parsed.stress,
    defaultPhoneString: [phone],
    realization: {
      ipa: phone.ipa,
      method: "default",
      rule: null,
      environment: null,
    },
  };
}

/**
 * Apply opt-in allophone rules to an ARPAbet-derived phoneme sequence.
 *
 * @param {Phoneme[]} phonemes
 * @param {{enabled?: boolean, language?: string, dialect?: string}} [config]
 * @returns {Phoneme[]}
 */
export function realizePhonemeSequence(phonemes, config = {}) {
  const enabled = config.enabled === true;
  if (!enabled || !Array.isArray(phonemes)) {
    return Array.isArray(phonemes) ? phonemes : [];
  }
  const dialect = config.dialect ?? "american_english";
  const language = config.language ?? "en";
  const realized = phonemes.map((entry, index, sequence) => {
    const env = defaultEnvironment(index, sequence.length, sequence);
    return {
      ...entry,
      realization: {
        ...entry.realization,
        environment: { ...env, language, dialect },
      },
    };
  });

  for (let i = 1; i < realized.length - 1; i++) {
    const cur = realized[i];
    const left = realized[i - 1];
    const right = realized[i + 1];
    const isIntervocalic =
      isVowel(left.sourceSymbol) &&
      isVowel(right.sourceSymbol) &&
      (cur.symbol === "T" || cur.symbol === "D");
    const leftStress = left.stress === "primary" || left.stress === "secondary";
    const rightUnstressed = right.stress === "unstressed";
    if (!isIntervocalic || !leftStress || !rightUnstressed) continue;

    realized[i] = {
      ...cur,
      realization: {
        ipa: "ɾ",
        method: "allophone_rule",
        rule: "american_english_intervocalic_flapping",
        environment: {
          ...cur.realization.environment,
          left_phone: left.realization.ipa,
          right_phone: right.realization.ipa,
          left_class: "vowel",
          right_class: "vowel",
          stress_context: "between stressed vowel and unstressed vowel",
          language,
          dialect,
        },
      },
    };
  }

  return realized;
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
 * @param {{allophoneRules?: {enabled?: boolean, language?: string, dialect?: string}}} [options]
 * @returns {Array<Phoneme & {start_ms: number, end_ms: number, timingSource: string}>}
 */
export function projectPhonemesIntoWordInterval(
  phonemes,
  startMs,
  endMs,
  source = "cmudict.proportional",
  options = {},
) {
  if (!Array.isArray(phonemes) || phonemes.length === 0) return [];
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs <= startMs) return [];

  const baseSequence = phonemes.map((token) => phonemeFromArpabet(token, "cmudict"));
  const sequence = realizePhonemeSequence(baseSequence, options.allophoneRules);
  const duration = endMs - startMs;
  const weights = sequence.map((p) => (isVowel(p.sourceSymbol) ? VOWEL_WEIGHT : CONSONANT_WEIGHT));
  const totalWeight = weights.reduce((sum, w) => sum + w, 0);

  const spans = [];
  let curMs = startMs;

  for (let i = 0; i < sequence.length; i++) {
    const proportionMs = (weights[i] / totalWeight) * duration;
    const spanStart = Math.round(curMs);
    // The last phoneme always snaps to endMs to avoid rounding drift.
    const spanEnd = i === sequence.length - 1 ? endMs : Math.round(curMs + proportionMs);

    spans.push({
      ...sequence[i],
      start_ms: spanStart,
      end_ms: spanEnd,
      timingSource: source,
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
