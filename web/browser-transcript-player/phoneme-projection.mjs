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

const STOP_BASES = new Set(["P", "B", "T", "D", "K", "G"]);
const FRICATIVE_BASES = new Set(["F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"]);
const NASAL_BASES = new Set(["M", "N", "NG"]);
const APPROXIMANT_LIQUID_BASES = new Set(["R", "L", "W", "Y"]);
const AFFRICATE_BASES = new Set(["CH", "JH"]);
const SILENCE_BASES = new Set(["SIL", "SP", "PAU"]);
const DIPHTHONG_BASES = new Set(["AW", "AY", "EY", "OW", "OY"]);

export const PHONE_CLASS_HEURISTICS = {
  vowel: {
    label: "vowel",
    expectedCues: ["stable_formants", "voicing", "energy_nucleus"],
    defaultMethod: "heuristic.vowel.formants",
  },
  diphthong: {
    label: "diphthong",
    expectedCues: ["formant_movement", "voicing", "energy_nucleus"],
    defaultMethod: "heuristic.formant.movement",
  },
  fricative: {
    label: "fricative",
    expectedCues: ["high_frequency_noise", "frication_band_energy", "voicing_optional"],
    defaultMethod: "heuristic.spectral.frication",
  },
  stop: {
    label: "stop",
    expectedCues: ["closure_low_energy", "release_burst", "post_release_onset"],
    defaultMethod: "heuristic.energy.stop_release",
  },
  nasal: {
    label: "nasal",
    expectedCues: ["voicing", "nasal_murmur", "reduced_high_frequency_energy"],
    defaultMethod: "heuristic.spectral.nasal_murmur",
  },
  approximant_liquid: {
    label: "approximant_liquid",
    expectedCues: ["voicing", "formant_transition", "low_obstruction"],
    defaultMethod: "heuristic.formant.transition",
  },
  affricate: {
    label: "affricate",
    expectedCues: ["stop_like_closure", "fricative_release"],
    defaultMethod: "heuristic.combined.affricate",
  },
  silence: {
    label: "silence",
    expectedCues: ["low_energy", "low_voicing"],
    defaultMethod: "heuristic.energy.silence",
  },
  other: {
    label: "other",
    expectedCues: ["duration_prior"],
    defaultMethod: "heuristic.combined",
  },
};

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

function phoneClassFromSymbol(sourceSymbol) {
  const base = parseArpabetToken(sourceSymbol).symbol;
  if (SILENCE_BASES.has(base)) return "silence";
  if (isVowel(base)) return DIPHTHONG_BASES.has(base) ? "diphthong" : "vowel";
  if (AFFRICATE_BASES.has(base)) return "affricate";
  if (STOP_BASES.has(base)) return "stop";
  if (FRICATIVE_BASES.has(base)) return "fricative";
  if (NASAL_BASES.has(base)) return "nasal";
  if (APPROXIMANT_LIQUID_BASES.has(base)) return "approximant_liquid";
  return "other";
}

function toPointList(values, type) {
  if (!Array.isArray(values)) return [];
  return values
    .map((ms) => Number(ms))
    .filter((ms) => Number.isFinite(ms))
    .map((ms) => ({ ms, type }));
}

function toSilenceBoundaryPoints(silences) {
  if (!Array.isArray(silences)) return [];
  const points = [];
  for (const silence of silences) {
    const start = Number(silence?.start_ms);
    const end = Number(silence?.end_ms);
    if (Number.isFinite(start)) points.push({ ms: start, type: "silence-start" });
    if (Number.isFinite(end)) points.push({ ms: end, type: "silence-end" });
  }
  return points;
}

function boundaryPointsInRange(landmarks, startMs, endMs) {
  const points = [
    ...toPointList(landmarks?.onsets, "onset"),
    ...toPointList(landmarks?.offsets, "offset"),
    ...toPointList(landmarks?.valleys, "valley"),
    ...toPointList(landmarks?.peaks, "peak"),
    ...toSilenceBoundaryPoints(landmarks?.silences),
  ].filter((point) => point.ms >= startMs && point.ms <= endMs);
  points.sort((left, right) => left.ms - right.ms);
  return points;
}

function nearestBoundaryCandidate(priorBoundaryMs, candidates, preferredTypes, toleranceMs) {
  let best = null;
  let bestCost = Number.POSITIVE_INFINITY;
  for (const candidate of candidates) {
    const distance = Math.abs(candidate.ms - priorBoundaryMs);
    if (distance > toleranceMs) continue;
    const preference = preferredTypes.includes(candidate.type) ? 0.6 : 1;
    const cost = distance * preference;
    if (cost < bestCost) {
      best = candidate;
      bestCost = cost;
    }
  }
  return best;
}

function preferredBoundaryTypes(leftClass, rightClass) {
  if (leftClass === "stop" || rightClass === "stop") return ["onset", "offset", "valley"];
  if (leftClass === "fricative" || rightClass === "fricative") return ["valley", "onset", "offset"];
  if (leftClass === "vowel" || rightClass === "vowel" || rightClass === "diphthong") return ["valley", "peak", "onset"];
  return ["valley", "onset", "offset", "silence-start", "silence-end"];
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normalizeBoundaries(boundaries, startMs, endMs, phoneCount) {
  if (!boundaries.length) return [];
  const duration = Math.max(0.001, endMs - startMs);
  const minDuration = Math.max(0.25, Math.min(1, duration / Math.max(1, phoneCount * 2)));
  const normalized = boundaries.map((boundary, index) => {
    if (index === 0) return startMs;
    if (index === boundaries.length - 1) return endMs;
    return clamp(boundary, startMs, endMs);
  });
  for (let i = 1; i < normalized.length; i++) {
    normalized[i] = Math.max(normalized[i], normalized[i - 1] + minDuration);
  }
  normalized[normalized.length - 1] = endMs;
  for (let i = normalized.length - 2; i >= 0; i--) {
    normalized[i] = Math.min(normalized[i], normalized[i + 1] - minDuration);
  }
  normalized[0] = startMs;
  return normalized;
}

function toVoicingPoints(voicing) {
  if (!Array.isArray(voicing)) return [];
  return voicing
    .map((point) => ({
      ms: Number(point?.time_ms ?? point?.ms),
      voiced:
        point?.voiced === true ||
        Number(point?.f0_hz) > 0 ||
        Number(point?.f0Hz) > 0,
    }))
    .filter((point) => Number.isFinite(point.ms));
}

function voicedRatioInRange(voicingPoints, startMs, endMs) {
  if (!voicingPoints.length) return null;
  const inRange = voicingPoints.filter((point) => point.ms >= startMs && point.ms <= endMs);
  if (!inRange.length) return null;
  const voicedCount = inRange.filter((point) => point.voiced).length;
  return voicedCount / inRange.length;
}

function formantStability(formants, startMs, endMs) {
  if (!Array.isArray(formants)) return null;
  const inRange = formants
    .map((point) => ({
      ms: Number(point?.time_ms ?? point?.ms),
      f1: Number(point?.f1_hz ?? point?.f1Hz),
      f2: Number(point?.f2_hz ?? point?.f2Hz),
    }))
    .filter((point) => Number.isFinite(point.ms) && Number.isFinite(point.f1) && Number.isFinite(point.f2))
    .filter((point) => point.ms >= startMs && point.ms <= endMs);
  if (inRange.length < 2) return null;
  const f1Range = Math.max(...inRange.map((point) => point.f1)) - Math.min(...inRange.map((point) => point.f1));
  const f2Range = Math.max(...inRange.map((point) => point.f2)) - Math.min(...inRange.map((point) => point.f2));
  return { f1Range, f2Range };
}

function spectralBandMeans(level, startMs, endMs) {
  if (!level?.frames?.length || !Number.isFinite(level?.hopMs) || !Number.isFinite(level?.binHz)) {
    return null;
  }
  const startIndex = Math.max(0, Math.floor(startMs / level.hopMs));
  const endIndex = Math.min(level.frames.length - 1, Math.ceil(endMs / level.hopMs));
  if (endIndex < startIndex) return null;

  let lowSum = 0;
  let highSum = 0;
  let bins = 0;
  const lowMaxBin = Math.max(1, Math.floor(1200 / level.binHz));
  const highMinBin = Math.max(1, Math.floor(3000 / level.binHz));
  const highMaxBin = Math.max(highMinBin, Math.floor(7000 / level.binHz));

  for (let frameIndex = startIndex; frameIndex <= endIndex; frameIndex++) {
    const frame = level.frames[frameIndex];
    if (!frame) continue;
    for (let bin = 1; bin <= Math.min(lowMaxBin, frame.length - 1); bin++) {
      lowSum += Number(frame[bin]) || 0;
    }
    for (let bin = Math.min(highMinBin, frame.length - 1); bin <= Math.min(highMaxBin, frame.length - 1); bin++) {
      highSum += Number(frame[bin]) || 0;
    }
    bins += 1;
  }
  if (bins === 0) return null;
  return {
    lowDb: lowSum / (bins * lowMaxBin),
    highDb: highSum / (bins * Math.max(1, highMaxBin - highMinBin + 1)),
  };
}

function acousticEvidenceForPhone({
  span,
  phoneClass,
  landmarks,
  boundaryPoints,
  spectrogramLevel,
  voicingPoints,
  formants,
}) {
  const featuresUsed = ["duration.prior"];
  let confidence = 0.42;
  const peaks = toPointList(landmarks?.peaks, "peak")
    .map((point) => point.ms)
    .filter((ms) => ms >= span.start_ms && ms <= span.end_ms);
  const voicedRatio = voicedRatioInRange(voicingPoints, span.start_ms, span.end_ms);
  const formant = formantStability(formants, span.start_ms, span.end_ms);
  const spectral = spectralBandMeans(spectrogramLevel, span.start_ms, span.end_ms);
  const localBoundaries = boundaryPoints.filter((point) => point.ms >= span.start_ms - 25 && point.ms <= span.end_ms + 25);

  if ((phoneClass === "vowel" || phoneClass === "diphthong") && peaks.length > 0) {
    confidence += 0.18;
    featuresUsed.push("energy.peak_nucleus");
  }
  if ((phoneClass === "vowel" || phoneClass === "approximant_liquid" || phoneClass === "nasal") && voicedRatio !== null) {
    if (voicedRatio >= 0.55) {
      confidence += 0.12;
      featuresUsed.push("voicing.ratio");
    } else {
      confidence -= 0.08;
    }
  }
  if (phoneClass === "fricative" && spectral) {
    if (spectral.highDb > spectral.lowDb + 4) {
      confidence += 0.2;
      featuresUsed.push("spectrogram.high_frequency_noise");
    } else {
      confidence -= 0.06;
    }
  }
  if (phoneClass === "nasal" && spectral) {
    if (spectral.lowDb >= spectral.highDb - 1) {
      confidence += 0.15;
      featuresUsed.push("spectrogram.nasal_murmur_balance");
    } else {
      confidence -= 0.05;
    }
  }
  if ((phoneClass === "vowel" || phoneClass === "approximant_liquid") && formant) {
    if (formant.f1Range < 450 && formant.f2Range < 900) {
      confidence += 0.15;
      featuresUsed.push("formants.stability");
    } else if (phoneClass === "diphthong" && formant.f1Range > 150) {
      confidence += 0.1;
      featuresUsed.push("formants.movement");
    }
  }
  if ((phoneClass === "stop" || phoneClass === "affricate") && localBoundaries.some((point) => point.type === "onset" || point.type === "offset")) {
    confidence += 0.18;
    featuresUsed.push("energy.release_or_closure");
  }

  confidence = clamp(confidence, 0.08, 0.95);
  const profile = PHONE_CLASS_HEURISTICS[phoneClass] ?? PHONE_CLASS_HEURISTICS.other;
  const method = featuresUsed.length > 1 ? profile.defaultMethod : "projected.proportional";
  return { confidence, featuresUsed, method };
}

function scorePronunciationCandidate(candidate, evidence) {
  if (!Array.isArray(candidate?.phonemes) || candidate.phonemes.length === 0) return 0;
  const classes = candidate.phonemes.map((token) => phoneClassFromSymbol(token));
  const vowelCount = classes.filter((kind) => kind === "vowel" || kind === "diphthong").length;
  const fricativeCount = classes.filter((kind) => kind === "fricative").length;
  const stopCount = classes.filter((kind) => kind === "stop" || kind === "affricate").length;
  let score = 0.5;
  if (vowelCount > 0) {
    const peakCount = evidence.peaksInWord;
    score += Math.max(-0.12, Math.min(0.18, (Math.min(peakCount, vowelCount) - Math.abs(peakCount - vowelCount)) * 0.06));
  }
  if (fricativeCount > 0 && evidence.spectralHighBias) {
    score += 0.09;
  }
  if (stopCount > 0 && evidence.onsetOffsetCount > 0) {
    score += 0.08;
  }
  return clamp(score, 0.05, 0.95);
}

function normalizePronunciationCandidates(pronunciationCandidates, fallbackId = "candidate-1") {
  if (!Array.isArray(pronunciationCandidates) || pronunciationCandidates.length === 0) return [];
  return pronunciationCandidates
    .map((candidate, index) => {
      if (Array.isArray(candidate)) {
        return { id: `candidate-${index + 1}`, phonemes: candidate };
      }
      if (candidate && Array.isArray(candidate.phonemes)) {
        return {
          id: candidate.id ?? `candidate-${index + 1}`,
          phonemes: candidate.phonemes,
        };
      }
      return null;
    })
    .filter(Boolean)
    .map((candidate, index) => ({
      id: candidate.id ?? (index === 0 ? fallbackId : `candidate-${index + 1}`),
      phonemes: candidate.phonemes,
    }));
}

/**
 * Segment known pronunciation candidates into phone spans using proportional
 * priors and optional acoustic/spectral evidence.
 *
 * @param {{
 *   word?: string,
 *   wordStartMs: number,
 *   wordEndMs: number,
 *   pronunciationCandidates: Array<string[]|{id?: string, phonemes: string[]}>,
 *   energyLandmarks?: {onsets?: number[], offsets?: number[], valleys?: number[], peaks?: number[], silences?: Array<{start_ms:number,end_ms:number}>},
 *   spectrogram?: {levels?: Array<{id?: string, hopMs: number, binHz: number, frames: ArrayLike<ArrayLike<number>>}>},
 *   formants?: Array<{time_ms?: number, ms?: number, f1_hz?: number, f1Hz?: number, f2_hz?: number, f2Hz?: number}>,
 *   voicing?: Array<{time_ms?: number, ms?: number, voiced?: boolean, f0_hz?: number, f0Hz?: number}>,
 *   allophoneRules?: {enabled?: boolean, language?: string, dialect?: string},
 *   enforceWordBounds?: boolean,
 * }} input
 * @returns {{
 *   word: string|null,
 *   word_start_ms: number,
 *   word_end_ms: number,
 *   pronunciation: string[],
 *   candidate_pronunciation_id: string|null,
 *   pronunciation_scores: Array<{id: string, score: number}>,
 *   phoneSpans: Array<Phoneme & {
 *     phone: string,
 *     phoneClass: string,
 *     prior_start_ms: number,
 *     prior_end_ms: number,
 *     start_ms: number,
 *     end_ms: number,
 *     resolved_start_ms: number,
 *     resolved_end_ms: number,
 *     method: string,
 *     confidence: number,
 *     features_used: string[],
 *     boundary_uncertainty_ms: number,
 *     candidate_pronunciation_id: string|null,
 *   }>
 * }}
 */
export function segmentKnownPronunciationIntoPhoneSpans(input = {}) {
  const wordStartMs = Number(input.wordStartMs);
  const wordEndMs = Number(input.wordEndMs);
  const word = input.word ?? null;
  if (!Number.isFinite(wordStartMs) || !Number.isFinite(wordEndMs) || wordEndMs <= wordStartMs) {
    return {
      word,
      word_start_ms: wordStartMs,
      word_end_ms: wordEndMs,
      pronunciation: [],
      candidate_pronunciation_id: null,
      pronunciation_scores: [],
      phoneSpans: [],
    };
  }

  const candidates = normalizePronunciationCandidates(input.pronunciationCandidates);
  if (!candidates.length) {
    return {
      word,
      word_start_ms: wordStartMs,
      word_end_ms: wordEndMs,
      pronunciation: [],
      candidate_pronunciation_id: null,
      pronunciation_scores: [],
      phoneSpans: [],
    };
  }

  const boundaryPoints = boundaryPointsInRange(input.energyLandmarks, wordStartMs, wordEndMs);
  const spectrogramLevel = input.spectrogram?.levels?.[0] ?? null;
  const voicingPoints = toVoicingPoints(input.voicing);
  const peaksInWord = toPointList(input.energyLandmarks?.peaks, "peak").filter(
    (point) => point.ms >= wordStartMs && point.ms <= wordEndMs,
  ).length;
  const spectralWindow = spectralBandMeans(spectrogramLevel, wordStartMs, wordEndMs);
  const evidence = {
    peaksInWord,
    onsetOffsetCount: [...toPointList(input.energyLandmarks?.onsets, "onset"), ...toPointList(input.energyLandmarks?.offsets, "offset")]
      .filter((point) => point.ms >= wordStartMs && point.ms <= wordEndMs).length,
    spectralHighBias: spectralWindow ? spectralWindow.highDb > spectralWindow.lowDb + 2 : false,
  };

  const pronunciationScores = candidates.map((candidate) => ({
    id: candidate.id,
    score: scorePronunciationCandidate(candidate, evidence),
  }));
  const winner = pronunciationScores[0];
  const selectedCandidate = candidates.find((candidate) => candidate.id === winner.id) ?? candidates[0];

  const priorSpans = projectPhonemesIntoWordInterval(
    selectedCandidate.phonemes,
    wordStartMs,
    wordEndMs,
    "projected.proportional",
    { allophoneRules: input.allophoneRules },
  );
  if (!priorSpans.length) {
    return {
      word,
      word_start_ms: wordStartMs,
      word_end_ms: wordEndMs,
      pronunciation: selectedCandidate.phonemes,
      candidate_pronunciation_id: selectedCandidate.id,
      pronunciation_scores: pronunciationScores,
      phoneSpans: [],
    };
  }

  const boundaries = [wordStartMs];
  const boundaryUncertaintyByIndex = new Array(priorSpans.length).fill(24);
  for (let i = 0; i < priorSpans.length - 1; i++) {
    const leftClass = phoneClassFromSymbol(priorSpans[i].sourceSymbol);
    const rightClass = phoneClassFromSymbol(priorSpans[i + 1].sourceSymbol);
    const priorBoundary = priorSpans[i].end_ms;
    const candidate = nearestBoundaryCandidate(
      priorBoundary,
      boundaryPoints,
      preferredBoundaryTypes(leftClass, rightClass),
      48,
    );
    boundaries.push(candidate ? candidate.ms : priorBoundary);
    boundaryUncertaintyByIndex[i] = candidate ? Math.abs(candidate.ms - priorBoundary) : 32;
    boundaryUncertaintyByIndex[i + 1] = candidate ? Math.abs(candidate.ms - priorBoundary) : boundaryUncertaintyByIndex[i + 1];
  }
  boundaries.push(wordEndMs);
  const normalizedBoundaries = normalizeBoundaries(
    boundaries,
    input.enforceWordBounds === false ? boundaries[0] : wordStartMs,
    input.enforceWordBounds === false ? boundaries[boundaries.length - 1] : wordEndMs,
    priorSpans.length,
  );

  const phoneSpans = priorSpans.map((span, index) => {
    const phoneClass = phoneClassFromSymbol(span.sourceSymbol ?? span.symbol);
    const start = normalizedBoundaries[index];
    const end = normalizedBoundaries[index + 1];
    const evidenceForPhone = acousticEvidenceForPhone({
      span: { ...span, start_ms: start, end_ms: end },
      phoneClass,
      landmarks: input.energyLandmarks,
      boundaryPoints,
      spectrogramLevel,
      voicingPoints,
      formants: input.formants,
    });
    return {
      ...span,
      phone: span.realization?.ipa ?? span.defaultPhoneString?.[0]?.ipa ?? "?",
      phoneClass,
      prior_start_ms: span.start_ms,
      prior_end_ms: span.end_ms,
      start_ms: start,
      end_ms: end,
      resolved_start_ms: start,
      resolved_end_ms: end,
      method: evidenceForPhone.method,
      confidence: evidenceForPhone.confidence,
      features_used: evidenceForPhone.featuresUsed,
      boundary_uncertainty_ms: Math.max(1, boundaryUncertaintyByIndex[index] ?? 24),
      candidate_pronunciation_id: selectedCandidate.id,
      timingSource: evidenceForPhone.method,
    };
  });

  return {
    word,
    word_start_ms: wordStartMs,
    word_end_ms: wordEndMs,
    pronunciation: selectedCandidate.phonemes,
    candidate_pronunciation_id: selectedCandidate.id,
    pronunciation_scores: pronunciationScores,
    phoneSpans,
  };
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
