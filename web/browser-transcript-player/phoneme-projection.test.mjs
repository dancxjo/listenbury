/**
 * phoneme-projection.test.mjs
 *
 * Tests for the phoneme projection module.
 *
 * Run with:
 *   node --test web/browser-transcript-player/phoneme-projection.test.mjs
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
  defaultPhoneStringAsString,
  isVowel,
  phonemeFromArpabet,
  projectPhonemesIntoWordInterval,
  realizePhonemeSequence,
  stressPattern,
} from "./phoneme-projection.mjs";

// ---------------------------------------------------------------------------
// isVowel
// ---------------------------------------------------------------------------

test("isVowel returns true for plain vowel base", () => {
  assert.ok(isVowel("AH"));
  assert.ok(isVowel("IY"));
  assert.ok(isVowel("OW"));
  assert.ok(isVowel("EY"));
});

test("isVowel strips trailing stress digit before checking", () => {
  assert.ok(isVowel("AH0"));
  assert.ok(isVowel("IY1"));
  assert.ok(isVowel("OW2"));
});

test("isVowel returns false for consonants", () => {
  assert.ok(!isVowel("TH"));
  assert.ok(!isVowel("R"));
  assert.ok(!isVowel("K"));
  assert.ok(!isVowel("D"));
});

// ---------------------------------------------------------------------------
// projectPhonemesIntoWordInterval
// ---------------------------------------------------------------------------

test("returns empty array for empty phoneme list", () => {
  const spans = projectPhonemesIntoWordInterval([], 0, 300);
  assert.deepEqual(spans, []);
});

test("returns empty array when endMs <= startMs", () => {
  const spans = projectPhonemesIntoWordInterval(["TH", "R", "IY1"], 300, 300);
  assert.deepEqual(spans, []);
  const spans2 = projectPhonemesIntoWordInterval(["K"], 500, 400);
  assert.deepEqual(spans2, []);
});

test("single phoneme spans the full interval", () => {
  const spans = projectPhonemesIntoWordInterval(["AH1"], 100, 400);
  assert.equal(spans.length, 1);
  assert.equal(spans[0].start_ms, 100);
  assert.equal(spans[0].end_ms, 400);
  assert.equal(spans[0].symbol, "AH");
  assert.equal(spans[0].sourceSymbol, "AH1");
});

test("TH R IY1 projects proportionally with vowel weighting", () => {
  // TH = consonant (weight 1), R = consonant (weight 1), IY1 = vowel (weight 2)
  // Total weight = 4; duration = 300 ms
  // TH: 75ms, R: 75ms, IY1: 150ms
  const spans = projectPhonemesIntoWordInterval(["TH", "R", "IY1"], 6010, 6310);
  assert.equal(spans.length, 3);
  assert.equal(spans[0].symbol, "TH");
  assert.equal(spans[0].start_ms, 6010);
  // TH gets 75ms (300 * 1/4)
  assert.equal(spans[0].end_ms, 6085); // 6010 + 75
  assert.equal(spans[1].symbol, "R");
  assert.equal(spans[1].start_ms, 6085);
  assert.equal(spans[1].end_ms, 6160); // 6085 + 75
  assert.equal(spans[2].symbol, "IY");
  assert.equal(spans[2].sourceSymbol, "IY1");
  assert.equal(spans[2].start_ms, 6160);
  // Last phoneme always snaps to endMs.
  assert.equal(spans[2].end_ms, 6310);
});

test("spans preserve monotonic ordering", () => {
  const phonemes = ["D", "AA1", "K", "T", "ER0"]; // DOCTOR
  const spans = projectPhonemesIntoWordInterval(phonemes, 0, 500);
  for (let i = 1; i < spans.length; i++) {
    assert.ok(
      spans[i].start_ms >= spans[i - 1].end_ms,
      `span ${i} starts before previous ends`,
    );
  }
});

test("last span ends exactly at endMs", () => {
  const phonemes = ["F", "IH0", "TS", "JH", "EH1", "R", "AH0", "L", "D"]; // FITZGERALD
  const spans = projectPhonemesIntoWordInterval(phonemes, 1000, 1357);
  const last = spans[spans.length - 1];
  assert.equal(last.end_ms, 1357);
});

test("first span starts exactly at startMs", () => {
  const spans = projectPhonemesIntoWordInterval(["HH", "AH0", "L", "OW1"], 200, 600);
  assert.equal(spans[0].start_ms, 200);
});

test("all spans stay within [startMs, endMs]", () => {
  const phonemes = ["Z", "AY1", "L", "AH0", "F", "OW2", "N"]; // XYLOPHONE
  const spans = projectPhonemesIntoWordInterval(phonemes, 5985, 6342);
  for (const span of spans) {
    assert.ok(span.start_ms >= 5985, `${span.sourceSymbol} start ${span.start_ms} < 5985`);
    assert.ok(span.end_ms <= 6342, `${span.sourceSymbol} end ${span.end_ms} > 6342`);
  }
});

test("default source label is cmudict.proportional", () => {
  const spans = projectPhonemesIntoWordInterval(["K"], 0, 100);
  assert.equal(spans[0].timingSource, "cmudict.proportional");
});

test("custom source label is forwarded", () => {
  const spans = projectPhonemesIntoWordInterval(["K"], 0, 100, "energy.assisted");
  assert.equal(spans[0].timingSource, "energy.assisted");
});

test("maps ARPABET symbols to default IPA with stress metadata preserved", () => {
  const phoneme = phonemeFromArpabet("IY1");
  assert.equal(phoneme.symbol, "IY");
  assert.equal(phoneme.sourceSymbol, "IY1");
  assert.equal(phoneme.stress, "primary");
  assert.deepEqual(phoneme.defaultPhoneString, [{ ipa: "iː", sourceSymbol: "IY1", status: "mapped" }]);
  assert.equal(phoneme.realization.ipa, "iː");
  assert.equal(phoneme.realization.method, "default");
});

test("unknown ARPABET symbols are explicit and safe", () => {
  const phoneme = phonemeFromArpabet("QH9");
  assert.equal(phoneme.symbol, "QH9");
  assert.equal(phoneme.stress, null);
  assert.equal(phoneme.defaultPhoneString[0].ipa, "?QH9");
  assert.equal(phoneme.defaultPhoneString[0].status, "unknown_symbol");
});

test("applies intervocalic flapping allophone rule by default", () => {
  const base = ["AE1", "T", "ER0"].map((token) => phonemeFromArpabet(token));
  const realized = realizePhonemeSequence(base, { dialect: "american_english" });
  assert.equal(realized[1].symbol, "T");
  assert.equal(realized[1].realization.ipa, "ɾ");
  assert.equal(realized[1].realization.method, "allophone_rule");
  assert.equal(realized[1].realization.rule, "american_english_intervocalic_flapping");
  assert.equal(
    realized[1].realization.environment?.stress_context,
    "between stressed vowel and unstressed vowel",
  );
});

test("can disable allophone rules explicitly", () => {
  const spans = projectPhonemesIntoWordInterval(["AE1", "T", "ER0"], 0, 300, "cmudict.proportional", {
    allophoneRules: { enabled: false },
  });
  assert.equal(spans[1].realization.ipa, "t");
  assert.equal(spans[1].realization.method, "default");
});

test("projected spans preserve default and realized IPA separately", () => {
  const spans = projectPhonemesIntoWordInterval(
    ["AE1", "T", "ER0"],
    0,
    300,
    "cmudict.proportional",
    { allophoneRules: { enabled: true } },
  );
  assert.equal(spans[1].sourceSymbol, "T");
  assert.equal(spans[1].defaultPhoneString[0].ipa, "t");
  assert.equal(spans[1].realization.ipa, "ɾ");
});

test("defaultPhoneStringAsString prefers explicit as_string before phone IPA fallback", () => {
  assert.equal(
    defaultPhoneStringAsString({ as_string: "tʃ", phones: [{ ipa: "ignored" }] }),
    "tʃ",
  );
  assert.equal(
    defaultPhoneStringAsString([{ ipa: "a" }, { ipa: "ɪ" }]),
    "aɪ",
  );
  assert.equal(defaultPhoneStringAsString(null), "");
});

// ---------------------------------------------------------------------------
// stressPattern
// ---------------------------------------------------------------------------

test("stressPattern extracts vowel stress digits", () => {
  // TH=consonant, R=consonant, IY1=vowel(1)
  assert.equal(stressPattern(["TH", "R", "IY1"]), "1");
});

test("stressPattern for multi-vowel word", () => {
  // OW1 K EY1 (OKAY)
  assert.equal(stressPattern(["OW1", "K", "EY1"]), "11");
});

test("stressPattern defaults missing stress digit to 0", () => {
  // A vowel without a stress digit defaults to 0
  assert.equal(stressPattern(["AH"]), "0");
});

test("stressPattern returns empty string for all-consonant phonemes", () => {
  assert.equal(stressPattern(["TH", "R", "K"]), "");
});
