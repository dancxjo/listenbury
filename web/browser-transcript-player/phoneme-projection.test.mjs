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
  isVowel,
  projectPhonemesIntoWordInterval,
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
  assert.equal(spans[0].symbol, "AH1");
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
  assert.equal(spans[2].symbol, "IY1");
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
    assert.ok(span.start_ms >= 5985, `${span.symbol} start ${span.start_ms} < 5985`);
    assert.ok(span.end_ms <= 6342, `${span.symbol} end ${span.end_ms} > 6342`);
  }
});

test("default source label is cmudict.proportional", () => {
  const spans = projectPhonemesIntoWordInterval(["K"], 0, 100);
  assert.equal(spans[0].source, "cmudict.proportional");
});

test("custom source label is forwarded", () => {
  const spans = projectPhonemesIntoWordInterval(["K"], 0, 100, "energy.assisted");
  assert.equal(spans[0].source, "energy.assisted");
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
