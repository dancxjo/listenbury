/**
 * viterbi-phone-alignment.test.mjs
 *
 * Tests for the Viterbi phone-alignment WaveDeck rendering utilities.
 *
 * Run with:
 *   node --test web/browser-transcript-player/viterbi-phone-alignment.test.mjs
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  phoneSpansFromHypotheses,
  drawViterbiPhoneSpans,
} from "./viterbi-phone-alignment.mjs";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/** Build a minimal pronunciation_alignment SpanHypothesis. */
function makeHyp(label, startMs, endMs, confidence = 0.72, method = "viterbi.fused_heuristic") {
  return {
    id: `hyp-${label}`,
    kind: "pronunciation_alignment",
    label,
    startMs,
    endMs,
    score: confidence,
    confidence,
    source: "viterbi_alignment",
    featuresUsed: ["viterbi.forced_alignment"],
    status: "provisional",
    provenance: {
      phone: label,
      phone_class: "vowel",
      method,
      assigned_frames: 3,
      word_start_ms: startMs,
      word_end_ms: endMs,
      path_score: confidence,
      emission_evidence: [
        {
          frame_start_ms: startMs,
          detected_class: "vowel_or_sonorant",
          features: ["zcr.low", "energy.voiced"],
          match_score: confidence,
        },
      ],
      boundary_evidence: {
        start: { frame_start_ms: startMs, rms_energy: 0.06, spectral_flux: 0.04, zcr: 0.05 },
        end: { frame_end_ms: endMs, rms_energy: 0.05, spectral_flux: 0.03, zcr: 0.06 },
      },
      conflicts: [],
      alternative_boundaries: { start_ms: startMs + 10, end_ms: endMs - 10, method: "viterbi.proportional_fallback", score: 0.30 },
    },
  };
}

/** Create a minimal mock Canvas 2D context that records operations. */
function makeMockCtx() {
  const ops = [];
  return {
    ops,
    save: () => ops.push("save"),
    restore: () => ops.push("restore"),
    beginPath: () => ops.push("beginPath"),
    rect: (x, y, w, h) => ops.push(`rect(${x},${y},${w},${h})`),
    clip: () => ops.push("clip"),
    fillRect: (x, y, w, h) => ops.push(`fillRect(${x},${y},${w},${h})`),
    strokeRect: (x, y, w, h) => ops.push(`strokeRect(${x},${y},${w},${h})`),
    fillText: (t, x, y) => ops.push(`fillText(${t},${x},${y})`),
    set fillStyle(_) {},
    set strokeStyle(_) {},
    set lineWidth(_) {},
    set font(_) {},
    set textAlign(_) {},
    set textBaseline(_) {},
  };
}

// ---------------------------------------------------------------------------
// phoneSpansFromHypotheses
// ---------------------------------------------------------------------------

describe("phoneSpansFromHypotheses", () => {
  it("returns empty array for null input", () => {
    assert.deepEqual(phoneSpansFromHypotheses(null), []);
  });

  it("returns empty array for empty array", () => {
    assert.deepEqual(phoneSpansFromHypotheses([]), []);
  });

  it("ignores hypotheses with non-pronunciation_alignment kind", () => {
    const hyps = [
      { kind: "speech_boundary", label: "onset", startMs: 0, endMs: 0, confidence: 0.7, source: "endpoint_detector", provenance: {} },
    ];
    assert.equal(phoneSpansFromHypotheses(hyps).length, 0);
  });

  it("converts pronunciation_alignment hypotheses to phone spans", () => {
    const hyps = [
      makeHyp("θ", 5985, 6078),
      makeHyp("ɹ", 6078, 6154),
      makeHyp("iː", 6154, 6342),
    ];
    const spans = phoneSpansFromHypotheses(hyps);
    assert.equal(spans.length, 3);
    assert.equal(spans[0].label, "θ");
    assert.equal(spans[1].label, "ɹ");
    assert.equal(spans[2].label, "iː");
  });

  it("extracts startMs and endMs correctly", () => {
    const hyps = [makeHyp("θ", 5985, 6078)];
    const [span] = phoneSpansFromHypotheses(hyps);
    assert.equal(span.startMs, 5985);
    assert.equal(span.endMs, 6078);
  });

  it("extracts method from provenance", () => {
    const hyps = [makeHyp("θ", 100, 200, 0.72, "viterbi.fused_heuristic")];
    const [span] = phoneSpansFromHypotheses(hyps);
    assert.equal(span.method, "viterbi.fused_heuristic");
  });

  it("defaults method to proportional_fallback when provenance is absent", () => {
    const hyp = {
      kind: "pronunciation_alignment",
      label: "θ",
      startMs: 100,
      endMs: 200,
      confidence: 0.30,
      source: "viterbi_alignment",
      provenance: null,
    };
    const [span] = phoneSpansFromHypotheses([hyp]);
    assert.equal(span.method, "viterbi.proportional_fallback");
  });

  it("extracts path_score from provenance", () => {
    const hyps = [makeHyp("θ", 100, 200, 0.65)];
    const [span] = phoneSpansFromHypotheses(hyps);
    assert.ok(span.pathScore >= 0.0 && span.pathScore <= 1.0);
  });

  it("extracts emissionEvidence array", () => {
    const hyps = [makeHyp("θ", 100, 200)];
    const [span] = phoneSpansFromHypotheses(hyps);
    assert.ok(Array.isArray(span.emissionEvidence));
    assert.equal(span.emissionEvidence.length, 1);
    assert.ok("detected_class" in span.emissionEvidence[0]);
  });

  it("extracts boundaryEvidence object", () => {
    const hyps = [makeHyp("θ", 100, 200)];
    const [span] = phoneSpansFromHypotheses(hyps);
    assert.ok(span.boundaryEvidence !== null);
    assert.ok("start" in span.boundaryEvidence);
    assert.ok("end" in span.boundaryEvidence);
  });

  it("extracts conflicts array", () => {
    const hyp = {
      ...makeHyp("s", 100, 200),
      provenance: {
        ...makeHyp("s", 100, 200).provenance,
        conflicts: [{ frame_start_ms: 110, detected_class: "silence_noise", expected_class: "fricative" }],
      },
    };
    const [span] = phoneSpansFromHypotheses([hyp]);
    assert.equal(span.conflicts.length, 1);
    assert.equal(span.conflicts[0].detected_class, "silence_noise");
  });

  it("extracts alternativeBoundary from provenance", () => {
    const hyps = [makeHyp("θ", 100, 200)];
    const [span] = phoneSpansFromHypotheses(hyps);
    assert.ok(span.alternativeBoundary !== null);
    assert.ok("start_ms" in span.alternativeBoundary);
    assert.ok("end_ms" in span.alternativeBoundary);
  });

  // ---- Vowel-heavy word: "audio" = [ɔː, d, iː, oʊ] --------------------

  it("vowel-heavy word spans are all converted correctly", () => {
    const hyps = [
      makeHyp("ɔː", 0, 30),
      makeHyp("d", 30, 40),
      makeHyp("iː", 40, 60),
      makeHyp("oʊ", 60, 80),
    ];
    const spans = phoneSpansFromHypotheses(hyps);
    assert.equal(spans.length, 4);
    // Verify monotonic ordering is preserved.
    for (let i = 1; i < spans.length; i++) {
      assert.ok(spans[i].startMs >= spans[i - 1].startMs, "spans out of order");
    }
  });

  // ---- Fricative + vowel: "see" = [s, iː] ------------------------------

  it("fricative plus vowel spans are converted with correct labels", () => {
    const hyps = [
      makeHyp("s", 1000, 1030, 0.72, "viterbi.fused_heuristic"),
      makeHyp("iː", 1030, 1060, 0.72, "viterbi.fused_heuristic"),
    ];
    const spans = phoneSpansFromHypotheses(hyps);
    assert.equal(spans.length, 2);
    assert.equal(spans[0].label, "s");
    assert.equal(spans[1].label, "iː");
  });

  // ---- Stop + vowel: "key" = [k, iː] -----------------------------------

  it("stop plus vowel spans are converted with correct method", () => {
    const hyps = [
      makeHyp("k", 2000, 2020, 0.68, "viterbi.fused_heuristic"),
      makeHyp("iː", 2020, 2050, 0.72, "viterbi.fused_heuristic"),
    ];
    const spans = phoneSpansFromHypotheses(hyps);
    assert.equal(spans.length, 2);
    assert.equal(spans[0].method, "viterbi.fused_heuristic");
  });

  // ---- Noisy fallback --------------------------------------------------

  it("proportional fallback spans have fallback method", () => {
    const fallbackHyp = {
      ...makeHyp("θ", 0, 100, 0.30, "viterbi.proportional_fallback"),
    };
    const [span] = phoneSpansFromHypotheses([fallbackHyp]);
    assert.equal(span.method, "viterbi.proportional_fallback");
  });

  it("source field is preserved", () => {
    const hyps = [makeHyp("θ", 100, 200)];
    const [span] = phoneSpansFromHypotheses(hyps);
    assert.equal(span.source, "viterbi_alignment");
  });
});

// ---------------------------------------------------------------------------
// drawViterbiPhoneSpans
// ---------------------------------------------------------------------------

describe("drawViterbiPhoneSpans", () => {
  const metrics = { cssHeight: 100 };
  const msToX = (ms) => ms / 10; // 1px per 10ms

  it("does nothing when spans array is empty", () => {
    const ctx = makeMockCtx();
    drawViterbiPhoneSpans(ctx, metrics, [], msToX);
    assert.ok(!ctx.ops.some((op) => op.startsWith("fillRect")));
  });

  it("does nothing when ctx is null", () => {
    // Should not throw.
    assert.doesNotThrow(() => drawViterbiPhoneSpans(null, metrics, [makeHyp("θ", 100, 200)], msToX));
  });

  it("draws a fillRect for each span", () => {
    const ctx = makeMockCtx();
    const spans = phoneSpansFromHypotheses([
      makeHyp("θ", 100, 200),
      makeHyp("iː", 200, 350),
    ]);
    drawViterbiPhoneSpans(ctx, metrics, spans, msToX);
    const fillRects = ctx.ops.filter((op) => op.startsWith("fillRect"));
    assert.equal(fillRects.length, 2);
  });

  it("draws a strokeRect for each span", () => {
    const ctx = makeMockCtx();
    const spans = phoneSpansFromHypotheses([makeHyp("s", 0, 300)]);
    drawViterbiPhoneSpans(ctx, metrics, spans, msToX);
    assert.ok(ctx.ops.some((op) => op.startsWith("strokeRect")));
  });

  it("draws a text label for each span wide enough", () => {
    const ctx = makeMockCtx();
    const spans = phoneSpansFromHypotheses([makeHyp("iː", 0, 1000)]);
    drawViterbiPhoneSpans(ctx, metrics, spans, msToX);
    assert.ok(ctx.ops.some((op) => op.startsWith("fillText(iː,")));
  });

  it("saves and restores ctx state", () => {
    const ctx = makeMockCtx();
    const spans = phoneSpansFromHypotheses([makeHyp("θ", 0, 100)]);
    drawViterbiPhoneSpans(ctx, metrics, spans, msToX);
    assert.ok(ctx.ops.includes("save"));
    assert.ok(ctx.ops.includes("restore"));
  });

  it("accepts custom phoneRowTop and phoneRowHeight via metrics", () => {
    const ctx = makeMockCtx();
    const customMetrics = { cssHeight: 200, phoneRowTop: 50, phoneRowHeight: 30 };
    const spans = phoneSpansFromHypotheses([makeHyp("k", 0, 500)]);
    drawViterbiPhoneSpans(ctx, customMetrics, spans, msToX);
    // fillRect should use the custom row top.
    const fillRects = ctx.ops.filter((op) => op.startsWith("fillRect("));
    assert.ok(fillRects.length > 0);
    const parts = fillRects[0].replace("fillRect(", "").replace(")", "").split(",");
    const y = Number(parts[1]);
    assert.equal(y, 50);
  });
});
