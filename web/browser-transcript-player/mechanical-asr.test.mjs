import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  boundaryHypothesesFromLandmarks,
  drawBoundaryHypotheses,
} from "./mechanical-asr.mjs";

describe("boundaryHypothesesFromLandmarks", () => {
  it("returns empty array for null landmarks", () => {
    assert.deepEqual(boundaryHypothesesFromLandmarks(null), []);
  });

  it("returns empty array for empty landmarks", () => {
    const hyps = boundaryHypothesesFromLandmarks({
      onsets: [],
      offsets: [],
      silences: [],
      valleys: [],
    });
    assert.equal(hyps.length, 0);
  });

  it("generates a speech_start hypothesis for each onset", () => {
    const hyps = boundaryHypothesesFromLandmarks({ onsets: [100, 500] });
    const starts = hyps.filter((h) => h.label === "speech_start");
    assert.equal(starts.length, 2);
    assert.equal(starts[0].startMs, 100);
    assert.equal(starts[1].startMs, 500);
  });

  it("speech_start hypothesis has correct fields", () => {
    const [h] = boundaryHypothesesFromLandmarks({ onsets: [200] });
    assert.equal(h.kind, "speech_boundary");
    assert.equal(h.label, "speech_start");
    assert.equal(h.startMs, 200);
    assert.equal(h.endMs, 200);
    assert.equal(h.source, "endpoint_detector");
    assert.equal(h.status, "provisional");
    assert.ok(h.confidence > 0 && h.confidence <= 1);
    assert.ok(Array.isArray(h.featuresUsed) && h.featuresUsed.length > 0);
    assert.ok(typeof h.id === "string");
  });

  it("generates a speech_end hypothesis for each offset", () => {
    const hyps = boundaryHypothesesFromLandmarks({ offsets: [800] });
    assert.equal(hyps.length, 1);
    assert.equal(hyps[0].label, "speech_end");
    assert.equal(hyps[0].startMs, 800);
  });

  it("generates a pause_candidate for each silence", () => {
    const hyps = boundaryHypothesesFromLandmarks({
      silences: [{ start_ms: 1000, end_ms: 1400 }],
    });
    assert.equal(hyps.length, 1);
    const h = hyps[0];
    assert.equal(h.kind, "pause_candidate");
    assert.equal(h.label, "pause");
    assert.equal(h.startMs, 1000);
    assert.equal(h.endMs, 1400);
  });

  it("long pause gets higher confidence than short pause", () => {
    const [long] = boundaryHypothesesFromLandmarks({
      silences: [{ start_ms: 0, end_ms: 500 }],
    });
    const [short] = boundaryHypothesesFromLandmarks({
      silences: [{ start_ms: 0, end_ms: 50 }],
    });
    assert.ok(long.confidence > short.confidence);
  });

  it("generates a boundary_candidate for each valley", () => {
    const hyps = boundaryHypothesesFromLandmarks({ valleys: [750] });
    assert.equal(hyps.length, 1);
    assert.equal(hyps[0].label, "boundary_candidate");
  });

  it("each hypothesis has a unique id", () => {
    const hyps = boundaryHypothesesFromLandmarks({
      onsets: [100, 200],
      offsets: [300],
    });
    const ids = hyps.map((h) => h.id);
    const unique = new Set(ids);
    assert.equal(unique.size, ids.length);
  });

  it("mixed landmarks produce correct total count", () => {
    const hyps = boundaryHypothesesFromLandmarks({
      onsets: [100, 600],
      offsets: [400, 900],
      valleys: [200, 700],
      silences: [{ start_ms: 1000, end_ms: 1200 }],
    });
    assert.equal(hyps.length, 7);
  });
});

describe("drawBoundaryHypotheses", () => {
  function makeMockCtx() {
    const ops = [];
    return {
      ops,
      save: () => ops.push("save"),
      restore: () => ops.push("restore"),
      beginPath: () => ops.push("beginPath"),
      moveTo: (x, y) => ops.push(`moveTo(${x},${y})`),
      lineTo: (x, y) => ops.push(`lineTo(${x},${y})`),
      stroke: () => ops.push("stroke"),
      fillRect: (x, y, w, h) => ops.push(`fillRect(${x},${y},${w},${h})`),
      setLineDash: (arr) => ops.push(`setLineDash(${JSON.stringify(arr)})`),
      set strokeStyle(_) {},
      set lineWidth(_) {},
      set fillStyle(_) {},
    };
  }

  it("does nothing when hypotheses is empty", () => {
    const ctx = makeMockCtx();
    drawBoundaryHypotheses(ctx, { cssHeight: 100 }, [], (ms) => ms);
    // Only save/restore should have been called, with nothing in between.
    assert.ok(!ctx.ops.includes("stroke"));
  });

  it("draws a pause_candidate as fillRect", () => {
    const ctx = makeMockCtx();
    const hyp = {
      id: "1",
      kind: "pause_candidate",
      label: "pause",
      startMs: 100,
      endMs: 300,
      score: 0.8,
      confidence: 0.75,
      source: "endpoint_detector",
      featuresUsed: [],
      status: "provisional",
      provenance: {},
    };
    drawBoundaryHypotheses(ctx, { cssHeight: 80 }, [hyp], (ms) => ms);
    assert.ok(ctx.ops.some((op) => op.startsWith("fillRect(")));
  });

  it("draws a speech_start as a vertical line (stroke)", () => {
    const ctx = makeMockCtx();
    const hyp = {
      id: "2",
      kind: "speech_boundary",
      label: "speech_start",
      startMs: 200,
      endMs: 200,
      score: 0.75,
      confidence: 0.65,
      source: "endpoint_detector",
      featuresUsed: [],
      status: "provisional",
      provenance: {},
    };
    drawBoundaryHypotheses(ctx, { cssHeight: 80 }, [hyp], (ms) => ms);
    assert.ok(ctx.ops.includes("stroke"));
  });
});
