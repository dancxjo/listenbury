import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  HypothesisLattice,
  fuseHypotheses,
  weightedConfidence,
  renderAlternativeHypothesisChips,
  drawHypothesisConfidenceHeatmap,
} from "./hypothesis-lattice.mjs";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let _nextId = 1;
function nextId() {
  return `test-hyp-${_nextId++}`;
}

function makeWordCandidate(label, startMs, endMs, confidence) {
  return {
    id: nextId(),
    kind: "word_candidate",
    label,
    startMs,
    endMs,
    score: confidence,
    confidence,
    source: "manual",
    featuresUsed: [],
    status: "provisional",
    provenance: null,
  };
}

function makeBoundary(label, ms, confidence, source = "endpoint_detector") {
  return {
    id: nextId(),
    kind: "speech_boundary",
    label,
    startMs: ms,
    endMs: ms,
    score: confidence,
    confidence,
    source,
    featuresUsed: ["energy.onset"],
    status: "provisional",
    provenance: null,
  };
}

// ---------------------------------------------------------------------------
// HypothesisLattice – structure
// ---------------------------------------------------------------------------

describe("HypothesisLattice – structure", () => {
  it("starts empty", () => {
    const lattice = new HypothesisLattice();
    assert.equal(lattice.activeHypotheses().length, 0);
    assert.equal(lattice.allHypotheses().length, 0);
    assert.equal(lattice.edges.length, 0);
  });

  it("competing word candidates can coexist", () => {
    const lattice = new HypothesisLattice();
    lattice.add(makeWordCandidate("testing", 1000, 1300, 0.72));
    lattice.add(makeWordCandidate("texting", 1000, 1300, 0.19));
    lattice.add(makeWordCandidate("test in", 1000, 1300, 0.07));
    assert.equal(lattice.activeHypotheses().length, 3);
    assert.equal(lattice.allHypotheses().length, 3);
  });

  it("add returns the hypothesis id", () => {
    const lattice = new HypothesisLattice();
    const h = makeWordCandidate("testing", 1000, 1300, 0.72);
    const id = lattice.add(h);
    assert.equal(id, h.id);
  });

  it("hypotheses can be connected with typed edges", () => {
    const lattice = new HypothesisLattice();
    const h1 = makeWordCandidate("testing", 1000, 1300, 0.72);
    const h2 = makeBoundary("speech_start", 1000, 0.65);
    lattice.add(h1);
    lattice.add(h2);
    lattice.connect(h2.id, h1.id, "supports", 0.8);
    const edges = lattice.edgesFrom(h2.id);
    assert.equal(edges.length, 1);
    assert.equal(edges[0].kind, "supports");
    assert.equal(edges[0].to, h1.id);
  });

  it("hypotheses can contradict each other", () => {
    const lattice = new HypothesisLattice();
    const b1 = makeBoundary("speech_start_asr", 1000, 0.8, "manual");
    const b2 = makeBoundary("speech_start_energy", 1050, 0.7, "endpoint_detector");
    lattice.add(b1);
    lattice.add(b2);
    lattice.connect(b1.id, b2.id, "contradicts", 1.0);
    const edge = lattice.edgesFrom(b1.id)[0];
    assert.equal(edge.kind, "contradicts");
  });
});

// ---------------------------------------------------------------------------
// HypothesisLattice – revision
// ---------------------------------------------------------------------------

describe("HypothesisLattice – revision", () => {
  it("revise marks old hypothesis as revised", () => {
    const lattice = new HypothesisLattice();
    const h1 = makeWordCandidate("testing", 1000, 1300, 0.72);
    lattice.add(h1);
    const h2 = makeWordCandidate("texting", 1000, 1300, 0.85);
    lattice.revise(h1.id, h2);
    const old = lattice.allHypotheses().find((h) => h.id === h1.id);
    assert.equal(old.status, "revised");
  });

  it("revised hypothesis remains in allHypotheses", () => {
    const lattice = new HypothesisLattice();
    const h1 = makeWordCandidate("testing", 1000, 1300, 0.72);
    lattice.add(h1);
    const h2 = makeWordCandidate("texting", 1000, 1300, 0.85);
    lattice.revise(h1.id, h2);
    assert.equal(lattice.allHypotheses().length, 2);
  });

  it("active hypotheses excludes revised entries", () => {
    const lattice = new HypothesisLattice();
    const h1 = makeWordCandidate("testing", 1000, 1300, 0.72);
    lattice.add(h1);
    const h2 = makeWordCandidate("texting", 1000, 1300, 0.85);
    lattice.revise(h1.id, h2);
    const active = lattice.activeHypotheses();
    assert.equal(active.length, 1);
    assert.equal(active[0].label, "texting");
  });

  it("revise adds a revision_of edge from new to old", () => {
    const lattice = new HypothesisLattice();
    const h1 = makeWordCandidate("testing", 1000, 1300, 0.72);
    lattice.add(h1);
    const h2 = makeWordCandidate("texting", 1000, 1300, 0.85);
    const newId = lattice.revise(h1.id, h2);
    const edges = lattice.edgesFrom(newId);
    assert.equal(edges.length, 1);
    assert.equal(edges[0].kind, "revision_of");
    assert.equal(edges[0].to, h1.id);
  });

  it("multiple revisions chain correctly", () => {
    const lattice = new HypothesisLattice();
    const h1 = makeWordCandidate("testing", 1000, 1300, 0.40);
    lattice.add(h1);
    const h2 = makeWordCandidate("testing", 1000, 1300, 0.60);
    const id2 = lattice.revise(h1.id, h2);
    const h3 = makeWordCandidate("testing", 1000, 1300, 0.80);
    lattice.revise(id2, h3);
    assert.equal(lattice.allHypotheses().length, 3);
    assert.equal(lattice.activeHypotheses().length, 1);
    assert.equal(lattice.activeHypotheses()[0].confidence, 0.80);
  });
});

// ---------------------------------------------------------------------------
// weightedConfidence
// ---------------------------------------------------------------------------

describe("weightedConfidence", () => {
  it("returns 0 for empty input", () => {
    assert.equal(weightedConfidence({}), 0);
  });

  it("returns 1.0 when all signals are 1.0", () => {
    const result = weightedConfidence({
      asrConfidence: 1.0,
      energyAlignmentQuality: 1.0,
      phoneSegmentationAgreement: 1.0,
      pronunciationFit: 1.0,
      spectralEvidence: 1.0,
      prosodyConsistency: 1.0,
      timingCoherence: 1.0,
      mechanicalRecognizerScore: 1.0,
    });
    assert.ok(Math.abs(result - 1.0) < 1e-5);
  });

  it("ignores undefined signals", () => {
    const r1 = weightedConfidence({ asrConfidence: 0.8 });
    assert.ok(r1 > 0 && r1 <= 1);
    assert.ok(Math.abs(r1 - 0.8) < 1e-5); // only one signal present
  });

  it("ASR confidence has the highest weight", () => {
    const highAsr = weightedConfidence({ asrConfidence: 1.0, spectralEvidence: 0.0 });
    const lowAsr = weightedConfidence({ asrConfidence: 0.0, spectralEvidence: 1.0 });
    assert.ok(highAsr > lowAsr);
  });

  it("result is clamped to [0, 1]", () => {
    const r = weightedConfidence({ asrConfidence: 2.0 });
    assert.ok(r <= 1.0);
    assert.ok(r >= 0.0);
  });
});

// ---------------------------------------------------------------------------
// fuseHypotheses
// ---------------------------------------------------------------------------

describe("fuseHypotheses", () => {
  it("returns null for an empty lattice", () => {
    const lattice = new HypothesisLattice();
    assert.equal(fuseHypotheses(lattice), null);
  });

  it("resolves the highest confidence candidate", () => {
    const lattice = new HypothesisLattice();
    lattice.add(makeWordCandidate("testing", 1000, 1300, 0.72));
    lattice.add(makeWordCandidate("texting", 1000, 1300, 0.19));
    lattice.add(makeWordCandidate("test in", 1000, 1300, 0.07));
    const result = fuseHypotheses(lattice);
    assert.equal(result.resolved.label, "testing");
    assert.ok(result.confidence > 0.5);
  });

  it("strong external evidence can flip the winner", () => {
    const lattice = new HypothesisLattice();
    const hLow = makeWordCandidate("texting", 1000, 1300, 0.19);
    const hHigh = makeWordCandidate("testing", 1000, 1300, 0.72);
    lattice.add(hLow);
    lattice.add(hHigh);

    const evidence = new Map([
      [
        hLow.id,
        {
          asrConfidence: 0.95,
          energyAlignmentQuality: 0.90,
          mechanicalRecognizerScore: 0.88,
        },
      ],
    ]);
    const result = fuseHypotheses(lattice, evidence);
    assert.equal(result.resolved.label, "texting");
  });

  it("conflicting boundary candidates are classified correctly", () => {
    const lattice = new HypothesisLattice();
    const b1 = makeBoundary("speech_start_asr", 1000, 0.8, "manual");
    const b2 = makeBoundary("speech_start_energy", 1000, 0.7, "endpoint_detector");
    lattice.add(b1);
    lattice.add(b2);
    lattice.connect(b1.id, b2.id, "contradicts", 1.0);
    const result = fuseHypotheses(lattice);
    assert.ok(result.conflictingIds.length > 0);
    assert.ok(!result.conflictingSummary.includes("no conflicting"));
  });

  it("explicit supports edge marks candidate as supporting", () => {
    const lattice = new HypothesisLattice();
    // Non-overlapping spans: one is a boundary, one is a word.
    const word = makeWordCandidate("testing", 1000, 1300, 0.72);
    const boundary = makeBoundary("speech_start", 1000, 0.65);
    boundary.endMs = 1000; // point boundary, no overlap with word [1000,1300]? actually it does overlap
    // Make boundary non-overlapping to avoid the fallback conflict rule
    boundary.startMs = 500;
    boundary.endMs = 500;
    lattice.add(word);
    lattice.add(boundary);
    lattice.connect(boundary.id, word.id, "supports", 0.8);
    const result = fuseHypotheses(lattice);
    assert.equal(result.resolved.label, "testing");
    assert.ok(result.supportingIds.includes(boundary.id));
  });

  it("result has provenance with fusion strategy", () => {
    const lattice = new HypothesisLattice();
    lattice.add(makeWordCandidate("testing", 1000, 1300, 0.72));
    const result = fuseHypotheses(lattice);
    assert.equal(result.provenance.fusion, "first_pass_weighted_average");
    assert.ok(result.provenance.candidateCount >= 1);
  });

  it("provenance records all candidate scores", () => {
    const lattice = new HypothesisLattice();
    lattice.add(makeWordCandidate("testing", 1000, 1300, 0.72));
    lattice.add(makeWordCandidate("texting", 1000, 1300, 0.19));
    const result = fuseHypotheses(lattice);
    assert.equal(result.provenance.scores.length, 2);
  });

  it("resolved span has updated confidence", () => {
    const lattice = new HypothesisLattice();
    const h = makeWordCandidate("testing", 1000, 1300, 0.72);
    lattice.add(h);
    const result = fuseHypotheses(lattice);
    assert.equal(result.resolved.confidence, result.confidence);
  });

  it("overlapping spans without explicit edge are treated as conflicting", () => {
    const lattice = new HypothesisLattice();
    // Both span [1000, 1300] — temporal overlap → conflict proxy
    lattice.add(makeWordCandidate("testing", 1000, 1300, 0.72));
    lattice.add(makeWordCandidate("texting", 1000, 1300, 0.19));
    const result = fuseHypotheses(lattice);
    assert.ok(result.conflictingIds.length > 0);
  });

  it("non-overlapping spans without explicit edge are supporting", () => {
    const lattice = new HypothesisLattice();
    const w = makeWordCandidate("testing", 1000, 1300, 0.72);
    const b = makeBoundary("speech_start", 800, 0.65); // ms=800, non-overlapping
    b.startMs = 800;
    b.endMs = 800;
    lattice.add(w);
    lattice.add(b);
    const result = fuseHypotheses(lattice);
    assert.ok(result.supportingIds.includes(b.id));
  });
});

// ---------------------------------------------------------------------------
// drawHypothesisConfidenceHeatmap
// ---------------------------------------------------------------------------

describe("drawHypothesisConfidenceHeatmap", () => {
  function makeMockCtx() {
    const ops = [];
    return {
      ops,
      save: () => ops.push("save"),
      restore: () => ops.push("restore"),
      fillRect: (x, y, w, h) => ops.push(`fillRect(${x},${y},${w},${h})`),
      set fillStyle(_) {},
    };
  }

  it("does nothing when lattice is empty", () => {
    const ctx = makeMockCtx();
    const lattice = new HypothesisLattice();
    drawHypothesisConfidenceHeatmap(ctx, { cssHeight: 100 }, lattice, null, (ms) => ms);
    assert.ok(!ctx.ops.includes("fillRect"));
  });

  it("draws a heatmap band for each active hypothesis", () => {
    const ctx = makeMockCtx();
    const lattice = new HypothesisLattice();
    lattice.add(makeWordCandidate("testing", 100, 300, 0.72));
    lattice.add(makeWordCandidate("texting", 100, 300, 0.19));
    drawHypothesisConfidenceHeatmap(ctx, { cssHeight: 100 }, lattice, null, (ms) => ms);
    const fillCalls = ctx.ops.filter((op) => op.startsWith("fillRect"));
    assert.equal(fillCalls.length, 2);
  });
});
