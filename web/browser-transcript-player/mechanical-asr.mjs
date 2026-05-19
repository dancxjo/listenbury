/**
 * @fileoverview Mechanical ASR hypothesis utilities for WaveDeck.
 *
 * Converts pre-computed acoustic energy-landmark objects into lightweight
 * SpanHypothesis values that can be rendered as diagnostic overlays on the
 * waveform / spectrogram timeline.
 *
 * These are client-side JS equivalents of the Rust generators in
 * src/audio/boundary.rs and src/audio/phone_class.rs.  They operate on the
 * same JSON shape that the Rust server returns in the acoustic analysis
 * response, so no extra round-trips are needed.
 */

// ---------------------------------------------------------------------------
// SpanHypothesis shape
// ---------------------------------------------------------------------------

/**
 * @typedef {"speech_boundary"|"pause_candidate"|"phone_class_candidate"|"template_match"|"pronunciation_alignment"} HypothesisKind
 * @typedef {"endpoint_detector"|"phone_classifier"|"dtw_template_matcher"|"viterbi_alignment"|"manual"} HypothesisSource
 * @typedef {"provisional"|"revised"|"confirmed"|"rejected"} HypothesisStatus
 *
 * @typedef {Object} SpanHypothesis
 * @property {string}            id
 * @property {HypothesisKind}    kind
 * @property {string}            label
 * @property {number}            startMs
 * @property {number}            endMs
 * @property {number}            score        - raw score from the generator
 * @property {number}            confidence   - 0.0–1.0 normalised confidence
 * @property {HypothesisSource}  source
 * @property {string[]}          featuresUsed
 * @property {HypothesisStatus}  status
 * @property {*}                 provenance
 */

let _nextId = 1;
function nextId() {
  return `mech-${_nextId++}`;
}

// ---------------------------------------------------------------------------
// Boundary hypothesis generator
// ---------------------------------------------------------------------------

/**
 * Convert energy-landmark data (as returned by /api/live-session-acoustic.json
 * or the AcousticAnalysis JSON) into SpanHypothesis objects.
 *
 * @param {Object}  landmarks
 * @param {number[]}  [landmarks.onsets]   - onset timestamps in ms
 * @param {number[]}  [landmarks.offsets]  - offset timestamps in ms
 * @param {Array<{start_ms:number,end_ms:number}>} [landmarks.silences]
 * @param {number[]}  [landmarks.valleys]  - energy-valley timestamps in ms
 * @returns {SpanHypothesis[]}
 */
export function boundaryHypothesesFromLandmarks(landmarks) {
  if (!landmarks) return [];
  const hyps = [];

  for (const ms of landmarks.onsets ?? []) {
    hyps.push({
      id: nextId(),
      kind: "speech_boundary",
      label: "speech_start",
      startMs: ms,
      endMs: ms,
      score: 0.75,
      confidence: 0.65,
      source: "endpoint_detector",
      featuresUsed: ["energy.onset"],
      status: "provisional",
      provenance: { type: "onset", ms },
    });
  }

  for (const ms of landmarks.offsets ?? []) {
    hyps.push({
      id: nextId(),
      kind: "speech_boundary",
      label: "speech_end",
      startMs: ms,
      endMs: ms,
      score: 0.70,
      confidence: 0.60,
      source: "endpoint_detector",
      featuresUsed: ["energy.offset"],
      status: "provisional",
      provenance: { type: "offset", ms },
    });
  }

  for (const silence of landmarks.silences ?? []) {
    const durationMs = (silence.end_ms ?? 0) - (silence.start_ms ?? 0);
    hyps.push({
      id: nextId(),
      kind: "pause_candidate",
      label: "pause",
      startMs: silence.start_ms ?? 0,
      endMs: silence.end_ms ?? 0,
      score: 0.80,
      confidence: durationMs >= 400 ? 0.88 : durationMs >= 150 ? 0.75 : 0.55,
      source: "endpoint_detector",
      featuresUsed: ["energy.silence"],
      status: "provisional",
      provenance: { type: "silence", start_ms: silence.start_ms, end_ms: silence.end_ms, durationMs },
    });
  }

  for (const ms of landmarks.valleys ?? []) {
    hyps.push({
      id: nextId(),
      kind: "speech_boundary",
      label: "boundary_candidate",
      startMs: ms,
      endMs: ms,
      score: 0.45,
      confidence: 0.40,
      source: "endpoint_detector",
      featuresUsed: ["energy.valley"],
      status: "provisional",
      provenance: { type: "valley", ms },
    });
  }

  return hyps;
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

/**
 * Colour map for boundary hypothesis labels.
 * @param {string} label
 * @returns {string} CSS colour string
 */
function boundaryColour(label) {
  switch (label) {
    case "speech_start":
      return "rgba(103, 224, 116, 0.90)";
    case "speech_end":
      return "rgba(255, 118, 117, 0.90)";
    case "boundary_candidate":
      return "rgba(255, 211, 74, 0.85)";
    default:
      return "rgba(180, 180, 255, 0.80)";
  }
}

/**
 * Draw mechanical boundary-candidate hypotheses onto a Canvas 2D context.
 *
 * `msToCanvasX` must be a function that converts a millisecond timestamp
 * to a pixel X position on the canvas.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {{ cssHeight: number }}    metrics
 * @param {SpanHypothesis[]}         hypotheses
 * @param {function(number): number} msToCanvasX
 */
export function drawBoundaryHypotheses(ctx, metrics, hypotheses, msToCanvasX) {
  if (!ctx || !Array.isArray(hypotheses) || hypotheses.length === 0) return;

  const bottom = metrics.cssHeight - 1;
  const top = Math.max(0, bottom - metrics.cssHeight * 0.22);

  ctx.save();
  ctx.lineWidth = 1.5;

  for (const hyp of hypotheses) {
    if (hyp.kind === "pause_candidate") {
      const x0 = msToCanvasX(hyp.startMs);
      const x1 = msToCanvasX(hyp.endMs);
      ctx.fillStyle = "rgba(200, 200, 255, 0.08)";
      ctx.fillRect(Math.min(x0, x1), top, Math.abs(x1 - x0), bottom - top);
      continue;
    }
    // Point boundary
    const x = msToCanvasX(hyp.startMs);
    ctx.strokeStyle = boundaryColour(hyp.label);
    // Dashed line to distinguish from energy-snapping markers
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.restore();
}

// ---------------------------------------------------------------------------
// Tests (exported for use with node --test)
// ---------------------------------------------------------------------------

if (typeof process !== "undefined" && process?.env?.NODE_TEST_CONTEXT) {
  // Tests are in a separate file; nothing to run here.
}
