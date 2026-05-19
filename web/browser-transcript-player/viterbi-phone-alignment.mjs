/**
 * @fileoverview WaveDeck rendering utilities for Viterbi-resolved phone spans.
 *
 * Converts `pronunciation_alignment` SpanHypothesis values (produced by the
 * Rust Viterbi aligner) into labelled IPA regions on the WaveDeck
 * spectrogram/waveform canvas.
 *
 * The provenance shape emitted by the Rust `viterbi_align_pronunciation`
 * function includes:
 *   - method: "viterbi.fused_heuristic" | "viterbi.proportional_fallback"
 *   - path_score: 0.0–1.0
 *   - emission_evidence: array of per-frame evidence objects
 *   - boundary_evidence: { start, end }
 *   - conflicts: array of conflict objects
 *   - alternative_boundaries: proportional-fallback alternative or null
 *
 * Exports:
 *   - phoneSpansFromHypotheses(hypotheses)
 *   - drawViterbiPhoneSpans(ctx, metrics, spans, msToCanvasX)
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * @typedef {Object} PhoneSpan
 * @property {string}  label          - IPA symbol or ARPAbet symbol
 * @property {number}  startMs        - span start in milliseconds
 * @property {number}  endMs          - span end in milliseconds
 * @property {number}  confidence     - 0.0–1.0
 * @property {string}  method         - e.g. "viterbi.fused_heuristic"
 * @property {number}  pathScore      - overall Viterbi path score
 * @property {Array}   emissionEvidence
 * @property {Object}  boundaryEvidence
 * @property {Array}   conflicts
 * @property {Object|null} alternativeBoundary
 * @property {string}  source         - original hypothesis source
 */

// ---------------------------------------------------------------------------
// phoneSpansFromHypotheses
// ---------------------------------------------------------------------------

/**
 * Convert an array of `pronunciation_alignment` SpanHypothesis objects into
 * PhoneSpan records suitable for rendering.
 *
 * Hypotheses that are not `pronunciation_alignment` kind are silently ignored.
 *
 * @param {Array<{
 *   kind: string,
 *   label: string,
 *   startMs: number,
 *   endMs: number,
 *   confidence: number,
 *   score: number,
 *   source: string,
 *   provenance: Object,
 * }>} hypotheses
 * @returns {PhoneSpan[]}
 */
export function phoneSpansFromHypotheses(hypotheses) {
  if (!Array.isArray(hypotheses)) return [];
  return hypotheses
    .filter((h) => h.kind === "pronunciation_alignment")
    .map((h) => {
      const prov = h.provenance ?? {};
      return {
        label: h.label,
        startMs: h.startMs,
        endMs: h.endMs,
        confidence: h.confidence ?? 0.0,
        method: prov.method ?? "viterbi.proportional_fallback",
        pathScore: prov.path_score ?? 0.0,
        emissionEvidence: prov.emission_evidence ?? [],
        boundaryEvidence: prov.boundary_evidence ?? { start: null, end: null },
        conflicts: prov.conflicts ?? [],
        alternativeBoundary: prov.alternative_boundaries ?? null,
        source: h.source,
      };
    });
}

// ---------------------------------------------------------------------------
// Colour helpers
// ---------------------------------------------------------------------------

/**
 * Return an RGBA fill colour for a phone span based on its method and
 * confidence.
 *
 * Viterbi-aligned spans use a cool blue; proportional fallback spans use a
 * muted amber to signal reduced confidence.
 *
 * @param {PhoneSpan} span
 * @returns {string} CSS colour string
 */
function phoneSpanFillColour(span) {
  const alpha = (0.25 + span.confidence * 0.45).toFixed(2);
  if (span.method === "viterbi.fused_heuristic") {
    return `rgba(80, 160, 240, ${alpha})`;
  }
  // proportional fallback
  return `rgba(200, 160, 60, ${alpha})`;
}

/**
 * Return a border colour for a phone span.
 *
 * Spans with unresolved conflicts get an orange tint; others follow fill.
 *
 * @param {PhoneSpan} span
 * @returns {string}
 */
function phoneSpanBorderColour(span) {
  if (Array.isArray(span.conflicts) && span.conflicts.length > 0) {
    return "rgba(255, 140, 0, 0.85)";
  }
  if (span.method === "viterbi.fused_heuristic") {
    return "rgba(40, 120, 200, 0.90)";
  }
  return "rgba(160, 120, 40, 0.80)";
}

// ---------------------------------------------------------------------------
// drawViterbiPhoneSpans
// ---------------------------------------------------------------------------

/**
 * Draw Viterbi-resolved phone spans as labelled IPA regions on a Canvas 2D
 * context.
 *
 * Each span is rendered as:
 *  - a filled semi-transparent rectangle spanning [startMs, endMs]
 *  - a border whose colour indicates confidence / conflict
 *  - an IPA label centred inside the rectangle
 *
 * @param {CanvasRenderingContext2D}  ctx
 * @param {{ cssHeight: number, phoneRowTop?: number, phoneRowHeight?: number }} metrics
 * @param {PhoneSpan[]}              spans
 * @param {function(number): number} msToCanvasX - converts ms to canvas px X
 */
export function drawViterbiPhoneSpans(ctx, metrics, spans, msToCanvasX) {
  if (!ctx || !Array.isArray(spans) || spans.length === 0) return;

  const rowTop = metrics.phoneRowTop ?? Math.round(metrics.cssHeight * 0.68);
  const rowHeight = metrics.phoneRowHeight ?? Math.round(metrics.cssHeight * 0.28);
  const rowBottom = rowTop + rowHeight;

  ctx.save();

  for (const span of spans) {
    const x0 = msToCanvasX(span.startMs);
    const x1 = msToCanvasX(span.endMs);
    const width = Math.max(x1 - x0, 1);

    // Fill.
    ctx.fillStyle = phoneSpanFillColour(span);
    ctx.fillRect(x0, rowTop, width, rowHeight);

    // Border.
    ctx.strokeStyle = phoneSpanBorderColour(span);
    ctx.lineWidth = 1.0;
    ctx.strokeRect(x0, rowTop, width, rowHeight);

    // IPA label — only draw if there is enough horizontal space.
    if (width >= 6) {
      const fontSize = Math.min(rowHeight * 0.55, 13);
      ctx.font = `${fontSize}px serif`;
      ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      // Clip the label to the span width so it never bleeds into adjacent spans.
      ctx.save();
      ctx.beginPath();
      ctx.rect(x0, rowTop, width, rowHeight);
      ctx.clip();
      ctx.fillText(span.label, x0 + width / 2, rowTop + rowHeight / 2);
      ctx.restore();
    }
  }

  ctx.restore();
}
