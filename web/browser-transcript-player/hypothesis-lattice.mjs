/**
 * @fileoverview Hypothesis lattice and first-pass fusion layer for WaveDeck.
 *
 * This module mirrors the Rust lattice in src/audio/lattice.rs and provides
 * a client-side representation of competing span hypotheses together with a
 * first-pass fusion scorer and visualisation helpers for the WaveDeck UI.
 *
 * Multiple competing interpretations (word candidates, boundary candidates,
 * pronunciations, phone segmentations) can coexist in the lattice and be
 * fused into a single resolved span with provenance.
 */

// ---------------------------------------------------------------------------
// Typedefs
// ---------------------------------------------------------------------------

/**
 * @typedef {"supports"|"contradicts"|"refines"|"contains"|"aligned_to"|"derived_from"|"revision_of"} EdgeKind
 *
 * @typedef {Object} HypothesisEdge
 * @property {string}   from    - source hypothesis id
 * @property {string}   to      - target hypothesis id
 * @property {EdgeKind} kind
 * @property {number}   weight  - 0.0–1.0
 *
 * @typedef {Object} FusionInput
 * @property {number|undefined} asrConfidence
 * @property {number|undefined} energyAlignmentQuality
 * @property {number|undefined} phoneSegmentationAgreement
 * @property {number|undefined} pronunciationFit
 * @property {number|undefined} spectralEvidence
 * @property {number|undefined} prosodyConsistency
 * @property {number|undefined} timingCoherence
 * @property {number|undefined} mechanicalRecognizerScore
 *
 * @typedef {Object} FusionResult
 * @property {import('./mechanical-asr.mjs').SpanHypothesis} resolved
 * @property {number}   confidence
 * @property {string[]} supportingIds
 * @property {string[]} conflictingIds
 * @property {string}   supportingSummary
 * @property {string}   conflictingSummary
 * @property {Object}   provenance
 */

// ---------------------------------------------------------------------------
// HypothesisLattice
// ---------------------------------------------------------------------------

/**
 * A graph of competing and collaborating span hypotheses.
 *
 * Hypotheses are never deleted; instead their status is updated to "revised"
 * or "rejected" so the full revision history remains inspectable.
 */
export class HypothesisLattice {
  constructor() {
    /** @type {import('./mechanical-asr.mjs').SpanHypothesis[]} */
    this.hypotheses = [];
    /** @type {HypothesisEdge[]} */
    this.edges = [];
  }

  /**
   * Add a hypothesis and return its id.
   * @param {import('./mechanical-asr.mjs').SpanHypothesis} hypothesis
   * @returns {string} the hypothesis id
   */
  add(hypothesis) {
    this.hypotheses.push(hypothesis);
    return hypothesis.id;
  }

  /**
   * Connect two hypotheses with a typed, weighted edge.
   * @param {string}   from
   * @param {string}   to
   * @param {EdgeKind} kind
   * @param {number}   [weight=1.0]
   */
  connect(from, to, kind, weight = 1.0) {
    this.edges.push({ from, to, kind, weight });
  }

  /**
   * Mark an existing hypothesis as revised and add the replacement.
   *
   * A "revision_of" edge is added from the new hypothesis to the old one.
   * @param {string} oldId
   * @param {import('./mechanical-asr.mjs').SpanHypothesis} revised
   * @returns {string} the new hypothesis id
   */
  revise(oldId, revised) {
    const old = this.hypotheses.find((h) => h.id === oldId);
    if (old) old.status = "revised";
    const newId = revised.id;
    this.hypotheses.push(revised);
    this.edges.push({ from: newId, to: oldId, kind: "revision_of", weight: 1.0 });
    return newId;
  }

  /**
   * Return only hypotheses that are currently active (not revised/rejected).
   * @returns {import('./mechanical-asr.mjs').SpanHypothesis[]}
   */
  activeHypotheses() {
    return this.hypotheses.filter(
      (h) => h.status !== "revised" && h.status !== "rejected"
    );
  }

  /**
   * Return all hypotheses, including superseded / revised ones.
   * @returns {import('./mechanical-asr.mjs').SpanHypothesis[]}
   */
  allHypotheses() {
    return this.hypotheses;
  }

  /**
   * Return all edges that originate from a given hypothesis id.
   * @param {string} id
   * @returns {HypothesisEdge[]}
   */
  edgesFrom(id) {
    return this.edges.filter((e) => e.from === id);
  }

  /**
   * Return all edges that point to a given hypothesis id.
   * @param {string} id
   * @returns {HypothesisEdge[]}
   */
  edgesTo(id) {
    return this.edges.filter((e) => e.to === id);
  }
}

// ---------------------------------------------------------------------------
// FusionInput helpers
// ---------------------------------------------------------------------------

/**
 * Compute a weighted average over the available evidence signals.
 *
 * Weights are heuristic; they can be tuned in follow-up work.
 *
 * @param {FusionInput} input
 * @returns {number} confidence in 0.0–1.0
 */
export function weightedConfidence(input) {
  const signals = [
    [input.asrConfidence, 3.0],
    [input.energyAlignmentQuality, 1.5],
    [input.phoneSegmentationAgreement, 1.0],
    [input.pronunciationFit, 1.0],
    [input.spectralEvidence, 0.75],
    [input.prosodyConsistency, 0.5],
    [input.timingCoherence, 1.25],
    [input.mechanicalRecognizerScore, 1.0],
  ];
  let totalWeight = 0;
  let weightedSum = 0;
  for (const [value, weight] of signals) {
    if (value != null) {
      weightedSum += value * weight;
      totalWeight += weight;
    }
  }
  if (totalWeight === 0) return 0;
  return Math.min(1, Math.max(0, weightedSum / totalWeight));
}

// ---------------------------------------------------------------------------
// First-pass fusion scorer
// ---------------------------------------------------------------------------

/**
 * Score each competing candidate in `lattice` using the provided `evidence`
 * map and return the best {@link FusionResult}.
 *
 * `evidence` is a Map from hypothesis id → {@link FusionInput}.
 *
 * Returns `null` when the lattice has no active hypotheses.
 *
 * @param {HypothesisLattice}      lattice
 * @param {Map<string,FusionInput>} [evidence]
 * @returns {FusionResult|null}
 */
export function fuseHypotheses(lattice, evidence = new Map()) {
  const actives = lattice.activeHypotheses();
  if (actives.length === 0) return null;

  // External evidence is weighted 3× more than the hypothesis's own confidence
  // because it aggregates multiple independent signals.
  const EXTERNAL_WEIGHT = 3.0;

  const scored = actives.map((hyp) => {
    const extra = evidence.has(hyp.id)
      ? weightedConfidence(evidence.get(hyp.id))
      : 0;
    const fused =
      extra > 0
        ? Math.min(1, (hyp.confidence + extra * EXTERNAL_WEIGHT) / (1 + EXTERNAL_WEIGHT))
        : hyp.confidence;
    return { hyp, fusedConfidence: fused };
  });

  // Sort descending by fused confidence.
  scored.sort((a, b) => b.fusedConfidence - a.fusedConfidence);

  const best = scored[0];
  const rest = scored.slice(1);

  const supportingIds = [];
  const conflictingIds = [];
  const supportingLabels = [];
  const conflictingLabels = [];

  for (const other of rest) {
    const edge = lattice.edges.find(
      (e) =>
        (e.from === best.hyp.id && e.to === other.hyp.id) ||
        (e.from === other.hyp.id && e.to === best.hyp.id)
    );

    let isConflicting;
    if (edge?.kind === "contradicts") {
      isConflicting = true;
    } else if (edge?.kind === "supports" || edge?.kind === "aligned_to") {
      isConflicting = false;
    } else {
      // No explicit edge: treat temporal overlap as a conflict proxy.
      isConflicting =
        other.hyp.startMs < best.hyp.endMs && other.hyp.endMs > best.hyp.startMs;
    }

    if (isConflicting) {
      conflictingIds.push(other.hyp.id);
      conflictingLabels.push(`${other.hyp.label} (${other.fusedConfidence.toFixed(2)})`);
    } else {
      supportingIds.push(other.hyp.id);
      supportingLabels.push(`${other.hyp.label} (${other.fusedConfidence.toFixed(2)})`);
    }
  }

  const resolved = { ...best.hyp, confidence: best.fusedConfidence };

  return {
    resolved,
    confidence: best.fusedConfidence,
    supportingIds,
    conflictingIds,
    supportingSummary:
      supportingLabels.length === 0
        ? "no supporting candidates"
        : supportingLabels.join(", "),
    conflictingSummary:
      conflictingLabels.length === 0
        ? "no conflicting candidates"
        : conflictingLabels.join(", "),
    provenance: {
      fusion: "first_pass_weighted_average",
      evidenceSources: evidence.size,
      candidateCount: scored.length,
      scores: scored.map((s) => ({
        id: s.hyp.id,
        label: s.hyp.label,
        sourceConfidence: s.hyp.confidence,
        fusedConfidence: s.fusedConfidence,
      })),
    },
  };
}

// ---------------------------------------------------------------------------
// WaveDeck visualisation helpers
// ---------------------------------------------------------------------------

/**
 * Render alternative hypothesis chips onto a WaveDeck lane element.
 *
 * For each hypothesis in the lattice that is NOT the resolved best candidate,
 * a "ghost chip" (low-opacity, dashed-border span element) is inserted above
 * the normal word chip row so the user can see competing interpretations.
 *
 * @param {HTMLElement}      container  - the lane's chip container element
 * @param {FusionResult}     result     - output of {@link fuseHypotheses}
 * @param {HypothesisLattice} lattice   - the full lattice (for superseded hyps)
 * @param {function(number): number} msToPercent - converts ms to a CSS left %
 * @param {function(number, number): number} durationToWidth - ms duration → CSS width %
 */
export function renderAlternativeHypothesisChips(
  container,
  result,
  lattice,
  msToPercent,
  durationToWidth
) {
  if (!container || !result) return;

  // Remove any previously rendered alternative chips.
  for (const old of container.querySelectorAll(".alt-hypothesis-chip")) {
    old.remove();
  }

  const allActive = lattice.activeHypotheses();
  for (const hyp of allActive) {
    if (hyp.id === result.resolved.id) continue; // skip the resolved winner

    const isConflicting = result.conflictingIds.includes(hyp.id);
    const chip = document.createElement("span");
    chip.className = "alt-hypothesis-chip";
    chip.dataset.hypothesisId = hyp.id;
    chip.dataset.kind = hyp.kind;
    chip.dataset.label = hyp.label;
    chip.dataset.confidence = String(hyp.confidence);
    chip.dataset.conflicting = isConflicting ? "true" : "false";
    chip.textContent = hyp.label;
    chip.title =
      `${hyp.label} (${hyp.source}) — confidence: ${hyp.confidence.toFixed(2)}\n` +
      (isConflicting ? "⚡ conflicts with resolved span" : "🔵 alternative candidate");

    const leftPct = msToPercent(hyp.startMs);
    const widthPct = durationToWidth(hyp.startMs, hyp.endMs);
    chip.style.cssText = [
      `position:absolute`,
      `left:${leftPct}%`,
      `width:${Math.max(widthPct, 0.5)}%`,
      `opacity:${(hyp.confidence * 0.55 + 0.1).toFixed(2)}`,
      `border:1.5px dashed ${isConflicting ? "#ff6b6b" : "#74b9ff"}`,
      `border-radius:3px`,
      `font-size:0.7em`,
      `padding:1px 3px`,
      `background:${isConflicting ? "rgba(255,107,107,0.08)" : "rgba(116,185,255,0.08)"}`,
      `color:${isConflicting ? "#d63031" : "#0984e3"}`,
      `pointer-events:auto`,
      `cursor:help`,
      `white-space:nowrap`,
      `overflow:hidden`,
      `text-overflow:ellipsis`,
      `box-sizing:border-box`,
    ].join(";");

    container.appendChild(chip);
  }
}

/**
 * Draw a confidence heatmap overlay onto a Canvas 2D context showing the
 * relative confidence of all active hypotheses in the lattice.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {{ cssHeight: number }}   metrics
 * @param {HypothesisLattice}       lattice
 * @param {FusionResult|null}       result   - if null, all hyps get equal weight
 * @param {function(number): number} msToCanvasX
 */
export function drawHypothesisConfidenceHeatmap(
  ctx,
  metrics,
  lattice,
  result,
  msToCanvasX
) {
  if (!ctx) return;
  const actives = lattice.activeHypotheses();
  if (actives.length === 0) return;

  const resolvedId = result?.resolved?.id;
  const height = metrics.cssHeight;
  const bandHeight = Math.max(4, height * 0.06);
  const y = height - bandHeight - 1;

  ctx.save();
  for (const hyp of actives) {
    const x0 = msToCanvasX(hyp.startMs);
    const x1 = msToCanvasX(hyp.endMs);
    const w = Math.max(2, Math.abs(x1 - x0));
    const isResolved = hyp.id === resolvedId;
    const alpha = (hyp.confidence * 0.6 + 0.1).toFixed(2);
    ctx.fillStyle = isResolved
      ? `rgba(0,200,100,${alpha})`
      : `rgba(100,160,255,${alpha})`;
    ctx.fillRect(Math.min(x0, x1), y, w, bandHeight);
  }
  ctx.restore();
}
