//! Hypothesis lattice and first-pass fusion layer.
//!
//! The lattice collects competing [`SpanHypothesis`] values from multiple
//! evidence sources and lets them support or contradict each other via
//! typed [`HypothesisEdge`]s.  A first-pass [`fuse_hypotheses`] scorer
//! combines the evidence into a [`FusionResult`] with a resolved candidate,
//! confidence, and provenance.
//!
//! [`SpeechHypothesisEngine`] is the first-class top-level fusion pipeline. It
//! composes multiple evidence sources (acoustic, phonetic/pronunciation, ASR
//! stability, visual speech), standardizes confidence handling, and produces
//! stable/revisable span partitions with inspectable debug traces.
//!
//! ## Design
//!
//! The lattice is append-only for hypotheses.  Existing hypotheses can have
//! their status updated (e.g. marked [`HypothesisStatus::Revised`]) but are
//! never removed so that provenance is preserved.  This satisfies the
//! requirement that superseded hypotheses remain inspectable.
//!
//! The fusion scorer is deliberately simple: it performs a weighted average
//! over the available evidence signals, with heuristic weights that can be
//! tuned in a follow-up.  The scorer does **not** require all signals to be
//! present; it uses only the signals that were provided.

use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

use crate::audio::hypothesis::{HypothesisStatus, SpanHypothesis, SpanHypothesisId};

// ---------------------------------------------------------------------------
// Edge kind
// ---------------------------------------------------------------------------

/// The semantic relationship between two hypotheses in the lattice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HypothesisEdgeKind {
    /// One hypothesis provides supporting evidence for another.
    Supports,
    /// One hypothesis is inconsistent with / contradicts another.
    Contradicts,
    /// One hypothesis is a more precise version of another.
    Refines,
    /// One hypothesis fully contains the span of another.
    Contains,
    /// Two hypotheses are temporally aligned (same or very similar timing).
    AlignedTo,
    /// One hypothesis is derived from another by a deterministic transform.
    DerivedFrom,
    /// One hypothesis supersedes / revises another (the target is now stale).
    RevisionOf,
}

// ---------------------------------------------------------------------------
// Edge
// ---------------------------------------------------------------------------

/// A directed edge between two hypotheses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisEdge {
    /// Source hypothesis identifier.
    pub from: SpanHypothesisId,
    /// Target hypothesis identifier.
    pub to: SpanHypothesisId,
    /// Semantic kind of the relationship.
    pub kind: HypothesisEdgeKind,
    /// Optional scalar weight on the edge (0.0–1.0).
    pub weight: f32,
}

// ---------------------------------------------------------------------------
// Lattice
// ---------------------------------------------------------------------------

/// A graph of competing and collaborating span hypotheses.
///
/// Hypotheses are never deleted from the lattice; instead their
/// [`HypothesisStatus`] is updated to `Revised` or `Rejected` so the full
/// revision history remains inspectable.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HypothesisLattice {
    /// All hypotheses ever added to this lattice (including superseded ones).
    pub hypotheses: Vec<SpanHypothesis>,
    /// Directed edges between hypotheses.
    pub edges: Vec<HypothesisEdge>,
}

impl HypothesisLattice {
    /// Create an empty lattice.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a hypothesis and return a clone of its identifier.
    pub fn add(&mut self, hypothesis: SpanHypothesis) -> SpanHypothesisId {
        let id = hypothesis.id.clone();
        self.hypotheses.push(hypothesis);
        id
    }

    /// Connect two hypotheses with a typed, weighted edge.
    pub fn connect(
        &mut self,
        from: SpanHypothesisId,
        to: SpanHypothesisId,
        kind: HypothesisEdgeKind,
        weight: f32,
    ) {
        self.edges.push(HypothesisEdge {
            from,
            to,
            kind,
            weight,
        });
    }

    /// Mark an existing hypothesis as revised and add the replacement.
    ///
    /// The old hypothesis is updated to [`HypothesisStatus::Revised`] and
    /// a [`HypothesisEdgeKind::RevisionOf`] edge is added from the new one
    /// to the old one so the full history remains inspectable.
    pub fn revise(
        &mut self,
        old_id: &SpanHypothesisId,
        revised: SpanHypothesis,
    ) -> SpanHypothesisId {
        if let Some(old) = self.hypotheses.iter_mut().find(|h| &h.id == old_id) {
            old.status = HypothesisStatus::Revised;
        }
        let old_id = old_id.clone();
        let new_id = revised.id.clone();
        self.hypotheses.push(revised);
        self.edges.push(HypothesisEdge {
            from: new_id.clone(),
            to: old_id,
            kind: HypothesisEdgeKind::RevisionOf,
            weight: 1.0,
        });
        new_id
    }

    /// Return only hypotheses that are currently active (not revised/rejected).
    pub fn active_hypotheses(&self) -> Vec<&SpanHypothesis> {
        self.hypotheses
            .iter()
            .filter(|h| {
                h.status != HypothesisStatus::Revised && h.status != HypothesisStatus::Rejected
            })
            .collect()
    }

    /// Return all hypotheses, including superseded / revised ones.
    pub fn all_hypotheses(&self) -> &[SpanHypothesis] {
        &self.hypotheses
    }

    /// Return all edges that originate from a given hypothesis id.
    pub fn edges_from(&self, id: &SpanHypothesisId) -> Vec<&HypothesisEdge> {
        self.edges.iter().filter(|e| &e.from == id).collect()
    }

    /// Return all edges that point to a given hypothesis id.
    pub fn edges_to(&self, id: &SpanHypothesisId) -> Vec<&HypothesisEdge> {
        self.edges.iter().filter(|e| &e.to == id).collect()
    }
}

// ---------------------------------------------------------------------------
// Fusion input
// ---------------------------------------------------------------------------

/// Evidence signals fed into the first-pass fusion scorer.
///
/// All fields are optional; the scorer weights only the signals that are
/// actually provided.  This makes it possible to call the scorer even when
/// some evidence sources have not yet produced a result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FusionInput {
    /// Normalised ASR confidence from Whisper (0.0–1.0).
    pub asr_confidence: Option<f32>,
    /// Quality of energy-landmark snapping (0.0–1.0).
    pub energy_alignment_quality: Option<f32>,
    /// Agreement score between phone segmentation and expected pronunciation (0.0–1.0).
    pub phone_segmentation_agreement: Option<f32>,
    /// Pronunciation fit score (0.0–1.0).
    pub pronunciation_fit: Option<f32>,
    /// Score from spectral evidence (0.0–1.0).
    pub spectral_evidence: Option<f32>,
    /// Prosody consistency score (0.0–1.0).
    pub prosody_consistency: Option<f32>,
    /// Timing coherence score (0.0–1.0; penalises impossible orderings).
    pub timing_coherence: Option<f32>,
    /// Aggregate mechanical recogniser score (0.0–1.0, e.g. from DTW/Viterbi).
    pub mechanical_recognizer_score: Option<f32>,
    /// Time-synchronised backend visual speech evidence (0.0–1.0).
    pub visual_speech_evidence: Option<f32>,
}

impl FusionInput {
    fn has_signal(&self) -> bool {
        self.asr_confidence.is_some()
            || self.energy_alignment_quality.is_some()
            || self.phone_segmentation_agreement.is_some()
            || self.pronunciation_fit.is_some()
            || self.spectral_evidence.is_some()
            || self.prosody_consistency.is_some()
            || self.timing_coherence.is_some()
            || self.mechanical_recognizer_score.is_some()
            || self.visual_speech_evidence.is_some()
    }

    fn normalized(mut self) -> Self {
        self.asr_confidence = self.asr_confidence.map(|v| v.clamp(0.0, 1.0));
        self.energy_alignment_quality = self.energy_alignment_quality.map(|v| v.clamp(0.0, 1.0));
        self.phone_segmentation_agreement =
            self.phone_segmentation_agreement.map(|v| v.clamp(0.0, 1.0));
        self.pronunciation_fit = self.pronunciation_fit.map(|v| v.clamp(0.0, 1.0));
        self.spectral_evidence = self.spectral_evidence.map(|v| v.clamp(0.0, 1.0));
        self.prosody_consistency = self.prosody_consistency.map(|v| v.clamp(0.0, 1.0));
        self.timing_coherence = self.timing_coherence.map(|v| v.clamp(0.0, 1.0));
        self.mechanical_recognizer_score =
            self.mechanical_recognizer_score.map(|v| v.clamp(0.0, 1.0));
        self.visual_speech_evidence = self.visual_speech_evidence.map(|v| v.clamp(0.0, 1.0));
        self
    }

    /// Compute a weighted average of the available evidence signals.
    ///
    /// Weights are heuristic and can be tuned in a follow-up.
    pub fn weighted_confidence(&self) -> f32 {
        let mut total_weight = 0.0_f32;
        let mut weighted_sum = 0.0_f32;

        let mut push = |value: Option<f32>, weight: f32| {
            if let Some(v) = value {
                weighted_sum += v * weight;
                total_weight += weight;
            }
        };

        push(self.asr_confidence, 3.0);
        push(self.energy_alignment_quality, 1.5);
        push(self.phone_segmentation_agreement, 1.0);
        push(self.pronunciation_fit, 1.0);
        push(self.spectral_evidence, 0.75);
        push(self.prosody_consistency, 0.5);
        push(self.timing_coherence, 1.25);
        push(self.mechanical_recognizer_score, 1.0);
        push(self.visual_speech_evidence, 0.9);

        if total_weight > 0.0 {
            (weighted_sum / total_weight).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Fusion result
// ---------------------------------------------------------------------------

/// Outcome of the first-pass fusion scorer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionResult {
    /// The hypothesis that best explains all the evidence.
    pub resolved: SpanHypothesis,
    /// Fused confidence (0.0–1.0), combining all evidence sources.
    pub confidence: f32,
    /// Identifiers of hypotheses that support the resolved candidate.
    pub supporting_ids: Vec<SpanHypothesisId>,
    /// Identifiers of hypotheses that contradict the resolved candidate.
    pub conflicting_ids: Vec<SpanHypothesisId>,
    /// Human-readable summary of the supporting evidence.
    pub supporting_summary: String,
    /// Human-readable summary of the conflicting evidence.
    pub conflicting_summary: String,
    /// Machine-readable provenance of the score breakdown.
    pub provenance: serde_json::Value,
}

// ---------------------------------------------------------------------------
// First-pass fusion scorer
// ---------------------------------------------------------------------------

/// Score each competing candidate in `lattice` using the provided `evidence`
/// map and return the best [`FusionResult`].
///
/// `evidence` is a slice of `(hypothesis_id, FusionInput)` pairs that supply
/// external evidence for specific hypotheses.  The scorer:
///
/// 1. Blends each hypothesis's own `confidence` with any matching
///    [`FusionInput`] to produce a fused score.
/// 2. Picks the highest-scoring active hypothesis as the resolved candidate.
/// 3. Classifies the remaining candidates as supporting or conflicting by
///    checking explicit lattice edges first, then falling back to temporal
///    overlap as a proxy for conflict.
/// 4. Returns `None` when the lattice has no active hypotheses.
pub fn fuse_hypotheses(
    lattice: &HypothesisLattice,
    evidence: &[(SpanHypothesisId, FusionInput)],
) -> Option<FusionResult> {
    let actives = lattice.active_hypotheses();
    if actives.is_empty() {
        return None;
    }

    // Build an id → weighted_confidence map from the supplied evidence.
    let evidence_map: std::collections::HashMap<&str, f32> = evidence
        .iter()
        .map(|(id, input)| (id.0.as_str(), input.weighted_confidence()))
        .collect();

    // Compute a fused confidence for each active hypothesis.
    struct Scored<'a> {
        hyp: &'a SpanHypothesis,
        fused_confidence: f32,
    }

    // Weight for external multi-source evidence relative to the hypothesis's
    // own single-source confidence.  External fusion evidence aggregates
    // multiple independent signals, so it deserves higher weight.
    const EXTERNAL_EVIDENCE_WEIGHT: f32 = 3.0;

    let mut scored: Vec<Scored> = actives
        .into_iter()
        .map(|hyp| {
            let extra = evidence_map.get(hyp.id.0.as_str()).copied().unwrap_or(0.0);
            // Blend the hypothesis's own confidence with any extra fusion signal.
            // External evidence is weighted more heavily because it aggregates
            // multiple independent sources.
            let fused = if extra > 0.0 {
                (hyp.confidence + extra * EXTERNAL_EVIDENCE_WEIGHT)
                    / (1.0 + EXTERNAL_EVIDENCE_WEIGHT)
            } else {
                hyp.confidence
            };
            Scored {
                hyp,
                fused_confidence: fused.clamp(0.0, 1.0),
            }
        })
        .collect();

    // Sort descending by fused confidence.
    scored.sort_by(|a, b| {
        b.fused_confidence
            .partial_cmp(&a.fused_confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let best = &scored[0];

    // Classify the remaining candidates as supporting or conflicting.
    let mut supporting_ids = Vec::new();
    let mut conflicting_ids = Vec::new();
    let mut supporting_labels = Vec::new();
    let mut conflicting_labels = Vec::new();

    for other in &scored[1..] {
        let edge_kind = lattice
            .edges
            .iter()
            .find(|e| {
                (&e.from == &best.hyp.id && &e.to == &other.hyp.id)
                    || (&e.from == &other.hyp.id && &e.to == &best.hyp.id)
            })
            .map(|e| &e.kind);

        let is_conflicting = match edge_kind {
            Some(HypothesisEdgeKind::Contradicts) => true,
            Some(HypothesisEdgeKind::Supports | HypothesisEdgeKind::AlignedTo) => false,
            _ => {
                // No explicit edge: treat temporally overlapping spans as conflicting.
                other.hyp.start_ms < best.hyp.end_ms && other.hyp.end_ms > best.hyp.start_ms
            }
        };

        if is_conflicting {
            conflicting_ids.push(other.hyp.id.clone());
            conflicting_labels.push(format!(
                "{} ({:.2})",
                other.hyp.label, other.fused_confidence
            ));
        } else {
            supporting_ids.push(other.hyp.id.clone());
            supporting_labels.push(format!(
                "{} ({:.2})",
                other.hyp.label, other.fused_confidence
            ));
        }
    }

    let mut resolved = best.hyp.clone();
    resolved.confidence = best.fused_confidence;

    let provenance = json!({
        "fusion": "first_pass_weighted_average",
        "evidence_sources": evidence.len(),
        "candidate_count": scored.len(),
        "scores": scored.iter().map(|s| json!({
            "id": s.hyp.id.0,
            "label": s.hyp.label,
            "source_confidence": s.hyp.confidence,
            "fused_confidence": s.fused_confidence,
        })).collect::<Vec<_>>(),
    });

    Some(FusionResult {
        confidence: best.fused_confidence,
        resolved,
        supporting_ids,
        conflicting_ids,
        supporting_summary: if supporting_labels.is_empty() {
            "no supporting candidates".to_string()
        } else {
            supporting_labels.join(", ")
        },
        conflicting_summary: if conflicting_labels.is_empty() {
            "no conflicting candidates".to_string()
        } else {
            conflicting_labels.join(", ")
        },
        provenance,
    })
}

// ---------------------------------------------------------------------------
// Top-level fusion engine
// ---------------------------------------------------------------------------

/// Composable source of fusion evidence for [`SpeechHypothesisEngine`].
pub trait SpeechEvidenceSource: Send + Sync {
    /// Stable source name used in debug traces.
    fn name(&self) -> &'static str;
    /// Produce evidence for hypotheses currently in `lattice`.
    fn collect(&self, lattice: &HypothesisLattice) -> Vec<(SpanHypothesisId, FusionInput)>;
}

/// Debug record showing which source produced which fusion signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceTraceEntry {
    pub source: String,
    pub hypothesis_id: SpanHypothesisId,
    pub input: FusionInput,
}

/// Result of the top-level speech-hypothesis fusion pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechHypothesisFusion {
    /// Copy of the lattice after stability classification.
    pub lattice: HypothesisLattice,
    /// Best-scoring fused hypothesis result.
    pub fusion: FusionResult,
    /// IDs considered stable enough to commit.
    pub stable_span_ids: Vec<SpanHypothesisId>,
    /// IDs that should remain revisable.
    pub revisable_span_ids: Vec<SpanHypothesisId>,
    /// Per-source evidence records for debugging/inspection.
    pub evidence_trace: Vec<EvidenceTraceEntry>,
}

/// First-class speech hypothesis fusion pipeline.
///
/// The default engine wires acoustic, phonetic/pronunciation, transcript
/// stability/ASR, and visual speech evidence sources.
pub struct SpeechHypothesisEngine {
    sources: Vec<Box<dyn SpeechEvidenceSource>>,
    stable_confidence_threshold: f32,
}

impl Default for SpeechHypothesisEngine {
    fn default() -> Self {
        Self::with_default_sources()
    }
}

impl SpeechHypothesisEngine {
    /// Create an empty engine with no evidence sources.
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            stable_confidence_threshold: 0.75,
        }
    }

    /// Create an engine with built-in evidence sources.
    pub fn with_default_sources() -> Self {
        let mut engine = Self::new();
        engine.add_source(AcousticEvidenceSource);
        engine.add_source(PhoneticEvidenceSource);
        engine.add_source(TranscriptStabilityEvidenceSource);
        engine.add_source(VisualSpeechEvidenceSource);
        engine
    }

    /// Add a new composable evidence source.
    pub fn add_source<S: SpeechEvidenceSource + 'static>(&mut self, source: S) {
        self.sources.push(Box::new(source));
    }

    /// Fuse all active hypotheses and classify them as stable/revisable.
    pub fn fuse(&self, lattice: &HypothesisLattice) -> Option<SpeechHypothesisFusion> {
        let mut merged: HashMap<String, FusionInput> = HashMap::new();
        let mut trace = Vec::new();

        for source in &self.sources {
            for (id, raw_input) in source.collect(lattice) {
                let input = raw_input.normalized();
                if !input.has_signal() {
                    continue;
                }
                let key = id.0.clone();
                merged
                    .entry(key)
                    .and_modify(|existing| merge_fusion_input(existing, &input))
                    .or_insert_with(|| input.clone());
                trace.push(EvidenceTraceEntry {
                    source: source.name().to_string(),
                    hypothesis_id: id,
                    input,
                });
            }
        }

        let mut evidence_pairs = Vec::with_capacity(merged.len());
        for (id, input) in &merged {
            evidence_pairs.push((SpanHypothesisId(id.clone()), input.clone()));
        }

        let fusion = fuse_hypotheses(lattice, &evidence_pairs)?;

        let mut stable_span_ids = Vec::new();
        let mut revisable_span_ids = Vec::new();
        for hypothesis in lattice.active_hypotheses() {
            let conf = merged
                .get(&hypothesis.id.0)
                .map(FusionInput::weighted_confidence)
                .unwrap_or(hypothesis.confidence)
                .clamp(0.0, 1.0);
            if conf >= self.stable_confidence_threshold {
                stable_span_ids.push(hypothesis.id.clone());
            } else {
                revisable_span_ids.push(hypothesis.id.clone());
            }
        }

        let mut classified_lattice = lattice.clone();
        for hypothesis in &mut classified_lattice.hypotheses {
            if stable_span_ids.contains(&hypothesis.id) {
                hypothesis.status = HypothesisStatus::Confirmed;
            }
        }

        Some(SpeechHypothesisFusion {
            lattice: classified_lattice,
            fusion,
            stable_span_ids,
            revisable_span_ids,
            evidence_trace: trace,
        })
    }
}

fn merge_signal(existing: Option<f32>, incoming: Option<f32>) -> Option<f32> {
    // Equal-weight averaging keeps merged confidence stable when multiple
    // sources publish the same signal type without introducing source-order
    // dependence.
    match (existing, incoming) {
        (Some(a), Some(b)) => Some(((a + b) * 0.5).clamp(0.0, 1.0)),
        (None, Some(b)) => Some(b.clamp(0.0, 1.0)),
        (Some(a), None) => Some(a.clamp(0.0, 1.0)),
        (None, None) => None,
    }
}

fn merge_fusion_input(target: &mut FusionInput, incoming: &FusionInput) {
    target.asr_confidence = merge_signal(target.asr_confidence, incoming.asr_confidence);
    target.energy_alignment_quality = merge_signal(
        target.energy_alignment_quality,
        incoming.energy_alignment_quality,
    );
    target.phone_segmentation_agreement = merge_signal(
        target.phone_segmentation_agreement,
        incoming.phone_segmentation_agreement,
    );
    target.pronunciation_fit = merge_signal(target.pronunciation_fit, incoming.pronunciation_fit);
    target.spectral_evidence = merge_signal(target.spectral_evidence, incoming.spectral_evidence);
    target.prosody_consistency =
        merge_signal(target.prosody_consistency, incoming.prosody_consistency);
    target.timing_coherence = merge_signal(target.timing_coherence, incoming.timing_coherence);
    target.mechanical_recognizer_score = merge_signal(
        target.mechanical_recognizer_score,
        incoming.mechanical_recognizer_score,
    );
    target.visual_speech_evidence = merge_signal(
        target.visual_speech_evidence,
        incoming.visual_speech_evidence,
    );
}

fn provenance_f32(hypothesis: &SpanHypothesis, key: &str) -> Option<f32> {
    hypothesis
        .provenance
        .get(key)
        .and_then(|value| value.as_f64())
        .map(|value| value as f32)
        .map(|value| value.clamp(0.0, 1.0))
}

struct AcousticEvidenceSource;

impl SpeechEvidenceSource for AcousticEvidenceSource {
    fn name(&self) -> &'static str {
        "acoustic"
    }

    fn collect(&self, lattice: &HypothesisLattice) -> Vec<(SpanHypothesisId, FusionInput)> {
        lattice
            .active_hypotheses()
            .into_iter()
            .map(|hypothesis| {
                let mut input = FusionInput::default();
                input.spectral_evidence = Some(hypothesis.score.clamp(0.0, 1.0));
                input.energy_alignment_quality = match hypothesis.source {
                    crate::audio::hypothesis::HypothesisSource::EndpointDetector => {
                        Some(hypothesis.confidence.clamp(0.0, 1.0))
                    }
                    _ => provenance_f32(hypothesis, "energy_alignment_quality"),
                };
                (hypothesis.id.clone(), input)
            })
            .collect()
    }
}

struct PhoneticEvidenceSource;

impl SpeechEvidenceSource for PhoneticEvidenceSource {
    fn name(&self) -> &'static str {
        "phonetic"
    }

    fn collect(&self, lattice: &HypothesisLattice) -> Vec<(SpanHypothesisId, FusionInput)> {
        lattice
            .active_hypotheses()
            .into_iter()
            .filter_map(|hypothesis| {
                let mut input = FusionInput::default();
                match hypothesis.source {
                    crate::audio::hypothesis::HypothesisSource::PhoneClassifier
                    | crate::audio::hypothesis::HypothesisSource::DtwTemplateMatcher
                    | crate::audio::hypothesis::HypothesisSource::ViterbiAlignment => {
                        input.mechanical_recognizer_score = Some(hypothesis.confidence);
                        input.phone_segmentation_agreement =
                            Some((hypothesis.score + hypothesis.confidence) * 0.5);
                        input.pronunciation_fit = provenance_f32(hypothesis, "pronunciation_fit")
                            .or(Some(
                                (hypothesis.score * 0.7 + hypothesis.confidence * 0.3)
                                    .clamp(0.0, 1.0),
                            ));
                    }
                    _ => {}
                }

                if input.has_signal() {
                    Some((hypothesis.id.clone(), input))
                } else {
                    None
                }
            })
            .collect()
    }
}

struct TranscriptStabilityEvidenceSource;

impl SpeechEvidenceSource for TranscriptStabilityEvidenceSource {
    fn name(&self) -> &'static str {
        "transcript_stability"
    }

    fn collect(&self, lattice: &HypothesisLattice) -> Vec<(SpanHypothesisId, FusionInput)> {
        lattice
            .active_hypotheses()
            .into_iter()
            .filter_map(|hypothesis| {
                let mut input = FusionInput::default();
                input.asr_confidence = provenance_f32(hypothesis, "asr_confidence");

                let stability = provenance_f32(hypothesis, "transcript_stability")
                    .or_else(|| provenance_f32(hypothesis, "stable_prefix_ratio"));
                if let Some(stability) = stability {
                    input.timing_coherence = Some(stability);
                    input.prosody_consistency = Some((stability * 0.85).clamp(0.0, 1.0));
                }

                if input.has_signal() {
                    Some((hypothesis.id.clone(), input))
                } else {
                    None
                }
            })
            .collect()
    }
}

struct VisualSpeechEvidenceSource;

impl SpeechEvidenceSource for VisualSpeechEvidenceSource {
    fn name(&self) -> &'static str {
        "visual_speech"
    }

    fn collect(&self, lattice: &HypothesisLattice) -> Vec<(SpanHypothesisId, FusionInput)> {
        lattice
            .active_hypotheses()
            .into_iter()
            .filter_map(|hypothesis| {
                let visual = provenance_f32(hypothesis, "visual_speech_evidence").or_else(|| {
                    if matches!(
                        hypothesis.source,
                        crate::audio::hypothesis::HypothesisSource::VisualSpeech
                    ) {
                        Some(hypothesis.confidence)
                    } else {
                        None
                    }
                });
                visual.map(|visual_speech_evidence| {
                    (
                        hypothesis.id.clone(),
                        FusionInput {
                            visual_speech_evidence: Some(visual_speech_evidence),
                            ..FusionInput::default()
                        },
                    )
                })
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::hypothesis::{HypothesisSource, HypothesisStatus, SpanHypothesisKind};
    use serde_json::json;

    fn make_word_candidate(
        label: &str,
        start_ms: u64,
        end_ms: u64,
        confidence: f32,
    ) -> SpanHypothesis {
        SpanHypothesis::new(
            SpanHypothesisKind::WordCandidate,
            label,
            start_ms,
            end_ms,
            confidence,
            confidence,
            HypothesisSource::Manual,
            vec![],
            json!(null),
        )
    }

    fn make_word_candidate_with_provenance(
        label: &str,
        start_ms: u64,
        end_ms: u64,
        confidence: f32,
        provenance: serde_json::Value,
    ) -> SpanHypothesis {
        let mut hypothesis = make_word_candidate(label, start_ms, end_ms, confidence);
        hypothesis.provenance = provenance;
        hypothesis
    }

    fn make_boundary(
        label: &str,
        ms: u64,
        confidence: f32,
        source: HypothesisSource,
        features: Vec<String>,
    ) -> SpanHypothesis {
        SpanHypothesis::new(
            SpanHypothesisKind::SpeechBoundary,
            label,
            ms,
            ms,
            confidence,
            confidence,
            source,
            features,
            json!(null),
        )
    }

    // -----------------------------------------------------------------------
    // Lattice structure
    // -----------------------------------------------------------------------

    #[test]
    fn competing_word_candidates_coexist_in_lattice() {
        let mut lattice = HypothesisLattice::new();
        lattice.add(make_word_candidate("testing", 1000, 1300, 0.72));
        lattice.add(make_word_candidate("texting", 1000, 1300, 0.19));
        lattice.add(make_word_candidate("test in", 1000, 1300, 0.07));
        assert_eq!(lattice.active_hypotheses().len(), 3);
        assert_eq!(lattice.all_hypotheses().len(), 3);
    }

    #[test]
    fn lattice_preserves_all_hypotheses_after_revision() {
        let mut lattice = HypothesisLattice::new();
        let h1 = make_word_candidate("testing", 1000, 1300, 0.72);
        let h1_id = h1.id.clone();
        lattice.add(h1);

        let h2 = make_word_candidate("texting", 1000, 1300, 0.85);
        lattice.revise(&h1_id, h2);

        // Both old and new should still be present.
        assert_eq!(lattice.all_hypotheses().len(), 2);
        // The old one should be marked revised.
        let old = lattice
            .all_hypotheses()
            .iter()
            .find(|h| h.id == h1_id)
            .unwrap();
        assert_eq!(old.status, HypothesisStatus::Revised);
    }

    #[test]
    fn active_hypotheses_excludes_revised() {
        let mut lattice = HypothesisLattice::new();
        let h1 = make_word_candidate("testing", 1000, 1300, 0.72);
        let h1_id = h1.id.clone();
        lattice.add(h1);
        let h2 = make_word_candidate("texting", 1000, 1300, 0.85);
        lattice.revise(&h1_id, h2);

        let active = lattice.active_hypotheses();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].label, "texting");
    }

    #[test]
    fn revision_adds_revision_of_edge() {
        let mut lattice = HypothesisLattice::new();
        let h1 = make_word_candidate("testing", 1000, 1300, 0.72);
        let h1_id = h1.id.clone();
        lattice.add(h1);
        let h2 = make_word_candidate("texting", 1000, 1300, 0.85);
        let h2_id = lattice.revise(&h1_id, h2);

        let edges = lattice.edges_from(&h2_id);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].kind, HypothesisEdgeKind::RevisionOf);
        assert_eq!(edges[0].to, h1_id);
    }

    #[test]
    fn hypotheses_can_support_each_other() {
        let mut lattice = HypothesisLattice::new();
        let h1 = make_word_candidate("testing", 1000, 1300, 0.72);
        let h2 = make_boundary(
            "speech_start",
            1000,
            0.65,
            HypothesisSource::EndpointDetector,
            vec!["energy.onset".to_string()],
        );
        let h1_id = h1.id.clone();
        let h2_id = h2.id.clone();
        lattice.add(h1);
        lattice.add(h2);
        lattice.connect(
            h2_id.clone(),
            h1_id.clone(),
            HypothesisEdgeKind::Supports,
            0.8,
        );
        let edges = lattice.edges_from(&h2_id);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].kind, HypothesisEdgeKind::Supports);
    }

    #[test]
    fn conflicting_boundary_hypotheses_are_distinguishable() {
        let mut lattice = HypothesisLattice::new();
        let b1 = make_boundary(
            "speech_start_asr",
            1000,
            0.8,
            HypothesisSource::Manual,
            vec!["asr.timing".to_string()],
        );
        let b2 = make_boundary(
            "speech_start_energy",
            1050,
            0.7,
            HypothesisSource::EndpointDetector,
            vec!["energy.onset".to_string()],
        );
        let b1_id = b1.id.clone();
        let b2_id = b2.id.clone();
        lattice.add(b1);
        lattice.add(b2);
        lattice.connect(
            b1_id.clone(),
            b2_id.clone(),
            HypothesisEdgeKind::Contradicts,
            1.0,
        );
        assert_eq!(lattice.active_hypotheses().len(), 2);
        let result = fuse_hypotheses(&lattice, &[]).unwrap();
        assert!(!result.conflicting_ids.is_empty());
    }

    // -----------------------------------------------------------------------
    // Fusion scorer
    // -----------------------------------------------------------------------

    #[test]
    fn fusion_resolves_highest_confidence_candidate() {
        let mut lattice = HypothesisLattice::new();
        lattice.add(make_word_candidate("testing", 1000, 1300, 0.72));
        lattice.add(make_word_candidate("texting", 1000, 1300, 0.19));
        lattice.add(make_word_candidate("test in", 1000, 1300, 0.07));

        let result = fuse_hypotheses(&lattice, &[]).unwrap();
        assert_eq!(result.resolved.label, "testing");
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn fusion_boosted_by_asr_and_energy_evidence_can_flip_winner() {
        let mut lattice = HypothesisLattice::new();
        let h_low = make_word_candidate("texting", 1000, 1300, 0.19);
        let h_high = make_word_candidate("testing", 1000, 1300, 0.72);
        let low_id = h_low.id.clone();
        lattice.add(h_low);
        lattice.add(h_high);

        // Strong external evidence for the lower-confidence candidate.
        let evidence = vec![(
            low_id,
            FusionInput {
                asr_confidence: Some(0.95),
                energy_alignment_quality: Some(0.90),
                mechanical_recognizer_score: Some(0.88),
                ..Default::default()
            },
        )];
        let result = fuse_hypotheses(&lattice, &evidence).unwrap();
        assert_eq!(result.resolved.label, "texting");
    }

    #[test]
    fn fusion_classifies_conflicting_and_supporting_correctly() {
        let mut lattice = HypothesisLattice::new();
        let h1 = make_word_candidate("testing", 1000, 1300, 0.72);
        let h2 = make_word_candidate("texting", 1000, 1300, 0.19);
        let h2_id = h2.id.clone();
        let h1_id = h1.id.clone();
        lattice.add(h1);
        lattice.add(h2);
        lattice.connect(
            h2_id.clone(),
            h1_id.clone(),
            HypothesisEdgeKind::Contradicts,
            1.0,
        );

        let result = fuse_hypotheses(&lattice, &[]).unwrap();
        assert!(result.conflicting_ids.contains(&h2_id));
        assert!(!result.conflicting_summary.contains("no conflicting"));
    }

    #[test]
    fn fusion_result_preserves_provenance_json() {
        let mut lattice = HypothesisLattice::new();
        lattice.add(make_word_candidate("testing", 1000, 1300, 0.72));
        let result = fuse_hypotheses(&lattice, &[]).unwrap();
        assert_eq!(result.provenance["fusion"], "first_pass_weighted_average");
        assert!(result.provenance["candidate_count"].as_u64().unwrap() >= 1);
    }

    #[test]
    fn fusion_on_empty_lattice_returns_none() {
        let lattice = HypothesisLattice::new();
        assert!(fuse_hypotheses(&lattice, &[]).is_none());
    }

    #[test]
    fn fusion_input_weighted_confidence_uses_all_signals() {
        let input = FusionInput {
            asr_confidence: Some(1.0),
            energy_alignment_quality: Some(1.0),
            phone_segmentation_agreement: Some(1.0),
            pronunciation_fit: Some(1.0),
            spectral_evidence: Some(1.0),
            prosody_consistency: Some(1.0),
            timing_coherence: Some(1.0),
            mechanical_recognizer_score: Some(1.0),
            visual_speech_evidence: Some(1.0),
        };
        assert!((input.weighted_confidence() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn fusion_input_zero_signals_returns_zero_confidence() {
        let input = FusionInput::default();
        assert_eq!(input.weighted_confidence(), 0.0);
    }

    #[test]
    fn fusion_result_serializes_to_json() {
        let mut lattice = HypothesisLattice::new();
        lattice.add(make_word_candidate("testing", 1000, 1300, 0.72));
        let result = fuse_hypotheses(&lattice, &[]).unwrap();
        let json = serde_json::to_string(&result).expect("serialise");
        assert!(json.contains("resolved"));
        assert!(json.contains("confidence"));
        assert!(json.contains("provenance"));
    }

    #[test]
    fn speech_hypothesis_engine_uses_multiple_default_evidence_sources() {
        let mut lattice = HypothesisLattice::new();
        lattice.add(make_word_candidate_with_provenance(
            "testing",
            1000,
            1300,
            0.40,
            json!({
                "asr_confidence": 0.91,
                "transcript_stability": 0.88,
                "visual_speech_evidence": 0.82,
            }),
        ));
        lattice.add(make_boundary(
            "speech_start",
            1000,
            0.72,
            HypothesisSource::EndpointDetector,
            vec!["energy.onset".to_string()],
        ));
        lattice.add(SpanHypothesis::new(
            SpanHypothesisKind::PhoneClassCandidate,
            "fricative",
            1020,
            1060,
            0.68,
            0.66,
            HypothesisSource::PhoneClassifier,
            vec!["spectral_flux".to_string()],
            json!(null),
        ));

        let engine = SpeechHypothesisEngine::with_default_sources();
        let output = engine.fuse(&lattice).expect("fused");

        let unique_sources: std::collections::HashSet<&str> = output
            .evidence_trace
            .iter()
            .map(|entry| entry.source.as_str())
            .collect();
        assert!(unique_sources.len() >= 3);
    }

    #[test]
    fn speech_hypothesis_engine_applies_stability_and_rescoring() {
        let mut lattice = HypothesisLattice::new();
        let low_acoustic_high_stability = make_word_candidate_with_provenance(
            "texting",
            1000,
            1300,
            0.25,
            json!({
                "asr_confidence": 0.94,
                "transcript_stability": 0.90,
                "stable_prefix_ratio": 0.89,
                "visual_speech_evidence": 0.87,
            }),
        );
        let high_acoustic_low_stability = make_word_candidate_with_provenance(
            "testing",
            1000,
            1300,
            0.78,
            json!({
                "asr_confidence": 0.25,
                "transcript_stability": 0.20,
            }),
        );
        let low_id = low_acoustic_high_stability.id.clone();
        let high_id = high_acoustic_low_stability.id.clone();
        lattice.add(low_acoustic_high_stability);
        lattice.add(high_acoustic_low_stability);
        lattice.connect(
            low_id.clone(),
            high_id.clone(),
            HypothesisEdgeKind::Contradicts,
            1.0,
        );

        let engine = SpeechHypothesisEngine::with_default_sources();
        let output = engine.fuse(&lattice).expect("fused");

        assert_eq!(output.fusion.resolved.label, "texting");
        assert!(output.stable_span_ids.contains(&low_id));
        assert!(output.revisable_span_ids.contains(&high_id));

        let stable = output
            .lattice
            .all_hypotheses()
            .iter()
            .find(|h| h.id == low_id)
            .expect("stable span");
        assert_eq!(stable.status, HypothesisStatus::Confirmed);
    }

    #[test]
    fn speech_hypothesis_fusion_output_serializes_for_debugging() {
        let mut lattice = HypothesisLattice::new();
        lattice.add(make_word_candidate_with_provenance(
            "testing",
            1000,
            1300,
            0.55,
            json!({
                "asr_confidence": 0.85,
                "transcript_stability": 0.83,
            }),
        ));

        let engine = SpeechHypothesisEngine::with_default_sources();
        let output = engine.fuse(&lattice).expect("fused");
        let encoded = serde_json::to_string(&output).expect("serialise");
        assert!(encoded.contains("stable_span_ids"));
        assert!(encoded.contains("evidence_trace"));
        assert!(encoded.contains("fusion"));
    }

    #[test]
    fn lattice_serializes_and_deserializes_round_trip() {
        let mut lattice = HypothesisLattice::new();
        let h1 = make_word_candidate("testing", 1000, 1300, 0.72);
        let h2 = make_word_candidate("texting", 1000, 1300, 0.19);
        let h1_id = h1.id.clone();
        let h2_id = h2.id.clone();
        lattice.add(h1);
        lattice.add(h2);
        lattice.connect(h1_id, h2_id, HypothesisEdgeKind::Contradicts, 1.0);

        let json = serde_json::to_string(&lattice).expect("serialise");
        let restored: HypothesisLattice = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(restored.hypotheses.len(), 2);
        assert_eq!(restored.edges.len(), 1);
        assert_eq!(restored.edges[0].kind, HypothesisEdgeKind::Contradicts);
    }
}
