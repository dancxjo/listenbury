//! Hypothesis lattice and first-pass fusion layer.
//!
//! The lattice collects competing [`SpanHypothesis`] values from multiple
//! evidence sources and lets them support or contradict each other via
//! typed [`HypothesisEdge`]s.  A first-pass [`fuse_hypotheses`] scorer
//! combines the evidence into a [`FusionResult`] with a resolved candidate,
//! confidence, and provenance.
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
    pub fn revise(&mut self, old_id: &SpanHypothesisId, revised: SpanHypothesis) -> SpanHypothesisId {
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
}

impl FusionInput {
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
        };
        assert!((input.weighted_confidence() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn fusion_input_zero_signals_returns_zero_confidence() {
        let input = FusionInput::default();
        assert_eq!(input.weighted_confidence(), 0.0);
    }

    #[test]
    fn fusion_result_serialises_to_json() {
        let mut lattice = HypothesisLattice::new();
        lattice.add(make_word_candidate("testing", 1000, 1300, 0.72));
        let result = fuse_hypotheses(&lattice, &[]).unwrap();
        let json = serde_json::to_string(&result).expect("serialise");
        assert!(json.contains("resolved"));
        assert!(json.contains("confidence"));
        assert!(json.contains("provenance"));
    }

    #[test]
    fn lattice_serialises_and_deserialises_round_trip() {
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
