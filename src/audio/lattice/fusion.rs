use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::audio::hypothesis::{SpanHypothesis, SpanHypothesisId};

use super::weights::FusionWeights;
use super::{HypothesisEdgeKind, HypothesisLattice};

/// Evidence signals fed into the first-pass fusion scorer.
///
/// All fields are optional; the scorer weights only the signals that are
/// actually provided. This makes it possible to call the scorer even when
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
    pub(crate) fn has_signal(&self) -> bool {
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

    pub(crate) fn normalized(mut self) -> Self {
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

    /// Compute a weighted average of the available evidence signals using the
    /// supplied [`FusionWeights`] configuration.
    ///
    /// Only signals that are `Some` contribute to the average; missing signals
    /// do not drag the score down.
    pub fn weighted_confidence_with(&self, weights: &FusionWeights) -> f32 {
        let mut total_weight = 0.0_f32;
        let mut weighted_sum = 0.0_f32;

        let mut push = |value: Option<f32>, weight: f32| {
            if let Some(v) = value {
                weighted_sum += v * weight;
                total_weight += weight;
            }
        };

        push(self.asr_confidence, weights.asr_confidence);
        push(
            self.energy_alignment_quality,
            weights.energy_alignment_quality,
        );
        push(
            self.phone_segmentation_agreement,
            weights.phone_segmentation_agreement,
        );
        push(self.pronunciation_fit, weights.pronunciation_fit);
        push(self.spectral_evidence, weights.spectral_evidence);
        push(self.prosody_consistency, weights.prosody_consistency);
        push(self.timing_coherence, weights.timing_coherence);
        push(
            self.mechanical_recognizer_score,
            weights.mechanical_recognizer_score,
        );
        push(self.visual_speech_evidence, weights.visual_speech_evidence);

        if total_weight > 0.0 {
            (weighted_sum / total_weight).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Compute a weighted average of the available evidence signals using the
    /// default [`FusionWeights`] (i.e. [`FusionProfile::Default`]).
    ///
    /// This is a convenience wrapper around [`Self::weighted_confidence_with`]
    /// and preserves the original heuristic behaviour.
    pub fn weighted_confidence(&self) -> f32 {
        self.weighted_confidence_with(&FusionWeights::default())
    }
}

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

/// Score each competing candidate in `lattice` using the provided `evidence`
/// map and return the best [`FusionResult`].
///
/// `evidence` is a slice of `(hypothesis_id, FusionInput)` pairs that supply
/// external evidence for specific hypotheses. The scorer:
///
/// 1. Blends each hypothesis's own `confidence` with any matching
///    [`FusionInput`] to produce a fused score using the supplied `weights`.
/// 2. Picks the highest-scoring active hypothesis as the resolved candidate.
/// 3. Classifies the remaining candidates as supporting or conflicting by
///    checking explicit lattice edges first, then falling back to temporal
///    overlap as a proxy for conflict.
/// 4. Returns `None` when the lattice has no active hypotheses.
///
/// Pass `&FusionWeights::default()` to reproduce the original heuristic
/// behaviour.
pub fn fuse_hypotheses(
    lattice: &HypothesisLattice,
    evidence: &[(SpanHypothesisId, FusionInput)],
    weights: &FusionWeights,
) -> Option<FusionResult> {
    let actives = lattice.active_hypotheses();
    if actives.is_empty() {
        return None;
    }

    let evidence_map: std::collections::HashMap<&str, f32> = evidence
        .iter()
        .map(|(id, input)| (id.0.as_str(), input.weighted_confidence_with(weights)))
        .collect();

    struct Scored<'a> {
        hyp: &'a SpanHypothesis,
        fused_confidence: f32,
    }

    let external_evidence_blend = weights.external_evidence_blend;

    let mut scored: Vec<Scored> = actives
        .into_iter()
        .map(|hyp| {
            let extra = evidence_map.get(hyp.id.0.as_str()).copied().unwrap_or(0.0);
            let fused = if extra > 0.0 {
                (hyp.confidence + extra * external_evidence_blend) / (1.0 + external_evidence_blend)
            } else {
                hyp.confidence
            };
            Scored {
                hyp,
                fused_confidence: fused.clamp(0.0, 1.0),
            }
        })
        .collect();

    scored.sort_by(|a, b| {
        b.fused_confidence
            .partial_cmp(&a.fused_confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let best = &scored[0];

    let mut supporting_ids = Vec::new();
    let mut conflicting_ids = Vec::new();
    let mut supporting_labels = Vec::new();
    let mut conflicting_labels = Vec::new();

    for other in &scored[1..] {
        let edge_kind = lattice
            .edges
            .iter()
            .find(|e| {
                (e.from == best.hyp.id && e.to == other.hyp.id)
                    || (e.from == other.hyp.id && e.to == best.hyp.id)
            })
            .map(|e| &e.kind);

        let is_conflicting = match edge_kind {
            Some(HypothesisEdgeKind::Contradicts) => true,
            Some(HypothesisEdgeKind::Supports | HypothesisEdgeKind::AlignedTo) => false,
            _ => other.hyp.start_ms < best.hyp.end_ms && other.hyp.end_ms > best.hyp.start_ms,
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
