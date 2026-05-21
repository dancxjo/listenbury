use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::audio::hypothesis::{HypothesisStatus, SpanHypothesisId};

use super::sources::{
    AcousticEvidenceSource, PhoneticEvidenceSource, TranscriptStabilityEvidenceSource,
    VisualSpeechEvidenceSource,
};
use super::{
    EvidenceTraceEntry, FusionInput, FusionResult, HypothesisLattice, SpeechEvidenceSource,
    fuse_hypotheses,
};

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
