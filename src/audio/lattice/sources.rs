use crate::audio::hypothesis::{HypothesisSource, SpanHypothesis, SpanHypothesisId};

use super::{FusionInput, HypothesisLattice, SpeechEvidenceSource};

pub(crate) struct AcousticEvidenceSource;

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
                    HypothesisSource::EndpointDetector => {
                        Some(hypothesis.confidence.clamp(0.0, 1.0))
                    }
                    _ => provenance_f32(hypothesis, "energy_alignment_quality"),
                };
                (hypothesis.id.clone(), input)
            })
            .collect()
    }
}

pub(crate) struct PhoneticEvidenceSource;

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
                    HypothesisSource::PhoneClassifier
                    | HypothesisSource::DtwTemplateMatcher
                    | HypothesisSource::ViterbiAlignment => {
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

pub(crate) struct TranscriptStabilityEvidenceSource;

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

pub(crate) struct VisualSpeechEvidenceSource;

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
                    if matches!(hypothesis.source, HypothesisSource::VisualSpeech) {
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

fn provenance_f32(hypothesis: &SpanHypothesis, key: &str) -> Option<f32> {
    hypothesis
        .provenance
        .get(key)
        .and_then(|value| value.as_f64())
        .map(|value| value as f32)
        .map(|value| value.clamp(0.0, 1.0))
}
