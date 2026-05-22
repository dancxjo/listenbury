//! Endpoint and speech-boundary hypothesis generator.
//!
//! Converts the energy-landmark outputs already computed by
//! [`crate::audio::acoustic`] into [`SpanHypothesis`] values:
//!
//! | Landmark type | Hypothesis kind       | Label(s)                                 |
//! |---------------|-----------------------|------------------------------------------|
//! | onset/offset  | `SpeechBoundary`      | `speech_start` / `speech_end` / `speech_region` |
//! | silence       | `PauseCandidate`      | `pause`                                  |
//! | valley        | `SpeechBoundary`      | `boundary_candidate`                     |

use serde_json::json;

use crate::audio::acoustic::EnergyLandmarks;
use crate::audio::features::AcousticFeatureStream;
use crate::audio::hypothesis::{
    HypothesisSource, HypothesisStatus, SpanHypothesis, SpanHypothesisId, SpanHypothesisKind,
};
use crate::segmentation::{
    BoundaryEvidence, BoundaryHypothesis, BoundaryKind, generate_landmark_hypotheses,
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Generate boundary and pause hypotheses from pre-computed energy landmarks.
///
/// The optional `features` stream is used to improve the confidence estimate
/// for onset and offset points; pass `None` if it is unavailable.
pub fn generate_boundary_hypotheses(
    landmarks: &EnergyLandmarks,
    features: Option<&AcousticFeatureStream>,
) -> Vec<SpanHypothesis> {
    generate_landmark_hypotheses(landmarks, features)
        .into_iter()
        .map(boundary_to_span_hypothesis)
        .collect()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Derive a confidence score from the RMS energy around `ms`.
fn boundary_to_span_hypothesis(hypothesis: BoundaryHypothesis) -> SpanHypothesis {
    let kind = to_span_kind(&hypothesis);
    let label = compatibility_label(&hypothesis);
    let start_ms = hypothesis.start_time.max(0.0).floor() as u64;
    let end_ms = hypothesis.end_time.max(0.0).ceil() as u64;
    let confidence = hypothesis.confidence.clamp(0.0, 1.0);
    let features_used = hypothesis
        .evidence
        .iter()
        .map(boundary_evidence_to_feature)
        .collect::<Vec<_>>();
    let score = confidence;

    SpanHypothesis {
        id: SpanHypothesisId::new(),
        kind,
        label,
        start_ms,
        end_ms,
        score,
        confidence,
        source: HypothesisSource::EndpointDetector,
        features_used,
        status: HypothesisStatus::Provisional,
        provenance: json!({
            "kind": hypothesis.kind,
            "evidence": hypothesis.evidence,
        }),
    }
}

fn to_span_kind(hypothesis: &BoundaryHypothesis) -> SpanHypothesisKind {
    match hypothesis.kind {
        BoundaryKind::SpeechRegion => SpanHypothesisKind::SpeechBoundary,
        BoundaryKind::SyllableIsland => SpanHypothesisKind::SyllableCandidate,
        BoundaryKind::PossibleWordRegion => SpanHypothesisKind::WordCandidate,
        BoundaryKind::NoiseEvent => {
            if hypothesis.evidence.contains(&BoundaryEvidence::SilenceGap) {
                SpanHypothesisKind::PauseCandidate
            } else {
                SpanHypothesisKind::SpeechBoundary
            }
        }
    }
}

fn compatibility_label(hypothesis: &BoundaryHypothesis) -> String {
    match hypothesis.kind {
        BoundaryKind::SpeechRegion => {
            if hypothesis
                .evidence
                .contains(&BoundaryEvidence::EnergyRise)
                && !hypothesis.evidence.contains(&BoundaryEvidence::EnergyFall)
            {
                "speech_start".to_string()
            } else if hypothesis
                .evidence
                .contains(&BoundaryEvidence::EnergyFall)
                && !hypothesis.evidence.contains(&BoundaryEvidence::EnergyRise)
            {
                "speech_end".to_string()
            } else if (hypothesis.start_time - hypothesis.end_time).abs() < f32::EPSILON {
                "speech_start".to_string()
            } else {
                "speech_region".to_string()
            }
        }
        BoundaryKind::SyllableIsland => "syllable_island".to_string(),
        BoundaryKind::PossibleWordRegion => "possible_word_region".to_string(),
        BoundaryKind::NoiseEvent => {
            if hypothesis.evidence.contains(&BoundaryEvidence::SilenceGap) {
                "pause".to_string()
            } else {
                "boundary_candidate".to_string()
            }
        }
    }
}

fn boundary_evidence_to_feature(evidence: &BoundaryEvidence) -> String {
    match evidence {
        BoundaryEvidence::EnergyRise => "energy.onset",
        BoundaryEvidence::EnergyFall => "energy.offset",
        BoundaryEvidence::VoicingOnset => "voicing.onset",
        BoundaryEvidence::VoicingOffset => "voicing.offset",
        BoundaryEvidence::FormantOnset => "formant.onset",
        BoundaryEvidence::FormantOffset => "formant.offset",
        BoundaryEvidence::VowelNucleus => "vowel.nucleus",
        BoundaryEvidence::SpectralFluxPeak => "spectral.flux_peak",
        BoundaryEvidence::SilenceGap => "energy.silence",
        BoundaryEvidence::NoiseRejected => "noise.rejected",
        BoundaryEvidence::MatchesExpectedPhoneShape => "phone.shape_match",
    }
    .to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::acoustic::{EnergyLandmarks, EnergySilence};

    fn empty_landmarks() -> EnergyLandmarks {
        EnergyLandmarks {
            onsets: Vec::new(),
            offsets: Vec::new(),
            valleys: Vec::new(),
            silences: Vec::new(),
            peaks: Vec::new(),
        }
    }

    #[test]
    fn empty_landmarks_produce_no_hypotheses() {
        let hyps = generate_boundary_hypotheses(&empty_landmarks(), None);
        assert!(hyps.is_empty());
    }

    #[test]
    fn onset_produces_speech_start_hypothesis() {
        let mut lm = empty_landmarks();
        lm.onsets.push(500);
        let hyps = generate_boundary_hypotheses(&lm, None);
        assert_eq!(hyps.len(), 1);
        let h = &hyps[0];
        assert_eq!(h.kind, SpanHypothesisKind::SpeechBoundary);
        assert_eq!(h.label, "speech_start");
        assert_eq!(h.start_ms, 500);
        assert_eq!(h.end_ms, 500);
        assert_eq!(h.source, HypothesisSource::EndpointDetector);
        assert!(h.features_used.contains(&"energy.onset".to_string()));
    }

    #[test]
    fn offset_produces_speech_end_hypothesis() {
        let mut lm = empty_landmarks();
        lm.offsets.push(1200);
        let hyps = generate_boundary_hypotheses(&lm, None);
        assert_eq!(hyps.len(), 1);
        assert_eq!(hyps[0].label, "speech_end");
    }

    #[test]
    fn silence_produces_pause_candidate() {
        let mut lm = empty_landmarks();
        lm.silences.push(EnergySilence {
            start_ms: 2000,
            end_ms: 2300,
        });
        let hyps = generate_boundary_hypotheses(&lm, None);
        assert_eq!(hyps.len(), 1);
        let h = &hyps[0];
        assert_eq!(h.kind, SpanHypothesisKind::PauseCandidate);
        assert_eq!(h.label, "pause");
        assert_eq!(h.start_ms, 2000);
        assert_eq!(h.end_ms, 2300);
    }

    #[test]
    fn valley_produces_boundary_candidate() {
        let mut lm = empty_landmarks();
        lm.valleys.push(750);
        let hyps = generate_boundary_hypotheses(&lm, None);
        assert_eq!(hyps.len(), 1);
        assert_eq!(hyps[0].label, "boundary_candidate");
    }

    #[test]
    fn mixed_landmarks_produce_correct_counts() {
        let lm = EnergyLandmarks {
            onsets: vec![100, 600],
            offsets: vec![400, 900],
            valleys: vec![200, 700],
            silences: vec![EnergySilence {
                start_ms: 1000,
                end_ms: 1200,
            }],
            peaks: vec![150, 650],
        };
        let hyps = generate_boundary_hypotheses(&lm, None);
        // 2 onsets + 2 offsets + 2 valleys + 1 silence = 7
        assert_eq!(hyps.len(), 7);
    }

    #[test]
    fn hypotheses_are_provisional_by_default() {
        let mut lm = empty_landmarks();
        lm.onsets.push(300);
        lm.silences.push(EnergySilence {
            start_ms: 500,
            end_ms: 700,
        });
        for hyp in generate_boundary_hypotheses(&lm, None) {
            assert_eq!(hyp.status, HypothesisStatus::Provisional);
        }
    }
}
