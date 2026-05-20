//! Endpoint and speech-boundary hypothesis generator.
//!
//! Converts the energy-landmark outputs already computed by
//! [`crate::audio::acoustic`] into [`SpanHypothesis`] values:
//!
//! | Landmark type | Hypothesis kind       | Label                |
//! |---------------|-----------------------|----------------------|
//! | onset         | `SpeechBoundary`      | `speech_start`       |
//! | offset        | `SpeechBoundary`      | `speech_end`         |
//! | silence       | `PauseCandidate`      | `pause`              |
//! | valley        | `SpeechBoundary`      | `boundary_candidate` |

use serde_json::json;

use crate::audio::acoustic::EnergyLandmarks;
use crate::audio::features::AcousticFeatureStream;
use crate::audio::hypothesis::{
    HypothesisSource, HypothesisStatus, SpanHypothesis, SpanHypothesisId, SpanHypothesisKind,
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
    let mut hyps = Vec::new();

    for &ms in &landmarks.onsets {
        let conf = energy_confidence_at(features, ms).unwrap_or(0.65);
        hyps.push(SpanHypothesis {
            id: SpanHypothesisId::new(),
            kind: SpanHypothesisKind::SpeechBoundary,
            label: "speech_start".to_string(),
            start_ms: ms,
            end_ms: ms,
            score: 0.75,
            confidence: conf,
            source: HypothesisSource::EndpointDetector,
            features_used: vec!["energy.onset".to_string()],
            status: HypothesisStatus::Provisional,
            provenance: json!({ "type": "onset", "ms": ms }),
        });
    }

    for &ms in &landmarks.offsets {
        let conf = energy_confidence_at(features, ms).unwrap_or(0.60);
        hyps.push(SpanHypothesis {
            id: SpanHypothesisId::new(),
            kind: SpanHypothesisKind::SpeechBoundary,
            label: "speech_end".to_string(),
            start_ms: ms,
            end_ms: ms,
            score: 0.70,
            confidence: conf,
            source: HypothesisSource::EndpointDetector,
            features_used: vec!["energy.offset".to_string()],
            status: HypothesisStatus::Provisional,
            provenance: json!({ "type": "offset", "ms": ms }),
        });
    }

    for silence in &landmarks.silences {
        let duration_ms = silence.end_ms.saturating_sub(silence.start_ms);
        hyps.push(SpanHypothesis {
            id: SpanHypothesisId::new(),
            kind: SpanHypothesisKind::PauseCandidate,
            label: "pause".to_string(),
            start_ms: silence.start_ms,
            end_ms: silence.end_ms,
            score: 0.80,
            confidence: confidence_for_pause_duration(duration_ms),
            source: HypothesisSource::EndpointDetector,
            features_used: vec!["energy.silence".to_string()],
            status: HypothesisStatus::Provisional,
            provenance: json!({
                "type": "silence",
                "start_ms": silence.start_ms,
                "end_ms": silence.end_ms,
                "duration_ms": duration_ms,
            }),
        });
    }

    for &ms in &landmarks.valleys {
        hyps.push(SpanHypothesis {
            id: SpanHypothesisId::new(),
            kind: SpanHypothesisKind::SpeechBoundary,
            label: "boundary_candidate".to_string(),
            start_ms: ms,
            end_ms: ms,
            score: 0.45,
            confidence: 0.40,
            source: HypothesisSource::EndpointDetector,
            features_used: vec!["energy.valley".to_string()],
            status: HypothesisStatus::Provisional,
            provenance: json!({ "type": "valley", "ms": ms }),
        });
    }

    hyps
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Derive a confidence score from the RMS energy around `ms`.
fn energy_confidence_at(features: Option<&AcousticFeatureStream>, ms: u64) -> Option<f32> {
    let stream = features?;
    let frame = stream
        .frames
        .iter()
        .find(|f| f.frame_start_ms <= ms && f.frame_end_ms >= ms)?;
    Some((frame.rms_energy * 8.0).clamp(0.25, 0.92))
}

/// Longer pauses get slightly higher confidence (true silence vs. brief dip).
fn confidence_for_pause_duration(duration_ms: u64) -> f32 {
    if duration_ms >= 400 {
        0.88
    } else if duration_ms >= 150 {
        0.75
    } else {
        0.55
    }
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
