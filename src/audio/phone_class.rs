//! First-pass coarse phone-class hypothesis generator.
//!
//! Uses heuristic spectral and energy features to emit phone-class guesses
//! over short analysis windows.  The classifier intentionally uses only
//! cheap, explainable features so its decisions can be inspected.
//!
//! ## Classes
//!
//! | Label               | Rough cue                                   |
//! |---------------------|---------------------------------------------|
//! | `vowel_or_sonorant` | Low ZCR, high energy, low-band dominant     |
//! | `fricative`         | High ZCR + strong high-frequency energy     |
//! | `stop_closure`      | Very low energy, low ZCR                    |
//! | `stop_burst`        | High spectral flux + short energy spike     |
//! | `nasal`             | Moderate ZCR, murmur-dominated low band     |
//! | `approximant_liquid`| Moderate ZCR, mid-energy, slow spectral change |
//! | `silence_noise`     | Energy below floor                          |
//! | `unknown`           | None of the above                           |

use serde_json::json;

use crate::audio::features::{AcousticFeatureFrame, AcousticFeatureStream};
use crate::audio::hypothesis::{
    HypothesisSource, HypothesisStatus, SpanHypothesis, SpanHypothesisId, SpanHypothesisKind,
};

// ---------------------------------------------------------------------------
// Coarse phone-class enum
// ---------------------------------------------------------------------------

/// Coarse phone-class labels emitted by the heuristic classifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoarsePhoneClass {
    VowelOrSonorant,
    Fricative,
    StopClosure,
    StopBurst,
    Nasal,
    ApproximantLiquid,
    SilenceNoise,
    Unknown,
}

impl CoarsePhoneClass {
    /// Canonical label string.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::VowelOrSonorant => "vowel_or_sonorant",
            Self::Fricative => "fricative",
            Self::StopClosure => "stop_closure",
            Self::StopBurst => "stop_burst",
            Self::Nasal => "nasal",
            Self::ApproximantLiquid => "approximant_liquid",
            Self::SilenceNoise => "silence_noise",
            Self::Unknown => "unknown",
        }
    }
}

// ---------------------------------------------------------------------------
// Frame-level classifier
// ---------------------------------------------------------------------------

/// Classify a single [`AcousticFeatureFrame`] into a [`CoarsePhoneClass`].
///
/// Returns the class and the list of feature names that drove the decision.
pub fn classify_frame(frame: &AcousticFeatureFrame) -> (CoarsePhoneClass, Vec<String>) {
    let mut features_used: Vec<String> = Vec::new();

    // Silence / noise: energy is below the floor.
    if frame.rms_energy < 0.005 {
        features_used.push("energy.silence".to_string());
        return (CoarsePhoneClass::SilenceNoise, features_used);
    }

    // Fricative: high ZCR + strong high-frequency energy relative to low-band.
    if frame.zero_crossing_rate > 0.15
        && frame.high_band_energy_db > frame.low_band_energy_db - 10.0
    {
        features_used.push("zcr.high".to_string());
        features_used.push("band.high_freq".to_string());
        return (CoarsePhoneClass::Fricative, features_used);
    }

    // Stop burst: sudden high spectral flux with a noticeable energy spike.
    if frame.spectral_flux > 0.15 && frame.rms_energy > 0.04 {
        features_used.push("spectral_flux.high".to_string());
        features_used.push("energy.burst".to_string());
        return (CoarsePhoneClass::StopBurst, features_used);
    }

    // Stop closure: very low energy and very low ZCR (pre-burst silence).
    if frame.rms_energy < 0.015 && frame.zero_crossing_rate < 0.08 {
        features_used.push("energy.low_closure".to_string());
        features_used.push("zcr.low".to_string());
        return (CoarsePhoneClass::StopClosure, features_used);
    }

    // Nasal: moderate ZCR, low high-frequency energy, stable spectrum.
    if frame.zero_crossing_rate < 0.10
        && frame.high_band_energy_db < frame.low_band_energy_db - 15.0
        && frame.spectral_flux < 0.08
    {
        features_used.push("zcr.moderate".to_string());
        features_used.push("band.low_dominated".to_string());
        features_used.push("spectral_flux.stable".to_string());
        return (CoarsePhoneClass::Nasal, features_used);
    }

    // Vowel / sonorant: low ZCR, moderate-to-high energy, low-band dominant.
    if frame.zero_crossing_rate < 0.12
        && frame.rms_energy > 0.02
        && frame.low_band_energy_db > frame.high_band_energy_db
    {
        features_used.push("zcr.low".to_string());
        features_used.push("energy.voiced".to_string());
        features_used.push("band.low_dominant".to_string());
        return (CoarsePhoneClass::VowelOrSonorant, features_used);
    }

    // Approximant / liquid: moderate ZCR, moderate energy, slow spectral change.
    if frame.zero_crossing_rate < 0.18 && frame.rms_energy > 0.01 && frame.spectral_flux < 0.12 {
        features_used.push("zcr.moderate".to_string());
        features_used.push("energy.moderate".to_string());
        features_used.push("spectral_flux.slow".to_string());
        return (CoarsePhoneClass::ApproximantLiquid, features_used);
    }

    features_used.push("unknown".to_string());
    (CoarsePhoneClass::Unknown, features_used)
}

// ---------------------------------------------------------------------------
// Stream-level generator
// ---------------------------------------------------------------------------

/// Emit one [`SpanHypothesis`] per frame in `stream`.
pub fn generate_phone_class_hypotheses(stream: &AcousticFeatureStream) -> Vec<SpanHypothesis> {
    stream
        .frames
        .iter()
        .map(|frame| {
            let (class, features_used) = classify_frame(frame);
            let label = class.as_str().to_string();
            let confidence = match class {
                CoarsePhoneClass::SilenceNoise | CoarsePhoneClass::Unknown => 0.40,
                CoarsePhoneClass::StopBurst | CoarsePhoneClass::Fricative => 0.65,
                _ => 0.55,
            };
            SpanHypothesis {
                id: SpanHypothesisId::new(),
                kind: SpanHypothesisKind::PhoneClassCandidate,
                label: label.clone(),
                start_ms: frame.frame_start_ms,
                end_ms: frame.frame_end_ms,
                score: confidence,
                confidence,
                source: HypothesisSource::PhoneClassifier,
                features_used,
                status: HypothesisStatus::Provisional,
                provenance: json!({
                    "class": label,
                    "rms_energy": frame.rms_energy,
                    "zcr": frame.zero_crossing_rate,
                    "spectral_flux": frame.spectral_flux,
                    "low_band_db": frame.low_band_energy_db,
                    "high_band_db": frame.high_band_energy_db,
                }),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(
        rms: f32,
        peak: f32,
        zcr: f32,
        flux: f32,
        low_db: f32,
        high_db: f32,
        start_ms: u64,
    ) -> AcousticFeatureFrame {
        AcousticFeatureFrame {
            frame_start_ms: start_ms,
            frame_end_ms: start_ms + 10,
            rms_energy: rms,
            peak_amplitude: peak,
            zero_crossing_rate: zcr,
            spectral_flux: flux,
            low_band_energy_db: low_db,
            high_band_energy_db: high_db,
        }
    }

    #[test]
    fn silence_below_energy_floor_classifies_as_silence_noise() {
        let (class, features) = classify_frame(&frame(0.002, 0.003, 0.05, 0.02, -50.0, -55.0, 0));
        assert_eq!(class, CoarsePhoneClass::SilenceNoise);
        assert!(features.contains(&"energy.silence".to_string()));
    }

    #[test]
    fn high_zcr_and_high_freq_classifies_as_fricative() {
        // High ZCR (0.20 > 0.15) and high-band close to low-band.
        let (class, features) = classify_frame(&frame(0.04, 0.06, 0.20, 0.05, -20.0, -15.0, 10));
        assert_eq!(class, CoarsePhoneClass::Fricative);
        assert!(features.contains(&"zcr.high".to_string()));
        assert!(features.contains(&"band.high_freq".to_string()));
    }

    #[test]
    fn high_flux_and_energy_classifies_as_stop_burst() {
        let (class, features) = classify_frame(&frame(0.08, 0.12, 0.12, 0.20, -18.0, -30.0, 20));
        assert_eq!(class, CoarsePhoneClass::StopBurst);
        assert!(features.contains(&"spectral_flux.high".to_string()));
    }

    #[test]
    fn low_energy_low_zcr_classifies_as_stop_closure() {
        let (class, features) = classify_frame(&frame(0.010, 0.014, 0.04, 0.02, -40.0, -50.0, 30));
        assert_eq!(class, CoarsePhoneClass::StopClosure);
        assert!(features.contains(&"energy.low_closure".to_string()));
    }

    #[test]
    fn low_zcr_high_energy_low_band_dominant_classifies_as_vowel() {
        let (class, features) = classify_frame(&frame(0.06, 0.09, 0.05, 0.04, -12.0, -25.0, 40));
        assert_eq!(class, CoarsePhoneClass::VowelOrSonorant);
        assert!(features.contains(&"energy.voiced".to_string()));
    }

    #[test]
    fn generate_phone_class_hypotheses_produces_one_per_frame() {
        let stream = AcousticFeatureStream {
            hop_ms: 10.0,
            frames: vec![
                frame(0.06, 0.09, 0.05, 0.04, -12.0, -25.0, 0),
                frame(0.04, 0.06, 0.20, 0.05, -20.0, -15.0, 10),
                frame(0.002, 0.003, 0.05, 0.02, -50.0, -55.0, 20),
            ],
        };
        let hyps = generate_phone_class_hypotheses(&stream);
        assert_eq!(hyps.len(), 3);
        for hyp in &hyps {
            assert_eq!(hyp.kind, SpanHypothesisKind::PhoneClassCandidate);
            assert_eq!(hyp.source, HypothesisSource::PhoneClassifier);
            assert_eq!(hyp.status, HypothesisStatus::Provisional);
            assert!(hyp.confidence > 0.0 && hyp.confidence <= 1.0);
        }
    }

    #[test]
    fn phone_class_hypothesis_provenance_contains_zcr() {
        let stream = AcousticFeatureStream {
            hop_ms: 10.0,
            frames: vec![frame(0.06, 0.09, 0.07, 0.03, -12.0, -25.0, 0)],
        };
        let hyps = generate_phone_class_hypotheses(&stream);
        let prov = hyps[0].provenance.as_object().expect("object");
        assert!(prov.contains_key("zcr"));
    }
}
