//! Glottal source and voicing descriptors for source/filter analysis.
//!
//! These types describe the acoustic properties of the glottal excitation
//! signal on a per-frame basis.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Voicing estimate for a single analysis frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VoicingEstimate {
    /// Estimated fundamental frequency in Hz. `None` if the frame is unvoiced.
    pub f0_hz: Option<f32>,
    /// Confidence in the F0 estimate (0.0–1.0). Zero when unvoiced.
    pub f0_confidence: f32,
    /// Probability that this frame is voiced (0.0–1.0).
    pub voicing_probability: f32,
    /// Harmonics-to-noise ratio proxy in dB. Higher values indicate a more
    /// periodic signal.
    pub hnr_db: f32,
}

/// Glottal source estimate for a single analysis frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlottalSourceEstimate {
    /// Spectral tilt in dB/octave. Negative values (e.g. −6) indicate a
    /// falling spectrum typical of modal phonation. Values near zero indicate
    /// a flat, breathy, or aspirated source.
    pub spectral_tilt_db_per_octave: f32,
    /// Breathiness proxy: proportion of aspiration noise in the source
    /// (0.0 = fully modal, 1.0 = fully breathy/whispered).
    pub breathiness: f32,
    /// Open-quotient estimate (0.0–1.0). A value around 0.5 indicates typical
    /// modal phonation; higher values indicate a more open/breathy glottis.
    pub open_quotient: f32,
}

/// Noise / frication estimate for a single analysis frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NoiseEstimate {
    /// Frication / turbulence energy proxy (0.0–1.0, normalised).
    /// High values indicate fricative or affricate noise.
    pub frication_energy: f32,
    /// Fraction of total energy attributable to noise rather than periodic
    /// harmonics (0.0–1.0).
    pub noise_ratio: f32,
}

// ---------------------------------------------------------------------------
// Default implementations
// ---------------------------------------------------------------------------

impl Default for VoicingEstimate {
    fn default() -> Self {
        Self {
            f0_hz: None,
            f0_confidence: 0.0,
            voicing_probability: 0.0,
            hnr_db: -20.0,
        }
    }
}

impl Default for GlottalSourceEstimate {
    fn default() -> Self {
        Self {
            spectral_tilt_db_per_octave: -6.0,
            breathiness: 0.0,
            open_quotient: 0.5,
        }
    }
}

impl Default for NoiseEstimate {
    fn default() -> Self {
        Self {
            frication_energy: 0.0,
            noise_ratio: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sensible() {
        let v = VoicingEstimate::default();
        assert!(v.f0_hz.is_none());
        assert_eq!(v.voicing_probability, 0.0);
        assert!(v.hnr_db < 0.0);

        let g = GlottalSourceEstimate::default();
        assert!(g.spectral_tilt_db_per_octave < 0.0);
        assert_eq!(g.breathiness, 0.0);
        assert!((g.open_quotient - 0.5).abs() < 1e-6);

        let n = NoiseEstimate::default();
        assert_eq!(n.frication_energy, 0.0);
        assert_eq!(n.noise_ratio, 0.0);
    }

    #[test]
    fn voicing_estimate_serialization_round_trips() {
        let v = VoicingEstimate {
            f0_hz: Some(220.0),
            f0_confidence: 0.85,
            voicing_probability: 0.9,
            hnr_db: 12.3,
        };
        let json = serde_json::to_string(&v).unwrap();
        let back: VoicingEstimate = serde_json::from_str(&json).unwrap();
        assert_eq!(v, back);
    }
}
