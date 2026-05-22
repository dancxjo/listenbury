//! Vocal-tract filter descriptors for source/filter analysis.
//!
//! These types summarise the resonant properties of the vocal tract on a
//! per-frame basis, built primarily from formant-peak evidence.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single formant estimate with frequency, bandwidth, amplitude, and
/// confidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FormantEstimation {
    /// Formant frequency in Hz.
    pub frequency_hz: f32,
    /// Bandwidth (−3 dB width) in Hz. `None` if not estimated.
    pub bandwidth_hz: Option<f32>,
    /// Relative amplitude of this formant peak in dBFS.
    pub amplitude_db: f32,
    /// Confidence in this formant estimate (0.0–1.0).
    pub confidence: f32,
}

/// Full vocal-tract filter estimate for a single analysis frame.
///
/// The four formant slots map to the conventional acoustic phonetics labels
/// F1–F4.  Any slot may be `None` when the analysis has insufficient evidence
/// to place a peak there.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VocalTractFilterEstimate {
    /// F1 – first formant (≈ 300–1 000 Hz for most vowels).
    pub f1: Option<FormantEstimation>,
    /// F2 – second formant (≈ 700–2 500 Hz for most vowels).
    pub f2: Option<FormantEstimation>,
    /// F3 – third formant (≈ 1 800–3 500 Hz).
    pub f3: Option<FormantEstimation>,
    /// F4 – fourth formant (≈ 3 000–5 000 Hz).
    pub f4: Option<FormantEstimation>,
    /// Nasality placeholder (0.0 = oral, 1.0 = maximally nasal).
    /// `None` when no nasality detector is active.
    pub nasality: Option<f32>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_has_no_formants() {
        let est = VocalTractFilterEstimate::default();
        assert!(est.f1.is_none());
        assert!(est.f2.is_none());
        assert!(est.f3.is_none());
        assert!(est.f4.is_none());
        assert!(est.nasality.is_none());
    }

    #[test]
    fn formant_estimation_round_trips_json() {
        let fe = FormantEstimation {
            frequency_hz: 500.0,
            bandwidth_hz: Some(80.0),
            amplitude_db: -3.0,
            confidence: 0.75,
        };
        let json = serde_json::to_string(&fe).unwrap();
        let back: FormantEstimation = serde_json::from_str(&json).unwrap();
        assert_eq!(fe, back);
    }
}
