//! Cheap acoustic feature frames for mechanical speech-recognition generators.
//!
//! [`AcousticFeatureFrame`] is a compact per-frame summary that all mechanical
//! recognisers can consume.  [`build_feature_stream`] derives it from the
//! energy envelope and an optional spectrogram level that are already computed
//! by the acoustic analyser.

use serde::{Deserialize, Serialize};

use crate::audio::acoustic::{EnergyEnvelope, SpectrogramLevel};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Compact acoustic features for a single analysis frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcousticFeatureFrame {
    /// Frame start time in milliseconds.
    pub frame_start_ms: u64,
    /// Frame end time in milliseconds.
    pub frame_end_ms: u64,
    /// Root-mean-square energy (linear, not dB).
    pub rms_energy: f32,
    /// Peak absolute sample amplitude.
    pub peak_amplitude: f32,
    /// Zero-crossing rate: fraction of adjacent sample pairs that cross zero (0.0–1.0).
    pub zero_crossing_rate: f32,
    /// Spectral flux: mean absolute difference between this and the previous spectral frame.
    /// 0.0 if no previous frame or no spectrogram level is available.
    pub spectral_flux: f32,
    /// Mean spectral energy in the low band (0–1 200 Hz), in dBFS.
    pub low_band_energy_db: f32,
    /// Mean spectral energy in the high band (3 000–7 000 Hz), in dBFS.
    pub high_band_energy_db: f32,
}

/// A time-ordered stream of compact acoustic feature frames.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcousticFeatureStream {
    /// Hop size between frames in milliseconds.
    pub hop_ms: f32,
    /// Ordered frames.
    pub frames: Vec<AcousticFeatureFrame>,
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Derive an [`AcousticFeatureStream`] from existing acoustic analysis outputs.
///
/// * `samples`         — raw mono PCM samples at `sample_rate`.
/// * `sample_rate`     — sample rate in Hz.
/// * `energy_envelope` — pre-computed energy envelope (hop / window in ms).
/// * `level`           — optional spectrogram level for spectral features.
///
/// When `level` is `None` the spectral flux and band-energy fields are filled
/// with `−60 dB` (a conventional silence floor).
pub fn build_feature_stream(
    samples: &[f32],
    sample_rate: u32,
    energy_envelope: &EnergyEnvelope,
    level: Option<&SpectrogramLevel>,
) -> AcousticFeatureStream {
    let hop_samples = ((sample_rate as f32 * energy_envelope.hop_ms) / 1000.0)
        .round()
        .max(1.0) as usize;
    let window_samples = ((sample_rate as f32 * energy_envelope.window_ms) / 1000.0)
        .round()
        .max(1.0) as usize;

    let mut frames: Vec<AcousticFeatureFrame> = Vec::with_capacity(energy_envelope.frames.len());
    let mut prev_spec: Option<Vec<f32>> = None;

    for (frame_index, energy_frame) in energy_envelope.frames.iter().enumerate() {
        let start = (frame_index * hop_samples).min(samples.len());
        let end = (start + window_samples).min(samples.len());
        let frame_samples = &samples[start..end];

        let rms_energy = energy_frame.rms_energy;
        let peak_amplitude = energy_frame.peak_energy;
        let zero_crossing_rate = compute_zero_crossing_rate(frame_samples);

        let (low_band_energy_db, high_band_energy_db, spectral_flux) = if let Some(lvl) = level {
            let spec_frame = spectral_frame_for_ms(lvl, energy_frame.frame_start_ms);
            let (low, high) = band_energies(lvl, spec_frame.as_deref());
            let flux = compute_spectral_flux(spec_frame.as_deref(), prev_spec.as_deref());
            prev_spec = spec_frame;
            (low, high, flux)
        } else {
            prev_spec = None;
            (-60.0, -60.0, 0.0)
        };

        frames.push(AcousticFeatureFrame {
            frame_start_ms: energy_frame.frame_start_ms,
            frame_end_ms: energy_frame.frame_end_ms,
            rms_energy,
            peak_amplitude,
            zero_crossing_rate,
            spectral_flux,
            low_band_energy_db,
            high_band_energy_db,
        });
    }

    AcousticFeatureStream {
        hop_ms: energy_envelope.hop_ms,
        frames,
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Fraction of adjacent sample pairs that change sign (0.0–1.0).
pub(crate) fn compute_zero_crossing_rate(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let crossings = samples
        .windows(2)
        .filter(|pair| {
            let a = pair[0];
            let b = pair[1];
            (a >= 0.0 && b < 0.0) || (a < 0.0 && b >= 0.0)
        })
        .count();
    crossings as f32 / (samples.len() - 1) as f32
}

/// Return the spectrogram frame whose start is closest to `ms`.
fn spectral_frame_for_ms(level: &SpectrogramLevel, ms: u64) -> Option<Vec<f32>> {
    if level.hop_ms <= 0.0 || level.frames.is_empty() {
        return None;
    }
    let frame_index = ((ms as f32) / level.hop_ms).floor() as usize;
    level.frames.get(frame_index).cloned()
}

/// Mean dBFS in the low (0–1 200 Hz) and high (3 000–7 000 Hz) bands.
fn band_energies(level: &SpectrogramLevel, spec_frame: Option<&[f32]>) -> (f32, f32) {
    let spec_frame = match spec_frame {
        Some(f) if !f.is_empty() => f,
        _ => return (-60.0, -60.0),
    };
    if level.bin_hz <= 0.0 {
        return (-60.0, -60.0);
    }
    let low_max =
        ((1200.0 / level.bin_hz).floor() as usize).min(spec_frame.len().saturating_sub(1));
    let high_min =
        ((3000.0 / level.bin_hz).floor() as usize).min(spec_frame.len().saturating_sub(1));
    let high_max =
        ((7000.0 / level.bin_hz).floor() as usize).min(spec_frame.len().saturating_sub(1));

    let low = if low_max >= 1 {
        spec_frame[1..=low_max].iter().copied().sum::<f32>() / low_max as f32
    } else {
        -60.0
    };
    let high = if high_min <= high_max {
        let count = high_max - high_min + 1;
        spec_frame[high_min..=high_max]
            .iter()
            .copied()
            .sum::<f32>()
            / count as f32
    } else {
        -60.0
    };
    (low, high)
}

/// Mean absolute spectral difference between the current and previous frame,
/// normalised to 0.0–1.0.
fn compute_spectral_flux(current: Option<&[f32]>, previous: Option<&[f32]>) -> f32 {
    let (cur, prev) = match (current, previous) {
        (Some(c), Some(p)) if c.len() == p.len() && !c.is_empty() => (c, p),
        _ => return 0.0,
    };
    let sum: f32 = cur
        .iter()
        .zip(prev.iter())
        .map(|(c, p)| (c - p).abs())
        .sum();
    (sum / cur.len() as f32).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::acoustic::{EnergyFrame, EnergyEnvelope};

    fn make_envelope(frames: Vec<EnergyFrame>) -> EnergyEnvelope {
        EnergyEnvelope {
            window_ms: 20.0,
            hop_ms: 10.0,
            frames,
        }
    }

    fn make_energy_frame(start_ms: u64, rms: f32, peak: f32) -> EnergyFrame {
        EnergyFrame {
            frame_start_ms: start_ms,
            frame_end_ms: start_ms + 10,
            rms_energy: rms,
            peak_energy: peak,
            dbfs: if rms > 0.0 {
                20.0 * rms.log10()
            } else {
                -120.0
            },
        }
    }

    #[test]
    fn zero_crossing_rate_is_zero_for_constant_signal() {
        let samples = vec![0.5f32; 100];
        assert_eq!(compute_zero_crossing_rate(&samples), 0.0);
    }

    #[test]
    fn zero_crossing_rate_is_high_for_alternating_signal() {
        let samples: Vec<f32> = (0..100).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 }).collect();
        let zcr = compute_zero_crossing_rate(&samples);
        assert!(zcr > 0.9, "expected high ZCR, got {zcr}");
    }

    #[test]
    fn build_feature_stream_has_same_frame_count_as_envelope() {
        let sample_rate = 16_000u32;
        let samples: Vec<f32> = (0..3200)
            .map(|i| ((2.0 * std::f32::consts::PI * 440.0 * i as f32) / sample_rate as f32).sin())
            .collect();
        let envelope = make_envelope(vec![
            make_energy_frame(0, 0.05, 0.07),
            make_energy_frame(10, 0.08, 0.10),
            make_energy_frame(20, 0.06, 0.09),
        ]);
        let stream = build_feature_stream(&samples, sample_rate, &envelope, None);
        assert_eq!(stream.frames.len(), 3);
        assert_eq!(stream.hop_ms, 10.0);
    }

    #[test]
    fn build_feature_stream_rms_matches_envelope() {
        let sample_rate = 16_000u32;
        let samples = vec![0.0f32; 4800];
        let energy_frame = make_energy_frame(0, 0.042, 0.06);
        let envelope = make_envelope(vec![energy_frame.clone()]);
        let stream = build_feature_stream(&samples, sample_rate, &envelope, None);
        assert_eq!(stream.frames.len(), 1);
        assert!((stream.frames[0].rms_energy - 0.042).abs() < 1e-6);
    }

    #[test]
    fn spectral_flux_is_zero_without_previous_frame() {
        let sample_rate = 16_000u32;
        let samples: Vec<f32> = (0..1600)
            .map(|i| ((2.0 * std::f32::consts::PI * 440.0 * i as f32) / sample_rate as f32).sin())
            .collect();
        let envelope = make_envelope(vec![make_energy_frame(0, 0.05, 0.07)]);
        // With one frame there is no previous frame; spectral_flux should be 0.
        let stream = build_feature_stream(&samples, sample_rate, &envelope, None);
        assert_eq!(stream.frames[0].spectral_flux, 0.0);
    }
}
