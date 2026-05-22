//! Cheap acoustic feature frames for mechanical speech-recognition generators.
//!
//! [`AcousticFeatureFrame`] is a compact per-frame summary that all mechanical
//! recognisers can consume. [`build_feature_stream`] derives it from the
//! energy envelope and an optional spectrogram level that are already computed
//! by the acoustic analyser.

use serde::{Deserialize, Serialize};

use crate::audio::acoustic::{EnergyEnvelope, SpectrogramLevel};
use crate::audio::noise_floor::AdaptiveNoiseFloor;

const DEFAULT_DB_FLOOR: f32 = -60.0;
const DEFAULT_BAND_SUMMARY_DB: [f32; 4] = [DEFAULT_DB_FLOOR; 4];
const MIN_SPEECH_FLOOR_RMS: f32 = 0.0005;
const SPEECH_ENERGY_OVER_FLOOR_RATIO: f32 = 1.7;
const SPEECH_MAX_ZCR: f32 = 0.18;
const SPEECH_MAX_SPECTRAL_FLUX: f32 = 0.25;
const SPEECH_BAND_SHAPE_MARGIN_DB: f32 = 6.0;
const SPEECH_MAX_NOISE_LIKENESS: f32 = 0.72;

/// Speech-layer alias for framewise acoustic features.
pub type SpeechFrameFeatures = AcousticFeatureFrame;
/// Speech-layer alias for a stream of framewise acoustic features.
pub type SpeechFeatureStream = AcousticFeatureStream;

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
    /// Spectral centroid in Hz.
    pub spectral_centroid_hz: f32,
    /// Spectral rolloff (85% cumulative energy) in Hz.
    pub spectral_rolloff_hz: f32,
    /// Coarse broadband noise-likeness estimate (0.0–1.0).
    pub broadband_noise_likeness: f32,
    /// Adaptive room-noise floor estimate (RMS).
    pub noise_floor_rms: f32,
    /// Energy surplus over floor, normalized by floor.
    pub energy_over_noise: f32,
    /// SNR-ish measure in dB (frame RMS vs. tracked noise floor).
    pub snr_db: f32,
    /// Spectral deviation from learned room-floor spectrum (0.0–1.0).
    pub spectral_deviation: f32,
    /// Optional per-band summary (low, low-mid, upper-mid, high) in dBFS.
    pub band_energy_db: [f32; 4],
}

impl Default for AcousticFeatureFrame {
    fn default() -> Self {
        Self {
            frame_start_ms: 0,
            frame_end_ms: 0,
            rms_energy: 0.0,
            peak_amplitude: 0.0,
            zero_crossing_rate: 0.0,
            spectral_flux: 0.0,
            low_band_energy_db: DEFAULT_DB_FLOOR,
            high_band_energy_db: DEFAULT_DB_FLOOR,
            spectral_centroid_hz: 0.0,
            spectral_rolloff_hz: 0.0,
            broadband_noise_likeness: 0.0,
            noise_floor_rms: 0.0,
            energy_over_noise: 0.0,
            snr_db: 0.0,
            spectral_deviation: 0.0,
            band_energy_db: DEFAULT_BAND_SUMMARY_DB,
        }
    }
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
/// When `level` is `None` the spectral fields are filled with conventional
/// silence defaults while preserving deterministic temporal features.
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
    let mut spectral_floor: Option<Vec<f32>> = None;
    let mut noise_floor = AdaptiveNoiseFloor::default();

    for (frame_index, energy_frame) in energy_envelope.frames.iter().enumerate() {
        let start = (frame_index * hop_samples).min(samples.len());
        let end = (start + window_samples).min(samples.len());
        let frame_samples = &samples[start..end];

        let rms_energy = energy_frame.rms_energy;
        let peak_amplitude = energy_frame.peak_energy;
        let zero_crossing_rate = compute_zero_crossing_rate(frame_samples);

        let (
            low_band_energy_db,
            high_band_energy_db,
            spectral_flux,
            spectral_centroid_hz,
            spectral_rolloff_hz,
            band_energy_db,
            spec_frame,
        ) = if let Some(lvl) = level {
            let current_spec = spectral_frame_for_ms(lvl, energy_frame.frame_start_ms);
            let (low, high) = band_energies(lvl, current_spec.as_deref());
            let flux = compute_spectral_flux(current_spec.as_deref(), prev_spec.as_deref());
            let centroid_hz = compute_spectral_centroid_hz(lvl, current_spec.as_deref());
            let rolloff_hz = compute_spectral_rolloff_hz(lvl, current_spec.as_deref(), 0.85);
            let band_summary = band_energy_summary(lvl, current_spec.as_deref());
            (
                low,
                high,
                flux,
                centroid_hz,
                rolloff_hz,
                band_summary,
                current_spec,
            )
        } else {
            (
                DEFAULT_DB_FLOOR,
                DEFAULT_DB_FLOOR,
                0.0,
                0.0,
                0.0,
                DEFAULT_BAND_SUMMARY_DB,
                None,
            )
        };

        let broadband_noise_likeness = compute_broadband_noise_likeness(
            zero_crossing_rate,
            spectral_flux,
            high_band_energy_db,
            low_band_energy_db,
        );
        let speech_like = is_probable_speech(
            rms_energy,
            noise_floor.current_rms(),
            zero_crossing_rate,
            spectral_flux,
            low_band_energy_db,
            high_band_energy_db,
            broadband_noise_likeness,
        );
        let floor_observation = noise_floor.observe(rms_energy, speech_like);

        let spectral_deviation = match spec_frame.as_deref() {
            Some(current) => {
                let deviation = spectral_deviation_from_floor(current, spectral_floor.as_deref());
                if !speech_like {
                    spectral_floor = Some(update_spectral_floor(
                        spectral_floor.as_deref(),
                        current,
                        0.015,
                    ));
                }
                deviation
            }
            None => 0.0,
        };

        prev_spec = spec_frame;

        frames.push(AcousticFeatureFrame {
            frame_start_ms: energy_frame.frame_start_ms,
            frame_end_ms: energy_frame.frame_end_ms,
            rms_energy,
            peak_amplitude,
            zero_crossing_rate,
            spectral_flux,
            low_band_energy_db,
            high_band_energy_db,
            spectral_centroid_hz,
            spectral_rolloff_hz,
            broadband_noise_likeness,
            noise_floor_rms: floor_observation.noise_floor_rms,
            energy_over_noise: floor_observation.energy_over_noise,
            snr_db: floor_observation.snr_db,
            spectral_deviation,
            band_energy_db,
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
        _ => return (DEFAULT_DB_FLOOR, DEFAULT_DB_FLOOR),
    };
    if level.bin_hz <= 0.0 {
        return (DEFAULT_DB_FLOOR, DEFAULT_DB_FLOOR);
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
        DEFAULT_DB_FLOOR
    };
    let high = if high_min <= high_max {
        let count = high_max - high_min + 1;
        spec_frame[high_min..=high_max].iter().copied().sum::<f32>() / count as f32
    } else {
        DEFAULT_DB_FLOOR
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

fn db_to_linear(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

fn compute_spectral_centroid_hz(level: &SpectrogramLevel, spec_frame: Option<&[f32]>) -> f32 {
    let spec = match spec_frame {
        Some(frame) if !frame.is_empty() && level.bin_hz > 0.0 => frame,
        _ => return 0.0,
    };
    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    for (idx, value_db) in spec.iter().enumerate() {
        let value = db_to_linear(*value_db).max(0.0);
        let hz = idx as f32 * level.bin_hz;
        numerator += hz * value;
        denominator += value;
    }
    if denominator <= f32::EPSILON {
        0.0
    } else {
        numerator / denominator
    }
}

fn compute_spectral_rolloff_hz(
    level: &SpectrogramLevel,
    spec_frame: Option<&[f32]>,
    threshold: f32,
) -> f32 {
    let spec = match spec_frame {
        Some(frame) if !frame.is_empty() && level.bin_hz > 0.0 => frame,
        _ => return 0.0,
    };
    let total: f32 = spec.iter().map(|value| db_to_linear(*value).max(0.0)).sum();
    if total <= f32::EPSILON {
        return 0.0;
    }
    let target = total * threshold.clamp(0.0, 1.0);
    let mut cumulative = 0.0f32;
    for (idx, value_db) in spec.iter().enumerate() {
        cumulative += db_to_linear(*value_db).max(0.0);
        if cumulative >= target {
            return idx as f32 * level.bin_hz;
        }
    }
    (spec.len().saturating_sub(1) as f32) * level.bin_hz
}

fn mean_db_in_range(spec: &[f32], bin_hz: f32, min_hz: f32, max_hz: f32) -> f32 {
    if spec.is_empty() || bin_hz <= 0.0 || min_hz > max_hz {
        return DEFAULT_DB_FLOOR;
    }
    let start = ((min_hz / bin_hz).floor() as usize).min(spec.len().saturating_sub(1));
    let end = ((max_hz / bin_hz).floor() as usize).min(spec.len().saturating_sub(1));
    if end < start {
        return DEFAULT_DB_FLOOR;
    }
    let count = end - start + 1;
    spec[start..=end].iter().copied().sum::<f32>() / count as f32
}

fn band_energy_summary(level: &SpectrogramLevel, spec_frame: Option<&[f32]>) -> [f32; 4] {
    let spec = match spec_frame {
        Some(frame) if !frame.is_empty() => frame,
        _ => return DEFAULT_BAND_SUMMARY_DB,
    };
    [
        mean_db_in_range(spec, level.bin_hz, 0.0, 600.0),
        mean_db_in_range(spec, level.bin_hz, 600.0, 1800.0),
        mean_db_in_range(spec, level.bin_hz, 1800.0, 3600.0),
        mean_db_in_range(spec, level.bin_hz, 3600.0, 8000.0),
    ]
}

fn compute_broadband_noise_likeness(
    zcr: f32,
    spectral_flux: f32,
    high_band_energy_db: f32,
    low_band_energy_db: f32,
) -> f32 {
    let zcr_term = (zcr / 0.28).clamp(0.0, 1.0);
    let flux_term = (spectral_flux / 0.30).clamp(0.0, 1.0);
    let tilt_term = ((high_band_energy_db - low_band_energy_db + 12.0) / 24.0).clamp(0.0, 1.0);
    (zcr_term * 0.45 + flux_term * 0.30 + tilt_term * 0.25).clamp(0.0, 1.0)
}

fn is_probable_speech(
    rms: f32,
    floor_rms: f32,
    zcr: f32,
    spectral_flux: f32,
    low_band_energy_db: f32,
    high_band_energy_db: f32,
    noise_likeness: f32,
) -> bool {
    let floor = floor_rms.max(MIN_SPEECH_FLOOR_RMS);
    let energetic = rms >= floor * SPEECH_ENERGY_OVER_FLOOR_RATIO;
    let periodicish = zcr <= SPEECH_MAX_ZCR && spectral_flux <= SPEECH_MAX_SPECTRAL_FLUX;
    let speech_band_shape = low_band_energy_db >= high_band_energy_db - SPEECH_BAND_SHAPE_MARGIN_DB;
    energetic && periodicish && speech_band_shape && noise_likeness < SPEECH_MAX_NOISE_LIKENESS
}

fn spectral_deviation_from_floor(current: &[f32], floor: Option<&[f32]>) -> f32 {
    let floor = match floor {
        Some(f) if f.len() == current.len() && !f.is_empty() => f,
        _ => return 0.0,
    };
    let delta_sum: f32 = current
        .iter()
        .zip(floor.iter())
        .map(|(cur, base)| (cur - base).abs())
        .sum();
    ((delta_sum / current.len() as f32) / 30.0).clamp(0.0, 1.0)
}

fn update_spectral_floor(previous: Option<&[f32]>, current: &[f32], alpha: f32) -> Vec<f32> {
    let alpha = alpha.clamp(0.0, 1.0);
    match previous {
        Some(prev) if prev.len() == current.len() => prev
            .iter()
            .zip(current.iter())
            .map(|(base, cur)| base * (1.0 - alpha) + cur * alpha)
            .collect(),
        _ => current.to_vec(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::acoustic::{EnergyEnvelope, EnergyFrame, analyze_mono_samples};

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

    fn deterministic_noise(len: usize) -> Vec<f32> {
        // Numerical Recipes LCG parameters: quick deterministic pseudo-noise for tests.
        const LCG_A: u32 = 1_664_525;
        const LCG_C: u32 = 1_013_904_223;
        let mut state: u32 = 0x1234_5678;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
                let unit = ((state >> 8) as f32) / ((u32::MAX >> 8) as f32);
                (unit * 2.0 - 1.0) * 0.18
            })
            .collect()
    }

    fn sine_wave(sample_rate: u32, hz: f32, seconds: f32, amp: f32) -> Vec<f32> {
        let count = (sample_rate as f32 * seconds).round() as usize;
        (0..count)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * hz * t).sin() * amp
            })
            .collect()
    }

    fn click_burst(sample_rate: u32, seconds: f32) -> Vec<f32> {
        let count = (sample_rate as f32 * seconds).round() as usize;
        let mut samples = vec![0.0; count];
        let step = (sample_rate / 20) as usize;
        for idx in (0..count).step_by(step.max(1)) {
            samples[idx] = if (idx / step).is_multiple_of(2) {
                0.9
            } else {
                -0.9
            };
        }
        samples
    }

    fn vowelish_harmonic(sample_rate: u32, seconds: f32) -> Vec<f32> {
        let count = (sample_rate as f32 * seconds).round() as usize;
        (0..count)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                let f0 = 140.0;
                (2.0 * std::f32::consts::PI * f0 * t).sin() * 0.24
                    + (2.0 * std::f32::consts::PI * f0 * 2.0 * t).sin() * 0.12
                    + (2.0 * std::f32::consts::PI * f0 * 3.0 * t).sin() * 0.07
            })
            .collect()
    }

    fn stream_from_samples(samples: &[f32], sample_rate: u32) -> AcousticFeatureStream {
        let analysis = analyze_mono_samples(samples, sample_rate);
        build_feature_stream(
            samples,
            sample_rate,
            &analysis.energy_envelope,
            analysis.spectrogram.levels.first(),
        )
    }

    fn mean_metric(
        stream: &AcousticFeatureStream,
        f: impl Fn(&AcousticFeatureFrame) -> f32,
    ) -> f32 {
        if stream.frames.is_empty() {
            return 0.0;
        }
        stream.frames.iter().map(f).sum::<f32>() / stream.frames.len() as f32
    }

    #[test]
    fn zero_crossing_rate_is_zero_for_constant_signal() {
        let samples = vec![0.5f32; 100];
        assert_eq!(compute_zero_crossing_rate(&samples), 0.0);
    }

    #[test]
    fn zero_crossing_rate_is_high_for_alternating_signal() {
        let samples: Vec<f32> = (0..100)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
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
        let stream = build_feature_stream(&samples, sample_rate, &envelope, None);
        assert_eq!(stream.frames[0].spectral_flux, 0.0);
    }

    #[test]
    fn synthetic_profiles_are_distinguishable() {
        let sr = 16_000;
        let silence = vec![0.0f32; sr as usize / 2];
        let noise = deterministic_noise(sr as usize / 2);
        let sine = sine_wave(sr, 220.0, 0.5, 0.22);
        let clicks = click_burst(sr, 0.5);
        let vowelish = vowelish_harmonic(sr, 0.5);

        let silence_stream = stream_from_samples(&silence, sr);
        let noise_stream = stream_from_samples(&noise, sr);
        let sine_stream = stream_from_samples(&sine, sr);
        let click_stream = stream_from_samples(&clicks, sr);
        let vowel_stream = stream_from_samples(&vowelish, sr);

        let silence_rms = mean_metric(&silence_stream, |f| f.rms_energy);
        let noise_rms = mean_metric(&noise_stream, |f| f.rms_energy);
        let sine_zcr = mean_metric(&sine_stream, |f| f.zero_crossing_rate);
        let noise_zcr = mean_metric(&noise_stream, |f| f.zero_crossing_rate);
        let click_flux = mean_metric(&click_stream, |f| f.spectral_flux);
        let sine_flux = mean_metric(&sine_stream, |f| f.spectral_flux);
        let noise_likeness_noise = mean_metric(&noise_stream, |f| f.broadband_noise_likeness);
        let noise_likeness_vowel = mean_metric(&vowel_stream, |f| f.broadband_noise_likeness);

        assert!(silence_rms < 0.0015, "silence rms too high: {silence_rms}");
        assert!(noise_rms > silence_rms * 10.0);
        assert!(noise_zcr > sine_zcr, "expected noise zcr > sine zcr");
        assert!(
            click_flux > sine_flux,
            "expected clicks to have stronger flux"
        );
        assert!(
            noise_likeness_noise > noise_likeness_vowel,
            "expected broadband noise to score noisier than vowel-like harmonic"
        );
    }

    #[test]
    fn adaptive_noise_floor_resists_short_transients() {
        let sr = 16_000;
        let mut samples = deterministic_noise(sr as usize / 2)
            .iter()
            .map(|s| s * 0.35)
            .collect::<Vec<_>>();
        let burst_start = sr as usize / 5;
        let burst_end = burst_start + (sr as usize / 20);
        for sample in &mut samples[burst_start..burst_end] {
            *sample += 0.45;
        }

        let stream = stream_from_samples(&samples, sr);
        let start_frame = stream.frames.len() / 8;
        let end_frame = (stream.frames.len() * 7) / 8;
        let before = stream.frames[start_frame].noise_floor_rms;
        let after = stream.frames[end_frame].noise_floor_rms;

        assert!(
            (after - before).abs() < 0.01,
            "noise floor drifted too far after transient: before={before} after={after}"
        );
    }
}
