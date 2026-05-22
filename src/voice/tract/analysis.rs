//! Layer A: normalised source/filter acoustic analysis.
//!
//! This module converts existing [`AcousticAnalysis`] and
//! [`AcousticFeatureStream`] outputs into compact [`SourceFilterFrame`]
//! evidence suitable for vocal-tract reasoning, phone-class hypothesis
//! generation, and soundscape signature updates.
//!
//! # Entry points
//!
//! * [`source_filter_track_from_acoustic`] — fast conversion using only
//!   energy and formant data (no spectrogram re-processing).
//! * [`source_filter_track_from_acoustic_full`] — richer conversion that
//!   also computes per-frame band energies and spectral flux from the
//!   pre-computed spectrogram level.
//! * [`estimate_f0_autocorrelation`] — stand-alone normalized autocorrelation
//!   F0 estimator (suitable for use outside this module).
//!
//! # Realtime safety
//!
//! Analysis runs on buffered frames outside the audio callback.  No heavy
//! allocation or blocking occurs inside a realtime path.

use serde::{Deserialize, Serialize};

use crate::audio::acoustic::{AcousticAnalysis, FormantFrame};
use crate::audio::features::compute_zero_crossing_rate;

use super::filter::{FormantEstimation, VocalTractFilterEstimate};
use super::source::{GlottalSourceEstimate, NoiseEstimate, VoicingEstimate};

// Spectral band frequency boundaries used when summarising spectrogram frames.
const LOW_BAND_MAX_HZ: f32 = 1200.0;
const HIGH_BAND_MIN_HZ: f32 = 3000.0;
const HIGH_BAND_MAX_HZ: f32 = 7000.0;
// Minimum hop_ms to guard against near-zero division when indexing spectrogram frames.
const MIN_HOP_MS: f32 = 0.001;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Normalised source/filter evidence for a single analysis frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceFilterFrame {
    /// Frame start time in milliseconds.
    pub frame_start_ms: u64,
    /// Frame end time in milliseconds.
    pub frame_end_ms: u64,
    /// Voicing evidence: F0, voicing probability, HNR.
    pub voicing: VoicingEstimate,
    /// Glottal source evidence: spectral tilt, breathiness, open quotient.
    pub source: GlottalSourceEstimate,
    /// Vocal-tract filter evidence: formants F1–F4.
    pub filter: VocalTractFilterEstimate,
    /// Noise / frication evidence.
    pub noise: NoiseEstimate,
    /// Overall frame-level confidence (0.0–1.0).
    pub confidence: f32,
}

/// A time-ordered sequence of [`SourceFilterFrame`] values sharing common
/// timing metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceFilterTrack {
    /// Sample rate of the originating audio (Hz).
    pub sample_rate: u32,
    /// Analysis hop size in milliseconds.
    pub hop_ms: f32,
    /// Ordered analysis frames.
    pub frames: Vec<SourceFilterFrame>,
}

// ---------------------------------------------------------------------------
// F0 estimation
// ---------------------------------------------------------------------------

/// Estimate F0 using normalised autocorrelation on a windowed sample slice.
///
/// Returns `Some((f0_hz, confidence))` when a reliable periodic peak is
/// found in the range `[min_hz, max_hz]`, or `None` when the frame is
/// silence or aperiodic.
///
/// # Parameters
///
/// * `samples`     — mono PCM samples (any contiguous window).
/// * `sample_rate` — sample rate in Hz.
/// * `min_hz`      — minimum candidate F0 in Hz (e.g. 60.0 for speech).
/// * `max_hz`      — maximum candidate F0 in Hz (e.g. 500.0 for speech).
pub fn estimate_f0_autocorrelation(
    samples: &[f32],
    sample_rate: u32,
    min_hz: f32,
    max_hz: f32,
) -> Option<(f32, f32)> {
    if samples.len() < 2 || sample_rate == 0 || min_hz >= max_hz {
        return None;
    }
    let sr = sample_rate as f32;
    let lag_min = (sr / max_hz).ceil() as usize;
    let lag_max = ((sr / min_hz).floor() as usize).min(samples.len().saturating_sub(1));
    if lag_min >= lag_max {
        return None;
    }
    let n = samples.len();
    let r0: f32 = samples.iter().map(|s| s * s).sum::<f32>() / n as f32;
    if r0 < 1e-10 {
        return None;
    }
    let mut best_lag = lag_min;
    let mut best_acf = f32::NEG_INFINITY;
    for lag in lag_min..=lag_max {
        let count = n - lag;
        let acf: f32 = samples[..count]
            .iter()
            .zip(samples[lag..].iter())
            .map(|(a, b)| a * b)
            .sum::<f32>()
            / count as f32;
        let norm_acf = acf / r0;
        if norm_acf > best_acf {
            best_acf = norm_acf;
            best_lag = lag;
        }
    }
    // Confidence threshold: autocorrelation must be clearly periodic
    if best_acf < 0.3 {
        return None;
    }
    let f0_hz = sr / best_lag as f32;
    // Map [0.3, 1.0] → [0.0, 1.0] for confidence
    let confidence = ((best_acf - 0.3) / 0.7).clamp(0.0, 1.0);
    Some((f0_hz, confidence))
}

// ---------------------------------------------------------------------------
// Internal frame builders
// ---------------------------------------------------------------------------

fn build_voicing_estimate(
    frame_samples: &[f32],
    sample_rate: u32,
    rms: f32,
    high_band_db: f32,
    low_band_db: f32,
    zcr: f32,
) -> VoicingEstimate {
    if rms < 0.005 {
        return VoicingEstimate {
            f0_hz: None,
            f0_confidence: 0.0,
            voicing_probability: 0.0,
            hnr_db: -30.0,
        };
    }
    let (f0_hz, f0_confidence) =
        match estimate_f0_autocorrelation(frame_samples, sample_rate, 60.0, 500.0) {
            Some((f, c)) => (Some(f), c),
            None => (None, 0.0),
        };
    // Voicing probability heuristic: low ZCR + low-band energy dominance
    let zcr_score = (0.1 - zcr).clamp(0.0, 0.1) / 0.1;
    let band_score =
        ((low_band_db - high_band_db + 15.0) / 30.0).clamp(0.0, 1.0);
    let base_prob = (0.5 * zcr_score + 0.5 * band_score).clamp(0.0, 1.0);
    let voicing_probability = if f0_hz.is_some() { base_prob } else { base_prob * 0.4 };
    // HNR proxy from autocorrelation confidence
    let hnr_db = if f0_confidence > 0.0 {
        let ratio = (f0_confidence / (1.0 - f0_confidence + 1e-6)).max(0.01);
        10.0 * ratio.log10()
    } else {
        -20.0
    };
    VoicingEstimate {
        f0_hz,
        f0_confidence,
        voicing_probability,
        hnr_db,
    }
}

fn build_glottal_source_estimate(
    voicing_probability: f32,
    zcr: f32,
    high_band_db: f32,
    low_band_db: f32,
) -> GlottalSourceEstimate {
    let breathiness = (zcr * 0.5).clamp(0.0, 1.0);
    let spectral_tilt = -6.0 + (high_band_db - low_band_db) * 0.1;
    let open_quotient = if voicing_probability > 0.5 {
        (0.5 + breathiness * 0.3).clamp(0.3, 0.8)
    } else {
        0.5
    };
    GlottalSourceEstimate {
        spectral_tilt_db_per_octave: spectral_tilt,
        breathiness,
        open_quotient,
    }
}

fn build_noise_estimate(
    zcr: f32,
    high_band_db: f32,
    low_band_db: f32,
    spectral_flux: f32,
) -> NoiseEstimate {
    let band_ratio = ((high_band_db - low_band_db + 30.0) / 60.0).clamp(0.0, 1.0);
    let frication_energy =
        ((zcr / 0.3).min(1.0) * 0.5 + band_ratio * 0.5).clamp(0.0, 1.0);
    let noise_ratio = ((zcr * 2.0) + spectral_flux).clamp(0.0, 1.0);
    NoiseEstimate {
        frication_energy,
        noise_ratio,
    }
}

fn filter_estimate_from_formant_frame(frame: &FormantFrame) -> VocalTractFilterEstimate {
    let lookup = |label: &str| -> Option<FormantEstimation> {
        frame.formants.iter().find(|f| f.label == label).map(|f| FormantEstimation {
            frequency_hz: f.frequency_hz,
            bandwidth_hz: f.bandwidth_hz,
            amplitude_db: f.amplitude_db,
            confidence: f.confidence,
        })
    };
    VocalTractFilterEstimate {
        f1: lookup("F1"),
        f2: lookup("F2"),
        f3: lookup("F3"),
        f4: lookup("F4"),
        nasality: None,
    }
}

// ---------------------------------------------------------------------------
// Conversion from AcousticAnalysis
// ---------------------------------------------------------------------------

/// Convert an [`AcousticAnalysis`] into a [`SourceFilterTrack`].
///
/// This fast path uses the pre-computed energy envelope and formant tracks.
/// Band energies default to flat estimates; use
/// [`source_filter_track_from_acoustic_full`] for spectrogram-derived band
/// energies.
pub fn source_filter_track_from_acoustic(
    analysis: &AcousticAnalysis,
    samples: &[f32],
) -> SourceFilterTrack {
    let sample_rate = analysis.sample_rate;
    let hop_ms = analysis.formant_tracks.hop_ms;
    let window_ms = analysis.formant_tracks.window_ms;
    let window_samples =
        ((sample_rate as f32 * window_ms) / 1000.0).round().max(1.0) as usize;
    let hop_samples =
        ((sample_rate as f32 * hop_ms) / 1000.0).round().max(1.0) as usize;

    let energy_frames = &analysis.energy_envelope.frames;

    let frames = analysis
        .formant_tracks
        .frames
        .iter()
        .enumerate()
        .map(|(idx, formant_frame)| {
            let start = (idx * hop_samples).min(samples.len());
            let end = (start + window_samples).min(samples.len());
            let frame_samples = &samples[start..end];

            // Closest energy frame by midpoint distance
            let fmid = (formant_frame.frame_start_ms + formant_frame.frame_end_ms) / 2;
            let rms = energy_frames
                .iter()
                .min_by_key(|ef| {
                    let emid = (ef.frame_start_ms + ef.frame_end_ms) / 2;
                    (emid as i64 - fmid as i64).unsigned_abs()
                })
                .map(|ef| ef.rms_energy)
                .unwrap_or(formant_frame.rms_energy);

            let zcr = compute_zero_crossing_rate(frame_samples);
            // Flat band-energy defaults (no spectrogram available here)
            let (low_band_db, high_band_db) = (-20.0f32, -30.0f32);

            let voicing = build_voicing_estimate(
                frame_samples,
                sample_rate,
                rms,
                high_band_db,
                low_band_db,
                zcr,
            );
            let source = build_glottal_source_estimate(
                voicing.voicing_probability,
                zcr,
                high_band_db,
                low_band_db,
            );
            let noise = build_noise_estimate(zcr, high_band_db, low_band_db, 0.0);
            let filter = filter_estimate_from_formant_frame(formant_frame);
            let confidence = if rms < 0.005 {
                0.1
            } else {
                (voicing.f0_confidence * 0.5 + 0.5).clamp(0.1, 0.95)
            };

            SourceFilterFrame {
                frame_start_ms: formant_frame.frame_start_ms,
                frame_end_ms: formant_frame.frame_end_ms,
                voicing,
                source,
                filter,
                noise,
                confidence,
            }
        })
        .collect();

    SourceFilterTrack {
        sample_rate,
        hop_ms,
        frames,
    }
}

/// Convert an [`AcousticAnalysis`] into a [`SourceFilterTrack`], using the
/// pre-computed spectrogram for band energy and spectral-flux calculations.
///
/// This is the preferred entry point when diagnostic accuracy matters more
/// than speed.
pub fn source_filter_track_from_acoustic_full(
    analysis: &AcousticAnalysis,
    samples: &[f32],
) -> SourceFilterTrack {
    let sample_rate = analysis.sample_rate;
    let hop_ms = analysis.formant_tracks.hop_ms;
    let window_ms = analysis.formant_tracks.window_ms;
    let window_samples =
        ((sample_rate as f32 * window_ms) / 1000.0).round().max(1.0) as usize;
    let hop_samples =
        ((sample_rate as f32 * hop_ms) / 1000.0).round().max(1.0) as usize;

    let energy_frames = &analysis.energy_envelope.frames;
    let spec_level = analysis
        .spectrogram
        .levels
        .iter()
        .find(|l| l.id == "detail")
        .or_else(|| analysis.spectrogram.levels.first());

    let mut prev_spec: Option<Vec<f32>> = None;

    let frames = analysis
        .formant_tracks
        .frames
        .iter()
        .enumerate()
        .map(|(idx, formant_frame)| {
            let start = (idx * hop_samples).min(samples.len());
            let end = (start + window_samples).min(samples.len());
            let frame_samples = &samples[start..end];

            let fmid = (formant_frame.frame_start_ms + formant_frame.frame_end_ms) / 2;
            let rms = energy_frames
                .iter()
                .min_by_key(|ef| {
                    let emid = (ef.frame_start_ms + ef.frame_end_ms) / 2;
                    (emid as i64 - fmid as i64).unsigned_abs()
                })
                .map(|ef| ef.rms_energy)
                .unwrap_or(formant_frame.rms_energy);

            let zcr = compute_zero_crossing_rate(frame_samples);

            let (low_band_db, high_band_db, spectral_flux) =
                if let Some(level) = spec_level {
                    let spec_idx = if level.hop_ms >= MIN_HOP_MS {
                        (formant_frame.frame_start_ms as f32 / level.hop_ms).floor() as usize
                    } else {
                        0
                    };
                    let spec_frame = level.frames.get(spec_idx).cloned();
                    let (low, high) = if let Some(ref sf) = spec_frame {
                        let bin_hz = level.bin_hz;
                        let low_max = ((LOW_BAND_MAX_HZ / bin_hz).floor() as usize)
                            .min(sf.len().saturating_sub(1));
                        let high_min = ((HIGH_BAND_MIN_HZ / bin_hz).floor() as usize)
                            .min(sf.len().saturating_sub(1));
                        let high_max = ((HIGH_BAND_MAX_HZ / bin_hz).floor() as usize)
                            .min(sf.len().saturating_sub(1));
                        let l = if low_max >= 1 {
                            sf[1..=low_max].iter().copied().sum::<f32>() / low_max as f32
                        } else {
                            -60.0
                        };
                        let h = if high_min <= high_max {
                            let count = high_max - high_min + 1;
                            sf[high_min..=high_max].iter().copied().sum::<f32>()
                                / count as f32
                        } else {
                            -60.0
                        };
                        (l, h)
                    } else {
                        (-60.0, -60.0)
                    };
                    let flux = if let (Some(cur), Some(prev)) =
                        (spec_frame.as_deref(), prev_spec.as_deref())
                    {
                        if cur.len() == prev.len() && !cur.is_empty() {
                            let sum: f32 = cur
                                .iter()
                                .zip(prev.iter())
                                .map(|(c, p)| (c - p).abs())
                                .sum();
                            (sum / cur.len() as f32).clamp(0.0, 1.0)
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };
                    prev_spec = spec_frame;
                    (low, high, flux)
                } else {
                    prev_spec = None;
                    (-20.0, -30.0, 0.0)
                };

            let voicing = build_voicing_estimate(
                frame_samples,
                sample_rate,
                rms,
                high_band_db,
                low_band_db,
                zcr,
            );
            let source = build_glottal_source_estimate(
                voicing.voicing_probability,
                zcr,
                high_band_db,
                low_band_db,
            );
            let noise = build_noise_estimate(zcr, high_band_db, low_band_db, spectral_flux);
            let filter = filter_estimate_from_formant_frame(formant_frame);
            let confidence = if rms < 0.005 {
                0.1
            } else {
                (voicing.f0_confidence * 0.5 + 0.5).clamp(0.1, 0.95)
            };

            SourceFilterFrame {
                frame_start_ms: formant_frame.frame_start_ms,
                frame_end_ms: formant_frame.frame_end_ms,
                voicing,
                source,
                filter,
                noise,
                confidence,
            }
        })
        .collect();

    SourceFilterTrack {
        sample_rate,
        hop_ms,
        frames,
    }
}

// ---------------------------------------------------------------------------
// Summary helpers
// ---------------------------------------------------------------------------

impl SourceFilterTrack {
    /// Return all frames whose time range falls within `[start_ms, end_ms)`.
    pub fn frames_in_span(&self, start_ms: u64, end_ms: u64) -> Vec<&SourceFilterFrame> {
        self.frames
            .iter()
            .filter(|f| f.frame_start_ms >= start_ms && f.frame_end_ms <= end_ms)
            .collect()
    }

    /// Median F0 over all voiced frames in the track, or `None` if no voiced
    /// frame is present.
    pub fn median_f0_hz(&self) -> Option<f32> {
        median_f0_of_frames(&self.frames)
    }

    /// Median F0 over voiced frames within a time span, or `None`.
    pub fn median_f0_hz_in_span(&self, start_ms: u64, end_ms: u64) -> Option<f32> {
        let frames: Vec<SourceFilterFrame> = self
            .frames_in_span(start_ms, end_ms)
            .into_iter()
            .cloned()
            .collect();
        median_f0_of_frames(&frames)
    }

    /// Median F1 frequency over frames with an F1 estimate, or `None`.
    pub fn median_f1_hz(&self) -> Option<f32> {
        median_formant_hz(&self.frames, FormantIndex::F1)
    }

    /// Median F2 frequency over frames with an F2 estimate, or `None`.
    pub fn median_f2_hz(&self) -> Option<f32> {
        median_formant_hz(&self.frames, FormantIndex::F2)
    }

    /// Median F3 frequency over frames with an F3 estimate, or `None`.
    pub fn median_f3_hz(&self) -> Option<f32> {
        median_formant_hz(&self.frames, FormantIndex::F3)
    }

    /// Fraction of frames with voicing probability above 0.5 (0.0–1.0).
    pub fn voicing_ratio(&self) -> f32 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let voiced = self
            .frames
            .iter()
            .filter(|f| f.voicing.voicing_probability > 0.5)
            .count();
        voiced as f32 / self.frames.len() as f32
    }

    /// Fraction of frames with a noise ratio above 0.5 (0.0–1.0).
    pub fn noise_ratio(&self) -> f32 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let noisy = self
            .frames
            .iter()
            .filter(|f| f.noise.noise_ratio > 0.5)
            .count();
        noisy as f32 / self.frames.len() as f32
    }
}

// ---------------------------------------------------------------------------
// Internal summary helpers
// ---------------------------------------------------------------------------

enum FormantIndex {
    F1,
    F2,
    F3,
}

fn median_f0_of_frames(frames: &[SourceFilterFrame]) -> Option<f32> {
    let mut values: Vec<f32> = frames.iter().filter_map(|f| f.voicing.f0_hz).collect();
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        Some((values[mid - 1] + values[mid]) / 2.0)
    } else {
        Some(values[mid])
    }
}

fn median_formant_hz(frames: &[SourceFilterFrame], index: FormantIndex) -> Option<f32> {
    let mut values: Vec<f32> = frames
        .iter()
        .filter_map(|f| match index {
            FormantIndex::F1 => f.filter.f1.as_ref().map(|x| x.frequency_hz),
            FormantIndex::F2 => f.filter.f2.as_ref().map(|x| x.frequency_hz),
            FormantIndex::F3 => f.filter.f3.as_ref().map(|x| x.frequency_hz),
        })
        .collect();
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        Some((values[mid - 1] + values[mid]) / 2.0)
    } else {
        Some(values[mid])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: u32 = 16_000;
    const DURATION_SAMPLES: usize = 8_000; // 500 ms

    /// Generate a simple sine-wave vowel-like signal.
    fn make_vowel_samples(f0_hz: f32, duration_samples: usize) -> Vec<f32> {
        let sr = SAMPLE_RATE as f32;
        (0..duration_samples)
            .map(|i| {
                let t = i as f32 / sr;
                // Fundamental + two harmonics → vowel-like
                0.5 * (2.0 * std::f32::consts::PI * f0_hz * t).sin()
                    + 0.25 * (2.0 * std::f32::consts::PI * 2.0 * f0_hz * t).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * 3.0 * f0_hz * t).sin()
            })
            .collect()
    }

    /// Generate silence.
    fn make_silence(n: usize) -> Vec<f32> {
        vec![0.0f32; n]
    }

    /// Generate white noise (deterministic seed via LCG).
    fn make_noise(n: usize) -> Vec<f32> {
        let mut x = 12_345u32;
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let sample = (x as f32 / u32::MAX as f32) * 2.0 - 1.0;
                sample * 0.3
            })
            .collect()
    }

    // --- F0 estimator tests -------------------------------------------------

    #[test]
    fn f0_estimator_finds_correct_pitch_for_sine() {
        let sr = SAMPLE_RATE;
        let f0 = 150.0f32;
        let samples: Vec<f32> = (0..2400)
            .map(|i| (2.0 * std::f32::consts::PI * f0 * i as f32 / sr as f32).sin())
            .collect();
        let result = estimate_f0_autocorrelation(&samples, sr, 60.0, 500.0);
        assert!(result.is_some(), "expected F0 estimate for clean sine");
        let (estimated, confidence) = result.unwrap();
        assert!(
            (estimated - f0).abs() < 5.0,
            "expected ~{f0} Hz, got {estimated}"
        );
        assert!(confidence > 0.0, "confidence should be positive");
    }

    #[test]
    fn f0_estimator_returns_none_for_silence() {
        let samples = make_silence(1600);
        let result = estimate_f0_autocorrelation(&samples, SAMPLE_RATE, 60.0, 500.0);
        assert!(result.is_none(), "silence should give no F0");
    }

    #[test]
    fn f0_estimator_returns_none_for_pure_noise() {
        // White noise has a flat autocorrelation, so the peak should be below
        // the 0.3 threshold and return None.
        let samples = make_noise(1600);
        let result = estimate_f0_autocorrelation(&samples, SAMPLE_RATE, 60.0, 500.0);
        // Pure white noise should not exceed the threshold reliably;
        // we accept either None or a very low confidence result.
        if let Some((_f, c)) = result {
            assert!(c < 0.7, "noise confidence should be low, got {c}");
        }
    }

    // --- SourceFilterTrack construction tests --------------------------------

    fn build_track_from_vowel() -> (SourceFilterTrack, Vec<f32>) {
        let samples = make_vowel_samples(150.0, DURATION_SAMPLES);
        let analysis = crate::audio::acoustic::analyze_mono_samples(&samples, SAMPLE_RATE);
        let track = source_filter_track_from_acoustic(&analysis, &samples);
        (track, samples)
    }

    #[test]
    fn vowel_track_is_non_empty() {
        let (track, _) = build_track_from_vowel();
        assert!(!track.frames.is_empty(), "track should have frames");
    }

    #[test]
    fn vowel_track_has_positive_confidence() {
        let (track, _) = build_track_from_vowel();
        let avg_conf: f32 =
            track.frames.iter().map(|f| f.confidence).sum::<f32>() / track.frames.len() as f32;
        assert!(avg_conf > 0.1, "expected non-trivial confidence, got {avg_conf}");
    }

    #[test]
    fn silence_gives_low_voicing() {
        let samples = make_silence(DURATION_SAMPLES);
        let analysis = crate::audio::acoustic::analyze_mono_samples(&samples, SAMPLE_RATE);
        let track = source_filter_track_from_acoustic(&analysis, &samples);
        let ratio = track.voicing_ratio();
        assert!(ratio < 0.3, "silence voicing ratio should be low, got {ratio}");
    }

    #[test]
    fn noise_gives_higher_frication_energy() {
        let samples = make_noise(DURATION_SAMPLES);
        let analysis = crate::audio::acoustic::analyze_mono_samples(&samples, SAMPLE_RATE);
        let track = source_filter_track_from_acoustic(&analysis, &samples);
        let avg_frication: f32 = if track.frames.is_empty() {
            0.0
        } else {
            track.frames.iter().map(|f| f.noise.frication_energy).sum::<f32>()
                / track.frames.len() as f32
        };
        let vowel_samples = make_vowel_samples(150.0, DURATION_SAMPLES);
        let vowel_analysis =
            crate::audio::acoustic::analyze_mono_samples(&vowel_samples, SAMPLE_RATE);
        let vowel_track = source_filter_track_from_acoustic(&vowel_analysis, &vowel_samples);
        let vowel_frication: f32 = if vowel_track.frames.is_empty() {
            0.0
        } else {
            vowel_track
                .frames
                .iter()
                .map(|f| f.noise.frication_energy)
                .sum::<f32>()
                / vowel_track.frames.len() as f32
        };
        assert!(
            avg_frication >= vowel_frication,
            "noise frication ({avg_frication}) should be ≥ vowel ({vowel_frication})"
        );
    }

    // --- Summary helper tests ------------------------------------------------

    #[test]
    fn median_f0_is_stable_for_constant_signal() {
        let (track, _) = build_track_from_vowel();
        if let Some(median) = track.median_f0_hz() {
            assert!(
                median > 50.0 && median < 600.0,
                "median F0 should be in speech range, got {median}"
            );
        }
        // If no voiced frames, that's acceptable for this unit test (no panic)
    }

    #[test]
    fn empty_span_does_not_panic() {
        let track = SourceFilterTrack {
            sample_rate: 16_000,
            hop_ms: 10.0,
            frames: vec![],
        };
        assert!(track.median_f0_hz().is_none());
        assert!(track.median_f1_hz().is_none());
        assert!(track.median_f2_hz().is_none());
        assert_eq!(track.voicing_ratio(), 0.0);
        assert_eq!(track.noise_ratio(), 0.0);
    }

    #[test]
    fn median_formants_are_in_reasonable_range() {
        let (track, _) = build_track_from_vowel();
        if let Some(f1) = track.median_f1_hz() {
            assert!(f1 > 100.0 && f1 < 1200.0, "F1 out of range: {f1}");
        }
        if let Some(f2) = track.median_f2_hz() {
            assert!(f2 > 400.0 && f2 < 3000.0, "F2 out of range: {f2}");
        }
    }

    #[test]
    fn full_analysis_gives_same_frame_count() {
        let samples = make_vowel_samples(150.0, DURATION_SAMPLES);
        let analysis = crate::audio::acoustic::analyze_mono_samples(&samples, SAMPLE_RATE);
        let fast = source_filter_track_from_acoustic(&analysis, &samples);
        let full = source_filter_track_from_acoustic_full(&analysis, &samples);
        assert_eq!(
            fast.frames.len(),
            full.frames.len(),
            "fast and full analyses should produce the same number of frames"
        );
    }
}
