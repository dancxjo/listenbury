//! Layer B: Klatt-style source/filter phone renderer.
//!
//! A small deterministic synthesiser that renders controlled diagnostic audio
//! from explicit per-phone targets.  This is **not** a neural vocoder; it is
//! a transparent, testable phone-lab for analysing, comparing, and inverting
//! acoustic hypotheses.
//!
//! # Architecture
//!
//! ```text
//!  F0 / noise source ──→ [ glottal pulse / aspiration mixer ]
//!                                        │
//!                        ┌──────────────▼──────────────────┐
//!                        │ F1 resonator                     │
//!                        │ F2 resonator    formant cascade  │
//!                        │ F3 resonator                     │
//!                        │ F4 resonator (optional)          │
//!                        └──────────────┬──────────────────┘
//!                                       │
//!                            amplitude × gain → PCM output
//! ```
//!
//! # Entry points
//!
//! * [`render_phone`] — render a single [`PhoneRenderTarget`] to mono PCM.
//! * [`render_phone_string`] — render a sequence with short inter-phone
//!   crossfades to avoid discontinuity clicks.
//!
//! # Realtime safety
//!
//! These functions are pure / deterministic and operate on owned data.  They
//! are intentionally **not** realtime-safe (they allocate freely) and should
//! be called on buffered / offline paths only.

use super::targets::{PhoneRenderTarget, VocalTractFilterTarget};

mod coarticulation;
mod params;
mod trajectory;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Renderer configuration shared across a synthesis session.
#[derive(Debug, Clone, PartialEq)]
pub struct KlattRenderConfig {
    /// Output sample rate in Hz.
    pub sample_rate: u32,
    /// Length of the cosine crossfade applied between adjacent phones in
    /// milliseconds.  Set to zero to disable crossfading.
    pub crossfade_ms: f32,
    /// Overall gain applied to the output (linear, 0.0–1.0).
    pub gain: f32,
}

impl Default for KlattRenderConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            crossfade_ms: 10.0,
            gain: 0.7,
        }
    }
}

// ---------------------------------------------------------------------------
// Second-order formant resonator
// ---------------------------------------------------------------------------

/// A second-order IIR resonator (Klatt 1980 style).
///
/// Transfer function (z-domain):
///
/// ```text
///             (1 − r)
/// H(z) = ───────────────────────────────
///          1 − 2r cos(ω) z⁻¹ + r² z⁻²
/// ```
///
/// where `r = exp(−π B / Fs)` and `ω = 2π F / Fs`.
#[derive(Debug, Clone)]
struct FormantResonator {
    /// Filter coefficient b0 = 1 − r.
    b0: f32,
    /// Filter coefficient a1 = −2r cos(ω).
    a1: f32,
    /// Filter coefficient a2 = r².
    a2: f32,
    /// Previous output sample y[n-1].
    y1: f32,
    /// Output sample y[n-2].
    y2: f32,
}

impl FormantResonator {
    /// Create a resonator tuned to `freq_hz` with bandwidth `bw_hz` at
    /// the given `sample_rate`.
    fn new(freq_hz: f32, bw_hz: f32, sample_rate: u32) -> Self {
        let sr = sample_rate as f32;
        let r = (-std::f32::consts::PI * bw_hz / sr).exp();
        let omega = 2.0 * std::f32::consts::PI * freq_hz / sr;
        Self {
            b0: 1.0 - r,
            a1: -2.0 * r * omega.cos(),
            a2: r * r,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Process one sample through the resonator and advance internal state.
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x - self.a1 * self.y1 - self.a2 * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    /// Reset internal state (prevents discontinuities between phones when
    /// the resonator is reused).
    fn reset(&mut self) {
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Source generators
// ---------------------------------------------------------------------------

/// Generate a sequence of glottal-pulse-like samples.
///
/// Uses a cosine-tapered decaying impulse train as a simple approximation to
/// a Liljencrants-Fant glottal pulse (adequate for diagnostic synthesis).
fn generate_glottal_source(n_samples: usize, f0_hz: f32, sample_rate: u32) -> Vec<f32> {
    let sr = sample_rate as f32;
    let period = sr / f0_hz;
    let mut out = vec![0.0f32; n_samples];
    let mut phase = 0.0f32;
    for s in out.iter_mut() {
        // Sawtooth that resets on each glottal cycle (open phase)
        let v = 1.0 - (2.0 * phase / period).min(1.0); // ramp 1→0 over open phase
        *s = v;
        phase += 1.0;
        if phase >= period {
            phase -= period;
        }
    }
    out
}

/// Generate white noise using a simple LCG.
fn generate_noise(n_samples: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_add(1);
    (0..n_samples)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Core rendering
// ---------------------------------------------------------------------------

/// Render a single [`PhoneRenderTarget`] to mono PCM samples.
///
/// The output length is `(duration_ms / 1000) * sample_rate` samples.
pub fn render_phone(target: &PhoneRenderTarget, config: &KlattRenderConfig) -> Vec<f32> {
    let sr = config.sample_rate;
    let n_samples = ((target.duration_ms as f32 / 1000.0) * sr as f32).round() as usize;
    if n_samples == 0 {
        return vec![];
    }

    let breathiness = target.source.as_ref().map(|s| s.breathiness).unwrap_or(0.0);
    let spectral_tilt = target
        .source
        .as_ref()
        .map(|s| s.spectral_tilt_db_per_octave)
        .unwrap_or(-6.0);

    // --- Source signal ---
    let (voiced_source, noise_source) = if let Some(f0) = target.f0_hz {
        let voiced = generate_glottal_source(n_samples, f0.max(20.0), sr);
        let noise = generate_noise(n_samples, 0xDEAD_BEEF);
        (voiced, noise)
    } else {
        // Unvoiced: pure noise source
        let noise = generate_noise(n_samples, 0xBEEF_CAFE);
        (vec![0.0f32; n_samples], noise)
    };

    // Mix voiced + noise according to breathiness
    let mixed: Vec<f32> = voiced_source
        .iter()
        .zip(noise_source.iter())
        .map(|(v, n)| {
            let voiced_gain = (1.0 - breathiness).clamp(0.0, 1.0);
            let noise_gain = breathiness.clamp(0.0, 1.0);
            v * voiced_gain + n * noise_gain
        })
        .collect();

    // Apply spectral tilt as a gentle one-pole low-pass (first-order IIR).
    // Tilt: more negative → more attenuation at high frequencies.
    // The divisor 40.0 maps typical speech tilt values (−6 to −3 dB/oct)
    // to a stable IIR coefficient range of ~[−0.15, −0.075].  A 6 dB/oct
    // roll-off corresponds to one pole at the Nyquist edge; this rough
    // mapping keeps the filter well inside the unit circle without needing
    // explicit frequency-domain design.
    let tilt_coeff = (spectral_tilt / 40.0).clamp(-0.95, 0.0);
    let tilt_b0 = 1.0 + tilt_coeff;
    let tilt_a1 = -tilt_coeff;
    let mut tilt_state = 0.0f32;
    let tilted: Vec<f32> = mixed
        .iter()
        .map(|&x| {
            let y = tilt_b0 * x + tilt_a1 * tilt_state;
            tilt_state = y;
            y
        })
        .collect();

    // --- Formant cascade ---
    let filtered = apply_formant_cascade(&tilted, target.filter.as_ref(), config);

    // --- Amplitude scaling ---
    let peak = filtered.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    let norm = if peak > 1e-6 { 1.0 / peak } else { 1.0 };

    filtered
        .iter()
        .map(|&s| s * norm * target.amplitude * config.gain)
        .collect()
}

/// Apply a cascade of formant resonators to the input signal.
fn apply_formant_cascade(
    input: &[f32],
    filter: Option<&VocalTractFilterTarget>,
    config: &KlattRenderConfig,
) -> Vec<f32> {
    let Some(f) = filter else {
        // No formant filter: pass through with mild low-pass to soften clicks
        return apply_simple_lowpass(input, 4000.0, config.sample_rate);
    };

    let mut resonators = vec![
        FormantResonator::new(f.f1_hz, f.f1_bw_hz.max(10.0), config.sample_rate),
        FormantResonator::new(f.f2_hz, f.f2_bw_hz.max(10.0), config.sample_rate),
        FormantResonator::new(f.f3_hz, f.f3_bw_hz.max(10.0), config.sample_rate),
    ];
    if let (Some(f4_hz), Some(f4_bw)) = (f.f4_hz, f.f4_bw_hz) {
        resonators.push(FormantResonator::new(
            f4_hz,
            f4_bw.max(10.0),
            config.sample_rate,
        ));
    }

    // Formant amplitude gains (convert dB to linear)
    let amps = [
        db_to_linear(f.f1_amp_db),
        db_to_linear(f.f2_amp_db),
        db_to_linear(f.f3_amp_db),
        db_to_linear(f.f4_amp_db.unwrap_or(0.0)),
    ];

    let mut signal = input.to_vec();
    for (res, &amp) in resonators.iter_mut().zip(amps.iter()) {
        res.reset();
        signal = signal.iter().map(|&x| res.process(x) * amp).collect();
    }
    signal
}

/// Simple single-pole low-pass for unfiltered phones (stops, etc.).
fn apply_simple_lowpass(input: &[f32], cutoff_hz: f32, sample_rate: u32) -> Vec<f32> {
    let sr = sample_rate as f32;
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
    let dt = 1.0 / sr;
    let alpha = dt / (rc + dt);
    let mut prev = 0.0f32;
    input
        .iter()
        .map(|&x| {
            let y = alpha * x + (1.0 - alpha) * prev;
            prev = y;
            y
        })
        .collect()
}

fn db_to_linear(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

// ---------------------------------------------------------------------------
// Multi-phone rendering with crossfade
// ---------------------------------------------------------------------------

/// Render a sequence of phones to mono PCM, applying cosine crossfades
/// between adjacent phones to suppress discontinuity clicks.
///
/// The total duration is the sum of all individual `duration_ms` values.
pub fn render_phone_string(targets: &[PhoneRenderTarget], config: &KlattRenderConfig) -> Vec<f32> {
    let coarticulated = coarticulation::apply_neighbor_influence(targets);
    let trajectory = trajectory::trajectory_targets_from_phones(
        &coarticulated,
        trajectory::TrajectoryConfig::default(),
    );
    let mut trajectory_config = config.clone();
    trajectory_config.crossfade_ms = trajectory_config.crossfade_ms.min(4.0);
    render_phone_sequence(&trajectory, &trajectory_config)
}

fn render_phone_sequence(targets: &[PhoneRenderTarget], config: &KlattRenderConfig) -> Vec<f32> {
    if targets.is_empty() {
        return vec![];
    }

    let sr = config.sample_rate;
    let crossfade_samples = ((config.crossfade_ms / 1000.0) * sr as f32).round() as usize;

    // Render all phones individually
    let segments: Vec<Vec<f32>> = targets.iter().map(|t| render_phone(t, config)).collect();

    // Total output length
    let total_samples: usize = segments.iter().map(|s| s.len()).sum();
    if total_samples == 0 {
        return vec![];
    }

    let mut output = vec![0.0f32; total_samples];
    let mut write_pos = 0usize;

    for (idx, seg) in segments.iter().enumerate() {
        let n = seg.len();
        // Apply fade-in ramp at the start of each segment (except the first)
        // and fade-out ramp at the end (except the last).
        let fade_in = if idx > 0 { crossfade_samples.min(n) } else { 0 };
        let fade_out = if idx < segments.len() - 1 {
            crossfade_samples.min(n)
        } else {
            0
        };

        for (i, &sample) in seg.iter().enumerate() {
            let gain = if i < fade_in {
                // Cosine fade-in: 0 → 1 over fade_in samples
                let t = i as f32 / fade_in as f32;
                0.5 * (1.0 - (std::f32::consts::PI * (1.0 - t)).cos())
            } else if i >= n - fade_out {
                // Cosine fade-out: 1 → 0 over fade_out samples
                let t = (i - (n - fade_out)) as f32 / fade_out as f32;
                0.5 * (1.0 + (std::f32::consts::PI * t).cos())
            } else {
                1.0
            };
            let out_idx = write_pos + i;
            if out_idx < output.len() {
                output[out_idx] += sample * gain;
            }
        }
        write_pos += n;
    }

    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linguistic::phonology::Phone;
    use crate::linguistic::phonology::PhoneString;
    use crate::voice::tract::targets::{
        default_english_phone_targets, phone_render_targets_from_string, GlottalSourceTarget,
        VocalTractFilterTarget,
    };

    fn vowel_target(f0: f32) -> PhoneRenderTarget {
        let table = default_english_phone_targets();
        let phone = Phone::new_ipa("ɑ");
        let ps = PhoneString {
            phones: vec![phone],
        };
        let mut targets = phone_render_targets_from_string(&ps, Some(f0), 0.7, &table);
        targets[0].duration_ms = 100;
        targets.remove(0)
    }

    fn fricative_target() -> PhoneRenderTarget {
        let table = default_english_phone_targets();
        let phone = Phone::new_ipa("s");
        let ps = PhoneString {
            phones: vec![phone],
        };
        let mut targets = phone_render_targets_from_string(&ps, None, 0.7, &table);
        targets[0].duration_ms = 80;
        targets.remove(0)
    }

    #[test]
    fn render_vowel_produces_non_empty_pcm_with_correct_duration() {
        let config = KlattRenderConfig::default();
        let target = vowel_target(150.0);
        let pcm = render_phone(&target, &config);
        let expected_samples = ((100.0 / 1000.0) * config.sample_rate as f32).round() as usize;
        assert_eq!(
            pcm.len(),
            expected_samples,
            "PCM length should match duration"
        );
        assert!(!pcm.is_empty());
    }

    #[test]
    fn render_vowel_has_significant_energy() {
        let config = KlattRenderConfig::default();
        let target = vowel_target(150.0);
        let pcm = render_phone(&target, &config);
        let rms: f32 = (pcm.iter().map(|s| s * s).sum::<f32>() / pcm.len() as f32).sqrt();
        assert!(
            rms > 0.01,
            "vowel should have significant energy, got rms={rms}"
        );
    }

    #[test]
    fn render_fricative_produces_noise_like_signal() {
        let config = KlattRenderConfig::default();
        let target = fricative_target();
        let pcm = render_phone(&target, &config);
        assert!(!pcm.is_empty(), "fricative PCM should not be empty");
        // Noise-like: high ZCR
        let crossings = pcm
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();
        let zcr = crossings as f32 / (pcm.len() - 1) as f32;
        assert!(zcr > 0.05, "fricative should have high ZCR, got {zcr}");
    }

    #[test]
    fn render_fricative_has_no_required_f0() {
        let target = fricative_target();
        assert!(target.f0_hz.is_none(), "/s/ should be unvoiced (no F0)");
    }

    #[test]
    fn render_phone_string_concatenates_durations_correctly() {
        let config = KlattRenderConfig {
            crossfade_ms: 0.0,
            ..Default::default()
        };
        let table = default_english_phone_targets();
        let ps = PhoneString {
            phones: vec![
                Phone::new_ipa("s"),
                Phone::new_ipa("ɑ"),
                Phone::new_ipa("t"),
            ],
        };
        let targets = phone_render_targets_from_string(&ps, Some(150.0), 0.7, &table);
        let total_dur_ms: u64 = targets.iter().map(|t| t.duration_ms).sum();
        let pcm = render_phone_string(&targets, &config);
        let expected_samples =
            ((total_dur_ms as f32 / 1000.0) * config.sample_rate as f32).round() as usize;
        assert_eq!(pcm.len(), expected_samples);
    }

    #[test]
    fn adjacent_phones_avoid_discontinuity_spikes() {
        let config = KlattRenderConfig {
            crossfade_ms: 10.0,
            ..Default::default()
        };
        let table = default_english_phone_targets();
        let ps = PhoneString {
            phones: vec![Phone::new_ipa("s"), Phone::new_ipa("ɑ")],
        };
        let targets = phone_render_targets_from_string(&ps, Some(150.0), 0.7, &table);
        let pcm = render_phone_string(&targets, &config);
        // Look for large amplitude jumps between consecutive samples
        let max_jump = pcm
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_jump < 0.6,
            "crossfade should prevent large amplitude jumps; max jump = {max_jump}"
        );
    }

    #[test]
    fn render_empty_phone_string_returns_empty() {
        let config = KlattRenderConfig::default();
        let pcm = render_phone_string(&[], &config);
        assert!(pcm.is_empty());
    }

    #[test]
    fn explicit_filter_overrides_default() {
        let config = KlattRenderConfig::default();
        let target = PhoneRenderTarget {
            phone: Phone::new_ipa("i"),
            duration_ms: 50,
            f0_hz: Some(200.0),
            amplitude: 0.8,
            vibrato: None,
            source: Some(GlottalSourceTarget {
                breathiness: 0.0,
                open_quotient: 0.5,
                spectral_tilt_db_per_octave: -6.0,
            }),
            filter: Some(VocalTractFilterTarget {
                f1_hz: 300.0,
                f1_bw_hz: 60.0,
                f1_amp_db: 0.0,
                f2_hz: 2300.0,
                f2_bw_hz: 90.0,
                f2_amp_db: -3.0,
                f3_hz: 3000.0,
                f3_bw_hz: 150.0,
                f3_amp_db: -6.0,
                f4_hz: None,
                f4_bw_hz: None,
                f4_amp_db: None,
            }),
        };
        let pcm = render_phone(&target, &config);
        let expected = ((50.0 / 1000.0) * config.sample_rate as f32).round() as usize;
        assert_eq!(pcm.len(), expected);
        let rms: f32 = (pcm.iter().map(|s| s * s).sum::<f32>() / pcm.len() as f32).sqrt();
        assert!(rms > 0.001, "should have energy with explicit filter");
    }
}
