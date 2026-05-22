//! Vibrato as an intentional prosodic pitch modulation.
//!
//! [`Vibrato`] models periodic pitch deviation in **cents** over time. The
//! returned frequency multiplier is `2^(offset_cents / 1200)`, so a depth of
//! `±100` cents corresponds to modulation across one semitone around the base
//! pitch.

use std::f32::consts::TAU;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::prosody::pitch_curve::PitchCurve;

/// Parameterized vibrato envelope and oscillator.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vibrato {
    /// Oscillation rate in Hz.
    pub rate_hz: f32,
    /// Peak modulation depth in cents after the ramp is complete.
    pub depth_cents: f32,
    /// Delay before vibrato begins.
    pub onset: Duration,
    /// Fade-in duration from zero depth to full depth.
    pub ramp: Duration,
    /// Phase offset in radians.
    pub phase: f32,
}

impl Vibrato {
    /// Construct a vibrato model.
    #[inline]
    pub fn new(
        rate_hz: f32,
        depth_cents: f32,
        onset: Duration,
        ramp: Duration,
        phase: f32,
    ) -> Self {
        Self {
            rate_hz,
            depth_cents,
            onset,
            ramp,
            phase,
        }
    }

    /// Sample cents offset at time `t`.
    #[inline]
    pub fn sample_cents_offset(&self, t: Duration) -> f32 {
        if self.depth_cents == 0.0 || t < self.onset {
            return 0.0;
        }

        let elapsed = (t - self.onset).as_secs_f32();
        let ramp_gain = if self.ramp.is_zero() {
            1.0
        } else {
            (elapsed / self.ramp.as_secs_f32()).clamp(0.0, 1.0)
        };
        let phase = self.phase + TAU * self.rate_hz * elapsed;

        self.depth_cents * ramp_gain * phase.sin()
    }

    /// Sample multiplicative pitch factor at time `t`.
    ///
    /// This converts cents to a frequency ratio via `2^(cents/1200)`.
    #[inline]
    pub fn sample_multiplier(&self, t: Duration) -> f32 {
        2.0_f32.powf(self.sample_cents_offset(t) / 1200.0)
    }

    /// Apply vibrato to a base frequency value.
    #[inline]
    pub fn apply_to_hz(&self, base_hz: f32, t: Duration) -> f32 {
        base_hz * self.sample_multiplier(t)
    }

    /// Sample a vibrato-modulated pitch over a base [`PitchCurve`].
    #[inline]
    pub fn sample_over_curve_hz(&self, curve: &PitchCurve, t: Duration) -> f32 {
        self.apply_to_hz(curve.sample_hz(t), t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prosody::pitch_curve::{Interpolation, PitchPoint};

    fn approx_eq(actual: f32, expected: f32, tol: f32) {
        assert!(
            (actual - expected).abs() <= tol,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn zero_depth_produces_no_offset() {
        let vibrato = Vibrato::new(
            5.0,
            0.0,
            Duration::from_millis(0),
            Duration::from_millis(250),
            0.0,
        );

        approx_eq(
            vibrato.sample_cents_offset(Duration::from_millis(500)),
            0.0,
            1e-6,
        );
        approx_eq(
            vibrato.sample_multiplier(Duration::from_millis(500)),
            1.0,
            1e-6,
        );
    }

    #[test]
    fn delayed_onset_has_no_offset_before_onset() {
        let vibrato = Vibrato::new(
            6.0,
            40.0,
            Duration::from_millis(300),
            Duration::from_millis(0),
            std::f32::consts::FRAC_PI_2,
        );

        approx_eq(
            vibrato.sample_cents_offset(Duration::from_millis(299)),
            0.0,
            1e-6,
        );
    }

    #[test]
    fn ramp_reaches_full_depth_after_ramp_duration_with_static_phase() {
        let vibrato = Vibrato::new(
            0.0,
            60.0,
            Duration::from_millis(100),
            Duration::from_millis(200),
            std::f32::consts::FRAC_PI_2,
        );

        approx_eq(
            vibrato.sample_cents_offset(Duration::from_millis(200)),
            30.0,
            1e-4,
        );
        approx_eq(
            vibrato.sample_cents_offset(Duration::from_millis(300)),
            60.0,
            1e-4,
        );
    }

    #[test]
    fn periodic_offsets_match_known_phase_points() {
        let vibrato = Vibrato::new(2.0, 30.0, Duration::ZERO, Duration::ZERO, 0.0);

        approx_eq(
            vibrato.sample_cents_offset(Duration::from_millis(0)),
            0.0,
            1e-5,
        );
        approx_eq(
            vibrato.sample_cents_offset(Duration::from_millis(125)),
            30.0,
            1e-3,
        );
        approx_eq(
            vibrato.sample_cents_offset(Duration::from_millis(250)),
            0.0,
            1e-3,
        );
        approx_eq(
            vibrato.sample_cents_offset(Duration::from_millis(375)),
            -30.0,
            1e-3,
        );
    }

    #[test]
    fn can_compose_vibrato_over_pitch_curve() {
        let curve = PitchCurve::new(
            vec![
                PitchPoint::new(Duration::ZERO, 440.0),
                PitchPoint::new(Duration::from_millis(1_000), 440.0),
            ],
            Interpolation::Linear,
        )
        .unwrap();

        let vibrato = Vibrato::new(2.0, 100.0, Duration::ZERO, Duration::ZERO, 0.0);
        let expected = 440.0 * 2.0_f32.powf(100.0 / 1200.0);
        approx_eq(
            vibrato.sample_over_curve_hz(&curve, Duration::from_millis(125)),
            expected,
            1e-4,
        );
    }
}
