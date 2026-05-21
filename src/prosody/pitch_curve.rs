//! Continuous sung pitch over time.
//!
//! [`PitchCurve`] is Listenbury's core representation for sung intonation.
//! It models pitch as time-varying frequency targets in Hz so renderers can
//! express continuous vocal motion (glides, scoops, transitions) instead of
//! forcing one fixed MIDI note per syllable.

use std::time::Duration;

use crate::prosody::note_target::NoteTarget;

/// A pitch control point in a curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitchPoint {
    /// Timestamp relative to phrase/utterance start.
    pub t: Duration,
    /// Target frequency in Hz.
    pub hz: f32,
}

impl PitchPoint {
    /// Construct a pitch point.
    #[inline]
    pub fn new(t: Duration, hz: f32) -> Self {
        Self { t, hz }
    }
}

/// Interpolation strategy between neighboring [`PitchPoint`] values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    /// Hold each point value until the next point (mechanical / step behavior).
    Step,
    /// Linear interpolation between adjacent points.
    Linear,
    /// Smoothed interpolation using a `smoothstep` easing curve.
    Smooth,
}

impl Default for Interpolation {
    fn default() -> Self {
        Self::Smooth
    }
}

/// Continuous pitch trajectory represented by time-ordered points.
#[derive(Debug, Clone, PartialEq)]
pub struct PitchCurve {
    /// Ordered pitch control points.
    pub points: Vec<PitchPoint>,
    /// Interpolation mode used between control points.
    pub interpolation: Interpolation,
}

impl PitchCurve {
    /// Create a curve from explicit points.
    ///
    /// Returns `None` when `points` is empty.
    pub fn new(mut points: Vec<PitchPoint>, interpolation: Interpolation) -> Option<Self> {
        if points.is_empty() {
            return None;
        }
        points.sort_by_key(|point| point.t);
        Some(Self {
            points,
            interpolation,
        })
    }

    /// Build a constant curve for one [`NoteTarget`].
    ///
    /// The resulting curve spans `[onset, onset + duration]` and uses the
    /// note's frequency at both endpoints.
    pub fn from_note_target(note: &NoteTarget, interpolation: Interpolation) -> Self {
        Self::from_note_targets([note], interpolation)
            .expect("from_note_target always has one note")
    }

    /// Build a curve from one or more note targets.
    ///
    /// Each note contributes a point at its onset; a final endpoint is added at
    /// the end of the last note to preserve the final sustain.
    ///
    /// Returns `None` when no notes are provided or when the final note end time
    /// overflows `u64` milliseconds.
    pub fn from_note_targets<'a, I>(notes: I, interpolation: Interpolation) -> Option<Self>
    where
        I: IntoIterator<Item = &'a NoteTarget>,
    {
        let mut notes: Vec<&NoteTarget> = notes.into_iter().collect();
        if notes.is_empty() {
            return None;
        }

        notes.sort_by_key(|note| note.onset.millis);

        let mut points = Vec::with_capacity(notes.len() + 1);
        for note in &notes {
            let t = Duration::from_millis(note.onset.millis);
            let hz = note.pitch.frequency_hz() as f32;
            push_or_replace(&mut points, PitchPoint::new(t, hz));
        }

        let last = notes.last().expect("checked non-empty");
        let end_millis = last.onset.millis.checked_add(last.duration.millis)?;
        let end_t = Duration::from_millis(end_millis);
        let end_hz = last.pitch.frequency_hz() as f32;
        push_or_replace(&mut points, PitchPoint::new(end_t, end_hz));

        Self::new(points, interpolation)
    }

    /// Sample pitch at `t`.
    ///
    /// Sampling outside the curve bounds clamps to the first/last point.
    pub fn sample_hz(&self, t: Duration) -> f32 {
        let first = self
            .points
            .first()
            .expect("PitchCurve points are non-empty");
        let last = self.points.last().expect("PitchCurve points are non-empty");

        if t <= first.t {
            return first.hz;
        }
        if t >= last.t {
            return last.hz;
        }

        if let Ok(index) = self.points.binary_search_by_key(&t, |point| point.t) {
            return self.points[index].hz;
        }

        let upper = self.points.partition_point(|point| point.t < t);
        let lower = upper.saturating_sub(1);
        let a = self.points[lower];
        let b = self.points[upper];

        if a.t == b.t {
            return b.hz;
        }

        let segment = (b.t - a.t).as_secs_f32();
        let elapsed = (t - a.t).as_secs_f32();
        let alpha = (elapsed / segment).clamp(0.0, 1.0);

        match self.interpolation {
            Interpolation::Step => a.hz,
            Interpolation::Linear => lerp(a.hz, b.hz, alpha),
            Interpolation::Smooth => {
                let smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha);
                lerp(a.hz, b.hz, smooth_alpha)
            }
        }
    }

    /// Inclusive time span covered by this curve.
    pub fn span(&self) -> (Duration, Duration) {
        let first = self
            .points
            .first()
            .expect("PitchCurve points are non-empty");
        let last = self.points.last().expect("PitchCurve points are non-empty");
        (first.t, last.t)
    }
}

fn push_or_replace(points: &mut Vec<PitchPoint>, point: PitchPoint) {
    if let Some(prev) = points.last_mut() {
        if prev.t == point.t {
            *prev = point;
            return;
        }
    }
    points.push(point);
}

#[inline]
fn lerp(a: f32, b: f32, alpha: f32) -> f32 {
    a + (b - a) * alpha
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prosody::note_target::{
        MidiNote, NoteArticulation, NoteDuration, PitchTarget, TimePoint, Velocity,
    };

    fn make_note(midi: u8, onset_ms: u64, duration_ms: u64) -> NoteTarget {
        NoteTarget {
            pitch: PitchTarget::new(MidiNote::new(midi).unwrap()),
            onset: TimePoint::from_millis(onset_ms),
            duration: NoteDuration::from_millis(duration_ms),
            velocity: Velocity::mezzo_forte(),
            articulation: NoteArticulation::Neutral,
        }
    }

    fn approx_eq(actual: f32, expected: f32, tol: f32) {
        assert!(
            (actual - expected).abs() <= tol,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn single_note_curve_is_constant() {
        let note = make_note(69, 0, 1_000);
        let curve = PitchCurve::from_note_target(&note, Interpolation::Smooth);
        let expected = note.pitch.frequency_hz() as f32;

        approx_eq(curve.sample_hz(Duration::from_millis(0)), expected, 1e-4);
        approx_eq(curve.sample_hz(Duration::from_millis(500)), expected, 1e-4);
        approx_eq(
            curve.sample_hz(Duration::from_millis(1_000)),
            expected,
            1e-4,
        );
    }

    #[test]
    fn linear_interpolation_transitions_between_two_pitches() {
        let curve = PitchCurve::new(
            vec![
                PitchPoint::new(Duration::from_millis(0), 440.0),
                PitchPoint::new(Duration::from_millis(1_000), 660.0),
            ],
            Interpolation::Linear,
        )
        .unwrap();

        approx_eq(curve.sample_hz(Duration::from_millis(500)), 550.0, 1e-4);
    }

    #[test]
    fn step_interpolation_holds_previous_pitch_until_next_point() {
        let curve = PitchCurve::new(
            vec![
                PitchPoint::new(Duration::from_millis(0), 440.0),
                PitchPoint::new(Duration::from_millis(1_000), 660.0),
            ],
            Interpolation::Step,
        )
        .unwrap();

        approx_eq(curve.sample_hz(Duration::from_millis(500)), 440.0, 1e-4);
    }

    #[test]
    fn sampling_outside_bounds_clamps_to_curve_edges() {
        let curve = PitchCurve::new(
            vec![
                PitchPoint::new(Duration::from_millis(100), 330.0),
                PitchPoint::new(Duration::from_millis(300), 550.0),
            ],
            Interpolation::Linear,
        )
        .unwrap();

        approx_eq(curve.sample_hz(Duration::from_millis(0)), 330.0, 1e-4);
        approx_eq(curve.sample_hz(Duration::from_millis(1_000)), 550.0, 1e-4);
    }

    #[test]
    fn builds_curve_from_note_target_sequence() {
        let n1 = make_note(69, 0, 500);
        let n2 = make_note(72, 500, 250);

        let curve = PitchCurve::from_note_targets([&n1, &n2], Interpolation::Smooth).unwrap();

        assert_eq!(curve.points.len(), 3);
        assert_eq!(curve.points[0].t, Duration::from_millis(0));
        assert_eq!(curve.points[1].t, Duration::from_millis(500));
        assert_eq!(curve.points[2].t, Duration::from_millis(750));
        approx_eq(curve.points[0].hz, n1.pitch.frequency_hz() as f32, 1e-4);
        approx_eq(curve.points[2].hz, n2.pitch.frequency_hz() as f32, 1e-4);
    }
}
