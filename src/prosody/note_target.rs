//! Vocal intent target for sung material.
//!
//! [`NoteTarget`] is the lowest-level musical target representation for sung
//! content in Listenbury. It describes a fully specified note event that can be
//! attached to a syllable, vowel nucleus, phone span, or vocal gesture.
//!
//! This is **not** a MIDI event writer or an audio synthesis primitive. It
//! captures *sung intent*: what pitch a voice renderer should aim for, how long
//! the note should last, how loud, and with what articulation. Future modules
//! such as `pitch_curve`, `syllable`, and `singing` will consume these targets.

use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};

// ─── MIDI note ───────────────────────────────────────────────────────────────

/// A MIDI note number in the range `0..=127`.
///
/// MIDI note 69 is A4 (440 Hz at standard tuning). The value is validated on
/// construction; values outside `0..=127` are rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MidiNote(u8);

impl MidiNote {
    /// Create a new [`MidiNote`] from a raw byte value.
    ///
    /// Returns `None` if `value` is greater than 127.
    #[inline]
    pub fn new(value: u8) -> Option<Self> {
        if value <= 127 {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Return the underlying MIDI note number.
    #[inline]
    pub fn as_u8(self) -> u8 {
        self.0
    }

    /// Compute the equal-temperament frequency in Hz for this note, ignoring
    /// any additional cents offset.
    ///
    /// Formula: `440.0 × 2^((note − 69) / 12)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use listenbury::prosody::note_target::MidiNote;
    /// let a4 = MidiNote::new(69).unwrap();
    /// assert!((a4.to_hz() - 440.0).abs() < 1e-6);
    /// ```
    #[inline]
    pub fn to_hz(self) -> f64 {
        440.0 * 2.0_f64.powf((self.0 as f64 - 69.0) / 12.0)
    }
}

impl fmt::Display for MidiNote {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MIDI({})", self.0)
    }
}

// ─── Cents offset ────────────────────────────────────────────────────────────

/// A microtonal offset in cents (hundredths of a semitone).
///
/// Typical ranges for pitch-bend or fine tuning lie within `±100` cents
/// (one semitone), but the field does not enforce a bound because extended
/// microtonality or deliberate pitch curves may exceed that range.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CentsOffset(f64);

impl CentsOffset {
    /// Create a new [`CentsOffset`].
    #[inline]
    pub fn new(cents: f64) -> Self {
        Self(cents)
    }

    /// Return the offset value in cents.
    #[inline]
    pub fn as_f64(self) -> f64 {
        self.0
    }

    /// Zero offset – no microtonal correction.
    #[inline]
    pub fn zero() -> Self {
        Self(0.0)
    }
}

impl Default for CentsOffset {
    fn default() -> Self {
        Self::zero()
    }
}

// ─── Pitch target ────────────────────────────────────────────────────────────

/// The pitch specification for a sung note.
///
/// A [`PitchTarget`] combines a MIDI note number with an optional microtonal
/// cents offset, allowing sub-semitone tuning corrections. Both the raw note
/// and the corrected frequency are available via accessors.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PitchTarget {
    /// The MIDI note number (`0..=127`).
    pub note: MidiNote,
    /// Additional tuning offset in cents.
    pub tuning: CentsOffset,
}

impl PitchTarget {
    /// Create a new [`PitchTarget`] with no microtonal offset.
    #[inline]
    pub fn new(note: MidiNote) -> Self {
        Self {
            note,
            tuning: CentsOffset::zero(),
        }
    }

    /// Create a [`PitchTarget`] with an explicit cents offset.
    #[inline]
    pub fn with_tuning(note: MidiNote, tuning: CentsOffset) -> Self {
        Self { note, tuning }
    }

    /// Compute the frequency in Hz, incorporating the cents offset.
    ///
    /// Formula: `440 × 2^((note − 69 + cents/100) / 12)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use listenbury::prosody::note_target::{CentsOffset, MidiNote, PitchTarget};
    /// let a4 = PitchTarget::new(MidiNote::new(69).unwrap());
    /// assert!((a4.frequency_hz() - 440.0).abs() < 1e-6);
    ///
    /// // A4 + 100 cents = A#4 / Bb4 ≈ 466.16 Hz
    /// let bb4 = PitchTarget::with_tuning(MidiNote::new(69).unwrap(), CentsOffset::new(100.0));
    /// let a_sharp_4 = PitchTarget::new(MidiNote::new(70).unwrap());
    /// assert!((bb4.frequency_hz() - a_sharp_4.frequency_hz()).abs() < 1e-6);
    /// ```
    #[inline]
    pub fn frequency_hz(&self) -> f64 {
        let semitones = self.note.as_u8() as f64 - 69.0 + self.tuning.as_f64() / 100.0;
        440.0 * 2.0_f64.powf(semitones / 12.0)
    }
}

// ─── TimePoint ───────────────────────────────────────────────────────────────

/// A point in time expressed as milliseconds from the start of a phrase or
/// utterance.
///
/// This intentionally avoids binding note targets to wall-clock time or sample
/// offsets, keeping the representation decoupled from audio rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TimePoint {
    /// Milliseconds from the beginning of the containing phrase or utterance.
    pub millis: u64,
}

impl TimePoint {
    /// Create a new [`TimePoint`] at the given millisecond offset.
    #[inline]
    pub fn from_millis(millis: u64) -> Self {
        Self { millis }
    }
}

// ─── NoteDuration ────────────────────────────────────────────────────────────

/// The sustain duration of a note, guaranteed to be non-negative.
///
/// Negative durations are made unrepresentable; construction returns `None`
/// for out-of-range inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NoteDuration {
    /// Duration in milliseconds.
    pub millis: u64,
}

impl NoteDuration {
    /// Create a duration from a millisecond count.
    ///
    /// Always succeeds because `u64` cannot be negative.
    #[inline]
    pub fn from_millis(millis: u64) -> Self {
        Self { millis }
    }

    /// Create a duration from a [`std::time::Duration`].
    ///
    /// Fractional sub-millisecond values are truncated.
    #[inline]
    pub fn from_duration(d: Duration) -> Self {
        Self {
            millis: d.as_millis() as u64,
        }
    }
}

// ─── Velocity ────────────────────────────────────────────────────────────────

/// A MIDI-compatible velocity value in the range `1..=127`.
///
/// Velocity 0 is conventionally a note-off in MIDI and is therefore excluded
/// here to avoid ambiguity. Values outside `1..=127` are rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Velocity(u8);

impl Velocity {
    /// Create a new [`Velocity`].
    ///
    /// Returns `None` if `value` is 0 or greater than 127.
    #[inline]
    pub fn new(value: u8) -> Option<Self> {
        if (1..=127).contains(&value) {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Return the underlying velocity byte.
    #[inline]
    pub fn as_u8(self) -> u8 {
        self.0
    }

    /// Mezzo-forte default (`64`).
    #[inline]
    pub fn mezzo_forte() -> Self {
        Self(64)
    }

    /// Normalise to `[0.0, 1.0]`.
    #[inline]
    pub fn as_f32(self) -> f32 {
        self.0 as f32 / 127.0
    }
}

impl Default for Velocity {
    fn default() -> Self {
        Self::mezzo_forte()
    }
}

// ─── Articulation ────────────────────────────────────────────────────────────

/// Qualitative articulation hint for a sung note.
///
/// These values convey vocal intent to a future voice renderer. They do not
/// prescribe a specific audio synthesis strategy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoteArticulation {
    /// Default articulation – no special treatment.
    #[default]
    Neutral,
    /// Smooth connection to the following note.
    Legato,
    /// Short and detached.
    Staccato,
    /// Extra emphasis at the attack.
    Accented,
    /// Held for full written value without a hard release.
    Tenuto,
}

// ─── NoteTarget ──────────────────────────────────────────────────────────────

/// A fully specified sung note target.
///
/// [`NoteTarget`] is the primary type in this module. It encodes all the
/// information a voice renderer needs to realise a single sung note:
///
/// - **pitch** – MIDI note number plus optional microtonal correction
/// - **onset** – when the note begins (phrase-relative ms)
/// - **duration** – how long the note is sustained
/// - **velocity** – loudness or intensity intent
/// - **articulation** – legato / staccato / accented / tenuto / neutral
///
/// This type is *not* a MIDI event and does not carry channel, program, or
/// transport metadata. It is a semantic description of vocal intent.
///
/// # Example
///
/// ```
/// # use listenbury::prosody::note_target::{
/// #     MidiNote, NoteArticulation, NoteDuration, NoteTarget,
/// #     PitchTarget, TimePoint, Velocity,
/// # };
/// let note = NoteTarget {
///     pitch: PitchTarget::new(MidiNote::new(69).unwrap()), // A4
///     onset: TimePoint::from_millis(0),
///     duration: NoteDuration::from_millis(500),
///     velocity: Velocity::mezzo_forte(),
///     articulation: NoteArticulation::Legato,
/// };
/// assert!((note.pitch.frequency_hz() - 440.0).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoteTarget {
    /// Pitch specification (MIDI note + microtonal offset).
    pub pitch: PitchTarget,
    /// Onset time relative to the containing phrase or utterance.
    pub onset: TimePoint,
    /// Sustain duration of the note.
    pub duration: NoteDuration,
    /// Loudness or intensity intent.
    pub velocity: Velocity,
    /// Qualitative articulation hint.
    pub articulation: NoteArticulation,
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a4_midi_69_is_440_hz() {
        let a4 = MidiNote::new(69).unwrap();
        let diff = (a4.to_hz() - 440.0).abs();
        assert!(diff < 1e-6, "expected ~440 Hz, got {}", a4.to_hz());
    }

    #[test]
    fn octave_above_doubles_frequency() {
        let a4 = MidiNote::new(69).unwrap();
        let a5 = MidiNote::new(81).unwrap();
        let ratio = a5.to_hz() / a4.to_hz();
        assert!(
            (ratio - 2.0).abs() < 1e-9,
            "expected ratio 2.0, got {}",
            ratio
        );
    }

    #[test]
    fn octave_below_halves_frequency() {
        let a4 = MidiNote::new(69).unwrap();
        let a3 = MidiNote::new(57).unwrap();
        let ratio = a4.to_hz() / a3.to_hz();
        assert!(
            (ratio - 2.0).abs() < 1e-9,
            "expected ratio 2.0, got {}",
            ratio
        );
    }

    #[test]
    fn cents_offset_100_equals_one_semitone() {
        let a4_with_offset =
            PitchTarget::with_tuning(MidiNote::new(69).unwrap(), CentsOffset::new(100.0));
        let a_sharp_4 = PitchTarget::new(MidiNote::new(70).unwrap());
        let diff = (a4_with_offset.frequency_hz() - a_sharp_4.frequency_hz()).abs();
        assert!(diff < 1e-6, "expected same freq, diff was {}", diff);
    }

    #[test]
    fn cents_offset_negative_50_is_quarter_tone_below() {
        let a4_flat = PitchTarget::with_tuning(MidiNote::new(69).unwrap(), CentsOffset::new(-50.0));
        let a4 = PitchTarget::new(MidiNote::new(69).unwrap());
        // Flattened by half a semitone → ratio should be 2^(-50/1200)
        let expected_ratio = 2.0_f64.powf(-50.0 / 1200.0);
        let actual_ratio = a4_flat.frequency_hz() / a4.frequency_hz();
        assert!(
            (actual_ratio - expected_ratio).abs() < 1e-9,
            "expected ratio {}, got {}",
            expected_ratio,
            actual_ratio
        );
    }

    #[test]
    fn zero_cents_offset_does_not_change_frequency() {
        let a4_plain = PitchTarget::new(MidiNote::new(69).unwrap());
        let a4_zero = PitchTarget::with_tuning(MidiNote::new(69).unwrap(), CentsOffset::zero());
        let diff = (a4_plain.frequency_hz() - a4_zero.frequency_hz()).abs();
        assert!(diff < 1e-12);
    }

    #[test]
    fn midi_note_0_is_valid() {
        assert!(MidiNote::new(0).is_some());
    }

    #[test]
    fn midi_note_127_is_valid() {
        assert!(MidiNote::new(127).is_some());
    }

    #[test]
    fn midi_note_128_is_invalid() {
        assert!(MidiNote::new(128).is_none());
    }

    #[test]
    fn velocity_1_is_valid() {
        assert!(Velocity::new(1).is_some());
    }

    #[test]
    fn velocity_127_is_valid() {
        assert!(Velocity::new(127).is_some());
    }

    #[test]
    fn velocity_0_is_invalid() {
        assert!(Velocity::new(0).is_none(), "velocity 0 should be rejected");
    }

    #[test]
    fn velocity_128_is_invalid() {
        assert!(
            Velocity::new(128).is_none(),
            "velocity 128 should be rejected"
        );
    }

    #[test]
    fn default_velocity_is_mezzo_forte() {
        assert_eq!(Velocity::default().as_u8(), 64);
    }

    #[test]
    fn velocity_normalises_to_0_to_1() {
        let v = Velocity::new(127).unwrap();
        assert!((v.as_f32() - 1.0).abs() < 1e-6);
        let v1 = Velocity::new(1).unwrap();
        assert!(v1.as_f32() > 0.0);
        assert!(v1.as_f32() < 1.0 / 64.0);
    }

    #[test]
    fn note_target_round_trip_serde() {
        let note = NoteTarget {
            pitch: PitchTarget::new(MidiNote::new(60).unwrap()),
            onset: TimePoint::from_millis(100),
            duration: NoteDuration::from_millis(250),
            velocity: Velocity::mezzo_forte(),
            articulation: NoteArticulation::Staccato,
        };
        let json = serde_json::to_string(&note).unwrap();
        let decoded: NoteTarget = serde_json::from_str(&json).unwrap();
        assert_eq!(note, decoded);
    }

    #[test]
    fn default_articulation_is_neutral() {
        assert_eq!(NoteArticulation::default(), NoteArticulation::Neutral);
    }
}
