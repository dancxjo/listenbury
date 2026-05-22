//! Singing phrase model for chaining syllables into a melody.
//!
//! A [`SungPhrase`] is a semantic singing plan: a sequence of [`SungSyllable`]
//! values with phrase-level pitch, timing, and expressive intent derived from
//! the constituent syllables.
//!
//! This module sits **one layer above** individual syllables and **one layer
//! below** audio rendering.  It is *not* a MIDI sequencer, score notation
//! system, or audio synthesis engine.  Its job is to aggregate syllable-level
//! musical intent into a phrase that voice renderers can consume.
//!
//! # Example
//!
//! ```
//! use listenbury::prosody::singing::{AppendError, SungPhrase};
//! use listenbury::prosody::syllable::{PhoneSpan, SungSyllable, TimedPhoneRef};
//! use listenbury::prosody::note_target::{
//!     MidiNote, NoteArticulation, NoteDuration, NoteTarget, PitchTarget, TimePoint, Velocity,
//! };
//! use listenbury::linguistic::phonology::Phone;
//!
//! fn timed(ipa: &str, s: u64, e: u64) -> TimedPhoneRef {
//!     TimedPhoneRef::new(
//!         Phone::new_ipa(ipa),
//!         TimePoint::from_millis(s),
//!         TimePoint::from_millis(e),
//!     ).unwrap()
//! }
//!
//! fn note(midi: u8, onset_ms: u64, duration_ms: u64) -> NoteTarget {
//!     NoteTarget {
//!         pitch: PitchTarget::new(MidiNote::new(midi).unwrap()),
//!         onset: TimePoint::from_millis(onset_ms),
//!         duration: NoteDuration::from_millis(duration_ms),
//!         velocity: Velocity::mezzo_forte(),
//!         articulation: NoteArticulation::Neutral,
//!     }
//! }
//!
//! // "hel" on C4 (MIDI 60), "lo" on G4 (MIDI 67)
//! let hel = SungSyllable::new(
//!     "hel",
//!     vec![timed("h", 0, 30), timed("ɛ", 30, 180), timed("l", 180, 240)],
//!     PhoneSpan::new(0, 1).unwrap(),
//!     PhoneSpan::new(1, 2).unwrap(),
//!     PhoneSpan::new(2, 3).unwrap(),
//!     None,
//!     Some(note(60, 0, 240)),
//! ).unwrap();
//!
//! let lo = SungSyllable::new(
//!     "lo",
//!     vec![timed("l", 240, 270), timed("oʊ", 270, 490)],
//!     PhoneSpan::new(0, 1).unwrap(),
//!     PhoneSpan::new(1, 2).unwrap(),
//!     PhoneSpan::new(2, 2).unwrap(),
//!     None,
//!     Some(note(67, 240, 250)),
//! ).unwrap();
//!
//! let mut phrase = SungPhrase::new();
//! phrase.push(hel).unwrap();
//! phrase.push(lo).unwrap();
//!
//! assert_eq!(phrase.text(), "hello");
//! assert_eq!(phrase.total_duration_millis(), Some(490));
//! ```

use crate::prosody::note_target::TimePoint;
use crate::prosody::pitch_curve::{Interpolation, PitchCurve};
use crate::prosody::syllable::SungSyllable;

// ─── PhraseGap ───────────────────────────────────────────────────────────────

/// A rest or silence detected between two consecutive [`SungSyllable`] values.
///
/// A gap is present when the end time of a syllable is strictly less than the
/// start time of the next.  Zero-length boundaries (back-to-back syllables)
/// are not gaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhraseGap {
    /// Start of the silence — the end of the preceding syllable.
    pub start: TimePoint,
    /// End of the silence — the start of the following syllable.
    pub end: TimePoint,
}

impl PhraseGap {
    /// Duration of the gap in milliseconds.
    #[inline]
    pub fn duration_millis(&self) -> u64 {
        self.end.millis.saturating_sub(self.start.millis)
    }
}

// ─── AppendError ─────────────────────────────────────────────────────────────

/// Reasons a syllable cannot be appended to a [`SungPhrase`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppendError {
    /// The syllable carries no timed phones so its position cannot be determined.
    NoTiming,
    /// The new syllable starts before the previous one ends (overlapping timing).
    OverlapWithPrevious,
}

// ─── SungPhrase ──────────────────────────────────────────────────────────────

/// A sequence of [`SungSyllable`] values forming a singable phrase.
///
/// `SungPhrase` represents a **semantic singing plan**: syllables in text and
/// musical order, with phrase-level duration and a combined pitch curve derived
/// from the individual syllables.  It is renderer-agnostic and carries no audio
/// synthesis state.
///
/// Syllables must be appended in non-overlapping time order via [`push`].
/// Gaps (rests) between syllables are permitted and are reported by [`gaps`].
///
/// [`push`]: SungPhrase::push
/// [`gaps`]: SungPhrase::gaps
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SungPhrase {
    /// Syllables in text/musical order.
    pub syllables: Vec<SungSyllable>,
}

impl SungPhrase {
    /// Create an empty phrase.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a syllable to the phrase.
    ///
    /// The syllable must have timed phones (otherwise its position in time is
    /// unknown) and its start time must be ≥ the end time of the last syllable
    /// already in the phrase.
    ///
    /// # Errors
    ///
    /// - [`AppendError::NoTiming`] – the syllable (or the last existing one)
    ///   has no phone timing data.
    /// - [`AppendError::OverlapWithPrevious`] – the new syllable starts before
    ///   the previous one ends.
    pub fn push(&mut self, syllable: SungSyllable) -> Result<(), AppendError> {
        let start = syllable.start_time().ok_or(AppendError::NoTiming)?;
        if let Some(prev) = self.syllables.last() {
            let prev_end = prev.end_time().ok_or(AppendError::NoTiming)?;
            if start.millis < prev_end.millis {
                return Err(AppendError::OverlapWithPrevious);
            }
        }
        self.syllables.push(syllable);
        Ok(())
    }

    /// Concatenated source text across all syllables, in order.
    pub fn text(&self) -> String {
        self.syllables.iter().map(|s| s.text.as_str()).collect()
    }

    /// Start time of the first syllable, or `None` if the phrase is empty.
    pub fn start_time(&self) -> Option<TimePoint> {
        self.syllables.first().and_then(|s| s.start_time())
    }

    /// End time of the last syllable, or `None` if the phrase is empty.
    pub fn end_time(&self) -> Option<TimePoint> {
        self.syllables.last().and_then(|s| s.end_time())
    }

    /// Total phrase duration in milliseconds.
    ///
    /// Spans from the start of the first syllable to the end of the last,
    /// including any inter-syllable gaps.  Returns `None` for an empty phrase.
    pub fn total_duration_millis(&self) -> Option<u64> {
        Some(
            self.end_time()?
                .millis
                .saturating_sub(self.start_time()?.millis),
        )
    }

    /// Detect rests (gaps) between consecutive syllables.
    ///
    /// Returns one [`PhraseGap`] for each pair of adjacent syllables where the
    /// end of the first is strictly earlier than the start of the second.
    /// Back-to-back syllables (gap of zero ms) do not produce an entry.
    pub fn gaps(&self) -> Vec<PhraseGap> {
        let mut result = Vec::new();
        for pair in self.syllables.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            if let (Some(a_end), Some(b_start)) = (a.end_time(), b.start_time()) {
                if b_start.millis > a_end.millis {
                    result.push(PhraseGap {
                        start: a_end,
                        end: b_start,
                    });
                }
            }
        }
        result
    }

    /// Derive a phrase-level [`PitchCurve`] from syllable note targets.
    ///
    /// Each syllable that carries a [`NoteTarget`] contributes its note to the
    /// combined curve.  Syllables without note targets are silently skipped.
    /// The supplied `interpolation` strategy is used between all control points.
    ///
    /// Returns `None` when no syllable in the phrase has a note target.
    pub fn phrase_pitch_curve(&self, interpolation: Interpolation) -> Option<PitchCurve> {
        let notes: Vec<_> = self
            .syllables
            .iter()
            .filter_map(|s| s.note.as_ref())
            .collect();
        PitchCurve::from_note_targets(notes, interpolation)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linguistic::phonology::Phone;
    use crate::prosody::note_target::{
        MidiNote, NoteArticulation, NoteDuration, NoteTarget, PitchTarget, TimePoint, Velocity,
    };
    use crate::prosody::pitch_curve::Interpolation;
    use crate::prosody::syllable::{PhoneSpan, SungSyllable, TimedPhoneRef};

    // ── helpers ──────────────────────────────────────────────────────────────

    fn timed_phone(ipa: &str, start_ms: u64, end_ms: u64) -> TimedPhoneRef {
        TimedPhoneRef::new(
            Phone::new_ipa(ipa),
            TimePoint::from_millis(start_ms),
            TimePoint::from_millis(end_ms),
        )
        .expect("valid timed phone")
    }

    fn make_note(midi: u8, onset_ms: u64, duration_ms: u64) -> NoteTarget {
        NoteTarget {
            pitch: PitchTarget::new(MidiNote::new(midi).unwrap()),
            onset: TimePoint::from_millis(onset_ms),
            duration: NoteDuration::from_millis(duration_ms),
            velocity: Velocity::mezzo_forte(),
            articulation: NoteArticulation::Neutral,
        }
    }

    /// Build a CVC syllable at a specific time window.
    fn make_syllable(
        text: &str,
        onset_ipa: &str,
        nucleus_ipa: &str,
        coda_ipa: &str,
        start_ms: u64,
        end_ms: u64,
        note: Option<NoteTarget>,
    ) -> SungSyllable {
        // onset: [start_ms, start_ms+20), nucleus: [+20, end_ms-20), coda: [end_ms-20, end_ms)
        let mid1 = start_ms + 20;
        let mid2 = end_ms.saturating_sub(20);
        let phones = vec![
            timed_phone(onset_ipa, start_ms, mid1),
            timed_phone(nucleus_ipa, mid1, mid2),
            timed_phone(coda_ipa, mid2, end_ms),
        ];
        SungSyllable::new(
            text,
            phones,
            PhoneSpan::new(0, 1).unwrap(),
            PhoneSpan::new(1, 2).unwrap(),
            PhoneSpan::new(2, 3).unwrap(),
            None,
            note,
        )
        .expect("valid syllable")
    }

    /// Build a CV syllable (no coda) at a specific time window.
    fn make_cv_syllable(
        text: &str,
        onset_ipa: &str,
        nucleus_ipa: &str,
        start_ms: u64,
        end_ms: u64,
        note: Option<NoteTarget>,
    ) -> SungSyllable {
        let mid = start_ms + 20;
        let phones = vec![
            timed_phone(onset_ipa, start_ms, mid),
            timed_phone(nucleus_ipa, mid, end_ms),
        ];
        SungSyllable::new(
            text,
            phones,
            PhoneSpan::new(0, 1).unwrap(),
            PhoneSpan::new(1, 2).unwrap(),
            PhoneSpan::new(2, 2).unwrap(),
            None,
            note,
        )
        .expect("valid syllable")
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn two_syllable_melody_hel_c4_lo_g4() {
        // "hel" on C4 (MIDI 60), "lo" on G4 (MIDI 67)
        let hel = make_syllable("hel", "h", "ɛ", "l", 0, 240, Some(make_note(60, 0, 240)));
        let lo = make_cv_syllable("lo", "l", "oʊ", 240, 490, Some(make_note(67, 240, 250)));

        let mut phrase = SungPhrase::new();
        phrase.push(hel).unwrap();
        phrase.push(lo).unwrap();

        assert_eq!(phrase.text(), "hello");
        assert_eq!(phrase.syllables.len(), 2);
    }

    #[test]
    fn phrase_duration_spans_all_syllables() {
        let hel = make_syllable("hel", "h", "ɛ", "l", 0, 240, None);
        let lo = make_cv_syllable("lo", "l", "oʊ", 240, 490, None);

        let mut phrase = SungPhrase::new();
        phrase.push(hel).unwrap();
        phrase.push(lo).unwrap();

        assert_eq!(phrase.total_duration_millis(), Some(490));
    }

    #[test]
    fn empty_phrase_has_no_duration() {
        let phrase = SungPhrase::new();
        assert_eq!(phrase.total_duration_millis(), None);
    }

    #[test]
    fn phrase_pitch_curve_spans_syllable_boundaries() {
        use std::time::Duration;

        // C4 = MIDI 60 ≈ 261.63 Hz, G4 = MIDI 67 ≈ 392.00 Hz
        let hel = make_syllable("hel", "h", "ɛ", "l", 0, 240, Some(make_note(60, 0, 240)));
        let lo = make_cv_syllable("lo", "l", "oʊ", 240, 490, Some(make_note(67, 240, 250)));

        let mut phrase = SungPhrase::new();
        phrase.push(hel).unwrap();
        phrase.push(lo).unwrap();

        let curve = phrase
            .phrase_pitch_curve(Interpolation::Linear)
            .expect("phrase has notes");

        // Before the phrase starts, curve clamps to C4.
        let c4_hz = MidiNote::new(60).unwrap().to_hz() as f32;
        let g4_hz = MidiNote::new(67).unwrap().to_hz() as f32;

        let sampled_start = curve.sample_hz(Duration::from_millis(0));
        let sampled_end = curve.sample_hz(Duration::from_millis(490));

        assert!((sampled_start - c4_hz).abs() < 0.1, "start ≈ C4");
        assert!((sampled_end - g4_hz).abs() < 0.1, "end ≈ G4");

        // Midway through the transition (at t=240 the curve should reach G4).
        let sampled_mid = curve.sample_hz(Duration::from_millis(240));
        assert!((sampled_mid - g4_hz).abs() < 0.1, "at t=240 ≈ G4");
    }

    #[test]
    fn phrase_pitch_curve_is_none_without_notes() {
        let hel = make_syllable("hel", "h", "ɛ", "l", 0, 240, None);
        let lo = make_cv_syllable("lo", "l", "oʊ", 240, 490, None);

        let mut phrase = SungPhrase::new();
        phrase.push(hel).unwrap();
        phrase.push(lo).unwrap();

        assert!(phrase.phrase_pitch_curve(Interpolation::Linear).is_none());
    }

    #[test]
    fn gap_between_syllables_is_detected() {
        // "hel" ends at 240 ms, "lo" starts at 340 ms → 100 ms gap.
        let hel = make_syllable("hel", "h", "ɛ", "l", 0, 240, None);
        let lo = make_cv_syllable("lo", "l", "oʊ", 340, 590, None);

        let mut phrase = SungPhrase::new();
        phrase.push(hel).unwrap();
        phrase.push(lo).unwrap();

        let gaps = phrase.gaps();
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].start, TimePoint::from_millis(240));
        assert_eq!(gaps[0].end, TimePoint::from_millis(340));
        assert_eq!(gaps[0].duration_millis(), 100);
    }

    #[test]
    fn back_to_back_syllables_have_no_gap() {
        let hel = make_syllable("hel", "h", "ɛ", "l", 0, 240, None);
        let lo = make_cv_syllable("lo", "l", "oʊ", 240, 490, None);

        let mut phrase = SungPhrase::new();
        phrase.push(hel).unwrap();
        phrase.push(lo).unwrap();

        assert!(phrase.gaps().is_empty());
    }

    #[test]
    fn overlapping_syllable_is_rejected() {
        // "hel" ends at 240; "lo" starts at 200 — overlaps by 40 ms.
        let hel = make_syllable("hel", "h", "ɛ", "l", 0, 240, None);
        let lo = make_cv_syllable("lo", "l", "oʊ", 200, 450, None);

        let mut phrase = SungPhrase::new();
        phrase.push(hel).unwrap();
        let err = phrase.push(lo).unwrap_err();

        assert_eq!(err, AppendError::OverlapWithPrevious);
    }

    #[test]
    fn single_syllable_phrase_has_correct_duration() {
        let hel = make_syllable("hel", "h", "ɛ", "l", 50, 290, None);
        let mut phrase = SungPhrase::new();
        phrase.push(hel).unwrap();
        assert_eq!(phrase.total_duration_millis(), Some(240));
        assert_eq!(phrase.start_time(), Some(TimePoint::from_millis(50)));
        assert_eq!(phrase.end_time(), Some(TimePoint::from_millis(290)));
    }
}
