//! Structured syllable representation for prosody and singing.
//!
//! A [`Syllable`] captures the phonological structure of one syllable using the
//! existing [`PhoneString`] type from the phonology layer: onset consonants,
//! nucleus vowel(s), and coda consonants — plus the source-index span back into
//! the originating [`Phoneme`] slice, an optional stress level, the variety
//! profile name that produced the parse, and diagnostics explaining any
//! non-trivial parse decisions.
//!
//! For singing, [`SungSyllable`] binds phone timing and musical targets to the
//! same onset/nucleus/coda structure. A syllable may carry a note-level intent
//! (`note`, optional `pitch_curve`, optional `vibrato`), but pitch-bearing
//! material is tracked explicitly via the nucleus span so consonant attack and
//! release phones are not blindly treated as sustained pitch.
//!
//! The canonical way to render a syllable sequence is
//! [`crate::prosody::syllabification::syllables_to_ipa`], which produces
//! notation like `ˈɛk.stɹʌ` or `ˈæt.lʌs`.
//!
//! [`Phoneme`]: crate::linguistic::phonology::Phoneme
//! [`PhoneString`]: crate::linguistic::phonology::PhoneString

use serde::{Deserialize, Serialize};

use crate::linguistic::phonology::{Phone, PhoneStatus, PhoneString, Stress};
use crate::prosody::note_target::{NoteTarget, TimePoint};
use crate::prosody::pitch_curve::PitchCurve;
use crate::prosody::vibrato::Vibrato;

// ─── Diagnostic ──────────────────────────────────────────────────────────────

/// Classification of a [`SyllableDiagnostic`] entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticKind {
    /// An onset cluster was found to be legal under the active profile.
    LegalOnset,
    /// An onset cluster was rejected by the active profile and a shorter
    /// (or empty) onset was used instead.
    RejectedOnset,
    /// More than one valid syllabification existed; the most onset-maximal
    /// legal parse was chosen.
    AmbiguousSyllabification,
    /// No fully legal parse was found; a best-effort fallback was used.
    FallbackParse,
    /// A consonant was treated as a syllabic nucleus (e.g. syllabic /l̩/, /n̩/).
    SyllabicConsonant,
    /// The decision was variety-specific (i.e. it differs across profiles).
    VarietySpecific,
}

/// A single diagnostic note attached to a [`Syllable`].
///
/// Diagnostics explain syllabification decisions — accepted onsets, rejected
/// clusters, fallback parses — so that phonological bugs are visible without
/// needing a debugger.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SyllableDiagnostic {
    /// What kind of event this diagnostic records.
    pub kind: DiagnosticKind,
    /// Human-readable description of the event.
    pub message: String,
}

impl SyllableDiagnostic {
    pub fn new(kind: DiagnosticKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

// ─── Syllable ─────────────────────────────────────────────────────────────────

/// Inclusive/exclusive source index span into the original `&[Phoneme]` slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceSpan {
    /// Inclusive start index into the source `&[Phoneme]` slice.
    pub start: usize,
    /// Exclusive end index into the source `&[Phoneme]` slice.
    pub end: usize,
}

/// A half-open span (`start..end`) over syllable phone indices.
///
/// Spans are validated by [`PhoneSpan::new`] and used by [`SungSyllable`] to
/// identify onset, nucleus, coda, and pitch-bearing ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhoneSpan {
    /// Inclusive start index.
    pub start: usize,
    /// Exclusive end index.
    pub end: usize,
}

impl PhoneSpan {
    /// Create a validated phone span where `start <= end`.
    pub fn new(start: usize, end: usize) -> Result<Self, NucleusSpanError> {
        if start > end {
            return Err(NucleusSpanError::Inverted);
        }
        Ok(Self { start, end })
    }

    #[inline]
    pub fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

/// A phone plus phrase-relative timing metadata.
///
/// This is a lightweight adapter over Listenbury's existing [`Phone`] type so
/// prosody can attach musical intent without reworking phonology.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimedPhoneRef {
    pub phone: Phone,
    pub start: TimePoint,
    pub end: TimePoint,
}

impl TimedPhoneRef {
    /// Build a timed phone. Returns `None` when `end < start`.
    pub fn new(phone: Phone, start: TimePoint, end: TimePoint) -> Option<Self> {
        if end.millis < start.millis {
            return None;
        }
        Some(Self { phone, start, end })
    }
}

/// Construction errors for [`SungSyllable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NucleusSpanError {
    /// A span had `start > end`.
    Inverted,
    /// Span points outside the phone vector bounds.
    OutOfBounds,
    /// The nucleus span must include at least one phone.
    EmptyNucleus,
    /// Onset/nucleus/coda spans overlapped or were out of order.
    OverlappingOrUnordered,
    /// Onset/nucleus/coda must partition the full phone list (`0..phones.len()`).
    NonContiguousStructure,
}

/// Singable syllable with phone timing and optional musical targets.
#[derive(Debug, Clone, PartialEq)]
pub struct SungSyllable {
    /// Source orthography (e.g. `"hel"` or `"lo"`).
    pub text: String,
    /// Timed phones for this syllable.
    pub phones: Vec<TimedPhoneRef>,
    /// Onset phone span in `phones`.
    pub onset: PhoneSpan,
    /// Pitch-bearing nucleus span in `phones`.
    pub nucleus: PhoneSpan,
    /// Coda phone span in `phones`.
    pub coda: PhoneSpan,
    /// Stress metadata, when available.
    pub stress: Option<Stress>,
    /// Optional note target for sung realization.
    pub note: Option<NoteTarget>,
    /// Optional continuous pitch override/shape.
    pub pitch_curve: Option<PitchCurve>,
    /// Optional vibrato intent layered on top of note/curve pitch.
    pub vibrato: Option<Vibrato>,
}

impl SungSyllable {
    /// Construct a validated singable syllable.
    ///
    /// The onset, nucleus, and coda spans must be ordered, non-overlapping, and
    /// contiguous partitions of the full `phones` slice. The nucleus is required
    /// to be non-empty so pitch-bearing phones are always representable.
    pub fn new(
        text: impl Into<String>,
        phones: Vec<TimedPhoneRef>,
        onset: PhoneSpan,
        nucleus: PhoneSpan,
        coda: PhoneSpan,
        stress: Option<Stress>,
        note: Option<NoteTarget>,
    ) -> Result<Self, NucleusSpanError> {
        let len = phones.len();
        for span in [onset, nucleus, coda] {
            if span.start > span.end {
                return Err(NucleusSpanError::Inverted);
            }
            if span.end > len {
                return Err(NucleusSpanError::OutOfBounds);
            }
        }
        if nucleus.is_empty() {
            return Err(NucleusSpanError::EmptyNucleus);
        }
        if onset.end > nucleus.start || nucleus.end > coda.start {
            return Err(NucleusSpanError::OverlappingOrUnordered);
        }
        if onset.start != 0
            || onset.end != nucleus.start
            || nucleus.end != coda.start
            || coda.end != len
        {
            return Err(NucleusSpanError::NonContiguousStructure);
        }

        Ok(Self {
            text: text.into(),
            phones,
            onset,
            nucleus,
            coda,
            stress,
            note,
            pitch_curve: None,
            vibrato: None,
        })
    }

    /// The default pitch-bearing span(s) for this syllable.
    ///
    /// Today this is the nucleus span, which matches typical singing where
    /// consonants form attack/release and vowel material carries sustained pitch.
    pub fn pitch_bearing_spans(&self) -> [PhoneSpan; 1] {
        [self.nucleus]
    }

    /// Start time from the first phone (or `None` when phone list is empty).
    pub fn start_time(&self) -> Option<TimePoint> {
        self.phones.first().map(|phone| phone.start)
    }

    /// End time from the last phone (or `None` when phone list is empty).
    pub fn end_time(&self) -> Option<TimePoint> {
        self.phones.last().map(|phone| phone.end)
    }

    /// Syllable duration derived from timed phone boundaries.
    pub fn duration_millis(&self) -> Option<u64> {
        Some(
            self.end_time()?
                .millis
                .saturating_sub(self.start_time()?.millis),
        )
    }

    /// Attach an optional pitch curve.
    pub fn with_pitch_curve(mut self, pitch_curve: Option<PitchCurve>) -> Self {
        self.pitch_curve = pitch_curve;
        self
    }

    /// Attach an optional vibrato target.
    pub fn with_vibrato(mut self, vibrato: Option<Vibrato>) -> Self {
        self.vibrato = vibrato;
        self
    }
}

/// A phonological syllable produced by the syllabifier.
///
/// Each constituent is stored as a [`PhoneString`] (a `Vec<Phone>`) where
/// every [`Phone`] carries its IPA surface form in `phone.ipa`:
///
/// | Field | Contents |
/// |-------|----------|
/// | `onset`   | Onset consonant phones, e.g. `[s, t, ɹ]` |
/// | `nucleus` | Nucleus phone(s), e.g. `[ɛ]` or `[eɪ]` for a diphthong |
/// | `coda`    | Coda consonant phones, e.g. `[k]` |
///
/// Diphthongs (`aɪ`, `eɪ`, `oʊ`, …) and affricates (`tʃ`, `dʒ`) appear as
/// a single `Phone` whose `.ipa` is the multi-character IPA string, matching
/// the phoneme's [`realization.ipa`][`crate::linguistic::phonology::Realization`].
///
/// The `source_span.start..source_span.end` span indexes back into the `&[Phoneme]`
/// slice passed to the syllabifier, enabling downstream code to recover
/// timing, allophone, and morphological data without re-parsing.
///
/// # Example
///
/// ```
/// use listenbury::prosody::syllable::{SourceSpan, Syllable};
/// use listenbury::linguistic::phonology::{Phone, PhoneString, Stress};
///
/// // Syllable representing /ˈɛk/ in "extra"
/// let syl = Syllable {
///     onset:   PhoneString::empty(),
///     nucleus: PhoneString { phones: vec![Phone::new_ipa("ɛ")] },
///     coda:    PhoneString { phones: vec![Phone::new_ipa("k")] },
///     source_span: SourceSpan { start: 0, end: 2 },
///     stress: Some(Stress::Primary),
///     variety: "General American English".into(),
///     diagnostics: vec![],
/// };
/// assert_eq!(syl.nucleus.to_ipa(), "ɛ");
/// assert_eq!(syl.phones_to_ipa(), "ɛk");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Syllable {
    /// Onset consonant phones, in sequence order.
    pub onset: PhoneString,
    /// Nucleus phone(s).  Usually a single phone; two for a diphthong when
    /// the phone inventory encodes them as separate phones rather than one.
    pub nucleus: PhoneString,
    /// Coda consonant phones, in sequence order.
    pub coda: PhoneString,
    /// Source span into the original `&[Phoneme]` slice.
    pub source_span: SourceSpan,
    /// Stress level inferred from the nucleus phone's corresponding
    /// [`Phoneme.stress`][`crate::linguistic::phonology::Phoneme`] field.
    pub stress: Option<Stress>,
    /// Display name of the [`crate::prosody::phonotactics::PhonotacticProfile`]
    /// that produced this syllable.
    pub variety: String,
    /// Diagnostics generated during syllabification of this syllable.
    pub diagnostics: Vec<SyllableDiagnostic>,
}

impl Syllable {
    /// Iterate over all [`Phone`]s in onset → nucleus → coda order.
    pub fn phones(&self) -> impl Iterator<Item = &Phone> {
        self.onset
            .phones
            .iter()
            .chain(self.nucleus.phones.iter())
            .chain(self.coda.phones.iter())
    }

    /// Concatenate all phones in this syllable into a single IPA string.
    ///
    /// No stress marker or inter-syllable dot is included; use
    /// [`syllables_to_ipa`][`crate::prosody::syllabification::syllables_to_ipa`]
    /// for a fully-rendered transcription.
    ///
    /// # Example
    ///
    /// ```
    /// use listenbury::prosody::syllable::{SourceSpan, Syllable};
    /// use listenbury::linguistic::phonology::{Phone, PhoneString};
    ///
    /// let syl = Syllable {
    ///     onset:   PhoneString { phones: vec![
    ///         Phone::new_ipa("s"), Phone::new_ipa("t"), Phone::new_ipa("ɹ"),
    ///     ]},
    ///     nucleus: PhoneString { phones: vec![Phone::new_ipa("ʌ")] },
    ///     coda:    PhoneString::empty(),
    ///     source_span: SourceSpan { start: 2, end: 6 },
    ///     stress: None,
    ///     variety: "General American English".into(),
    ///     diagnostics: vec![],
    /// };
    /// assert_eq!(syl.phones_to_ipa(), "stɹʌ");
    /// ```
    pub fn phones_to_ipa(&self) -> String {
        self.phones().map(|p| p.ipa.as_str()).collect()
    }

    /// Return `true` if this syllable has no nucleus — a degenerate consonant
    /// cluster returned when no vowel was found in the input.
    pub fn is_nucleus_empty(&self) -> bool {
        self.nucleus.phones.is_empty()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prosody::note_target::{
        MidiNote, NoteArticulation, NoteDuration, PitchTarget, TimePoint, Velocity,
    };
    use crate::prosody::pitch_curve::{Interpolation, PitchCurve, PitchPoint};
    use std::time::Duration;

    fn phone(ipa: &str) -> Phone {
        Phone {
            ipa: ipa.to_string(),
            source_symbol: None,
            status: PhoneStatus::Mapped,
        }
    }

    fn ps(phones: &[&str]) -> PhoneString {
        PhoneString {
            phones: phones.iter().map(|s| phone(s)).collect(),
        }
    }

    fn timed_phone(ipa: &str, start_ms: u64, end_ms: u64) -> TimedPhoneRef {
        TimedPhoneRef::new(
            phone(ipa),
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

    fn syl(onset: &[&str], nucleus: &[&str], coda: &[&str]) -> Syllable {
        Syllable {
            onset: ps(onset),
            nucleus: ps(nucleus),
            coda: ps(coda),
            source_span: SourceSpan {
                start: 0,
                end: onset.len() + nucleus.len() + coda.len(),
            },
            stress: None,
            variety: "General American English".into(),
            diagnostics: vec![],
        }
    }

    #[test]
    fn phones_iterates_onset_nucleus_coda_in_order() {
        let s = syl(&["s", "t", "ɹ"], &["ʌ"], &[]);
        let got: Vec<&str> = s.phones().map(|p| p.ipa.as_str()).collect();
        assert_eq!(got, vec!["s", "t", "ɹ", "ʌ"]);
    }

    #[test]
    fn phones_to_ipa_concatenates_with_no_separator() {
        let s = syl(&["s", "t", "ɹ"], &["ʌ"], &[]);
        assert_eq!(s.phones_to_ipa(), "stɹʌ");
    }

    #[test]
    fn phones_to_ipa_includes_coda() {
        // /ɛk/
        let s = syl(&[], &["ɛ"], &["k"]);
        assert_eq!(s.phones_to_ipa(), "ɛk");
    }

    #[test]
    fn diphthong_nucleus_is_single_phone_entry() {
        // /eɪ/ as one Phone whose .ipa is "eɪ"
        let s = syl(&["p"], &["eɪ"], &[]);
        assert_eq!(s.phones_to_ipa(), "peɪ");
        assert_eq!(s.nucleus.phones.len(), 1);
        assert_eq!(s.nucleus.phones[0].ipa, "eɪ");
    }

    #[test]
    fn affricate_onset_is_single_phone_entry() {
        // /tʃ/ as one Phone whose .ipa is "tʃ"
        let s = syl(&["tʃ"], &["ɪ"], &["p"]);
        assert_eq!(s.phones_to_ipa(), "tʃɪp");
        assert_eq!(s.onset.phones.len(), 1);
        assert_eq!(s.onset.phones[0].ipa, "tʃ");
    }

    #[test]
    fn is_nucleus_empty_when_no_nucleus() {
        let s = syl(&["s"], &[], &[]);
        assert!(s.is_nucleus_empty());
    }

    #[test]
    fn is_nucleus_empty_false_when_nucleus_present() {
        let s = syl(&[], &["ɛ"], &[]);
        assert!(!s.is_nucleus_empty());
    }

    #[test]
    fn diagnostic_construction() {
        let d = SyllableDiagnostic::new(DiagnosticKind::RejectedOnset, "/tl/ is not legal");
        assert_eq!(d.kind, DiagnosticKind::RejectedOnset);
        assert_eq!(d.message, "/tl/ is not legal");
    }

    #[test]
    fn phone_new_ipa_helper() {
        let p = Phone::new_ipa("ɹ");
        assert_eq!(p.ipa, "ɹ");
        assert_eq!(p.status, PhoneStatus::Mapped);
        assert!(p.source_symbol.is_none());
    }

    #[test]
    fn phone_string_to_ipa() {
        let ps = PhoneString {
            phones: vec![
                Phone::new_ipa("s"),
                Phone::new_ipa("t"),
                Phone::new_ipa("ɹ"),
                Phone::new_ipa("ʌ"),
            ],
        };
        assert_eq!(ps.to_ipa(), "stɹʌ");
    }

    #[test]
    fn phone_string_empty() {
        let ps = PhoneString::empty();
        assert!(ps.phones.is_empty());
        assert_eq!(ps.to_ipa(), "");
    }

    #[test]
    fn sung_cv_syllable_marks_vowel_nucleus_as_pitch_bearing() {
        let phones = vec![timed_phone("h", 0, 40), timed_phone("ɛ", 40, 220)];
        let note = make_note(60, 0, 220);

        let curve = PitchCurve::new(
            vec![
                PitchPoint::new(Duration::from_millis(0), 261.63),
                PitchPoint::new(Duration::from_millis(220), 261.63),
            ],
            Interpolation::Linear,
        )
        .expect("non-empty");

        let syllable = SungSyllable::new(
            "he",
            phones,
            PhoneSpan::new(0, 1).unwrap(),
            PhoneSpan::new(1, 2).unwrap(),
            PhoneSpan::new(2, 2).unwrap(),
            Some(Stress::Primary),
            Some(note),
        )
        .unwrap()
        .with_pitch_curve(Some(curve))
        .with_vibrato(Some(Vibrato::new(
            5.5,
            20.0,
            Duration::from_millis(40),
            Duration::from_millis(60),
            0.0,
        )));

        assert_eq!(
            syllable.pitch_bearing_spans(),
            [PhoneSpan { start: 1, end: 2 }]
        );
        assert_eq!(syllable.phones[1].phone.ipa, "ɛ");
        assert!(syllable.note.is_some());
        assert!(syllable.pitch_curve.is_some());
        assert!(syllable.vibrato.is_some());
    }

    #[test]
    fn sung_cvc_syllable_separates_attack_nucleus_and_release() {
        let phones = vec![
            timed_phone("h", 0, 30),
            timed_phone("ɛ", 30, 180),
            timed_phone("l", 180, 240),
        ];
        let syllable = SungSyllable::new(
            "hel",
            phones,
            PhoneSpan::new(0, 1).unwrap(),
            PhoneSpan::new(1, 2).unwrap(),
            PhoneSpan::new(2, 3).unwrap(),
            Some(Stress::Primary),
            Some(make_note(60, 0, 240)),
        )
        .unwrap();

        assert_eq!(syllable.onset, PhoneSpan { start: 0, end: 1 });
        assert_eq!(syllable.nucleus, PhoneSpan { start: 1, end: 2 });
        assert_eq!(syllable.coda, PhoneSpan { start: 2, end: 3 });
        assert_eq!(syllable.duration_millis(), Some(240));
    }

    #[test]
    fn sung_syllable_without_note_target_is_valid_for_spoken_prosody() {
        let phones = vec![timed_phone("l", 0, 40), timed_phone("oʊ", 40, 220)];
        let syllable = SungSyllable::new(
            "lo",
            phones,
            PhoneSpan::new(0, 1).unwrap(),
            PhoneSpan::new(1, 2).unwrap(),
            PhoneSpan::new(2, 2).unwrap(),
            Some(Stress::Unstressed),
            None,
        )
        .unwrap();

        assert!(syllable.note.is_none());
        assert_eq!(
            syllable.pitch_bearing_spans(),
            [PhoneSpan { start: 1, end: 2 }]
        );
    }

    #[test]
    fn sung_syllable_rejects_invalid_nucleus_spans() {
        let phones = vec![timed_phone("h", 0, 30), timed_phone("ɛ", 30, 180)];

        let empty_nucleus = SungSyllable::new(
            "he",
            phones.clone(),
            PhoneSpan::new(0, 1).unwrap(),
            PhoneSpan::new(1, 1).unwrap(),
            PhoneSpan::new(1, 2).unwrap(),
            None,
            None,
        );
        assert_eq!(empty_nucleus, Err(NucleusSpanError::EmptyNucleus));

        let out_of_bounds_nucleus = SungSyllable::new(
            "he",
            phones.clone(),
            PhoneSpan::new(0, 1).unwrap(),
            PhoneSpan::new(1, 3).unwrap(),
            PhoneSpan::new(3, 3).unwrap(),
            None,
            None,
        );
        assert_eq!(out_of_bounds_nucleus, Err(NucleusSpanError::OutOfBounds));

        let non_contiguous = SungSyllable::new(
            "he",
            phones,
            PhoneSpan::new(0, 0).unwrap(),
            PhoneSpan::new(1, 2).unwrap(),
            PhoneSpan::new(2, 2).unwrap(),
            None,
            None,
        );
        assert_eq!(
            non_contiguous,
            Err(NucleusSpanError::NonContiguousStructure)
        );
    }
}
