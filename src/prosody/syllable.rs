//! Structured syllable representation for prosody and singing.
//!
//! A [`Syllable`] captures the phonological structure of one syllable using the
//! existing [`PhoneString`] type from the phonology layer: onset consonants,
//! nucleus vowel(s), and coda consonants — plus the source-index span back into
//! the originating [`Phoneme`] slice, an optional stress level, the variety
//! profile name that produced the parse, and diagnostics explaining any
//! non-trivial parse decisions.
//!
//! The canonical way to render a syllable sequence is
//! [`crate::prosody::syllabification::syllables_to_ipa`], which produces
//! notation like `ˈɛk.stɹʌ` or `ˈæt.lʌs`.
//!
//! [`Phoneme`]: crate::linguistic::phonology::Phoneme
//! [`PhoneString`]: crate::linguistic::phonology::PhoneString

use serde::{Deserialize, Serialize};

use crate::linguistic::phonology::{Phone, PhoneStatus, PhoneString, Stress};

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
/// use listenbury::prosody::syllable::Syllable;
/// use listenbury::linguistic::phonology::{Phone, PhoneString, Stress};
///
/// // Syllable representing /ˈɛk/ in "extra"
/// let syl = Syllable {
///     onset:   PhoneString::empty(),
///     nucleus: PhoneString { phones: vec![Phone::new_ipa("ɛ")] },
///     coda:    PhoneString { phones: vec![Phone::new_ipa("k")] },
///     source_span: listenbury::prosody::syllable::SourceSpan { start: 0, end: 2 },
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
    /// use listenbury::prosody::syllable::Syllable;
    /// use listenbury::linguistic::phonology::{Phone, PhoneString};
    ///
    /// let syl = Syllable {
    ///     onset:   PhoneString { phones: vec![
    ///         Phone::new_ipa("s"), Phone::new_ipa("t"), Phone::new_ipa("ɹ"),
    ///     ]},
    ///     nucleus: PhoneString { phones: vec![Phone::new_ipa("ʌ")] },
    ///     coda:    PhoneString::empty(),
    ///     source_span: listenbury::prosody::syllable::SourceSpan { start: 2, end: 6 },
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
}
