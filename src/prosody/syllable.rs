//! Structured syllable representation for prosody and singing.
//!
//! A [`Syllable`] captures the phonological structure of one syllable as IPA
//! phone strings: onset consonants, nucleus vowel(s), and coda consonants —
//! plus the source-index span back into the originating [`Phoneme`] slice,
//! an optional stress level, the variety profile name that produced the parse,
//! and diagnostics explaining any non-trivial parse decisions.
//!
//! The canonical way to render a syllable sequence is
//! [`crate::prosody::syllabification::syllables_to_ipa`], which produces
//! notation like `ˈɛk.stɹʌ` or `ˈæt.ləs`.
//!
//! [`Phoneme`]: crate::linguistic::phonology::Phoneme

use serde::{Deserialize, Serialize};

use crate::linguistic::phonology::Stress;

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

/// A phonological syllable produced by the syllabifier.
///
/// Each field stores **IPA strings** (one string per phone, as produced by
/// [`crate::linguistic::phonology::Phoneme::realization`]):
///
/// | Field | Contents |
/// |-------|----------|
/// | `onset`   | Onset consonant IPA strings, e.g. `["s", "t", "ɹ"]` |
/// | `nucleus` | Nucleus IPA string(s), e.g. `["ɛ"]` or `["eɪ"]` |
/// | `coda`    | Coda consonant IPA strings, e.g. `["k"]` |
///
/// Diphthongs (`aɪ`, `eɪ`, `oʊ`, …) and affricates (`tʃ`, `dʒ`) appear as
/// single multi-character IPA strings matching the phone's
/// [`realization.ipa`][`crate::linguistic::phonology::Realization`].
///
/// The `source_start..source_end` span indexes back into the `&[Phoneme]`
/// slice that was passed to the syllabifier, allowing downstream code to
/// recover timing, allophone, and morphological data without re-parsing.
///
/// # Example
///
/// ```
/// use listenbury::prosody::syllable::{DiagnosticKind, Syllable, SyllableDiagnostic};
/// use listenbury::linguistic::phonology::Stress;
///
/// // Syllable representing /ˈɛk/ in "extra"
/// let syl = Syllable {
///     onset:  vec![],
///     nucleus: vec!["ɛ".into()],
///     coda:   vec!["k".into()],
///     source_start: 0,
///     source_end:   2,
///     stress: Some(Stress::Primary),
///     variety: "General American English".into(),
///     diagnostics: vec![],
/// };
/// assert_eq!(syl.nucleus, ["ɛ"]);
/// assert_eq!(syl.phones_to_ipa(), "ɛk");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Syllable {
    /// IPA strings for onset consonants, in sequence order.
    pub onset: Vec<String>,
    /// IPA string(s) for the nucleus (usually one; two for a diphthong if
    /// the phone inventory encodes them as two phones rather than one).
    pub nucleus: Vec<String>,
    /// IPA strings for coda consonants, in sequence order.
    pub coda: Vec<String>,
    /// Inclusive start index into the source `&[Phoneme]` slice.
    pub source_start: usize,
    /// Exclusive end index into the source `&[Phoneme]` slice.
    pub source_end: usize,
    /// Stress level inferred from the nucleus phone's `stress` field.
    pub stress: Option<Stress>,
    /// Display name of the [`crate::prosody::phonotactics::PhonotacticProfile`]
    /// that produced this syllable.
    pub variety: String,
    /// Diagnostics generated during syllabification of this syllable.
    pub diagnostics: Vec<SyllableDiagnostic>,
}

impl Syllable {
    /// Iterate over all IPA strings in onset → nucleus → coda order.
    pub fn phones(&self) -> impl Iterator<Item = &str> {
        self.onset
            .iter()
            .chain(self.nucleus.iter())
            .chain(self.coda.iter())
            .map(String::as_str)
    }

    /// Concatenate all phones in this syllable into a single IPA string.
    ///
    /// No stress marker or dot is included; use
    /// [`syllables_to_ipa`][`crate::prosody::syllabification::syllables_to_ipa`]
    /// for a fully-rendered transcription.
    ///
    /// # Example
    ///
    /// ```
    /// use listenbury::prosody::syllable::Syllable;
    ///
    /// let syl = Syllable {
    ///     onset:  vec!["s".into(), "t".into(), "ɹ".into()],
    ///     nucleus: vec!["ʌ".into()],
    ///     coda:   vec![],
    ///     source_start: 2,
    ///     source_end:   6,
    ///     stress: None,
    ///     variety: "General American English".into(),
    ///     diagnostics: vec![],
    /// };
    /// assert_eq!(syl.phones_to_ipa(), "stɹʌ");
    /// ```
    pub fn phones_to_ipa(&self) -> String {
        self.phones().collect()
    }

    /// Return `true` if this syllable has no nucleus — a degenerate consonant
    /// cluster returned when no vowel was found in the input.
    pub fn is_nucleus_empty(&self) -> bool {
        self.nucleus.is_empty()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn syl(onset: &[&str], nucleus: &[&str], coda: &[&str]) -> Syllable {
        Syllable {
            onset:  onset.iter().map(|s| s.to_string()).collect(),
            nucleus: nucleus.iter().map(|s| s.to_string()).collect(),
            coda:   coda.iter().map(|s| s.to_string()).collect(),
            source_start: 0,
            source_end:   onset.len() + nucleus.len() + coda.len(),
            stress: None,
            variety: "General American English".into(),
            diagnostics: vec![],
        }
    }

    #[test]
    fn phones_iterates_onset_nucleus_coda_in_order() {
        let s = syl(&["s", "t", "ɹ"], &["ʌ"], &[]);
        let got: Vec<&str> = s.phones().collect();
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
    fn diphthong_nucleus_is_single_entry() {
        // /eɪ/ as one IPA string
        let s = syl(&["p"], &["eɪ"], &[]);
        assert_eq!(s.phones_to_ipa(), "peɪ");
    }

    #[test]
    fn affricate_onset_is_single_entry() {
        // /tʃ/ as one IPA string
        let s = syl(&["tʃ"], &["ɪ"], &["p"]);
        assert_eq!(s.phones_to_ipa(), "tʃɪp");
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
        let d = SyllableDiagnostic::new(DiagnosticKind::RejectedOnset, "tl is not legal");
        assert_eq!(d.kind, DiagnosticKind::RejectedOnset);
        assert_eq!(d.message, "tl is not legal");
    }
}
