use std::collections::HashMap;
use std::sync::OnceLock;

use crate::linguistic::{
    orthography::OrthographicWord,
    phoneme::{Phoneme, PhonemeSeq, PhonemeText, PhonemeTextUnit},
    pronounce::{OrthographyToPhonemes, PhonologyError},
    variety::LinguisticVariety,
};

/// How a pronunciation was resolved during lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PronunciationStatus {
    /// The word was found in the dictionary exactly (case-insensitive match).
    Exact,
    /// The word was found after normalization (e.g. punctuation stripping).
    Normalized,
    /// The word was not found; a guess was provided by another means.
    Guessed,
    /// The word was not found and no pronunciation is available.
    Missing,
}

/// The result of a CMUdict pronunciation lookup for a single word.
///
/// Contains the original query, the normalized lookup key used, the source
/// identifier, **all** candidate pronunciation sequences, and a status flag
/// indicating how (or whether) the word was resolved.
///
/// # Example
///
/// ```
/// use listenbury::linguistic::CmudictPronouncer;
/// use listenbury::linguistic::PronunciationStatus;
///
/// let p = CmudictPronouncer::bundled();
/// let entry = p.lookup_entry("three");
/// assert_eq!(entry.status, PronunciationStatus::Exact);
/// assert!(!entry.candidates.is_empty());
/// assert_eq!(entry.source, "cmudict");
/// // All candidate pronunciations are returned.
/// assert!(entry.candidates.len() >= 1);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PronunciationEntry {
    /// The original (un-normalized) word text that was queried.
    pub original: String,
    /// The normalized form used for the dictionary lookup.
    pub lookup: String,
    /// The pronunciation source identifier (always `"cmudict"` for this impl).
    pub source: &'static str,
    /// All candidate pronunciations ordered by dictionary priority.
    ///
    /// Each inner `Vec<CmuPhoneme>` is one pronunciation variant.  Empty when
    /// `status` is [`PronunciationStatus::Missing`] or
    /// [`PronunciationStatus::Guessed`].
    pub candidates: Vec<Vec<CmuPhoneme>>,
    /// How the pronunciation was resolved.
    pub status: PronunciationStatus,
}

/// Stress level for a CMUdict vowel phoneme.
///
/// CMUdict encodes stress by appending a digit to vowel symbols:
/// `0` = unstressed, `1` = primary stress, `2` = secondary stress.
/// Consonant phonemes carry no stress marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stress {
    /// Primary (strongest) stress, encoded as `1` in CMUdict.
    Primary,
    /// Secondary stress, encoded as `2` in CMUdict.
    Secondary,
    /// Explicitly unstressed vowel, encoded as `0` in CMUdict.
    Unstressed,
}

/// A single ARPAbet phoneme parsed from CMUdict, optionally carrying stress.
///
/// For example the CMUdict token `"OW1"` is represented as
/// `CmuPhoneme { base: "OW".into(), stress: Some(Stress::Primary) }`,
/// while the consonant `"K"` becomes
/// `CmuPhoneme { base: "K".into(), stress: None }`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CmuPhoneme {
    /// The base ARPAbet symbol without the stress digit (e.g. `"OW"`, `"K"`).
    pub base: String,
    /// Stress level for vowel phonemes; `None` for consonants.
    pub stress: Option<Stress>,
}

impl CmuPhoneme {
    /// Parse a raw CMUdict token such as `"OW1"`, `"K"`, or `"EY0"` into a
    /// [`CmuPhoneme`].
    pub fn parse(token: &str) -> Self {
        let stress = token.chars().last().and_then(|c| match c {
            '1' => Some(Stress::Primary),
            '2' => Some(Stress::Secondary),
            '0' => Some(Stress::Unstressed),
            _ => None,
        });
        let base = if stress.is_some() {
            // Strip the trailing stress digit.
            token[..token.len() - 1].to_string()
        } else {
            token.to_string()
        };
        Self { base, stress }
    }

    /// Convert this phoneme to a neutral [`Phoneme`] using only the base symbol.
    pub fn to_phoneme(&self) -> Phoneme {
        Phoneme::new(&self.base)
    }
}

/// A fast in-memory CMUdict-backed English pronunciation service.
///
/// The dictionary is indexed by lowercased orthographic keys for
/// case-insensitive lookup.  Each key maps to one or more pronunciation
/// variants; [`lookup`] returns the first (primary) pronunciation.
///
/// # Building
///
/// ```
/// use listenbury::linguistic::CmudictPronouncer;
/// let pronouncer = CmudictPronouncer::bundled();
/// ```
///
/// # Example
///
/// ```
/// use listenbury::linguistic::{CmudictPronouncer, Stress};
/// let pronouncer = CmudictPronouncer::bundled();
/// let phones = pronouncer.lookup("okay").expect("in dictionary");
/// assert_eq!(phones[0].base, "OW");
/// assert_eq!(phones[0].stress, Some(Stress::Secondary));
/// ```
pub struct CmudictPronouncer {
    entries: HashMap<Box<str>, Vec<Vec<CmuPhoneme>>>,
}

impl CmudictPronouncer {
    /// Build a [`CmudictPronouncer`] from the CMUdict data bundled with this
    /// crate.
    pub fn bundled() -> Self {
        Self::from_str(BUNDLED_CMUDICT)
    }

    /// Parse a string in CMUdict format and build the pronunciation index.
    ///
    /// Lines beginning with `;;;` are comments and are ignored.  Blank lines
    /// are also ignored.  Alternate-pronunciation entries such as `WORD(2)` are
    /// stored alongside the primary pronunciation.
    pub fn from_str(data: &str) -> Self {
        let mut entries: HashMap<Box<str>, Vec<Vec<CmuPhoneme>>> = HashMap::new();

        for line in data.lines() {
            let line = line.trim();
            if line.starts_with(";;;") || line.is_empty() {
                continue;
            }

            let mut parts = line.split_ascii_whitespace();
            let word_raw = match parts.next() {
                Some(w) => w,
                None => continue,
            };

            // Strip the alternate-pronunciation index, e.g. `WORD(2)` → `WORD`.
            let word = match word_raw.find('(') {
                Some(idx) => &word_raw[..idx],
                None => word_raw,
            };

            let key: Box<str> = word.to_lowercase().into_boxed_str();

            let phonemes: Vec<CmuPhoneme> = parts.map(CmuPhoneme::parse).collect();
            if phonemes.is_empty() {
                continue;
            }

            entries.entry(key).or_default().push(phonemes);
        }

        Self { entries }
    }

    /// Return the primary (first) pronunciation for `word`, or `None` if the
    /// word is not in the dictionary.
    ///
    /// The lookup is case-insensitive.
    pub fn lookup(&self, word: &str) -> Option<&[CmuPhoneme]> {
        let key = word.to_lowercase();
        self.entries
            .get(key.as_str())
            .and_then(|v| v.first())
            .map(|v| v.as_slice())
    }

    /// Return all pronunciation variants for `word`, or `None` if absent.
    pub fn lookup_all(&self, word: &str) -> Option<&Vec<Vec<CmuPhoneme>>> {
        let key = word.to_lowercase();
        self.entries.get(key.as_str())
    }

    /// Look up `word` and return a [`PronunciationEntry`] that includes **all**
    /// candidate pronunciations, the resolution status, and provenance.
    ///
    /// The lookup strategy is:
    /// 1. Try an exact case-insensitive match first.
    /// 2. If that fails, strip leading/trailing punctuation and retry
    ///    (e.g. `"three,"` → `"three"`).
    /// 3. If still not found, return an entry with status
    ///    [`PronunciationStatus::Missing`] and an empty `candidates` list.
    ///
    /// # Example
    ///
    /// ```
    /// use listenbury::linguistic::{CmudictPronouncer, PronunciationStatus};
    /// let p = CmudictPronouncer::bundled();
    ///
    /// let entry = p.lookup_entry("THREE");
    /// assert_eq!(entry.status, PronunciationStatus::Exact);
    /// assert!(!entry.candidates.is_empty());
    ///
    /// let entry = p.lookup_entry("xyzzyqux");
    /// assert_eq!(entry.status, PronunciationStatus::Missing);
    /// assert!(entry.candidates.is_empty());
    /// ```
    pub fn lookup_entry(&self, word: &str) -> PronunciationEntry {
        // 1. Exact case-insensitive lookup.
        let exact_key = word.to_lowercase();
        if let Some(variants) = self.entries.get(exact_key.as_str()) {
            return PronunciationEntry {
                original: word.to_string(),
                lookup: exact_key,
                source: "cmudict",
                candidates: variants.clone(),
                status: PronunciationStatus::Exact,
            };
        }

        // 2. Normalized lookup: strip leading/trailing non-alphabetic chars,
        //    collapse apostrophes in common contractions.
        let normalized = normalize_for_lookup(word);
        if normalized != exact_key {
            if let Some(variants) = self.entries.get(normalized.as_str()) {
                return PronunciationEntry {
                    original: word.to_string(),
                    lookup: normalized,
                    source: "cmudict",
                    candidates: variants.clone(),
                    status: PronunciationStatus::Normalized,
                };
            }
        }

        // 3. Not found.
        PronunciationEntry {
            original: word.to_string(),
            lookup: if normalized.is_empty() {
                exact_key
            } else {
                normalized
            },
            source: "cmudict",
            candidates: Vec::new(),
            status: PronunciationStatus::Missing,
        }
    }

    /// Return the number of entries in the dictionary.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when the dictionary is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Global lazy-initialized singleton backed by the bundled CMUdict data.
static BUNDLED_INSTANCE: OnceLock<CmudictPronouncer> = OnceLock::new();

/// Return a reference to the process-wide singleton [`CmudictPronouncer`]
/// backed by the bundled dictionary, initializing it on first call.
pub fn bundled() -> &'static CmudictPronouncer {
    BUNDLED_INSTANCE.get_or_init(CmudictPronouncer::bundled)
}

impl OrthographyToPhonemes for CmudictPronouncer {
    fn realize_word(
        &self,
        _variety: &LinguisticVariety,
        word: &OrthographicWord,
    ) -> Result<PhonemeSeq, PhonologyError> {
        let phones = self
            .lookup(&word.text)
            .ok_or_else(|| PhonologyError::UnsupportedWord {
                word: word.text.clone(),
            })?;
        Ok(PhonemeSeq::new(
            phones.iter().map(CmuPhoneme::to_phoneme).collect(),
        ))
    }

    /// Realize free-form text into phoneme text units.
    ///
    /// Tokenization rules:
    /// - Alphabetic characters accumulate into the current word token.
    /// - ASCII whitespace flushes the current word and inserts a
    ///   [`PhonemeTextUnit::WordBoundary`] *between* consecutive words.
    /// - `.`, `,`, `;`, `:`, `!`, `?` flush the current word and insert a
    ///   [`PhonemeTextUnit::PhraseBoundary`].
    /// - All other characters (digits, apostrophes, hyphens, etc.) are
    ///   silently skipped.  Words that straddle such characters are treated as
    ///   separate tokens (e.g. `"it's"` becomes two tokens: `"it"` and `"s"`).
    fn realize_text(
        &self,
        variety: &LinguisticVariety,
        text: &str,
    ) -> Result<PhonemeText, PhonologyError> {
        let mut units: Vec<PhonemeTextUnit> = Vec::new();
        let mut current_word = String::new();
        let mut pending_word_boundary = false;

        let flush_word = |current_word: &mut String,
                          units: &mut Vec<PhonemeTextUnit>,
                          pending: &mut bool|
         -> Result<(), PhonologyError> {
            if current_word.is_empty() {
                return Ok(());
            }
            if *pending {
                units.push(PhonemeTextUnit::WordBoundary);
            }
            let ortho = OrthographicWord::new(current_word.as_str());
            let phonemes = self.realize_word(variety, &ortho)?;
            units.push(PhonemeTextUnit::Word {
                orthography: ortho,
                phonemes,
            });
            current_word.clear();
            *pending = true;
            Ok(())
        };

        for ch in text.chars() {
            if ch.is_alphabetic() {
                current_word.push(ch);
            } else if ch.is_ascii_whitespace() {
                flush_word(&mut current_word, &mut units, &mut pending_word_boundary)?;
            } else if matches!(ch, '.' | ',' | ';' | ':' | '!' | '?') {
                flush_word(&mut current_word, &mut units, &mut pending_word_boundary)?;
                units.push(PhonemeTextUnit::PhraseBoundary);
                pending_word_boundary = false;
            }
        }

        flush_word(&mut current_word, &mut units, &mut pending_word_boundary)?;

        Ok(PhonemeText::new(units))
    }
}

/// The CMUdict data shipped with this crate.
static BUNDLED_CMUDICT: &str = include_str!("../../data/cmudict.dict");

/// Normalize a raw transcript word for CMUdict lookup.
///
/// - Strips leading and trailing non-alphabetic characters (punctuation, etc.).
/// - Lowercases the result.
/// - Preserves internal apostrophes so that contractions like `"it's"` or
///   `"don't"` survive intact for CMUdict entries that include them.
fn normalize_for_lookup(word: &str) -> String {
    // Strip leading non-alpha chars.
    let stripped = word.trim_matches(|c: char| !c.is_alphabetic());
    stripped.to_lowercase()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn pronouncer() -> CmudictPronouncer {
        CmudictPronouncer::bundled()
    }

    fn variety() -> LinguisticVariety {
        use crate::linguistic::variety::Phonology;
        LinguisticVariety::untagged("test", Phonology::new("test"))
    }

    // ------------------------------------------------------------------
    // CmuPhoneme parsing
    // ------------------------------------------------------------------

    #[test]
    fn parses_vowel_with_primary_stress() {
        let p = CmuPhoneme::parse("OW1");
        assert_eq!(p.base, "OW");
        assert_eq!(p.stress, Some(Stress::Primary));
    }

    #[test]
    fn parses_vowel_with_secondary_stress() {
        let p = CmuPhoneme::parse("AH2");
        assert_eq!(p.base, "AH");
        assert_eq!(p.stress, Some(Stress::Secondary));
    }

    #[test]
    fn parses_unstressed_vowel() {
        let p = CmuPhoneme::parse("ER0");
        assert_eq!(p.base, "ER");
        assert_eq!(p.stress, Some(Stress::Unstressed));
    }

    #[test]
    fn parses_consonant_without_stress() {
        let p = CmuPhoneme::parse("K");
        assert_eq!(p.base, "K");
        assert_eq!(p.stress, None);
    }

    #[test]
    fn to_phoneme_uses_base_only() {
        let p = CmuPhoneme::parse("EY1");
        assert_eq!(p.to_phoneme().symbol, "EY");
    }

    // ------------------------------------------------------------------
    // Dictionary parsing
    // ------------------------------------------------------------------

    #[test]
    fn parses_dict_comment_lines() {
        let dict = ";;; This is a comment\nOKAY  OW1 K EY1\n";
        let p = CmudictPronouncer::from_str(dict);
        assert!(p.lookup("okay").is_some());
    }

    #[test]
    fn parses_alternate_pronunciations() {
        let dict = "THE  DH AH0\nTHE(2)  DH AH1\nTHE(3)  DH IY0\n";
        let p = CmudictPronouncer::from_str(dict);
        let all = p.lookup_all("the").expect("found");
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn lookup_is_case_insensitive() {
        let p = pronouncer();
        assert!(p.lookup("OKAY").is_some());
        assert!(p.lookup("Okay").is_some());
        assert!(p.lookup("okay").is_some());
    }

    #[test]
    fn lookup_returns_none_for_unknown_word() {
        let p = pronouncer();
        assert!(p.lookup("xyzzyqux").is_none());
    }

    // ------------------------------------------------------------------
    // Required acceptance-criteria words
    // ------------------------------------------------------------------

    #[test]
    fn lookup_okay_preserves_stress() {
        let p = pronouncer();
        let phones = p.lookup("okay").expect("okay in dictionary");
        // OKAY  OW2 K EY1
        assert_eq!(phones[0].base, "OW");
        assert_eq!(phones[0].stress, Some(Stress::Secondary));
        assert_eq!(phones[1].base, "K");
        assert_eq!(phones[1].stress, None);
        assert_eq!(phones[2].base, "EY");
        assert_eq!(phones[2].stress, Some(Stress::Primary));
    }

    #[test]
    fn lookup_doctor() {
        let p = pronouncer();
        let phones = p.lookup("doctor").expect("doctor in dictionary");
        // DOCTOR  D AA1 K T ER0
        let bases: Vec<&str> = phones.iter().map(|ph| ph.base.as_str()).collect();
        assert_eq!(bases, vec!["D", "AA", "K", "T", "ER"]);
    }

    #[test]
    fn lookup_fitzgerald() {
        let p = pronouncer();
        let phones = p.lookup("fitzgerald").expect("fitzgerald in dictionary");
        // FITZGERALD  F IH0 T S JH EH1 R AH0 L D
        let bases: Vec<&str> = phones.iter().map(|ph| ph.base.as_str()).collect();
        assert_eq!(
            bases,
            vec!["F", "IH", "T", "S", "JH", "EH", "R", "AH", "L", "D"]
        );
    }

    #[test]
    fn lookup_xylophone() {
        let p = pronouncer();
        let phones = p.lookup("xylophone").expect("xylophone in dictionary");
        // XYLOPHONE  Z AY1 L AH0 F OW2 N
        let bases: Vec<&str> = phones.iter().map(|ph| ph.base.as_str()).collect();
        assert_eq!(bases, vec!["Z", "AY", "L", "AH", "F", "OW", "N"]);
    }

    #[test]
    fn xylophone_secondary_stress_preserved() {
        let p = pronouncer();
        let phones = p.lookup("xylophone").expect("xylophone in dictionary");
        // OW2 = secondary stress
        let ow = phones
            .iter()
            .find(|ph| ph.base == "OW")
            .expect("OW phoneme");
        assert_eq!(ow.stress, Some(Stress::Secondary));
    }

    // ------------------------------------------------------------------
    // OrthographyToPhonemes interface
    // ------------------------------------------------------------------

    #[test]
    fn realize_word_returns_base_phoneme_seq() {
        let p = pronouncer();
        let v = variety();
        let word = OrthographicWord::new("okay");
        let seq = p.realize_word(&v, &word).expect("realize okay");
        let symbols: Vec<&str> = seq.phonemes.iter().map(|ph| ph.symbol.as_str()).collect();
        assert_eq!(symbols, vec!["OW", "K", "EY"]);
    }

    #[test]
    fn realize_word_unknown_returns_error() {
        let p = pronouncer();
        let v = variety();
        let word = OrthographicWord::new("xyzzyqux");
        let err = p.realize_word(&v, &word).unwrap_err();
        assert!(matches!(err, PhonologyError::UnsupportedWord { .. }));
    }

    #[test]
    fn realize_text_word_boundaries() {
        let p = pronouncer();
        let v = variety();
        let text = p.realize_text(&v, "okay doctor").expect("realize text");
        assert_eq!(text.units.len(), 3);
        assert!(
            matches!(&text.units[0], PhonemeTextUnit::Word { orthography, .. } if orthography.text == "okay")
        );
        assert_eq!(text.units[1], PhonemeTextUnit::WordBoundary);
        assert!(
            matches!(&text.units[2], PhonemeTextUnit::Word { orthography, .. } if orthography.text == "doctor")
        );
    }

    #[test]
    fn realize_text_phrase_boundary() {
        let p = pronouncer();
        let v = variety();
        let text = p.realize_text(&v, "okay, doctor").expect("realize text");
        // okay + PhraseBoundary + doctor
        assert_eq!(text.units.len(), 3);
        assert!(matches!(&text.units[0], PhonemeTextUnit::Word { .. }));
        assert_eq!(text.units[1], PhonemeTextUnit::PhraseBoundary);
        assert!(matches!(&text.units[2], PhonemeTextUnit::Word { .. }));
    }

    #[test]
    fn bundled_dict_has_reasonable_size() {
        let p = pronouncer();
        // The bundled dictionary should have at least 100 entries.
        assert!(
            p.len() >= 100,
            "expected at least 100 entries, got {}",
            p.len()
        );
    }

    // ------------------------------------------------------------------
    // PronunciationEntry / lookup_entry
    // ------------------------------------------------------------------

    #[test]
    fn lookup_entry_exact_match() {
        let p = pronouncer();
        let entry = p.lookup_entry("three");
        assert_eq!(entry.status, PronunciationStatus::Exact);
        assert_eq!(entry.original, "three");
        assert_eq!(entry.source, "cmudict");
        assert!(
            !entry.candidates.is_empty(),
            "should have at least one pronunciation"
        );
    }

    #[test]
    fn lookup_entry_case_insensitive_is_exact() {
        let p = pronouncer();
        let entry = p.lookup_entry("THREE");
        assert_eq!(entry.status, PronunciationStatus::Exact);
        assert!(!entry.candidates.is_empty());
    }

    #[test]
    fn lookup_entry_returns_all_candidates() {
        // "the" has three variants in CMUdict
        let dict = "THE  DH AH0\nTHE(2)  DH AH1\nTHE(3)  DH IY0\n";
        let p = CmudictPronouncer::from_str(dict);
        let entry = p.lookup_entry("the");
        assert_eq!(entry.candidates.len(), 3);
        assert_eq!(entry.status, PronunciationStatus::Exact);
    }

    #[test]
    fn lookup_entry_normalized_strips_trailing_punctuation() {
        let p = pronouncer();
        // "three," should normalize to "three" and still find a result.
        let entry = p.lookup_entry("three,");
        assert_eq!(entry.status, PronunciationStatus::Normalized);
        assert!(!entry.candidates.is_empty());
        assert_eq!(entry.original, "three,");
        assert_eq!(entry.lookup, "three");
    }

    #[test]
    fn lookup_entry_normalized_strips_leading_punctuation() {
        let p = pronouncer();
        let entry = p.lookup_entry("\"hello");
        assert_eq!(entry.status, PronunciationStatus::Normalized);
        assert_eq!(entry.lookup, "hello");
        assert!(!entry.candidates.is_empty());
    }

    #[test]
    fn lookup_entry_unknown_word_is_missing() {
        let p = pronouncer();
        let entry = p.lookup_entry("xyzzyqux");
        assert_eq!(entry.status, PronunciationStatus::Missing);
        assert!(entry.candidates.is_empty());
        assert_eq!(entry.source, "cmudict");
    }

    #[test]
    fn lookup_entry_phonemes_match_lookup() {
        let p = pronouncer();
        let entry = p.lookup_entry("doctor");
        assert_eq!(entry.status, PronunciationStatus::Exact);
        let bases: Vec<&str> = entry.candidates[0]
            .iter()
            .map(|ph| ph.base.as_str())
            .collect();
        assert_eq!(bases, vec!["D", "AA", "K", "T", "ER"]);
    }

    #[test]
    fn normalize_for_lookup_strips_both_ends() {
        assert_eq!(normalize_for_lookup("\"hello!\""), "hello");
    }

    #[test]
    fn normalize_for_lookup_lowercases() {
        assert_eq!(normalize_for_lookup("Hello"), "hello");
    }
}
