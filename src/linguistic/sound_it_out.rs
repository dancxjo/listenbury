use crate::linguistic::{
    orthography::OrthographicWord,
    phoneme::{Phoneme, PhonemeSeq, PhonemeText, PhonemeTextUnit},
    pronounce::{OrthographyToPhonemes, PhonologyError},
    variety::{LinguisticVariety, VarietyTag},
};

/// Optional phonological environment constraint for a grapheme rule.
///
/// Both fields are currently informational; environment-sensitive dispatch is not
/// yet implemented in [`SoundItOutPronouncer`] but the field is available for
/// richer rule sets in the future.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Environment {
    /// Grapheme string that must immediately precede the target (if any).
    pub preceding: Option<String>,
    /// Grapheme string that must immediately follow the target (if any).
    pub following: Option<String>,
}

impl Environment {
    pub fn new(preceding: Option<impl Into<String>>, following: Option<impl Into<String>>) -> Self {
        Self {
            preceding: preceding.map(Into::into),
            following: following.map(Into::into),
        }
    }
}

/// A single grapheme-to-phoneme mapping rule.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphemeRule {
    /// The orthographic grapheme (one or more characters) to match.
    pub grapheme: String,
    /// The phoneme sequence to produce.
    pub phonemes: PhonemeSeq,
    /// Optional phonological environment constraint (reserved for future use).
    pub environment: Option<Environment>,
    /// Higher-priority rules are preferred when multiple rules have the same
    /// grapheme length.
    pub priority: u16,
}

impl GraphemeRule {
    pub fn new(grapheme: impl Into<String>, phonemes: PhonemeSeq) -> Self {
        Self {
            grapheme: grapheme.into(),
            phonemes,
            environment: None,
            priority: 0,
        }
    }

    pub fn with_priority(mut self, priority: u16) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_environment(mut self, env: Environment) -> Self {
        self.environment = Some(env);
        self
    }
}

/// A collection of grapheme-to-phoneme rules for a particular linguistic variety.
#[derive(Debug, Clone)]
pub struct SoundItOutRules {
    pub variety: VarietyTag,
    pub mappings: Vec<GraphemeRule>,
}

impl SoundItOutRules {
    pub fn new(variety: VarietyTag, mappings: Vec<GraphemeRule>) -> Self {
        Self { variety, mappings }
    }

    /// Build rules for a transparent orthography where each listed grapheme is
    /// itself the phoneme symbol.
    pub fn one_to_one<I, S>(variety: VarietyTag, graphemes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::new(
            variety,
            graphemes
                .into_iter()
                .map(|grapheme| {
                    let grapheme = grapheme.into();
                    GraphemeRule::new(
                        grapheme.clone(),
                        PhonemeSeq::new(vec![Phoneme::new(grapheme)]),
                    )
                })
                .collect(),
        )
    }

    /// Build rules for a transparent orthography where each listed grapheme
    /// maps to exactly one explicit phoneme symbol.
    pub fn one_to_one_with_symbols<I, G, P>(variety: VarietyTag, mappings: I) -> Self
    where
        I: IntoIterator<Item = (G, P)>,
        G: Into<String>,
        P: Into<String>,
    {
        Self::new(
            variety,
            mappings
                .into_iter()
                .map(|(grapheme, phoneme)| {
                    GraphemeRule::new(
                        grapheme,
                        PhonemeSeq::new(vec![Phoneme::new(phoneme.into())]),
                    )
                })
                .collect(),
        )
    }

    /// Pre-built rule set for Esperanto.
    ///
    /// Esperanto has a perfectly regular, transparent orthography: every letter
    /// maps to exactly one phoneme.  Special letters (ĉ ĝ ĥ ĵ ŝ ŭ) are single
    /// Unicode code-points and are handled before the plain ASCII fallbacks.
    pub fn esperanto() -> Self {
        Self::one_to_one_with_symbols(
            VarietyTag::new("eo"),
            vec![
                // Special Esperanto letters (digraph-like phonemes as single codepoints)
                ("ĉ", "t͡ʃ"),
                ("ĝ", "d͡ʒ"),
                ("ĥ", "x"),
                ("ĵ", "ʒ"),
                ("ŝ", "ʃ"),
                ("ŭ", "w"),
                // Vowels
                ("a", "a"),
                ("e", "e"),
                ("i", "i"),
                ("o", "o"),
                ("u", "u"),
                // Consonants
                ("b", "b"),
                ("c", "t͡s"),
                ("d", "d"),
                ("f", "f"),
                ("g", "g"),
                ("h", "h"),
                ("j", "j"),
                ("k", "k"),
                ("l", "l"),
                ("m", "m"),
                ("n", "n"),
                ("p", "p"),
                ("r", "r"),
                ("s", "s"),
                ("t", "t"),
                ("v", "v"),
                ("z", "z"),
            ],
        )
    }

    /// Pre-built rule set for Classical Latin (restored/classical pronunciation).
    ///
    /// Key features:
    /// - `qu` → /kʷ/ (longer match, listed before `q`)
    /// - `ae`, `oe`, `au` → diphthongs
    /// - `c` → /k/ (always velar in Classical Latin)
    /// - `v` → /w/
    /// - `x` → /k/ + /s/
    pub fn classical_latin() -> Self {
        Self::new(
            VarietyTag::new("la-class"),
            vec![
                // Multi-grapheme rules first (longest-match will select them anyway,
                // but listing them first is a useful convention for readability).
                rule("qu", &["kʷ"]),
                rule("ae", &["ae̯"]),
                rule("oe", &["oe̯"]),
                rule("au", &["au̯"]),
                rule("æ", &["ae̯"]),
                rule("œ", &["oe̯"]),
                // Long vowels marked with macrons.
                rule("ā", &["aː"]),
                rule("ē", &["eː"]),
                rule("ī", &["iː"]),
                rule("ō", &["oː"]),
                rule("ū", &["uː"]),
                rule("ȳ", &["yː"]),
                // Vowels
                rule("a", &["a"]),
                rule("e", &["e"]),
                rule("i", &["i"]),
                rule("o", &["o"]),
                rule("u", &["u"]),
                rule("y", &["y"]),
                // Consonants
                rule("b", &["b"]),
                rule("c", &["k"]),
                rule("d", &["d"]),
                rule("f", &["f"]),
                rule("g", &["g"]),
                rule("h", &["h"]),
                rule("j", &["j"]),
                rule("k", &["k"]),
                rule("l", &["l"]),
                rule("m", &["m"]),
                rule("n", &["n"]),
                rule("p", &["p"]),
                rule("q", &["k"]),
                rule("r", &["r"]),
                rule("s", &["s"]),
                rule("t", &["t"]),
                rule("v", &["w"]),
                rule("x", &["k", "s"]),
                rule("z", &["z"]),
            ],
        )
    }

    /// Pre-built rule set for modern Turkish.
    ///
    /// Turkish is close to phonemic at the letter level.  The rule for `ğ`
    /// uses /ɰ/ as a deterministic approximation; in real Turkish it often
    /// lengthens or smooths adjacent vowels rather than surfacing as a stable
    /// consonant. The combining dot rule also tolerates decomposed dotted-i
    /// input.
    pub fn turkish() -> Self {
        let mut rules = Self::one_to_one_with_symbols(
            VarietyTag::new("tr"),
            vec![
                ("a", "a"),
                ("b", "b"),
                ("c", "d͡ʒ"),
                ("ç", "t͡ʃ"),
                ("d", "d"),
                ("e", "e"),
                ("f", "f"),
                ("g", "g"),
                ("ğ", "ɰ"),
                ("h", "h"),
                ("ı", "ɯ"),
                ("i", "i"),
                ("j", "ʒ"),
                ("k", "k"),
                ("l", "l"),
                ("m", "m"),
                ("n", "n"),
                ("o", "o"),
                ("ö", "ø"),
                ("p", "p"),
                ("r", "r"),
                ("s", "s"),
                ("ş", "ʃ"),
                ("t", "t"),
                ("u", "u"),
                ("ü", "y"),
                ("v", "v"),
                ("y", "j"),
                ("z", "z"),
            ],
        );
        rules.mappings.push(rule("'", &[]));
        rules.mappings.push(rule("\u{307}", &[]));
        rules
    }

    /// Approximate English fallback rules using the ARPAbet symbols already
    /// used by CMUdict and the Riper path.
    ///
    /// This is intentionally a fallback, not a replacement for CMUdict. It is
    /// deterministic and broad enough for unknown words, acronyms, names, and
    /// misspellings to remain pronounceable when no dictionary entry exists.
    pub fn english_arpabet_fallback() -> Self {
        Self::new(
            VarietyTag::new("en-US-fallback"),
            vec![
                rule("tion", &["SH", "AH", "N"]),
                rule("sion", &["ZH", "AH", "N"]),
                rule("ough", &["OW"]),
                rule("eigh", &["EY"]),
                rule("tch", &["CH"]),
                rule("kn", &["N"]),
                rule("wr", &["R"]),
                rule("wh", &["W"]),
                rule("qu", &["K", "W"]),
                rule("ck", &["K"]),
                rule("ch", &["CH"]),
                rule("sh", &["SH"]),
                rule("th", &["TH"]),
                rule("ph", &["F"]),
                rule("ng", &["NG"]),
                rule("ee", &["IY"]),
                rule("ea", &["IY"]),
                rule("oo", &["UW"]),
                rule("ou", &["AW"]),
                rule("ow", &["OW"]),
                rule("ai", &["EY"]),
                rule("ay", &["EY"]),
                rule("oa", &["OW"]),
                rule("oi", &["OY"]),
                rule("oy", &["OY"]),
                rule("au", &["AO"]),
                rule("aw", &["AO"]),
                rule("a", &["AH"]),
                rule("b", &["B"]),
                rule("c", &["K"]),
                rule("d", &["D"]),
                rule("e", &["EH"]),
                rule("f", &["F"]),
                rule("g", &["G"]),
                rule("h", &["HH"]),
                rule("i", &["IH"]),
                rule("j", &["JH"]),
                rule("k", &["K"]),
                rule("l", &["L"]),
                rule("m", &["M"]),
                rule("n", &["N"]),
                rule("o", &["OW"]),
                rule("p", &["P"]),
                rule("q", &["K"]),
                rule("r", &["R"]),
                rule("s", &["S"]),
                rule("t", &["T"]),
                rule("u", &["AH"]),
                rule("v", &["V"]),
                rule("w", &["W"]),
                rule("x", &["K", "S"]),
                rule("y", &["IY"]),
                rule("z", &["Z"]),
                rule("'", &[]),
            ],
        )
    }
}

/// A deterministic grapheme-to-phoneme pronouncer based on declarative rules.
///
/// The pronouncer works by scanning normalized input left-to-right and greedily
/// selecting the rule whose `grapheme` is the **longest** prefix of the
/// remaining text. When two rules match a grapheme of equal length the one with
/// the higher `priority` value is chosen.
pub struct SoundItOutPronouncer {
    rules: SoundItOutRules,
}

impl SoundItOutPronouncer {
    pub fn new(rules: SoundItOutRules) -> Self {
        Self { rules }
    }

    /// Apply the grapheme rules to a single word string, returning the phoneme
    /// sequence. The input is case-normalized before matching.
    fn apply_rules_to_word(&self, text: &str) -> Result<PhonemeSeq, PhonologyError> {
        let normalized = self.normalize_word(text);
        let mut remaining = normalized.as_str();
        let mut phonemes: Vec<Phoneme> = Vec::new();

        while !remaining.is_empty() {
            // Choose the rule whose grapheme is the longest prefix of `remaining`.
            // Ties in length are broken by `priority` (higher wins).
            let best = self
                .rules
                .mappings
                .iter()
                .filter(|r| remaining.starts_with(r.grapheme.as_str()))
                .max_by_key(|r| (r.grapheme.chars().count(), r.priority));

            match best {
                Some(rule) => {
                    phonemes.extend(rule.phonemes.phonemes.iter().cloned());
                    // Advance by the byte length of the matched grapheme.
                    remaining = &remaining[rule.grapheme.len()..];
                }
                None => {
                    return Err(PhonologyError::UnsupportedWord {
                        word: text.to_string(),
                    });
                }
            }
        }

        Ok(PhonemeSeq::new(phonemes))
    }

    fn normalize_word(&self, text: &str) -> String {
        if self.rules.variety.0 == "tr" {
            turkish_lowercase(text)
        } else {
            text.to_lowercase()
        }
    }
}

impl OrthographyToPhonemes for SoundItOutPronouncer {
    fn realize_word(
        &self,
        _variety: &LinguisticVariety,
        word: &OrthographicWord,
    ) -> Result<PhonemeSeq, PhonologyError> {
        self.apply_rules_to_word(&word.text)
    }

    /// Realize free-form text into phoneme text units.
    ///
    /// Tokenization rules:
    /// - Alphabetic characters accumulate into the current word token.
    /// - ASCII whitespace flushes the current word and inserts a
    ///   [`PhonemeTextUnit::WordBoundary`] *between* consecutive words.
    /// - `.`, `,`, `;`, `:`, `!`, `?` flush the current word and insert a
    ///   [`PhonemeTextUnit::PhraseBoundary`].
    /// - All other characters are silently skipped (e.g. apostrophes, hyphens).
    fn realize_text(
        &self,
        _variety: &LinguisticVariety,
        text: &str,
    ) -> Result<PhonemeText, PhonologyError> {
        let mut units: Vec<PhonemeTextUnit> = Vec::new();
        let mut current_word = String::new();
        // `pending_word_boundary` tracks whether we should insert a WordBoundary
        // before the next word (i.e. we already emitted at least one word and the
        // last separator was whitespace, not a phrase boundary).
        let mut pending_word_boundary = false;

        let flush_word = |current_word: &mut String,
                          units: &mut Vec<PhonemeTextUnit>,
                          pending: &mut bool,
                          pronouncer: &SoundItOutPronouncer|
         -> Result<(), PhonologyError> {
            if current_word.is_empty() {
                return Ok(());
            }
            if *pending {
                units.push(PhonemeTextUnit::WordBoundary);
            }
            let ortho = OrthographicWord::new(current_word.as_str());
            let phonemes = pronouncer.apply_rules_to_word(current_word)?;
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
                flush_word(
                    &mut current_word,
                    &mut units,
                    &mut pending_word_boundary,
                    self,
                )?;
            } else if matches!(ch, '.' | ',' | ';' | ':' | '!' | '?') {
                flush_word(
                    &mut current_word,
                    &mut units,
                    &mut pending_word_boundary,
                    self,
                )?;
                units.push(PhonemeTextUnit::PhraseBoundary);
                // After a phrase boundary the next word should not be preceded by
                // an additional WordBoundary.
                pending_word_boundary = false;
            }
            // All other characters are silently ignored.
        }

        // Flush any remaining word at end of input.
        flush_word(
            &mut current_word,
            &mut units,
            &mut pending_word_boundary,
            self,
        )?;

        Ok(PhonemeText::new(units))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Convenience constructor for a [`GraphemeRule`] from a grapheme string and a
/// slice of phoneme symbol strings.
fn rule(grapheme: &str, symbols: &[&str]) -> GraphemeRule {
    GraphemeRule::new(
        grapheme,
        PhonemeSeq::new(symbols.iter().map(|s| Phoneme::new(*s)).collect()),
    )
}

fn turkish_lowercase(text: &str) -> String {
    let mut lowered = String::new();
    for ch in text.chars() {
        match ch {
            'I' => lowered.push('ı'),
            'İ' => lowered.push('i'),
            _ => lowered.extend(ch.to_lowercase()),
        }
    }
    lowered
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn esperanto() -> SoundItOutPronouncer {
        SoundItOutPronouncer::new(SoundItOutRules::esperanto())
    }

    fn latin() -> SoundItOutPronouncer {
        SoundItOutPronouncer::new(SoundItOutRules::classical_latin())
    }

    fn english_fallback() -> SoundItOutPronouncer {
        SoundItOutPronouncer::new(SoundItOutRules::english_arpabet_fallback())
    }

    fn turkish() -> SoundItOutPronouncer {
        SoundItOutPronouncer::new(SoundItOutRules::turkish())
    }

    fn variety() -> LinguisticVariety {
        use crate::linguistic::variety::Phonology;
        LinguisticVariety::untagged("test", Phonology::new("test"))
    }

    fn symbols(seq: &PhonemeSeq) -> Vec<&str> {
        seq.phonemes.iter().map(|p| p.symbol.as_str()).collect()
    }

    // ------------------------------------------------------------------
    // Esperanto
    // ------------------------------------------------------------------

    #[test]
    fn esperanto_saluton() {
        let p = esperanto();
        let word = OrthographicWord::new("saluton");
        let seq = p.realize_word(&variety(), &word).unwrap();
        assert_eq!(symbols(&seq), vec!["s", "a", "l", "u", "t", "o", "n"]);
    }

    #[test]
    fn esperanto_cambro() {
        let p = esperanto();
        let word = OrthographicWord::new("ĉambro");
        let seq = p.realize_word(&variety(), &word).unwrap();
        assert_eq!(symbols(&seq), vec!["t͡ʃ", "a", "m", "b", "r", "o"]);
    }

    #[test]
    fn esperanto_uppercase_normalized() {
        let p = esperanto();
        let word = OrthographicWord::new("SALUTON");
        let seq = p.realize_word(&variety(), &word).unwrap();
        assert_eq!(symbols(&seq), vec!["s", "a", "l", "u", "t", "o", "n"]);
    }

    #[test]
    fn esperanto_realize_text_word_boundaries() {
        let p = esperanto();
        let text = p.realize_text(&variety(), "saluton ĉambro").unwrap();
        assert_eq!(text.units.len(), 3);
        assert!(matches!(
            &text.units[0],
            PhonemeTextUnit::Word { orthography, .. } if orthography.text == "saluton"
        ));
        assert_eq!(text.units[1], PhonemeTextUnit::WordBoundary);
        assert!(matches!(
            &text.units[2],
            PhonemeTextUnit::Word { orthography, .. } if orthography.text == "ĉambro"
        ));
    }

    #[test]
    fn esperanto_realize_text_phrase_boundary() {
        let p = esperanto();
        let text = p.realize_text(&variety(), "saluton, ĉambro").unwrap();
        // Word, PhraseBoundary, Word
        assert_eq!(text.units.len(), 3);
        assert!(matches!(&text.units[0], PhonemeTextUnit::Word { .. }));
        assert_eq!(text.units[1], PhonemeTextUnit::PhraseBoundary);
        assert!(matches!(&text.units[2], PhonemeTextUnit::Word { .. }));
    }

    #[test]
    fn esperanto_c_is_affricate() {
        let p = esperanto();
        let word = OrthographicWord::new("celo");
        let seq = p.realize_word(&variety(), &word).unwrap();
        assert_eq!(symbols(&seq), vec!["t͡s", "e", "l", "o"]);
    }

    #[test]
    fn one_to_one_rules_map_graphemes_to_themselves() {
        let rules = SoundItOutRules::one_to_one(VarietyTag::new("test"), ["a", "ĉ"]);
        let p = SoundItOutPronouncer::new(rules);
        let word = OrthographicWord::new("aĉ");
        let seq = p.realize_word(&variety(), &word).unwrap();
        assert_eq!(symbols(&seq), vec!["a", "ĉ"]);
    }

    #[test]
    fn one_to_one_rules_can_use_explicit_symbols() {
        let rules = SoundItOutRules::one_to_one_with_symbols(
            VarietyTag::new("test"),
            [("c", "t͡s"), ("a", "a")],
        );
        let p = SoundItOutPronouncer::new(rules);
        let word = OrthographicWord::new("ca");
        let seq = p.realize_word(&variety(), &word).unwrap();
        assert_eq!(symbols(&seq), vec!["t͡s", "a"]);
    }

    // ------------------------------------------------------------------
    // Classical Latin
    // ------------------------------------------------------------------

    #[test]
    fn latin_vita() {
        let p = latin();
        let word = OrthographicWord::new("vita");
        let seq = p.realize_word(&variety(), &word).unwrap();
        assert_eq!(symbols(&seq), vec!["w", "i", "t", "a"]);
    }

    #[test]
    fn english_fallback_sounds_out_unknown_tokens_as_arpabet() {
        let p = english_fallback();
        let word = OrthographicWord::new("mbrola");
        let seq = p.realize_word(&variety(), &word).unwrap();
        assert_eq!(symbols(&seq), vec!["M", "B", "R", "OW", "L", "AH"]);
    }

    #[test]
    fn english_fallback_handles_common_digraphs_and_apostrophes() {
        let p = english_fallback();
        let word = OrthographicWord::new("quoth's");
        let seq = p.realize_word(&variety(), &word).unwrap();
        assert_eq!(symbols(&seq), vec!["K", "W", "OW", "TH", "S"]);
    }

    #[test]
    fn latin_quod_longest_match() {
        let p = latin();
        let word = OrthographicWord::new("quod");
        let seq = p.realize_word(&variety(), &word).unwrap();
        // `qu` must be selected over `q` alone (longest-match).
        assert_eq!(symbols(&seq), vec!["kʷ", "o", "d"]);
    }

    #[test]
    fn latin_x_is_two_phonemes() {
        let p = latin();
        let word = OrthographicWord::new("rex");
        let seq = p.realize_word(&variety(), &word).unwrap();
        assert_eq!(symbols(&seq), vec!["r", "e", "k", "s"]);
    }

    #[test]
    fn latin_ae_diphthong() {
        let p = latin();
        // "aetas" starts with the "ae" diphthong.
        let word = OrthographicWord::new("aetas");
        let seq = p.realize_word(&variety(), &word).unwrap();
        // "ae" should be matched as a single diphthong, not two separate vowels.
        assert_eq!(symbols(&seq), vec!["ae̯", "t", "a", "s"]);
    }

    #[test]
    fn latin_macrons_mark_long_vowels() {
        let p = latin();
        let word = OrthographicWord::new("Rōmānī");
        let seq = p.realize_word(&variety(), &word).unwrap();
        assert_eq!(symbols(&seq), vec!["r", "oː", "m", "aː", "n", "iː"]);
    }

    #[test]
    fn latin_ligature_diphthongs_are_supported() {
        let p = latin();
        let word = OrthographicWord::new("cælum");
        let seq = p.realize_word(&variety(), &word).unwrap();
        assert_eq!(symbols(&seq), vec!["k", "ae̯", "l", "u", "m"]);
    }

    #[test]
    fn latin_realize_text_multi_word() {
        let p = latin();
        let text = p.realize_text(&variety(), "vita brevis").unwrap();
        assert_eq!(text.units.len(), 3);
        assert!(matches!(&text.units[0], PhonemeTextUnit::Word { .. }));
        assert_eq!(text.units[1], PhonemeTextUnit::WordBoundary);
        assert!(matches!(&text.units[2], PhonemeTextUnit::Word { .. }));
    }

    // ------------------------------------------------------------------
    // Turkish
    // ------------------------------------------------------------------

    #[test]
    fn turkish_maps_diacritic_letters() {
        let p = turkish();
        let word = OrthographicWord::new("çığ");
        let seq = p.realize_word(&variety(), &word).unwrap();
        assert_eq!(symbols(&seq), vec!["t͡ʃ", "ɯ", "ɰ"]);
    }

    #[test]
    fn turkish_uppercase_dotted_i_normalized() {
        let p = turkish();
        let word = OrthographicWord::new("İstanbul");
        let seq = p.realize_word(&variety(), &word).unwrap();
        assert_eq!(symbols(&seq), vec!["i", "s", "t", "a", "n", "b", "u", "l"]);
    }

    #[test]
    fn turkish_uppercase_dotless_i_normalized() {
        let p = turkish();
        let word = OrthographicWord::new("IĞDIR");
        let seq = p.realize_word(&variety(), &word).unwrap();
        assert_eq!(symbols(&seq), vec!["ɯ", "ɰ", "d", "ɯ", "r"]);
    }

    #[test]
    fn turkish_apostrophe_is_ignored_inside_words() {
        let p = turkish();
        let word = OrthographicWord::new("Ankara'da");
        let seq = p.realize_word(&variety(), &word).unwrap();
        assert_eq!(symbols(&seq), vec!["a", "n", "k", "a", "r", "a", "d", "a"]);
    }

    // ------------------------------------------------------------------
    // Longest-match generic behaviour
    // ------------------------------------------------------------------

    #[test]
    fn longest_match_preferred_over_shorter() {
        // Build a custom pronouncer where "qu" and "q" are both present.
        // Longest-match should prefer "qu".
        let rules = SoundItOutRules::new(
            VarietyTag::new("test"),
            vec![rule("q", &["k"]), rule("qu", &["kʷ"]), rule("u", &["u"])],
        );
        let p = SoundItOutPronouncer::new(rules);
        let word = OrthographicWord::new("qu");
        let seq = p.realize_word(&variety(), &word).unwrap();
        assert_eq!(symbols(&seq), vec!["kʷ"]);
    }

    #[test]
    fn unsupported_grapheme_returns_error() {
        // Build a minimal pronouncer that only knows "a".
        let rules = SoundItOutRules::new(VarietyTag::new("test"), vec![rule("a", &["a"])]);
        let p = SoundItOutPronouncer::new(rules);
        let result = p.realize_word(&variety(), &OrthographicWord::new("b"));
        assert!(matches!(
            result,
            Err(PhonologyError::UnsupportedWord { .. })
        ));
    }
}
