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
    pub fn new(
        preceding: Option<impl Into<String>>,
        following: Option<impl Into<String>>,
    ) -> Self {
        Self {
            preceding: preceding.map(Into::into),
            following: following.map(Into::into),
        }
    }
}

/// A single grapheme-to-phoneme mapping rule.
#[derive(Debug, Clone, PartialEq, Eq)]
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

    /// Pre-built rule set for Esperanto.
    ///
    /// Esperanto has a perfectly regular, transparent orthography: every letter
    /// maps to exactly one phoneme.  Special letters (ĉ ĝ ĥ ĵ ŝ ŭ) are single
    /// Unicode code-points and are handled before the plain ASCII fallbacks.
    pub fn esperanto() -> Self {
        Self::new(
            VarietyTag::new("eo"),
            vec![
                // Special Esperanto letters (digraph-like phonemes as single codepoints)
                rule("ĉ", &["t͡ʃ"]),
                rule("ĝ", &["d͡ʒ"]),
                rule("ĥ", &["x"]),
                rule("ĵ", &["ʒ"]),
                rule("ŝ", &["ʃ"]),
                rule("ŭ", &["w"]),
                // Vowels
                rule("a", &["a"]),
                rule("e", &["e"]),
                rule("i", &["i"]),
                rule("o", &["o"]),
                rule("u", &["u"]),
                // Consonants
                rule("b", &["b"]),
                rule("c", &["t͡s"]),
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
                rule("r", &["r"]),
                rule("s", &["s"]),
                rule("t", &["t"]),
                rule("v", &["v"]),
                rule("z", &["z"]),
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
}

/// A deterministic grapheme-to-phoneme pronouncer based on declarative rules.
///
/// The pronouncer works by scanning the (lowercased) input left-to-right and
/// greedily selecting the rule whose `grapheme` is the **longest** prefix of the
/// remaining text.  When two rules match a grapheme of equal length the one with
/// the higher `priority` value is chosen.
pub struct SoundItOutPronouncer {
    rules: SoundItOutRules,
}

impl SoundItOutPronouncer {
    pub fn new(rules: SoundItOutRules) -> Self {
        Self { rules }
    }

    /// Apply the grapheme rules to a single word string, returning the phoneme
    /// sequence.  The input is lowercased before matching.
    fn apply_rules_to_word(&self, text: &str) -> Result<PhonemeSeq, PhonologyError> {
        let normalized = text.to_lowercase();
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

        let flush_word =
            |current_word: &mut String,
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
                flush_word(&mut current_word, &mut units, &mut pending_word_boundary, self)?;
            } else if matches!(ch, '.' | ',' | ';' | ':' | '!' | '?') {
                flush_word(&mut current_word, &mut units, &mut pending_word_boundary, self)?;
                units.push(PhonemeTextUnit::PhraseBoundary);
                // After a phrase boundary the next word should not be preceded by
                // an additional WordBoundary.
                pending_word_boundary = false;
            }
            // All other characters are silently ignored.
        }

        // Flush any remaining word at end of input.
        flush_word(&mut current_word, &mut units, &mut pending_word_boundary, self)?;

        Ok(PhonemeText::new(units))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Convenience constructor for a [`GraphemeRule`] from a grapheme string and a
/// slice of IPA symbol strings.
fn rule(grapheme: &str, symbols: &[&str]) -> GraphemeRule {
    GraphemeRule::new(
        grapheme,
        PhonemeSeq::new(symbols.iter().map(|s| Phoneme::new(*s)).collect()),
    )
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
    fn latin_realize_text_multi_word() {
        let p = latin();
        let text = p.realize_text(&variety(), "vita brevis").unwrap();
        assert_eq!(text.units.len(), 3);
        assert!(matches!(&text.units[0], PhonemeTextUnit::Word { .. }));
        assert_eq!(text.units[1], PhonemeTextUnit::WordBoundary);
        assert!(matches!(&text.units[2], PhonemeTextUnit::Word { .. }));
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
        assert!(matches!(result, Err(PhonologyError::UnsupportedWord { .. })));
    }
}
