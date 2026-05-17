use anyhow::Result;
use thiserror::Error;

use crate::mouth::piper_native::phoneme::{PiperPhoneme, PiperPhonemeSequence};
use crate::mouth::piper_native::text::{
    NormalizedToken, ProsodyBoundaryHint, ProsodyCommitment, TextNormalizationError, TextNormalizer,
};

pub trait GraphemeToPhoneme {
    fn phonemize(&self, text: &str) -> Result<PiperPhonemeSequence>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhonemizedUnit {
    pub phonemes: PiperPhonemeSequence,
    pub length_hints: Vec<PhoneLengthHint>,
    pub boundary: ProsodyBoundaryHint,
    pub commitment: ProsodyCommitment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhoneLengthClass {
    Short,
    Medium,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhoneLengthHint {
    pub symbol: String,
    pub class: PhoneLengthClass,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum G2pError {
    #[error(transparent)]
    Normalization(#[from] TextNormalizationError),
    #[error("unsupported word `{word}` for native Piper simple English G2P")]
    UnsupportedWord { word: String },
    #[error("unsupported initial `{initial}` for native Piper simple English G2P")]
    UnsupportedInitial { initial: char },
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SimpleEnglishG2p {
    normalizer: TextNormalizer,
}

impl SimpleEnglishG2p {
    pub fn phonemize_unit(&self, text: &str) -> std::result::Result<PhonemizedUnit, G2pError> {
        let normalized = self.normalizer.normalize(text)?;
        let mut symbols = Vec::new();

        let pronounceable_count = normalized
            .tokens
            .iter()
            .filter(|token| !matches!(token, NormalizedToken::PhraseBreak))
            .count();
        let mut emitted_pronounceable = 0usize;

        for token in &normalized.tokens {
            match token {
                NormalizedToken::Word(word) => {
                    let word_symbols = word_to_phones(word)
                        .ok_or_else(|| G2pError::UnsupportedWord { word: word.clone() })?;
                    symbols.extend(word_symbols.iter().copied().map(String::from));
                    emitted_pronounceable += 1;
                    if emitted_pronounceable < pronounceable_count {
                        symbols.push(" ".to_string());
                    }
                }
                NormalizedToken::Initial(initial) => {
                    let initial_symbols = initial_to_phones(*initial)
                        .ok_or(G2pError::UnsupportedInitial { initial: *initial })?;
                    symbols.extend(initial_symbols.iter().copied().map(String::from));
                    emitted_pronounceable += 1;
                    if emitted_pronounceable < pronounceable_count {
                        symbols.push(" ".to_string());
                    }
                }
                NormalizedToken::PhraseBreak => {
                    if !matches!(symbols.last(), Some(last) if last == "|") {
                        symbols.push("|".to_string());
                    }
                }
            }
        }

        if matches!(
            normalized.boundary,
            ProsodyBoundaryHint::SentenceTerminalCandidate
        ) && !matches!(symbols.last(), Some(last) if last == "|")
        {
            symbols.push("|".to_string());
        }

        let length_hints = symbols
            .iter()
            .map(|symbol| PhoneLengthHint {
                symbol: symbol.clone(),
                class: if symbol == " " || symbol == "|" {
                    PhoneLengthClass::Short
                } else {
                    PhoneLengthClass::Medium
                },
            })
            .collect();

        Ok(PhonemizedUnit {
            phonemes: PiperPhonemeSequence {
                phonemes: symbols.into_iter().map(PiperPhoneme).collect(),
            },
            length_hints,
            boundary: normalized.boundary,
            commitment: normalized.commitment,
        })
    }
}

impl GraphemeToPhoneme for SimpleEnglishG2p {
    fn phonemize(&self, text: &str) -> Result<PiperPhonemeSequence> {
        Ok(self.phonemize_unit(text)?.phonemes)
    }
}

fn word_to_phones(word: &str) -> Option<&'static [&'static str]> {
    match word {
        "okay" => Some(&["OW", "K", "EY"]),
        "i" => Some(&["AY"]),
        "see" => Some(&["S", "IY"]),
        "doctor" => Some(&["D", "AA", "K", "T", "ER"]),
        "king" => Some(&["K", "IH", "NG"]),
        "scott" => Some(&["S", "K", "AA", "T"]),
        "fitzgerald" => Some(&["F", "IH", "TS", "JH", "EH", "R", "AH", "L", "D"]),
        _ => None,
    }
}

fn initial_to_phones(initial: char) -> Option<&'static [&'static str]> {
    match initial.to_ascii_lowercase() {
        'f' => Some(&["EH", "F"]),
        'j' => Some(&["JH", "EY"]),
        'r' => Some(&["AA", "R"]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symbols(sequence: &PiperPhonemeSequence) -> Vec<String> {
        sequence.phonemes.iter().map(|p| p.0.clone()).collect()
    }

    #[test]
    fn phonemizes_okay_sentence() {
        let g2p = SimpleEnglishG2p::default();
        let unit = g2p.phonemize_unit("Okay.").expect("phonemize");
        assert_eq!(symbols(&unit.phonemes), vec!["OW", "K", "EY", "|"]);
        assert_eq!(
            unit.boundary,
            ProsodyBoundaryHint::SentenceTerminalCandidate
        );
        assert_eq!(unit.commitment, ProsodyCommitment::Provisional);
    }

    #[test]
    fn phonemizes_i_see_sentence() {
        let g2p = SimpleEnglishG2p::default();
        let unit = g2p.phonemize_unit("I see.").expect("phonemize");
        assert_eq!(symbols(&unit.phonemes), vec!["AY", " ", "S", "IY", "|"]);
    }

    #[test]
    fn phonemizes_honorific_word() {
        let g2p = SimpleEnglishG2p::default();
        let unit = g2p.phonemize_unit("Dr. King").expect("phonemize");
        assert_eq!(
            symbols(&unit.phonemes),
            vec!["D", "AA", "K", "T", "ER", " ", "K", "IH", "NG"]
        );
        assert_eq!(unit.boundary, ProsodyBoundaryHint::None);
    }

    #[test]
    fn phonemizes_initials_and_words() {
        let g2p = SimpleEnglishG2p::default();
        let unit = g2p
            .phonemize_unit("F. Scott Fitzgerald")
            .expect("phonemize");
        assert_eq!(
            symbols(&unit.phonemes),
            vec![
                "EH", "F", " ", "S", "K", "AA", "T", " ", "F", "IH", "TS", "JH", "EH", "R", "AH",
                "L", "D"
            ]
        );
    }

    #[test]
    fn unsupported_words_return_clear_errors() {
        let g2p = SimpleEnglishG2p::default();
        let error = g2p
            .phonemize_unit("xylophone")
            .expect_err("unknown word should fail");
        assert_eq!(
            error,
            G2pError::UnsupportedWord {
                word: "xylophone".to_string()
            }
        );
        assert_eq!(
            error.to_string(),
            "unsupported word `xylophone` for native Piper simple English G2P"
        );
    }
}
