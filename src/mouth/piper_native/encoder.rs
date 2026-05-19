use crate::linguistic::phoneme::{PhonemeText, PhonemeTextUnit};
use crate::mouth::piper_native::phoneme::{PiperPhoneme, PiperPhonemeSequence};

const WORD_SEPARATOR: &str = " ";
const PHRASE_BREAK: &str = "|";

/// Encodes a neutral [`PhonemeText`] into a Piper-specific [`PiperPhonemeSequence`].
///
/// This is the boundary layer between the shared phonological representation
/// and the Piper ONNX synthesis backend.  It maps:
///
/// - [`PhonemeTextUnit::Word`] → the word's phoneme symbols in sequence.
/// - [`PhonemeTextUnit::WordBoundary`] → the `" "` inter-word separator token.
/// - [`PhonemeTextUnit::PhraseBoundary`] → the `"|"` phrase-pause token.
///
/// Consecutive phrase-break tokens are deduplicated (Piper treats multiple
/// `"|"` entries identically to one, but the deduplication keeps the sequence
/// clean).
#[derive(Debug, Default, Clone, Copy)]
pub struct PiperEncoder;

impl PiperEncoder {
    /// Encode `phoneme_text` into a [`PiperPhonemeSequence`] ready for ID
    /// look-up via [`PiperPhonemeSequence::to_piper_ids`].
    pub fn encode(&self, phoneme_text: &PhonemeText) -> PiperPhonemeSequence {
        let mut phonemes = Vec::new();
        for unit in &phoneme_text.units {
            match unit {
                PhonemeTextUnit::Word { phonemes: word_phonemes, .. } => {
                    phonemes.extend(
                        word_phonemes
                            .phonemes
                            .iter()
                            .map(|p| PiperPhoneme(p.symbol.clone())),
                    );
                }
                PhonemeTextUnit::WordBoundary => {
                    phonemes.push(PiperPhoneme(WORD_SEPARATOR.to_string()));
                }
                PhonemeTextUnit::PhraseBoundary => {
                    if !matches!(phonemes.last(), Some(last) if last.0 == PHRASE_BREAK) {
                        phonemes.push(PiperPhoneme(PHRASE_BREAK.to_string()));
                    }
                }
            }
        }
        PiperPhonemeSequence { phonemes }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linguistic::orthography::OrthographicWord;
    use crate::linguistic::phoneme::{Phoneme, PhonemeSeq, PhonemeText, PhonemeTextUnit};

    fn word_unit(ortho: &str, symbols: &[&str]) -> PhonemeTextUnit {
        PhonemeTextUnit::Word {
            orthography: OrthographicWord::new(ortho),
            phonemes: PhonemeSeq::new(symbols.iter().map(|s| Phoneme::new(*s)).collect()),
        }
    }

    fn symbols(seq: &PiperPhonemeSequence) -> Vec<&str> {
        seq.phonemes.iter().map(|p| p.0.as_str()).collect()
    }

    #[test]
    fn encodes_single_word() {
        let text = PhonemeText::new(vec![word_unit("okay", &["OW", "K", "EY"])]);
        let seq = PiperEncoder.encode(&text);
        assert_eq!(symbols(&seq), vec!["OW", "K", "EY"]);
    }

    #[test]
    fn encodes_word_boundary_as_space() {
        let text = PhonemeText::new(vec![
            word_unit("i", &["AY"]),
            PhonemeTextUnit::WordBoundary,
            word_unit("see", &["S", "IY"]),
        ]);
        let seq = PiperEncoder.encode(&text);
        assert_eq!(symbols(&seq), vec!["AY", " ", "S", "IY"]);
    }

    #[test]
    fn encodes_phrase_boundary_as_pipe() {
        let text = PhonemeText::new(vec![
            word_unit("okay", &["OW", "K", "EY"]),
            PhonemeTextUnit::PhraseBoundary,
        ]);
        let seq = PiperEncoder.encode(&text);
        assert_eq!(symbols(&seq), vec!["OW", "K", "EY", "|"]);
    }

    #[test]
    fn deduplicates_consecutive_phrase_breaks() {
        let text = PhonemeText::new(vec![
            word_unit("okay", &["OW", "K", "EY"]),
            PhonemeTextUnit::PhraseBoundary,
            PhonemeTextUnit::PhraseBoundary,
        ]);
        let seq = PiperEncoder.encode(&text);
        assert_eq!(symbols(&seq), vec!["OW", "K", "EY", "|"]);
    }
}
