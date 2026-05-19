pub mod orthography;
pub mod phoneme;
pub mod pronounce;
pub mod variety;

pub use orthography::OrthographicWord;
pub use phoneme::{Phoneme, PhonemeSeq, PhonemeText, PhonemeTextUnit};
pub use pronounce::{OrthographyToPhonemes, PhonologyError};
pub use variety::{Lexicon, LinguisticRuntimeProfile, LinguisticVariety, Phonology, VarietyTag};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_tagged_varieties_and_untagged_session_variety() {
        let english = LinguisticVariety::tagged(
            VarietyTag::new("en_US"),
            "English (US)",
            Phonology::new("General American"),
        );
        let french = LinguisticVariety::tagged(
            VarietyTag::new("fr_CA"),
            "French (Canada)",
            Phonology::new("Canadian French"),
        );
        let custom = LinguisticVariety::untagged("Session Blend", Phonology::new("Custom Session"));

        assert_eq!(english.tag, Some(VarietyTag::new("en_US")));
        assert_eq!(french.tag, Some(VarietyTag::new("fr_CA")));
        assert_eq!(custom.tag, None);
    }

    #[test]
    fn constructs_word_phoneme_sequence() {
        let word = OrthographicWord::new("okay");
        let sequence = PhonemeSeq::new(vec![
            Phoneme::new("OW"),
            Phoneme::new("K"),
            Phoneme::new("EY"),
        ]);

        let unit = PhonemeTextUnit::Word {
            orthography: word.clone(),
            phonemes: sequence.clone(),
        };

        assert_eq!(word.text, "okay");
        assert_eq!(sequence.phonemes.len(), 3);
        assert_eq!(
            unit,
            PhonemeTextUnit::Word {
                orthography: OrthographicWord::new("okay"),
                phonemes: PhonemeSeq::new(vec![
                    Phoneme::new("OW"),
                    Phoneme::new("K"),
                    Phoneme::new("EY")
                ]),
            }
        );
    }

    #[test]
    fn constructs_phoneme_text_with_boundaries() {
        let text = PhonemeText::new(vec![
            PhonemeTextUnit::Word {
                orthography: OrthographicWord::new("hello"),
                phonemes: PhonemeSeq::new(vec![
                    Phoneme::new("HH"),
                    Phoneme::new("AH"),
                    Phoneme::new("L"),
                    Phoneme::new("OW"),
                ]),
            },
            PhonemeTextUnit::WordBoundary,
            PhonemeTextUnit::Word {
                orthography: OrthographicWord::new("world"),
                phonemes: PhonemeSeq::new(vec![
                    Phoneme::new("W"),
                    Phoneme::new("ER"),
                    Phoneme::new("L"),
                    Phoneme::new("D"),
                ]),
            },
            PhonemeTextUnit::PhraseBoundary,
        ]);

        assert_eq!(text.units.len(), 4);
        assert_eq!(text.units[1], PhonemeTextUnit::WordBoundary);
        assert_eq!(text.units[3], PhonemeTextUnit::PhraseBoundary);
    }
}
