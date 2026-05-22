pub mod arpabet;
pub mod cmudict;
pub mod environment;
pub mod inventory;
pub mod language_pack;
#[cfg(feature = "tts-riper")]
pub mod language_pack_rules;
pub mod language_variety;
pub mod orthography;
pub mod phone;
pub mod phoneme;
pub mod phonology;
pub mod pronounce;
pub mod realization;
pub mod rule_registry;
pub mod service;
pub mod sound_it_out;
pub mod variety;

pub use cmudict::{CmuPhoneme, CmudictPronouncer, PronunciationEntry, PronunciationStatus, Stress};
pub use inventory::general_american_english;
pub use language_pack::{LanguagePack, english_us_language_pack};
pub use language_variety::{LanguageVariety, LanguageVarietyLookupError, english_us_variety};
pub use orthography::OrthographicWord;
pub use phoneme::{Phoneme, PhonemeSeq, PhonemeText, PhonemeTextUnit};
pub use phonology::{
    Phone, PhoneComparisonMode, PhoneDecompositionPolicy, PhoneEqualityOptions, PhoneStatus,
    PhoneString, PhonemeDefinition, PhonemeId, PhonemicInventory, RealizedPhone, SourceSymbol,
    VarietyId, VarietyImplementationStatus, phone_comparison_key, phones_equivalent,
};
pub use pronounce::{OrthographyToPhonemes, PhonologyError};
pub use rule_registry::{
    InventoryData, PhonotacticData, RuleFragment, RuleProfile, RuleRegistry, RuleRegistryError,
    VarietyRuleData,
};
pub use service::PronunciationService;
pub use sound_it_out::{Environment, GraphemeRule, SoundItOutPronouncer, SoundItOutRules};
pub use variety::{
    EnglishVariety, Lexicon, LinguisticRuntimeProfile, LinguisticVariety, Phonology, VarietyTag,
};

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
