use super::phonology::{
    PhonemeClass, PhonemeSchema, default_phone_string_for_arpabet, feature_bundle_for_arpabet,
};
use super::rule_registry::RuleRegistry;

pub use super::phonology::{PhonemeDefinition, PhonemeId, PhonemicInventory, SourceSymbol};

/// Build the General American English phonemic inventory.
///
/// Every phoneme is defined with:
/// - a [`PhonemeId`] (ARPABET base as lowercase, e.g. `"ae"`)
/// - canonical IPA from the ARPABET default phone mapping
/// - source symbols for both ARPABET and IPA schemas
/// - a default phone realization, including multi-phone affricates and diphthongs
/// - broad phoneme class(es)
pub fn general_american_english() -> PhonemicInventory {
    RuleRegistry::builtin()
        .inventory("en-US-GA")
        .expect("built-in registry should include en-US-GA")
}

/// ARPABET → IPA mapping used to populate the English phoneme inventory.
/// Each row: (arpabet_base, ipa, is_vowel)
pub(crate) fn english_phoneme_table() -> Vec<PhonemeDefinition> {
    let rows: &[(&str, &str, bool)] = &[
        // Vowels / nuclei
        ("AA", "ɑ", true),
        ("AE", "æ", true),
        ("AH", "ʌ", true),
        ("AO", "ɔ", true),
        ("AW", "aʊ", true),
        ("AY", "aɪ", true),
        ("EH", "ɛ", true),
        ("ER", "ɝ", true),
        ("EY", "eɪ", true),
        ("IH", "ɪ", true),
        ("IY", "iː", true),
        ("OW", "oʊ", true),
        ("OY", "ɔɪ", true),
        ("UH", "ʊ", true),
        ("UW", "uː", true),
        // Consonants
        ("B", "b", false),
        ("CH", "tʃ", false),
        ("D", "d", false),
        ("DH", "ð", false),
        ("F", "f", false),
        ("G", "ɡ", false),
        ("HH", "h", false),
        ("JH", "dʒ", false),
        ("K", "k", false),
        ("L", "l", false),
        ("M", "m", false),
        ("N", "n", false),
        ("NG", "ŋ", false),
        ("P", "p", false),
        ("R", "ɹ", false),
        ("S", "s", false),
        ("SH", "ʃ", false),
        ("T", "t", false),
        ("TH", "θ", false),
        ("V", "v", false),
        ("W", "w", false),
        ("Y", "j", false),
        ("Z", "z", false),
        ("ZH", "ʒ", false),
    ];
    rows.iter()
        .map(|(arpabet, ipa, is_vowel)| {
            let id = PhonemeId::new(arpabet.to_lowercase());
            let default_phone_string = default_phone_string_for_arpabet(arpabet, arpabet);
            let classes = if *is_vowel {
                vec![PhonemeClass::Vowel]
            } else {
                vec![PhonemeClass::Consonant]
            };
            PhonemeDefinition {
                id,
                ipa: ipa.to_string(),
                source_symbols: vec![
                    SourceSymbol {
                        schema: PhonemeSchema::Arpabet,
                        symbol: arpabet.to_string(),
                    },
                    SourceSymbol {
                        schema: PhonemeSchema::Ipa,
                        symbol: ipa.to_string(),
                    },
                ],
                default_phone_string,
                classes,
                features: feature_bundle_for_arpabet(arpabet),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linguistic::VarietyId;

    #[test]
    fn general_american_inventory_has_expected_id() {
        let inv = general_american_english();
        assert_eq!(inv.id, VarietyId::new("en-US-GA"));
    }

    #[test]
    fn inventory_can_find_vowel_phonemes_by_ipa() {
        let inv = general_american_english();
        let def = inv.find_by_ipa("æ").expect("æ should be in GA inventory");
        assert!(def.classes.contains(&PhonemeClass::Vowel));
    }

    #[test]
    fn inventory_can_find_consonant_by_ipa() {
        let inv = general_american_english();
        let def = inv.find_by_ipa("ɹ").expect("ɹ should be in GA inventory");
        assert!(def.classes.contains(&PhonemeClass::Consonant));
    }
}
