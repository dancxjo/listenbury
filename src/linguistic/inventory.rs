use serde::{Deserialize, Serialize};

use crate::linguistic::arpabet::{default_phone_string_for_arpabet, feature_bundle_for_arpabet};
use crate::linguistic::phone::Stress;
use crate::linguistic::phone::{Phone, PhoneEqualityOptions, PhoneString, phones_equivalent};
use crate::linguistic::rule_registry::RuleRegistry;

/// Stable identifier for a phonological variety.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct VarietyId(pub String);

impl VarietyId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

/// Stable identifier for a phoneme within a variety's inventory.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PhonemeId(pub String);

impl PhonemeId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

/// A source-schema symbol that maps to a phoneme in the inventory.
///
/// For example, the CMU Pronouncing Dictionary symbol `"AE"` maps to the IPA
/// phoneme `/æ/`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceSymbol {
    /// The schema this symbol comes from.
    pub schema: PhonemeSchema,
    /// The symbol string, e.g. `"AE"`, `"æ"`.
    pub symbol: String,
}

/// A single phoneme entry in a [`PhonemicInventory`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhonemeDefinition {
    /// Stable identifier for this phoneme.
    pub id: PhonemeId,
    /// Canonical IPA representation.
    pub ipa: String,
    /// Source symbols across different encoding schemas.
    pub source_symbols: Vec<SourceSymbol>,
    /// Default phone realization as a sequence of zero or more phones.
    ///
    /// Most phonemes realize as a single [`Phone`]; affricates, diphthongs,
    /// and syllabic consonants may realize as multiple phones.
    pub default_phone_string: PhoneString,
    /// Broad phonological class(es) for this phoneme.
    pub classes: Vec<PhonemeClass>,
    /// Distinctive-feature bundle used for environment matching.
    pub features: FeatureBundle,
}

impl PhonemeDefinition {
    /// Return the canonical default [`Phone`] (first element of
    /// `default_phone_string`), or a freshly constructed phone from the IPA
    /// string if the phone string is empty.
    pub fn default_phone(&self) -> Phone {
        self.default_phone_string
            .phones
            .first()
            .cloned()
            .unwrap_or_else(|| Phone::mapped(self.ipa.clone()))
    }
}

/// The phoneme inventory and phone comparison policy for a specific linguistic
/// variety.
///
/// This is the phonological backbone consumed by
/// [`crate::prosody::phonotactics::EnglishPhonotactics`] and the syllabifier.  It is *not* the same as
/// [`crate::linguistic::variety::LinguisticVariety`], which handles runtime
/// configuration; `PhonemicInventory` is purely about phonological facts.
///
/// Construct via [`crate::linguistic::variety::EnglishVariety::phonemic_inventory`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhonemicInventory {
    /// Stable identifier for this variety.
    pub id: VarietyId,
    /// ISO 639 language code, e.g. `"en"`.
    pub language: String,
    /// Human-readable label, e.g. `"General American English"`.
    pub label: String,
    /// All phonemes in this variety's inventory.
    pub phonemes: Vec<PhonemeDefinition>,
    /// Phone comparison policy.
    pub phone_equality: PhoneEqualityOptions,
}

impl PhonemicInventory {
    /// Return all phoneme definitions that belong to `class`.
    pub fn phonemes_of_class(&self, class: PhonemeClass) -> Vec<&PhonemeDefinition> {
        self.phonemes
            .iter()
            .filter(|def| def.classes.contains(&class) || def.features.matches_class(class, None))
            .collect()
    }

    /// Look up the phoneme definition whose canonical IPA matches `ipa`
    /// using this inventory's equality policy.
    pub fn find_by_ipa(&self, ipa: &str) -> Option<&PhonemeDefinition> {
        let query = Phone::mapped(ipa);
        self.phonemes.iter().find(|def| {
            let canonical = Phone::mapped(def.ipa.clone());
            phones_equivalent(&query, &canonical, &self.phone_equality)
        })
    }

    /// Return the feature bundle for a phone in this inventory.
    ///
    /// Lookup checks both canonical phoneme IPA and the realized phones in each
    /// default phone string. Unknown phones default to permissively voiced so
    /// borrowed or experimental symbols do not lose pitch unless the inventory
    /// explicitly marks them voiceless.
    pub fn features_for_phone(&self, phone: &Phone) -> FeatureBundle {
        self.phonemes
            .iter()
            .find(|def| {
                let canonical = Phone::mapped(def.ipa.clone());
                phones_equivalent(phone, &canonical, &self.phone_equality)
                    || def
                        .default_phone_string
                        .phones
                        .iter()
                        .any(|candidate| phones_equivalent(phone, candidate, &self.phone_equality))
            })
            .map(|def| def.features)
            .unwrap_or_else(FeatureBundle::unknown_phone)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhonemeSchema {
    Arpabet,
    Cmudict,
    ArpabetSurface,
    Ipa,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WordPosition {
    Singleton,
    WordInitial,
    WordMedial,
    WordFinal,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Environment {
    pub left_phone: Option<String>,
    pub right_phone: Option<String>,
    pub left_class: Option<String>,
    pub right_class: Option<String>,
    pub word_position: Option<WordPosition>,
    pub syllable_position: Option<String>,
    pub stress_context: Option<String>,
    pub phrase_position: Option<String>,
    pub language: Option<String>,
    pub dialect: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhonemeClass {
    Vowel,
    Consonant,
    AlveolarStop,
    AlveolarNasal,
    VelarStop,
    VelarConsonant,
    Sonorant,
    Obstruent,
    Continuant,
    Coronal,
    Dorsal,
    Labial,
    Nasal,
    Liquid,
    Glide,
    Sibilant,
    HighVowel,
    UnstressedVowel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MajorClass {
    Vowel,
    Consonant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Place {
    Bilabial,
    Labiodental,
    Dental,
    Alveolar,
    Postalveolar,
    Palatal,
    Velar,
    Glottal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VowelHeight {
    High,
    Mid,
    Low,
    Rhotic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VowelBackness {
    Front,
    Central,
    Back,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Roundedness {
    Rounded,
    Unrounded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Manner {
    Stop,
    Nasal,
    Fricative,
    Affricate,
    Liquid,
    Glide,
    Vowel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Voicing {
    Voiced,
    Voiceless,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FeatureBundle {
    pub major: MajorClass,
    pub place: Option<Place>,
    pub vowel_height: Option<VowelHeight>,
    pub vowel_backness: Option<VowelBackness>,
    pub roundedness: Option<Roundedness>,
    pub manner: Option<Manner>,
    pub voicing: Option<Voicing>,
    pub syllabic: bool,
}

impl FeatureBundle {
    pub fn unknown_phone() -> Self {
        FeatureBundle {
            major: MajorClass::Consonant,
            place: None,
            vowel_height: None,
            vowel_backness: None,
            roundedness: None,
            manner: None,
            voicing: None,
            syllabic: false,
        }
    }

    pub fn is_voiced(self) -> bool {
        !matches!(self.voicing, Some(Voicing::Voiceless))
    }

    pub fn matches_class(self, class: PhonemeClass, stress: Option<Stress>) -> bool {
        match class {
            PhonemeClass::Vowel => self.major == MajorClass::Vowel,
            PhonemeClass::Consonant => self.major == MajorClass::Consonant,
            PhonemeClass::AlveolarStop => {
                self.place == Some(Place::Alveolar) && self.manner == Some(Manner::Stop)
            }
            PhonemeClass::AlveolarNasal => {
                self.place == Some(Place::Alveolar) && self.manner == Some(Manner::Nasal)
            }
            PhonemeClass::VelarStop => {
                self.place == Some(Place::Velar) && self.manner == Some(Manner::Stop)
            }
            PhonemeClass::VelarConsonant => {
                self.major == MajorClass::Consonant && self.place == Some(Place::Velar)
            }
            PhonemeClass::Sonorant => self.is_sonorant(),
            PhonemeClass::Obstruent => self.is_obstruent(),
            PhonemeClass::Continuant => self.is_continuant(),
            PhonemeClass::Coronal => self.is_coronal(),
            PhonemeClass::Dorsal => self.is_dorsal(),
            PhonemeClass::Labial => self.is_labial(),
            PhonemeClass::Nasal => self.manner == Some(Manner::Nasal),
            PhonemeClass::Liquid => self.manner == Some(Manner::Liquid),
            PhonemeClass::Glide => self.manner == Some(Manner::Glide),
            PhonemeClass::Sibilant => self.is_sibilant(),
            PhonemeClass::HighVowel => self.is_high_vowel(),
            PhonemeClass::UnstressedVowel => {
                self.major == MajorClass::Vowel && stress == Some(Stress::Unstressed)
            }
        }
    }

    pub fn is_sonorant(self) -> bool {
        self.major == MajorClass::Vowel
            || matches!(
                self.manner,
                Some(Manner::Nasal | Manner::Liquid | Manner::Glide)
            )
    }

    pub fn is_obstruent(self) -> bool {
        self.major == MajorClass::Consonant
            && matches!(
                self.manner,
                Some(Manner::Stop | Manner::Fricative | Manner::Affricate)
            )
    }

    pub fn is_continuant(self) -> bool {
        matches!(
            self.manner,
            Some(Manner::Fricative | Manner::Liquid | Manner::Glide | Manner::Vowel)
        )
    }

    pub fn is_coronal(self) -> bool {
        matches!(
            self.place,
            Some(Place::Dental | Place::Alveolar | Place::Postalveolar)
        )
    }

    pub fn is_dorsal(self) -> bool {
        matches!(self.place, Some(Place::Palatal | Place::Velar))
    }

    pub fn is_labial(self) -> bool {
        matches!(self.place, Some(Place::Bilabial | Place::Labiodental))
    }

    pub fn is_sibilant(self) -> bool {
        matches!(self.place, Some(Place::Alveolar | Place::Postalveolar))
            && matches!(self.manner, Some(Manner::Fricative | Manner::Affricate))
    }

    pub fn is_high_vowel(self) -> bool {
        self.major == MajorClass::Vowel && self.vowel_height == Some(VowelHeight::High)
    }
}

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
