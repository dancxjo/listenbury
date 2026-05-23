use std::collections::{BTreeMap, HashSet};
use std::sync::OnceLock;

use serde::Deserialize;
use thiserror::Error;

use crate::linguistic::PhonemicInventory;
use crate::linguistic::inventory::general_american_english;

static EN_US_PACK: OnceLock<LanguagePack> = OnceLock::new();

const EN_US_MANIFEST_PATH: &str = "../../data/language-varieties/en-US/manifest.toml";
const EN_US_PHONOLOGY_PATH: &str = "../../data/language-varieties/en-US/phonology.toml";
const EN_US_INVENTORY_PATH: &str = "../../data/language-varieties/en-US/phoneme-inventory.toml";
const EN_US_VOWEL_TARGETS_PATH: &str = "../../data/language-varieties/en-US/vowel-targets.toml";
const EN_US_ACOUSTIC_TWEAKS_PATH: &str = "../../data/language-varieties/en-US/acoustic-tweaks.toml";
const EN_US_SPELLING_RULES_PATH: &str = "../../data/language-varieties/en-US/spelling-rules.toml";
const EN_US_LEXICON_PATH: &str = "../../data/language-varieties/en-US/lexicon.toml";
const EN_US_PRONUNCIATION_RULES_PATH: &str =
    "../../data/language-varieties/en-US/pronunciation-rules.toml";
const EN_US_PRONUNCIATION_RULES_JSON_PATH: &str =
    "../../data/language-varieties/en-US/pronunciation-rules.json";
const EN_US_MORPHOPHONOLOGY_PATH: &str = "../../data/language-varieties/en-US/morphophonology.toml";
const EN_US_PHRASE_RULES_PATH: &str = "../../data/language-varieties/en-US/phrase-rules.toml";
const EN_US_PROSODY_RULES_PATH: &str = "../../data/language-varieties/en-US/prosody-rules.toml";
const EN_US_PUNCTUATION_RULES_PATH: &str =
    "../../data/language-varieties/en-US/punctuation-rules.toml";
const EN_US_NORMALIZATION_NUMBERS_PATH: &str =
    "../../data/language-varieties/en-US/normalization/numbers.toml";
const EN_US_NORMALIZATION_LETTERS_PATH: &str =
    "../../data/language-varieties/en-US/normalization/letters.toml";
const EN_US_NORMALIZATION_SYMBOLS_PATH: &str =
    "../../data/language-varieties/en-US/normalization/symbols.toml";
const EN_US_NORMALIZATION_EMOJI_PATH: &str =
    "../../data/language-varieties/en-US/normalization/emoji.toml";
const EN_US_MBROLA_US1_MAP_PATH: &str =
    "../../data/language-varieties/en-US/backend-maps/mbrola-us1.toml";
const EN_US_MBROLA_US3_MAP_PATH: &str =
    "../../data/language-varieties/en-US/backend-maps/mbrola-us3.toml";
const EN_US_RIPER_MAP_PATH: &str = "../../data/language-varieties/en-US/backend-maps/riper.toml";
const EN_US_KLATT_MAP_PATH: &str = "../../data/language-varieties/en-US/backend-maps/klatt.toml";
const EN_US_PIPER_MAP_PATH: &str = "../../data/language-varieties/en-US/backend-maps/piper.toml";
const EN_US_SINGING_MAP_PATH: &str =
    "../../data/language-varieties/en-US/backend-maps/singing.toml";
const LANGUAGE_PACK_SOURCE_INVENTORY_PATH: &str =
    "docs/architecture/language-pack-source-inventory.md";

/// Listenbury-native language/variety pack model.
///
/// Schema intent is documented in
/// [`docs/architecture/language-pack-source-inventory.md`](../../docs/architecture/language-pack-source-inventory.md).
// `Eq` is intentionally omitted because `PhonemicInventory` currently derives
// `PartialEq` but not `Eq`.
#[derive(Debug, Clone, PartialEq)]
pub struct LanguagePack {
    pub manifest: LanguagePackManifest,
    pub inventory: PhonemicInventory,
    pub phonology: PhonologyProfile,
    pub inventory_profile: InventoryProfileSection,
    pub vowel_targets: VowelTargetsSection,
    pub acoustic_tweaks: AcousticTweaksSection,
    pub spelling_rules: SpellingRulesSection,
    pub lexicon: LexiconSection,
    pub pronunciation_rules: PronunciationRulesSection,
    pub morphophonology: MorphophonologySection,
    pub phrase_rules: PhraseRulesSection,
    pub prosody_rules: ProsodyRulesSection,
    pub punctuation_rules: PunctuationRulesSection,
    pub normalization: NormalizationTables,
    pub backend_maps: BackendMapRegistry,
    pub provenance: Vec<SourceProvenance>,
}

impl LanguagePack {
    pub fn backend_map(&self, backend_id: &str) -> Option<&BTreeMap<String, String>> {
        self.backend_maps.get(backend_id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct LanguagePackManifest {
    pub id: String,
    pub label: String,
    pub language: String,
    pub inventory_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct PhonologyProfile {
    pub arpabet_vowels: Vec<String>,
    pub nucleus_symbols: Vec<String>,
    pub vowel_phone_chars: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InventoryProfileSection {
    #[serde(default = "default_inventory_profile")]
    pub profile: String,
    pub vowels: Vec<String>,
    pub consonants: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VowelTargetsSection {
    pub profile: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticTweaksSection {
    pub policy: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpellingRulesSection {
    pub profile: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexiconSection {
    pub profile: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PronunciationRulesSection {
    pub policy: String,
    pub seed_rule_table_json: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct PronunciationPolicySection {
    pub profile: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MorphophonologySection {
    pub profile: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhraseRulesSection {
    pub profile: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProsodyRulesSection {
    pub profile: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PunctuationRulesSection {
    pub profile: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Split across multiple files in `normalization/`, so this is assembled from
/// separately-deserialized sections in `load_english_us_pack`.
pub struct NormalizationTables {
    pub numbers: NormalizationSection,
    pub letters: NormalizationSection,
    pub symbols: NormalizationSection,
    pub emoji: NormalizationSection,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormalizationSection {
    pub profile: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Validated backend map registry. Construct through `from_maps` so required
/// renderer map keys are always present.
pub struct BackendMapRegistry {
    maps: BTreeMap<String, BTreeMap<String, String>>,
}

impl BackendMapRegistry {
    fn from_maps(
        maps: BTreeMap<String, BTreeMap<String, String>>,
    ) -> Result<Self, LanguagePackDataError> {
        let required = [
            "riper",
            "klatt",
            "mbrola-us1",
            "mbrola-us3",
            "piper",
            "singing",
        ];
        let missing: Vec<&str> = required
            .into_iter()
            .filter(|backend| !maps.contains_key(*backend))
            .collect();
        if !missing.is_empty() {
            return Err(LanguagePackDataError::MissingBackendMaps { missing });
        }
        Ok(Self { maps })
    }

    pub fn get(&self, backend_id: &str) -> Option<&BTreeMap<String, String>> {
        self.maps.get(backend_id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceProvenance {
    pub section: String,
    pub source: String,
    pub source_path: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct BackendMapPack {
    map: BTreeMap<String, String>,
}

#[derive(Debug, Error)]
pub enum LanguagePackDataError {
    #[error("failed parsing {path}: {source}")]
    Parse {
        path: &'static str,
        source: toml::de::Error,
    },
    #[error("invalid section `{section}` in {path}: {reason}")]
    InvalidSection {
        path: &'static str,
        section: &'static str,
        reason: String,
    },
    #[error("missing required backend maps: {missing:?}")]
    MissingBackendMaps { missing: Vec<&'static str> },
}

pub fn english_us_language_pack() -> &'static LanguagePack {
    EN_US_PACK.get_or_init(|| load_english_us_pack().expect("en-US language pack should be valid"))
}

fn load_english_us_pack() -> Result<LanguagePack, LanguagePackDataError> {
    let manifest: LanguagePackManifest = parse_toml(
        include_str!("../../data/language-varieties/en-US/manifest.toml"),
        EN_US_MANIFEST_PATH,
    )?;
    let phonology: PhonologyProfile = parse_toml(
        include_str!("../../data/language-varieties/en-US/phonology.toml"),
        EN_US_PHONOLOGY_PATH,
    )?;
    let inventory_profile: InventoryProfileSection = parse_toml(
        include_str!("../../data/language-varieties/en-US/phoneme-inventory.toml"),
        EN_US_INVENTORY_PATH,
    )?;
    let vowel_targets: VowelTargetsSection = parse_toml(
        include_str!("../../data/language-varieties/en-US/vowel-targets.toml"),
        EN_US_VOWEL_TARGETS_PATH,
    )?;
    let acoustic_tweaks: AcousticTweaksSection = parse_toml(
        include_str!("../../data/language-varieties/en-US/acoustic-tweaks.toml"),
        EN_US_ACOUSTIC_TWEAKS_PATH,
    )?;
    let spelling_rules: SpellingRulesSection = parse_toml(
        include_str!("../../data/language-varieties/en-US/spelling-rules.toml"),
        EN_US_SPELLING_RULES_PATH,
    )?;
    let lexicon: LexiconSection = parse_toml(
        include_str!("../../data/language-varieties/en-US/lexicon.toml"),
        EN_US_LEXICON_PATH,
    )?;
    let pronunciation_policy: PronunciationPolicySection = parse_toml(
        include_str!("../../data/language-varieties/en-US/pronunciation-rules.toml"),
        EN_US_PRONUNCIATION_RULES_PATH,
    )?;
    let morphophonology: MorphophonologySection = parse_toml(
        include_str!("../../data/language-varieties/en-US/morphophonology.toml"),
        EN_US_MORPHOPHONOLOGY_PATH,
    )?;
    let phrase_rules: PhraseRulesSection = parse_toml(
        include_str!("../../data/language-varieties/en-US/phrase-rules.toml"),
        EN_US_PHRASE_RULES_PATH,
    )?;
    let prosody_rules: ProsodyRulesSection = parse_toml(
        include_str!("../../data/language-varieties/en-US/prosody-rules.toml"),
        EN_US_PROSODY_RULES_PATH,
    )?;
    let punctuation_rules: PunctuationRulesSection = parse_toml(
        include_str!("../../data/language-varieties/en-US/punctuation-rules.toml"),
        EN_US_PUNCTUATION_RULES_PATH,
    )?;

    let normalization = NormalizationTables {
        numbers: parse_toml(
            include_str!("../../data/language-varieties/en-US/normalization/numbers.toml"),
            EN_US_NORMALIZATION_NUMBERS_PATH,
        )?,
        letters: parse_toml(
            include_str!("../../data/language-varieties/en-US/normalization/letters.toml"),
            EN_US_NORMALIZATION_LETTERS_PATH,
        )?,
        symbols: parse_toml(
            include_str!("../../data/language-varieties/en-US/normalization/symbols.toml"),
            EN_US_NORMALIZATION_SYMBOLS_PATH,
        )?,
        emoji: parse_toml(
            include_str!("../../data/language-varieties/en-US/normalization/emoji.toml"),
            EN_US_NORMALIZATION_EMOJI_PATH,
        )?,
    };

    validate_non_empty(
        EN_US_INVENTORY_PATH,
        "inventory.profile",
        &inventory_profile.profile,
    )?;
    validate_non_empty(
        EN_US_VOWEL_TARGETS_PATH,
        "vowel_targets.profile",
        &vowel_targets.profile,
    )?;
    validate_non_empty(
        EN_US_ACOUSTIC_TWEAKS_PATH,
        "acoustic_tweaks.policy",
        &acoustic_tweaks.policy,
    )?;
    validate_non_empty(
        EN_US_SPELLING_RULES_PATH,
        "spelling_rules.profile",
        &spelling_rules.profile,
    )?;
    validate_non_empty(EN_US_LEXICON_PATH, "lexicon.profile", &lexicon.profile)?;
    validate_non_empty(
        EN_US_PRONUNCIATION_RULES_PATH,
        "pronunciation_rules.profile",
        &pronunciation_policy.profile,
    )?;
    validate_non_empty(
        EN_US_MORPHOPHONOLOGY_PATH,
        "morphophonology.profile",
        &morphophonology.profile,
    )?;
    validate_non_empty(
        EN_US_PHRASE_RULES_PATH,
        "phrase_rules.profile",
        &phrase_rules.profile,
    )?;
    validate_non_empty(
        EN_US_PROSODY_RULES_PATH,
        "prosody_rules.profile",
        &prosody_rules.profile,
    )?;
    validate_non_empty(
        EN_US_PUNCTUATION_RULES_PATH,
        "punctuation_rules.profile",
        &punctuation_rules.profile,
    )?;
    validate_non_empty(
        EN_US_NORMALIZATION_NUMBERS_PATH,
        "normalization.numbers.profile",
        &normalization.numbers.profile,
    )?;
    validate_non_empty(
        EN_US_NORMALIZATION_LETTERS_PATH,
        "normalization.letters.profile",
        &normalization.letters.profile,
    )?;
    validate_non_empty(
        EN_US_NORMALIZATION_SYMBOLS_PATH,
        "normalization.symbols.profile",
        &normalization.symbols.profile,
    )?;
    validate_non_empty(
        EN_US_NORMALIZATION_EMOJI_PATH,
        "normalization.emoji.profile",
        &normalization.emoji.profile,
    )?;

    let nucleus_symbols: HashSet<String> = phonology.nucleus_symbols.iter().cloned().collect();
    if !phonology
        .arpabet_vowels
        .iter()
        .all(|vowel| nucleus_symbols.contains(vowel))
    {
        return Err(LanguagePackDataError::InvalidSection {
            path: EN_US_PHONOLOGY_PATH,
            section: "phonology.arpabet_vowels",
            reason: "every arpabet vowel must also be listed in nucleus_symbols".to_string(),
        });
    }
    if phonology.arpabet_vowels.is_empty() {
        return Err(LanguagePackDataError::InvalidSection {
            path: EN_US_PHONOLOGY_PATH,
            section: "phonology.arpabet_vowels",
            reason: "at least one ARPABET vowel is required".to_string(),
        });
    }
    if phonology.nucleus_symbols.is_empty() {
        return Err(LanguagePackDataError::InvalidSection {
            path: EN_US_PHONOLOGY_PATH,
            section: "phonology.nucleus_symbols",
            reason: "at least one nucleus symbol is required".to_string(),
        });
    }
    if inventory_profile.vowels.is_empty() {
        return Err(LanguagePackDataError::InvalidSection {
            path: EN_US_INVENTORY_PATH,
            section: "inventory.vowels",
            reason: "at least one inventory vowel is required".to_string(),
        });
    }
    if inventory_profile.consonants.is_empty() {
        return Err(LanguagePackDataError::InvalidSection {
            path: EN_US_INVENTORY_PATH,
            section: "inventory.consonants",
            reason: "at least one inventory consonant is required".to_string(),
        });
    }

    let mbrola_us1: BackendMapPack = parse_toml(
        include_str!("../../data/language-varieties/en-US/backend-maps/mbrola-us1.toml"),
        EN_US_MBROLA_US1_MAP_PATH,
    )?;
    let mbrola_us3: BackendMapPack = parse_toml(
        include_str!("../../data/language-varieties/en-US/backend-maps/mbrola-us3.toml"),
        EN_US_MBROLA_US3_MAP_PATH,
    )?;
    let riper: BackendMapPack = parse_toml(
        include_str!("../../data/language-varieties/en-US/backend-maps/riper.toml"),
        EN_US_RIPER_MAP_PATH,
    )?;
    let klatt: BackendMapPack = parse_toml(
        include_str!("../../data/language-varieties/en-US/backend-maps/klatt.toml"),
        EN_US_KLATT_MAP_PATH,
    )?;
    let piper: BackendMapPack = parse_toml(
        include_str!("../../data/language-varieties/en-US/backend-maps/piper.toml"),
        EN_US_PIPER_MAP_PATH,
    )?;
    let singing: BackendMapPack = parse_toml(
        include_str!("../../data/language-varieties/en-US/backend-maps/singing.toml"),
        EN_US_SINGING_MAP_PATH,
    )?;

    let mut backend_maps = BTreeMap::new();
    backend_maps.insert("mbrola-us1".to_string(), mbrola_us1.map);
    backend_maps.insert("mbrola-us3".to_string(), mbrola_us3.map);
    backend_maps.insert("riper".to_string(), riper.map);
    backend_maps.insert("klatt".to_string(), klatt.map);
    backend_maps.insert("piper".to_string(), piper.map);
    backend_maps.insert("singing".to_string(), singing.map);
    let backend_maps = BackendMapRegistry::from_maps(backend_maps)?;

    let pronunciation_rules_json =
        include_str!("../../data/language-varieties/en-US/pronunciation-rules.json");
    if pronunciation_rules_json.trim().is_empty() {
        return Err(LanguagePackDataError::InvalidSection {
            path: EN_US_PRONUNCIATION_RULES_JSON_PATH,
            section: "pronunciation_rules.seed_rule_table_json",
            reason: "JSON seed table cannot be empty".to_string(),
        });
    }

    let provenance = vec![
        SourceProvenance {
            section: "language-pack-source-inventory".to_string(),
            source: "listenbury-architecture".to_string(),
            source_path: LANGUAGE_PACK_SOURCE_INVENTORY_PATH,
        },
        SourceProvenance {
            section: "manifest".to_string(),
            source: "language-pack-data".to_string(),
            source_path: EN_US_MANIFEST_PATH,
        },
        SourceProvenance {
            section: "phonology".to_string(),
            source: "language-pack-data".to_string(),
            source_path: EN_US_PHONOLOGY_PATH,
        },
        SourceProvenance {
            section: "inventory-profile".to_string(),
            source: "language-pack-data".to_string(),
            source_path: EN_US_INVENTORY_PATH,
        },
        SourceProvenance {
            section: "vowel-targets".to_string(),
            source: "language-pack-data".to_string(),
            source_path: EN_US_VOWEL_TARGETS_PATH,
        },
        SourceProvenance {
            section: "acoustic-tweaks".to_string(),
            source: "language-pack-data".to_string(),
            source_path: EN_US_ACOUSTIC_TWEAKS_PATH,
        },
        SourceProvenance {
            section: "spelling-rules".to_string(),
            source: "language-pack-data".to_string(),
            source_path: EN_US_SPELLING_RULES_PATH,
        },
        SourceProvenance {
            section: "lexicon".to_string(),
            source: "language-pack-data".to_string(),
            source_path: EN_US_LEXICON_PATH,
        },
        SourceProvenance {
            section: "pronunciation-rules".to_string(),
            source: "language-pack-data".to_string(),
            source_path: EN_US_PRONUNCIATION_RULES_PATH,
        },
        SourceProvenance {
            section: "morphophonology".to_string(),
            source: "language-pack-data".to_string(),
            source_path: EN_US_MORPHOPHONOLOGY_PATH,
        },
        SourceProvenance {
            section: "phrase-rules".to_string(),
            source: "language-pack-data".to_string(),
            source_path: EN_US_PHRASE_RULES_PATH,
        },
        SourceProvenance {
            section: "prosody-rules".to_string(),
            source: "language-pack-data".to_string(),
            source_path: EN_US_PROSODY_RULES_PATH,
        },
        SourceProvenance {
            section: "punctuation-rules".to_string(),
            source: "language-pack-data".to_string(),
            source_path: EN_US_PUNCTUATION_RULES_PATH,
        },
        SourceProvenance {
            section: "normalization".to_string(),
            source: "language-pack-data".to_string(),
            source_path: EN_US_NORMALIZATION_NUMBERS_PATH,
        },
    ];

    Ok(LanguagePack {
        manifest,
        inventory: general_american_english(),
        phonology,
        inventory_profile,
        vowel_targets,
        acoustic_tweaks,
        spelling_rules,
        lexicon,
        pronunciation_rules: PronunciationRulesSection {
            policy: pronunciation_policy.profile,
            seed_rule_table_json: pronunciation_rules_json,
        },
        morphophonology,
        phrase_rules,
        prosody_rules,
        punctuation_rules,
        normalization,
        backend_maps,
        provenance,
    })
}

fn parse_toml<T: for<'de> Deserialize<'de>>(
    data: &'static str,
    path: &'static str,
) -> Result<T, LanguagePackDataError> {
    toml::from_str(data).map_err(|source| LanguagePackDataError::Parse { path, source })
}

fn default_inventory_profile() -> String {
    "en-us-phoneme-inventory-v1".to_string()
}

fn validate_non_empty(
    path: &'static str,
    section: &'static str,
    value: &str,
) -> Result<(), LanguagePackDataError> {
    if value.trim().is_empty() {
        return Err(LanguagePackDataError::InvalidSection {
            path,
            section,
            reason: "must not be empty".to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn en_us_pack_exposes_all_required_sections() {
        let pack = english_us_language_pack();
        assert_eq!(pack.manifest.id, "en-US");
        assert!(!pack.phonology.arpabet_vowels.is_empty());
        assert!(!pack.pronunciation_rules.seed_rule_table_json.is_empty());
        assert!(pack.backend_map("riper").is_some());
        assert!(pack.backend_map("klatt").is_some());
        assert!(pack.backend_map("mbrola-us1").is_some());
        assert!(pack.backend_map("piper").is_some());
        assert!(pack.backend_map("singing").is_some());
    }

    #[test]
    fn en_us_pack_inventory_is_renderer_neutral() {
        let pack = english_us_language_pack();
        let er = pack
            .inventory
            .find_by_ipa("ɝ")
            .expect("inventory should expose canonical IPA entries");
        assert_eq!(er.ipa, "ɝ");
    }

    #[test]
    fn en_us_pack_carries_source_inventory_provenance() {
        let pack = english_us_language_pack();
        assert!(
            pack.provenance
                .iter()
                .any(|item| item.source_path == LANGUAGE_PACK_SOURCE_INVENTORY_PATH),
            "schema should link back to language-pack source inventory doc"
        );
    }
}
