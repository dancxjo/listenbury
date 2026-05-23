use std::collections::{BTreeMap, HashSet};
use std::sync::OnceLock;

use serde::Deserialize;
use thiserror::Error;

use crate::linguistic::PhonemicInventory;
use crate::linguistic::inventory::general_american_english;
use crate::linguistic::variety::EnglishVariety;

static EN_US_PACK: OnceLock<LanguagePack> = OnceLock::new();
static EN_GB_RP_PACK: OnceLock<LanguagePack> = OnceLock::new();

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
const EN_GB_RP_MANIFEST_PATH: &str = "../../data/language-varieties/en-GB-RP/manifest.toml";
const EN_GB_RP_PHONOLOGY_PATH: &str = "../../data/language-varieties/en-GB-RP/phonology.toml";
const EN_GB_RP_INVENTORY_PATH: &str =
    "../../data/language-varieties/en-GB-RP/phoneme-inventory.toml";
const EN_GB_RP_VOWEL_TARGETS_PATH: &str =
    "../../data/language-varieties/en-GB-RP/vowel-targets.toml";
const EN_GB_RP_ACOUSTIC_TWEAKS_PATH: &str =
    "../../data/language-varieties/en-GB-RP/acoustic-tweaks.toml";
const EN_GB_RP_SPELLING_RULES_PATH: &str =
    "../../data/language-varieties/en-GB-RP/spelling-rules.toml";
const EN_GB_RP_LEXICON_PATH: &str = "../../data/language-varieties/en-GB-RP/lexicon.toml";
const EN_GB_RP_PRONUNCIATION_RULES_PATH: &str =
    "../../data/language-varieties/en-GB-RP/pronunciation-rules.toml";
const EN_GB_RP_PRONUNCIATION_RULES_JSON_PATH: &str =
    "../../data/language-varieties/en-GB-RP/pronunciation-rules.json";
const EN_GB_RP_MORPHOPHONOLOGY_PATH: &str =
    "../../data/language-varieties/en-GB-RP/morphophonology.toml";
const EN_GB_RP_PHRASE_RULES_PATH: &str = "../../data/language-varieties/en-GB-RP/phrase-rules.toml";
const EN_GB_RP_PROSODY_RULES_PATH: &str =
    "../../data/language-varieties/en-GB-RP/prosody-rules.toml";
const EN_GB_RP_PUNCTUATION_RULES_PATH: &str =
    "../../data/language-varieties/en-GB-RP/punctuation-rules.toml";
const EN_GB_RP_NORMALIZATION_NUMBERS_PATH: &str =
    "../../data/language-varieties/en-GB-RP/normalization/numbers.toml";
const EN_GB_RP_NORMALIZATION_LETTERS_PATH: &str =
    "../../data/language-varieties/en-GB-RP/normalization/letters.toml";
const EN_GB_RP_NORMALIZATION_SYMBOLS_PATH: &str =
    "../../data/language-varieties/en-GB-RP/normalization/symbols.toml";
const EN_GB_RP_NORMALIZATION_EMOJI_PATH: &str =
    "../../data/language-varieties/en-GB-RP/normalization/emoji.toml";
const EN_GB_RP_MBROLA_EN1_MAP_PATH: &str =
    "../../data/language-varieties/en-GB-RP/backend-maps/mbrola-en1.toml";
const EN_GB_RP_RIPER_MAP_PATH: &str =
    "../../data/language-varieties/en-GB-RP/backend-maps/riper.toml";
const EN_GB_RP_KLATT_MAP_PATH: &str =
    "../../data/language-varieties/en-GB-RP/backend-maps/klatt.toml";
const EN_GB_RP_PIPER_MAP_PATH: &str =
    "../../data/language-varieties/en-GB-RP/backend-maps/piper.toml";
const EN_GB_RP_SINGING_MAP_PATH: &str =
    "../../data/language-varieties/en-GB-RP/backend-maps/singing.toml";
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
    pub pronunciation_rule_catalog_json: &'static str,
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
        required: &'static [&'static str],
    ) -> Result<Self, LanguagePackDataError> {
        let missing: Vec<&str> = required
            .iter()
            .copied()
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

#[derive(Clone, Copy)]
struct StaticBackendMapData {
    id: &'static str,
    data: &'static str,
    path: &'static str,
}

#[derive(Clone, Copy)]
struct StaticLanguagePackData {
    manifest: (&'static str, &'static str),
    phonology: (&'static str, &'static str),
    inventory_profile: (&'static str, &'static str),
    vowel_targets: (&'static str, &'static str),
    acoustic_tweaks: (&'static str, &'static str),
    spelling_rules: (&'static str, &'static str),
    lexicon: (&'static str, &'static str),
    pronunciation_policy: (&'static str, &'static str),
    pronunciation_rules_json: (&'static str, &'static str),
    morphophonology: (&'static str, &'static str),
    phrase_rules: (&'static str, &'static str),
    prosody_rules: (&'static str, &'static str),
    punctuation_rules: (&'static str, &'static str),
    normalization_numbers: (&'static str, &'static str),
    normalization_letters: (&'static str, &'static str),
    normalization_symbols: (&'static str, &'static str),
    normalization_emoji: (&'static str, &'static str),
    backend_maps: &'static [StaticBackendMapData],
    required_backend_maps: &'static [&'static str],
    inventory: fn() -> PhonemicInventory,
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

pub fn english_rp_language_pack() -> &'static LanguagePack {
    EN_GB_RP_PACK
        .get_or_init(|| load_english_rp_pack().expect("en-GB-RP language pack should be valid"))
}

fn load_english_us_pack() -> Result<LanguagePack, LanguagePackDataError> {
    load_static_language_pack(StaticLanguagePackData {
        manifest: (
            include_str!("../../data/language-varieties/en-US/manifest.toml"),
            EN_US_MANIFEST_PATH,
        ),
        phonology: (
            include_str!("../../data/language-varieties/en-US/phonology.toml"),
            EN_US_PHONOLOGY_PATH,
        ),
        inventory_profile: (
            include_str!("../../data/language-varieties/en-US/phoneme-inventory.toml"),
            EN_US_INVENTORY_PATH,
        ),
        vowel_targets: (
            include_str!("../../data/language-varieties/en-US/vowel-targets.toml"),
            EN_US_VOWEL_TARGETS_PATH,
        ),
        acoustic_tweaks: (
            include_str!("../../data/language-varieties/en-US/acoustic-tweaks.toml"),
            EN_US_ACOUSTIC_TWEAKS_PATH,
        ),
        spelling_rules: (
            include_str!("../../data/language-varieties/en-US/spelling-rules.toml"),
            EN_US_SPELLING_RULES_PATH,
        ),
        lexicon: (
            include_str!("../../data/language-varieties/en-US/lexicon.toml"),
            EN_US_LEXICON_PATH,
        ),
        pronunciation_policy: (
            include_str!("../../data/language-varieties/en-US/pronunciation-rules.toml"),
            EN_US_PRONUNCIATION_RULES_PATH,
        ),
        pronunciation_rules_json: (
            include_str!("../../data/language-varieties/en-US/pronunciation-rules.json"),
            EN_US_PRONUNCIATION_RULES_JSON_PATH,
        ),
        morphophonology: (
            include_str!("../../data/language-varieties/en-US/morphophonology.toml"),
            EN_US_MORPHOPHONOLOGY_PATH,
        ),
        phrase_rules: (
            include_str!("../../data/language-varieties/en-US/phrase-rules.toml"),
            EN_US_PHRASE_RULES_PATH,
        ),
        prosody_rules: (
            include_str!("../../data/language-varieties/en-US/prosody-rules.toml"),
            EN_US_PROSODY_RULES_PATH,
        ),
        punctuation_rules: (
            include_str!("../../data/language-varieties/en-US/punctuation-rules.toml"),
            EN_US_PUNCTUATION_RULES_PATH,
        ),
        normalization_numbers: (
            include_str!("../../data/language-varieties/en-US/normalization/numbers.toml"),
            EN_US_NORMALIZATION_NUMBERS_PATH,
        ),
        normalization_letters: (
            include_str!("../../data/language-varieties/en-US/normalization/letters.toml"),
            EN_US_NORMALIZATION_LETTERS_PATH,
        ),
        normalization_symbols: (
            include_str!("../../data/language-varieties/en-US/normalization/symbols.toml"),
            EN_US_NORMALIZATION_SYMBOLS_PATH,
        ),
        normalization_emoji: (
            include_str!("../../data/language-varieties/en-US/normalization/emoji.toml"),
            EN_US_NORMALIZATION_EMOJI_PATH,
        ),
        backend_maps: &[
            StaticBackendMapData {
                id: "mbrola-us1",
                data: include_str!(
                    "../../data/language-varieties/en-US/backend-maps/mbrola-us1.toml"
                ),
                path: EN_US_MBROLA_US1_MAP_PATH,
            },
            StaticBackendMapData {
                id: "mbrola-us3",
                data: include_str!(
                    "../../data/language-varieties/en-US/backend-maps/mbrola-us3.toml"
                ),
                path: EN_US_MBROLA_US3_MAP_PATH,
            },
            StaticBackendMapData {
                id: "riper",
                data: include_str!("../../data/language-varieties/en-US/backend-maps/riper.toml"),
                path: EN_US_RIPER_MAP_PATH,
            },
            StaticBackendMapData {
                id: "klatt",
                data: include_str!("../../data/language-varieties/en-US/backend-maps/klatt.toml"),
                path: EN_US_KLATT_MAP_PATH,
            },
            StaticBackendMapData {
                id: "piper",
                data: include_str!("../../data/language-varieties/en-US/backend-maps/piper.toml"),
                path: EN_US_PIPER_MAP_PATH,
            },
            StaticBackendMapData {
                id: "singing",
                data: include_str!("../../data/language-varieties/en-US/backend-maps/singing.toml"),
                path: EN_US_SINGING_MAP_PATH,
            },
        ],
        required_backend_maps: &[
            "riper",
            "klatt",
            "mbrola-us1",
            "mbrola-us3",
            "piper",
            "singing",
        ],
        inventory: general_american_english,
    })
}

fn load_english_rp_pack() -> Result<LanguagePack, LanguagePackDataError> {
    load_static_language_pack(StaticLanguagePackData {
        manifest: (
            include_str!("../../data/language-varieties/en-GB-RP/manifest.toml"),
            EN_GB_RP_MANIFEST_PATH,
        ),
        phonology: (
            include_str!("../../data/language-varieties/en-GB-RP/phonology.toml"),
            EN_GB_RP_PHONOLOGY_PATH,
        ),
        inventory_profile: (
            include_str!("../../data/language-varieties/en-GB-RP/phoneme-inventory.toml"),
            EN_GB_RP_INVENTORY_PATH,
        ),
        vowel_targets: (
            include_str!("../../data/language-varieties/en-GB-RP/vowel-targets.toml"),
            EN_GB_RP_VOWEL_TARGETS_PATH,
        ),
        acoustic_tweaks: (
            include_str!("../../data/language-varieties/en-GB-RP/acoustic-tweaks.toml"),
            EN_GB_RP_ACOUSTIC_TWEAKS_PATH,
        ),
        spelling_rules: (
            include_str!("../../data/language-varieties/en-GB-RP/spelling-rules.toml"),
            EN_GB_RP_SPELLING_RULES_PATH,
        ),
        lexicon: (
            include_str!("../../data/language-varieties/en-GB-RP/lexicon.toml"),
            EN_GB_RP_LEXICON_PATH,
        ),
        pronunciation_policy: (
            include_str!("../../data/language-varieties/en-GB-RP/pronunciation-rules.toml"),
            EN_GB_RP_PRONUNCIATION_RULES_PATH,
        ),
        pronunciation_rules_json: (
            include_str!("../../data/language-varieties/en-GB-RP/pronunciation-rules.json"),
            EN_GB_RP_PRONUNCIATION_RULES_JSON_PATH,
        ),
        morphophonology: (
            include_str!("../../data/language-varieties/en-GB-RP/morphophonology.toml"),
            EN_GB_RP_MORPHOPHONOLOGY_PATH,
        ),
        phrase_rules: (
            include_str!("../../data/language-varieties/en-GB-RP/phrase-rules.toml"),
            EN_GB_RP_PHRASE_RULES_PATH,
        ),
        prosody_rules: (
            include_str!("../../data/language-varieties/en-GB-RP/prosody-rules.toml"),
            EN_GB_RP_PROSODY_RULES_PATH,
        ),
        punctuation_rules: (
            include_str!("../../data/language-varieties/en-GB-RP/punctuation-rules.toml"),
            EN_GB_RP_PUNCTUATION_RULES_PATH,
        ),
        normalization_numbers: (
            include_str!("../../data/language-varieties/en-GB-RP/normalization/numbers.toml"),
            EN_GB_RP_NORMALIZATION_NUMBERS_PATH,
        ),
        normalization_letters: (
            include_str!("../../data/language-varieties/en-GB-RP/normalization/letters.toml"),
            EN_GB_RP_NORMALIZATION_LETTERS_PATH,
        ),
        normalization_symbols: (
            include_str!("../../data/language-varieties/en-GB-RP/normalization/symbols.toml"),
            EN_GB_RP_NORMALIZATION_SYMBOLS_PATH,
        ),
        normalization_emoji: (
            include_str!("../../data/language-varieties/en-GB-RP/normalization/emoji.toml"),
            EN_GB_RP_NORMALIZATION_EMOJI_PATH,
        ),
        backend_maps: &[
            StaticBackendMapData {
                id: "mbrola-en1",
                data: include_str!(
                    "../../data/language-varieties/en-GB-RP/backend-maps/mbrola-en1.toml"
                ),
                path: EN_GB_RP_MBROLA_EN1_MAP_PATH,
            },
            StaticBackendMapData {
                id: "riper",
                data: include_str!(
                    "../../data/language-varieties/en-GB-RP/backend-maps/riper.toml"
                ),
                path: EN_GB_RP_RIPER_MAP_PATH,
            },
            StaticBackendMapData {
                id: "klatt",
                data: include_str!(
                    "../../data/language-varieties/en-GB-RP/backend-maps/klatt.toml"
                ),
                path: EN_GB_RP_KLATT_MAP_PATH,
            },
            StaticBackendMapData {
                id: "piper",
                data: include_str!(
                    "../../data/language-varieties/en-GB-RP/backend-maps/piper.toml"
                ),
                path: EN_GB_RP_PIPER_MAP_PATH,
            },
            StaticBackendMapData {
                id: "singing",
                data: include_str!(
                    "../../data/language-varieties/en-GB-RP/backend-maps/singing.toml"
                ),
                path: EN_GB_RP_SINGING_MAP_PATH,
            },
        ],
        required_backend_maps: &["riper", "klatt", "mbrola-en1", "piper", "singing"],
        inventory: received_pronunciation_inventory,
    })
}

fn received_pronunciation_inventory() -> PhonemicInventory {
    EnglishVariety::ReceivedPronunciation.phonemic_inventory()
}

fn load_static_language_pack(
    data: StaticLanguagePackData,
) -> Result<LanguagePack, LanguagePackDataError> {
    let manifest: LanguagePackManifest = parse_toml(data.manifest.0, data.manifest.1)?;
    let phonology: PhonologyProfile = parse_toml(data.phonology.0, data.phonology.1)?;
    let inventory_profile: InventoryProfileSection =
        parse_toml(data.inventory_profile.0, data.inventory_profile.1)?;
    let vowel_targets: VowelTargetsSection =
        parse_toml(data.vowel_targets.0, data.vowel_targets.1)?;
    let acoustic_tweaks: AcousticTweaksSection =
        parse_toml(data.acoustic_tweaks.0, data.acoustic_tweaks.1)?;
    let spelling_rules: SpellingRulesSection =
        parse_toml(data.spelling_rules.0, data.spelling_rules.1)?;
    let lexicon: LexiconSection = parse_toml(data.lexicon.0, data.lexicon.1)?;
    let pronunciation_policy: PronunciationPolicySection =
        parse_toml(data.pronunciation_policy.0, data.pronunciation_policy.1)?;
    let morphophonology: MorphophonologySection =
        parse_toml(data.morphophonology.0, data.morphophonology.1)?;
    let phrase_rules: PhraseRulesSection = parse_toml(data.phrase_rules.0, data.phrase_rules.1)?;
    let prosody_rules: ProsodyRulesSection =
        parse_toml(data.prosody_rules.0, data.prosody_rules.1)?;
    let punctuation_rules: PunctuationRulesSection =
        parse_toml(data.punctuation_rules.0, data.punctuation_rules.1)?;

    let normalization = NormalizationTables {
        numbers: parse_toml(data.normalization_numbers.0, data.normalization_numbers.1)?,
        letters: parse_toml(data.normalization_letters.0, data.normalization_letters.1)?,
        symbols: parse_toml(data.normalization_symbols.0, data.normalization_symbols.1)?,
        emoji: parse_toml(data.normalization_emoji.0, data.normalization_emoji.1)?,
    };

    validate_non_empty(
        data.inventory_profile.1,
        "inventory.profile",
        &inventory_profile.profile,
    )?;
    validate_non_empty(
        data.vowel_targets.1,
        "vowel_targets.profile",
        &vowel_targets.profile,
    )?;
    validate_non_empty(
        data.acoustic_tweaks.1,
        "acoustic_tweaks.policy",
        &acoustic_tweaks.policy,
    )?;
    validate_non_empty(
        data.spelling_rules.1,
        "spelling_rules.profile",
        &spelling_rules.profile,
    )?;
    validate_non_empty(data.lexicon.1, "lexicon.profile", &lexicon.profile)?;
    validate_non_empty(
        data.pronunciation_policy.1,
        "pronunciation_rules.profile",
        &pronunciation_policy.profile,
    )?;
    validate_non_empty(
        data.morphophonology.1,
        "morphophonology.profile",
        &morphophonology.profile,
    )?;
    validate_non_empty(
        data.phrase_rules.1,
        "phrase_rules.profile",
        &phrase_rules.profile,
    )?;
    validate_non_empty(
        data.prosody_rules.1,
        "prosody_rules.profile",
        &prosody_rules.profile,
    )?;
    validate_non_empty(
        data.punctuation_rules.1,
        "punctuation_rules.profile",
        &punctuation_rules.profile,
    )?;
    validate_non_empty(
        data.normalization_numbers.1,
        "normalization.numbers.profile",
        &normalization.numbers.profile,
    )?;
    validate_non_empty(
        data.normalization_letters.1,
        "normalization.letters.profile",
        &normalization.letters.profile,
    )?;
    validate_non_empty(
        data.normalization_symbols.1,
        "normalization.symbols.profile",
        &normalization.symbols.profile,
    )?;
    validate_non_empty(
        data.normalization_emoji.1,
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
            path: data.phonology.1,
            section: "phonology.arpabet_vowels",
            reason: "every arpabet vowel must also be listed in nucleus_symbols".to_string(),
        });
    }
    if phonology.arpabet_vowels.is_empty() {
        return Err(LanguagePackDataError::InvalidSection {
            path: data.phonology.1,
            section: "phonology.arpabet_vowels",
            reason: "at least one ARPABET vowel is required".to_string(),
        });
    }
    if phonology.nucleus_symbols.is_empty() {
        return Err(LanguagePackDataError::InvalidSection {
            path: data.phonology.1,
            section: "phonology.nucleus_symbols",
            reason: "at least one nucleus symbol is required".to_string(),
        });
    }
    if inventory_profile.vowels.is_empty() {
        return Err(LanguagePackDataError::InvalidSection {
            path: data.inventory_profile.1,
            section: "inventory.vowels",
            reason: "at least one inventory vowel is required".to_string(),
        });
    }
    if inventory_profile.consonants.is_empty() {
        return Err(LanguagePackDataError::InvalidSection {
            path: data.inventory_profile.1,
            section: "inventory.consonants",
            reason: "at least one inventory consonant is required".to_string(),
        });
    }

    let mut backend_maps = BTreeMap::new();
    for map_data in data.backend_maps {
        let map: BackendMapPack = parse_toml(map_data.data, map_data.path)?;
        backend_maps.insert(map_data.id.to_string(), map.map);
    }
    let backend_maps = BackendMapRegistry::from_maps(backend_maps, data.required_backend_maps)?;

    let pronunciation_rules_json = data.pronunciation_rules_json.0;
    if pronunciation_rules_json.trim().is_empty() {
        return Err(LanguagePackDataError::InvalidSection {
            path: data.pronunciation_rules_json.1,
            section: "pronunciation_rules.pronunciation_rule_catalog_json",
            reason: "JSON pronunciation rule catalog cannot be empty".to_string(),
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
            source_path: data.manifest.1,
        },
        SourceProvenance {
            section: "phonology".to_string(),
            source: "language-pack-data".to_string(),
            source_path: data.phonology.1,
        },
        SourceProvenance {
            section: "inventory-profile".to_string(),
            source: "language-pack-data".to_string(),
            source_path: data.inventory_profile.1,
        },
        SourceProvenance {
            section: "vowel-targets".to_string(),
            source: "language-pack-data".to_string(),
            source_path: data.vowel_targets.1,
        },
        SourceProvenance {
            section: "acoustic-tweaks".to_string(),
            source: "language-pack-data".to_string(),
            source_path: data.acoustic_tweaks.1,
        },
        SourceProvenance {
            section: "spelling-rules".to_string(),
            source: "language-pack-data".to_string(),
            source_path: data.spelling_rules.1,
        },
        SourceProvenance {
            section: "lexicon".to_string(),
            source: "language-pack-data".to_string(),
            source_path: data.lexicon.1,
        },
        SourceProvenance {
            section: "pronunciation-rules".to_string(),
            source: "language-pack-data".to_string(),
            source_path: data.pronunciation_policy.1,
        },
        SourceProvenance {
            section: "morphophonology".to_string(),
            source: "language-pack-data".to_string(),
            source_path: data.morphophonology.1,
        },
        SourceProvenance {
            section: "phrase-rules".to_string(),
            source: "language-pack-data".to_string(),
            source_path: data.phrase_rules.1,
        },
        SourceProvenance {
            section: "prosody-rules".to_string(),
            source: "language-pack-data".to_string(),
            source_path: data.prosody_rules.1,
        },
        SourceProvenance {
            section: "punctuation-rules".to_string(),
            source: "language-pack-data".to_string(),
            source_path: data.punctuation_rules.1,
        },
        SourceProvenance {
            section: "normalization".to_string(),
            source: "language-pack-data".to_string(),
            source_path: data.normalization_numbers.1,
        },
    ];

    Ok(LanguagePack {
        manifest,
        inventory: (data.inventory)(),
        phonology,
        inventory_profile,
        vowel_targets,
        acoustic_tweaks,
        spelling_rules,
        lexicon,
        pronunciation_rules: PronunciationRulesSection {
            policy: pronunciation_policy.profile,
            pronunciation_rule_catalog_json: pronunciation_rules_json,
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
        assert!(
            !pack
                .pronunciation_rules
                .pronunciation_rule_catalog_json
                .is_empty()
        );
        assert!(pack.backend_map("riper").is_some());
        assert!(pack.backend_map("klatt").is_some());
        assert!(pack.backend_map("mbrola-us1").is_some());
        assert!(pack.backend_map("piper").is_some());
        assert!(pack.backend_map("singing").is_some());
    }

    #[test]
    fn en_gb_rp_pack_exposes_mbrola_en1_datapack_map() {
        let pack = english_rp_language_pack();
        assert_eq!(pack.manifest.id, "en-GB-RP");
        assert_eq!(pack.manifest.inventory_id, "en-GB-RP");
        assert!(pack.backend_map("mbrola-en1").is_some());
        assert_eq!(
            pack.backend_map("mbrola-en1")
                .and_then(|map| map.get("OW1"))
                .map(String::as_str),
            Some("@U")
        );
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
