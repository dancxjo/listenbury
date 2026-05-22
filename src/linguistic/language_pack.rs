use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde::Deserialize;
use thiserror::Error;

static EN_US_PACK: OnceLock<LanguagePack> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanguagePack {
    pub manifest: LanguagePackManifest,
    pub phoneme_inventory: PhonemeInventorySection,
    pub vowel_targets: VowelTargetsSection,
    pub acoustic_tweaks: AcousticTweaksSection,
    pub spelling_rules: SpellingRulesSection,
    pub pronunciation_rules: PronunciationRulesSection,
    pub morphophonology: MorphophonologySection,
    pub prosody_rules: ProsodyRulesSection,
    pub punctuation_rules: PunctuationRulesSection,
    pub backend_maps: BTreeMap<String, BTreeMap<String, String>>,
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
pub struct PhonemeInventorySection {
    pub vowels: Vec<String>,
    pub consonants: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct VowelTargetsSection {
    pub profile: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct AcousticTweaksSection {
    pub policy: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SpellingRulesSection {
    pub profile: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PronunciationRulesSection {
    pub policy: String,
    pub seed_rule_table_json: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct MorphophonologySection {
    pub profile: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ProsodyRulesSection {
    pub profile: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct PunctuationRulesSection {
    pub profile: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct BackendMapPack {
    map: BTreeMap<String, String>,
}

#[derive(Debug, Error)]
enum LanguagePackDataError {
    #[error("failed parsing {path}: {source}")]
    Parse {
        path: &'static str,
        source: toml::de::Error,
    },
}

pub fn english_us_language_pack() -> &'static LanguagePack {
    EN_US_PACK.get_or_init(|| load_english_us_pack().expect("en-US language pack should be valid"))
}

fn load_english_us_pack() -> Result<LanguagePack, LanguagePackDataError> {
    let manifest: LanguagePackManifest = toml::from_str(include_str!(
        "../../data/language-varieties/en-US/manifest.toml"
    ))
    .map_err(|source| LanguagePackDataError::Parse {
        path: "../../data/language-varieties/en-US/manifest.toml",
        source,
    })?;
    let phoneme_inventory: PhonemeInventorySection = toml::from_str(include_str!(
        "../../data/language-varieties/en-US/phoneme-inventory.toml"
    ))
    .map_err(|source| LanguagePackDataError::Parse {
        path: "../../data/language-varieties/en-US/phoneme-inventory.toml",
        source,
    })?;
    let vowel_targets: VowelTargetsSection = toml::from_str(include_str!(
        "../../data/language-varieties/en-US/vowel-targets.toml"
    ))
    .map_err(|source| LanguagePackDataError::Parse {
        path: "../../data/language-varieties/en-US/vowel-targets.toml",
        source,
    })?;
    let acoustic_tweaks: AcousticTweaksSection = toml::from_str(include_str!(
        "../../data/language-varieties/en-US/acoustic-tweaks.toml"
    ))
    .map_err(|source| LanguagePackDataError::Parse {
        path: "../../data/language-varieties/en-US/acoustic-tweaks.toml",
        source,
    })?;
    let spelling_rules: SpellingRulesSection = toml::from_str(include_str!(
        "../../data/language-varieties/en-US/spelling-rules.toml"
    ))
    .map_err(|source| LanguagePackDataError::Parse {
        path: "../../data/language-varieties/en-US/spelling-rules.toml",
        source,
    })?;
    let pronunciation_policy: SpellingRulesSection = toml::from_str(include_str!(
        "../../data/language-varieties/en-US/pronunciation-rules.toml"
    ))
    .map_err(|source| LanguagePackDataError::Parse {
        path: "../../data/language-varieties/en-US/pronunciation-rules.toml",
        source,
    })?;
    let morphophonology: MorphophonologySection = toml::from_str(include_str!(
        "../../data/language-varieties/en-US/morphophonology.toml"
    ))
    .map_err(|source| LanguagePackDataError::Parse {
        path: "../../data/language-varieties/en-US/morphophonology.toml",
        source,
    })?;
    let prosody_rules: ProsodyRulesSection = toml::from_str(include_str!(
        "../../data/language-varieties/en-US/prosody-rules.toml"
    ))
    .map_err(|source| LanguagePackDataError::Parse {
        path: "../../data/language-varieties/en-US/prosody-rules.toml",
        source,
    })?;
    let punctuation_rules: PunctuationRulesSection = toml::from_str(include_str!(
        "../../data/language-varieties/en-US/punctuation-rules.toml"
    ))
    .map_err(|source| LanguagePackDataError::Parse {
        path: "../../data/language-varieties/en-US/punctuation-rules.toml",
        source,
    })?;
    let mbrola_us1: BackendMapPack = toml::from_str(include_str!(
        "../../data/language-varieties/en-US/backend-maps/mbrola-us1.toml"
    ))
    .map_err(|source| LanguagePackDataError::Parse {
        path: "../../data/language-varieties/en-US/backend-maps/mbrola-us1.toml",
        source,
    })?;
    let mbrola_us3: BackendMapPack = toml::from_str(include_str!(
        "../../data/language-varieties/en-US/backend-maps/mbrola-us3.toml"
    ))
    .map_err(|source| LanguagePackDataError::Parse {
        path: "../../data/language-varieties/en-US/backend-maps/mbrola-us3.toml",
        source,
    })?;
    let riper: BackendMapPack = toml::from_str(include_str!(
        "../../data/language-varieties/en-US/backend-maps/riper.toml"
    ))
    .map_err(|source| LanguagePackDataError::Parse {
        path: "../../data/language-varieties/en-US/backend-maps/riper.toml",
        source,
    })?;

    let mut backend_maps = BTreeMap::new();
    backend_maps.insert("mbrola-us1".to_string(), mbrola_us1.map);
    backend_maps.insert("mbrola-us3".to_string(), mbrola_us3.map);
    backend_maps.insert("riper".to_string(), riper.map);

    Ok(LanguagePack {
        manifest,
        phoneme_inventory,
        vowel_targets,
        acoustic_tweaks,
        spelling_rules,
        pronunciation_rules: PronunciationRulesSection {
            policy: pronunciation_policy.profile,
            seed_rule_table_json: include_str!(
                "../../data/language-varieties/en-US/pronunciation-rules.json"
            ),
        },
        morphophonology,
        prosody_rules,
        punctuation_rules,
        backend_maps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn en_us_pack_exposes_all_required_sections() {
        let pack = english_us_language_pack();
        assert_eq!(pack.manifest.id, "en-US");
        assert!(!pack.phoneme_inventory.vowels.is_empty());
        assert!(!pack.pronunciation_rules.seed_rule_table_json.is_empty());
        assert!(pack.backend_map("riper").is_some());
        assert!(pack.backend_map("mbrola-us1").is_some());
    }
}
