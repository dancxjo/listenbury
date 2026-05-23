use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::OnceLock;

use serde::Deserialize;
use thiserror::Error;

static EN_US_VARIETY: OnceLock<LanguageVariety> = OnceLock::new();
static EN_GB_RP_VARIETY: OnceLock<LanguageVariety> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanguageVariety {
    pub id: String,
    pub label: String,
    pub language: String,
    pub inventory_id: String,
    arpabet_vowels: HashSet<String>,
    nucleus_symbols: HashSet<String>,
    vowel_phone_chars: HashSet<char>,
    backend_maps: HashMap<String, BTreeMap<String, String>>,
}

impl LanguageVariety {
    pub fn is_arpabet_vowel(&self, symbol: &str) -> bool {
        let normalized = normalize_arpabet_symbol(symbol);
        self.arpabet_vowels.contains(&normalized)
    }

    pub fn is_nucleus_symbol(&self, symbol: &str) -> bool {
        let base = symbol.trim_end_matches(|ch: char| ch.is_ascii_digit());
        self.nucleus_symbols.contains(base)
            || self.nucleus_symbols.contains(&base.to_ascii_uppercase())
    }

    pub fn is_vowel_phone(&self, phone: &str) -> bool {
        if self.is_arpabet_vowel(phone) {
            return true;
        }
        phone.chars().any(|ch| {
            ch.to_lowercase()
                .any(|normalized| self.vowel_phone_chars.contains(&normalized))
        })
    }

    pub fn backend_map(
        &self,
        backend_id: &str,
    ) -> Result<&BTreeMap<String, String>, LanguageVarietyLookupError> {
        self.backend_maps.get(backend_id).ok_or_else(|| {
            LanguageVarietyLookupError::UnknownBackend {
                variety_id: self.id.clone(),
                backend_id: backend_id.to_string(),
            }
        })
    }

    pub fn map_backend_symbol(
        &self,
        backend_id: &str,
        phone: &str,
    ) -> Result<String, LanguageVarietyLookupError> {
        let map = self.backend_map(backend_id)?;
        if let Some(mapped) = map.get(phone) {
            return Ok(mapped.clone());
        }
        let stressless = phone.trim_end_matches(|ch: char| ch.is_ascii_digit());
        if stressless != phone
            && let Some(mapped) = map.get(stressless)
        {
            return Ok(mapped.clone());
        }
        Err(LanguageVarietyLookupError::UnknownPhone {
            variety_id: self.id.clone(),
            backend_id: backend_id.to_string(),
            phone: phone.to_string(),
        })
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LanguageVarietyLookupError {
    #[error("language variety `{variety_id}` has no backend map `{backend_id}`")]
    UnknownBackend {
        variety_id: String,
        backend_id: String,
    },
    #[error(
        "phone `{phone}` is not mapped for backend `{backend_id}` in language variety `{variety_id}`"
    )]
    UnknownPhone {
        variety_id: String,
        backend_id: String,
        phone: String,
    },
}

#[derive(Debug, Error)]
enum LanguageVarietyDataError {
    #[error("failed parsing {path}: {source}")]
    Parse {
        path: &'static str,
        source: toml::de::Error,
    },
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
struct ManifestPack {
    id: String,
    label: String,
    language: String,
    inventory_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
struct PhonologyPack {
    arpabet_vowels: Vec<String>,
    nucleus_symbols: Vec<String>,
    vowel_phone_chars: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct BackendMapPack {
    map: BTreeMap<String, String>,
}

pub fn english_us_variety() -> &'static LanguageVariety {
    EN_US_VARIETY.get_or_init(|| {
        load_english_us_variety().expect("en-US language-variety datapack should be valid")
    })
}

pub fn english_rp_variety() -> &'static LanguageVariety {
    EN_GB_RP_VARIETY.get_or_init(|| {
        load_english_rp_variety().expect("en-GB-RP language-variety datapack should be valid")
    })
}

fn load_english_us_variety() -> Result<LanguageVariety, LanguageVarietyDataError> {
    let manifest: ManifestPack = toml::from_str(include_str!(
        "../../data/language-varieties/en-US/manifest.toml"
    ))
    .map_err(|source| LanguageVarietyDataError::Parse {
        path: "../../data/language-varieties/en-US/manifest.toml",
        source,
    })?;
    let phonology: PhonologyPack = toml::from_str(include_str!(
        "../../data/language-varieties/en-US/phonology.toml"
    ))
    .map_err(|source| LanguageVarietyDataError::Parse {
        path: "../../data/language-varieties/en-US/phonology.toml",
        source,
    })?;
    let mbrola_us1: BackendMapPack = toml::from_str(include_str!(
        "../../data/language-varieties/en-US/backend-maps/mbrola-us1.toml"
    ))
    .map_err(|source| LanguageVarietyDataError::Parse {
        path: "../../data/language-varieties/en-US/backend-maps/mbrola-us1.toml",
        source,
    })?;
    let mbrola_us3: BackendMapPack = toml::from_str(include_str!(
        "../../data/language-varieties/en-US/backend-maps/mbrola-us3.toml"
    ))
    .map_err(|source| LanguageVarietyDataError::Parse {
        path: "../../data/language-varieties/en-US/backend-maps/mbrola-us3.toml",
        source,
    })?;

    let mut backend_maps = HashMap::new();
    backend_maps.insert("mbrola-us1".to_string(), mbrola_us1.map);
    backend_maps.insert("mbrola-us3".to_string(), mbrola_us3.map);

    Ok(LanguageVariety {
        id: manifest.id,
        label: manifest.label,
        language: manifest.language,
        inventory_id: manifest.inventory_id,
        arpabet_vowels: phonology
            .arpabet_vowels
            .into_iter()
            .map(|value| value.to_ascii_uppercase())
            .collect(),
        nucleus_symbols: phonology.nucleus_symbols.into_iter().collect(),
        vowel_phone_chars: phonology
            .vowel_phone_chars
            .into_iter()
            .flat_map(|symbol| symbol.chars().collect::<Vec<_>>())
            .flat_map(|ch| ch.to_lowercase().collect::<Vec<_>>())
            .collect(),
        backend_maps,
    })
}

fn load_english_rp_variety() -> Result<LanguageVariety, LanguageVarietyDataError> {
    let manifest: ManifestPack = toml::from_str(include_str!(
        "../../data/language-varieties/en-GB-RP/manifest.toml"
    ))
    .map_err(|source| LanguageVarietyDataError::Parse {
        path: "../../data/language-varieties/en-GB-RP/manifest.toml",
        source,
    })?;
    let phonology: PhonologyPack = toml::from_str(include_str!(
        "../../data/language-varieties/en-GB-RP/phonology.toml"
    ))
    .map_err(|source| LanguageVarietyDataError::Parse {
        path: "../../data/language-varieties/en-GB-RP/phonology.toml",
        source,
    })?;
    let mbrola_en1: BackendMapPack = toml::from_str(include_str!(
        "../../data/language-varieties/en-GB-RP/backend-maps/mbrola-en1.toml"
    ))
    .map_err(|source| LanguageVarietyDataError::Parse {
        path: "../../data/language-varieties/en-GB-RP/backend-maps/mbrola-en1.toml",
        source,
    })?;

    let mut backend_maps = HashMap::new();
    backend_maps.insert("mbrola-en1".to_string(), mbrola_en1.map);

    Ok(LanguageVariety {
        id: manifest.id,
        label: manifest.label,
        language: manifest.language,
        inventory_id: manifest.inventory_id,
        arpabet_vowels: phonology
            .arpabet_vowels
            .into_iter()
            .map(|value| value.to_ascii_uppercase())
            .collect(),
        nucleus_symbols: phonology.nucleus_symbols.into_iter().collect(),
        vowel_phone_chars: phonology
            .vowel_phone_chars
            .into_iter()
            .flat_map(|symbol| symbol.chars().collect::<Vec<_>>())
            .flat_map(|ch| ch.to_lowercase().collect::<Vec<_>>())
            .collect(),
        backend_maps,
    })
}

fn normalize_arpabet_symbol(symbol: &str) -> String {
    symbol
        .trim()
        .trim_matches(|ch: char| ch.is_ascii_digit() || ch == '"' || ch == '\'')
        .to_ascii_uppercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn en_us_datapack_loads_with_expected_inventory() {
        let variety = english_us_variety();
        assert_eq!(variety.id, "en-US");
        assert_eq!(variety.inventory_id, "en-US-GA");
        assert!(variety.is_arpabet_vowel("AY1"));
        assert!(variety.is_nucleus_symbol("ə"));
    }

    #[test]
    fn mbrola_maps_preserve_known_symbols() {
        let variety = english_us_variety();
        assert_eq!(
            variety
                .map_backend_symbol("mbrola-us1", "OW1")
                .expect("OW1 should map"),
            "oU"
        );
        assert_eq!(
            variety
                .map_backend_symbol("mbrola-us3", "ER1")
                .expect("ER1 should map"),
            "r="
        );
    }

    #[test]
    fn rp_datapack_maps_mbrola_en1_symbols() {
        let variety = english_rp_variety();
        assert_eq!(variety.id, "en-GB-RP");
        assert_eq!(variety.inventory_id, "en-GB-RP");
        assert_eq!(
            variety
                .map_backend_symbol("mbrola-en1", "OW1")
                .expect("OW1 should map"),
            "@U"
        );
    }

    #[test]
    fn unknown_phone_error_includes_variety_id() {
        let variety = english_us_variety();
        let error = variety
            .map_backend_symbol("mbrola-us1", "NOT_A_PHONE")
            .expect_err("unknown phone should fail");
        assert_eq!(
            error,
            LanguageVarietyLookupError::UnknownPhone {
                variety_id: "en-US".to_string(),
                backend_id: "mbrola-us1".to_string(),
                phone: "NOT_A_PHONE".to_string(),
            }
        );
        assert!(error.to_string().contains("en-US"));
    }
}
