use std::collections::BTreeMap;

use thiserror::Error;

use crate::linguistic::english_us_language_pack;

/// Voice-specific mapping from Listenbury phone symbols to MBROLA symbols.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MbrolaSymbolMap {
    map: BTreeMap<String, String>,
}

impl MbrolaSymbolMap {
    pub fn new(map: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>) -> Self {
        Self {
            map: map
                .into_iter()
                .map(|(from, to)| (from.into(), to.into()))
                .collect(),
        }
    }

    /// A conservative US-English starter map. Real voices may need overrides.
    pub fn us1_starter() -> Self {
        let map = english_us_language_pack()
            .backend_map("mbrola-us1")
            .expect("en-US language pack should define mbrola-us1 backend map");
        Self::new(map.iter().map(|(from, to)| (from.clone(), to.clone())))
    }

    pub fn identity() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    /// Starter map for the `us3` voice inventory, whose diphthongs use
    /// uppercase SAMPA-style symbols like `EI`, `AI`, and `@U`.
    pub fn us3_starter() -> Self {
        let map = english_us_language_pack()
            .backend_map("mbrola-us3")
            .expect("en-US language pack should define mbrola-us3 backend map");
        Self::new(map.iter().map(|(from, to)| (from.clone(), to.clone())))
    }

    pub fn map_phone(&self, phone: &str) -> Result<String, UnmappedPhone> {
        if let Some(symbol) = self.map.get(phone) {
            return Ok(symbol.clone());
        }
        let stressless = phone.trim_end_matches(|ch: char| ch.is_ascii_digit());
        if stressless != phone
            && let Some(symbol) = self.map.get(stressless)
        {
            return Ok(symbol.clone());
        }
        if self.map.is_empty() {
            return Ok(phone.to_string());
        }
        Err(UnmappedPhone {
            phone: phone.to_string(),
        })
    }

    pub fn insert(&mut self, from: impl Into<String>, to: impl Into<String>) {
        self.map.insert(from.into(), to.into());
    }
}

impl Default for MbrolaSymbolMap {
    fn default() -> Self {
        Self::us1_starter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn us3_maps_cmu_r_colored_vowels_to_voice_symbol() {
        let map = MbrolaSymbolMap::us3_starter();

        assert_eq!(map.map_phone("ER").unwrap(), "r=");
        assert_eq!(map.map_phone("ER1").unwrap(), "r=");
        assert_eq!(map.map_phone("ɝ").unwrap(), "r=");
    }

    #[test]
    fn us1_maps_representative_phones_from_datapack() {
        let map = MbrolaSymbolMap::us1_starter();
        assert_eq!(map.map_phone("OW1").unwrap(), "oU");
        assert_eq!(map.map_phone("tʃ").unwrap(), "tS");
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("phone `{phone}` is not mapped for this MBROLA voice")]
pub struct UnmappedPhone {
    pub phone: String,
}
