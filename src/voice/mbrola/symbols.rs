use std::collections::BTreeMap;

use thiserror::Error;

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
        Self::new([
            ("_", "_"),
            ("pau", "_"),
            ("sil", "_"),
            ("h", "h"),
            ("l", "l"),
            ("m", "m"),
            ("b", "b"),
            ("i", "i"),
            ("iː", "i"),
            ("ɪ", "I"),
            ("ɛ", "E"),
            ("æ", "{"),
            ("ɑ", "A"),
            ("ɡ", "g"),
            ("ɔ", "O"),
            ("ʊ", "U"),
            ("u", "u"),
            ("uː", "u"),
            ("ʌ", "V"),
            ("ə", "@"),
            ("ɝ", "3"),
            ("ɚ", "3"),
            ("oʊ", "oU"),
            ("aɪ", "aI"),
            ("ɑɪ", "aI"),
            ("eɪ", "eI"),
            ("aʊ", "aU"),
            ("ɔɪ", "OI"),
            ("p", "p"),
            ("t", "t"),
            ("k", "k"),
            ("d", "d"),
            ("g", "g"),
            ("n", "n"),
            ("ŋ", "N"),
            ("f", "f"),
            ("v", "v"),
            ("θ", "T"),
            ("ð", "D"),
            ("s", "s"),
            ("z", "z"),
            ("ʃ", "S"),
            ("ʒ", "Z"),
            ("r", "r"),
            ("ɹ", "r"),
            ("w", "w"),
            ("j", "j"),
            ("tʃ", "tS"),
            ("dʒ", "dZ"),
            ("HH", "h"),
            ("AA", "A"),
            ("AE", "{"),
            ("AH", "@"),
            ("AO", "O"),
            ("AW", "aU"),
            ("AY", "aI"),
            ("EH", "E"),
            ("ER", "3"),
            ("EY", "eI"),
            ("IH", "I"),
            ("IY", "i"),
            ("OW", "oU"),
            ("OY", "OI"),
            ("UH", "U"),
            ("UW", "u"),
            ("AH0", "@"),
            ("AH1", "V"),
            ("AA1", "A"),
            ("AE1", "{"),
            ("AO1", "O"),
            ("EH1", "E"),
            ("IH1", "I"),
            ("UW1", "u"),
            ("OW1", "oU"),
            ("AY1", "aI"),
            ("EY1", "eI"),
            ("IY0", "i"),
            ("IY1", "i"),
            ("L", "l"),
            ("M", "m"),
            ("N", "n"),
            ("NG", "N"),
            ("B", "b"),
            ("P", "p"),
            ("T", "t"),
            ("D", "d"),
            ("K", "k"),
            ("G", "g"),
            ("F", "f"),
            ("V", "v"),
            ("TH", "T"),
            ("DH", "D"),
            ("S", "s"),
            ("Z", "z"),
            ("SH", "S"),
            ("ZH", "Z"),
            ("R", "r"),
            ("W", "w"),
            ("Y", "j"),
            ("CH", "tS"),
            ("JH", "dZ"),
        ])
    }

    pub fn identity() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    /// Starter map for the `us3` voice inventory, whose diphthongs use
    /// uppercase SAMPA-style symbols like `EI`, `AI`, and `@U`.
    pub fn us3_starter() -> Self {
        let mut map = Self::us1_starter();
        for (from, to) in [
            ("eɪ", "EI"),
            ("aɪ", "AI"),
            ("ɑɪ", "AI"),
            ("oʊ", "@U"),
            ("EY", "EI"),
            ("EY1", "EI"),
            ("EY0", "EI"),
            ("AY", "AI"),
            ("AY1", "AI"),
            ("AY0", "AI"),
            ("OW", "@U"),
            ("OW1", "@U"),
            ("OW0", "@U"),
            ("ɝ", "r="),
            ("ɚ", "r="),
            ("DX", "4"),
            ("ɾ", "4"),
        ] {
            map.insert(from, to);
        }
        map
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

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("phone `{phone}` is not mapped for this MBROLA voice")]
pub struct UnmappedPhone {
    pub phone: String,
}
