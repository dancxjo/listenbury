use thiserror::Error;

use crate::mouth::riper::config::PiperVoiceConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PiperPhoneme(pub String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PiperPhonemeSequence {
    pub phonemes: Vec<PiperPhoneme>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PiperIdSequence {
    pub ids: Vec<i64>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PiperPhonemeIdConversionError {
    #[error("unknown Piper phoneme symbol `{symbol}`")]
    UnknownPhoneme { symbol: String },
}

impl PiperPhonemeSequence {
    pub fn to_piper_ids(
        &self,
        config: &PiperVoiceConfig,
    ) -> Result<PiperIdSequence, PiperPhonemeIdConversionError> {
        let mut ids = Vec::new();
        for phoneme in &self.phonemes {
            let mapped = config.phoneme_id_map.get(&phoneme.0).ok_or_else(|| {
                PiperPhonemeIdConversionError::UnknownPhoneme {
                    symbol: phoneme.0.clone(),
                }
            })?;
            ids.extend(mapped);
        }
        Ok(PiperIdSequence { ids })
    }

    pub fn to_piper_ids_compatible(
        &self,
        config: &PiperVoiceConfig,
    ) -> Result<PiperIdSequence, PiperPhonemeIdConversionError> {
        self.to_piper_ids(config).or_else(|_| {
            espeak_compatible_sequence(self, config)
                .and_then(|sequence| sequence.to_piper_ids(config))
        })
    }
}

pub fn espeak_compatible_sequence(
    sequence: &PiperPhonemeSequence,
    config: &PiperVoiceConfig,
) -> Result<PiperPhonemeSequence, PiperPhonemeIdConversionError> {
    let mut symbols = vec![PiperPhoneme("^".to_string())];
    for phoneme in &sequence.phonemes {
        let expanded = expand_espeak_phoneme(&phoneme.0, config).ok_or_else(|| {
            PiperPhonemeIdConversionError::UnknownPhoneme {
                symbol: phoneme.0.clone(),
            }
        })?;
        symbols.extend(expanded.into_iter().map(PiperPhoneme));
    }
    symbols.push(PiperPhoneme("$".to_string()));

    let mut interspersed = Vec::with_capacity(symbols.len().saturating_mul(2).saturating_sub(1));
    for (index, symbol) in symbols.into_iter().enumerate() {
        if index > 0 {
            interspersed.push(PiperPhoneme("_".to_string()));
        }
        interspersed.push(symbol);
    }

    Ok(PiperPhonemeSequence {
        phonemes: interspersed,
    })
}

fn expand_espeak_phoneme(symbol: &str, config: &PiperVoiceConfig) -> Option<Vec<String>> {
    let expanded = match symbol {
        "AA" => &["ɑ"][..],
        "AH0" => &["ə"],
        "AH1" | "AH2" => &["ʌ"],
        "AH" => &["ə"],
        "AY" => &["a", "ɪ"],
        "AE" => &["æ"],
        "AO" => &["ɔ"],
        "AW" => &["a", "ʊ"],
        "B" => &["b"],
        "CH" => &["t", "ʃ"],
        "D" => &["d"],
        "DH" => &["ð"],
        "EH" => &["ɛ"],
        "ER" => &["ɚ"],
        "EY" => &["ˈ", "e", "ɪ"],
        "F" => &["f"],
        "G" => &["ɡ"],
        "HH" => &["h"],
        "IH" => &["ɪ"],
        "IY" => &["i"],
        "JH" => &["d", "ʒ"],
        "K" => &["k"],
        "L" => &["l"],
        "M" => &["m"],
        "N" => &["n"],
        "NG" => &["ŋ"],
        "OW" => &["o", "ʊ"],
        "OY" => &["ɔ", "ɪ"],
        "P" => &["p"],
        "R" => &["ɹ"],
        "S" => &["s"],
        "SH" => &["ʃ"],
        "T" => &["t"],
        "TH" => &["θ"],
        "TS" => &["t", "s"],
        "UH" => &["ʊ"],
        "UW" => &["u"],
        "V" => &["v"],
        "W" => &["w"],
        "Y" => &["j"],
        "Z" => &["z"],
        "ZH" => &["ʒ"],
        "|" => &["."],
        _ if config.phoneme_id_map.contains_key(symbol) => return Some(vec![symbol.to_string()]),
        _ => return None,
    };

    expanded
        .iter()
        .all(|symbol| config.phoneme_id_map.contains_key(*symbol))
        .then(|| {
            expanded
                .iter()
                .map(|symbol| (*symbol).to_string())
                .collect()
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_from_json(json: &str) -> PiperVoiceConfig {
        PiperVoiceConfig::from_json_str(json).expect("voice config should parse")
    }

    fn sequence(symbols: &[&str]) -> PiperPhonemeSequence {
        PiperPhonemeSequence {
            phonemes: symbols
                .iter()
                .map(|symbol| PiperPhoneme((*symbol).to_string()))
                .collect(),
        }
    }

    #[test]
    fn converts_single_id_mappings() {
        let config = config_from_json(
            r#"
            {
              "audio": { "sample_rate": 22050 },
              "phoneme_id_map": {
                "a": [1],
                "t": [2]
              }
            }
            "#,
        );
        let ids = sequence(&["a", "t"])
            .to_piper_ids(&config)
            .expect("known symbols should convert");
        assert_eq!(ids, PiperIdSequence { ids: vec![1, 2] });
    }

    #[test]
    fn flattens_multi_id_mappings_in_order() {
        let config = config_from_json(
            r#"
            {
              "audio": { "sample_rate": 22050 },
              "phoneme_id_map": {
                "a": [1, 2],
                "t": [3]
              }
            }
            "#,
        );
        let ids = sequence(&["a", "t"])
            .to_piper_ids(&config)
            .expect("known symbols should convert");
        assert_eq!(ids, PiperIdSequence { ids: vec![1, 2, 3] });
    }

    #[test]
    fn supports_separator_symbols() {
        let config = config_from_json(
            r#"
            {
              "audio": { "sample_rate": 22050 },
              "phoneme_id_map": {
                "a": [1],
                " ": [3],
                "t": [2]
              }
            }
            "#,
        );
        let ids = sequence(&["a", " ", "t"])
            .to_piper_ids(&config)
            .expect("known symbols should convert");
        assert_eq!(ids, PiperIdSequence { ids: vec![1, 3, 2] });
    }

    #[test]
    fn returns_clear_error_for_unknown_phoneme() {
        let config = config_from_json(
            r#"
            {
              "audio": { "sample_rate": 22050 },
              "phoneme_id_map": {
                "a": [1]
              }
            }
            "#,
        );
        let error = sequence(&["a", "z"])
            .to_piper_ids(&config)
            .expect_err("unknown symbol should return an error");
        assert_eq!(
            error,
            PiperPhonemeIdConversionError::UnknownPhoneme {
                symbol: "z".to_string()
            }
        );
        assert_eq!(error.to_string(), "unknown Piper phoneme symbol `z`");
    }

    #[test]
    fn empty_sequence_returns_empty_ids() {
        let config = config_from_json(
            r#"
            {
              "audio": { "sample_rate": 22050 },
              "phoneme_id_map": {
                "a": [1]
              }
            }
            "#,
        );
        let ids = sequence(&[])
            .to_piper_ids(&config)
            .expect("empty sequence should convert");
        assert_eq!(ids, PiperIdSequence { ids: Vec::new() });
    }

    #[test]
    fn compatible_conversion_expands_ah_stress_for_espeak_voice_maps() {
        let config = config_from_json(
            r#"
            {
              "audio": { "sample_rate": 22050 },
              "phoneme_id_map": {
                "^": [1],
                "_": [2],
                "$": [3],
                "ə": [4],
                "ʌ": [5]
              }
            }
            "#,
        );
        let ids = sequence(&["AH0", "AH1", "AH2"])
            .to_piper_ids_compatible(&config)
            .expect("stress-specific AH symbols should expand for eSpeak maps");
        assert_eq!(
            ids,
            PiperIdSequence {
                ids: vec![1, 2, 4, 2, 5, 2, 5, 2, 3]
            }
        );
    }
}
