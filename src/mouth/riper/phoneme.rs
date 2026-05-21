use thiserror::Error;

use crate::mouth::riper::config::PiperVoiceConfig;

const PIPER_PAD: &str = "_";
const PIPER_BOS: &str = "^";
const PIPER_EOS: &str = "$";
const PIPER_WORD_SEPARATOR: &str = " ";

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
        if config_has_piper_framing(config) {
            return self.to_piper_framed_ids(config).or_else(|_| {
                espeak_compatible_sequence(self, config)
                    .and_then(|sequence| sequence.to_piper_framed_ids(config))
            });
        }

        self.to_piper_ids(config).or_else(|_| {
            espeak_compatible_sequence(self, config)
                .and_then(|sequence| sequence.to_piper_ids(config))
        })
    }

    fn to_piper_framed_ids(
        &self,
        config: &PiperVoiceConfig,
    ) -> Result<PiperIdSequence, PiperPhonemeIdConversionError> {
        let mut ids = Vec::new();
        extend_symbol_ids(&mut ids, PIPER_BOS, config)?;
        extend_symbol_ids(&mut ids, PIPER_PAD, config)?;
        for phoneme in &self.phonemes {
            if phoneme.0 == PIPER_WORD_SEPARATOR {
                continue;
            }
            extend_symbol_ids(&mut ids, &phoneme.0, config)?;
            extend_symbol_ids(&mut ids, PIPER_PAD, config)?;
        }
        extend_symbol_ids(&mut ids, PIPER_EOS, config)?;
        Ok(PiperIdSequence { ids })
    }
}

pub fn espeak_compatible_sequence(
    sequence: &PiperPhonemeSequence,
    config: &PiperVoiceConfig,
) -> Result<PiperPhonemeSequence, PiperPhonemeIdConversionError> {
    let mut symbols = Vec::new();
    for phoneme in &sequence.phonemes {
        let expanded = expand_espeak_phoneme(&phoneme.0, config).ok_or_else(|| {
            PiperPhonemeIdConversionError::UnknownPhoneme {
                symbol: phoneme.0.clone(),
            }
        })?;
        symbols.extend(expanded.into_iter().map(PiperPhoneme));
    }
    Ok(PiperPhonemeSequence { phonemes: symbols })
}

fn config_has_piper_framing(config: &PiperVoiceConfig) -> bool {
    [PIPER_BOS, PIPER_PAD, PIPER_EOS]
        .iter()
        .all(|symbol| config.phoneme_id_map.contains_key(*symbol))
}

fn extend_symbol_ids(
    ids: &mut Vec<i64>,
    symbol: &str,
    config: &PiperVoiceConfig,
) -> Result<(), PiperPhonemeIdConversionError> {
    let mapped = config.phoneme_id_map.get(symbol).ok_or_else(|| {
        PiperPhonemeIdConversionError::UnknownPhoneme {
            symbol: symbol.to_string(),
        }
    })?;
    ids.extend(mapped);
    Ok(())
}

fn expand_espeak_phoneme(symbol: &str, config: &PiperVoiceConfig) -> Option<Vec<String>> {
    let stress_marker = match symbol.chars().next_back() {
        Some('1') => Some("ˈ"),
        Some('2') => Some("ˌ"),
        _ => None,
    };
    let base_symbol = symbol
        .strip_suffix(['0', '1', '2'])
        .filter(|base| is_arpabet_vowel(base))
        .unwrap_or(symbol);

    let expanded = match (symbol, base_symbol) {
        (PIPER_WORD_SEPARATOR, _) => &[][..],
        ("AH0", _) => &["ə"][..],
        ("AH1" | "AH2", _) => &["ʌ"],
        (_, "AA") => &["ɑ"],
        (_, "AH") => &["ə"],
        (_, "AY") => &["a", "ɪ"],
        (_, "AE") => &["æ"],
        (_, "AO") => &["ɔ"],
        (_, "AW") => &["a", "ʊ"],
        (_, "B") => &["b"],
        (_, "CH") => &["t", "ʃ"],
        (_, "D") => &["d"],
        (_, "DH") => &["ð"],
        (_, "EH") => &["ɛ"],
        (_, "ER") => &["ɚ"],
        (_, "EY") => &["e", "ɪ"],
        (_, "F") => &["f"],
        (_, "G") => &["ɡ"],
        (_, "HH") => &["h"],
        (_, "IH") => &["ɪ"],
        (_, "IY") => &["i"],
        (_, "JH") => &["d", "ʒ"],
        (_, "K") => &["k"],
        (_, "L") => &["l"],
        (_, "M") => &["m"],
        (_, "N") => &["n"],
        (_, "NG") => &["ŋ"],
        (_, "OW") => &["o", "ʊ"],
        (_, "OY") => &["ɔ", "ɪ"],
        (_, "P") => &["p"],
        (_, "R") => &["ɹ"],
        (_, "S") => &["s"],
        (_, "SH") => &["ʃ"],
        (_, "T") => &["t"],
        (_, "TH") => &["θ"],
        (_, "TS") => &["t", "s"],
        (_, "UH") => &["ʊ"],
        (_, "UW") => &["u"],
        (_, "V") => &["v"],
        (_, "W") => &["w"],
        (_, "Y") => &["j"],
        (_, "Z") => &["z"],
        (_, "ZH") => &["ʒ"],
        (_, "|") => &["."],
        _ if config.phoneme_id_map.contains_key(symbol) => return Some(vec![symbol.to_string()]),
        _ => return None,
    };

    if !expanded
        .iter()
        .all(|symbol| config.phoneme_id_map.contains_key(*symbol))
    {
        return None;
    }

    let mut output = Vec::new();
    if let Some(marker) = stress_marker
        && config.phoneme_id_map.contains_key(marker)
    {
        output.push(marker.to_string());
    }
    output.extend(expanded.iter().map(|symbol| (*symbol).to_string()));
    Some(output)
}

fn is_arpabet_vowel(symbol: &str) -> bool {
    matches!(
        symbol,
        "AA" | "AE"
            | "AH"
            | "AO"
            | "AW"
            | "AY"
            | "EH"
            | "ER"
            | "EY"
            | "IH"
            | "IY"
            | "OW"
            | "OY"
            | "UH"
            | "UW"
    )
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
    fn compatible_sequence_expands_without_piper_padding_or_word_separator_tokens() {
        let config = config_from_json(
            r#"
            {
              "audio": { "sample_rate": 22050 },
              "phoneme_id_map": {
                "^": [1],
                "_": [2],
                "$": [3],
                " ": [4],
                ".": [5],
                "a": [6],
                "ɪ": [7],
                "s": [8],
                "i": [9]
              }
            }
            "#,
        );

        let compatible =
            espeak_compatible_sequence(&sequence(&["AY", " ", "S", "IY", "|"]), &config)
                .expect("ARPAbet symbols should expand to Piper codepoints");

        assert_eq!(sequence_symbols(&compatible), vec!["a", "ɪ", "s", "i", "."]);
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

    #[test]
    fn compatible_conversion_frames_already_compatible_espeak_symbols() {
        let config = config_from_json(
            r#"
            {
              "audio": { "sample_rate": 22050 },
              "phoneme_id_map": {
                "^": [1],
                "_": [2],
                "$": [3],
                " ": [4],
                ".": [5],
                "ð": [6],
                "ɪ": [7],
                "s": [8]
              }
            }
            "#,
        );

        let ids = sequence(&["ð", "ɪ", "s", " "])
            .to_piper_ids_compatible(&config)
            .expect("already compatible eSpeak symbols should still get Piper framing");

        assert_eq!(
            ids,
            PiperIdSequence {
                ids: vec![1, 2, 6, 2, 7, 2, 8, 2, 3]
            }
        );
    }

    #[test]
    fn compatible_conversion_expands_stressed_non_ah_vowels_for_espeak_voice_maps() {
        let config = config_from_json(
            r#"
            {
              "audio": { "sample_rate": 22050 },
              "phoneme_id_map": {
                "^": [1],
                "_": [2],
                "$": [3],
                "ɛ": [4],
                "ɪ": [5],
                "ɑ": [6],
                "a": [7],
                "ˈ": [8],
                "ˌ": [9]
              }
            }
            "#,
        );
        let ids = sequence(&["EH0", "IH0", "AA1", "AY2"])
            .to_piper_ids_compatible(&config)
            .expect("stress-specific non-AH vowels should expand for eSpeak maps");
        assert_eq!(
            ids,
            PiperIdSequence {
                ids: vec![1, 2, 4, 2, 5, 2, 8, 2, 6, 2, 9, 2, 7, 2, 5, 2, 3]
            }
        );
    }

    fn sequence_symbols(sequence: &PiperPhonemeSequence) -> Vec<&str> {
        sequence
            .phonemes
            .iter()
            .map(|phoneme| phoneme.0.as_str())
            .collect()
    }
}
