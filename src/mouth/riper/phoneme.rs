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
}
