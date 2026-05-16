use std::collections::HashMap;

use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub struct PiperVoiceConfig {
    pub sample_rate_hz: u32,
    pub phoneme_id_map: HashMap<String, Vec<i64>>,
    pub num_speakers: Option<u32>,
    pub speaker_id_map: HashMap<String, u32>,
    pub length_scale: Option<f32>,
    pub noise_scale: Option<f32>,
    pub noise_w: Option<f32>,
    pub model_metadata: HashMap<String, String>,
}

#[derive(Debug, Error)]
pub enum PiperVoiceConfigError {
    #[error("failed to parse Piper voice config JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),
    #[error("missing required Piper voice config field `{field}`")]
    MissingField { field: &'static str },
    #[error("invalid Piper voice config field `{field}`: {reason}")]
    InvalidField {
        field: &'static str,
        reason: String,
    },
}

impl PiperVoiceConfig {
    pub fn from_json_str(json: &str) -> Result<Self, PiperVoiceConfigError> {
        let value: Value = serde_json::from_str(json)?;
        Self::from_value(&value)
    }

    pub fn from_value(value: &Value) -> Result<Self, PiperVoiceConfigError> {
        let sample_rate_hz = parse_required_u32(
            value,
            &[&["audio", "sample_rate"], &["sample_rate"]],
            "audio.sample_rate",
        )?;
        let phoneme_id_map = parse_phoneme_id_map(
            find_value(value, &[&["phoneme_id_map"], &["phoneme_map"]]).ok_or(
                PiperVoiceConfigError::MissingField {
                    field: "phoneme_id_map",
                },
            )?,
        )?;
        let speaker_id_map = find_value(value, &[&["speaker_id_map"], &["speaker_map"]])
            .map(parse_speaker_id_map)
            .transpose()?
            .unwrap_or_default();
        let num_speakers = parse_optional_u32(
            value,
            &[&["num_speakers"], &["speaker_count"]],
            "num_speakers",
        )?
        .or_else(|| {
            if speaker_id_map.is_empty() {
                None
            } else {
                u32::try_from(speaker_id_map.len()).ok()
            }
        });

        Ok(Self {
            sample_rate_hz,
            phoneme_id_map,
            num_speakers,
            speaker_id_map,
            length_scale: parse_optional_f32(
                value,
                &[&["inference", "length_scale"], &["length_scale"]],
                "inference.length_scale",
            )?,
            noise_scale: parse_optional_f32(
                value,
                &[&["inference", "noise_scale"], &["noise_scale"]],
                "inference.noise_scale",
            )?,
            noise_w: parse_optional_f32(
                value,
                &[&["inference", "noise_w"], &["noise_w"]],
                "inference.noise_w",
            )?,
            model_metadata: collect_metadata(value),
        })
    }
}

fn find_value<'a>(root: &'a Value, paths: &[&[&str]]) -> Option<&'a Value> {
    paths.iter().find_map(|path| {
        let mut current = root;
        for segment in *path {
            current = current.get(*segment)?;
        }
        Some(current)
    })
}

fn parse_required_u32(
    root: &Value,
    paths: &[&[&str]],
    field: &'static str,
) -> Result<u32, PiperVoiceConfigError> {
    let value = find_value(root, paths).ok_or(PiperVoiceConfigError::MissingField { field })?;
    parse_u32(value, field)
}

fn parse_optional_u32(
    root: &Value,
    paths: &[&[&str]],
    field: &'static str,
) -> Result<Option<u32>, PiperVoiceConfigError> {
    find_value(root, paths)
        .map(|value| parse_u32(value, field))
        .transpose()
}

fn parse_u32(value: &Value, field: &'static str) -> Result<u32, PiperVoiceConfigError> {
    let number = value
        .as_u64()
        .ok_or_else(|| invalid_field(field, "expected an unsigned integer"))?;
    u32::try_from(number).map_err(|_| invalid_field(field, "value exceeds u32 range"))
}

fn parse_optional_f32(
    root: &Value,
    paths: &[&[&str]],
    field: &'static str,
) -> Result<Option<f32>, PiperVoiceConfigError> {
    find_value(root, paths)
        .map(|value| parse_f32(value, field))
        .transpose()
}

fn parse_f32(value: &Value, field: &'static str) -> Result<f32, PiperVoiceConfigError> {
    let number = value
        .as_f64()
        .ok_or_else(|| invalid_field(field, "expected a number"))?;
    if !number.is_finite() || number < f32::MIN as f64 || number > f32::MAX as f64 {
        return Err(invalid_field(field, "value is out of f32 range"));
    }
    Ok(number as f32)
}

fn parse_phoneme_id_map(
    value: &Value,
) -> Result<HashMap<String, Vec<i64>>, PiperVoiceConfigError> {
    let entries = value
        .as_object()
        .ok_or_else(|| invalid_field("phoneme_id_map", "expected an object"))?;
    let mut phoneme_id_map = HashMap::with_capacity(entries.len());
    for (phoneme, ids) in entries {
        let ids = match ids {
            Value::Array(values) => values
                .iter()
                .map(|value| parse_i64(value, "phoneme_id_map"))
                .collect::<Result<Vec<_>, _>>()?,
            _ => vec![parse_i64(ids, "phoneme_id_map")?],
        };
        phoneme_id_map.insert(phoneme.clone(), ids);
    }
    Ok(phoneme_id_map)
}

fn parse_speaker_id_map(value: &Value) -> Result<HashMap<String, u32>, PiperVoiceConfigError> {
    let entries = value
        .as_object()
        .ok_or_else(|| invalid_field("speaker_id_map", "expected an object"))?;
    let mut speaker_id_map = HashMap::with_capacity(entries.len());
    for (speaker, id) in entries {
        speaker_id_map.insert(speaker.clone(), parse_u32(id, "speaker_id_map")?);
    }
    Ok(speaker_id_map)
}

fn parse_i64(value: &Value, field: &'static str) -> Result<i64, PiperVoiceConfigError> {
    value
        .as_i64()
        .ok_or_else(|| invalid_field(field, "expected an integer"))
}

fn collect_metadata(root: &Value) -> HashMap<String, String> {
    let mut metadata = HashMap::new();
    for section in ["model", "voice", "metadata"] {
        let Some(Value::Object(values)) = root.get(section) else {
            continue;
        };
        for (key, value) in values {
            if let Some(value) = metadata_scalar(value) {
                metadata.insert(format!("{section}.{key}"), value);
            }
        }
    }
    metadata
}

fn metadata_scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn invalid_field(field: &'static str, reason: impl Into<String>) -> PiperVoiceConfigError {
    PiperVoiceConfigError::InvalidField {
        field,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_voice_config_core_fields_and_metadata() {
        let config = PiperVoiceConfig::from_json_str(
            r#"
            {
              "audio": { "sample_rate": 22050 },
              "phoneme_id_map": {
                "_": [0],
                "a": [1, 2],
                "sil": 3
              },
              "num_speakers": 2,
              "speaker_id_map": {
                "alice": 0,
                "bob": 1
              },
              "inference": {
                "length_scale": 1.1,
                "noise_scale": 0.667,
                "noise_w": 0.8
              },
              "model": {
                "name": "Test Voice",
                "version": 1
              },
              "voice": {
                "language": "en_US"
              }
            }
            "#,
        )
        .expect("voice config should parse");

        assert_eq!(config.sample_rate_hz, 22_050);
        assert_eq!(config.phoneme_id_map["_"], vec![0]);
        assert_eq!(config.phoneme_id_map["a"], vec![1, 2]);
        assert_eq!(config.phoneme_id_map["sil"], vec![3]);
        assert_eq!(config.num_speakers, Some(2));
        assert_eq!(config.speaker_id_map["alice"], 0);
        assert_eq!(config.speaker_id_map["bob"], 1);
        assert_eq!(config.length_scale, Some(1.1));
        assert_eq!(config.noise_scale, Some(0.667));
        assert_eq!(config.noise_w, Some(0.8));
        assert_eq!(
            config.model_metadata.get("model.name"),
            Some(&"Test Voice".to_string())
        );
        assert_eq!(
            config.model_metadata.get("voice.language"),
            Some(&"en_US".to_string())
        );
    }

    #[test]
    fn infers_speaker_count_from_speaker_id_map() {
        let config = PiperVoiceConfig::from_json_str(
            r#"
            {
              "sample_rate": 16000,
              "phoneme_map": {
                "a": [1]
              },
              "speaker_id_map": {
                "narrator": 0,
                "assistant": 1,
                "guest": 2
              }
            }
            "#,
        )
        .expect("voice config should parse");

        assert_eq!(config.sample_rate_hz, 16_000);
        assert_eq!(config.num_speakers, Some(3));
    }

    #[test]
    fn missing_required_fields_return_clear_errors() {
        let missing_sample_rate = PiperVoiceConfig::from_json_str(
            r#"
            {
              "phoneme_id_map": {
                "a": [1]
              }
            }
            "#,
        )
        .expect_err("sample rate is required");
        assert_eq!(
            missing_sample_rate.to_string(),
            "missing required Piper voice config field `audio.sample_rate`"
        );

        let missing_phoneme_map = PiperVoiceConfig::from_json_str(
            r#"
            {
              "audio": { "sample_rate": 22050 }
            }
            "#,
        )
        .expect_err("phoneme map is required");
        assert_eq!(
            missing_phoneme_map.to_string(),
            "missing required Piper voice config field `phoneme_id_map`"
        );
    }
}
