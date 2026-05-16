use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use ort::session::Session;

use crate::audio::frame::AudioFrame;
use crate::mouth::backend::TtsBackend;

use super::PiperVoiceConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PiperModelContract {
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
}

#[derive(Debug)]
pub struct NativePiperBackend {
    config: PiperVoiceConfig,
    model_path: PathBuf,
    session: Option<Session>,
}

impl NativePiperBackend {
    pub fn load(model_path: impl AsRef<Path>, config: PiperVoiceConfig) -> Result<Self> {
        validate_config(&config)?;

        let model_path = model_path.as_ref().to_path_buf();
        if !model_path.is_file() {
            bail!(
                "Piper ONNX model file not found at {}",
                model_path.display()
            );
        }

        let mut session_builder =
            Session::builder().context("failed to create Piper ONNX session builder")?;
        let session = session_builder
            .commit_from_file(&model_path)
            .with_context(|| {
                format!(
                    "failed to load Piper ONNX model from {}",
                    model_path.display()
                )
            })?;

        Ok(Self {
            config,
            model_path,
            session: Some(session),
        })
    }

    pub fn validate_model_contract(&self) -> Result<PiperModelContract> {
        let session = self
            .session
            .as_ref()
            .context("Piper ONNX session has not been loaded")?;

        let input_names = session
            .inputs()
            .iter()
            .map(|input| input.name().to_string())
            .collect::<Vec<_>>();
        let output_names = session
            .outputs()
            .iter()
            .map(|output| output.name().to_string())
            .collect::<Vec<_>>();

        ensure!(
            !input_names.is_empty(),
            "Piper ONNX model `{}` exposes no inputs",
            self.model_path.display()
        );
        ensure!(
            !output_names.is_empty(),
            "Piper ONNX model `{}` exposes no outputs",
            self.model_path.display()
        );
        ensure!(
            input_names.iter().all(|name| !name.trim().is_empty()),
            "Piper ONNX model `{}` has an unnamed input",
            self.model_path.display()
        );
        ensure!(
            output_names.iter().all(|name| !name.trim().is_empty()),
            "Piper ONNX model `{}` has an unnamed output",
            self.model_path.display()
        );

        Ok(PiperModelContract {
            input_names,
            output_names,
        })
    }

    pub fn config(&self) -> &PiperVoiceConfig {
        &self.config
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    #[cfg(test)]
    fn unloaded_for_tests(model_path: PathBuf, config: PiperVoiceConfig) -> Self {
        Self {
            config,
            model_path,
            session: None,
        }
    }
}

impl TtsBackend for NativePiperBackend {
    fn synthesize(&mut self, _text: &str) -> Result<Vec<AudioFrame>> {
        bail!(
            "Native Piper synthesis is not implemented yet for {}",
            self.model_path.display()
        );
    }
}

fn validate_config(config: &PiperVoiceConfig) -> Result<()> {
    ensure!(
        config.sample_rate_hz > 0,
        "missing required Piper voice config field `audio.sample_rate`"
    );
    ensure!(
        !config.phoneme_id_map.is_empty(),
        "missing required Piper voice config field `phoneme_id_map`"
    );
    if let Some(num_speakers) = config.num_speakers {
        ensure!(
            num_speakers > 0,
            "invalid Piper voice config field `num_speakers`: expected a value greater than zero"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;

    fn voice_config() -> PiperVoiceConfig {
        PiperVoiceConfig {
            sample_rate_hz: 22_050,
            phoneme_id_map: HashMap::from([("a".to_string(), vec![1])]),
            num_speakers: None,
            speaker_id_map: HashMap::new(),
            length_scale: None,
            noise_scale: None,
            noise_w: None,
            model_metadata: HashMap::new(),
        }
    }

    fn unique_path(label: &str) -> PathBuf {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should advance")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "listenbury-native-piper-{label}-{}-{ts}.onnx",
            std::process::id()
        ))
    }

    #[test]
    fn load_returns_clear_error_for_missing_model_file() {
        let model_path = unique_path("missing-model");
        let error = NativePiperBackend::load(&model_path, voice_config())
            .expect_err("missing model should fail");

        assert_eq!(
            error.to_string(),
            format!(
                "Piper ONNX model file not found at {}",
                model_path.display()
            )
        );
    }

    #[test]
    fn load_rejects_missing_phoneme_map_before_session_creation() {
        let model_path = unique_path("config-error");
        fs::write(&model_path, b"placeholder").expect("placeholder model file");

        let mut config = voice_config();
        config.phoneme_id_map.clear();

        let error = NativePiperBackend::load(&model_path, config)
            .expect_err("empty phoneme map should fail");
        assert_eq!(
            error.to_string(),
            "missing required Piper voice config field `phoneme_id_map`"
        );

        let _ = fs::remove_file(model_path);
    }

    #[test]
    fn synthesize_returns_clear_unimplemented_error() {
        let model_path = unique_path("unimplemented");
        let mut backend =
            NativePiperBackend::unloaded_for_tests(model_path.clone(), voice_config());

        let error = backend
            .synthesize("hello")
            .expect_err("native synthesis is not implemented");
        assert_eq!(
            error.to_string(),
            format!(
                "Native Piper synthesis is not implemented yet for {}",
                model_path.display()
            )
        );
    }
}
