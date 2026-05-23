use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::hearing::breath::{BreathGroupConfig, DEFAULT_VAD_FRAME_MS};
use crate::hearing::vad::VadBackendKind;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListenburyConfig {
    #[serde(default)]
    pub breath_group: BreathGroupConfigSerde,
    #[serde(default)]
    pub vad: Option<VadProfile>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BreathGroupConfigSerde {
    pub open_after_speech_frames: usize,
    pub close_after_silence_frames: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct VadProfile {
    pub backend: VadBackendKind,
    pub rms_threshold: f32,
    pub hangover_ms: u64,
    pub min_speech_ms: u64,
    pub noise_floor: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VadProfileToml {
    vad: VadProfile,
}

impl Default for ListenburyConfig {
    fn default() -> Self {
        let breath = BreathGroupConfig::default();
        Self {
            breath_group: BreathGroupConfigSerde {
                open_after_speech_frames: breath.open_after_speech_frames,
                close_after_silence_frames: breath.close_after_silence_frames,
            },
            vad: None,
        }
    }
}

impl Default for BreathGroupConfigSerde {
    fn default() -> Self {
        let breath = BreathGroupConfig::default();
        Self {
            open_after_speech_frames: breath.open_after_speech_frames,
            close_after_silence_frames: breath.close_after_silence_frames,
        }
    }
}

impl VadProfile {
    pub fn read_toml(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let toml = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read VAD profile at {}", path.display()))?;
        Self::from_toml_str(&toml)
            .with_context(|| format!("failed to parse VAD profile at {}", path.display()))
    }

    pub fn from_toml_str(toml: &str) -> Result<Self> {
        let profile = toml::from_str::<VadProfileToml>(toml)?.vad;
        profile.validate()?;
        Ok(profile)
    }

    pub fn to_toml(self) -> String {
        format!(
            "[vad]\nbackend = \"{}\"\nrms_threshold = {:.6}\nhangover_ms = {}\nmin_speech_ms = {}\nnoise_floor = {:.6}\n",
            self.backend.as_str(),
            self.rms_threshold,
            self.hangover_ms,
            self.min_speech_ms,
            self.noise_floor
        )
    }

    pub fn breath_group_config(self) -> BreathGroupConfig {
        BreathGroupConfig {
            open_after_speech_frames: frames_for_duration_ms(
                self.min_speech_ms,
                DEFAULT_VAD_FRAME_MS,
            )
            .max(1),
            close_after_silence_frames: frames_for_duration_ms(
                self.hangover_ms,
                DEFAULT_VAD_FRAME_MS,
            )
            .max(1),
            max_group_frames: None,
        }
    }

    fn validate(self) -> Result<()> {
        anyhow::ensure!(
            self.rms_threshold.is_finite() && self.rms_threshold > 0.0,
            "VAD profile rms_threshold must be finite and greater than zero"
        );
        anyhow::ensure!(
            self.noise_floor.is_finite() && self.noise_floor >= 0.0,
            "VAD profile noise_floor must be finite and non-negative"
        );
        anyhow::ensure!(
            self.hangover_ms > 0,
            "VAD profile hangover_ms must be greater than zero"
        );
        anyhow::ensure!(
            self.min_speech_ms > 0,
            "VAD profile min_speech_ms must be greater than zero"
        );
        Ok(())
    }
}

const fn frames_for_duration_ms(duration_ms: u64, frame_ms: u64) -> usize {
    if duration_ms == 0 || frame_ms == 0 {
        return 0;
    }
    duration_ms.div_ceil(frame_ms) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vad_profile_round_trips_calibration_toml() {
        let profile = VadProfile::from_toml_str(
            "[vad]\nbackend = \"energy\"\nrms_threshold = 0.069579\nhangover_ms = 180\nmin_speech_ms = 120\nnoise_floor = 0.032066\n",
        )
        .expect("profile should parse");

        assert_eq!(profile.backend, VadBackendKind::Energy);
        assert_eq!(profile.breath_group_config().open_after_speech_frames, 12);
        assert_eq!(profile.breath_group_config().close_after_silence_frames, 18);
        assert!(profile.to_toml().contains("rms_threshold = 0.069579"));
    }
}
