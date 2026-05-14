use serde::{Deserialize, Serialize};

use crate::hearing::breath::BreathGroupConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListenburyConfig {
    pub breath_group: BreathGroupConfigSerde,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BreathGroupConfigSerde {
    pub open_after_speech_frames: usize,
    pub close_after_silence_frames: usize,
}

impl Default for ListenburyConfig {
    fn default() -> Self {
        let breath = BreathGroupConfig::default();
        Self {
            breath_group: BreathGroupConfigSerde {
                open_after_speech_frames: breath.open_after_speech_frames,
                close_after_silence_frames: breath.close_after_silence_frames,
            },
        }
    }
}
