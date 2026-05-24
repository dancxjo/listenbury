use anyhow::{Result, bail};

use crate::audio::frame::AudioFrame;
use crate::vocoder::{
    BackendCapabilities, BackendFamily, SpeechSynthesizer, VocoderDescriptor, VocoderInput,
};

pub struct NeuralSourceFilterBackend;

impl NeuralSourceFilterBackend {
    pub fn descriptor() -> VocoderDescriptor {
        VocoderDescriptor {
            id: "source-filter-neural",
            family: BackendFamily::NeuralSourceFilter,
            capabilities: BackendCapabilities {
                accepts_phone_timed: false,
                accepts_partial_prosody: true,
                accepts_coarse_text: false,
                accepts_mel: false,
                accepts_mel_f0: true,
                honors_explicit_duration: true,
                honors_explicit_f0: true,
                honors_vibrato: true,
                streaming_safe: false,
            },
            sample_rate_hz: 24_000,
            backend_kind: None,
            detail: None,
            notes: &["Future neural source-filter singing path stub."],
        }
    }
}

impl SpeechSynthesizer for NeuralSourceFilterBackend {
    fn id(&self) -> &'static str {
        Self::descriptor().id
    }

    fn descriptor(&self) -> VocoderDescriptor {
        Self::descriptor()
    }

    fn render(&mut self, _input: VocoderInput<'_>) -> Result<Vec<AudioFrame>> {
        bail!(
            "vocoder `source-filter-neural` is registered but not implemented yet; expected MelF0 or SourceFilter input, no model/runtime adapter is wired"
        )
    }
}
