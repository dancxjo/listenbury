use anyhow::{Result, bail};

use crate::audio::frame::AudioFrame;
use crate::vocoder::{
    BackendCapabilities, BackendFamily, VocoderBackend, VocoderDescriptor, VocoderInput,
};

pub struct HifiganBackend;

impl HifiganBackend {
    pub fn descriptor() -> VocoderDescriptor {
        VocoderDescriptor {
            id: "hifigan",
            family: BackendFamily::NeuralVocoder,
            capabilities: BackendCapabilities {
                accepts_phone_timed: false,
                accepts_partial_prosody: false,
                accepts_coarse_text: false,
                accepts_mel: true,
                accepts_mel_f0: true,
                honors_explicit_duration: false,
                honors_explicit_f0: false,
                honors_vibrato: false,
                streaming_safe: false,
            },
            sample_rate_hz: 22_050,
            backend_kind: None,
            detail: None,
            notes: &["Future neural mel vocoder stub."],
        }
    }
}

impl VocoderBackend for HifiganBackend {
    fn id(&self) -> &'static str {
        Self::descriptor().id
    }

    fn descriptor(&self) -> VocoderDescriptor {
        Self::descriptor()
    }

    fn render(&mut self, _input: VocoderInput<'_>) -> Result<Vec<AudioFrame>> {
        bail!(
            "vocoder `hifigan` is registered but not implemented yet; expected Mel or MelF0 input, no ONNX model/runtime adapter is wired"
        )
    }
}
