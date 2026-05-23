use anyhow::{Result, bail};

use crate::audio::frame::AudioFrame;
use crate::vocoder::{
    BackendCapabilities, BackendFamily, VocoderBackend, VocoderDescriptor, VocoderInput,
};

pub struct DiffwaveBackend;

impl DiffwaveBackend {
    pub fn descriptor() -> VocoderDescriptor {
        VocoderDescriptor {
            id: "diffwave",
            family: BackendFamily::NeuralVocoder,
            capabilities: BackendCapabilities {
                accepts_phone_timed: false,
                accepts_partial_prosody: false,
                accepts_coarse_text: false,
                accepts_mel: true,
                accepts_mel_f0: false,
                honors_explicit_duration: false,
                honors_explicit_f0: false,
                honors_vibrato: false,
                streaming_safe: false,
            },
            sample_rate_hz: 22_050,
            backend_kind: None,
            detail: None,
            notes: &["Future diffusion vocoder stub."],
        }
    }
}

impl VocoderBackend for DiffwaveBackend {
    fn id(&self) -> &'static str {
        Self::descriptor().id
    }

    fn descriptor(&self) -> VocoderDescriptor {
        Self::descriptor()
    }

    fn render(&mut self, _input: VocoderInput<'_>) -> Result<Vec<AudioFrame>> {
        bail!(
            "vocoder `diffwave` is registered but not implemented yet; expected Mel input, no model/runtime adapter is wired"
        )
    }
}
