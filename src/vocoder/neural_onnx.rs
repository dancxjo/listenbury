use anyhow::{Result, bail};

use crate::audio::frame::AudioFrame;
use crate::vocoder::{
    BackendCapabilities, BackendFamily, VocoderBackend, VocoderDescriptor, VocoderInput,
};

pub struct RiperOnnxDirectBackend;

impl RiperOnnxDirectBackend {
    pub fn descriptor() -> VocoderDescriptor {
        VocoderDescriptor {
            id: "riper-onnx-direct",
            family: BackendFamily::Placeholder,
            capabilities: BackendCapabilities {
                accepts_phone_timed: false,
                accepts_partial_prosody: true,
                accepts_coarse_text: false,
                accepts_mel: false,
                accepts_mel_f0: false,
                honors_explicit_duration: false,
                honors_explicit_f0: false,
                honors_vibrato: false,
                streaming_safe: false,
            },
            sample_rate_hz: 24_000,
            backend_kind: None,
            detail: None,
            notes: &["Direct Riper ONNX control surface is a compile-safe placeholder."],
        }
    }
}

impl VocoderBackend for RiperOnnxDirectBackend {
    fn id(&self) -> &'static str {
        Self::descriptor().id
    }

    fn descriptor(&self) -> VocoderDescriptor {
        Self::descriptor()
    }

    fn render(&mut self, _input: VocoderInput<'_>) -> Result<Vec<AudioFrame>> {
        bail!(
            "vocoder `riper-onnx-direct` is registered but not implemented yet; accepts PartialProsody input, but direct F0/duration controls are not wired"
        )
    }
}
