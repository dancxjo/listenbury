//! Shared stretch-synth backend interface for sung rendering.
//!
//! The module is called `vocoder` for continuity, but it intentionally covers
//! formant/source-filter and diphone renderers alongside neural vocoder stubs.

mod bigvgan;
mod capability;
mod diffwave;
mod hifigan;
mod input;
mod klatt;
mod mbrola;
mod neural_onnx;
mod piper;
mod registry;
mod riper;
mod source_filter;

use anyhow::Result;

use crate::audio::frame::AudioFrame;

pub use capability::{BackendCapabilities, BackendFamily, VocoderDescriptor};
pub use hifigan::HifiganBackend;
pub use input::{MelConfig, MelFrame, MelScale, MelSpectrogram, MelTensorLayout, VocoderInput};
pub use registry::{
    SingDemoBackendSelector, VocoderConfig, backend_by_id, backend_for_option, list_backends,
};

pub trait VocoderBackend {
    fn id(&self) -> &'static str;
    fn descriptor(&self) -> VocoderDescriptor;
    fn render(&mut self, input: VocoderInput<'_>) -> Result<Vec<AudioFrame>>;
}
