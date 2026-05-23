use anyhow::{Context, Result, bail};

use crate::audio::frame::AudioFrame;
use crate::vocoder::{BackendFamily, VocoderBackend, VocoderDescriptor, VocoderInput};
use crate::voice::articulator::{SungBackendDetail, SungBackendKind};

use super::klatt::KlattBackend;

pub struct RiperKlattFallbackBackend {
    inner: KlattBackend,
}

impl RiperKlattFallbackBackend {
    pub fn new() -> Self {
        Self {
            inner: KlattBackend,
        }
    }

    pub fn descriptor() -> VocoderDescriptor {
        let mut capabilities = KlattBackend::descriptor().capabilities;
        capabilities.accepts_phone_timed = true;
        VocoderDescriptor {
            id: "riper-klatt-fallback",
            family: BackendFamily::Placeholder,
            capabilities,
            sample_rate_hz: KlattBackend::descriptor().sample_rate_hz,
            backend_kind: Some(SungBackendKind::RiperKlattFallback),
            detail: Some(SungBackendDetail::PhoneTimedViaKlattFallback),
            notes: &[
                "Riper sing-demo currently routes through an explicit RiperKlattFallback sung path.",
                "Riper's current sung stretch-synth path is Klatt source/filter until the ONNX path grows direct F0 and duration controls.",
            ],
        }
    }
}

impl Default for RiperKlattFallbackBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl VocoderBackend for RiperKlattFallbackBackend {
    fn id(&self) -> &'static str {
        Self::descriptor().id
    }

    fn descriptor(&self) -> VocoderDescriptor {
        Self::descriptor()
    }

    fn render(&mut self, input: VocoderInput<'_>) -> Result<Vec<AudioFrame>> {
        match input {
            VocoderInput::RenderPlan(_) | VocoderInput::PhoneTimed(_) => self
                .inner
                .render(input)
                .context("riper klatt fallback failed to render the shared phone-timed plan"),
            _ => bail!("riper-klatt-fallback backend requires phone-timed input"),
        }
    }
}
