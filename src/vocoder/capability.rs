use crate::voice::articulator::{SungBackendDetail, SungBackendKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendFamily {
    FormantSourceFilter,
    DiphoneTdPsola,
    TextTtsProcess,
    NeuralVocoder,
    NeuralSourceFilter,
    Placeholder,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub accepts_phone_timed: bool,
    pub accepts_partial_prosody: bool,
    pub accepts_coarse_text: bool,
    pub accepts_mel: bool,
    pub accepts_mel_f0: bool,
    pub honors_explicit_duration: bool,
    pub honors_explicit_f0: bool,
    pub honors_vibrato: bool,
    pub streaming_safe: bool,
}

impl BackendCapabilities {
    pub const fn unsupported() -> Self {
        Self {
            accepts_phone_timed: false,
            accepts_partial_prosody: false,
            accepts_coarse_text: false,
            accepts_mel: false,
            accepts_mel_f0: false,
            honors_explicit_duration: false,
            honors_explicit_f0: false,
            honors_vibrato: false,
            streaming_safe: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VocoderDescriptor {
    pub id: &'static str,
    pub family: BackendFamily,
    pub capabilities: BackendCapabilities,
    pub sample_rate_hz: u32,
    pub backend_kind: SungBackendKind,
    pub detail: SungBackendDetail,
    pub notes: &'static [&'static str],
}
