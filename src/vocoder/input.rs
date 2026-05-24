use crate::voice::articulator::{
    PartialProsodyPhone, PhoneTimedRenderTarget, PitchHint, RenderPlan,
};

#[derive(Debug, Clone, PartialEq)]
pub struct MelFrame {
    pub bins: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MelConfig {
    pub sample_rate_hz: u32,
    pub hop_samples: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub n_mels: usize,
    pub f_min_hz: f32,
    pub f_max_hz: Option<f32>,
    pub scale: MelScale,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MelScale {
    LinearEnergy,
    NaturalLogEnergy,
    Log10Energy,
    DynamicRangeCompressed,
    ModelSpecific(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct MelSpectrogram {
    pub config: MelConfig,
    pub frames: Vec<MelFrame>,
}

impl MelConfig {
    pub fn test_default(n_mels: usize) -> Self {
        Self {
            sample_rate_hz: 22_050,
            hop_samples: 256,
            n_fft: 1024,
            win_length: 1024,
            n_mels,
            f_min_hz: 0.0,
            f_max_hz: Some(8_000.0),
            scale: MelScale::NaturalLogEnergy,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelTensorLayout {
    FramesBins,      // [T, M]
    BinsFrames,      // [M, T]
    BatchFramesBins, // [1, T, M]
    BatchBinsFrames, // [1, M, T]
}

pub enum VocoderInput<'a> {
    RenderPlan(&'a RenderPlan),
    PhoneTimed(&'a [PhoneTimedRenderTarget]),
    PartialProsody {
        text: &'a str,
        phones: &'a [PartialProsodyPhone],
        pitch_hints: &'a [PitchHint],
    },
    CoarseText {
        text: &'a str,
        ssml_hint: Option<&'a str>,
    },
    Mel(&'a MelSpectrogram),
    MelF0 {
        mel: &'a MelSpectrogram,
        f0_hz: &'a [f32],
        voiced: &'a [bool],
    },
    SourceFilter {
        f0_hz: &'a [f32],
        voiced: &'a [bool],
        spectral: &'a [f32],
        aperiodicity: &'a [f32],
    },
}
