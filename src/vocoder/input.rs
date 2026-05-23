use crate::voice::articulator::{
    PartialProsodyPhone, PhoneTimedRenderTarget, PitchHint, RenderPlan,
};

#[derive(Debug, Clone, PartialEq)]
pub struct MelFrame {
    pub bins: Vec<f32>,
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
    Mel(&'a [MelFrame]),
    MelF0 {
        mel: &'a [MelFrame],
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
