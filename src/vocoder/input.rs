use crate::acoustic::MelFrame;
use crate::voice::articulator::{
    PartialProsodyPhone, PhoneTimedRenderTarget, PitchHint, RenderPlan,
};
use crate::voice::tract::SourceFilterTrack;

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
    SourceFilterTrack(&'a SourceFilterTrack),
    SourceFilter {
        f0_hz: &'a [f32],
        voiced: &'a [bool],
        spectral: &'a [f32],
        aperiodicity: &'a [f32],
    },
}
