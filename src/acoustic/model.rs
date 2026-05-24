use anyhow::Result;

use crate::prosody::singing::SungPhrase;
use crate::voice::articulator::PhoneTimedRenderTarget;
use crate::voice::tract::SourceFilterTrack;

#[derive(Debug, Clone, PartialEq)]
pub struct MelFrame {
    pub bins: Vec<f32>,
}

pub type SingingPlan = SungPhrase;

pub enum AcousticInput<'a> {
    PhoneTimed(&'a [PhoneTimedRenderTarget]),
    Singing(&'a SingingPlan),
    SourceFilterTrack(&'a SourceFilterTrack),
}

#[derive(Debug, Clone, PartialEq)]
pub struct AcousticFrameTrack {
    pub mel: Vec<MelFrame>,
    pub f0_hz: Vec<f32>,
    pub voiced: Vec<bool>,
    pub sample_rate_hz: u32,
    pub hop_samples: usize,
}

pub trait AcousticModelBackend {
    fn id(&self) -> &'static str;

    fn generate(&mut self, input: AcousticInput<'_>) -> Result<AcousticFrameTrack>;
}
