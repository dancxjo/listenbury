use crate::audio::frame::AudioFrame;
use crate::mouth::planner::MouthSyntheticPlan;

pub trait TextToSpeech {
    fn enqueue(&mut self, plan: MouthSyntheticPlan) -> anyhow::Result<()>;
    fn poll_audio(&mut self) -> anyhow::Result<Vec<AudioFrame>>;
    fn stop(&mut self) -> anyhow::Result<()>;
}
