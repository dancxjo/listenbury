use crate::audio::frame::AudioFrame;
use crate::mouth::planner::SpeechPlan;

pub trait TextToSpeech {
    fn enqueue(&mut self, plan: SpeechPlan) -> anyhow::Result<()>;
    fn poll_audio(&mut self) -> anyhow::Result<Vec<AudioFrame>>;
    fn stop(&mut self) -> anyhow::Result<()>;
}
