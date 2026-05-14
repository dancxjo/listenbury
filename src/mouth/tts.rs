use crate::mouth::planner::SpeechPlan;

pub trait TextToSpeech {
    fn enqueue(&mut self, plan: SpeechPlan) -> anyhow::Result<()>;
}
