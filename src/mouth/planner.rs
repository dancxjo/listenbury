use crate::mind::llm::LlmEvent;

#[derive(Debug, Clone)]
pub enum SpeechPlan {
    Backchannel(String),
    Clause(String),
    FullTurn(String),
}

#[derive(Debug, Clone)]
pub enum MouthCommand {
    Speak(SpeechPlan),
    FadeOut { millis: u64 },
    StopNow,
}

#[derive(Debug, Default)]
pub struct SpeechPlanner {
    buffer: String,
}

impl SpeechPlanner {
    pub fn ingest(&mut self, events: &[LlmEvent]) -> Option<SpeechPlan> {
        for event in events {
            match event {
                LlmEvent::Token { text } => self.buffer.push_str(text),
                LlmEvent::Completed => {
                    let text = std::mem::take(&mut self.buffer);
                    return Some(SpeechPlan::FullTurn(text));
                }
                LlmEvent::Cancelled | LlmEvent::Error { .. } => {
                    self.buffer.clear();
                }
            }
        }

        None
    }
}
