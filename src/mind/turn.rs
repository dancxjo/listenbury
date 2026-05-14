use crate::event::HearingEvent;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnState {
    Idle,
    UserSpeaking,
    UserPausedMaybeContinuing,
    PeteThinking,
    PeteSpeaking,
    CollisionLikely,
}

#[derive(Debug, Clone, Copy)]
pub struct TurnTracker {
    state: TurnState,
}

impl TurnTracker {
    pub fn new() -> Self {
        Self {
            state: TurnState::Idle,
        }
    }

    pub fn state(&self) -> TurnState {
        self.state
    }

    pub fn on_hearing_event(&mut self, event: &HearingEvent) {
        match event {
            HearingEvent::SpeechStarted => {
                if self.state == TurnState::PeteSpeaking {
                    self.state = TurnState::CollisionLikely;
                } else {
                    self.state = TurnState::UserSpeaking;
                }
            }
            HearingEvent::PauseStarted => {
                if self.state == TurnState::UserSpeaking {
                    self.state = TurnState::UserPausedMaybeContinuing;
                }
            }
            HearingEvent::BreathGroupClosed { .. } => {
                self.state = TurnState::PeteThinking;
            }
            HearingEvent::SpeechContinued { .. } | HearingEvent::BreathGroupOpened { .. } => {}
        }
    }

    pub fn on_pete_speech_started(&mut self) {
        self.state = TurnState::PeteSpeaking;
    }
}

impl Default for TurnTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::event::HearingEvent;

    use super::{TurnState, TurnTracker};

    #[test]
    fn turn_tracker_enters_collision_state() {
        let mut tracker = TurnTracker::default();
        tracker.on_pete_speech_started();
        tracker.on_hearing_event(&HearingEvent::SpeechStarted);
        assert_eq!(tracker.state(), TurnState::CollisionLikely);
    }
}
