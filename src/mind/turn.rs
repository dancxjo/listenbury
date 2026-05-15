use crate::event::HearingEvent;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnState {
    Idle,
    UserSpeaking,
    UserPauseLikelyContinuing,
    UserTurnComplete,
    PeteThinking,
    PeteSpeaking,
    PeteInterrupted,
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
                if !matches!(
                    self.state,
                    TurnState::PeteSpeaking | TurnState::PeteInterrupted
                ) {
                    self.state = TurnState::UserSpeaking;
                }
            }
            HearingEvent::PauseStarted => {
                if matches!(
                    self.state,
                    TurnState::UserSpeaking | TurnState::PeteInterrupted
                ) {
                    self.state = TurnState::UserPauseLikelyContinuing;
                }
            }
            HearingEvent::BreathGroupClosed { .. } => {
                self.state = TurnState::UserTurnComplete;
            }
            HearingEvent::SpeechContinued { .. } | HearingEvent::BreathGroupOpened { .. } => {}
        }
    }

    pub fn on_pete_thinking_started(&mut self) {
        self.state = TurnState::PeteThinking;
    }

    pub fn on_pete_speech_started(&mut self) {
        self.state = TurnState::PeteSpeaking;
    }

    pub fn on_pete_interrupted(&mut self) {
        self.state = TurnState::PeteInterrupted;
    }
}

impl Default for TurnTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use crate::event::HearingEvent;
    use crate::hearing::breath::{BreathGroupEndReason, BreathGroupId};

    use super::{TurnState, TurnTracker};

    #[test]
    fn turn_tracker_keeps_pete_speaking_for_short_barge_in_blips() {
        let mut tracker = TurnTracker::default();
        tracker.on_pete_speech_started();
        tracker.on_hearing_event(&HearingEvent::SpeechStarted);
        assert_eq!(tracker.state(), TurnState::PeteSpeaking);
    }

    #[test]
    fn turn_tracker_marks_interrupted_when_policy_escalates() {
        let mut tracker = TurnTracker::default();
        tracker.on_pete_speech_started();
        tracker.on_pete_interrupted();
        assert_eq!(tracker.state(), TurnState::PeteInterrupted);
    }

    #[test]
    fn turn_tracker_marks_user_turn_complete_when_breath_group_closes() {
        let mut tracker = TurnTracker::default();
        tracker.on_hearing_event(&HearingEvent::BreathGroupClosed {
            id: BreathGroupId(Uuid::nil()),
            reason: BreathGroupEndReason::Silence,
        });
        assert_eq!(tracker.state(), TurnState::UserTurnComplete);
    }
}
