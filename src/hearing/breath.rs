use uuid::Uuid;

use crate::event::HearingEvent;
use crate::hearing::vad::VadResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BreathGroupId(pub Uuid);

#[derive(Debug, Clone, PartialEq)]
pub enum BreathGroupEndReason {
    Silence,
    Timeout,
    Overlap,
    Cancelled,
}

#[derive(Debug, Clone, Copy)]
pub struct BreathGroupConfig {
    pub open_after_speech_frames: usize,
    pub close_after_silence_frames: usize,
}

impl Default for BreathGroupConfig {
    fn default() -> Self {
        Self {
            open_after_speech_frames: 3,
            close_after_silence_frames: 10,
        }
    }
}

#[derive(Debug)]
pub struct BreathGroupSegmenter {
    config: BreathGroupConfig,
    speech_frames: usize,
    silence_frames: usize,
    active_group: Option<BreathGroupId>,
}

impl BreathGroupSegmenter {
    pub fn new(config: BreathGroupConfig) -> Self {
        Self {
            config,
            speech_frames: 0,
            silence_frames: 0,
            active_group: None,
        }
    }

    pub fn process(&mut self, vad: VadResult) -> Vec<HearingEvent> {
        let mut events = Vec::new();

        if vad.is_speech {
            self.silence_frames = 0;
            if let Some(_id) = self.active_group {
                events.push(HearingEvent::SpeechContinued {
                    speech_prob: vad.speech_prob,
                });
            } else {
                self.speech_frames += 1;
                if self.speech_frames >= self.config.open_after_speech_frames {
                    let id = BreathGroupId(Uuid::new_v4());
                    self.active_group = Some(id);
                    self.speech_frames = 0;
                    events.push(HearingEvent::SpeechStarted);
                    events.push(HearingEvent::BreathGroupOpened { id });
                }
            }
            return events;
        }

        self.speech_frames = 0;
        if let Some(id) = self.active_group {
            if self.silence_frames == 0 {
                events.push(HearingEvent::PauseStarted);
            }
            self.silence_frames += 1;
            if self.silence_frames >= self.config.close_after_silence_frames {
                self.silence_frames = 0;
                self.active_group = None;
                events.push(HearingEvent::BreathGroupClosed {
                    id,
                    reason: BreathGroupEndReason::Silence,
                });
            }
        }

        events
    }
}

impl Default for BreathGroupSegmenter {
    fn default() -> Self {
        Self::new(BreathGroupConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn speech() -> VadResult {
        VadResult {
            speech_prob: 0.9,
            is_speech: true,
        }
    }

    fn silence() -> VadResult {
        VadResult {
            speech_prob: 0.0,
            is_speech: false,
        }
    }

    #[test]
    fn breath_group_opens_after_configured_speech_frames() {
        let mut segmenter = BreathGroupSegmenter::default();

        let _ = segmenter.process(speech());
        let _ = segmenter.process(speech());
        let events = segmenter.process(speech());

        assert!(
            events
                .iter()
                .any(|ev| matches!(ev, HearingEvent::BreathGroupOpened { .. }))
        );
    }

    #[test]
    fn breath_group_bridges_short_silence() {
        let mut segmenter = BreathGroupSegmenter::default();
        for _ in 0..3 {
            let _ = segmenter.process(speech());
        }

        for _ in 0..9 {
            let events = segmenter.process(silence());
            assert!(
                !events
                    .iter()
                    .any(|ev| matches!(ev, HearingEvent::BreathGroupClosed { .. }))
            );
        }

        let events = segmenter.process(speech());
        assert!(
            !events
                .iter()
                .any(|ev| matches!(ev, HearingEvent::BreathGroupClosed { .. }))
        );
    }

    #[test]
    fn breath_group_closes_after_configured_silence_frames() {
        let mut segmenter = BreathGroupSegmenter::default();
        for _ in 0..3 {
            let _ = segmenter.process(speech());
        }

        let mut closed = false;
        for _ in 0..10 {
            let events = segmenter.process(silence());
            if events
                .iter()
                .any(|ev| matches!(ev, HearingEvent::BreathGroupClosed { .. }))
            {
                closed = true;
            }
        }

        assert!(closed);
    }
}
