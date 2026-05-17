use uuid::Uuid;

use crate::event::HearingEvent;
use crate::hearing::vad::VadResult;

pub const DEFAULT_VAD_FRAME_MS: u64 = 10;
pub const DEFAULT_CONVERSATIONAL_TURN_SILENCE_MS: u64 = 800;

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
            close_after_silence_frames: frames_for_duration_ms(
                DEFAULT_CONVERSATIONAL_TURN_SILENCE_MS,
                DEFAULT_VAD_FRAME_MS,
            ),
        }
    }
}

const fn frames_for_duration_ms(duration_ms: u64, frame_ms: u64) -> usize {
    if duration_ms == 0 || frame_ms == 0 {
        return 0;
    }
    duration_ms.div_ceil(frame_ms) as usize
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

        for _ in 0..(BreathGroupConfig::default().close_after_silence_frames - 1) {
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
        for _ in 0..BreathGroupConfig::default().close_after_silence_frames {
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

    #[test]
    fn default_close_wait_matches_conversational_turn_silence() {
        assert_eq!(
            BreathGroupConfig::default().close_after_silence_frames as u64 * DEFAULT_VAD_FRAME_MS,
            DEFAULT_CONVERSATIONAL_TURN_SILENCE_MS
        );
    }
}
