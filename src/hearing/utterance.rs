use std::collections::VecDeque;

use crate::audio::AudioFrame;
use crate::hearing::vad::VadResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UtteranceSmootherConfig {
    pub speech_start_frames: usize,
    pub speech_end_silence_frames: usize,
    pub min_utterance_ms: u64,
    pub pre_roll_ms: u64,
    pub post_roll_ms: u64,
    pub frame_ms: u64,
}

impl Default for UtteranceSmootherConfig {
    fn default() -> Self {
        Self {
            speech_start_frames: 3,
            speech_end_silence_frames: 30,
            min_utterance_ms: 250,
            pre_roll_ms: 200,
            post_roll_ms: 300,
            frame_ms: 10,
        }
    }
}

impl UtteranceSmootherConfig {
    fn pre_roll_frames(self) -> usize {
        frames_for_duration_ms(self.pre_roll_ms, self.frame_ms)
    }

    fn post_roll_frames(self) -> usize {
        frames_for_duration_ms(self.post_roll_ms, self.frame_ms)
    }

    fn end_after_silence_frames(self) -> usize {
        self.speech_end_silence_frames
            .max(self.post_roll_frames())
            .max(1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UtteranceSmootherState {
    Idle,
    MaybeSpeech,
    InSpeech,
    MaybeSilence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UtteranceEndReason {
    Silence,
    Timeout,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UtteranceSmootherEvent {
    SpeechStarted {
        pre_roll_frames: usize,
        speech_prob: f32,
    },
    SpeechEnded {
        reason: UtteranceEndReason,
        frames: Vec<AudioFrame>,
        duration_ms: u64,
        pre_roll_frames: usize,
        post_roll_frames: usize,
    },
    UtteranceDropped {
        reason: UtteranceEndReason,
        frames: Vec<AudioFrame>,
        duration_ms: u64,
        pre_roll_frames: usize,
        post_roll_frames: usize,
    },
}

#[derive(Debug)]
pub struct UtteranceSmoother {
    config: UtteranceSmootherConfig,
    state: UtteranceSmootherState,
    pre_roll: VecDeque<AudioFrame>,
    pending_speech: Vec<AudioFrame>,
    active_frames: Vec<AudioFrame>,
    consecutive_voiced_frames: usize,
    consecutive_silence_frames: usize,
    active_pre_roll_frames: usize,
    last_voiced_active_len: usize,
}

impl UtteranceSmoother {
    pub fn new(config: UtteranceSmootherConfig) -> Self {
        Self {
            config,
            state: UtteranceSmootherState::Idle,
            pre_roll: VecDeque::new(),
            pending_speech: Vec::new(),
            active_frames: Vec::new(),
            consecutive_voiced_frames: 0,
            consecutive_silence_frames: 0,
            active_pre_roll_frames: 0,
            last_voiced_active_len: 0,
        }
    }

    pub fn state(&self) -> UtteranceSmootherState {
        self.state
    }

    pub fn process(&mut self, vad: VadResult, frame: AudioFrame) -> Vec<UtteranceSmootherEvent> {
        let mut events = Vec::new();
        if vad.is_speech {
            self.observe_speech_frame(frame, vad.speech_prob, &mut events);
        } else {
            self.observe_silence_frame(frame, &mut events);
        }
        events
    }

    pub fn finish(&mut self) -> Vec<UtteranceSmootherEvent> {
        match self.state {
            UtteranceSmootherState::InSpeech | UtteranceSmootherState::MaybeSilence => {
                vec![self.close_active_utterance(UtteranceEndReason::Timeout)]
            }
            UtteranceSmootherState::MaybeSpeech => {
                let pending = std::mem::take(&mut self.pending_speech);
                for frame in pending {
                    self.push_pre_roll(frame);
                }
                self.reset_idle();
                Vec::new()
            }
            UtteranceSmootherState::Idle => Vec::new(),
        }
    }

    fn observe_speech_frame(
        &mut self,
        frame: AudioFrame,
        speech_prob: f32,
        events: &mut Vec<UtteranceSmootherEvent>,
    ) {
        match self.state {
            UtteranceSmootherState::Idle => {
                self.state = UtteranceSmootherState::MaybeSpeech;
                self.pending_speech.push(frame);
                self.consecutive_voiced_frames = 1;
                if self.consecutive_voiced_frames >= self.config.speech_start_frames.max(1) {
                    events.push(self.open_active_utterance(speech_prob));
                }
            }
            UtteranceSmootherState::MaybeSpeech => {
                self.pending_speech.push(frame);
                self.consecutive_voiced_frames = self.consecutive_voiced_frames.saturating_add(1);
                if self.consecutive_voiced_frames >= self.config.speech_start_frames.max(1) {
                    events.push(self.open_active_utterance(speech_prob));
                }
            }
            UtteranceSmootherState::InSpeech | UtteranceSmootherState::MaybeSilence => {
                self.active_frames.push(frame);
                self.last_voiced_active_len = self.active_frames.len();
                self.consecutive_silence_frames = 0;
                self.state = UtteranceSmootherState::InSpeech;
            }
        }
    }

    fn observe_silence_frame(
        &mut self,
        frame: AudioFrame,
        events: &mut Vec<UtteranceSmootherEvent>,
    ) {
        match self.state {
            UtteranceSmootherState::Idle => self.push_pre_roll(frame),
            UtteranceSmootherState::MaybeSpeech => {
                let pending = std::mem::take(&mut self.pending_speech);
                for pending_frame in pending {
                    self.push_pre_roll(pending_frame);
                }
                self.push_pre_roll(frame);
                self.reset_idle();
            }
            UtteranceSmootherState::InSpeech | UtteranceSmootherState::MaybeSilence => {
                self.active_frames.push(frame);
                self.consecutive_silence_frames = self.consecutive_silence_frames.saturating_add(1);
                self.state = UtteranceSmootherState::MaybeSilence;
                if self.consecutive_silence_frames >= self.config.end_after_silence_frames() {
                    events.push(self.close_active_utterance(UtteranceEndReason::Silence));
                }
            }
        }
    }

    fn open_active_utterance(&mut self, speech_prob: f32) -> UtteranceSmootherEvent {
        self.active_pre_roll_frames = self.pre_roll.len();
        self.active_frames.extend(self.pre_roll.drain(..));
        self.active_frames.append(&mut self.pending_speech);
        self.last_voiced_active_len = self.active_frames.len();
        self.consecutive_voiced_frames = 0;
        self.consecutive_silence_frames = 0;
        self.state = UtteranceSmootherState::InSpeech;
        UtteranceSmootherEvent::SpeechStarted {
            pre_roll_frames: self.active_pre_roll_frames,
            speech_prob,
        }
    }

    fn close_active_utterance(&mut self, reason: UtteranceEndReason) -> UtteranceSmootherEvent {
        let frames = std::mem::take(&mut self.active_frames);
        let pre_roll_frames = self.active_pre_roll_frames.min(frames.len());
        let voiced_end = self.last_voiced_active_len.min(frames.len());
        let utterance_frames = voiced_end.saturating_sub(pre_roll_frames);
        let post_roll_frames = frames.len().saturating_sub(voiced_end);
        let duration_ms = utterance_frames as u64 * self.config.frame_ms;
        let next_pre_roll_seed: Vec<AudioFrame> = frames.iter().skip(voiced_end).cloned().collect();
        self.reset_idle();
        for frame in next_pre_roll_seed {
            self.push_pre_roll(frame);
        }

        if duration_ms < self.config.min_utterance_ms {
            UtteranceSmootherEvent::UtteranceDropped {
                reason,
                frames,
                duration_ms,
                pre_roll_frames,
                post_roll_frames,
            }
        } else {
            UtteranceSmootherEvent::SpeechEnded {
                reason,
                frames,
                duration_ms,
                pre_roll_frames,
                post_roll_frames,
            }
        }
    }

    fn push_pre_roll(&mut self, frame: AudioFrame) {
        self.pre_roll.push_back(frame);
        let max_frames = self.config.pre_roll_frames();
        while self.pre_roll.len() > max_frames {
            self.pre_roll.pop_front();
        }
    }

    fn reset_idle(&mut self) {
        self.state = UtteranceSmootherState::Idle;
        self.pending_speech.clear();
        self.consecutive_voiced_frames = 0;
        self.consecutive_silence_frames = 0;
        self.active_pre_roll_frames = 0;
        self.last_voiced_active_len = 0;
    }
}

impl Default for UtteranceSmoother {
    fn default() -> Self {
        Self::new(UtteranceSmootherConfig::default())
    }
}

const fn frames_for_duration_ms(duration_ms: u64, frame_ms: u64) -> usize {
    if duration_ms == 0 || frame_ms == 0 {
        return 0;
    }
    duration_ms.div_ceil(frame_ms) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::ExactTimestamp;

    fn test_config() -> UtteranceSmootherConfig {
        UtteranceSmootherConfig::default()
    }

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

    fn frame(sample: f32) -> AudioFrame {
        AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![sample; 160],
            voice_signatures: Vec::new(),
        }
    }

    fn feed(
        smoother: &mut UtteranceSmoother,
        vad: VadResult,
        count: usize,
    ) -> Vec<UtteranceSmootherEvent> {
        let mut events = Vec::new();
        for index in 0..count {
            events.extend(smoother.process(vad, frame(index as f32)));
        }
        events
    }

    #[test]
    fn silence_only_emits_no_utterance() {
        let mut smoother = UtteranceSmoother::new(test_config());
        let events = feed(&mut smoother, silence(), 100);

        assert!(events.is_empty());
        assert_eq!(smoother.state(), UtteranceSmootherState::Idle);
    }

    #[test]
    fn single_voiced_blip_emits_no_utterance() {
        let mut smoother = UtteranceSmoother::new(test_config());
        let mut events = feed(&mut smoother, speech(), 1);
        events.extend(feed(&mut smoother, silence(), 10));

        assert!(events.is_empty());
        assert_eq!(smoother.state(), UtteranceSmootherState::Idle);
    }

    #[test]
    fn three_voiced_frames_start_an_utterance() {
        let mut smoother = UtteranceSmoother::new(test_config());
        let events = feed(&mut smoother, speech(), 3);

        assert!(
            events
                .iter()
                .any(|event| matches!(event, UtteranceSmootherEvent::SpeechStarted { .. }))
        );
        assert_eq!(smoother.state(), UtteranceSmootherState::InSpeech);
    }

    #[test]
    fn short_silent_gaps_inside_speech_do_not_end_the_utterance() {
        let mut smoother = UtteranceSmoother::new(test_config());
        let mut events = feed(&mut smoother, speech(), 30);
        events.extend(feed(&mut smoother, silence(), 29));
        events.extend(feed(&mut smoother, speech(), 5));

        assert!(
            !events
                .iter()
                .any(|event| matches!(event, UtteranceSmootherEvent::SpeechEnded { .. }))
        );
        assert_eq!(smoother.state(), UtteranceSmootherState::InSpeech);
    }

    #[test]
    fn thirty_silent_frames_end_the_utterance() {
        let mut smoother = UtteranceSmoother::new(test_config());
        let mut events = feed(&mut smoother, speech(), 30);
        events.extend(feed(&mut smoother, silence(), 30));

        assert!(
            events
                .iter()
                .any(|event| matches!(event, UtteranceSmootherEvent::SpeechEnded { .. }))
        );
        assert_eq!(smoother.state(), UtteranceSmootherState::Idle);
    }

    #[test]
    fn utterances_shorter_than_min_duration_are_dropped() {
        let mut smoother = UtteranceSmoother::new(test_config());
        let mut events = feed(&mut smoother, speech(), 10);
        events.extend(feed(&mut smoother, silence(), 30));

        assert!(
            events
                .iter()
                .any(|event| matches!(event, UtteranceSmootherEvent::UtteranceDropped { .. }))
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, UtteranceSmootherEvent::SpeechEnded { .. }))
        );
    }

    #[test]
    fn pre_roll_frames_are_included_in_emitted_utterance_buffer() {
        let mut smoother = UtteranceSmoother::new(test_config());
        let mut events = feed(&mut smoother, silence(), 20);
        events.extend(feed(&mut smoother, speech(), 25));
        events.extend(feed(&mut smoother, silence(), 30));

        let ended = events
            .iter()
            .find_map(|event| match event {
                UtteranceSmootherEvent::SpeechEnded {
                    frames,
                    pre_roll_frames,
                    ..
                } => Some((frames, *pre_roll_frames)),
                _ => None,
            })
            .expect("accepted utterance should end");
        assert_eq!(ended.1, 20);
        assert_eq!(ended.0.len(), 20 + 25 + 30);
    }

    #[test]
    fn post_roll_frames_are_included_before_finalization() {
        let mut smoother = UtteranceSmoother::new(test_config());
        let mut events = feed(&mut smoother, speech(), 25);
        events.extend(feed(&mut smoother, silence(), 30));

        let post_roll_frames = events
            .iter()
            .find_map(|event| match event {
                UtteranceSmootherEvent::SpeechEnded {
                    post_roll_frames, ..
                } => Some(*post_roll_frames),
                _ => None,
            })
            .expect("accepted utterance should end");
        assert_eq!(post_roll_frames, 30);
    }
}
