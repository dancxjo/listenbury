#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use super::*;
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use listenbury::speech::transcript::{
    TranscriptCandidateEvent, TranscriptCandidateId, TranscriptReplacementReason,
};

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
#[derive(Debug, Clone, PartialEq)]
pub(super) struct TranscriptStabilityState {
    pub(super) candidate_id: TranscriptCandidateId,
    pub(super) stable_text: String,
    pub(super) unstable_text: String,
    pub(super) confidence: Option<f32>,
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
impl TranscriptStabilityState {
    pub(super) fn from_parts(
        candidate_id: TranscriptCandidateId,
        text: &str,
        stable_prefix_len: usize,
        confidence: Option<f32>,
    ) -> Self {
        let split = stable_prefix_len.min(text.len());
        let split = if text.is_char_boundary(split) {
            split
        } else {
            text.char_indices()
                .find_map(|(idx, ch)| {
                    let end = idx + ch.len_utf8();
                    (end >= split).then_some(end)
                })
                .unwrap_or(text.len())
        };
        let (stable_text, unstable_text) = text.split_at(split);
        Self {
            candidate_id,
            stable_text: stable_text.to_string(),
            unstable_text: unstable_text.to_string(),
            confidence,
        }
    }
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
#[derive(Debug, Default)]
pub(super) struct TranscriptSpeculativePlanner {
    active_candidate: Option<TranscriptCandidateId>,
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
impl TranscriptSpeculativePlanner {
    pub(super) fn observe(
        &mut self,
        event: &TranscriptCandidateEvent,
    ) -> Option<TranscriptStabilityState> {
        match event {
            TranscriptCandidateEvent::CandidateStarted { id } => {
                self.active_candidate = Some(*id);
                None
            }
            TranscriptCandidateEvent::CandidateUpdated {
                id,
                text,
                stable_prefix_len,
                confidence,
            } => {
                self.active_candidate = Some(*id);
                Some(TranscriptStabilityState::from_parts(
                    *id,
                    text,
                    *stable_prefix_len,
                    *confidence,
                ))
            }
            TranscriptCandidateEvent::CandidateReplaced { new, .. } => {
                self.active_candidate = Some(*new);
                None
            }
            TranscriptCandidateEvent::CandidateFinalized {
                id,
                text,
                confidence,
            } => {
                if self.active_candidate == Some(*id) {
                    self.active_candidate = None;
                }
                Some(TranscriptStabilityState::from_parts(
                    *id,
                    text,
                    text.len(),
                    *confidence,
                ))
            }
            TranscriptCandidateEvent::CandidateCancelled { id } => {
                if self.active_candidate == Some(*id) {
                    self.active_candidate = None;
                }
                None
            }
        }
    }
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
pub(super) enum ContinueEarEvent {
    ListeningStarted {
        device: String,
        sample_rate_hz: u32,
        channels: u16,
        vad: VadBackendKind,
    },
    SpeechStarted,
    SpeechStopped,
    AuditoryObservation {
        text: String,
    },
    EnvironmentalSound {
        sound: EnvironmentalSound,
    },
    SelfVoiceHeard {
        delay_ms: i64,
        gain: f32,
        confidence: f32,
    },
    OverlapDetected {
        self_confidence: f32,
        external_confidence: f32,
        duration_ms: u64,
    },
    Transcript {
        text: String,
        timed_word_stream: TimedWordStream,
        occurred_at: ExactTimestamp,
    },
    TranscriptCandidate {
        event: TranscriptCandidateEvent,
        stability: Option<TranscriptStabilityState>,
        occurred_at: ExactTimestamp,
    },
    Error {
        message: String,
    },
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
impl ContinueEarEvent {
    pub(super) fn to_message(&self) -> String {
        match self {
            Self::ListeningStarted {
                device,
                sample_rate_hz,
                channels,
                vad,
            } => format!(
                "listening_started: device={device:?} sample_rate_hz={sample_rate_hz} channels={channels} vad={}",
                vad.as_str()
            ),
            Self::SpeechStarted => "speech_started".to_string(),
            Self::SpeechStopped => "speech_stopped".to_string(),
            Self::AuditoryObservation { text } => text.clone(),
            Self::EnvironmentalSound { sound } => sound.description.clone(),
            Self::SelfVoiceHeard {
                delay_ms,
                gain,
                confidence,
            } => format!(
                "Pete's own playback is audible in the microphone but has been excluded from ASR. delay_ms={delay_ms} gain={gain:.2} confidence={confidence:.2}"
            ),
            Self::OverlapDetected {
                self_confidence,
                external_confidence,
                duration_ms,
            } => format!(
                "Someone began speaking while Pete was speaking. self_confidence={self_confidence:.2} external_confidence={external_confidence:.2} duration_ms={duration_ms}"
            ),
            Self::Transcript { text, .. } => format!("Heard: {}", text.trim()),
            Self::TranscriptCandidate {
                event,
                stability,
                occurred_at: _,
            } => match event {
                TranscriptCandidateEvent::CandidateStarted { id } => {
                    format!("transcript_candidate_started: id={}", id.0)
                }
                TranscriptCandidateEvent::CandidateUpdated { id, .. }
                | TranscriptCandidateEvent::CandidateFinalized { id, .. } => {
                    if let Some(state) = stability {
                        format!(
                            "transcript_candidate_state: id={} stable={:?} unstable={:?} confidence={:?}",
                            id.0, state.stable_text, state.unstable_text, state.confidence
                        )
                    } else {
                        format!("transcript_candidate_state: id={}", id.0)
                    }
                }
                TranscriptCandidateEvent::CandidateReplaced { old, new, reason } => {
                    let reason = match reason {
                        TranscriptReplacementReason::HeadChanged { stable_prefix_len } => {
                            format!("head_changed stable_prefix_len={stable_prefix_len}")
                        }
                        TranscriptReplacementReason::Restarted => "restarted".to_string(),
                    };
                    format!(
                        "transcript_candidate_replaced: old={} new={} reason={}",
                        old.0, new.0, reason
                    )
                }
                TranscriptCandidateEvent::CandidateCancelled { id } => {
                    format!("transcript_candidate_cancelled: id={}", id.0)
                }
            },
            Self::Error { message } => format!("error: {message}"),
        }
    }

    pub(super) fn direct_prompt_packet(&self) -> Option<PromptPacket> {
        match self {
            Self::Transcript { text, .. } => Some(PromptPacket::heard(text.clone())),
            Self::AuditoryObservation { text } => Some(PromptPacket::ear_observation(text.clone())),
            Self::EnvironmentalSound { sound } => {
                Some(PromptPacket::ear_observation(sound.description.clone()))
            }
            Self::SpeechStopped => Some(PromptPacket::ear_observation(
                "External speech stopped; prepare a response if appropriate, but wait for the quiet turn gap before speaking.".to_string(),
            )),
            Self::SelfVoiceHeard { .. } | Self::OverlapDetected { .. } => {
                Some(PromptPacket::ear_observation(self.to_message()))
            }
            Self::ListeningStarted { .. }
            | Self::SpeechStarted
            | Self::TranscriptCandidate { .. }
            | Self::Error { .. } => None,
        }
    }
}
