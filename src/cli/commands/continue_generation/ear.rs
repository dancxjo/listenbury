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
            Self::SelfVoiceHeard { .. } | Self::OverlapDetected { .. } => {
                Some(PromptPacket::ear_observation(self.to_message()))
            }
            Self::ListeningStarted { .. }
            | Self::SpeechStarted
            | Self::SpeechStopped
            | Self::Error { .. } => None,
        }
    }
}
