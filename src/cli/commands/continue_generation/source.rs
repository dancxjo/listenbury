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
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum SourceCommand {
    RunTypeScript { source: String },
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SourceCommandExecution {
    pub(super) message: String,
    pub(super) runtime_events: Vec<ContinueRuntimeEvent>,
}
