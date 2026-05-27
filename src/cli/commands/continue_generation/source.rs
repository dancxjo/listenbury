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
pub(super) fn execute_source_command(command: &SourceCommand) -> SourceCommandExecution {
    match command {
        SourceCommand::RunTypeScript { source } => execute_typescript_source(source),
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SourceCommandExecution {
    pub(super) message: String,
    pub(super) runtime_events: Vec<ContinueRuntimeEvent>,
}
