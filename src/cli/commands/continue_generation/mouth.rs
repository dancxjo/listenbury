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
pub(super) enum ContinueMouthCommand {
    Speak {
        id: u64,
        text: String,
        interrupt: bool,
    },
    Shutup,
    Pause,
    Resume,
    Shutdown,
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
pub(super) fn mouth_command_for_runtime_event(
    event: &ContinueRuntimeEvent,
) -> Option<(ContinueMouthCommand, bool)> {
    match event {
        ContinueRuntimeEvent::UtteranceCompleted {
            id,
            content,
            interrupt,
        } => {
            let content = clean_spoken_content(content)?;
            Some((
                ContinueMouthCommand::Speak {
                    id: *id,
                    text: content,
                    interrupt: *interrupt,
                },
                true,
            ))
        }
        ContinueRuntimeEvent::SpeechControl { command } => {
            let command = match command {
                SpeechControlCommand::Shutup => ContinueMouthCommand::Shutup,
                SpeechControlCommand::Pause => ContinueMouthCommand::Pause,
                SpeechControlCommand::Resume => ContinueMouthCommand::Resume,
            };
            Some((command, false))
        }
        ContinueRuntimeEvent::SourceCommand { .. } => None,
    }
}
