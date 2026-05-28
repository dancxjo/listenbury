#[cfg(any(test, feature = "asr-whisper"))]
use super::*;

#[cfg(any(test, feature = "asr-whisper"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum SourceCommand {
    RunTypeScript { source: String },
}

#[cfg(any(test, feature = "asr-whisper"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SourceCommandExecution {
    pub(super) message: String,
    pub(super) runtime_events: Vec<ContinueRuntimeEvent>,
}
