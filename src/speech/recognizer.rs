use crate::audio::frame::AudioFrame;
use crate::speech::transcript::{TranscriptCandidateEvent, TranscriptChunk};
use crate::word::TranscriptWord;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingPartialKind {
    FinalOnly,
    Approximate,
    TokenStreaming,
}

impl StreamingPartialKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FinalOnly => "final_only",
            Self::Approximate => "approximate",
            Self::TokenStreaming => "token_streaming",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingRecognizerBackend {
    pub source: &'static str,
    pub partial_kind: StreamingPartialKind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreamingRecognition {
    pub text: String,
    pub words: Vec<TranscriptWord>,
    pub candidate_events: Vec<TranscriptCandidateEvent>,
    pub backend: StreamingRecognizerBackend,
}

pub trait SpeechRecognizer {
    fn push_frame(&mut self, frame: &AudioFrame) -> anyhow::Result<()>;

    /// Polls recognizer output for all currently buffered audio.
    ///
    /// Implementations are expected to consume/drain any pending internal audio buffer
    /// when producing these chunks.
    fn poll_chunks(&mut self) -> anyhow::Result<Vec<TranscriptChunk>>;
}

pub trait StreamingSpeechRecognizer: SpeechRecognizer {
    fn poll_streaming(&mut self, is_final: bool) -> anyhow::Result<StreamingRecognition>;

    fn flush(&mut self) -> anyhow::Result<StreamingRecognition> {
        self.poll_streaming(true)
    }

    fn backend(&self) -> StreamingRecognizerBackend;
}
