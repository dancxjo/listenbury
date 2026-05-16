use crate::audio::frame::AudioFrame;
use crate::speech::transcript::TranscriptChunk;

pub trait SpeechRecognizer {
    fn push_frame(&mut self, frame: &AudioFrame) -> anyhow::Result<()>;

    /// Polls recognizer output for all currently buffered audio.
    ///
    /// Implementations are expected to consume/drain any pending internal audio buffer
    /// when producing these chunks.
    fn poll_chunks(&mut self) -> anyhow::Result<Vec<TranscriptChunk>>;
}
