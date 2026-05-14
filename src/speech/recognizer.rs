use crate::audio::frame::AudioFrame;
use crate::speech::transcript::TranscriptChunk;

pub trait SpeechRecognizer {
    fn push_frame(&mut self, frame: &AudioFrame) -> anyhow::Result<()>;
    fn poll_chunks(&mut self) -> anyhow::Result<Vec<TranscriptChunk>>;
}
