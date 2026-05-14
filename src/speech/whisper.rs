use crate::audio::frame::AudioFrame;
use crate::speech::recognizer::SpeechRecognizer;
use crate::speech::transcript::TranscriptChunk;

pub struct WhisperSpeechRecognizer {
    ctx: whisper_cpp_plus::WhisperContext,
    pending: Vec<f32>,
    sample_rate_hz: u32,
}

impl WhisperSpeechRecognizer {
    pub fn new(model_path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let ctx = whisper_cpp_plus::WhisperContext::new(model_path.as_ref())?;

        Ok(Self {
            ctx,
            pending: Vec::new(),
            sample_rate_hz: 16_000,
        })
    }

    fn accept_frame(&mut self, frame: &AudioFrame) -> anyhow::Result<()> {
        anyhow::ensure!(
            frame.sample_rate_hz == self.sample_rate_hz,
            "Whisper expects {} Hz audio; got {} Hz",
            self.sample_rate_hz,
            frame.sample_rate_hz
        );

        anyhow::ensure!(
            frame.channels == 1,
            "Whisper expects mono audio; got {} channels",
            frame.channels
        );

        self.pending.extend_from_slice(&frame.samples);
        Ok(())
    }
}

impl SpeechRecognizer for WhisperSpeechRecognizer {
    fn push_frame(&mut self, frame: &AudioFrame) -> anyhow::Result<()> {
        self.accept_frame(frame)
    }

    fn poll_chunks(&mut self) -> anyhow::Result<Vec<TranscriptChunk>> {
        if self.pending.is_empty() {
            return Ok(Vec::new());
        }

        let audio = std::mem::take(&mut self.pending);
        let text = self.ctx.transcribe(&audio)?;
        let text = text.trim();

        if text.is_empty() {
            return Ok(Vec::new());
        }

        Ok(vec![TranscriptChunk {
            text: text.to_owned(),
            is_final: true,
        }])
    }
}
