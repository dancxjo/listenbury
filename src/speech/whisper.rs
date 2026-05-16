use crate::audio::frame::AudioFrame;
use crate::developer_diagnostics_enabled;
use crate::speech::recognizer::SpeechRecognizer;
use crate::speech::transcript::TranscriptChunk;
use std::sync::OnceLock;
use whisper_cpp_plus::whisper_cpp_plus_sys as whisper_ffi;

pub struct WhisperSpeechRecognizer {
    ctx: whisper_cpp_plus::WhisperContext,
    pending: Vec<f32>,
    sample_rate_hz: u32,
}

impl WhisperSpeechRecognizer {
    pub fn new(model_path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        Self::new_with_log_suppression(model_path, false)
    }

    pub fn new_quiet(model_path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        Self::new_with_log_suppression(model_path, true)
    }

    fn new_with_log_suppression(
        model_path: impl AsRef<std::path::Path>,
        suppress_logs: bool,
    ) -> anyhow::Result<Self> {
        configure_whisper_logging(suppress_logs);
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

fn configure_whisper_logging(suppress_logs: bool) {
    static LOGGING_CONFIGURED: OnceLock<()> = OnceLock::new();
    if suppress_logs || !developer_diagnostics_enabled() {
        LOGGING_CONFIGURED.get_or_init(|| unsafe {
            whisper_ffi::whisper_log_set(Some(drop_whisper_log), std::ptr::null_mut());
        });
    }
}

unsafe extern "C" fn drop_whisper_log(
    _level: whisper_ffi::ggml_log_level,
    _text: *const std::ffi::c_char,
    _user_data: *mut std::ffi::c_void,
) {
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
