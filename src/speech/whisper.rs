use crate::audio::frame::AudioFrame;
use crate::developer_diagnostics_enabled;
use crate::speech::recognizer::SpeechRecognizer;
use crate::speech::transcript::{
    TranscriptCandidateEvent, TranscriptCandidateTracker, TranscriptChunk,
};
use std::sync::OnceLock;
use whisper_cpp_plus::whisper_cpp_plus_sys as whisper_ffi;

pub struct WhisperSpeechRecognizer {
    ctx: whisper_cpp_plus::WhisperContext,
    pending: Vec<f32>,
    sample_rate_hz: u32,
    input_silence_padding_ms: u64,
    candidate_tracker: TranscriptCandidateTracker,
}

const DEFAULT_INPUT_SILENCE_PADDING_MS: u64 = 250;

impl WhisperSpeechRecognizer {
    pub fn new(model_path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        Self::new_with_log_suppression(model_path, false)
    }

    pub fn new_quiet(model_path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        Self::new_with_log_suppression(model_path, true)
    }

    /// Creates a quiet recognizer without synthetic edge silence.
    ///
    /// This is intended for refinement passes that already operate over a wider rolling
    /// context window.
    pub fn new_quiet_without_input_padding(
        model_path: impl AsRef<std::path::Path>,
    ) -> anyhow::Result<Self> {
        Self::new_with_log_suppression_and_padding(model_path, true, 0)
    }

    fn new_with_log_suppression(
        model_path: impl AsRef<std::path::Path>,
        suppress_logs: bool,
    ) -> anyhow::Result<Self> {
        Self::new_with_log_suppression_and_padding(
            model_path,
            suppress_logs,
            DEFAULT_INPUT_SILENCE_PADDING_MS,
        )
    }

    fn new_with_log_suppression_and_padding(
        model_path: impl AsRef<std::path::Path>,
        suppress_logs: bool,
        input_silence_padding_ms: u64,
    ) -> anyhow::Result<Self> {
        configure_whisper_logging(suppress_logs);
        let ctx = whisper_cpp_plus::WhisperContext::new(model_path.as_ref())?;

        Ok(Self {
            ctx,
            pending: Vec::new(),
            sample_rate_hz: 16_000,
            input_silence_padding_ms,
            candidate_tracker: TranscriptCandidateTracker::new(),
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

    fn poll_transcript_text(&mut self) -> anyhow::Result<Option<String>> {
        if self.pending.is_empty() {
            return Ok(None);
        }

        let audio = std::mem::take(&mut self.pending);
        let audio =
            pad_samples_with_silence(audio, self.sample_rate_hz, self.input_silence_padding_ms);
        let text = self.ctx.transcribe(&audio)?;
        let text = text.trim();

        if text.is_empty() {
            return Ok(None);
        }

        Ok(Some(text.to_owned()))
    }

    /// Emits candidate lifecycle events for recognized audio.
    ///
    /// The current Whisper integration is final-only, so each recognition result maps to
    /// `CandidateStarted -> CandidateFinalized`. This method is the seam for future
    /// partial/streaming ASR to emit updates and replacements.
    ///
    /// ⚠️ This method and `poll_chunks` consume the same pending audio.
    /// Callers must treat them as alternative polling APIs and should not call both expecting
    /// duplicated output for the same buffered frames.
    pub fn poll_candidate_events(&mut self) -> anyhow::Result<Vec<TranscriptCandidateEvent>> {
        self.poll_candidate_events_with_finality(true)
    }

    pub fn poll_candidate_events_with_finality(
        &mut self,
        is_final: bool,
    ) -> anyhow::Result<Vec<TranscriptCandidateEvent>> {
        let Some(text) = self.poll_transcript_text()? else {
            return Ok(Vec::new());
        };

        Ok(self
            .candidate_tracker
            .ingest_candidate(text, None, is_final))
    }
}

fn pad_samples_with_silence(audio: Vec<f32>, sample_rate_hz: u32, padding_ms: u64) -> Vec<f32> {
    if audio.is_empty() || sample_rate_hz == 0 || padding_ms == 0 {
        return audio;
    }

    let padding_samples = (u64::from(sample_rate_hz) * padding_ms).div_ceil(1_000) as usize;
    if padding_samples == 0 {
        return audio;
    }

    let mut padded = Vec::with_capacity(
        audio
            .len()
            .saturating_add(padding_samples.saturating_mul(2)),
    );
    padded.extend(std::iter::repeat_n(0.0, padding_samples));
    padded.extend(audio);
    padded.extend(std::iter::repeat_n(0.0, padding_samples));
    padded
}

fn configure_whisper_logging(suppress_logs: bool) {
    static LOGGING_CONFIGURED: OnceLock<()> = OnceLock::new();
    if suppress_logs || !developer_diagnostics_enabled() {
        LOGGING_CONFIGURED.get_or_init(|| unsafe {
            whisper_ffi::whisper_log_set(Some(drop_whisper_log), std::ptr::null_mut());
        });
    }
}

#[cfg(test)]
mod tests {
    use super::pad_samples_with_silence;

    #[test]
    fn pads_samples_with_silence_on_both_ends() {
        let padded = pad_samples_with_silence(vec![0.5, -0.5], 1_000, 2);

        assert_eq!(padded, vec![0.0, 0.0, 0.5, -0.5, 0.0, 0.0]);
    }

    #[test]
    fn silence_padding_uses_ceiling_sample_count() {
        let padded = pad_samples_with_silence(vec![1.0], 16_000, 1);

        assert_eq!(padded.len(), 33);
        assert_eq!(padded[16], 1.0);
    }

    #[test]
    fn zero_padding_leaves_samples_unchanged() {
        let padded = pad_samples_with_silence(vec![0.25], 16_000, 0);

        assert_eq!(padded, vec![0.25]);
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

    /// Returns final transcript chunks for all currently buffered audio.
    ///
    /// ⚠️ This method and [`WhisperSpeechRecognizer::poll_candidate_events`] are alternative
    /// consumers over the same pending buffer. Calling one drains the audio for both paths.
    ///
    /// Prefer `poll_candidate_events` for new integrations and use this method as
    /// compatibility sugar until a unified transcript event stream fully replaces chunk polling.
    fn poll_chunks(&mut self) -> anyhow::Result<Vec<TranscriptChunk>> {
        let Some(text) = self.poll_transcript_text()? else {
            return Ok(Vec::new());
        };

        Ok(vec![TranscriptChunk {
            text,
            is_final: true,
        }])
    }
}
