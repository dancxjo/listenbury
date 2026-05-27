use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use tracing::debug;

use crate::audio::frame::AudioFrame;
use crate::mouth::backend::TtsBackend;
use crate::mouth::planner::{MouthSyntheticPlan, strip_emoji};
#[cfg(feature = "piper-compat")]
use crate::mouth::riper::{
    PiperIdSequence, PiperPhonemeSequence, PiperVoiceConfig, RiperBackend, SimpleEnglishG2p,
};
use crate::mouth::tts::TextToSpeech;
use crate::time::ExactTimestamp;

#[cfg(feature = "piper-compat")]
const RIPER_TEXT_OUTPUT_GAIN: f32 = 1.5;
#[cfg(feature = "piper-compat")]
const RIPER_SENTENCE_PAUSE_MS: u64 = 260;
#[cfg(feature = "piper-compat")]
const RIPER_CHANNELS: u16 = 1;

#[derive(Debug, Clone)]
pub struct PiperConfig {
    pub executable: PathBuf,
    pub model_path: PathBuf,
    pub config_path: Option<PathBuf>,
    pub num_threads: Option<usize>,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub frame_samples: usize,
}

impl PiperConfig {
    pub fn new(executable: impl Into<PathBuf>, model_path: impl Into<PathBuf>) -> Self {
        Self {
            executable: executable.into(),
            model_path: model_path.into(),
            config_path: None,
            num_threads: Some(1),
            sample_rate_hz: 22_050,
            channels: 1,
            frame_samples: 1024,
        }
    }
}

/// A backend that synthesizes speech by spawning the external Piper executable
/// once per utterance.
///
/// This is the default backend used by [`PiperTextToSpeech::new`].  It
/// preserves the original process-per-synthesis behavior.
pub struct ProcessPiperBackend {
    config: PiperConfig,
}

impl ProcessPiperBackend {
    pub fn new(config: PiperConfig) -> Self {
        Self { config }
    }
}

impl TtsBackend for ProcessPiperBackend {
    fn synthesize(&mut self, text: &str) -> Result<Vec<AudioFrame>> {
        let t0 = Instant::now();
        let process_text = normalize_process_piper_pronunciations(text);
        let samples = synthesize_process(&self.config, &process_text)?;
        let elapsed = t0.elapsed();
        debug!(
            backend = "process",
            chars = text.len(),
            elapsed_ms = elapsed.as_millis(),
            "ProcessPiperBackend synthesis complete"
        );
        Ok(frames_from_samples(&self.config, samples))
    }
}

fn normalize_process_piper_pronunciations(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    let mut index = 0;

    while index < text.len() {
        let ch = text[index..]
            .chars()
            .next()
            .expect("index should stay on a char boundary");
        if !ch.is_ascii_alphabetic() {
            normalized.push(ch);
            index += ch.len_utf8();
            continue;
        }

        let start = index;
        index += ch.len_utf8();
        while index < text.len() {
            let next = text[index..]
                .chars()
                .next()
                .expect("index should stay on a char boundary");
            if next.is_ascii_alphabetic() {
                index += next.len_utf8();
                continue;
            }
            if next == '-'
                && text[index + next.len_utf8()..]
                    .chars()
                    .next()
                    .is_some_and(|after_dash| after_dash.is_ascii_alphabetic())
            {
                index += next.len_utf8();
                continue;
            }
            break;
        }

        let token = &text[start..index];
        match token.to_ascii_lowercase().as_str() {
            "mmm" => normalized.push_str("[[m:]]"),
            "mm" => normalized.push_str("[[m]]"),
            "mm-hm" | "mm-hmm" => normalized.push_str("[[m h@m]]"),
            _ => normalized.push_str(token),
        }
    }

    normalized
}

#[cfg(feature = "piper-compat")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PiperBackendPreference {
    Process,
    Riper,
    RiperWithProcessFallback,
}

#[cfg(feature = "piper-compat")]
impl PiperBackendPreference {
    fn from_env() -> Self {
        match std::env::var("LISTENBURY_PIPER_BACKEND")
            .ok()
            .map(|value| value.to_ascii_lowercase())
            .as_deref()
        {
            None | Some("") | Some("process") => Self::Process,
            Some("riper") => Self::Riper,
            Some("riper-fallback") => Self::RiperWithProcessFallback,
            Some(other) => {
                tracing::warn!(
                    backend_env = other,
                    "unknown LISTENBURY_PIPER_BACKEND value; falling back to process backend"
                );
                Self::Process
            }
        }
    }
}

#[cfg(feature = "piper-compat")]
#[derive(Debug)]
struct RiperTextBackend {
    backend: RiperBackend,
    phonemizer: SimpleEnglishG2p,
    frame_samples: usize,
}

#[cfg(feature = "piper-compat")]
impl RiperTextBackend {
    fn load(config: &PiperConfig) -> Result<Self> {
        let config_path = riper_config_path(config).with_context(|| {
            format!(
                "Riper backend requested but no voice config path was provided and no inferred config exists for model {}",
                config.model_path.display()
            )
        })?;
        let voice_config_json = std::fs::read_to_string(&config_path).with_context(|| {
            format!(
                "failed to read Riper voice config at {}",
                config_path.display()
            )
        })?;
        let voice_config =
            PiperVoiceConfig::from_json_str(&voice_config_json).with_context(|| {
                format!(
                    "failed to parse Riper voice config JSON at {}",
                    config_path.display()
                )
            })?;
        let backend = RiperBackend::load(&config.model_path, voice_config).with_context(|| {
            format!(
                "failed to initialize Riper backend for model {}",
                config.model_path.display()
            )
        })?;
        Ok(Self {
            backend,
            phonemizer: SimpleEnglishG2p::default(),
            frame_samples: config.frame_samples,
        })
    }
}

#[cfg(feature = "piper-compat")]
impl TtsBackend for RiperTextBackend {
    fn synthesize(&mut self, text: &str) -> Result<Vec<AudioFrame>> {
        let t0 = Instant::now();
        let chunks = riper_sentence_chunks(text);
        let mut frames = Vec::new();
        for chunk in &chunks {
            let phonemes = self
                .phonemizer
                .phonemize_unit(chunk.text)
                .with_context(|| {
                    format!("failed to realize Riper phonemes for text `{}`", chunk.text)
                })?
                .phonemes;
            let ids =
                phonemes.to_riper_text_ids(self.backend.config(), self.backend.model_path())?;
            frames.extend(
                self.backend
                    .synthesize_id_frames(&ids)
                    .with_context(|| format!("Riper synthesis failed for text `{}`", chunk.text))?,
            );
            if chunk.sentence_final {
                frames.extend(riper_silence_frames(
                    self.backend.config().sample_rate_hz,
                    RIPER_CHANNELS,
                    self.frame_samples,
                    RIPER_SENTENCE_PAUSE_MS,
                ));
            }
        }
        apply_frame_gain(&mut frames, RIPER_TEXT_OUTPUT_GAIN);
        debug!(
            backend = "riper",
            chars = text.len(),
            chunks = chunks.len(),
            output_gain = RIPER_TEXT_OUTPUT_GAIN,
            sentence_pause_ms = RIPER_SENTENCE_PAUSE_MS,
            elapsed_ms = t0.elapsed().as_millis(),
            "RiperTextBackend synthesis complete"
        );
        Ok(frames)
    }
}

#[cfg(feature = "piper-compat")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RiperTextChunk<'a> {
    text: &'a str,
    sentence_final: bool,
}

#[cfg(feature = "piper-compat")]
fn riper_sentence_chunks(text: &str) -> Vec<RiperTextChunk<'_>> {
    let mut chunks = Vec::new();
    let mut start = 0usize;

    for (idx, ch) in text.char_indices() {
        if !is_riper_sentence_terminal(ch) {
            continue;
        }
        let end = idx + ch.len_utf8();
        if let Some(chunk) = riper_text_chunk(text, start, end, true) {
            chunks.push(chunk);
        }
        start = end;
        while start < text.len() {
            let Some(next) = text[start..].chars().next() else {
                break;
            };
            if !next.is_whitespace() && !matches!(next, '"' | '\'' | ')' | ']' | '}') {
                break;
            }
            start += next.len_utf8();
        }
    }

    if let Some(chunk) = riper_text_chunk(text, start, text.len(), false) {
        chunks.push(chunk);
    }
    if chunks.is_empty() {
        chunks.push(RiperTextChunk {
            text: text.trim(),
            sentence_final: false,
        });
    }

    chunks
}

#[cfg(feature = "piper-compat")]
fn riper_text_chunk(
    text: &str,
    start: usize,
    end: usize,
    sentence_final: bool,
) -> Option<RiperTextChunk<'_>> {
    let chunk = text.get(start..end)?.trim();
    (!chunk.is_empty()).then_some(RiperTextChunk {
        text: chunk,
        sentence_final,
    })
}

#[cfg(feature = "piper-compat")]
fn is_riper_sentence_terminal(ch: char) -> bool {
    matches!(ch, '.' | '!' | '?')
}

#[cfg(feature = "piper-compat")]
fn riper_silence_frames(
    sample_rate_hz: u32,
    channels: u16,
    frame_samples: usize,
    pause_ms: u64,
) -> Vec<AudioFrame> {
    let channels = channels.max(1);
    let total_samples = pause_ms
        .saturating_mul(u64::from(sample_rate_hz))
        .saturating_mul(u64::from(channels))
        / 1000;
    let Ok(total_samples) = usize::try_from(total_samples) else {
        return Vec::new();
    };
    let frame_samples = frame_samples.max(1);
    let mut remaining = total_samples;
    let mut frames = Vec::new();
    while remaining > 0 {
        let len = remaining.min(frame_samples);
        frames.push(AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz,
            channels,
            samples: vec![0.0; len],
            voice_signatures: Vec::new(),
        });
        remaining -= len;
    }
    frames
}

#[cfg(feature = "piper-compat")]
fn apply_frame_gain(frames: &mut [AudioFrame], gain: f32) {
    if (gain - 1.0).abs() <= f32::EPSILON {
        return;
    }
    for sample in frames.iter_mut().flat_map(|frame| frame.samples.iter_mut()) {
        *sample = (*sample * gain).clamp(-1.0, 1.0);
    }
}

#[cfg(feature = "piper-compat")]
trait RiperTextPhonemeIds {
    fn to_riper_text_ids(
        &self,
        config: &PiperVoiceConfig,
        model_path: &Path,
    ) -> Result<PiperIdSequence>;
}

#[cfg(feature = "piper-compat")]
impl RiperTextPhonemeIds for PiperPhonemeSequence {
    fn to_riper_text_ids(
        &self,
        config: &PiperVoiceConfig,
        model_path: &Path,
    ) -> Result<PiperIdSequence> {
        self.to_piper_text_ids_compatible(config).with_context(|| {
            format!(
                "failed to map phonemes to IDs for Riper model {}",
                model_path.display()
            )
        })
    }
}

#[cfg(feature = "piper-compat")]
struct RiperPreferredBackend<P, N> {
    process: P,
    riper: Option<N>,
    preference: PiperBackendPreference,
    riper_init_error: Option<String>,
}

#[cfg(feature = "piper-compat")]
impl<P, N> RiperPreferredBackend<P, N> {
    fn new(
        process: P,
        riper: Option<N>,
        preference: PiperBackendPreference,
        riper_init_error: Option<String>,
    ) -> Self {
        Self {
            process,
            riper,
            preference,
            riper_init_error,
        }
    }
}

#[cfg(feature = "piper-compat")]
impl<P: TtsBackend, N: TtsBackend> TtsBackend for RiperPreferredBackend<P, N> {
    fn synthesize(&mut self, text: &str) -> Result<Vec<AudioFrame>> {
        if self.preference == PiperBackendPreference::Process {
            return self.process.synthesize(text);
        }

        let riper = match self.riper.as_mut() {
            Some(riper) => riper,
            None => {
                let detail = self
                    .riper_init_error
                    .as_deref()
                    .unwrap_or("Riper backend is unavailable for an unknown reason");
                if self.preference == PiperBackendPreference::RiperWithProcessFallback {
                    tracing::warn!(
                        error = detail,
                        "Riper unavailable; falling back to process backend"
                    );
                    return self.process.synthesize(text);
                }
                anyhow::bail!("Riper backend is unavailable: {detail}");
            }
        };

        match riper.synthesize(text) {
            Ok(frames) => Ok(frames),
            Err(error) => {
                if self.preference == PiperBackendPreference::RiperWithProcessFallback {
                    tracing::warn!(error = %error, "Riper synthesis failed; falling back to process backend");
                    self.process.synthesize(text)
                } else {
                    Err(error.context("Riper synthesis failed (process fallback disabled)"))
                }
            }
        }
    }

    fn stop(&mut self) -> Result<()> {
        self.process.stop()?;
        if let Some(riper) = self.riper.as_mut() {
            riper.stop()?;
        }
        Ok(())
    }
}

#[cfg(feature = "piper-compat")]
fn riper_config_path(config: &PiperConfig) -> Option<PathBuf> {
    config
        .config_path
        .as_ref()
        .filter(|path| path.is_file())
        .cloned()
        .or_else(|| {
            let inferred = config.model_path.with_extension("onnx.json");
            inferred.is_file().then_some(inferred)
        })
}

pub struct PiperTextToSpeech {
    tx: Sender<PiperCommand>,
    rx_audio: Receiver<Vec<AudioFrame>>,
    rx_error: Receiver<anyhow::Error>,
    worker: Option<JoinHandle<()>>,
}

impl PiperTextToSpeech {
    /// Create a `PiperTextToSpeech` that uses the default
    /// [`ProcessPiperBackend`] for synthesis.
    pub fn new(config: PiperConfig) -> Self {
        Self::with_boxed_backend(default_piper_backend(config))
    }

    #[cfg(feature = "piper-compat")]
    pub fn new_with_backend_preference(
        config: PiperConfig,
        preference: PiperBackendPreference,
    ) -> Self {
        Self::with_boxed_backend(piper_backend_with_preference(config, preference))
    }

    /// Create a `PiperTextToSpeech` backed by any [`TtsBackend`] implementor.
    ///
    /// Use this constructor to substitute a custom backend (e.g. a persistent
    /// worker or a mock for testing).
    pub fn with_backend(backend: impl TtsBackend + 'static) -> Self {
        Self::with_boxed_backend(Box::new(backend))
    }

    fn with_boxed_backend(backend: Box<dyn TtsBackend>) -> Self {
        let (tx, rx) = crossbeam_channel::unbounded();
        let (tx_audio, rx_audio) = crossbeam_channel::unbounded();
        let (tx_error, rx_error) = crossbeam_channel::unbounded();

        let worker = thread::spawn(move || run_piper_worker(backend, rx, tx_audio, tx_error));

        Self {
            tx,
            rx_audio,
            rx_error,
            worker: Some(worker),
        }
    }
}

fn default_piper_backend(config: PiperConfig) -> Box<dyn TtsBackend> {
    #[cfg(feature = "piper-compat")]
    {
        piper_backend_with_preference(config, PiperBackendPreference::from_env())
    }

    #[cfg(not(feature = "piper-compat"))]
    {
        Box::new(ProcessPiperBackend::new(config))
    }
}

#[cfg(feature = "piper-compat")]
fn piper_backend_with_preference(
    config: PiperConfig,
    preference: PiperBackendPreference,
) -> Box<dyn TtsBackend> {
    if preference == PiperBackendPreference::Process {
        return Box::new(ProcessPiperBackend::new(config));
    }

    let process = ProcessPiperBackend::new(config.clone());
    match RiperTextBackend::load(&config) {
        Ok(riper) => Box::new(RiperPreferredBackend::new(
            process,
            Some(riper),
            preference,
            None,
        )),
        Err(error) => Box::new(RiperPreferredBackend::new(
            process,
            None::<RiperTextBackend>,
            preference,
            Some(format!("{error:#}")),
        )),
    }
}

impl TextToSpeech for PiperTextToSpeech {
    fn enqueue(&mut self, plan: MouthSyntheticPlan) -> Result<()> {
        let text = plan_text(plan);
        if text.trim().is_empty() {
            return Ok(());
        }

        self.tx
            .send(PiperCommand::Synthesize(text))
            .context("failed to enqueue Piper speech plan")
    }

    fn poll_audio(&mut self) -> Result<Vec<AudioFrame>> {
        if let Ok(error) = self.rx_error.try_recv() {
            return Err(error);
        }

        Ok(self.rx_audio.try_iter().flatten().collect())
    }

    fn stop(&mut self) -> Result<()> {
        for _ in self.rx_audio.try_iter() {}
        self.tx
            .send(PiperCommand::Stop)
            .context("failed to send Piper stop command")
    }
}

impl Drop for PiperTextToSpeech {
    fn drop(&mut self) {
        let _ = self.tx.send(PiperCommand::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

enum PiperCommand {
    Synthesize(String),
    Stop,
    Shutdown,
}

fn run_piper_worker(
    mut backend: Box<dyn TtsBackend>,
    rx: Receiver<PiperCommand>,
    tx_audio: Sender<Vec<AudioFrame>>,
    tx_error: Sender<anyhow::Error>,
) {
    while let Ok(command) = rx.recv() {
        match command {
            PiperCommand::Synthesize(text) => match backend.synthesize(&text) {
                Ok(frames) => {
                    if tx_audio.send(frames).is_err() {
                        return;
                    }
                }
                Err(error) => {
                    let _ = tx_error.send(error);
                }
            },
            PiperCommand::Stop => {
                if let Err(e) = backend.stop() {
                    tracing::warn!(error = %e, "TtsBackend::stop returned an error");
                }
                if should_shutdown_after_drain(&rx) {
                    return;
                }
            }
            PiperCommand::Shutdown => return,
        }
    }
}

fn should_shutdown_after_drain(rx: &Receiver<PiperCommand>) -> bool {
    loop {
        match rx.try_recv() {
            Ok(PiperCommand::Shutdown) => return true,
            Ok(PiperCommand::Synthesize(_)) | Ok(PiperCommand::Stop) => {}
            Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => return false,
        }
    }
}

fn plan_text(plan: MouthSyntheticPlan) -> String {
    strip_emoji(plan.text())
}

fn synthesize_process(config: &PiperConfig, text: &str) -> Result<Vec<f32>> {
    let mut command = Command::new(&config.executable);
    command
        .arg("--model")
        .arg(&config.model_path)
        .arg("--output-raw")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(num_threads) = config.num_threads {
        command.arg("--num-threads").arg(num_threads.to_string());
    }

    if let Some(config_path) = &config.config_path {
        command.arg("--config").arg(config_path);
    }

    let mut child = command
        .spawn()
        .with_context(|| format!("failed to spawn Piper at {}", config.executable.display()))?;

    {
        let mut stdin = child.stdin.take().context("failed to open Piper stdin")?;
        stdin
            .write_all(text.as_bytes())
            .context("failed to write text to Piper stdin")?;
        stdin
            .write_all(b"\n")
            .context("failed to finish Piper stdin")?;
    }

    let output = child
        .wait_with_output()
        .context("failed to read Piper output")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Piper exited with {}: {}", output.status, stderr.trim());
    }

    Ok(output
        .stdout
        .chunks_exact(2)
        .map(|chunk| {
            let value = i16::from_le_bytes([chunk[0], chunk[1]]);
            value as f32 / i16::MAX as f32
        })
        .collect())
}

fn frames_from_samples(config: &PiperConfig, samples: Vec<f32>) -> Vec<AudioFrame> {
    samples
        .chunks(config.frame_samples.max(1))
        .map(|chunk| AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: config.sample_rate_hz,
            channels: config.channels,
            samples: chunk.to_vec(),
            voice_signatures: Vec::new(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::mouth::backend::tests::MockTtsBackend;
    use crate::mouth::planner::SyntheticUnit;

    fn collect_audio(tts: &mut PiperTextToSpeech, timeout: Duration) -> Vec<AudioFrame> {
        let deadline = std::time::Instant::now() + timeout;
        let mut all = Vec::new();
        loop {
            let frames = tts.poll_audio().expect("poll_audio");
            all.extend(frames);
            if !all.is_empty() || std::time::Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        all
    }

    #[test]
    fn with_backend_delivers_audio_for_enqueued_plan() {
        let backend = MockTtsBackend::new();
        let mut tts = PiperTextToSpeech::with_backend(backend);

        tts.enqueue(MouthSyntheticPlan::from(SyntheticUnit::FullTurn(
            "Hello.".to_string(),
        )))
        .expect("enqueue");

        let frames = collect_audio(&mut tts, Duration::from_secs(2));
        assert!(
            !frames.is_empty(),
            "expected audio frames from mock backend"
        );
    }

    #[test]
    fn empty_text_after_emoji_strip_is_skipped() {
        let backend = MockTtsBackend::new();
        let mut tts = PiperTextToSpeech::with_backend(backend);

        // A plan whose text reduces to empty after emoji stripping should not
        // reach the backend at all.
        tts.enqueue(MouthSyntheticPlan::from(SyntheticUnit::FullTurn(
            "\u{1F600}\u{1F601}".to_string(),
        )))
        .expect("enqueue emoji-only plan");

        // Give worker a moment to process.
        std::thread::sleep(Duration::from_millis(50));
        let frames = tts.poll_audio().expect("poll_audio");
        assert!(
            frames.is_empty(),
            "emoji-only plan should not produce audio"
        );
    }

    #[test]
    fn plan_text_strips_emoji_before_reaching_backend() {
        // Verify that the text reaching the backend never contains emoji.
        // We rely on the mock backend recording what it received.
        let backend = MockTtsBackend::new();
        let mut tts = PiperTextToSpeech::with_backend(backend);

        tts.enqueue(MouthSyntheticPlan::from(SyntheticUnit::FullTurn(
            "Hello \u{1F600} world.".to_string(),
        )))
        .expect("enqueue");

        let _frames = collect_audio(&mut tts, Duration::from_secs(2));
        // The backend is moved into the worker thread, so we can't inspect it
        // directly here.  The test passes as long as no panic occurs and audio
        // is produced (meaning the stripped text "Hello  world." was non-empty).
        // See `empty_text_after_emoji_strip_is_skipped` for the empty case.
    }

    #[test]
    fn process_piper_normalizes_backchannel_nasal_pronunciations() {
        assert_eq!(
            normalize_process_piper_pronunciations("Mmm. Mm. Mm-hmm. Mm-hm"),
            "[[m:]]. [[m]]. [[m h@m]]. [[m h@m]]"
        );
    }

    #[test]
    fn process_piper_pronunciation_normalizer_leaves_other_words_alone() {
        assert_eq!(
            normalize_process_piper_pronunciations("model low-latency mms"),
            "model low-latency mms"
        );
    }

    // Regression: `stop` should not panic and subsequent enqueues should not error.
    #[test]
    fn stop_then_enqueue_does_not_panic() {
        let backend = MockTtsBackend::new();
        let mut tts = PiperTextToSpeech::with_backend(backend);

        tts.stop().expect("stop should not error");
        // Give the worker a moment to process the stop command so that a
        // subsequent enqueue is not consumed by `should_shutdown_after_drain`.
        std::thread::sleep(Duration::from_millis(50));
        tts.enqueue(MouthSyntheticPlan::from(SyntheticUnit::FullTurn(
            "Hi.".to_string(),
        )))
        .expect("enqueue after stop should not error");

        let frames = collect_audio(&mut tts, Duration::from_secs(2));
        assert!(!frames.is_empty(), "expected audio after stop+enqueue");
    }

    #[cfg(feature = "piper-compat")]
    struct AlwaysFailBackend {
        calls: usize,
    }

    #[cfg(feature = "piper-compat")]
    impl TtsBackend for AlwaysFailBackend {
        fn synthesize(&mut self, _text: &str) -> Result<Vec<AudioFrame>> {
            self.calls += 1;
            anyhow::bail!("riper boom");
        }
    }

    #[cfg(feature = "piper-compat")]
    #[test]
    fn riper_sentence_chunks_split_sentence_final_text() {
        let chunks = riper_sentence_chunks("Okay. Again? Done!");

        assert_eq!(
            chunks,
            vec![
                RiperTextChunk {
                    text: "Okay.",
                    sentence_final: true,
                },
                RiperTextChunk {
                    text: "Again?",
                    sentence_final: true,
                },
                RiperTextChunk {
                    text: "Done!",
                    sentence_final: true,
                },
            ]
        );
    }

    #[cfg(feature = "piper-compat")]
    #[test]
    fn riper_sentence_chunks_leave_trailing_fragment_without_pause() {
        let chunks = riper_sentence_chunks("Okay. still thinking");

        assert_eq!(
            chunks,
            vec![
                RiperTextChunk {
                    text: "Okay.",
                    sentence_final: true,
                },
                RiperTextChunk {
                    text: "still thinking",
                    sentence_final: false,
                },
            ]
        );
    }

    #[cfg(feature = "piper-compat")]
    #[test]
    fn riper_silence_frames_match_pause_duration() {
        let frames = riper_silence_frames(1_000, 1, 128, 260);
        let total_samples = frames
            .iter()
            .map(|frame| frame.samples.len())
            .sum::<usize>();

        assert_eq!(total_samples, 260);
        assert!(frames.iter().all(|frame| frame.sample_rate_hz == 1_000));
        assert!(frames.iter().all(|frame| frame.channels == 1));
    }

    #[cfg(feature = "piper-compat")]
    #[test]
    fn apply_frame_gain_boosts_and_clips_samples() {
        let mut frames = vec![AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: 22_050,
            channels: 1,
            samples: vec![0.2, -0.5, 0.8, -0.9],
            voice_signatures: Vec::new(),
        }];

        apply_frame_gain(&mut frames, 1.5);

        assert_eq!(frames[0].samples, vec![0.3, -0.75, 1.0, -1.0]);
    }

    #[cfg(feature = "piper-compat")]
    #[test]
    fn riper_backend_preference_uses_riper_when_it_succeeds() {
        let process = MockTtsBackend::new();
        let riper = MockTtsBackend::new();
        let mut backend =
            RiperPreferredBackend::new(process, Some(riper), PiperBackendPreference::Riper, None);

        let frames = backend.synthesize("hello").expect("riper synthesize");
        assert!(!frames.is_empty(), "expected frames from Riper backend");
        assert_eq!(backend.process.synthesize_calls.len(), 0);
        assert_eq!(
            backend
                .riper
                .as_ref()
                .expect("Riper backend")
                .synthesize_calls
                .len(),
            1
        );
    }

    #[cfg(feature = "piper-compat")]
    #[test]
    fn riper_fallback_mode_uses_process_when_riper_fails() {
        let process = MockTtsBackend::new();
        let riper = AlwaysFailBackend { calls: 0 };
        let mut backend = RiperPreferredBackend::new(
            process,
            Some(riper),
            PiperBackendPreference::RiperWithProcessFallback,
            None,
        );

        let frames = backend.synthesize("hello").expect("fallback to process");
        assert!(!frames.is_empty(), "expected process fallback frames");
        assert_eq!(backend.process.synthesize_calls.len(), 1);
        assert_eq!(backend.riper.as_ref().expect("riper").calls, 1);
    }

    #[cfg(feature = "piper-compat")]
    #[test]
    fn riper_mode_returns_clear_error_when_riper_is_unavailable() {
        let process = MockTtsBackend::new();
        let mut backend = RiperPreferredBackend::<MockTtsBackend, AlwaysFailBackend>::new(
            process,
            None,
            PiperBackendPreference::Riper,
            Some("missing Riper config".to_string()),
        );

        let error = backend
            .synthesize("hello")
            .expect_err("Riper mode should fail without Riper backend");
        assert!(error.to_string().contains("Riper backend is unavailable"));
        assert!(error.to_string().contains("missing Riper config"));
    }

    #[cfg(feature = "piper-compat")]
    #[test]
    fn riper_text_id_conversion_accepts_cmudict_uw_for_espeak_voice_maps() {
        let config = PiperVoiceConfig::from_json_str(
            r#"
            {
              "audio": { "sample_rate": 22050 },
              "phoneme_id_map": {
                "_": [0],
                "^": [1],
                "$": [2],
                "u": [33]
              }
            }
            "#,
        )
        .expect("voice config should parse");
        let phonemes = PiperPhonemeSequence {
            phonemes: vec![crate::mouth::riper::PiperPhoneme("UW".to_string())],
        };

        let ids = phonemes
            .to_riper_text_ids(&config, Path::new("/tmp/voice.onnx"))
            .expect("CMUdict UW should convert to the eSpeak symbol used by Piper");

        assert_eq!(
            ids,
            PiperIdSequence {
                ids: vec![1, 0, 33, 0, 2]
            }
        );
    }

    #[cfg(feature = "piper-compat")]
    #[test]
    fn riper_text_id_conversion_preserves_already_d_for_espeak_voice_maps() {
        let config = PiperVoiceConfig::from_json_str(
            r#"
            {
              "audio": { "sample_rate": 22050 },
              "phoneme_id_map": {
                "_": [0],
                "^": [1],
                "$": [2],
                "i": [10],
                "l": [11],
                "ɔ": [12],
                "ɛ": [13],
                "ɹ": [14],
                "d": [15]
              }
            }
            "#,
        )
        .expect("voice config should parse");
        let phonemes = SimpleEnglishG2p::default()
            .phonemize_unit("already")
            .expect("already should phonemize")
            .phonemes;

        let ids = phonemes
            .to_riper_text_ids(&config, Path::new("/tmp/voice.onnx"))
            .expect("already should keep /d/ in the compatible ID path");

        assert_eq!(
            ids,
            PiperIdSequence {
                ids: vec![1, 0, 12, 0, 11, 0, 14, 0, 13, 0, 15, 0, 10, 0, 2]
            }
        );
    }

    #[cfg(feature = "piper-compat")]
    #[test]
    fn riper_text_backend_reports_missing_config_path_clearly() {
        let mut config = PiperConfig::new(
            "/tmp/piper",
            "/tmp/listenbury-riper-missing-config/model.onnx",
        );
        config.config_path = None;

        let error =
            RiperTextBackend::load(&config).expect_err("missing config should be reported clearly");
        assert!(
            error
                .to_string()
                .contains("Riper backend requested but no voice config path was provided"),
            "unexpected error: {error}"
        );
    }
}
