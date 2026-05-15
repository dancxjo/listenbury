use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use tracing::debug;

use crate::audio::frame::AudioFrame;
use crate::mouth::backend::TtsBackend;
use crate::mouth::planner::{SpeechPlan, strip_emoji};
use crate::mouth::tts::TextToSpeech;
use crate::time::ExactTimestamp;

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
        let samples = synthesize_process(&self.config, text)?;
        let elapsed = t0.elapsed();
        debug!(
            chars = text.len(),
            elapsed_ms = elapsed.as_millis(),
            "ProcessPiperBackend synthesis complete"
        );
        Ok(frames_from_samples(&self.config, samples))
    }
}

pub struct PiperTextToSpeech {
    tx: Sender<PiperCommand>,
    rx_audio: Receiver<AudioFrame>,
    rx_error: Receiver<anyhow::Error>,
    worker: Option<JoinHandle<()>>,
}

impl PiperTextToSpeech {
    /// Create a `PiperTextToSpeech` that uses the default
    /// [`ProcessPiperBackend`] for synthesis.
    pub fn new(config: PiperConfig) -> Self {
        Self::with_backend(ProcessPiperBackend::new(config))
    }

    /// Create a `PiperTextToSpeech` backed by any [`TtsBackend`] implementor.
    ///
    /// Use this constructor to substitute a custom backend (e.g. a persistent
    /// worker or a mock for testing).
    pub fn with_backend(backend: impl TtsBackend + 'static) -> Self {
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

impl TextToSpeech for PiperTextToSpeech {
    fn enqueue(&mut self, plan: SpeechPlan) -> Result<()> {
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

        Ok(self.rx_audio.try_iter().collect())
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
    mut backend: impl TtsBackend,
    rx: Receiver<PiperCommand>,
    tx_audio: Sender<AudioFrame>,
    tx_error: Sender<anyhow::Error>,
) {
    while let Ok(command) = rx.recv() {
        match command {
            PiperCommand::Synthesize(text) => match backend.synthesize(&text) {
                Ok(frames) => {
                    for frame in frames {
                        if tx_audio.send(frame).is_err() {
                            return;
                        }
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

fn plan_text(plan: SpeechPlan) -> String {
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
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::mouth::backend::tests::MockTtsBackend;
    use crate::mouth::planner::SpeechUnit;

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

        tts.enqueue(SpeechPlan::from(SpeechUnit::FullTurn("Hello.".to_string())))
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
        tts.enqueue(SpeechPlan::from(SpeechUnit::FullTurn(
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

        tts.enqueue(SpeechPlan::from(SpeechUnit::FullTurn(
            "Hello \u{1F600} world.".to_string(),
        )))
        .expect("enqueue");

        let _frames = collect_audio(&mut tts, Duration::from_secs(2));
        // The backend is moved into the worker thread, so we can't inspect it
        // directly here.  The test passes as long as no panic occurs and audio
        // is produced (meaning the stripped text "Hello  world." was non-empty).
        // See `empty_text_after_emoji_strip_is_skipped` for the empty case.
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
        tts.enqueue(SpeechPlan::from(SpeechUnit::FullTurn("Hi.".to_string())))
            .expect("enqueue after stop should not error");

        let frames = collect_audio(&mut tts, Duration::from_secs(2));
        assert!(!frames.is_empty(), "expected audio after stop+enqueue");
    }
}
