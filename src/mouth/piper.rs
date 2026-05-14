use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender, TryRecvError};

use crate::audio::frame::AudioFrame;
use crate::mouth::planner::SpeechPlan;
use crate::mouth::tts::TextToSpeech;
use crate::time::ExactTimestamp;

#[derive(Debug, Clone)]
pub struct PiperConfig {
    pub executable: PathBuf,
    pub model_path: PathBuf,
    pub config_path: Option<PathBuf>,
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
            sample_rate_hz: 22_050,
            channels: 1,
            frame_samples: 1024,
        }
    }
}

pub struct PiperTextToSpeech {
    tx: Sender<PiperCommand>,
    rx_audio: Receiver<AudioFrame>,
    rx_error: Receiver<anyhow::Error>,
    worker: Option<JoinHandle<()>>,
}

impl PiperTextToSpeech {
    pub fn new(config: PiperConfig) -> Self {
        let (tx, rx) = crossbeam_channel::unbounded();
        let (tx_audio, rx_audio) = crossbeam_channel::unbounded();
        let (tx_error, rx_error) = crossbeam_channel::unbounded();

        let worker = thread::spawn(move || run_piper_worker(config, rx, tx_audio, tx_error));

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
    config: PiperConfig,
    rx: Receiver<PiperCommand>,
    tx_audio: Sender<AudioFrame>,
    tx_error: Sender<anyhow::Error>,
) {
    while let Ok(command) = rx.recv() {
        match command {
            PiperCommand::Synthesize(text) => match synthesize(&config, &text) {
                Ok(samples) => {
                    for frame in frames_from_samples(&config, samples) {
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
    plan.text().to_string()
}

fn synthesize(config: &PiperConfig, text: &str) -> Result<Vec<f32>> {
    let mut command = Command::new(&config.executable);
    command
        .arg("--model")
        .arg(&config.model_path)
        .arg("--output-raw")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

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
