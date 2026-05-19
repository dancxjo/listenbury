#[cfg(feature = "asr-whisper")]
use crate::cli::MicTranscribeCommand;
use crate::cli::TranscribeCommand;
#[cfg(feature = "asr-whisper")]
use crate::cli::model_paths::resolve_whisper_model;
#[cfg(feature = "asr-whisper")]
use anyhow::Context;
use anyhow::Result;
#[cfg(feature = "asr-whisper")]
use listenbury::audio::read_wav_as_whisper_frames;
#[cfg(feature = "asr-whisper")]
use listenbury::speech::recognizer::SpeechRecognizer;

#[cfg(feature = "asr-whisper")]
pub(crate) fn run_transcribe(command: TranscribeCommand) -> Result<()> {
    let Some(input_wav) = command.input_wav else {
        return super::mic_transcribe::run_mic_transcribe(MicTranscribeCommand {
            seconds: command.seconds,
            until_ctrl_c: command.until_ctrl_c,
            whisper_model: command.whisper_model,
            refine_whisper_model: command.refine_whisper_model,
            refine_window_seconds: command.refine_window_seconds,
            refine_interval_ms: command.refine_interval_ms,
            vad: command.vad,
            web: command.web,
            web_host: command.web_host,
            web_port: command.web_port,
        });
    };
    anyhow::ensure!(
        !command.web,
        "`transcribe --web` is microphone-only; omit the input WAV path"
    );

    let model_path = resolve_whisper_model(command.whisper_model)?;
    let mut recognizer = listenbury::WhisperSpeechRecognizer::new(&model_path)
        .with_context(|| format!("failed to load Whisper model at {}", model_path.display()))?;
    let frames = read_wav_as_whisper_frames(&input_wav, 1_600)
        .with_context(|| format!("failed to read WAV at {}", input_wav.display()))?;

    for frame in &frames {
        recognizer.push_frame(frame)?;
    }

    let chunks = recognizer.poll_chunks()?;
    if chunks.is_empty() {
        println!("No transcript chunks produced.");
        return Ok(());
    }

    for chunk in chunks {
        println!("{chunk:?}");
    }

    Ok(())
}

#[cfg(not(feature = "asr-whisper"))]
pub(crate) fn run_transcribe(_command: TranscribeCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `asr-whisper` feature")
}
