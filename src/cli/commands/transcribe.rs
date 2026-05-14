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
    let model_path = resolve_whisper_model(command.whisper_model)?;
    let mut recognizer = listenbury::WhisperSpeechRecognizer::new(&model_path)
        .with_context(|| format!("failed to load Whisper model at {}", model_path.display()))?;
    let frames = read_wav_as_whisper_frames(&command.input_wav, 1_600)
        .with_context(|| format!("failed to read WAV at {}", command.input_wav.display()))?;

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
