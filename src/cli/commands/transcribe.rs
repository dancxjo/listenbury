use crate::cli::TranscribeSyntheticCommand;
#[cfg(feature = "asr-whisper")]
use crate::cli::model_paths::resolve_whisper_model;
#[cfg(feature = "asr-whisper")]
use anyhow::Context;
use anyhow::Result;
#[cfg(feature = "asr-whisper")]
use listenbury::audio::frame::AudioFrame;
#[cfg(feature = "asr-whisper")]
use listenbury::speech::recognizer::SpeechRecognizer;
#[cfg(feature = "asr-whisper")]
use listenbury::time::ExactTimestamp;

#[cfg(feature = "asr-whisper")]
pub(crate) fn run_transcribe_synthetic(command: TranscribeSyntheticCommand) -> Result<()> {
    let model_path = resolve_whisper_model(command.whisper_model)?;
    let mut recognizer = listenbury::WhisperSpeechRecognizer::new(&model_path)
        .with_context(|| format!("failed to load Whisper model at {}", model_path.display()))?;

    recognizer.push_frame(&AudioFrame {
        captured_at: ExactTimestamp::now(),
        sample_rate_hz: 16_000,
        channels: 1,
        samples: vec![0.0; 16_000],
    })?;

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
pub(crate) fn run_transcribe_synthetic(_command: TranscribeSyntheticCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `asr-whisper` feature")
}
