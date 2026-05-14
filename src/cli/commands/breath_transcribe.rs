use crate::cli::BreathTranscribeCommand;
#[cfg(feature = "asr-whisper")]
use crate::cli::model_paths::resolve_whisper_model;
#[cfg(feature = "asr-whisper")]
use anyhow::Context;
use anyhow::Result;
#[cfg(feature = "asr-whisper")]
use listenbury::audio::read_wav_as_audio_frames;
#[cfg(feature = "asr-whisper")]
use listenbury::speech::recognizer::SpeechRecognizer;
#[cfg(feature = "asr-whisper")]
use listenbury::{BreathAsrConfig, WhisperSpeechRecognizer, collect_breath_segments};

#[cfg(feature = "asr-whisper")]
pub(crate) fn run_breath_transcribe(command: BreathTranscribeCommand) -> Result<()> {
    let model_path = resolve_whisper_model(command.whisper_model)?;
    let mut recognizer = WhisperSpeechRecognizer::new(&model_path)
        .with_context(|| format!("failed to load Whisper model at {}", model_path.display()))?;

    let frames = read_wav_as_audio_frames(&command.input_wav, 160)
        .with_context(|| format!("failed to read WAV {}", command.input_wav.display()))?;
    let config = BreathAsrConfig {
        pre_roll_ms: command.pre_roll_ms,
        trailing_pad_ms: command.trailing_pad_ms,
        min_group_ms: command.min_group_ms,
        max_group_ms: command.max_group_ms,
    };
    let segments = collect_breath_segments(&frames, config)?;

    if segments.is_empty() {
        println!("No breath groups detected.");
        return Ok(());
    }

    for (idx, segment) in segments.iter().enumerate() {
        for frame in &segment.frames {
            recognizer.push_frame(frame)?;
        }
        let transcript = recognizer
            .poll_chunks()?
            .into_iter()
            .map(|chunk| chunk.text)
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "group={} start_ms={} end_ms={} duration_ms={} start_unix_nanos={} end_unix_nanos={} transcript={}",
            idx + 1,
            segment.start_ms,
            segment.end_ms,
            segment.duration_ms(),
            segment.start_captured_at.unix_nanos,
            segment.end_captured_at.unix_nanos,
            transcript
        );
    }

    Ok(())
}

#[cfg(not(feature = "asr-whisper"))]
pub(crate) fn run_breath_transcribe(_command: BreathTranscribeCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `asr-whisper` feature")
}
