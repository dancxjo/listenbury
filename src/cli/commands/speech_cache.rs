use crate::cli::{SpeechCacheCommand, SpeechCachePrewarmCommand};
use anyhow::Result;

#[cfg(feature = "tts-piper")]
use crate::cli::model_paths::resolve_piper_voice;
#[cfg(feature = "tts-piper")]
use crate::cli::piper::{collect_tts_audio, piper_config_for_voice};
#[cfg(feature = "tts-piper")]
use listenbury::mouth::cache::{CachedTextToSpeech, FileSpeechCache};
#[cfg(feature = "tts-piper")]
use listenbury::mouth::planner::{SpeechPlan, SpeechUnit, DEFAULT_SAFE_BACKCHANNELS};
#[cfg(feature = "tts-piper")]
use listenbury::mouth::tts::TextToSpeech;
#[cfg(feature = "tts-piper")]
use listenbury::PiperTextToSpeech;
#[cfg(feature = "tts-piper")]
use std::path::PathBuf;
#[cfg(feature = "tts-piper")]
use std::time::Duration;

#[cfg(feature = "tts-piper")]
pub(crate) fn run_speech_cache(command: SpeechCacheCommand) -> Result<()> {
    match command {
        SpeechCacheCommand::Prewarm(command) => run_speech_cache_prewarm(command),
    }
}

#[cfg(not(feature = "tts-piper"))]
pub(crate) fn run_speech_cache(_command: SpeechCacheCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `tts-piper` feature")
}

#[cfg(feature = "tts-piper")]
#[derive(Debug)]
struct SpeechCachePrewarmOptions {
    piper_bin: PathBuf,
    piper_voice: PathBuf,
    listenbury_home: PathBuf,
}

#[cfg(feature = "tts-piper")]
fn run_speech_cache_prewarm(command: SpeechCachePrewarmCommand) -> Result<()> {
    let options = SpeechCachePrewarmOptions::from_command(command)?;
    let config = piper_config_for_voice(&options.piper_bin, &options.piper_voice)?;
    let mut tts = CachedTextToSpeech::new(
        PiperTextToSpeech::new(config.clone()),
        FileSpeechCache::for_piper(&options.listenbury_home, &config),
    );

    for text in DEFAULT_SAFE_BACKCHANNELS {
        let plan = SpeechPlan::from(SpeechUnit::Backchannel((*text).to_string()));
        tts.enqueue(plan)?;
        let frames = collect_tts_audio(&mut tts, Duration::from_secs(30))?;
        println!("warmed backchannel \"{text}\" ({} frames)", frames.len());
    }

    Ok(())
}

#[cfg(feature = "tts-piper")]
impl SpeechCachePrewarmOptions {
    fn from_command(command: SpeechCachePrewarmCommand) -> Result<Self> {
        let piper_bin = command
            .piper_bin
            .or_else(|| std::env::var_os("LISTENBURY_PIPER_BIN").map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("piper"));
        let piper_voice = resolve_piper_voice(command.piper_voice)?;
        let listenbury_home = command
            .listenbury_home
            .or_else(|| std::env::var_os("LISTENBURY_HOME").map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from(".listenbury"));

        Ok(Self {
            piper_bin,
            piper_voice,
            listenbury_home,
        })
    }
}
