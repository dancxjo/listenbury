use crate::cli::SyntheticCacheCommand;
#[cfg(feature = "tts-piper")]
use crate::cli::SyntheticCachePrewarmCommand;
use anyhow::Result;

#[cfg(feature = "tts-piper")]
use crate::cli::model_paths::resolve_piper_voice;
#[cfg(feature = "tts-piper")]
use crate::cli::piper::{collect_tts_audio, piper_config_for_voice, resolve_piper_bin};
#[cfg(feature = "tts-piper")]
use listenbury::PiperTextToSpeech;
#[cfg(feature = "tts-piper")]
use listenbury::mouth::cache::{CachedTextToSpeech, FileSyntheticCache};
#[cfg(feature = "tts-piper")]
use listenbury::mouth::planner::{MouthSyntheticPlan, SyntheticPlannerConfig, SyntheticUnit};
#[cfg(feature = "tts-piper")]
use listenbury::mouth::tts::TextToSpeech;
#[cfg(feature = "tts-piper")]
use std::path::PathBuf;
#[cfg(feature = "tts-piper")]
use std::time::Duration;

#[cfg(feature = "tts-piper")]
pub(crate) fn run_synthetic_cache(command: SyntheticCacheCommand) -> Result<()> {
    match command {
        SyntheticCacheCommand::Prewarm(command) => run_synthetic_cache_prewarm(command),
    }
}

#[cfg(not(feature = "tts-piper"))]
pub(crate) fn run_synthetic_cache(_command: SyntheticCacheCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `tts-piper` feature")
}

#[cfg(feature = "tts-piper")]
#[derive(Debug)]
struct SyntheticCachePrewarmOptions {
    piper_bin: PathBuf,
    piper_voice: PathBuf,
    listenbury_home: PathBuf,
}

#[cfg(feature = "tts-piper")]
fn run_synthetic_cache_prewarm(command: SyntheticCachePrewarmCommand) -> Result<()> {
    let options = SyntheticCachePrewarmOptions::from_command(command)?;
    let config = piper_config_for_voice(&options.piper_bin, &options.piper_voice)?;
    let mut tts = CachedTextToSpeech::new(
        PiperTextToSpeech::new(config.clone()),
        FileSyntheticCache::for_piper(&options.listenbury_home, &config),
    );

    for text in SyntheticPlannerConfig::default().safe_backchannels {
        let plan = MouthSyntheticPlan::from(SyntheticUnit::Backchannel(text.clone()));
        tts.enqueue(plan)?;
        let frames = collect_tts_audio(&mut tts, Duration::from_secs(30))?;
        println!("warmed backchannel \"{text}\" ({} frames)", frames.len());
    }

    Ok(())
}

#[cfg(feature = "tts-piper")]
impl SyntheticCachePrewarmOptions {
    fn from_command(command: SyntheticCachePrewarmCommand) -> Result<Self> {
        let piper_bin = resolve_piper_bin(command.piper_bin)?;
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
