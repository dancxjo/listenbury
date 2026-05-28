use crate::cli::SyntheticCacheCommand;
use crate::cli::SyntheticCachePrewarmCommand;
use anyhow::Result;

use crate::cli::model_paths::resolve_piper_voice;
use crate::cli::piper::{collect_tts_audio, piper_config_for_voice, resolve_piper_bin};
use listenbury::PiperTextToSpeech;
use listenbury::mouth::cache::{CachedTextToSpeech, FileSyntheticCache};
use listenbury::mouth::planner::{MouthSyntheticPlan, SyntheticPlannerConfig, SyntheticUnit};
use listenbury::mouth::tts::TextToSpeech;
use std::path::PathBuf;
use std::time::Duration;

pub(crate) fn run_synthetic_cache(command: SyntheticCacheCommand) -> Result<()> {
    match command {
        SyntheticCacheCommand::Prewarm(command) => run_synthetic_cache_prewarm(command),
    }
}

#[derive(Debug)]
struct SyntheticCachePrewarmOptions {
    piper_bin: PathBuf,
    piper_voice: PathBuf,
    listenbury_home: PathBuf,
}

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
