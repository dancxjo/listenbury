mod commands;
#[cfg(any(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
mod model_paths;
#[cfg(feature = "tts-piper")]
mod piper;

use anyhow::Result;
use clap::{Args, CommandFactory, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "listenbury", version, about = "Low-latency PETE runtime")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    FakeTurn(TextCommand),
    DemoVad,
    LlamaTurn(LlamaTurnCommand),
    TranscribeSynthetic(TranscribeSyntheticCommand),
    PiperSay(PiperSayCommand),
    RoundTripWav(RoundTripWavCommand),
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },
    SpeechCache {
        #[command(subcommand)]
        command: SpeechCacheCommand,
    },
}

#[derive(Debug, Args)]
pub(crate) struct TextCommand {
    #[arg(required = true, num_args = 1.., trailing_var_arg = true)]
    pub(crate) text: Vec<String>,
}

#[derive(Debug, Args)]
pub(crate) struct LlamaTurnCommand {
    #[arg(long, alias = "model-path")]
    pub(crate) llm_model: Option<PathBuf>,
    #[arg(required = true, num_args = 1.., trailing_var_arg = true)]
    pub(crate) prompt: Vec<String>,
}

#[derive(Debug, Args)]
pub(crate) struct TranscribeSyntheticCommand {
    #[arg(long, alias = "model-path")]
    pub(crate) whisper_model: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct PiperSayCommand {
    #[arg(long)]
    pub(crate) piper_bin: Option<PathBuf>,
    #[arg(long, alias = "model-path")]
    pub(crate) piper_voice: Option<PathBuf>,
    #[arg(required = true, num_args = 1.., trailing_var_arg = true)]
    pub(crate) words: Vec<String>,
}

#[derive(Debug, Args)]
pub(crate) struct RoundTripWavCommand {
    pub(crate) input_wav: PathBuf,
    #[arg(long)]
    pub(crate) whisper_model: Option<PathBuf>,
    #[arg(long)]
    pub(crate) llm_model: Option<PathBuf>,
    #[arg(long)]
    pub(crate) piper_bin: Option<PathBuf>,
    #[arg(long)]
    pub(crate) piper_voice: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
pub(crate) enum ModelsCommand {
    Fetch,
    Status,
    Path,
}

#[derive(Debug, Subcommand)]
pub(crate) enum SpeechCacheCommand {
    Prewarm(SpeechCachePrewarmCommand),
}

#[derive(Debug, Args)]
pub(crate) struct SpeechCachePrewarmCommand {
    #[arg(long)]
    pub(crate) piper_bin: Option<PathBuf>,
    #[arg(long)]
    pub(crate) piper_voice: Option<PathBuf>,
    #[arg(long)]
    pub(crate) listenbury_home: Option<PathBuf>,
}

pub(crate) fn run() -> Result<()> {
    let cli = Cli::parse();
    let Some(command) = cli.command else {
        let mut root = Cli::command();
        root.print_help()?;
        println!();
        return Ok(());
    };

    match command {
        Command::FakeTurn(cmd) => commands::run_fake_turn(cmd.text.join(" ")),
        Command::DemoVad => commands::run_demo_vad(),
        Command::LlamaTurn(cmd) => commands::run_llama_turn(cmd),
        Command::TranscribeSynthetic(cmd) => commands::run_transcribe_synthetic(cmd),
        Command::PiperSay(cmd) => commands::run_piper_say(cmd),
        Command::RoundTripWav(cmd) => commands::run_round_trip_wav(cmd),
        Command::Models { command } => commands::run_models(command),
        Command::SpeechCache { command } => commands::run_speech_cache(command),
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "asr-whisper")]
    use super::*;
    #[cfg(feature = "asr-whisper")]
    use clap::Parser;

    #[cfg(feature = "asr-whisper")]
    #[test]
    fn transcribe_synthetic_accepts_default_model() {
        let cli = Cli::try_parse_from(["listenbury", "transcribe-synthetic"])
            .expect("transcribe-synthetic should not require a model path");

        let Some(Command::TranscribeSynthetic(command)) = cli.command else {
            panic!("expected transcribe-synthetic command");
        };
        assert!(command.whisper_model.is_none());
    }
}
