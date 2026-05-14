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
    VadTrace(VadTraceCommand),
    BreathTranscribe(BreathTranscribeCommand),
    RecordWav(RecordWavCommand),
    PlayWav(PlayWavCommand),
    LlamaTurn(LlamaTurnCommand),
    Transcribe(TranscribeCommand),
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
pub(crate) struct RecordWavCommand {
    pub(crate) output_wav: PathBuf,
    #[arg(long, default_value_t = 5)]
    pub(crate) seconds: u64,
}

#[derive(Debug, Args)]
pub(crate) struct PlayWavCommand {
    pub(crate) input_wav: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct VadTraceCommand {
    pub(crate) input_wav: PathBuf,
    #[arg(long)]
    pub(crate) jsonl: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct BreathTranscribeCommand {
    pub(crate) input_wav: PathBuf,
    #[arg(long)]
    pub(crate) whisper_model: Option<PathBuf>,
    #[arg(long, default_value_t = 100)]
    pub(crate) pre_roll_ms: u64,
    #[arg(long, default_value_t = 100)]
    pub(crate) trailing_pad_ms: u64,
    #[arg(long, default_value_t = 150)]
    pub(crate) min_group_ms: u64,
    #[arg(long, default_value_t = 15_000)]
    pub(crate) max_group_ms: u64,
}

#[derive(Debug, Args)]
pub(crate) struct LlamaTurnCommand {
    #[arg(long, alias = "model-path")]
    pub(crate) llm_model: Option<PathBuf>,
    #[arg(required = true, num_args = 1.., trailing_var_arg = true)]
    pub(crate) prompt: Vec<String>,
}

#[derive(Debug, Args)]
pub(crate) struct TranscribeCommand {
    pub(crate) input_wav: PathBuf,
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
        Command::VadTrace(cmd) => commands::run_vad_trace(cmd),
        Command::BreathTranscribe(cmd) => commands::run_breath_transcribe(cmd),
        Command::RecordWav(cmd) => commands::run_record_wav(cmd),
        Command::PlayWav(cmd) => commands::run_play_wav(cmd),
        Command::LlamaTurn(cmd) => commands::run_llama_turn(cmd),
        Command::Transcribe(cmd) => commands::run_transcribe(cmd),
        Command::PiperSay(cmd) => commands::run_piper_say(cmd),
        Command::RoundTripWav(cmd) => commands::run_round_trip_wav(cmd),
        Command::Models { command } => commands::run_models(command),
        Command::SpeechCache { command } => commands::run_speech_cache(command),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn transcribe_accepts_input_and_default_model() {
        let cli = Cli::try_parse_from(["listenbury", "transcribe", "welcome.wav"])
            .expect("transcribe should parse an input path and optional model path");

        let Some(Command::Transcribe(command)) = cli.command else {
            panic!("expected transcribe command");
        };
        assert_eq!(command.input_wav, PathBuf::from("welcome.wav"));
        assert!(command.whisper_model.is_none());
    }

    #[test]
    fn record_wav_parses_seconds_and_output_path() {
        let cli =
            Cli::try_parse_from(["listenbury", "record-wav", "out/mic.wav", "--seconds", "5"])
                .expect("record-wav should parse");

        let Some(Command::RecordWav(command)) = cli.command else {
            panic!("expected record-wav command");
        };
        assert_eq!(command.output_wav, PathBuf::from("out/mic.wav"));
        assert_eq!(command.seconds, 5);
    }

    #[test]
    fn play_wav_parses_input_path() {
        let cli = Cli::try_parse_from(["listenbury", "play-wav", "out/listenbury-round-trip.wav"])
            .expect("play-wav should parse");

        let Some(Command::PlayWav(command)) = cli.command else {
            panic!("expected play-wav command");
        };
        assert_eq!(
            command.input_wav,
            PathBuf::from("out/listenbury-round-trip.wav")
        );
    }

    #[test]
    fn vad_trace_parses_input_and_jsonl() {
        let cli = Cli::try_parse_from([
            "listenbury",
            "vad-trace",
            "samples/silence-16k-mono.wav",
            "--jsonl",
            "out/vad-trace.jsonl",
        ])
        .expect("vad-trace should parse");

        let Some(Command::VadTrace(command)) = cli.command else {
            panic!("expected vad-trace command");
        };
        assert_eq!(
            command.input_wav,
            PathBuf::from("samples/silence-16k-mono.wav")
        );
        assert_eq!(command.jsonl, Some(PathBuf::from("out/vad-trace.jsonl")));
    }

    #[test]
    fn breath_transcribe_parses_config() {
        let cli = Cli::try_parse_from([
            "listenbury",
            "breath-transcribe",
            "samples/hello-16k-mono.wav",
            "--pre-roll-ms",
            "60",
            "--trailing-pad-ms",
            "120",
            "--min-group-ms",
            "80",
            "--max-group-ms",
            "3000",
        ])
        .expect("breath-transcribe should parse");

        let Some(Command::BreathTranscribe(command)) = cli.command else {
            panic!("expected breath-transcribe command");
        };
        assert_eq!(
            command.input_wav,
            PathBuf::from("samples/hello-16k-mono.wav")
        );
        assert_eq!(command.pre_roll_ms, 60);
        assert_eq!(command.trailing_pad_ms, 120);
        assert_eq!(command.min_group_ms, 80);
        assert_eq!(command.max_group_ms, 3000);
    }
}
