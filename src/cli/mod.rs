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
use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
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
    MicTranscribe(MicTranscribeCommand),
    RecordWav(RecordWavCommand),
    PlayWav(PlayWavCommand),
    LlamaTurn(LlamaTurnCommand),
    Transcribe(TranscribeCommand),
    Say(SayCommand),
    RoundTripWav(RoundTripWavCommand),
    LiveHalfDuplex(LiveHalfDuplexCommand),
    DogfoodTwo(DogfoodTwoCommand),
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
pub(crate) struct MicTranscribeCommand {
    #[arg(long, default_value_t = 30)]
    pub(crate) seconds: u64,
    #[arg(long)]
    pub(crate) until_ctrl_c: bool,
    #[arg(long, alias = "model-path")]
    pub(crate) whisper_model: Option<PathBuf>,
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
pub(crate) struct SayCommand {
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

#[derive(Debug, Args)]
pub(crate) struct DogfoodTwoCommand {
    /// Initial utterance that instance A speaks on turn one.
    #[arg(long, default_value = "Hello.")]
    pub(crate) seed: String,
    /// Total number of TTS→ASR exchanges to run (hard-capped at 32).
    #[arg(long, default_value_t = 6)]
    pub(crate) turns: usize,
    /// Maximum tokens the LLM may generate per turn.
    #[arg(long, default_value_t = 128)]
    pub(crate) max_tokens: u32,
    #[arg(long, alias = "model-path")]
    pub(crate) whisper_model: Option<PathBuf>,
    #[arg(long)]
    pub(crate) llm_model: Option<PathBuf>,
    #[arg(long)]
    pub(crate) piper_bin: Option<PathBuf>,
    /// Piper voice used by instance A.
    #[arg(long)]
    pub(crate) piper_voice_a: Option<PathBuf>,
    /// Piper voice used by instance B (defaults to the same voice as A).
    #[arg(long)]
    pub(crate) piper_voice_b: Option<PathBuf>,
    /// Write a structured JSONL trace to this path.
    #[arg(long)]
    pub(crate) jsonl: Option<PathBuf>,
    /// Save per-turn WAV files to this directory.
    #[arg(long)]
    pub(crate) save_audio_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum ModelProfile {
    #[default]
    Tiny,
}

#[derive(Debug, Args)]
pub(crate) struct LiveHalfDuplexCommand {
    #[arg(long, default_value_t = 30)]
    pub(crate) seconds: u64,
    #[arg(long, value_enum, default_value_t = ModelProfile::Tiny)]
    pub(crate) model_profile: ModelProfile,
    #[arg(long)]
    pub(crate) no_backchannels: bool,
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
        Command::MicTranscribe(cmd) => commands::run_mic_transcribe(cmd),
        Command::RecordWav(cmd) => commands::run_record_wav(cmd),
        Command::PlayWav(cmd) => commands::run_play_wav(cmd),
        Command::LlamaTurn(cmd) => commands::run_llama_turn(cmd),
        Command::Transcribe(cmd) => commands::run_transcribe(cmd),
        Command::Say(cmd) => commands::run_say(cmd),
        Command::RoundTripWav(cmd) => commands::run_round_trip_wav(cmd),
        Command::LiveHalfDuplex(cmd) => commands::run_live_half_duplex(cmd),
        Command::DogfoodTwo(cmd) => commands::run_dogfood_two(cmd),
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
    fn say_accepts_text_and_default_piper_options() {
        let cli = Cli::try_parse_from(["listenbury", "say", "hello", "there"])
            .expect("say should parse text and optional Piper options");

        let Some(Command::Say(command)) = cli.command else {
            panic!("expected say command");
        };
        assert!(command.piper_bin.is_none());
        assert!(command.piper_voice.is_none());
        assert_eq!(command.words, ["hello", "there"]);
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

    #[test]
    fn mic_transcribe_parses_seconds_by_default() {
        let cli = Cli::try_parse_from(["listenbury", "mic-transcribe"])
            .expect("mic-transcribe should parse with defaults");

        let Some(Command::MicTranscribe(command)) = cli.command else {
            panic!("expected mic-transcribe command");
        };
        assert_eq!(command.seconds, 30);
        assert!(!command.until_ctrl_c);
        assert!(command.whisper_model.is_none());
    }

    #[test]
    fn mic_transcribe_parses_until_ctrl_c_and_model() {
        let cli = Cli::try_parse_from([
            "listenbury",
            "mic-transcribe",
            "--until-ctrl-c",
            "--model-path",
            "models/ggml-base.en.bin",
        ])
        .expect("mic-transcribe should parse until-ctrl-c and model path");

        let Some(Command::MicTranscribe(command)) = cli.command else {
            panic!("expected mic-transcribe command");
        };
        assert!(command.until_ctrl_c);
        assert_eq!(
            command.whisper_model,
            Some(PathBuf::from("models/ggml-base.en.bin"))
        );
    }

    #[test]
    fn live_half_duplex_parses_defaults() {
        let cli = Cli::try_parse_from(["listenbury", "live-half-duplex"])
            .expect("live-half-duplex should parse with defaults");

        let Some(Command::LiveHalfDuplex(command)) = cli.command else {
            panic!("expected live-half-duplex command");
        };
        assert_eq!(command.seconds, 30);
        assert_eq!(command.model_profile, ModelProfile::Tiny);
        assert!(!command.no_backchannels);
    }

    #[test]
    fn live_half_duplex_parses_optional_flags() {
        let cli = Cli::try_parse_from([
            "listenbury",
            "live-half-duplex",
            "--seconds",
            "12",
            "--model-profile",
            "tiny",
            "--no-backchannels",
        ])
        .expect("live-half-duplex should parse optional flags");

        let Some(Command::LiveHalfDuplex(command)) = cli.command else {
            panic!("expected live-half-duplex command");
        };
        assert_eq!(command.seconds, 12);
        assert_eq!(command.model_profile, ModelProfile::Tiny);
        assert!(command.no_backchannels);
    }

    #[test]
    fn dogfood_two_parses_defaults() {
        let cli = Cli::try_parse_from(["listenbury", "dogfood-two"])
            .expect("dogfood-two should parse with all defaults");

        let Some(Command::DogfoodTwo(command)) = cli.command else {
            panic!("expected dogfood-two command");
        };
        assert_eq!(command.seed, "Hello.");
        assert_eq!(command.turns, 6);
        assert_eq!(command.max_tokens, 128);
        assert!(command.whisper_model.is_none());
        assert!(command.llm_model.is_none());
        assert!(command.piper_bin.is_none());
        assert!(command.piper_voice_a.is_none());
        assert!(command.piper_voice_b.is_none());
        assert!(command.jsonl.is_none());
        assert!(command.save_audio_dir.is_none());
    }

    #[test]
    fn dogfood_two_parses_all_flags() {
        let cli = Cli::try_parse_from([
            "listenbury",
            "dogfood-two",
            "--seed",
            "Hi there.",
            "--turns",
            "4",
            "--max-tokens",
            "64",
            "--whisper-model",
            "models/ggml-tiny.bin",
            "--llm-model",
            "models/tiny.gguf",
            "--piper-bin",
            "/usr/bin/piper",
            "--piper-voice-a",
            "voices/a.onnx",
            "--piper-voice-b",
            "voices/b.onnx",
            "--jsonl",
            "out/dogfood-two.jsonl",
            "--save-audio-dir",
            "out/dogfood-two-audio",
        ])
        .expect("dogfood-two should parse all flags");

        let Some(Command::DogfoodTwo(command)) = cli.command else {
            panic!("expected dogfood-two command");
        };
        assert_eq!(command.seed, "Hi there.");
        assert_eq!(command.turns, 4);
        assert_eq!(command.max_tokens, 64);
        assert_eq!(
            command.whisper_model,
            Some(PathBuf::from("models/ggml-tiny.bin"))
        );
        assert_eq!(
            command.llm_model,
            Some(PathBuf::from("models/tiny.gguf"))
        );
        assert_eq!(
            command.piper_bin,
            Some(PathBuf::from("/usr/bin/piper"))
        );
        assert_eq!(
            command.piper_voice_a,
            Some(PathBuf::from("voices/a.onnx"))
        );
        assert_eq!(
            command.piper_voice_b,
            Some(PathBuf::from("voices/b.onnx"))
        );
        assert_eq!(
            command.jsonl,
            Some(PathBuf::from("out/dogfood-two.jsonl"))
        );
        assert_eq!(
            command.save_audio_dir,
            Some(PathBuf::from("out/dogfood-two-audio"))
        );
    }
}
