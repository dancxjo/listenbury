mod commands;
#[cfg(feature = "model-download")]
mod download_progress;
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
use listenbury::VadBackendKind;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "listenbury", version, about = "Low-latency PETE runtime")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    #[command(about = "Transcribe microphone audio or a WAV file")]
    Transcribe(TranscribeCommand),
    #[command(about = "Speak text aloud")]
    Say(SayCommand),
    #[command(about = "Compare process-backed and native Piper synthesis")]
    PiperCompare(PiperCompareCommand),
    #[command(
        alias = "live-half-duplex",
        about = "Listen and reply in a live voice loop"
    )]
    Listen(LiveHalfDuplexCommand),
    #[command(alias = "llama-turn", about = "Ask the local language model")]
    Ask(LlamaTurnCommand),
    #[command(about = "Run a raw local language model completion")]
    Complete(LlamaTurnCommand),
    #[command(alias = "round-trip-wav", about = "Reply to a WAV file with speech")]
    Reply(RoundTripWavCommand),
    #[command(about = "Host the browser transcript viewer as a local web UI")]
    Web(WebCommand),
    #[command(about = "Fetch and inspect local model assets")]
    Models {
        #[command(subcommand)]
        command: Option<ModelsCommand>,
    },
    #[command(hide = true)]
    Dev {
        #[command(subcommand)]
        command: DevCommand,
    },
}

#[derive(Debug, Subcommand)]
enum DevCommand {
    FakeTurn(TextCommand),
    DemoVad,
    VadTrace(VadTraceCommand),
    BreathTranscribe(BreathTranscribeCommand),
    MicTranscribe(MicTranscribeCommand),
    RecordWav(RecordWavCommand),
    PlayWav(PlayWavCommand),
    LlamaTurn(LlamaTurnCommand),
    Continue(ContinueCommand),
    TraceViewerExport(TraceViewerExportCommand),
    RoundTripWav(RoundTripWavCommand),
    LiveHalfDuplex(LiveHalfDuplexCommand),
    DogfoodTwo(DogfoodTwoCommand),
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
    #[arg(long, value_enum, default_value_t = VadBackendOption::WebRtc)]
    pub(crate) vad: VadBackendOption,
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
    #[arg(long, value_enum, default_value_t = VadBackendOption::WebRtc)]
    pub(crate) vad: VadBackendOption,
}

#[derive(Debug, Args)]
pub(crate) struct LlamaTurnCommand {
    #[arg(long, alias = "model-path")]
    pub(crate) llm_model: Option<PathBuf>,
    /// Number of llama.cpp layers to offload to the GPU. Use 0 for CPU-only LLM inference.
    #[arg(long)]
    pub(crate) llm_gpu_layers: Option<u32>,
    /// Prompt framing to apply before generation.
    #[arg(long, value_enum, default_value_t = PromptMode::Spoken)]
    pub(crate) mode: PromptMode,
    /// Maximum tokens the LLM may generate.
    #[arg(long, default_value_t = 48)]
    pub(crate) max_tokens: u32,
    #[arg(required = true, num_args = 1.., trailing_var_arg = true)]
    pub(crate) prompt: Vec<String>,
}

#[derive(Debug, Args)]
pub(crate) struct ContinueCommand {
    #[arg(long, alias = "model-path")]
    pub(crate) llm_model: Option<PathBuf>,
    /// Number of llama.cpp layers to offload to the GPU. Use 0 for CPU-only LLM inference.
    #[arg(long)]
    pub(crate) llm_gpu_layers: Option<u32>,
    #[arg(long)]
    pub(crate) piper_bin: Option<PathBuf>,
    #[arg(long)]
    pub(crate) piper_voice: Option<PathBuf>,
    #[arg(long)]
    pub(crate) whisper_model: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = VadBackendOption::WebRtc)]
    pub(crate) vad: VadBackendOption,
    /// Prompt framing to apply to the initial prompt only. Stdin appends are inserted raw.
    #[arg(long, value_enum, default_value_t = PromptMode::Raw)]
    pub(crate) mode: PromptMode,
    /// Optional maximum generated-token cap. By default, continue until Ctrl-C or context fills.
    #[arg(long)]
    pub(crate) max_tokens: Option<u32>,
    /// llama.cpp context size for the live session.
    #[arg(long, default_value_t = 8192)]
    pub(crate) context_size: u32,
    /// Number of recent listened/spoken turns to keep verbatim before summarizing older turns.
    #[arg(long, default_value_t = 8)]
    pub(crate) verbatim_turns: usize,
    /// Continuous VAD speech duration before TTS auto-pauses while Pete is speaking.
    #[arg(long, default_value_t = 250)]
    pub(crate) tts_vad_pause_ms: u64,
    /// Time to listen after an auto-pause before clearing or resuming TTS.
    #[arg(long, default_value_t = 700)]
    pub(crate) tts_vad_listen_ms: u64,
    /// Run a synthetic duplex diagnostic scenario instead of the live mic/speaker loop.
    #[arg(long, value_enum)]
    pub(crate) duplex_trace_scenario: Option<DuplexTraceScenarioOption>,
    /// Write structured JSONL trace output (required when --duplex-trace-scenario is set).
    #[arg(long)]
    pub(crate) jsonl: Option<PathBuf>,
    /// Initial prompt text. If omitted, generation starts from Pete's continuous-awareness seed.
    #[arg(num_args = 0.., trailing_var_arg = true)]
    pub(crate) prompt: Vec<String>,
}

#[derive(Debug, Args)]
pub(crate) struct TraceViewerExportCommand {
    /// Runtime trace JSONL generated by `listenbury listen --jsonl ...` or `listenbury dev continue --jsonl ...`.
    pub(crate) input_jsonl: PathBuf,
    /// Destination browser viewer payload JSON.
    pub(crate) output_json: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct WebCommand {
    #[arg(long, default_value = "127.0.0.1")]
    pub(crate) host: String,
    #[arg(long, default_value_t = 8787)]
    pub(crate) port: u16,
    #[arg(long)]
    pub(crate) payload: Option<PathBuf>,
    #[arg(long)]
    pub(crate) trace: Option<PathBuf>,
    #[arg(long)]
    pub(crate) open: bool,
}

#[derive(Debug, Args)]
pub(crate) struct TranscribeCommand {
    pub(crate) input_wav: Option<PathBuf>,
    #[arg(long, alias = "model-path")]
    pub(crate) whisper_model: Option<PathBuf>,
    #[arg(long, default_value_t = 30)]
    pub(crate) seconds: u64,
    #[arg(long)]
    pub(crate) until_ctrl_c: bool,
    #[arg(long, value_enum, default_value_t = VadBackendOption::WebRtc)]
    pub(crate) vad: VadBackendOption,
}

#[derive(Debug, Args)]
pub(crate) struct SayCommand {
    #[arg(long)]
    pub(crate) piper_bin: Option<PathBuf>,
    #[arg(long, alias = "model-path")]
    pub(crate) piper_voice: Option<PathBuf>,
    #[arg(long)]
    pub(crate) output_wav: Option<PathBuf>,
    #[arg(required = true, num_args = 1.., trailing_var_arg = true)]
    pub(crate) words: Vec<String>,
}

#[derive(Debug, Args)]
pub(crate) struct PiperCompareCommand {
    #[arg(long)]
    pub(crate) piper_bin: Option<PathBuf>,
    #[arg(long, alias = "model-path")]
    pub(crate) piper_voice: Option<PathBuf>,
    #[arg(long)]
    pub(crate) native_voice: Option<PathBuf>,
    #[arg(long)]
    pub(crate) native_config: Option<PathBuf>,
    #[arg(long)]
    pub(crate) process_output_wav: Option<PathBuf>,
    #[arg(long)]
    pub(crate) native_output_wav: Option<PathBuf>,
    #[arg(long)]
    pub(crate) phonemes: Option<String>,
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
    /// Number of llama.cpp layers to offload to the GPU. Use 0 for CPU-only LLM inference.
    #[arg(long)]
    pub(crate) llm_gpu_layers: Option<u32>,
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
    /// Number of llama.cpp layers to offload to the GPU. Defaults lower for CUDA live mode so ASR and LLM fit together.
    #[arg(long)]
    pub(crate) llm_gpu_layers: Option<u32>,
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
pub(crate) enum PromptMode {
    Raw,
    #[default]
    Spoken,
    Chat,
    Inner,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum ModelProfile {
    #[default]
    Tiny,
}

#[derive(Debug, Args)]
pub(crate) struct LiveHalfDuplexCommand {
    /// Stop listening after this many seconds. By default, listen until Ctrl-C.
    #[arg(long)]
    pub(crate) seconds: Option<u64>,
    #[arg(long)]
    pub(crate) jsonl: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = ModelProfile::Tiny)]
    pub(crate) model_profile: ModelProfile,
    #[arg(long)]
    pub(crate) no_backchannels: bool,
    #[arg(long)]
    pub(crate) whisper_model: Option<PathBuf>,
    #[arg(long)]
    pub(crate) llm_model: Option<PathBuf>,
    /// Number of llama.cpp layers to offload to the GPU. Defaults lower for CUDA live mode so ASR and LLM fit together.
    #[arg(long)]
    pub(crate) llm_gpu_layers: Option<u32>,
    #[arg(long)]
    pub(crate) piper_bin: Option<PathBuf>,
    #[arg(long)]
    pub(crate) piper_voice: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = VadBackendOption::WebRtc)]
    pub(crate) vad: VadBackendOption,
    /// Start the WaveDeck browser viewer alongside the listen loop (live events streamed via SSE).
    #[arg(long)]
    pub(crate) web: bool,
    /// Host for the embedded web viewer (requires --web).
    #[arg(long, default_value = "127.0.0.1")]
    pub(crate) web_host: String,
    /// Port for the embedded web viewer (requires --web).
    #[arg(long, default_value_t = 8787)]
    pub(crate) web_port: u16,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum VadBackendOption {
    Energy,
    #[default]
    #[value(name = "webrtc")]
    WebRtc,
    Silero,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub(crate) enum DuplexTraceScenarioOption {
    #[value(name = "overlap-yield")]
    OverlapYield,
}

impl VadBackendOption {
    pub(crate) fn as_backend_kind(self) -> VadBackendKind {
        match self {
            Self::Energy => VadBackendKind::Energy,
            Self::WebRtc => VadBackendKind::WebRtc,
            Self::Silero => VadBackendKind::Silero,
        }
    }
}

#[derive(Debug, Subcommand)]
pub(crate) enum ModelsCommand {
    Menu,
    Fetch(ModelsFetchCommand),
    List,
    Use(ModelsUseCommand),
    Status,
    Path,
}

#[derive(Debug, Args)]
pub(crate) struct ModelsFetchCommand {
    /// Fetch one model bundle by name instead of the currently selected models.
    pub(crate) model: Option<String>,
    /// Fetch every registered asset.
    #[arg(long)]
    pub(crate) all: bool,
    /// Maximum number of model assets to download at once.
    #[arg(long, default_value_t = 2)]
    pub(crate) jobs: usize,
}

#[derive(Debug, Args)]
pub(crate) struct ModelsUseCommand {
    #[arg(value_enum)]
    pub(crate) kind: ModelsUseKind,
    pub(crate) model: String,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub(crate) enum ModelsUseKind {
    Llm,
    Voice,
    Whisper,
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

    listenbury::set_developer_diagnostics_enabled(matches!(&command, Command::Dev { .. }));

    match command {
        Command::Transcribe(cmd) => commands::run_transcribe(cmd),
        Command::Say(cmd) => commands::run_say(cmd),
        Command::PiperCompare(cmd) => commands::run_piper_compare(cmd),
        Command::Listen(cmd) => commands::run_live_half_duplex(cmd),
        Command::Ask(cmd) => commands::run_llama_turn(cmd),
        Command::Complete(mut cmd) => {
            cmd.mode = PromptMode::Raw;
            commands::run_llama_turn(cmd)
        }
        Command::Reply(cmd) => commands::run_round_trip_wav(cmd),
        Command::Web(cmd) => commands::run_web(cmd),
        Command::Models { command } => commands::run_models(command),
        Command::Dev { command } => run_dev(command),
    }
}

fn run_dev(command: DevCommand) -> Result<()> {
    match command {
        DevCommand::FakeTurn(cmd) => commands::run_fake_turn(cmd.text.join(" ")),
        DevCommand::DemoVad => commands::run_demo_vad(),
        DevCommand::VadTrace(cmd) => commands::run_vad_trace(cmd),
        DevCommand::BreathTranscribe(cmd) => commands::run_breath_transcribe(cmd),
        DevCommand::MicTranscribe(cmd) => commands::run_mic_transcribe(cmd),
        DevCommand::RecordWav(cmd) => commands::run_record_wav(cmd),
        DevCommand::PlayWav(cmd) => commands::run_play_wav(cmd),
        DevCommand::LlamaTurn(cmd) => commands::run_llama_turn(cmd),
        DevCommand::Continue(cmd) => commands::run_continue(cmd),
        DevCommand::TraceViewerExport(cmd) => commands::run_trace_viewer_export(cmd),
        DevCommand::RoundTripWav(cmd) => commands::run_round_trip_wav(cmd),
        DevCommand::LiveHalfDuplex(cmd) => commands::run_live_half_duplex(cmd),
        DevCommand::DogfoodTwo(cmd) => commands::run_dogfood_two(cmd),
        DevCommand::SpeechCache { command } => commands::run_speech_cache(command),
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
        assert_eq!(command.input_wav, Some(PathBuf::from("welcome.wav")));
        assert!(command.whisper_model.is_none());
        assert_eq!(command.seconds, 30);
        assert!(!command.until_ctrl_c);
        assert_eq!(command.vad, VadBackendOption::WebRtc);
    }

    #[test]
    fn transcribe_without_input_uses_mic_defaults() {
        let cli = Cli::try_parse_from(["listenbury", "transcribe"])
            .expect("transcribe should parse without input for mic capture");

        let Some(Command::Transcribe(command)) = cli.command else {
            panic!("expected transcribe command");
        };
        assert!(command.input_wav.is_none());
        assert!(command.whisper_model.is_none());
        assert_eq!(command.seconds, 30);
        assert!(!command.until_ctrl_c);
        assert_eq!(command.vad, VadBackendOption::WebRtc);
    }

    #[test]
    fn transcribe_without_input_accepts_mic_options() {
        let cli = Cli::try_parse_from([
            "listenbury",
            "transcribe",
            "--seconds",
            "5",
            "--until-ctrl-c",
            "--model-path",
            "models/ggml-base.en.bin",
            "--vad",
            "webrtc",
        ])
        .expect("transcribe should parse mic capture options");

        let Some(Command::Transcribe(command)) = cli.command else {
            panic!("expected transcribe command");
        };
        assert!(command.input_wav.is_none());
        assert_eq!(command.seconds, 5);
        assert!(command.until_ctrl_c);
        assert_eq!(
            command.whisper_model,
            Some(PathBuf::from("models/ggml-base.en.bin"))
        );
        assert_eq!(command.vad, VadBackendOption::WebRtc);
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
        assert!(command.output_wav.is_none());
        assert_eq!(command.words, ["hello", "there"]);
    }

    #[test]
    fn say_accepts_output_wav_override() {
        let cli = Cli::try_parse_from([
            "listenbury",
            "say",
            "--output-wav",
            "out/test.wav",
            "hello",
            "there",
        ])
        .expect("say should parse an optional output WAV path");

        let Some(Command::Say(command)) = cli.command else {
            panic!("expected say command");
        };
        assert_eq!(command.output_wav, Some(PathBuf::from("out/test.wav")));
        assert_eq!(command.words, ["hello", "there"]);
    }

    #[test]
    fn piper_compare_parses_text_and_optional_outputs() {
        let cli = Cli::try_parse_from([
            "listenbury",
            "piper-compare",
            "--process-output-wav",
            "out/process.wav",
            "--native-output-wav",
            "out/native.wav",
            "I",
            "see.",
        ])
        .expect("piper-compare should parse text and optional output paths");

        let Some(Command::PiperCompare(command)) = cli.command else {
            panic!("expected piper-compare command");
        };
        assert_eq!(
            command.process_output_wav,
            Some(PathBuf::from("out/process.wav"))
        );
        assert_eq!(
            command.native_output_wav,
            Some(PathBuf::from("out/native.wav"))
        );
        assert_eq!(command.words, ["I", "see."]);
    }

    #[test]
    fn record_wav_parses_seconds_and_output_path() {
        let cli = Cli::try_parse_from([
            "listenbury",
            "dev",
            "record-wav",
            "out/mic.wav",
            "--seconds",
            "5",
        ])
        .expect("record-wav should parse");

        let Some(Command::Dev {
            command: DevCommand::RecordWav(command),
        }) = cli.command
        else {
            panic!("expected record-wav command");
        };
        assert_eq!(command.output_wav, PathBuf::from("out/mic.wav"));
        assert_eq!(command.seconds, 5);
    }

    #[test]
    fn play_wav_parses_input_path() {
        let cli = Cli::try_parse_from([
            "listenbury",
            "dev",
            "play-wav",
            "out/listenbury-round-trip.wav",
        ])
        .expect("play-wav should parse");

        let Some(Command::Dev {
            command: DevCommand::PlayWav(command),
        }) = cli.command
        else {
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
            "dev",
            "vad-trace",
            "samples/silence-16k-mono.wav",
            "--jsonl",
            "out/vad-trace.jsonl",
        ])
        .expect("vad-trace should parse");

        let Some(Command::Dev {
            command: DevCommand::VadTrace(command),
        }) = cli.command
        else {
            panic!("expected vad-trace command");
        };
        assert_eq!(
            command.input_wav,
            PathBuf::from("samples/silence-16k-mono.wav")
        );
        assert_eq!(command.jsonl, Some(PathBuf::from("out/vad-trace.jsonl")));
        assert_eq!(command.vad, VadBackendOption::WebRtc);
    }

    #[test]
    fn breath_transcribe_parses_config() {
        let cli = Cli::try_parse_from([
            "listenbury",
            "dev",
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

        let Some(Command::Dev {
            command: DevCommand::BreathTranscribe(command),
        }) = cli.command
        else {
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
        let cli = Cli::try_parse_from(["listenbury", "dev", "mic-transcribe"])
            .expect("mic-transcribe should parse with defaults");

        let Some(Command::Dev {
            command: DevCommand::MicTranscribe(command),
        }) = cli.command
        else {
            panic!("expected mic-transcribe command");
        };
        assert_eq!(command.seconds, 30);
        assert!(!command.until_ctrl_c);
        assert!(command.whisper_model.is_none());
        assert_eq!(command.vad, VadBackendOption::WebRtc);
    }

    #[test]
    fn mic_transcribe_parses_until_ctrl_c_and_model() {
        let cli = Cli::try_parse_from([
            "listenbury",
            "dev",
            "mic-transcribe",
            "--until-ctrl-c",
            "--model-path",
            "models/ggml-base.en.bin",
        ])
        .expect("mic-transcribe should parse until-ctrl-c and model path");

        let Some(Command::Dev {
            command: DevCommand::MicTranscribe(command),
        }) = cli.command
        else {
            panic!("expected mic-transcribe command");
        };
        assert!(command.until_ctrl_c);
        assert_eq!(
            command.whisper_model,
            Some(PathBuf::from("models/ggml-base.en.bin"))
        );
        assert_eq!(command.vad, VadBackendOption::WebRtc);
    }

    #[test]
    fn live_half_duplex_parses_defaults() {
        let cli = Cli::try_parse_from(["listenbury", "listen"])
            .expect("listen should parse with defaults");

        let Some(Command::Listen(command)) = cli.command else {
            panic!("expected listen command");
        };
        assert_eq!(command.seconds, None);
        assert!(command.jsonl.is_none());
        assert_eq!(command.model_profile, ModelProfile::Tiny);
        assert!(!command.no_backchannels);
        assert_eq!(command.vad, VadBackendOption::WebRtc);
        assert!(!command.web);
        assert_eq!(command.web_host, "127.0.0.1");
        assert_eq!(command.web_port, 8787);
    }

    #[test]
    fn listen_parses_web_flag() {
        let cli = Cli::try_parse_from([
            "listenbury",
            "listen",
            "--web",
            "--web-host",
            "0.0.0.0",
            "--web-port",
            "9000",
        ])
        .expect("listen should parse --web flags");

        let Some(Command::Listen(command)) = cli.command else {
            panic!("expected listen command");
        };
        assert!(command.web);
        assert_eq!(command.web_host, "0.0.0.0");
        assert_eq!(command.web_port, 9000);
    }

    #[test]
    fn live_half_duplex_parses_optional_flags() {
        let cli = Cli::try_parse_from([
            "listenbury",
            "listen",
            "--seconds",
            "12",
            "--jsonl",
            "out/live-trace.jsonl",
            "--model-profile",
            "tiny",
            "--no-backchannels",
        ])
        .expect("listen should parse optional flags");

        let Some(Command::Listen(command)) = cli.command else {
            panic!("expected listen command");
        };
        assert_eq!(command.seconds, Some(12));
        assert_eq!(command.jsonl, Some(PathBuf::from("out/live-trace.jsonl")));
        assert_eq!(command.model_profile, ModelProfile::Tiny);
        assert!(command.no_backchannels);
        assert_eq!(command.vad, VadBackendOption::WebRtc);
    }

    #[test]
    fn vad_options_parse_for_vad_trace_mic_and_live_half_duplex() {
        let trace = Cli::try_parse_from([
            "listenbury",
            "dev",
            "vad-trace",
            "samples/hello-16k-mono.wav",
            "--vad",
            "webrtc",
        ])
        .expect("vad-trace should parse --vad webrtc");
        let Some(Command::Dev {
            command: DevCommand::VadTrace(trace_command),
        }) = trace.command
        else {
            panic!("expected vad-trace command");
        };
        assert_eq!(trace_command.vad, VadBackendOption::WebRtc);

        let mic = Cli::try_parse_from(["listenbury", "dev", "mic-transcribe", "--vad", "silero"])
            .expect("mic-transcribe should parse --vad silero");
        let Some(Command::Dev {
            command: DevCommand::MicTranscribe(mic_command),
        }) = mic.command
        else {
            panic!("expected mic-transcribe command");
        };
        assert_eq!(mic_command.vad, VadBackendOption::Silero);

        let live = Cli::try_parse_from(["listenbury", "listen", "--vad", "energy"])
            .expect("listen should parse --vad energy");
        let Some(Command::Listen(live_command)) = live.command else {
            panic!("expected listen command");
        };
        assert_eq!(live_command.vad, VadBackendOption::Energy);
    }

    #[test]
    fn dogfood_two_parses_defaults() {
        let cli = Cli::try_parse_from(["listenbury", "dev", "dogfood-two"])
            .expect("dogfood-two should parse with all defaults");

        let Some(Command::Dev {
            command: DevCommand::DogfoodTwo(command),
        }) = cli.command
        else {
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
    fn dev_continue_parses_defaults_and_prompt() {
        let cli = Cli::try_parse_from(["listenbury", "dev", "continue", "seed", "prompt"])
            .expect("dev continue should parse with a seed prompt");

        let Some(Command::Dev {
            command: DevCommand::Continue(command),
        }) = cli.command
        else {
            panic!("expected continue command");
        };
        assert!(command.llm_model.is_none());
        assert!(command.piper_bin.is_none());
        assert!(command.piper_voice.is_none());
        assert!(command.whisper_model.is_none());
        assert_eq!(command.vad, VadBackendOption::WebRtc);
        assert_eq!(command.mode, PromptMode::Raw);
        assert_eq!(command.max_tokens, None);
        assert_eq!(command.context_size, 8192);
        assert_eq!(command.verbatim_turns, 8);
        assert_eq!(command.tts_vad_pause_ms, 250);
        assert_eq!(command.tts_vad_listen_ms, 700);
        assert!(command.duplex_trace_scenario.is_none());
        assert!(command.jsonl.is_none());
        assert_eq!(command.prompt, ["seed", "prompt"]);
    }

    #[test]
    fn dev_continue_accepts_optional_token_cap() {
        let cli = Cli::try_parse_from(["listenbury", "dev", "continue", "--max-tokens", "64"])
            .expect("dev continue should parse an optional token cap");

        let Some(Command::Dev {
            command: DevCommand::Continue(command),
        }) = cli.command
        else {
            panic!("expected continue command");
        };
        assert_eq!(command.max_tokens, Some(64));
    }

    #[test]
    fn dev_continue_accepts_mouth_overrides() {
        let cli = Cli::try_parse_from([
            "listenbury",
            "dev",
            "continue",
            "--piper-bin",
            "/usr/bin/piper",
            "--piper-voice",
            "voices/pete.onnx",
        ])
        .expect("dev continue should parse optional mouth overrides");

        let Some(Command::Dev {
            command: DevCommand::Continue(command),
        }) = cli.command
        else {
            panic!("expected continue command");
        };
        assert_eq!(command.piper_bin, Some(PathBuf::from("/usr/bin/piper")));
        assert_eq!(command.piper_voice, Some(PathBuf::from("voices/pete.onnx")));
    }

    #[test]
    fn dev_continue_accepts_ear_overrides() {
        let cli = Cli::try_parse_from([
            "listenbury",
            "dev",
            "continue",
            "--whisper-model",
            "models/ggml-base.en.bin",
            "--vad",
            "energy",
            "--tts-vad-pause-ms",
            "180",
            "--tts-vad-listen-ms",
            "900",
        ])
        .expect("dev continue should parse optional ear overrides");

        let Some(Command::Dev {
            command: DevCommand::Continue(command),
        }) = cli.command
        else {
            panic!("expected continue command");
        };
        assert_eq!(
            command.whisper_model,
            Some(PathBuf::from("models/ggml-base.en.bin"))
        );
        assert_eq!(command.vad, VadBackendOption::Energy);
        assert_eq!(command.tts_vad_pause_ms, 180);
        assert_eq!(command.tts_vad_listen_ms, 900);
    }

    #[test]
    fn dev_continue_accepts_duplex_trace_scenario_and_jsonl() {
        let cli = Cli::try_parse_from([
            "listenbury",
            "dev",
            "continue",
            "--duplex-trace-scenario",
            "overlap-yield",
            "--jsonl",
            "out/duplex-trace.jsonl",
        ])
        .expect("dev continue should parse duplex trace scenario options");

        let Some(Command::Dev {
            command: DevCommand::Continue(command),
        }) = cli.command
        else {
            panic!("expected continue command");
        };
        assert_eq!(
            command.duplex_trace_scenario,
            Some(DuplexTraceScenarioOption::OverlapYield)
        );
        assert_eq!(command.jsonl, Some(PathBuf::from("out/duplex-trace.jsonl")));
    }

    #[test]
    fn dev_trace_viewer_export_parses_input_and_output_paths() {
        let cli = Cli::try_parse_from([
            "listenbury",
            "dev",
            "trace-viewer-export",
            "out/live-trace.jsonl",
            "examples/browser-transcript-player/live-trace.sample.viewer.json",
        ])
        .expect("trace-viewer-export should parse");

        let Some(Command::Dev {
            command: DevCommand::TraceViewerExport(command),
        }) = cli.command
        else {
            panic!("expected trace-viewer-export command");
        };
        assert_eq!(command.input_jsonl, PathBuf::from("out/live-trace.jsonl"));
        assert_eq!(
            command.output_json,
            PathBuf::from("examples/browser-transcript-player/live-trace.sample.viewer.json")
        );
    }

    #[test]
    fn web_command_parses_defaults() {
        let cli = Cli::try_parse_from(["listenbury", "web"]).expect("web should parse defaults");

        let Some(Command::Web(command)) = cli.command else {
            panic!("expected web command");
        };
        assert_eq!(command.host, "127.0.0.1");
        assert_eq!(command.port, 8787);
        assert!(command.payload.is_none());
        assert!(command.trace.is_none());
        assert!(!command.open);
    }

    #[test]
    fn web_command_parses_all_flags() {
        let cli = Cli::try_parse_from([
            "listenbury",
            "web",
            "--host",
            "0.0.0.0",
            "--port",
            "9090",
            "--payload",
            "out/viewer-payload.json",
            "--trace",
            "out/live.jsonl",
            "--open",
        ])
        .expect("web should parse optional flags");

        let Some(Command::Web(command)) = cli.command else {
            panic!("expected web command");
        };
        assert_eq!(command.host, "0.0.0.0");
        assert_eq!(command.port, 9090);
        assert_eq!(
            command.payload,
            Some(PathBuf::from("out/viewer-payload.json"))
        );
        assert_eq!(command.trace, Some(PathBuf::from("out/live.jsonl")));
        assert!(command.open);
    }

    #[test]
    fn dogfood_two_parses_all_flags() {
        let cli = Cli::try_parse_from([
            "listenbury",
            "dev",
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

        let Some(Command::Dev {
            command: DevCommand::DogfoodTwo(command),
        }) = cli.command
        else {
            panic!("expected dogfood-two command");
        };
        assert_eq!(command.seed, "Hi there.");
        assert_eq!(command.turns, 4);
        assert_eq!(command.max_tokens, 64);
        assert_eq!(
            command.whisper_model,
            Some(PathBuf::from("models/ggml-tiny.bin"))
        );
        assert_eq!(command.llm_model, Some(PathBuf::from("models/tiny.gguf")));
        assert_eq!(command.piper_bin, Some(PathBuf::from("/usr/bin/piper")));
        assert_eq!(command.piper_voice_a, Some(PathBuf::from("voices/a.onnx")));
        assert_eq!(command.piper_voice_b, Some(PathBuf::from("voices/b.onnx")));
        assert_eq!(command.jsonl, Some(PathBuf::from("out/dogfood-two.jsonl")));
        assert_eq!(
            command.save_audio_dir,
            Some(PathBuf::from("out/dogfood-two-audio"))
        );
    }

    #[test]
    fn top_level_help_keeps_diagnostics_hidden() {
        let mut command = Cli::command();
        let help = command.render_help().to_string();

        for visible in [
            "transcribe",
            "say",
            "listen",
            "ask",
            "reply",
            "web",
            "models",
        ] {
            assert!(
                help.contains(visible),
                "missing {visible} from help:\n{help}"
            );
        }

        for hidden in [
            "fake-turn",
            "demo-vad",
            "vad-trace",
            "breath-transcribe",
            "mic-transcribe",
            "record-wav",
            "play-wav",
            "llama-turn",
            "round-trip-wav",
            "live-half-duplex",
            "dogfood-two",
            "speech-cache",
            "dev",
        ] {
            assert!(!help.contains(hidden), "leaked {hidden} into help:\n{help}");
        }
    }

    #[test]
    fn models_fetch_parses_jobs() {
        let cli = Cli::try_parse_from(["listenbury", "models", "fetch", "--jobs", "4"])
            .expect("models fetch should parse jobs");

        let Some(Command::Models {
            command: Some(ModelsCommand::Fetch(command)),
        }) = cli.command
        else {
            panic!("expected models fetch command");
        };
        assert_eq!(command.jobs, 4);
    }

    #[test]
    fn models_accepts_bare_menu() {
        let cli = Cli::try_parse_from(["listenbury", "models"])
            .expect("bare models should parse as the interactive menu");

        let Some(Command::Models { command: None }) = cli.command else {
            panic!("expected bare models command");
        };
    }

    #[test]
    fn models_menu_parses_explicitly() {
        let cli = Cli::try_parse_from(["listenbury", "models", "menu"])
            .expect("models menu should parse");

        let Some(Command::Models {
            command: Some(ModelsCommand::Menu),
        }) = cli.command
        else {
            panic!("expected models menu command");
        };
    }
}
