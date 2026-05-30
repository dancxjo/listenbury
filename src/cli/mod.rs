mod args;
mod commands;
#[cfg(feature = "model-download")]
mod download_progress;
mod live_session;
mod model_paths;
mod piper;

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use live_session::{LiveSession, LiveSessionConfig};

pub(crate) use args::*;
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
    #[command(about = "Run Pete's continuous stream of consciousness")]
    Go(GoCommand),
    #[command(name = "go1", about = "Run the previous continuous runtime")]
    Go1(GoCommand),
    #[command(
        name = "harmony-go",
        alias = "go-harmony",
        about = "Run Pete's continuous runtime through the clean Harmony renderer/parser"
    )]
    HarmonyGo(GoCommand),
    #[command(about = "Render a raw MBROLA .pho file to WAV")]
    MbrolaRender(MbrolaRenderCommand),
    #[command(about = "Sing the ragtime demo phrase")]
    Sing(SingDemoCommand),
    #[command(
        alias = "piper-compare",
        about = "Compare process-backed Piper and Riper synthesis"
    )]
    RiperCompare(RiperCompareCommand),
    #[command(
        alias = "monkey-do",
        about = "Transcribe a WAV and echo the same text back with matched prosody hints"
    )]
    Echo(EchoCommand),
    #[command(
        alias = "live-half-duplex",
        about = "Listen and reply in a live voice loop"
    )]
    Listen(LiveHalfDuplexCommand),
    #[command(alias = "llama-turn", about = "Ask the local language model")]
    Ask(LlamaTurnCommand),
    #[command(about = "Run a raw local language model completion")]
    Complete(LlamaTurnCommand),
    #[command(about = "Run raw local LLM experiments with optional TTS")]
    Draft {
        #[command(subcommand)]
        command: DraftCommand,
    },
    #[command(alias = "round-trip-wav", about = "Reply to a WAV file with speech")]
    Reply(RoundTripWavCommand),
    #[command(about = "Host the WaveDeck session viewer as a local web UI")]
    Web(WebCommand),
    #[command(about = "Fetch and inspect local model assets")]
    Models {
        #[command(subcommand)]
        command: Option<ModelsCommand>,
    },
    #[command(about = "Forge, inspect, and audit neural diphone caches (requires piper-compat)")]
    Diphone {
        #[command(subcommand)]
        command: DiphoneCommand,
    },
    #[command(about = "Calibrate and compare VAD backends on captured or fixture audio")]
    Vad {
        #[command(subcommand)]
        command: VadCommand,
    },
    #[command(about = "Run diagnostic tracing tools")]
    Debug {
        #[command(subcommand)]
        command: DebugCommand,
    },
    #[command(hide = true)]
    Dev {
        #[command(subcommand)]
        command: DevCommand,
    },
}

#[derive(Debug, Subcommand)]
enum DebugCommand {
    #[command(about = "Record an end-to-end loop timing trace")]
    LoopTrace(LoopTraceCommand),
    #[command(about = "Record live mic audio and render a VAD waveform/spectrogram diagnostic")]
    EarScope(EarScopeCommand),
    #[command(about = "Stream one LLM thought into the mouth pipeline")]
    ThinkSay(ThinkSayCommand),
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
    ProsodyPlan(ProsodyPlanCommand),
    MelRoundtrip(MelRoundtripCommand),
    SingDemo(SingDemoCommand),
    RoundTripWav(RoundTripWavCommand),
    LiveHalfDuplex(LiveHalfDuplexCommand),
    DogfoodTwo(DogfoodTwoCommand),
    #[command(
        about = "Print a JSON debug snapshot of the soundscape: sources, hypotheses, voice counts, and overlaps"
    )]
    SoundscapeDebug(SoundscapeDebugCommand),
    SyntheticCache {
        #[command(subcommand)]
        command: SyntheticCacheCommand,
    },
    #[command(about = "Build or inspect the neural diphone cache (requires piper-compat)")]
    DiphoneCache {
        #[command(subcommand)]
        command: DiphoneCacheCommand,
    },
    #[command(
        about = "Inspect a MBROLA voice database: inventory, statistics, and license manifest status"
    )]
    MbrolaInventory(MbrolaInventoryCommand),
    #[command(
        about = "Audit a MBROLA voice database against a .pho plan: check diphone coverage and fallback strategies"
    )]
    MbrolaAudit(MbrolaAuditCommand),
}

#[derive(Debug, Subcommand)]
enum DraftCommand {
    #[command(
        name = "pete-line",
        about = "Run the raw consciousness prompt with open-mouth TTS control"
    )]
    PeteLine(DraftPeteLineCommand),
}

pub(crate) fn run() -> Result<()> {
    let cli = Cli::parse();
    let Some(command) = cli.command else {
        let mut root = Cli::command();
        root.print_help()?;
        println!();
        return Ok(());
    };

    listenbury::set_developer_diagnostics_enabled(matches!(
        &command,
        Command::Debug { .. }
            | Command::Dev { .. }
            | Command::Listen(LiveHalfDuplexCommand { duplex: true, .. })
    ));

    match command {
        Command::Transcribe(cmd) => commands::run_transcribe(cmd),
        Command::Say(cmd) => commands::run_say(cmd),
        Command::Go(cmd) => commands::run_go(cmd),
        Command::Go1(cmd) => commands::run_go1(cmd),
        Command::HarmonyGo(cmd) => commands::run_harmony_go(cmd),
        Command::MbrolaRender(cmd) => commands::run_mbrola_render(cmd),
        Command::Sing(cmd) => commands::run_sing_demo(cmd),
        Command::RiperCompare(cmd) => commands::run_riper_compare(cmd),
        Command::Echo(cmd) => commands::run_echo(cmd),
        Command::Listen(cmd) => run_live_session(LiveSessionConfig::from_listen_command(cmd)),
        Command::Ask(cmd) => commands::run_llama_turn(cmd),
        Command::Complete(mut cmd) => {
            cmd.mode = PromptMode::Raw;
            commands::run_llama_turn(cmd)
        }
        Command::Draft { command } => match command {
            DraftCommand::PeteLine(cmd) => commands::run_draft_pete_line(cmd),
        },
        Command::Reply(cmd) => commands::run_round_trip_wav(cmd),
        Command::Web(cmd) => commands::run_web(cmd),
        Command::Models { command } => commands::run_models(command),
        Command::Diphone { command } => commands::run_diphone(command),
        Command::Vad { command } => commands::run_vad(command),
        Command::Debug { command } => run_debug(command),
        Command::Dev { command } => run_dev(command),
    }
}

fn run_debug(command: DebugCommand) -> Result<()> {
    match command {
        DebugCommand::LoopTrace(cmd) => commands::run_loop_trace(cmd),
        DebugCommand::EarScope(cmd) => commands::run_ear_scope(cmd),
        DebugCommand::ThinkSay(cmd) => commands::run_think_say(cmd),
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
        DevCommand::Continue(cmd) => {
            run_live_session(LiveSessionConfig::from_continue_command(cmd))
        }
        DevCommand::TraceViewerExport(cmd) => commands::run_trace_viewer_export(cmd),
        DevCommand::ProsodyPlan(cmd) => commands::run_prosody_plan(cmd),
        DevCommand::MelRoundtrip(cmd) => commands::run_mel_roundtrip(cmd),
        DevCommand::SingDemo(cmd) => commands::run_sing_demo(cmd),
        DevCommand::RoundTripWav(cmd) => commands::run_round_trip_wav(cmd),
        DevCommand::LiveHalfDuplex(cmd) => {
            run_live_session(LiveSessionConfig::from_listen_command(cmd))
        }
        DevCommand::DogfoodTwo(cmd) => commands::run_dogfood_two(cmd),
        DevCommand::SoundscapeDebug(cmd) => commands::run_soundscape_debug(cmd),
        DevCommand::SyntheticCache { command } => commands::run_synthetic_cache(command),
        DevCommand::DiphoneCache { command } => commands::run_diphone_cache(command),
        DevCommand::MbrolaInventory(cmd) => commands::run_mbrola_inventory(cmd),
        DevCommand::MbrolaAudit(cmd) => commands::run_mbrola_audit(cmd),
    }
}

fn run_live_session(config: LiveSessionConfig) -> Result<()> {
    let mut session = LiveSession::new(config)?;
    let run_result = session.run();
    let shutdown_result = session.shutdown();
    run_result.and(shutdown_result)
}

#[cfg(test)]
mod tests;
