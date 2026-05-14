use anyhow::{Context, Result};
use clap::{Args, CommandFactory, Parser, Subcommand};
#[cfg(feature = "model-download")]
use indicatif::{ProgressBar, ProgressStyle};
use listenbury::audio::frame::AudioFrame;
use listenbury::hearing::breath::BreathGroupSegmenter;
use listenbury::hearing::vad::{EnergyVad, VoiceActivityDetector};
use listenbury::mind::llm::{GenerationRequest, LlmEngine, LlmEvent, MockLlmEngine};
#[cfg(feature = "model-download")]
use listenbury::models::{
    default_asset_paths, default_assets_status, fetch_default_assets,
    manifest::DEFAULT_MODELS,
    paths::{asset_path, resolve_listenbury_home},
    FetchOutcome,
};
#[cfg(feature = "tts-piper")]
use listenbury::mouth::cache::{CachedTextToSpeech, FileSpeechCache};
use listenbury::mouth::planner::SpeechPlanner;
#[cfg(feature = "tts-piper")]
use listenbury::mouth::planner::{DEFAULT_SAFE_BACKCHANNELS, SpeechPlan, SpeechUnit};
#[cfg(all(feature = "asr-whisper", feature = "llm-llama-cpp", feature = "tts-piper"))]
use listenbury::mouth::planner::ExpressiveUnit;
#[cfg(feature = "tts-piper")]
use listenbury::mouth::tts::TextToSpeech;
#[cfg(feature = "asr-whisper")]
use listenbury::speech::recognizer::SpeechRecognizer;
use listenbury::time::ExactTimestamp;
#[cfg(feature = "llm-llama-cpp")]
use listenbury::{LlamaCppConfig, LlamaCppEngine};
#[cfg(feature = "tts-piper")]
use listenbury::{PiperConfig, PiperTextToSpeech};
#[cfg(feature = "model-download")]
use owo_colors::OwoColorize;
#[cfg(feature = "llm-llama-cpp")]
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
#[cfg(feature = "tts-piper")]
use std::time::{Duration, Instant};

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
struct TextCommand {
    #[arg(required = true, num_args = 1.., trailing_var_arg = true)]
    text: Vec<String>,
}

#[derive(Debug, Args)]
struct LlamaTurnCommand {
    #[arg(long, alias = "model-path")]
    llm_model: Option<PathBuf>,
    #[arg(required = true, num_args = 1.., trailing_var_arg = true)]
    prompt: Vec<String>,
}

#[derive(Debug, Args)]
struct TranscribeSyntheticCommand {
    model_path: String,
}

#[derive(Debug, Args)]
struct PiperSayCommand {
    #[arg(long)]
    piper_bin: Option<PathBuf>,
    #[arg(long, alias = "model-path")]
    piper_voice: Option<PathBuf>,
    #[arg(required = true, num_args = 1.., trailing_var_arg = true)]
    words: Vec<String>,
}

#[derive(Debug, Args)]
struct RoundTripWavCommand {
    input_wav: PathBuf,
    #[arg(long)]
    whisper_model: Option<PathBuf>,
    #[arg(long)]
    llm_model: Option<PathBuf>,
    #[arg(long)]
    piper_bin: Option<PathBuf>,
    #[arg(long)]
    piper_voice: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum ModelsCommand {
    Fetch,
    Status,
    Path,
}

#[derive(Debug, Subcommand)]
enum SpeechCacheCommand {
    Prewarm(SpeechCachePrewarmCommand),
}

#[derive(Debug, Args)]
struct SpeechCachePrewarmCommand {
    #[arg(long)]
    piper_bin: Option<PathBuf>,
    #[arg(long)]
    piper_voice: Option<PathBuf>,
    #[arg(long)]
    listenbury_home: Option<PathBuf>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    let Some(command) = cli.command else {
        let mut root = Cli::command();
        root.print_help()?;
        println!();
        return Ok(());
    };

    match command {
        Command::FakeTurn(cmd) => run_fake_turn(cmd.text.join(" ")),
        Command::DemoVad => run_demo_vad(),
        Command::LlamaTurn(cmd) => run_llama_turn(cmd),
        Command::TranscribeSynthetic(cmd) => run_transcribe_synthetic(cmd.model_path),
        Command::PiperSay(cmd) => run_piper_say(cmd),
        Command::RoundTripWav(cmd) => run_round_trip_wav(
            cmd.input_wav,
            RoundTripWavOptions {
                whisper_model: cmd.whisper_model,
                llm_model: cmd.llm_model,
                piper_bin: cmd.piper_bin,
                piper_voice: cmd.piper_voice,
            },
        ),
        Command::Models { command } => run_models(command),
        Command::SpeechCache { command } => run_speech_cache(command),
    }
}

#[derive(Debug, Default)]
#[cfg_attr(
    not(all(
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )),
    allow(dead_code)
)]
struct RoundTripWavOptions {
    whisper_model: Option<PathBuf>,
    llm_model: Option<PathBuf>,
    piper_bin: Option<PathBuf>,
    piper_voice: Option<PathBuf>,
}

#[cfg(feature = "model-download")]
fn run_models(command: ModelsCommand) -> Result<()> {
    match command {
        ModelsCommand::Path => {
            let home = resolve_listenbury_home()?;
            println!("{}={}", "listenbury_home".cyan(), home.display());
            println!("{}={}", "models_dir".cyan(), home.join("models").display());
            println!("{}={}", "bin_dir".cyan(), home.join("bin").display());
            for (asset, path) in default_asset_paths()? {
                println!("{}={}", asset.id.cyan(), path.display());
            }
            Ok(())
        }
        ModelsCommand::Status => {
            for status in default_assets_status()? {
                let state = if status.present {
                    "present".green().to_string()
                } else {
                    "missing".red().to_string()
                };
                println!(
                    "{} {} {}",
                    status.asset_id.bold(),
                    state,
                    status.path.display()
                );
            }
            Ok(())
        }
        ModelsCommand::Fetch => {
            let spinner = ProgressBar::new_spinner();
            let style = ProgressStyle::with_template("{spinner:.cyan} {msg}")
                .context("failed to create spinner style")?;
            spinner.set_style(style);
            spinner.enable_steady_tick(std::time::Duration::from_millis(100));
            spinner.set_message("Fetching default model assets...");

            let results = fetch_default_assets()?;
            spinner.finish_and_clear();
            let mut had_failure = false;
            for result in results {
                match result.outcome {
                    FetchOutcome::SkippedExisting => {
                        println!(
                            "{} {} {}",
                            result.asset_id.bold(),
                            "skipped".yellow(),
                            result.path.display()
                        );
                    }
                    FetchOutcome::Downloaded => {
                        println!(
                            "{} {} {}",
                            result.asset_id.bold(),
                            "downloaded".green(),
                            result.path.display()
                        );
                    }
                    FetchOutcome::Failed => {
                        had_failure = true;
                        println!(
                            "{} {} {} ({})",
                            result.asset_id.bold(),
                            "failed".red(),
                            result.path.display(),
                            result.error.as_deref().unwrap_or("unknown error")
                        );
                    }
                }
            }
            if had_failure {
                anyhow::bail!("one or more model assets failed to fetch");
            }
            Ok(())
        }
    }
}

#[cfg(not(feature = "model-download"))]
fn run_models(_command: ModelsCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `model-download` feature")
}

#[cfg(feature = "tts-piper")]
fn run_speech_cache(command: SpeechCacheCommand) -> Result<()> {
    match command {
        SpeechCacheCommand::Prewarm(command) => run_speech_cache_prewarm(command),
    }
}

#[cfg(not(feature = "tts-piper"))]
fn run_speech_cache(_command: SpeechCacheCommand) -> Result<()> {
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

#[cfg(feature = "tts-piper")]
fn run_fake_turn(user_text: String) -> Result<()> {
    let mut llm = MockLlmEngine::with_response(vec!["I ".into(), "heard ".into(), "you.".into()]);
    let request = GenerationRequest {
        prompt: format!("User said: {user_text}"),
        max_tokens: None,
    };

    let id = llm.start(request).context("failed to start generation")?;
    let mut planner = SpeechPlanner::default();

    loop {
        let events = llm.poll(id)?;
        if events.is_empty() {
            continue;
        }

        for event in &events {
            if let LlmEvent::Token { text } = event {
                print!("{text}");
            }
        }

        for unit in planner.ingest(&events) {
            println!();
            println!("ExpressiveUnit: {unit:?}");
        }

        if events.iter().any(|event| {
            matches!(
                event,
                LlmEvent::Completed | LlmEvent::Cancelled | LlmEvent::Error { .. }
            )
        }) {
            break;
        }
    }

    Ok(())
}

#[cfg(feature = "llm-llama-cpp")]
fn run_llama_turn(command: LlamaTurnCommand) -> Result<()> {
    let args = LlamaTurnArgs::from_command(command)?;
    let model_path = resolve_llm_model(args.llm_model)?;
    let config = LlamaCppConfig {
        model_path,
        ..Default::default()
    };
    let mut llm = LlamaCppEngine::new(config).context("failed to initialize llama.cpp engine")?;
    let id = llm
        .start(GenerationRequest {
            prompt: args.prompt,
            max_tokens: None,
        })
        .context("failed to start llama.cpp generation")?;

    loop {
        let events = llm.poll(id)?;
        if events.is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(5));
            continue;
        }

        for event in &events {
            match event {
                LlmEvent::Token { text } => {
                    print!("{text}");
                    std::io::stdout().flush()?;
                }
                LlmEvent::Error { message } => {
                    anyhow::bail!("llama.cpp generation failed: {message}");
                }
                LlmEvent::Completed | LlmEvent::Cancelled => {}
            }
        }

        if events.iter().any(|event| {
            matches!(
                event,
                LlmEvent::Completed | LlmEvent::Cancelled | LlmEvent::Error { .. }
            )
        }) {
            println!();
            break;
        }
    }

    Ok(())
}

#[cfg(not(feature = "llm-llama-cpp"))]
fn run_llama_turn(_command: LlamaTurnCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `llm-llama-cpp` feature")
}

#[cfg(feature = "llm-llama-cpp")]
#[derive(Debug)]
struct LlamaTurnArgs {
    llm_model: Option<PathBuf>,
    prompt: String,
}

#[cfg(feature = "llm-llama-cpp")]
impl LlamaTurnArgs {
    fn from_command(command: LlamaTurnCommand) -> Result<Self> {
        let mut prompt = command.prompt;
        let mut llm_model = command.llm_model;

        if llm_model.is_none() && prompt.first().is_some_and(|word| word.ends_with(".gguf")) {
            llm_model = Some(PathBuf::from(prompt.remove(0)));
        }

        anyhow::ensure!(
            !prompt.is_empty(),
            "missing prompt; try `llama-turn \"hello\"`"
        );

        Ok(Self {
            llm_model,
            prompt: prompt.join(" "),
        })
    }
}

#[cfg(feature = "llm-llama-cpp")]
fn resolve_llm_model(explicit: Option<PathBuf>) -> Result<PathBuf> {
    resolve_model_path(
        explicit,
        "LISTENBURY_LLM_MODEL",
        "llama.cpp model",
        "--llm-model",
        Some("tinyllama-q4-k-m"),
        |path| path.extension().is_some_and(|ext| ext == "gguf"),
    )
}

fn run_demo_vad() -> Result<()> {
    let mut vad = EnergyVad::new(0.02);
    let mut segmenter = BreathGroupSegmenter::default();

    let amplitudes = [
        0.0_f32, 0.0, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    for amp in amplitudes {
        let frame = AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![amp; 160],
        };
        let vad_result = vad.process_frame(&frame)?;
        for event in segmenter.process(vad_result) {
            println!("{event:?}");
        }
    }

    Ok(())
}

#[cfg(feature = "asr-whisper")]
fn run_transcribe_synthetic(model_path: String) -> Result<()> {
    let mut recognizer = listenbury::WhisperSpeechRecognizer::new(&model_path)
        .with_context(|| format!("failed to load Whisper model at {model_path}"))?;

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
fn run_transcribe_synthetic(_model_path: String) -> Result<()> {
    anyhow::bail!("listenbury was built without the `asr-whisper` feature")
}

#[cfg(feature = "tts-piper")]
fn run_piper_say(command: PiperSayCommand) -> Result<()> {
    let piper_args = PiperSayArgs::from_command(command)?;
    let piper_bin = resolve_piper_bin(piper_args.piper_bin);
    let piper_voice = resolve_piper_voice(piper_args.piper_voice)?;
    let mut tts = PiperTextToSpeech::new(piper_config_for_voice(piper_bin, piper_voice)?);
    tts.enqueue(SpeechPlan::from(SpeechUnit::FullTurn(piper_args.text)))?;
    let frames = collect_tts_audio(&mut tts, Duration::from_secs(30))?;

    std::fs::create_dir_all("out").context("failed to create out directory")?;
    let output_path = std::path::Path::new("out/listenbury-piper-test.wav");
    write_wav(output_path, &frames)?;

    let sample_count: usize = frames.iter().map(|frame| frame.samples.len()).sum();
    println!(
        "Wrote {} frames / {} samples to {}",
        frames.len(),
        sample_count,
        output_path.display()
    );

    Ok(())
}

#[cfg(not(feature = "tts-piper"))]
fn run_piper_say(_command: PiperSayCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `tts-piper` feature")
}

#[cfg(feature = "tts-piper")]
#[derive(Debug)]
struct PiperSayArgs {
    piper_bin: Option<PathBuf>,
    piper_voice: Option<PathBuf>,
    text: String,
}

#[cfg(feature = "tts-piper")]
impl PiperSayArgs {
    fn from_command(command: PiperSayCommand) -> Result<Self> {
        let mut words = command.words;
        let mut piper_bin = command.piper_bin;
        let mut piper_voice = command.piper_voice;

        if piper_bin.is_none() && words.first().is_some_and(|word| looks_like_piper_bin(word)) {
            piper_bin = Some(PathBuf::from(words.remove(0)));
        }

        if piper_voice.is_none() && words.first().is_some_and(|word| word.ends_with(".onnx")) {
            piper_voice = Some(PathBuf::from(words.remove(0)));
        }

        anyhow::ensure!(
            !words.is_empty(),
            "missing text to speak; try `piper-say hello`"
        );

        Ok(Self {
            piper_bin,
            piper_voice,
            text: words.join(" "),
        })
    }
}

#[cfg(feature = "tts-piper")]
fn looks_like_piper_bin(word: &str) -> bool {
    let path = Path::new(word);
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.contains("piper"))
}

#[cfg(feature = "tts-piper")]
fn resolve_piper_bin(explicit: Option<PathBuf>) -> PathBuf {
    explicit
        .or_else(|| std::env::var_os("LISTENBURY_PIPER_BIN").map(PathBuf::from))
        .or_else(|| find_piper_executable("piper"))
        .or_else(|| find_piper_executable("piper-tts.piper-cli"))
        .unwrap_or_else(|| PathBuf::from("piper"))
}

#[cfg(feature = "tts-piper")]
fn find_piper_executable(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path)
            .map(|dir| dir.join(name))
            .find(|candidate| candidate.is_file())
    })
}

#[cfg(feature = "tts-piper")]
fn resolve_piper_voice(explicit: Option<PathBuf>) -> Result<PathBuf> {
    resolve_model_path(
        explicit,
        "LISTENBURY_PIPER_VOICE",
        "Piper voice",
        "--piper-voice",
        Some("piper-lessac-medium"),
        |path| path.extension().is_some_and(|ext| ext == "onnx"),
    )
}

#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn run_round_trip_wav(input_wav: PathBuf, options: RoundTripWavOptions) -> Result<()> {
    let paths = RoundTripModelPaths::discover(options)?;
    let frames = read_wav_as_audio_frames(&input_wav, 1_600)?;

    let mut recognizer = listenbury::WhisperSpeechRecognizer::new(&paths.whisper_model)
        .with_context(|| {
            format!(
                "failed to load Whisper model at {}",
                paths.whisper_model.display()
            )
        })?;
    for frame in &frames {
        recognizer.push_frame(frame)?;
    }

    let transcript = recognizer
        .poll_chunks()?
        .into_iter()
        .map(|chunk| chunk.text)
        .collect::<Vec<_>>()
        .join(" ");
    println!("Heard: {transcript}");

    let mut llm = LlamaCppEngine::new(LlamaCppConfig {
        model_path: paths.llm_model.clone(),
        ..Default::default()
    })
    .with_context(|| {
        format!(
            "failed to initialize llama.cpp with {}",
            paths.llm_model.display()
        )
    })?;

    let generation_id = llm
        .start(GenerationRequest {
            prompt: build_round_trip_prompt(&transcript),
            max_tokens: Some(96),
        })
        .context("failed to start llama.cpp generation")?;

    let mut planner = SpeechPlanner::default();
    let mut last_plan = None;
    loop {
        let events = llm.poll(generation_id)?;
        if events.is_empty() {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        }

        for event in &events {
            match event {
                LlmEvent::Token { text } => {
                    print!("{text}");
                    std::io::stdout().flush()?;
                }
                LlmEvent::Error { message } => {
                    anyhow::bail!("llama.cpp generation failed: {message}");
                }
                LlmEvent::Completed | LlmEvent::Cancelled => {}
            }
        }

        let emitted = planner.ingest(&events);
        if let Some(plan) = emitted.into_iter().rev().find_map(|u| match u {
            ExpressiveUnit::Speech(plan) => Some(plan),
            ExpressiveUnit::Face(_) => None,
        }) {
            last_plan = Some(plan);
        }

        if events.iter().any(|event| {
            matches!(
                event,
                LlmEvent::Completed | LlmEvent::Cancelled | LlmEvent::Error { .. }
            )
        }) {
            println!();
            break;
        }
    }

    let plan = last_plan.unwrap_or_else(|| {
        SpeechPlan::from(SpeechUnit::FullTurn(
            "I heard you, but I lost my words.".to_string(),
        ))
    });

    let mut tts =
        PiperTextToSpeech::new(piper_config_for_voice(paths.piper_bin, paths.piper_voice)?);
    tts.enqueue(plan)?;
    let audio = collect_tts_audio(&mut tts, Duration::from_secs(30))?;

    std::fs::create_dir_all("out").context("failed to create out directory")?;
    let output_path = Path::new("out/listenbury-round-trip.wav");
    write_wav(output_path, &audio)?;
    println!("Wrote {}", output_path.display());

    Ok(())
}

#[cfg(not(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
)))]
fn run_round_trip_wav(_input_wav: PathBuf, _options: RoundTripWavOptions) -> Result<()> {
    anyhow::bail!(
        "listenbury was built without the `asr-whisper`, `llm-llama-cpp`, and `tts-piper` features"
    )
}

#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
#[derive(Debug, Clone)]
struct RoundTripModelPaths {
    whisper_model: PathBuf,
    llm_model: PathBuf,
    piper_bin: PathBuf,
    piper_voice: PathBuf,
}

#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
impl RoundTripModelPaths {
    fn discover(options: RoundTripWavOptions) -> Result<Self> {
        Ok(Self {
            whisper_model: resolve_model_path(
                options.whisper_model,
                "LISTENBURY_WHISPER_MODEL",
                "Whisper model",
                "--whisper-model",
                Some("whisper-tiny-en"),
                |path| {
                    path.extension().is_some_and(|ext| ext == "bin")
                        && path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .is_some_and(|name| name.contains("ggml"))
                },
            )?,
            llm_model: resolve_llm_model(options.llm_model)?,
            piper_bin: resolve_piper_bin(options.piper_bin),
            piper_voice: resolve_model_path(
                options.piper_voice,
                "LISTENBURY_PIPER_VOICE",
                "Piper voice",
                "--piper-voice",
                Some("piper-lessac-medium"),
                |path| path.extension().is_some_and(|ext| ext == "onnx"),
            )?,
        })
    }
}

#[cfg(any(feature = "llm-llama-cpp", feature = "tts-piper"))]
fn resolve_model_path(
    explicit: Option<PathBuf>,
    env_var: &str,
    label: &str,
    flag: &str,
    default_asset_id: Option<&str>,
    matches: impl Fn(&Path) -> bool,
) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path);
    }

    if let Some(path) = std::env::var_os(env_var) {
        return Ok(PathBuf::from(path));
    }

    #[cfg(feature = "model-download")]
    if let Some(asset_id) = default_asset_id {
        let path = default_asset_path(asset_id)?;
        if is_non_empty_file(&path) {
            return Ok(path);
        }
    }

    if let Some(path) = discover_model_file(&matches)? {
        return Ok(path);
    }

    let fetch_hint = if default_asset_id.is_some() && cfg!(feature = "model-download") {
        ", or run `cargo run -- models fetch`"
    } else {
        ""
    };
    anyhow::bail!("could not discover {label}; set {env_var}, pass {flag}{fetch_hint}")
}

#[cfg(feature = "model-download")]
fn default_asset_path(asset_id: &str) -> Result<PathBuf> {
    let Some(asset) = DEFAULT_MODELS.iter().find(|asset| asset.id == asset_id) else {
        anyhow::bail!("default model asset `{asset_id}` is not registered");
    };
    let home = resolve_listenbury_home()?;
    Ok(asset_path(&home, asset))
}

#[cfg(feature = "model-download")]
fn is_non_empty_file(path: &Path) -> bool {
    path.metadata().map(|meta| meta.len() > 0).unwrap_or(false)
}

fn discover_model_file(matches: &impl Fn(&Path) -> bool) -> Result<Option<PathBuf>> {
    let models_dir = Path::new("models");
    if !models_dir.exists() {
        return Ok(None);
    }

    let mut stack = vec![models_dir.to_path_buf()];
    let mut found = Vec::new();

    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)
            .with_context(|| format!("failed to read model directory {}", dir.display()))?
        {
            let entry = entry
                .with_context(|| format!("failed to inspect model directory {}", dir.display()))?;
            let path = entry.path();
            let file_type = entry
                .file_type()
                .with_context(|| format!("failed to inspect {}", path.display()))?;
            if file_type.is_dir() {
                stack.push(path);
            } else if file_type.is_file() && matches(&path) {
                found.push(path);
            }
        }
    }

    found.sort();
    Ok(found.into_iter().next())
}

#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn build_round_trip_prompt(transcript: &str) -> String {
    format!(
        "<|system|>\nYou are Pete, speaking aloud through a TTS system.\nWrite in short, complete spoken sentences.\nDo not rely on long subordinate clauses.\nPrefer natural sentence boundaries.\nEach sentence should be speakable on its own.</s>\n<|user|>\n{transcript}</s>\n<|assistant|>\n"
    )
}

#[cfg(feature = "tts-piper")]
fn write_wav(path: &std::path::Path, frames: &[AudioFrame]) -> Result<()> {
    let Some(first_frame) = frames.first() else {
        anyhow::bail!("cannot write WAV without audio frames");
    };

    let spec = hound::WavSpec {
        channels: first_frame.channels,
        sample_rate: first_frame.sample_rate_hz,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("failed to create WAV at {}", path.display()))?;

    for frame in frames {
        anyhow::ensure!(
            frame.channels == first_frame.channels,
            "Piper frame channel count changed from {} to {}",
            first_frame.channels,
            frame.channels
        );
        anyhow::ensure!(
            frame.sample_rate_hz == first_frame.sample_rate_hz,
            "Piper frame sample rate changed from {} to {}",
            first_frame.sample_rate_hz,
            frame.sample_rate_hz
        );

        for sample in &frame.samples {
            writer.write_sample(f32_to_i16(*sample))?;
        }
    }

    writer.finalize()?;
    Ok(())
}

#[cfg(feature = "tts-piper")]
fn f32_to_i16(sample: f32) -> i16 {
    let sample = sample.clamp(-1.0, 1.0);
    (sample * i16::MAX as f32) as i16
}

#[cfg(feature = "tts-piper")]
fn read_piper_sample_rate_hz(path: &std::path::Path) -> Result<Option<u32>> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read Piper config at {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse Piper config at {}", path.display()))?;

    Ok(value
        .get("audio")
        .and_then(|audio| audio.get("sample_rate"))
        .and_then(|sample_rate| sample_rate.as_u64())
        .and_then(|sample_rate| u32::try_from(sample_rate).ok()))
}

#[cfg(feature = "tts-piper")]
fn piper_config_for_voice(
    piper_bin: impl Into<PathBuf>,
    model_path: impl Into<PathBuf>,
) -> Result<PiperConfig> {
    let piper_bin = piper_bin.into();
    let model_path = prepare_piper_model_path(&piper_bin, model_path.into())?;
    let inferred_config_path = model_path.with_extension("onnx.json");
    let mut config = PiperConfig::new(piper_bin, model_path);
    if inferred_config_path.exists() {
        if let Some(sample_rate_hz) = read_piper_sample_rate_hz(&inferred_config_path)? {
            config.sample_rate_hz = sample_rate_hz;
        }
        config.config_path = Some(inferred_config_path);
    }
    Ok(config)
}

#[cfg(feature = "tts-piper")]
fn prepare_piper_model_path(piper_bin: &Path, model_path: PathBuf) -> Result<PathBuf> {
    if !uses_snap_piper(piper_bin) || !has_hidden_component(&model_path) {
        return Ok(model_path);
    }

    let destination_dir = Path::new("out/piper-models");
    std::fs::create_dir_all(destination_dir)
        .context("failed to create Snap-readable Piper model directory")?;

    let model_filename = model_path
        .file_name()
        .context("Piper model path has no filename")?;
    let copied_model_path = destination_dir.join(model_filename);
    copy_if_needed(&model_path, &copied_model_path)?;

    let config_path = model_path.with_extension("onnx.json");
    if config_path.exists() {
        let config_filename = config_path
            .file_name()
            .context("Piper config path has no filename")?;
        copy_if_needed(&config_path, &destination_dir.join(config_filename))?;
    }

    Ok(copied_model_path)
}

#[cfg(feature = "tts-piper")]
fn uses_snap_piper(piper_bin: &Path) -> bool {
    piper_bin
        .to_str()
        .is_some_and(|path| path.starts_with("/snap/bin/") || path.contains("piper-tts.piper-cli"))
}

#[cfg(feature = "tts-piper")]
fn has_hidden_component(path: &Path) -> bool {
    path.components().any(|component| {
        component
            .as_os_str()
            .to_str()
            .is_some_and(|part| part.starts_with('.') && part != "." && part != "..")
    })
}

#[cfg(feature = "tts-piper")]
fn copy_if_needed(source: &Path, destination: &Path) -> Result<()> {
    let should_copy = match (source.metadata(), destination.metadata()) {
        (Ok(source_meta), Ok(destination_meta)) => source_meta.len() != destination_meta.len(),
        (Ok(_), Err(_)) => true,
        (Err(error), _) => {
            return Err(error).with_context(|| format!("failed to inspect {}", source.display()));
        }
    };

    if should_copy {
        std::fs::copy(source, destination).with_context(|| {
            format!(
                "failed to copy Piper asset from {} to {}",
                source.display(),
                destination.display()
            )
        })?;
    }

    Ok(())
}

#[cfg(feature = "tts-piper")]
fn collect_tts_audio(tts: &mut impl TextToSpeech, timeout: Duration) -> Result<Vec<AudioFrame>> {
    let deadline = Instant::now() + timeout;
    let quiet_after_audio = Duration::from_millis(100);
    let mut frames = Vec::new();
    let mut last_audio_at = None;

    while Instant::now() < deadline {
        let new_frames = tts.poll_audio()?;
        if new_frames.is_empty() {
            if let Some(last_audio_at) = last_audio_at {
                if Instant::now().duration_since(last_audio_at) >= quiet_after_audio {
                    break;
                }
            }
        } else {
            frames.extend(new_frames);
            last_audio_at = Some(Instant::now());
        }

        std::thread::sleep(Duration::from_millis(10));
    }

    if frames.is_empty() {
        anyhow::bail!("Piper produced no audio frames before timeout");
    }

    Ok(frames)
}

#[cfg(feature = "tts-piper")]
#[cfg_attr(
    not(all(
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )),
    allow(dead_code)
)]
fn read_wav_as_audio_frames(path: &Path, frame_samples: usize) -> Result<Vec<AudioFrame>> {
    anyhow::ensure!(frame_samples > 0, "frame_samples must be greater than zero");

    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open WAV at {}", path.display()))?;
    let spec = reader.spec();

    anyhow::ensure!(
        spec.channels == 1,
        "expected mono WAV input at {}; got {} channels",
        path.display(),
        spec.channels
    );
    anyhow::ensure!(
        spec.sample_rate == 16_000,
        "expected 16 kHz WAV input at {}; got {} Hz",
        path.display(),
        spec.sample_rate
    );
    anyhow::ensure!(
        spec.sample_format == hound::SampleFormat::Int,
        "expected integer PCM WAV input at {}; floating-point WAV is not supported yet",
        path.display()
    );

    let samples = match spec.bits_per_sample {
        1..=8 => reader
            .samples::<i8>()
            .map(|sample| sample.map(|sample| sample as f32 / 128.0))
            .collect::<std::result::Result<Vec<_>, _>>()
            .with_context(|| format!("failed to read PCM samples from {}", path.display()))?,
        9..=16 => reader
            .samples::<i16>()
            .map(|sample| sample.map(|sample| sample as f32 / i16::MAX as f32))
            .collect::<std::result::Result<Vec<_>, _>>()
            .with_context(|| format!("failed to read PCM samples from {}", path.display()))?,
        17..=32 => {
            let scale = if spec.bits_per_sample == 32 {
                i32::MAX as f32
            } else {
                ((1_i64 << (spec.bits_per_sample - 1)) - 1) as f32
            };
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|sample| sample as f32 / scale))
                .collect::<std::result::Result<Vec<_>, _>>()
                .with_context(|| format!("failed to read PCM samples from {}", path.display()))?
        }
        bits => anyhow::bail!(
            "unsupported PCM bit depth {bits} for WAV input at {}",
            path.display()
        ),
    };

    Ok(samples
        .chunks(frame_samples)
        .map(|chunk| AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: spec.sample_rate,
            channels: spec.channels,
            samples: chunk.to_vec(),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "tts-piper")]
    use super::*;
    #[cfg(feature = "tts-piper")]
    use std::fs;

    #[cfg(feature = "tts-piper")]
    fn unique_test_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("listenbury-{name}-{}.wav", std::process::id()))
    }

    #[cfg(feature = "tts-piper")]
    const FLOAT_TOLERANCE: f32 = 0.0001;

    #[cfg(feature = "llm-llama-cpp")]
    #[test]
    fn llama_turn_args_treats_single_argument_as_prompt() {
        let args = LlamaTurnArgs::from_command(LlamaTurnCommand {
            llm_model: None,
            prompt: vec!["hello".to_string()],
        })
        .expect("single argument should be prompt");

        assert!(args.llm_model.is_none());
        assert_eq!(args.prompt, "hello");
    }

    #[cfg(feature = "llm-llama-cpp")]
    #[test]
    fn llama_turn_args_accepts_legacy_model_position() {
        let args = LlamaTurnArgs::from_command(LlamaTurnCommand {
            llm_model: None,
            prompt: vec!["tinyllama.gguf".to_string(), "hello".to_string()],
        })
        .expect("legacy model path should be accepted");

        assert_eq!(args.llm_model, Some(PathBuf::from("tinyllama.gguf")));
        assert_eq!(args.prompt, "hello");
    }

    #[cfg(feature = "tts-piper")]
    #[test]
    fn piper_say_args_treats_single_word_as_text() {
        let args = PiperSayArgs::from_command(PiperSayCommand {
            piper_bin: None,
            piper_voice: None,
            words: vec!["hello".to_string()],
        })
        .expect("single word should be text");

        assert!(args.piper_bin.is_none());
        assert!(args.piper_voice.is_none());
        assert_eq!(args.text, "hello");
    }

    #[cfg(feature = "tts-piper")]
    #[test]
    fn piper_say_args_accepts_legacy_piper_bin_position() {
        let args = PiperSayArgs::from_command(PiperSayCommand {
            piper_bin: None,
            piper_voice: None,
            words: vec![
                "/snap/bin/piper-tts.piper-cli".to_string(),
                "hello".to_string(),
            ],
        })
        .expect("legacy Piper executable should be accepted");

        assert_eq!(
            args.piper_bin,
            Some(PathBuf::from("/snap/bin/piper-tts.piper-cli"))
        );
        assert!(args.piper_voice.is_none());
        assert_eq!(args.text, "hello");
    }

    #[cfg(feature = "tts-piper")]
    #[test]
    fn piper_say_args_accepts_legacy_voice_position() {
        let args = PiperSayArgs::from_command(PiperSayCommand {
            piper_bin: None,
            piper_voice: None,
            words: vec![
                "/snap/bin/piper-tts.piper-cli".to_string(),
                "voice.onnx".to_string(),
                "hello".to_string(),
            ],
        })
        .expect("legacy Piper executable and voice should be accepted");

        assert_eq!(
            args.piper_bin,
            Some(PathBuf::from("/snap/bin/piper-tts.piper-cli"))
        );
        assert_eq!(args.piper_voice, Some(PathBuf::from("voice.onnx")));
        assert_eq!(args.text, "hello");
    }

    #[cfg(feature = "tts-piper")]
    #[test]
    fn read_wav_as_audio_frames_chunks_pcm_samples() {
        let path = unique_test_path("mono-16k");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).expect("WAV should be created");
        writer
            .write_sample(i16::MIN)
            .expect("sample write should succeed");
        writer
            .write_sample(0_i16)
            .expect("sample write should succeed");
        writer
            .write_sample(i16::MAX)
            .expect("sample write should succeed");
        writer.finalize().expect("WAV should finalize");

        let frames = read_wav_as_audio_frames(&path, 2).expect("WAV should be read");
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].sample_rate_hz, 16_000);
        assert_eq!(frames[0].channels, 1);
        assert_eq!(frames[0].samples.len(), 2);
        assert_eq!(frames[1].samples.len(), 1);
        assert!(frames[0].samples[0] <= -1.0 + FLOAT_TOLERANCE);
        assert!(frames[1].samples[0] >= 1.0 - FLOAT_TOLERANCE);

        fs::remove_file(path).expect("temporary WAV should be removed");
    }

    #[cfg(feature = "tts-piper")]
    #[test]
    fn read_wav_as_audio_frames_rejects_wrong_sample_rate() {
        let path = unique_test_path("wrong-rate");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 8_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).expect("WAV should be created");
        writer
            .write_sample(0_i16)
            .expect("sample write should succeed");
        writer.finalize().expect("WAV should finalize");

        let error = read_wav_as_audio_frames(&path, 1600).expect_err("sample rate should fail");
        assert!(error.to_string().contains("expected 16 kHz WAV input"));

        fs::remove_file(path).expect("temporary WAV should be removed");
    }
}
