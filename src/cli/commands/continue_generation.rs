use crate::cli::ContinueCommand;
use anyhow::Result;

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use crate::cli::commands::cpal_diag::play_audio_frames;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use crate::cli::commands::llama::build_prompt;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use crate::cli::commands::mic_transcribe::transcribe_group;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use crate::cli::model_paths::{
    llm_runtime_placement, resolve_llm_model, resolve_piper_voice, resolve_whisper_model,
};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use crate::cli::piper::{piper_config_for_voice, resolve_piper_bin};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use anyhow::Context;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use cpal::{FromSample, Sample, SizedSample};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::event::HearingEvent;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::hearing::breath::{BreathGroupId, BreathGroupSegmenter};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::hearing::vad::{VoiceActivityDetector, create_vad_backend};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::mind::llm::{GenerationId, GenerationRequest, LlmEngine, LlmEvent};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::mouth::planner::{SpeechPlan, SpeechUnit, strip_emoji};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::mouth::tts::TextToSpeech;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::{AudioFrame, ExactTimestamp, LlamaCppConfig, LlamaCppEngine, PiperTextToSpeech};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::{VadBackendKind, WhisperSpeechRecognizer};
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use seams::sentence_detector::dialog_detector::SentenceDetectorDialog;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use std::collections::{HashMap, VecDeque};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use std::io::{BufRead, Write};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use std::path::PathBuf;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use std::sync::Arc;
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use std::sync::OnceLock;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use std::thread::JoinHandle;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const DEFAULT_CONTINUE_PROMPT: &str = "You are Pete Listenbury, an experiment in artificial awareness. Please continuously generate thoughts as new input arrives from the outside world. Try to understand what's going on around you and make new friends.";
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const LIVE_EVENT_INSTRUCTIONS: &str = "Live events may appear in the transcript while you are generating.\nTreat them as observations from outside.\nDo not copy live event delimiters or runtime event text.\nDo not write system, assistant, analysis, channel, or template tokens.\nContinue naturally as Pete.\nYou may emit speech at any time by surrounding it with inline speech tags:\n<sp>words to say aloud :)</sp>\nEmoji inside speech tags are instructions to your countenance, not words to speak.";
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const INLINE_SPEECH_START_MARKER: &str = "<sp>";
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const INLINE_SPEECH_END_MARKER: &str = "</sp>";
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
const CALLBACK_SAMPLE_CAPACITY: usize = 16_384;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
const WEBRTC_VAD_SAMPLE_RATE_HZ: u32 = 16_000;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
const MONO_CHANNELS: u16 = 1;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
const TIME_EVENT_INTERVAL: Duration = Duration::from_secs(10);
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "llm-llama-cpp-cuda",
    feature = "tts-piper"
))]
const DEFAULT_CONTINUE_LLAMA_GPU_LAYERS: Option<u32> = Some(999);
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    not(feature = "llm-llama-cpp-cuda"),
    feature = "tts-piper"
))]
const DEFAULT_CONTINUE_LLAMA_GPU_LAYERS: Option<u32> = None;

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
pub(crate) fn run_continue(command: ContinueCommand) -> Result<()> {
    let max_tokens = command
        .max_tokens
        .map(|max_tokens| usize::try_from(max_tokens).context("max_tokens does not fit in usize"))
        .transpose()?;
    if let Some(max_tokens) = max_tokens {
        anyhow::ensure!(max_tokens > 0, "max_tokens must be greater than zero");
    }
    anyhow::ensure!(
        command.context_size > 0,
        "context_size must be greater than zero"
    );

    let model_path = resolve_llm_model(command.llm_model)?;
    let llm_placement = llm_runtime_placement(
        &model_path,
        command.llm_gpu_layers,
        DEFAULT_CONTINUE_LLAMA_GPU_LAYERS,
    )?;
    let config = LlamaCppConfig {
        model_path,
        gpu_layers: llm_placement.gpu_layers,
        cpu_only: llm_placement.cpu_only,
        context_size: command.context_size,
        ..Default::default()
    };

    let initial_prompt = build_initial_prompt(&command.prompt);
    let (prompt, stop) = build_prompt(command.mode, &initial_prompt);
    let mut llm = LlamaCppEngine::new(config).context("failed to initialize llama.cpp engine")?;
    let id = llm
        .start(GenerationRequest {
            prompt,
            max_tokens,
            stop,
        })
        .context("failed to start continued llama.cpp generation")?;
    let piper_bin = resolve_piper_bin(command.piper_bin)?;
    let piper_voice = resolve_piper_voice(command.piper_voice)?;
    let whisper_model = resolve_whisper_model(command.whisper_model)?;
    let vad_backend = command.vad.as_backend_kind();
    let capture_enabled = Arc::new(AtomicBool::new(true));
    let mut mouth = ContinueMouth::new(
        PiperTextToSpeech::new(piper_config_for_voice(piper_bin, piper_voice)?),
        Arc::clone(&capture_enabled),
    );
    let (_ear, ear_rx) = ContinueEar::start(ContinueEarConfig {
        whisper_model,
        vad_backend,
        capture_enabled: Arc::clone(&capture_enabled),
    })?;

    let interrupted = Arc::new(AtomicBool::new(false));
    ctrlc::set_handler({
        let interrupted = Arc::clone(&interrupted);
        move || {
            interrupted.store(true, Ordering::Relaxed);
        }
    })
    .context("failed to install Ctrl-C handler")?;

    let (stdin_tx, stdin_rx) =
        crossbeam_channel::unbounded::<std::result::Result<String, String>>();
    std::thread::Builder::new()
        .name("listenbury-dev-continue-stdin".to_string())
        .spawn(move || {
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if stdin_tx.send(Ok(line)).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = stdin_tx.send(Err(error.to_string()));
                        break;
                    }
                }
            }
        })
        .context("failed to spawn stdin reader")?;

    eprintln!(
        "listenbury dev continue: streaming one generation; stdin lines, mic transcripts, and 10s time events append to the live context. Ctrl-C cancels."
    );

    let mut cancelled = false;
    let mut next_time_event_at = Instant::now() + TIME_EVENT_INTERVAL;
    let mut speech_events = SpeechEventDetector::default();
    loop {
        if interrupted.load(Ordering::Relaxed) && !cancelled {
            llm.cancel(id)?;
            cancelled = true;
        }

        append_pending_live_events(&mut llm, id, &stdin_rx, &ear_rx, &mut next_time_event_at)?;

        if mouth.is_throttling_llm() {
            if mouth.drain_ready_audio("listenbury dev continue speech")? {
                if !mouth.is_throttling_llm() {
                    llm.set_paused(id, false)
                        .context("failed to resume continued llama.cpp generation")?;
                }
            } else {
                std::thread::sleep(Duration::from_millis(5));
            }
            continue;
        }

        let events = llm.poll(id)?;
        if events.is_empty() {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        }

        let generation_terminal = events.iter().any(|event| {
            matches!(
                event,
                LlmEvent::Completed | LlmEvent::Cancelled | LlmEvent::Error { .. }
            )
        });

        for event in &events {
            match event {
                LlmEvent::Token { text } => {
                    print!("{text}");
                    std::io::stdout().flush()?;
                    for speech_event in speech_events.ingest(text) {
                        if !generation_terminal {
                            llm.append_prompt(id, wrap_runtime_event(&speech_event.to_message()))
                                .context(
                                    "failed to append runtime speech event to live generation",
                                )?;
                        }
                        if mouth.enqueue_runtime_event(&speech_event)? {
                            if !generation_terminal {
                                llm.set_paused(id, true)
                                    .context("failed to throttle continued llama.cpp generation")?;
                            }
                        }
                    }
                }
                LlmEvent::Error { message } => {
                    anyhow::bail!("continued llama.cpp generation failed: {message}");
                }
                LlmEvent::Completed | LlmEvent::Cancelled => {}
            }
        }

        if generation_terminal {
            println!();
            while mouth.is_throttling_llm() {
                if !mouth.drain_ready_audio("listenbury dev continue speech")? {
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
            break;
        }
    }

    Ok(())
}

#[cfg(not(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
)))]
pub(crate) fn run_continue(_command: ContinueCommand) -> Result<()> {
    anyhow::bail!(
        "listenbury dev continue requires the `audio-cpal`, `asr-whisper`, `llm-llama-cpp`, and `tts-piper` features"
    )
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn wrap_live_input(text: &str) -> String {
    format!(
        "\n\n--- LIVE EVENT: user ---\n{}\n--- END LIVE EVENT ---\n\n",
        text.trim()
    )
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn wrap_time_event(message: &str) -> String {
    format!("\n\n--- LIVE EVENT: clock ---\n{message}\n--- END LIVE EVENT ---\n\n")
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn wrap_heard_input(text: &str) -> String {
    format!(
        "\n\n--- LIVE EVENT: ear ---\nHeard: {}\n--- END LIVE EVENT ---\n\n",
        text.trim()
    )
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn wrap_runtime_event(message: &str) -> String {
    format!("\n\n--- LIVE EVENT: runtime ---\n{message}\n--- END LIVE EVENT ---\n\n")
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn current_time_message() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::ZERO);
    format!(
        "The current Unix time is {}.{:03} seconds.",
        now.as_secs(),
        now.subsec_millis()
    )
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn append_pending_live_events(
    llm: &mut LlamaCppEngine,
    id: GenerationId,
    stdin_rx: &crossbeam_channel::Receiver<std::result::Result<String, String>>,
    ear_rx: &crossbeam_channel::Receiver<ContinueEarEvent>,
    next_time_event_at: &mut Instant,
) -> Result<()> {
    let now = Instant::now();
    if now >= *next_time_event_at {
        llm.append_prompt(id, wrap_time_event(&current_time_message()))
            .context("failed to append time event to live generation")?;
        *next_time_event_at = now + TIME_EVENT_INTERVAL;
    }

    for stdin_event in stdin_rx.try_iter() {
        match stdin_event {
            Ok(text) => llm
                .append_prompt(id, wrap_live_input(&text))
                .context("failed to append stdin text to live generation")?,
            Err(message) => anyhow::bail!("failed to read stdin: {message}"),
        }
    }

    for ear_event in ear_rx.try_iter() {
        match ear_event {
            ContinueEarEvent::Transcript(text) => {
                eprintln!("[dev continue] heard: {text}");
                llm.append_prompt(id, wrap_heard_input(&text))
                    .context("failed to append heard speech to live generation")?;
            }
            ContinueEarEvent::Error(message) => {
                anyhow::bail!("dev continue ear failed: {message}");
            }
        }
    }

    Ok(())
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
struct ContinueEarConfig {
    whisper_model: PathBuf,
    vad_backend: VadBackendKind,
    capture_enabled: Arc<AtomicBool>,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
enum ContinueEarEvent {
    Transcript(String),
    Error(String),
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
struct ContinueEar {
    stop: Arc<AtomicBool>,
    _stream: cpal::Stream,
    processor: Option<JoinHandle<()>>,
    asr: Option<JoinHandle<()>>,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
impl ContinueEar {
    fn start(
        config: ContinueEarConfig,
    ) -> Result<(Self, crossbeam_channel::Receiver<ContinueEarEvent>)> {
        let mut recognizer =
            WhisperSpeechRecognizer::new(&config.whisper_model).with_context(|| {
                format!(
                    "failed to load Whisper model at {}",
                    config.whisper_model.display()
                )
            })?;
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow::anyhow!("no default input device available"))?;
        let device_name = device
            .name()
            .unwrap_or_else(|_| "<unknown input device>".to_string());
        let supported_config = device
            .default_input_config()
            .with_context(|| format!("failed to read default input config for {device_name}"))?;
        let stream_config = supported_config.config();
        let input_sample_rate_hz = stream_config.sample_rate.0;
        let input_channels = stream_config.channels;
        anyhow::ensure!(
            input_channels > 0,
            "default input device reported zero channels"
        );

        let stop = Arc::new(AtomicBool::new(false));
        let (sample_tx, sample_rx) = crossbeam_channel::bounded::<f32>(CALLBACK_SAMPLE_CAPACITY);
        let (asr_tx, asr_rx) = crossbeam_channel::bounded::<Vec<AudioFrame>>(8);
        let (event_tx, event_rx) = crossbeam_channel::unbounded::<ContinueEarEvent>();
        let dropped_in_callback = Arc::new(AtomicUsize::new(0));
        let err_fn = |err| eprintln!("input stream error: {err}");
        let stream = match supported_config.sample_format() {
            cpal::SampleFormat::F32 => build_continue_input_stream::<f32>(
                &device,
                &stream_config,
                sample_tx.clone(),
                Arc::clone(&dropped_in_callback),
                Arc::clone(&config.capture_enabled),
                err_fn,
            )?,
            cpal::SampleFormat::F64 => build_continue_input_stream::<f64>(
                &device,
                &stream_config,
                sample_tx.clone(),
                Arc::clone(&dropped_in_callback),
                Arc::clone(&config.capture_enabled),
                err_fn,
            )?,
            cpal::SampleFormat::I8 => build_continue_input_stream::<i8>(
                &device,
                &stream_config,
                sample_tx.clone(),
                Arc::clone(&dropped_in_callback),
                Arc::clone(&config.capture_enabled),
                err_fn,
            )?,
            cpal::SampleFormat::I16 => build_continue_input_stream::<i16>(
                &device,
                &stream_config,
                sample_tx.clone(),
                Arc::clone(&dropped_in_callback),
                Arc::clone(&config.capture_enabled),
                err_fn,
            )?,
            cpal::SampleFormat::I32 => build_continue_input_stream::<i32>(
                &device,
                &stream_config,
                sample_tx.clone(),
                Arc::clone(&dropped_in_callback),
                Arc::clone(&config.capture_enabled),
                err_fn,
            )?,
            cpal::SampleFormat::I64 => build_continue_input_stream::<i64>(
                &device,
                &stream_config,
                sample_tx.clone(),
                Arc::clone(&dropped_in_callback),
                Arc::clone(&config.capture_enabled),
                err_fn,
            )?,
            cpal::SampleFormat::U8 => build_continue_input_stream::<u8>(
                &device,
                &stream_config,
                sample_tx.clone(),
                Arc::clone(&dropped_in_callback),
                Arc::clone(&config.capture_enabled),
                err_fn,
            )?,
            cpal::SampleFormat::U16 => build_continue_input_stream::<u16>(
                &device,
                &stream_config,
                sample_tx.clone(),
                Arc::clone(&dropped_in_callback),
                Arc::clone(&config.capture_enabled),
                err_fn,
            )?,
            cpal::SampleFormat::U32 => build_continue_input_stream::<u32>(
                &device,
                &stream_config,
                sample_tx.clone(),
                Arc::clone(&dropped_in_callback),
                Arc::clone(&config.capture_enabled),
                err_fn,
            )?,
            cpal::SampleFormat::U64 => build_continue_input_stream::<u64>(
                &device,
                &stream_config,
                sample_tx,
                Arc::clone(&dropped_in_callback),
                Arc::clone(&config.capture_enabled),
                err_fn,
            )?,
            sample_format => anyhow::bail!("unsupported input sample format: {sample_format:?}"),
        };
        stream
            .play()
            .with_context(|| format!("failed to start capture from {device_name}"))?;

        eprintln!(
            "dev continue ear listening on {device_name}: {} Hz, {} channel(s), vad={}.",
            input_sample_rate_hz,
            input_channels,
            config.vad_backend.as_str()
        );

        let stop_for_asr = Arc::clone(&stop);
        let event_tx_for_asr = event_tx.clone();
        let asr = std::thread::Builder::new()
            .name("listenbury-dev-continue-asr".to_string())
            .spawn(move || {
                while !stop_for_asr.load(Ordering::Relaxed) {
                    match asr_rx.recv_timeout(Duration::from_millis(20)) {
                        Ok(frames) => match transcribe_group(&frames, &mut recognizer) {
                            Ok(text) if !text.is_empty() => {
                                if event_tx_for_asr
                                    .send(ContinueEarEvent::Transcript(text))
                                    .is_err()
                                {
                                    return;
                                }
                            }
                            Ok(_) => {}
                            Err(error) => {
                                let _ = event_tx_for_asr
                                    .send(ContinueEarEvent::Error(error.to_string()));
                            }
                        },
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return,
                    }
                }
            })
            .context("failed to spawn dev continue ASR worker")?;

        let stop_for_processor = Arc::clone(&stop);
        let processor = std::thread::Builder::new()
            .name("listenbury-dev-continue-ear".to_string())
            .spawn(move || {
                if let Err(error) = run_continue_ear_processor(
                    sample_rx,
                    asr_tx,
                    stop_for_processor,
                    config.vad_backend,
                    input_sample_rate_hz,
                    input_channels,
                ) {
                    let _ = event_tx.send(ContinueEarEvent::Error(error.to_string()));
                }
            })
            .context("failed to spawn dev continue ear worker")?;

        Ok((
            Self {
                stop,
                _stream: stream,
                processor: Some(processor),
                asr: Some(asr),
            },
            event_rx,
        ))
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
impl Drop for ContinueEar {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.processor.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.asr.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
struct ContinueEarState {
    vad: Box<dyn VoiceActivityDetector>,
    segmenter: BreathGroupSegmenter,
    active_groups: HashMap<BreathGroupId, Vec<AudioFrame>>,
    frame_time_ms: u64,
    last_vad_state: Option<bool>,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn run_continue_ear_processor(
    sample_rx: crossbeam_channel::Receiver<f32>,
    asr_tx: crossbeam_channel::Sender<Vec<AudioFrame>>,
    stop: Arc<AtomicBool>,
    vad_backend: VadBackendKind,
    input_sample_rate_hz: u32,
    input_channels: u16,
) -> Result<()> {
    let input_frame_samples =
        frame_samples_per_callback_frame(input_sample_rate_hz, input_channels);
    let (frame_sample_rate_hz, frame_channels) =
        vad_frame_format(vad_backend, input_sample_rate_hz, input_channels);
    let mut pending = VecDeque::<f32>::new();
    let mut state = ContinueEarState {
        vad: create_vad_backend(vad_backend)?,
        segmenter: BreathGroupSegmenter::default(),
        active_groups: HashMap::new(),
        frame_time_ms: 0,
        last_vad_state: None,
    };

    while !stop.load(Ordering::Relaxed) {
        match sample_rx.recv_timeout(Duration::from_millis(20)) {
            Ok(sample) => pending.push_back(sample),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
        while let Ok(sample) = sample_rx.try_recv() {
            pending.push_back(sample);
        }
        drain_pending_continue_ear_frames(
            &mut pending,
            input_frame_samples,
            input_sample_rate_hz,
            input_channels,
            frame_sample_rate_hz,
            frame_channels,
            &mut state,
            &asr_tx,
        )?;
    }

    while let Ok(sample) = sample_rx.try_recv() {
        pending.push_back(sample);
    }
    drain_pending_continue_ear_frames(
        &mut pending,
        input_frame_samples,
        input_sample_rate_hz,
        input_channels,
        frame_sample_rate_hz,
        frame_channels,
        &mut state,
        &asr_tx,
    )?;
    for (_, frames) in state.active_groups.drain() {
        if !frames.is_empty() && asr_tx.send(frames).is_err() {
            break;
        }
    }

    Ok(())
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn drain_pending_continue_ear_frames(
    pending: &mut VecDeque<f32>,
    input_frame_samples: usize,
    input_sample_rate_hz: u32,
    input_channels: u16,
    frame_sample_rate_hz: u32,
    frame_channels: u16,
    state: &mut ContinueEarState,
    asr_tx: &crossbeam_channel::Sender<Vec<AudioFrame>>,
) -> Result<()> {
    while pending.len() >= input_frame_samples {
        let mut samples = Vec::with_capacity(input_frame_samples);
        for _ in 0..input_frame_samples {
            if let Some(sample) = pending.pop_front() {
                samples.push(sample);
            }
        }
        if samples.len() < input_frame_samples {
            break;
        }
        let samples = convert_frame_samples(
            &samples,
            input_sample_rate_hz,
            input_channels,
            frame_sample_rate_hz,
            frame_channels,
        );
        let frame = AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: frame_sample_rate_hz,
            channels: frame_channels,
            samples,
        };
        process_continue_ear_frame(frame, state, asr_tx)?;
    }
    Ok(())
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn process_continue_ear_frame(
    frame: AudioFrame,
    state: &mut ContinueEarState,
    asr_tx: &crossbeam_channel::Sender<Vec<AudioFrame>>,
) -> Result<()> {
    let frame_duration_ms = frame_duration_ms(&frame);
    let vad_result = state.vad.process_frame(&frame)?;
    if listenbury::developer_diagnostics_enabled()
        && state.last_vad_state != Some(vad_result.is_speech)
    {
        eprintln!(
            "[dev continue ear] vad t_ms={} speech={} prob={:.3}",
            state.frame_time_ms, vad_result.is_speech, vad_result.speech_prob
        );
        state.last_vad_state = Some(vad_result.is_speech);
    }

    let events = state.segmenter.process(vad_result);
    for event in &events {
        if let HearingEvent::BreathGroupOpened { id } = event {
            state.active_groups.entry(*id).or_default();
        }
    }
    for group in state.active_groups.values_mut() {
        group.push(frame.clone());
    }
    for event in events {
        if let HearingEvent::BreathGroupClosed { id, .. } = event {
            if let Some(group_frames) = state.active_groups.remove(&id) {
                if !group_frames.is_empty() && asr_tx.send(group_frames).is_err() {
                    return Ok(());
                }
            }
        }
    }
    state.frame_time_ms = state.frame_time_ms.saturating_add(frame_duration_ms);
    Ok(())
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn build_continue_input_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_tx: crossbeam_channel::Sender<f32>,
    dropped_in_callback: Arc<AtomicUsize>,
    capture_enabled: Arc<AtomicBool>,
    err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream>
where
    T: Sample + SizedSample,
    f32: FromSample<T>,
{
    device
        .build_input_stream(
            config,
            move |data: &[T], _| {
                if !capture_enabled.load(Ordering::Relaxed) {
                    return;
                }
                for sample in data {
                    if sample_tx.try_send(sample.to_sample::<f32>()).is_err() {
                        dropped_in_callback.fetch_add(1, Ordering::Relaxed);
                    }
                }
            },
            err_fn,
            None,
        )
        .context("failed to build input stream")
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn vad_frame_format(
    vad_backend: VadBackendKind,
    input_sample_rate_hz: u32,
    input_channels: u16,
) -> (u32, u16) {
    match vad_backend {
        VadBackendKind::WebRtc => (WEBRTC_VAD_SAMPLE_RATE_HZ, MONO_CHANNELS),
        VadBackendKind::Energy | VadBackendKind::Silero => (input_sample_rate_hz, input_channels),
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn convert_frame_samples(
    samples: &[f32],
    input_sample_rate_hz: u32,
    input_channels: u16,
    frame_sample_rate_hz: u32,
    frame_channels: u16,
) -> Vec<f32> {
    if input_sample_rate_hz == frame_sample_rate_hz && input_channels == frame_channels {
        return samples.to_vec();
    }

    let mut converted = if input_channels != frame_channels && frame_channels == MONO_CHANNELS {
        mix_to_mono(samples, input_channels)
    } else {
        samples.to_vec()
    };

    if input_sample_rate_hz != frame_sample_rate_hz {
        converted = resample_linear(&converted, input_sample_rate_hz, frame_sample_rate_hz);
    }

    converted
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn frame_samples_per_callback_frame(sample_rate_hz: u32, channels: u16) -> usize {
    let samples_per_channel = usize::try_from(sample_rate_hz / 100).unwrap_or(1).max(1);
    samples_per_channel.saturating_mul(usize::from(channels).max(1))
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn frame_duration_ms(frame: &AudioFrame) -> u64 {
    if frame.sample_rate_hz == 0 || frame.channels == 0 {
        return 0;
    }
    let samples_per_channel = frame.samples.len() as f64 / f64::from(frame.channels);
    ((samples_per_channel / f64::from(frame.sample_rate_hz)) * 1000.0).round() as u64
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn mix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    let channel_count = usize::from(channels).max(1);
    if channel_count == 1 {
        return samples.to_vec();
    }
    samples
        .chunks_exact(channel_count)
        .map(|frame| frame.iter().sum::<f32>() / f32::from(channels))
        .collect()
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn resample_linear(samples: &[f32], source_rate_hz: u32, target_rate_hz: u32) -> Vec<f32> {
    if samples.is_empty() || source_rate_hz == target_rate_hz {
        return samples.to_vec();
    }

    let output_len = ((samples.len() as f64 * f64::from(target_rate_hz))
        / f64::from(source_rate_hz))
    .round() as usize;
    let mut output = Vec::with_capacity(output_len);
    let source_step = f64::from(source_rate_hz) / f64::from(target_rate_hz);

    for output_idx in 0..output_len {
        let source_pos = output_idx as f64 * source_step;
        let left_idx = source_pos.floor() as usize;
        let right_idx = (left_idx + 1).min(samples.len() - 1);
        let fraction = (source_pos - left_idx as f64) as f32;
        let left = samples[left_idx];
        let right = samples[right_idx];
        output.push(left + (right - left) * fraction);
    }

    output
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
struct ContinueMouth {
    tts: PiperTextToSpeech,
    queued_text: VecDeque<String>,
    capture_enabled: Arc<AtomicBool>,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
impl ContinueMouth {
    fn new(tts: PiperTextToSpeech, capture_enabled: Arc<AtomicBool>) -> Self {
        Self {
            tts,
            queued_text: VecDeque::new(),
            capture_enabled,
        }
    }

    fn enqueue_runtime_event(&mut self, event: &ContinueRuntimeEvent) -> Result<bool> {
        let ContinueRuntimeEvent::UtteranceCompleted { content, .. } = event else {
            return Ok(false);
        };
        if strip_emoji(content).trim().is_empty() {
            return Ok(false);
        }

        self.tts
            .enqueue(SpeechPlan::from(SpeechUnit::CompleteSentence(
                content.to_string(),
            )))?;
        self.queued_text.push_back(content.to_string());
        Ok(true)
    }

    fn is_throttling_llm(&self) -> bool {
        !self.queued_text.is_empty()
    }

    fn drain_ready_audio(&mut self, source: &str) -> Result<bool> {
        let frames = self.tts.poll_audio()?;
        if frames.is_empty() {
            return Ok(false);
        }

        let spoken_text = self.queued_text.pop_front().unwrap_or_default();
        eprintln!("[dev continue] speaking: {spoken_text}");
        self.capture_enabled.store(false, Ordering::Relaxed);
        let playback = play_audio_frames(&frames, source);
        self.capture_enabled.store(true, Ordering::Relaxed);
        playback?;
        Ok(true)
    }
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn build_initial_prompt(prompt_words: &[String]) -> String {
    let seed = if prompt_words.is_empty() {
        DEFAULT_CONTINUE_PROMPT.to_string()
    } else {
        prompt_words.join(" ")
    };
    format!("{seed}\n\n{LIVE_EVENT_INSTRUCTIONS}\n\n")
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
#[derive(Debug, Default)]
struct SpeechEventDetector {
    pending: String,
    in_speech: bool,
    next_utterance_id: u64,
    current_utterance_id: Option<u64>,
    current_utterance_content: String,
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
#[derive(Debug, Clone, PartialEq, Eq)]
enum ContinueRuntimeEvent {
    UtteranceStarted { id: u64 },
    UtteranceCompleted { id: u64, content: String },
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
impl ContinueRuntimeEvent {
    fn to_message(&self) -> String {
        match self {
            Self::UtteranceStarted { id } => {
                format!("utterance_started: id={id}")
            }
            Self::UtteranceCompleted { id, content } => {
                format!(
                    "utterance_completed: id={id}\ncontent:\n{}",
                    sanitize_runtime_event_content(content)
                )
            }
        }
    }
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
impl SpeechEventDetector {
    fn ingest(&mut self, text: &str) -> Vec<ContinueRuntimeEvent> {
        self.pending.push_str(text);
        let mut events = Vec::new();

        loop {
            if self.in_speech {
                let Some(next_marker) = self.next_speech_close_marker() else {
                    self.commit_pending_speech_text_before_marker_prefix(&mut events);
                    return events;
                };
                let marker_text = next_marker.text;
                let speech_text = self.pending[..next_marker.index].to_string();
                let marker_end = next_marker.index + marker_text.len();
                self.pending.drain(..marker_end);

                self.append_speech_text(&speech_text, &mut events);
                if let Some(event) = self.flush_current_utterance() {
                    events.push(event);
                }
                if next_marker.kind == SpeechMarkerKind::Start {
                    events.push(self.start_utterance());
                }
            } else {
                let Some(next_marker) = next_speech_marker(&self.pending, SpeechMarkerKind::Start)
                else {
                    self.trim_pending_to_marker_prefix();
                    return events;
                };
                let marker_end = next_marker.index + next_marker.text.len();
                self.pending.drain(..marker_end);
                events.push(self.start_utterance());
            }
        }
    }

    fn start_utterance(&mut self) -> ContinueRuntimeEvent {
        self.in_speech = true;
        self.current_utterance_content.clear();
        self.open_utterance()
    }

    fn open_utterance(&mut self) -> ContinueRuntimeEvent {
        let id = self.next_utterance_id;
        self.next_utterance_id += 1;
        self.current_utterance_id = Some(id);
        ContinueRuntimeEvent::UtteranceStarted { id }
    }

    fn append_speech_text(&mut self, text: &str, events: &mut Vec<ContinueRuntimeEvent>) {
        if text.is_empty() {
            return;
        }
        self.ensure_utterance_started_if_needed(text, events);
        self.current_utterance_content.push_str(text);
        self.emit_completed_sentences(events);
    }

    fn ensure_utterance_started_if_needed(
        &mut self,
        text: &str,
        events: &mut Vec<ContinueRuntimeEvent>,
    ) {
        if self.current_utterance_id.is_none() && !text.trim().is_empty() {
            events.push(self.start_utterance());
        }
    }

    fn emit_completed_sentences(&mut self, events: &mut Vec<ContinueRuntimeEvent>) {
        while let Some(end) = seams_sentence_end(&self.current_utterance_content) {
            let sentence = self.current_utterance_content[..end].trim().to_string();
            self.current_utterance_content.drain(..end);

            if let Some(id) = self.current_utterance_id.take() {
                events.push(ContinueRuntimeEvent::UtteranceCompleted {
                    id,
                    content: sentence,
                });
            }

            let leading_whitespace = self.current_utterance_content.len()
                - self.current_utterance_content.trim_start().len();
            if leading_whitespace > 0 {
                self.current_utterance_content.drain(..leading_whitespace);
            }
            if !self.current_utterance_content.trim().is_empty() {
                events.push(self.open_utterance());
            }
        }
    }

    fn flush_current_utterance(&mut self) -> Option<ContinueRuntimeEvent> {
        self.in_speech = false;
        let id = self.current_utterance_id.take()?;
        let content = self.current_utterance_content.trim().to_string();
        self.current_utterance_content.clear();
        if content.is_empty() {
            return None;
        }
        Some(ContinueRuntimeEvent::UtteranceCompleted { id, content })
    }

    fn next_speech_close_marker(&self) -> Option<SpeechMarker> {
        next_any_speech_marker(&self.pending)
    }

    fn commit_pending_speech_text_before_marker_prefix(
        &mut self,
        events: &mut Vec<ContinueRuntimeEvent>,
    ) {
        let keep = longest_marker_prefix_suffix_len(&self.pending);
        let emit_len = self.pending.len() - keep;
        let speech_text = self.pending[..emit_len].to_string();
        self.pending = self.pending[emit_len..].to_string();
        self.append_speech_text(&speech_text, events);
    }

    fn trim_pending_to_marker_prefix(&mut self) {
        let keep = longest_marker_prefix_suffix_len(&self.pending);
        if keep == 0 {
            self.pending.clear();
        } else {
            let start = self.pending.len() - keep;
            self.pending = self.pending[start..].to_string();
        }
    }
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpeechMarkerKind {
    Start,
    End,
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SpeechMarker {
    kind: SpeechMarkerKind,
    index: usize,
    text: &'static str,
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn next_any_speech_marker(text: &str) -> Option<SpeechMarker> {
    [
        next_speech_marker(text, SpeechMarkerKind::Start),
        next_speech_marker(text, SpeechMarkerKind::End),
    ]
    .into_iter()
    .flatten()
    .min_by_key(|marker| marker.index)
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn next_speech_marker(text: &str, kind: SpeechMarkerKind) -> Option<SpeechMarker> {
    speech_markers(kind)
        .into_iter()
        .filter_map(|marker| {
            text.find(marker).map(|index| SpeechMarker {
                kind,
                index,
                text: marker,
            })
        })
        .min_by_key(|marker| marker.index)
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn speech_markers(kind: SpeechMarkerKind) -> [&'static str; 1] {
    match kind {
        SpeechMarkerKind::Start => [INLINE_SPEECH_START_MARKER],
        SpeechMarkerKind::End => [INLINE_SPEECH_END_MARKER],
    }
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn all_speech_markers() -> [&'static str; 2] {
    [INLINE_SPEECH_START_MARKER, INLINE_SPEECH_END_MARKER]
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn longest_marker_prefix_suffix_len(text: &str) -> usize {
    all_speech_markers()
        .into_iter()
        .flat_map(|marker| {
            marker
                .char_indices()
                .skip(1)
                .map(|(index, _)| index)
                .chain(std::iter::once(marker.len()))
                .filter(|&len| len <= text.len())
                .filter(|&len| text.ends_with(&marker[..len]))
        })
        .max()
        .unwrap_or(0)
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn sentence_detector() -> Option<&'static SentenceDetectorDialog> {
    static DETECTOR: OnceLock<Option<SentenceDetectorDialog>> = OnceLock::new();
    DETECTOR
        .get_or_init(|| SentenceDetectorDialog::new().ok())
        .as_ref()
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn seams_sentence_end(text: &str) -> Option<usize> {
    let detector = sentence_detector()?;
    let sentences = detector.detect_sentences_borrowed(text).ok()?;
    let mut search_from = 0;
    for sentence in sentences {
        if let Some(rel) = text[search_from..].find(sentence.raw_content) {
            let start = search_from + rel;
            let end = start + sentence.raw_content.len();
            search_from = end;
            if sentence.raw_content.trim().ends_with(['.', '?', '!']) {
                return Some(end);
            }
        }
    }
    None
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn sanitize_runtime_event_content(content: &str) -> String {
    content
        .replace("--- END LIVE EVENT ---", "[end live event]")
        .replace("--- LIVE EVENT:", "[live event]")
}

#[cfg(test)]
mod tests {
    use super::{
        ContinueRuntimeEvent, SpeechEventDetector, build_initial_prompt, wrap_heard_input,
        wrap_live_input, wrap_runtime_event, wrap_time_event,
    };

    #[test]
    fn stdin_append_is_wrapped_as_live_input() {
        assert_eq!(
            wrap_live_input("turn toward the window\n"),
            "\n\n--- LIVE EVENT: user ---\nturn toward the window\n--- END LIVE EVENT ---\n\n"
        );
    }

    #[test]
    fn time_append_is_wrapped_as_live_input() {
        assert_eq!(
            wrap_time_event("The current Unix time is 42.000 seconds."),
            "\n\n--- LIVE EVENT: clock ---\nThe current Unix time is 42.000 seconds.\n--- END LIVE EVENT ---\n\n"
        );
    }

    #[test]
    fn heard_speech_is_wrapped_as_live_input() {
        assert_eq!(
            wrap_heard_input("hello from the room\n"),
            "\n\n--- LIVE EVENT: ear ---\nHeard: hello from the room\n--- END LIVE EVENT ---\n\n"
        );
    }

    #[test]
    fn runtime_event_is_wrapped_as_live_input() {
        assert_eq!(
            wrap_runtime_event("utterance_started: id=0"),
            "\n\n--- LIVE EVENT: runtime ---\nutterance_started: id=0\n--- END LIVE EVENT ---\n\n"
        );
        assert_eq!(
            ContinueRuntimeEvent::UtteranceStarted { id: 7 }.to_message(),
            "utterance_started: id=7"
        );
    }

    #[test]
    fn initial_prompt_includes_live_event_hygiene() {
        let prompt = build_initial_prompt(&["Think continuously.".to_string()]);

        assert!(prompt.starts_with("Think continuously.\n\n"));
        assert!(prompt.contains("Live events may appear in the transcript"));
        assert!(prompt.contains("Do not copy live event delimiters or runtime event text."));
        assert!(prompt.contains("Do not write system, assistant, analysis, channel"));
        assert!(prompt.contains("<sp>words to say aloud :)</sp>"));
        assert!(!prompt.contains("--- SPEECH ---"));
        assert!(prompt.contains("Emoji inside speech tags are instructions to your countenance"));
    }

    #[test]
    fn speech_detector_parses_inline_speech() {
        let mut detector = SpeechEventDetector::default();

        assert_eq!(
            detector
                .ingest("<sp>:) This is how I speak. Parse here. And here. And here...live</sp>"),
            vec![
                ContinueRuntimeEvent::UtteranceStarted { id: 0 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 0,
                    content: ":) This is how I speak.".to_string()
                },
                ContinueRuntimeEvent::UtteranceStarted { id: 1 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 1,
                    content: "Parse here.".to_string()
                },
                ContinueRuntimeEvent::UtteranceStarted { id: 2 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 2,
                    content: "And here.".to_string()
                },
                ContinueRuntimeEvent::UtteranceStarted { id: 3 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 3,
                    content: "And here...live".to_string()
                }
            ]
        );
    }

    #[test]
    fn speech_detector_handles_split_inline_marker() {
        let mut detector = SpeechEventDetector::default();

        assert!(detector.ingest("<s").is_empty());
        assert_eq!(
            detector.ingest("p>Hello</sp>"),
            vec![
                ContinueRuntimeEvent::UtteranceStarted { id: 0 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 0,
                    content: "Hello".to_string()
                }
            ]
        );
    }

    #[test]
    fn speech_detector_emits_utterance_started_on_marker() {
        let mut detector = SpeechEventDetector::default();

        assert_eq!(
            detector.ingest("thinking <sp>Hello"),
            vec![ContinueRuntimeEvent::UtteranceStarted { id: 0 }]
        );
    }

    #[test]
    fn speech_detector_handles_split_marker() {
        let mut detector = SpeechEventDetector::default();

        assert!(detector.ingest("thinking <s").is_empty());
        assert_eq!(
            detector.ingest("p>Hello"),
            vec![ContinueRuntimeEvent::UtteranceStarted { id: 0 }]
        );
    }

    #[test]
    fn speech_detector_rearms_after_end_marker() {
        let mut detector = SpeechEventDetector::default();

        assert_eq!(
            detector.ingest("<sp>First</sp>Later <sp>Second"),
            vec![
                ContinueRuntimeEvent::UtteranceStarted { id: 0 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 0,
                    content: "First".to_string()
                },
                ContinueRuntimeEvent::UtteranceStarted { id: 1 }
            ]
        );
    }

    #[test]
    fn speech_detector_emits_utterance_completed_on_end_marker() {
        let mut detector = SpeechEventDetector::default();

        assert_eq!(
            detector.ingest("<sp>Hello there.</sp>"),
            vec![
                ContinueRuntimeEvent::UtteranceStarted { id: 0 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 0,
                    content: "Hello there.".to_string()
                }
            ]
        );
    }

    #[test]
    fn speech_detector_treats_nested_start_as_recovery_boundary() {
        let mut detector = SpeechEventDetector::default();

        assert_eq!(
            detector.ingest("<sp>Hello<sp>what happens up here?"),
            vec![
                ContinueRuntimeEvent::UtteranceStarted { id: 0 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 0,
                    content: "Hello".to_string()
                },
                ContinueRuntimeEvent::UtteranceStarted { id: 1 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 1,
                    content: "what happens up here?".to_string()
                }
            ]
        );
    }

    #[test]
    fn speech_detector_emits_complete_sentences_from_head_before_end_marker() {
        let mut detector = SpeechEventDetector::default();

        assert_eq!(
            detector.ingest("<sp>First sentence. Second"),
            vec![
                ContinueRuntimeEvent::UtteranceStarted { id: 0 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 0,
                    content: "First sentence.".to_string()
                },
                ContinueRuntimeEvent::UtteranceStarted { id: 1 }
            ]
        );
        assert!(detector.ingest(" sentence").is_empty());
        assert_eq!(
            detector.ingest(".</sp>"),
            vec![ContinueRuntimeEvent::UtteranceCompleted {
                id: 1,
                content: "Second sentence.".to_string()
            }]
        );
    }

    #[test]
    fn speech_detector_captures_content_across_chunks() {
        let mut detector = SpeechEventDetector::default();

        assert_eq!(
            detector.ingest("<sp>Hello "),
            vec![ContinueRuntimeEvent::UtteranceStarted { id: 0 }]
        );
        assert_eq!(
            detector.ingest("there.</sp>"),
            vec![ContinueRuntimeEvent::UtteranceCompleted {
                id: 0,
                content: "Hello there.".to_string()
            }]
        );
    }

    #[test]
    fn speech_detector_ignores_legacy_block_delimiters() {
        let mut detector = SpeechEventDetector::default();

        assert!(
            detector
                .ingest("--- SPEECH ---no fallback--- END SPEECH ---")
                .is_empty()
        );
    }
}
