use crate::cli::LiveHalfDuplexCommand;
use anyhow::Result;

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use crate::cli::ModelProfile;
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
use crate::cli::commands::mic_transcribe::transcribe_group;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use crate::cli::model_paths::{resolve_llm_model, resolve_piper_voice, resolve_whisper_model};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use crate::cli::piper::{collect_tts_audio, piper_config_for_voice, resolve_piper_bin};
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
use listenbury::audio::ring::make_audio_ring;
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
use listenbury::hearing::vad::{EnergyVad, VoiceActivityDetector};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::mind::llm::{GenerationRequest, LlmEngine, LlmEvent};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::mouth::planner::{ExpressiveUnit, SpeechPlan, SpeechPlanner, SpeechUnit};
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
use listenbury::speech::recognizer::SpeechRecognizer;
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
use std::collections::{HashMap, VecDeque};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use std::time::{Duration, Instant};

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
const AUDIO_RING_CAPACITY: usize = 256;

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
#[derive(Debug)]
struct LiveHalfDuplexState {
    vad: EnergyVad,
    segmenter: BreathGroupSegmenter,
    active_groups: HashMap<BreathGroupId, Vec<AudioFrame>>,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
#[derive(Debug, Clone)]
struct LiveHalfDuplexModelPaths {
    whisper_model: std::path::PathBuf,
    llm_model: std::path::PathBuf,
    piper_bin: std::path::PathBuf,
    piper_voice: std::path::PathBuf,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
impl LiveHalfDuplexModelPaths {
    fn discover(command: &LiveHalfDuplexCommand) -> Result<Self> {
        Ok(Self {
            whisper_model: resolve_whisper_model(command.whisper_model.clone())?,
            llm_model: resolve_llm_model(command.llm_model.clone())?,
            piper_bin: resolve_piper_bin(command.piper_bin.clone())?,
            piper_voice: resolve_piper_voice(command.piper_voice.clone())?,
        })
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
pub(crate) fn run_live_half_duplex(command: LiveHalfDuplexCommand) -> Result<()> {
    anyhow::ensure!(command.seconds > 0, "--seconds must be greater than zero");

    let paths = LiveHalfDuplexModelPaths::discover(&command)?;
    let mut recognizer = listenbury::WhisperSpeechRecognizer::new(&paths.whisper_model)
        .with_context(|| {
            format!(
                "failed to load Whisper model at {}",
                paths.whisper_model.display()
            )
        })?;
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
    let mut tts = PiperTextToSpeech::new(piper_config_for_voice(
        paths.piper_bin.clone(),
        paths.piper_voice.clone(),
    )?);

    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("no default input device available"))?;
    let input_name = input_device
        .name()
        .unwrap_or_else(|_| "<unknown input device>".to_string());
    let supported_input = input_device
        .default_input_config()
        .with_context(|| format!("failed to read default input config for {input_name}"))?;
    let stream_config = supported_input.config();
    let input_sample_rate_hz = stream_config.sample_rate.0;
    let input_channels = stream_config.channels;
    anyhow::ensure!(
        input_channels > 0,
        "default input device reported zero channels"
    );

    let capture_enabled = Arc::new(AtomicBool::new(true));
    let (sample_tx, sample_rx) = crossbeam_channel::bounded::<f32>(CALLBACK_SAMPLE_CAPACITY);
    let dropped_in_callback = Arc::new(AtomicUsize::new(0));
    let dropped_in_ring = Arc::new(AtomicUsize::new(0));
    let err_fn = |err| eprintln!("input stream error: {err}");
    let stream = match supported_input.sample_format() {
        cpal::SampleFormat::F32 => build_input_stream::<f32>(
            &input_device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            Arc::clone(&capture_enabled),
            err_fn,
        )?,
        cpal::SampleFormat::F64 => build_input_stream::<f64>(
            &input_device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            Arc::clone(&capture_enabled),
            err_fn,
        )?,
        cpal::SampleFormat::I8 => build_input_stream::<i8>(
            &input_device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            Arc::clone(&capture_enabled),
            err_fn,
        )?,
        cpal::SampleFormat::I16 => build_input_stream::<i16>(
            &input_device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            Arc::clone(&capture_enabled),
            err_fn,
        )?,
        cpal::SampleFormat::I32 => build_input_stream::<i32>(
            &input_device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            Arc::clone(&capture_enabled),
            err_fn,
        )?,
        cpal::SampleFormat::I64 => build_input_stream::<i64>(
            &input_device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            Arc::clone(&capture_enabled),
            err_fn,
        )?,
        cpal::SampleFormat::U8 => build_input_stream::<u8>(
            &input_device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            Arc::clone(&capture_enabled),
            err_fn,
        )?,
        cpal::SampleFormat::U16 => build_input_stream::<u16>(
            &input_device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            Arc::clone(&capture_enabled),
            err_fn,
        )?,
        cpal::SampleFormat::U32 => build_input_stream::<u32>(
            &input_device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            Arc::clone(&capture_enabled),
            err_fn,
        )?,
        cpal::SampleFormat::U64 => build_input_stream::<u64>(
            &input_device,
            &stream_config,
            sample_tx,
            Arc::clone(&dropped_in_callback),
            Arc::clone(&capture_enabled),
            err_fn,
        )?,
        sample_format => anyhow::bail!("unsupported input sample format: {sample_format:?}"),
    };
    stream
        .play()
        .with_context(|| format!("failed to start capture from {input_name}"))?;

    println!(
        "live-half-duplex listening on {input_name}: {} Hz, {} channel(s).",
        input_sample_rate_hz, input_channels
    );
    println!("half-duplex mode: no barge-in, no interruption during Pete's speech.");

    let stop_deadline = Instant::now() + Duration::from_secs(command.seconds);
    let input_frame_samples =
        frame_samples_per_callback_frame(input_sample_rate_hz, input_channels);
    let (mut ring_tx, mut ring_rx) = make_audio_ring(AUDIO_RING_CAPACITY);
    let mut pending = VecDeque::<f32>::new();
    let mut state = LiveHalfDuplexState {
        vad: EnergyVad::default(),
        segmenter: BreathGroupSegmenter::default(),
        active_groups: HashMap::new(),
    };
    let mut turns = 0usize;

    while Instant::now() < stop_deadline {
        match sample_rx.recv_timeout(Duration::from_millis(20)) {
            Ok(sample) => pending.push_back(sample),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
        while let Ok(sample) = sample_rx.try_recv() {
            pending.push_back(sample);
        }
        drain_pending_into_ring(
            &mut pending,
            input_frame_samples,
            input_sample_rate_hz,
            input_channels,
            &mut ring_tx,
            &dropped_in_ring,
        );
        let closed_groups = process_ring_frames(&mut ring_rx, &mut state)?;
        for group_frames in closed_groups {
            let transcript = transcribe_group(&group_frames, &mut recognizer)?;
            let transcript = transcript.trim();
            if transcript.is_empty() {
                continue;
            }

            println!("Heard: {transcript}");
            capture_enabled.store(false, Ordering::SeqCst);
            let plan = generate_speech_plan(
                &mut llm,
                transcript,
                command.model_profile,
                command.no_backchannels,
            )?;
            tts.enqueue(plan)?;
            let audio = collect_tts_audio(&mut tts, Duration::from_secs(30))?;
            play_audio_frames(&audio, "live-half-duplex response")?;
            capture_enabled.store(true, Ordering::SeqCst);
            turns += 1;
            println!("Listening...");
        }
    }

    drop(stream);

    println!(
        "live-half-duplex finished: turns={}, callback_drops={}, ring_drops={}",
        turns,
        dropped_in_callback.load(Ordering::Relaxed),
        dropped_in_ring.load(Ordering::Relaxed),
    );
    Ok(())
}

#[cfg(not(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
)))]
pub(crate) fn run_live_half_duplex(_command: LiveHalfDuplexCommand) -> Result<()> {
    anyhow::bail!(
        "listenbury live-half-duplex requires the `audio-cpal`, `asr-whisper`, `llm-llama-cpp`, and `tts-piper` features"
    )
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn process_live_frame(
    frame: AudioFrame,
    state: &mut LiveHalfDuplexState,
) -> Result<Vec<Vec<AudioFrame>>> {
    let vad_result = state.vad.process_frame(&frame)?;
    let events = state.segmenter.process(vad_result);
    for event in &events {
        if let HearingEvent::BreathGroupOpened { id } = event {
            state.active_groups.entry(*id).or_default();
        }
    }
    for group in state.active_groups.values_mut() {
        group.push(frame.clone());
    }

    let mut closed_groups = Vec::new();
    for event in events {
        if let HearingEvent::BreathGroupClosed { id, .. } = event {
            if let Some(group_frames) = state.active_groups.remove(&id) {
                closed_groups.push(group_frames);
            }
        }
    }
    Ok(closed_groups)
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn process_ring_frames(
    ring_rx: &mut listenbury::audio::ring::AudioRingRx,
    state: &mut LiveHalfDuplexState,
) -> Result<Vec<Vec<AudioFrame>>> {
    let mut closed_groups = Vec::new();
    while let Some(frame) = ring_rx.try_pop() {
        closed_groups.extend(process_live_frame(frame, state)?);
    }
    Ok(closed_groups)
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn generate_speech_plan(
    llm: &mut LlamaCppEngine,
    transcript: &str,
    model_profile: ModelProfile,
    no_backchannels: bool,
) -> Result<SpeechPlan> {
    let generation_id = llm
        .start(GenerationRequest {
            prompt: build_prompt(transcript),
            max_tokens: Some(max_tokens(model_profile)),
        })
        .context("failed to start llama.cpp generation")?;

    let mut planner = SpeechPlanner::default();
    let mut spoken_parts = Vec::new();
    loop {
        let events = llm.poll(generation_id)?;
        if events.is_empty() {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        }

        for event in &events {
            if let LlmEvent::Error { message } = event {
                anyhow::bail!("llama.cpp generation failed: {message}");
            }
        }
        for unit in planner.ingest(&events) {
            let ExpressiveUnit::Speech(plan) = unit else {
                continue;
            };
            if no_backchannels && matches!(plan.unit(), SpeechUnit::Backchannel(_)) {
                continue;
            }
            spoken_parts.push(plan.text().to_string());
        }

        if events.iter().any(is_terminal_llm_event) {
            break;
        }
    }

    let text = spoken_parts.join(" ");
    let text = text.trim();
    let response = if text.is_empty() {
        "I heard you, but I lost my words.".to_string()
    } else {
        text.to_string()
    };
    Ok(SpeechPlan::from(SpeechUnit::FullTurn(response)))
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn build_prompt(transcript: &str) -> String {
    format!(
        "<|system|>\nYou are Pete, speaking aloud through a TTS system.\nWrite in short, complete spoken sentences.\nDo not rely on long subordinate clauses.\nPrefer natural sentence boundaries.\nEach sentence should be speakable on its own.</s>\n<|user|>\n{transcript}</s>\n<|assistant|>\n"
    )
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn max_tokens(model_profile: ModelProfile) -> usize {
    match model_profile {
        ModelProfile::Tiny => 96,
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn is_terminal_llm_event(event: &LlmEvent) -> bool {
    matches!(
        event,
        LlmEvent::Completed | LlmEvent::Cancelled | LlmEvent::Error { .. }
    )
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn drain_pending_into_ring(
    pending: &mut VecDeque<f32>,
    input_frame_samples: usize,
    input_sample_rate_hz: u32,
    input_channels: u16,
    ring_tx: &mut listenbury::audio::ring::AudioRingTx,
    dropped_in_ring: &AtomicUsize,
) {
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
        let frame = AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: input_sample_rate_hz,
            channels: input_channels,
            samples,
        };
        if ring_tx.try_push(frame).is_err() {
            dropped_in_ring.fetch_add(1, Ordering::Relaxed);
        }
    }
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
fn build_input_stream<T>(
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
