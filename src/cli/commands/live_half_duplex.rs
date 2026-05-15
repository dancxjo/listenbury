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
use listenbury::RuntimePacket;
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
use listenbury::hearing::vad::VadBackendKind;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::hearing::vad::{VoiceActivityDetector, create_vad_backend};
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use listenbury::mind::llm::LlmEvent;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::mind::llm::{GenerationRequest, LlmEngine};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::mouth::planner::FaceCommand;
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use listenbury::mouth::planner::{ExpressiveUnit, MouthCommand, SpeechPlan, SpeechUnit};
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
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use listenbury::{ConversationController, FillerContext};
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
const FILLER_SILENCE_DURATION_MS: u64 = 1_200;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
const AUDIO_DRAIN_QUIET_THRESHOLD_MS: u64 = 100;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
const POST_PLAYBACK_TTS_GRACE_MS: u64 = 1_500;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
const NANOS_PER_MILLI: u128 = 1_000_000;
const WEBRTC_VAD_SAMPLE_RATE_HZ: u32 = 16_000;
const MONO_CHANNELS: u16 = 1;

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
struct LiveHalfDuplexState {
    vad: Box<dyn VoiceActivityDetector>,
    segmenter: BreathGroupSegmenter,
    active_groups: HashMap<BreathGroupId, Vec<AudioFrame>>,
    self_hearing: listenbury::SelfHearingState,
    controller: ConversationController,
    frame_time_ms: u64,
    last_vad_state: Option<bool>,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
impl std::fmt::Debug for LiveHalfDuplexState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveHalfDuplexState")
            .field("vad", &"dyn VoiceActivityDetector")
            .field("segmenter", &self.segmenter)
            .field("active_groups", &self.active_groups)
            .field("self_hearing", &self.self_hearing)
            .field("controller", &self.controller)
            .field("frame_time_ms", &self.frame_time_ms)
            .field("last_vad_state", &self.last_vad_state)
            .finish()
    }
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
    if let Some(seconds) = command.seconds {
        anyhow::ensure!(seconds > 0, "--seconds must be greater than zero");
    }

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
        "live-half-duplex listening on {input_name}: {} Hz, {} channel(s), vad={}.",
        input_sample_rate_hz,
        input_channels,
        command.vad.as_backend_kind().as_str()
    );
    println!("half-duplex mode: no barge-in, no interruption during Pete's speech.");

    let stop_deadline = command
        .seconds
        .map(|seconds| Instant::now() + Duration::from_secs(seconds));
    let vad_backend = command.vad.as_backend_kind();
    let (frame_sample_rate_hz, frame_channels) =
        vad_frame_format(vad_backend, input_sample_rate_hz, input_channels);
    let input_frame_samples =
        frame_samples_per_callback_frame(input_sample_rate_hz, input_channels);
    let (mut ring_tx, mut ring_rx) = make_audio_ring(AUDIO_RING_CAPACITY);
    let mut pending = VecDeque::<f32>::new();
    let mut state = LiveHalfDuplexState {
        vad: create_vad_backend(vad_backend)?,
        segmenter: BreathGroupSegmenter::default(),
        active_groups: HashMap::new(),
        self_hearing: listenbury::SelfHearingState::default(),
        controller: ConversationController::default(),
        frame_time_ms: 0,
        last_vad_state: None,
    };
    let mut turns = 0usize;

    while stop_deadline.is_none_or(|deadline| Instant::now() < deadline) {
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
            frame_sample_rate_hz,
            frame_channels,
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
            state
                .controller
                .record_runtime_packet(RuntimePacket::TranscriptUpdated {
                    text: transcript.to_string(),
                    confidence: 1.0,
                });
            state.controller.apply_safe_boundary_updates();
            capture_enabled.store(false, Ordering::SeqCst);
            stream_speech_to_tts(
                &mut llm,
                &mut tts,
                transcript,
                command.model_profile,
                command.no_backchannels,
                &mut state.self_hearing,
                &mut state.controller,
                turns as u64 + 1,
            )?;
            state.controller.apply_safe_boundary_updates();
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
    if state
        .self_hearing
        .suppression_decision_at(frame.captured_at)
        == listenbury::SuppressionDecision::Suppress
    {
        // Pete is speaking or the echo-tail window is still active; drop the frame
        // so that VAD/ASR cannot transcribe Pete's own voice.
        return Ok(vec![]);
    }
    let frame_duration_ms = frame_duration_ms(&frame);
    let vad_result = state.vad.process_frame(&frame)?;
    if listenbury::developer_diagnostics_enabled()
        && state.last_vad_state != Some(vad_result.is_speech)
    {
        println!(
            "vad t_ms={} speech={} prob={:.3}",
            state.frame_time_ms, vad_result.is_speech, vad_result.speech_prob
        );
        state.last_vad_state = Some(vad_result.is_speech);
    }
    let events = state.segmenter.process(vad_result);
    let now_ms = unix_nanos_to_millis(frame.captured_at.unix_nanos);
    for event in &events {
        state.controller.on_hearing_event(event, now_ms);
        match event {
            HearingEvent::SpeechStarted => {
                state
                    .controller
                    .record_runtime_packet(RuntimePacket::UserStartedSpeaking);
            }
            HearingEvent::BreathGroupClosed { .. } => {
                state
                    .controller
                    .record_runtime_packet(RuntimePacket::UserStoppedSpeaking);
                state.controller.apply_safe_boundary_updates();
            }
            HearingEvent::SpeechContinued { .. }
            | HearingEvent::PauseStarted
            | HearingEvent::BreathGroupOpened { .. } => {}
        }
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
    state.frame_time_ms = state.frame_time_ms.saturating_add(frame_duration_ms);
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
fn stream_speech_to_tts(
    llm: &mut LlamaCppEngine,
    tts: &mut impl TextToSpeech,
    transcript: &str,
    model_profile: ModelProfile,
    no_backchannels: bool,
    self_hearing: &mut listenbury::SelfHearingState,
    controller: &mut ConversationController,
    user_turn_id: u64,
) -> Result<()> {
    controller.turn_tracker.on_pete_thinking_started();
    let generation_id = llm
        .start(GenerationRequest {
            prompt: build_prompt(transcript),
            max_tokens: Some(max_tokens(model_profile)),
            stop: live_half_duplex_stops(),
        })
        .context("failed to start llama.cpp generation")?;

    let llm_started_at_ms = unix_nanos_to_millis(ExactTimestamp::now().unix_nanos);
    let llm_started_at = Instant::now();
    eprintln!(
        "[live-half-duplex] controller turn state after llm start: {:?}",
        controller.turn_tracker.state()
    );
    let mut current_spoken_text = String::new();
    let mut main_llm_has_emitted_token = false;
    let mut main_llm_has_safe_speech_unit = false;
    let mut filler_attempted = false;
    let mut played_any_audio = false;
    loop {
        let events = llm.poll(generation_id)?;
        if events.is_empty() {
            if !filler_attempted
                && !main_llm_has_emitted_token
                && llm_started_at.elapsed() >= Duration::from_millis(FILLER_SILENCE_DURATION_MS)
            {
                let now_ms = unix_nanos_to_millis(ExactTimestamp::now().unix_nanos);
                filler_attempted = true;
                if let Some(filler_plan) = maybe_plan_cached_backchannel(
                    controller,
                    transcript,
                    no_backchannels,
                    user_turn_id,
                    llm_started_at_ms,
                    now_ms,
                    main_llm_has_emitted_token,
                    main_llm_has_safe_speech_unit,
                ) {
                    eprintln!(
                        "[live-half-duplex] controller filler decision: speaking backchannel {:?}",
                        filler_plan.unit()
                    );
                    let filler_text = filler_plan.text().to_string();
                    current_spoken_text = filler_text.clone();
                    tts.enqueue(filler_plan)?;
                    controller.record_runtime_packet(RuntimePacket::SpeechUnitCommitted {
                        text: filler_text,
                    });
                    controller.apply_safe_boundary_updates();
                }
            }
            played_any_audio |= drain_ready_tts_audio(
                tts,
                &current_spoken_text,
                self_hearing,
                "live-half-duplex response",
                controller,
            )?;
            std::thread::sleep(Duration::from_millis(5));
            continue;
        }

        for event in &events {
            if let LlmEvent::Error { message } = event {
                anyhow::bail!("llama.cpp generation failed: {message}");
            }
        }
        if events
            .iter()
            .any(|event| matches!(event, LlmEvent::Token { .. }))
        {
            main_llm_has_emitted_token = true;
        }
        for unit in planner_units_from_events(controller, &events, no_backchannels) {
            match unit {
                ExpressiveUnit::Speech(plan) => {
                    let text = plan.text().to_string();
                    current_spoken_text = text.clone();
                    main_llm_has_safe_speech_unit = true;
                    tts.enqueue(plan)?;
                    controller.record_runtime_packet(RuntimePacket::SpeechUnitCommitted { text });
                    controller.apply_safe_boundary_updates();
                }
                ExpressiveUnit::Face(command) => {
                    eprintln!("[live-half-duplex] face event: {command:?}");
                    let emoji = match command {
                        FaceCommand::SetEmoji(emoji) => emoji,
                        FaceCommand::Clear => String::new(),
                    };
                    controller.record_runtime_packet(RuntimePacket::FaceChanged { emoji });
                    controller.apply_safe_boundary_updates();
                }
            }
        }
        played_any_audio |= drain_ready_tts_audio(
            tts,
            &current_spoken_text,
            self_hearing,
            "live-half-duplex response",
            controller,
        )?;

        if events.iter().any(is_terminal_llm_event) {
            break;
        }
    }

    let flushed_audio = flush_tts_audio(
        tts,
        &current_spoken_text,
        self_hearing,
        "live-half-duplex response",
        Duration::from_secs(30),
        played_any_audio,
        controller,
    )?;
    played_any_audio |= flushed_audio;
    if !played_any_audio {
        current_spoken_text = "I heard you, but I lost my words.".to_string();
        tts.enqueue(SpeechPlan::from(SpeechUnit::FullTurn(
            current_spoken_text.clone(),
        )))?;
        let played_fallback = flush_tts_audio(
            tts,
            &current_spoken_text,
            self_hearing,
            "live-half-duplex response fallback",
            Duration::from_secs(30),
            false,
            controller,
        )?;
        anyhow::ensure!(
            played_fallback,
            "Piper produced no audio frames before timeout"
        );
    }
    self_hearing.mark_output_finished();
    controller.on_pete_speech_finished();
    eprintln!(
        "[self-hearing] playback finished; tail window active until unix_ns={:?}",
        self_hearing.output_expected_until.map(|t| t.unix_nanos)
    );
    Ok(())
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
fn planner_units_from_events(
    controller: &mut ConversationController,
    events: &[LlmEvent],
    no_backchannels: bool,
) -> Vec<ExpressiveUnit> {
    controller
        .ingest_llm_events(events)
        .into_iter()
        .filter_map(|unit| match unit {
            ExpressiveUnit::Speech(plan)
                if no_backchannels && matches!(plan.unit(), SpeechUnit::Backchannel(_)) =>
            {
                None
            }
            ExpressiveUnit::Speech(plan) if is_thinking_leak(plan.text()) => None,
            _ => Some(unit),
        })
        .collect()
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
fn is_thinking_leak(text: &str) -> bool {
    let text = text
        .trim()
        .trim_matches(['"', '\'', '`'])
        .trim_start_matches(|ch: char| ch == '-' || ch.is_whitespace())
        .to_ascii_lowercase();

    [
        "<think>",
        "assistant should",
        "the assistant should",
        "the assistant's response",
        "the user asks",
        "the user asked",
        "user asks",
        "user asked",
        "the instructions",
        "instructions:",
        "we should respond",
        "we should produce",
        "we have to output",
        "we need to",
        "need to answer",
        "write only the words",
        "let's craft",
        "short reply:",
        "or we can do",
    ]
    .iter()
    .any(|prefix| text.starts_with(prefix))
        || text.contains(" the instructions:")
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
fn maybe_plan_cached_backchannel(
    controller: &mut ConversationController,
    transcript: &str,
    no_backchannels: bool,
    user_turn_id: u64,
    llm_started_at_ms: u64,
    now_ms: u64,
    main_llm_has_emitted_token: bool,
    main_llm_has_safe_speech_unit: bool,
) -> Option<SpeechPlan> {
    if no_backchannels {
        return None;
    }
    let ctx = FillerContext {
        turn_state: controller.turn_tracker.state(),
        transcript_so_far: Some(transcript.to_string()),
        vad_confidence: 0.0,
        silence_duration_ms: now_ms.saturating_sub(llm_started_at_ms),
        main_llm_started_at_ms: Some(llm_started_at_ms),
        main_llm_has_emitted_token,
        main_llm_has_safe_speech_unit,
        user_interrupted_recently: false,
        now_ms,
        user_turn_id: Some(user_turn_id),
    };
    match controller.decide_filler_command(&ctx) {
        Some(MouthCommand::Speak(plan)) => Some(plan),
        Some(MouthCommand::FadeOut { .. }) | Some(MouthCommand::StopNow) | None => None,
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn drain_ready_tts_audio(
    tts: &mut impl TextToSpeech,
    spoken_text: &str,
    self_hearing: &mut listenbury::SelfHearingState,
    source: &str,
    controller: &mut ConversationController,
) -> Result<bool> {
    let frames = tts.poll_audio()?;
    if frames.is_empty() {
        return Ok(false);
    }
    let audio_dur = tts_audio_duration(&frames);
    controller.on_pete_speech_started();
    controller.record_runtime_packet(RuntimePacket::TtsQueueChanged {
        queued_ms: u64::try_from(audio_dur.as_millis()).unwrap_or(u64::MAX),
    });
    controller.apply_safe_boundary_updates();
    self_hearing.mark_output_started(spoken_text, audio_dur);
    eprintln!(
        "[self-hearing] suppression window opened: utterance={:?} duration={audio_dur:?}",
        self_hearing.current_utterance_text.as_deref().unwrap_or("")
    );
    play_audio_frames(&frames, source)?;
    Ok(true)
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn flush_tts_audio(
    tts: &mut impl TextToSpeech,
    spoken_text: &str,
    self_hearing: &mut listenbury::SelfHearingState,
    source: &str,
    timeout: Duration,
    prior_audio_played: bool,
    controller: &mut ConversationController,
) -> Result<bool> {
    let quiet_after_audio = Duration::from_millis(AUDIO_DRAIN_QUIET_THRESHOLD_MS);
    let post_playback_grace = Duration::from_millis(POST_PLAYBACK_TTS_GRACE_MS);
    let deadline = Instant::now() + timeout;
    let mut played_any_audio = false;
    let mut last_audio_at = prior_audio_played.then(Instant::now);

    while Instant::now() < deadline {
        if drain_ready_tts_audio(tts, spoken_text, self_hearing, source, controller)? {
            played_any_audio = true;
            last_audio_at = Some(Instant::now());
            continue;
        }
        if let Some(last_audio_at) = last_audio_at {
            let quiet_threshold = if played_any_audio {
                quiet_after_audio
            } else {
                post_playback_grace
            };
            if Instant::now().duration_since(last_audio_at) >= quiet_threshold {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    Ok(played_any_audio)
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn unix_nanos_to_millis(unix_nanos: u128) -> u64 {
    u64::try_from(unix_nanos / NANOS_PER_MILLI).unwrap_or(u64::MAX)
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn build_prompt(transcript: &str) -> String {
    format!(
        "<|system|>\nYou are Pete, speaking aloud through a TTS system.\nWrite one assistant turn only.\nDo not prethink, reason aloud, or describe what you are about to do.\nRespond only with the exact text Pete should speak.\nDo not mention the assistant, the user, instructions, reasoning, drafting, possible replies, or quoted prompt text.\nWrite in short, complete spoken sentences.\nDo not rely on long subordinate clauses.\nPrefer natural sentence boundaries.\nEach sentence should be speakable on its own.</s>\n<|user|>\n{transcript}</s>\n<|assistant|>\n"
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
fn live_half_duplex_stops() -> Vec<String> {
    vec![
        "</s>".to_string(),
        "\n<|user|>".to_string(),
        "\n<|assistant|>".to_string(),
        "\n<|system|>".to_string(),
        "<|user|>".to_string(),
        "<|assistant|>".to_string(),
        "<|system|>".to_string(),
        "\nUser:".to_string(),
        "\nPete:".to_string(),
        "\nAssistant:".to_string(),
    ]
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
    frame_sample_rate_hz: u32,
    frame_channels: u16,
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
        if ring_tx.try_push(frame).is_err() {
            dropped_in_ring.fetch_add(1, Ordering::Relaxed);
        }
    }
}

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
fn tts_audio_duration(frames: &[AudioFrame]) -> Duration {
    let Some(first) = frames.first() else {
        return Duration::ZERO;
    };
    let channels = usize::from(first.channels).max(1);
    let sample_rate = first.sample_rate_hz;
    if sample_rate == 0 {
        return Duration::ZERO;
    }
    let total_samples: usize = frames.iter().map(|f| f.samples.len()).sum();
    let samples_per_channel = total_samples / channels;
    Duration::from_secs_f64(samples_per_channel as f64 / f64::from(sample_rate))
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

#[cfg(test)]
mod tests {
    use super::{
        convert_frame_samples, live_half_duplex_stops, maybe_plan_cached_backchannel,
        planner_units_from_events, vad_frame_format,
    };
    use listenbury::hearing::vad::VadBackendKind;
    use listenbury::mind::llm::LlmEvent;
    use listenbury::mouth::planner::{ExpressiveUnit, SpeechUnit};
    use listenbury::{ConversationController, RuntimePacket};

    fn token(text: &str) -> LlmEvent {
        LlmEvent::Token {
            text: text.to_string(),
        }
    }

    #[test]
    fn planner_units_emit_speech_before_completed_event() {
        let mut controller = ConversationController::default();
        let emitted_before_completed =
            planner_units_from_events(&mut controller, &[token("I think that works.")], false);
        assert!(matches!(
            emitted_before_completed.first(),
            Some(ExpressiveUnit::Speech(_))
        ));

        let emitted_on_completed =
            planner_units_from_events(&mut controller, &[LlmEvent::Completed], false);
        assert!(emitted_on_completed.is_empty());
    }

    #[test]
    fn planner_units_still_filter_backchannels() {
        let mut controller = ConversationController::default();
        let without_filter = planner_units_from_events(
            &mut controller,
            &[token("Okay. This should still be spoken.")],
            false,
        );
        assert!(without_filter.iter().any(|unit| matches!(
            unit,
            ExpressiveUnit::Speech(plan) if matches!(plan.unit(), SpeechUnit::Backchannel(_))
        )));

        let mut controller = ConversationController::default();
        let with_filter = planner_units_from_events(
            &mut controller,
            &[token("Okay. This should still be spoken.")],
            true,
        );
        assert!(with_filter.iter().all(|unit| !matches!(
            unit,
            ExpressiveUnit::Speech(plan) if matches!(plan.unit(), SpeechUnit::Backchannel(_))
        )));
    }

    #[test]
    fn planner_units_drop_thinking_leaks() {
        let mut controller = ConversationController::default();
        let units = planner_units_from_events(
            &mut controller,
            &[
                token("<thought>this should be a thought</thought> "),
                token("<thinking>Or is it thinking</thinking> "),
                token("Yes, I can hear you."),
            ],
            false,
        );

        assert_eq!(units.len(), 1);
        assert!(matches!(
            units.first(),
            Some(ExpressiveUnit::Speech(plan)) if plan.text() == "Yes, I can hear you."
        ));
    }

    #[test]
    fn planner_units_drop_preamble_leaks() {
        let mut controller = ConversationController::default();
        let units = planner_units_from_events(
            &mut controller,
            &[
                token("We have to output Pete's spoken response. "),
                token("\"Write only the words Pete should say aloud.\" "),
                token("Yes, I can hear you."),
            ],
            false,
        );

        assert_eq!(units.len(), 1);
        assert!(matches!(
            units.first(),
            Some(ExpressiveUnit::Speech(plan)) if plan.text() == "Yes, I can hear you."
        ));
    }

    #[test]
    fn planner_units_preserve_face_event_order() {
        let mut controller = ConversationController::default();
        let units = planner_units_from_events(&mut controller, &[token("Okay 🙂 I see.")], false);
        assert!(matches!(units.first(), Some(ExpressiveUnit::Speech(_))));
        assert!(matches!(units.get(1), Some(ExpressiveUnit::Face(_))));
        assert!(matches!(units.get(2), Some(ExpressiveUnit::Speech(_))));
    }

    #[test]
    fn live_half_duplex_stops_at_chat_boundaries() {
        let stops = live_half_duplex_stops();
        assert!(stops.iter().any(|stop| stop == "</s>"));
        assert!(stops.iter().any(|stop| stop == "\n<|user|>"));
        assert!(stops.iter().any(|stop| stop == "\n<|assistant|>"));
        assert!(stops.iter().any(|stop| stop == "\nUser:"));
    }

    #[test]
    fn webrtc_vad_frames_use_supported_mono_rate() {
        assert_eq!(
            vad_frame_format(VadBackendKind::WebRtc, 44_100, 2),
            (16_000, 1)
        );
        assert_eq!(
            vad_frame_format(VadBackendKind::Energy, 44_100, 2),
            (44_100, 2)
        );
    }

    #[test]
    fn webrtc_conversion_turns_44100_stereo_10ms_into_16000_mono_10ms() {
        let input = vec![1.0; 882];
        let converted = convert_frame_samples(&input, 44_100, 2, 16_000, 1);

        assert_eq!(converted.len(), 160);
        assert!(
            converted
                .iter()
                .all(|sample| (*sample - 1.0).abs() < 0.0001)
        );
    }

    #[test]
    fn filler_planning_can_emit_cached_backchannel_before_safe_speech() {
        let mut controller = ConversationController::default();
        controller.turn_tracker.on_pete_thinking_started();

        let first = maybe_plan_cached_backchannel(
            &mut controller,
            "Can you explain this?",
            false,
            42,
            10_000,
            11_200,
            false,
            false,
        );
        assert!(matches!(
            first.as_ref().map(|plan| plan.unit()),
            Some(SpeechUnit::Backchannel(text)) if text == "Let me think."
        ));

        if let Some(plan) = first {
            controller.record_runtime_packet(RuntimePacket::SpeechUnitCommitted {
                text: plan.text().to_string(),
            });
            controller.apply_safe_boundary_updates();
        }
        assert!(
            controller
                .runtime_context()
                .iter()
                .any(|packet| matches!(packet, RuntimePacket::BackchannelPlayed { .. }))
        );

        let second = maybe_plan_cached_backchannel(
            &mut controller,
            "Can you explain this?",
            false,
            42,
            10_100,
            11_300,
            false,
            false,
        );
        assert!(second.is_none());
    }

    #[test]
    fn filler_planning_stays_silent_before_delay_or_after_tokens() {
        let mut controller = ConversationController::default();
        controller.turn_tracker.on_pete_thinking_started();

        let too_early = maybe_plan_cached_backchannel(
            &mut controller,
            "Can you explain this?",
            false,
            42,
            10_000,
            10_500,
            false,
            false,
        );
        assert!(too_early.is_none());

        let after_token = maybe_plan_cached_backchannel(
            &mut controller,
            "Can you explain this?",
            false,
            43,
            20_000,
            21_300,
            true,
            false,
        );
        assert!(after_token.is_none());
    }
}
