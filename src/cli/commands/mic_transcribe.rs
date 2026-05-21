use crate::cli::MicTranscribeCommand;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use crate::cli::model_paths::{resolve_refine_whisper_model, resolve_whisper_model};
use anyhow::Result;

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use anyhow::Context;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use cpal::{FromSample, Sample, SizedSample};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::audio::capture::{
    boost_current_thread_for_capture, callback_sample_queue_capacity,
};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::audio::ring::make_audio_ring;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::event::HearingEvent;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::hearing::breath::{
    BreathGroupConfig, BreathGroupId, BreathGroupSegmenter, DEFAULT_VAD_FRAME_MS,
};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::hearing::vad::{VoiceActivityDetector, create_vad_backend};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::live_trace::{LiveTraceRecorder, SseBroadcaster};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::speech::recognizer::SpeechRecognizer;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::speech::transcript::TranscriptCandidateEvent;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::word::{
    TimedWordStream, TranscriptWord, WordStreamId, WordStreamSource,
    transcript_to_energy_snapped_word_stream, transcript_to_word_stream,
};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::{AudioFrame, ExactTimestamp, WhisperSpeechRecognizer};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use serde_json::json;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use std::collections::{HashMap, VecDeque};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use std::time::{Duration, Instant};

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
const AUDIO_RING_CAPACITY: usize = 256;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
const WHISPER_SAMPLE_RATE_HZ: u32 = 16_000;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
const WHISPER_FRAME_SAMPLES: usize = 160;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
const WEBRTC_VAD_SAMPLE_RATE_HZ: u32 = 16_000;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
const MONO_CHANNELS: u16 = 1;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
const WEB_TRANSCRIBE_PROSPECTIVE_INITIAL_MS: u64 = 300;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
const WEB_TRANSCRIBE_PROSPECTIVE_INTERVAL_MS: u64 = 250;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
const WEB_TRANSCRIBE_BREATH_GROUP_SILENCE_MS: u64 = 350;

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
struct MicTranscribeState {
    vad: Box<dyn VoiceActivityDetector>,
    segmenter: BreathGroupSegmenter,
    active_groups: HashMap<BreathGroupId, Vec<AudioFrame>>,
    frame_time_ms: u64,
    last_vad_state: Option<bool>,
    groups_closed: usize,
    transcripts_emitted: usize,
    recognizer: WhisperSpeechRecognizer,
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
struct WebTranscribeState {
    vad: Box<dyn VoiceActivityDetector>,
    segmenter: BreathGroupSegmenter,
    active_groups: HashMap<BreathGroupId, ActiveWebTranscribeGroup>,
    frame_time_ms: u64,
    last_vad_state: Option<bool>,
    groups_closed: usize,
    transcripts_emitted: usize,
    recognizer: WhisperSpeechRecognizer,
    candidate_planner: WebTranscriptSpeculativePlanner,
    live_trace: LiveTraceRecorder<SseBroadcaster>,
    live_trace_turn: u64,
    live_audio: listenbury::web::LiveSessionAudioStore,
    next_stream_id: u64,
    finalized_segments: VecDeque<FinalizedAsrSegment>,
    finalized_segments_duration_ms: u64,
    max_finalized_segments_duration_ms: u64,
    next_refine_at: Instant,
    refine_interval: Duration,
    refine_tx: crossbeam_channel::Sender<RefinementWorkItem>,
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
struct ActiveWebTranscribeGroup {
    frames: Vec<AudioFrame>,
    opened_at_ms: u64,
    next_prospective_at_ms: u64,
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
#[derive(Debug, Clone)]
struct FinalizedAsrSegment {
    frames: Vec<AudioFrame>,
    duration_ms: u64,
    text: String,
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
impl ActiveWebTranscribeGroup {
    fn new(opened_at_ms: u64) -> Self {
        Self {
            frames: Vec::new(),
            opened_at_ms,
            next_prospective_at_ms: opened_at_ms
                .saturating_add(WEB_TRANSCRIBE_PROSPECTIVE_INITIAL_MS),
        }
    }
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
#[derive(Debug, Clone)]
struct RefinementWorkItem {
    frames: Vec<AudioFrame>,
    observed_at: ExactTimestamp,
    segment_count: usize,
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
#[derive(Debug, Clone)]
struct WebTranscriptStabilityState {
    candidate_id: listenbury::speech::transcript::TranscriptCandidateId,
    stable_text: String,
    unstable_text: String,
    confidence: Option<f32>,
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
#[derive(Debug, Default)]
struct WebTranscriptSpeculativePlanner {
    active_candidate: Option<listenbury::speech::transcript::TranscriptCandidateId>,
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
impl WebTranscriptSpeculativePlanner {
    fn observe(&mut self, event: &TranscriptCandidateEvent) -> Option<WebTranscriptStabilityState> {
        match event {
            TranscriptCandidateEvent::CandidateStarted { id } => {
                self.active_candidate = Some(*id);
                None
            }
            TranscriptCandidateEvent::CandidateUpdated {
                id,
                text,
                stable_prefix_len,
                confidence,
            } => {
                self.active_candidate = Some(*id);
                Some(WebTranscriptStabilityState::from_parts(
                    *id,
                    text,
                    *stable_prefix_len,
                    *confidence,
                ))
            }
            TranscriptCandidateEvent::CandidateReplaced { new, .. } => {
                self.active_candidate = Some(*new);
                None
            }
            TranscriptCandidateEvent::CandidateFinalized {
                id,
                text,
                confidence,
            } => {
                if self.active_candidate == Some(*id) {
                    self.active_candidate = None;
                }
                Some(WebTranscriptStabilityState::from_parts(
                    *id,
                    text,
                    text.len(),
                    *confidence,
                ))
            }
            TranscriptCandidateEvent::CandidateCancelled { id } => {
                if self.active_candidate == Some(*id) {
                    self.active_candidate = None;
                }
                None
            }
        }
    }
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
impl WebTranscriptStabilityState {
    fn from_parts(
        candidate_id: listenbury::speech::transcript::TranscriptCandidateId,
        text: &str,
        stable_prefix_len: usize,
        confidence: Option<f32>,
    ) -> Self {
        let split = stable_prefix_len.min(text.len());
        let split = if text.is_char_boundary(split) {
            split
        } else {
            text.char_indices()
                .find_map(|(idx, ch)| {
                    let end = idx + ch.len_utf8();
                    (end >= split).then_some(end)
                })
                .unwrap_or(text.len())
        };
        let (stable_text, unstable_text) = text.split_at(split);
        Self {
            candidate_id,
            stable_text: stable_text.to_string(),
            unstable_text: unstable_text.to_string(),
            confidence,
        }
    }
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
pub(crate) fn run_mic_transcribe(command: MicTranscribeCommand) -> Result<()> {
    if command.web {
        return run_web_mic_transcribe(command);
    }

    if !command.until_ctrl_c {
        anyhow::ensure!(
            command.seconds > 0,
            "--seconds must be greater than zero unless --until-ctrl-c is set"
        );
    }

    let model_path = resolve_whisper_model(command.whisper_model)?;
    let recognizer = WhisperSpeechRecognizer::new(&model_path)
        .with_context(|| format!("failed to load Whisper model at {}", model_path.display()))?;
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

    let stop_requested = Arc::new(AtomicBool::new(false));
    ctrlc::set_handler({
        let stop_requested = Arc::clone(&stop_requested);
        move || {
            stop_requested.store(true, Ordering::SeqCst);
        }
    })
    .context("failed to register Ctrl-C handler")?;

    let sample_capacity = callback_sample_queue_capacity(input_sample_rate_hz, input_channels);
    let (sample_tx, sample_rx) = crossbeam_channel::bounded::<f32>(sample_capacity);
    let dropped_in_callback = Arc::new(AtomicUsize::new(0));
    let dropped_in_ring = Arc::new(AtomicUsize::new(0));
    let err_fn = |err| eprintln!("input stream error: {err}");

    let stream = match supported_config.sample_format() {
        cpal::SampleFormat::F32 => build_input_stream::<f32>(
            &device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            err_fn,
        )?,
        cpal::SampleFormat::F64 => build_input_stream::<f64>(
            &device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            err_fn,
        )?,
        cpal::SampleFormat::I8 => build_input_stream::<i8>(
            &device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            err_fn,
        )?,
        cpal::SampleFormat::I16 => build_input_stream::<i16>(
            &device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            err_fn,
        )?,
        cpal::SampleFormat::I32 => build_input_stream::<i32>(
            &device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            err_fn,
        )?,
        cpal::SampleFormat::I64 => build_input_stream::<i64>(
            &device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            err_fn,
        )?,
        cpal::SampleFormat::U8 => build_input_stream::<u8>(
            &device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            err_fn,
        )?,
        cpal::SampleFormat::U16 => build_input_stream::<u16>(
            &device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            err_fn,
        )?,
        cpal::SampleFormat::U32 => build_input_stream::<u32>(
            &device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            err_fn,
        )?,
        cpal::SampleFormat::U64 => build_input_stream::<u64>(
            &device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            err_fn,
        )?,
        sample_format => anyhow::bail!("unsupported input sample format: {sample_format:?}"),
    };

    stream
        .play()
        .with_context(|| format!("failed to start capture from {device_name}"))?;

    println!(
        "mic-transcribe listening on {device_name}: {} Hz, {} channel(s), vad={}, sample_queue={}. Press Ctrl-C to stop.",
        input_sample_rate_hz,
        input_channels,
        command.vad.as_backend_kind().as_str(),
        sample_capacity
    );
    boost_current_thread_for_capture("mic-transcribe");

    let stop_deadline = if command.until_ctrl_c {
        None
    } else {
        Some(Instant::now() + Duration::from_secs(command.seconds))
    };
    let input_frame_samples =
        frame_samples_per_callback_frame(input_sample_rate_hz, input_channels);
    let (mut ring_tx, mut ring_rx) = make_audio_ring(AUDIO_RING_CAPACITY);
    let mut pending = VecDeque::<f32>::new();
    let vad_backend = command.vad.as_backend_kind();
    let (frame_sample_rate_hz, frame_channels) =
        vad_frame_format(vad_backend, input_sample_rate_hz, input_channels);
    let mut state = MicTranscribeState {
        vad: create_vad_backend(vad_backend)?,
        segmenter: BreathGroupSegmenter::default(),
        active_groups: HashMap::new(),
        frame_time_ms: 0,
        last_vad_state: None,
        groups_closed: 0,
        transcripts_emitted: 0,
        recognizer,
    };

    loop {
        if stop_requested.load(Ordering::SeqCst) {
            println!("received Ctrl-C, stopping capture...");
            break;
        }
        if let Some(deadline) = stop_deadline
            && Instant::now() >= deadline
        {
            println!(
                "capture timeout reached ({}s), stopping...",
                command.seconds
            );
            break;
        }

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
        process_ring_frames(&mut ring_rx, &mut state)?;
    }

    drop(stream);

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
    process_ring_frames(&mut ring_rx, &mut state)?;

    if !state.active_groups.is_empty() {
        println!(
            "forcing transcription of {} active breath group(s) on shutdown",
            state.active_groups.len()
        );
    }
    for (id, frames) in state.active_groups.drain() {
        state.groups_closed += 1;
        println!("breath-group forced-close id={id:?} reason=shutdown");
        let text = transcribe_group(&frames, &mut state.recognizer)?.text;
        if text.is_empty() {
            println!("transcript group={} text=<empty>", state.groups_closed);
        } else {
            state.transcripts_emitted += 1;
            println!("transcript group={} text={}", state.groups_closed, text);
        }
    }

    let callback_drops = dropped_in_callback.load(Ordering::Relaxed);
    let ring_drops = dropped_in_ring.load(Ordering::Relaxed);
    println!(
        "mic-transcribe finished: closed_groups={}, non_empty_transcripts={}, callback_drops={}, ring_drops={}",
        state.groups_closed, state.transcripts_emitted, callback_drops, ring_drops
    );

    Ok(())
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn run_web_mic_transcribe(command: MicTranscribeCommand) -> Result<()> {
    anyhow::ensure!(
        command.refine_window_seconds > 0,
        "--refine-window-seconds must be greater than zero"
    );
    anyhow::ensure!(
        command.refine_interval_ms > 0,
        "--refine-interval-ms must be greater than zero"
    );

    let fast_model_path = resolve_whisper_model(command.whisper_model)?;
    let refine_model_path = resolve_refine_whisper_model(command.refine_whisper_model)?;
    let recognizer = WhisperSpeechRecognizer::new_quiet(&fast_model_path).with_context(|| {
        format!(
            "failed to load Whisper model at {}",
            fast_model_path.display()
        )
    })?;

    let broadcaster = SseBroadcaster::new();
    let server_broadcaster = broadcaster.clone();
    let live_audio = listenbury::web::LiveSessionAudioStore::new();
    let server_live_audio = live_audio.clone();
    let live_visual_speech = listenbury::web::LiveSessionVisualSpeechStore::new();
    let capture_enabled = Arc::new(AtomicBool::new(true));
    let (browser_audio_tx, browser_audio_rx) = crossbeam_channel::bounded::<AudioFrame>(128);
    let bind_host = command.web_host.clone();
    let server = listenbury::web::bind(listenbury::web::ServeConfig {
        host: bind_host.clone(),
        port: command.web_port,
        payload: None,
        trace: None,
        broadcaster: Some(server_broadcaster),
        live_audio: Some(server_live_audio),
        live_visual_speech: Some(live_visual_speech),
        input_control: listenbury::web::WebInputControl::new(
            Some(Arc::clone(&capture_enabled)),
            Some(browser_audio_tx),
        ),
    })
    .context("failed to start embedded transcription web viewer")?;
    let web_port = server.local_addr().port();
    let browser_host = browser_host_for_bind_host(&bind_host);
    let screenplay_url = format!("http://{}:{}/screenplay", browser_host, web_port);
    let viewer_url = format!("http://{}:{}/", browser_host, web_port);
    std::thread::spawn(move || {
        if let Err(error) = server.serve() {
            eprintln!("embedded web server error: {error:#}");
        }
    });
    println!("Listenbury transcription screenplay available at {screenplay_url}");
    println!("WaveDeck live session view available at {viewer_url}");

    let trace_started_at = ExactTimestamp::now();
    let mut live_trace = LiveTraceRecorder::new(trace_started_at, broadcaster.clone());
    live_trace.emit_now(0, "capture_started", ExactTimestamp::now())?;

    let (refine_tx, refine_rx) = crossbeam_channel::bounded::<RefinementWorkItem>(1);
    let refine_broadcaster = broadcaster.clone();
    std::thread::Builder::new()
        .name("listenbury-transcribe-refiner".to_string())
        .spawn(move || {
            run_refinement_worker(
                refine_model_path,
                refine_rx,
                refine_broadcaster,
                trace_started_at,
            );
        })
        .context("failed to spawn transcription refinement worker")?;

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

    let stop_requested = Arc::new(AtomicBool::new(false));
    ctrlc::set_handler({
        let stop_requested = Arc::clone(&stop_requested);
        move || {
            stop_requested.store(true, Ordering::SeqCst);
        }
    })
    .context("failed to register Ctrl-C handler")?;

    let sample_capacity = callback_sample_queue_capacity(input_sample_rate_hz, input_channels);
    let (sample_tx, sample_rx) = crossbeam_channel::bounded::<f32>(sample_capacity);
    let dropped_in_callback = Arc::new(AtomicUsize::new(0));
    let dropped_in_ring = Arc::new(AtomicUsize::new(0));
    let err_fn = |err| eprintln!("input stream error: {err}");
    let stream = match supported_config.sample_format() {
        cpal::SampleFormat::F32 => build_input_stream_with_capture_control::<f32>(
            &device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            Some(Arc::clone(&capture_enabled)),
            err_fn,
        )?,
        cpal::SampleFormat::F64 => build_input_stream_with_capture_control::<f64>(
            &device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            Some(Arc::clone(&capture_enabled)),
            err_fn,
        )?,
        cpal::SampleFormat::I8 => build_input_stream_with_capture_control::<i8>(
            &device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            Some(Arc::clone(&capture_enabled)),
            err_fn,
        )?,
        cpal::SampleFormat::I16 => build_input_stream_with_capture_control::<i16>(
            &device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            Some(Arc::clone(&capture_enabled)),
            err_fn,
        )?,
        cpal::SampleFormat::I32 => build_input_stream_with_capture_control::<i32>(
            &device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            Some(Arc::clone(&capture_enabled)),
            err_fn,
        )?,
        cpal::SampleFormat::I64 => build_input_stream_with_capture_control::<i64>(
            &device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            Some(Arc::clone(&capture_enabled)),
            err_fn,
        )?,
        cpal::SampleFormat::U8 => build_input_stream_with_capture_control::<u8>(
            &device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            Some(Arc::clone(&capture_enabled)),
            err_fn,
        )?,
        cpal::SampleFormat::U16 => build_input_stream_with_capture_control::<u16>(
            &device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            Some(Arc::clone(&capture_enabled)),
            err_fn,
        )?,
        cpal::SampleFormat::U32 => build_input_stream_with_capture_control::<u32>(
            &device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            Some(Arc::clone(&capture_enabled)),
            err_fn,
        )?,
        cpal::SampleFormat::U64 => build_input_stream_with_capture_control::<u64>(
            &device,
            &stream_config,
            sample_tx.clone(),
            Arc::clone(&dropped_in_callback),
            Some(Arc::clone(&capture_enabled)),
            err_fn,
        )?,
        sample_format => anyhow::bail!("unsupported input sample format: {sample_format:?}"),
    };
    stream
        .play()
        .with_context(|| format!("failed to start capture from {device_name}"))?;

    let vad_backend = command.vad.as_backend_kind();
    println!(
        "transcribe --web listening on {device_name}: {} Hz, {} channel(s), vad={}, sample_queue={}. Press Ctrl-C to stop.",
        input_sample_rate_hz,
        input_channels,
        vad_backend.as_str(),
        sample_capacity
    );
    boost_current_thread_for_capture("transcribe --web");
    let mut started = live_trace.event(0, "listening_started", ExactTimestamp::now());
    started.text = Some(format!(
        "device={device_name:?} sample_rate_hz={input_sample_rate_hz} channels={input_channels} vad={}",
        vad_backend.as_str()
    ));
    live_trace.emit(started)?;

    let input_frame_samples =
        frame_samples_per_callback_frame(input_sample_rate_hz, input_channels);
    let (mut ring_tx, mut ring_rx) = make_audio_ring(AUDIO_RING_CAPACITY);
    let mut pending = VecDeque::<f32>::new();
    let (frame_sample_rate_hz, frame_channels) =
        vad_frame_format(vad_backend, input_sample_rate_hz, input_channels);
    let mut state = WebTranscribeState {
        vad: create_vad_backend(vad_backend)?,
        segmenter: BreathGroupSegmenter::new(web_transcribe_breath_group_config()),
        active_groups: HashMap::new(),
        frame_time_ms: 0,
        last_vad_state: None,
        groups_closed: 0,
        transcripts_emitted: 0,
        recognizer,
        candidate_planner: WebTranscriptSpeculativePlanner::default(),
        live_trace,
        live_trace_turn: 0,
        live_audio,
        next_stream_id: 1,
        finalized_segments: VecDeque::new(),
        finalized_segments_duration_ms: 0,
        max_finalized_segments_duration_ms: command.refine_window_seconds.saturating_mul(1_000),
        next_refine_at: Instant::now(),
        refine_interval: Duration::from_millis(command.refine_interval_ms),
        refine_tx,
    };

    loop {
        if stop_requested.load(Ordering::SeqCst) {
            println!("received Ctrl-C, stopping capture...");
            break;
        }

        drain_browser_audio_into_ring(&browser_audio_rx, &mut ring_tx, &dropped_in_ring);
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
        process_web_ring_frames(&mut ring_rx, &mut state)?;
    }

    drop(stream);

    drain_browser_audio_into_ring(&browser_audio_rx, &mut ring_tx, &dropped_in_ring);
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
    process_web_ring_frames(&mut ring_rx, &mut state)?;

    let forced_groups = state
        .active_groups
        .drain()
        .map(|(_, group)| group)
        .collect::<Vec<_>>();
    for group in forced_groups {
        state.groups_closed += 1;
        let output = transcribe_group(&group.frames, &mut state.recognizer)?;
        emit_web_transcribe_output(
            &mut state,
            output,
            &group.frames,
            group.opened_at_ms,
            true,
            ExactTimestamp::now(),
        )?;
    }

    let callback_drops = dropped_in_callback.load(Ordering::Relaxed);
    let ring_drops = dropped_in_ring.load(Ordering::Relaxed);
    println!(
        "transcribe --web finished: closed_groups={}, non_empty_transcripts={}, callback_drops={}, ring_drops={}",
        state.groups_closed, state.transcripts_emitted, callback_drops, ring_drops
    );

    Ok(())
}

#[cfg(not(all(feature = "asr-whisper", feature = "audio-cpal")))]
pub(crate) fn run_mic_transcribe(_command: MicTranscribeCommand) -> Result<()> {
    anyhow::bail!("listenbury mic-transcribe requires the `audio-cpal` and `asr-whisper` features")
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn process_live_frame(frame: AudioFrame, state: &mut MicTranscribeState) -> Result<()> {
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
    for event in &events {
        if let HearingEvent::BreathGroupOpened { id } = event {
            println!("breath-group open id={id:?} t_ms={}", state.frame_time_ms);
            state.active_groups.entry(*id).or_default();
        }
    }

    for group in state.active_groups.values_mut() {
        group.push(frame.clone());
    }

    for event in events {
        if let HearingEvent::BreathGroupClosed { id, reason } = event {
            state.groups_closed += 1;
            println!(
                "breath-group close id={id:?} t_ms={} reason={reason:?}",
                state.frame_time_ms.saturating_add(frame_duration_ms)
            );
            if let Some(group_frames) = state.active_groups.remove(&id) {
                let text = transcribe_group(&group_frames, &mut state.recognizer)?.text;
                if text.is_empty() {
                    println!("transcript group={} text=<empty>", state.groups_closed);
                } else {
                    state.transcripts_emitted += 1;
                    println!("transcript group={} text={}", state.groups_closed, text);
                }
            } else {
                println!(
                    "transcript group={} text=<missing audio>",
                    state.groups_closed
                );
            }
        }
    }

    state.frame_time_ms = state.frame_time_ms.saturating_add(frame_duration_ms);
    Ok(())
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
pub(super) struct TranscribeGroupOutput {
    pub(super) text: String,
    pub(super) words: Vec<TranscriptWord>,
    pub(super) candidate_events: Vec<TranscriptCandidateEvent>,
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
pub(super) fn transcribe_group(
    frames: &[AudioFrame],
    recognizer: &mut WhisperSpeechRecognizer,
) -> Result<TranscribeGroupOutput> {
    transcribe_group_with_finality(frames, recognizer, true)
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
pub(super) fn transcribe_group_with_finality(
    frames: &[AudioFrame],
    recognizer: &mut WhisperSpeechRecognizer,
    is_final: bool,
) -> Result<TranscribeGroupOutput> {
    let whisper_frames = prepare_whisper_frames(frames, WHISPER_FRAME_SAMPLES)?;
    if whisper_frames.is_empty() {
        return Ok(TranscribeGroupOutput {
            text: String::new(),
            words: Vec::new(),
            candidate_events: Vec::new(),
        });
    }
    for frame in &whisper_frames {
        recognizer.push_frame(frame)?;
    }
    let Some((transcript, candidate_events)) =
        recognizer.poll_timed_transcript_with_finality(is_final)?
    else {
        return Ok(TranscribeGroupOutput {
            text: String::new(),
            words: Vec::new(),
            candidate_events: Vec::new(),
        });
    };
    let text = candidate_events
        .iter()
        .filter_map(|event| match event {
            TranscriptCandidateEvent::CandidateFinalized { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ");
    Ok(TranscribeGroupOutput {
        text: text.trim().to_string(),
        words: transcript.words,
        candidate_events,
    })
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
#[allow(clippy::too_many_arguments)]
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
            voice_signatures: Vec::new(),
        };
        if ring_tx.try_push(frame).is_err() {
            dropped_in_ring.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn process_ring_frames(
    ring_rx: &mut listenbury::audio::ring::AudioRingRx,
    state: &mut MicTranscribeState,
) -> Result<()> {
    while let Some(frame) = ring_rx.try_pop() {
        process_live_frame(frame, state)?;
    }
    Ok(())
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn process_web_ring_frames(
    ring_rx: &mut listenbury::audio::ring::AudioRingRx,
    state: &mut WebTranscribeState,
) -> Result<()> {
    while let Some(frame) = ring_rx.try_pop() {
        process_web_transcribe_frame(frame, state)?;
    }
    Ok(())
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn drain_browser_audio_into_ring(
    browser_audio_rx: &crossbeam_channel::Receiver<AudioFrame>,
    ring_tx: &mut listenbury::audio::ring::AudioRingTx,
    dropped_in_ring: &AtomicUsize,
) {
    while let Ok(frame) = browser_audio_rx.try_recv() {
        if ring_tx.try_push(frame).is_err() {
            dropped_in_ring.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn process_web_transcribe_frame(frame: AudioFrame, state: &mut WebTranscribeState) -> Result<()> {
    state.live_audio.push_frame(frame.clone());
    let frame_duration_ms = frame_duration_ms(&frame);
    let vad_result = state.vad.process_frame(&frame)?;

    if state.last_vad_state != Some(vad_result.is_speech) {
        let turn = state.live_trace_turn.saturating_add(1);
        let kind = if vad_result.is_speech {
            "speech_started"
        } else {
            "speech_stopped"
        };
        let mut event = state.live_trace.event(turn, kind, ExactTimestamp::now());
        event.confidence = Some(vad_result.speech_prob);
        state.live_trace.emit(event)?;
        if listenbury::developer_diagnostics_enabled() {
            println!(
                "vad t_ms={} speech={} prob={:.3}",
                state.frame_time_ms, vad_result.is_speech, vad_result.speech_prob
            );
        }
        state.last_vad_state = Some(vad_result.is_speech);
    }

    let events = state.segmenter.process(vad_result);
    for event in &events {
        if let HearingEvent::BreathGroupOpened { id } = event {
            let turn = state.live_trace_turn.saturating_add(1);
            let mut trace_event =
                state
                    .live_trace
                    .event(turn, "breath_group_opened", ExactTimestamp::now());
            trace_event.group_id = Some(format!("{id:?}"));
            state.live_trace.emit(trace_event)?;
            state
                .active_groups
                .entry(*id)
                .or_insert_with(|| ActiveWebTranscribeGroup::new(state.frame_time_ms));
        }
    }

    let mut prospective_groups = Vec::new();
    for group in state.active_groups.values_mut() {
        group.frames.push(frame.clone());
        if state.frame_time_ms >= group.next_prospective_at_ms {
            prospective_groups.push((group.frames.clone(), group.opened_at_ms));
            group.next_prospective_at_ms = state
                .frame_time_ms
                .saturating_add(WEB_TRANSCRIBE_PROSPECTIVE_INTERVAL_MS);
        }
    }
    for (frames, opened_at_ms) in prospective_groups {
        let output = transcribe_group_with_finality(&frames, &mut state.recognizer, false)?;
        emit_web_transcribe_output(
            state,
            output,
            &frames,
            opened_at_ms,
            false,
            ExactTimestamp::now(),
        )?;
    }

    for event in events {
        if let HearingEvent::BreathGroupClosed { id, reason } = event {
            state.groups_closed += 1;
            let turn = state.live_trace_turn.saturating_add(1);
            let mut trace_event =
                state
                    .live_trace
                    .event(turn, "breath_group_closed", ExactTimestamp::now());
            trace_event.group_id = Some(format!("{id:?}"));
            trace_event.reason = Some(format!("{reason:?}"));
            state.live_trace.emit(trace_event)?;

            if let Some(group) = state.active_groups.remove(&id) {
                let output = transcribe_group(&group.frames, &mut state.recognizer)?;
                let finalized_text = output.text.clone();
                if confirms_existing_finalized_segments(state, &finalized_text) {
                    let segment_count = state.finalized_segments.len();
                    emit_web_confirmed_transcript(
                        state,
                        finalized_text,
                        ExactTimestamp::now(),
                        segment_count,
                    )?;
                    continue;
                }
                emit_web_transcribe_output(
                    state,
                    output,
                    &group.frames,
                    group.opened_at_ms,
                    true,
                    ExactTimestamp::now(),
                )?;
                if !finalized_text.is_empty() {
                    append_finalized_asr_segment(state, group.frames, finalized_text);
                    queue_refinement_if_due(state);
                }
            }
        }
    }

    state.frame_time_ms = state.frame_time_ms.saturating_add(frame_duration_ms);
    Ok(())
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn emit_web_transcribe_output(
    state: &mut WebTranscribeState,
    output: TranscribeGroupOutput,
    audio_frames: &[AudioFrame],
    word_timing_offset_ms: u64,
    is_final: bool,
    occurred_at: ExactTimestamp,
) -> Result<()> {
    let turn = state.live_trace_turn.saturating_add(1);
    for event in output.candidate_events {
        let stability = state.candidate_planner.observe(&event);
        emit_web_candidate_trace_event(
            &mut state.live_trace,
            turn,
            &event,
            stability.as_ref(),
            occurred_at,
        )?;
    }

    if is_final && !output.text.is_empty() {
        state.live_trace_turn = turn;
        state.transcripts_emitted += 1;
        println!(
            "transcript group={} text={}",
            state.groups_closed, output.text
        );

        let stream = if output.words.is_empty() {
            live_asr_text_to_word_stream(WordStreamId(state.next_stream_id), &output.text)
        } else {
            let mut stream = transcript_to_energy_snapped_word_stream(
                WordStreamId(state.next_stream_id),
                &output.words,
                audio_frames,
            );
            offset_timed_word_stream_to_session(&mut stream, word_timing_offset_ms);
            stream.source = WordStreamSource::LiveAsr;
            stream
        };
        state.next_stream_id = state.next_stream_id.saturating_add(1);

        let mut transcript_event = state.live_trace.event(turn, "transcript", occurred_at);
        transcript_event.text = Some(output.text);
        state.live_trace.emit(transcript_event)?;

        let mut stream_event = state
            .live_trace
            .event(turn, "asr_timed_word_stream", occurred_at);
        stream_event.artifact =
            Some(serde_json::to_value(stream).context("serialize ASR word stream")?);
        state.live_trace.emit(stream_event)?;
    }

    Ok(())
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn offset_timed_word_stream_to_session(stream: &mut TimedWordStream, offset_ms: u64) {
    if offset_ms == 0 {
        return;
    }
    for word in &mut stream.words {
        if let Some(timing) = word.timing.as_mut() {
            timing.start_ms = timing.start_ms.saturating_add(offset_ms);
            timing.end_ms = timing.end_ms.saturating_add(offset_ms);
        }
    }
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn emit_web_confirmed_transcript(
    state: &mut WebTranscribeState,
    text: String,
    occurred_at: ExactTimestamp,
    segment_count: usize,
) -> Result<()> {
    let turn = state.live_trace_turn.saturating_add(1);
    state.live_trace_turn = turn;
    println!("confirmed transcript segments={segment_count} text={text}");

    let mut event = state
        .live_trace
        .event(turn, "transcript_confirmed", occurred_at);
    event.text = Some(text.clone());
    event.artifact = Some(json!({
        "source": "whisper-large-v3-turbo",
        "input": "consecutive_finalized_asr_segments",
        "segment_count": segment_count,
        "text": text,
    }));
    state.live_trace.emit(event)
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn web_transcribe_breath_group_config() -> BreathGroupConfig {
    BreathGroupConfig {
        close_after_silence_frames: WEB_TRANSCRIBE_BREATH_GROUP_SILENCE_MS
            .div_ceil(DEFAULT_VAD_FRAME_MS)
            .try_into()
            .unwrap_or(usize::MAX),
        max_group_frames: None,
        ..Default::default()
    }
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn emit_web_candidate_trace_event(
    trace: &mut LiveTraceRecorder<SseBroadcaster>,
    turn: u64,
    event: &TranscriptCandidateEvent,
    stability: Option<&WebTranscriptStabilityState>,
    occurred_at: ExactTimestamp,
) -> Result<()> {
    let mut candidate_event = trace.event(turn, "transcript_candidate", occurred_at);
    candidate_event.text = Some(match event {
        TranscriptCandidateEvent::CandidateStarted { id } => {
            format!("candidate_started id={}", id.0)
        }
        TranscriptCandidateEvent::CandidateUpdated { id, .. } => {
            format!("candidate_updated id={}", id.0)
        }
        TranscriptCandidateEvent::CandidateReplaced { old, new, reason } => {
            format!(
                "candidate_replaced old={} new={} reason={reason:?}",
                old.0, new.0
            )
        }
        TranscriptCandidateEvent::CandidateFinalized { id, .. } => {
            format!("candidate_finalized id={}", id.0)
        }
        TranscriptCandidateEvent::CandidateCancelled { id } => {
            format!("candidate_cancelled id={}", id.0)
        }
    });
    if let Some(stability) = stability {
        candidate_event.artifact = Some(json!({
            "candidate_id": stability.candidate_id.0,
            "stable_text": stability.stable_text,
            "unstable_text": stability.unstable_text,
            "confidence": stability.confidence,
            "source": "fast",
        }));
    }
    trace.emit(candidate_event)
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn append_finalized_asr_segment(
    state: &mut WebTranscribeState,
    frames: Vec<AudioFrame>,
    text: String,
) {
    let duration_ms = total_frame_duration_ms(&frames);
    if frames.is_empty() || duration_ms == 0 || text.trim().is_empty() {
        return;
    }

    state.finalized_segments_duration_ms = state
        .finalized_segments_duration_ms
        .saturating_add(duration_ms);
    state.finalized_segments.push_back(FinalizedAsrSegment {
        frames,
        duration_ms,
        text,
    });

    trim_finalized_asr_segments(
        &mut state.finalized_segments,
        &mut state.finalized_segments_duration_ms,
        state.max_finalized_segments_duration_ms,
    );
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn trim_finalized_asr_segments(
    segments: &mut VecDeque<FinalizedAsrSegment>,
    duration_ms: &mut u64,
    max_duration_ms: u64,
) {
    while *duration_ms > max_duration_ms {
        let Some(segment) = segments.pop_front() else {
            *duration_ms = 0;
            break;
        };
        *duration_ms = (*duration_ms).saturating_sub(segment.duration_ms);
    }
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn finalized_segments_text(segments: &VecDeque<FinalizedAsrSegment>) -> String {
    segments
        .iter()
        .map(|segment| segment.text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn normalized_transcript_for_confirmation(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches(|ch: char| ch.is_ascii_punctuation())
        .to_ascii_lowercase()
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn confirms_existing_finalized_segments(state: &WebTranscribeState, text: &str) -> bool {
    confirms_finalized_segments(&state.finalized_segments, text)
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn confirms_finalized_segments(segments: &VecDeque<FinalizedAsrSegment>, text: &str) -> bool {
    if segments.len() < 2 {
        return false;
    }
    let previous = finalized_segments_text(segments);
    !previous.is_empty()
        && normalized_transcript_for_confirmation(&previous)
            == normalized_transcript_for_confirmation(text)
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn queue_refinement_if_due(state: &mut WebTranscribeState) {
    if state.finalized_segments.len() < 2 || Instant::now() < state.next_refine_at {
        return;
    }
    let Some(work) =
        finalized_segments_refinement_work(&state.finalized_segments, ExactTimestamp::now())
    else {
        return;
    };
    match state.refine_tx.try_send(work) {
        Ok(()) => {
            state.next_refine_at = Instant::now() + state.refine_interval;
        }
        Err(crossbeam_channel::TrySendError::Full(_)) => {}
        Err(crossbeam_channel::TrySendError::Disconnected(_)) => {}
    }
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn finalized_segments_refinement_work(
    segments: &VecDeque<FinalizedAsrSegment>,
    observed_at: ExactTimestamp,
) -> Option<RefinementWorkItem> {
    if segments.len() < 2 {
        return None;
    }

    let frame_count = segments
        .iter()
        .map(|segment| segment.frames.len())
        .sum::<usize>();
    if frame_count == 0 {
        return None;
    }

    let mut frames = Vec::with_capacity(frame_count);
    for segment in segments {
        frames.extend(segment.frames.iter().cloned());
    }

    Some(RefinementWorkItem {
        frames,
        observed_at,
        segment_count: segments.len(),
    })
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn run_refinement_worker(
    model_path: std::path::PathBuf,
    rx: crossbeam_channel::Receiver<RefinementWorkItem>,
    broadcaster: SseBroadcaster,
    trace_started_at: ExactTimestamp,
) {
    let recognizer_result = WhisperSpeechRecognizer::new_quiet_without_input_padding(&model_path);
    let mut recognizer = match recognizer_result {
        Ok(recognizer) => recognizer,
        Err(error) => {
            let mut trace = LiveTraceRecorder::new(trace_started_at, broadcaster);
            let mut event = trace.event(0, "transcription_refinement_error", ExactTimestamp::now());
            event.text = Some(format!("failed to load refinement model: {error:#}"));
            let _ = trace.emit(event);
            return;
        }
    };
    let mut trace = LiveTraceRecorder::new(trace_started_at, broadcaster);

    while let Ok(work) = rx.recv() {
        match transcribe_group_with_finality(&work.frames, &mut recognizer, true) {
            Ok(output) if !output.text.is_empty() => {
                let mut event = trace.event(0, "transcript_confirmed", work.observed_at);
                event.text = Some(output.text.clone());
                event.artifact = Some(json!({
                    "source": "whisper-large-v3-turbo",
                    "input": "consecutive_finalized_asr_segments",
                    "segment_count": work.segment_count,
                    "window_ms": total_frame_duration_ms(&work.frames),
                    "text": output.text,
                }));
                let _ = trace.emit(event);
            }
            Ok(_) => {}
            Err(error) => {
                let mut event =
                    trace.event(0, "transcription_refinement_error", ExactTimestamp::now());
                event.text = Some(error.to_string());
                let _ = trace.emit(event);
            }
        }
    }
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn total_frame_duration_ms(frames: &[AudioFrame]) -> u64 {
    frames.iter().fold(0u64, |total, frame| {
        total.saturating_add(frame_duration_ms(frame))
    })
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn live_asr_text_to_word_stream(stream_id: WordStreamId, transcript: &str) -> TimedWordStream {
    let words = transcript
        .split_whitespace()
        .map(|word| TranscriptWord {
            text: word.to_string(),
            start_ms: None,
            end_ms: None,
            confidence: None,
        })
        .collect::<Vec<_>>();
    let mut stream = transcript_to_word_stream(stream_id, &words);
    stream.source = WordStreamSource::LiveAsr;
    stream
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn browser_host_for_bind_host(bind_host: &str) -> String {
    match bind_host {
        "0.0.0.0" => "127.0.0.1".to_string(),
        "::" => "[::1]".to_string(),
        _ => {
            let looks_like_ipv6 =
                bind_host.contains(':') && !bind_host.starts_with('[') && !bind_host.ends_with(']');
            if looks_like_ipv6 {
                format!("[{bind_host}]")
            } else {
                bind_host.to_string()
            }
        }
    }
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn prepare_whisper_frames(frames: &[AudioFrame], frame_samples: usize) -> Result<Vec<AudioFrame>> {
    anyhow::ensure!(frame_samples > 0, "frame_samples must be greater than zero");
    let Some(first) = frames.first() else {
        return Ok(Vec::new());
    };
    anyhow::ensure!(first.channels > 0, "input audio frame has zero channels");

    let source_rate_hz = first.sample_rate_hz;
    let source_channels = first.channels;
    let mut interleaved = Vec::new();
    for frame in frames {
        anyhow::ensure!(
            frame.sample_rate_hz == source_rate_hz,
            "audio frame sample rate changed mid-group ({} -> {})",
            source_rate_hz,
            frame.sample_rate_hz
        );
        anyhow::ensure!(
            frame.channels == source_channels,
            "audio frame channel count changed mid-group ({} -> {})",
            source_channels,
            frame.channels
        );
        interleaved.extend_from_slice(&frame.samples);
    }

    let mono = mix_to_mono(&interleaved, source_channels);
    let resampled = resample_linear(&mono, source_rate_hz, WHISPER_SAMPLE_RATE_HZ);
    Ok(resampled
        .chunks(frame_samples)
        .map(|chunk| AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: WHISPER_SAMPLE_RATE_HZ,
            channels: 1,
            samples: chunk.to_vec(),
            voice_signatures: Vec::new(),
        })
        .collect())
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn vad_frame_format(
    vad_backend: listenbury::hearing::vad::VadBackendKind,
    input_sample_rate_hz: u32,
    input_channels: u16,
) -> (u32, u16) {
    match vad_backend {
        listenbury::hearing::vad::VadBackendKind::WebRtc => {
            (WEBRTC_VAD_SAMPLE_RATE_HZ, MONO_CHANNELS)
        }
        listenbury::hearing::vad::VadBackendKind::Energy
        | listenbury::hearing::vad::VadBackendKind::Silero => {
            (input_sample_rate_hz, input_channels)
        }
    }
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
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

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn frame_samples_per_callback_frame(sample_rate_hz: u32, channels: u16) -> usize {
    let samples_per_channel = usize::try_from(sample_rate_hz / 100).unwrap_or(1).max(1);
    samples_per_channel.saturating_mul(usize::from(channels).max(1))
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn frame_duration_ms(frame: &AudioFrame) -> u64 {
    if frame.sample_rate_hz == 0 || frame.channels == 0 {
        return 0;
    }
    let samples_per_channel = frame.samples.len() as f64 / f64::from(frame.channels);
    ((samples_per_channel / f64::from(frame.sample_rate_hz)) * 1000.0).round() as u64
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn mix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    let channel_count = usize::from(channels);
    if channel_count == 1 {
        return samples.to_vec();
    }
    samples
        .chunks_exact(channel_count)
        .map(|frame| frame.iter().sum::<f32>() / f32::from(channels))
        .collect()
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
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

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn build_input_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_tx: crossbeam_channel::Sender<f32>,
    dropped_in_callback: Arc<AtomicUsize>,
    err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream>
where
    T: Sample + SizedSample,
    f32: FromSample<T>,
{
    build_input_stream_with_capture_control(
        device,
        config,
        sample_tx,
        dropped_in_callback,
        None,
        err_fn,
    )
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn build_input_stream_with_capture_control<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_tx: crossbeam_channel::Sender<f32>,
    dropped_in_callback: Arc<AtomicUsize>,
    capture_enabled: Option<Arc<AtomicBool>>,
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
                if capture_enabled
                    .as_ref()
                    .is_some_and(|enabled| !enabled.load(Ordering::Relaxed))
                {
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

#[cfg(all(test, feature = "asr-whisper", feature = "audio-cpal"))]
mod tests {
    use super::{
        FinalizedAsrSegment, WEB_TRANSCRIBE_BREATH_GROUP_SILENCE_MS, confirms_finalized_segments,
        convert_frame_samples, finalized_segments_refinement_work, total_frame_duration_ms,
        trim_finalized_asr_segments, vad_frame_format, web_transcribe_breath_group_config,
    };
    use listenbury::hearing::breath::DEFAULT_VAD_FRAME_MS;
    use listenbury::hearing::vad::VadBackendKind;
    use listenbury::{AudioFrame, ExactTimestamp};
    use std::collections::VecDeque;

    fn test_frame(sample_count: usize) -> AudioFrame {
        AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![0.1; sample_count],
            voice_signatures: Vec::new(),
        }
    }

    fn test_segment(frame_count: usize) -> FinalizedAsrSegment {
        let frames = vec![test_frame(160); frame_count];
        let duration_ms = total_frame_duration_ms(&frames);
        FinalizedAsrSegment {
            frames,
            duration_ms,
            text: format!("segment {frame_count}"),
        }
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
    fn web_transcribe_groups_close_on_short_pause_without_hard_timeout() {
        let config = web_transcribe_breath_group_config();

        assert_eq!(
            config.close_after_silence_frames as u64 * DEFAULT_VAD_FRAME_MS,
            WEB_TRANSCRIBE_BREATH_GROUP_SILENCE_MS
        );
        assert_eq!(config.max_group_frames, None);
    }

    #[test]
    fn refinement_work_combines_multiple_finalized_segments() {
        let mut segments = VecDeque::new();
        segments.push_back(test_segment(2));
        segments.push_back(test_segment(3));

        let work = finalized_segments_refinement_work(&segments, ExactTimestamp::now())
            .expect("two finalized segments should queue refinement");

        assert_eq!(work.segment_count, 2);
        assert_eq!(work.frames.len(), 5);
        assert_eq!(total_frame_duration_ms(&work.frames), 50);
    }

    #[test]
    fn refinement_work_waits_for_a_transition_between_segments() {
        let mut segments = VecDeque::new();
        segments.push_back(test_segment(2));

        assert!(finalized_segments_refinement_work(&segments, ExactTimestamp::now()).is_none());
    }

    #[test]
    fn finalized_segment_window_trims_whole_old_segments() {
        let mut segments = VecDeque::new();
        segments.push_back(test_segment(2));
        segments.push_back(test_segment(3));
        segments.push_back(test_segment(4));
        let mut duration_ms = segments.iter().fold(0u64, |total, segment| {
            total.saturating_add(segment.duration_ms)
        });

        trim_finalized_asr_segments(&mut segments, &mut duration_ms, 70);

        assert_eq!(segments.len(), 2);
        assert_eq!(duration_ms, 70);
        assert_eq!(
            segments.front().map(|segment| segment.frames.len()),
            Some(3)
        );
    }

    #[test]
    fn combined_prior_segments_are_treated_as_confirmation() {
        let mut segments = VecDeque::new();
        segments.push_back(FinalizedAsrSegment {
            frames: vec![test_frame(160)],
            duration_ms: 10,
            text: "Hello, can you hear me?".to_string(),
        });
        segments.push_back(FinalizedAsrSegment {
            frames: vec![test_frame(160)],
            duration_ms: 10,
            text: "My name is Travis.".to_string(),
        });

        assert!(confirms_finalized_segments(
            &segments,
            "Hello, can you hear me? My name is Travis."
        ));
    }
}
