use std::time::Duration;

use anyhow::Context;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::append_mock_downstream_trace;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::real_payload;
use listenbury::{
    MockLoopTraceConfig, mock_interaction_trace, summarize_latency, write_trace_jsonl,
};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use serde_json::json;

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use std::collections::{HashMap, VecDeque};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use std::time::Instant;

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use cpal::{FromSample, Sample, SizedSample};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::audio::capture::{
    boost_current_thread_for_capture, callback_sample_queue_capacity,
};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::audio::{AudioFormat, SampleKind, normalize_interleaved_f32};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::event::HearingEvent;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::hearing::breath::{BreathGroupEndReason, BreathGroupId, BreathGroupSegmenter};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::hearing::vad::{VadBackendKind, create_vad_backend_with_profile};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::speech::recognizer::SpeechRecognizer;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::{AudioFrame, ExactTimestamp, TraceEvent, WhisperSpeechRecognizer};

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use crate::cli::model_paths::resolve_whisper_model;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use crate::cli::resolve_vad_config;
use crate::cli::{LoopTraceCommand, LoopTraceProfile};

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
const WHISPER_SAMPLE_RATE_HZ: u32 = 16_000;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
const WHISPER_FRAME_SAMPLES: usize = 160;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
const WEBRTC_VAD_SAMPLE_RATE_HZ: u32 = 16_000;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
const MONO_CHANNELS: u16 = 1;

pub(crate) fn run_loop_trace(command: LoopTraceCommand) -> anyhow::Result<()> {
    let events = match command.effective_profile() {
        LoopTraceProfile::Mock => mock_loop_trace(&command),
        LoopTraceProfile::Ear => real_ear_loop_trace(&command)?,
    };
    write_trace_jsonl(&command.write, &events)
        .with_context(|| format!("write loop trace {}", command.write.display()))?;

    let summary = summarize_latency(&events);
    if command.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!("{}", summary.format_pretty());
        println!();
        println!("trace: {}", command.write.display());
    }

    Ok(())
}

fn mock_loop_trace(command: &LoopTraceCommand) -> Vec<listenbury::TraceEvent> {
    let config = MockLoopTraceConfig {
        duration: Duration::from_secs(command.duration),
        self_hearing: !command.no_self_hearing,
    };
    if command.mock_mic || command.mock_llm || command.mock_mouth {
        eprintln!(
            "note: loop-trace mock profile uses the synthetic mock path; explicit mock flags were accepted"
        );
    }
    mock_interaction_trace(config)
}

#[cfg(not(all(feature = "asr-whisper", feature = "audio-cpal")))]
fn real_ear_loop_trace(_command: &LoopTraceCommand) -> anyhow::Result<Vec<listenbury::TraceEvent>> {
    anyhow::bail!("loop-trace --profile ear requires the `audio-cpal` and `asr-whisper` features")
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn real_ear_loop_trace(command: &LoopTraceCommand) -> anyhow::Result<Vec<TraceEvent>> {
    anyhow::ensure!(
        command.duration > 0,
        "--duration must be greater than zero for --profile ear"
    );

    let model_path = resolve_whisper_model(command.whisper_model.clone())?;
    let mut recognizer = WhisperSpeechRecognizer::new_quiet(&model_path)
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
    let sample_format = supported_config.sample_format();
    let stream_config = supported_config.config();
    let input_sample_rate_hz = stream_config.sample_rate.0;
    let input_channels = stream_config.channels;
    anyhow::ensure!(
        input_channels > 0,
        "default input device reported zero channels"
    );

    let vad_config = resolve_vad_config(command.vad, command.vad_profile.as_deref())?;
    let vad_backend = vad_config.backend;
    let (frame_sample_rate_hz, frame_channels) =
        vad_frame_format(vad_backend, input_sample_rate_hz, input_channels);
    let input_frame_samples =
        frame_samples_per_callback_frame(input_sample_rate_hz, input_channels);
    let sample_capacity = callback_sample_queue_capacity(input_sample_rate_hz, input_channels);
    let (sample_tx, sample_rx) = crossbeam_channel::bounded::<f32>(sample_capacity);
    let dropped_in_callback = Arc::new(AtomicUsize::new(0));
    let err_fn = |err| eprintln!("input stream error: {err}");
    let stream = match sample_format {
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

    let started_at = Instant::now();
    let mut events = Vec::new();
    stream
        .play()
        .with_context(|| format!("failed to start capture from {device_name}"))?;
    events.push(TraceEvent::new(
        started_at.elapsed(),
        "audio_capture",
        "capture_start",
        Some(real_payload(json!({
            "source": "real_mic",
            "selected_input_device": device_name,
            "sample_rate_hz": input_sample_rate_hz,
            "channel_count": input_channels,
            "frame_size": input_frame_samples,
            "sample_format": format!("{sample_format:?}"),
            "vad_backend": vad_backend.as_str(),
            "requested_duration_ms": command.duration.saturating_mul(1_000)
        }))),
    ));

    println!(
        "loop-trace ear profile listening on {device_name}: {} Hz, {} channel(s), vad={}. Speak one utterance.",
        input_sample_rate_hz,
        input_channels,
        vad_backend.as_str()
    );
    boost_current_thread_for_capture("loop-trace");

    let mut pending = VecDeque::<f32>::new();
    let mut state = EarTraceState {
        vad: create_vad_backend_with_profile(vad_backend, vad_config.profile.as_ref())?,
        vad_backend,
        segmenter: vad_config
            .profile
            .map(|profile| BreathGroupSegmenter::new(profile.breath_group_config()))
            .unwrap_or_default(),
        active_groups: HashMap::new(),
        last_vad_state: None,
        frame_index: 0,
        first_asr_final_elapsed: None,
    };
    let deadline = started_at + Duration::from_secs(command.duration);
    while Instant::now() < deadline && state.first_asr_final_elapsed.is_none() {
        match sample_rx.recv_timeout(Duration::from_millis(20)) {
            Ok(sample) => pending.push_back(sample),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
        while let Ok(sample) = sample_rx.try_recv() {
            pending.push_back(sample);
        }
        process_pending_samples(
            &mut pending,
            &mut state,
            &mut recognizer,
            &model_path,
            input_frame_samples,
            input_sample_rate_hz,
            input_channels,
            frame_sample_rate_hz,
            frame_channels,
            started_at,
            &mut events,
        )?;
    }

    drop(stream);
    while let Ok(sample) = sample_rx.try_recv() {
        pending.push_back(sample);
    }
    process_pending_samples(
        &mut pending,
        &mut state,
        &mut recognizer,
        &model_path,
        input_frame_samples,
        input_sample_rate_hz,
        input_channels,
        frame_sample_rate_hz,
        frame_channels,
        started_at,
        &mut events,
    )?;

    if state.first_asr_final_elapsed.is_none() {
        force_close_active_groups(
            &mut state,
            &mut recognizer,
            &model_path,
            started_at,
            &mut events,
        )?;
    }

    events.push(TraceEvent::new(
        started_at.elapsed(),
        "audio_capture",
        "capture_end",
        Some(real_payload(json!({
            "source": "real_mic",
            "callback_drops": dropped_in_callback.load(Ordering::Relaxed)
        }))),
    ));

    if let Some(asr_final_elapsed) = state.first_asr_final_elapsed {
        append_mock_downstream_trace(&mut events, asr_final_elapsed, !command.no_self_hearing);
        events.sort_by_key(|event| event.monotonic_ns);
    }

    Ok(events)
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
struct ActiveEarGroup {
    frames: Vec<AudioFrame>,
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
struct EarTraceState {
    vad: Box<dyn listenbury::hearing::vad::VoiceActivityDetector>,
    vad_backend: VadBackendKind,
    segmenter: BreathGroupSegmenter,
    active_groups: HashMap<BreathGroupId, ActiveEarGroup>,
    last_vad_state: Option<bool>,
    frame_index: u64,
    first_asr_final_elapsed: Option<Duration>,
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
#[allow(clippy::too_many_arguments)]
fn process_pending_samples(
    pending: &mut VecDeque<f32>,
    state: &mut EarTraceState,
    recognizer: &mut WhisperSpeechRecognizer,
    model_path: &std::path::Path,
    input_frame_samples: usize,
    input_sample_rate_hz: u32,
    input_channels: u16,
    frame_sample_rate_hz: u32,
    frame_channels: u16,
    started_at: Instant,
    events: &mut Vec<TraceEvent>,
) -> anyhow::Result<()> {
    while pending.len() >= input_frame_samples && state.first_asr_final_elapsed.is_none() {
        let mut samples = Vec::with_capacity(input_frame_samples);
        for _ in 0..input_frame_samples {
            if let Some(sample) = pending.pop_front() {
                samples.push(sample);
            }
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
        process_ear_frame(frame, state, recognizer, model_path, started_at, events)?;
    }
    Ok(())
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn process_ear_frame(
    frame: AudioFrame,
    state: &mut EarTraceState,
    recognizer: &mut WhisperSpeechRecognizer,
    model_path: &std::path::Path,
    started_at: Instant,
    events: &mut Vec<TraceEvent>,
) -> anyhow::Result<()> {
    let observed_at = started_at.elapsed();
    events.push(TraceEvent::new(
        observed_at,
        "audio_capture",
        "audio_frame_received",
        Some(real_payload(json!({
            "source": "real_mic",
            "frame_index": state.frame_index,
            "sample_rate_hz": frame.sample_rate_hz,
            "channel_count": frame.channels,
            "frame_size": frame.samples.len()
        }))),
    ));
    state.frame_index = state.frame_index.saturating_add(1);

    let vad_result = state.vad.process_frame(&frame)?;
    let vad_transition = match (state.last_vad_state, vad_result.is_speech) {
        (Some(true), false) => Some("speech_end"),
        (previous, true) if previous != Some(true) => Some("speech_start"),
        _ => None,
    };
    if let Some(kind) = vad_transition {
        events.push(TraceEvent::new(
            started_at.elapsed(),
            "vad",
            kind,
            Some(real_payload(json!({
                "backend": state.vad_backend.as_str(),
                "speech_prob": vad_result.speech_prob
            }))),
        ));
    }
    state.last_vad_state = Some(vad_result.is_speech);

    let hearing_events = state.segmenter.process(vad_result);
    for event in &hearing_events {
        if let HearingEvent::BreathGroupOpened { id } = event {
            state
                .active_groups
                .entry(*id)
                .or_insert_with(|| ActiveEarGroup { frames: Vec::new() });
        }
    }
    for group in state.active_groups.values_mut() {
        group.frames.push(frame.clone());
    }
    for event in hearing_events {
        if let HearingEvent::BreathGroupClosed { id, reason } = event
            && let Some(group) = state.active_groups.remove(&id)
        {
            transcribe_closed_group(
                group, reason, state, recognizer, model_path, started_at, events,
            )?;
        }
    }

    Ok(())
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn force_close_active_groups(
    state: &mut EarTraceState,
    recognizer: &mut WhisperSpeechRecognizer,
    model_path: &std::path::Path,
    started_at: Instant,
    events: &mut Vec<TraceEvent>,
) -> anyhow::Result<()> {
    let groups = std::mem::take(&mut state.active_groups);
    if groups.is_empty() {
        return Ok(());
    }
    if state.last_vad_state == Some(true) {
        events.push(TraceEvent::new(
            started_at.elapsed(),
            "vad",
            "speech_end",
            Some(real_payload(json!({
                "backend": "shutdown",
                "reason": "duration_elapsed"
            }))),
        ));
    }
    for (_, group) in groups {
        transcribe_closed_group(
            group,
            BreathGroupEndReason::Timeout,
            state,
            recognizer,
            model_path,
            started_at,
            events,
        )?;
        if state.first_asr_final_elapsed.is_some() {
            break;
        }
    }
    Ok(())
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn transcribe_closed_group(
    group: ActiveEarGroup,
    reason: BreathGroupEndReason,
    state: &mut EarTraceState,
    recognizer: &mut WhisperSpeechRecognizer,
    model_path: &std::path::Path,
    started_at: Instant,
    events: &mut Vec<TraceEvent>,
) -> anyhow::Result<()> {
    if group.frames.is_empty() {
        return Ok(());
    }
    let asr_started_at = Instant::now();
    let whisper_frames = prepare_whisper_frames(&group.frames, WHISPER_FRAME_SAMPLES)?;
    for frame in &whisper_frames {
        recognizer.push_frame(frame)?;
    }
    let output = recognizer.poll_timed_transcript_with_finality(true)?;
    let asr_elapsed = started_at.elapsed();
    let transcription_duration_ms = asr_started_at.elapsed().as_secs_f64() * 1_000.0;
    let confidence = average_word_confidence(&output.words);
    let final_text = output.text;
    events.push(TraceEvent::new(
        asr_elapsed,
        "asr",
        "final_result",
        Some(real_payload(json!({
            "backend": output.backend.source,
            "partial_kind": output.backend.partial_kind.as_str(),
            "model_path": model_path.display().to_string(),
            "model_name": model_path.file_name().and_then(|name| name.to_str()),
            "transcription_duration_ms": transcription_duration_ms,
            "final_text": final_text,
            "text": final_text,
            "confidence": confidence,
            "breath_group_end_reason": format!("{reason:?}")
        }))),
    ));
    state.first_asr_final_elapsed.get_or_insert(asr_elapsed);
    Ok(())
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn average_word_confidence(words: &[listenbury::word::TranscriptWord]) -> Option<f32> {
    let mut sum = 0.0;
    let mut count = 0;
    for confidence in words.iter().filter_map(|word| word.confidence) {
        sum += confidence;
        count += 1;
    }
    (count > 0).then(|| sum / count as f32)
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
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

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn frame_samples_per_callback_frame(sample_rate_hz: u32, channels: u16) -> usize {
    let samples_per_channel = usize::try_from(sample_rate_hz / 100).unwrap_or(1).max(1);
    samples_per_channel.saturating_mul(usize::from(channels).max(1))
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
    normalize_interleaved_f32(
        samples,
        AudioFormat::new(input_sample_rate_hz, input_channels, SampleKind::F32),
        AudioFormat::new(frame_sample_rate_hz, frame_channels, SampleKind::F32),
        "loop_trace_ear_frame",
    )
    .expect("validated loop trace frame formats should normalize")
    .samples
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn prepare_whisper_frames(
    frames: &[AudioFrame],
    frame_samples: usize,
) -> anyhow::Result<Vec<AudioFrame>> {
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

    let resampled = normalize_interleaved_f32(
        &interleaved,
        AudioFormat::new(source_rate_hz, source_channels, SampleKind::F32),
        AudioFormat::new(WHISPER_SAMPLE_RATE_HZ, MONO_CHANNELS, SampleKind::F32),
        "loop_trace_whisper_input",
    )?
    .samples;
    Ok(resampled
        .chunks(frame_samples)
        .map(|chunk| AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: WHISPER_SAMPLE_RATE_HZ,
            channels: MONO_CHANNELS,
            samples: chunk.to_vec(),
            voice_signatures: Vec::new(),
        })
        .collect())
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn build_input_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_tx: crossbeam_channel::Sender<f32>,
    dropped_in_callback: Arc<AtomicUsize>,
    err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
) -> anyhow::Result<cpal::Stream>
where
    T: Sample + SizedSample,
    f32: FromSample<T>,
{
    device
        .build_input_stream(
            config,
            move |data: &[T], _| {
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
