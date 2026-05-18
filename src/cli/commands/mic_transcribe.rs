use crate::cli::MicTranscribeCommand;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use crate::cli::model_paths::resolve_whisper_model;
use anyhow::Result;

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use anyhow::Context;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use cpal::{FromSample, Sample, SizedSample};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::audio::ring::make_audio_ring;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::event::HearingEvent;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::hearing::breath::{BreathGroupId, BreathGroupSegmenter};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::hearing::vad::{VoiceActivityDetector, create_vad_backend};
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::speech::recognizer::SpeechRecognizer;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::speech::transcript::TranscriptCandidateEvent;
#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
use listenbury::{AudioFrame, ExactTimestamp, WhisperSpeechRecognizer};
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
const CALLBACK_SAMPLE_CAPACITY: usize = 16_384;
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
pub(crate) fn run_mic_transcribe(command: MicTranscribeCommand) -> Result<()> {
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

    let (sample_tx, sample_rx) = crossbeam_channel::bounded::<f32>(CALLBACK_SAMPLE_CAPACITY);
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
        "mic-transcribe listening on {device_name}: {} Hz, {} channel(s), vad={}. Press Ctrl-C to stop.",
        input_sample_rate_hz,
        input_channels,
        command.vad.as_backend_kind().as_str()
    );

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
        if let Some(deadline) = stop_deadline {
            if Instant::now() >= deadline {
                println!(
                    "capture timeout reached ({}s), stopping...",
                    command.seconds
                );
                break;
            }
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

#[cfg(not(all(feature = "asr-whisper", feature = "audio-cpal")))]
pub(crate) fn run_mic_transcribe(_command: MicTranscribeCommand) -> Result<()> {
    anyhow::bail!("listenbury mic-transcribe requires the `audio-cpal` and `asr-whisper` features")
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
fn process_live_frame(frame: AudioFrame, state: &mut MicTranscribeState) -> Result<()> {
    let frame_duration_ms = frame_duration_ms(&frame);
    let vad_result = state.vad.process_frame(&frame)?;

    if state.last_vad_state != Some(vad_result.is_speech) {
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
            candidate_events: Vec::new(),
        });
    }
    for frame in &whisper_frames {
        recognizer.push_frame(frame)?;
    }
    let candidate_events = recognizer.poll_candidate_events_with_finality(is_final)?;
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
        candidate_events,
    })
}

#[cfg(all(feature = "asr-whisper", feature = "audio-cpal"))]
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

#[cfg(all(test, feature = "asr-whisper", feature = "audio-cpal"))]
mod tests {
    use super::{convert_frame_samples, vad_frame_format};
    use listenbury::hearing::vad::VadBackendKind;

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
}
