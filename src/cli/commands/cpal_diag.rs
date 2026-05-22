use crate::cli::{PlayWavCommand, RecordWavCommand};
use anyhow::Result;

#[cfg(feature = "audio-cpal")]
use anyhow::Context;
#[cfg(feature = "audio-cpal")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(feature = "audio-cpal")]
use cpal::{FromSample, Sample, SizedSample};
#[cfg(feature = "audio-cpal")]
use listenbury::audio::frame::AudioFrame;
#[cfg(feature = "audio-cpal")]
use listenbury::audio::{read_wav_frames, write_wav};
#[cfg(feature = "audio-cpal")]
use listenbury::time::ExactTimestamp;
#[cfg(feature = "audio-cpal")]
use std::collections::VecDeque;
#[cfg(feature = "audio-cpal")]
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
#[cfg(feature = "audio-cpal")]
use std::time::{Duration, Instant};

#[cfg(feature = "audio-cpal")]
const CALLBACK_CHANNEL_CAPACITY: usize = 16_384;

#[cfg(feature = "audio-cpal")]
pub(crate) fn run_record_wav(command: RecordWavCommand) -> Result<()> {
    anyhow::ensure!(
        command.seconds > 0,
        "--seconds must be greater than zero for record-wav"
    );

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
    let sample_rate = stream_config.sample_rate.0;
    let channels = stream_config.channels;

    let parent = command
        .output_wav
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    if let Some(parent) = parent {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create output directory {}",
                parent.to_string_lossy()
            )
        })?;
    }

    let (sample_tx, sample_rx) = crossbeam_channel::bounded::<f32>(CALLBACK_CHANNEL_CAPACITY);
    let err_fn = |err| eprintln!("input stream error: {err}");

    let stream = match supported_config.sample_format() {
        cpal::SampleFormat::F32 => {
            build_input_stream::<f32>(&device, &stream_config, sample_tx.clone(), err_fn)?
        }
        cpal::SampleFormat::F64 => {
            build_input_stream::<f64>(&device, &stream_config, sample_tx.clone(), err_fn)?
        }
        cpal::SampleFormat::I8 => {
            build_input_stream::<i8>(&device, &stream_config, sample_tx.clone(), err_fn)?
        }
        cpal::SampleFormat::I16 => {
            build_input_stream::<i16>(&device, &stream_config, sample_tx.clone(), err_fn)?
        }
        cpal::SampleFormat::I32 => {
            build_input_stream::<i32>(&device, &stream_config, sample_tx.clone(), err_fn)?
        }
        cpal::SampleFormat::I64 => {
            build_input_stream::<i64>(&device, &stream_config, sample_tx.clone(), err_fn)?
        }
        cpal::SampleFormat::U8 => {
            build_input_stream::<u8>(&device, &stream_config, sample_tx.clone(), err_fn)?
        }
        cpal::SampleFormat::U16 => {
            build_input_stream::<u16>(&device, &stream_config, sample_tx.clone(), err_fn)?
        }
        cpal::SampleFormat::U32 => {
            build_input_stream::<u32>(&device, &stream_config, sample_tx.clone(), err_fn)?
        }
        cpal::SampleFormat::U64 => {
            build_input_stream::<u64>(&device, &stream_config, sample_tx.clone(), err_fn)?
        }
        sample_format => anyhow::bail!("unsupported input sample format: {sample_format:?}"),
    };

    stream
        .play()
        .with_context(|| format!("failed to start capture from {device_name}"))?;

    let mut samples = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(command.seconds);
    while Instant::now() < deadline {
        match sample_rx.recv_timeout(Duration::from_millis(20)) {
            Ok(sample) => samples.push(sample),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
    drop(stream);

    while let Ok(sample) = sample_rx.try_recv() {
        samples.push(sample);
    }

    anyhow::ensure!(
        !samples.is_empty(),
        "record-wav captured no samples from {device_name}"
    );

    let frame_count = samples.len() / usize::from(channels);
    let audio = vec![AudioFrame {
        captured_at: ExactTimestamp::now(),
        sample_rate_hz: sample_rate,
        channels,
        samples,
        voice_signatures: Vec::new(),
    }];
    write_wav(&command.output_wav, &audio)?;

    println!(
        "Recorded with {device_name}: {} Hz, {channels} channel(s), {frame_count} frames -> {}",
        sample_rate,
        command.output_wav.display()
    );

    Ok(())
}

#[cfg(feature = "audio-cpal")]
pub(crate) fn run_play_wav(command: PlayWavCommand) -> Result<()> {
    let frames = read_wav_frames(&command.input_wav, 2_048)
        .with_context(|| format!("failed to read WAV {}", command.input_wav.display()))?;
    play_audio_frames(&frames, &command.input_wav.display().to_string())
}

#[cfg(not(feature = "audio-cpal"))]
pub(crate) fn run_record_wav(_command: RecordWavCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `audio-cpal` feature")
}

#[cfg(not(feature = "audio-cpal"))]
pub(crate) fn run_play_wav(_command: PlayWavCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `audio-cpal` feature")
}

#[cfg(feature = "audio-cpal")]
fn build_input_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_tx: crossbeam_channel::Sender<f32>,
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
                    let _ = sample_tx.try_send(sample.to_sample::<f32>());
                }
            },
            err_fn,
            None,
        )
        .context("failed to build input stream")
}

#[cfg(feature = "audio-cpal")]
fn build_output_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    samples: Arc<Vec<f32>>,
    playback_cursor: Arc<AtomicUsize>,
    playback_paused: Arc<AtomicBool>,
    err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream>
where
    T: Sample + SizedSample + FromSample<f32>,
{
    device
        .build_output_stream(
            config,
            move |output: &mut [T], _| {
                for out in output.iter_mut() {
                    if playback_paused.load(Ordering::Relaxed) {
                        *out = T::from_sample(0.0);
                        continue;
                    }
                    let idx = playback_cursor.fetch_add(1, Ordering::Relaxed);
                    let sample = samples.get(idx).copied().unwrap_or(0.0);
                    *out = T::from_sample(sample);
                }
            },
            err_fn,
            None,
        )
        .context("failed to build output stream")
}

#[cfg(feature = "audio-cpal")]
fn build_output_queue_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_queue: Arc<Mutex<VecDeque<f32>>>,
    err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream>
where
    T: Sample + SizedSample + FromSample<f32>,
{
    device
        .build_output_stream(
            config,
            move |output: &mut [T], _| {
                let mut queue = sample_queue.lock().expect("audio sample queue poisoned");
                for out in output.iter_mut() {
                    let sample = queue.pop_front().unwrap_or(0.0);
                    *out = T::from_sample(sample);
                }
            },
            err_fn,
            None,
        )
        .context("failed to build streaming output stream")
}

#[cfg(feature = "audio-cpal")]
pub(crate) struct PreparedAudioPlayback {
    device: cpal::Device,
    stream_config: cpal::StreamConfig,
    sample_format: cpal::SampleFormat,
    pub(crate) device_name: String,
    pub(crate) sample_rate_hz: u32,
    pub(crate) channels: u16,
    pub(crate) samples: Arc<Vec<f32>>,
}

#[cfg(feature = "audio-cpal")]
impl PreparedAudioPlayback {
    pub(crate) fn sample_count(&self) -> usize {
        self.samples.len()
    }

    pub(crate) fn duration(&self) -> Duration {
        playback_duration(self.sample_count(), self.sample_rate_hz, self.channels)
    }

    pub(crate) fn as_audio_frame(&self, captured_at: ExactTimestamp) -> AudioFrame {
        AudioFrame {
            captured_at,
            sample_rate_hz: self.sample_rate_hz,
            channels: self.channels,
            samples: self.samples.as_ref().clone(),
            voice_signatures: Vec::new(),
        }
    }

    pub(crate) fn build_stream(
        &self,
        playback_cursor: Arc<AtomicUsize>,
        playback_paused: Arc<AtomicBool>,
    ) -> Result<cpal::Stream> {
        let err_fn = |err| eprintln!("output stream error: {err}");
        match self.sample_format {
            cpal::SampleFormat::F32 => build_output_stream::<f32>(
                &self.device,
                &self.stream_config,
                Arc::clone(&self.samples),
                playback_cursor,
                playback_paused,
                err_fn,
            ),
            cpal::SampleFormat::F64 => build_output_stream::<f64>(
                &self.device,
                &self.stream_config,
                Arc::clone(&self.samples),
                playback_cursor,
                playback_paused,
                err_fn,
            ),
            cpal::SampleFormat::I8 => build_output_stream::<i8>(
                &self.device,
                &self.stream_config,
                Arc::clone(&self.samples),
                playback_cursor,
                playback_paused,
                err_fn,
            ),
            cpal::SampleFormat::I16 => build_output_stream::<i16>(
                &self.device,
                &self.stream_config,
                Arc::clone(&self.samples),
                playback_cursor,
                playback_paused,
                err_fn,
            ),
            cpal::SampleFormat::I32 => build_output_stream::<i32>(
                &self.device,
                &self.stream_config,
                Arc::clone(&self.samples),
                playback_cursor,
                playback_paused,
                err_fn,
            ),
            cpal::SampleFormat::I64 => build_output_stream::<i64>(
                &self.device,
                &self.stream_config,
                Arc::clone(&self.samples),
                playback_cursor,
                playback_paused,
                err_fn,
            ),
            cpal::SampleFormat::U8 => build_output_stream::<u8>(
                &self.device,
                &self.stream_config,
                Arc::clone(&self.samples),
                playback_cursor,
                playback_paused,
                err_fn,
            ),
            cpal::SampleFormat::U16 => build_output_stream::<u16>(
                &self.device,
                &self.stream_config,
                Arc::clone(&self.samples),
                playback_cursor,
                playback_paused,
                err_fn,
            ),
            cpal::SampleFormat::U32 => build_output_stream::<u32>(
                &self.device,
                &self.stream_config,
                Arc::clone(&self.samples),
                playback_cursor,
                playback_paused,
                err_fn,
            ),
            cpal::SampleFormat::U64 => build_output_stream::<u64>(
                &self.device,
                &self.stream_config,
                Arc::clone(&self.samples),
                playback_cursor,
                playback_paused,
                err_fn,
            ),
            sample_format => anyhow::bail!("unsupported output sample format: {sample_format:?}"),
        }
    }

    fn build_queue_stream(&self, sample_queue: Arc<Mutex<VecDeque<f32>>>) -> Result<cpal::Stream> {
        let err_fn = |err| eprintln!("output stream error: {err}");
        match self.sample_format {
            cpal::SampleFormat::F32 => build_output_queue_stream::<f32>(
                &self.device,
                &self.stream_config,
                sample_queue,
                err_fn,
            ),
            cpal::SampleFormat::F64 => build_output_queue_stream::<f64>(
                &self.device,
                &self.stream_config,
                sample_queue,
                err_fn,
            ),
            cpal::SampleFormat::I8 => build_output_queue_stream::<i8>(
                &self.device,
                &self.stream_config,
                sample_queue,
                err_fn,
            ),
            cpal::SampleFormat::I16 => build_output_queue_stream::<i16>(
                &self.device,
                &self.stream_config,
                sample_queue,
                err_fn,
            ),
            cpal::SampleFormat::I32 => build_output_queue_stream::<i32>(
                &self.device,
                &self.stream_config,
                sample_queue,
                err_fn,
            ),
            cpal::SampleFormat::I64 => build_output_queue_stream::<i64>(
                &self.device,
                &self.stream_config,
                sample_queue,
                err_fn,
            ),
            cpal::SampleFormat::U8 => build_output_queue_stream::<u8>(
                &self.device,
                &self.stream_config,
                sample_queue,
                err_fn,
            ),
            cpal::SampleFormat::U16 => build_output_queue_stream::<u16>(
                &self.device,
                &self.stream_config,
                sample_queue,
                err_fn,
            ),
            cpal::SampleFormat::U32 => build_output_queue_stream::<u32>(
                &self.device,
                &self.stream_config,
                sample_queue,
                err_fn,
            ),
            cpal::SampleFormat::U64 => build_output_queue_stream::<u64>(
                &self.device,
                &self.stream_config,
                sample_queue,
                err_fn,
            ),
            sample_format => anyhow::bail!("unsupported output sample format: {sample_format:?}"),
        }
    }
}

#[cfg(feature = "audio-cpal")]
pub(crate) fn prepare_audio_playback(
    frames: &[AudioFrame],
    source: &str,
) -> Result<PreparedAudioPlayback> {
    let Some(first_frame) = frames.first() else {
        anyhow::bail!("no audio frames available for playback from {source}");
    };
    let sample_rate = first_frame.sample_rate_hz;
    let channels = first_frame.channels;
    anyhow::ensure!(
        sample_rate > 0,
        "audio from {source} has invalid sample rate"
    );
    anyhow::ensure!(
        channels > 0,
        "audio from {source} has invalid channel count"
    );

    let total_samples: usize = frames.iter().map(|frame| frame.samples.len()).sum();
    let mut audio_samples = Vec::with_capacity(total_samples);
    for frame in frames {
        anyhow::ensure!(
            frame.sample_rate_hz == sample_rate,
            "audio from {source} changed sample rate mid-stream ({} -> {})",
            sample_rate,
            frame.sample_rate_hz
        );
        anyhow::ensure!(
            frame.channels == channels,
            "audio from {source} changed channel count mid-stream ({} -> {})",
            channels,
            frame.channels
        );
        audio_samples.extend_from_slice(&frame.samples);
    }

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow::anyhow!("no default output device available"))?;
    let device_name = device
        .name()
        .unwrap_or_else(|_| "<unknown output device>".to_string());
    let output_config = select_output_config(&device, sample_rate, channels)?;
    let output_sample_rate = output_config.sample_rate_hz;
    let output_channels = output_config.channels;
    let audio_samples = convert_audio_samples(
        &audio_samples,
        sample_rate,
        channels,
        output_sample_rate,
        output_channels,
    );
    anyhow::ensure!(
        !audio_samples.is_empty(),
        "audio from {source} had no samples after output conversion"
    );

    Ok(PreparedAudioPlayback {
        device,
        stream_config: output_config.stream_config,
        sample_format: output_config.sample_format,
        device_name,
        sample_rate_hz: output_sample_rate,
        channels: output_channels,
        samples: Arc::new(audio_samples),
    })
}

#[cfg(feature = "audio-cpal")]
fn convert_frames_for_output(
    frames: &[AudioFrame],
    source: &str,
    target_sample_rate_hz: u32,
    target_channels: u16,
) -> Result<Vec<f32>> {
    let Some(first_frame) = frames.first() else {
        return Ok(Vec::new());
    };
    let sample_rate = first_frame.sample_rate_hz;
    let channels = first_frame.channels;
    anyhow::ensure!(
        sample_rate > 0,
        "audio from {source} has invalid sample rate"
    );
    anyhow::ensure!(
        channels > 0,
        "audio from {source} has invalid channel count"
    );

    let total_samples: usize = frames.iter().map(|frame| frame.samples.len()).sum();
    let mut audio_samples = Vec::with_capacity(total_samples);
    for frame in frames {
        anyhow::ensure!(
            frame.sample_rate_hz == sample_rate,
            "audio from {source} changed sample rate mid-stream ({} -> {})",
            sample_rate,
            frame.sample_rate_hz
        );
        anyhow::ensure!(
            frame.channels == channels,
            "audio from {source} changed channel count mid-stream ({} -> {})",
            channels,
            frame.channels
        );
        audio_samples.extend_from_slice(&frame.samples);
    }

    Ok(convert_audio_samples(
        &audio_samples,
        sample_rate,
        channels,
        target_sample_rate_hz,
        target_channels,
    ))
}

#[cfg(feature = "audio-cpal")]
pub(crate) fn play_audio_frames(frames: &[AudioFrame], source: &str) -> Result<()> {
    let playback = prepare_audio_playback(frames, source)?;
    let playback_cursor = Arc::new(AtomicUsize::new(0));
    let playback_paused = Arc::new(AtomicBool::new(false));
    let done_threshold = playback.sample_count();
    let stream = playback.build_stream(Arc::clone(&playback_cursor), playback_paused)?;
    stream
        .play()
        .with_context(|| format!("failed to start playback on {}", playback.device_name))?;

    while playback_cursor.load(Ordering::Relaxed) < done_threshold {
        std::thread::sleep(Duration::from_millis(10));
    }
    std::thread::sleep(Duration::from_millis(20));
    drop(stream);

    let audio_duration = playback.duration();
    println!(
        "Played with {}: {} Hz, {} channel(s), {:.2}s from {source}",
        playback.device_name,
        playback.sample_rate_hz,
        playback.channels,
        audio_duration.as_secs_f64(),
    );

    Ok(())
}

#[cfg(feature = "audio-cpal")]
pub(crate) fn play_audio_frame_stream(
    frame_rx: crossbeam_channel::Receiver<Vec<AudioFrame>>,
    source: &str,
) -> Result<()> {
    let first_frames = frame_rx
        .recv()
        .with_context(|| format!("no audio frames available for playback from {source}"))?;
    let playback = prepare_audio_playback(&first_frames, source)?;
    let sample_queue = Arc::new(Mutex::new(VecDeque::from(
        playback.samples.as_ref().clone(),
    )));
    let stream = playback.build_queue_stream(Arc::clone(&sample_queue))?;
    stream
        .play()
        .with_context(|| format!("failed to start playback on {}", playback.device_name))?;

    let mut total_samples = playback.sample_count();
    for frames in frame_rx {
        let samples =
            convert_frames_for_output(&frames, source, playback.sample_rate_hz, playback.channels)?;
        if samples.is_empty() {
            continue;
        }
        total_samples += samples.len();
        sample_queue
            .lock()
            .expect("audio sample queue poisoned")
            .extend(samples);
    }

    while !sample_queue
        .lock()
        .expect("audio sample queue poisoned")
        .is_empty()
    {
        std::thread::sleep(Duration::from_millis(10));
    }
    std::thread::sleep(Duration::from_millis(20));
    drop(stream);

    let audio_duration =
        playback_duration(total_samples, playback.sample_rate_hz, playback.channels);
    println!(
        "Played with {}: {} Hz, {} channel(s), {:.2}s from {source}",
        playback.device_name,
        playback.sample_rate_hz,
        playback.channels,
        audio_duration.as_secs_f64(),
    );

    Ok(())
}

#[cfg(feature = "audio-cpal")]
struct OutputConfig {
    sample_format: cpal::SampleFormat,
    sample_rate_hz: u32,
    channels: u16,
    stream_config: cpal::StreamConfig,
}

#[cfg(feature = "audio-cpal")]
fn select_output_config(
    device: &cpal::Device,
    sample_rate: u32,
    channels: u16,
) -> Result<OutputConfig> {
    if let Ok(default_config) = device.default_output_config() {
        return Ok(output_config_from_supported(default_config));
    }

    let candidates = device
        .supported_output_configs()
        .context("failed to list output stream configs")?
        .collect::<Vec<_>>();
    let desired_rate = cpal::SampleRate(sample_rate);
    let selected = candidates
        .iter()
        .find(|config| {
            config.channels() == channels
                && config.min_sample_rate() <= desired_rate
                && desired_rate <= config.max_sample_rate()
        })
        .or_else(|| candidates.first())
        .ok_or_else(|| anyhow::anyhow!("no output stream configs available"))?;
    let selected_rate = if selected.min_sample_rate() <= desired_rate
        && desired_rate <= selected.max_sample_rate()
    {
        desired_rate
    } else {
        selected.max_sample_rate()
    };
    Ok(output_config_from_supported(
        (*selected).with_sample_rate(selected_rate),
    ))
}

#[cfg(feature = "audio-cpal")]
fn output_config_from_supported(config: cpal::SupportedStreamConfig) -> OutputConfig {
    let sample_format = config.sample_format();
    let sample_rate_hz = config.sample_rate().0;
    let channels = config.channels();
    let stream_config = config.config();
    OutputConfig {
        sample_format,
        sample_rate_hz,
        channels,
        stream_config,
    }
}

#[cfg(feature = "audio-cpal")]
fn convert_audio_samples(
    samples: &[f32],
    source_sample_rate_hz: u32,
    source_channels: u16,
    target_sample_rate_hz: u32,
    target_channels: u16,
) -> Vec<f32> {
    let mut converted = convert_channels(samples, source_channels, target_channels);
    if source_sample_rate_hz != target_sample_rate_hz {
        converted = resample_interleaved(
            &converted,
            source_sample_rate_hz,
            target_sample_rate_hz,
            target_channels,
        );
    }
    converted
}

#[cfg(feature = "audio-cpal")]
fn convert_channels(samples: &[f32], source_channels: u16, target_channels: u16) -> Vec<f32> {
    if source_channels == target_channels {
        return samples.to_vec();
    }

    if target_channels == 1 {
        return mix_to_mono(samples, source_channels);
    }

    let source_channel_count = usize::from(source_channels).max(1);
    let target_channel_count = usize::from(target_channels).max(1);
    if source_channel_count == 1 {
        let mut converted = Vec::with_capacity(samples.len().saturating_mul(target_channel_count));
        for sample in samples {
            for _ in 0..target_channel_count {
                converted.push(*sample);
            }
        }
        return converted;
    }

    let mut converted = Vec::with_capacity(
        samples
            .len()
            .saturating_div(source_channel_count)
            .saturating_mul(target_channel_count),
    );
    for frame in samples.chunks_exact(source_channel_count) {
        for channel_idx in 0..target_channel_count {
            converted.push(frame[channel_idx.min(source_channel_count - 1)]);
        }
    }
    converted
}

#[cfg(feature = "audio-cpal")]
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

#[cfg(feature = "audio-cpal")]
fn resample_interleaved(
    samples: &[f32],
    source_rate_hz: u32,
    target_rate_hz: u32,
    channels: u16,
) -> Vec<f32> {
    let channel_count = usize::from(channels).max(1);
    let frame_count = samples.len() / channel_count;
    if samples.is_empty() || frame_count == 0 || source_rate_hz == target_rate_hz {
        return samples.to_vec();
    }

    let output_frame_count = ((frame_count as f64 * f64::from(target_rate_hz))
        / f64::from(source_rate_hz))
    .round() as usize;
    let mut output = Vec::with_capacity(output_frame_count.saturating_mul(channel_count));
    let source_step = f64::from(source_rate_hz) / f64::from(target_rate_hz);

    for output_frame_idx in 0..output_frame_count {
        let source_pos = output_frame_idx as f64 * source_step;
        let left_frame_idx = source_pos.floor() as usize;
        let right_frame_idx = (left_frame_idx + 1).min(frame_count - 1);
        let fraction = (source_pos - left_frame_idx as f64) as f32;
        for channel_idx in 0..channel_count {
            let left = samples[left_frame_idx * channel_count + channel_idx];
            let right = samples[right_frame_idx * channel_count + channel_idx];
            output.push(left + (right - left) * fraction);
        }
    }

    output
}

#[cfg(feature = "audio-cpal")]
fn playback_duration(total_samples: usize, sample_rate: u32, channels: u16) -> Duration {
    let sample_frames = total_samples as f64 / f64::from(channels);
    Duration::from_secs_f64(sample_frames / f64::from(sample_rate))
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "audio-cpal")]
    use super::playback_duration;
    #[cfg(feature = "audio-cpal")]
    use std::time::Duration;

    #[cfg(feature = "audio-cpal")]
    #[test]
    fn playback_duration_uses_channels_and_rate() {
        assert_eq!(playback_duration(96_000, 48_000, 2), Duration::from_secs(1));
    }
}
