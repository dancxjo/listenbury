use crate::cli::{PlayWavCommand, RecordWavCommand};
use anyhow::Result;

#[cfg(feature = "audio-cpal")]
use anyhow::Context;
#[cfg(feature = "audio-cpal")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(feature = "audio-cpal")]
use cpal::{FromSample, Sample, SizedSample, SupportedStreamConfigRange};
#[cfg(feature = "audio-cpal")]
use listenbury::audio::frame::AudioFrame;
#[cfg(feature = "audio-cpal")]
use listenbury::audio::{read_wav_frames, write_wav};
#[cfg(feature = "audio-cpal")]
use listenbury::time::ExactTimestamp;
#[cfg(feature = "audio-cpal")]
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
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
pub(crate) fn play_audio_frames(frames: &[AudioFrame], source: &str) -> Result<()> {
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
    let supported = select_output_config(&device, sample_rate, channels)?;
    let stream_config = supported
        .with_sample_rate(cpal::SampleRate(sample_rate))
        .config();

    let playback_cursor = Arc::new(AtomicUsize::new(0));
    let samples = Arc::new(audio_samples);
    let done_threshold = samples.len();
    let err_fn = |err| eprintln!("output stream error: {err}");
    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => build_output_stream::<f32>(
            &device,
            &stream_config,
            Arc::clone(&samples),
            Arc::clone(&playback_cursor),
            err_fn,
        )?,
        cpal::SampleFormat::F64 => build_output_stream::<f64>(
            &device,
            &stream_config,
            Arc::clone(&samples),
            Arc::clone(&playback_cursor),
            err_fn,
        )?,
        cpal::SampleFormat::I8 => build_output_stream::<i8>(
            &device,
            &stream_config,
            Arc::clone(&samples),
            Arc::clone(&playback_cursor),
            err_fn,
        )?,
        cpal::SampleFormat::I16 => build_output_stream::<i16>(
            &device,
            &stream_config,
            Arc::clone(&samples),
            Arc::clone(&playback_cursor),
            err_fn,
        )?,
        cpal::SampleFormat::I32 => build_output_stream::<i32>(
            &device,
            &stream_config,
            Arc::clone(&samples),
            Arc::clone(&playback_cursor),
            err_fn,
        )?,
        cpal::SampleFormat::I64 => build_output_stream::<i64>(
            &device,
            &stream_config,
            Arc::clone(&samples),
            Arc::clone(&playback_cursor),
            err_fn,
        )?,
        cpal::SampleFormat::U8 => build_output_stream::<u8>(
            &device,
            &stream_config,
            Arc::clone(&samples),
            Arc::clone(&playback_cursor),
            err_fn,
        )?,
        cpal::SampleFormat::U16 => build_output_stream::<u16>(
            &device,
            &stream_config,
            Arc::clone(&samples),
            Arc::clone(&playback_cursor),
            err_fn,
        )?,
        cpal::SampleFormat::U32 => build_output_stream::<u32>(
            &device,
            &stream_config,
            Arc::clone(&samples),
            Arc::clone(&playback_cursor),
            err_fn,
        )?,
        cpal::SampleFormat::U64 => build_output_stream::<u64>(
            &device,
            &stream_config,
            Arc::clone(&samples),
            Arc::clone(&playback_cursor),
            err_fn,
        )?,
        sample_format => anyhow::bail!("unsupported output sample format: {sample_format:?}"),
    };
    stream
        .play()
        .with_context(|| format!("failed to start playback on {device_name}"))?;

    while playback_cursor.load(Ordering::Relaxed) < done_threshold {
        std::thread::sleep(Duration::from_millis(10));
    }
    std::thread::sleep(Duration::from_millis(20));
    drop(stream);

    let audio_duration = playback_duration(total_samples, sample_rate, channels);
    println!(
        "Played with {device_name}: {} Hz, {channels} channel(s), {:.2}s from {source}",
        sample_rate,
        audio_duration.as_secs_f64(),
    );

    Ok(())
}

#[cfg(feature = "audio-cpal")]
fn select_output_config(
    device: &cpal::Device,
    sample_rate: u32,
    channels: u16,
) -> Result<SupportedStreamConfigRange> {
    let mut candidates = device
        .supported_output_configs()
        .context("failed to list output stream configs")?;
    let desired_rate = cpal::SampleRate(sample_rate);
    candidates
        .find(|config| {
            config.channels() == channels
                && config.min_sample_rate() <= desired_rate
                && desired_rate <= config.max_sample_rate()
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "default output device does not support {} Hz with {} channel(s)",
                sample_rate,
                channels
            )
        })
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
