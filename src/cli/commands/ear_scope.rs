use crate::cli::EarScopeCommand;
use crate::cli::resolve_vad_config;
use anyhow::{Context, Result};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SizedSample};
use listenbury::audio::capture::{
    boost_current_thread_for_capture, callback_sample_queue_capacity,
};
use listenbury::audio::{
    AudioFormat, MONO_CHANNELS, SampleKind, WHISPER_SAMPLE_RATE_HZ, WavExportOptions,
    WavSampleEncoding, analyze_mono_samples, normalize_interleaved_f32, write_wav_with_report,
};
use listenbury::event::HearingEvent;
use listenbury::hearing::breath::{BreathGroupConfig, BreathGroupId, BreathGroupSegmenter};
use listenbury::hearing::vad::{VadBackendKind, create_vad_backend_with_profile};
use listenbury::{AudioFrame, ExactTimestamp};
use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

const FRAME_SAMPLES: usize = 160;
const FRAME_MS: u64 = 10;
const TARGET_CHANNELS: u16 = MONO_CHANNELS;
const TARGET_SAMPLE_RATE_HZ: u32 = WHISPER_SAMPLE_RATE_HZ;

pub(crate) fn run_ear_scope(command: EarScopeCommand) -> Result<()> {
    anyhow::ensure!(
        command.duration > 0,
        "--duration must be greater than zero for ear-scope"
    );
    ensure_parent_dir(&command.output_wav)?;
    ensure_parent_dir(&command.output_png)?;
    ensure_parent_dir(&command.output_jsonl)?;

    let capture = capture_normalized_mic_frames(command.duration)?;
    anyhow::ensure!(
        !capture.frames.is_empty(),
        "ear-scope captured no 16 kHz frames from {}",
        capture.device_name
    );

    let vad_config = resolve_vad_config(command.vad, command.vad_profile.as_deref())?;
    let diagnostics = collect_diagnostics(
        &capture.frames,
        vad_config.backend,
        vad_config.profile,
        capture.dropped_in_callback,
    )?;

    write_wav_with_report(
        &command.output_wav,
        &capture.frames,
        WavExportOptions {
            sample_rate_hz: Some(TARGET_SAMPLE_RATE_HZ),
            channels: Some(TARGET_CHANNELS),
            sample_encoding: WavSampleEncoding::PcmI16,
        },
    )
    .with_context(|| format!("failed to write {}", command.output_wav.display()))?;
    write_diagnostics_jsonl(&command.output_jsonl, &diagnostics.frames)
        .with_context(|| format!("failed to write {}", command.output_jsonl.display()))?;
    render_ear_scope_png(&command.output_png, &capture.frames, &diagnostics)
        .with_context(|| format!("failed to write {}", command.output_png.display()))?;

    println!(
        "ear-scope captured {} frame(s) from {}: input={} Hz/{} channel(s)/{} -> 16000 Hz mono",
        capture.frames.len(),
        capture.device_name,
        capture.input_sample_rate_hz,
        capture.input_channels,
        capture.sample_format
    );
    if capture.dropped_in_callback > 0 {
        println!("callback_drops: {}", capture.dropped_in_callback);
    }
    println!("wav: {}", command.output_wav.display());
    println!("png: {}", command.output_png.display());
    println!("jsonl: {}", command.output_jsonl.display());

    Ok(())
}

struct EarScopeCapture {
    frames: Vec<AudioFrame>,
    device_name: String,
    input_sample_rate_hz: u32,
    input_channels: u16,
    sample_format: String,
    dropped_in_callback: usize,
}

fn capture_normalized_mic_frames(duration_secs: u64) -> Result<EarScopeCapture> {
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
    anyhow::ensure!(input_channels > 0, "{device_name} reported zero channels");

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

    println!(
        "ear-scope listening on {device_name}: {} Hz, {} channel(s), {sample_format:?}",
        input_sample_rate_hz, input_channels
    );
    boost_current_thread_for_capture("ear-scope");

    let target_frames = usize::try_from(duration_secs)
        .unwrap_or(usize::MAX / 100)
        .saturating_mul(100);
    let mut frames = Vec::with_capacity(target_frames);
    let mut pending = VecDeque::<f32>::new();
    let started_at = Instant::now();
    let deadline = started_at + Duration::from_secs(duration_secs);
    stream
        .play()
        .with_context(|| format!("failed to start capture from {device_name}"))?;
    while frames.len() < target_frames && Instant::now() < deadline + Duration::from_millis(250) {
        match sample_rx.recv_timeout(Duration::from_millis(20)) {
            Ok(sample) => pending.push_back(sample),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
        while let Ok(sample) = sample_rx.try_recv() {
            pending.push_back(sample);
        }
        drain_pending_capture_frames(
            &mut pending,
            &mut frames,
            target_frames,
            input_frame_samples,
            input_sample_rate_hz,
            input_channels,
        )?;
    }
    drop(stream);
    while frames.len() < target_frames {
        match sample_rx.try_recv() {
            Ok(sample) => pending.push_back(sample),
            Err(_) => break,
        }
        drain_pending_capture_frames(
            &mut pending,
            &mut frames,
            target_frames,
            input_frame_samples,
            input_sample_rate_hz,
            input_channels,
        )?;
    }
    if !frames.is_empty() && frames.len() < target_frames {
        frames.resize_with(target_frames, || AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: TARGET_SAMPLE_RATE_HZ,
            channels: TARGET_CHANNELS,
            samples: vec![0.0; FRAME_SAMPLES],
            voice_signatures: Vec::new(),
        });
    }

    Ok(EarScopeCapture {
        frames,
        device_name,
        input_sample_rate_hz,
        input_channels,
        sample_format: format!("{sample_format:?}"),
        dropped_in_callback: dropped_in_callback.load(Ordering::Relaxed),
    })
}

fn drain_pending_capture_frames(
    pending: &mut VecDeque<f32>,
    frames: &mut Vec<AudioFrame>,
    target_frames: usize,
    input_frame_samples: usize,
    input_sample_rate_hz: u32,
    input_channels: u16,
) -> Result<()> {
    while pending.len() >= input_frame_samples && frames.len() < target_frames {
        let mut raw_frame = Vec::with_capacity(input_frame_samples);
        for _ in 0..input_frame_samples {
            if let Some(sample) = pending.pop_front() {
                raw_frame.push(sample);
            }
        }
        let normalized = normalize_interleaved_f32(
            &raw_frame,
            AudioFormat::new(input_sample_rate_hz, input_channels, SampleKind::F32),
            AudioFormat::new(TARGET_SAMPLE_RATE_HZ, TARGET_CHANNELS, SampleKind::F32),
            "ear_scope_vad_input",
        )?;
        let mut samples = normalized.samples;
        if samples.len() < FRAME_SAMPLES {
            samples.resize(FRAME_SAMPLES, 0.0);
        } else if samples.len() > FRAME_SAMPLES {
            samples.truncate(FRAME_SAMPLES);
        }
        frames.push(AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: TARGET_SAMPLE_RATE_HZ,
            channels: TARGET_CHANNELS,
            samples,
            voice_signatures: Vec::new(),
        });
    }
    Ok(())
}

fn frame_samples_per_callback_frame(sample_rate_hz: u32, channels: u16) -> usize {
    let samples_per_channel = usize::try_from(sample_rate_hz / 100).unwrap_or(1).max(1);
    samples_per_channel.saturating_mul(usize::from(channels).max(1))
}

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

#[derive(Debug)]
struct EarScopeDiagnostics {
    frames: Vec<EarScopeFrameDiagnostic>,
    utterance_regions: Vec<UtteranceRegion>,
}

#[derive(Debug, Clone, Serialize)]
struct EarScopeFrameDiagnostic {
    kind: &'static str,
    timestamp: f64,
    timestamp_ms: u64,
    frame_index: u64,
    rms: f32,
    peak: f32,
    vad_backend: &'static str,
    vad_speech: bool,
    vad_speech_prob: f32,
    speech_state: &'static str,
    utterance_active: bool,
    events: Vec<String>,
    callback_drops: usize,
}

#[derive(Debug, Clone)]
struct UtteranceRegion {
    start_frame: usize,
    end_frame: usize,
    reason: String,
}

fn collect_diagnostics(
    frames: &[AudioFrame],
    backend: VadBackendKind,
    profile: Option<listenbury::VadProfile>,
    callback_drops: usize,
) -> Result<EarScopeDiagnostics> {
    let breath_config = profile
        .map(|profile| profile.breath_group_config())
        .unwrap_or_default();
    let mut vad = create_vad_backend_with_profile(backend, profile.as_ref())?;
    let mut segmenter = BreathGroupSegmenter::new(breath_config);
    let mut active_groups = HashMap::<BreathGroupId, usize>::new();
    let mut utterance_regions = Vec::new();
    let mut diagnostics = Vec::with_capacity(frames.len());

    for (index, frame) in frames.iter().enumerate() {
        let rms = rms(&frame.samples);
        let peak = peak(&frame.samples);
        let vad_result = vad.process_frame(frame)?;
        let hearing_events = segmenter.process(vad_result);
        let mut event_names = Vec::new();

        for event in hearing_events {
            match event {
                HearingEvent::SpeechStarted => event_names.push("speech_started".to_string()),
                HearingEvent::SpeechContinued { .. } => {
                    event_names.push("speech_continued".to_string())
                }
                HearingEvent::PauseStarted => event_names.push("pause_started".to_string()),
                HearingEvent::BreathGroupOpened { id } => {
                    let start_frame = breath_group_start_frame(index, breath_config);
                    active_groups.insert(id, start_frame);
                    event_names.push("utterance_start".to_string());
                }
                HearingEvent::BreathGroupClosed { id, reason } => {
                    let start_frame = active_groups.remove(&id).unwrap_or(index);
                    utterance_regions.push(UtteranceRegion {
                        start_frame,
                        end_frame: index.saturating_add(1),
                        reason: format!("{reason:?}").to_lowercase(),
                    });
                    event_names.push("utterance_end".to_string());
                }
            }
        }

        let timestamp_ms = (index as u64).saturating_mul(FRAME_MS);
        diagnostics.push(EarScopeFrameDiagnostic {
            kind: "ear_scope_frame",
            timestamp: timestamp_ms as f64 / 1000.0,
            timestamp_ms,
            frame_index: index as u64,
            rms,
            peak,
            vad_backend: backend.as_str(),
            vad_speech: vad_result.is_speech,
            vad_speech_prob: vad_result.speech_prob,
            speech_state: if vad_result.is_speech {
                "speech"
            } else {
                "silence"
            },
            utterance_active: !active_groups.is_empty(),
            events: event_names,
            callback_drops,
        });
    }

    for start_frame in active_groups.into_values() {
        utterance_regions.push(UtteranceRegion {
            start_frame,
            end_frame: frames.len(),
            reason: "capture_end".to_string(),
        });
    }

    Ok(EarScopeDiagnostics {
        frames: diagnostics,
        utterance_regions,
    })
}

fn breath_group_start_frame(index: usize, config: BreathGroupConfig) -> usize {
    index
        .saturating_add(1)
        .saturating_sub(config.open_after_speech_frames.max(1))
}

fn write_diagnostics_jsonl(path: &Path, frames: &[EarScopeFrameDiagnostic]) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    for frame in frames {
        serde_json::to_writer(&mut writer, frame)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq = samples.iter().map(|sample| sample * sample).sum::<f32>();
    (sum_sq / samples.len() as f32).sqrt()
}

fn peak(samples: &[f32]) -> f32 {
    samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0f32, f32::max)
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output directory {}", parent.display()))?;
    }
    Ok(())
}

fn render_ear_scope_png(
    path: &Path,
    frames: &[AudioFrame],
    diagnostics: &EarScopeDiagnostics,
) -> Result<()> {
    let samples = frames
        .iter()
        .flat_map(|frame| frame.samples.iter().copied())
        .collect::<Vec<_>>();
    let analysis = analyze_mono_samples(&samples, TARGET_SAMPLE_RATE_HZ);
    let spectrogram = analysis
        .spectrogram
        .levels
        .iter()
        .find(|level| level.id == "overview")
        .or_else(|| analysis.spectrogram.levels.first())
        .context("acoustic analysis produced no spectrogram levels")?;

    let duration_secs = samples.len() as f32 / TARGET_SAMPLE_RATE_HZ as f32;
    let width = ((duration_secs * 140.0).round() as usize).clamp(1000, 2000);
    let margin_left = 118usize;
    let margin_right = 24usize;
    let top = 22usize;
    let gap = 14usize;
    let waveform_h = 170usize;
    let rms_h = 112usize;
    let vad_h = 62usize;
    let utterance_h = 62usize;
    let spectrogram_h = 330usize;
    let bottom = 38usize;
    let height = top
        + waveform_h
        + gap
        + rms_h
        + gap
        + vad_h
        + gap
        + utterance_h
        + gap
        + spectrogram_h
        + bottom;
    let plot_x = margin_left;
    let plot_w = width.saturating_sub(margin_left + margin_right).max(1);

    let mut canvas = Canvas::new(width, height, rgb(250, 251, 252));
    let waveform_y = top;
    let rms_y = waveform_y + waveform_h + gap;
    let vad_y = rms_y + rms_h + gap;
    let utterance_y = vad_y + vad_h + gap;
    let spectrogram_y = utterance_y + utterance_h + gap;

    let panels = [
        ("WAVEFORM", waveform_y, waveform_h),
        ("RMS / ENERGY", rms_y, rms_h),
        ("VAD FRAMES", vad_y, vad_h),
        ("UTTERANCE REGIONS", utterance_y, utterance_h),
        ("SPECTROGRAM", spectrogram_y, spectrogram_h),
    ];
    for (label, y, h) in panels {
        draw_panel(&mut canvas, plot_x, y, plot_w, h, label, duration_secs);
    }

    draw_waveform(
        &mut canvas,
        &samples,
        plot_x,
        waveform_y,
        plot_w,
        waveform_h,
    );
    draw_rms_curve(
        &mut canvas,
        &diagnostics.frames,
        plot_x,
        rms_y,
        plot_w,
        rms_h,
    );
    draw_vad_frames(
        &mut canvas,
        &diagnostics.frames,
        plot_x,
        vad_y,
        plot_w,
        vad_h,
    );
    draw_utterance_regions(
        &mut canvas,
        &diagnostics.utterance_regions,
        frames.len(),
        plot_x,
        utterance_y,
        plot_w,
        utterance_h,
    );
    draw_spectrogram(
        &mut canvas,
        spectrogram,
        plot_x,
        spectrogram_y,
        plot_w,
        spectrogram_h,
    );
    draw_time_axis(
        &mut canvas,
        plot_x,
        spectrogram_y + spectrogram_h,
        plot_w,
        duration_secs,
    );

    canvas.write_png(path)
}

fn draw_panel(
    canvas: &mut Canvas,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    label: &str,
    duration_secs: f32,
) {
    canvas.fill_rect(x, y, w, h, rgb(255, 255, 255));
    canvas.stroke_rect(x, y, w, h, rgb(197, 203, 211));
    canvas.draw_text(14, y + 10, label, rgb(34, 42, 52), 2);
    let seconds = duration_secs.ceil().max(1.0) as usize;
    for second in 0..=seconds {
        let gx = x + ((second as f32 / seconds as f32) * w as f32).round() as usize;
        canvas.draw_vline(gx, y + 1, h.saturating_sub(2), rgb(235, 238, 242));
    }
}

fn draw_waveform(canvas: &mut Canvas, samples: &[f32], x: usize, y: usize, w: usize, h: usize) {
    if samples.is_empty() || w == 0 || h == 0 {
        return;
    }
    let max_abs = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0f32, f32::max)
        .max(0.02);
    let center_y = y + h / 2;
    canvas.draw_hline(x, center_y, w, rgb(217, 222, 228));
    for col in 0..w {
        let start = col * samples.len() / w;
        let end = ((col + 1) * samples.len() / w)
            .min(samples.len())
            .max(start + 1);
        let mut min_sample = 1.0f32;
        let mut max_sample = -1.0f32;
        for sample in &samples[start..end] {
            min_sample = min_sample.min(*sample);
            max_sample = max_sample.max(*sample);
        }
        let y1 = sample_to_y(max_sample, max_abs, y, h);
        let y2 = sample_to_y(min_sample, max_abs, y, h);
        canvas.draw_vline(
            x + col,
            y1.min(y2),
            y1.abs_diff(y2).max(1),
            rgb(29, 100, 126),
        );
    }
}

fn sample_to_y(sample: f32, max_abs: f32, y: usize, h: usize) -> usize {
    let normalized = (sample / max_abs).clamp(-1.0, 1.0);
    let center = y as f32 + h as f32 / 2.0;
    let amp = h as f32 * 0.45;
    (center - normalized * amp)
        .round()
        .clamp(y as f32, (y + h - 1) as f32) as usize
}

fn draw_rms_curve(
    canvas: &mut Canvas,
    diagnostics: &[EarScopeFrameDiagnostic],
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) {
    if diagnostics.is_empty() {
        return;
    }
    let max_rms = diagnostics
        .iter()
        .map(|frame| frame.rms)
        .fold(0.0f32, f32::max)
        .max(0.01);
    let baseline = y + h - 12;
    canvas.draw_hline(x, baseline, w, rgb(217, 222, 228));
    let mut prev = None;
    for (index, frame) in diagnostics.iter().enumerate() {
        let px = x + (index * w / diagnostics.len()).min(w.saturating_sub(1));
        let normalized = (frame.rms / max_rms).clamp(0.0, 1.0);
        let py = baseline.saturating_sub((normalized * (h as f32 - 22.0)).round() as usize);
        if let Some((prev_x, prev_y)) = prev {
            canvas.draw_line(prev_x, prev_y, px, py, rgb(127, 80, 34));
        }
        prev = Some((px, py));
    }
}

fn draw_vad_frames(
    canvas: &mut Canvas,
    diagnostics: &[EarScopeFrameDiagnostic],
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) {
    if diagnostics.is_empty() {
        return;
    }
    for (index, frame) in diagnostics.iter().enumerate() {
        let rel_x1 = index * w / diagnostics.len();
        let rel_x2 = ((index + 1) * w / diagnostics.len()).max(rel_x1 + 1);
        let x1 = x + rel_x1;
        let x2 = x + rel_x2;
        let color = if frame.vad_speech {
            rgb(38, 149, 107)
        } else {
            rgb(226, 230, 235)
        };
        canvas.fill_rect(
            x1,
            y + 12,
            x2.saturating_sub(x1),
            h.saturating_sub(24),
            color,
        );
        if frame.events.iter().any(|event| event == "speech_started") {
            canvas.draw_vline(x1, y + 5, h.saturating_sub(10), rgb(22, 102, 73));
        }
    }
}

fn draw_utterance_regions(
    canvas: &mut Canvas,
    regions: &[UtteranceRegion],
    frame_count: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) {
    if frame_count == 0 {
        return;
    }
    for region in regions {
        let x1 = x + region.start_frame.min(frame_count) * w / frame_count;
        let x2 = x + region.end_frame.min(frame_count) * w / frame_count;
        let color = if region.reason == "capture_end" {
            rgb(212, 129, 58)
        } else {
            rgb(72, 112, 168)
        };
        canvas.fill_rect(
            x1,
            y + 12,
            x2.saturating_sub(x1).max(1),
            h.saturating_sub(24),
            color,
        );
    }
}

fn draw_spectrogram(
    canvas: &mut Canvas,
    level: &listenbury::audio::acoustic::SpectrogramLevel,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) {
    if level.frames.is_empty() || level.bin_count == 0 {
        return;
    }
    for col in 0..w {
        let frame_index = (col * level.frames.len() / w).min(level.frames.len() - 1);
        let bins = &level.frames[frame_index];
        for row in 0..h {
            let bin = ((h - 1 - row) * bins.len() / h).min(bins.len() - 1);
            let value = bins[bin];
            canvas.put_pixel(x + col, y + row, spectrogram_color(value));
        }
    }
}

fn spectrogram_color(db: f32) -> Rgb {
    let t = ((db + 96.0) / 96.0).clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.35 {
        let p = t / 0.35;
        (
            lerp(18.0, 34.0, p),
            lerp(24.0, 75.0, p),
            lerp(34.0, 91.0, p),
        )
    } else if t < 0.72 {
        let p = (t - 0.35) / 0.37;
        (
            lerp(34.0, 224.0, p),
            lerp(75.0, 146.0, p),
            lerp(91.0, 67.0, p),
        )
    } else {
        let p = (t - 0.72) / 0.28;
        (
            lerp(224.0, 255.0, p),
            lerp(146.0, 246.0, p),
            lerp(67.0, 226.0, p),
        )
    };
    rgb(r as u8, g as u8, b as u8)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn draw_time_axis(canvas: &mut Canvas, x: usize, y: usize, w: usize, duration_secs: f32) {
    let seconds = duration_secs.ceil().max(1.0) as usize;
    for second in 0..=seconds {
        let gx = x + ((second as f32 / seconds as f32) * w as f32).round() as usize;
        canvas.draw_vline(gx, y, 6, rgb(91, 101, 112));
        canvas.draw_text(
            gx.saturating_sub(8),
            y + 12,
            &format!("{second}s"),
            rgb(91, 101, 112),
            1,
        );
    }
}

#[derive(Debug, Clone, Copy)]
struct Rgb {
    r: u8,
    g: u8,
    b: u8,
}

const fn rgb(r: u8, g: u8, b: u8) -> Rgb {
    Rgb { r, g, b }
}

struct Canvas {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

impl Canvas {
    fn new(width: usize, height: usize, fill: Rgb) -> Self {
        let mut canvas = Self {
            width,
            height,
            pixels: vec![0; width.saturating_mul(height).saturating_mul(3)],
        };
        canvas.fill_rect(0, 0, width, height, fill);
        canvas
    }

    fn put_pixel(&mut self, x: usize, y: usize, color: Rgb) {
        if x >= self.width || y >= self.height {
            return;
        }
        let index = (y * self.width + x) * 3;
        self.pixels[index] = color.r;
        self.pixels[index + 1] = color.g;
        self.pixels[index + 2] = color.b;
    }

    fn fill_rect(&mut self, x: usize, y: usize, w: usize, h: usize, color: Rgb) {
        for row in y..y.saturating_add(h).min(self.height) {
            for col in x..x.saturating_add(w).min(self.width) {
                self.put_pixel(col, row, color);
            }
        }
    }

    fn stroke_rect(&mut self, x: usize, y: usize, w: usize, h: usize, color: Rgb) {
        if w == 0 || h == 0 {
            return;
        }
        self.draw_hline(x, y, w, color);
        self.draw_hline(x, y + h - 1, w, color);
        self.draw_vline(x, y, h, color);
        self.draw_vline(x + w - 1, y, h, color);
    }

    fn draw_hline(&mut self, x: usize, y: usize, w: usize, color: Rgb) {
        for col in x..x.saturating_add(w).min(self.width) {
            self.put_pixel(col, y, color);
        }
    }

    fn draw_vline(&mut self, x: usize, y: usize, h: usize, color: Rgb) {
        for row in y..y.saturating_add(h).min(self.height) {
            self.put_pixel(x, row, color);
        }
    }

    fn draw_line(&mut self, x0: usize, y0: usize, x1: usize, y1: usize, color: Rgb) {
        let mut x0 = x0 as isize;
        let mut y0 = y0 as isize;
        let x1 = x1 as isize;
        let y1 = y1 as isize;
        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -(y1 - y0).abs();
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            if x0 >= 0 && y0 >= 0 {
                self.put_pixel(x0 as usize, y0 as usize, color);
            }
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
    }

    fn draw_text(&mut self, x: usize, y: usize, text: &str, color: Rgb, scale: usize) {
        let mut cursor = x;
        for ch in text.chars() {
            draw_glyph(
                self,
                cursor,
                y,
                ch.to_ascii_uppercase(),
                color,
                scale.max(1),
            );
            cursor += 6 * scale.max(1);
        }
    }

    fn write_png(&self, path: &Path) -> Result<()> {
        let bytes = encode_png_rgb(self.width, self.height, &self.pixels)?;
        std::fs::write(path, bytes)
            .with_context(|| format!("failed to create PNG at {}", path.display()))?;
        Ok(())
    }
}

fn draw_glyph(canvas: &mut Canvas, x: usize, y: usize, ch: char, color: Rgb, scale: usize) {
    let glyph = glyph_5x7(ch);
    for (row, bits) in glyph.iter().enumerate() {
        for (col, byte) in bits.as_bytes().iter().enumerate() {
            if *byte == b'1' {
                canvas.fill_rect(x + col * scale, y + row * scale, scale, scale, color);
            }
        }
    }
}

fn glyph_5x7(ch: char) -> [&'static str; 7] {
    match ch {
        'A' => [
            "01110", "10001", "10001", "11111", "10001", "10001", "10001",
        ],
        'B' => [
            "11110", "10001", "10001", "11110", "10001", "10001", "11110",
        ],
        'C' => [
            "01111", "10000", "10000", "10000", "10000", "10000", "01111",
        ],
        'D' => [
            "11110", "10001", "10001", "10001", "10001", "10001", "11110",
        ],
        'E' => [
            "11111", "10000", "10000", "11110", "10000", "10000", "11111",
        ],
        'F' => [
            "11111", "10000", "10000", "11110", "10000", "10000", "10000",
        ],
        'G' => [
            "01111", "10000", "10000", "10011", "10001", "10001", "01111",
        ],
        'H' => [
            "10001", "10001", "10001", "11111", "10001", "10001", "10001",
        ],
        'I' => [
            "11111", "00100", "00100", "00100", "00100", "00100", "11111",
        ],
        'J' => [
            "00111", "00010", "00010", "00010", "10010", "10010", "01100",
        ],
        'K' => [
            "10001", "10010", "10100", "11000", "10100", "10010", "10001",
        ],
        'L' => [
            "10000", "10000", "10000", "10000", "10000", "10000", "11111",
        ],
        'M' => [
            "10001", "11011", "10101", "10101", "10001", "10001", "10001",
        ],
        'N' => [
            "10001", "11001", "10101", "10011", "10001", "10001", "10001",
        ],
        'O' => [
            "01110", "10001", "10001", "10001", "10001", "10001", "01110",
        ],
        'P' => [
            "11110", "10001", "10001", "11110", "10000", "10000", "10000",
        ],
        'Q' => [
            "01110", "10001", "10001", "10001", "10101", "10010", "01101",
        ],
        'R' => [
            "11110", "10001", "10001", "11110", "10100", "10010", "10001",
        ],
        'S' => [
            "01111", "10000", "10000", "01110", "00001", "00001", "11110",
        ],
        'T' => [
            "11111", "00100", "00100", "00100", "00100", "00100", "00100",
        ],
        'U' => [
            "10001", "10001", "10001", "10001", "10001", "10001", "01110",
        ],
        'V' => [
            "10001", "10001", "10001", "10001", "10001", "01010", "00100",
        ],
        'W' => [
            "10001", "10001", "10001", "10101", "10101", "10101", "01010",
        ],
        'X' => [
            "10001", "10001", "01010", "00100", "01010", "10001", "10001",
        ],
        'Y' => [
            "10001", "10001", "01010", "00100", "00100", "00100", "00100",
        ],
        'Z' => [
            "11111", "00001", "00010", "00100", "01000", "10000", "11111",
        ],
        '0' => [
            "01110", "10001", "10011", "10101", "11001", "10001", "01110",
        ],
        '1' => [
            "00100", "01100", "00100", "00100", "00100", "00100", "01110",
        ],
        '2' => [
            "01110", "10001", "00001", "00010", "00100", "01000", "11111",
        ],
        '3' => [
            "11110", "00001", "00001", "01110", "00001", "00001", "11110",
        ],
        '4' => [
            "00010", "00110", "01010", "10010", "11111", "00010", "00010",
        ],
        '5' => [
            "11111", "10000", "10000", "11110", "00001", "00001", "11110",
        ],
        '6' => [
            "01110", "10000", "10000", "11110", "10001", "10001", "01110",
        ],
        '7' => [
            "11111", "00001", "00010", "00100", "01000", "01000", "01000",
        ],
        '8' => [
            "01110", "10001", "10001", "01110", "10001", "10001", "01110",
        ],
        '9' => [
            "01110", "10001", "10001", "01111", "00001", "00001", "01110",
        ],
        '/' => [
            "00001", "00010", "00010", "00100", "01000", "01000", "10000",
        ],
        '-' => [
            "00000", "00000", "00000", "11111", "00000", "00000", "00000",
        ],
        ':' => [
            "00000", "00100", "00100", "00000", "00100", "00100", "00000",
        ],
        '.' => [
            "00000", "00000", "00000", "00000", "00000", "01100", "01100",
        ],
        _ => [
            "00000", "00000", "00000", "00000", "00000", "00000", "00000",
        ],
    }
}

fn encode_png_rgb(width: usize, height: usize, rgb_pixels: &[u8]) -> Result<Vec<u8>> {
    anyhow::ensure!(width > 0 && height > 0, "PNG dimensions must be non-zero");
    anyhow::ensure!(
        rgb_pixels.len() == width.saturating_mul(height).saturating_mul(3),
        "RGB pixel buffer length does not match PNG dimensions"
    );

    let mut scanlines = Vec::with_capacity(height.saturating_mul(width.saturating_mul(3) + 1));
    for row in 0..height {
        scanlines.push(0);
        let start = row * width * 3;
        scanlines.extend_from_slice(&rgb_pixels[start..start + width * 3]);
    }

    let mut out = Vec::new();
    out.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(width as u32).to_be_bytes());
    ihdr.extend_from_slice(&(height as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    write_png_chunk(&mut out, b"IHDR", &ihdr)?;
    let idat = zlib_store_blocks(&scanlines);
    write_png_chunk(&mut out, b"IDAT", &idat)?;
    write_png_chunk(&mut out, b"IEND", &[])?;
    Ok(out)
}

fn write_png_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) -> Result<()> {
    let len = u32::try_from(data.len()).context("PNG chunk exceeds u32 length")?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(kind.len() + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
    Ok(())
}

fn zlib_store_blocks(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + data.len() / 65_535 * 5 + 6);
    out.extend_from_slice(&[0x78, 0x01]);
    let mut offset = 0;
    while offset < data.len() {
        let remaining = data.len() - offset;
        let block_len = remaining.min(65_535);
        let is_last = offset + block_len >= data.len();
        out.push(if is_last { 0x01 } else { 0x00 });
        let len = block_len as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(&data[offset..offset + block_len]);
        offset += block_len;
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

fn adler32(data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65_521;
    let mut a = 1u32;
    let mut b = 0u32;
    for byte in data {
        a = (a + u32::from(*byte)) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }
    (b << 16) | a
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for byte in data {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_encoder_writes_png_signature() {
        let pixels = vec![255u8; 2 * 2 * 3];
        let bytes = encode_png_rgb(2, 2, &pixels).expect("PNG encoding should work");
        assert_eq!(&bytes[..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
        assert!(bytes.windows(4).any(|chunk| chunk == b"IHDR"));
        assert!(bytes.windows(4).any(|chunk| chunk == b"IDAT"));
        assert!(bytes.windows(4).any(|chunk| chunk == b"IEND"));
    }

    #[test]
    fn breath_group_start_frame_backdates_opening_frame() {
        let config = BreathGroupConfig {
            open_after_speech_frames: 3,
            close_after_silence_frames: 10,
            max_group_frames: None,
        };
        assert_eq!(breath_group_start_frame(2, config), 0);
        assert_eq!(breath_group_start_frame(10, config), 8);
    }
}
