use crate::cli::{
    RecordWavCommand, VadCalibrateCommand, VadCalibrateRoomCommand, VadCommand, VadCompareCommand,
};
use anyhow::{Context, Result};
use listenbury::audio::read_wav_as_audio_frames;
use listenbury::config::VadProfile;
use listenbury::hearing::vad::{VadBackendKind, create_vad_backend};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::io::Write;
use std::path::{Path, PathBuf};

const CALIBRATION_FRAME_SAMPLES: usize = 160;
const DEFAULT_HANGOVER_MS: u64 = 180;
const DEFAULT_MIN_SPEECH_MS: u64 = 120;
// Heuristic mapping from energy-over-noise to a bounded pseudo-probability.
const SPEECH_PROB_ENERGY_OVER_NOISE_NORMALIZER: f32 = 3.0;
// Threshold picks are intentionally conservative: require being above both
// high-percentile RMS and a multiple of measured noise-floor percentile.
const RECOMMENDED_RMS_THRESHOLD_QUANTILE: f32 = 0.95;
const RECOMMENDED_NOISE_FLOOR_QUANTILE: f32 = 0.9;
const RECOMMENDED_NOISE_GATE_MULTIPLIER: f32 = 2.0;

pub(crate) fn run_vad(command: VadCommand) -> Result<()> {
    match command {
        VadCommand::CalibrateRoom(command) => run_vad_calibrate_room(command),
        VadCommand::Calibrate(command) => run_vad_calibrate(command),
        VadCommand::Compare(command) => run_vad_compare(command),
    }
}

fn run_vad_calibrate_room(command: VadCalibrateRoomCommand) -> Result<()> {
    let frames = load_audio_or_capture(command.audio, command.seconds)?;
    let report = build_single_backend_report(command.vad.as_backend_kind(), &frames, None)?;
    emit_report(
        &CalibrationReport {
            mode: "calibrate-room",
            audio_frames: frames.len(),
            room_baseline: report.baseline.clone(),
            backend_reports: vec![report],
        },
        command.json.as_deref(),
        command.toml.as_deref(),
    )
}

fn run_vad_calibrate(command: VadCalibrateCommand) -> Result<()> {
    let frames = read_fixture_frames(&command.audio)?;
    let labels = read_labels(command.labels.as_deref())?;
    let report =
        build_single_backend_report(command.vad.as_backend_kind(), &frames, labels.as_deref())?;
    emit_report(
        &CalibrationReport {
            mode: "calibrate",
            audio_frames: frames.len(),
            room_baseline: report.baseline.clone(),
            backend_reports: vec![report],
        },
        command.json.as_deref(),
        command.toml.as_deref(),
    )
}

fn run_vad_compare(command: VadCompareCommand) -> Result<()> {
    let frames = read_fixture_frames(&command.audio)?;
    let labels = read_labels(command.labels.as_deref())?;
    let mut backend_reports = Vec::with_capacity(command.backends.len());
    for backend in command.backends {
        backend_reports.push(build_single_backend_report(
            backend.as_backend_kind(),
            &frames,
            labels.as_deref(),
        )?);
    }
    let room_baseline = backend_reports
        .first()
        .map(|report| report.baseline.clone())
        .unwrap_or_default();
    emit_report(
        &CalibrationReport {
            mode: "compare",
            audio_frames: frames.len(),
            room_baseline,
            backend_reports,
        },
        command.json.as_deref(),
        None,
    )
}

fn emit_report(
    report: &CalibrationReport,
    json_path: Option<&Path>,
    toml_path: Option<&Path>,
) -> Result<()> {
    let json = serde_json::to_string_pretty(report)?;
    println!("{json}");

    if let Some(path) = json_path {
        write_text(path, &json)?;
        println!("wrote calibration JSON: {}", path.display());
    }

    if let Some(profile) = report
        .backend_reports
        .first()
        .and_then(|backend| backend.recommended_profile.clone())
    {
        let toml = profile.to_toml();
        println!("{toml}");
        if let Some(path) = toml_path {
            write_text(path, &toml)?;
            println!("wrote VAD profile TOML: {}", path.display());
        }
    }

    Ok(())
}

fn build_single_backend_report(
    backend: VadBackendKind,
    frames: &[listenbury::AudioFrame],
    labels: Option<&[LabelInterval]>,
) -> Result<BackendCalibrationReport> {
    let baseline = compute_room_baseline(frames)?;
    match create_vad_backend(backend) {
        Ok(mut detector) => {
            let mut frame_stats = Vec::with_capacity(frames.len());
            let mut t_ms = 0u64;
            for frame in frames {
                let result = detector.process_frame(frame)?;
                frame_stats.push(FrameStat {
                    start_ms: t_ms,
                    end_ms: t_ms.saturating_add(frame_duration_ms(frame)),
                    speech: result.is_speech,
                    speech_prob: result.speech_prob,
                });
                t_ms = frame_stats.last().map(|frame| frame.end_ms).unwrap_or(t_ms);
            }
            let metrics = labels
                .map(|label_intervals| compute_labeled_metrics(&frame_stats, label_intervals));
            let profile = recommend_profile(backend, &baseline);
            Ok(BackendCalibrationReport {
                backend: backend.as_str().to_string(),
                available: true,
                error: None,
                confidence: confidence_summary(&frame_stats),
                metrics,
                baseline,
                recommended_profile: Some(profile),
            })
        }
        Err(error) => Ok(BackendCalibrationReport {
            backend: backend.as_str().to_string(),
            available: false,
            error: Some(error.to_string()),
            confidence: ConfidenceSummary::default(),
            metrics: None,
            baseline,
            recommended_profile: None,
        }),
    }
}

fn compute_room_baseline(frames: &[listenbury::AudioFrame]) -> Result<RoomBaseline> {
    let analysis = listenbury::analyze_audio_frames(frames)
        .context("failed to compute acoustic analysis for VAD calibration")?;
    let feature_frames = &analysis.feature_stream.frames;
    let mut rms = Vec::with_capacity(feature_frames.len());
    let mut peak = Vec::with_capacity(feature_frames.len());
    let mut zcr = Vec::with_capacity(feature_frames.len());
    let mut speech_prob = Vec::with_capacity(feature_frames.len());
    let mut noise_floor = Vec::with_capacity(feature_frames.len());
    let mut band_low = Vec::with_capacity(feature_frames.len());
    let mut band_low_mid = Vec::with_capacity(feature_frames.len());
    let mut band_mid_high = Vec::with_capacity(feature_frames.len());
    let mut band_high = Vec::with_capacity(feature_frames.len());

    for frame in feature_frames {
        rms.push(frame.rms_energy);
        peak.push(frame.peak_amplitude);
        zcr.push(frame.zero_crossing_rate);
        speech_prob.push(
            (frame.energy_over_noise / SPEECH_PROB_ENERGY_OVER_NOISE_NORMALIZER).clamp(0.0, 1.0),
        );
        noise_floor.push(frame.noise_floor_rms);
        band_low.push(frame.band_energy_db[0]);
        band_low_mid.push(frame.band_energy_db[1]);
        band_mid_high.push(frame.band_energy_db[2]);
        band_high.push(frame.band_energy_db[3]);
    }

    Ok(RoomBaseline {
        rms: Distribution::from_values(&rms),
        peak: Distribution::from_values(&peak),
        zero_crossing_rate: Distribution::from_values(&zcr),
        speech_probability: Distribution::from_values(&speech_prob),
        noise_floor_rms: Distribution::from_values(&noise_floor),
        spectral_bands_db: SpectralBandSummary {
            low: Distribution::from_values(&band_low),
            low_mid: Distribution::from_values(&band_low_mid),
            mid_high: Distribution::from_values(&band_mid_high),
            high: Distribution::from_values(&band_high),
        },
        recommended_initial_thresholds: {
            let noise_gate_threshold = quantile(&noise_floor, RECOMMENDED_NOISE_FLOOR_QUANTILE)
                * RECOMMENDED_NOISE_GATE_MULTIPLIER;
            RecommendedThresholds {
                rms_threshold: quantile(&rms, RECOMMENDED_RMS_THRESHOLD_QUANTILE)
                    .max(noise_gate_threshold),
                noise_floor: quantile(&noise_floor, 0.5),
                hangover_ms: DEFAULT_HANGOVER_MS,
                min_speech_ms: DEFAULT_MIN_SPEECH_MS,
            }
        },
    })
}

fn recommend_profile(backend: VadBackendKind, baseline: &RoomBaseline) -> VadProfile {
    VadProfile {
        backend,
        rms_threshold: baseline.recommended_initial_thresholds.rms_threshold,
        hangover_ms: baseline.recommended_initial_thresholds.hangover_ms,
        min_speech_ms: baseline.recommended_initial_thresholds.min_speech_ms,
        noise_floor: baseline.recommended_initial_thresholds.noise_floor,
    }
}

fn confidence_summary(frame_stats: &[FrameStat]) -> ConfidenceSummary {
    let probs = frame_stats
        .iter()
        .map(|frame| frame.speech_prob)
        .collect::<Vec<_>>();
    ConfidenceSummary {
        all: Distribution::from_values(&probs),
        speech_frames: Distribution::from_values(
            &frame_stats
                .iter()
                .filter(|frame| frame.speech)
                .map(|frame| frame.speech_prob)
                .collect::<Vec<_>>(),
        ),
        non_speech_frames: Distribution::from_values(
            &frame_stats
                .iter()
                .filter(|frame| !frame.speech)
                .map(|frame| frame.speech_prob)
                .collect::<Vec<_>>(),
        ),
    }
}

fn compute_labeled_metrics(frame_stats: &[FrameStat], labels: &[LabelInterval]) -> LabeledMetrics {
    let mut tp = 0usize;
    let mut tn = 0usize;
    let mut fp = 0usize;
    let mut fn_count = 0usize;
    let mut noise_duration_ms = 0u64;

    let truth = frame_stats
        .iter()
        .map(|frame| {
            let midpoint = frame.start_ms.saturating_add(frame.end_ms) / 2;
            match label_at(labels, midpoint) {
                Some(LabelKind::Speech) => {
                    if frame.speech {
                        tp += 1;
                    } else {
                        fn_count += 1;
                    }
                    true
                }
                Some(LabelKind::SilenceOrNoise) => {
                    noise_duration_ms = noise_duration_ms
                        .saturating_add(frame.end_ms.saturating_sub(frame.start_ms));
                    if frame.speech {
                        fp += 1;
                    } else {
                        tn += 1;
                    }
                    false
                }
                None => false,
            }
        })
        .collect::<Vec<_>>();

    let onsets = transition_points(frame_stats, &truth, false, true);
    let offsets = transition_points(frame_stats, &truth, true, false);
    let onset_latency_ms = latency_summary(&onsets, frame_stats, true);
    let offset_latency_ms = latency_summary(&offsets, frame_stats, false);
    let noise_trigger_rate_hz = if noise_duration_ms == 0 {
        0.0
    } else {
        fp as f32 / (noise_duration_ms as f32 / 1000.0)
    };

    LabeledMetrics {
        true_positives: tp,
        true_negatives: tn,
        false_positives: fp,
        false_negatives: fn_count,
        noise_trigger_rate_hz,
        onset_latency_ms,
        offset_latency_ms,
    }
}

fn transition_points(
    frame_stats: &[FrameStat],
    truth: &[bool],
    previous: bool,
    current: bool,
) -> Vec<u64> {
    let mut transitions = Vec::new();
    for (index, frame) in frame_stats.iter().enumerate() {
        let prev = if index == 0 {
            previous
        } else {
            truth[index - 1]
        };
        if prev == previous && truth[index] == current {
            transitions.push(frame.start_ms);
        }
    }
    transitions
}

fn latency_summary(
    transitions_ms: &[u64],
    frame_stats: &[FrameStat],
    target_speech: bool,
) -> Option<LatencySummary> {
    if transitions_ms.is_empty() {
        return None;
    }
    let mut latencies = Vec::new();
    for transition in transitions_ms {
        if let Some(frame) = frame_stats
            .iter()
            .find(|frame| frame.start_ms >= *transition && frame.speech == target_speech)
        {
            latencies.push(frame.start_ms.saturating_sub(*transition) as f32);
        }
    }
    if latencies.is_empty() {
        return None;
    }
    Some(LatencySummary {
        mean_ms: latencies.iter().sum::<f32>() / latencies.len() as f32,
        p95_ms: quantile(&latencies, 0.95),
    })
}

fn read_labels(path: Option<&Path>) -> Result<Option<Vec<LabelInterval>>> {
    let Some(path) = path else {
        return Ok(None);
    };
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read labels file {}", path.display()))?;
    let labels = serde_json::from_str::<Vec<LabelInterval>>(&raw)
        .with_context(|| format!("failed to parse labels file {}", path.display()))?;
    Ok(Some(labels))
}

fn label_at(labels: &[LabelInterval], t_ms: u64) -> Option<LabelKind> {
    labels
        .iter()
        .find(|label| label.start_ms <= t_ms && t_ms < label.end_ms)
        .map(|label| label.label)
}

fn read_fixture_frames(path: &Path) -> Result<Vec<listenbury::AudioFrame>> {
    read_wav_as_audio_frames(path, CALIBRATION_FRAME_SAMPLES)
        .with_context(|| format!("failed to read WAV {}", path.display()))
}

fn load_audio_or_capture(
    audio: Option<PathBuf>,
    seconds: u64,
) -> Result<Vec<listenbury::AudioFrame>> {
    if let Some(path) = audio {
        return read_fixture_frames(&path);
    }
    let temp_path = std::env::temp_dir().join(format!(
        "listenbury-vad-calibration-room-{}-{}.wav",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    ));
    super::run_record_wav(RecordWavCommand {
        output_wav: temp_path.clone(),
        seconds,
    })?;
    let frames = read_fixture_frames(&temp_path);
    let _ = std::fs::remove_file(&temp_path);
    frames
}

fn frame_duration_ms(frame: &listenbury::AudioFrame) -> u64 {
    if frame.sample_rate_hz == 0 || frame.channels == 0 {
        return 0;
    }
    let samples_per_channel = frame.samples.len() as f64 / f64::from(frame.channels);
    ((samples_per_channel / f64::from(frame.sample_rate_hz)) * 1000.0).round() as u64
}

fn write_text(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }
    let mut writer = std::io::BufWriter::new(
        std::fs::File::create(path)
            .with_context(|| format!("failed to create {}", path.display()))?,
    );
    writer.write_all(content.as_bytes())?;
    writer.flush()?;
    Ok(())
}

fn quantile(values: &[f32], q: f32) -> f32 {
    let mut sorted = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if sorted.is_empty() {
        return 0.0;
    }
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let clamped_q = q.clamp(0.0, 1.0);
    let idx = ((sorted.len() - 1) as f32 * clamped_q).round() as usize;
    sorted[idx]
}

#[derive(Debug, Serialize)]
struct CalibrationReport {
    mode: &'static str,
    audio_frames: usize,
    room_baseline: RoomBaseline,
    backend_reports: Vec<BackendCalibrationReport>,
}

#[derive(Debug, Clone, Serialize, Default)]
struct RoomBaseline {
    rms: Distribution,
    peak: Distribution,
    spectral_bands_db: SpectralBandSummary,
    zero_crossing_rate: Distribution,
    speech_probability: Distribution,
    noise_floor_rms: Distribution,
    recommended_initial_thresholds: RecommendedThresholds,
}

#[derive(Debug, Clone, Serialize, Default)]
struct SpectralBandSummary {
    low: Distribution,
    low_mid: Distribution,
    mid_high: Distribution,
    high: Distribution,
}

#[derive(Debug, Clone, Serialize, Default)]
struct RecommendedThresholds {
    rms_threshold: f32,
    hangover_ms: u64,
    min_speech_ms: u64,
    noise_floor: f32,
}

#[derive(Debug, Clone, Serialize, Default)]
struct Distribution {
    min: f32,
    p50: f32,
    p90: f32,
    p95: f32,
    p99: f32,
    max: f32,
    mean: f32,
}

impl Distribution {
    fn from_values(values: &[f32]) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        Self {
            min: quantile(values, 0.0),
            p50: quantile(values, 0.5),
            p90: quantile(values, 0.9),
            p95: quantile(values, 0.95),
            p99: quantile(values, 0.99),
            max: quantile(values, 1.0),
            mean: values.iter().sum::<f32>() / values.len() as f32,
        }
    }
}

#[derive(Debug, Serialize)]
struct BackendCalibrationReport {
    backend: String,
    available: bool,
    error: Option<String>,
    confidence: ConfidenceSummary,
    metrics: Option<LabeledMetrics>,
    baseline: RoomBaseline,
    recommended_profile: Option<VadProfile>,
}

#[derive(Debug, Clone, Serialize, Default)]
struct ConfidenceSummary {
    all: Distribution,
    speech_frames: Distribution,
    non_speech_frames: Distribution,
}

#[derive(Debug, Clone, Serialize)]
struct LabeledMetrics {
    true_positives: usize,
    true_negatives: usize,
    false_positives: usize,
    false_negatives: usize,
    noise_trigger_rate_hz: f32,
    onset_latency_ms: Option<LatencySummary>,
    offset_latency_ms: Option<LatencySummary>,
}

#[derive(Debug, Clone, Serialize)]
struct LatencySummary {
    mean_ms: f32,
    p95_ms: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum LabelKind {
    Speech,
    SilenceOrNoise,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LabelInterval {
    start_ms: u64,
    end_ms: u64,
    label: LabelKind,
}

#[derive(Debug, Clone)]
struct FrameStat {
    start_ms: u64,
    end_ms: u64,
    speech: bool,
    speech_prob: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    const EXPECTED_NOISE_TRIGGER_RATE_HZ_MIN: f32 = 39.0;

    #[test]
    fn distribution_computes_expected_summary() {
        let summary = Distribution::from_values(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(summary.min, 0.0);
        assert_eq!(summary.max, 4.0);
        assert_eq!(summary.p50, 2.0);
        assert_eq!(summary.p90, 4.0);
        assert_eq!(summary.mean, 2.0);
    }

    #[test]
    fn labeled_metrics_compute_fp_fn_and_latency() {
        let frames = vec![
            FrameStat {
                start_ms: 0,
                end_ms: 10,
                speech: false,
                speech_prob: 0.1,
            },
            FrameStat {
                start_ms: 10,
                end_ms: 20,
                speech: true,
                speech_prob: 0.9,
            },
            FrameStat {
                start_ms: 20,
                end_ms: 30,
                speech: true,
                speech_prob: 0.85,
            },
            FrameStat {
                start_ms: 30,
                end_ms: 40,
                speech: false,
                speech_prob: 0.2,
            },
            FrameStat {
                start_ms: 40,
                end_ms: 50,
                speech: true,
                speech_prob: 0.8,
            },
        ];
        let labels = vec![
            LabelInterval {
                start_ms: 0,
                end_ms: 10,
                label: LabelKind::SilenceOrNoise,
            },
            LabelInterval {
                start_ms: 10,
                end_ms: 35,
                label: LabelKind::Speech,
            },
            LabelInterval {
                start_ms: 35,
                end_ms: 50,
                label: LabelKind::SilenceOrNoise,
            },
        ];

        let metrics = compute_labeled_metrics(&frames, &labels);
        assert_eq!(metrics.true_positives, 2);
        assert_eq!(metrics.false_negatives, 1);
        assert_eq!(metrics.false_positives, 1);
        assert_eq!(metrics.true_negatives, 1);
        assert!(metrics.noise_trigger_rate_hz > EXPECTED_NOISE_TRIGGER_RATE_HZ_MIN);
        let onset = metrics.onset_latency_ms.expect("expected onset latency");
        assert_eq!(onset.mean_ms, 0.0);
    }
}
