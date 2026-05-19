use serde::{Deserialize, Serialize};

use crate::audio::frame::AudioFrame;

const DEFAULT_MIN_DB: f32 = -96.0;
const DEFAULT_MAX_DB: f32 = 0.0;
const ENERGY_WINDOW_MS: f32 = 20.0;
const ENERGY_HOP_MS: f32 = 10.0;
const ENERGY_DB_FLOOR: f32 = -120.0;
const FORMANT_WINDOW_MS: f32 = 25.0;
const FORMANT_HOP_MS: f32 = 10.0;
const FORMANT_FFT_SIZE: usize = 2048;
const FORMANT_MIN_HZ: f32 = 90.0;
const FORMANT_MAX_HZ: f32 = 5_500.0;
const FORMANT_MIN_SEPARATION_HZ: f32 = 250.0;
const FORMANT_MAX_COUNT: usize = 4;
const EPSILON: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcousticAnalysis {
    pub sample_rate: u32,
    pub sample_count: usize,
    pub duration_ms: f32,
    pub spectrogram: SpectrogramAnalysis,
    pub energy_envelope: EnergyEnvelope,
    pub energy_landmarks: EnergyLandmarks,
    pub formant_tracks: FormantTracks,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpectrogramAnalysis {
    pub sample_rate: u32,
    pub sample_count: usize,
    pub duration_ms: f32,
    pub db_scale: bool,
    pub min_db: f32,
    pub max_db: f32,
    pub levels: Vec<SpectrogramLevel>,
    pub analysis_mode: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpectrogramLevel {
    pub id: String,
    pub sample_rate: u32,
    pub window_name: String,
    pub window_size: usize,
    pub hop_size: usize,
    pub hop_ms: f32,
    pub fft_size: usize,
    pub bin_count: usize,
    pub bin_hz: f32,
    pub nyquist_hz: f32,
    pub db_scale: bool,
    pub min_value: f32,
    pub max_value: f32,
    pub frame_duration_ms: f32,
    pub frame_count: usize,
    pub sample_count: usize,
    pub reused_frame_count: usize,
    pub frames: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyEnvelope {
    pub window_ms: f32,
    pub hop_ms: f32,
    pub frames: Vec<EnergyFrame>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyFrame {
    pub frame_start_ms: u64,
    pub frame_end_ms: u64,
    pub rms_energy: f32,
    pub peak_energy: f32,
    pub dbfs: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyLandmarks {
    pub onsets: Vec<u64>,
    pub offsets: Vec<u64>,
    pub valleys: Vec<u64>,
    pub silences: Vec<EnergySilence>,
    pub peaks: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnergySilence {
    pub start_ms: u64,
    pub end_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FormantTracks {
    pub window_ms: f32,
    pub hop_ms: f32,
    pub method: String,
    pub frames: Vec<FormantFrame>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FormantFrame {
    pub frame_start_ms: u64,
    pub frame_end_ms: u64,
    pub rms_energy: f32,
    pub formants: Vec<FormantEstimate>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FormantEstimate {
    pub label: String,
    pub frequency_hz: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bandwidth_hz: Option<f32>,
    pub amplitude_db: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy)]
struct SpectrogramLevelConfig {
    id: &'static str,
    window_size: usize,
    hop_size: usize,
    fft_size: usize,
}

pub fn analyze_audio_frames(frames: &[AudioFrame]) -> Option<AcousticAnalysis> {
    let (samples, sample_rate) = audio_frames_to_mono_samples(frames)?;
    Some(analyze_mono_samples(&samples, sample_rate))
}

pub fn analyze_mono_samples(samples: &[f32], sample_rate: u32) -> AcousticAnalysis {
    let spectrogram = analyze_spectrogram_samples(samples, sample_rate);
    let energy_envelope = build_energy_envelope(samples, sample_rate);
    let energy_landmarks = detect_energy_landmarks(&energy_envelope);
    let formant_tracks = track_formants(samples, sample_rate);
    AcousticAnalysis {
        sample_rate,
        sample_count: samples.len(),
        duration_ms: duration_ms(samples.len(), sample_rate),
        spectrogram,
        energy_envelope,
        energy_landmarks,
        formant_tracks,
    }
}

fn audio_frames_to_mono_samples(frames: &[AudioFrame]) -> Option<(Vec<f32>, u32)> {
    let first = frames
        .iter()
        .find(|frame| frame.sample_rate_hz > 0 && frame.channels > 0)?;
    let sample_rate = first.sample_rate_hz;
    let channels = first.channels as usize;
    let mut samples = Vec::new();
    for frame in frames {
        if frame.sample_rate_hz != sample_rate || frame.channels as usize != channels {
            continue;
        }
        let per_channel_samples = frame.samples.len() / channels;
        for sample_index in 0..per_channel_samples {
            let mut mono = 0.0f32;
            for channel in 0..channels {
                mono += frame.samples[sample_index * channels + channel];
            }
            samples.push(mono / channels as f32);
        }
    }
    if samples.is_empty() {
        None
    } else {
        Some((samples, sample_rate))
    }
}

fn analyze_spectrogram_samples(samples: &[f32], sample_rate: u32) -> SpectrogramAnalysis {
    let levels = default_spectrogram_levels(sample_rate)
        .into_iter()
        .map(|level| build_spectrogram_level(samples, sample_rate, level))
        .collect();
    SpectrogramAnalysis {
        sample_rate,
        sample_count: samples.len(),
        duration_ms: duration_ms(samples.len(), sample_rate),
        db_scale: true,
        min_db: DEFAULT_MIN_DB,
        max_db: DEFAULT_MAX_DB,
        levels,
        analysis_mode: "rust-full".to_string(),
    }
}

fn default_spectrogram_levels(sample_rate: u32) -> Vec<SpectrogramLevelConfig> {
    let sample_rate = sample_rate as f32;
    vec![
        SpectrogramLevelConfig {
            id: "overview",
            window_size: 2048,
            hop_size: (sample_rate * 0.02).round().max(1.0) as usize,
            fft_size: 2048,
        },
        SpectrogramLevelConfig {
            id: "detail",
            window_size: 1024,
            hop_size: (sample_rate * 0.005).round().max(1.0) as usize,
            fft_size: 1024,
        },
        SpectrogramLevelConfig {
            id: "fine",
            window_size: 512,
            hop_size: (sample_rate * 0.0025).round().max(1.0) as usize,
            fft_size: 512,
        },
    ]
}

fn build_spectrogram_level(
    samples: &[f32],
    sample_rate: u32,
    config: SpectrogramLevelConfig,
) -> SpectrogramLevel {
    let frame_count = frame_count_for_sample_count(samples.len(), config.hop_size);
    let window = hann_window(config.window_size);
    let frames = (0..frame_count)
        .map(|frame_index| {
            analyze_spectral_frame(samples, frame_index * config.hop_size, config, &window)
        })
        .collect::<Vec<_>>();
    let bin_count = config.fft_size / 2 + 1;
    let hop_ms = config.hop_size as f32 / sample_rate as f32 * 1000.0;
    SpectrogramLevel {
        id: config.id.to_string(),
        sample_rate,
        window_name: "hann".to_string(),
        window_size: config.window_size,
        hop_size: config.hop_size,
        hop_ms,
        fft_size: config.fft_size,
        bin_count,
        bin_hz: sample_rate as f32 / config.fft_size as f32,
        nyquist_hz: sample_rate as f32 / 2.0,
        db_scale: true,
        min_value: DEFAULT_MIN_DB,
        max_value: DEFAULT_MAX_DB,
        frame_duration_ms: hop_ms,
        frame_count,
        sample_count: samples.len(),
        reused_frame_count: 0,
        frames,
    }
}

fn analyze_spectral_frame(
    samples: &[f32],
    start_sample: usize,
    config: SpectrogramLevelConfig,
    window: &[f64],
) -> Vec<f32> {
    let mut real = vec![0.0f64; config.fft_size];
    let mut imag = vec![0.0f64; config.fft_size];
    for index in 0..config.window_size {
        real[index] = f64::from(*samples.get(start_sample + index).unwrap_or(&0.0)) * window[index];
    }
    fft_in_place(&mut real, &mut imag);

    let bin_count = config.fft_size / 2 + 1;
    let mut bins = Vec::with_capacity(bin_count);
    for index in 0..bin_count {
        let magnitude = real[index].hypot(imag[index]) / config.window_size.max(1) as f64;
        let db = (20.0 * (magnitude + EPSILON).log10()) as f32;
        bins.push(db.clamp(DEFAULT_MIN_DB, DEFAULT_MAX_DB));
    }
    bins
}

fn build_energy_envelope(samples: &[f32], sample_rate: u32) -> EnergyEnvelope {
    let window_samples = ((sample_rate as f32 * ENERGY_WINDOW_MS) / 1000.0)
        .round()
        .max(1.0) as usize;
    let hop_samples = ((sample_rate as f32 * ENERGY_HOP_MS) / 1000.0)
        .round()
        .max(1.0) as usize;
    let mut frames = Vec::new();
    for frame_start in (0..samples.len()).step_by(hop_samples) {
        let frame_end = samples.len().min(frame_start + window_samples);
        if frame_end <= frame_start {
            continue;
        }
        let mut rms_squared_sum = 0.0f32;
        let mut peak = 0.0f32;
        let mut count = 0usize;
        for sample in &samples[frame_start..frame_end] {
            let abs = sample.abs();
            peak = peak.max(abs);
            rms_squared_sum += sample * sample;
            count += 1;
        }
        let rms = if count > 0 {
            (rms_squared_sum / count as f32).sqrt()
        } else {
            0.0
        };
        let dbfs = if rms > 0.0 {
            20.0 * rms.log10()
        } else {
            ENERGY_DB_FLOOR
        };
        frames.push(EnergyFrame {
            frame_start_ms: ((frame_start as u64 * 1000) / sample_rate as u64),
            frame_end_ms: (((frame_end as u64 * 1000) / sample_rate as u64)
                .max(((frame_start as u64 + 1) * 1000) / sample_rate as u64)),
            rms_energy: rms,
            peak_energy: peak,
            dbfs,
        });
    }
    EnergyEnvelope {
        window_ms: ENERGY_WINDOW_MS,
        hop_ms: ENERGY_HOP_MS,
        frames,
    }
}

fn detect_energy_landmarks(envelope: &EnergyEnvelope) -> EnergyLandmarks {
    let frames = &envelope.frames;
    if frames.is_empty() {
        return EnergyLandmarks {
            onsets: Vec::new(),
            offsets: Vec::new(),
            valleys: Vec::new(),
            silences: Vec::new(),
            peaks: Vec::new(),
        };
    }

    let energies = frames
        .iter()
        .map(|frame| frame.rms_energy)
        .collect::<Vec<_>>();
    let max_energy = energies.iter().copied().fold(0.0f32, f32::max);
    let mut sorted = energies.clone();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let noise_floor = sorted.get(sorted.len() / 20).copied().unwrap_or(0.0);
    let silence_threshold = (max_energy * 0.45).min((max_energy * 0.08).max(noise_floor * 1.35));
    let onset_rise_threshold = max_energy * 0.07;
    let offset_fall_threshold = max_energy * 0.07;

    let mut landmarks = EnergyLandmarks {
        onsets: Vec::new(),
        offsets: Vec::new(),
        valleys: Vec::new(),
        silences: Vec::new(),
        peaks: Vec::new(),
    };
    let mut silence_start = None;
    for index in 0..frames.len() {
        let current = energies[index];
        let previous = if index > 0 {
            energies[index - 1]
        } else {
            current
        };
        let next = energies.get(index + 1).copied().unwrap_or(current);
        let center_ms = (frames[index].frame_start_ms + frames[index].frame_end_ms) / 2;

        if current <= silence_threshold {
            silence_start.get_or_insert(frames[index].frame_start_ms);
        } else if let Some(start_ms) = silence_start.take() {
            landmarks.silences.push(EnergySilence {
                start_ms,
                end_ms: frames[index].frame_start_ms,
            });
        }

        if index == 0 || index == frames.len() - 1 {
            continue;
        }
        if current >= previous
            && current >= next
            && current >= (silence_threshold * 1.5).max(noise_floor * 1.8)
        {
            landmarks.peaks.push(center_ms);
        }
        if current <= previous && current <= next && current <= (previous + next) * 0.55 {
            landmarks.valleys.push(center_ms);
        }
        if current - previous >= onset_rise_threshold
            && current > silence_threshold
            && previous <= silence_threshold * 1.25
        {
            landmarks.onsets.push(frames[index].frame_start_ms);
        }
        if previous - current >= offset_fall_threshold
            && previous > silence_threshold
            && current <= silence_threshold * 1.25
        {
            landmarks.offsets.push(frames[index].frame_start_ms);
        }
    }
    if let Some(start_ms) = silence_start {
        if let Some(last) = frames.last() {
            landmarks.silences.push(EnergySilence {
                start_ms,
                end_ms: last.frame_end_ms,
            });
        }
    }
    landmarks
}

fn track_formants(samples: &[f32], sample_rate: u32) -> FormantTracks {
    let window_samples = ((sample_rate as f32 * FORMANT_WINDOW_MS) / 1000.0)
        .round()
        .max(1.0) as usize;
    let hop_samples = ((sample_rate as f32 * FORMANT_HOP_MS) / 1000.0)
        .round()
        .max(1.0) as usize;
    let fft_size = FORMANT_FFT_SIZE.max(window_samples.next_power_of_two());
    let window = hann_window(window_samples);
    let mut frames = Vec::new();

    for frame_start in (0..samples.len()).step_by(hop_samples) {
        let frame_end = samples.len().min(frame_start + window_samples);
        if frame_end <= frame_start {
            continue;
        }
        let rms = rms_energy(&samples[frame_start..frame_end]);
        let formants = estimate_frame_formants(
            samples,
            sample_rate,
            frame_start,
            frame_end,
            fft_size,
            &window,
            rms,
        );
        frames.push(FormantFrame {
            frame_start_ms: (frame_start as u64 * 1000) / sample_rate as u64,
            frame_end_ms: ((frame_end as u64 * 1000) / sample_rate as u64)
                .max(((frame_start as u64 + 1) * 1000) / sample_rate as u64),
            rms_energy: rms,
            formants,
        });
    }

    FormantTracks {
        window_ms: FORMANT_WINDOW_MS,
        hop_ms: FORMANT_HOP_MS,
        method: "smoothed-spectrum-peaks".to_string(),
        frames,
    }
}

fn estimate_frame_formants(
    samples: &[f32],
    sample_rate: u32,
    frame_start: usize,
    frame_end: usize,
    fft_size: usize,
    window: &[f64],
    rms: f32,
) -> Vec<FormantEstimate> {
    if rms <= 0.000_1 {
        return Vec::new();
    }

    let mut real = vec![0.0f64; fft_size];
    let mut imag = vec![0.0f64; fft_size];
    for (index, sample_index) in (frame_start..frame_end).enumerate() {
        real[index] = f64::from(samples[sample_index]) * window[index];
    }
    fft_in_place(&mut real, &mut imag);

    let bin_hz = sample_rate as f32 / fft_size as f32;
    let min_bin = ((FORMANT_MIN_HZ / bin_hz).floor() as usize).max(1);
    let max_bin =
        ((FORMANT_MAX_HZ.min(sample_rate as f32 / 2.0) / bin_hz).ceil() as usize).min(fft_size / 2);
    if max_bin <= min_bin + 2 {
        return Vec::new();
    }

    let magnitudes_db = (0..=fft_size / 2)
        .map(|bin| {
            let magnitude = real[bin].hypot(imag[bin]) / (frame_end - frame_start).max(1) as f64;
            (20.0 * (magnitude + EPSILON).log10()) as f32
        })
        .collect::<Vec<_>>();
    let smoothing_bins = ((150.0 / bin_hz).round() as usize).max(2);
    let smoothed = smooth_spectrum_db(&magnitudes_db, smoothing_bins);
    let noise_floor = percentile(&smoothed[min_bin..=max_bin], 0.20);

    let mut peaks = Vec::<FormantPeak>::new();
    for bin in (min_bin + 1)..max_bin {
        let value = smoothed[bin];
        if value <= smoothed[bin - 1] || value < smoothed[bin + 1] {
            continue;
        }
        let prominence_db = value - noise_floor;
        if prominence_db < 3.0 {
            continue;
        }
        peaks.push(FormantPeak {
            frequency_hz: bin as f32 * bin_hz,
            amplitude_db: value,
            bandwidth_hz: estimate_bandwidth_hz(&smoothed, bin, min_bin, max_bin, bin_hz),
            confidence: (prominence_db / 24.0).clamp(0.0, 1.0),
        });
    }

    peaks.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
    let mut selected = Vec::<FormantPeak>::new();
    for peak in peaks {
        if selected
            .iter()
            .any(|prior| (prior.frequency_hz - peak.frequency_hz).abs() < FORMANT_MIN_SEPARATION_HZ)
        {
            continue;
        }
        selected.push(peak);
        if selected.len() >= FORMANT_MAX_COUNT {
            break;
        }
    }
    selected.sort_by(|left, right| left.frequency_hz.total_cmp(&right.frequency_hz));
    selected
        .into_iter()
        .enumerate()
        .map(|(index, peak)| FormantEstimate {
            label: format!("F{}", index + 1),
            frequency_hz: peak.frequency_hz,
            bandwidth_hz: Some(peak.bandwidth_hz),
            amplitude_db: peak.amplitude_db,
            confidence: peak.confidence,
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct FormantPeak {
    frequency_hz: f32,
    amplitude_db: f32,
    bandwidth_hz: f32,
    confidence: f32,
}

fn rms_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_squares = samples.iter().map(|sample| sample * sample).sum::<f32>();
    (sum_squares / samples.len() as f32).sqrt()
}

fn smooth_spectrum_db(values: &[f32], radius: usize) -> Vec<f32> {
    let mut smoothed = Vec::with_capacity(values.len());
    for index in 0..values.len() {
        let start = index.saturating_sub(radius);
        let end = (index + radius).min(values.len().saturating_sub(1));
        let count = (end - start + 1) as f32;
        let sum = values[start..=end].iter().sum::<f32>();
        smoothed.push(sum / count);
    }
    smoothed
}

fn estimate_bandwidth_hz(
    spectrum_db: &[f32],
    peak_bin: usize,
    min_bin: usize,
    max_bin: usize,
    bin_hz: f32,
) -> f32 {
    let half_power_db = spectrum_db[peak_bin] - 3.0;
    let mut left = peak_bin;
    while left > min_bin && spectrum_db[left] > half_power_db {
        left -= 1;
    }
    let mut right = peak_bin;
    while right < max_bin && spectrum_db[right] > half_power_db {
        right += 1;
    }
    ((right.saturating_sub(left)).max(1) as f32 * bin_hz).max(bin_hz)
}

fn percentile(values: &[f32], percentile: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let index =
        ((sorted.len().saturating_sub(1)) as f32 * percentile.clamp(0.0, 1.0)).round() as usize;
    sorted[index]
}

fn frame_count_for_sample_count(sample_count: usize, hop_size: usize) -> usize {
    if sample_count == 0 {
        0
    } else {
        (sample_count - 1) / hop_size.max(1) + 1
    }
}

fn duration_ms(sample_count: usize, sample_rate: u32) -> f32 {
    if sample_rate == 0 {
        0.0
    } else {
        sample_count as f32 / sample_rate as f32 * 1000.0
    }
}

fn hann_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|index| {
            0.5 * (1.0
                - ((2.0 * std::f64::consts::PI * index as f64) / (size.max(2) - 1) as f64).cos())
        })
        .collect()
}

fn fft_in_place(real: &mut [f64], imag: &mut [f64]) {
    let size = real.len();
    let bits = size.trailing_zeros();
    for index in 0..size {
        let reversed = index.reverse_bits() >> (usize::BITS - bits);
        if reversed > index {
            real.swap(index, reversed);
            imag.swap(index, reversed);
        }
    }

    let mut step = 2;
    while step <= size {
        let half_step = step / 2;
        let angle_step = -2.0 * std::f64::consts::PI / step as f64;
        for offset in (0..size).step_by(step) {
            for pair in 0..half_step {
                let angle = angle_step * pair as f64;
                let cos = angle.cos();
                let sin = angle.sin();
                let even = offset + pair;
                let odd = even + half_step;
                let t_real = real[odd] * cos - imag[odd] * sin;
                let t_imag = real[odd] * sin + imag[odd] * cos;
                real[odd] = real[even] - t_real;
                imag[odd] = imag[even] - t_imag;
                real[even] += t_real;
                imag[even] += t_imag;
            }
        }
        step <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::ExactTimestamp;

    #[test]
    fn analyzes_spectrogram_and_energy_from_ingested_frames() {
        let sample_rate_hz = 16_000;
        let samples = (0..1600)
            .map(|index| {
                ((2.0 * std::f32::consts::PI * 440.0 * index as f32) / sample_rate_hz as f32).sin()
            })
            .collect::<Vec<_>>();
        let frames = vec![AudioFrame {
            captured_at: ExactTimestamp { unix_nanos: 0 },
            sample_rate_hz,
            channels: 1,
            samples,
            voice_signatures: Vec::new(),
        }];

        let analysis = analyze_audio_frames(&frames).expect("analysis");
        assert_eq!(analysis.sample_rate, sample_rate_hz);
        assert_eq!(analysis.spectrogram.levels.len(), 3);
        assert_eq!(analysis.spectrogram.levels[0].id, "overview");
        assert_eq!(analysis.spectrogram.levels[1].id, "detail");
        assert_eq!(analysis.spectrogram.levels[2].id, "fine");
        assert!(analysis.spectrogram.levels[0].hop_ms > analysis.spectrogram.levels[1].hop_ms);
        assert!(analysis.spectrogram.levels[1].hop_ms > analysis.spectrogram.levels[2].hop_ms);
        assert!(!analysis.energy_envelope.frames.is_empty());
        assert!(!analysis.formant_tracks.frames.is_empty());
        assert_eq!(analysis.formant_tracks.method, "smoothed-spectrum-peaks");
    }

    #[test]
    fn tracks_ordered_formant_estimates_in_acoustic_artifact() {
        let sample_rate_hz = 16_000;
        let samples = (0..3200)
            .map(|index| {
                let t = index as f32 / sample_rate_hz as f32;
                (2.0 * std::f32::consts::PI * 700.0 * t).sin() * 0.6
                    + (2.0 * std::f32::consts::PI * 1_250.0 * t).sin() * 0.35
                    + (2.0 * std::f32::consts::PI * 2_600.0 * t).sin() * 0.25
            })
            .collect::<Vec<_>>();

        let analysis = analyze_mono_samples(&samples, sample_rate_hz);
        let frame = analysis
            .formant_tracks
            .frames
            .iter()
            .find(|frame| !frame.formants.is_empty())
            .expect("at least one voiced frame should expose formant estimates");

        assert!(frame.formants.len() <= FORMANT_MAX_COUNT);
        for (index, formant) in frame.formants.iter().enumerate() {
            assert_eq!(formant.label, format!("F{}", index + 1));
            assert!(formant.frequency_hz >= FORMANT_MIN_HZ);
            assert!(formant.frequency_hz <= FORMANT_MAX_HZ);
            assert!(formant.confidence >= 0.0);
            assert!(formant.confidence <= 1.0);
            if let Some(previous) = index.checked_sub(1).and_then(|i| frame.formants.get(i)) {
                assert!(previous.frequency_hz < formant.frequency_hz);
            }
        }

        let json = serde_json::to_string(&analysis).expect("serialize analysis");
        assert!(json.contains("\"formantTracks\""));
    }
}
