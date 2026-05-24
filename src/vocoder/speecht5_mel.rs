use std::f64::consts::PI;

use anyhow::{Result, ensure};

use crate::acoustic::MelFrame;
use crate::vocoder::{
    LogCompression, MelConfig, MelNormalization, MelScale, SPEECHT5_HIFIGAN_MEL_CONTRACT,
};

const SPEECHT5_MEL_FLOOR: f64 = 1.0e-10;

#[derive(Debug, Clone, PartialEq)]
pub struct SpeechT5MelExtraction {
    pub mel: Vec<MelFrame>,
    pub sample_rate_hz: u32,
    pub hop_samples: usize,
    pub frame_length_samples: usize,
    pub fft_length_samples: usize,
}

pub fn extract_speecht5_log_mel(samples: &[f32]) -> Result<SpeechT5MelExtraction> {
    extract_log_mel_with_config(samples, &SPEECHT5_HIFIGAN_MEL_CONTRACT)
}

fn extract_log_mel_with_config(
    samples: &[f32],
    config: &MelConfig,
) -> Result<SpeechT5MelExtraction> {
    validate_speecht5_config(config)?;
    ensure!(
        !samples.is_empty(),
        "SpeechT5 mel extraction received empty audio"
    );

    let frame_length = config.win_length;
    let hop_length = config.hop_length;
    let fft_length = config.n_fft;
    let padded = if config.center {
        reflect_pad(samples, frame_length / 2)
    } else {
        samples.to_vec()
    };
    ensure!(
        padded.len() >= frame_length,
        "SpeechT5 mel extraction needs at least {frame_length} padded samples"
    );

    let window = periodic_hann(frame_length);
    let mel_filters = slaney_mel_filter_bank(config);
    let num_frames = 1 + (padded.len() - frame_length) / hop_length;
    let mut mel = Vec::with_capacity(num_frames);

    for frame_index in 0..num_frames {
        let start = frame_index * hop_length;
        let frame = &padded[start..start + frame_length];
        let spectrum = amplitude_spectrum(frame, &window, fft_length);
        let bins = (0..config.n_mels)
            .map(|mel_index| {
                let energy = spectrum
                    .iter()
                    .zip(&mel_filters[mel_index])
                    .map(|(magnitude, weight)| magnitude * weight)
                    .sum::<f64>()
                    .max(SPEECHT5_MEL_FLOOR);
                energy.log10() as f32
            })
            .collect::<Vec<_>>();
        mel.push(MelFrame { bins });
    }

    Ok(SpeechT5MelExtraction {
        mel,
        sample_rate_hz: config.sample_rate_hz,
        hop_samples: hop_length,
        frame_length_samples: frame_length,
        fft_length_samples: fft_length,
    })
}

fn validate_speecht5_config(config: &MelConfig) -> Result<()> {
    ensure!(
        config.sample_rate_hz == 16_000,
        "SpeechT5 mel extraction requires 16 kHz audio, got {} Hz",
        config.sample_rate_hz
    );
    ensure!(
        config.win_length == 1_024 && config.hop_length == 256 && config.n_fft == 1_024,
        "SpeechT5 mel extraction requires 1024-sample window/FFT and 256-sample hop"
    );
    ensure!(
        config.n_mels == 80,
        "SpeechT5 mel extraction requires 80 mel bins"
    );
    ensure!(
        config.center,
        "SpeechT5 mel extraction requires centered reflect padding"
    );
    ensure!(
        matches!(config.scale, MelScale::Slaney)
            && matches!(config.log_base, LogCompression::Log10)
            && matches!(config.normalize, MelNormalization::Floor { min } if (min + 10.0).abs() < f32::EPSILON),
        "SpeechT5 mel extraction requires Slaney log10 mel with a 1e-10 floor"
    );
    Ok(())
}

fn periodic_hann(length: usize) -> Vec<f64> {
    (0..length)
        .map(|index| 0.5 - 0.5 * ((2.0 * PI * index as f64) / length as f64).cos())
        .collect()
}

fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    if samples.is_empty() || pad == 0 {
        return samples.to_vec();
    }

    let mut padded = Vec::with_capacity(samples.len() + pad * 2);
    for position in -(pad as isize)..(samples.len() as isize + pad as isize) {
        padded.push(samples[reflect_index(position, samples.len())]);
    }
    padded
}

fn reflect_index(mut index: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = 2 * len as isize - 2;
    index %= period;
    if index < 0 {
        index += period;
    }
    if index >= len as isize {
        (period - index) as usize
    } else {
        index as usize
    }
}

fn amplitude_spectrum(frame: &[f32], window: &[f64], fft_length: usize) -> Vec<f64> {
    let frequency_bins = fft_length / 2 + 1;
    (0..frequency_bins)
        .map(|bin| {
            let mut real = 0.0;
            let mut imag = 0.0;
            for (sample_index, sample) in frame.iter().enumerate() {
                let windowed = f64::from(*sample) * window[sample_index];
                let phase = -2.0 * PI * bin as f64 * sample_index as f64 / fft_length as f64;
                real += windowed * phase.cos();
                imag += windowed * phase.sin();
            }
            real.hypot(imag)
        })
        .collect()
}

fn slaney_mel_filter_bank(config: &MelConfig) -> Vec<Vec<f64>> {
    let frequency_bins = config.n_fft / 2 + 1;
    let min_hz = f64::from(config.f_min_hz);
    let max_hz = f64::from(
        config
            .f_max_hz
            .unwrap_or(config.sample_rate_hz as f32 / 2.0),
    );
    let min_mel = hertz_to_slaney_mel(min_hz);
    let max_mel = hertz_to_slaney_mel(max_hz);
    let mel_points = (0..config.n_mels + 2)
        .map(|index| {
            let t = index as f64 / (config.n_mels + 1) as f64;
            slaney_mel_to_hertz(min_mel + (max_mel - min_mel) * t)
        })
        .collect::<Vec<_>>();

    let fft_freqs = (0..frequency_bins)
        .map(|index| index as f64 * config.sample_rate_hz as f64 / config.n_fft as f64)
        .collect::<Vec<_>>();

    let mut filters = vec![vec![0.0; frequency_bins]; config.n_mels];
    for mel_index in 0..config.n_mels {
        let left = mel_points[mel_index];
        let center = mel_points[mel_index + 1];
        let right = mel_points[mel_index + 2];
        let down_width = (center - left).max(f64::EPSILON);
        let up_width = (right - center).max(f64::EPSILON);
        let enorm = 2.0 / (right - left).max(f64::EPSILON);
        for (freq_index, freq) in fft_freqs.iter().enumerate() {
            let down = (*freq - left) / down_width;
            let up = (right - *freq) / up_width;
            filters[mel_index][freq_index] = down.min(up).max(0.0) * enorm;
        }
    }
    filters
}

fn hertz_to_slaney_mel(freq: f64) -> f64 {
    let min_log_hz = 1_000.0;
    let min_log_mel = 15.0;
    let logstep = 27.0 / 6.4_f64.ln();
    if freq >= min_log_hz {
        min_log_mel + (freq / min_log_hz).ln() * logstep
    } else {
        3.0 * freq / 200.0
    }
}

fn slaney_mel_to_hertz(mel: f64) -> f64 {
    let min_log_hz = 1_000.0;
    let min_log_mel = 15.0;
    let logstep = 6.4_f64.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        200.0 * mel / 3.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_padding_matches_numpy_reflect_shape_for_short_vectors() {
        assert_eq!(
            reflect_pad(&[0.0, 1.0, 2.0], 2),
            vec![2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0]
        );
    }

    #[test]
    fn extracts_speecht5_log_mel_contract_frames() {
        let samples = (0..16_000)
            .map(|index| {
                let phase = 2.0 * std::f32::consts::PI * 220.0 * index as f32 / 16_000.0;
                phase.sin() * 0.1
            })
            .collect::<Vec<_>>();

        let extraction = extract_speecht5_log_mel(&samples).expect("mel extraction");

        assert_eq!(extraction.sample_rate_hz, 16_000);
        assert_eq!(extraction.hop_samples, 256);
        assert_eq!(extraction.frame_length_samples, 1_024);
        assert_eq!(extraction.fft_length_samples, 1_024);
        assert_eq!(extraction.mel.len(), 63);
        assert!(extraction.mel.iter().all(|frame| frame.bins.len() == 80));
        assert!(
            extraction
                .mel
                .iter()
                .flat_map(|frame| &frame.bins)
                .all(|bin| bin.is_finite() && *bin >= -10.0)
        );
    }
}
