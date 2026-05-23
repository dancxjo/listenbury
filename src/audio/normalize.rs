use anyhow::Result;

pub const WHISPER_SAMPLE_RATE_HZ: u32 = 16_000;
pub const MONO_CHANNELS: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleKind {
    F32,
    I16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioFormat {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub sample_kind: SampleKind,
}

impl AudioFormat {
    pub const fn new(sample_rate_hz: u32, channels: u16, sample_kind: SampleKind) -> Self {
        Self {
            sample_rate_hz,
            channels,
            sample_kind,
        }
    }

    pub const fn asr_whisper_input() -> Self {
        Self::new(WHISPER_SAMPLE_RATE_HZ, MONO_CHANNELS, SampleKind::F32)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioConversionOp {
    SanitizeInvalidSamples,
    ClampOutOfRangeSamples,
    DownmixToMono,
    ExpandMonoToMultiChannel {
        target_channels: u16,
    },
    RemapChannels {
        source_channels: u16,
        target_channels: u16,
    },
    Resample,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioConversionReport {
    pub source: AudioFormat,
    pub target: AudioFormat,
    pub operations: Vec<AudioConversionOp>,
    pub warnings: Vec<String>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedAudio {
    pub samples: Vec<f32>,
    pub report: AudioConversionReport,
}

pub fn normalize_interleaved_f32(
    samples: &[f32],
    source: AudioFormat,
    target: AudioFormat,
    reason: impl Into<String>,
) -> Result<NormalizedAudio> {
    anyhow::ensure!(
        source.sample_rate_hz > 0,
        "source sample rate must be greater than zero"
    );
    anyhow::ensure!(
        source.channels > 0,
        "source channels must be greater than zero"
    );
    anyhow::ensure!(
        target.sample_rate_hz > 0,
        "target sample rate must be greater than zero"
    );
    anyhow::ensure!(
        target.channels > 0,
        "target channels must be greater than zero"
    );

    let mut operations = Vec::new();
    let mut warnings = Vec::new();

    let channel_count = usize::from(source.channels);
    let remainder = samples.len() % channel_count;
    let aligned_len = samples.len().saturating_sub(remainder);
    if remainder > 0 {
        warnings.push(format!(
            "discarded {remainder} trailing sample(s) that did not fit source channel layout"
        ));
    }

    let mut converted = sanitize_samples(&samples[..aligned_len], &mut operations, &mut warnings);
    converted = convert_channels_with_report(
        &converted,
        source.channels,
        target.channels,
        &mut operations,
    );
    if source.sample_rate_hz != target.sample_rate_hz {
        converted = resample_interleaved_linear(
            &converted,
            source.sample_rate_hz,
            target.sample_rate_hz,
            target.channels,
        );
        operations.push(AudioConversionOp::Resample);
    }

    Ok(NormalizedAudio {
        samples: converted,
        report: AudioConversionReport {
            source,
            target,
            operations,
            warnings,
            reason: reason.into(),
        },
    })
}

pub fn convert_channels(samples: &[f32], source_channels: u16, target_channels: u16) -> Vec<f32> {
    convert_channels_with_report(samples, source_channels, target_channels, &mut Vec::new())
}

fn convert_channels_with_report(
    samples: &[f32],
    source_channels: u16,
    target_channels: u16,
    operations: &mut Vec<AudioConversionOp>,
) -> Vec<f32> {
    if source_channels == target_channels {
        return samples.to_vec();
    }

    if target_channels == MONO_CHANNELS {
        operations.push(AudioConversionOp::DownmixToMono);
        return mix_to_mono(samples, source_channels);
    }

    let source_channel_count = usize::from(source_channels).max(1);
    let target_channel_count = usize::from(target_channels).max(1);
    if source_channel_count == 1 {
        operations.push(AudioConversionOp::ExpandMonoToMultiChannel { target_channels });
        let mut converted = Vec::with_capacity(samples.len().saturating_mul(target_channel_count));
        for sample in samples {
            converted.extend(std::iter::repeat_n(*sample, target_channel_count));
        }
        return converted;
    }

    operations.push(AudioConversionOp::RemapChannels {
        source_channels,
        target_channels,
    });
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

pub fn mix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    let channel_count = usize::from(channels).max(1);
    if channel_count == 1 {
        return samples.to_vec();
    }
    samples
        .chunks_exact(channel_count)
        .map(|frame| frame.iter().sum::<f32>() / f32::from(channels))
        .collect()
}

pub fn resample_interleaved_linear(
    samples: &[f32],
    source_rate_hz: u32,
    target_rate_hz: u32,
    channels: u16,
) -> Vec<f32> {
    let channel_count = usize::from(channels).max(1);
    if channel_count == 1 {
        return resample_linear(samples, source_rate_hz, target_rate_hz);
    }

    let frame_count = samples.len() / channel_count;
    if frame_count == 0 || source_rate_hz == target_rate_hz {
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

pub fn resample_linear(samples: &[f32], source_rate_hz: u32, target_rate_hz: u32) -> Vec<f32> {
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

pub fn normalize_signed_sample(sample: i64, bits_per_sample: u16) -> f32 {
    if bits_per_sample == 0 {
        // A zero-bit PCM declaration is invalid; return silence to keep callers total.
        return 0.0;
    }
    let positive_scale = ((1_i64 << (bits_per_sample - 1)) - 1) as f32;
    let negative_scale = (1_i64 << (bits_per_sample - 1)) as f32;
    if sample < 0 {
        sample as f32 / negative_scale
    } else {
        sample as f32 / positive_scale
    }
}

pub fn f32_to_i16(sample: f32) -> i16 {
    let sanitized = if sample.is_finite() {
        sample.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let scaled = (sanitized * (i16::MAX as f32 + 1.0)).round();
    scaled.clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

pub fn pad_silence_to_len(samples: &[f32], target_len: usize) -> Vec<f32> {
    if samples.len() >= target_len {
        return samples.to_vec();
    }
    let mut padded = Vec::with_capacity(target_len);
    padded.extend_from_slice(samples);
    padded.resize(target_len, 0.0);
    padded
}

pub fn trim_trailing_silence(samples: &[f32], threshold: f32) -> Vec<f32> {
    let threshold = threshold.abs();
    let end = samples
        .iter()
        .rposition(|sample| sample.abs() > threshold)
        .map(|idx| idx + 1)
        .unwrap_or(0);
    samples[..end].to_vec()
}

fn sanitize_samples(
    samples: &[f32],
    operations: &mut Vec<AudioConversionOp>,
    warnings: &mut Vec<String>,
) -> Vec<f32> {
    let mut saw_invalid = false;
    let mut saw_clamp = false;
    let mut sanitized = Vec::with_capacity(samples.len());
    for sample in samples {
        if !sample.is_finite() {
            saw_invalid = true;
            sanitized.push(0.0);
            continue;
        }
        let clamped = sample.clamp(-1.0, 1.0);
        if clamped != *sample {
            saw_clamp = true;
        }
        sanitized.push(clamped);
    }

    if saw_invalid {
        operations.push(AudioConversionOp::SanitizeInvalidSamples);
        warnings.push("replaced NaN or infinite samples with silence".to_string());
    }
    if saw_clamp {
        operations.push(AudioConversionOp::ClampOutOfRangeSamples);
        warnings.push("clamped out-of-range samples to [-1.0, 1.0]".to_string());
    }

    sanitized
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLOAT_TOLERANCE: f32 = 0.0001;

    #[test]
    fn converts_48khz_stereo_to_16khz_mono_for_asr() {
        let input = vec![0.5; 48_000 * 2];
        let converted = normalize_interleaved_f32(
            &input,
            AudioFormat::new(48_000, 2, SampleKind::F32),
            AudioFormat::asr_whisper_input(),
            "whisper_input",
        )
        .expect("conversion should succeed");

        assert_eq!(converted.samples.len(), 16_000);
        assert!(
            converted
                .samples
                .iter()
                .all(|sample| (*sample - 0.5).abs() <= FLOAT_TOLERANCE)
        );
        assert!(
            converted
                .report
                .operations
                .contains(&AudioConversionOp::DownmixToMono)
        );
        assert!(
            converted
                .report
                .operations
                .contains(&AudioConversionOp::Resample)
        );
    }

    #[test]
    fn converts_44100_mono_to_48000_stereo_for_playback() {
        let input = vec![0.25; 44_100];
        let converted = normalize_interleaved_f32(
            &input,
            AudioFormat::new(44_100, 1, SampleKind::F32),
            AudioFormat::new(48_000, 2, SampleKind::F32),
            "playback_device",
        )
        .expect("conversion should succeed");

        assert_eq!(converted.samples.len(), 48_000 * 2);
        assert!(
            converted
                .report
                .operations
                .contains(&AudioConversionOp::ExpandMonoToMultiChannel { target_channels: 2 })
        );
        assert!(
            converted
                .report
                .operations
                .contains(&AudioConversionOp::Resample)
        );
    }

    #[test]
    fn integer_pcm_round_trips_sanely() {
        let input = [i16::MIN, -12_345, 0, 12_345, i16::MAX];
        let restored = input
            .iter()
            .map(|sample| normalize_signed_sample(i64::from(*sample), 16))
            .map(f32_to_i16)
            .collect::<Vec<_>>();
        assert_eq!(restored, input);
    }

    #[test]
    fn silence_stays_silence_after_conversion() {
        let input = vec![0.0; 48_000 * 2];
        let converted = normalize_interleaved_f32(
            &input,
            AudioFormat::new(48_000, 2, SampleKind::F32),
            AudioFormat::asr_whisper_input(),
            "silence_test",
        )
        .expect("conversion should succeed");

        assert!(converted.samples.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn conversion_report_tracks_nan_and_clipping_cleanup() {
        let input = vec![f32::NAN, f32::INFINITY, -2.0, 2.0];
        let converted = normalize_interleaved_f32(
            &input,
            AudioFormat::new(16_000, 1, SampleKind::F32),
            AudioFormat::new(16_000, 1, SampleKind::F32),
            "cleanup_test",
        )
        .expect("conversion should succeed");

        assert_eq!(converted.samples, vec![0.0, 0.0, -1.0, 1.0]);
        assert!(
            converted
                .report
                .operations
                .contains(&AudioConversionOp::SanitizeInvalidSamples)
        );
        assert!(
            converted
                .report
                .operations
                .contains(&AudioConversionOp::ClampOutOfRangeSamples)
        );
        assert_eq!(converted.report.warnings.len(), 2);
    }

    #[test]
    fn silence_padding_and_trimming_helpers_are_explicit() {
        assert_eq!(pad_silence_to_len(&[0.1, 0.2], 4), vec![0.1, 0.2, 0.0, 0.0]);
        assert_eq!(trim_trailing_silence(&[0.1, 0.0, 0.0], 0.0001), vec![0.1]);
    }
}
