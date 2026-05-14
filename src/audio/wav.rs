use std::path::Path;

use anyhow::{Context, Result};

use crate::audio::frame::AudioFrame;
use crate::time::ExactTimestamp;

pub fn read_wav_as_audio_frames(path: &Path, frame_samples: usize) -> Result<Vec<AudioFrame>> {
    anyhow::ensure!(frame_samples > 0, "frame_samples must be greater than zero");

    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open WAV at {}", path.display()))?;
    let spec = reader.spec();

    anyhow::ensure!(
        spec.channels == 1,
        "expected mono WAV input at {}; got {} channels",
        path.display(),
        spec.channels
    );
    anyhow::ensure!(
        spec.sample_rate == 16_000,
        "expected 16 kHz WAV input at {}; got {} Hz",
        path.display(),
        spec.sample_rate
    );
    anyhow::ensure!(
        spec.sample_format == hound::SampleFormat::Int,
        "expected integer PCM WAV input at {}; floating-point WAV is not supported yet",
        path.display()
    );

    let samples = match spec.bits_per_sample {
        1..=8 => reader
            .samples::<i8>()
            .map(|sample| {
                sample
                    .map(|sample| normalize_signed_sample(i64::from(sample), spec.bits_per_sample))
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .with_context(|| format!("failed to read PCM samples from {}", path.display()))?,
        9..=16 => reader
            .samples::<i16>()
            .map(|sample| {
                sample
                    .map(|sample| normalize_signed_sample(i64::from(sample), spec.bits_per_sample))
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .with_context(|| format!("failed to read PCM samples from {}", path.display()))?,
        17..=32 => reader
            .samples::<i32>()
            .map(|sample| {
                sample
                    .map(|sample| normalize_signed_sample(i64::from(sample), spec.bits_per_sample))
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .with_context(|| format!("failed to read PCM samples from {}", path.display()))?,
        bits => anyhow::bail!(
            "unsupported PCM bit depth {bits} for WAV input at {}",
            path.display()
        ),
    };

    Ok(samples
        .chunks(frame_samples)
        .map(|chunk| AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: spec.sample_rate,
            channels: spec.channels,
            samples: chunk.to_vec(),
        })
        .collect())
}

pub fn write_wav(path: &Path, frames: &[AudioFrame]) -> Result<()> {
    let Some(first_frame) = frames.first() else {
        anyhow::bail!("cannot write WAV without audio frames");
    };

    let spec = hound::WavSpec {
        channels: first_frame.channels,
        sample_rate: first_frame.sample_rate_hz,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("failed to create WAV at {}", path.display()))?;

    for frame in frames {
        anyhow::ensure!(
            frame.channels == first_frame.channels,
            "frame channel count changed from {} to {}",
            first_frame.channels,
            frame.channels
        );
        anyhow::ensure!(
            frame.sample_rate_hz == first_frame.sample_rate_hz,
            "frame sample rate changed from {} to {}",
            first_frame.sample_rate_hz,
            frame.sample_rate_hz
        );

        for sample in &frame.samples {
            writer.write_sample(f32_to_i16(*sample))?;
        }
    }

    writer.finalize()?;
    Ok(())
}

fn normalize_signed_sample(sample: i64, bits_per_sample: u16) -> f32 {
    let positive_scale = ((1_i64 << (bits_per_sample - 1)) - 1) as f32;
    let negative_scale = (1_i64 << (bits_per_sample - 1)) as f32;
    if sample < 0 {
        sample as f32 / negative_scale
    } else {
        sample as f32 / positive_scale
    }
}

fn f32_to_i16(sample: f32) -> i16 {
    let sample = sample.clamp(-1.0, 1.0);
    if sample < 0.0 {
        (sample * -(i16::MIN as f32)).round() as i16
    } else {
        (sample * i16::MAX as f32).round() as i16
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use super::*;

    const FLOAT_TOLERANCE: f32 = 0.0001;

    fn unique_test_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("listenbury-{name}-{}.wav", std::process::id()))
    }

    #[test]
    fn write_and_read_wav_round_trip_preserves_metadata() {
        let path = unique_test_path("round-trip");
        let frames = vec![
            AudioFrame {
                captured_at: ExactTimestamp::now(),
                sample_rate_hz: 16_000,
                channels: 1,
                samples: vec![-1.0, -0.25],
            },
            AudioFrame {
                captured_at: ExactTimestamp::now(),
                sample_rate_hz: 16_000,
                channels: 1,
                samples: vec![0.0, 0.5, 1.0],
            },
        ];

        write_wav(&path, &frames).expect("WAV should be written");
        let got = read_wav_as_audio_frames(&path, 2).expect("WAV should be read");

        assert_eq!(got.len(), 3);
        assert_eq!(got[0].sample_rate_hz, 16_000);
        assert_eq!(got[0].channels, 1);
        assert!((got[0].samples[0] + 1.0).abs() <= FLOAT_TOLERANCE);
        assert!((got[1].samples[1] - 0.5).abs() <= FLOAT_TOLERANCE);
        assert!((got[2].samples[0] - 1.0).abs() <= FLOAT_TOLERANCE);

        fs::remove_file(path).expect("temporary WAV should be removed");
    }

    #[test]
    fn read_wav_rejects_wrong_sample_rate() {
        let path = unique_test_path("wrong-rate");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 8_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).expect("WAV should be created");
        writer
            .write_sample(0_i16)
            .expect("sample write should succeed");
        writer.finalize().expect("WAV should finalize");

        let error = read_wav_as_audio_frames(&path, 1600).expect_err("sample rate should fail");
        assert!(error.to_string().contains("expected 16 kHz WAV input"));

        fs::remove_file(path).expect("temporary WAV should be removed");
    }

    #[test]
    fn read_wav_rejects_stereo_input() {
        let path = unique_test_path("stereo");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).expect("WAV should be created");
        writer
            .write_sample(0_i16)
            .expect("left sample write should succeed");
        writer
            .write_sample(0_i16)
            .expect("right sample write should succeed");
        writer.finalize().expect("WAV should finalize");

        let error = read_wav_as_audio_frames(&path, 1600).expect_err("channels should fail");
        assert!(error.to_string().contains("expected mono WAV input"));

        fs::remove_file(path).expect("temporary WAV should be removed");
    }
}
