use std::path::Path;

use anyhow::{Context, Result};

use crate::audio::frame::AudioFrame;
use crate::audio::normalize::{
    AudioConversionReport, AudioFormat, MONO_CHANNELS, SampleKind, WHISPER_SAMPLE_RATE_HZ,
    f32_to_i16, normalize_interleaved_f32, normalize_signed_sample,
};
use crate::time::ExactTimestamp;

pub fn read_wav_as_audio_frames(path: &Path, frame_samples: usize) -> Result<Vec<AudioFrame>> {
    Ok(read_wav_as_audio_frames_with_report(path, frame_samples)?.0)
}

pub fn read_wav_as_audio_frames_with_report(
    path: &Path,
    frame_samples: usize,
) -> Result<(Vec<AudioFrame>, AudioConversionReport)> {
    read_wav_as_mono_16khz_frames(path, frame_samples)
}

pub fn read_wav_as_whisper_frames(path: &Path, frame_samples: usize) -> Result<Vec<AudioFrame>> {
    Ok(read_wav_as_whisper_frames_with_report(path, frame_samples)?.0)
}

pub fn read_wav_as_whisper_frames_with_report(
    path: &Path,
    frame_samples: usize,
) -> Result<(Vec<AudioFrame>, AudioConversionReport)> {
    read_wav_as_mono_16khz_frames(path, frame_samples)
}

fn read_wav_as_mono_16khz_frames(
    path: &Path,
    frame_samples: usize,
) -> Result<(Vec<AudioFrame>, AudioConversionReport)> {
    anyhow::ensure!(frame_samples > 0, "frame_samples must be greater than zero");

    let frames = read_wav_frames(path, frame_samples)?;
    let Some(first) = frames.first() else {
        return Ok((
            frames,
            AudioConversionReport {
                source: AudioFormat::asr_whisper_input(),
                target: AudioFormat::asr_whisper_input(),
                operations: Vec::new(),
                warnings: vec!["wav input contained no frames".to_string()],
                reason: "whisper_input".to_string(),
            },
        ));
    };

    let sample_rate_hz = first.sample_rate_hz;
    let channels = first.channels;
    anyhow::ensure!(
        channels > 0,
        "WAV input at {} reported zero channels",
        path.display()
    );

    let mut samples = Vec::new();
    for frame in &frames {
        anyhow::ensure!(
            frame.sample_rate_hz == sample_rate_hz,
            "WAV {} changed sample rate mid-stream ({} -> {})",
            path.display(),
            sample_rate_hz,
            frame.sample_rate_hz
        );
        anyhow::ensure!(
            frame.channels == channels,
            "WAV {} changed channel count mid-stream ({} -> {})",
            path.display(),
            channels,
            frame.channels
        );
        samples.extend_from_slice(&frame.samples);
    }

    let normalized = normalize_interleaved_f32(
        &samples,
        AudioFormat::new(sample_rate_hz, channels, SampleKind::F32),
        AudioFormat::asr_whisper_input(),
        "whisper_input",
    )?;
    let resampled_samples = normalized.samples;

    let frames = resampled_samples
        .chunks(frame_samples)
        .map(|chunk| AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: WHISPER_SAMPLE_RATE_HZ,
            channels: MONO_CHANNELS,
            samples: chunk.to_vec(),
            voice_signatures: Vec::new(),
        })
        .collect();
    Ok((frames, normalized.report))
}

pub fn read_wav_frames(path: &Path, frame_samples: usize) -> Result<Vec<AudioFrame>> {
    anyhow::ensure!(frame_samples > 0, "frame_samples must be greater than zero");

    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open WAV at {}", path.display()))?;
    let spec = reader.spec();

    let samples = match spec.sample_format {
        hound::SampleFormat::Int => match spec.bits_per_sample {
            1..=8 => reader
                .samples::<i8>()
                .map(|sample| {
                    sample.map(|sample| {
                        normalize_signed_sample(i64::from(sample), spec.bits_per_sample)
                    })
                })
                .collect::<std::result::Result<Vec<_>, _>>()
                .with_context(|| format!("failed to read PCM samples from {}", path.display()))?,
            9..=16 => reader
                .samples::<i16>()
                .map(|sample| {
                    sample.map(|sample| {
                        normalize_signed_sample(i64::from(sample), spec.bits_per_sample)
                    })
                })
                .collect::<std::result::Result<Vec<_>, _>>()
                .with_context(|| format!("failed to read PCM samples from {}", path.display()))?,
            17..=32 => reader
                .samples::<i32>()
                .map(|sample| {
                    sample.map(|sample| {
                        normalize_signed_sample(i64::from(sample), spec.bits_per_sample)
                    })
                })
                .collect::<std::result::Result<Vec<_>, _>>()
                .with_context(|| format!("failed to read PCM samples from {}", path.display()))?,
            bits => anyhow::bail!(
                "unsupported PCM bit depth {bits} for WAV input at {}",
                path.display()
            ),
        },
        hound::SampleFormat::Float => {
            anyhow::ensure!(
                spec.bits_per_sample == 32,
                "unsupported floating-point bit depth {} for WAV input at {}",
                spec.bits_per_sample,
                path.display()
            );
            reader
                .samples::<f32>()
                .map(|sample| sample.map(|pcm| pcm.clamp(-1.0, 1.0)))
                .collect::<std::result::Result<Vec<_>, _>>()
                .with_context(|| format!("failed to read PCM samples from {}", path.display()))?
        }
    };

    Ok(samples
        .chunks(frame_samples)
        .map(|chunk| AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: spec.sample_rate,
            channels: spec.channels,
            samples: chunk.to_vec(),
            voice_signatures: Vec::new(),
        })
        .collect())
}

pub fn write_wav(path: &Path, frames: &[AudioFrame]) -> Result<()> {
    let bytes = write_wav_bytes(frames)?;
    std::fs::write(path, bytes)
        .with_context(|| format!("failed to create WAV at {}", path.display()))?;
    Ok(())
}

pub fn write_wav_bytes(frames: &[AudioFrame]) -> Result<Vec<u8>> {
    let Some(first_frame) = frames.first() else {
        anyhow::bail!("cannot write WAV without audio frames");
    };
    anyhow::ensure!(
        first_frame.channels > 0,
        "cannot write WAV with zero channels"
    );

    let mut pcm = Vec::<u8>::new();
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
            pcm.extend_from_slice(&f32_to_i16(*sample).to_le_bytes());
        }
    }

    let data_len = u32::try_from(pcm.len()).context("WAV PCM payload exceeds u32 size")?;
    let riff_len = 36u32
        .checked_add(data_len)
        .context("WAV RIFF payload exceeds u32 size")?;
    let byte_rate = first_frame
        .sample_rate_hz
        .checked_mul(u32::from(first_frame.channels))
        .and_then(|value| value.checked_mul(2))
        .context("WAV byte rate overflows u32")?;
    let block_align = first_frame
        .channels
        .checked_mul(2)
        .context("WAV block alignment overflows u16")?;

    let mut out = Vec::with_capacity(44usize.saturating_add(pcm.len()));
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_len.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&first_frame.channels.to_le_bytes());
    out.extend_from_slice(&first_frame.sample_rate_hz.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(&pcm);
    Ok(out)
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
                voice_signatures: Vec::new(),
            },
            AudioFrame {
                captured_at: ExactTimestamp::now(),
                sample_rate_hz: 16_000,
                channels: 1,
                samples: vec![0.0, 0.5, 1.0],
                voice_signatures: Vec::new(),
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
    fn read_wav_as_audio_frames_resamples_to_16khz() {
        let path = unique_test_path("resample");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 22_050,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).expect("WAV should be created");
        for _ in 0..22_050 {
            writer
                .write_sample(i16::MAX)
                .expect("sample write should succeed");
        }
        writer.finalize().expect("WAV should finalize");

        let (got, report) =
            read_wav_as_audio_frames_with_report(&path, 1_600).expect("WAV should convert");

        assert_eq!(got.len(), 10);
        assert!(
            got.iter()
                .all(|frame| frame.sample_rate_hz == WHISPER_SAMPLE_RATE_HZ)
        );
        assert!(got.iter().all(|frame| frame.channels == MONO_CHANNELS));
        assert!(got.iter().all(|frame| frame.samples.len() == 1_600));
        assert_eq!(report.reason, "whisper_input");

        fs::remove_file(path).expect("temporary WAV should be removed");
    }

    #[test]
    fn read_wav_as_audio_frames_mixes_stereo_to_mono() {
        let path = unique_test_path("audio-stereo");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).expect("WAV should be created");
        for _ in 0..16_000 {
            writer
                .write_sample(i16::MAX)
                .expect("left sample should write");
            writer
                .write_sample(0_i16)
                .expect("right sample should write");
        }
        writer.finalize().expect("WAV should finalize");

        let got = read_wav_as_audio_frames(&path, 1_600).expect("WAV should convert");

        assert_eq!(got.len(), 10);
        assert!(got.iter().all(|frame| frame.channels == MONO_CHANNELS));
        assert!((got[0].samples[0] - 0.5).abs() <= FLOAT_TOLERANCE);

        fs::remove_file(path).expect("temporary WAV should be removed");
    }

    #[test]
    fn read_wav_frames_accepts_non_16khz_and_stereo() {
        let path = unique_test_path("playback");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44_100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).expect("WAV should be created");
        writer
            .write_sample(0_i16)
            .expect("left sample should write");
        writer
            .write_sample(0_i16)
            .expect("right sample should write");
        writer.finalize().expect("WAV should finalize");

        let got = read_wav_frames(&path, 4).expect("generic WAV reader should succeed");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].sample_rate_hz, 44_100);
        assert_eq!(got[0].channels, 2);

        fs::remove_file(path).expect("temporary WAV should be removed");
    }

    #[test]
    fn read_wav_as_whisper_frames_mixes_stereo_and_resamples_to_16khz() {
        let path = unique_test_path("whisper-convert");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 32_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).expect("WAV should be created");
        for _ in 0..32_000 {
            writer
                .write_sample(i16::MAX)
                .expect("left sample should write");
            writer
                .write_sample(0_i16)
                .expect("right sample should write");
        }
        writer.finalize().expect("WAV should finalize");

        let got = read_wav_as_whisper_frames(&path, 1_600).expect("WAV should convert");

        assert_eq!(got.len(), 10);
        assert!(
            got.iter()
                .all(|frame| frame.sample_rate_hz == WHISPER_SAMPLE_RATE_HZ)
        );
        assert!(got.iter().all(|frame| frame.channels == MONO_CHANNELS));
        assert!(got.iter().all(|frame| frame.samples.len() == 1_600));
        assert!((got[0].samples[0] - 0.5).abs() <= FLOAT_TOLERANCE);

        fs::remove_file(path).expect("temporary WAV should be removed");
    }

    #[test]
    fn normalize_signed_sample_maps_boundaries() {
        assert_eq!(normalize_signed_sample(0, 8), 0.0);
        assert_eq!(normalize_signed_sample(-128, 8), -1.0);
        assert_eq!(normalize_signed_sample(127, 8), 1.0);

        assert_eq!(normalize_signed_sample(-32_768, 16), -1.0);
        assert_eq!(normalize_signed_sample(32_767, 16), 1.0);

        assert_eq!(normalize_signed_sample(-8_388_608, 24), -1.0);
        assert_eq!(normalize_signed_sample(8_388_607, 24), 1.0);

        assert_eq!(normalize_signed_sample(i32::MIN as i64, 32), -1.0);
        assert_eq!(normalize_signed_sample(i32::MAX as i64, 32), 1.0);
    }

    #[test]
    fn f32_to_i16_clamps_and_rounds_samples() {
        assert_eq!(f32_to_i16(-1.5), i16::MIN);
        assert_eq!(f32_to_i16(-1.0), i16::MIN);
        assert_eq!(f32_to_i16(0.0), 0);
        assert_eq!(f32_to_i16(0.5), 16_384);
        assert_eq!(f32_to_i16(1.0), i16::MAX);
        assert_eq!(f32_to_i16(1.5), i16::MAX);
    }
}
