use crate::audio::frame::AudioFrame;
#[cfg(feature = "vad-webrtc")]
use anyhow::Context;

#[cfg(feature = "vad-webrtc")]
const WEBRTC_ENERGY_FALLBACK_THRESHOLD_RMS: f32 = 0.08;
#[cfg(feature = "vad-webrtc")]
const WEBRTC_MIN_SPEECH_RMS: f32 = 0.015;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VadBackendKind {
    #[default]
    Energy,
    WebRtc,
    Silero,
}

impl VadBackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Energy => "energy",
            Self::WebRtc => "webrtc",
            Self::Silero => "silero",
        }
    }
}

impl std::fmt::Display for VadBackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VadResult {
    pub speech_prob: f32,
    pub is_speech: bool,
}

pub trait VoiceActivityDetector {
    fn process_frame(&mut self, frame: &AudioFrame) -> anyhow::Result<VadResult>;
}

pub fn create_vad_backend(kind: VadBackendKind) -> anyhow::Result<Box<dyn VoiceActivityDetector>> {
    match kind {
        VadBackendKind::Energy => Ok(Box::new(EnergyVad::default())),
        VadBackendKind::WebRtc => {
            #[cfg(feature = "vad-webrtc")]
            {
                Ok(Box::new(WebRtcVad::default()))
            }
            #[cfg(not(feature = "vad-webrtc"))]
            {
                anyhow::bail!(
                    "VAD backend '{}' requires the `vad-webrtc` feature",
                    kind.as_str()
                );
            }
        }
        VadBackendKind::Silero => {
            #[cfg(feature = "vad-silero")]
            {
                anyhow::bail!(
                    "VAD backend '{}' is not implemented yet in listenbury; use --vad energy or --vad webrtc",
                    kind.as_str()
                );
            }
            #[cfg(not(feature = "vad-silero"))]
            {
                anyhow::bail!(
                    "VAD backend '{}' requires the `vad-silero` feature, but this backend is not implemented yet in listenbury",
                    kind.as_str()
                );
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EnergyVad {
    threshold_rms: f32,
}

impl EnergyVad {
    pub fn new(threshold_rms: f32) -> Self {
        Self { threshold_rms }
    }

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
        (sum_sq / samples.len() as f32).sqrt()
    }
}

impl Default for EnergyVad {
    fn default() -> Self {
        Self::new(0.02)
    }
}

impl VoiceActivityDetector for EnergyVad {
    fn process_frame(&mut self, frame: &AudioFrame) -> anyhow::Result<VadResult> {
        let rms = Self::rms(&frame.samples);
        let speech_prob = (rms / self.threshold_rms).clamp(0.0, 1.0);
        Ok(VadResult {
            speech_prob,
            is_speech: rms >= self.threshold_rms,
        })
    }
}

#[cfg(feature = "vad-webrtc")]
pub struct WebRtcVad {
    engine: webrtc_vad::Vad,
    sample_rate_hz: Option<u32>,
    energy_fallback: EnergyVad,
}

#[cfg(feature = "vad-webrtc")]
impl Default for WebRtcVad {
    fn default() -> Self {
        Self {
            engine: webrtc_vad::Vad::new_with_mode(webrtc_vad::VadMode::VeryAggressive),
            sample_rate_hz: None,
            energy_fallback: EnergyVad::new(WEBRTC_ENERGY_FALLBACK_THRESHOLD_RMS),
        }
    }
}

#[cfg(feature = "vad-webrtc")]
impl VoiceActivityDetector for WebRtcVad {
    fn process_frame(&mut self, frame: &AudioFrame) -> anyhow::Result<VadResult> {
        anyhow::ensure!(
            frame.sample_rate_hz > 0,
            "audio frame sample rate must be non-zero"
        );
        anyhow::ensure!(
            frame.channels > 0,
            "audio frame channel count must be non-zero"
        );
        anyhow::ensure!(
            frame.samples.len() % usize::from(frame.channels) == 0,
            "audio frame sample count ({}) is not divisible by channel count ({})",
            frame.samples.len(),
            frame.channels
        );

        let sample_rate = sample_rate_from_hz(frame.sample_rate_hz).with_context(|| {
            format!(
                "WebRTC VAD only supports 8000/16000/32000/48000 Hz, got {} Hz",
                frame.sample_rate_hz
            )
        })?;
        if self.sample_rate_hz != Some(frame.sample_rate_hz) {
            self.engine.set_sample_rate(sample_rate);
            self.sample_rate_hz = Some(frame.sample_rate_hz);
        }

        let centered_frame = mean_centered_frame(frame);
        let centered_rms = EnergyVad::rms(&centered_frame.samples);
        let mono_i16 = to_mono_i16(frame);
        ensure_supported_frame_length(frame.sample_rate_hz, mono_i16.len())?;
        let is_webrtc_speech = self
            .engine
            .is_voice_segment(&mono_i16)
            .map_err(|_| anyhow::anyhow!("invalid WebRTC VAD frame length"))?;
        let fallback = self.energy_fallback.process_frame(&centered_frame)?;
        let is_speech =
            (is_webrtc_speech && centered_rms >= WEBRTC_MIN_SPEECH_RMS) || fallback.is_speech;
        Ok(VadResult {
            speech_prob: if is_webrtc_speech {
                (centered_rms / WEBRTC_MIN_SPEECH_RMS).clamp(0.0, 1.0)
            } else {
                fallback.speech_prob
            },
            is_speech,
        })
    }
}

#[cfg(feature = "vad-webrtc")]
fn sample_rate_from_hz(sample_rate_hz: u32) -> anyhow::Result<webrtc_vad::SampleRate> {
    match sample_rate_hz {
        8_000 => Ok(webrtc_vad::SampleRate::Rate8kHz),
        16_000 => Ok(webrtc_vad::SampleRate::Rate16kHz),
        32_000 => Ok(webrtc_vad::SampleRate::Rate32kHz),
        48_000 => Ok(webrtc_vad::SampleRate::Rate48kHz),
        _ => anyhow::bail!("unsupported sample rate: {sample_rate_hz}"),
    }
}

#[cfg(feature = "vad-webrtc")]
fn ensure_supported_frame_length(sample_rate_hz: u32, mono_samples: usize) -> anyhow::Result<()> {
    let samples_10ms = usize::try_from(sample_rate_hz / 100).unwrap_or(0);
    let valid_lengths = [samples_10ms, samples_10ms * 2, samples_10ms * 3];
    anyhow::ensure!(
        valid_lengths.contains(&mono_samples),
        "WebRTC VAD requires mono frames with 10/20/30ms duration; got {mono_samples} samples at {sample_rate_hz} Hz"
    );
    Ok(())
}

#[cfg(feature = "vad-webrtc")]
fn to_mono_i16(frame: &AudioFrame) -> Vec<i16> {
    if frame.channels == 0 {
        return Vec::new();
    }
    let channel_count = usize::from(frame.channels);
    let mono_samples = frame
        .samples
        .chunks_exact(channel_count)
        .map(|chunk| chunk.iter().sum::<f32>() / f32::from(frame.channels))
        .collect::<Vec<_>>();
    if mono_samples.is_empty() {
        return Vec::new();
    }
    let mean = mono_samples.iter().sum::<f32>() / mono_samples.len() as f32;
    mono_samples
        .into_iter()
        .map(|mono| {
            let centered = mono - mean;
            (centered.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16
        })
        .collect()
}

#[cfg(feature = "vad-webrtc")]
fn mean_centered_frame(frame: &AudioFrame) -> AudioFrame {
    if frame.samples.is_empty() {
        return AudioFrame {
            captured_at: frame.captured_at,
            sample_rate_hz: frame.sample_rate_hz,
            channels: frame.channels,
            samples: Vec::new(),
        };
    }

    let mean = frame.samples.iter().sum::<f32>() / frame.samples.len() as f32;
    AudioFrame {
        captured_at: frame.captured_at,
        sample_rate_hz: frame.sample_rate_hz,
        channels: frame.channels,
        samples: frame.samples.iter().map(|sample| sample - mean).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExactTimestamp;

    #[test]
    fn creates_energy_backend() {
        let mut vad = create_vad_backend(VadBackendKind::Energy).unwrap();
        let frame = AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![0.0; 160],
        };
        let result = vad.process_frame(&frame).unwrap();
        assert!(!result.is_speech);
    }

    #[test]
    fn silero_backend_returns_clear_error() {
        let error = create_vad_backend(VadBackendKind::Silero)
            .err()
            .expect("silero backend should currently fail");
        let message = error.to_string();
        assert!(message.contains("silero"));
        assert!(message.contains("not implemented"));
    }

    #[cfg(feature = "vad-webrtc")]
    #[test]
    fn creates_webrtc_backend_when_feature_enabled() {
        let mut vad = create_vad_backend(VadBackendKind::WebRtc).unwrap();
        let frame = AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![0.0; 160],
        };
        let result = vad.process_frame(&frame).unwrap();
        assert!(!result.is_speech);
    }

    #[cfg(feature = "vad-webrtc")]
    #[test]
    fn webrtc_backend_keeps_energy_fallback_for_loud_frames() {
        let mut vad = create_vad_backend(VadBackendKind::WebRtc).unwrap();
        let frame = AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: 16_000,
            channels: 1,
            samples: (0..160)
                .map(|index| if index % 2 == 0 { 0.10 } else { -0.10 })
                .collect(),
        };

        let result = vad.process_frame(&frame).unwrap();

        assert!(result.is_speech);
        assert!(result.speech_prob > 0.0);
    }

    #[cfg(feature = "vad-webrtc")]
    #[test]
    fn webrtc_backend_ignores_dc_offset_frames() {
        let mut vad = create_vad_backend(VadBackendKind::WebRtc).unwrap();
        let frame = AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![0.04; 160],
        };

        let result = vad.process_frame(&frame).unwrap();

        assert!(!result.is_speech);
        assert!(result.speech_prob > 0.0);
    }

    #[cfg(feature = "vad-webrtc")]
    #[test]
    fn webrtc_backend_ignores_low_energy_non_voice_frames() {
        let mut vad = create_vad_backend(VadBackendKind::WebRtc).unwrap();
        let frame = AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: 16_000,
            channels: 1,
            samples: (0..160)
                .map(|index| if index % 2 == 0 { 0.005 } else { -0.005 })
                .collect(),
        };

        let result = vad.process_frame(&frame).unwrap();

        assert!(!result.is_speech);
    }

    #[cfg(not(feature = "vad-webrtc"))]
    #[test]
    fn webrtc_backend_requires_feature() {
        let error = create_vad_backend(VadBackendKind::WebRtc)
            .err()
            .expect("webrtc backend should fail without feature");
        assert!(error.to_string().contains("vad-webrtc"));
    }
}
