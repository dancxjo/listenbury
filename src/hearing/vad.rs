use crate::audio::frame::AudioFrame;
#[cfg(feature = "vad-webrtc")]
use anyhow::Context;

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
            engine: webrtc_vad::Vad::new_with_mode(webrtc_vad::VadMode::Quality),
            sample_rate_hz: None,
            energy_fallback: EnergyVad::default(),
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

        let mono_i16 = to_mono_i16(frame);
        ensure_supported_frame_length(frame.sample_rate_hz, mono_i16.len())?;
        let is_speech = self
            .engine
            .is_voice_segment(&mono_i16)
            .map_err(|_| anyhow::anyhow!("invalid WebRTC VAD frame length"))?;
        let fallback = self.energy_fallback.process_frame(frame)?;
        Ok(VadResult {
            speech_prob: if is_speech { 1.0 } else { fallback.speech_prob },
            is_speech: is_speech || fallback.is_speech,
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
    frame
        .samples
        .chunks_exact(channel_count)
        .map(|chunk| {
            let mono = chunk.iter().sum::<f32>() / f32::from(frame.channels);
            (mono.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16
        })
        .collect()
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

    #[cfg(not(feature = "vad-webrtc"))]
    #[test]
    fn webrtc_backend_requires_feature() {
        let error = create_vad_backend(VadBackendKind::WebRtc)
            .err()
            .expect("webrtc backend should fail without feature");
        assert!(error.to_string().contains("vad-webrtc"));
    }
}
