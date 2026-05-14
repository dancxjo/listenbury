use crate::audio::frame::AudioFrame;

#[derive(Debug, Clone, Copy)]
pub struct VadResult {
    pub speech_prob: f32,
    pub is_speech: bool,
}

pub trait VoiceActivityDetector {
    fn process_frame(&mut self, frame: &AudioFrame) -> anyhow::Result<VadResult>;
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
