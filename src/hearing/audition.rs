use std::sync::{Arc, Mutex};

use crate::audio::frame::AudioFrame;
use crate::hearing::suppression::{
    SpeakerReferenceDecision, SpeakerReferenceMask, SuppressionDecision,
};
use crate::time::ExactTimestamp;

const AUDITION_MIN_VOICE_ENERGY: f32 = 0.0004;
const AUDITION_MIN_MIXED_CORRELATION: f32 = 0.10;
const AUDITION_MIN_SELF_GAIN: f32 = 0.05;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditoryRouting {
    EchoOnly,
    MixedSpeech,
    ExternalOnly,
    SilenceOrNoise,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelfVoiceEstimate {
    pub correlation: f32,
    pub residual_ratio: f32,
    pub delay_ms: i64,
    pub gain: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExternalVoiceEstimate {
    pub residual_energy: f32,
    pub vad_candidate: bool,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseEstimate {
    pub energy: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AuditoryFrameAnalysis {
    pub captured_at: ExactTimestamp,
    pub frame: AudioFrame,
    pub self_voice: SelfVoiceEstimate,
    pub external_voice: ExternalVoiceEstimate,
    pub noise: NoiseEstimate,
    pub routing: AuditoryRouting,
    pub residual_frame: AudioFrame,
    pub self_frame: AudioFrame,
}

impl AuditoryFrameAnalysis {
    pub fn external_residual_frame(&self) -> &AudioFrame {
        &self.residual_frame
    }
}

#[derive(Debug, Clone)]
pub struct AuditorySceneAnalyzer {
    speaker_reference: Arc<Mutex<SpeakerReferenceMask>>,
}

impl AuditorySceneAnalyzer {
    pub fn new(speaker_reference: Arc<Mutex<SpeakerReferenceMask>>) -> Self {
        Self { speaker_reference }
    }

    pub fn analyze(&self, frame: AudioFrame) -> anyhow::Result<AuditoryFrameAnalysis> {
        let speaker = {
            let mut speaker_reference = self
                .speaker_reference
                .lock()
                .map_err(|_| anyhow::anyhow!("speaker reference mask lock poisoned"))?;
            speaker_reference.analyze_frame(&frame)
        };
        Ok(analyze_speaker_reference(frame, speaker))
    }
}

pub fn analyze_speaker_reference(
    frame: AudioFrame,
    speaker: SpeakerReferenceDecision,
) -> AuditoryFrameAnalysis {
    let frame_energy = mean_square_energy(&frame.samples);
    let residual_energy = mean_square_energy(&speaker.residual_frame.samples);
    let residual_is_voice = residual_energy >= AUDITION_MIN_VOICE_ENERGY;
    let self_confidence = self_voice_confidence(&speaker);
    let external_confidence = external_voice_confidence(residual_energy, speaker.residual_ratio);
    let routing = if speaker.decision == SuppressionDecision::Suppress {
        AuditoryRouting::EchoOnly
    } else if residual_is_voice
        && speaker.correlation >= AUDITION_MIN_MIXED_CORRELATION
        && speaker.gain.abs() >= AUDITION_MIN_SELF_GAIN
    {
        AuditoryRouting::MixedSpeech
    } else if frame_energy >= AUDITION_MIN_VOICE_ENERGY {
        AuditoryRouting::ExternalOnly
    } else {
        AuditoryRouting::SilenceOrNoise
    };

    AuditoryFrameAnalysis {
        captured_at: frame.captured_at,
        frame,
        self_voice: SelfVoiceEstimate {
            correlation: speaker.correlation,
            residual_ratio: speaker.residual_ratio,
            delay_ms: speaker.delay_ms,
            gain: speaker.gain,
            confidence: self_confidence,
        },
        external_voice: ExternalVoiceEstimate {
            residual_energy,
            vad_candidate: residual_is_voice,
            confidence: external_confidence,
        },
        noise: NoiseEstimate {
            energy: if routing == AuditoryRouting::SilenceOrNoise {
                frame_energy
            } else {
                0.0
            },
        },
        routing,
        residual_frame: speaker.residual_frame,
        self_frame: speaker.self_frame,
    }
}

fn self_voice_confidence(speaker: &SpeakerReferenceDecision) -> f32 {
    let residual_fit = 1.0 - speaker.residual_ratio;
    (speaker.correlation * residual_fit.clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

fn external_voice_confidence(residual_energy: f32, residual_ratio: f32) -> f32 {
    let energy_confidence = (residual_energy / AUDITION_MIN_VOICE_ENERGY)
        .sqrt()
        .clamp(0.0, 1.0);
    (energy_confidence * residual_ratio.clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

fn mean_square_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use crate::audio::frame::AudioFrame;
    use crate::hearing::audition::{AuditoryRouting, AuditorySceneAnalyzer};
    use crate::hearing::suppression::SpeakerReferenceMask;
    use crate::time::ExactTimestamp;

    #[test]
    fn pure_delayed_playback_is_echo_only() {
        let (analyzer, reference, started_at) = analyzer_with_reference();
        let mic = reference[800..960]
            .iter()
            .map(|sample| sample * 0.4)
            .collect::<Vec<_>>();

        let analysis = analyzer
            .analyze(frame_at(started_at.unix_nanos + 50_000_000, mic))
            .expect("analysis should succeed");

        assert_eq!(analysis.routing, AuditoryRouting::EchoOnly);
        assert!(analysis.self_voice.confidence > 0.9);
    }

    #[test]
    fn speech_mixed_with_playback_routes_residual_to_asr() {
        let (analyzer, reference, started_at) = analyzer_with_reference();
        let user = test_noise(160).into_iter().rev().collect::<Vec<_>>();
        let mic = reference[800..960]
            .iter()
            .zip(user)
            .map(|(echo, user)| echo * 0.25 + user * 0.8)
            .collect::<Vec<_>>();

        let analysis = analyzer
            .analyze(frame_at(started_at.unix_nanos + 50_000_000, mic.clone()))
            .expect("analysis should succeed");

        assert_eq!(analysis.routing, AuditoryRouting::MixedSpeech);
        assert_ne!(analysis.external_residual_frame().samples, mic);
        assert!(analysis.external_voice.vad_candidate);
    }

    #[test]
    fn external_only_speech_remains_external_only() {
        let analyzer =
            AuditorySceneAnalyzer::new(Arc::new(Mutex::new(SpeakerReferenceMask::new())));
        let analysis = analyzer
            .analyze(frame_at(1_000_000_000, test_noise(160)))
            .expect("analysis should succeed");

        assert_eq!(analysis.routing, AuditoryRouting::ExternalOnly);
    }

    #[test]
    fn silence_or_noise_does_not_become_external_speech() {
        let analyzer =
            AuditorySceneAnalyzer::new(Arc::new(Mutex::new(SpeakerReferenceMask::new())));
        let analysis = analyzer
            .analyze(frame_at(1_000_000_000, vec![0.001; 160]))
            .expect("analysis should succeed");

        assert_eq!(analysis.routing, AuditoryRouting::SilenceOrNoise);
        assert!(!analysis.external_voice.vad_candidate);
    }

    fn analyzer_with_reference() -> (AuditorySceneAnalyzer, Vec<f32>, ExactTimestamp) {
        let sample_rate_hz = 16_000;
        let started_at = ExactTimestamp {
            unix_nanos: 1_000_000_000,
        };
        let reference = test_noise(3_200);
        let frames = vec![AudioFrame {
            captured_at: started_at,
            sample_rate_hz,
            channels: 1,
            samples: reference.clone(),
        }];
        let mask = Arc::new(Mutex::new(SpeakerReferenceMask::new()));
        mask.lock()
            .expect("speaker reference lock should be available")
            .mark_output_started(&frames, started_at);
        (AuditorySceneAnalyzer::new(mask), reference, started_at)
    }

    fn frame_at(unix_nanos: u128, samples: Vec<f32>) -> AudioFrame {
        AudioFrame {
            captured_at: ExactTimestamp { unix_nanos },
            sample_rate_hz: 16_000,
            channels: 1,
            samples,
        }
    }

    fn test_noise(len: usize) -> Vec<f32> {
        let mut state = 0x1234_5678u32;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let value = ((state >> 8) as f32 / 16_777_215.0) * 2.0 - 1.0;
                value * 0.5
            })
            .collect()
    }
}
