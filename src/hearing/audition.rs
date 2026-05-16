use std::sync::{Arc, Mutex};

use crate::audio::frame::AudioFrame;
use crate::hearing::suppression::{
    SpeakerReferenceDecision, SpeakerReferenceMask, SuppressionDecision,
};
use crate::time::ExactTimestamp;

const AUDITION_MIN_VOICE_ENERGY: f32 = 0.0004;
const AUDITION_MIN_MIXED_CORRELATION: f32 = 0.10;
const AUDITION_MIN_SELF_GAIN: f32 = 0.05;
const AUDITION_MIN_SPEECH_ZERO_CROSSING_RATE: f32 = 0.08;
const AUDITION_MIN_ENVIRONMENTAL_ENERGY: f32 = 0.00008;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditoryRouting {
    EchoOnly,
    MixedSelfAndExternal,
    ExternalSpeechCandidate,
    EnvironmentalSoundCandidate,
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
pub struct ExternalEstimate {
    pub residual_energy: f32,
    pub vad_candidate: bool,
    pub confidence: f32,
}

pub type ExternalVoiceEstimate = ExternalEstimate;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseEstimate {
    pub energy: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AuditoryFrameAnalysis {
    pub captured_at: ExactTimestamp,
    pub frame: AudioFrame,
    pub self_voice: SelfVoiceEstimate,
    pub external: ExternalEstimate,
    pub noise: NoiseEstimate,
    pub routing: AuditoryRouting,
    pub residual_frame: Option<AudioFrame>,
    pub self_frame: AudioFrame,
}

impl AuditoryFrameAnalysis {
    pub fn external_residual_frame(&self) -> Option<&AudioFrame> {
        self.residual_frame.as_ref()
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
    let speech_like = is_speech_candidate(&frame.samples, frame_energy);
    let self_confidence = self_voice_confidence(&speaker);
    let external_confidence = external_voice_confidence(residual_energy, speaker.residual_ratio);
    let routing = if speaker.decision == SuppressionDecision::Suppress {
        AuditoryRouting::EchoOnly
    } else if residual_is_voice
        && speaker.correlation >= AUDITION_MIN_MIXED_CORRELATION
        && speaker.gain.abs() >= AUDITION_MIN_SELF_GAIN
    {
        AuditoryRouting::MixedSelfAndExternal
    } else if speech_like {
        AuditoryRouting::ExternalSpeechCandidate
    } else if frame_energy >= AUDITION_MIN_ENVIRONMENTAL_ENERGY {
        AuditoryRouting::EnvironmentalSoundCandidate
    } else {
        AuditoryRouting::SilenceOrNoise
    };
    let residual_frame = match routing {
        AuditoryRouting::MixedSelfAndExternal => Some(speaker.residual_frame.clone()),
        AuditoryRouting::ExternalSpeechCandidate => Some(frame.clone()),
        AuditoryRouting::EchoOnly
        | AuditoryRouting::EnvironmentalSoundCandidate
        | AuditoryRouting::SilenceOrNoise => None,
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
        external: ExternalEstimate {
            residual_energy,
            vad_candidate: matches!(
                routing,
                AuditoryRouting::MixedSelfAndExternal | AuditoryRouting::ExternalSpeechCandidate
            ),
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
        residual_frame,
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

fn is_speech_candidate(samples: &[f32], energy: f32) -> bool {
    energy >= AUDITION_MIN_VOICE_ENERGY
        && zero_crossing_rate(samples) >= AUDITION_MIN_SPEECH_ZERO_CROSSING_RATE
}

fn zero_crossing_rate(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let crossings = samples
        .windows(2)
        .filter(|pair| {
            let a = pair[0];
            let b = pair[1];
            (a >= 0.0 && b < 0.0) || (a < 0.0 && b >= 0.0)
        })
        .count();
    crossings as f32 / (samples.len() - 1) as f32
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
            .zip(user.iter().copied())
            .map(|(echo, user)| echo * 0.25 + user * 0.8)
            .collect::<Vec<_>>();

        let analysis = analyzer
            .analyze(frame_at(started_at.unix_nanos + 50_000_000, mic.clone()))
            .expect("analysis should succeed");

        assert_eq!(analysis.routing, AuditoryRouting::MixedSelfAndExternal);
        let residual = analysis
            .external_residual_frame()
            .expect("mixed frame should provide residual");
        assert_ne!(residual.samples, mic);
        assert!(analysis.external.vad_candidate);
        assert!(
            mean_square_error(&residual.samples, &user) < mean_square_error(&mic, &user),
            "residual should preserve external speech better than the raw mixed frame"
        );
    }

    #[test]
    fn external_only_speech_remains_external_speech_candidate() {
        let analyzer =
            AuditorySceneAnalyzer::new(Arc::new(Mutex::new(SpeakerReferenceMask::new())));
        let analysis = analyzer
            .analyze(frame_at(1_000_000_000, test_noise(160)))
            .expect("analysis should succeed");

        assert_eq!(analysis.routing, AuditoryRouting::ExternalSpeechCandidate);
    }

    #[test]
    fn salient_non_speech_event_becomes_environmental_candidate() {
        let analyzer =
            AuditorySceneAnalyzer::new(Arc::new(Mutex::new(SpeakerReferenceMask::new())));
        let tone = (0..160)
            .map(|idx| {
                let phase = idx as f32 * std::f32::consts::TAU * 300.0 / 16_000.0;
                phase.sin() * 0.06
            })
            .collect::<Vec<_>>();
        let analysis = analyzer
            .analyze(frame_at(1_000_000_000, tone))
            .expect("analysis should succeed");

        assert_eq!(
            analysis.routing,
            AuditoryRouting::EnvironmentalSoundCandidate
        );
        assert!(!analysis.external.vad_candidate);
    }

    #[test]
    fn silence_or_noise_does_not_become_external_speech() {
        let analyzer =
            AuditorySceneAnalyzer::new(Arc::new(Mutex::new(SpeakerReferenceMask::new())));
        let analysis = analyzer
            .analyze(frame_at(1_000_000_000, vec![0.001; 160]))
            .expect("analysis should succeed");

        assert_eq!(analysis.routing, AuditoryRouting::SilenceOrNoise);
        assert!(!analysis.external.vad_candidate);
    }

    #[test]
    fn delayed_room_echo_in_tail_window_remains_echo_only() {
        let (analyzer, reference, started_at) = analyzer_with_reference();
        let delayed_echo = reference[2_720..2_880]
            .iter()
            .map(|sample| sample * 0.35)
            .collect::<Vec<_>>();

        let analysis = analyzer
            .analyze(frame_at(started_at.unix_nanos + 260_000_000, delayed_echo))
            .expect("analysis should succeed");

        assert_eq!(analysis.routing, AuditoryRouting::EchoOnly);
        assert_eq!(analysis.self_voice.delay_ms, 90);
        assert!(analysis.self_voice.correlation > 0.99);
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

    fn mean_square_error(actual: &[f32], expected: &[f32]) -> f32 {
        actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| {
                let delta = actual - expected;
                delta * delta
            })
            .sum::<f32>()
            / actual.len().max(1) as f32
    }
}
