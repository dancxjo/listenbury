use std::sync::{Arc, Mutex};

use crate::audio::frame::AudioFrame;
use crate::hearing::suppression::{
    SpeakerReferenceDecision, SpeakerReferenceMask, SuppressionDecision,
};
use crate::time::ExactTimestamp;

const AUDITION_MIN_VOICE_ENERGY: f32 = 0.0004;
const AUDITION_MIN_VOICE_RMS: f32 = 0.020;
const AUDITION_MIN_MIXED_CORRELATION: f32 = 0.10;
const AUDITION_MIN_SELF_GAIN: f32 = 0.05;
const AUDITION_MIN_ENVIRONMENTAL_RMS: f32 = 0.012;
const AUDITION_BASELINE_NOISE_RMS: f32 = 0.003;
const AUDITION_VOICE_SCORE_THRESHOLD: f32 = 0.38;
const AUDITION_STRONG_VOICE_SCORE_THRESHOLD: f32 = 0.45;
const AUDITION_ENVIRONMENT_SCORE_THRESHOLD: f32 = 0.53;
const AUDITION_ENVIRONMENT_COOLDOWN_FRAMES: u8 = 25;
/// Env score threshold that bypasses hysteresis: a clearly exceptional transient.
const AUDITION_ENVIRONMENT_STRONG_SCORE_THRESHOLD: f32 = 0.78;
/// Brightness below this value identifies a low-spectral (thump/sub-bass) sound.
/// Brightness is derived as `zero_crossing_rate * 2.0`, so this threshold
/// corresponds roughly to a fundamental frequency below ~100 Hz.
const AUDITION_ENVIRONMENT_LOW_SPECTRAL_BRIGHTNESS: f32 = 0.04;
/// Consecutive frames a low-spectral candidate must remain salient before acceptance.
const AUDITION_ENVIRONMENT_HYSTERESIS_FRAMES: u8 = 3;

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
pub struct AuditoryFrameDiagnostics {
    pub rms: f32,
    pub zero_crossing_rate: f32,
    pub brightness: f32,
    pub voice_score: f32,
    pub environment_score: f32,
    pub noise_floor_rms: f32,
    pub routing_reason: &'static str,
    /// Number of consecutive frames that have qualified as an environmental candidate.
    /// Used to track hysteresis buildup for low-spectral (thump-like) sounds.
    pub environmental_hysteresis_frames: u8,
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
    pub diagnostics: AuditoryFrameDiagnostics,
}

impl AuditoryFrameAnalysis {
    pub fn external_residual_frame(&self) -> Option<&AudioFrame> {
        self.residual_frame.as_ref()
    }
}

#[derive(Debug, Clone)]
pub struct AuditorySceneAnalyzer {
    speaker_reference: Arc<Mutex<SpeakerReferenceMask>>,
    state: Arc<Mutex<AuditorySceneState>>,
}

impl AuditorySceneAnalyzer {
    pub fn new(speaker_reference: Arc<Mutex<SpeakerReferenceMask>>) -> Self {
        Self {
            speaker_reference,
            state: Arc::new(Mutex::new(AuditorySceneState::default())),
        }
    }

    pub fn analyze(&self, frame: AudioFrame) -> anyhow::Result<AuditoryFrameAnalysis> {
        let speaker = {
            let mut speaker_reference = self
                .speaker_reference
                .lock()
                .map_err(|_| anyhow::anyhow!("speaker reference mask lock poisoned"))?;
            speaker_reference.analyze_frame(&frame)
        };
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("auditory scene state lock poisoned"))?;
        Ok(analyze_speaker_reference_with_state(
            frame, speaker, &mut state,
        ))
    }
}

pub fn analyze_speaker_reference(
    frame: AudioFrame,
    speaker: SpeakerReferenceDecision,
) -> AuditoryFrameAnalysis {
    let mut state = AuditorySceneState::default();
    analyze_speaker_reference_with_state(frame, speaker, &mut state)
}

#[derive(Debug, Clone)]
struct AuditorySceneState {
    noise_floor_rms: f32,
    speech_like_frames: u8,
    environmental_cooldown_frames: u8,
    /// Consecutive frames that have qualified as an environmental candidate.
    /// Resets to zero whenever a frame does not qualify.
    environmental_candidate_frames: u8,
}

impl Default for AuditorySceneState {
    fn default() -> Self {
        Self {
            noise_floor_rms: AUDITION_BASELINE_NOISE_RMS,
            speech_like_frames: 0,
            environmental_cooldown_frames: 0,
            environmental_candidate_frames: 0,
        }
    }
}

fn analyze_speaker_reference_with_state(
    frame: AudioFrame,
    speaker: SpeakerReferenceDecision,
    state: &mut AuditorySceneState,
) -> AuditoryFrameAnalysis {
    let frame_energy = mean_square_energy(&frame.samples);
    let frame_rms = frame_energy.sqrt();
    let residual_energy = mean_square_energy(&speaker.residual_frame.samples);
    let residual_is_voice = residual_energy >= AUDITION_MIN_VOICE_ENERGY;
    let features = frame_features(&frame, frame_rms, state.noise_floor_rms);
    let speech_like = features.voice_score >= AUDITION_VOICE_SCORE_THRESHOLD
        && frame_rms >= AUDITION_MIN_VOICE_RMS;
    let self_confidence = self_voice_confidence(&speaker);
    let external_confidence = external_voice_confidence(residual_energy, speaker.residual_ratio);
    if speech_like {
        state.speech_like_frames = state.speech_like_frames.saturating_add(1).min(8);
    } else {
        state.speech_like_frames = 0;
    }
    if state.environmental_cooldown_frames > 0 {
        state.environmental_cooldown_frames -= 1;
    }

    let environmental_candidate = frame_rms >= AUDITION_MIN_ENVIRONMENTAL_RMS
        && features.environment_score >= AUDITION_ENVIRONMENT_SCORE_THRESHOLD;
    if environmental_candidate {
        state.environmental_candidate_frames =
            state.environmental_candidate_frames.saturating_add(1);
    } else {
        state.environmental_candidate_frames = 0;
    }
    // A low-spectral sound (very low brightness / ZCR, typical of thumps and
    // sub-bass handling noise) requires sustained salience across several frames
    // before it is accepted, unless it is a clearly exceptional transient.
    let is_low_spectral = features.brightness < AUDITION_ENVIRONMENT_LOW_SPECTRAL_BRIGHTNESS;
    let is_strong_transient =
        features.environment_score >= AUDITION_ENVIRONMENT_STRONG_SCORE_THRESHOLD;
    let hysteresis_met = !is_low_spectral
        || is_strong_transient
        || state.environmental_candidate_frames >= AUDITION_ENVIRONMENT_HYSTERESIS_FRAMES;
    let (routing, routing_reason) = if speaker.decision == SuppressionDecision::Suppress {
        (AuditoryRouting::EchoOnly, "speaker_reference_suppressed")
    } else if residual_is_voice
        && speaker.correlation >= AUDITION_MIN_MIXED_CORRELATION
        && speaker.gain.abs() >= AUDITION_MIN_SELF_GAIN
    {
        (
            AuditoryRouting::MixedSelfAndExternal,
            "speaker_reference_mixed_with_residual",
        )
    } else if speech_like && features.voice_score >= AUDITION_STRONG_VOICE_SCORE_THRESHOLD {
        (
            AuditoryRouting::ExternalSpeechCandidate,
            "strong_voice_score",
        )
    } else if speech_like && state.speech_like_frames >= 2 {
        (
            AuditoryRouting::ExternalSpeechCandidate,
            "sustained_voice_score",
        )
    } else if environmental_candidate && !hysteresis_met {
        (
            AuditoryRouting::SilenceOrNoise,
            "environmental_hysteresis_building",
        )
    } else if environmental_candidate && state.environmental_cooldown_frames == 0 {
        state.environmental_cooldown_frames = AUDITION_ENVIRONMENT_COOLDOWN_FRAMES;
        (
            AuditoryRouting::EnvironmentalSoundCandidate,
            if is_strong_transient {
                "environmental_transient_accepted"
            } else {
                "salient_non_speech_sound"
            },
        )
    } else if environmental_candidate {
        (
            AuditoryRouting::SilenceOrNoise,
            "environmental_candidate_rate_limited",
        )
    } else {
        (AuditoryRouting::SilenceOrNoise, "below_voice_threshold")
    };
    update_noise_floor(state, frame_rms, routing, speech_like);

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
        diagnostics: AuditoryFrameDiagnostics {
            rms: frame_rms,
            zero_crossing_rate: features.zero_crossing_rate,
            brightness: features.brightness,
            voice_score: features.voice_score,
            environment_score: features.environment_score,
            noise_floor_rms: state.noise_floor_rms,
            routing_reason,
            environmental_hysteresis_frames: state.environmental_candidate_frames,
        },
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

fn update_noise_floor(
    state: &mut AuditorySceneState,
    rms: f32,
    routing: AuditoryRouting,
    speech_like: bool,
) {
    if speech_like
        || matches!(
            routing,
            AuditoryRouting::EchoOnly | AuditoryRouting::MixedSelfAndExternal
        )
    {
        return;
    }
    let alpha = if routing == AuditoryRouting::SilenceOrNoise {
        0.04
    } else {
        0.005
    };
    let capped_rms = rms.min(state.noise_floor_rms * 3.0);
    state.noise_floor_rms = (state.noise_floor_rms * (1.0 - alpha) + capped_rms * alpha)
        .max(AUDITION_BASELINE_NOISE_RMS);
}

#[derive(Debug, Clone, Copy)]
struct FrameFeatures {
    zero_crossing_rate: f32,
    brightness: f32,
    voice_score: f32,
    environment_score: f32,
}

fn frame_features(frame: &AudioFrame, rms: f32, noise_floor_rms: f32) -> FrameFeatures {
    let zero_crossing_rate = zero_crossing_rate(&frame.samples);
    let brightness = (zero_crossing_rate * 2.0).clamp(0.0, 1.0);
    let voice_band_score = band_score(zero_crossing_rate, 0.025, 0.12, 0.30);
    let periodicity = pitch_periodicity(&frame.samples, frame.sample_rate_hz);
    let energy_score = ((rms - AUDITION_MIN_VOICE_RMS) / 0.040).clamp(0.0, 1.0);
    let voice_score = (energy_score
        * (voice_band_score * 0.72 + periodicity * voice_band_score.min(0.85) * 0.28))
        .clamp(0.0, 1.0);
    let salience = ((rms - noise_floor_rms * 1.8)
        / noise_floor_rms.max(AUDITION_BASELINE_NOISE_RMS))
    .clamp(0.0, 1.0);
    let crest = crest_factor(&frame.samples, rms);
    let transient_score = ((crest - 4.0) / 5.0).clamp(0.0, 1.0);
    let non_voice_score = (1.0 - voice_score).clamp(0.0, 1.0);
    let environment_score =
        (salience * non_voice_score * (0.55 + transient_score * 0.35 + brightness * 0.10))
            .clamp(0.0, 1.0);

    FrameFeatures {
        zero_crossing_rate,
        brightness,
        voice_score,
        environment_score,
    }
}

fn band_score(value: f32, low: f32, peak: f32, high: f32) -> f32 {
    if value <= low || value >= high {
        0.0
    } else if value <= peak {
        ((value - low) / (peak - low)).clamp(0.0, 1.0)
    } else {
        ((high - value) / (high - peak)).clamp(0.0, 1.0)
    }
}

fn pitch_periodicity(samples: &[f32], sample_rate_hz: u32) -> f32 {
    if samples.len() < 24 || sample_rate_hz == 0 {
        return 0.0;
    }
    let min_lag = (sample_rate_hz / 400).max(1) as usize;
    let max_lag = (sample_rate_hz / 90).max(min_lag as u32) as usize;
    let max_lag = max_lag.min(samples.len().saturating_sub(2));
    if min_lag >= max_lag {
        return 0.0;
    }

    let mut best = 0.0f32;
    for lag in min_lag..=max_lag {
        let mut dot = 0.0f32;
        let mut left_energy = 0.0f32;
        let mut right_energy = 0.0f32;
        for idx in lag..samples.len() {
            let left = samples[idx - lag];
            let right = samples[idx];
            dot += left * right;
            left_energy += left * left;
            right_energy += right * right;
        }
        let denom = (left_energy * right_energy).sqrt();
        if denom > f32::EPSILON {
            best = best.max((dot / denom).clamp(0.0, 1.0));
        }
    }
    best
}

fn crest_factor(samples: &[f32], rms: f32) -> f32 {
    if samples.is_empty() || rms <= f32::EPSILON {
        return 0.0;
    }
    samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0, f32::max)
        / rms
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
            .zip(user.clone())
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
            .analyze(frame_at(1_000_000_000, synthetic_voice(160)))
            .expect("analysis should succeed");

        assert_eq!(
            analysis.routing,
            AuditoryRouting::ExternalSpeechCandidate,
            "{:?}",
            analysis.diagnostics
        );
        assert!(analysis.external.vad_candidate);
        assert!(analysis.diagnostics.voice_score >= 0.45);
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
            AuditoryRouting::EnvironmentalSoundCandidate,
            "{:?}",
            analysis.diagnostics
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
    fn low_level_fan_noise_routes_mostly_to_silence_or_noise() {
        let analyzer =
            AuditorySceneAnalyzer::new(Arc::new(Mutex::new(SpeakerReferenceMask::new())));
        let mut external_count = 0;
        let mut environmental_count = 0;
        let mut silence_count = 0;

        for frame_index in 0..80 {
            let analysis = analyzer
                .analyze(frame_at(
                    1_000_000_000 + frame_index * 10_000_000,
                    low_level_fan_noise(160),
                ))
                .expect("analysis should succeed");
            match analysis.routing {
                AuditoryRouting::ExternalSpeechCandidate => external_count += 1,
                AuditoryRouting::EnvironmentalSoundCandidate => environmental_count += 1,
                AuditoryRouting::SilenceOrNoise => silence_count += 1,
                AuditoryRouting::EchoOnly | AuditoryRouting::MixedSelfAndExternal => {}
            }
        }

        assert_eq!(external_count, 0);
        assert!(
            environmental_count <= 2,
            "low-level fan should not become frame-level events"
        );
        assert!(silence_count >= 78);
    }

    #[test]
    fn transient_tick_routes_environmental_only_briefly() {
        let analyzer =
            AuditorySceneAnalyzer::new(Arc::new(Mutex::new(SpeakerReferenceMask::new())));
        let mut environmental_count = 0;
        let mut external_count = 0;

        for frame_index in 0..40 {
            let samples = if frame_index == 10 {
                transient_tick()
            } else {
                vec![0.0; 160]
            };
            let analysis = analyzer
                .analyze(frame_at(1_000_000_000 + frame_index * 10_000_000, samples))
                .expect("analysis should succeed");
            match analysis.routing {
                AuditoryRouting::EnvironmentalSoundCandidate => environmental_count += 1,
                AuditoryRouting::ExternalSpeechCandidate => external_count += 1,
                _ => {}
            }
        }

        assert_eq!(external_count, 0);
        assert_eq!(environmental_count, 1);
    }

    #[test]
    fn repeated_noise_frames_are_rate_limited_environmental_not_speech() {
        let analyzer =
            AuditorySceneAnalyzer::new(Arc::new(Mutex::new(SpeakerReferenceMask::new())));
        let mut external_count = 0;
        let mut environmental_count = 0;

        for frame_index in 0..100 {
            let analysis = analyzer
                .analyze(frame_at(
                    1_000_000_000 + frame_index * 10_000_000,
                    mid_level_fan_noise(160),
                ))
                .expect("analysis should succeed");
            match analysis.routing {
                AuditoryRouting::ExternalSpeechCandidate => external_count += 1,
                AuditoryRouting::EnvironmentalSoundCandidate => environmental_count += 1,
                _ => {}
            }
        }

        assert_eq!(external_count, 0);
        assert!(
            environmental_count <= 4,
            "sustained noise should be state, not per-frame news"
        );
    }

    #[test]
    fn delayed_room_echo_in_tail_window_remains_echo_only() {
        let (analyzer, reference, started_at) = analyzer_with_reference();
        let delayed_echo_delay_ms = 90u128;
        let delayed_echo_capture_offset_ms = 260u128;
        let delayed_echo_start =
            ((delayed_echo_capture_offset_ms - delayed_echo_delay_ms) * 16_000 / 1_000) as usize;
        let delayed_echo_end = delayed_echo_start + 160;
        let delayed_echo = reference[delayed_echo_start..delayed_echo_end]
            .iter()
            .map(|sample| sample * 0.35)
            .collect::<Vec<_>>();

        let analysis = analyzer
            .analyze(frame_at(
                started_at.unix_nanos + delayed_echo_capture_offset_ms * 1_000_000,
                delayed_echo,
            ))
            .expect("analysis should succeed");

        assert_eq!(analysis.routing, AuditoryRouting::EchoOnly);
        assert_eq!(analysis.self_voice.delay_ms, delayed_echo_delay_ms as i64);
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

    fn synthetic_voice(len: usize) -> Vec<f32> {
        let sample_rate = 16_000.0f32;
        let fundamental_hz = 180.0f32;
        (0..len)
            .map(|idx| {
                let t = idx as f32 / sample_rate;
                let phase = std::f32::consts::TAU * fundamental_hz * t;
                let harmonics = phase.sin() * 0.55
                    + (phase * 2.0).sin() * 0.30
                    + (phase * 3.0).sin() * 0.22
                    + (phase * 5.0).sin() * 0.18
                    + (phase * 7.0).sin() * 0.12;
                let aspiration = if idx % 5 < 2 { 0.05 } else { -0.05 };
                (harmonics + aspiration) * 0.14
            })
            .collect()
    }

    fn low_level_fan_noise(len: usize) -> Vec<f32> {
        fan_noise(len, 0.006)
    }

    fn mid_level_fan_noise(len: usize) -> Vec<f32> {
        fan_noise(len, 0.025)
    }

    fn fan_noise(len: usize, amplitude: f32) -> Vec<f32> {
        let sample_rate = 16_000.0f32;
        let freq_hz = 120.0f32;
        (0..len)
            .map(|idx| {
                let t = idx as f32 / sample_rate;
                let hum = (std::f32::consts::TAU * freq_hz * t).sin() * amplitude;
                let ripple = if idx % 17 == 0 {
                    amplitude * 0.08
                } else {
                    -amplitude * 0.02
                };
                hum + ripple
            })
            .collect()
    }

    fn transient_tick() -> Vec<f32> {
        let mut samples = vec![0.0; 160];
        samples[8] = 0.50;
        samples[9] = -0.35;
        samples[10] = 0.20;
        samples
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

    /// A single-frame low-frequency thump (low ZCR, low brightness) must not
    /// immediately route as an environmental candidate.  The hysteresis window
    /// must build up first.
    #[test]
    fn brief_low_frequency_thump_is_suppressed_by_hysteresis() {
        let analyzer =
            AuditorySceneAnalyzer::new(Arc::new(Mutex::new(SpeakerReferenceMask::new())));
        // Simulate a single frame of a very low-frequency (sub-100 Hz) loud
        // thump: high RMS, very low ZCR / brightness, moderate env_score.
        let thump = low_freq_thump(160);
        let analysis = analyzer
            .analyze(frame_at(1_000_000_000, thump))
            .expect("analysis should succeed");
        assert_eq!(
            analysis.routing,
            AuditoryRouting::SilenceOrNoise,
            "single thump frame should be held by hysteresis, not routed immediately: {:?}",
            analysis.diagnostics
        );
        assert_eq!(
            analysis.diagnostics.routing_reason, "environmental_hysteresis_building",
            "routing reason should indicate hysteresis building"
        );
    }

    /// A low-frequency thump that is sustained across the hysteresis window
    /// must eventually produce an `EnvironmentalSoundCandidate`.
    #[test]
    fn sustained_low_frequency_sound_routes_environmental_after_hysteresis() {
        let analyzer =
            AuditorySceneAnalyzer::new(Arc::new(Mutex::new(SpeakerReferenceMask::new())));
        let mut environmental_count = 0;
        let mut last_reason = "";

        for frame_index in 0..10 {
            let analysis = analyzer
                .analyze(frame_at(
                    1_000_000_000 + frame_index * 10_000_000,
                    low_freq_thump(160),
                ))
                .expect("analysis should succeed");
            if analysis.routing == AuditoryRouting::EnvironmentalSoundCandidate {
                environmental_count += 1;
            }
            last_reason = analysis.diagnostics.routing_reason;
        }

        assert!(
            environmental_count >= 1,
            "sustained low-frequency sound should eventually route as environmental (got 0 in 10 frames); last reason={last_reason}"
        );
    }

    /// Synthesise a loud sub-100 Hz thump: high RMS, very low ZCR/brightness.
    fn low_freq_thump(len: usize) -> Vec<f32> {
        let sample_rate = 16_000.0f32;
        let freq_hz = 60.0f32; // 60 Hz fundamental → very low brightness
        (0..len)
            .map(|idx| {
                let t = idx as f32 / sample_rate;
                (std::f32::consts::TAU * freq_hz * t).sin() * 0.45
            })
            .collect()
    }
}
