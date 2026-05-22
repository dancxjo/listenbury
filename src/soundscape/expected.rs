use serde::{Deserialize, Serialize};

use crate::audio::AudioFrame;
use crate::soundscape::{AttributionEvidence, SourceId, TimePoint, TimeRange};

const DEFAULT_MIN_OVERLAP_RATIO: f32 = 0.30;
const DEFAULT_MIN_SAMPLE_COVERAGE: f32 = 0.30;
const DEFAULT_MIN_CORRELATION: f32 = 0.65;

/// Planned or generated playback represented as an expected acoustic trace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpectedSound {
    pub source_id: SourceId,
    pub expected_range: TimeRange,
    pub expected_text: Option<String>,
    pub expected_samples: Vec<f32>,
    pub confidence: f32,
}

/// Observed microphone audio represented for source attribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservedSound {
    pub range: TimeRange,
    pub transcript_hypotheses: Vec<TranscriptHypothesis>,
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptHypothesis {
    pub text: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlaybackMatchConfig {
    pub max_timing_skew_ms: u64,
    pub min_overlap_ratio: f32,
    pub min_sample_coverage: f32,
    pub min_correlation: f32,
}

impl PlaybackMatchConfig {
    pub fn with_max_timing_skew_ms(max_timing_skew_ms: u64) -> Self {
        Self {
            max_timing_skew_ms,
            ..Self::default()
        }
    }
}

impl Default for PlaybackMatchConfig {
    fn default() -> Self {
        Self {
            max_timing_skew_ms: 180,
            min_overlap_ratio: DEFAULT_MIN_OVERLAP_RATIO,
            min_sample_coverage: DEFAULT_MIN_SAMPLE_COVERAGE,
            min_correlation: DEFAULT_MIN_CORRELATION,
        }
    }
}

impl ObservedSound {
    /// Adapter from the existing mic frame model to observed sound.
    pub fn from_audio_frame(frame: &AudioFrame) -> Self {
        let frame_samples_per_channel = frame.samples.len() / usize::from(frame.channels.max(1));
        let duration_ms = if frame.sample_rate_hz == 0 {
            0
        } else {
            ((frame_samples_per_channel as u128).saturating_mul(1_000)
                / u128::from(frame.sample_rate_hz))
            .min(u128::from(u64::MAX)) as u64
        };
        let start_ms = (frame.captured_at.unix_nanos / 1_000_000).min(u128::from(u64::MAX)) as u64;
        let end_ms = start_ms.saturating_add(duration_ms);
        Self {
            range: TimeRange::new(
                TimePoint::from_millis(start_ms),
                TimePoint::from_millis(end_ms),
            ),
            transcript_hypotheses: Vec::new(),
            samples: frame.samples.clone(),
        }
    }
}

/// Returns playback attribution evidence when observed audio matches expected playback.
pub fn playback_match_evidence(
    expected: &ExpectedSound,
    observed: &ObservedSound,
    config: PlaybackMatchConfig,
) -> Option<AttributionEvidence> {
    let skew_ms = expected
        .expected_range
        .start
        .millis
        .abs_diff(observed.range.start.millis);
    if skew_ms > config.max_timing_skew_ms {
        return None;
    }

    let overlap = overlap_millis(expected.expected_range, observed.range);
    let expected_duration = expected.expected_range.duration_millis().max(1);
    let overlap_ratio = (overlap as f32 / expected_duration as f32).clamp(0.0, 1.0);
    if overlap_ratio < config.min_overlap_ratio {
        return None;
    }

    let sample_coverage = if expected.expected_samples.is_empty() {
        1.0
    } else {
        (observed.samples.len().min(expected.expected_samples.len()) as f32
            / expected.expected_samples.len() as f32)
            .clamp(0.0, 1.0)
    };
    if sample_coverage < config.min_sample_coverage {
        return None;
    }

    let correlation = sample_correlation(&expected.expected_samples, &observed.samples);
    if correlation < config.min_correlation {
        return None;
    }

    let confidence =
        (expected.confidence * overlap_ratio * sample_coverage * correlation).clamp(0.0, 1.0);
    Some(AttributionEvidence::MatchesPlaybackBuffer { confidence })
}

fn overlap_millis(expected: TimeRange, observed: TimeRange) -> u64 {
    let start = expected.start.millis.max(observed.start.millis);
    let end = expected.end.millis.min(observed.end.millis);
    end.saturating_sub(start)
}

fn sample_correlation(expected: &[f32], observed: &[f32]) -> f32 {
    let len = expected.len().min(observed.len());
    if len == 0 {
        return 0.0;
    }
    let expected = &expected[..len];
    let observed = &observed[..len];
    let mut dot = 0.0f32;
    let mut expected_energy = 0.0f32;
    let mut observed_energy = 0.0f32;
    for (&a, &b) in expected.iter().zip(observed) {
        dot += a * b;
        expected_energy += a * a;
        observed_energy += b * b;
    }
    if expected_energy <= f32::EPSILON || observed_energy <= f32::EPSILON {
        return 0.0;
    }
    (dot.abs() / (expected_energy.sqrt() * observed_energy.sqrt())).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use crate::audio::AudioFrame;
    use crate::soundscape::{
        AttributionEvidence, SourceId, TimePoint, TimeRange,
        expected::{ExpectedSound, ObservedSound, PlaybackMatchConfig, playback_match_evidence},
    };
    use crate::time::ExactTimestamp;

    #[test]
    fn exact_match_yields_playback_buffer_evidence() {
        let expected = expected_sound(1_000, 1_200, samples(64, 0.9));
        let observed = observed_sound(1_000, 1_200, samples(64, 0.9));

        let evidence =
            playback_match_evidence(&expected, &observed, PlaybackMatchConfig::default())
                .expect("expected an exact match");

        assert!(matches!(
            evidence,
            AttributionEvidence::MatchesPlaybackBuffer { confidence } if confidence > 0.85
        ));
    }

    #[test]
    fn delayed_match_within_skew_still_matches() {
        let expected = expected_sound(2_000, 2_200, samples(64, 0.75));
        let observed = observed_sound(2_080, 2_280, samples(64, 0.75));

        let evidence = playback_match_evidence(
            &expected,
            &observed,
            PlaybackMatchConfig::with_max_timing_skew_ms(120),
        )
        .expect("expected delayed match within skew tolerance");

        assert!(matches!(
            evidence,
            AttributionEvidence::MatchesPlaybackBuffer { confidence } if confidence > 0.45
        ));
    }

    #[test]
    fn partial_match_still_emits_evidence_with_lower_confidence() {
        let expected = expected_sound(3_000, 3_240, samples(120, 0.8));
        let observed = observed_sound(3_020, 3_140, samples(60, 0.8));

        let evidence =
            playback_match_evidence(&expected, &observed, PlaybackMatchConfig::default())
                .expect("expected partial match evidence");

        assert!(matches!(
            evidence,
            AttributionEvidence::MatchesPlaybackBuffer { confidence } if (0.10..0.40).contains(&confidence)
        ));
    }

    #[test]
    fn non_match_does_not_emit_playback_evidence() {
        let expected = expected_sound(4_000, 4_200, samples(64, 0.7));
        let observed = observed_sound(4_350, 4_550, samples(64, -0.7));

        assert!(
            playback_match_evidence(&expected, &observed, PlaybackMatchConfig::default()).is_none()
        );
    }

    #[test]
    fn observed_sound_can_be_adapted_from_audio_frame() {
        let frame = AudioFrame {
            captured_at: ExactTimestamp {
                unix_nanos: 8_000_000_000,
            },
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![0.1; 160],
            voice_signatures: Vec::new(),
        };
        let observed = ObservedSound::from_audio_frame(&frame);

        assert_eq!(observed.range.start.millis, 8_000);
        assert_eq!(observed.range.end.millis, 8_010);
        assert_eq!(observed.samples.len(), 160);
        assert!(observed.transcript_hypotheses.is_empty());
    }

    fn expected_sound(start_ms: u64, end_ms: u64, expected_samples: Vec<f32>) -> ExpectedSound {
        ExpectedSound {
            source_id: SourceId::new(),
            expected_range: TimeRange::new(
                TimePoint::from_millis(start_ms),
                TimePoint::from_millis(end_ms),
            ),
            expected_text: Some("pete playback".to_string()),
            expected_samples,
            confidence: 0.95,
        }
    }

    fn observed_sound(start_ms: u64, end_ms: u64, samples: Vec<f32>) -> ObservedSound {
        ObservedSound {
            range: TimeRange::new(
                TimePoint::from_millis(start_ms),
                TimePoint::from_millis(end_ms),
            ),
            transcript_hypotheses: Vec::new(),
            samples,
        }
    }

    fn samples(len: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|idx| {
                let phase = idx as f32 * std::f32::consts::TAU * 190.0 / 16_000.0;
                phase.sin() * scale
            })
            .collect()
    }
}
