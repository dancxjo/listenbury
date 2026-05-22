use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::audio::AudioFrame;
use crate::soundscape::{SoundSource, SourceId, SourceKind, TimeRange, VoiceSignatureId};

/// Stable identifier for a speaker embedding cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClusterId(pub Uuid);

impl ClusterId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ClusterId {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared context for source attribution over incoming audio frames.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SoundscapeContext {
    /// Current active sources from the wider soundscape state.
    pub active_sources: Vec<SoundSource>,
    /// Optional source expected to be producing local playback output.
    pub expected_playback_source: Option<SourceId>,
    /// Optional source expected to be producing local self output.
    pub expected_self_output_source: Option<SourceId>,
}

/// A time-bounded source-identity hypothesis with supporting evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceHypothesis {
    pub source_id: Option<SourceId>,
    pub kind: SourceKind,
    pub range: TimeRange,
    pub confidence: f32,
    pub evidence: Vec<AttributionEvidence>,
}

/// Supporting evidence for source attribution decisions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttributionEvidence {
    MatchesExpectedPlayback {
        source_id: SourceId,
        confidence: f32,
    },
    MatchesExpectedSelfOutput {
        source_id: SourceId,
        confidence: f32,
    },
    MatchesPlaybackBuffer {
        confidence: f32,
    },
    MatchesVoiceSignature {
        signature_id: VoiceSignatureId,
        confidence: f32,
    },
    SpeakerEmbeddingCluster {
        cluster_id: ClusterId,
        confidence: f32,
    },
    PitchContinuity {
        confidence: f32,
    },
    SpectralContinuity {
        confidence: f32,
    },
    LexicalContinuity {
        confidence: f32,
    },
    EnergyChange,
    OverlapDetected,
}

pub trait SourceAttributor {
    fn attribute(
        &mut self,
        frame: &AudioFrame,
        context: &SoundscapeContext,
    ) -> Vec<SourceHypothesis>;
}

#[cfg(test)]
mod tests {
    use crate::soundscape::{
        AttributionEvidence, SourceHypothesis, SourceId, SourceKind, TimePoint, TimeRange,
        VoiceSignatureId,
    };

    #[test]
    fn combines_multiple_evidence_items_into_single_hypothesis() {
        let signature_id = VoiceSignatureId::new();
        let expected_playback_source = SourceId::new();

        let hypothesis = SourceHypothesis {
            source_id: Some(expected_playback_source),
            kind: SourceKind::Playback,
            range: TimeRange::new(TimePoint::from_millis(2_000), TimePoint::from_millis(2_250)),
            confidence: 0.88,
            evidence: vec![
                AttributionEvidence::MatchesExpectedPlayback {
                    source_id: expected_playback_source,
                    confidence: 0.93,
                },
                AttributionEvidence::MatchesVoiceSignature {
                    signature_id,
                    confidence: 0.71,
                },
                AttributionEvidence::SpectralContinuity { confidence: 0.77 },
                AttributionEvidence::OverlapDetected,
            ],
        };

        assert_eq!(hypothesis.evidence.len(), 4);
        assert!(matches!(
            hypothesis.evidence[0],
            AttributionEvidence::MatchesExpectedPlayback { .. }
        ));
        assert!(matches!(
            hypothesis.evidence[1],
            AttributionEvidence::MatchesVoiceSignature { .. }
        ));
        assert!(matches!(
            hypothesis.evidence[2],
            AttributionEvidence::SpectralContinuity { .. }
        ));
        assert!(matches!(
            hypothesis.evidence[3],
            AttributionEvidence::OverlapDetected
        ));
    }
}
