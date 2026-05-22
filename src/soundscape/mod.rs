//! Soundscape-first data model for source-attributed listening.
//!
//! This module keeps source identity independent from transcript text so that
//! playback, voices, room noise, and other audible events can coexist in one
//! timeline.

pub mod attribution;
pub mod criteria;
pub mod debug;
pub mod event;
pub mod expected;
pub mod frame;
pub mod isolation;
pub mod overlap;
pub mod signature;
pub mod source;
pub mod time;
pub mod transcript;
pub mod voice_count;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub use attribution::{
    AttributionEvidence, ClusterId, SoundscapeContext, SourceAttributor, SourceHypothesis,
};
pub use criteria::{IsolationPolicy, SourceCriterion, SourceOperation};
pub use debug::{
    DebugHypothesis, DebugOverlapMixture, DebugSource, DebugTranscriptEvent, SoundscapeDebugView,
};
pub use event::{
    AcousticContribution, AcousticMixture, EventId, MixtureId, SoundEvent, SoundEventKind,
};
pub use expected::{
    ExpectedSound, ObservedSound, PlaybackMatchConfig, TranscriptHypothesis,
    playback_match_evidence,
};
pub use frame::SoundscapeFrame;
pub use isolation::{
    AudioSpan, IsolationEvaluation, NoopSourceSeparator, PlaybackCancellationSeparator,
    SeparationMethod, SeparationRequest, SeparationResult, SourceSeparator, SuppressionTarget,
    TrackingTarget, apply_separation_requests, evaluate_policies, self_hearing_suppression_policy,
};
pub use overlap::{MixtureComponent, OverlapMixture, detect_overlaps};
pub use signature::{
    FormantProfile, PitchProfile, ProsodyProfile, RateProfile, TimbreProfile, VoiceSignature,
    VoiceSignatureId, VoiceSignatureMatch, VoiceSignatureObservation,
};
pub use source::{SoundSource, SourceId, SourceKind, SourceLabel};
pub use time::{TimePoint, TimeRange};
pub use transcript::{AcousticMixtureId, SourceAttributedTranscript};
pub use voice_count::{
    RollingVoiceCountEstimator, VoiceActivityFrame, VoiceCount, VoiceCountConfig,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SoundscapeId(pub Uuid);

impl SoundscapeId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for SoundscapeId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VoiceId(pub Uuid);

impl VoiceId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for VoiceId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Soundscape {
    pub id: SoundscapeId,
    pub voices: Vec<Voice>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Voice {
    pub id: VoiceId,
    pub label: VoiceLabel,
    pub kind: VoiceKind,
    pub signatures: Vec<VoiceSignatureId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoiceLabel {
    Pete,
    Named(String),
    Unknown { ordinal: u32 },
    Cluster(String),
    Background,
    Environment,
}

impl VoiceLabel {
    pub fn display_label(&self) -> String {
        match self {
            Self::Pete => "PETE".to_string(),
            Self::Named(name) => name.trim().to_uppercase(),
            Self::Unknown { ordinal } => format!("UNKNOWN VOICE #{ordinal}"),
            Self::Cluster(name) => format!("VOICE CLUSTER {}", name.trim().to_uppercase()),
            Self::Background => "BACKGROUND VOICE".to_string(),
            Self::Environment => "ENVIRONMENT".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoiceKind {
    Pete,
    Human,
    Environment,
    Device,
    Unknown,
    Cluster,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceAttribution {
    pub voice_id: VoiceId,
    pub confidence: f32,
    pub role: VoiceRoleInSpan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoiceRoleInSpan {
    Speaker,
    Listener,
    SelfVoiceLeakage,
    Background,
    Overlap,
    Echo,
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::VoiceLabel;

    #[test]
    fn voice_labels_render_screenplay_friendly_cues() {
        assert_eq!(VoiceLabel::Pete.display_label(), "PETE");
        assert_eq!(
            VoiceLabel::Unknown { ordinal: 2 }.display_label(),
            "UNKNOWN VOICE #2"
        );
        assert_eq!(VoiceLabel::Named("Travis".into()).display_label(), "TRAVIS");
        assert_eq!(VoiceLabel::Background.display_label(), "BACKGROUND VOICE");
    }
}
