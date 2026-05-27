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
pub mod pipeline;
pub mod signature;
pub mod source;
pub mod time;
pub mod transcript;
pub mod voice_count;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{span::SpanId, time::ExactTimestamp};

const PETE_KNOWN_VOICE_LABEL: &str = "PETE";

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
pub use pipeline::SoundscapePipelineAdapter;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VoiceEnrollmentSampleId(pub Uuid);

impl VoiceEnrollmentSampleId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for VoiceEnrollmentSampleId {
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
pub struct KnownVoice {
    pub id: VoiceId,
    pub label: String,
    pub kind: VoiceKind,
    pub enrollment_samples: Vec<VoiceEnrollmentSampleId>,
    pub created_at: ExactTimestamp,
    pub notes: Option<String>,
}

impl KnownVoice {
    pub fn pete(created_at: ExactTimestamp) -> Self {
        Self {
            id: VoiceId::new(),
            label: PETE_KNOWN_VOICE_LABEL.to_string(),
            kind: VoiceKind::Pete,
            enrollment_samples: Vec::new(),
            created_at,
            notes: Some("Known self-voice identity".to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnrollmentSource {
    ManualLabel,
    GeneratedTts,
    ExplicitEnrollmentCommand,
    ImportedFixture,
    ClusteringPromotion,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnrollmentQuality {
    Low,
    Medium,
    High,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingRef {
    pub backend: String,
    pub key: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoiceEnrollmentSample {
    pub id: VoiceEnrollmentSampleId,
    pub voice_id: VoiceId,
    pub audio_span_id: SpanId,
    pub source: EnrollmentSource,
    pub quality: EnrollmentQuality,
    pub embedding_ref: Option<EmbeddingRef>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct KnownVoiceRegistry {
    pub voices: Vec<KnownVoice>,
    pub enrollment_samples: Vec<VoiceEnrollmentSample>,
    #[serde(default)]
    pub voice_entity_associations: Vec<VoiceEntityAssociation>,
}

impl KnownVoiceRegistry {
    pub fn ensure_pete_voice(&mut self, created_at: ExactTimestamp) -> VoiceId {
        if let Some(existing) = self
            .voices
            .iter()
            .find(|voice| voice.kind == VoiceKind::Pete)
        {
            return existing.id;
        }
        let pete = KnownVoice::pete(created_at);
        let id = pete.id;
        self.voices.push(pete);
        id
    }

    pub fn associate_voice_with_entity(
        &mut self,
        voice_id: VoiceId,
        entity_node_id: impl Into<String>,
        entity_label: Option<String>,
        confidence: f32,
        source: VoiceEntityAssociationSource,
        associated_at: ExactTimestamp,
    ) -> VoiceEntityAssociation {
        let association = VoiceEntityAssociation {
            voice_id,
            entity_node_id: entity_node_id.into(),
            entity_label,
            confidence: confidence.clamp(0.0, 1.0),
            source,
            associated_at,
        };
        if let Some(existing) = self.voice_entity_associations.iter_mut().find(|existing| {
            existing.voice_id == association.voice_id
                && existing.entity_node_id == association.entity_node_id
        }) {
            *existing = association.clone();
        } else {
            self.voice_entity_associations.push(association.clone());
        }
        association
    }

    pub fn entities_for_voice(&self, voice_id: VoiceId) -> Vec<&VoiceEntityAssociation> {
        self.voice_entity_associations
            .iter()
            .filter(|association| association.voice_id == voice_id)
            .collect()
    }

    pub fn voices_for_entity(&self, entity_node_id: &str) -> Vec<&VoiceEntityAssociation> {
        self.voice_entity_associations
            .iter()
            .filter(|association| association.entity_node_id == entity_node_id)
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceEntityAssociation {
    pub voice_id: VoiceId,
    pub entity_node_id: String,
    pub entity_label: Option<String>,
    pub confidence: f32,
    pub source: VoiceEntityAssociationSource,
    pub associated_at: ExactTimestamp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoiceEntityAssociationSource {
    Manual,
    ExplicitUserStatement,
    EntityExtractionCommand,
    EnrollmentMetadata,
    Inference,
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
    #[serde(default)]
    pub span_id: Option<SpanId>,
    pub confidence: f32,
    #[serde(default)]
    pub source: VoiceAttributionSource,
    #[serde(default)]
    pub alternatives: Vec<VoiceAttributionAlternative>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceAttributionAlternative {
    pub voice_id: VoiceId,
    pub confidence: f32,
    pub source: VoiceAttributionSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoiceAttributionSource {
    Unknown,
    ManualLabel,
    EnrollmentMatch,
    GeneratedTts,
    Clustering,
    Heuristic,
    MockMatcher,
}

impl Default for VoiceAttributionSource {
    fn default() -> Self {
        Self::Unknown
    }
}

pub trait VoiceMatcher {
    fn attribute(
        &self,
        span_id: SpanId,
        signature_ids: &[VoiceSignatureId],
        registry: &KnownVoiceRegistry,
    ) -> Vec<VoiceAttribution>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MockVoiceMatcher;

impl VoiceMatcher for MockVoiceMatcher {
    fn attribute(
        &self,
        span_id: SpanId,
        _signature_ids: &[VoiceSignatureId],
        registry: &KnownVoiceRegistry,
    ) -> Vec<VoiceAttribution> {
        // Intentional seam: this mock ignores `signature_ids` and emits one
        // deterministic attribution so callers can exercise
        // registry/attribution plumbing before a real matcher is wired. A real
        // matcher should validate signature compatibility before attribution.
        registry
            .voices
            .first()
            .map(|voice| VoiceAttribution {
                voice_id: voice.id,
                span_id: Some(span_id),
                confidence: 0.5,
                source: VoiceAttributionSource::MockMatcher,
                alternatives: Vec::new(),
            })
            .into_iter()
            .collect()
    }
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
    use super::{
        EnrollmentQuality, EnrollmentSource, KnownVoice, KnownVoiceRegistry, VoiceAttribution,
        VoiceAttributionAlternative, VoiceAttributionSource, VoiceEnrollmentSample,
        VoiceEnrollmentSampleId, VoiceEntityAssociationSource, VoiceId, VoiceKind, VoiceLabel,
    };
    use crate::{span::SpanId, time::ExactTimestamp};

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

    #[test]
    fn registry_serializes_with_enrollment_provenance() {
        let created_at = ExactTimestamp::from_unix_nanos(1_750_000_000_000_000_000);
        let voice_id = VoiceId::new();
        let sample_id = VoiceEnrollmentSampleId::new();
        let span_id = SpanId(1);
        let registry = KnownVoiceRegistry {
            voices: vec![KnownVoice {
                id: voice_id,
                label: "TRAVIS".to_string(),
                kind: VoiceKind::Human,
                enrollment_samples: vec![sample_id],
                created_at,
                notes: Some("Manual enrollment from WaveDeck".to_string()),
            }],
            enrollment_samples: vec![VoiceEnrollmentSample {
                id: sample_id,
                voice_id,
                audio_span_id: span_id,
                source: EnrollmentSource::ManualLabel,
                quality: EnrollmentQuality::High,
                embedding_ref: None,
            }],
            voice_entity_associations: Vec::new(),
        };

        let json = serde_json::to_string(&registry).expect("registry should serialize");
        let decoded: KnownVoiceRegistry =
            serde_json::from_str(&json).expect("registry should deserialize");
        assert_eq!(decoded, registry);
    }

    #[test]
    fn supports_zero_or_many_attributions_per_span() {
        let span_id = SpanId(42);
        let attributions = vec![
            VoiceAttribution {
                voice_id: VoiceId::new(),
                span_id: Some(span_id),
                confidence: 0.72,
                source: VoiceAttributionSource::EnrollmentMatch,
                alternatives: vec![VoiceAttributionAlternative {
                    voice_id: VoiceId::new(),
                    confidence: 0.61,
                    source: VoiceAttributionSource::Heuristic,
                }],
            },
            VoiceAttribution {
                voice_id: VoiceId::new(),
                span_id: Some(span_id),
                confidence: 0.55,
                source: VoiceAttributionSource::Clustering,
                alternatives: vec![],
            },
        ];

        let empty: Vec<VoiceAttribution> = Vec::new();
        assert!(empty.is_empty());
        assert_eq!(
            attributions
                .iter()
                .filter(|item| item.span_id == Some(span_id))
                .count(),
            2
        );
        assert_eq!(attributions[0].alternatives.len(), 1);
    }

    #[test]
    fn attribution_defaults_keep_legacy_payloads_optional() {
        let voice_id = VoiceId::new();
        let payload = serde_json::json!({
            "voice_id": voice_id,
            "confidence": 0.64
        });

        let attribution: VoiceAttribution =
            serde_json::from_value(payload).expect("legacy payload should deserialize");
        assert_eq!(attribution.voice_id, voice_id);
        assert_eq!(attribution.span_id, None);
        assert_eq!(attribution.source, VoiceAttributionSource::Unknown);
        assert!(attribution.alternatives.is_empty());
    }

    #[test]
    fn pete_is_distinct_known_voice_identity() {
        let pete = KnownVoice::pete(ExactTimestamp::from_unix_nanos(1_000));
        let travis = KnownVoice {
            id: VoiceId::new(),
            label: "TRAVIS".to_string(),
            kind: VoiceKind::Human,
            enrollment_samples: Vec::new(),
            created_at: ExactTimestamp::from_unix_nanos(2_000),
            notes: None,
        };

        assert_eq!(pete.kind, VoiceKind::Pete);
        assert_ne!(pete.id, travis.id);
    }

    #[test]
    fn registry_associates_voice_ids_with_entity_nodes() {
        let mut registry = KnownVoiceRegistry::default();
        let voice_id = VoiceId::new();

        let association = registry.associate_voice_with_entity(
            voice_id,
            "person:travis",
            Some("Travis".to_string()),
            0.91,
            VoiceEntityAssociationSource::ExplicitUserStatement,
            ExactTimestamp::from_unix_nanos(10),
        );

        assert_eq!(association.entity_node_id, "person:travis");
        assert_eq!(registry.entities_for_voice(voice_id).len(), 1);
        assert_eq!(registry.voices_for_entity("person:travis").len(), 1);
    }
}
