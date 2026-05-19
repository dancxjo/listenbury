use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Opaque identifier for a voice signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VoiceSignatureId(pub Uuid);

impl VoiceSignatureId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for VoiceSignatureId {
    fn default() -> Self {
        Self::new()
    }
}

/// Coarse label for a detected voice within an audio frame.
///
/// Labels are intentionally neutral: they represent evidence or hypotheses
/// about who is speaking, not a definitive registry lookup.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VoiceSignatureLabel {
    /// No speaker could be identified (e.g., silence, background noise).
    Unknown,
    /// The primary user's voice.
    User,
    /// Pete's own TTS output heard back through the microphone.
    PeteSelfVoice,
    /// A voice identified by a human-readable name.
    Named(String),
    /// A voice grouped by an unsupervised clustering algorithm.
    Cluster(String),
}

/// How a [`VoiceSignature`] was produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoiceSignatureSource {
    /// Assigned manually (e.g., by the user or a test fixture).
    Manual,
    /// Detected by self-voice suppression comparing microphone input to Pete's
    /// speaker output.
    SelfVoiceSuppression,
    /// Produced by a speaker embedding model.
    EmbeddingModel,
    /// Produced by an unsupervised clustering algorithm.
    Clustering,
    /// Inferred from a lightweight rule or heuristic.
    Heuristic,
}

/// A lightweight annotation that associates a speaker hypothesis with a
/// portion of audio.
///
/// Audio frames carry **zero or more** voice signatures to reflect the
/// reality that a frame may contain silence, a single speaker, or
/// overlapping speakers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceSignature {
    pub id: VoiceSignatureId,
    pub label: VoiceSignatureLabel,
    /// Confidence score in `[0.0, 1.0]` for this hypothesis.
    pub confidence: f32,
    pub source: VoiceSignatureSource,
}

impl VoiceSignature {
    /// Construct a new [`VoiceSignature`] with a freshly generated ID.
    pub fn new(
        label: VoiceSignatureLabel,
        confidence: f32,
        source: VoiceSignatureSource,
    ) -> Self {
        Self {
            id: VoiceSignatureId::new(),
            label,
            confidence,
            source,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_signature_id_is_unique() {
        let a = VoiceSignatureId::new();
        let b = VoiceSignatureId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn voice_signature_label_variants_are_debug() {
        let labels = [
            VoiceSignatureLabel::Unknown,
            VoiceSignatureLabel::User,
            VoiceSignatureLabel::PeteSelfVoice,
            VoiceSignatureLabel::Named("Alice".to_string()),
            VoiceSignatureLabel::Cluster("cluster-0".to_string()),
        ];
        for label in &labels {
            let _ = format!("{label:?}");
        }
    }

    #[test]
    fn voice_signature_new_assigns_distinct_ids() {
        let a = VoiceSignature::new(VoiceSignatureLabel::User, 0.9, VoiceSignatureSource::Manual);
        let b = VoiceSignature::new(
            VoiceSignatureLabel::Unknown,
            0.5,
            VoiceSignatureSource::Heuristic,
        );
        assert_ne!(a.id, b.id);
    }
}
