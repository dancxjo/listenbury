//! Shared span hypothesis model for mechanical speech-recognition generators.
//!
//! Every mechanical recogniser (endpoint detector, phone-class classifier,
//! DTW template matcher, Viterbi aligner) emits [`SpanHypothesis`] values.
//! This gives the rest of the pipeline a uniform interface for early,
//! provisional acoustic guesses that can be scored, visualised, and fused.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Identifier
// ---------------------------------------------------------------------------

/// Unique identifier for a single [`SpanHypothesis`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpanHypothesisId(pub String);

impl SpanHypothesisId {
    /// Generate a new random identifier.
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

impl Default for SpanHypothesisId {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Kind
// ---------------------------------------------------------------------------

/// Coarse kind of acoustic event a hypothesis represents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanHypothesisKind {
    /// A likely speech onset or offset, or a within-utterance boundary.
    SpeechBoundary,
    /// A candidate word span derived from acoustic cues.
    WordCandidate,
    /// A coarse phone-class label (e.g. vowel, fricative).
    PhoneClassCandidate,
    /// A fine-grained phone-level guess.
    PhoneCandidate,
    /// A candidate syllable boundary or syllable nucleus.
    SyllableCandidate,
    /// A pause or silence interval.
    PauseCandidate,
    /// A template-matched known sound.
    TemplateMatch,
    /// A phone aligned to a known pronunciation via forced alignment.
    PronunciationAlignment,
}

// ---------------------------------------------------------------------------
// Source
// ---------------------------------------------------------------------------

/// Which mechanical generator produced a hypothesis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HypothesisSource {
    /// Energy-based endpoint / speech-boundary detector.
    EndpointDetector,
    /// Heuristic phone-class classifier.
    PhoneClassifier,
    /// DTW template matcher.
    DtwTemplateMatcher,
    /// Viterbi forced-alignment over a known pronunciation.
    ViterbiAlignment,
    /// Backend-derived mouth-motion evidence from a local camera stream.
    VisualSpeech,
    /// Manually created (e.g. for testing or seeding).
    Manual,
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

/// Lifecycle state of a hypothesis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HypothesisStatus {
    /// Freshly emitted; not yet reviewed or confirmed.
    Provisional,
    /// Has been superseded by a revised estimate.
    Revised,
    /// Confirmed by a downstream stage.
    Confirmed,
    /// Rejected / suppressed.
    Rejected,
}

// ---------------------------------------------------------------------------
// The hypothesis itself
// ---------------------------------------------------------------------------

/// A provisional span hypothesis produced by a mechanical recogniser.
///
/// Hypotheses record everything needed to understand why they were made:
/// timing, score, confidence, source generator, and the features that drove
/// the decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpanHypothesis {
    /// Unique identifier.
    pub id: SpanHypothesisId,
    /// Semantic kind.
    pub kind: SpanHypothesisKind,
    /// Human-readable label (phone class name, template name, …).
    pub label: String,
    /// Start time in milliseconds from the audio origin.
    pub start_ms: u64,
    /// End time in milliseconds from the audio origin.
    pub end_ms: u64,
    /// Raw score from the generator (interpretation is generator-specific).
    pub score: f32,
    /// Normalised confidence in 0.0..=1.0.
    pub confidence: f32,
    /// Which generator produced this hypothesis.
    pub source: HypothesisSource,
    /// Names of the acoustic features that were used.
    pub features_used: Vec<String>,
    /// Lifecycle status.
    pub status: HypothesisStatus,
    /// Free-form JSON provenance record.
    pub provenance: serde_json::Value,
}

impl SpanHypothesis {
    /// Convenience constructor.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kind: SpanHypothesisKind,
        label: impl Into<String>,
        start_ms: u64,
        end_ms: u64,
        score: f32,
        confidence: f32,
        source: HypothesisSource,
        features_used: Vec<String>,
        provenance: serde_json::Value,
    ) -> Self {
        Self {
            id: SpanHypothesisId::new(),
            kind,
            label: label.into(),
            start_ms,
            end_ms,
            score,
            confidence,
            source,
            features_used,
            status: HypothesisStatus::Provisional,
            provenance,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn span_hypothesis_serialises_to_camel_case_json() {
        let hyp = SpanHypothesis::new(
            SpanHypothesisKind::SpeechBoundary,
            "speech_start",
            1000,
            1000,
            0.75,
            0.80,
            HypothesisSource::EndpointDetector,
            vec!["energy.onset".to_string()],
            json!({ "type": "onset" }),
        );

        let json = serde_json::to_string(&hyp).expect("serialise");
        assert!(json.contains("\"startMs\""));
        assert!(json.contains("\"endMs\""));
        assert!(json.contains("\"featuresUsed\""));
        // SpanHypothesisKind uses snake_case (not camelCase).
        assert!(json.contains("\"speech_boundary\""));
        assert!(json.contains("\"endpoint_detector\""));
        assert!(json.contains("\"provisional\""));
    }

    #[test]
    fn span_hypothesis_id_is_unique() {
        let a = SpanHypothesisId::new();
        let b = SpanHypothesisId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn span_hypothesis_default_status_is_provisional() {
        let hyp = SpanHypothesis::new(
            SpanHypothesisKind::PauseCandidate,
            "pause",
            500,
            700,
            0.8,
            0.75,
            HypothesisSource::EndpointDetector,
            vec![],
            json!(null),
        );
        assert_eq!(hyp.status, HypothesisStatus::Provisional);
    }
}
