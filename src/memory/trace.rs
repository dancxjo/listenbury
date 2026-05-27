use crate::time::ExactTimestamp;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A voice label captured in a memory trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerRole {
    Pete,
    Named(String),
    UnknownVoice { ordinal: u32 },
    BackgroundVoice,
    Environment,
}

/// A semantic referent explicitly extracted from text.
///
/// `node_id` is a stable graph referent ID such as `person:travis`; source
/// artifacts should link to it instead of becoming the referent themselves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryEntityMention {
    pub node_id: String,
    pub label: String,
    pub kind: String,
    pub confidence: f32,
    pub span_start: usize,
    pub span_end: usize,
}

/// A precomputed picture vector derived from a transient camera frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryImageVector {
    pub image_id: String,
    pub source: String,
    pub width: u32,
    pub height: u32,
    pub vector: Vec<f32>,
    pub content_node_id: Option<String>,
    pub retained_image: bool,
}

/// A precomputed voice vector derived from heard speech.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryVoiceVector {
    pub voice_signature_id: String,
    pub voice_node_id: String,
    pub source: String,
    pub span_id: Option<u64>,
    pub vector: Vec<f32>,
    pub confidence: f32,
}

/// Field/property updates Pete explicitly applies to an existing or provisional graph node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryGraphNodeFieldUpdate {
    pub node_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub fields: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_text: Option<String>,
    pub confidence: f32,
}

/// A single runtime trace event emitted by the Listenbury engine.
///
/// Traces are produced on the hot path but consumed asynchronously via a
/// [`MemorySink`].  They capture what the system experienced — heard speech,
/// generated text, mouth playback, timed word streams, auditory observations,
/// overlap events, and recall results — without imposing any database schema.
///
/// [`MemorySink`]: crate::memory::sink::MemorySink
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MemoryTrace {
    /// A conversation turn (voice utterance or Pete response) was finalised.
    ConversationTurnFinalized {
        speaker: SpeakerRole,
        text: String,
        occurred_at: ExactTimestamp,
    },
    /// A timed word stream reached its end and was summarised.
    TimedWordStreamFinalized {
        stream_id: String,
        summary: String,
        occurred_at: ExactTimestamp,
    },
    /// The mouth began playing back a synthesised utterance.
    MouthPlaybackStarted {
        utterance_id: u64,
        text: String,
        occurred_at: ExactTimestamp,
    },
    /// The mouth finished playing back a synthesised utterance.
    MouthPlaybackCompleted {
        utterance_id: u64,
        text: String,
        occurred_at: ExactTimestamp,
    },
    /// The auditory scene produced a noteworthy observation.
    AuditorySceneObservation {
        description: String,
        /// Normalized salience in `[0.0, 1.0]`.
        salience: f32,
        occurred_at: ExactTimestamp,
    },
    /// Two speakers were detected talking at the same time.
    OverlapDetected {
        description: String,
        occurred_at: ExactTimestamp,
    },
    /// A recall query returned a result that was used by the system.
    RecallResultUsed {
        query: String,
        result_summary: String,
        occurred_at: ExactTimestamp,
    },
    /// Pete explicitly ran entity extraction over text.
    EntityExtractionPerformed {
        source_text: String,
        entities: Vec<MemoryEntityMention>,
        occurred_at: ExactTimestamp,
    },
    /// Pete explicitly updated fields/properties on a graph node.
    GraphNodeFieldsUpdated {
        update: MemoryGraphNodeFieldUpdate,
        occurred_at: ExactTimestamp,
    },
    /// A camera frame was vectorized without retaining raw image bytes.
    ImageVectorCaptured {
        image: MemoryImageVector,
        captured_at: ExactTimestamp,
    },
    /// A heard voice segment was assigned a signature ID and vector.
    VoiceVectorCaptured {
        voice: MemoryVoiceVector,
        captured_at: ExactTimestamp,
    },
}

impl MemoryTrace {
    /// Return the stable snake_case discriminator for this trace.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::ConversationTurnFinalized { .. } => "conversation_turn_finalized",
            Self::TimedWordStreamFinalized { .. } => "timed_word_stream_finalized",
            Self::MouthPlaybackStarted { .. } => "mouth_playback_started",
            Self::MouthPlaybackCompleted { .. } => "mouth_playback_completed",
            Self::AuditorySceneObservation { .. } => "auditory_scene_observation",
            Self::OverlapDetected { .. } => "overlap_detected",
            Self::RecallResultUsed { .. } => "recall_result_used",
            Self::EntityExtractionPerformed { .. } => "entity_extraction_performed",
            Self::GraphNodeFieldsUpdated { .. } => "graph_node_fields_updated",
            Self::ImageVectorCaptured { .. } => "image_vector_captured",
            Self::VoiceVectorCaptured { .. } => "voice_vector_captured",
        }
    }

    /// Return the timestamp at which the runtime observed this trace.
    pub fn occurred_at(&self) -> ExactTimestamp {
        match self {
            Self::ConversationTurnFinalized { occurred_at, .. }
            | Self::TimedWordStreamFinalized { occurred_at, .. }
            | Self::MouthPlaybackStarted { occurred_at, .. }
            | Self::MouthPlaybackCompleted { occurred_at, .. }
            | Self::AuditorySceneObservation { occurred_at, .. }
            | Self::OverlapDetected { occurred_at, .. }
            | Self::RecallResultUsed { occurred_at, .. }
            | Self::EntityExtractionPerformed { occurred_at, .. }
            | Self::GraphNodeFieldsUpdated { occurred_at, .. } => *occurred_at,
            Self::ImageVectorCaptured { captured_at, .. }
            | Self::VoiceVectorCaptured { captured_at, .. } => *captured_at,
        }
    }
}
