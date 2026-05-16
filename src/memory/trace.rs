use crate::time::ExactTimestamp;
use serde::{Deserialize, Serialize};

/// The role of a speaker in a conversation turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerRole {
    User,
    Pete,
    Unknown,
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
    /// A conversation turn (user utterance or Pete response) was finalised.
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
}
