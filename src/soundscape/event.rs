use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::soundscape::{SourceId, TimeRange};

/// Stable identifier for a single attributed sound event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventId(pub Uuid);

impl EventId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for EventId {
    fn default() -> Self {
        Self::new()
    }
}

/// Stable identifier for a mixture that combines multiple sources/events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MixtureId(pub Uuid);

impl MixtureId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for MixtureId {
    fn default() -> Self {
        Self::new()
    }
}

/// A single audible event attributed to one source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoundEvent {
    pub id: EventId,
    pub source_id: SourceId,
    pub kind: SoundEventKind,
    pub range: TimeRange,
    pub confidence: f32,
}

/// Sound event taxonomy independent from transcript role terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SoundEventKind {
    VoiceActivity,
    PlaybackActivity,
    EnvironmentalNoise,
    Echo,
    Unknown,
}

/// Per-source contribution in an acoustic mixture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticContribution {
    pub source_id: SourceId,
    pub gain: f32,
}

/// Multi-source blend for one frame range.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticMixture {
    pub id: MixtureId,
    pub event_ids: Vec<EventId>,
    pub contributions: Vec<AcousticContribution>,
}
