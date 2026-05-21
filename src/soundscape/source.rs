use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Stable identifier for a sound source in the soundscape model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceId(pub Uuid);

impl SourceId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for SourceId {
    fn default() -> Self {
        Self::new()
    }
}

/// A source that contributes audible energy inside a frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoundSource {
    pub id: SourceId,
    pub kind: SourceKind,
    pub label: SourceLabel,
    /// Source attribution confidence in the inclusive range `[0.0, 1.0]`.
    pub confidence: f32,
}

/// Neutral source kinds for source-attributed listening.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceKind {
    Voice,
    SyntheticVoice,
    KnownSelfVoice,
    Playback,
    EnvironmentalNoise,
    Unknown,
}

/// Neutral source labels for script-friendly rendering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceLabel {
    NamedVoice(String),
    UnknownVoice { ordinal: u32 },
    BackgroundVoice { ordinal: u32 },
    Playback(String),
    RoomNoise,
    Custom(String),
}

impl SourceLabel {
    pub fn display_label(&self) -> String {
        match self {
            Self::NamedVoice(name) => format!("_{} VOICE_", name.trim().to_uppercase()),
            Self::UnknownVoice { ordinal } => format!("_UNKNOWN VOICE #{ordinal}_"),
            Self::BackgroundVoice { ordinal } => format!("_BACKGROUND VOICE #{ordinal}_"),
            Self::Playback(name) => format!("_{} PLAYBACK_", name.trim().to_uppercase()),
            Self::RoomNoise => "_ROOM NOISE_".to_string(),
            Self::Custom(label) => format!("_{}_", label.trim().to_uppercase()),
        }
    }
}
