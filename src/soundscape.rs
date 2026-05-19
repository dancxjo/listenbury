use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::audio::VoiceSignatureId;

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
        assert_eq!(
            VoiceLabel::Background.display_label(),
            "BACKGROUND VOICE"
        );
    }
}
