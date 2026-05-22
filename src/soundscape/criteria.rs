use serde::{Deserialize, Serialize};

use crate::soundscape::{SourceId, VoiceSignatureId};

/// Criteria used to select one or more sound sources in a soundscape frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceCriterion {
    KnownSource(SourceId),
    NotKnownSource(SourceId),
    MatchesSignature(VoiceSignatureId),
    HumanSpeech,
    SyntheticSpeech,
    Playback,
    Foreground,
    Background,
    UnknownVoice,
    CurrentlyAddressingPete,
}

/// Action to apply to sources matched by a [`SourceCriterion`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceOperation {
    Suppress,
    Enhance,
    Extract,
    Track,
}

/// Declarative source isolation policy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IsolationPolicy {
    pub operation: SourceOperation,
    pub criterion: SourceCriterion,
    pub strength: f32,
}

impl IsolationPolicy {
    pub fn suppress(criterion: SourceCriterion, strength: f32) -> Self {
        Self {
            operation: SourceOperation::Suppress,
            criterion,
            strength,
        }
    }

    pub fn track(criterion: SourceCriterion, strength: f32) -> Self {
        Self {
            operation: SourceOperation::Track,
            criterion,
            strength,
        }
    }
}
