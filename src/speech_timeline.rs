use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub Uuid);

impl SessionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TurnId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UtteranceId(pub Uuid);

impl UtteranceId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for UtteranceId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SyntheticUnitId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TranscriptRevisionId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpanId(pub Uuid);

impl SpanId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for SpanId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AudioClipId(pub Uuid);

impl AudioClipId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for AudioClipId {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_roundtrip_through_json() {
        let session = SessionId::new();
        let turn = TurnId(7);
        let utterance = UtteranceId::new();
        let synthetic_unit = SyntheticUnitId(42);
        let revision = TranscriptRevisionId(9);
        let span = SpanId::new();
        let clip = AudioClipId::new();

        let session_json = serde_json::to_string(&session).expect("serialize SessionId");
        let turn_json = serde_json::to_string(&turn).expect("serialize TurnId");
        let utterance_json = serde_json::to_string(&utterance).expect("serialize UtteranceId");
        let synthetic_json =
            serde_json::to_string(&synthetic_unit).expect("serialize SyntheticUnitId");
        let revision_json =
            serde_json::to_string(&revision).expect("serialize TranscriptRevisionId");
        let span_json = serde_json::to_string(&span).expect("serialize SpanId");
        let clip_json = serde_json::to_string(&clip).expect("serialize AudioClipId");

        assert_eq!(
            serde_json::from_str::<SessionId>(&session_json).expect("deserialize SessionId"),
            session
        );
        assert_eq!(
            serde_json::from_str::<TurnId>(&turn_json).expect("deserialize TurnId"),
            turn
        );
        assert_eq!(
            serde_json::from_str::<UtteranceId>(&utterance_json).expect("deserialize UtteranceId"),
            utterance
        );
        assert_eq!(
            serde_json::from_str::<SyntheticUnitId>(&synthetic_json)
                .expect("deserialize SyntheticUnitId"),
            synthetic_unit
        );
        assert_eq!(
            serde_json::from_str::<TranscriptRevisionId>(&revision_json)
                .expect("deserialize TranscriptRevisionId"),
            revision
        );
        assert_eq!(
            serde_json::from_str::<SpanId>(&span_json).expect("deserialize SpanId"),
            span
        );
        assert_eq!(
            serde_json::from_str::<AudioClipId>(&clip_json).expect("deserialize AudioClipId"),
            clip
        );
    }
}
