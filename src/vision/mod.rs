use crate::time::ExactTimestamp;

pub mod speech;

pub use speech::{
    AvSyncConfig, EvidenceScore, PhonemeClass, VisualEvidenceStatus, VisualProvenance,
    VisualSpeechClaim, VisualSpeechClaimKind, VisualSpeechFrame, VisualSpeechTrace, VowelShape,
    extract_visual_speech_frame_from_rgba, visual_claim_hypotheses,
    visual_fusion_inputs_for_phone_hypotheses, visual_speech_claims_for_phone_hypotheses,
};

#[derive(Debug, Clone)]
pub struct VisionFrame {
    pub captured_at: ExactTimestamp,
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
}
