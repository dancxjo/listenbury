use crate::time::ExactTimestamp;

#[cfg(target_os = "linux")]
pub mod linux_video;
pub mod speech;
pub mod vector;

#[cfg(target_os = "linux")]
pub use linux_video::{
    LinuxVideoCaptureConfig, NativeVideoCaptureHandle, ffmpeg_linux_video_args,
    spawn_linux_video_vector_capture,
};
pub use speech::{
    AvSyncConfig, EvidenceScore, PhonemeClass, VisualEvidenceStatus, VisualProvenance,
    VisualSpeechClaim, VisualSpeechClaimKind, VisualSpeechFrame, VisualSpeechTrace, VowelShape,
    extract_visual_speech_frame_from_rgba, visual_claim_hypotheses,
    visual_fusion_inputs_for_phone_hypotheses, visual_speech_claims_for_phone_hypotheses,
};
pub use vector::{IMAGE_VECTOR_DIMS, ImageVectorObservation, vectorize_rgba_frame};

#[derive(Debug, Clone)]
pub struct VisionFrame {
    pub captured_at: ExactTimestamp,
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
}
