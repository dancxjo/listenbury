use crate::time::ExactTimestamp;

#[derive(Debug, Clone)]
pub struct VisionFrame {
    pub captured_at: ExactTimestamp,
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
}
