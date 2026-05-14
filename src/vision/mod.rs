#[derive(Debug, Clone)]
pub struct VisionFrame {
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
}
