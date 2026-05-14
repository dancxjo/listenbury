#[derive(Debug, Clone)]
pub struct AudioFrame {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}
