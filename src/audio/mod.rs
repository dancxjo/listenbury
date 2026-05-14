//! Realtime rule:
//! audio callbacks must not allocate, block, call models, log heavily, or acquire contended locks.
//! They should only move PCM through bounded realtime-safe buffers.

pub mod frame;
pub mod ring;

use crate::audio::frame::AudioFrame;

pub trait AudioInput {
    fn poll_frames(&mut self) -> anyhow::Result<Vec<AudioFrame>>;
}

pub trait AudioOutput {
    fn push_frame(&mut self, frame: AudioFrame) -> anyhow::Result<()>;
}
