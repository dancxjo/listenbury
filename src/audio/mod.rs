//! Realtime rule:
//! audio callbacks must not allocate, block, call models, log heavily, or acquire contended locks.
//! They should only move PCM through bounded realtime-safe buffers.

pub mod acoustic;
pub mod frame;
pub mod ring;
pub mod voice_signature;
pub mod wav;

pub use crate::audio::frame::AudioFrame;
pub use acoustic::{AcousticAnalysis, analyze_audio_frames, analyze_mono_samples};
pub use voice_signature::{
    VoiceSignature, VoiceSignatureId, VoiceSignatureLabel, VoiceSignatureSource,
};
pub use wav::{
    read_wav_as_audio_frames, read_wav_as_whisper_frames, read_wav_frames, write_wav,
    write_wav_bytes,
};

pub trait AudioInput {
    fn poll_frames(&mut self) -> anyhow::Result<Vec<AudioFrame>>;
}

pub trait AudioOutput {
    fn push_frame(&mut self, frame: AudioFrame) -> anyhow::Result<()>;
}
