//! Realtime rule:
//! audio callbacks must not allocate, block, call models, log heavily, or acquire contended locks.
//! They should only move PCM through bounded realtime-safe buffers.

pub mod acoustic;
pub mod boundary;
pub mod capture;
pub mod dtw;
pub mod features;
pub mod frame;
pub mod hypothesis;
pub mod lattice;
pub mod noise_floor;
pub mod phone_class;
pub mod ring;
pub mod speech_likelihood;
pub mod streaming_prosody;
pub mod viterbi;
pub mod voice_signature;
pub mod wav;

pub use crate::audio::frame::AudioFrame;
pub use acoustic::{
    AcousticAnalysis, analyze_audio_frames, analyze_mono_samples,
    segment_pronunciation_with_acoustics,
};
pub use boundary::generate_boundary_hypotheses;
pub use dtw::{DtwTemplate, DtwTemplateMatcher};
pub use features::{AcousticFeatureFrame, AcousticFeatureStream, build_feature_stream};
pub use hypothesis::{
    HypothesisSource, HypothesisStatus, SpanHypothesis, SpanHypothesisId, SpanHypothesisKind,
};
pub use lattice::{
    EvidenceTraceEntry, FusionInput, FusionProfile, FusionResult, FusionWeights, HypothesisEdge,
    HypothesisEdgeKind, HypothesisLattice, SpeechEvidenceSource, SpeechHypothesisEngine,
    SpeechHypothesisFusion, fuse_hypotheses,
};
pub use noise_floor::{AdaptiveNoiseFloor, NoiseFloorConfig, NoiseFloorObservation};
pub use phone_class::{CoarsePhoneClass, classify_frame, generate_phone_class_hypotheses};
pub use speech_likelihood::{
    SpeechLikelihood, SpeechLikelihoodConfig, build_speech_likelihood_stream,
};
pub use viterbi::{PhoneState, viterbi_align_pronunciation};
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
