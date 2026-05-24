//! Acoustic model stage between timed symbolic plans and waveform vocoders.
//!
//! Acoustic models own duration-controlled frame layout. Vocoders consume the
//! resulting mel/F0 tracks and render waveforms.

mod model;
pub mod neural;
pub mod registry;
pub mod source_filter;

pub use model::{AcousticFrameTrack, AcousticInput, AcousticModelBackend, MelFrame, SingingPlan};
pub use neural::{
    FastSpeech2AcousticModel, MatchaAcousticModel, NeuralAcousticModel, NeuralAcousticModelKind,
    NeuralAcousticOnnxConfig, NeuralAcousticTensorNames, NeuralAcousticTrackContract,
    NeuralMelOutputLayout, NeuralPhoneIdMap, SpeechT5AcousticModel, VitsPiperAcousticModel,
};
pub use registry::{AcousticModelDescriptor, acoustic_model_by_id, list_acoustic_models};
pub use source_filter::{
    MelTemporalDiscontinuityStats, SourceFilterAcousticModel, mel_frame_delta_energy,
    phone_timed_to_source_filter_track, source_filter_track_to_mel_f0,
    summarize_mel_temporal_discontinuity, temporal_smooth_mel_frames,
};
