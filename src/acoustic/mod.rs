//! Acoustic model stage between timed symbolic plans and waveform vocoders.
//!
//! Acoustic models own duration-controlled frame layout. Vocoders consume the
//! resulting mel/F0 tracks and render waveforms.

mod model;
pub mod registry;
pub mod source_filter;

pub use model::{AcousticFrameTrack, AcousticInput, AcousticModelBackend, MelFrame, SingingPlan};
pub use registry::{AcousticModelDescriptor, acoustic_model_by_id, list_acoustic_models};
pub use source_filter::{
    SourceFilterAcousticModel, phone_timed_to_source_filter_track, source_filter_track_to_mel_f0,
};
