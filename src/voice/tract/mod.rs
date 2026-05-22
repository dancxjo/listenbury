//! Vocal-tract source/filter analysis and synthesis.
//!
//! This module is split into two focused layers:
//!
//! **Layer A — [`analysis`]**: normalised source/filter acoustic analysis.
//! Converts existing [`crate::audio::acoustic::AcousticAnalysis`] outputs
//! into compact [`analysis::SourceFilterFrame`] / [`analysis::SourceFilterTrack`]
//! evidence suitable for vocal-tract reasoning, phone-class hypothesis
//! generation, and soundscape signature updates.
//!
//! **Layer B — [`klatt`]**: a small deterministic Klatt-style source/filter
//! synthesiser that renders controlled diagnostic phones and syllables from
//! explicit acoustic targets.
//!
//! # Integration seams
//!
//! - [`analysis::source_filter_track_from_acoustic`] accepts
//!   [`crate::audio::acoustic::AcousticAnalysis`] and re-uses the existing
//!   formant tracks rather than replacing them.
//! - [`targets::phone_render_targets_from_string`] accepts
//!   [`crate::linguistic::phonology::PhoneString`] and the shared
//!   [`crate::linguistic::phonology::Phone`] type.
//! - [`targets::render_targets_from_syllable`] and
//!   [`targets::render_targets_from_sung_syllable`] accept the existing
//!   [`crate::prosody::syllable::Syllable`] /
//!   [`crate::prosody::syllable::SungSyllable`] types.
//!
//! # Bridge note
//!
//! `soundscape::signature::VoiceSignature` and `audio::voice_signature` cover
//! overlapping territory.  `SourceFilterTrack` outputs (F0, formants) should
//! be used to populate [`crate::soundscape::signature::VoiceSignatureObservation`]
//! fields (`pitch_profile`, `formant_profile`).  The two signature worlds
//! should eventually be unified; this module avoids forking the concept
//! further by not introducing a third signature type.
//!
//! TODO: add a helper `source_filter_track_to_observation()` that maps
//! `SourceFilterTrack` → `VoiceSignatureObservation` for direct ingestion by
//! `soundscape::signature::VoiceSignature::update_with_observation`.

pub mod analysis;
pub mod filter;
pub mod klatt;
pub mod source;
pub mod targets;

// ---------------------------------------------------------------------------
// Convenience re-exports
// ---------------------------------------------------------------------------

pub use analysis::{
    SourceFilterFrame, SourceFilterTrack, estimate_f0_autocorrelation,
    source_filter_track_from_acoustic, source_filter_track_from_acoustic_full,
};
pub use filter::{FormantEstimation, VocalTractFilterEstimate};
pub use klatt::{KlattRenderConfig, render_phone, render_phone_string};
pub use source::{GlottalSourceEstimate, NoiseEstimate, VoicingEstimate};
pub use targets::{
    GlottalSourceTarget, PhoneAcousticTarget, PhoneRenderTarget, VocalTractFilterTarget,
    default_english_phone_targets, klatt_targets_from_features, phone_render_targets_from_string,
    render_targets_from_sung_syllable, render_targets_from_syllable,
};
