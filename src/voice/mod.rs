//! Voice execution-refinement passes for sung and chant-like output.
//!
//! This module provides late-stage processing of articulation plans produced
//! by upstream prosody and articulator components. It does not model prosody
//! itself; rather, it refines an already-planned sequence of phone gestures
//! to account for cross-boundary effects.
//!
//! # Modules
//!
//! - [`articulator`] – converts a [`crate::prosody::singing::SungPhrase`] into
//!   a renderer-neutral [`articulator::ArticulatorPlan`] containing ordered
//!   phone gestures, a pitch curve, and an energy curve.
//! - [`coarticulation`] – smoothing phone boundaries: legato transitions,
//!   unvoiced consonant pitch suppression, coda time borrowing, and boundary
//!   classification.

pub mod articulator;
pub mod coarticulation;
pub mod tract;
pub mod vocal_plausibility;

pub use articulator::{
    articulate, backend_detail_expectation, is_phone_voiced, klatt_targets_from_articulator_plan,
    ArticulatorPlan, EnergyCurve, EnergyPoint, GestureSpan, SungBackendDetail, SungBackendKind,
    SyllableRenderSpan,
};
pub use coarticulation::{
    coarticulate, BoundaryKind, CoarticulatedPlan, PhoneGesture, PhoneRole, RefinedGesture,
    VocalGesturePlan,
};
pub use tract::{
    default_english_phone_targets, estimate_f0_autocorrelation, phone_render_targets_from_string,
    render_phone, render_phone_string, render_targets_from_sung_syllable,
    render_targets_from_syllable, source_filter_track_from_acoustic,
    source_filter_track_from_acoustic_full, FormantEstimation, GlottalSourceEstimate,
    GlottalSourceTarget, KlattRenderConfig, NoiseEstimate, PhoneAcousticTarget, PhoneRenderTarget,
    SourceFilterFrame, SourceFilterTrack, VocalTractFilterEstimate, VocalTractFilterTarget,
    VoicingEstimate,
};
pub use vocal_plausibility::{
    assess_vocal_plausibility, VocalPlausibility, VocalPlausibilityConfig,
};
