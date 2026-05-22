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
pub mod mbrola;
pub mod tract;
pub mod vocal_plausibility;

pub use articulator::{
    ArticulatorPlan, EnergyCurve, EnergyPoint, GestureSpan, PartialProsodyPhone, PitchHint,
    RenderPlan, SungBackendDetail, SungBackendKind, SyllableRenderSpan, articulate,
    articulate_with_decomposition_policy, articulate_with_inventory,
    articulate_with_inventory_and_decomposition_policy, backend_detail_expectation,
    coarse_text_render_plan, is_phone_voiced, klatt_targets_from_articulator_plan,
    partial_prosody_render_plan, render_plan_for_backend,
};
pub use coarticulation::{
    BoundaryKind, CoarticulatedPlan, PhoneGesture, PhoneRole, RefinedGesture, VocalGesturePlan,
    coarticulate,
};
pub use mbrola::{
    MbrolaPhone, MbrolaPitchTarget, MbrolaRenderer, MbrolaRendererConfig, MbrolaSymbolMap,
    MbrolaVoice, PhoneTimedPlan, PhoneTimedRenderer, RenderReport, phone_timed_plan_to_pho,
    prosody_timing_plan_to_phone_timed_plan, read_pho_file, write_pho_file,
};
pub use tract::{
    FormantEstimation, GlottalSourceEstimate, GlottalSourceTarget, KlattRenderConfig,
    NoiseEstimate, PhoneAcousticTarget, PhoneRenderTarget, SourceFilterFrame, SourceFilterTrack,
    VocalTractFilterEstimate, VocalTractFilterTarget, VoicingEstimate,
    default_english_phone_targets, estimate_f0_autocorrelation, phone_render_targets_from_string,
    render_phone, render_phone_string, render_targets_from_sung_syllable,
    render_targets_from_syllable, source_filter_track_from_acoustic,
    source_filter_track_from_acoustic_full,
};
pub use vocal_plausibility::{
    VocalPlausibility, VocalPlausibilityConfig, assess_vocal_plausibility,
};
