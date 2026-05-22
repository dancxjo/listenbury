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

pub use articulator::{ArticulatorPlan, EnergyCurve, EnergyPoint, articulate, is_phone_voiced};
pub use coarticulation::{
    BoundaryKind, CoarticulatedPlan, PhoneGesture, PhoneRole, RefinedGesture, VocalGesturePlan,
    coarticulate,
};
