//! Voice execution-refinement passes for sung and chant-like output.
//!
//! This module provides late-stage processing of articulation plans produced
//! by upstream prosody and articulator components. It does not model prosody
//! itself; rather, it refines an already-planned sequence of phone gestures
//! to account for cross-boundary effects.
//!
//! # Modules
//!
//! - [`coarticulation`] – smoothing phone boundaries: legato transitions,
//!   unvoiced consonant pitch suppression, coda time borrowing, and boundary
//!   classification.

pub mod coarticulation;

pub use coarticulation::{
    BoundaryKind, CoarticulatedPlan, PhoneGesture, PhoneRole, RefinedGesture, VocalGesturePlan,
    coarticulate,
};
