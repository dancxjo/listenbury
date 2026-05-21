//! Hypothesis lattice and first-pass fusion layer.
//!
//! The lattice collects competing [`SpanHypothesis`] values from multiple
//! evidence sources and lets them support or contradict each other via
//! typed [`HypothesisEdge`]s. A first-pass [`fuse_hypotheses`] scorer
//! combines the evidence into a [`FusionResult`] with a resolved candidate,
//! confidence, and provenance.
//!
//! [`SpeechHypothesisEngine`] is the first-class top-level fusion pipeline. It
//! composes multiple evidence sources (acoustic, phonetic/pronunciation, ASR
//! stability, visual speech), standardizes confidence handling, and produces
//! stable/revisable span partitions with inspectable debug traces.
//!
//! Weighting policy is configured via [`FusionWeights`] and [`FusionProfile`],
//! keeping numeric heuristics out of the scoring mechanics.

mod engine;
mod evidence;
mod fusion;
mod graph;
mod sources;
mod weights;

pub use engine::{SpeechHypothesisEngine, SpeechHypothesisFusion};
pub use evidence::{EvidenceTraceEntry, SpeechEvidenceSource};
pub use fusion::{FusionInput, FusionResult, fuse_hypotheses};
pub use graph::{HypothesisEdge, HypothesisEdgeKind, HypothesisLattice};
pub use weights::{FusionProfile, FusionWeights};

#[cfg(test)]
mod tests;
