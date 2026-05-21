use serde::{Deserialize, Serialize};

use crate::audio::hypothesis::SpanHypothesisId;

use super::{FusionInput, HypothesisLattice};

/// Composable source of fusion evidence for [`SpeechHypothesisEngine`].
pub trait SpeechEvidenceSource: Send + Sync {
    /// Stable source name used in debug traces.
    fn name(&self) -> &'static str;
    /// Produce evidence for hypotheses currently in `lattice`.
    fn collect(&self, lattice: &HypothesisLattice) -> Vec<(SpanHypothesisId, FusionInput)>;
}

/// Debug record showing which source produced which fusion signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceTraceEntry {
    pub source: String,
    pub hypothesis_id: SpanHypothesisId,
    pub input: FusionInput,
}
