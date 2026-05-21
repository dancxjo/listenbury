use serde::{Deserialize, Serialize};

use crate::audio::hypothesis::{HypothesisStatus, SpanHypothesis, SpanHypothesisId};

/// The semantic relationship between two hypotheses in the lattice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HypothesisEdgeKind {
    /// One hypothesis provides supporting evidence for another.
    Supports,
    /// One hypothesis is inconsistent with / contradicts another.
    Contradicts,
    /// One hypothesis is a more precise version of another.
    Refines,
    /// One hypothesis fully contains the span of another.
    Contains,
    /// Two hypotheses are temporally aligned (same or very similar timing).
    AlignedTo,
    /// One hypothesis is derived from another by a deterministic transform.
    DerivedFrom,
    /// One hypothesis supersedes / revises another (the target is now stale).
    RevisionOf,
}

/// A directed edge between two hypotheses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisEdge {
    /// Source hypothesis identifier.
    pub from: SpanHypothesisId,
    /// Target hypothesis identifier.
    pub to: SpanHypothesisId,
    /// Semantic kind of the relationship.
    pub kind: HypothesisEdgeKind,
    /// Optional scalar weight on the edge (0.0–1.0).
    pub weight: f32,
}

/// A graph of competing and collaborating span hypotheses.
///
/// Hypotheses are never deleted from the lattice; instead their
/// [`HypothesisStatus`] is updated to `Revised` or `Rejected` so the full
/// revision history remains inspectable.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HypothesisLattice {
    /// All hypotheses ever added to this lattice (including superseded ones).
    pub hypotheses: Vec<SpanHypothesis>,
    /// Directed edges between hypotheses.
    pub edges: Vec<HypothesisEdge>,
}

impl HypothesisLattice {
    /// Create an empty lattice.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a hypothesis and return a clone of its identifier.
    pub fn add(&mut self, hypothesis: SpanHypothesis) -> SpanHypothesisId {
        let id = hypothesis.id.clone();
        self.hypotheses.push(hypothesis);
        id
    }

    /// Connect two hypotheses with a typed, weighted edge.
    pub fn connect(
        &mut self,
        from: SpanHypothesisId,
        to: SpanHypothesisId,
        kind: HypothesisEdgeKind,
        weight: f32,
    ) {
        self.edges.push(HypothesisEdge {
            from,
            to,
            kind,
            weight,
        });
    }

    /// Mark an existing hypothesis as revised and add the replacement.
    ///
    /// The old hypothesis is updated to [`HypothesisStatus::Revised`] and
    /// a [`HypothesisEdgeKind::RevisionOf`] edge is added from the new one
    /// to the old one so the full history remains inspectable.
    pub fn revise(
        &mut self,
        old_id: &SpanHypothesisId,
        revised: SpanHypothesis,
    ) -> SpanHypothesisId {
        if let Some(old) = self.hypotheses.iter_mut().find(|h| &h.id == old_id) {
            old.status = HypothesisStatus::Revised;
        }
        let old_id = old_id.clone();
        let new_id = revised.id.clone();
        self.hypotheses.push(revised);
        self.edges.push(HypothesisEdge {
            from: new_id.clone(),
            to: old_id,
            kind: HypothesisEdgeKind::RevisionOf,
            weight: 1.0,
        });
        new_id
    }

    /// Return only hypotheses that are currently active (not revised/rejected).
    pub fn active_hypotheses(&self) -> Vec<&SpanHypothesis> {
        self.hypotheses
            .iter()
            .filter(|h| {
                h.status != HypothesisStatus::Revised && h.status != HypothesisStatus::Rejected
            })
            .collect()
    }

    /// Return all hypotheses, including superseded / revised ones.
    pub fn all_hypotheses(&self) -> &[SpanHypothesis] {
        &self.hypotheses
    }

    /// Return all edges that originate from a given hypothesis id.
    pub fn edges_from(&self, id: &SpanHypothesisId) -> Vec<&HypothesisEdge> {
        self.edges.iter().filter(|e| &e.from == id).collect()
    }

    /// Return all edges that point to a given hypothesis id.
    pub fn edges_to(&self, id: &SpanHypothesisId) -> Vec<&HypothesisEdge> {
        self.edges.iter().filter(|e| &e.to == id).collect()
    }
}
