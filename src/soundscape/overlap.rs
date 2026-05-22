//! Heuristic detector and representation for overlapping voice-like sources.
//!
//! When multiple speakers talk at the same time the type system should not
//! pretend the world politely takes turns.  This module provides:
//!
//! * [`MixtureComponent`] — one voice-like participant in an overlapping window.
//! * [`OverlapMixture`] — a time range containing one or more concurrent
//!   voice-like sources, derived from [`SourceHypothesis`] inputs.
//! * [`detect_overlaps`] — a heuristic first-pass detector that groups
//!   voice-like hypotheses that share any time overlap into [`OverlapMixture`]
//!   windows.

use serde::{Deserialize, Serialize};

use crate::soundscape::{AttributionEvidence, SourceHypothesis, SourceKind, TimePoint, TimeRange};

/// One voice-like participant in an overlapping acoustic mixture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MixtureComponent {
    /// The attribution hypothesis for this participant.
    pub source_hypothesis: SourceHypothesis,
    /// Relative linear energy contribution estimated for this component.
    ///
    /// Normalised to `[0.0, 1.0]` across all components in the same mixture,
    /// or `None` when no energy estimate is available.
    pub relative_energy: Option<f32>,
}

/// A time range in which one or more concurrent voice-like sources are
/// hypothesised to be active.
///
/// A single-component mixture (`voice_count() == 1`) represents an
/// unambiguous, non-overlapping segment.  A multi-component mixture
/// (`is_overlapping() == true`) indicates detected overlap.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverlapMixture {
    /// Span covering all components' ranges.
    pub range: TimeRange,
    /// The concurrent voice-like participants in this window.
    pub components: Vec<MixtureComponent>,
    /// Overall detection confidence in `[0.0, 1.0]`.
    ///
    /// For single-component mixtures this is the hypothesis confidence.
    /// For multi-component mixtures it is the mean confidence boosted by any
    /// [`AttributionEvidence::OverlapDetected`] hints — each hint adds `0.1`,
    /// and multiple hints accumulate (capped at `1.0`).
    pub confidence: f32,
}

impl OverlapMixture {
    /// Estimated number of simultaneous voice-like sources in this region.
    pub fn voice_count(&self) -> usize {
        self.components.len()
    }

    /// Returns `true` when more than one voice-like source is concurrently
    /// active in this window.
    pub fn is_overlapping(&self) -> bool {
        self.components.len() > 1
    }
}

/// Heuristic overlap detector over a flat slice of [`SourceHypothesis`] values.
///
/// Voice-like hypotheses that share any time overlap are merged into a single
/// [`OverlapMixture`] via union-find grouping.  Non-voice hypotheses
/// (e.g. [`SourceKind::Playback`] or [`SourceKind::EnvironmentalNoise`]) are
/// ignored.
///
/// Detection signals consulted:
/// - Multiple voice-like hypotheses covering the same time range.
/// - [`AttributionEvidence::OverlapDetected`] present on any component
///   (boosts confidence by `0.1` per hint, capped at `1.0`).
/// - Distinct `source_id` values active over an overlapping range.
///
/// The returned vector is unordered; each entry covers a disjoint time span
/// of potentially-concurrent activity.
pub fn detect_overlaps(hypotheses: &[SourceHypothesis]) -> Vec<OverlapMixture> {
    let voice_like: Vec<&SourceHypothesis> =
        hypotheses.iter().filter(|h| is_voice_like(h)).collect();

    if voice_like.is_empty() {
        return Vec::new();
    }

    let n = voice_like.len();
    let mut parent: Vec<usize> = (0..n).collect();

    for i in 0..n {
        for j in (i + 1)..n {
            if ranges_overlap(voice_like[i].range, voice_like[j].range) {
                union(&mut parent, i, j);
            }
        }
    }

    // Group indices by their root.
    let mut clusters: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        clusters.entry(root).or_default().push(i);
    }

    let mut result: Vec<OverlapMixture> = clusters
        .values()
        .map(|indices| {
            let components: Vec<MixtureComponent> = indices
                .iter()
                .map(|&i| MixtureComponent {
                    source_hypothesis: voice_like[i].clone(),
                    relative_energy: None,
                })
                .collect();

            let start = components
                .iter()
                .map(|c| c.source_hypothesis.range.start.millis)
                .min()
                .unwrap_or(0);
            let end = components
                .iter()
                .map(|c| c.source_hypothesis.range.end.millis)
                .max()
                .unwrap_or(0);
            let range = TimeRange::new(TimePoint::from_millis(start), TimePoint::from_millis(end));

            let mean_confidence = components
                .iter()
                .map(|c| c.source_hypothesis.confidence)
                .sum::<f32>()
                / components.len() as f32;

            let overlap_bonus: f32 = components
                .iter()
                .flat_map(|c| &c.source_hypothesis.evidence)
                .filter(|e| matches!(e, AttributionEvidence::OverlapDetected))
                .count() as f32
                * 0.1;

            let confidence = (mean_confidence + overlap_bonus).clamp(0.0, 1.0);

            OverlapMixture {
                range,
                components,
                confidence,
            }
        })
        .collect();

    // Deterministic order: earliest start first.
    result.sort_by_key(|m| m.range.start.millis);
    result
}

fn is_voice_like(hypothesis: &SourceHypothesis) -> bool {
    matches!(
        hypothesis.kind,
        SourceKind::Voice
            | SourceKind::SyntheticVoice
            | SourceKind::KnownSelfVoice
            | SourceKind::Unknown
    )
}

fn ranges_overlap(left: TimeRange, right: TimeRange) -> bool {
    left.start.millis < right.end.millis && right.start.millis < left.end.millis
}

fn find(parent: &mut Vec<usize>, i: usize) -> usize {
    let mut root = i;
    while parent[root] != root {
        root = parent[root];
    }
    // Path compression.
    let mut curr = i;
    while curr != root {
        let next = parent[curr];
        parent[curr] = root;
        curr = next;
    }
    root
}

fn union(parent: &mut Vec<usize>, a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent[rb] = ra;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soundscape::{AttributionEvidence, SourceId, SourceKind, TimePoint, TimeRange};

    fn range(start: u64, end: u64) -> TimeRange {
        TimeRange::new(TimePoint::from_millis(start), TimePoint::from_millis(end))
    }

    fn voice(start: u64, end: u64) -> SourceHypothesis {
        SourceHypothesis {
            source_id: None,
            kind: SourceKind::Voice,
            range: range(start, end),
            confidence: 0.85,
            evidence: vec![],
        }
    }

    fn voice_with_id(id: SourceId, start: u64, end: u64) -> SourceHypothesis {
        SourceHypothesis {
            source_id: Some(id),
            kind: SourceKind::Voice,
            range: range(start, end),
            confidence: 0.9,
            evidence: vec![],
        }
    }

    // ------------------------------------------------------------------
    // Non-overlap cases
    // ------------------------------------------------------------------

    #[test]
    fn empty_input_produces_no_mixtures() {
        let mixtures = detect_overlaps(&[]);
        assert!(mixtures.is_empty());
    }

    #[test]
    fn single_voice_produces_non_overlap_mixture() {
        let mixtures = detect_overlaps(&[voice(0, 500)]);

        assert_eq!(mixtures.len(), 1);
        assert!(!mixtures[0].is_overlapping());
        assert_eq!(mixtures[0].voice_count(), 1);
        assert_eq!(mixtures[0].range, range(0, 500));
    }

    #[test]
    fn non_overlapping_voices_produce_separate_mixtures() {
        // Speaker A: 0–400 ms, Speaker B: 500–900 ms — no overlap.
        let mixtures = detect_overlaps(&[voice(0, 400), voice(500, 900)]);

        assert_eq!(mixtures.len(), 2);
        assert!(mixtures.iter().all(|m| !m.is_overlapping()));
        // Sorted earliest-first.
        assert_eq!(mixtures[0].range, range(0, 400));
        assert_eq!(mixtures[1].range, range(500, 900));
    }

    #[test]
    fn non_voice_hypotheses_are_filtered_out() {
        let playback = SourceHypothesis {
            kind: SourceKind::Playback,
            source_id: None,
            range: range(0, 600),
            confidence: 0.9,
            evidence: vec![],
        };
        // The playback overlaps the voice in time, but must not be grouped.
        let mixtures = detect_overlaps(&[playback, voice(200, 700)]);

        assert_eq!(mixtures.len(), 1);
        assert_eq!(mixtures[0].voice_count(), 1);
        assert!(matches!(
            mixtures[0].components[0].source_hypothesis.kind,
            SourceKind::Voice
        ));
    }

    // ------------------------------------------------------------------
    // Partial overlap
    // ------------------------------------------------------------------

    #[test]
    fn partial_overlap_merges_into_one_mixture() {
        // Speaker A: 0–600, Speaker B: 400–900 — overlap 400–600 (200 ms).
        let mixtures = detect_overlaps(&[voice(0, 600), voice(400, 900)]);

        assert_eq!(mixtures.len(), 1);
        assert!(mixtures[0].is_overlapping());
        assert_eq!(mixtures[0].voice_count(), 2);
        assert_eq!(mixtures[0].range, range(0, 900));
    }

    #[test]
    fn three_way_chain_overlap_forms_single_mixture() {
        // A: 0–400, B: 300–700, C: 600–1000.
        // A∩B and B∩C; A and C do not overlap directly.
        let mixtures = detect_overlaps(&[voice(0, 400), voice(300, 700), voice(600, 1000)]);

        assert_eq!(mixtures.len(), 1);
        assert_eq!(mixtures[0].voice_count(), 3);
        assert_eq!(mixtures[0].range, range(0, 1000));
    }

    // ------------------------------------------------------------------
    // Full overlap
    // ------------------------------------------------------------------

    #[test]
    fn full_overlap_identical_ranges_detected() {
        // Both speakers active over exactly the same window.
        let mixtures = detect_overlaps(&[voice(0, 800), voice(0, 800)]);

        assert_eq!(mixtures.len(), 1);
        assert!(mixtures[0].is_overlapping());
        assert_eq!(mixtures[0].voice_count(), 2);
        assert_eq!(mixtures[0].range, range(0, 800));
    }

    #[test]
    fn one_voice_contained_within_another_is_full_overlap() {
        // Speaker A: 0–1000, Speaker B: 200–600 (completely inside A).
        let mixtures = detect_overlaps(&[voice(0, 1000), voice(200, 600)]);

        assert_eq!(mixtures.len(), 1);
        assert!(mixtures[0].is_overlapping());
        assert_eq!(mixtures[0].voice_count(), 2);
    }

    // ------------------------------------------------------------------
    // Confidence / evidence
    // ------------------------------------------------------------------

    #[test]
    fn overlap_evidence_boosts_confidence_above_mean() {
        let h1 = voice(0, 500);
        let h2 = SourceHypothesis {
            evidence: vec![AttributionEvidence::OverlapDetected],
            ..voice(200, 700)
        };
        let mixtures = detect_overlaps(&[h1, h2]);

        assert_eq!(mixtures.len(), 1);
        // Mean confidence is 0.85; overlap bonus of 0.1 should push it higher.
        assert!(mixtures[0].confidence > 0.85);
    }

    #[test]
    fn known_source_ids_are_preserved_in_components() {
        let id_a = SourceId::new();
        let id_b = SourceId::new();
        let mixtures =
            detect_overlaps(&[voice_with_id(id_a, 0, 500), voice_with_id(id_b, 300, 800)]);

        assert_eq!(mixtures.len(), 1);
        let ids: Vec<Option<SourceId>> = mixtures[0]
            .components
            .iter()
            .map(|c| c.source_hypothesis.source_id)
            .collect();
        assert!(ids.contains(&Some(id_a)));
        assert!(ids.contains(&Some(id_b)));
    }
}
