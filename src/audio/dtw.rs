//! Simple DTW (Dynamic Time Warping) template-matcher scaffold.
//!
//! Provides:
//! - [`DtwTemplate`]      — a named sequence of feature vectors.
//! - [`dtw_align`]        — the core DTW distance + alignment-path function.
//! - [`DtwTemplateMatcher`] — registry that matches a query against templates.
//!
//! The matcher returns [`SpanHypothesis`] objects with kind
//! [`SpanHypothesisKind::TemplateMatch`].

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::audio::hypothesis::{
    HypothesisSource, HypothesisStatus, SpanHypothesis, SpanHypothesisId, SpanHypothesisKind,
};

// ---------------------------------------------------------------------------
// Template
// ---------------------------------------------------------------------------

/// A named template: a sequence of feature vectors to match against.
///
/// Each inner `Vec<f32>` is one analysis frame's feature vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DtwTemplate {
    /// Human-readable name (e.g. a word or phone label).
    pub name: String,
    /// Sequence of feature vectors, one per analysis frame.
    pub frames: Vec<Vec<f32>>,
}

impl DtwTemplate {
    /// Create a template from a label and a feature matrix.
    pub fn new(name: impl Into<String>, frames: Vec<Vec<f32>>) -> Self {
        Self {
            name: name.into(),
            frames,
        }
    }
}

// ---------------------------------------------------------------------------
// Match result
// ---------------------------------------------------------------------------

/// Result of matching a query sequence against one template.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DtwMatch {
    /// Name of the matched template.
    pub template_name: String,
    /// Normalised DTW distance (lower = better match).
    pub score: f32,
    /// Monotonic warp path as `[query_index, template_index]` pairs.
    pub alignment_path: Vec<[usize; 2]>,
}

// ---------------------------------------------------------------------------
// Core DTW
// ---------------------------------------------------------------------------

/// Compute the DTW distance and warp path between two feature sequences.
///
/// Returns `(normalised_distance, path)`.  The distance is the total
/// cumulative cost divided by the path length, so it is comparable across
/// templates of different lengths.  Lower values indicate a better match.
pub fn dtw_align(query: &[Vec<f32>], template: &[Vec<f32>]) -> (f32, Vec<[usize; 2]>) {
    let n = query.len();
    let m = template.len();
    if n == 0 || m == 0 {
        return (f32::INFINITY, Vec::new());
    }

    // Accumulated cost matrix.
    let mut cost = vec![vec![f32::INFINITY; m]; n];
    cost[0][0] = euclidean_distance(&query[0], &template[0]);
    for j in 1..m {
        cost[0][j] = cost[0][j - 1] + euclidean_distance(&query[0], &template[j]);
    }
    for i in 1..n {
        cost[i][0] = cost[i - 1][0] + euclidean_distance(&query[i], &template[0]);
    }
    for i in 1..n {
        for j in 1..m {
            let local = euclidean_distance(&query[i], &template[j]);
            let min_prev = cost[i - 1][j]
                .min(cost[i][j - 1])
                .min(cost[i - 1][j - 1]);
            cost[i][j] = local + min_prev;
        }
    }

    let total = cost[n - 1][m - 1];
    let path = backtrace_path(&cost, n, m);
    let path_len = path.len().max(1) as f32;
    (total / path_len, path)
}

fn backtrace_path(cost: &[Vec<f32>], n: usize, m: usize) -> Vec<[usize; 2]> {
    let mut path = Vec::new();
    let mut i = n - 1;
    let mut j = m - 1;
    path.push([i, j]);
    while i > 0 || j > 0 {
        if i == 0 {
            j -= 1;
        } else if j == 0 {
            i -= 1;
        } else {
            let diag = cost[i - 1][j - 1];
            let up = cost[i - 1][j];
            let left = cost[i][j - 1];
            if diag <= up && diag <= left {
                i -= 1;
                j -= 1;
            } else if up <= left {
                i -= 1;
            } else {
                j -= 1;
            }
        }
        path.push([i, j]);
    }
    path.reverse();
    path
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    a[..n]
        .iter()
        .zip(b[..n].iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

// ---------------------------------------------------------------------------
// Matcher registry
// ---------------------------------------------------------------------------

/// Template matcher: stores named templates and returns best-N matches.
#[derive(Debug, Clone, Default)]
pub struct DtwTemplateMatcher {
    templates: Vec<DtwTemplate>,
}

impl DtwTemplateMatcher {
    /// Create an empty matcher.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a template.
    pub fn register(&mut self, template: DtwTemplate) {
        self.templates.push(template);
    }

    /// Match `query` against all registered templates and return up to
    /// `top_n` matches sorted by ascending score (better = lower).
    pub fn match_query(&self, query: &[Vec<f32>], top_n: usize) -> Vec<DtwMatch> {
        let mut results: Vec<DtwMatch> = self
            .templates
            .iter()
            .map(|template| {
                let (score, path) = dtw_align(query, &template.frames);
                DtwMatch {
                    template_name: template.name.clone(),
                    score,
                    alignment_path: path,
                }
            })
            .collect();
        results.sort_by(|a, b| a.score.total_cmp(&b.score));
        results.truncate(top_n);
        results
    }

    /// Match and emit [`SpanHypothesis`] values for the best-N matches.
    pub fn match_hypotheses(
        &self,
        query: &[Vec<f32>],
        start_ms: u64,
        end_ms: u64,
        top_n: usize,
    ) -> Vec<SpanHypothesis> {
        self.match_query(query, top_n)
            .into_iter()
            .map(|m| {
                // Convert DTW distance → confidence: lower distance ⟹ higher confidence.
                let confidence = (1.0 / (1.0 + m.score)).clamp(0.0, 1.0);
                SpanHypothesis {
                    id: SpanHypothesisId::new(),
                    kind: SpanHypothesisKind::TemplateMatch,
                    label: m.template_name.clone(),
                    start_ms,
                    end_ms,
                    score: m.score,
                    confidence,
                    source: HypothesisSource::DtwTemplateMatcher,
                    features_used: vec!["dtw.euclidean".to_string()],
                    status: HypothesisStatus::Provisional,
                    provenance: json!({
                        "template": m.template_name,
                        "dtw_score": m.score,
                        "path_length": m.alignment_path.len(),
                    }),
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_frames(data: &[&[f32]]) -> Vec<Vec<f32>> {
        data.iter().map(|row| row.to_vec()).collect()
    }

    #[test]
    fn dtw_identical_sequences_have_zero_distance() {
        let frames = vec_frames(&[&[1.0, 0.0], &[2.0, 1.0], &[3.0, 2.0]]);
        let (dist, path) = dtw_align(&frames, &frames);
        assert!(dist < 0.001, "expected near-zero distance, got {dist}");
        assert!(!path.is_empty());
    }

    #[test]
    fn dtw_path_starts_at_origin_and_ends_at_corner() {
        let q = vec_frames(&[&[1.0], &[2.0], &[3.0]]);
        let t = vec_frames(&[&[1.0], &[2.0]]);
        let (_, path) = dtw_align(&q, &t);
        assert_eq!(path[0], [0, 0]);
        assert_eq!(*path.last().unwrap(), [q.len() - 1, t.len() - 1]);
    }

    #[test]
    fn dtw_path_is_monotonic() {
        let q = vec_frames(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
        let t = vec_frames(&[&[0.0], &[1.5], &[3.0]]);
        let (_, path) = dtw_align(&q, &t);
        for w in path.windows(2) {
            let [i0, j0] = w[0];
            let [i1, j1] = w[1];
            assert!(i1 >= i0 && j1 >= j0, "path must be non-decreasing");
        }
    }

    #[test]
    fn matcher_returns_best_match_first() {
        let mut matcher = DtwTemplateMatcher::new();
        // Template "a": identical to the query.
        matcher.register(DtwTemplate::new(
            "a",
            vec_frames(&[&[1.0], &[2.0], &[3.0]]),
        ));
        // Template "b": very different.
        matcher.register(DtwTemplate::new(
            "b",
            vec_frames(&[&[10.0], &[20.0], &[30.0]]),
        ));
        let query = vec_frames(&[&[1.0], &[2.0], &[3.0]]);
        let matches = matcher.match_query(&query, 2);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].template_name, "a");
        assert!(matches[0].score < matches[1].score);
    }

    #[test]
    fn matcher_top_n_truncates_results() {
        let mut matcher = DtwTemplateMatcher::new();
        for name in ["x", "y", "z"] {
            matcher.register(DtwTemplate::new(name, vec_frames(&[&[1.0], &[2.0]])));
        }
        let query = vec_frames(&[&[1.0], &[2.0]]);
        let matches = matcher.match_query(&query, 2);
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn match_hypotheses_returns_span_hypothesis_with_correct_kind() {
        let mut matcher = DtwTemplateMatcher::new();
        matcher.register(DtwTemplate::new("foo", vec_frames(&[&[0.5], &[1.0]])));
        let query = vec_frames(&[&[0.5], &[1.0]]);
        let hyps = matcher.match_hypotheses(&query, 100, 200, 1);
        assert_eq!(hyps.len(), 1);
        assert_eq!(hyps[0].kind, SpanHypothesisKind::TemplateMatch);
        assert_eq!(hyps[0].label, "foo");
        assert_eq!(hyps[0].start_ms, 100);
        assert_eq!(hyps[0].end_ms, 200);
        assert!(hyps[0].confidence > 0.5);
    }

    #[test]
    fn dtw_empty_query_returns_infinity() {
        let (dist, path) = dtw_align(&[], &[vec![1.0]]);
        assert!(dist.is_infinite());
        assert!(path.is_empty());
    }

    #[test]
    fn dtw_empty_template_returns_infinity() {
        let (dist, path) = dtw_align(&[vec![1.0]], &[]);
        assert!(dist.is_infinite());
        assert!(path.is_empty());
    }
}
