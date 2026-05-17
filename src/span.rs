//! Generic span model shared across listening/speaking/memory modalities.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

use crate::time::ExactTimestamp;

/// Unique identifier for a source text timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TextId(pub u64);

/// Unique identifier for a span within a text timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpanId(pub u64);

/// Cursor position in a text timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Cursor(pub u64);

/// Text timeline metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Text {
    pub id: TextId,
    pub created_at: ExactTimestamp,
}

/// Data modality represented by a span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Modality {
    Audio,
    Phoneme,
    Word,
    Clause,
    BreathGroup,
    Prosody,
    Emotion,
    Semantic,
    Topic,
    Episode,
    Memory,
}

/// Lifecycle state for a span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpanState {
    Hypothesis,
    Stable,
    Committed,
    Revised,
    Deprecated,
}

/// Prior version of a span that was superseded by a revision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpanRevision<T> {
    pub state: SpanState,
    pub start: Cursor,
    pub end: Option<Cursor>,
    pub contents: T,
    pub confidence: f32,
    pub stability: f32,
}

/// Generic span over a text timeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Span<T> {
    pub id: SpanId,
    pub text_id: TextId,
    pub modality: Modality,
    pub state: SpanState,
    pub start: Cursor,
    pub end: Option<Cursor>,
    pub contents: T,
    pub confidence: f32,
    pub stability: f32,
    pub revisions: Vec<SpanRevision<T>>,
}

/// Cursor offsets of an aligned sub-span within a larger span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlignmentOffset {
    pub start: Cursor,
    pub end: Option<Cursor>,
}

/// Semantic relationship between two aligned spans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlignmentKind {
    Equivalent,
    Contains,
    Overlaps,
    Derived,
}

/// Cross-modal edge between two spans.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Alignment {
    pub source: SpanId,
    pub target: SpanId,
    pub confidence: f32,
    pub relation: AlignmentKind,
    pub source_offset: Option<AlignmentOffset>,
    pub target_offset: Option<AlignmentOffset>,
}

impl Alignment {
    pub fn new(source: SpanId, target: SpanId, confidence: f32, relation: AlignmentKind) -> Self {
        Self {
            source,
            target,
            confidence,
            relation,
            source_offset: None,
            target_offset: None,
        }
    }

    pub fn with_offsets(
        mut self,
        source_offset: Option<AlignmentOffset>,
        target_offset: Option<AlignmentOffset>,
    ) -> Self {
        self.source_offset = source_offset;
        self.target_offset = target_offset;
        self
    }
}

/// Many-to-many span alignment graph with directional edges and bidirectional traversal.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AlignmentGraph {
    pub alignments: Vec<Alignment>,
    outgoing: HashMap<SpanId, Vec<usize>>,
    incoming: HashMap<SpanId, Vec<usize>>,
}

impl AlignmentGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_alignment(&mut self, alignment: Alignment) {
        let index = self.alignments.len();
        self.outgoing
            .entry(alignment.source)
            .or_default()
            .push(index);
        self.incoming
            .entry(alignment.target)
            .or_default()
            .push(index);
        self.alignments.push(alignment);
    }

    pub fn alignments_from(&self, span_id: SpanId) -> Vec<&Alignment> {
        self.outgoing
            .get(&span_id)
            .map(|indices| indices.iter().map(|&i| &self.alignments[i]).collect())
            .unwrap_or_default()
    }

    pub fn alignments_to(&self, span_id: SpanId) -> Vec<&Alignment> {
        self.incoming
            .get(&span_id)
            .map(|indices| indices.iter().map(|&i| &self.alignments[i]).collect())
            .unwrap_or_default()
    }

    pub fn aligned_targets(&self, span_id: SpanId) -> Vec<SpanId> {
        self.alignments_from(span_id)
            .into_iter()
            .map(|alignment| alignment.target)
            .collect()
    }

    pub fn aligned_sources(&self, span_id: SpanId) -> Vec<SpanId> {
        self.alignments_to(span_id)
            .into_iter()
            .map(|alignment| alignment.source)
            .collect()
    }

    pub fn neighbors_bidirectional(&self, span_id: SpanId) -> Vec<SpanId> {
        let mut seen = HashSet::new();
        self.aligned_targets(span_id)
            .into_iter()
            .chain(self.aligned_sources(span_id))
            .filter(|neighbor| seen.insert(*neighbor))
            .collect()
    }

    pub fn reconstruct_path(&self, start: SpanId, end: SpanId) -> Option<Vec<SpanId>> {
        if start == end {
            return Some(vec![start]);
        }

        let mut queue = VecDeque::from([start]);
        let mut parent: HashMap<SpanId, SpanId> = HashMap::new();
        let mut seen = HashSet::from([start]);

        while let Some(current) = queue.pop_front() {
            for neighbor in self.neighbors_bidirectional(current) {
                if !seen.insert(neighbor) {
                    continue;
                }
                parent.insert(neighbor, current);
                if neighbor == end {
                    let mut path = vec![end];
                    let mut step = end;
                    while let Some(prev) = parent.get(&step) {
                        path.push(*prev);
                        if *prev == start {
                            break;
                        }
                        step = *prev;
                    }
                    path.reverse();
                    return Some(path);
                }
                queue.push_back(neighbor);
            }
        }

        None
    }
}

impl<T: Clone> Span<T> {
    pub fn new_provisional(
        id: SpanId,
        text_id: TextId,
        modality: Modality,
        start: Cursor,
        end: Option<Cursor>,
        contents: T,
        confidence: f32,
        stability: f32,
    ) -> Self {
        Self::new_hypothesis(
            id, text_id, modality, start, end, contents, confidence, stability,
        )
    }

    pub fn new_hypothesis(
        id: SpanId,
        text_id: TextId,
        modality: Modality,
        start: Cursor,
        end: Option<Cursor>,
        contents: T,
        confidence: f32,
        stability: f32,
    ) -> Self {
        Self {
            id,
            text_id,
            modality,
            state: SpanState::Hypothesis,
            start,
            end,
            contents,
            confidence,
            stability,
            revisions: Vec::new(),
        }
    }

    pub fn stabilize(&mut self) -> bool {
        self.transition_to(SpanState::Stable)
    }

    pub fn commit(&mut self) -> bool {
        self.transition_to(SpanState::Committed)
    }

    pub fn deprecate(&mut self) -> bool {
        self.transition_to(SpanState::Deprecated)
    }

    pub fn transition_to(&mut self, next_state: SpanState) -> bool {
        let allowed = match (self.state, next_state) {
            (current, next) if current == next => true,
            (SpanState::Hypothesis, SpanState::Stable)
            | (SpanState::Hypothesis, SpanState::Committed)
            | (SpanState::Hypothesis, SpanState::Deprecated)
            | (SpanState::Stable, SpanState::Committed)
            | (SpanState::Stable, SpanState::Revised)
            | (SpanState::Stable, SpanState::Deprecated)
            | (SpanState::Committed, SpanState::Revised)
            | (SpanState::Committed, SpanState::Deprecated)
            | (SpanState::Revised, SpanState::Stable)
            | (SpanState::Revised, SpanState::Committed)
            | (SpanState::Revised, SpanState::Deprecated) => true,
            _ => false,
        };

        if allowed {
            self.state = next_state;
        }
        allowed
    }

    pub fn revise(
        &mut self,
        start: Cursor,
        end: Option<Cursor>,
        contents: T,
        confidence: f32,
        stability: f32,
    ) -> bool {
        if !matches!(self.state, SpanState::Stable | SpanState::Committed) {
            return false;
        }

        self.revisions.push(SpanRevision {
            state: self.state,
            start: self.start,
            end: self.end,
            contents: self.contents.clone(),
            confidence: self.confidence,
            stability: self.stability,
        });

        self.start = start;
        self.end = end;
        self.contents = contents;
        self.confidence = confidence;
        self.stability = stability;
        self.state = SpanState::Revised;
        true
    }

    pub fn contains_span<U>(&self, inner: &Span<U>) -> bool {
        if inner.start < self.start {
            return false;
        }

        match (self.end, inner.end) {
            (Some(outer_end), Some(inner_end)) => inner_end <= outer_end,
            (Some(_), None) => false,
            (None, _) => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supports_open_ended_spans() {
        let span = Span::new_hypothesis(
            SpanId(1),
            TextId(7),
            Modality::Word,
            Cursor(42),
            None,
            "hello".to_string(),
            0.8,
            0.2,
        );

        assert_eq!(span.end, None);
        assert_eq!(span.state, SpanState::Hypothesis);
    }

    #[test]
    fn supports_provisional_to_committed_lifecycle() {
        let mut span = Span::new_hypothesis(
            SpanId(1),
            TextId(1),
            Modality::Phoneme,
            Cursor(0),
            Some(Cursor(5)),
            "hə".to_string(),
            0.75,
            0.3,
        );

        assert!(span.stabilize());
        assert_eq!(span.state, SpanState::Stable);
        assert!(span.commit());
        assert_eq!(span.state, SpanState::Committed);
        assert!(!span.transition_to(SpanState::Hypothesis));
    }

    #[test]
    fn revisions_preserve_history() {
        let mut span = Span::new_hypothesis(
            SpanId(3),
            TextId(11),
            Modality::Semantic,
            Cursor(1),
            Some(Cursor(4)),
            "old".to_string(),
            0.6,
            0.4,
        );
        assert!(span.stabilize());

        assert!(span.revise(Cursor(1), Some(Cursor(5)), "new".to_string(), 0.95, 0.9));

        assert_eq!(span.state, SpanState::Revised);
        assert_eq!(span.contents, "new");
        assert_eq!(span.revisions.len(), 1);
        assert_eq!(span.revisions[0].state, SpanState::Stable);
        assert_eq!(span.revisions[0].contents, "old");
    }

    #[test]
    fn modalities_are_serializable() {
        let encoded = serde_json::to_string(&Modality::BreathGroup).expect("modality encodes");
        let decoded: Modality = serde_json::from_str(&encoded).expect("modality decodes");
        assert_eq!(decoded, Modality::BreathGroup);
    }

    #[test]
    fn supports_nesting_checks() {
        let outer = Span::new_hypothesis(
            SpanId(1),
            TextId(1),
            Modality::Clause,
            Cursor(0),
            Some(Cursor(20)),
            "outer".to_string(),
            1.0,
            1.0,
        );
        let inner = Span::new_hypothesis(
            SpanId(2),
            TextId(1),
            Modality::Word,
            Cursor(5),
            Some(Cursor(10)),
            "inner".to_string(),
            1.0,
            1.0,
        );
        let not_inner = Span::new_hypothesis(
            SpanId(3),
            TextId(1),
            Modality::Word,
            Cursor(5),
            Some(Cursor(25)),
            "wide".to_string(),
            1.0,
            1.0,
        );

        assert!(outer.contains_span(&inner));
        assert!(!outer.contains_span(&not_inner));
    }

    #[test]
    fn open_ended_outer_span_contains_bounded_and_open_ended_inner() {
        let outer = Span::new_provisional(
            SpanId(1),
            TextId(1),
            Modality::Episode,
            Cursor(10),
            None,
            "outer".to_string(),
            1.0,
            1.0,
        );
        let bounded_inner = Span::new_provisional(
            SpanId(2),
            TextId(1),
            Modality::Topic,
            Cursor(12),
            Some(Cursor(24)),
            "bounded".to_string(),
            1.0,
            1.0,
        );
        let open_inner = Span::new_provisional(
            SpanId(3),
            TextId(1),
            Modality::Topic,
            Cursor(16),
            None,
            "open".to_string(),
            1.0,
            1.0,
        );

        assert!(outer.contains_span(&bounded_inner));
        assert!(outer.contains_span(&open_inner));
    }

    #[test]
    fn alignment_graph_supports_many_to_many_and_ambiguous_edges() {
        let p1 = SpanId(100);
        let p2 = SpanId(101);
        let w1 = SpanId(200);
        let w2 = SpanId(201);

        let mut graph = AlignmentGraph::new();
        graph.add_alignment(Alignment::new(p1, w1, 0.95, AlignmentKind::Equivalent));
        graph.add_alignment(Alignment::new(p1, w2, 0.51, AlignmentKind::Overlaps));
        graph.add_alignment(Alignment::new(p2, w1, 0.88, AlignmentKind::Equivalent));

        assert_eq!(graph.alignments_from(p1).len(), 2);
        assert_eq!(graph.alignments_to(w1).len(), 2);
        assert_eq!(graph.aligned_targets(p1), vec![w1, w2]);
    }

    #[test]
    fn alignment_offsets_are_preserved() {
        let source_offset = AlignmentOffset {
            start: Cursor(120),
            end: Some(Cursor(240)),
        };
        let target_offset = AlignmentOffset {
            start: Cursor(8),
            end: Some(Cursor(14)),
        };

        let mut graph = AlignmentGraph::new();
        graph.add_alignment(
            Alignment::new(SpanId(1), SpanId(2), 0.9, AlignmentKind::Contains)
                .with_offsets(Some(source_offset), Some(target_offset)),
        );

        let alignment = graph.alignments_from(SpanId(1))[0];
        assert_eq!(alignment.source_offset, Some(source_offset));
        assert_eq!(alignment.target_offset, Some(target_offset));
    }

    #[test]
    fn reconstructs_phoneme_to_word_to_audio_path_bidirectionally() {
        let phoneme = SpanId(10);
        let word = SpanId(20);
        let audio = SpanId(30);

        let mut graph = AlignmentGraph::new();
        graph.add_alignment(Alignment::new(
            phoneme,
            word,
            0.93,
            AlignmentKind::Equivalent,
        ));
        graph.add_alignment(Alignment::new(word, audio, 0.99, AlignmentKind::Derived));

        let phoneme_to_audio = graph.reconstruct_path(phoneme, audio).expect("path exists");
        assert_eq!(phoneme_to_audio, vec![phoneme, word, audio]);

        let audio_to_phoneme = graph
            .reconstruct_path(audio, phoneme)
            .expect("reverse path exists");
        assert_eq!(audio_to_phoneme, vec![audio, word, phoneme]);
    }
}
