//! Generic span model shared across listening/speaking/memory modalities.

use serde::{Deserialize, Serialize};

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
}
