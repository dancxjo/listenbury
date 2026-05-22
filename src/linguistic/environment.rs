use crate::linguistic::inventory::PhonemeClass;
pub use crate::linguistic::inventory::{Environment, WordPosition};
use serde::{Deserialize, Serialize};

use crate::linguistic::phoneme::Phoneme;
use crate::linguistic::realization::RealizationConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PositionPattern {
    Any,
    Singleton,
    Initial,
    Medial,
    Final,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StressPattern {
    Any,
    Stressed,
    Unstressed,
    Primary,
    Secondary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanState {
    Candidate,
    Stable,
    Committed,
    Invalidated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchCommitment {
    Provisional,
    Stable,
    Committed,
    Invalidated,
}

impl MatchCommitment {
    fn from_span_state(state: SpanState) -> Self {
        match state {
            SpanState::Candidate => Self::Provisional,
            SpanState::Stable => Self::Stable,
            SpanState::Committed => Self::Committed,
            SpanState::Invalidated => Self::Invalidated,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MorphemeKind {
    Prefix,
    Stem,
    Suffix,
    Clitic,
    CompoundMember,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartOfSpeech {
    Noun,
    Verb,
    Auxiliary,
    Determiner,
    Preposition,
    Pronoun,
    Adverb,
    Adjective,
    Conjunction,
    Particle,
    ProperName,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProsodicRole {
    Content,
    FunctionWeak,
    FunctionStrong,
    Contrastive,
    Focus,
    DirectAddress,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhraseBoundaryKind {
    None,
    Minor,
    Major,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TargetPattern {
    Symbol(String),
    Symbols(Vec<String>),
    PhonemeClass(PhonemeClass),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TimingPredicate {
    AtOrBeforeNow,
    AtNow,
    SpanState(SpanState),
    ConfidenceAtLeast(f32),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ContextPredicate {
    Symbol(String),
    PhoneIpa(String),
    PhonemeClass(PhonemeClass),
    Stress(StressPattern),
    MorphemeKind(MorphemeKind),
    WordText(String),
    Pos(PartOfSpeech),
    ProsodicRole(ProsodicRole),
    BoundaryKind(PhraseBoundaryKind),
    SpanState(SpanState),
    ConfidenceAtLeast(f32),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentPattern {
    pub target: TargetPattern,
    pub left: Vec<ContextPredicate>,
    pub right: Vec<ContextPredicate>,
    pub contains: Vec<ContextPredicate>,
    pub overlaps: Vec<ContextPredicate>,
    pub word_position: Option<PositionPattern>,
    pub syllable_position: Option<PositionPattern>,
    pub phrase_position: Option<PositionPattern>,
    pub stress: Option<StressPattern>,
    pub language: Option<String>,
    pub variety: Option<String>,
    pub timing: Vec<TimingPredicate>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextSide {
    Target,
    Left,
    Right,
    Contains,
    Overlaps,
    Timing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PredicateDiagnostic {
    pub side: ContextSide,
    pub predicate: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentMatch {
    pub rule: String,
    pub target: String,
    pub matched_environment: Environment,
    pub matched_predicates: Vec<PredicateDiagnostic>,
    pub commitment: MatchCommitment,
    pub result: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanKind {
    Phoneme,
    Syllable,
    Word,
    Clause,
    Prosody,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpanRef {
    pub id: String,
    pub kind: SpanKind,
    pub start_index: usize,
    pub end_index: usize,
    pub state: SpanState,
    pub confidence: f32,
    pub text: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimelineCursor {
    pub now_index: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct RuleMatchContext<'a> {
    pub sequence: &'a [Phoneme],
    pub index: usize,
    pub now: TimelineCursor,
    pub target_span: SpanRef,
    pub previous_span: Option<SpanRef>,
    pub next_span: Option<SpanRef>,
    pub parent_word_span: Option<SpanRef>,
    pub containing_syllable_span: Option<SpanRef>,
    pub containing_clause_span: Option<SpanRef>,
    pub neighboring_word_spans: Vec<SpanRef>,
    pub phrase_boundary: Option<PhraseBoundaryKind>,
    pub prosodic_role: Option<ProsodicRole>,
    pub morphology: Option<MorphemeKind>,
    pub part_of_speech: Option<PartOfSpeech>,
    pub span_state: SpanState,
    pub confidence: f32,
    pub language: String,
    pub variety: String,
}

impl<'a> RuleMatchContext<'a> {
    pub fn from_sequence(
        sequence: &'a [Phoneme],
        index: usize,
        config: &RealizationConfig,
    ) -> Self {
        let confidence = config.confidence.clamp(0.0, 1.0);
        let target_span = SpanRef {
            id: format!("phoneme:{index}"),
            kind: SpanKind::Phoneme,
            start_index: index,
            end_index: index + 1,
            state: config.span_state,
            confidence,
            text: Some(sequence[index].symbol.clone()),
        };
        let previous_span = index.checked_sub(1).map(|left| SpanRef {
            id: format!("phoneme:{left}"),
            kind: SpanKind::Phoneme,
            start_index: left,
            end_index: left + 1,
            state: config.span_state,
            confidence,
            text: Some(sequence[left].symbol.clone()),
        });
        let next_span = (index + 1 < sequence.len()).then(|| SpanRef {
            id: format!("phoneme:{}", index + 1),
            kind: SpanKind::Phoneme,
            start_index: index + 1,
            end_index: index + 2,
            state: config.span_state,
            confidence,
            text: Some(sequence[index + 1].symbol.clone()),
        });
        let parent_word_span = (!sequence.is_empty()).then(|| SpanRef {
            id: "word:0".to_string(),
            kind: SpanKind::Word,
            start_index: 0,
            end_index: sequence.len(),
            state: config.span_state,
            confidence,
            text: Some(
                sequence
                    .iter()
                    .map(|phoneme| phoneme.symbol.as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
            ),
        });

        Self {
            sequence,
            index,
            now: TimelineCursor {
                now_index: config.now_index,
            },
            target_span,
            previous_span,
            next_span,
            parent_word_span,
            containing_syllable_span: None,
            containing_clause_span: None,
            neighboring_word_spans: Vec::new(),
            phrase_boundary: config.phrase_boundary,
            prosodic_role: config.prosodic_role,
            morphology: config.morpheme_kind,
            part_of_speech: config.part_of_speech,
            span_state: config.span_state,
            confidence,
            language: config.language.clone(),
            variety: config.dialect.clone(),
        }
    }

    pub fn target(&self) -> &Phoneme {
        &self.sequence[self.index]
    }

    pub fn left(&self) -> Option<&Phoneme> {
        self.index.checked_sub(1).map(|index| &self.sequence[index])
    }

    pub fn right(&self) -> Option<&Phoneme> {
        (self.index + 1 < self.sequence.len()).then(|| &self.sequence[self.index + 1])
    }

    pub fn commitment(&self) -> MatchCommitment {
        MatchCommitment::from_span_state(self.span_state)
    }
}
