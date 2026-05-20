use serde::{Deserialize, Serialize};

use crate::mouth::riper::text::{NormalizedText, NormalizedToken};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SentenceAnalysis {
    pub tokens: Vec<TokenAnalysis>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenAnalysis {
    pub token_index: usize,
    pub word_index: Option<usize>,
    pub text: String,
    pub pos: PartOfSpeech,
    pub syntactic_role: Option<SyntacticRole>,
    pub prosodic_role: ProsodicRole,
    pub reduction: ReductionClass,
    pub reduction_diagnostic: Option<ReductionDiagnostic>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
pub enum SyntacticRole {
    InfinitivalMarker,
    PrepositionalObjectLink,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProsodicRole {
    Content,
    FunctionWeak,
    FunctionStrong,
    Contrastive,
    Focus,
    DirectAddress,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReductionClass {
    None,
    WeakFunctionWord,
    CliticLike,
    Contracted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReductionStatus {
    Applied,
    Blocked,
    Provisional,
    NotApplicable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReductionDiagnostic {
    pub word: String,
    pub word_index: usize,
    pub citation: String,
    pub realized: String,
    pub reason: String,
    pub status: ReductionStatus,
}

pub trait SentenceAnalyzer {
    fn analyze(&self, source_text: &str, normalized: &NormalizedText) -> SentenceAnalysis;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicSentenceAnalyzer;

impl SentenceAnalyzer for HeuristicSentenceAnalyzer {
    fn analyze(&self, source_text: &str, normalized: &NormalizedText) -> SentenceAnalysis {
        let word_slots = normalized
            .tokens
            .iter()
            .enumerate()
            .filter_map(|(token_index, token)| match token {
                NormalizedToken::Word(word) => Some((token_index, word.clone())),
                NormalizedToken::Initial(initial) => Some((token_index, initial.to_string())),
                NormalizedToken::PhraseBreak => None,
            })
            .collect::<Vec<_>>();
        let token_to_word_index = word_slots
            .iter()
            .enumerate()
            .map(|(word_index, (token_index, _))| (*token_index, word_index))
            .collect::<std::collections::HashMap<_, _>>();

        let tokens = normalized
            .tokens
            .iter()
            .enumerate()
            .map(|(token_index, token)| {
                let (token_text, base_pos) = match token {
                    NormalizedToken::Word(word) => (word.clone(), base_pos(word)),
                    NormalizedToken::Initial(initial) => (
                        initial.to_ascii_lowercase().to_string(),
                        PartOfSpeech::ProperName,
                    ),
                    NormalizedToken::PhraseBreak => ("|".to_string(), PartOfSpeech::Unknown),
                };
                let Some(word_index) = token_to_word_index.get(&token_index).copied() else {
                    return TokenAnalysis {
                        token_index,
                        word_index: None,
                        text: token_text,
                        pos: PartOfSpeech::Unknown,
                        syntactic_role: None,
                        prosodic_role: ProsodicRole::Content,
                        reduction: ReductionClass::None,
                        reduction_diagnostic: None,
                    };
                };

                if token_text != "to" {
                    let prosodic_role = if is_function_word(&token_text) {
                        ProsodicRole::FunctionWeak
                    } else {
                        ProsodicRole::Content
                    };
                    return TokenAnalysis {
                        token_index,
                        word_index: Some(word_index),
                        text: token_text,
                        pos: base_pos,
                        syntactic_role: None,
                        prosodic_role,
                        reduction: ReductionClass::None,
                        reduction_diagnostic: None,
                    };
                }

                let raw_token = normalized
                    .token_spans
                    .get(token_index)
                    .and_then(|span| source_text.get(span.clone()))
                    .unwrap_or("to");
                let prev = word_slots
                    .get(word_index.saturating_sub(1))
                    .map(|(_, text)| text.as_str());
                let prev_prev = word_index
                    .checked_sub(2)
                    .and_then(|idx| word_slots.get(idx))
                    .map(|(_, text)| text.as_str());
                let next = word_slots
                    .get(word_index + 1)
                    .map(|(_, text)| text.as_str());

                let citation = "T UW1".to_string();
                let reduced = "T AH0".to_string();

                let (pos, syntactic_role, prosodic_role, reduction, diagnostic) = classify_to_token(
                    raw_token, word_index, prev_prev, prev, next, &citation, &reduced,
                );

                TokenAnalysis {
                    token_index,
                    word_index: Some(word_index),
                    text: token_text,
                    pos,
                    syntactic_role,
                    prosodic_role,
                    reduction,
                    reduction_diagnostic: Some(diagnostic),
                }
            })
            .collect();

        SentenceAnalysis { tokens }
    }
}

fn classify_to_token(
    raw_token: &str,
    word_index: usize,
    prev_prev: Option<&str>,
    prev: Option<&str>,
    next: Option<&str>,
    citation: &str,
    reduced: &str,
) -> (
    PartOfSpeech,
    Option<SyntacticRole>,
    ProsodicRole,
    ReductionClass,
    ReductionDiagnostic,
) {
    let diagnostic = |realized: &str, reason: &str, status| ReductionDiagnostic {
        word: "to".to_string(),
        word_index,
        citation: citation.to_string(),
        realized: realized.to_string(),
        reason: reason.to_string(),
        status,
    };

    if raw_token.chars().all(|ch| ch.is_ascii_uppercase()) && raw_token.len() > 1 {
        return (
            PartOfSpeech::Preposition,
            Some(SyntacticRole::PrepositionalObjectLink),
            ProsodicRole::Contrastive,
            ReductionClass::None,
            diagnostic(citation, "contrastive_emphasis", ReductionStatus::Blocked),
        );
    }

    if raw_token.contains('/') || raw_token.contains('@') {
        return (
            PartOfSpeech::Preposition,
            Some(SyntacticRole::PrepositionalObjectLink),
            ProsodicRole::FunctionStrong,
            ReductionClass::None,
            diagnostic(
                citation,
                "explicit_phonetic_override",
                ReductionStatus::Blocked,
            ),
        );
    }

    if next.is_none() {
        return (
            PartOfSpeech::Particle,
            Some(SyntacticRole::InfinitivalMarker),
            ProsodicRole::FunctionWeak,
            ReductionClass::WeakFunctionWord,
            diagnostic(
                citation,
                "phrase_final_uncertainty",
                ReductionStatus::Provisional,
            ),
        );
    }

    if next == Some("be") && prev.is_none() {
        return (
            PartOfSpeech::Particle,
            Some(SyntacticRole::InfinitivalMarker),
            ProsodicRole::FunctionStrong,
            ReductionClass::None,
            diagnostic(
                citation,
                "citation_form_phrase_initial",
                ReductionStatus::Blocked,
            ),
        );
    }

    if prev_prev == Some("or") && prev == Some("not") && next == Some("be") {
        return (
            PartOfSpeech::Particle,
            Some(SyntacticRole::InfinitivalMarker),
            ProsodicRole::FunctionStrong,
            ReductionClass::None,
            diagnostic(
                citation,
                "quotation_or_citation_form",
                ReductionStatus::Blocked,
            ),
        );
    }

    if next.is_some_and(is_likely_verb) {
        return (
            PartOfSpeech::Particle,
            Some(SyntacticRole::InfinitivalMarker),
            ProsodicRole::FunctionWeak,
            ReductionClass::WeakFunctionWord,
            diagnostic(
                reduced,
                "unstressed_function_word_before_verb",
                ReductionStatus::Applied,
            ),
        );
    }

    (
        PartOfSpeech::Preposition,
        Some(SyntacticRole::PrepositionalObjectLink),
        ProsodicRole::FunctionStrong,
        ReductionClass::None,
        diagnostic(citation, "prepositional_to", ReductionStatus::Blocked),
    )
}

fn base_pos(word: &str) -> PartOfSpeech {
    if matches!(word, "to") {
        return PartOfSpeech::Preposition;
    }
    if is_pronoun(word) {
        return PartOfSpeech::Pronoun;
    }
    if is_determiner(word) {
        return PartOfSpeech::Determiner;
    }
    if is_conjunction(word) {
        return PartOfSpeech::Conjunction;
    }
    if is_auxiliary(word) {
        return PartOfSpeech::Auxiliary;
    }
    if is_likely_verb(word) {
        return PartOfSpeech::Verb;
    }
    PartOfSpeech::Noun
}

fn is_function_word(word: &str) -> bool {
    is_pronoun(word)
        || is_determiner(word)
        || is_conjunction(word)
        || is_auxiliary(word)
        || matches!(word, "to" | "for" | "of" | "and")
}

fn is_pronoun(word: &str) -> bool {
    matches!(
        word,
        "i" | "you" | "he" | "she" | "it" | "we" | "they" | "me" | "him" | "her" | "us" | "them"
    )
}

fn is_determiner(word: &str) -> bool {
    matches!(
        word,
        "a" | "an" | "the" | "this" | "that" | "these" | "those"
    )
}

fn is_conjunction(word: &str) -> bool {
    matches!(word, "and" | "or" | "but" | "not")
}

fn is_auxiliary(word: &str) -> bool {
    matches!(
        word,
        "be" | "am"
            | "is"
            | "are"
            | "was"
            | "were"
            | "been"
            | "do"
            | "does"
            | "did"
            | "have"
            | "has"
            | "had"
            | "will"
            | "would"
            | "should"
            | "could"
            | "may"
            | "might"
            | "must"
            | "can"
    )
}

fn is_likely_verb(word: &str) -> bool {
    matches!(
        word,
        "go" | "leave"
            | "remember"
            | "see"
            | "stay"
            | "be"
            | "try"
            | "need"
            | "want"
            | "make"
            | "take"
            | "get"
            | "keep"
            | "let"
            | "tell"
            | "call"
            | "put"
            | "ask"
    ) || has_likely_verb_suffix(word)
}

fn has_likely_verb_suffix(word: &str) -> bool {
    const COMMON_NON_VERB_ING: &[&str] = &["thing", "king", "morning", "ceiling"];
    const COMMON_NON_VERB_ED: &[&str] = &["red", "bed", "sled"];
    (word.len() >= 5 && word.ends_with("ing") && !COMMON_NON_VERB_ING.contains(&word))
        || (word.len() >= 4 && word.ends_with("ed") && !COMMON_NON_VERB_ED.contains(&word))
}
