use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::mouth::riper::prosody_audit::PhraseBoundaryKind;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedText {
    pub tokens: Vec<NormalizedToken>,
    pub token_spans: Vec<std::ops::Range<usize>>,
    pub boundary: ProsodyBoundaryHint,
    pub boundary_kind: PhraseBoundaryKind,
    pub commitment: ProsodyCommitment,
    pub punctuation_commitment: PunctuationCommitmentState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizedToken {
    Word(String),
    Initial(char),
    PhraseBreak,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProsodyBoundaryHint {
    None,
    PhraseBreak,
    PossibleSentenceEnd,
    FinalSentenceEnd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProsodyCommitment {
    Provisional,
    Prepared,
    Playable,
    Committed,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PunctuationCommitmentState {
    SafeToPrepare,
    SafeToPlay,
    FinalCadence,
}

pub trait PunctuationCommitmentClassifier {
    fn classify(&self, input: &str) -> PunctuationCommitmentState;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicPunctuationCommitmentClassifier;

impl PunctuationCommitmentClassifier for HeuristicPunctuationCommitmentClassifier {
    fn classify(&self, input: &str) -> PunctuationCommitmentState {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return PunctuationCommitmentState::SafeToPrepare;
        }

        if trimmed.ends_with("...") {
            return PunctuationCommitmentState::SafeToPrepare;
        }

        let mut chars = trimmed.chars();
        let Some(last) = chars.next_back() else {
            return PunctuationCommitmentState::SafeToPrepare;
        };

        match last {
            '!' | '?' => return PunctuationCommitmentState::SafeToPlay,
            '.' => {}
            _ => return PunctuationCommitmentState::SafeToPrepare,
        }

        let stem = trimmed[..trimmed.len() - last.len_utf8()].trim_end();
        let last_token = stem
            .split_ascii_whitespace()
            .next_back()
            .unwrap_or_default();
        if last_token.is_empty() {
            return PunctuationCommitmentState::SafeToPrepare;
        }

        if last_token.len() == 1 && last_token.chars().all(|ch| ch.is_ascii_alphabetic()) {
            return PunctuationCommitmentState::SafeToPrepare;
        }

        if last_token.chars().all(|ch| ch.is_ascii_digit()) {
            return PunctuationCommitmentState::SafeToPrepare;
        }

        if is_decimal_fragment(last_token) {
            return PunctuationCommitmentState::SafeToPrepare;
        }

        if looks_like_url_or_email(last_token) {
            return PunctuationCommitmentState::SafeToPrepare;
        }

        if is_title_case_honorific(last_token) {
            return PunctuationCommitmentState::SafeToPrepare;
        }

        PunctuationCommitmentState::SafeToPlay
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TextNormalizationError {
    #[error("cannot normalize empty text")]
    EmptyInput,
    #[error("unsupported character `{ch}` at byte offset {byte_offset}")]
    UnsupportedCharacter { ch: char, byte_offset: usize },
}

#[derive(Debug, Default, Clone, Copy)]
pub struct TextNormalizer;

impl TextNormalizer {
    pub fn normalize(&self, input: &str) -> Result<NormalizedText, TextNormalizationError> {
        let trim_offset = input.len() - input.trim_start().len();
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(TextNormalizationError::EmptyInput);
        }

        let vocative_detection = detect_vocative(trimmed, trim_offset);
        let mut tokens = Vec::new();
        let mut token_spans = Vec::new();
        let mut current = String::new();
        let mut current_start = None;
        let mut saw_phrase_break = false;
        let punctuation_commitment = HeuristicPunctuationCommitmentClassifier.classify(trimmed);
        let chars: Vec<(usize, char)> = trimmed.char_indices().collect();
        for (index, (byte_offset, ch)) in chars.iter().copied().enumerate() {
            let next = chars.get(index + 1).map(|(_, next)| *next);
            if ch.is_ascii_alphanumeric() || matches!(ch, '@' | '/' | '_') {
                if current.is_empty() {
                    current_start = Some(trim_offset + byte_offset);
                }
                current.push(ch);
                continue;
            }

            match ch {
                '\'' | '’' => {
                    if !current.is_empty() && next.is_some_and(|next| next.is_ascii_alphanumeric())
                    {
                        current.push('\'');
                    } else {
                        push_word_token(
                            &mut tokens,
                            &mut token_spans,
                            &mut current,
                            &mut current_start,
                            trim_offset + byte_offset,
                        );
                    }
                }
                '.' => {
                    if next.is_some_and(|next| should_treat_as_internal_dot(&current, next)) {
                        if current.is_empty() {
                            current_start = Some(trim_offset + byte_offset);
                        }
                        current.push(ch);
                        continue;
                    }
                    finalize_period_token(
                        &mut tokens,
                        &mut token_spans,
                        &mut current,
                        &mut current_start,
                        trim_offset + byte_offset + 1,
                    );
                }
                '!' | '?' => {
                    push_word_token(
                        &mut tokens,
                        &mut token_spans,
                        &mut current,
                        &mut current_start,
                        trim_offset + byte_offset,
                    );
                }
                ':' => {
                    if next.is_some_and(|next| next == '/') && looks_like_url_prefix(&current) {
                        if current.is_empty() {
                            current_start = Some(trim_offset + byte_offset);
                        }
                        current.push(':');
                        continue;
                    }
                    push_word_token(
                        &mut tokens,
                        &mut token_spans,
                        &mut current,
                        &mut current_start,
                        trim_offset + byte_offset,
                    );
                    push_phrase_break(
                        &mut tokens,
                        &mut token_spans,
                        trim_offset + byte_offset,
                        trim_offset + byte_offset + 1,
                    );
                    saw_phrase_break = true;
                }
                ',' | ';' => {
                    push_word_token(
                        &mut tokens,
                        &mut token_spans,
                        &mut current,
                        &mut current_start,
                        trim_offset + byte_offset,
                    );
                    let is_vocative_comma =
                        ch == ',' && vocative_detection.comma_offsets.contains(&byte_offset);
                    if !is_vocative_comma {
                        push_phrase_break(
                            &mut tokens,
                            &mut token_spans,
                            trim_offset + byte_offset,
                            trim_offset + byte_offset + 1,
                        );
                        saw_phrase_break = true;
                    }
                }
                ' ' | '\t' | '\n' | '\r' => {
                    push_word_token(
                        &mut tokens,
                        &mut token_spans,
                        &mut current,
                        &mut current_start,
                        trim_offset + byte_offset,
                    );
                }
                '"' | '(' | ')' | '[' | ']' | '{' | '}' => {
                    push_word_token(
                        &mut tokens,
                        &mut token_spans,
                        &mut current,
                        &mut current_start,
                        trim_offset + byte_offset,
                    );
                }
                _ => return Err(TextNormalizationError::UnsupportedCharacter { ch, byte_offset }),
            }
        }

        push_word_token(
            &mut tokens,
            &mut token_spans,
            &mut current,
            &mut current_start,
            trim_offset + trimmed.len(),
        );

        if tokens.is_empty() {
            return Err(TextNormalizationError::EmptyInput);
        }

        let boundary = if matches!(
            punctuation_commitment,
            PunctuationCommitmentState::SafeToPlay
        ) {
            ProsodyBoundaryHint::PossibleSentenceEnd
        } else if saw_phrase_break {
            ProsodyBoundaryHint::PhraseBreak
        } else {
            ProsodyBoundaryHint::None
        };
        let boundary_kind = classify_phrase_boundary_kind(
            trimmed,
            saw_phrase_break,
            boundary,
            !vocative_detection.spans.is_empty(),
        );

        Ok(NormalizedText {
            tokens,
            token_spans,
            boundary,
            boundary_kind,
            commitment: ProsodyCommitment::Provisional,
            punctuation_commitment,
        })
    }
}

fn classify_phrase_boundary_kind(
    input: &str,
    saw_phrase_break: bool,
    boundary: ProsodyBoundaryHint,
    has_vocative: bool,
) -> PhraseBoundaryKind {
    if has_vocative {
        return PhraseBoundaryKind::Vocative;
    }
    let Some(last) = input
        .trim_end_matches(|ch: char| ch.is_ascii_whitespace() || is_quote_or_bracket(ch))
        .chars()
        .next_back()
    else {
        return PhraseBoundaryKind::None;
    };
    match last {
        ',' => PhraseBoundaryKind::MinorPhrase,
        ';' | ':' => PhraseBoundaryKind::MajorPhrase,
        '!' => PhraseBoundaryKind::Exclamation,
        '?' => PhraseBoundaryKind::FinalRising,
        '.' => match boundary {
            ProsodyBoundaryHint::PossibleSentenceEnd | ProsodyBoundaryHint::FinalSentenceEnd => {
                PhraseBoundaryKind::FinalFalling
            }
            _ => PhraseBoundaryKind::PossibleFinal,
        },
        '-' | '—' | '–' | '(' | ')' | '[' | ']' => PhraseBoundaryKind::Parenthetical,
        _ if saw_phrase_break => PhraseBoundaryKind::MinorPhrase,
        _ => PhraseBoundaryKind::None,
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct VocativeDetection {
    spans: Vec<std::ops::Range<usize>>,
    comma_offsets: Vec<usize>,
}

pub(crate) fn detect_vocative_spans(input: &str) -> Vec<std::ops::Range<usize>> {
    let trim_offset = input.len() - input.trim_start().len();
    let trimmed = input.trim();
    detect_vocative(trimmed, trim_offset).spans
}

fn detect_vocative(trimmed: &str, trim_offset: usize) -> VocativeDetection {
    if trimmed.is_empty() {
        return VocativeDetection::default();
    }

    let comma_offsets = trimmed
        .match_indices(',')
        .map(|(idx, _)| idx)
        .collect::<Vec<_>>();
    if comma_offsets.is_empty() {
        return VocativeDetection::default();
    }

    let mut spans = Vec::new();
    let mut keep_commas = std::collections::BTreeSet::new();

    if let Some(last_comma) = comma_offsets.last().copied()
        && trimmed[..last_comma]
            .chars()
            .any(|ch| ch.is_ascii_alphabetic())
        && let Some(vocative_span) = detect_addressee_span(&trimmed[last_comma + 1..], false)
    {
        let start = trim_offset + last_comma + 1 + vocative_span.start;
        let end = trim_offset + last_comma + 1 + vocative_span.end;
        spans.push(start..end);
        keep_commas.insert(last_comma);
    }

    for window in comma_offsets.windows(2) {
        let [left_comma, right_comma] = [window[0], window[1]];
        if !has_discourse_cue(&trimmed[..left_comma]) {
            continue;
        }
        if let Some(vocative_span) =
            detect_addressee_span(&trimmed[left_comma + 1..right_comma], true)
        {
            let start = trim_offset + left_comma + 1 + vocative_span.start;
            let end = trim_offset + left_comma + 1 + vocative_span.end;
            spans.push(start..end);
            keep_commas.insert(left_comma);
            keep_commas.insert(right_comma);
        }
    }

    VocativeDetection {
        spans,
        comma_offsets: keep_commas.into_iter().collect(),
    }
}

fn detect_addressee_span(
    segment: &str,
    require_vocative_noun: bool,
) -> Option<std::ops::Range<usize>> {
    let leading_ws = segment.len() - segment.trim_start().len();
    let mut trimmed = segment.trim();
    if trimmed.is_empty() {
        return None;
    }

    let trailing_punctuation = trimmed
        .chars()
        .rev()
        .take_while(|ch| matches!(ch, '.' | '!' | '?'))
        .map(char::len_utf8)
        .sum::<usize>();
    if trailing_punctuation > 0 {
        trimmed = &trimmed[..trimmed.len() - trailing_punctuation];
        trimmed = trimmed.trim_end();
    }
    if trimmed.is_empty() {
        return None;
    }

    let words = trimmed
        .split_ascii_whitespace()
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    if words.is_empty() || words.len() > 3 {
        return None;
    }

    let lower_words = words
        .iter()
        .map(|word| {
            word.trim_matches(|ch: char| !ch.is_ascii_alphabetic())
                .to_ascii_lowercase()
        })
        .collect::<Vec<_>>();
    if lower_words.iter().any(|word| {
        matches!(
            word.as_str(),
            "who" | "which" | "that" | "where" | "when" | "unfortunately" | "however"
        )
    }) {
        return None;
    }
    let has_capitalized = words.iter().any(|word| {
        word.chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_uppercase())
    });
    let has_vocative_noun = lower_words
        .iter()
        .any(|word| is_likely_vocative_noun(word.as_str()));
    if (!has_capitalized && !has_vocative_noun) || (require_vocative_noun && !has_vocative_noun) {
        return None;
    }

    let span_start = leading_ws;
    let span_end = leading_ws + trimmed.len();
    Some(span_start..span_end)
}

fn has_discourse_cue(prefix: &str) -> bool {
    let words = prefix
        .split_ascii_whitespace()
        .map(|word| {
            word.trim_matches(|ch: char| !ch.is_ascii_alphabetic())
                .to_ascii_lowercase()
        })
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    if words.is_empty() {
        return false;
    }
    matches!(
        words.last().map(String::as_str),
        Some("listen" | "look" | "see")
    ) || words.ends_with(&["you".to_string(), "see".to_string()])
}

fn is_likely_vocative_noun(word: &str) -> bool {
    matches!(
        word,
        "professor"
            | "interlocutor"
            | "sir"
            | "madam"
            | "friend"
            | "friends"
            | "team"
            | "folks"
            | "everyone"
            | "everybody"
    )
}

fn is_quote_or_bracket(ch: char) -> bool {
    matches!(
        ch,
        '"' | '\'' | '“' | '”' | '‘' | '’' | ')' | ']' | '}' | '(' | '[' | '{'
    )
}

fn finalize_period_token(
    tokens: &mut Vec<NormalizedToken>,
    token_spans: &mut Vec<std::ops::Range<usize>>,
    current: &mut String,
    current_start: &mut Option<usize>,
    token_end: usize,
) {
    if current.is_empty() {
        return;
    }
    let start = current_start
        .take()
        .expect("token start should be tracked for non-empty token");

    if current.len() == 1 && current.chars().all(|ch| ch.is_ascii_alphabetic()) {
        let initial = current
            .chars()
            .next()
            .expect("single-character token should have one char")
            .to_ascii_lowercase();
        tokens.push(NormalizedToken::Initial(initial));
        token_spans.push(start..token_end);
        current.clear();
        return;
    }

    let original = current.clone();
    let lower = current.to_ascii_lowercase();
    current.clear();
    if is_title_case_honorific(&original)
        && let Some(expanded) = expand_known_abbreviation(&lower)
    {
        tokens.push(NormalizedToken::Word(expanded.to_string()));
        token_spans.push(start..token_end);
        return;
    }

    tokens.push(NormalizedToken::Word(lower));
    token_spans.push(start..token_end);
}

fn push_word_token(
    tokens: &mut Vec<NormalizedToken>,
    token_spans: &mut Vec<std::ops::Range<usize>>,
    current: &mut String,
    current_start: &mut Option<usize>,
    token_end: usize,
) {
    if current.is_empty() {
        return;
    }
    let start = current_start
        .take()
        .expect("token start should be tracked for non-empty token");
    let lower = current.to_ascii_lowercase();
    current.clear();
    tokens.push(NormalizedToken::Word(lower));
    token_spans.push(start..token_end);
}

fn push_phrase_break(
    tokens: &mut Vec<NormalizedToken>,
    token_spans: &mut Vec<std::ops::Range<usize>>,
    start: usize,
    end: usize,
) {
    if matches!(tokens.last(), Some(NormalizedToken::PhraseBreak)) {
        return;
    }
    tokens.push(NormalizedToken::PhraseBreak);
    token_spans.push(start..end);
}

fn expand_known_abbreviation(token: &str) -> Option<&'static str> {
    match token {
        "dr" => Some("doctor"),
        "mr" => Some("mister"),
        "mrs" => Some("missis"),
        "ms" => Some("miss"),
        "prof" => Some("professor"),
        _ => None,
    }
}

fn is_title_case_honorific(token: &str) -> bool {
    token
        .chars()
        .next()
        .is_some_and(|ch| ch.is_ascii_uppercase())
        && expand_known_abbreviation(&token.to_ascii_lowercase()).is_some()
}

fn looks_like_url_or_email(token: &str) -> bool {
    token.contains('@')
        || token.contains("://")
        || token.contains("www.")
        || looks_like_url_prefix(token)
}

fn looks_like_url_prefix(token: &str) -> bool {
    let lower = token.to_ascii_lowercase();
    lower.starts_with("http") || lower.starts_with("www")
}

fn is_decimal_fragment(token: &str) -> bool {
    token.split_once('.').is_some_and(|(left, right)| {
        !left.is_empty()
            && !right.is_empty()
            && left.chars().all(|ch| ch.is_ascii_digit())
            && right.chars().all(|ch| ch.is_ascii_digit())
    })
}

fn should_treat_as_internal_dot(current: &str, next: char) -> bool {
    (current.chars().last().is_some_and(|ch| ch.is_ascii_digit()) && next.is_ascii_digit())
        || (next.is_ascii_alphanumeric() && looks_like_url_or_email(current))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_honorific_and_sentence_boundary() {
        let normalized = TextNormalizer.normalize("Dr. King.").expect("normalize");
        assert_eq!(
            normalized.tokens,
            vec![
                NormalizedToken::Word("doctor".to_string()),
                NormalizedToken::Word("king".to_string())
            ]
        );
        assert_eq!(
            normalized.boundary,
            ProsodyBoundaryHint::PossibleSentenceEnd
        );
        assert_eq!(normalized.boundary_kind, PhraseBoundaryKind::FinalFalling);
        assert_eq!(normalized.commitment, ProsodyCommitment::Provisional);
        assert_eq!(
            normalized.punctuation_commitment,
            PunctuationCommitmentState::SafeToPlay
        );
    }

    #[test]
    fn keeps_initials_and_phrase_breaks() {
        let normalized = TextNormalizer
            .normalize("J. R. R., test")
            .expect("normalize");
        assert_eq!(
            normalized.tokens,
            vec![
                NormalizedToken::Initial('j'),
                NormalizedToken::Initial('r'),
                NormalizedToken::Initial('r'),
                NormalizedToken::PhraseBreak,
                NormalizedToken::Word("test".to_string())
            ]
        );
        assert_eq!(normalized.boundary, ProsodyBoundaryHint::PhraseBreak);
        assert_eq!(normalized.boundary_kind, PhraseBoundaryKind::MinorPhrase);
    }

    #[test]
    fn keeps_decimal_without_sentence_commitment() {
        let normalized = TextNormalizer.normalize("3.14").expect("normalize");
        assert_eq!(
            normalized.tokens,
            vec![NormalizedToken::Word("3.14".to_string())]
        );
        assert_eq!(normalized.boundary, ProsodyBoundaryHint::None);
        assert_eq!(normalized.boundary_kind, PhraseBoundaryKind::None);
        assert_eq!(
            normalized.punctuation_commitment,
            PunctuationCommitmentState::SafeToPrepare
        );
    }

    #[test]
    fn classifies_question_and_exclamation_boundaries() {
        let question = TextNormalizer
            .normalize("Is this ready?")
            .expect("normalize");
        assert_eq!(question.boundary_kind, PhraseBoundaryKind::FinalRising);

        let exclamation = TextNormalizer.normalize("Listen!").expect("normalize");
        assert_eq!(exclamation.boundary_kind, PhraseBoundaryKind::Exclamation);
    }

    #[test]
    fn detects_sentence_final_vocative_and_suppresses_comma_break() {
        let normalized = TextNormalizer
            .normalize("Thank you, Dave.")
            .expect("normalize");
        assert_eq!(normalized.boundary_kind, PhraseBoundaryKind::Vocative);
        assert_eq!(
            normalized.tokens,
            vec![
                NormalizedToken::Word("thank".to_string()),
                NormalizedToken::Word("you".to_string()),
                NormalizedToken::Word("dave".to_string()),
            ]
        );
    }

    #[test]
    fn detects_comma_surrounded_vocative_with_discourse_cues() {
        let listen = TextNormalizer
            .normalize("Listen, professor, this matters.")
            .expect("normalize");
        assert_eq!(listen.boundary_kind, PhraseBoundaryKind::Vocative);
        assert!(
            !listen
                .tokens
                .iter()
                .any(|token| matches!(token, NormalizedToken::PhraseBreak))
        );

        let see = TextNormalizer
            .normalize("You see, interlocutor, the system has revealed itself.")
            .expect("normalize");
        assert_eq!(see.boundary_kind, PhraseBoundaryKind::Vocative);
        assert!(
            !see.tokens
                .iter()
                .any(|token| matches!(token, NormalizedToken::PhraseBreak))
        );
    }

    #[test]
    fn keeps_parenthetical_and_appositional_commas_as_phrase_breaks() {
        let parenthetical = TextNormalizer
            .normalize("The machine, unfortunately, exploded.")
            .expect("normalize");
        assert_ne!(parenthetical.boundary_kind, PhraseBoundaryKind::Vocative);
        assert!(
            parenthetical
                .tokens
                .iter()
                .any(|token| matches!(token, NormalizedToken::PhraseBreak))
        );

        let apposition = TextNormalizer
            .normalize("My brother, who lives in Tacoma, arrived.")
            .expect("normalize");
        assert_ne!(apposition.boundary_kind, PhraseBoundaryKind::Vocative);
        assert!(
            apposition
                .tokens
                .iter()
                .any(|token| matches!(token, NormalizedToken::PhraseBreak))
        );
    }

    #[test]
    fn lowercase_honorific_stays_sentence_ending_candidate() {
        let normalized = TextNormalizer.normalize("dr.").expect("normalize");
        assert_eq!(
            normalized.tokens,
            vec![NormalizedToken::Word("dr".to_string())]
        );
        assert_eq!(
            normalized.boundary,
            ProsodyBoundaryHint::PossibleSentenceEnd
        );
        assert_eq!(
            normalized.punctuation_commitment,
            PunctuationCommitmentState::SafeToPlay
        );
    }

    #[test]
    fn keeps_ellipsis_provisional() {
        let normalized = TextNormalizer.normalize("Wait...").expect("normalize");
        assert_eq!(
            normalized.tokens,
            vec![NormalizedToken::Word("wait".to_string())]
        );
        assert_eq!(normalized.boundary, ProsodyBoundaryHint::None);
        assert_eq!(
            normalized.punctuation_commitment,
            PunctuationCommitmentState::SafeToPrepare
        );
    }

    #[test]
    fn keeps_url_and_email_periods_provisional() {
        let url = TextNormalizer
            .normalize("go to https://example.com")
            .expect("normalize");
        assert_eq!(url.boundary, ProsodyBoundaryHint::None);
        assert_eq!(
            url.punctuation_commitment,
            PunctuationCommitmentState::SafeToPrepare
        );

        let email = TextNormalizer
            .normalize("me@example.com")
            .expect("normalize");
        assert_eq!(email.boundary, ProsodyBoundaryHint::None);
        assert_eq!(
            email.punctuation_commitment,
            PunctuationCommitmentState::SafeToPrepare
        );
    }

    #[test]
    fn keeps_internal_apostrophes_in_contractions() {
        let normalized = TextNormalizer.normalize("It's ready").expect("normalize");
        assert_eq!(
            normalized.tokens,
            vec![
                NormalizedToken::Word("it's".to_string()),
                NormalizedToken::Word("ready".to_string())
            ]
        );

        let curly = TextNormalizer.normalize("It’s ready").expect("normalize");
        assert_eq!(curly.tokens, normalized.tokens);
    }

    #[test]
    fn treats_quote_apostrophes_as_punctuation() {
        let normalized = TextNormalizer.normalize("'Hello'").expect("normalize");
        assert_eq!(
            normalized.tokens,
            vec![NormalizedToken::Word("hello".to_string())]
        );
    }

    #[test]
    fn tracks_token_byte_spans_in_original_text() {
        let normalized = TextNormalizer
            .normalize("  F. Scott, \"okay\"  ")
            .expect("normalize");
        assert_eq!(
            normalized.tokens,
            vec![
                NormalizedToken::Initial('f'),
                NormalizedToken::Word("scott".to_string()),
                NormalizedToken::PhraseBreak,
                NormalizedToken::Word("okay".to_string())
            ]
        );
        assert_eq!(normalized.token_spans[0], 2..4);
        assert_eq!(normalized.token_spans[1], 5..10);
        assert_eq!(normalized.token_spans[2], 10..11);
        assert_eq!(normalized.token_spans[3], 13..17);
    }

    #[test]
    fn returns_clear_error_for_unsupported_characters() {
        let error = TextNormalizer
            .normalize("hello🙂")
            .expect_err("emoji should be unsupported");
        assert_eq!(
            error,
            TextNormalizationError::UnsupportedCharacter {
                ch: '🙂',
                byte_offset: 5
            }
        );
    }
}
