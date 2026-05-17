use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedText {
    pub tokens: Vec<NormalizedToken>,
    pub boundary: ProsodyBoundaryHint,
    pub commitment: ProsodyCommitment,
    pub punctuation_commitment: PunctuationCommitmentState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizedToken {
    Word(String),
    Initial(char),
    PhraseBreak,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProsodyBoundaryHint {
    None,
    PhraseBreak,
    PossibleSentenceEnd,
    FinalSentenceEnd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProsodyCommitment {
    Provisional,
    Prepared,
    Playable,
    Committed,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        let last_token = stem.split_ascii_whitespace().next_back().unwrap_or_default();
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
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(TextNormalizationError::EmptyInput);
        }

        let mut tokens = Vec::new();
        let mut current = String::new();
        let mut saw_phrase_break = false;
        let punctuation_commitment = HeuristicPunctuationCommitmentClassifier.classify(trimmed);
        let chars: Vec<(usize, char)> = trimmed.char_indices().collect();
        for (index, (byte_offset, ch)) in chars.iter().copied().enumerate() {
            let next = chars.get(index + 1).map(|(_, next)| *next);
            if ch.is_ascii_alphanumeric() || matches!(ch, '@' | '/' | '_') {
                current.push(ch);
                continue;
            }

            match ch {
                '\'' => {
                    return Err(TextNormalizationError::UnsupportedCharacter { ch, byte_offset });
                }
                '.' => {
                    if next.is_some_and(|next| should_treat_as_internal_dot(&current, next)) {
                        current.push(ch);
                        continue;
                    }
                    finalize_period_token(&mut tokens, &mut current);
                }
                '!' | '?' => {
                    push_word_token(&mut tokens, &mut current);
                }
                ':' => {
                    if next.is_some_and(|next| next == '/')
                        && current.to_ascii_lowercase().starts_with("http")
                    {
                        current.push(':');
                        continue;
                    }
                    push_word_token(&mut tokens, &mut current);
                    push_phrase_break(&mut tokens);
                    saw_phrase_break = true;
                }
                ',' | ';' => {
                    push_word_token(&mut tokens, &mut current);
                    push_phrase_break(&mut tokens);
                    saw_phrase_break = true;
                }
                ' ' | '\t' | '\n' | '\r' => {
                    push_word_token(&mut tokens, &mut current);
                }
                '"' | '(' | ')' | '[' | ']' | '{' | '}' => {
                    push_word_token(&mut tokens, &mut current);
                }
                _ => return Err(TextNormalizationError::UnsupportedCharacter { ch, byte_offset }),
            }
        }

        push_word_token(&mut tokens, &mut current);

        if tokens.is_empty() {
            return Err(TextNormalizationError::EmptyInput);
        }

        let boundary = if matches!(punctuation_commitment, PunctuationCommitmentState::SafeToPlay)
        {
            ProsodyBoundaryHint::PossibleSentenceEnd
        } else if saw_phrase_break {
            ProsodyBoundaryHint::PhraseBreak
        } else {
            ProsodyBoundaryHint::None
        };

        Ok(NormalizedText {
            tokens,
            boundary,
            commitment: ProsodyCommitment::Provisional,
            punctuation_commitment,
        })
    }
}

fn finalize_period_token(tokens: &mut Vec<NormalizedToken>, current: &mut String) {
    if current.is_empty() {
        return;
    }

    if current.len() == 1 && current.chars().all(|ch| ch.is_ascii_alphabetic()) {
        let initial = current
            .chars()
            .next()
            .expect("single-character token should have one char")
            .to_ascii_lowercase();
        tokens.push(NormalizedToken::Initial(initial));
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
        return;
    }

    tokens.push(NormalizedToken::Word(lower));
}

fn push_word_token(tokens: &mut Vec<NormalizedToken>, current: &mut String) {
    if current.is_empty() {
        return;
    }
    let lower = current.to_ascii_lowercase();
    current.clear();
    tokens.push(NormalizedToken::Word(lower));
}

fn push_phrase_break(tokens: &mut Vec<NormalizedToken>) {
    if matches!(tokens.last(), Some(NormalizedToken::PhraseBreak)) {
        return;
    }
    tokens.push(NormalizedToken::PhraseBreak);
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
    token.chars().next().is_some_and(|ch| ch.is_ascii_uppercase())
        && expand_known_abbreviation(&token.to_ascii_lowercase()).is_some()
}

fn looks_like_url_or_email(token: &str) -> bool {
    token.contains('@') || token.contains("://") || token.contains("www.")
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
    }

    #[test]
    fn keeps_decimal_without_sentence_commitment() {
        let normalized = TextNormalizer.normalize("3.14").expect("normalize");
        assert_eq!(
            normalized.tokens,
            vec![NormalizedToken::Word("3.14".to_string())]
        );
        assert_eq!(normalized.boundary, ProsodyBoundaryHint::None);
        assert_eq!(
            normalized.punctuation_commitment,
            PunctuationCommitmentState::SafeToPrepare
        );
    }

    #[test]
    fn lowercase_honorific_stays_sentence_ending_candidate() {
        let normalized = TextNormalizer.normalize("dr.").expect("normalize");
        assert_eq!(
            normalized.tokens,
            vec![NormalizedToken::Word("dr".to_string())]
        );
        assert_eq!(normalized.boundary, ProsodyBoundaryHint::PossibleSentenceEnd);
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
