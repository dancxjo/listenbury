use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedText {
    pub tokens: Vec<NormalizedToken>,
    pub boundary: ProsodyBoundaryHint,
    pub commitment: ProsodyCommitment,
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
        let mut saw_sentence_terminal = false;

        for (byte_offset, ch) in trimmed.char_indices() {
            if ch.is_ascii_alphabetic() {
                current.push(ch);
                continue;
            }

            match ch {
                '\'' | '-' | '0'..='9' => {
                    return Err(TextNormalizationError::UnsupportedCharacter { ch, byte_offset });
                }
                '.' => {
                    if finalize_period_token(&mut tokens, &mut current) {
                        saw_sentence_terminal = true;
                    }
                }
                '!' | '?' => {
                    push_word_token(&mut tokens, &mut current);
                    saw_sentence_terminal = true;
                }
                ',' | ';' | ':' => {
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

        let boundary = if saw_sentence_terminal {
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
        })
    }
}

fn finalize_period_token(tokens: &mut Vec<NormalizedToken>, current: &mut String) -> bool {
    if current.is_empty() {
        return false;
    }

    if current.len() == 1 {
        let initial = current
            .chars()
            .next()
            .expect("single-character token should have one char")
            .to_ascii_lowercase();
        tokens.push(NormalizedToken::Initial(initial));
        current.clear();
        return false;
    }

    let lower = current.to_ascii_lowercase();
    current.clear();
    if let Some(expanded) = expand_known_abbreviation(&lower) {
        tokens.push(NormalizedToken::Word(expanded.to_string()));
        return false;
    }

    tokens.push(NormalizedToken::Word(lower));
    true
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
