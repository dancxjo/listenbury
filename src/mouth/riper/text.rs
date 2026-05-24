use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::linguistic::language_pack_rules::english_punctuation_rule;
use crate::mouth::riper::prosody_audit::PhraseBoundaryKind;

const MAX_VOCATIVE_WORDS: usize = 3;
const COMMON_VOCATIVE_NOUNS: &[&str] = &[
    "mr",
    "mrs",
    "ms",
    "miss",
    "dr",
    "professor",
    "interlocutor",
    "sir",
    "madam",
    "friend",
    "friends",
    "team",
    "folks",
    "everyone",
    "everybody",
    "doctor",
    "captain",
    "mom",
    "dad",
    "buddy",
];

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
            if let Some(folded) = fold_latin_letter(ch) {
                if current.is_empty() {
                    current_start = Some(trim_offset + byte_offset);
                }
                current.push_str(folded);
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
                '-' | '–' | '—' => {
                    push_word_token(
                        &mut tokens,
                        &mut token_spans,
                        &mut current,
                        &mut current_start,
                        trim_offset + byte_offset,
                    );
                    if !is_infix_word_dash(&chars, index) {
                        push_phrase_break(
                            &mut tokens,
                            &mut token_spans,
                            trim_offset + byte_offset,
                            trim_offset + byte_offset + ch.len_utf8(),
                        );
                        saw_phrase_break = true;
                    }
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

fn fold_latin_letter(ch: char) -> Option<&'static str> {
    Some(match ch {
        'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' | 'Ā' | 'Ă' | 'Ą' | 'à' | 'á' | 'â' | 'ã' | 'ä' | 'å'
        | 'ā' | 'ă' | 'ą' => "a",
        'Æ' | 'æ' => "ae",
        'Ç' | 'Ć' | 'Ĉ' | 'Ċ' | 'Č' | 'ç' | 'ć' | 'ĉ' | 'ċ' | 'č' => "c",
        'Ð' | 'Ď' | 'Đ' | 'ð' | 'ď' | 'đ' => "d",
        'È' | 'É' | 'Ê' | 'Ë' | 'Ē' | 'Ĕ' | 'Ė' | 'Ę' | 'Ě' | 'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ĕ'
        | 'ė' | 'ę' | 'ě' => "e",
        'Ĝ' | 'Ğ' | 'Ġ' | 'Ģ' | 'ĝ' | 'ğ' | 'ġ' | 'ģ' => "g",
        'Ĥ' | 'Ħ' | 'ĥ' | 'ħ' => "h",
        'Ì' | 'Í' | 'Î' | 'Ï' | 'Ĩ' | 'Ī' | 'Ĭ' | 'Į' | 'İ' | 'ì' | 'í' | 'î' | 'ï' | 'ĩ' | 'ī'
        | 'ĭ' | 'į' | 'ı' => "i",
        'Ĵ' | 'ĵ' => "j",
        'Ķ' | 'ķ' | 'ĸ' => "k",
        'Ĺ' | 'Ļ' | 'Ľ' | 'Ŀ' | 'Ł' | 'ĺ' | 'ļ' | 'ľ' | 'ŀ' | 'ł' => "l",
        'Ñ' | 'Ń' | 'Ņ' | 'Ň' | 'ñ' | 'ń' | 'ņ' | 'ň' => "n",
        'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' | 'Ø' | 'Ō' | 'Ŏ' | 'Ő' | 'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø'
        | 'ō' | 'ŏ' | 'ő' => "o",
        'Œ' | 'œ' => "oe",
        'Ŕ' | 'Ŗ' | 'Ř' | 'ŕ' | 'ŗ' | 'ř' => "r",
        'Ś' | 'Ŝ' | 'Ş' | 'Š' | 'ś' | 'ŝ' | 'ş' | 'š' | 'ſ' => "s",
        'Ţ' | 'Ť' | 'Ŧ' | 'ţ' | 'ť' | 'ŧ' => "t",
        'Ù' | 'Ú' | 'Û' | 'Ü' | 'Ũ' | 'Ū' | 'Ŭ' | 'Ů' | 'Ű' | 'Ų' | 'ù' | 'ú' | 'û' | 'ü' | 'ũ'
        | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' => "u",
        'Ŵ' | 'ŵ' => "w",
        'Ý' | 'Ŷ' | 'Ÿ' | 'ý' | 'ÿ' | 'ŷ' => "y",
        'Ź' | 'Ż' | 'Ž' | 'ź' | 'ż' | 'ž' => "z",
        'Þ' | 'þ' => "th",
        'ß' => "ss",
        _ => return None,
    })
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
    if let Some(rule) = english_punctuation_rule(last) {
        return match rule.output_transformation.as_str() {
            "boundary:exclamation" => PhraseBoundaryKind::Exclamation,
            "boundary:final_rising" => PhraseBoundaryKind::FinalRising,
            "boundary:final_falling" => match boundary {
                ProsodyBoundaryHint::PossibleSentenceEnd
                | ProsodyBoundaryHint::FinalSentenceEnd => PhraseBoundaryKind::FinalFalling,
                _ => PhraseBoundaryKind::PossibleFinal,
            },
            "boundary:minor" => PhraseBoundaryKind::MinorPhrase,
            "boundary:major" => PhraseBoundaryKind::MajorPhrase,
            _ => PhraseBoundaryKind::None,
        };
    }
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

    if let Some(first_comma) = comma_offsets.first().copied()
        && trimmed[first_comma + 1..]
            .chars()
            .any(|ch| ch.is_ascii_alphabetic())
        && !comma_followed_by_place_name(trimmed, first_comma)
        && let Some(vocative_span) = detect_initial_addressee_span(&trimmed[..first_comma])
    {
        let start = trim_offset + vocative_span.start;
        let end = trim_offset + vocative_span.end;
        spans.push(start..end);
        keep_commas.insert(first_comma);
    }

    for window in comma_offsets.windows(2) {
        let [left_comma, right_comma] = [window[0], window[1]];
        let prefix = &trimmed[..left_comma];
        let segment = &trimmed[left_comma + 1..right_comma];
        let has_discourse_cue = has_discourse_cue(prefix);
        let has_direct_address_context =
            has_discourse_cue || has_mid_sentence_direct_address_context(prefix);
        if !has_direct_address_context
            || (!has_discourse_cue && is_likely_place_name_segment(segment))
        {
            continue;
        }
        if let Some(vocative_span) = detect_addressee_span(segment, false) {
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
    if words.is_empty() || words.len() > MAX_VOCATIVE_WORDS {
        return None;
    }

    let lower_words = words
        .iter()
        .map(|word| {
            word.trim_matches(|ch: char| !ch.is_ascii_alphabetic())
                .to_ascii_lowercase()
        })
        .collect::<Vec<_>>();
    if lower_words
        .iter()
        .any(|word| is_addressee_blocked_word(word))
    {
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
    if (require_vocative_noun || !has_capitalized) && !has_vocative_noun {
        return None;
    }

    let span_start = leading_ws;
    let span_end = leading_ws + trimmed.len();
    Some(span_start..span_end)
}

fn detect_initial_addressee_span(segment: &str) -> Option<std::ops::Range<usize>> {
    let span = detect_addressee_span(segment, false)?;
    let trimmed = &segment[span.clone()];
    let words = trimmed
        .split_ascii_whitespace()
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    let lower_words = words
        .iter()
        .map(|word| {
            word.trim_matches(|ch: char| !ch.is_ascii_alphabetic())
                .to_ascii_lowercase()
        })
        .collect::<Vec<_>>();
    if lower_words
        .iter()
        .any(|word| is_initial_vocative_blocked_word(word))
    {
        return None;
    }

    let has_vocative_noun = lower_words
        .iter()
        .any(|word| is_likely_vocative_noun(word.as_str()));
    let looks_like_proper_name = !trimmed.contains('.')
        && words.iter().all(|word| {
            word.chars()
                .find(|ch| ch.is_ascii_alphabetic())
                .is_some_and(|ch| ch.is_ascii_uppercase())
        });
    (has_vocative_noun || looks_like_proper_name).then_some(span)
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
        Some(
            "listen"
                | "look"
                | "see"
                | "hey"
                | "hello"
                | "hi"
                | "well"
                | "ok"
                | "okay"
                | "yes"
                | "no"
        )
    ) || matches!(words.as_slice(), [.., prev, last] if prev == "you" && last == "see")
}

fn has_mid_sentence_direct_address_context(prefix: &str) -> bool {
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
        Some("sorry" | "thanks" | "please" | "yes" | "no" | "ok" | "okay")
    ) || matches!(words.as_slice(), [.., prev, last] if prev == "thank" && last == "you")
        || matches!(
            words.as_slice(),
            [first, .., last]
                if matches!(first.as_str(), "i" | "we")
                    && matches!(
                        last.as_str(),
                        "agree"
                            | "know"
                            | "think"
                            | "mean"
                            | "suppose"
                            | "believe"
                            | "understand"
                            | "appreciate"
                            | "hear"
                    )
        )
}

fn is_likely_vocative_noun(word: &str) -> bool {
    COMMON_VOCATIVE_NOUNS.contains(&word)
}

fn is_likely_place_name_segment(segment: &str) -> bool {
    let words = segment
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

    let joined = words.join(" ");
    US_STATE_NAMES.contains(&joined.as_str())
        || PLACE_NAME_WORDS.contains(&joined.as_str())
        || words
            .iter()
            .all(|word| PLACE_NAME_WORDS.contains(&word.as_str()))
}

fn comma_followed_by_place_name(trimmed: &str, comma_offset: usize) -> bool {
    let next_segment_end = trimmed[comma_offset + 1..]
        .find(',')
        .map(|offset| comma_offset + 1 + offset)
        .unwrap_or(trimmed.len());
    is_likely_place_name_segment(&trimmed[comma_offset + 1..next_segment_end])
}

fn is_addressee_blocked_word(word: &str) -> bool {
    matches!(
        word,
        "a" | "an"
            | "the"
            | "this"
            | "that"
            | "these"
            | "those"
            | "i"
            | "we"
            | "you"
            | "he"
            | "she"
            | "it"
            | "they"
            | "who"
            | "which"
            | "where"
            | "when"
            | "unfortunately"
            | "however"
    )
}

fn is_initial_vocative_blocked_word(word: &str) -> bool {
    if is_addressee_blocked_word(word) {
        return true;
    }
    matches!(
        word,
        "well"
            | "now"
            | "then"
            | "so"
            | "yes"
            | "no"
            | "ok"
            | "okay"
            | "please"
            | "thanks"
            | "thank"
            | "hey"
            | "hello"
            | "hi"
            | "listen"
            | "look"
            | "see"
            | "monday"
            | "tuesday"
            | "wednesday"
            | "thursday"
            | "friday"
            | "saturday"
            | "sunday"
            | "january"
            | "february"
            | "march"
            | "april"
            | "may"
            | "june"
            | "july"
            | "august"
            | "september"
            | "october"
            | "november"
            | "december"
    )
}

const US_STATE_NAMES: &[&str] = &[
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "new hampshire",
    "new jersey",
    "new mexico",
    "new york",
    "north carolina",
    "north dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhode island",
    "south carolina",
    "south dakota",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west virginia",
    "wisconsin",
    "wyoming",
];

const PLACE_NAME_WORDS: &[&str] = &[
    "america",
    "canada",
    "mexico",
    "england",
    "scotland",
    "wales",
    "ireland",
    "france",
    "germany",
    "italy",
    "spain",
    "china",
    "japan",
    "korea",
    "india",
    "australia",
    "zealand",
];

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

fn is_infix_word_dash(chars: &[(usize, char)], index: usize) -> bool {
    let previous = index
        .checked_sub(1)
        .and_then(|previous| chars.get(previous))
        .is_some_and(|(_, ch)| ch.is_ascii_alphanumeric());
    let next = chars
        .get(index + 1)
        .is_some_and(|(_, ch)| ch.is_ascii_alphanumeric());
    previous && next
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
    fn splits_hyphenated_compounds_into_pronounceable_words() {
        let normalized = TextNormalizer
            .normalize("low-latency speech")
            .expect("normalize hyphenated compound");
        assert_eq!(
            normalized.tokens,
            vec![
                NormalizedToken::Word("low".to_string()),
                NormalizedToken::Word("latency".to_string()),
                NormalizedToken::Word("speech".to_string()),
            ]
        );
        assert_eq!(normalized.boundary, ProsodyBoundaryHint::None);
        assert_eq!(normalized.boundary_kind, PhraseBoundaryKind::None);
    }

    #[test]
    fn treats_standalone_dashes_as_phrase_breaks() {
        let normalized = TextNormalizer
            .normalize("ready - maybe")
            .expect("normalize dashed phrase");
        assert_eq!(
            normalized.tokens,
            vec![
                NormalizedToken::Word("ready".to_string()),
                NormalizedToken::PhraseBreak,
                NormalizedToken::Word("maybe".to_string()),
            ]
        );
        assert_eq!(normalized.boundary, ProsodyBoundaryHint::PhraseBreak);
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
    fn punctuation_boundary_rules_include_espeak_provenance() {
        let exclamation_rule = english_punctuation_rule('!').expect("exclamation rule");
        assert_eq!(exclamation_rule.rule_id, "punctuation_exclamation_boundary");
        assert_eq!(
            exclamation_rule.output_transformation,
            "boundary:exclamation"
        );
        assert_eq!(exclamation_rule.provenance.source, "espeak-ng-derived");
        assert_eq!(
            exclamation_rule.provenance.source_license,
            "GPL-3.0-or-later"
        );
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
    fn detects_initial_vocative_and_suppresses_direct_address_comma() {
        for fixture in ["Dave, thank you.", "Friends, listen closely."] {
            let normalized = TextNormalizer.normalize(fixture).expect("normalize");
            assert_eq!(normalized.boundary_kind, PhraseBoundaryKind::Vocative);
            assert!(
                !normalized
                    .tokens
                    .iter()
                    .any(|token| matches!(token, NormalizedToken::PhraseBreak)),
                "initial direct-address comma should not become a hard phrase break: {fixture}"
            );
        }
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

        let capitalized_name = TextNormalizer
            .normalize("Listen, Dave, this matters.")
            .expect("normalize");
        assert_eq!(capitalized_name.boundary_kind, PhraseBoundaryKind::Vocative);
        assert!(
            !capitalized_name
                .tokens
                .iter()
                .any(|token| matches!(token, NormalizedToken::PhraseBreak))
        );

        let greeting = TextNormalizer
            .normalize("Hey, Dave, listen.")
            .expect("normalize");
        assert_eq!(greeting.boundary_kind, PhraseBoundaryKind::Vocative);
        assert!(
            !greeting
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

        let thanks = TextNormalizer
            .normalize("Thank you, Dave, I appreciate it.")
            .expect("normalize");
        assert_eq!(thanks.boundary_kind, PhraseBoundaryKind::Vocative);
        assert!(
            !thanks
                .tokens
                .iter()
                .any(|token| matches!(token, NormalizedToken::PhraseBreak))
        );
    }

    #[test]
    fn keeps_discourse_and_temporal_leading_commas_as_phrase_breaks() {
        for fixture in ["Well, this matters.", "Monday, we leave."] {
            let normalized = TextNormalizer.normalize(fixture).expect("normalize");
            assert_ne!(normalized.boundary_kind, PhraseBoundaryKind::Vocative);
            assert!(
                normalized
                    .tokens
                    .iter()
                    .any(|token| matches!(token, NormalizedToken::PhraseBreak)),
                "non-vocative leading comma should remain a phrase break: {fixture}"
            );
        }
    }

    #[test]
    fn keeps_place_apposition_commas_as_phrase_breaks_not_vocatives() {
        let normalized = TextNormalizer
            .normalize("Seattle, Washington, a great city")
            .expect("normalize");
        assert_ne!(normalized.boundary_kind, PhraseBoundaryKind::Vocative);
        assert_eq!(
            normalized
                .tokens
                .iter()
                .filter(|token| matches!(token, NormalizedToken::PhraseBreak))
                .count(),
            2
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
    fn folds_latin_diacritics_for_pronounceable_words() {
        let normalized = TextNormalizer
            .normalize("No way, José!")
            .expect("normalize accented Latin text");
        assert_eq!(
            normalized.tokens,
            vec![
                NormalizedToken::Word("no".to_string()),
                NormalizedToken::Word("way".to_string()),
                NormalizedToken::Word("jose".to_string())
            ]
        );
        assert_eq!(normalized.token_spans[2], 8..13);
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
