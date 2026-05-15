use crate::mind::llm::LlmEvent;
use seams::sentence_detector::dialog_detector::SentenceDetectorDialog;
use std::sync::OnceLock;

/// Remove all emoji characters from a string, leaving only non-emoji text.
pub fn strip_emoji(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some((start, end)) = find_first_emoji_sequence(remaining, true) {
        result.push_str(&remaining[..start]);
        remaining = &remaining[end..];
    }
    result.push_str(remaining);
    result
}

/// Returns `true` if `ch` is a common base emoji character.
///
/// This covers the most-used Unicode emoji ranges (emoticons, symbols,
/// pictographs, etc.) but is **not** an exhaustive implementation of the full
/// Unicode emoji specification.
fn is_emoji_char(ch: char) -> bool {
    let cp = ch as u32;
    matches!(
        cp,
        0x00A9
            | 0x00AE
            | 0x2194..=0x2199
            | 0x21A9..=0x21AA
            | 0x231A..=0x231B
            | 0x23E9..=0x23F3
            | 0x23F8..=0x23FA
            | 0x25AA..=0x25AB
            | 0x25B6
            | 0x25C0
            | 0x25FB..=0x25FE
            | 0x2600..=0x27BF
            | 0x2934..=0x2935
            | 0x2B05..=0x2B07
            | 0x2B1B..=0x2B1C
            | 0x2B50
            | 0x2B55
            | 0x3030
            | 0x303D
            | 0x3297
            | 0x3299
            | 0x1F000..=0x1FFFF
    )
}

/// Finds the byte offset range `(start, end)` of the first emoji sequence in `text`.
///
/// An emoji sequence consists of a base emoji character, followed by zero or more
/// modifiers (e.g., skin tones, Variation Selector-16), and optionally chained
/// to other base emojis via Zero Width Joiners (ZWJ).
///
/// If `completed` is `false` and the sequence touches the end of the string,
/// this returns `None` so the caller can wait for more text to ensure the sequence
/// isn't split across chunks.
fn find_first_emoji_sequence(text: &str, completed: bool) -> Option<(usize, usize)> {
    let mut start = None;
    let mut end = 0;
    let mut expect_zwj_target = false;

    for (i, ch) in text.char_indices() {
        let is_mod = matches!(ch as u32, 0x1F3FB..=0x1F3FF) || ch == '\u{FE0F}';
        let is_base = !is_mod && is_emoji_char(ch);
        let is_zwj = ch == '\u{200D}';

        if start.is_none() {
            if is_base {
                start = Some(i);
                end = i + ch.len_utf8();
            }
            continue;
        }

        if expect_zwj_target {
            if is_base {
                end = i + ch.len_utf8();
                expect_zwj_target = false;
            } else {
                break;
            }
        } else {
            if is_mod {
                end = i + ch.len_utf8();
            } else if is_zwj {
                end = i + ch.len_utf8();
                expect_zwj_target = true;
            } else {
                break;
            }
        }
    }

    if let Some(s) = start {
        if !completed && end == text.len() {
            return None;
        }
        Some((s, end))
    } else {
        None
    }
}

/// A face expression command emitted when emoji are detected in LLM output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FaceCommand {
    /// Set Pete's facial expression to the given emoji.
    SetEmoji(String),
    /// Clear any active facial expression.
    Clear,
}

/// A single unit in the expressive output stream.
///
/// The planner emits an ordered sequence of `ExpressiveUnit`s. Speech units are
/// sent to TTS; face units update Pete's displayed countenance inline with
/// speech.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpressiveUnit {
    Speech(SpeechPlan),
    Face(FaceCommand),
}

/// Internal result type for boundary detection.
enum BoundaryResult {
    /// A sentence-ending boundary; the payload is the byte offset just after
    /// the terminating character.
    SentenceEnd(usize),
    /// An emoji character at `(start, end)` byte offsets within the buffer.
    EmojiMarker(usize, usize),
}

const MIN_NON_BACKCHANNEL_CHARS: usize = 12;
const MIN_SHORT_COMPLETE_CHARS: usize = 4;
pub const DEFAULT_SAFE_BACKCHANNELS: &[&str] = &[
    "Ah, okay.",
    "Alright.",
    "Got it.",
    "Got it, one second.",
    "Hang on.",
    "Hmm.",
    "I'm thinking.",
    "Okay.",
    "Okay, let me see.",
    "Okay, one moment.",
    "Okay, yeah.",
    "Right.",
    "Right, yeah.",
    "I see.",
    "Just a second.",
    "Just a moment.",
    "Let me see.",
    "Mm-hm.",
    "Mm.",
    "Let me think.",
    "Well, I dee-clare!",
    "One moment.",
    "One second.",
    "Sure.",
    "Sure, one second.",
    "That makes sense.",
    "Uh.",
    "Um.",
    "Yeah.",
];
const SAFE_DISCOURSE_MARKERS: &[&str] = &["Well,", "Okay,", "Right,", "So,"];
/// Short sentences that are safe to emit immediately even though they fall
/// below the `MIN_NON_BACKCHANNEL_CHARS` length guard.
const SAFE_SHORT_SENTENCES: &[&str] = &[
    "Yes.", "No.", "Yep.", "Nope.", "Sure.", "Okay.", "Right.", "Good.", "Great.", "Hi!",
];
const COMMON_ABBREVIATIONS: &[&str] = &[
    "dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.", "vs.", "etc.", "e.g.", "i.e.", "u.s.",
    "u.k.",
];

/// Persona- and language-specific heuristics used by [`SpeechPlanner`].
///
/// The defaults preserve the original English conversational behavior. Callers
/// can provide a custom config to swap in a different persona or language
/// without changing the planner's boundary detection machinery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechPlannerConfig {
    pub min_non_backchannel_chars: usize,
    pub safe_backchannels: Vec<String>,
    pub safe_discourse_markers: Vec<String>,
    pub safe_short_sentences: Vec<String>,
    pub common_abbreviations: Vec<String>,
}

impl Default for SpeechPlannerConfig {
    fn default() -> Self {
        Self {
            min_non_backchannel_chars: MIN_NON_BACKCHANNEL_CHARS,
            safe_backchannels: strings_from(DEFAULT_SAFE_BACKCHANNELS),
            safe_discourse_markers: strings_from(SAFE_DISCOURSE_MARKERS),
            safe_short_sentences: strings_from(SAFE_SHORT_SENTENCES),
            common_abbreviations: strings_from(COMMON_ABBREVIATIONS),
        }
    }
}

fn strings_from(entries: &[&str]) -> Vec<String> {
    entries.iter().map(|entry| (*entry).to_string()).collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeechUnit {
    Backchannel(String),
    DiscourseMarker(String),
    CompleteClause(String),
    CompleteSentence(String),
    FullTurn(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechPlan {
    unit: SpeechUnit,
}

impl SpeechPlan {
    pub fn new(unit: SpeechUnit) -> Self {
        Self { unit }
    }

    pub fn unit(&self) -> &SpeechUnit {
        &self.unit
    }

    pub fn text(&self) -> &str {
        match &self.unit {
            SpeechUnit::Backchannel(text)
            | SpeechUnit::DiscourseMarker(text)
            | SpeechUnit::CompleteClause(text)
            | SpeechUnit::CompleteSentence(text)
            | SpeechUnit::FullTurn(text) => text,
        }
    }
}

impl From<SpeechUnit> for SpeechPlan {
    fn from(unit: SpeechUnit) -> Self {
        Self::new(unit)
    }
}

#[derive(Debug, Clone)]
pub enum MouthCommand {
    Speak(SpeechPlan),
    FadeOut { millis: u64 },
    StopNow,
}

#[derive(Debug)]
pub struct SpeechPlanner {
    buffer: String,
    config: SpeechPlannerConfig,
    thought_filter: ThoughtMarkupFilter,
}

impl Default for SpeechPlanner {
    fn default() -> Self {
        Self::new(SpeechPlannerConfig::default())
    }
}

impl SpeechPlanner {
    pub fn new(config: SpeechPlannerConfig) -> Self {
        Self {
            buffer: String::new(),
            config,
            thought_filter: ThoughtMarkupFilter::default(),
        }
    }

    pub fn config(&self) -> &SpeechPlannerConfig {
        &self.config
    }

    pub fn ingest(&mut self, events: &[LlmEvent]) -> Vec<ExpressiveUnit> {
        let mut completed = false;
        for event in events {
            match event {
                LlmEvent::Token { text } => self.buffer.push_str(&self.thought_filter.push(text)),
                LlmEvent::Completed => {
                    self.buffer.push_str(&self.thought_filter.finish());
                    completed = true;
                }
                LlmEvent::Cancelled | LlmEvent::Error { .. } => {
                    self.buffer.clear();
                    self.thought_filter.reset();
                    return Vec::new();
                }
            }
        }

        self.emit_ready(completed)
    }

    fn emit_ready(&mut self, completed: bool) -> Vec<ExpressiveUnit> {
        let mut units = Vec::new();
        loop {
            match self.next_boundary(completed) {
                None => break,
                Some(BoundaryResult::SentenceEnd(end)) => {
                    let candidate = self.buffer[..end].trim().to_string();
                    if candidate.is_empty() {
                        self.buffer.drain(..end);
                        continue;
                    }
                    if let Some(unit) = classify_boundary_unit(&candidate, &self.config) {
                        units.push(ExpressiveUnit::Speech(unit.into()));
                        self.buffer.drain(..end);
                    } else if self.buffer[end..].chars().any(is_emoji_char) {
                        // Emoji follows in the remaining buffer (linear scan over
                        // a typically short conversational fragment, acceptable
                        // here).  Flush the sentence-punctuated text with an
                        // appropriate classification rather than waiting for more
                        // tokens, so face and speech events stay in order.
                        let unit = classify_text_before_emoji(&candidate, &self.config);
                        units.push(ExpressiveUnit::Speech(unit.into()));
                        self.buffer.drain(..end);
                    } else if let Some(next_rel) =
                        find_sentence_end(&self.buffer[end..], &self.config)
                    {
                        // The current boundary produced an unclassifiable short text.
                        // There is a further sentence boundary in the buffer, so merge
                        // the short prefix into the next chunk rather than blocking it.
                        let merged_end = end + next_rel;
                        let merged = self.buffer[..merged_end].trim().to_string();
                        if let Some(unit) = classify_boundary_unit(&merged, &self.config) {
                            units.push(ExpressiveUnit::Speech(unit.into()));
                            self.buffer.drain(..merged_end);
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                Some(BoundaryResult::EmojiMarker(start, end)) => {
                    let before = self.buffer[..start].trim().to_string();
                    if !before.is_empty() {
                        let unit = classify_text_before_emoji(&before, &self.config);
                        units.push(ExpressiveUnit::Speech(unit.into()));
                    }
                    let emoji = self.buffer[start..end].to_string();
                    units.push(ExpressiveUnit::Face(FaceCommand::SetEmoji(emoji)));
                    self.buffer.drain(..end);
                }
            }
        }

        if completed {
            let trailing = self.buffer.trim().to_string();
            if let Some(unit) = classify_completed_unit(&trailing, &self.config) {
                units.push(ExpressiveUnit::Speech(unit.into()));
            }
            self.buffer.clear();
        }

        units
    }

    fn next_boundary(&self, completed: bool) -> Option<BoundaryResult> {
        let emoji_boundary = find_first_emoji_sequence(&self.buffer, completed);
        let text_boundary = self.next_text_boundary(completed);

        match (emoji_boundary, text_boundary) {
            (Some((start, end)), Some(sentence_end)) if start < sentence_end => {
                Some(BoundaryResult::EmojiMarker(start, end))
            }
            (Some(_), Some(sentence_end)) => Some(BoundaryResult::SentenceEnd(sentence_end)),
            (Some((start, end)), None) => Some(BoundaryResult::EmojiMarker(start, end)),
            (None, Some(sentence_end)) => Some(BoundaryResult::SentenceEnd(sentence_end)),
            (None, None) => None,
        }
    }
}

#[derive(Debug, Default)]
struct ThoughtMarkupFilter {
    pending: String,
    hidden_until: Option<&'static str>,
}

impl ThoughtMarkupFilter {
    fn push(&mut self, text: &str) -> String {
        self.pending.push_str(text);
        self.drain(false)
    }

    fn finish(&mut self) -> String {
        self.drain(true)
    }

    fn reset(&mut self) {
        self.pending.clear();
        self.hidden_until = None;
    }

    fn drain(&mut self, completed: bool) -> String {
        let mut visible = String::new();
        loop {
            if let Some(close_tag) = self.hidden_until {
                if let Some(end) = find_ascii_case_insensitive(&self.pending, close_tag) {
                    self.pending.drain(..end + close_tag.len());
                    self.hidden_until = None;
                    continue;
                }
                if completed {
                    self.pending.clear();
                    self.hidden_until = None;
                } else {
                    keep_possible_tag_prefix(&mut self.pending, &[close_tag]);
                }
                break;
            }

            let Some((start, open_tag, close_tag)) = first_thought_open_tag(&self.pending) else {
                if completed {
                    visible.push_str(&self.pending);
                    self.pending.clear();
                } else {
                    let keep_from = possible_tag_prefix_start(&self.pending, THOUGHT_OPEN_TAGS);
                    visible.push_str(&self.pending[..keep_from]);
                    self.pending.drain(..keep_from);
                }
                break;
            };

            visible.push_str(&self.pending[..start]);
            self.pending.drain(..start + open_tag.len());
            self.hidden_until = Some(close_tag);
        }
        visible
    }
}

const THOUGHT_OPEN_TAGS: &[&str] = &["<think>", "<thinking>", "<thought>"];
const THOUGHT_TAG_PAIRS: &[(&str, &str)] = &[
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<thought>", "</thought>"),
];

fn first_thought_open_tag(text: &str) -> Option<(usize, &'static str, &'static str)> {
    THOUGHT_TAG_PAIRS
        .iter()
        .filter_map(|(open, close)| {
            find_ascii_case_insensitive(text, open).map(|index| (index, *open, *close))
        })
        .min_by_key(|(index, _, _)| *index)
}

fn find_ascii_case_insensitive(text: &str, needle: &str) -> Option<usize> {
    text.to_ascii_lowercase().find(needle)
}

fn keep_possible_tag_prefix(text: &mut String, tags: &[&str]) {
    let keep_from = possible_tag_prefix_start(text, tags);
    text.drain(..keep_from);
}

fn possible_tag_prefix_start(text: &str, tags: &[&str]) -> usize {
    (0..text.len())
        .find(|&index| {
            text.is_char_boundary(index)
                && tags.iter().any(|tag| {
                    let suffix = &text[index..];
                    !suffix.is_empty()
                        && suffix.len() < tag.len()
                        && tag.starts_with(&suffix.to_ascii_lowercase())
                })
        })
        .unwrap_or(text.len())
}

fn sentence_detector() -> Option<&'static SentenceDetectorDialog> {
    static DETECTOR: OnceLock<Option<SentenceDetectorDialog>> = OnceLock::new();
    DETECTOR
        .get_or_init(|| SentenceDetectorDialog::new().ok())
        .as_ref()
}

impl SpeechPlanner {
    fn next_text_boundary(&self, completed: bool) -> Option<usize> {
        let sentence = self.next_sentence_boundary();
        let clause = self.next_clause_boundary();
        let discourse = self.next_discourse_marker_boundary(completed);
        [sentence, clause, discourse].into_iter().flatten().min()
    }

    fn next_sentence_boundary(&self) -> Option<usize> {
        find_sentence_end(&self.buffer, &self.config)
    }

    fn next_clause_boundary(&self) -> Option<usize> {
        for (index, ch) in self.buffer.char_indices() {
            let end = index + ch.len_utf8();
            let is_end = end == self.buffer.len();
            let next_is_whitespace = self.buffer[end..]
                .chars()
                .next()
                .is_some_and(char::is_whitespace);
            if !(next_is_whitespace || is_end) {
                continue;
            }
            if matches!(ch, ';' | ':') {
                return Some(end);
            }
        }
        None
    }

    fn next_discourse_marker_boundary(&self, completed: bool) -> Option<usize> {
        for (index, ch) in self.buffer.char_indices() {
            if ch != ',' {
                continue;
            }
            let end = index + ch.len_utf8();
            let is_end = end == self.buffer.len();
            if is_end && !completed {
                continue;
            }
            let next_is_whitespace = self.buffer[end..]
                .chars()
                .next()
                .is_some_and(char::is_whitespace);
            if !(next_is_whitespace || is_end) {
                continue;
            }
            let candidate = self.buffer[..end].trim();
            if is_safe_discourse_marker(candidate, &self.config) {
                return Some(end);
            }
        }
        None
    }
}

fn classify_boundary_unit(text: &str, config: &SpeechPlannerConfig) -> Option<SpeechUnit> {
    if text.is_empty() {
        return None;
    }
    if is_safe_backchannel(text, config) || is_safe_short_sentence(text, config) {
        return Some(SpeechUnit::Backchannel(text.to_string()));
    }
    if is_safe_discourse_marker(text, config) {
        return Some(SpeechUnit::DiscourseMarker(text.to_string()));
    }
    if ends_with_sentence_punctuation(text) {
        if text.len() < config.min_non_backchannel_chars
            && meaningful_char_count(text) < MIN_SHORT_COMPLETE_CHARS
        {
            return None;
        }
        return Some(SpeechUnit::CompleteSentence(text.to_string()));
    }
    if text.len() < config.min_non_backchannel_chars {
        return None;
    }
    if text.ends_with([';', ':']) {
        return Some(SpeechUnit::CompleteClause(text.to_string()));
    }
    None
}

fn meaningful_char_count(text: &str) -> usize {
    text.chars()
        .filter(|ch| !ch.is_whitespace() && !matches!(ch, '.' | '?' | '!'))
        .count()
}

fn ends_with_sentence_punctuation(text: &str) -> bool {
    text.trim_end_matches(['"', '\'', '”', '’'])
        .ends_with(['.', '?', '!'])
}

fn classify_completed_unit(text: &str, config: &SpeechPlannerConfig) -> Option<SpeechUnit> {
    classify_boundary_unit(text, config)
}

/// Classify text that is being force-flushed because an emoji follows it.
///
/// Unlike [`classify_boundary_unit`], this bypasses the minimum-length guard so
/// that short but grammatically complete phrases (e.g. "That works.") are
/// emitted with the right type rather than falling back to `FullTurn`.
fn classify_text_before_emoji(text: &str, config: &SpeechPlannerConfig) -> SpeechUnit {
    if text.is_empty() {
        return SpeechUnit::FullTurn(text.to_string());
    }
    if is_safe_backchannel(text, config) {
        return SpeechUnit::Backchannel(text.to_string());
    }
    if is_safe_discourse_marker(text, config) {
        return SpeechUnit::DiscourseMarker(text.to_string());
    }
    if ends_with_sentence_punctuation(text) {
        return SpeechUnit::CompleteSentence(text.to_string());
    }
    if text.ends_with([';', ':']) {
        return SpeechUnit::CompleteClause(text.to_string());
    }
    SpeechUnit::FullTurn(text.to_string())
}

fn is_safe_backchannel(text: &str, config: &SpeechPlannerConfig) -> bool {
    config.safe_backchannels.iter().any(|entry| entry == text)
}

fn is_safe_short_sentence(text: &str, config: &SpeechPlannerConfig) -> bool {
    config
        .safe_short_sentences
        .iter()
        .any(|entry| entry == text)
}

fn is_safe_discourse_marker(text: &str, config: &SpeechPlannerConfig) -> bool {
    config
        .safe_discourse_markers
        .iter()
        .any(|entry| entry == text)
}

fn is_common_abbreviation(text: &str, config: &SpeechPlannerConfig) -> bool {
    let lowercase = text.trim().to_ascii_lowercase();
    config
        .common_abbreviations
        .iter()
        .any(|abbreviation| lowercase.ends_with(abbreviation))
}

/// Find the byte offset just after the first complete sentence in `text`.
///
/// Uses a simple punctuation scan first for the streaming hot path, then falls
/// back to the seams dialog detector for cases the deterministic scan misses.
fn find_sentence_end(text: &str, config: &SpeechPlannerConfig) -> Option<usize> {
    if let Some(end) = punctuation_sentence_end(text, config) {
        return Some(end);
    }
    if let Some(detector) = sentence_detector() {
        if let Ok(sentences) = detector.detect_sentences_borrowed(text) {
            let mut search_from = 0;
            for sentence in sentences {
                if let Some(rel) = text[search_from..].find(sentence.raw_content) {
                    let start = search_from + rel;
                    let end = start + sentence.raw_content.len();
                    search_from = end;
                    if sentence.raw_content.trim().ends_with(['.', '?', '!']) {
                        return Some(end);
                    }
                }
            }
        }
    }
    None
}

/// Deterministic punctuation-based sentence boundary scan used as a fallback
/// when the seams detector is unavailable.  Uses the abbreviation guard to
/// avoid splitting on `Dr.`, `Mr.`, etc.
fn punctuation_sentence_end(text: &str, config: &SpeechPlannerConfig) -> Option<usize> {
    for (index, ch) in text.char_indices() {
        let punctuation_end = index + ch.len_utf8();
        let end = closing_quote_end(text, punctuation_end);
        let is_end = end == text.len();
        let next_is_whitespace = text[end..].chars().next().is_some_and(char::is_whitespace);
        if !(next_is_whitespace || is_end) {
            continue;
        }
        match ch {
            '?' | '!' => return Some(end),
            '.' if !is_common_abbreviation(text[..punctuation_end].trim(), config) => {
                return Some(end);
            }
            _ => {}
        }
    }
    None
}

fn closing_quote_end(text: &str, start: usize) -> usize {
    let mut end = start;
    for ch in text[start..].chars() {
        if matches!(ch, '"' | '\'' | '”' | '’') {
            end += ch.len_utf8();
        } else {
            break;
        }
    }
    end
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(text: &str) -> LlmEvent {
        LlmEvent::Token {
            text: text.to_string(),
        }
    }

    fn speech(unit: SpeechUnit) -> ExpressiveUnit {
        ExpressiveUnit::Speech(SpeechPlan::from(unit))
    }

    fn face(emoji: &str) -> ExpressiveUnit {
        ExpressiveUnit::Face(FaceCommand::SetEmoji(emoji.to_string()))
    }

    #[test]
    fn partial_fragment_emits_nothing() {
        let mut planner = SpeechPlanner::default();
        let units = planner.ingest(&[token("I think that")]);
        assert!(units.is_empty());
    }

    #[test]
    fn complete_sentence_emits_unit() {
        let mut planner = SpeechPlanner::default();
        let units = planner.ingest(&[token("I think that works.")]);
        assert_eq!(
            units,
            vec![speech(SpeechUnit::CompleteSentence(
                "I think that works.".to_string()
            ))]
        );
    }

    #[test]
    fn quoted_sentence_emits_at_closing_quote() {
        let mut planner = SpeechPlanner::default();
        let units = planner.ingest(&[token("\"This is spoken.\" Yes, it is.")]);
        assert_eq!(
            units,
            vec![
                speech(SpeechUnit::CompleteSentence(
                    "\"This is spoken.\"".to_string()
                )),
                speech(SpeechUnit::CompleteSentence("Yes, it is.".to_string()))
            ]
        );
    }

    #[test]
    fn thought_tags_are_not_spoken() {
        let mut planner = SpeechPlanner::default();
        let units = planner.ingest(&[token(
            "<thought>this should be a thought</thought> <thinking>Or is it thinking</thinking> Yes, I can hear you.",
        )]);
        assert_eq!(
            units,
            vec![speech(SpeechUnit::CompleteSentence(
                "Yes, I can hear you.".to_string()
            ))]
        );
    }

    #[test]
    fn split_thought_tags_are_not_spoken() {
        let mut planner = SpeechPlanner::default();
        assert!(planner.ingest(&[token("<th")]).is_empty());
        assert!(planner.ingest(&[token("inking>hidden")]).is_empty());
        let units = planner.ingest(&[token("</thinking> Good.")]);
        assert_eq!(
            units,
            vec![speech(SpeechUnit::Backchannel("Good.".to_string()))]
        );
    }

    #[test]
    fn safe_backchannel_emits_early() {
        let mut planner = SpeechPlanner::default();
        let units = planner.ingest(&[token("Okay.")]);
        assert_eq!(
            units,
            vec![speech(SpeechUnit::Backchannel("Okay.".to_string()))]
        );
    }

    #[test]
    fn short_complete_name_answer_emits_early() {
        let mut planner = SpeechPlanner::default();
        let units = planner.ingest(&[token("Pete. I can")]);
        assert_eq!(
            units,
            vec![speech(SpeechUnit::CompleteSentence("Pete.".to_string()))]
        );
    }

    #[test]
    fn custom_config_controls_safe_backchannels() {
        let config = SpeechPlannerConfig {
            safe_backchannels: vec!["Indeed.".to_string()],
            safe_short_sentences: Vec::new(),
            ..SpeechPlannerConfig::default()
        };
        let mut planner = SpeechPlanner::new(config);

        let units = planner.ingest(&[token("Indeed.")]);
        assert_eq!(
            units,
            vec![speech(SpeechUnit::Backchannel("Indeed.".to_string()))]
        );
    }

    #[test]
    fn custom_config_can_make_default_short_sentence_wait() {
        let config = SpeechPlannerConfig {
            safe_backchannels: Vec::new(),
            safe_short_sentences: Vec::new(),
            ..SpeechPlannerConfig::default()
        };
        let mut planner = SpeechPlanner::new(config);

        let units = planner.ingest(&[token("Yep. I think that works.")]);
        assert_eq!(
            units,
            vec![speech(SpeechUnit::CompleteSentence(
                "Yep. I think that works.".to_string()
            ))]
        );
    }

    #[test]
    fn comma_fragment_without_allowlist_emits_nothing() {
        let mut planner = SpeechPlanner::default();
        let units = planner.ingest(&[token("Not exactly,")]);
        assert!(units.is_empty());
    }

    #[test]
    fn comma_clause_emits_when_sentence_completes() {
        let mut planner = SpeechPlanner::default();
        let units = planner.ingest(&[token("Not exactly, there is a catch.")]);
        assert_eq!(
            units,
            vec![speech(SpeechUnit::CompleteSentence(
                "Not exactly, there is a catch.".to_string()
            ))]
        );
    }

    #[test]
    fn custom_config_controls_discourse_markers() {
        let config = SpeechPlannerConfig {
            safe_discourse_markers: vec!["Pues,".to_string()],
            ..SpeechPlannerConfig::default()
        };
        let mut planner = SpeechPlanner::new(config);

        let units = planner.ingest(&[token("Pues, seguimos.")]);
        assert_eq!(
            units,
            vec![
                speech(SpeechUnit::DiscourseMarker("Pues,".to_string())),
                speech(SpeechUnit::CompleteSentence("seguimos.".to_string())),
            ]
        );
    }

    #[test]
    fn planner_does_not_split_common_abbreviation() {
        let mut planner = SpeechPlanner::default();
        let units = planner.ingest(&[token("Dr. Smith arrived.")]);
        assert_eq!(
            units,
            vec![speech(SpeechUnit::CompleteSentence(
                "Dr. Smith arrived.".to_string()
            ))]
        );
    }

    #[test]
    fn custom_config_controls_punctuation_abbreviations() {
        let config = SpeechPlannerConfig {
            common_abbreviations: vec!["sra.".to_string()],
            ..SpeechPlannerConfig::default()
        };

        assert_eq!(
            punctuation_sentence_end("Sra. Garcia llego.", &config),
            Some(18)
        );
    }

    // --- Emoji tests ---

    #[test]
    fn emoji_at_start_emits_face_then_speech() {
        let mut planner = SpeechPlanner::default();
        let units = planner.ingest(&[token("🙂 Okay.")]);
        assert_eq!(
            units,
            vec![
                face("🙂"),
                speech(SpeechUnit::Backchannel("Okay.".to_string())),
            ]
        );
    }

    #[test]
    fn emoji_in_middle_splits_speech() {
        let mut planner = SpeechPlanner::default();
        let units = planner.ingest(&[token("Okay 🙂 I see.")]);
        assert_eq!(
            units,
            vec![
                speech(SpeechUnit::FullTurn("Okay".to_string())),
                face("🙂"),
                speech(SpeechUnit::Backchannel("I see.".to_string())),
            ]
        );
    }

    #[test]
    fn emoji_at_end_follows_speech() {
        let mut planner = SpeechPlanner::default();
        let units = planner.ingest(&[token("That works. 😄")]);
        assert_eq!(
            units,
            vec![speech(SpeechUnit::CompleteSentence(
                "That works.".to_string()
            ))]
        );

        let units = planner.ingest(&[LlmEvent::Completed]);
        assert_eq!(units, vec![face("😄")]);
    }

    #[test]
    fn text_without_emoji_unaffected() {
        let mut planner = SpeechPlanner::default();
        let units = planner.ingest(&[token("I think that works.")]);
        assert_eq!(
            units,
            vec![speech(SpeechUnit::CompleteSentence(
                "I think that works.".to_string()
            ))]
        );
    }

    #[test]
    fn strip_emoji_removes_emoji_only() {
        assert_eq!(strip_emoji("Hello 🙂 world"), "Hello  world");
        assert_eq!(strip_emoji("No emoji here."), "No emoji here.");
        assert_eq!(strip_emoji("😄"), "");
    }

    #[test]
    fn emits_complete_sentence_before_completed_event() {
        let mut planner = SpeechPlanner::default();
        let first = planner.ingest(&[token("I think that")]);
        assert!(first.is_empty());

        let second = planner.ingest(&[token(" works. The next")]);
        assert_eq!(
            second,
            vec![speech(SpeechUnit::CompleteSentence(
                "I think that works.".to_string()
            ))]
        );

        let third = planner.ingest(&[token(" thing is timing.")]);
        assert_eq!(
            third,
            vec![speech(SpeechUnit::CompleteSentence(
                "The next thing is timing.".to_string()
            ))]
        );
    }

    #[test]
    fn emits_multiple_complete_units_from_one_batch() {
        let mut planner = SpeechPlanner::default();
        let units = planner.ingest(&[token("First sentence. Second sentence.")]);
        assert_eq!(
            units,
            vec![
                speech(SpeechUnit::CompleteSentence("First sentence.".to_string())),
                speech(SpeechUnit::CompleteSentence("Second sentence.".to_string()))
            ]
        );
    }

    #[test]
    fn cancelled_event_clears_buffered_fragment() {
        let mut planner = SpeechPlanner::default();
        assert!(planner.ingest(&[token("I think this")]).is_empty());
        assert!(planner.ingest(&[LlmEvent::Cancelled]).is_empty());
        assert!(planner.buffer.is_empty());
        let units = planner.ingest(&[token(" this definitely works now.")]);
        assert_eq!(
            units,
            vec![speech(SpeechUnit::CompleteSentence(
                "this definitely works now.".to_string()
            ))]
        );
    }

    #[test]
    fn short_allowlisted_sentence_emits_without_completed() {
        let mut planner = SpeechPlanner::default();
        assert_eq!(
            planner.ingest(&[token("Yes. I think")]),
            vec![speech(SpeechUnit::Backchannel("Yes.".to_string()))]
        );
    }

    #[test]
    fn short_sentence_does_not_block_later_sentence() {
        let mut planner = SpeechPlanner::default();
        let units = planner.ingest(&[token("Yes. I think that works.")]);
        assert_eq!(
            units,
            vec![
                speech(SpeechUnit::Backchannel("Yes.".to_string())),
                speech(SpeechUnit::CompleteSentence(
                    "I think that works.".to_string()
                )),
            ]
        );
    }

    #[test]
    fn unknown_short_prefix_merges_with_next_safe_sentence() {
        let mut planner = SpeechPlanner::default();
        let units = planner.ingest(&[token("Hm. I think that works.")]);
        assert_eq!(
            units,
            vec![speech(SpeechUnit::CompleteSentence(
                "Hm. I think that works.".to_string()
            ))]
        );
    }

    #[test]
    fn compound_emoji_not_split() {
        let mut planner = SpeechPlanner::default();
        // 👨‍👩‍👧‍👦 Family
        let units = planner.ingest(&[
            token("👨"),
            token("\u{200D}👩"),
            token("\u{200D}👧"),
            token("\u{200D}👦 Okay."),
        ]);
        assert_eq!(
            units,
            vec![
                face("👨‍👩‍👧‍👦"),
                speech(SpeechUnit::Backchannel("Okay.".to_string())),
            ]
        );
    }

    #[test]
    fn emoji_with_skin_tone_not_split() {
        let mut planner = SpeechPlanner::default();
        // 👋🏽 Waving hand + medium skin tone
        let units = planner.ingest(&[token("👋"), token("🏽"), token(" Hi!")]);
        assert_eq!(
            units,
            vec![
                face("👋🏽"),
                speech(SpeechUnit::Backchannel("Hi!".to_string())),
            ]
        );
    }

    #[test]
    fn strip_emoji_removes_compound_emojis() {
        assert_eq!(strip_emoji("Hello 👨‍👩‍👧‍👦 world"), "Hello  world");
        assert_eq!(strip_emoji("Hi 👋🏽 there"), "Hi  there");
    }
}
