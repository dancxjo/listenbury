use crate::mind::llm::LlmEvent;
use seams::sentence_detector::dialog_detector::SentenceDetectorDialog;
use std::sync::OnceLock;

/// Remove all emoji characters from a string, leaving only non-emoji text.
pub fn strip_emoji(text: &str) -> String {
    text.chars().filter(|&ch| !is_emoji_char(ch)).collect()
}

/// Returns `true` if `ch` is a common emoji character.
///
/// This covers the most-used Unicode emoji ranges (emoticons, symbols,
/// pictographs, etc.) but is **not** an exhaustive implementation of the full
/// Unicode emoji specification.  Sequences such as ZWJ chains or skin-tone
/// modifiers are handled character-by-character; isolated modifier codepoints
/// may not be detected.  This is intentionally lightweight—sufficient for
/// stripping conversational emoji from LLM output.
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
pub const DEFAULT_SAFE_BACKCHANNELS: &[&str] = &[
    "Okay.",
    "Right.",
    "I see.",
    "Mm-hm.",
    "Let me think.",
    "Well, boy howdy! I declare.",
    "That makes sense.",
];
const SAFE_DISCOURSE_MARKERS: &[&str] = &["Well,", "Okay,", "Right,", "So,"];
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

#[derive(Debug, Default)]
pub struct SpeechPlanner {
    buffer: String,
}

impl SpeechPlanner {
    pub fn ingest(&mut self, events: &[LlmEvent]) -> Vec<ExpressiveUnit> {
        let mut completed = false;
        for event in events {
            match event {
                LlmEvent::Token { text } => self.buffer.push_str(text),
                LlmEvent::Completed => {
                    completed = true;
                }
                LlmEvent::Cancelled | LlmEvent::Error { .. } => {
                    self.buffer.clear();
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
                    if let Some(unit) = classify_boundary_unit(&candidate) {
                        units.push(ExpressiveUnit::Speech(unit.into()));
                        self.buffer.drain(..end);
                    } else if self.buffer[end..].chars().any(is_emoji_char) {
                        // Emoji follows in the remaining buffer (linear scan over
                        // a typically short conversational fragment, acceptable
                        // here).  Flush the sentence-punctuated text with an
                        // appropriate classification rather than waiting for more
                        // tokens, so face and speech events stay in order.
                        let unit = classify_text_before_emoji(&candidate);
                        units.push(ExpressiveUnit::Speech(unit.into()));
                        self.buffer.drain(..end);
                    } else {
                        break;
                    }
                }
                Some(BoundaryResult::EmojiMarker(start, end)) => {
                    let before = self.buffer[..start].trim().to_string();
                    if !before.is_empty() {
                        let unit = classify_text_before_emoji(&before);
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
            if let Some(unit) = classify_completed_unit(&trailing) {
                units.push(ExpressiveUnit::Speech(unit.into()));
            }
            self.buffer.clear();
        }

        units
    }

    fn next_boundary(&self, completed: bool) -> Option<BoundaryResult> {
        let emoji_boundary = self
            .buffer
            .char_indices()
            .find_map(|(index, ch)| is_emoji_char(ch).then_some((index, index + ch.len_utf8())));
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
        let detector = sentence_detector()?;
        let sentences = detector.detect_sentences_borrowed(&self.buffer).ok()?;
        let mut search_from = 0;

        for sentence in sentences {
            let relative = self.buffer[search_from..].find(sentence.raw_content)?;
            let start = search_from + relative;
            let end = start + sentence.raw_content.len();
            search_from = end;

            let text = sentence.raw_content.trim();
            if !text.ends_with(['.', '?', '!']) {
                continue;
            }
            return Some(end);
        }

        None
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
            if is_safe_discourse_marker(candidate) {
                return Some(end);
            }
        }
        None
    }
}

fn classify_boundary_unit(text: &str) -> Option<SpeechUnit> {
    if text.is_empty() {
        return None;
    }
    if is_safe_backchannel(text) {
        return Some(SpeechUnit::Backchannel(text.to_string()));
    }
    if is_safe_discourse_marker(text) {
        return Some(SpeechUnit::DiscourseMarker(text.to_string()));
    }
    if text.len() < MIN_NON_BACKCHANNEL_CHARS {
        return None;
    }
    if text.ends_with(['.', '?', '!']) {
        return Some(SpeechUnit::CompleteSentence(text.to_string()));
    }
    if text.ends_with([';', ':']) {
        return Some(SpeechUnit::CompleteClause(text.to_string()));
    }
    None
}

fn classify_completed_unit(text: &str) -> Option<SpeechUnit> {
    classify_boundary_unit(text)
}

/// Classify text that is being force-flushed because an emoji follows it.
///
/// Unlike [`classify_boundary_unit`], this bypasses the minimum-length guard so
/// that short but grammatically complete phrases (e.g. "That works.") are
/// emitted with the right type rather than falling back to `FullTurn`.
fn classify_text_before_emoji(text: &str) -> SpeechUnit {
    if text.is_empty() {
        return SpeechUnit::FullTurn(text.to_string());
    }
    if is_safe_backchannel(text) {
        return SpeechUnit::Backchannel(text.to_string());
    }
    if is_safe_discourse_marker(text) {
        return SpeechUnit::DiscourseMarker(text.to_string());
    }
    if text.ends_with(['.', '?', '!']) {
        return SpeechUnit::CompleteSentence(text.to_string());
    }
    if text.ends_with([';', ':']) {
        return SpeechUnit::CompleteClause(text.to_string());
    }
    SpeechUnit::FullTurn(text.to_string())
}

fn is_safe_backchannel(text: &str) -> bool {
    DEFAULT_SAFE_BACKCHANNELS.iter().any(|entry| *entry == text)
}

fn is_safe_discourse_marker(text: &str) -> bool {
    SAFE_DISCOURSE_MARKERS.iter().any(|entry| *entry == text)
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
    fn safe_backchannel_emits_early() {
        let mut planner = SpeechPlanner::default();
        let units = planner.ingest(&[token("Okay.")]);
        assert_eq!(
            units,
            vec![speech(SpeechUnit::Backchannel("Okay.".to_string()))]
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
            vec![
                speech(SpeechUnit::CompleteSentence("That works.".to_string())),
                face("😄"),
            ]
        );
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
}
