use crate::mind::llm::LlmEvent;

const MIN_NON_BACKCHANNEL_CHARS: usize = 12;
const SAFE_BACKCHANNELS: &[&str] = &[
    "Okay.",
    "Right.",
    "I see.",
    "Mm-hm.",
    "Let me think.",
    "One thing jumps out.",
    "That makes sense.",
];
const SAFE_DISCOURSE_MARKERS: &[&str] = &["Well,", "Okay,", "Right,", "So,"];
const COMMON_ABBREVIATIONS: &[&str] = &[
    "dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.", "vs.", "etc.", "e.g.", "i.e.", "u.s.",
    "u.k.",
];

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
    pub fn ingest(&mut self, events: &[LlmEvent]) -> Vec<SpeechPlan> {
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

    fn emit_ready(&mut self, completed: bool) -> Vec<SpeechPlan> {
        let mut plans = Vec::new();
        while let Some(boundary) = self.next_boundary(completed) {
            let candidate = self.buffer[..boundary].trim();
            if candidate.is_empty() {
                self.buffer.drain(..boundary);
                continue;
            }

            let Some(unit) = classify_boundary_unit(candidate) else {
                break;
            };
            plans.push(unit.into());
            self.buffer.drain(..boundary);
        }

        if completed {
            let trailing = self.buffer.trim();
            if let Some(unit) = classify_completed_unit(trailing) {
                plans.push(unit.into());
            }
            self.buffer.clear();
        }

        plans
    }

    fn next_boundary(&self, completed: bool) -> Option<usize> {
        for (index, ch) in self.buffer.char_indices() {
            let boundary = index + ch.len_utf8();
            let is_end = boundary == self.buffer.len();
            let next_is_whitespace = self.buffer[boundary..]
                .chars()
                .next()
                .is_some_and(char::is_whitespace);
            if !(next_is_whitespace || is_end) {
                continue;
            }
            if is_end && !completed && ch == ',' {
                continue;
            }

            match ch {
                '.' => {
                    let candidate = self.buffer[..boundary].trim();
                    if is_common_abbreviation(candidate) {
                        continue;
                    }
                    return Some(boundary);
                }
                '?' | '!' | ';' | ':' => return Some(boundary),
                ',' => {
                    let candidate = self.buffer[..boundary].trim();
                    if is_safe_discourse_marker(candidate) {
                        return Some(boundary);
                    }
                }
                _ => {}
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
    if text.chars().count() < MIN_NON_BACKCHANNEL_CHARS {
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

fn is_safe_backchannel(text: &str) -> bool {
    SAFE_BACKCHANNELS.iter().any(|entry| *entry == text)
}

fn is_safe_discourse_marker(text: &str) -> bool {
    SAFE_DISCOURSE_MARKERS.iter().any(|entry| *entry == text)
}

fn is_common_abbreviation(text: &str) -> bool {
    let lowercase = text.trim().to_ascii_lowercase();
    COMMON_ABBREVIATIONS
        .iter()
        .any(|abbreviation| lowercase.ends_with(abbreviation))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(text: &str) -> LlmEvent {
        LlmEvent::Token {
            text: text.to_string(),
        }
    }

    #[test]
    fn partial_fragment_emits_nothing() {
        let mut planner = SpeechPlanner::default();
        let plans = planner.ingest(&[token("I think that")]);
        assert!(plans.is_empty());
    }

    #[test]
    fn complete_sentence_emits_unit() {
        let mut planner = SpeechPlanner::default();
        let plans = planner.ingest(&[token("I think that works.")]);
        assert_eq!(
            plans,
            vec![SpeechPlan::from(SpeechUnit::CompleteSentence(
                "I think that works.".to_string()
            ))]
        );
    }

    #[test]
    fn safe_backchannel_emits_early() {
        let mut planner = SpeechPlanner::default();
        let plans = planner.ingest(&[token("Okay.")]);
        assert_eq!(
            plans,
            vec![SpeechPlan::from(SpeechUnit::Backchannel(
                "Okay.".to_string()
            ))]
        );
    }

    #[test]
    fn comma_fragment_without_allowlist_emits_nothing() {
        let mut planner = SpeechPlanner::default();
        let plans = planner.ingest(&[token("Not exactly,")]);
        assert!(plans.is_empty());
    }

    #[test]
    fn comma_clause_emits_when_sentence_completes() {
        let mut planner = SpeechPlanner::default();
        let plans = planner.ingest(&[token("Not exactly, there is a catch.")]);
        assert_eq!(
            plans,
            vec![SpeechPlan::from(SpeechUnit::CompleteSentence(
                "Not exactly, there is a catch.".to_string()
            ))]
        );
    }

    #[test]
    fn planner_does_not_split_common_abbreviation() {
        let mut planner = SpeechPlanner::default();
        let plans = planner.ingest(&[token("Dr. Smith arrived.")]);
        assert_eq!(
            plans,
            vec![SpeechPlan::from(SpeechUnit::CompleteSentence(
                "Dr. Smith arrived.".to_string()
            ))]
        );
    }
}
