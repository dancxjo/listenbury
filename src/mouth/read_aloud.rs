use crate::text_stability::stable_prefix_len;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpeechCandidateId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechCandidateCommitment {
    Speculative,
    Prepared,
    Playable,
    Committed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadAloudCandidate {
    pub id: SpeechCandidateId,
    pub text: String,
    pub stable_prefix_len: usize,
    pub commitment: SpeechCandidateCommitment,
    pub safe_to_prepare_len: usize,
    pub safe_to_play_len: usize,
}

impl ReadAloudCandidate {
    pub fn mark_prepared(&mut self) {
        if !matches!(self.commitment, SpeechCandidateCommitment::Cancelled) {
            self.commitment = SpeechCandidateCommitment::Prepared;
        }
    }

    pub fn mark_playable(&mut self) {
        if !matches!(self.commitment, SpeechCandidateCommitment::Cancelled) {
            self.commitment = SpeechCandidateCommitment::Playable;
        }
    }

    pub fn mark_committed(&mut self) {
        if !matches!(self.commitment, SpeechCandidateCommitment::Cancelled) {
            self.commitment = SpeechCandidateCommitment::Committed;
        }
    }

    pub fn cancel(&mut self) {
        self.commitment = SpeechCandidateCommitment::Cancelled;
    }
}

pub trait ReadAloudAudioPreparer {
    type PreparedAudio;

    fn prepare(&mut self, candidate: &ReadAloudCandidate) -> Self::PreparedAudio;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadAloudCandidateEvent {
    CandidateStarted { id: SpeechCandidateId },
    CandidateUpdated { candidate: ReadAloudCandidate },
    CandidateReplaced {
        old: SpeechCandidateId,
        new: SpeechCandidateId,
        stable_prefix_len: usize,
    },
    CandidateCancelled { id: SpeechCandidateId },
}

#[derive(Debug, Default)]
pub struct ReadAloudCandidateTracker {
    next_id: u64,
    active: Option<ReadAloudCandidate>,
}

impl ReadAloudCandidateTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn active(&self) -> Option<&ReadAloudCandidate> {
        self.active.as_ref()
    }

    pub fn ingest_text(&mut self, text: impl Into<String>) -> Vec<ReadAloudCandidateEvent> {
        let text = text.into();
        if text.trim().is_empty() {
            return Vec::new();
        }

        let mut events = Vec::new();
        let (id, stable_prefix) = if let Some(active) = self.active.as_ref() {
            let stable = stable_prefix_len(&active.text, &text);
            if stable < active.text.len() {
                let old = active.id;
                events.push(ReadAloudCandidateEvent::CandidateCancelled { id: old });
                let new = self.next_id();
                events.push(ReadAloudCandidateEvent::CandidateReplaced {
                    old,
                    new,
                    stable_prefix_len: stable,
                });
                (new, stable)
            } else {
                (active.id, stable)
            }
        } else {
            let id = self.next_id();
            events.push(ReadAloudCandidateEvent::CandidateStarted { id });
            (id, 0)
        };

        let candidate = ReadAloudCandidate {
            id,
            safe_to_prepare_len: text.len(),
            safe_to_play_len: safe_to_play_len(&text),
            text,
            stable_prefix_len: stable_prefix,
            commitment: SpeechCandidateCommitment::Speculative,
        };

        self.active = Some(candidate.clone());
        events.push(ReadAloudCandidateEvent::CandidateUpdated { candidate });
        events
    }

    fn next_id(&mut self) -> SpeechCandidateId {
        self.next_id = self
            .next_id
            .checked_add(1)
            .expect("speech candidate id space exhausted");
        SpeechCandidateId(self.next_id)
    }
}

fn safe_to_play_len(input: &str) -> usize {
    let trimmed = input.trim_end();
    if trimmed.is_empty() {
        return 0;
    }

    if has_confident_sentence_end(trimmed) {
        return trimmed.len();
    }

    let words: Vec<&str> = trimmed.split_ascii_whitespace().collect();
    if words.len() <= 1 {
        return 0;
    }

    words
        .iter()
        .take(words.len() - 1)
        .fold(String::new(), |mut acc, word| {
            if !acc.is_empty() {
                acc.push(' ');
            }
            acc.push_str(word);
            acc
        })
        .len()
}

fn has_confident_sentence_end(trimmed: &str) -> bool {
    if trimmed.ends_with('!') || trimmed.ends_with('?') {
        return true;
    }

    if !trimmed.ends_with('.') {
        return false;
    }

    let stem = trimmed[..trimmed.len() - 1].trim_end();
    let last_token = stem.split_ascii_whitespace().next_back().unwrap_or_default();
    if last_token.is_empty() {
        return false;
    }

    if last_token.len() == 1 && last_token.chars().all(|ch| ch.is_ascii_alphabetic()) {
        return false;
    }

    if last_token.chars().all(|ch| ch.is_ascii_digit()) {
        return false;
    }

    let lower = last_token.to_ascii_lowercase();
    let is_honorific = matches!(lower.as_str(), "dr" | "mr" | "mrs" | "ms" | "prof")
        && last_token
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_uppercase());
    if is_honorific {
        return false;
    }

    if last_token.contains('@') || last_token.contains("://") || lower.starts_with("www.") {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn latest(events: &[ReadAloudCandidateEvent]) -> ReadAloudCandidate {
        match events.last().expect("candidate event") {
            ReadAloudCandidateEvent::CandidateUpdated { candidate } => candidate.clone(),
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn tracks_incremental_text_extension() {
        let mut tracker = ReadAloudCandidateTracker::new();

        let snapshots = [
            "I",
            "I read",
            "I read a",
            "I read a book",
            "I read a book by",
            "I read a book by F.",
            "I read a book by F. Scott",
            "I read a book by F. Scott Fitzgerald.",
        ];

        let mut last_id = None;
        let mut last_stable = 0;
        for snapshot in snapshots {
            let candidate = latest(&tracker.ingest_text(snapshot));
            if let Some(previous_id) = last_id {
                assert_eq!(candidate.id, previous_id);
                assert!(candidate.stable_prefix_len >= last_stable);
            }
            last_id = Some(candidate.id);
            last_stable = candidate.stable_prefix_len;
            assert_eq!(candidate.commitment, SpeechCandidateCommitment::Speculative);
            assert_eq!(candidate.safe_to_prepare_len, snapshot.len());
        }
    }

    #[test]
    fn head_replacement_cancels_and_restarts_candidate() {
        let mut tracker = ReadAloudCandidateTracker::new();

        let first = latest(&tracker.ingest_text("I read a book"));
        let second_events = tracker.ingest_text("You read a book");

        assert!(matches!(
            second_events.first(),
            Some(ReadAloudCandidateEvent::CandidateCancelled { id }) if *id == first.id
        ));
        assert!(second_events.iter().any(|event| matches!(
            event,
            ReadAloudCandidateEvent::CandidateReplaced {
                old,
                new: _,
                stable_prefix_len: 0
            } if *old == first.id
        )));
        let replacement = latest(&second_events);
        assert_ne!(replacement.id, first.id);
    }

    #[test]
    fn delays_sentence_commitment_for_f_scott_fitzgerald() {
        let mut tracker = ReadAloudCandidateTracker::new();

        let first = latest(&tracker.ingest_text("I read a book by F."));
        assert_eq!(first.safe_to_play_len, "I read a book by".len());
        assert_eq!(first.commitment, SpeechCandidateCommitment::Speculative);

        let second = latest(&tracker.ingest_text("I read a book by F. Scott Fitzgerald."));
        assert_eq!(second.id, first.id);
        assert_eq!(second.safe_to_play_len, second.text.len());
    }

    #[test]
    fn handles_dr_king_and_ordinary_sentence_endings() {
        let mut tracker = ReadAloudCandidateTracker::new();

        let honorific = latest(&tracker.ingest_text("Dr."));
        assert_eq!(honorific.safe_to_play_len, 0);

        let full_name = latest(&tracker.ingest_text("Dr. King."));
        assert_eq!(full_name.id, honorific.id);
        assert_eq!(full_name.safe_to_play_len, full_name.text.len());

        let ordinary = latest(&tracker.ingest_text("This is fine."));
        assert_eq!(ordinary.safe_to_play_len, ordinary.text.len());
    }

    #[test]
    fn commitment_states_progress_and_cancel() {
        let mut candidate = latest(&ReadAloudCandidateTracker::new().ingest_text("Hello world."));
        candidate.mark_prepared();
        assert_eq!(candidate.commitment, SpeechCandidateCommitment::Prepared);
        candidate.mark_playable();
        assert_eq!(candidate.commitment, SpeechCandidateCommitment::Playable);
        candidate.mark_committed();
        assert_eq!(candidate.commitment, SpeechCandidateCommitment::Committed);
        candidate.cancel();
        assert_eq!(candidate.commitment, SpeechCandidateCommitment::Cancelled);
    }
}
