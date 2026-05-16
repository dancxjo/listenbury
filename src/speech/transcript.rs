#[derive(Debug, Clone)]
pub struct TranscriptChunk {
    pub text: String,
    pub is_final: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TranscriptCandidateId(pub u64);

#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptReplacementReason {
    HeadChanged { stable_prefix_len: usize },
    Restarted,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptCandidateEvent {
    CandidateStarted {
        id: TranscriptCandidateId,
    },
    CandidateUpdated {
        id: TranscriptCandidateId,
        text: String,
        stable_prefix_len: usize,
        confidence: Option<f32>,
    },
    CandidateReplaced {
        old: TranscriptCandidateId,
        new: TranscriptCandidateId,
        reason: TranscriptReplacementReason,
    },
    CandidateFinalized {
        id: TranscriptCandidateId,
        text: String,
        confidence: Option<f32>,
    },
    CandidateCancelled {
        id: TranscriptCandidateId,
    },
}

#[derive(Debug, Default)]
pub struct TranscriptCandidateTracker {
    next_id: u64,
    active: Option<ActiveCandidate>,
}

#[derive(Debug)]
struct ActiveCandidate {
    id: TranscriptCandidateId,
    text: String,
}

impl TranscriptCandidateTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ingest_chunk(&mut self, chunk: TranscriptChunk) -> Vec<TranscriptCandidateEvent> {
        self.ingest_candidate(chunk.text, None, chunk.is_final)
    }

    /// Tracks candidate lifecycle transitions.
    ///
    /// This is the seam for future streaming/partial ASR: current final-only recognizers
    /// emit `CandidateStarted -> CandidateFinalized`, while future implementations can emit
    /// updates and replacements as hypotheses evolve.
    pub fn ingest_candidate(
        &mut self,
        text: impl Into<String>,
        confidence: Option<f32>,
        is_final: bool,
    ) -> Vec<TranscriptCandidateEvent> {
        let text = text.into();
        if text.is_empty() {
            return Vec::new();
        }

        let mut events = Vec::new();

        if let Some(active) = self.active.take() {
            if active.text == text {
                if is_final {
                    events.push(TranscriptCandidateEvent::CandidateFinalized {
                        id: active.id,
                        text,
                        confidence,
                    });
                } else {
                    let stable_prefix_len = text.len();
                    self.active = Some(ActiveCandidate {
                        id: active.id,
                        text: text.clone(),
                    });
                    events.push(TranscriptCandidateEvent::CandidateUpdated {
                        id: active.id,
                        text,
                        stable_prefix_len,
                        confidence,
                    });
                }
                return events;
            }

            let stable_prefix_len = stable_prefix_len(&active.text, &text);
            if stable_prefix_len < active.text.len() {
                let new_id = self.next_id();
                events.push(TranscriptCandidateEvent::CandidateReplaced {
                    old: active.id,
                    new: new_id,
                    reason: TranscriptReplacementReason::HeadChanged { stable_prefix_len },
                });
                events.push(TranscriptCandidateEvent::CandidateStarted { id: new_id });

                if is_final {
                    events.push(TranscriptCandidateEvent::CandidateFinalized {
                        id: new_id,
                        text,
                        confidence,
                    });
                } else {
                    self.active = Some(ActiveCandidate {
                        id: new_id,
                        text: text.clone(),
                    });
                    events.push(TranscriptCandidateEvent::CandidateUpdated {
                        id: new_id,
                        text,
                        stable_prefix_len,
                        confidence,
                    });
                }
                return events;
            }

            if is_final {
                events.push(TranscriptCandidateEvent::CandidateFinalized {
                    id: active.id,
                    text,
                    confidence,
                });
            } else {
                self.active = Some(ActiveCandidate {
                    id: active.id,
                    text: text.clone(),
                });
                events.push(TranscriptCandidateEvent::CandidateUpdated {
                    id: active.id,
                    text,
                    stable_prefix_len,
                    confidence,
                });
            }

            return events;
        }

        let id = self.next_id();
        events.push(TranscriptCandidateEvent::CandidateStarted { id });
        if is_final {
            events.push(TranscriptCandidateEvent::CandidateFinalized {
                id,
                text,
                confidence,
            });
        } else {
            let stable_prefix_len = text.len();
            self.active = Some(ActiveCandidate {
                id,
                text: text.clone(),
            });
            events.push(TranscriptCandidateEvent::CandidateUpdated {
                id,
                text,
                stable_prefix_len,
                confidence,
            });
        }

        events
    }

    fn next_id(&mut self) -> TranscriptCandidateId {
        self.next_id += 1;
        TranscriptCandidateId(self.next_id)
    }
}

pub fn shared_prefix_len(previous: &str, next: &str) -> usize {
    let mut len = 0;

    let mut previous_chars = previous.char_indices();
    let mut next_chars = next.char_indices();
    loop {
        match (previous_chars.next(), next_chars.next()) {
            (Some((idx, previous_char)), Some((_, next_char))) if previous_char == next_char => {
                len = idx + previous_char.len_utf8();
            }
            _ => break,
        }
    }

    len
}

pub fn stable_prefix_len(previous: &str, next: &str) -> usize {
    let shared = shared_prefix_len(previous, next);
    if shared == 0 {
        return 0;
    }

    if shared == previous.len() || shared == next.len() {
        return shared;
    }

    let boundary = last_word_boundary_at_or_before(previous, shared)
        .zip(last_word_boundary_at_or_before(next, shared))
        .map(|(previous_boundary, next_boundary)| previous_boundary.min(next_boundary));

    boundary.unwrap_or(shared)
}

fn last_word_boundary_at_or_before(text: &str, limit: usize) -> Option<usize> {
    let capped = limit.min(text.len());
    if capped == 0 || !text.is_char_boundary(capped) {
        return None;
    }

    let mut last_boundary = None;
    for (idx, ch) in text[..capped].char_indices() {
        if ch.is_whitespace() {
            last_boundary = Some(idx + ch.len_utf8());
        }
    }

    if capped < text.len() {
        let previous = text[..capped].chars().next_back();
        let next = text[capped..].chars().next();
        if let (Some(previous), Some(next)) = (previous, next)
            && (previous.is_whitespace() || next.is_whitespace())
        {
            last_boundary = Some(capped);
        }
    } else {
        last_boundary = Some(capped);
    }

    last_boundary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_prefix_detects_pure_extension() {
        assert_eq!(stable_prefix_len("hello", "hello world"), "hello".len());
    }

    #[test]
    fn stable_prefix_detects_correction_after_shared_prefix() {
        assert_eq!(
            stable_prefix_len("hello world", "hello there"),
            "hello ".len()
        );
    }

    #[test]
    fn stable_prefix_detects_novel_head() {
        assert_eq!(stable_prefix_len("goodbye", "hello"), 0);
    }

    #[test]
    fn stable_prefix_prefers_word_boundary_when_possible() {
        assert_eq!(
            stable_prefix_len("play music now", "play movie now"),
            "play ".len()
        );
    }

    #[test]
    fn final_only_chunks_emit_candidate_started_then_finalized() {
        let mut tracker = TranscriptCandidateTracker::new();

        let first = tracker.ingest_chunk(TranscriptChunk {
            text: "first".to_string(),
            is_final: true,
        });
        assert_eq!(
            first,
            vec![
                TranscriptCandidateEvent::CandidateStarted {
                    id: TranscriptCandidateId(1),
                },
                TranscriptCandidateEvent::CandidateFinalized {
                    id: TranscriptCandidateId(1),
                    text: "first".to_string(),
                    confidence: None,
                },
            ]
        );

        let second = tracker.ingest_chunk(TranscriptChunk {
            text: "second".to_string(),
            is_final: true,
        });
        assert_eq!(
            second,
            vec![
                TranscriptCandidateEvent::CandidateStarted {
                    id: TranscriptCandidateId(2),
                },
                TranscriptCandidateEvent::CandidateFinalized {
                    id: TranscriptCandidateId(2),
                    text: "second".to_string(),
                    confidence: None,
                },
            ]
        );
    }
}
