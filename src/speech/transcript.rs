use crate::text_stability::stable_prefix_len;

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

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptStabilityState {
    pub candidate_id: TranscriptCandidateId,
    pub text: String,
    pub stable_prefix_len: usize,
    pub stable_text: String,
    pub unstable_text: String,
    pub stable_word_prefix: Option<String>,
    pub stable_word_count: usize,
    pub confidence: Option<f32>,
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
            return if is_final {
                self.cancel_active()
            } else {
                Vec::new()
            };
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

    pub fn cancel_active(&mut self) -> Vec<TranscriptCandidateEvent> {
        let Some(active) = self.active.take() else {
            return Vec::new();
        };
        vec![TranscriptCandidateEvent::CandidateCancelled { id: active.id }]
    }

    fn next_id(&mut self) -> TranscriptCandidateId {
        // IDs intentionally start at 1.
        self.next_id = self
            .next_id
            .checked_add(1)
            .expect("transcript candidate id space exhausted");
        TranscriptCandidateId(self.next_id)
    }
}

impl TranscriptStabilityState {
    pub fn from_parts(
        candidate_id: TranscriptCandidateId,
        text: &str,
        stable_prefix_len: usize,
        confidence: Option<f32>,
    ) -> Self {
        let split = stable_prefix_len.min(text.len());
        let split = if text.is_char_boundary(split) {
            split
        } else {
            text.char_indices()
                .find_map(|(idx, ch)| {
                    let end = idx + ch.len_utf8();
                    (end >= split).then_some(end)
                })
                .unwrap_or(text.len())
        };
        let (stable_text, unstable_text) = text.split_at(split);
        let stable_word_split = if stable_text
            .chars()
            .next_back()
            .is_some_and(char::is_whitespace)
        {
            stable_text.trim_end().len()
        } else {
            stable_text
                .char_indices()
                .rev()
                .find_map(|(idx, ch)| ch.is_whitespace().then_some(idx + ch.len_utf8()))
                .unwrap_or_default()
        };
        let stable_word_prefix = stable_text[..stable_word_split].trim_end();
        Self {
            candidate_id,
            text: text.to_string(),
            stable_prefix_len: split,
            stable_text: stable_text.to_string(),
            unstable_text: unstable_text.to_string(),
            stable_word_prefix: (!stable_word_prefix.is_empty())
                .then(|| stable_word_prefix.to_string()),
            stable_word_count: stable_word_prefix.split_whitespace().count(),
            confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn nonfinal_extension_keeps_candidate_and_reports_stable_prefix() {
        let mut tracker = TranscriptCandidateTracker::new();

        let first = tracker.ingest_candidate("can you", None, false);
        assert_eq!(
            first,
            vec![
                TranscriptCandidateEvent::CandidateStarted {
                    id: TranscriptCandidateId(1),
                },
                TranscriptCandidateEvent::CandidateUpdated {
                    id: TranscriptCandidateId(1),
                    text: "can you".to_string(),
                    stable_prefix_len: "can you".len(),
                    confidence: None,
                },
            ]
        );

        let second = tracker.ingest_candidate("can you tell", None, false);
        assert_eq!(
            second,
            vec![TranscriptCandidateEvent::CandidateUpdated {
                id: TranscriptCandidateId(1),
                text: "can you tell".to_string(),
                stable_prefix_len: "can you".len(),
                confidence: None,
            },]
        );
    }

    #[test]
    fn correction_after_stable_prefix_replaces_candidate_with_shared_head() {
        let mut tracker = TranscriptCandidateTracker::new();
        let _ = tracker.ingest_candidate("can you tell", None, false);

        let events = tracker.ingest_candidate("can you help", None, false);
        assert_eq!(
            events,
            vec![
                TranscriptCandidateEvent::CandidateReplaced {
                    old: TranscriptCandidateId(1),
                    new: TranscriptCandidateId(2),
                    reason: TranscriptReplacementReason::HeadChanged {
                        stable_prefix_len: "can you ".len(),
                    },
                },
                TranscriptCandidateEvent::CandidateStarted {
                    id: TranscriptCandidateId(2),
                },
                TranscriptCandidateEvent::CandidateUpdated {
                    id: TranscriptCandidateId(2),
                    text: "can you help".to_string(),
                    stable_prefix_len: "can you ".len(),
                    confidence: None,
                },
            ]
        );
    }

    #[test]
    fn novel_head_restarts_candidate_from_zero_stable_prefix() {
        let mut tracker = TranscriptCandidateTracker::new();
        let _ = tracker.ingest_candidate("can you tell", None, false);

        let events = tracker.ingest_candidate("wait no actually", None, false);
        assert_eq!(
            events,
            vec![
                TranscriptCandidateEvent::CandidateReplaced {
                    old: TranscriptCandidateId(1),
                    new: TranscriptCandidateId(2),
                    reason: TranscriptReplacementReason::HeadChanged {
                        stable_prefix_len: 0,
                    },
                },
                TranscriptCandidateEvent::CandidateStarted {
                    id: TranscriptCandidateId(2),
                },
                TranscriptCandidateEvent::CandidateUpdated {
                    id: TranscriptCandidateId(2),
                    text: "wait no actually".to_string(),
                    stable_prefix_len: 0,
                    confidence: None,
                },
            ]
        );
    }

    #[test]
    fn final_empty_update_cancels_active_candidate() {
        let mut tracker = TranscriptCandidateTracker::new();
        let _ = tracker.ingest_candidate("can you tell", Some(0.4), false);

        let events = tracker.ingest_candidate("", None, true);
        assert_eq!(
            events,
            vec![TranscriptCandidateEvent::CandidateCancelled {
                id: TranscriptCandidateId(1),
            }]
        );
        assert!(tracker.cancel_active().is_empty());
    }

    #[test]
    fn partial_candidate_can_finalize_without_replacement() {
        let mut tracker = TranscriptCandidateTracker::new();
        let _ = tracker.ingest_candidate("can you tell", Some(0.4), false);

        let events = tracker.ingest_candidate("can you tell", Some(0.9), true);
        assert_eq!(
            events,
            vec![TranscriptCandidateEvent::CandidateFinalized {
                id: TranscriptCandidateId(1),
                text: "can you tell".to_string(),
                confidence: Some(0.9),
            }]
        );
    }

    #[test]
    fn stability_state_tracks_whole_word_prefix_separately_from_character_prefix() {
        let state = TranscriptStabilityState::from_parts(
            TranscriptCandidateId(7),
            "hello wor",
            "hello wor".len(),
            Some(0.6),
        );

        assert_eq!(state.stable_text, "hello wor");
        assert_eq!(state.stable_word_prefix.as_deref(), Some("hello"));
        assert_eq!(state.stable_word_count, 1);
        assert_eq!(state.unstable_text, "");
        assert_eq!(state.confidence, Some(0.6));
    }
}
