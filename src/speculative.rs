//! Shared speculative-candidate model for Listenbury pipelines.
//!
//! This module provides a stage-agnostic candidate shape that can represent:
//! - ASR transcript candidates,
//! - prompt/LLM text candidates,
//! - speech-unit candidates,
//! - speculative TTS audio,
//! - playback candidates.
//!
//! Candidates may be linked in a parent chain and only become committed when
//! callers explicitly confirm a safe boundary.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CandidateId(pub u64);

#[derive(Debug, Default)]
pub struct CandidateIdGenerator {
    next_id: u64,
}

impl CandidateIdGenerator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn next(&mut self) -> CandidateId {
        // IDs intentionally start at 1 to align with existing transcript candidate IDs.
        self.next_id = self
            .next_id
            .checked_add(1)
            .expect("speculative candidate id space exhausted");
        CandidateId(self.next_id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateStage {
    Transcript,
    Prompt,
    LlmText,
    SpeechUnit,
    TtsAudio,
    Playback,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FallbackKind {
    Silence,
    CachedBackchannel,
    Ellipsis,
    ThinkingFiller,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FallbackMetadata {
    pub kind: FallbackKind,
    pub reason: String,
}

impl FallbackMetadata {
    pub fn new(kind: FallbackKind, reason: impl Into<String>) -> Self {
        Self {
            kind,
            reason: reason.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Candidate<T> {
    pub id: CandidateId,
    pub parent: Option<CandidateId>,
    pub stage: CandidateStage,
    pub value: T,
    pub stable_head_key: String,
    pub committed: bool,
    pub confidence: Option<f32>,
    pub fallback: Option<FallbackMetadata>,
}

impl<T> Candidate<T> {
    pub fn new(
        id: CandidateId,
        parent: Option<CandidateId>,
        stage: CandidateStage,
        value: T,
        stable_head_key: impl Into<String>,
        confidence: Option<f32>,
    ) -> Self {
        Self {
            id,
            parent,
            stage,
            value,
            stable_head_key: stable_head_key.into(),
            committed: false,
            confidence,
            fallback: None,
        }
    }

    pub fn fallback(
        id: CandidateId,
        parent: Option<CandidateId>,
        stage: CandidateStage,
        value: T,
        stable_head_key: impl Into<String>,
        confidence: Option<f32>,
        fallback: FallbackMetadata,
    ) -> Self {
        let mut candidate = Self::new(id, parent, stage, value, stable_head_key, confidence);
        candidate.fallback = Some(fallback);
        candidate
    }

    pub fn is_fallback(&self) -> bool {
        self.fallback.is_some()
    }

    pub fn commit_at_boundary(&mut self, safe_boundary: bool) -> bool {
        if !safe_boundary {
            return false;
        }

        let changed = !self.committed;
        self.committed = true;
        changed
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StableHeadDecision {
    ExtensionReuse { stable_prefix_len: usize },
    SharedPrefixReuse { stable_prefix_len: usize },
    RestartFromNovelHead,
}

pub fn stable_head_decision(previous: &str, next: &str) -> StableHeadDecision {
    let stable_prefix_len = stable_prefix_len(previous, next);
    if stable_prefix_len == 0 {
        return StableHeadDecision::RestartFromNovelHead;
    }

    if stable_prefix_len == previous.len() && next.len() >= previous.len() {
        return StableHeadDecision::ExtensionReuse { stable_prefix_len };
    }

    StableHeadDecision::SharedPrefixReuse { stable_prefix_len }
}

pub fn is_pure_extension(previous: &str, next: &str) -> bool {
    matches!(
        stable_head_decision(previous, next),
        StableHeadDecision::ExtensionReuse { .. }
    )
}

pub fn shares_stable_prefix(previous: &str, next: &str) -> bool {
    stable_prefix_len(previous, next) > 0
}

pub fn head_changed_requires_restart(previous: &str, next: &str) -> bool {
    matches!(
        stable_head_decision(previous, next),
        StableHeadDecision::RestartFromNovelHead
    )
}

pub fn can_reuse_buffer(previous: &str, next: &str) -> bool {
    !head_changed_requires_restart(previous, next)
}

/// Character-level shared prefix length.
///
/// This does not enforce word boundaries.
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

/// Stable prefix length suitable for speculative head decisions.
///
/// Unlike [`shared_prefix_len`], this prefers full-word boundaries when both
/// strings diverge in the middle of a word.
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
    let mut capped = limit.min(text.len());
    while capped > 0 && !text.is_char_boundary(capped) {
        capped -= 1;
    }

    if capped == 0 {
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
    fn extension_reuse_is_detected() {
        let decision = stable_head_decision("hello", "hello world");
        assert_eq!(
            decision,
            StableHeadDecision::ExtensionReuse {
                stable_prefix_len: "hello".len()
            }
        );
        assert!(is_pure_extension("hello", "hello world"));
        assert!(can_reuse_buffer("hello", "hello world"));
    }

    #[test]
    fn shared_prefix_reuse_is_detected() {
        let decision = stable_head_decision("play music now", "play movie now");
        assert_eq!(
            decision,
            StableHeadDecision::SharedPrefixReuse {
                stable_prefix_len: "play ".len()
            }
        );
        assert!(shares_stable_prefix("play music now", "play movie now"));
        assert!(can_reuse_buffer("play music now", "play movie now"));
    }

    #[test]
    fn novel_head_requires_restart() {
        assert_eq!(
            stable_head_decision("goodbye", "hello there"),
            StableHeadDecision::RestartFromNovelHead
        );
        assert!(head_changed_requires_restart("goodbye", "hello there"));
        assert!(!can_reuse_buffer("goodbye", "hello there"));
    }

    #[test]
    fn fallback_candidate_creation_marks_metadata() {
        let mut ids = CandidateIdGenerator::new();
        let candidate = Candidate::fallback(
            ids.next(),
            None,
            CandidateStage::Playback,
            "Let me think.".to_string(),
            "thinking".to_string(),
            Some(0.4),
            FallbackMetadata::new(FallbackKind::ThinkingFiller, "llm still generating"),
        );

        assert!(candidate.is_fallback());
        assert_eq!(
            candidate.fallback,
            Some(FallbackMetadata::new(
                FallbackKind::ThinkingFiller,
                "llm still generating"
            ))
        );
    }

    #[test]
    fn commit_at_boundary_only_commits_when_safe() {
        let mut candidate = Candidate::new(
            CandidateId(1),
            None,
            CandidateStage::LlmText,
            "working".to_string(),
            "working".to_string(),
            None,
        );

        assert!(!candidate.committed);
        assert!(!candidate.commit_at_boundary(false));
        assert!(!candidate.committed);
        assert!(candidate.commit_at_boundary(true));
        assert!(candidate.committed);
        assert!(!candidate.commit_at_boundary(true));
    }

    #[test]
    fn candidate_ids_start_at_one() {
        let mut ids = CandidateIdGenerator::new();
        assert_eq!(ids.next(), CandidateId(1));
    }
}
