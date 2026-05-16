//! Canonical text-stability helpers shared across Listenbury pipeline stages.
//!
//! These utilities compute word-boundary-aware stable prefixes between two
//! evolving text hypotheses.  They are used by both the transcript candidate
//! tracker in [`crate::speech::transcript`] and the stage-agnostic speculative
//! candidate model in [`crate::speculative`] so that both layers share exactly
//! one implementation and cannot diverge around char boundaries, whitespace,
//! initials, or abbreviations.

/// Returns the length in bytes of the longest common character-level prefix
/// of `previous` and `next`.
///
/// This does not enforce word boundaries; see [`stable_prefix_len`] for the
/// word-boundary-aware variant.
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

/// Returns the length in bytes of the longest *stable* prefix of `previous`
/// that is still valid in `next`, snapped to a word boundary where possible.
///
/// Unlike [`shared_prefix_len`], when both strings diverge in the middle of a
/// word this function retreats to the last whitespace boundary so callers do
/// not commit partial words as stable heads.
///
/// # Examples
///
/// ```
/// use listenbury::stable_prefix_len;
/// assert_eq!(stable_prefix_len("hello", "hello world"), "hello".len());
/// assert_eq!(stable_prefix_len("play music now", "play movie now"), "play ".len());
/// assert_eq!(stable_prefix_len("goodbye", "hello"), 0);
/// ```
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

    // `shared` is always a valid char boundary because `shared_prefix_len` advances using
    // `char_indices` from both strings.
    boundary.unwrap_or(shared)
}

pub(crate) fn last_word_boundary_at_or_before(text: &str, limit: usize) -> Option<usize> {
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
    fn shared_prefix_len_returns_zero_for_completely_different_strings() {
        assert_eq!(shared_prefix_len("abc", "xyz"), 0);
    }

    #[test]
    fn shared_prefix_len_handles_identical_strings() {
        assert_eq!(shared_prefix_len("same", "same"), "same".len());
    }

    #[test]
    fn stable_prefix_treats_full_match_as_stable() {
        assert_eq!(stable_prefix_len("exact", "exact"), "exact".len());
    }
}
