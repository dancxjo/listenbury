//! Adapter layer that converts generated/outgoing text into a [`TimedWordStream`].
//!
//! The pipeline shape this module implements:
//!
//! ```text
//! text stream
//!   -> speech candidates
//!   -> speculative TTS/preparation
//!   -> playback
//!   -> timed word stream
//! ```
//!
//! # Overview
//!
//! - [`generated_text_to_word_stream`] converts raw generated text into a
//!   [`TimedWordStream`] with initial commitment [`WordCommitment::StableText`].
//! - [`attach_heuristic_timing`] distributes a known total duration proportionally
//!   across words by character count.
//! - [`attach_audio_frame_timing`] calculates the total duration from a slice of
//!   [`AudioFrame`]s and then calls [`attach_heuristic_timing`].
//! - [`PlaybackCursor`] tracks the current playback position and updates word
//!   commitment states as audio plays out or is interrupted.

use crate::audio::frame::AudioFrame;
use crate::word::stream::{
    BoundarySource, TextSpan, TimedWordStream, WordCommitment, WordId, WordNode, WordStreamId,
    WordStreamSource, WordTiming,
};

// ---------------------------------------------------------------------------
// Text → word stream adapter
// ---------------------------------------------------------------------------

/// Convert a generated text string into a [`TimedWordStream`] with
/// [`WordStreamSource::SyntheticSpeech`].
///
/// Each word is split by ASCII whitespace.  The resulting [`WordNode`]s start
/// with:
/// - `commitment`: [`WordCommitment::StableText`] (text is known but not yet
///   synthesised).
/// - `boundary_source`: [`BoundarySource::Predicted`] (timing is unknown until
///   TTS audio arrives).
/// - `timing`: `None` — attach timing later with [`attach_heuristic_timing`]
///   or [`attach_audio_frame_timing`].
/// - `lexical_span`: byte-level span within the original `text` string.
///
/// An empty or whitespace-only `text` produces an empty stream.
///
/// # Example
///
/// ```rust
/// use listenbury::word::tts_export::generated_text_to_word_stream;
/// use listenbury::word::{WordStreamId, WordStreamSource, WordCommitment};
///
/// let stream = generated_text_to_word_stream(WordStreamId(1), "hello there");
/// assert_eq!(stream.source, WordStreamSource::SyntheticSpeech);
/// assert_eq!(stream.words.len(), 2);
/// assert!(stream.words.iter().all(|w| w.timing.is_none()));
/// assert!(stream.words.iter().all(|w| w.commitment == WordCommitment::StableText));
/// ```
pub fn generated_text_to_word_stream(id: WordStreamId, text: &str) -> TimedWordStream {
    let word_nodes: Vec<WordNode> = split_words(text)
        .enumerate()
        .map(|(i, (word_text, span_start, span_end))| WordNode {
            id: WordId(i as u64 + 1),
            text: word_text.to_string(),
            lexical_span: Some(TextSpan {
                start: span_start,
                end: span_end,
            }),
            timing: None,
            timing_confidence: None,
            commitment: WordCommitment::StableText,
            boundary_source: BoundarySource::Predicted,
            audio_ref: None,
        })
        .collect();

    TimedWordStream {
        id,
        source: WordStreamSource::SyntheticSpeech,
        words: word_nodes,
    }
}

/// Iterate over words in `text`, yielding `(word_str, byte_start, byte_end)`.
///
/// Words are delimited by ASCII whitespace.  The byte spans are relative to
/// the beginning of `text`.
fn split_words(text: &str) -> impl Iterator<Item = (&str, usize, usize)> {
    SplitWordsIter {
        text,
        byte_pos: 0,
    }
}

struct SplitWordsIter<'a> {
    text: &'a str,
    byte_pos: usize,
}

impl<'a> Iterator for SplitWordsIter<'a> {
    type Item = (&'a str, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        // Skip leading whitespace.
        let remaining = &self.text[self.byte_pos..];
        let leading_ws = remaining
            .char_indices()
            .take_while(|(_, c)| c.is_ascii_whitespace())
            .count();
        self.byte_pos += remaining
            .char_indices()
            .take(leading_ws)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);

        if self.byte_pos >= self.text.len() {
            return None;
        }

        let remaining = &self.text[self.byte_pos..];
        // Find end of this word (next whitespace).
        let word_len = remaining
            .char_indices()
            .take_while(|(_, c)| !c.is_ascii_whitespace())
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);

        if word_len == 0 {
            return None;
        }

        let start = self.byte_pos;
        let end = start + word_len;
        let word_str = &self.text[start..end];
        self.byte_pos = end;
        Some((word_str, start, end))
    }
}

// ---------------------------------------------------------------------------
// Timing attachment
// ---------------------------------------------------------------------------

/// Distribute `total_duration_ms` across words in `stream` proportionally by
/// character count, updating `timing` and `boundary_source` on each word.
///
/// After this call every word with non-zero text will have a [`WordTiming`]
/// with [`BoundarySource::Predicted`].  Words whose text is empty are assigned
/// a zero-duration interval at the end of the stream.
///
/// `stream` may contain words at any commitment level; this function does not
/// change commitment states — use [`PlaybackCursor`] to advance those.
///
/// If the stream is empty the function returns immediately without modifying
/// anything.
pub fn attach_heuristic_timing(stream: &mut TimedWordStream, total_duration_ms: u64) {
    if stream.words.is_empty() {
        return;
    }

    // Total character count across all words (use 1 as minimum to avoid div/0).
    let total_chars: usize = stream.words.iter().map(|w| w.text.len().max(1)).sum();

    let mut cursor_ms: u64 = 0;
    let n = stream.words.len();

    for (i, word) in stream.words.iter_mut().enumerate() {
        let word_chars = word.text.len().max(1);
        // Last word gets all remaining duration to avoid rounding drift.
        let duration_ms = if i == n - 1 {
            total_duration_ms.saturating_sub(cursor_ms)
        } else {
            (total_duration_ms as u128 * word_chars as u128 / total_chars as u128) as u64
        };

        let start_ms = cursor_ms;
        let end_ms = cursor_ms + duration_ms;
        // WordTiming::new enforces start <= end; this is always true here.
        word.timing = WordTiming::new(start_ms, end_ms);
        word.boundary_source = BoundarySource::Predicted;
        cursor_ms = end_ms;
    }
}

/// Calculate the total playback duration from a slice of [`AudioFrame`]s and
/// call [`attach_heuristic_timing`] on `stream`.
///
/// The duration is derived from the number of PCM samples divided by the
/// sample rate.  If `frames` is empty the total duration is 0 ms and all words
/// receive zero-length timing anchored at 0.
///
/// # Panics
///
/// Does not panic.  Frames with a sample rate of 0 contribute 0 ms.
pub fn attach_audio_frame_timing(stream: &mut TimedWordStream, frames: &[AudioFrame]) {
    let total_ms = total_duration_ms(frames);
    attach_heuristic_timing(stream, total_ms);
}

/// Sum the playback duration of all `frames` in milliseconds.
fn total_duration_ms(frames: &[AudioFrame]) -> u64 {
    frames
        .iter()
        .map(|f| {
            if f.sample_rate_hz == 0 || f.channels == 0 {
                return 0u64;
            }
            // samples contains interleaved channel data.
            let per_channel = f.samples.len() / f.channels as usize;
            per_channel as u64 * 1_000 / f.sample_rate_hz as u64
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Playback cursor
// ---------------------------------------------------------------------------

/// Tracks the current position within a [`TimedWordStream`] during TTS
/// playback.
///
/// As audio is played out, call [`advance`][`PlaybackCursor::advance`] with
/// the elapsed milliseconds.  The cursor updates word commitment states:
///
/// - Words whose interval contains the current playback position are marked
///   [`WordCommitment::Played`].
/// - Words whose interval ends before the current position are marked
///   [`WordCommitment::Final`].
///
/// When playback is interrupted (e.g. because the user spoke), call
/// [`interrupt`][`PlaybackCursor::interrupt`] to mark all not-yet-final words
/// as [`WordCommitment::Cancelled`].
///
/// # Example
///
/// ```rust
/// use listenbury::word::tts_export::{generated_text_to_word_stream, attach_heuristic_timing, PlaybackCursor};
/// use listenbury::word::{WordStreamId, WordCommitment};
///
/// let mut stream = generated_text_to_word_stream(WordStreamId(1), "hello world");
/// attach_heuristic_timing(&mut stream, 1000);
/// // Promote words to Playable before playback begins.
/// for w in &mut stream.words { w.commitment = WordCommitment::Playable; }
///
/// let mut cursor = PlaybackCursor::new();
/// cursor.advance(&mut stream, 600);
/// // "hello" (ends around 455ms for 5 chars out of 11) is now Final.
/// // "world" is the current word being played.
/// assert_eq!(cursor.played_ms(), 600);
/// ```
#[derive(Debug, Default, Clone)]
pub struct PlaybackCursor {
    /// Current playback position in milliseconds.
    played_ms: u64,
    /// The [`WordId`] of the word that is currently being spoken, if any.
    current_word: Option<WordId>,
}

impl PlaybackCursor {
    /// Create a new cursor positioned at 0 ms.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current playback position in milliseconds from the start of the stream.
    pub fn played_ms(&self) -> u64 {
        self.played_ms
    }

    /// The word currently being spoken, or `None` if no word covers the current
    /// position.
    pub fn current_word(&self) -> Option<WordId> {
        self.current_word
    }

    /// Advance the playback cursor by `delta_ms` milliseconds.
    ///
    /// For each word in `stream` that has timing information:
    /// - If the word's interval ends at or before `played_ms`, mark it as
    ///   [`WordCommitment::Final`] (if it was `Played` or `Playable`).
    /// - If the word's interval contains `played_ms`, mark it as
    ///   [`WordCommitment::Played`] and record it as `current_word`.
    ///
    /// Words with commitment other than `Playable` or `Played` are not
    /// modified.
    pub fn advance(&mut self, stream: &mut TimedWordStream, delta_ms: u64) {
        self.played_ms = self.played_ms.saturating_add(delta_ms);
        self.current_word = None;

        for word in &mut stream.words {
            let Some(timing) = word.timing else { continue };

            match word.commitment {
                WordCommitment::Playable | WordCommitment::Played => {
                    if timing.end_ms <= self.played_ms {
                        word.commitment = WordCommitment::Final;
                    } else if timing.start_ms <= self.played_ms {
                        word.commitment = WordCommitment::Played;
                        self.current_word = Some(word.id);
                    }
                }
                _ => {}
            }
        }
    }

    /// Interrupt playback: mark every word that has not yet reached
    /// [`WordCommitment::Final`] as [`WordCommitment::Cancelled`].
    ///
    /// Also clears `current_word`.
    pub fn interrupt(&mut self, stream: &mut TimedWordStream) {
        self.current_word = None;
        for word in &mut stream.words {
            match word.commitment {
                WordCommitment::Final => {}
                _ => {
                    word.commitment = WordCommitment::Cancelled;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::ExactTimestamp;

    // -----------------------------------------------------------------------
    // generated_text_to_word_stream
    // -----------------------------------------------------------------------

    /// Basic conversion produces one WordNode per whitespace-delimited token.
    #[test]
    fn basic_text_produces_one_word_per_token() {
        let stream = generated_text_to_word_stream(WordStreamId(1), "hello there world");
        assert_eq!(stream.source, WordStreamSource::SyntheticSpeech);
        assert_eq!(stream.words.len(), 3);
        assert_eq!(stream.words[0].text, "hello");
        assert_eq!(stream.words[1].text, "there");
        assert_eq!(stream.words[2].text, "world");
    }

    /// All initial words start as StableText with no timing.
    #[test]
    fn initial_commitment_is_stable_text() {
        let stream = generated_text_to_word_stream(WordStreamId(2), "one two three");
        assert!(stream.words.iter().all(|w| w.commitment == WordCommitment::StableText));
        assert!(stream.words.iter().all(|w| w.timing.is_none()));
        assert!(stream.words.iter().all(|w| w.boundary_source == BoundarySource::Predicted));
    }

    /// Empty text produces an empty stream without panicking.
    #[test]
    fn empty_text_produces_empty_stream() {
        let stream = generated_text_to_word_stream(WordStreamId(3), "");
        assert_eq!(stream.source, WordStreamSource::SyntheticSpeech);
        assert!(stream.words.is_empty());
    }

    /// Whitespace-only text produces an empty stream.
    #[test]
    fn whitespace_only_text_produces_empty_stream() {
        let stream = generated_text_to_word_stream(WordStreamId(4), "   \t  \n  ");
        assert!(stream.words.is_empty());
    }

    /// Lexical spans in the resulting stream correspond to byte offsets within
    /// the original text.
    #[test]
    fn lexical_spans_match_original_text() {
        let text = "hi there";
        let stream = generated_text_to_word_stream(WordStreamId(5), text);
        assert_eq!(stream.words.len(), 2);
        let s0 = stream.words[0].lexical_span.unwrap();
        let s1 = stream.words[1].lexical_span.unwrap();
        assert_eq!(&text[s0.start..s0.end], "hi");
        assert_eq!(&text[s1.start..s1.end], "there");
    }

    /// Multiple spaces between words are handled correctly.
    #[test]
    fn multiple_spaces_between_words() {
        let stream = generated_text_to_word_stream(WordStreamId(6), "a   b");
        assert_eq!(stream.words.len(), 2);
        assert_eq!(stream.words[0].text, "a");
        assert_eq!(stream.words[1].text, "b");
    }

    /// Word IDs are assigned sequentially starting from 1.
    #[test]
    fn word_ids_are_sequential_from_one() {
        let stream = generated_text_to_word_stream(WordStreamId(7), "x y z");
        assert_eq!(stream.words[0].id, WordId(1));
        assert_eq!(stream.words[1].id, WordId(2));
        assert_eq!(stream.words[2].id, WordId(3));
    }

    // -----------------------------------------------------------------------
    // Incremental text extension
    // -----------------------------------------------------------------------

    /// Extending a stream by creating a new one from a longer text string
    /// and appending new words preserves already-existing words.
    #[test]
    fn incremental_text_extension() {
        // Simulate incremental generation: start with 2 words, then add 1 more.
        let stream1 = generated_text_to_word_stream(WordStreamId(10), "sure I");
        assert_eq!(stream1.words.len(), 2);

        // Later, the full text is "sure I can".  Build a new stream for the
        // extra word and append it to the first.
        let extra = generated_text_to_word_stream(WordStreamId(11), "can");
        assert_eq!(extra.words.len(), 1);

        // Combine: manually extend (real code can do the same).
        let mut combined = stream1.clone();
        let offset = combined.words.len() as u64;
        for mut w in extra.words {
            w.id = WordId(offset + w.id.0);
            combined.words.push(w);
        }

        assert_eq!(combined.words.len(), 3);
        assert_eq!(combined.words[0].text, "sure");
        assert_eq!(combined.words[2].text, "can");
        assert!(combined.words.iter().all(|w| w.commitment == WordCommitment::StableText));
    }

    // -----------------------------------------------------------------------
    // attach_heuristic_timing
    // -----------------------------------------------------------------------

    /// Heuristic timing distributes total duration proportionally and covers
    /// the full time range without gaps.
    #[test]
    fn heuristic_timing_covers_full_duration() {
        let mut stream = generated_text_to_word_stream(WordStreamId(20), "hello world");
        attach_heuristic_timing(&mut stream, 1000);

        assert!(stream.words.iter().all(|w| w.timing.is_some()));

        // First word starts at 0.
        assert_eq!(stream.words[0].timing.unwrap().start_ms, 0);
        // Last word ends at total_duration_ms.
        let last = stream.words.last().unwrap();
        assert_eq!(last.timing.unwrap().end_ms, 1000);
    }

    /// Longer words receive more time than shorter ones.
    #[test]
    fn longer_words_get_more_time() {
        let mut stream = generated_text_to_word_stream(WordStreamId(21), "hi extraordinary");
        attach_heuristic_timing(&mut stream, 1000);

        let dur_hi = stream.words[0].timing.unwrap().duration_ms();
        let dur_long = stream.words[1].timing.unwrap().duration_ms();
        assert!(
            dur_long > dur_hi,
            "longer word should get more time: {dur_long} vs {dur_hi}"
        );
    }

    /// Calling heuristic timing on an empty stream does not panic.
    #[test]
    fn heuristic_timing_empty_stream_no_panic() {
        let mut stream = TimedWordStream::new(WordStreamId(22), WordStreamSource::SyntheticSpeech);
        attach_heuristic_timing(&mut stream, 1000); // must not panic
        assert!(stream.words.is_empty());
    }

    // -----------------------------------------------------------------------
    // attach_audio_frame_timing
    // -----------------------------------------------------------------------

    /// attach_audio_frame_timing calculates duration from frames correctly.
    #[test]
    fn audio_frame_timing_matches_sample_count() {
        let sample_rate_hz = 22_050u32;
        // 22_050 samples at 22_050 Hz, mono = 1 second = 1_000 ms.
        let frame = AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz,
            channels: 1,
            samples: vec![0.0f32; 22_050],
        };

        let mut stream = generated_text_to_word_stream(WordStreamId(30), "one two three");
        attach_audio_frame_timing(&mut stream, &[frame]);

        let last = stream.words.last().unwrap();
        assert_eq!(last.timing.unwrap().end_ms, 1_000);
    }

    /// Empty frames slice results in zero-ms timing (words anchored at 0).
    #[test]
    fn audio_frame_timing_empty_frames() {
        let mut stream = generated_text_to_word_stream(WordStreamId(31), "hi there");
        attach_audio_frame_timing(&mut stream, &[]);

        // All words start and end at 0.
        for w in &stream.words {
            let t = w.timing.unwrap();
            assert_eq!(t.start_ms, 0);
            assert_eq!(t.end_ms, 0);
        }
    }

    // -----------------------------------------------------------------------
    // PlaybackCursor — committed playback timing
    // -----------------------------------------------------------------------

    /// Advancing the cursor past a word marks it as Final.
    #[test]
    fn advancing_cursor_marks_past_words_as_final() {
        let mut stream = generated_text_to_word_stream(WordStreamId(40), "hello world");
        attach_heuristic_timing(&mut stream, 1000);
        // Promote to Playable.
        for w in &mut stream.words {
            w.commitment = WordCommitment::Playable;
        }

        let mut cursor = PlaybackCursor::new();
        // "hello" = 5 chars, "world" = 5 chars → each ~500 ms.
        // Advance past the first word.
        cursor.advance(&mut stream, 600);

        assert_eq!(stream.words[0].commitment, WordCommitment::Final);
        assert_eq!(stream.words[1].commitment, WordCommitment::Played);
        assert_eq!(cursor.current_word(), Some(WordId(2)));
        assert_eq!(cursor.played_ms(), 600);
    }

    /// Advancing the cursor to the end marks all words as Final.
    #[test]
    fn advancing_to_end_marks_all_words_final() {
        let mut stream = generated_text_to_word_stream(WordStreamId(41), "one two");
        attach_heuristic_timing(&mut stream, 500);
        for w in &mut stream.words {
            w.commitment = WordCommitment::Playable;
        }

        let mut cursor = PlaybackCursor::new();
        cursor.advance(&mut stream, 500);

        assert!(stream.words.iter().all(|w| w.commitment == WordCommitment::Final));
        assert_eq!(cursor.current_word(), None);
    }

    /// The cursor correctly reports no current word before any advance.
    #[test]
    fn new_cursor_has_no_current_word() {
        let cursor = PlaybackCursor::new();
        assert_eq!(cursor.current_word(), None);
        assert_eq!(cursor.played_ms(), 0);
    }

    /// Multiple advance calls are cumulative.
    #[test]
    fn multiple_advances_are_cumulative() {
        let mut stream = generated_text_to_word_stream(WordStreamId(42), "a b c");
        attach_heuristic_timing(&mut stream, 300);
        for w in &mut stream.words {
            w.commitment = WordCommitment::Playable;
        }

        let mut cursor = PlaybackCursor::new();
        cursor.advance(&mut stream, 50);
        cursor.advance(&mut stream, 50);
        assert_eq!(cursor.played_ms(), 100);
    }

    // -----------------------------------------------------------------------
    // PlaybackCursor — speculative / cancelled words
    // -----------------------------------------------------------------------

    /// Interrupt marks all non-final words as Cancelled.
    #[test]
    fn interrupt_cancels_all_non_final_words() {
        let mut stream = generated_text_to_word_stream(WordStreamId(50), "one two three");
        attach_heuristic_timing(&mut stream, 1500);
        for w in &mut stream.words {
            w.commitment = WordCommitment::Playable;
        }

        // Advance past the first word.
        let mut cursor = PlaybackCursor::new();
        cursor.advance(&mut stream, 600);

        // "one" should be Final; "two" and "three" still Playable or Played.
        cursor.interrupt(&mut stream);

        assert_eq!(stream.words[0].commitment, WordCommitment::Final);
        assert_eq!(stream.words[1].commitment, WordCommitment::Cancelled);
        assert_eq!(stream.words[2].commitment, WordCommitment::Cancelled);
        assert_eq!(cursor.current_word(), None);
    }

    /// Words that start as StableText (not yet Playable) are also cancelled
    /// on interrupt.
    #[test]
    fn interrupt_cancels_stable_text_words() {
        let mut stream = generated_text_to_word_stream(WordStreamId(51), "speculative words here");
        // No timing attached; words remain StableText.

        let mut cursor = PlaybackCursor::new();
        cursor.interrupt(&mut stream);

        assert!(stream.words.iter().all(|w| w.commitment == WordCommitment::Cancelled));
    }

    // -----------------------------------------------------------------------
    // PlaybackCursor — interruption during playback
    // -----------------------------------------------------------------------

    /// Interrupting mid-playback leaves completed words as Final and cancels
    /// the rest.
    #[test]
    fn interruption_mid_playback_preserves_played_words() {
        let mut stream = generated_text_to_word_stream(WordStreamId(60), "a b c d");
        attach_heuristic_timing(&mut stream, 1000);
        for w in &mut stream.words {
            w.commitment = WordCommitment::Playable;
        }

        let mut cursor = PlaybackCursor::new();
        // Each word is ~250 ms (equal length).  Advance 300 ms to complete "a"
        // and start "b".
        cursor.advance(&mut stream, 300);

        // Interrupt: "a" is Final, "b" is currently Played, "c"/"d" are Playable.
        cursor.interrupt(&mut stream);

        assert_eq!(stream.words[0].commitment, WordCommitment::Final,  "a should be Final");
        assert_eq!(stream.words[1].commitment, WordCommitment::Cancelled, "b should be Cancelled");
        assert_eq!(stream.words[2].commitment, WordCommitment::Cancelled, "c should be Cancelled");
        assert_eq!(stream.words[3].commitment, WordCommitment::Cancelled, "d should be Cancelled");
    }
}
