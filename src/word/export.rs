//! Adapter layer that converts ASR transcript output into a [`TimedWordStream`].
//!
//! The pipeline shape this module implements:
//!
//! ```text
//! audio
//!   -> Whisper transcript/timestamps
//!   -> TimedWordStream
//!   -> optional acoustic refinement
//! ```
//!
//! # Usage
//!
//! Given a slice of [`TranscriptWord`] values produced by an ASR engine (e.g.
//! Whisper), call [`transcript_to_word_stream`] to obtain a
//! [`TimedWordStream`] with [`WordStreamSource::RecordedAudio`].
//!
//! Word timing produced by Whisper is preserved directly.  Later passes can
//! refine the boundaries by implementing the [`WordBoundaryRefiner`] trait and
//! calling [`WordBoundaryRefiner::refine`] on the resulting stream.  A no-op
//! placeholder ([`NoopWordBoundaryRefiner`]) is provided for use in tests and
//! pipelines that do not yet require acoustic alignment.

use crate::audio::frame::AudioFrame;
use crate::word::stream::{
    BoundarySource, TextSpan, TimedWordStream, WordCommitment, WordId, WordNode, WordStreamId,
    WordStreamSource, WordTiming,
};

// ---------------------------------------------------------------------------
// Input type
// ---------------------------------------------------------------------------

/// A single word as emitted by an ASR engine, with optional timing and
/// confidence information.
///
/// This is the primary input type for [`transcript_to_word_stream`].  Timing
/// fields are `Option` because some ASR backends return only the full
/// transcript text without per-word timestamps.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptWord {
    /// The textual form of this word (may include surrounding punctuation).
    pub text: String,
    /// Start of this word in the audio timeline, in milliseconds from the
    /// beginning of the audio buffer.  `None` if the ASR engine did not
    /// produce per-word timestamps.
    pub start_ms: Option<u64>,
    /// End of this word in the audio timeline, in milliseconds.  `None` if the
    /// ASR engine did not produce per-word timestamps.
    pub end_ms: Option<u64>,
    /// ASR confidence for this word's transcription (0.0 – 1.0), if available.
    pub confidence: Option<f32>,
}

// ---------------------------------------------------------------------------
// Adapter function
// ---------------------------------------------------------------------------

/// Convert a slice of [`TranscriptWord`]s from an ASR engine into a
/// [`TimedWordStream`] with [`WordStreamSource::RecordedAudio`].
///
/// Each output [`WordNode`] is assigned:
///
/// - `boundary_source`: [`BoundarySource::Whisper`] when both `start_ms` and
///   `end_ms` are present and form a valid interval (`start ≤ end`);
///   [`BoundarySource::Predicted`] otherwise.
/// - `commitment`: [`WordCommitment::Final`] for all words.
/// - `lexical_span`: byte-level span within the reconstructed transcript text
///   (words joined by single spaces).
/// - `timing_confidence`: forwarded directly from [`TranscriptWord::confidence`].
///
/// An empty `words` slice produces an empty stream, which is a valid value.
///
/// # Example
///
/// ```rust
/// use listenbury::word::export::{TranscriptWord, transcript_to_word_stream};
/// use listenbury::word::{WordStreamId, WordStreamSource};
///
/// let words = vec![
///     TranscriptWord { text: "hello".into(), start_ms: Some(100), end_ms: Some(500), confidence: Some(0.9) },
///     TranscriptWord { text: "world".into(), start_ms: Some(550), end_ms: Some(900), confidence: Some(0.85) },
/// ];
/// let stream = transcript_to_word_stream(WordStreamId(1), &words);
/// assert_eq!(stream.source, WordStreamSource::RecordedAudio);
/// assert_eq!(stream.words.len(), 2);
/// assert!(stream.words.iter().all(|w| w.timing.is_some()));
/// ```
pub fn transcript_to_word_stream(id: WordStreamId, words: &[TranscriptWord]) -> TimedWordStream {
    let mut byte_offset: usize = 0;

    let word_nodes: Vec<WordNode> = words
        .iter()
        .enumerate()
        .map(|(i, tw)| {
            let span_start = byte_offset;
            let span_end = span_start + tw.text.len();
            // Advance past this word and the trailing space separator.
            byte_offset = span_end + 1;

            let timing = match (tw.start_ms, tw.end_ms) {
                (Some(s), Some(e)) => WordTiming::new(s, e),
                _ => None,
            };

            let boundary_source = if timing.is_some() {
                BoundarySource::Whisper
            } else {
                BoundarySource::Predicted
            };

            WordNode {
                id: WordId(i as u64 + 1),
                text: tw.text.clone(),
                lexical_span: Some(TextSpan {
                    start: span_start,
                    end: span_end,
                }),
                timing,
                timing_confidence: tw.confidence,
                commitment: WordCommitment::Final,
                boundary_source,
                audio_ref: None,
            }
        })
        .collect();

    TimedWordStream {
        id,
        source: WordStreamSource::RecordedAudio,
        words: word_nodes,
    }
}

// ---------------------------------------------------------------------------
// Refinement seam
// ---------------------------------------------------------------------------

/// A seam for optional acoustic refinement of word boundaries in a
/// [`TimedWordStream`].
///
/// Implementations receive the raw audio frames alongside the stream produced
/// by [`transcript_to_word_stream`] and may update timing, confidence, and
/// [`BoundarySource`] metadata in-place.
///
/// The trait is intentionally left without a real implementation in the
/// initial release.  Use [`NoopWordBoundaryRefiner`] as a placeholder.
pub trait WordBoundaryRefiner {
    /// Refine word boundaries in `stream` using the provided `audio` frames.
    ///
    /// Implementations are free to modify `stream.words` in any way, including
    /// updating `timing`, `timing_confidence`, and `boundary_source` fields.
    fn refine(&self, audio: &[AudioFrame], stream: &mut TimedWordStream);
}

/// A no-op [`WordBoundaryRefiner`] that leaves the stream unchanged.
///
/// Use this as a placeholder until a real acoustic alignment implementation is
/// available.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopWordBoundaryRefiner;

impl WordBoundaryRefiner for NoopWordBoundaryRefiner {
    fn refine(&self, _audio: &[AudioFrame], _stream: &mut TimedWordStream) {
        // No-op: timing and boundaries are preserved as-is.
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // transcript_to_word_stream
    // ------------------------------------------------------------------

    /// Converting a typical Whisper transcript with word-level timing
    /// produces a RecordedAudio stream with all timing preserved.
    #[test]
    fn transcript_export_with_timing() {
        let words = vec![
            TranscriptWord {
                text: "hello".into(),
                start_ms: Some(100),
                end_ms: Some(500),
                confidence: Some(0.95),
            },
            TranscriptWord {
                text: "world".into(),
                start_ms: Some(550),
                end_ms: Some(900),
                confidence: Some(0.88),
            },
        ];

        let stream = transcript_to_word_stream(WordStreamId(1), &words);

        assert_eq!(stream.source, WordStreamSource::RecordedAudio);
        assert_eq!(stream.id, WordStreamId(1));
        assert_eq!(stream.words.len(), 2);

        // Timing preserved
        assert_eq!(
            stream.words[0].timing,
            Some(WordTiming {
                start_ms: 100,
                end_ms: 500
            })
        );
        assert_eq!(
            stream.words[1].timing,
            Some(WordTiming {
                start_ms: 550,
                end_ms: 900
            })
        );

        // Confidence forwarded
        assert_eq!(stream.words[0].timing_confidence, Some(0.95));
        assert_eq!(stream.words[1].timing_confidence, Some(0.88));

        // All words are Final with Whisper boundary source
        assert!(
            stream
                .words
                .iter()
                .all(|w| w.commitment == WordCommitment::Final)
        );
        assert!(
            stream
                .words
                .iter()
                .all(|w| w.boundary_source == BoundarySource::Whisper)
        );
    }

    /// Timing is correctly preserved from the TranscriptWord fields.
    #[test]
    fn timing_preservation() {
        let words = vec![TranscriptWord {
            text: "test".into(),
            start_ms: Some(200),
            end_ms: Some(600),
            confidence: None,
        }];

        let stream = transcript_to_word_stream(WordStreamId(2), &words);

        let w = &stream.words[0];
        let timing = w.timing.expect("timing should be present");
        assert_eq!(timing.start_ms, 200);
        assert_eq!(timing.end_ms, 600);
        assert_eq!(timing.duration_ms(), 400);
    }

    /// An empty word slice produces an empty stream without panicking.
    #[test]
    fn empty_transcript_produces_empty_stream() {
        let stream = transcript_to_word_stream(WordStreamId(3), &[]);

        assert_eq!(stream.source, WordStreamSource::RecordedAudio);
        assert_eq!(stream.id, WordStreamId(3));
        assert!(stream.words.is_empty());
    }

    /// Words without timing metadata get BoundarySource::Predicted and
    /// timing: None, while confidence is still forwarded if present.
    #[test]
    fn missing_timing_metadata_uses_predicted_boundary() {
        let words = vec![
            TranscriptWord {
                text: "no".into(),
                start_ms: None,
                end_ms: None,
                confidence: Some(0.7),
            },
            TranscriptWord {
                text: "timestamps".into(),
                start_ms: None,
                end_ms: None,
                confidence: None,
            },
        ];

        let stream = transcript_to_word_stream(WordStreamId(4), &words);

        assert_eq!(stream.words.len(), 2);
        assert!(stream.words.iter().all(|w| w.timing.is_none()));
        assert!(
            stream
                .words
                .iter()
                .all(|w| w.boundary_source == BoundarySource::Predicted)
        );

        // Confidence forwarded even when timing is absent
        assert_eq!(stream.words[0].timing_confidence, Some(0.7));
        assert_eq!(stream.words[1].timing_confidence, None);
    }

    /// Only start_ms present (no end_ms) — treated as missing timing.
    #[test]
    fn partial_timing_treated_as_missing() {
        let words = vec![TranscriptWord {
            text: "partial".into(),
            start_ms: Some(100),
            end_ms: None,
            confidence: None,
        }];

        let stream = transcript_to_word_stream(WordStreamId(5), &words);

        assert!(stream.words[0].timing.is_none());
        assert_eq!(stream.words[0].boundary_source, BoundarySource::Predicted);
    }

    /// Inverted timestamps (end < start) produce no timing entry and fall
    /// back to Predicted boundary source.
    #[test]
    fn inverted_timestamps_treated_as_missing() {
        let words = vec![TranscriptWord {
            text: "bad".into(),
            start_ms: Some(900),
            end_ms: Some(100),
            confidence: None,
        }];

        let stream = transcript_to_word_stream(WordStreamId(6), &words);

        assert!(stream.words[0].timing.is_none());
        assert_eq!(stream.words[0].boundary_source, BoundarySource::Predicted);
    }

    /// Lexical spans are byte-level offsets matching each word's position in
    /// the space-separated transcript text.
    #[test]
    fn lexical_spans_are_correct() {
        let words = vec![
            TranscriptWord {
                text: "abc".into(),
                start_ms: None,
                end_ms: None,
                confidence: None,
            },
            TranscriptWord {
                text: "de".into(),
                start_ms: None,
                end_ms: None,
                confidence: None,
            },
        ];

        let stream = transcript_to_word_stream(WordStreamId(7), &words);

        // "abc de" → "abc" at [0,3), "de" at [4,6)
        assert_eq!(
            stream.words[0].lexical_span,
            Some(TextSpan { start: 0, end: 3 })
        );
        assert_eq!(
            stream.words[1].lexical_span,
            Some(TextSpan { start: 4, end: 6 })
        );
    }

    // ------------------------------------------------------------------
    // NoopWordBoundaryRefiner
    // ------------------------------------------------------------------

    /// The no-op refiner leaves the stream unchanged.
    #[test]
    fn noop_refiner_does_not_modify_stream() {
        let words = vec![TranscriptWord {
            text: "unchanged".into(),
            start_ms: Some(0),
            end_ms: Some(400),
            confidence: Some(1.0),
        }];

        let mut stream = transcript_to_word_stream(WordStreamId(8), &words);
        let original = stream.clone();

        let refiner = NoopWordBoundaryRefiner;
        refiner.refine(&[], &mut stream);

        assert_eq!(stream, original);
    }
}
