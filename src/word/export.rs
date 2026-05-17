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

const MIN_REFINED_WORD_DURATION_MS: u64 = 20;
const BOUNDARY_SEARCH_RADIUS_MS: u64 = 120;

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

/// First-pass heuristic acoustic boundary refiner.
///
/// Uses a local energy-minimum search around each Whisper-adjacent boundary to
/// nudge boundaries toward likely pause regions in the waveform.
#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicAcousticWordBoundaryRefiner;

impl WordBoundaryRefiner for HeuristicAcousticWordBoundaryRefiner {
    fn refine(&self, audio: &[AudioFrame], stream: &mut TimedWordStream) {
        let Some(energy_per_ms) = energy_profile_per_ms(audio) else {
            return;
        };
        if stream.words.len() < 2 {
            return;
        }

        for i in 0..stream.words.len().saturating_sub(1) {
            let left = &stream.words[i];
            let right = &stream.words[i + 1];

            if left.boundary_source != BoundarySource::Whisper
                || right.boundary_source != BoundarySource::Whisper
            {
                continue;
            }

            let (Some(left_timing), Some(right_timing)) = (left.timing, right.timing) else {
                continue;
            };
            if left_timing.start_ms > left_timing.end_ms
                || right_timing.start_ms > right_timing.end_ms
            {
                continue;
            }
            let min_boundary_ms = left_timing
                .start_ms
                .saturating_add(MIN_REFINED_WORD_DURATION_MS);
            let max_boundary_ms = right_timing
                .end_ms
                .saturating_sub(MIN_REFINED_WORD_DURATION_MS);
            if min_boundary_ms > max_boundary_ms {
                continue;
            }

            let original_boundary_ms = left_timing.end_ms.min(right_timing.start_ms);
            let search_start = original_boundary_ms
                .saturating_sub(BOUNDARY_SEARCH_RADIUS_MS)
                .max(min_boundary_ms);
            let search_end = original_boundary_ms
                .saturating_add(BOUNDARY_SEARCH_RADIUS_MS)
                .min(max_boundary_ms)
                .min(energy_per_ms.len().saturating_sub(1) as u64);
            if search_start > search_end {
                continue;
            }

            let mut best_boundary_ms = original_boundary_ms;
            let mut best_energy = f32::INFINITY;
            for boundary_ms in search_start..=search_end {
                let energy = smoothed_energy_at_ms(&energy_per_ms, boundary_ms as usize);
                if energy < best_energy {
                    best_energy = energy;
                    best_boundary_ms = boundary_ms;
                }
            }

            if best_boundary_ms == left_timing.end_ms && best_boundary_ms == right_timing.start_ms {
                continue;
            }

            let (left_slice, right_slice) = stream.words.split_at_mut(i + 1);
            let left_mut = &mut left_slice[i];
            let right_mut = &mut right_slice[0];

            left_mut.timing = WordTiming::new(left_timing.start_ms, best_boundary_ms);
            right_mut.timing = WordTiming::new(best_boundary_ms, right_timing.end_ms);
            left_mut.boundary_source = BoundarySource::RefinedAcoustic;
            right_mut.boundary_source = BoundarySource::RefinedAcoustic;
        }
    }
}

fn energy_profile_per_ms(audio: &[AudioFrame]) -> Option<Vec<f32>> {
    let mut ms_energies = Vec::<f32>::new();
    let mut ms_counts = Vec::<u32>::new();
    let mut frame_offset_ms = 0u64;

    for frame in audio {
        if frame.sample_rate_hz == 0 || frame.channels == 0 {
            continue;
        }
        let channels = frame.channels as usize;
        let per_channel_samples = frame.samples.len() / channels;
        if per_channel_samples == 0 {
            continue;
        }
        let frame_duration_ms = per_channel_samples as u64 * 1_000 / frame.sample_rate_hz as u64;

        for sample_idx in 0..per_channel_samples {
            let mut mono = 0.0f32;
            for ch in 0..channels {
                mono += frame.samples[sample_idx * channels + ch].abs();
            }
            mono /= channels as f32;

            let ms_idx =
                frame_offset_ms + (sample_idx as u64 * 1_000 / frame.sample_rate_hz as u64);
            let ms_idx = ms_idx as usize;
            if ms_idx >= ms_energies.len() {
                ms_energies.resize(ms_idx + 1, 0.0);
                ms_counts.resize(ms_idx + 1, 0);
            }
            ms_energies[ms_idx] += mono;
            ms_counts[ms_idx] += 1;
        }

        frame_offset_ms = frame_offset_ms.saturating_add(frame_duration_ms);
    }

    if ms_energies.is_empty() {
        return None;
    }

    for (energy, count) in ms_energies.iter_mut().zip(ms_counts.iter()) {
        if *count > 0 {
            *energy /= *count as f32;
        }
    }

    Some(ms_energies)
}

fn smoothed_energy_at_ms(energy_per_ms: &[f32], ms_idx: usize) -> f32 {
    let start = ms_idx.saturating_sub(5);
    let end = (ms_idx + 5).min(energy_per_ms.len().saturating_sub(1));
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for v in &energy_per_ms[start..=end] {
        sum += *v;
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::ExactTimestamp;

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

    // ------------------------------------------------------------------
    // HeuristicAcousticWordBoundaryRefiner
    // ------------------------------------------------------------------

    #[test]
    fn heuristic_refiner_moves_boundary_toward_local_silence() {
        let words = vec![
            TranscriptWord {
                text: "hello".into(),
                start_ms: Some(0),
                end_ms: Some(330),
                confidence: Some(0.9),
            },
            TranscriptWord {
                text: "world".into(),
                start_ms: Some(330),
                end_ms: Some(800),
                confidence: Some(0.9),
            },
        ];
        let mut stream = transcript_to_word_stream(WordStreamId(9), &words);

        let mut samples = vec![1.0f32; 900];
        samples[290..410].fill(0.0);
        let audio = vec![AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: 1_000,
            channels: 1,
            samples,
        }];

        HeuristicAcousticWordBoundaryRefiner.refine(&audio, &mut stream);

        let left = stream.words[0].timing.expect("left timing");
        let right = stream.words[1].timing.expect("right timing");
        assert_ne!(left.end_ms, 330);
        assert_eq!(left.end_ms, right.start_ms);
        assert!((290..=410).contains(&left.end_ms));
        assert_eq!(
            stream.words[0].boundary_source,
            BoundarySource::RefinedAcoustic
        );
        assert_eq!(
            stream.words[1].boundary_source,
            BoundarySource::RefinedAcoustic
        );
    }

    #[test]
    fn heuristic_refiner_preserves_non_whisper_provenance() {
        let words = vec![
            TranscriptWord {
                text: "one".into(),
                start_ms: Some(0),
                end_ms: Some(200),
                confidence: None,
            },
            TranscriptWord {
                text: "two".into(),
                start_ms: Some(200),
                end_ms: Some(400),
                confidence: None,
            },
        ];
        let mut stream = transcript_to_word_stream(WordStreamId(10), &words);
        stream.words[0].boundary_source = BoundarySource::Manual;
        let original = stream.clone();

        let audio = vec![AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: 1_000,
            channels: 1,
            samples: vec![0.0; 500],
        }];

        HeuristicAcousticWordBoundaryRefiner.refine(&audio, &mut stream);
        assert_eq!(stream, original);
    }

    #[test]
    fn heuristic_refiner_no_audio_is_noop() {
        let words = vec![
            TranscriptWord {
                text: "keep".into(),
                start_ms: Some(0),
                end_ms: Some(100),
                confidence: None,
            },
            TranscriptWord {
                text: "same".into(),
                start_ms: Some(100),
                end_ms: Some(200),
                confidence: None,
            },
        ];
        let mut stream = transcript_to_word_stream(WordStreamId(11), &words);
        let original = stream.clone();

        HeuristicAcousticWordBoundaryRefiner.refine(&[], &mut stream);
        assert_eq!(stream, original);
    }
}
