//! Core timed-word stream types.
//!
//! Both incoming (recorded/transcribed) speech and outgoing (synthetic) speech
//! can be represented as a [`TimedWordStream`].  The [`WordStreamSource`] enum
//! distinguishes the origin, while [`WordCommitment`] tracks how confident or
//! stable each word's position in the timeline is.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Identifier newtypes
// ---------------------------------------------------------------------------

/// Unique identifier for a [`TimedWordStream`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WordStreamId(pub u64);

/// Unique identifier for a single [`WordNode`] within a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WordId(pub u64);

// ---------------------------------------------------------------------------
// Top-level stream
// ---------------------------------------------------------------------------

/// A sequence of timed words that originated from a single audio source or
/// text generation event.
///
/// The same type is used for:
/// - Recorded audio transcripts (e.g. Whisper ASR output).
/// - Live ASR hypothesis streams.
/// - Generated text that has been expanded into a word sequence.
/// - Synthetic speech words produced by a TTS engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimedWordStream {
    /// Stable identifier for this stream.
    pub id: WordStreamId,
    /// Where the words came from.
    pub source: WordStreamSource,
    /// The ordered list of words in this stream.
    pub words: Vec<WordNode>,
}

impl TimedWordStream {
    /// Create a new, empty stream with the given identifier and source.
    pub fn new(id: WordStreamId, source: WordStreamSource) -> Self {
        Self {
            id,
            source,
            words: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Source classification
// ---------------------------------------------------------------------------

/// Describes where the words in a [`TimedWordStream`] originated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WordStreamSource {
    /// Audio was recorded and subsequently transcribed (e.g. via Whisper ASR).
    RecordedAudio,
    /// Words came from a live ASR hypothesis that may still be revised.
    LiveAsr,
    /// Words were produced by a text generator (e.g. an LLM) without audio.
    GeneratedText,
    /// Words were synthesised into audio by a TTS engine (e.g. Piper).
    SyntheticSpeech,
}

// ---------------------------------------------------------------------------
// Word node
// ---------------------------------------------------------------------------

/// A single word within a [`TimedWordStream`], together with all metadata
/// needed to align it with audio or display it in a transcript UI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WordNode {
    /// Unique identifier for this word within its stream.
    pub id: WordId,
    /// The textual form of the word (may include surrounding punctuation).
    pub text: String,
    /// Position of this word in the original source text, if available.
    pub lexical_span: Option<TextSpan>,
    /// Aligned start/end timestamps in the audio timeline, if available.
    pub timing: Option<WordTiming>,
    /// How confident we are in the timing alignment (0.0 – 1.0).
    pub timing_confidence: Option<f32>,
    /// Commitment state: how stable or final this word's position is.
    pub commitment: WordCommitment,
    /// How the word boundary was detected or assigned.
    pub boundary_source: BoundarySource,
    /// Reference to the audio segment that corresponds to this word, if any.
    pub audio_ref: Option<AudioRef>,
    /// Pronunciation metadata for this word, if available.
    ///
    /// When present, the [`WordPronunciation`] carries the phoneme sequence
    /// and provenance information needed to render phoneme-level annotations
    /// in the timeline UI.  Absence means pronunciation has not yet been
    /// resolved for this word.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pronunciation: Option<WordPronunciation>,
}

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/// A byte-level span `[start, end)` within a UTF-8 text string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextSpan {
    /// Inclusive start byte offset.
    pub start: usize,
    /// Exclusive end byte offset.
    pub end: usize,
}

/// Start/end timestamps for a word in the audio timeline.
///
/// Invariant: `start_ms <= end_ms`.  Use [`WordTiming::new`] to construct
/// validated instances.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WordTiming {
    /// Start of the word in milliseconds from the beginning of the stream.
    pub start_ms: u64,
    /// End of the word in milliseconds from the beginning of the stream.
    pub end_ms: u64,
}

impl WordTiming {
    /// Create a new `WordTiming`, returning `None` if `end_ms < start_ms`.
    pub fn new(start_ms: u64, end_ms: u64) -> Option<Self> {
        if end_ms >= start_ms {
            Some(Self { start_ms, end_ms })
        } else {
            None
        }
    }

    /// Duration of the word in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.end_ms - self.start_ms
    }
}

/// Describes how stable or actionable a word's position in the timeline is.
///
/// The states progress roughly from speculative toward final, though
/// cancellation is always possible from any non-final state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WordCommitment {
    /// The word exists only as a hypothesis; timing and text may change.
    Hypothetical,
    /// The word text is stable but has not yet been prepared for playback.
    StableText,
    /// The word has been scheduled for synthesis/playback but not yet played.
    Prepared,
    /// Audio for this word is ready and playback is imminent.
    Playable,
    /// The word is currently being played back.
    Played,
    /// The word has been played and the result is confirmed.
    Final,
    /// The word was confirmed by a later, higher-quality pass over broader context.
    Confirmed,
    /// The word was abandoned before it could be played (e.g. interruption).
    Cancelled,
}

/// Indicates the algorithm or process that determined the word boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundarySource {
    /// Boundary came from Whisper's internal word-timestamp output.
    Whisper,
    /// Boundary was refined by acoustic forced-alignment.
    RefinedAcoustic,
    /// Boundary was estimated/predicted (e.g. by duration models).
    Predicted,
    /// Boundary was set by a playback cursor during live TTS output.
    PlaybackCursor,
    /// Boundary was set manually (e.g. by a human editor or test fixture).
    Manual,
}

/// A reference to a slice of audio data associated with a word.
///
/// Keeps the model serialisable without embedding raw audio bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioRef {
    /// Opaque identifier for the audio buffer (e.g. a UUID or path fragment).
    pub buffer_id: String,
    /// Byte offset of the first sample in the buffer.
    pub byte_offset: u64,
    /// Byte length of the slice.
    pub byte_len: u64,
}

// ---------------------------------------------------------------------------
// Pronunciation metadata
// ---------------------------------------------------------------------------

/// Indicates how a word's pronunciation was resolved during lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PronunciationLookupStatus {
    /// Word found in the dictionary with an exact (case-insensitive) match.
    Exact,
    /// Word found after normalization (e.g. punctuation stripping).
    Normalized,
    /// Word not found; a grapheme-to-phoneme guess was used.
    Guessed,
    /// Word not found and no pronunciation is available.
    Missing,
}

/// Pronunciation metadata for a word, derived from a pronunciation dictionary.
///
/// Carries the phoneme sequence and enough provenance information for the UI
/// to clearly label that these are dictionary-derived projections, not
/// acoustically measured timings.
///
/// # JSON shape
///
/// ```json
/// {
///   "source": "cmudict",
///   "lookup": "THREE",
///   "phonemes": ["TH", "R", "IY1"],
///   "stressPattern": "1",
///   "status": "exact"
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WordPronunciation {
    /// Pronunciation source identifier, e.g. `"cmudict"`.
    pub source: String,
    /// Normalized word form used for the dictionary lookup, e.g. `"THREE"`.
    pub lookup: String,
    /// ARPAbet phoneme symbols for the primary pronunciation.
    ///
    /// Stress digits (`0`, `1`, `2`) are preserved where present so the UI
    /// can convey syllable stress.
    pub phonemes: Vec<String>,
    /// Compact stress-digit string derived from vowel phonemes (e.g. `"1"`
    /// for primary stress).  Empty for all-consonant sequences.
    pub stress_pattern: String,
    /// How the pronunciation was resolved.
    pub status: PronunciationLookupStatus,
    /// Optional backend-computed phone segmentation metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phone_segmentation: Option<WordPhoneSegmentation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WordPhoneSegmentation {
    pub word: String,
    pub word_start_ms: u64,
    pub word_end_ms: u64,
    pub pronunciation: Vec<String>,
    pub candidate_pronunciation_id: Option<String>,
    #[serde(default)]
    pub pronunciation_scores: Vec<PronunciationCandidateScore>,
    #[serde(default)]
    pub phone_spans: Vec<WordPhoneSpan>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PronunciationCandidateScore {
    pub id: String,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WordPhoneSpan {
    pub source_symbol: String,
    pub phone: String,
    pub phone_class: String,
    pub prior_start_ms: u64,
    pub prior_end_ms: u64,
    pub start_ms: u64,
    pub end_ms: u64,
    pub resolved_start_ms: u64,
    pub resolved_end_ms: u64,
    pub method: String,
    pub confidence: f32,
    #[serde(default)]
    pub features_used: Vec<String>,
    pub boundary_uncertainty_ms: u64,
    pub candidate_pronunciation_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ids() -> (WordStreamId, WordId, WordId, WordId) {
        (WordStreamId(1), WordId(1), WordId(2), WordId(3))
    }

    /// Verify that a recorded-audio transcript stream can be constructed with
    /// full timing and confidence information (as produced by Whisper ASR).
    #[test]
    fn recorded_transcript_word_stream() {
        let (stream_id, w1, w2, w3) = make_ids();

        let stream = TimedWordStream {
            id: stream_id,
            source: WordStreamSource::RecordedAudio,
            words: vec![
                WordNode {
                    id: w1,
                    text: "hello".to_string(),
                    lexical_span: Some(TextSpan { start: 0, end: 5 }),
                    timing: Some(WordTiming {
                        start_ms: 200,
                        end_ms: 600,
                    }),
                    timing_confidence: Some(0.95),
                    commitment: WordCommitment::Final,
                    boundary_source: BoundarySource::Whisper,
                    audio_ref: None,
                    pronunciation: None,
                },
                WordNode {
                    id: w2,
                    text: "world".to_string(),
                    lexical_span: Some(TextSpan { start: 6, end: 11 }),
                    timing: Some(WordTiming {
                        start_ms: 650,
                        end_ms: 1050,
                    }),
                    timing_confidence: Some(0.91),
                    commitment: WordCommitment::Final,
                    boundary_source: BoundarySource::Whisper,
                    audio_ref: None,
                    pronunciation: None,
                },
                WordNode {
                    id: w3,
                    text: "today".to_string(),
                    lexical_span: Some(TextSpan { start: 12, end: 17 }),
                    timing: Some(WordTiming {
                        start_ms: 1100,
                        end_ms: 1500,
                    }),
                    timing_confidence: Some(0.88),
                    commitment: WordCommitment::Final,
                    boundary_source: BoundarySource::Whisper,
                    audio_ref: None,
                    pronunciation: None,
                },
            ],
        };

        assert_eq!(stream.source, WordStreamSource::RecordedAudio);
        assert_eq!(stream.words.len(), 3);
        assert!(stream.words.iter().all(|w| w.timing.is_some()));
        assert!(stream
            .words
            .iter()
            .all(|w| w.commitment == WordCommitment::Final));
    }

    /// Verify that a generated-text stream can be constructed *without* timing
    /// (words exist as stable text but no audio has been produced yet).
    #[test]
    fn generated_text_word_stream_without_timing() {
        let stream_id = WordStreamId(2);

        let words: Vec<WordNode> = ["sure", "I", "can", "help"]
            .iter()
            .enumerate()
            .map(|(i, &word)| {
                let start = if i == 0 {
                    0
                } else {
                    // rough byte offset estimate for test purposes
                    ["sure ", "I ", "can ", "help"][..i]
                        .iter()
                        .map(|s| s.len())
                        .sum()
                };
                let end = start + word.len();
                WordNode {
                    id: WordId(i as u64 + 1),
                    text: word.to_string(),
                    lexical_span: Some(TextSpan { start, end }),
                    timing: None,
                    timing_confidence: None,
                    commitment: WordCommitment::StableText,
                    boundary_source: BoundarySource::Manual,
                    audio_ref: None,
                    pronunciation: None,
                }
            })
            .collect();

        let stream = TimedWordStream {
            id: stream_id,
            source: WordStreamSource::GeneratedText,
            words,
        };

        assert_eq!(stream.source, WordStreamSource::GeneratedText);
        assert_eq!(stream.words.len(), 4);
        assert!(stream.words.iter().all(|w| w.timing.is_none()));
        assert!(stream
            .words
            .iter()
            .all(|w| w.commitment == WordCommitment::StableText));
    }

    /// Verify that a synthetic-speech stream can be constructed with playback
    /// timing (words aligned to a TTS audio buffer via a PlaybackCursor).
    #[test]
    fn synthetic_speech_word_stream_with_playback_timing() {
        let stream_id = WordStreamId(3);

        let buffer_id = "tts-buffer-001".to_string();
        let words: Vec<WordNode> = [
            ("hello", 0u64, 400u64, 0u64, 3200u64),
            ("there", 450, 900, 3200, 7200),
        ]
        .iter()
        .enumerate()
        .map(
            |(i, &(word, start_ms, end_ms, byte_offset, byte_len))| WordNode {
                id: WordId(i as u64 + 1),
                text: word.to_string(),
                lexical_span: None,
                timing: Some(WordTiming { start_ms, end_ms }),
                timing_confidence: Some(1.0),
                commitment: WordCommitment::Playable,
                boundary_source: BoundarySource::PlaybackCursor,
                audio_ref: Some(AudioRef {
                    buffer_id: buffer_id.clone(),
                    byte_offset,
                    byte_len,
                }),
                pronunciation: None,
            },
        )
        .collect();

        let stream = TimedWordStream {
            id: stream_id,
            source: WordStreamSource::SyntheticSpeech,
            words,
        };

        assert_eq!(stream.source, WordStreamSource::SyntheticSpeech);
        assert_eq!(stream.words.len(), 2);
        assert!(stream.words.iter().all(|w| w.timing.is_some()));
        assert!(stream.words.iter().all(|w| w.audio_ref.is_some()));
        assert_eq!(
            stream.words[0].audio_ref.as_ref().unwrap().buffer_id,
            "tts-buffer-001"
        );
    }

    /// Verify round-trip serialisation via serde_json.
    #[test]
    fn timed_word_stream_serialises_and_deserialises() {
        let stream = TimedWordStream {
            id: WordStreamId(42),
            source: WordStreamSource::LiveAsr,
            words: vec![WordNode {
                id: WordId(1),
                text: "test".to_string(),
                lexical_span: None,
                timing: Some(WordTiming {
                    start_ms: 100,
                    end_ms: 300,
                }),
                timing_confidence: Some(0.80),
                commitment: WordCommitment::Hypothetical,
                boundary_source: BoundarySource::Whisper,
                audio_ref: None,
                pronunciation: None,
            }],
        };

        let json = serde_json::to_string(&stream).expect("serialisation failed");
        let restored: TimedWordStream =
            serde_json::from_str(&json).expect("deserialisation failed");

        assert_eq!(stream, restored);
    }

    /// Verify `TimedWordStream::new` produces an empty stream.
    #[test]
    fn new_stream_is_empty() {
        let stream = TimedWordStream::new(WordStreamId(99), WordStreamSource::SyntheticSpeech);
        assert_eq!(stream.id, WordStreamId(99));
        assert_eq!(stream.source, WordStreamSource::SyntheticSpeech);
        assert!(stream.words.is_empty());
    }

    /// Verify `WordTiming::new` validates the start ≤ end invariant.
    #[test]
    fn word_timing_new_validates_invariant() {
        assert!(WordTiming::new(100, 300).is_some());
        assert!(WordTiming::new(100, 100).is_some()); // zero-duration is allowed
        assert!(WordTiming::new(300, 100).is_none()); // end before start is rejected
        let t = WordTiming::new(200, 600).unwrap();
        assert_eq!(t.duration_ms(), 400);
    }
}
