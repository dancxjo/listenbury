//! Shared timed-word data model for Listenbury pipelines.
//!
//! This module provides the common substrate connecting word-by-word read-aloud
//! with speculative TTS commitment (see issue #59) and browser transcript
//! playback with word-level scrubbing (see issue #61).
//!
//! The key insight: a transcript and a synthesised utterance are the same kind
//! of object once they become timed words.  One is *discovered* from sound; the
//! other is *created* into sound.  Both need word identity, timing, confidence,
//! commitment state, and audio linkage.
//!
//! # Module layout
//!
//! - [`mod`] (this file) — re-exports all public types from submodules.
//! - [`stream`] — [`TimedWordStream`], [`WordNode`], and supporting enums.
//! - [`export`] — [`transcript_to_word_stream`] adapter and [`WordBoundaryRefiner`] trait.
//! - [`tts_export`] — [`generated_text_to_word_stream`], [`attach_heuristic_timing`],
//!   [`attach_audio_frame_timing`], and [`PlaybackCursor`] for outgoing/synthetic speech.

pub mod export;
pub mod stream;
pub mod tts_export;

pub use export::{
    transcript_to_energy_snapped_word_stream, transcript_to_word_stream,
    HeuristicAcousticWordBoundaryRefiner, NoopWordBoundaryRefiner, TranscriptWord,
    WordBoundaryRefiner,
};
pub use stream::{
    AudioRef, BoundarySource, PronunciationCandidateScore, PronunciationLookupStatus, TextSpan,
    TimedWordStream, WordCommitment, WordId, WordNode, WordPhoneSegmentation, WordPhoneSpan,
    WordPronunciation, WordStreamId, WordStreamSource, WordTiming,
};
