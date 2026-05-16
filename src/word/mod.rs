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

pub mod stream;

pub use stream::{
    AudioRef, BoundarySource, TextSpan, TimedWordStream, WordCommitment, WordId, WordNode,
    WordStreamId, WordStreamSource, WordTiming,
};
