//! Prosody primitives for sung material.
//!
//! This module provides the lowest-level building blocks for expressing vocal
//! intent in Listenbury. The types here are semantic targets for a voice
//! renderer, not MIDI transport events or audio synthesis commands.
//!
//! The primary entry point is [`note_target::NoteTarget`], which fully
//! specifies a sung note: pitch (MIDI + microtonal offset), onset time,
//! duration, velocity, and articulation.

pub mod note_target;
pub mod pitch_curve;
pub mod singing;
pub mod syllable;
pub mod phonotactics;
pub mod syllabification;
pub mod vibrato;
