//! Listenbury-native memory trace model.
//!
//! This module provides a lightweight, non-blocking seam between the real-time
//! conversational loop and any persistent memory backend.
//!
//! # Overview
//!
//! The fast audio/LLM/TTS loop emits [`MemoryTrace`] events that are handed
//! to a [`MemorySink`].  The sink is responsible for forwarding those traces
//! to a background worker or cold storage without ever blocking the caller.
//!
//! ```text
//! runtime event -> MemoryTrace -> MemorySink -> background worker / journal
//! ```
//!
//! Memory failures (a full channel, a crashed worker) must not break
//! conversation.  The hot path continues regardless.
//!
//! # Default configuration
//!
//! Use [`NoopMemorySink`] when no persistent backend is required.  Replace it
//! with [`ChannelMemorySink`] when a background worker should process traces,
//! or with [`JournalMemorySink`] to write traces directly to a JSONL file.
//!
//! [`MemoryTrace`]: trace::MemoryTrace
//! [`MemorySink`]: sink::MemorySink
//! [`NoopMemorySink`]: sink::NoopMemorySink
//! [`ChannelMemorySink`]: sink::ChannelMemorySink
//! [`JournalMemorySink`]: journal::JournalMemorySink

pub mod journal;
pub mod sink;
pub mod trace;

pub use journal::{JournalEntry, JournalMemorySink, MemoryJournal};
pub use sink::{ChannelMemorySink, MemorySink, NoopMemorySink};
pub use trace::{MemoryTrace, SpeakerRole};
