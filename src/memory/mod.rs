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

pub mod embed;
pub mod journal;
pub mod known_voice;
pub mod neo4j;
pub mod qdrant;
pub mod sink;
pub mod trace;
pub mod worker;

pub use embed::EmbeddingProvider;
pub use journal::{JournalEntry, JournalMemorySink, MemoryJournal};
pub use known_voice::{
    DEFAULT_KNOWN_VOICE_REGISTRY_PATH, DeterministicKnownVoiceEmbeddingProvider,
    KNOWN_VOICE_EMBEDDING_BACKEND, KNOWN_VOICE_LOCALITY, KNOWN_VOICE_QDRANT_COLLECTION,
    KnownVoiceEmbeddingProvider, KnownVoiceMemoryStore, QdrantKnownVoiceMatcher,
};
pub use neo4j::{
    Neo4jHttpStore, Neo4jNode, Neo4jRelationship, Neo4jStore, Neo4jTraceWrite, Neo4jWriteResult,
    trace_write_for,
};
pub use qdrant::{
    DEFAULT_QDRANT_COLLECTION, PICTURE_QDRANT_COLLECTION, QdrantHttpStore, QdrantPoint,
    QdrantSearchHit, QdrantStore, VOICE_QDRANT_COLLECTION, VectorDocument,
    vector_documents_for_trace,
};
pub use sink::{ChannelMemorySink, MemorySink, NoopMemorySink};
pub use trace::{
    MemoryEntityMention, MemoryGraphNodeFieldUpdate, MemoryImageVector, MemorySceneRef,
    MemoryTrace, MemoryVoiceVector, SpeakerRole,
};
pub use worker::{ColdMemoryWorker, ColdMemoryWorkerConfig, ColdMemoryWorkerReport};
