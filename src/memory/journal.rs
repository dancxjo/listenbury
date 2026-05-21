//! Append-only warm memory journal.
//!
//! The journal persists [`MemoryTrace`] events to a [JSON Lines] file on disk.
//! It sits between the fast in-process runtime state and slower cold-storage
//! systems such as vector or graph databases.
//!
//! # Design
//!
//! ```text
//! runtime event -> MemoryTrace -> warm journal (JSONL)
//!                                        |
//!                              (background ingestion)
//!                                        |
//!                               cold storage / index
//! ```
//!
//! Each line of the file is one [`JournalEntry`] serialised as JSON.  The
//! format is intentionally simple so that other tools can consume it without
//! requiring Listenbury itself.
//!
//! # Resilience
//!
//! [`MemoryJournal::replay`] silently skips any line that cannot be parsed,
//! so a corrupt write (power loss, partial flush) never prevents replay of
//! the surrounding entries.
//!
//! [JSON Lines]: https://jsonlines.org/

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Context;
use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::memory::trace::MemoryTrace;
use crate::runtime_event::RuntimeEvent;

// ---------------------------------------------------------------------------
// JournalEntry
// ---------------------------------------------------------------------------

/// A single persisted entry in the JSONL journal.
///
/// The `timestamp` field records when the entry was *written* to the journal.
/// Each [`MemoryTrace`] variant also carries its own `occurred_at` field that
/// records when the event was *observed* by the runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEntry {
    /// ISO 8601 wall-clock timestamp of when this entry was appended.
    pub timestamp: String,
    /// The trace event that was recorded.
    pub trace: MemoryTrace,
    /// Canonical runtime envelope for trace correlation/replay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_event: Option<RuntimeEvent>,
}

// ---------------------------------------------------------------------------
// MemoryJournal
// ---------------------------------------------------------------------------

/// Append-only JSONL memory journal.
///
/// Open an existing journal (or create a new one) with [`MemoryJournal::open`],
/// then call [`MemoryJournal::append`] to record traces.  Use
/// [`MemoryJournal::replay`] to read all previously written entries.
///
/// The inner [`File`] is guarded by a [`Mutex`] so that a single journal
/// instance can be shared across threads via an [`Arc`] wrapper without
/// requiring `&mut self` on the caller side.
///
/// # Default path
///
/// `listenbury_data/memory/events.jsonl`
pub struct MemoryJournal {
    file: Mutex<File>,
    pub path: PathBuf,
}

impl MemoryJournal {
    /// The default path used when no explicit path is supplied.
    pub const DEFAULT_PATH: &'static str = "listenbury_data/memory/events.jsonl";

    /// Open (or create) a journal at `path`.
    ///
    /// All parent directories are created automatically.
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create journal directory {:?}", parent))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("open journal at {:?}", path))?;
        Ok(Self {
            file: Mutex::new(file),
            path,
        })
    }

    /// Append a single [`MemoryTrace`] to the journal.
    ///
    /// The call acquires an internal mutex and flushes one JSON line.  Callers
    /// on the hot audio/LLM path should dispatch this to a background thread
    /// rather than calling it directly.
    pub fn append(&self, trace: &MemoryTrace) -> anyhow::Result<()> {
        let entry = JournalEntry {
            timestamp: Utc::now().to_rfc3339(),
            trace: trace.clone(),
            runtime_event: Some(RuntimeEvent::from_memory_trace(trace)),
        };
        let mut line = serde_json::to_string(&entry).context("serialize journal entry")?;
        line.push('\n');
        let mut file = self.file.lock().expect("journal mutex poisoned");
        file.write_all(line.as_bytes())
            .context("write journal entry")?;
        file.flush().context("flush journal")?;
        Ok(())
    }

    /// Replay all valid entries from the journal file at `path`.
    ///
    /// Lines that cannot be parsed (truncated writes, encoding errors) are
    /// silently skipped so that a single corrupt entry does not prevent the
    /// rest of the journal from being read.
    ///
    /// Returns an empty `Vec` when the file does not exist.
    pub fn replay(path: impl AsRef<Path>) -> anyhow::Result<Vec<JournalEntry>> {
        let path = path.as_ref();
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => {
                return Err(e).with_context(|| format!("open journal for replay at {:?}", path));
            }
        };
        let reader = BufReader::new(file);
        let entries = reader
            .lines()
            .filter_map(|line| {
                let line = line.ok()?;
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    return None;
                }
                serde_json::from_str(trimmed).ok()
            })
            .collect();
        Ok(entries)
    }
}

// MemoryJournal is Send because the inner Mutex<File> is Send.
// It is Sync because all mutable access is serialised through the Mutex.
unsafe impl Send for MemoryJournal {}
unsafe impl Sync for MemoryJournal {}

// ---------------------------------------------------------------------------
// JournalMemorySink — MemorySink adapter for MemoryJournal
// ---------------------------------------------------------------------------

/// A [`crate::memory::sink::MemorySink`] that appends every trace to a
/// [`MemoryJournal`] through an `Arc`.
///
/// Append errors are logged via `tracing::warn` and never propagate to
/// the caller so that a journal write failure never breaks conversation.
#[derive(Clone)]
pub struct JournalMemorySink {
    journal: Arc<MemoryJournal>,
}

impl JournalMemorySink {
    /// Wrap an existing [`MemoryJournal`] in an `Arc` and return the sink.
    pub fn new(journal: Arc<MemoryJournal>) -> Self {
        Self { journal }
    }

    /// Open a new journal at `path` and return the sink together with an `Arc`
    /// handle to the journal for inspection / replay.
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<(Self, Arc<MemoryJournal>)> {
        let journal = Arc::new(MemoryJournal::open(path)?);
        Ok((Self::new(Arc::clone(&journal)), journal))
    }
}

impl crate::memory::sink::MemorySink for JournalMemorySink {
    fn submit(&self, trace: MemoryTrace) {
        if let Err(e) = self.journal.append(&trace) {
            tracing::warn!("journal append failed: {e:#}");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::sink::MemorySink as _;
    use crate::memory::trace::SpeakerRole;
    use crate::time::ExactTimestamp;

    fn sample_trace() -> MemoryTrace {
        MemoryTrace::ConversationTurnFinalized {
            speaker: SpeakerRole::UnknownVoice { ordinal: 1 },
            text: "Can you hear me?".to_string(),
            occurred_at: ExactTimestamp::now(),
        }
    }

    fn temp_path() -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "listenbury_journal_test_{}.jsonl",
            uuid::Uuid::new_v4()
        ));
        p
    }

    // -------------------------------------------------------------------------
    // append
    // -------------------------------------------------------------------------

    #[test]
    fn append_creates_file_and_writes_a_line() {
        let path = temp_path();
        let journal = MemoryJournal::open(&path).expect("open journal");
        journal.append(&sample_trace()).expect("append");

        let content = std::fs::read_to_string(&path).expect("read file");
        assert!(!content.trim().is_empty(), "file should not be empty");
        assert_eq!(
            content.lines().count(),
            1,
            "one trace should produce one line"
        );
    }

    #[test]
    fn append_multiple_traces_writes_multiple_lines() {
        let path = temp_path();
        let journal = MemoryJournal::open(&path).expect("open journal");
        for _ in 0..5 {
            journal.append(&sample_trace()).expect("append");
        }
        let content = std::fs::read_to_string(&path).expect("read file");
        assert_eq!(content.lines().count(), 5);
    }

    #[test]
    fn appended_line_is_valid_json_with_timestamp_and_trace() {
        let path = temp_path();
        let journal = MemoryJournal::open(&path).expect("open journal");
        journal.append(&sample_trace()).expect("append");

        let line = std::fs::read_to_string(&path).expect("read");
        let value: serde_json::Value =
            serde_json::from_str(line.trim()).expect("must be valid JSON");
        assert!(value.get("timestamp").is_some(), "must have timestamp");
        assert!(value.get("trace").is_some(), "must have trace");
        assert!(
            value.get("runtime_event").is_some(),
            "must have canonical runtime_event envelope"
        );
        assert_eq!(
            value["runtime_event"]["source"],
            serde_json::Value::String("memory_ingestion".to_string())
        );
    }

    // -------------------------------------------------------------------------
    // replay
    // -------------------------------------------------------------------------

    #[test]
    fn replay_returns_entries_in_order() {
        let path = temp_path();
        let journal = MemoryJournal::open(&path).expect("open journal");
        let texts = ["alpha", "beta", "gamma"];
        for t in &texts {
            journal
                .append(&MemoryTrace::ConversationTurnFinalized {
                    speaker: SpeakerRole::UnknownVoice { ordinal: 1 },
                    text: t.to_string(),
                    occurred_at: ExactTimestamp::now(),
                })
                .expect("append");
        }

        let entries = MemoryJournal::replay(&path).expect("replay");
        assert_eq!(entries.len(), 3);
        for (entry, expected) in entries.iter().zip(texts.iter()) {
            match &entry.trace {
                MemoryTrace::ConversationTurnFinalized { text, .. } => {
                    assert_eq!(text, expected);
                }
                other => panic!("unexpected variant: {:?}", other),
            }
        }
    }

    // -------------------------------------------------------------------------
    // empty journal
    // -------------------------------------------------------------------------

    #[test]
    fn replay_of_empty_journal_returns_empty_vec() {
        let path = temp_path();
        // Create an empty file.
        std::fs::write(&path, b"").expect("write empty file");

        let entries = MemoryJournal::replay(&path).expect("replay");
        assert!(entries.is_empty(), "empty file should produce no entries");
    }

    #[test]
    fn replay_of_nonexistent_file_returns_empty_vec() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "listenbury_journal_nonexistent_{}.jsonl",
            uuid::Uuid::new_v4()
        ));
        // Do NOT create the file.
        let entries = MemoryJournal::replay(&path).expect("replay should not error");
        assert!(entries.is_empty());
    }

    // -------------------------------------------------------------------------
    // corrupt line handling
    // -------------------------------------------------------------------------

    #[test]
    fn corrupt_lines_are_skipped_during_replay() {
        let path = temp_path();
        // Write a valid entry, a corrupt line, then another valid entry.
        let journal = MemoryJournal::open(&path).expect("open journal");
        journal.append(&sample_trace()).expect("append first");

        // Manually inject a corrupt line.
        use std::io::Write as _;
        let mut f = OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("re-open");
        writeln!(f, "{{not valid json}}").expect("inject corrupt line");

        journal.append(&sample_trace()).expect("append last");

        let entries = MemoryJournal::replay(&path).expect("replay");
        assert_eq!(
            entries.len(),
            2,
            "corrupt line should be skipped; two valid entries expected"
        );
    }

    // -------------------------------------------------------------------------
    // concurrent append safety
    // -------------------------------------------------------------------------

    #[test]
    fn concurrent_appends_do_not_corrupt_the_journal() {
        use std::thread;

        let path = temp_path();
        let journal = Arc::new(MemoryJournal::open(&path).expect("open journal"));

        let thread_count = 8;
        let appends_per_thread = 16;

        let handles: Vec<_> = (0..thread_count)
            .map(|_| {
                let j = Arc::clone(&journal);
                let trace = sample_trace();
                thread::spawn(move || {
                    for _ in 0..appends_per_thread {
                        j.append(&trace).expect("concurrent append");
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        let entries = MemoryJournal::replay(&path).expect("replay");
        assert_eq!(
            entries.len(),
            thread_count * appends_per_thread,
            "all concurrent entries should be present"
        );
    }

    // -------------------------------------------------------------------------
    // JournalMemorySink
    // -------------------------------------------------------------------------

    #[test]
    fn journal_memory_sink_appends_via_submit() {
        let path = temp_path();
        let (sink, _journal) = JournalMemorySink::open(&path).expect("open sink");
        sink.submit(sample_trace());
        sink.submit(sample_trace());

        let entries = MemoryJournal::replay(&path).expect("replay");
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn journal_memory_sink_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<JournalMemorySink>();
    }
}
