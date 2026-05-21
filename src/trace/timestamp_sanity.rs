//! Timestamp normalization and ordering helpers for live traces.
//!
//! The viewer and exporters should operate in one canonical elapsed-ms domain.
//! Producers can still emit local monotonic timestamps, approximate wall time,
//! sequence ids, and clock sync beacons; this module folds that metadata into a
//! stable trace before downstream span construction.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::live_trace::{LiveTraceEvent, TraceClockSync};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClockMapping {
    pub a: f64,
    pub b: f64,
}

impl ClockMapping {
    pub fn from_sync(sync: &TraceClockSync) -> Self {
        let a = sync
            .drift_ppm
            .map(|ppm| 1.0 + ppm / 1_000_000.0)
            .unwrap_or(1.0);
        let b = sync.canon_elapsed_ms as f64 - a * sync.local_mono as f64;
        Self { a, b }
    }

    pub fn from_sync_pair(first: &TraceClockSync, last: &TraceClockSync) -> Self {
        if first.local_mono == last.local_mono {
            return Self::from_sync(last);
        }
        let local_delta = last.local_mono as f64 - first.local_mono as f64;
        let canon_delta = last.canon_elapsed_ms as f64 - first.canon_elapsed_ms as f64;
        let a = canon_delta / local_delta;
        let b = first.canon_elapsed_ms as f64 - a * first.local_mono as f64;
        Self { a, b }
    }

    pub fn to_canonical_elapsed_ms(self, local_mono: u64) -> u64 {
        let mapped = self.a * local_mono as f64 + self.b;
        if !mapped.is_finite() || mapped <= 0.0 {
            0
        } else if mapped >= u64::MAX as f64 {
            u64::MAX
        } else {
            mapped.round() as u64
        }
    }
}

pub fn normalize_live_trace_events(events: &[LiveTraceEvent]) -> Vec<LiveTraceEvent> {
    let mappings = clock_mappings(events);
    let session_start_ns = inferred_session_start_ns(events);
    let mut normalized = events
        .iter()
        .cloned()
        .map(|event| normalize_event(event, &mappings, session_start_ns))
        .enumerate()
        .collect::<Vec<_>>();

    normalized.sort_by(|(left_index, left), (right_index, right)| {
        normalized_event_order(left, *left_index, right, *right_index)
    });
    normalized.into_iter().map(|(_, event)| event).collect()
}

fn normalize_event(
    mut event: LiveTraceEvent,
    mappings: &HashMap<String, ClockMapping>,
    session_start_ns: Option<u64>,
) -> LiveTraceEvent {
    let observed_elapsed_ms = event.elapsed_ms;
    let observed_unix_ns = event.t_unix_ns;
    let canonical_elapsed_ms = event
        .normalized_elapsed_ms
        .or_else(|| {
            let local_mono = event.ts_local_monotonic?;
            let mapping = mappings.get(&emitter_key(&event))?;
            Some(mapping.to_canonical_elapsed_ms(local_mono))
        })
        .unwrap_or(event.elapsed_ms);

    if canonical_elapsed_ms != observed_elapsed_ms {
        event.observed_elapsed_ms = Some(observed_elapsed_ms);
        event.elapsed_ms = canonical_elapsed_ms;
    }

    let canonical_unix_ns = event.normalized_unix_ns.or_else(|| {
        let start = session_start_ns?;
        Some(start.saturating_add(canonical_elapsed_ms.saturating_mul(1_000_000)))
    });
    if let Some(canonical_unix_ns) = canonical_unix_ns {
        if canonical_unix_ns != observed_unix_ns {
            event.observed_unix_ns = Some(observed_unix_ns);
            event.t_unix_ns = canonical_unix_ns;
        }
        event.normalized_unix_ns = Some(canonical_unix_ns);
    }
    event.normalized_elapsed_ms = Some(canonical_elapsed_ms);
    event.refresh_runtime_event();
    event
}

fn clock_mappings(events: &[LiveTraceEvent]) -> HashMap<String, ClockMapping> {
    let mut syncs_by_emitter = HashMap::<String, Vec<TraceClockSync>>::new();
    for event in events {
        let Some(sync) = event.clock_sync.clone() else {
            continue;
        };
        syncs_by_emitter
            .entry(emitter_key(event))
            .or_default()
            .push(sync);
    }

    syncs_by_emitter
        .into_iter()
        .map(|(emitter, mut syncs)| {
            syncs.sort_by_key(|sync| sync.local_mono);
            let mapping = match (syncs.first(), syncs.last()) {
                (Some(first), Some(last)) if syncs.len() > 1 => {
                    ClockMapping::from_sync_pair(first, last)
                }
                (_, Some(sync)) => ClockMapping::from_sync(sync),
                _ => ClockMapping { a: 1.0, b: 0.0 },
            };
            (emitter, mapping)
        })
        .collect()
}

fn inferred_session_start_ns(events: &[LiveTraceEvent]) -> Option<u64> {
    events
        .iter()
        .map(|event| {
            let unix_ns = event.normalized_unix_ns.unwrap_or(event.t_unix_ns);
            let elapsed_ms = event.normalized_elapsed_ms.unwrap_or(event.elapsed_ms);
            unix_ns.saturating_sub(elapsed_ms.saturating_mul(1_000_000))
        })
        .min()
}

pub fn emitter_key(event: &LiveTraceEvent) -> String {
    event
        .emitter_id
        .as_deref()
        .or(event.source.as_deref())
        .unwrap_or("unknown")
        .to_string()
}

fn normalized_event_order(
    left: &LiveTraceEvent,
    left_index: usize,
    right: &LiveTraceEvent,
    right_index: usize,
) -> Ordering {
    (
        left.elapsed_ms,
        left.t_unix_ns,
        emitter_key(left),
        left.seq_id,
        left.turn,
        left_index,
    )
        .cmp(&(
            right.elapsed_ms,
            right.t_unix_ns,
            emitter_key(right),
            right.seq_id,
            right.turn,
            right_index,
        ))
}

#[derive(Debug)]
struct Queued<T> {
    seq_id: u64,
    arrival: u64,
    event: T,
}

impl<T> PartialEq for Queued<T> {
    fn eq(&self, other: &Self) -> bool {
        self.seq_id == other.seq_id && self.arrival == other.arrival
    }
}

impl<T> Eq for Queued<T> {}

impl<T> PartialOrd for Queued<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Queued<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .seq_id
            .cmp(&self.seq_id)
            .then_with(|| other.arrival.cmp(&self.arrival))
    }
}

#[derive(Debug)]
pub struct TraceReorderBuffer<T> {
    next_seq: u64,
    window: usize,
    next_arrival: u64,
    heap: BinaryHeap<Queued<T>>,
}

impl<T> TraceReorderBuffer<T> {
    pub fn new(next_seq: u64, window: usize) -> Self {
        Self {
            next_seq,
            window: window.max(1),
            next_arrival: 0,
            heap: BinaryHeap::new(),
        }
    }

    pub fn insert_and_flush(&mut self, seq_id: u64, event: T) -> Vec<T> {
        self.heap.push(Queued {
            seq_id,
            arrival: self.next_arrival,
            event,
        });
        self.next_arrival = self.next_arrival.saturating_add(1);
        self.flush_ready()
    }

    pub fn flush_ready(&mut self) -> Vec<T> {
        let mut out = Vec::new();
        while self
            .heap
            .peek()
            .is_some_and(|top| top.seq_id == self.next_seq)
        {
            let queued = self.heap.pop().expect("peeked queued event should pop");
            self.next_seq = self.next_seq.saturating_add(1);
            out.push(queued.event);
        }

        while self.heap.len() > self.window {
            let queued = self.heap.pop().expect("heap above window should pop");
            self.next_seq = queued.seq_id.saturating_add(1);
            out.push(queued.event);
        }
        out
    }

    pub fn drain(mut self) -> Vec<T> {
        let mut out = Vec::new();
        while let Some(queued) = self.heap.pop() {
            out.push(queued.event);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live_trace::LiveTraceEvent;
    use crate::speech_timeline::SessionId;
    use crate::time::ExactTimestamp;

    fn ts(ms: u64) -> ExactTimestamp {
        ExactTimestamp {
            unix_nanos: u128::from(ms) * 1_000_000,
        }
    }

    fn event(kind: &str, elapsed_ms: u64) -> LiveTraceEvent {
        let mut event = LiveTraceEvent::new(SessionId::new(), 1, kind, ts(elapsed_ms), ts(0));
        event.normalized_elapsed_ms = None;
        event.normalized_unix_ns = None;
        event
    }

    #[test]
    fn pendulum_sequence_reorder_restores_delayed_flushes() {
        let mut buffer = TraceReorderBuffer::new(1, 16);
        let mut out = Vec::new();
        for seq in [1, 3, 2, 5, 4, 6] {
            out.extend(buffer.insert_and_flush(seq, seq));
        }
        out.extend(buffer.drain());

        assert_eq!(out, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn drift_mapping_keeps_remote_emitter_in_canonical_time() {
        let mut start_sync = event("clock_sync", 0);
        start_sync.emitter_id = Some("remote".to_string());
        start_sync.clock_sync = Some(TraceClockSync {
            local_mono: 0,
            canon_elapsed_ms: 0,
            canon_unix_ns: Some(0),
            drift_ppm: None,
        });

        let mut end_sync = event("clock_sync", 10_000);
        end_sync.emitter_id = Some("remote".to_string());
        end_sync.clock_sync = Some(TraceClockSync {
            local_mono: 10_001,
            canon_elapsed_ms: 10_000,
            canon_unix_ns: Some(10_000_000_000),
            drift_ppm: None,
        });

        let mut remote = event("remote_join", 0);
        remote.emitter_id = Some("remote".to_string());
        remote.ts_local_monotonic = Some(5_001);
        remote.elapsed_ms = 4_800;

        let normalized = normalize_live_trace_events(&[remote, end_sync, start_sync]);
        let remote = normalized
            .iter()
            .find(|event| event.kind == "remote_join")
            .expect("remote event should survive normalization");

        assert_eq!(remote.elapsed_ms, 5_000);
        assert_eq!(remote.observed_elapsed_ms, Some(4_800));
    }
}
