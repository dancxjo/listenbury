//! Golden-trace regression testing helpers.
//!
//! This module provides helpers for loading canonical fixture traces and
//! asserting structural properties of their converted [`ViewerPayload`]s.
//!
//! # Fixture layout
//!
//! Each fixture lives under `fixtures/traces/<name>/` and contains:
//!
//! - `input.jsonl` – the raw [`LiveTraceEvent`] stream in JSONL format
//! - `expected_key_spans.json` – a [`KeySpanAssertions`] document used as the
//!   primary automated assertions
//! - `expected_screenplay.txt` – a human-readable transcript for the session
//! - `expected_viewer_payload.json` – the full expected [`ViewerPayload`] for
//!   snapshot comparison (generated via `LISTENBURY_UPDATE_GOLDEN=1`)
//!
//! # Running golden tests
//!
//! ```text
//! cargo test --no-default-features --test golden_traces
//! ```
//!
//! # Regenerating expected payloads
//!
//! Set `LISTENBURY_UPDATE_GOLDEN=1` when running the tests to write
//! `expected_viewer_payload.json` for every fixture:
//!
//! ```text
//! LISTENBURY_UPDATE_GOLDEN=1 cargo test --no-default-features \
//!     --test golden_traces
//! ```
//!
//! See [`docs/golden-trace-fixtures.md`] for guidance on adding new fixtures.
//!
//! [`docs/golden-trace-fixtures.md`]: ../../docs/golden-trace-fixtures.md

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::trace::viewer_payload::{ViewerPayload, live_trace_jsonl_to_viewer_payload};

/// Replay a JSONL trace string and return the resulting [`ViewerPayload`].
///
/// This is the primary entry point for golden-trace tests.
pub fn replay_trace_jsonl(jsonl: &str) -> Result<ViewerPayload> {
    live_trace_jsonl_to_viewer_payload(jsonl)
}

/// A collection of key-span assertions for a golden trace fixture.
///
/// Stored in `expected_key_spans.json` inside each fixture directory.
/// Use [`KeySpanAssertions::from_json`] to load it and
/// [`KeySpanAssertions::check_all`] to evaluate against a [`ViewerPayload`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeySpanAssertions {
    /// Short human-readable description of what this fixture tests.
    pub description: String,
    /// Individual assertions over the viewer payload.
    pub assertions: Vec<KeySpanAssertion>,
}

/// A single structural assertion over a [`ViewerPayload`].
///
/// Serialized as a tagged JSON object with a `"kind"` discriminant field.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KeySpanAssertion {
    /// The payload must contain a word lane with the given label.
    HasLane { label: String },
    /// The payload must contain at least one viewer event with the given kind
    /// inside the named lane.
    HasEvent { lane: String, event_kind: String },
    /// The payload must contain at least one viewer marker with the given kind
    /// inside the named lane.
    HasMarker { lane: String, marker_kind: String },
    /// The payload must **not** contain any viewer event with the given kind.
    NoEvent { event_kind: String },
    /// The payload must **not** contain any viewer marker with the given kind.
    NoMarker { marker_kind: String },
    /// The first marker must appear at or before the second marker on the
    /// timeline (`at_ms` ordering).  Both markers must be present.
    MarkerOrdering {
        first_kind: String,
        second_kind: String,
    },
    /// The first event must start at or before the second event on the
    /// timeline (`start_ms` ordering).  Both events must be present.
    EventOrdering {
        first_kind: String,
        second_kind: String,
    },
    /// The marker must appear no later than `max_ms` milliseconds into the
    /// session (useful for latency-budget regression tests).
    LatencyBudget { marker_kind: String, max_ms: u64 },
}

/// The outcome of a single [`KeySpanAssertion`] check.
#[derive(Debug, Clone)]
pub enum AssertionOutcome {
    /// The assertion passed.
    Pass,
    /// The assertion failed.  Contains a human-readable description.
    Fail(String),
}

impl KeySpanAssertions {
    /// Deserialise from a JSON string (the `expected_key_spans.json` file).
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).context("parse key-span assertions JSON")
    }

    /// Evaluate all assertions against `payload`.
    ///
    /// Returns a list of failure messages; an empty list means all assertions
    /// passed.
    pub fn check_all(&self, payload: &ViewerPayload) -> Vec<String> {
        self.assertions
            .iter()
            .filter_map(|assertion| match assertion.check(payload) {
                AssertionOutcome::Pass => None,
                AssertionOutcome::Fail(msg) => Some(msg),
            })
            .collect()
    }
}

impl KeySpanAssertion {
    /// Evaluate this single assertion against `payload`.
    pub fn check(&self, payload: &ViewerPayload) -> AssertionOutcome {
        match self {
            Self::HasLane { label } => {
                if payload.streams.iter().any(|s| &s.label == label) {
                    AssertionOutcome::Pass
                } else {
                    let found: Vec<&str> =
                        payload.streams.iter().map(|s| s.label.as_str()).collect();
                    AssertionOutcome::Fail(format!(
                        "expected word lane {label:?} but found: {found:?}"
                    ))
                }
            }
            Self::HasEvent { lane, event_kind } => {
                if payload
                    .events
                    .iter()
                    .any(|e| &e.lane == lane && &e.kind == event_kind)
                {
                    AssertionOutcome::Pass
                } else {
                    AssertionOutcome::Fail(format!(
                        "expected viewer event with lane={lane:?} kind={event_kind:?}"
                    ))
                }
            }
            Self::HasMarker { lane, marker_kind } => {
                if payload
                    .markers
                    .iter()
                    .any(|m| &m.lane == lane && &m.kind == marker_kind)
                {
                    AssertionOutcome::Pass
                } else {
                    AssertionOutcome::Fail(format!(
                        "expected viewer marker with lane={lane:?} kind={marker_kind:?}"
                    ))
                }
            }
            Self::NoEvent { event_kind } => {
                if payload.events.iter().any(|e| &e.kind == event_kind) {
                    AssertionOutcome::Fail(format!(
                        "expected no viewer event with kind={event_kind:?} but found one"
                    ))
                } else {
                    AssertionOutcome::Pass
                }
            }
            Self::NoMarker { marker_kind } => {
                if payload.markers.iter().any(|m| &m.kind == marker_kind) {
                    AssertionOutcome::Fail(format!(
                        "expected no viewer marker with kind={marker_kind:?} but found one"
                    ))
                } else {
                    AssertionOutcome::Pass
                }
            }
            Self::MarkerOrdering {
                first_kind,
                second_kind,
            } => {
                let first_at = payload
                    .markers
                    .iter()
                    .find(|m| &m.kind == first_kind)
                    .map(|m| m.at_ms);
                let second_at = payload
                    .markers
                    .iter()
                    .find(|m| &m.kind == second_kind)
                    .map(|m| m.at_ms);
                match (first_at, second_at) {
                    (Some(a), Some(b)) if a <= b => AssertionOutcome::Pass,
                    (Some(a), Some(b)) => AssertionOutcome::Fail(format!(
                        "expected marker {first_kind:?} at {a}ms before \
                         {second_kind:?} at {b}ms, but order is reversed"
                    )),
                    (None, _) => AssertionOutcome::Fail(format!(
                        "marker {first_kind:?} not found (required for ordering check)"
                    )),
                    (_, None) => AssertionOutcome::Fail(format!(
                        "marker {second_kind:?} not found (required for ordering check)"
                    )),
                }
            }
            Self::EventOrdering {
                first_kind,
                second_kind,
            } => {
                let first_start = payload
                    .events
                    .iter()
                    .find(|e| &e.kind == first_kind)
                    .map(|e| e.start_ms);
                let second_start = payload
                    .events
                    .iter()
                    .find(|e| &e.kind == second_kind)
                    .map(|e| e.start_ms);
                match (first_start, second_start) {
                    (Some(a), Some(b)) if a <= b => AssertionOutcome::Pass,
                    (Some(a), Some(b)) => AssertionOutcome::Fail(format!(
                        "expected event {first_kind:?} starting at {a}ms before \
                         {second_kind:?} starting at {b}ms, but order is reversed"
                    )),
                    (None, _) => AssertionOutcome::Fail(format!(
                        "event {first_kind:?} not found (required for ordering check)"
                    )),
                    (_, None) => AssertionOutcome::Fail(format!(
                        "event {second_kind:?} not found (required for ordering check)"
                    )),
                }
            }
            Self::LatencyBudget {
                marker_kind,
                max_ms,
            } => match payload.markers.iter().find(|m| &m.kind == marker_kind) {
                Some(marker) if marker.at_ms <= *max_ms => AssertionOutcome::Pass,
                Some(marker) => AssertionOutcome::Fail(format!(
                    "latency budget exceeded: {marker_kind:?} appeared at {}ms \
                         but expected ≤ {max_ms}ms",
                    marker.at_ms
                )),
                None => AssertionOutcome::Fail(format!(
                    "marker {marker_kind:?} not found (required for latency budget check)"
                )),
            },
        }
    }
}

/// Compare two [`ViewerPayload`] values, returning a list of human-readable
/// differences.
///
/// Used by the golden-update mechanism when `LISTENBURY_UPDATE_GOLDEN` is not
/// set.  An empty list means the payloads are equivalent.
pub fn diff_viewer_payloads(expected: &ViewerPayload, actual: &ViewerPayload) -> Vec<String> {
    let mut diffs = Vec::new();

    let expected_json = serde_json::to_value(expected).unwrap_or_default();
    let actual_json = serde_json::to_value(actual).unwrap_or_default();

    if expected_json != actual_json {
        // Provide a summary diff at the top level rather than a recursive walk,
        // which keeps failure messages actionable without being overwhelming.
        let expected_str = serde_json::to_string_pretty(&expected_json).unwrap_or_default();
        let actual_str = serde_json::to_string_pretty(&actual_json).unwrap_or_default();

        if expected_str != actual_str {
            diffs.push(format!(
                "viewer payload mismatch:\n--- expected ---\n{expected_str}\n\
                 --- actual ---\n{actual_str}"
            ));
        }
    }

    diffs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live_trace::LiveTraceEvent;
    use crate::speech_timeline::SessionId;
    use crate::time::ExactTimestamp;
    use crate::trace::viewer_payload::live_trace_events_to_viewer_payload;

    fn ts(ms: u64) -> ExactTimestamp {
        ExactTimestamp {
            unix_nanos: u128::from(ms) * 1_000_000,
        }
    }

    fn make_event(turn: u64, kind: &str, elapsed_ms: u64) -> LiveTraceEvent {
        LiveTraceEvent::new(SessionId::new(), turn, kind, ts(elapsed_ms), ts(0))
    }

    #[test]
    fn has_lane_passes_when_lane_present() {
        let mut event = make_event(1, "transcript", 500);
        event.text = Some("hello".to_string());
        let payload = live_trace_events_to_viewer_payload(&[event]);

        let assertion = KeySpanAssertion::HasLane {
            label: "User transcript".to_string(),
        };
        assert!(
            matches!(assertion.check(&payload), AssertionOutcome::Pass),
            "HasLane should pass when the lane is present"
        );
    }

    #[test]
    fn has_lane_fails_when_lane_absent() {
        let payload = live_trace_events_to_viewer_payload(&[]);
        let assertion = KeySpanAssertion::HasLane {
            label: "Nonexistent lane".to_string(),
        };
        assert!(
            matches!(assertion.check(&payload), AssertionOutcome::Fail(_)),
            "HasLane should fail when the lane is absent"
        );
    }

    #[test]
    fn no_event_passes_when_event_absent() {
        let payload = live_trace_events_to_viewer_payload(&[]);
        let assertion = KeySpanAssertion::NoEvent {
            event_kind: "overlap".to_string(),
        };
        assert!(
            matches!(assertion.check(&payload), AssertionOutcome::Pass),
            "NoEvent should pass when no such event is present"
        );
    }

    #[test]
    fn no_event_fails_when_event_present() {
        let overlap_start = make_event(1, "overlap_started", 300);
        let overlap_end = make_event(1, "overlap_ended", 500);
        let payload = live_trace_events_to_viewer_payload(&[overlap_start, overlap_end]);
        let assertion = KeySpanAssertion::NoEvent {
            event_kind: "overlap".to_string(),
        };
        assert!(
            matches!(assertion.check(&payload), AssertionOutcome::Fail(_)),
            "NoEvent should fail when the event is present"
        );
    }

    #[test]
    fn marker_ordering_passes_when_first_precedes_second() {
        let first = make_event(1, "first_llm_token", 400);
        let second = make_event(1, "playback_started", 600);
        let payload = live_trace_events_to_viewer_payload(&[first, second]);
        let assertion = KeySpanAssertion::MarkerOrdering {
            first_kind: "first_llm_token".to_string(),
            second_kind: "playback_started".to_string(),
        };
        assert!(
            matches!(assertion.check(&payload), AssertionOutcome::Pass),
            "MarkerOrdering should pass when first_kind appears before second_kind"
        );
    }

    #[test]
    fn marker_ordering_fails_when_second_precedes_first() {
        let first = make_event(1, "playback_started", 400);
        let second = make_event(1, "first_llm_token", 600);
        let payload = live_trace_events_to_viewer_payload(&[first, second]);
        let assertion = KeySpanAssertion::MarkerOrdering {
            first_kind: "first_llm_token".to_string(),
            second_kind: "playback_started".to_string(),
        };
        // first_llm_token is at 600, playback_started is at 400 → ordering violated
        assert!(
            matches!(assertion.check(&payload), AssertionOutcome::Fail(_)),
            "MarkerOrdering should fail when second_kind appears before first_kind"
        );
    }

    #[test]
    fn latency_budget_passes_within_budget() {
        let event = make_event(1, "first_llm_token", 300);
        let payload = live_trace_events_to_viewer_payload(&[event]);
        let assertion = KeySpanAssertion::LatencyBudget {
            marker_kind: "first_llm_token".to_string(),
            max_ms: 500,
        };
        assert!(
            matches!(assertion.check(&payload), AssertionOutcome::Pass),
            "LatencyBudget should pass when marker appears within budget"
        );
    }

    #[test]
    fn latency_budget_fails_over_budget() {
        let event = make_event(1, "first_llm_token", 800);
        let payload = live_trace_events_to_viewer_payload(&[event]);
        let assertion = KeySpanAssertion::LatencyBudget {
            marker_kind: "first_llm_token".to_string(),
            max_ms: 500,
        };
        assert!(
            matches!(assertion.check(&payload), AssertionOutcome::Fail(_)),
            "LatencyBudget should fail when marker appears beyond budget"
        );
    }

    #[test]
    fn key_span_assertions_from_json_round_trips() {
        let json = r#"{
            "description": "test",
            "assertions": [
                { "kind": "has_lane", "label": "User transcript" },
                { "kind": "no_event", "event_kind": "overlap" },
                { "kind": "marker_ordering", "first_kind": "first_llm_token", "second_kind": "playback_started" }
            ]
        }"#;
        let parsed = KeySpanAssertions::from_json(json).expect("should parse");
        assert_eq!(parsed.description, "test");
        assert_eq!(parsed.assertions.len(), 3);
    }
}
