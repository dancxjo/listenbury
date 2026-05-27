use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::live_trace::LiveTraceEvent;
use crate::memory::trace::MemoryTrace;
use crate::speech_timeline::SessionId;
use crate::time::ExactTimestamp;

pub type EventId = String;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventSource {
    RuntimeTrace,
    BrowserInput,
    MemoryIngestion,
    Unknown(String),
}

impl EventSource {
    pub fn from_source_tag(source: Option<&str>) -> Self {
        match source.unwrap_or("runtime.trace") {
            "runtime.trace" => Self::RuntimeTrace,
            "browser.camera" => Self::BrowserInput,
            "memory.trace" => Self::MemoryIngestion,
            value => Self::Unknown(value.to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeEventSubtype {
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "domain", content = "event", rename_all = "snake_case")]
pub enum RuntimeEventKind {
    Hearing(RuntimeEventSubtype),
    Playback(RuntimeEventSubtype),
    Asr(RuntimeEventSubtype),
    TranscriptRevision(RuntimeEventSubtype),
    Llm(RuntimeEventSubtype),
    Suppression(RuntimeEventSubtype),
    BrowserInput(RuntimeEventSubtype),
    Diagnostics(RuntimeEventSubtype),
    SpanMutation(RuntimeEventSubtype),
    MemoryIngestion(RuntimeEventSubtype),
    Other(RuntimeEventSubtype),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypedRuntimeEvent {
    Hearing(RuntimeEventSubtype),
    Playback(RuntimeEventSubtype),
    Asr(RuntimeEventSubtype),
    TranscriptRevision(RuntimeEventSubtype),
    Llm(RuntimeEventSubtype),
    Suppression(RuntimeEventSubtype),
    BrowserInput(RuntimeEventSubtype),
    Diagnostics(RuntimeEventSubtype),
    SpanMutation(RuntimeEventSubtype),
    MemoryIngestion(RuntimeEventSubtype),
    Other(RuntimeEventSubtype),
}

impl From<TypedRuntimeEvent> for RuntimeEventKind {
    fn from(event: TypedRuntimeEvent) -> Self {
        match event {
            TypedRuntimeEvent::Hearing(subtype) => RuntimeEventKind::Hearing(subtype),
            TypedRuntimeEvent::Playback(subtype) => RuntimeEventKind::Playback(subtype),
            TypedRuntimeEvent::Asr(subtype) => RuntimeEventKind::Asr(subtype),
            TypedRuntimeEvent::TranscriptRevision(subtype) => {
                RuntimeEventKind::TranscriptRevision(subtype)
            }
            TypedRuntimeEvent::Llm(subtype) => RuntimeEventKind::Llm(subtype),
            TypedRuntimeEvent::Suppression(subtype) => RuntimeEventKind::Suppression(subtype),
            TypedRuntimeEvent::BrowserInput(subtype) => RuntimeEventKind::BrowserInput(subtype),
            TypedRuntimeEvent::Diagnostics(subtype) => RuntimeEventKind::Diagnostics(subtype),
            TypedRuntimeEvent::SpanMutation(subtype) => RuntimeEventKind::SpanMutation(subtype),
            TypedRuntimeEvent::MemoryIngestion(subtype) => {
                RuntimeEventKind::MemoryIngestion(subtype)
            }
            TypedRuntimeEvent::Other(subtype) => RuntimeEventKind::Other(subtype),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeEvent {
    pub id: EventId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<SessionId>,
    pub timestamp: ExactTimestamp,
    pub monotonic_ms: u64,
    pub source: EventSource,
    pub kind: RuntimeEventKind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub causality: Vec<EventId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub correlation: Vec<String>,
}

impl RuntimeEvent {
    /// Builds a canonical runtime event from a live trace event.
    ///
    /// Preferred path: producers attach a typed `runtime_event.kind` on `LiveTraceEvent`.
    /// Compatibility path: if no typed kind is present, this falls back to
    /// `legacy_classify_runtime_kind_from_string` so replay of historical traces remains stable.
    pub fn from_live_trace_event(event: &LiveTraceEvent) -> Self {
        let id = format!(
            "live:{}:{}:{}:{}",
            event
                .session_id
                .map(|session_id| session_id.0.to_string())
                .unwrap_or_else(|| "none".to_string()),
            event.turn,
            event.elapsed_ms,
            event.kind
        );
        let timestamp = ExactTimestamp {
            unix_nanos: u128::from(event.t_unix_ns),
        };
        let source = EventSource::from_source_tag(event.source.as_deref());
        let kind = event
            .runtime_event
            .as_ref()
            .map(|runtime| runtime.kind.clone())
            .unwrap_or_else(|| {
                legacy_classify_runtime_kind_from_string(
                    &event.kind,
                    event.text.as_deref(),
                    event.reason.as_deref(),
                    event.artifact.clone(),
                )
            });
        let mut causality = Vec::new();
        if let Some(turn_id) = event.turn_id {
            causality.push(format!("turn:{}", turn_id.0));
        }
        if let Some(utterance_id) = event.utterance_id {
            causality.push(format!("utterance:{}", utterance_id.0));
        }
        if let Some(synthetic_unit_id) = event.synthetic_unit_id {
            causality.push(format!("synthetic_unit:{}", synthetic_unit_id.0));
        }
        if let Some(transcript_revision_id) = event.transcript_revision_id {
            causality.push(format!("transcript_revision:{}", transcript_revision_id.0));
        }
        if let Some(span_id) = event.span_id {
            causality.push(format!("span:{}", span_id.0));
        }
        if let Some(audio_clip_id) = event.audio_clip_id {
            causality.push(format!("audio_clip:{}", audio_clip_id.0));
        }
        Self {
            id,
            session_id: event.session_id,
            timestamp,
            monotonic_ms: event.elapsed_ms,
            source,
            kind,
            causality,
            correlation: vec![format!("turn:{}", event.turn)],
        }
    }

    pub fn from_memory_trace(trace: &MemoryTrace) -> Self {
        let occurred_at = trace.occurred_at();
        let kind_name = trace.kind_name().to_string();
        let (text, reason, artifact): (Option<String>, Option<String>, Option<Value>) = match trace
        {
            MemoryTrace::ConversationTurnFinalized { text, .. }
            | MemoryTrace::TimedWordStreamFinalized { summary: text, .. }
            | MemoryTrace::MouthPlaybackStarted { text, .. }
            | MemoryTrace::MouthPlaybackCompleted { text, .. }
            | MemoryTrace::AuditorySceneObservation {
                description: text, ..
            }
            | MemoryTrace::OverlapDetected {
                description: text, ..
            }
            | MemoryTrace::RecallResultUsed {
                result_summary: text,
                ..
            }
            | MemoryTrace::AssistantAnalysisCaptured { text, .. } => {
                (Some(text.clone()), None, None)
            }
            MemoryTrace::EntityExtractionPerformed {
                source_text,
                entities,
                ..
            } => (
                Some(source_text.clone()),
                Some("entity_extraction".to_string()),
                Some(json!({ "entities": entities })),
            ),
            MemoryTrace::GraphNodeFieldsUpdated { update, .. } => (
                Some(format!("updated graph node {}", update.node_id)),
                Some("graph_node_fields_updated".to_string()),
                Some(json!({ "update": update })),
            ),
            MemoryTrace::ImageVectorCaptured { image, .. } => (
                Some(format!("image vector {}", image.image_id)),
                Some("image_vector".to_string()),
                Some(json!({ "image": image })),
            ),
            MemoryTrace::VoiceVectorCaptured { voice, .. } => (
                Some(format!("voice vector {}", voice.voice_signature_id)),
                Some("voice_vector".to_string()),
                Some(json!({ "voice": voice })),
            ),
        };
        let mut causality = Vec::new();
        if let MemoryTrace::TimedWordStreamFinalized { stream_id, .. } = trace {
            causality.push(format!("stream:{stream_id}"));
        }
        if let MemoryTrace::MouthPlaybackStarted { utterance_id, .. }
        | MemoryTrace::MouthPlaybackCompleted { utterance_id, .. } = trace
        {
            causality.push(format!("utterance:{utterance_id}"));
        }
        if let MemoryTrace::RecallResultUsed { query, .. } = trace {
            causality.push(format!("query:{query}"));
        }
        if let MemoryTrace::AssistantAnalysisCaptured { scene, .. } = trace {
            causality.push(format!("scene:{}", scene.node_id));
        }
        if let MemoryTrace::EntityExtractionPerformed { entities, .. } = trace {
            causality.extend(
                entities
                    .iter()
                    .map(|entity| format!("entity:{}", entity.node_id)),
            );
        }
        if let MemoryTrace::GraphNodeFieldsUpdated { update, .. } = trace {
            causality.push(format!("graph_node:{}", update.node_id));
        }
        if let MemoryTrace::ImageVectorCaptured { image, .. } = trace {
            causality.push(format!("image:{}", image.image_id));
            if let Some(content_node_id) = &image.content_node_id {
                causality.push(format!("visual_referent:{content_node_id}"));
            }
        }
        if let MemoryTrace::VoiceVectorCaptured { voice, .. } = trace {
            causality.push(format!("voice_signature:{}", voice.voice_signature_id));
            causality.push(format!("voice:{}", voice.voice_node_id));
        }
        let memory_kind_correlation = format!("memory_kind:{kind_name}");
        Self {
            id: format!("memory:{}:{}", occurred_at.unix_nanos, kind_name),
            session_id: None,
            timestamp: occurred_at,
            monotonic_ms: occurred_at.unix_nanos.saturating_div(1_000_000) as u64,
            source: EventSource::MemoryIngestion,
            kind: TypedRuntimeEvent::MemoryIngestion(RuntimeEventSubtype {
                kind: kind_name,
                text,
                reason,
                artifact,
            })
            .into(),
            causality,
            correlation: vec![memory_kind_correlation],
        }
    }
}

/// Legacy adapter for historical string-typed `LiveTraceEvent.kind` values.
///
/// # ⚠ REPLAY-ONLY — do not add new call sites
///
/// This function exists solely so that replaying historical `.jsonl` trace files
/// (recorded before `TypedRuntimeEvent` existed) still produces a sensible
/// `RuntimeEventKind`.  Every **live** runtime event producer should instead call
/// `LiveTraceEvent::set_runtime_kind` with an explicit `TypedRuntimeEvent` value.
///
/// ## Deletion checklist
///
/// Delete this function once **all** of the following are true:
///
/// - [ ] `src/cli/commands/live_half_duplex.rs` — all calls to `trace.emit_now`,
///   `trace.buffer_now`, `trace.emit`, and `trace.buffer` use `set_runtime_kind`
///   before emission.
/// - [ ] `src/web/server.rs` — the test/diagnostic `LiveTraceEvent::new("transcript", …)`
///   call uses a typed kind.
/// - [ ] Any other future live producer added to `src/` calls `set_runtime_kind`.
/// - [ ] Trace-replay code (e.g. `TraceSessionEnvelope` loading, golden-trace fixtures)
///   either (a) no longer depends on string-prefix inference, or (b) migrates to a
///   dedicated replay-only helper that is clearly separate from the live path.
///
/// ## How to verify readiness
///
/// Run `cargo test --no-default-features --lib runtime_event` and confirm that
/// `all_known_live_producers_use_typed_runtime_kind` passes.  That test currently
/// documents the *unfinished* migration by asserting that `event_used_legacy_classification`
/// returns `true` for freshly-constructed events — once all producers are migrated the
/// test should be inverted and this function should be deleted.
fn legacy_classify_runtime_kind_from_string(
    kind: &str,
    text: Option<&str>,
    reason: Option<&str>,
    artifact: Option<Value>,
) -> RuntimeEventKind {
    let subtype = RuntimeEventSubtype {
        kind: kind.to_string(),
        text: text.map(str::to_string),
        reason: reason.map(str::to_string),
        artifact,
    };
    if kind.starts_with("speech_")
        || kind.starts_with("breath_")
        || kind == "capture_started"
        || kind == "listening_started"
        || kind == "auditory_observation"
        || kind == "environment_observation"
        || kind == "environmental_sound"
        || kind == "overlap_detected"
    {
        RuntimeEventKind::Hearing(subtype)
    } else if kind.starts_with("playback_")
        || kind.starts_with("tts_")
        || kind.starts_with("echo_")
        || kind == "self_voice_heard"
    {
        RuntimeEventKind::Playback(subtype)
    } else if kind.starts_with("asr_") {
        RuntimeEventKind::Asr(subtype)
    } else if kind.starts_with("transcript_") || kind == "transcript" {
        RuntimeEventKind::TranscriptRevision(subtype)
    } else if kind.starts_with("llm_")
        || kind.starts_with("first_llm_")
        || kind == "token_emitted"
        || kind.starts_with("synthetic_unit_")
        || kind == "speculative_synthetic_unit_updated"
        || kind == "first_safe_synthetic_unit_emitted"
    {
        RuntimeEventKind::Llm(subtype)
    } else if kind.starts_with("self_hearing_suppression_") || kind.starts_with("yield_") {
        RuntimeEventKind::Suppression(subtype)
    } else if kind.starts_with("browser_") || kind.starts_with("visual_speech_") {
        RuntimeEventKind::BrowserInput(subtype)
    } else if kind.starts_with("span_") || kind.contains("alignment") {
        RuntimeEventKind::SpanMutation(subtype)
    } else if kind.starts_with("diagnostic") || kind.starts_with("debug_") {
        RuntimeEventKind::Diagnostics(subtype)
    } else {
        RuntimeEventKind::Other(subtype)
    }
}

/// Returns `true` when `event.runtime_event.kind` was inferred by the legacy string
/// fallback rather than set explicitly via [`LiveTraceEvent::set_runtime_kind`].
///
/// This is the **mechanical detection hook** for the migration tracker.  Use it in
/// tests to assert that a producer has (or has not yet) migrated to the typed path:
///
/// ```ignore
/// // A producer that still relies on string-only classification:
/// assert!(event_used_legacy_classification(&event));
///
/// // A producer that already calls set_runtime_kind:
/// assert!(!event_used_legacy_classification(&event));
/// ```
///
/// The function re-runs `legacy_classify_runtime_kind_from_string` on the event's
/// `kind` string and compares the result against the stored `runtime_event.kind`.
/// They match when no explicit typed kind was set; they differ when the producer
/// called `set_runtime_kind` with a domain that differs from the legacy inference.
///
/// # Note on ambiguous cases
/// If a producer calls `set_runtime_kind` but chooses the exact same domain
/// that the legacy classifier would also choose, this function conservatively
/// reports `true` (a false-positive in that narrow case).  This is acceptable —
/// the goal is to catch events where `set_runtime_kind` was never called at all,
/// and any producer that calls `set_runtime_kind` should be considered migrated
/// regardless of whether its chosen domain happens to match the legacy inference.
#[cfg(test)]
pub(crate) fn event_used_legacy_classification(event: &LiveTraceEvent) -> bool {
    let Some(runtime) = event.runtime_event.as_ref() else {
        return true;
    };
    let legacy_kind = legacy_classify_runtime_kind_from_string(
        &event.kind,
        event.text.as_deref(),
        event.reason.as_deref(),
        event.artifact.clone(),
    );
    runtime.kind == legacy_kind
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live_trace::{LiveTraceEvent, SessionId};
    use crate::memory::trace::MemoryEntityMention;

    fn ts(ms: u64) -> ExactTimestamp {
        ExactTimestamp {
            unix_nanos: u128::from(ms) * 1_000_000,
        }
    }

    #[test]
    fn runtime_event_roundtrips_with_stable_domain_encoding() {
        let mut event =
            LiveTraceEvent::new(SessionId::new(), 3, "asr_started", ts(1_350), ts(1_000));
        event.source = Some("runtime.trace".to_string());
        let runtime = RuntimeEvent::from_live_trace_event(&event);
        let json = serde_json::to_string(&runtime).expect("serialize runtime event");
        let decoded: RuntimeEvent = serde_json::from_str(&json).expect("deserialize runtime event");
        assert_eq!(decoded, runtime);
        assert!(json.contains("\"domain\":\"asr\""));
        assert!(json.contains("\"monotonic_ms\":350"));
    }

    #[test]
    fn memory_runtime_event_serializes_with_memory_source() {
        let trace = MemoryTrace::OverlapDetected {
            description: "two voices".to_string(),
            occurred_at: ts(2_000),
        };
        let runtime = RuntimeEvent::from_memory_trace(&trace);
        let json = serde_json::to_string(&runtime).expect("serialize runtime event");
        assert!(json.contains("\"source\":\"memory_ingestion\""));
        assert!(json.contains("\"domain\":\"memory_ingestion\""));
        let decoded: RuntimeEvent = serde_json::from_str(&json).expect("deserialize runtime event");
        assert_eq!(decoded.kind, runtime.kind);
    }

    #[test]
    fn entity_extraction_memory_event_keeps_entity_artifact_and_causality() {
        let trace = MemoryTrace::EntityExtractionPerformed {
            source_text: "My name is Travis".to_string(),
            entities: vec![MemoryEntityMention {
                node_id: "person:travis".to_string(),
                label: "Travis".to_string(),
                kind: "person".to_string(),
                confidence: 0.97,
                span_start: 11,
                span_end: 17,
            }],
            occurred_at: ts(2_500),
        };

        let runtime = RuntimeEvent::from_memory_trace(&trace);

        assert!(
            runtime
                .causality
                .contains(&"entity:person:travis".to_string())
        );
        let RuntimeEventKind::MemoryIngestion(subtype) = runtime.kind else {
            panic!("entity extraction should remain a memory ingestion event");
        };
        assert_eq!(subtype.reason.as_deref(), Some("entity_extraction"));
        assert_eq!(subtype.text.as_deref(), Some("My name is Travis"));
        assert!(
            subtype
                .artifact
                .as_ref()
                .and_then(|artifact| artifact.get("entities"))
                .and_then(Value::as_array)
                .is_some_and(|entities| entities.len() == 1)
        );
    }

    fn subtype(kind: &str) -> RuntimeEventSubtype {
        RuntimeEventSubtype {
            kind: kind.to_string(),
            text: None,
            reason: None,
            artifact: None,
        }
    }

    #[test]
    fn typed_runtime_event_converts_major_domains() {
        let cases = vec![
            (
                TypedRuntimeEvent::Hearing(subtype("hearing")),
                RuntimeEventKind::Hearing(subtype("hearing")),
            ),
            (
                TypedRuntimeEvent::Playback(subtype("playback")),
                RuntimeEventKind::Playback(subtype("playback")),
            ),
            (
                TypedRuntimeEvent::Asr(subtype("asr")),
                RuntimeEventKind::Asr(subtype("asr")),
            ),
            (
                TypedRuntimeEvent::Llm(subtype("llm")),
                RuntimeEventKind::Llm(subtype("llm")),
            ),
            (
                TypedRuntimeEvent::BrowserInput(subtype("browser_input")),
                RuntimeEventKind::BrowserInput(subtype("browser_input")),
            ),
            (
                TypedRuntimeEvent::Suppression(subtype("suppression")),
                RuntimeEventKind::Suppression(subtype("suppression")),
            ),
        ];

        for (typed, expected) in cases {
            let converted: RuntimeEventKind = typed.into();
            assert_eq!(converted, expected);
        }
    }

    #[test]
    fn prefers_typed_runtime_kind_when_present() {
        let mut event = LiveTraceEvent::new(
            SessionId::new(),
            7,
            "brand_new_event_name",
            ts(1_200),
            ts(1_000),
        );
        let typed_runtime_kind: RuntimeEventKind =
            TypedRuntimeEvent::BrowserInput(subtype("browser_pointer_move")).into();
        event.set_runtime_kind(typed_runtime_kind);

        let runtime = RuntimeEvent::from_live_trace_event(&event);
        assert_eq!(
            runtime.kind,
            RuntimeEventKind::BrowserInput(subtype("browser_pointer_move"))
        );
    }

    // -----------------------------------------------------------------------
    // Legacy classification detection tests
    // -----------------------------------------------------------------------

    /// Documents the current migration state: a freshly-constructed live trace event
    /// that never had `set_runtime_kind` called on it is detected as legacy-classified.
    ///
    /// This test should be updated (or deleted together with `legacy_classify_runtime_kind_from_string`)
    /// once all live producers migrate to the typed path.
    #[test]
    fn all_known_live_producers_use_typed_runtime_kind() {
        // Representative sample of kind strings emitted by live producers in
        // src/cli/commands/live_half_duplex.rs and src/web/server.rs.
        // All of these currently go through the legacy fallback because no
        // live producer calls set_runtime_kind() yet.
        let legacy_kinds = [
            "capture_started",
            "speech_started",
            "breath_group_opened",
            "breath_group_closed",
            "asr_started",
            "asr_finished",
            "transcript",
            "first_llm_token",
            "tts_enqueue_finished",
            "playback_finished",
        ];

        for kind in legacy_kinds {
            let event = LiveTraceEvent::new(SessionId::new(), 1, kind, ts(1_100), ts(1_000));
            assert!(
                event_used_legacy_classification(&event),
                "expected '{kind}' to use legacy classification — once this producer \
                 calls set_runtime_kind() flip this assertion to assert!(!...)"
            );
        }
    }

    /// Verifies that a producer which calls `set_runtime_kind` with a domain that
    /// differs from the legacy inference is correctly identified as **not** legacy.
    ///
    /// This is the "typed runtime event attachment on a live-produced trace event" test
    /// required by the acceptance criteria.
    #[test]
    fn event_with_typed_kind_bypasses_legacy_detection() {
        // "brand_new_event_name" has no legacy prefix, so the legacy fallback would
        // classify it as Other.  A producer that explicitly sets BrowserInput gets
        // a different domain → the event is NOT considered legacy-classified.
        let mut event = LiveTraceEvent::new(
            SessionId::new(),
            3,
            "brand_new_event_name",
            ts(1_100),
            ts(1_000),
        );
        event.set_runtime_kind(
            TypedRuntimeEvent::BrowserInput(subtype("brand_new_event_name")).into(),
        );

        assert!(
            !event_used_legacy_classification(&event),
            "a producer that calls set_runtime_kind should not be detected as legacy"
        );
        assert_eq!(
            event.runtime_event.as_ref().unwrap().kind,
            RuntimeEventKind::BrowserInput(subtype("brand_new_event_name")),
        );
    }

    /// Shows the ideal migration pattern for a live producer:
    /// create the event, then immediately call set_runtime_kind before emitting.
    #[test]
    fn live_producer_pattern_attaches_typed_runtime_event() {
        let session_id = SessionId::new();
        let mut event = LiveTraceEvent::new(session_id, 2, "capture_started", ts(1_050), ts(1_000));

        // The typed path: producer explicitly declares the domain.
        event.set_runtime_kind(
            TypedRuntimeEvent::Hearing(RuntimeEventSubtype {
                kind: "capture_started".to_string(),
                text: None,
                reason: None,
                artifact: None,
            })
            .into(),
        );

        let runtime = RuntimeEvent::from_live_trace_event(&event);
        assert!(
            matches!(runtime.kind, RuntimeEventKind::Hearing(_)),
            "live producer should attach a Hearing domain for capture_started"
        );
    }
}
