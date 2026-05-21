use serde::{Deserialize, Serialize};
use serde_json::Value;

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
        if let Some(speech_unit_id) = event.speech_unit_id {
            causality.push(format!("speech_unit:{}", speech_unit_id.0));
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
            } => (Some(text.clone()), None, None),
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
        Self {
            id: format!("memory:{}:{}", occurred_at.unix_nanos, kind_name),
            session_id: None,
            timestamp: occurred_at,
            monotonic_ms: occurred_at.unix_nanos.saturating_div(1_000_000) as u64,
            source: EventSource::MemoryIngestion,
            kind: RuntimeEventKind::MemoryIngestion(RuntimeEventSubtype {
                kind: kind_name.clone(),
                text,
                reason,
                artifact,
            }),
            causality,
            correlation: vec![format!("memory_kind:{kind_name}")],
        }
    }
}

/// Legacy adapter for historical stringly `LiveTraceEvent.kind` values.
///
/// Keep this only while traces/events still rely on string classification.
/// Delete after all live producers attach typed `runtime_event.kind`.
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
        || kind.starts_with("speech_unit_")
        || kind == "speculative_speech_updated"
        || kind == "first_safe_speech_unit_emitted"
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live_trace::{LiveTraceEvent, SessionId};

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
}
