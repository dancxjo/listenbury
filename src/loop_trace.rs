use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Duration;

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TraceEvent {
    pub monotonic_ns: u64,
    pub subsystem: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<Value>,
}

impl TraceEvent {
    pub fn new(
        at: Duration,
        subsystem: impl Into<String>,
        kind: impl Into<String>,
        payload: Option<Value>,
    ) -> Self {
        Self {
            monotonic_ns: at.as_nanos().min(u128::from(u64::MAX)) as u64,
            subsystem: subsystem.into(),
            kind: kind.into(),
            payload,
        }
    }

    pub fn monotonic_ms(&self) -> f64 {
        self.monotonic_ns as f64 / 1_000_000.0
    }

    pub fn payload_mode(&self) -> Option<&str> {
        self.payload
            .as_ref()
            .and_then(|payload| payload.get("mode"))
            .and_then(Value::as_str)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MockLoopTraceConfig {
    pub duration: Duration,
    pub self_hearing: bool,
}

impl Default for MockLoopTraceConfig {
    fn default() -> Self {
        Self {
            duration: Duration::from_secs(20),
            self_hearing: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LatencyBucket {
    pub label: String,
    pub start_event: String,
    pub end_event: String,
    pub duration_ms: f64,
    pub mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LatencySummary {
    pub total_events: usize,
    pub duration_ms: f64,
    pub buckets: Vec<LatencyBucket>,
}

impl LatencySummary {
    pub fn format_pretty(&self) -> String {
        let mut lines = vec![
            "Loop trace latency summary".to_string(),
            format!("events: {}", self.total_events),
            format!("trace duration: {:.1} ms", self.duration_ms),
            String::new(),
            "Major buckets:".to_string(),
        ];

        for bucket in &self.buckets {
            lines.push(format!(
                "  {:<34} {:>8.1} ms  {}",
                bucket.label, bucket.duration_ms, bucket.mode
            ));
        }

        lines.join("\n")
    }
}

pub fn write_trace_jsonl(path: impl AsRef<Path>, events: &[TraceEvent]) -> anyhow::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create trace directory {}", parent.display()))?;
    }

    let file =
        File::create(path).with_context(|| format!("create trace file {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for event in events {
        serde_json::to_writer(&mut writer, event).context("serialize loop trace event")?;
        writer
            .write_all(b"\n")
            .context("terminate loop trace JSONL line")?;
    }
    writer.flush().context("flush loop trace JSONL")
}

pub fn mock_interaction_trace(config: MockLoopTraceConfig) -> Vec<TraceEvent> {
    let requested_ms = config.duration.as_millis().min(u128::from(u64::MAX)) as u64;
    let mut trace = MockTraceBuilder::default();

    trace.emit(
        0,
        "audio_capture",
        "capture_start",
        mock_payload(json!({
            "source": "mock_mic",
            "requested_duration_ms": requested_ms
        })),
    );
    trace.emit(
        20,
        "audio_capture",
        "audio_frame_received",
        mock_payload(json!({"frame_index": 0, "sample_rate_hz": 16000, "samples": 320})),
    );
    trace.emit(
        40,
        "audio_capture",
        "audio_frame_received",
        mock_payload(json!({"frame_index": 1, "sample_rate_hz": 16000, "samples": 320})),
    );
    trace.emit(
        80,
        "vad",
        "frame_decision",
        mock_payload(json!({"backend": "mock_vad", "frame_index": 7, "is_speech": true, "speech_prob": 0.91})),
    );
    trace.emit(
        80,
        "utterance",
        "speech_start",
        mock_payload(json!({"backend": "mock_vad", "pre_roll_frames": 20})),
    );
    trace.emit(
        120,
        "asr",
        "partial_result",
        mock_payload(json!({"text": "hello", "stability": 0.54})),
    );
    trace.emit(
        320,
        "audio_capture",
        "audio_frame_received",
        mock_payload(json!({"frame_index": 15, "sample_rate_hz": 16000, "samples": 320})),
    );
    trace.emit(
        760,
        "asr",
        "partial_result",
        mock_payload(json!({"text": "hello listenbury", "stability": 0.82})),
    );
    trace.emit(
        900,
        "vad",
        "frame_decision",
        mock_payload(json!({"backend": "mock_vad", "frame_index": 89, "is_speech": false, "speech_prob": 0.04})),
    );
    trace.emit(
        900,
        "utterance",
        "speech_end",
        mock_payload(json!({"backend": "mock_vad", "reason": "Silence", "duration_ms": 820, "post_roll_frames": 30})),
    );
    trace.emit(
        930,
        "asr",
        "final_result",
        mock_payload(json!({"text": "Hello, Listenbury.", "confidence": 0.96})),
    );
    append_mock_downstream_trace(
        &mut trace.events,
        Duration::from_millis(930),
        config.self_hearing,
    );

    trace.events
}

pub fn append_mock_downstream_trace(
    events: &mut Vec<TraceEvent>,
    after_asr_final: Duration,
    self_hearing: bool,
) {
    let base_ms = after_asr_final.as_millis().min(u128::from(u64::MAX)) as u64;
    let mut trace = MockTraceBuilder::default();
    trace.emit(
        base_ms.saturating_add(30),
        "prompt",
        "assembly_start",
        mock_payload(json!({"turn": 1, "source": "mock_context"})),
    );
    trace.emit(
        base_ms.saturating_add(85),
        "prompt",
        "assembly_end",
        mock_payload(json!({"turn": 1, "prompt_bytes": 384})),
    );
    trace.emit(
        base_ms.saturating_add(95),
        "llm",
        "request_start",
        mock_payload(json!({"backend": "mock_llm", "max_tokens": 32})),
    );
    trace.emit(
        base_ms.saturating_add(190),
        "llm",
        "first_token",
        mock_payload(json!({"text": "Hi"})),
    );
    trace.emit(
        base_ms.saturating_add(320),
        "llm",
        "sentence_boundary",
        mock_payload(json!({"text": "Hi there.", "boundary": "sentence"})),
    );
    trace.emit(
        base_ms.saturating_add(340),
        "llm",
        "breath_group_boundary",
        mock_payload(json!({"text": "Hi there.", "boundary": "breath_group"})),
    );
    trace.emit(
        base_ms.saturating_add(350),
        "speech_planner",
        "first_unit",
        mock_payload(json!({"unit_index": 0, "text": "Hi"})),
    );
    trace.emit(
        base_ms.saturating_add(380),
        "audio_render",
        "render_start",
        mock_payload(json!({"backend": "mock_mouth"})),
    );
    trace.emit(
        base_ms.saturating_add(530),
        "audio_render",
        "render_end",
        mock_payload(json!({"audio_ms": 890, "sample_rate_hz": 16000})),
    );
    trace.emit(
        base_ms.saturating_add(560),
        "playback",
        "playback_start",
        mock_payload(json!({"device": "mock_speaker"})),
    );

    if self_hearing {
        trace.emit(
            base_ms.saturating_add(1_170),
            "self_hearing",
            "capture_start",
            mock_payload(json!({"source": "mock_loopback"})),
        );
        trace.emit(
            base_ms.saturating_add(1_250),
            "self_hearing",
            "transcription",
            mock_payload(json!({"text": "Hi there.", "classified_as": "self"})),
        );
    }

    trace.emit(
        base_ms.saturating_add(1_450),
        "playback",
        "playback_end",
        mock_payload(json!({"device": "mock_speaker"})),
    );

    events.extend(trace.events);
}

pub fn summarize_latency(events: &[TraceEvent]) -> LatencySummary {
    let mut first_by_key: BTreeMap<(&str, &str), &TraceEvent> = BTreeMap::new();
    for event in events {
        first_by_key
            .entry((event.subsystem.as_str(), event.kind.as_str()))
            .or_insert(event);
    }

    let mut buckets = Vec::new();
    push_bucket(
        &mut buckets,
        &first_by_key,
        "capture to utterance speech start",
        ("audio_capture", "capture_start"),
        ("utterance", "speech_start"),
    );
    push_bucket(
        &mut buckets,
        &first_by_key,
        "user utterance duration",
        ("utterance", "speech_start"),
        ("utterance", "speech_end"),
    );
    push_bucket(
        &mut buckets,
        &first_by_key,
        "utterance end to ASR final",
        ("utterance", "speech_end"),
        ("asr", "final_result"),
    );
    push_bucket(
        &mut buckets,
        &first_by_key,
        "prompt assembly",
        ("prompt", "assembly_start"),
        ("prompt", "assembly_end"),
    );
    push_bucket(
        &mut buckets,
        &first_by_key,
        "LLM time to first token",
        ("llm", "request_start"),
        ("llm", "first_token"),
    );
    push_bucket(
        &mut buckets,
        &first_by_key,
        "first token to first speech unit",
        ("llm", "first_token"),
        ("speech_planner", "first_unit"),
    );
    push_bucket(
        &mut buckets,
        &first_by_key,
        "audio render",
        ("audio_render", "render_start"),
        ("audio_render", "render_end"),
    );
    push_bucket(
        &mut buckets,
        &first_by_key,
        "render end to playback start",
        ("audio_render", "render_end"),
        ("playback", "playback_start"),
    );
    push_bucket(
        &mut buckets,
        &first_by_key,
        "utterance end to playback start",
        ("utterance", "speech_end"),
        ("playback", "playback_start"),
    );
    push_bucket(
        &mut buckets,
        &first_by_key,
        "capture to playback start",
        ("audio_capture", "capture_start"),
        ("playback", "playback_start"),
    );
    push_bucket(
        &mut buckets,
        &first_by_key,
        "playback duration",
        ("playback", "playback_start"),
        ("playback", "playback_end"),
    );
    push_bucket(
        &mut buckets,
        &first_by_key,
        "self-hearing transcription lag",
        ("self_hearing", "capture_start"),
        ("self_hearing", "transcription"),
    );

    let duration_ms = events
        .first()
        .zip(events.last())
        .map(|(first, last)| {
            last.monotonic_ns.saturating_sub(first.monotonic_ns) as f64 / 1_000_000.0
        })
        .unwrap_or_default();

    LatencySummary {
        total_events: events.len(),
        duration_ms,
        buckets,
    }
}

pub fn real_payload(payload: Value) -> Value {
    payload_with_mode(payload, "real")
}

pub fn mock_payload(payload: Value) -> Value {
    payload_with_mode(payload, "mock")
}

fn payload_with_mode(mut payload: Value, mode: &'static str) -> Value {
    if let Value::Object(map) = &mut payload {
        map.insert("mode".to_string(), Value::String(mode.to_string()));
    }
    payload
}

fn push_bucket(
    buckets: &mut Vec<LatencyBucket>,
    first_by_key: &BTreeMap<(&str, &str), &TraceEvent>,
    label: &str,
    start_key: (&str, &str),
    end_key: (&str, &str),
) {
    let (Some(start), Some(end)) = (first_by_key.get(&start_key), first_by_key.get(&end_key))
    else {
        return;
    };
    let duration_ms = end.monotonic_ns.saturating_sub(start.monotonic_ns) as f64 / 1_000_000.0;
    let mode = match (start.payload_mode(), end.payload_mode()) {
        (Some(start), Some(end)) if start == end => start.to_string(),
        (Some("real"), Some("mock")) | (Some("mock"), Some("real")) => "mixed".to_string(),
        (Some(mode), None) | (None, Some(mode)) => mode.to_string(),
        _ => "unknown".to_string(),
    };
    buckets.push(LatencyBucket {
        label: label.to_string(),
        start_event: format!("{}.{}", start.subsystem, start.kind),
        end_event: format!("{}.{}", end.subsystem, end.kind),
        duration_ms,
        mode,
    });
}

#[derive(Default)]
struct MockTraceBuilder {
    events: Vec<TraceEvent>,
}

impl MockTraceBuilder {
    fn emit(&mut self, at_ms: u64, subsystem: &str, kind: &str, payload: Value) {
        self.events.push(TraceEvent::new(
            Duration::from_millis(at_ms),
            subsystem,
            kind,
            Some(payload),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_interaction_emits_ordered_trace_events_and_sane_durations() {
        let events = mock_interaction_trace(MockLoopTraceConfig::default());
        assert!(events.len() >= 20);

        for window in events.windows(2) {
            assert!(
                window[0].monotonic_ns <= window[1].monotonic_ns,
                "trace events must be monotonic"
            );
        }

        assert!(
            events.iter().any(|event| {
                event.subsystem == "audio_capture" && event.kind == "capture_start"
            })
        );
        assert!(
            events
                .iter()
                .any(|event| event.subsystem == "llm" && event.kind == "first_token")
        );
        assert!(
            events
                .iter()
                .any(|event| { event.subsystem == "speech_planner" && event.kind == "first_unit" })
        );
        assert!(
            events
                .iter()
                .any(|event| event.subsystem == "playback" && event.kind == "playback_end")
        );

        let summary = summarize_latency(&events);
        assert_eq!(summary.total_events, events.len());
        assert!(summary.duration_ms > 2_000.0);

        let bucket = summary
            .buckets
            .iter()
            .find(|bucket| bucket.label == "utterance end to playback start")
            .expect("summary should include utterance end to playback start");
        assert!(bucket.duration_ms > 0.0);
        assert!(bucket.duration_ms < 1_000.0);

        let prompt_bucket = summary
            .buckets
            .iter()
            .find(|bucket| bucket.label == "prompt assembly")
            .expect("summary should include prompt assembly");
        assert_eq!(prompt_bucket.duration_ms, 55.0);
    }
}
