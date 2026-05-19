use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::soundscape::{SoundscapeId, VoiceAttribution, VoiceId, VoiceLabel};
pub use crate::speech_timeline::SessionId;
use crate::speech_timeline::{
    AudioClipId, SpanId as TimelineSpanId, SpeechUnitId, TranscriptRevisionId, TurnId, UtteranceId,
};
use crate::time::ExactTimestamp;

pub const TRACE_SESSION_FORMAT: &str = "listenbury.live-session.v1";
pub const TRACE_SESSION_METADATA_FILE: &str = "metadata.json";
pub const TRACE_SESSION_EVENTS_FILE: &str = "events.jsonl";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LiveTraceEvent {
    pub turn: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub soundscape_id: Option<SoundscapeId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_id: Option<VoiceId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_label: Option<VoiceLabel>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub voice_attributions: Vec<VoiceAttribution>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<SessionId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<TurnId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub utterance_id: Option<UtteranceId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speech_unit_id: Option<SpeechUnitId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcript_revision_id: Option<TranscriptRevisionId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span_id: Option<TimelineSpanId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_clip_id: Option<AudioClipId>,
    pub kind: String,
    pub t_unix_ns: u64,
    pub elapsed_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub face: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit_kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_until_unix_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact: Option<Value>,
}

impl LiveTraceEvent {
    pub fn new(
        session_id: SessionId,
        turn: u64,
        kind: impl Into<String>,
        at: ExactTimestamp,
        session_started_at: ExactTimestamp,
    ) -> Self {
        let t_unix_ns = unix_nanos_u64(at);
        let started_unix_ns = unix_nanos_u64(session_started_at);
        Self {
            turn,
            soundscape_id: None,
            voice_id: None,
            voice_label: None,
            voice_attributions: Vec::new(),
            session_id: Some(session_id),
            turn_id: Some(TurnId(turn)),
            utterance_id: None,
            speech_unit_id: None,
            transcript_revision_id: None,
            span_id: None,
            audio_clip_id: None,
            kind: kind.into(),
            t_unix_ns,
            elapsed_ms: t_unix_ns
                .saturating_sub(started_unix_ns)
                .saturating_div(1_000_000),
            text: None,
            confidence: None,
            group_id: None,
            reason: None,
            face: None,
            unit_kind: None,
            expected_until_unix_ns: None,
            artifact: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TraceRuntimeMetadata {
    pub command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub configuration: Map<String, Value>,
}

impl TraceRuntimeMetadata {
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            mode: None,
            configuration: Map::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TraceSessionMetadata {
    pub format: String,
    pub session_id: SessionId,
    pub session_started_at_unix_ns: u64,
    pub recorded_at_unix_ns: u64,
    pub events_path: String,
    pub runtime: TraceRuntimeMetadata,
}

impl TraceSessionMetadata {
    pub fn new(
        session_id: SessionId,
        session_started_at: ExactTimestamp,
        runtime: TraceRuntimeMetadata,
    ) -> Self {
        let session_started_at_unix_ns = unix_nanos_u64(session_started_at);
        Self {
            format: TRACE_SESSION_FORMAT.to_string(),
            session_id,
            session_started_at_unix_ns,
            recorded_at_unix_ns: session_started_at_unix_ns,
            events_path: TRACE_SESSION_EVENTS_FILE.to_string(),
            runtime,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TraceSessionEnvelope {
    pub metadata: TraceSessionMetadata,
    pub events: Vec<LiveTraceEvent>,
}

pub trait LiveTraceSink {
    fn emit(&mut self, event: LiveTraceEvent) -> anyhow::Result<()>;
}

impl LiveTraceSink for Vec<LiveTraceEvent> {
    fn emit(&mut self, event: LiveTraceEvent) -> anyhow::Result<()> {
        self.push(event);
        Ok(())
    }
}

impl<T> LiveTraceSink for Option<T>
where
    T: LiveTraceSink,
{
    fn emit(&mut self, event: LiveTraceEvent) -> anyhow::Result<()> {
        if let Some(sink) = self.as_mut() {
            sink.emit(event)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct JsonlTraceWriter {
    writer: BufWriter<File>,
    pub path: PathBuf,
}

impl JsonlTraceWriter {
    pub fn create(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create trace directory {}", parent.display()))?;
        }
        let file = File::create(&path)
            .with_context(|| format!("create trace file at {}", path.display()))?;
        Ok(Self {
            writer: BufWriter::new(file),
            path,
        })
    }

    pub fn write<T>(&mut self, value: &T) -> anyhow::Result<()>
    where
        T: Serialize,
    {
        serde_json::to_writer(&mut self.writer, value).context("serialize trace event")?;
        self.writer
            .write_all(b"\n")
            .context("terminate trace JSONL line")?;
        self.writer.flush().context("flush trace JSONL")?;
        Ok(())
    }
}

impl LiveTraceSink for JsonlTraceWriter {
    fn emit(&mut self, event: LiveTraceEvent) -> anyhow::Result<()> {
        self.write(&event)
    }
}

#[derive(Debug)]
pub struct TraceSessionWriter {
    pub metadata: TraceSessionMetadata,
    pub metadata_path: PathBuf,
    pub events_path: PathBuf,
    events: JsonlTraceWriter,
}

impl TraceSessionWriter {
    pub fn create(
        directory: impl AsRef<Path>,
        metadata: TraceSessionMetadata,
    ) -> anyhow::Result<Self> {
        let directory = directory.as_ref();
        std::fs::create_dir_all(directory)
            .with_context(|| format!("create trace session directory {}", directory.display()))?;
        let metadata_path = directory.join(TRACE_SESSION_METADATA_FILE);
        let events_path = directory.join(&metadata.events_path);
        let metadata_json =
            serde_json::to_vec_pretty(&metadata).context("serialize trace session metadata")?;
        std::fs::write(&metadata_path, metadata_json)
            .with_context(|| format!("write trace metadata {}", metadata_path.display()))?;
        let events = JsonlTraceWriter::create(&events_path)?;
        Ok(Self {
            metadata,
            metadata_path,
            events_path,
            events,
        })
    }

    pub fn write<T>(&mut self, value: &T) -> anyhow::Result<()>
    where
        T: Serialize,
    {
        self.events.write(value)
    }
}

impl LiveTraceSink for TraceSessionWriter {
    fn emit(&mut self, event: LiveTraceEvent) -> anyhow::Result<()> {
        self.events.emit(event)
    }
}

#[derive(Debug)]
pub enum DiskTraceWriter {
    Jsonl(JsonlTraceWriter),
    Session(TraceSessionWriter),
}

impl DiskTraceWriter {
    pub fn create(path: impl AsRef<Path>, metadata: TraceSessionMetadata) -> anyhow::Result<Self> {
        let path = path.as_ref();
        if trace_path_looks_like_jsonl(path) {
            Ok(Self::Jsonl(JsonlTraceWriter::create(path)?))
        } else {
            Ok(Self::Session(TraceSessionWriter::create(path, metadata)?))
        }
    }
}

impl LiveTraceSink for DiskTraceWriter {
    fn emit(&mut self, event: LiveTraceEvent) -> anyhow::Result<()> {
        match self {
            Self::Jsonl(writer) => writer.emit(event),
            Self::Session(writer) => writer.emit(event),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingSuppression {
    turn: u64,
    expected_until: ExactTimestamp,
}

#[derive(Debug, Clone, PartialEq)]
struct PendingTurn {
    turn: u64,
    events: Vec<LiveTraceEvent>,
}

#[derive(Debug)]
pub struct LiveTraceRecorder<S> {
    session_id: SessionId,
    session_started_at: ExactTimestamp,
    sink: S,
    pending_turn: Option<PendingTurn>,
    pending_suppression: Option<PendingSuppression>,
}

impl<S> LiveTraceRecorder<S>
where
    S: LiveTraceSink,
{
    pub fn with_session_id(
        session_id: SessionId,
        session_started_at: ExactTimestamp,
        sink: S,
    ) -> Self {
        Self {
            session_id,
            session_started_at,
            sink,
            pending_turn: None,
            pending_suppression: None,
        }
    }

    pub fn new(session_started_at: ExactTimestamp, sink: S) -> Self {
        Self::with_session_id(SessionId::new(), session_started_at, sink)
    }

    pub fn session_id(&self) -> SessionId {
        self.session_id
    }

    pub fn into_sink(self) -> S {
        self.sink
    }

    pub fn event(&self, turn: u64, kind: impl Into<String>, at: ExactTimestamp) -> LiveTraceEvent {
        LiveTraceEvent::new(self.session_id, turn, kind, at, self.session_started_at)
    }

    pub fn emit(&mut self, event: LiveTraceEvent) -> anyhow::Result<()> {
        self.sink.emit(event)
    }

    pub fn emit_now(
        &mut self,
        turn: u64,
        kind: impl Into<String>,
        at: ExactTimestamp,
    ) -> anyhow::Result<()> {
        self.emit(self.event(turn, kind, at))
    }

    pub fn buffer(&mut self, event: LiveTraceEvent) {
        let pending = self.pending_turn.get_or_insert_with(|| PendingTurn {
            turn: event.turn,
            events: Vec::new(),
        });
        if pending.turn != event.turn {
            tracing::warn!(
                pending_turn = pending.turn,
                new_turn = event.turn,
                "discarding uncommitted live trace turn"
            );
            debug_assert_eq!(
                pending.turn, event.turn,
                "unexpected live trace turn rollover without commit/discard"
            );
            self.pending_turn = Some(PendingTurn {
                turn: event.turn,
                events: vec![event],
            });
            return;
        }
        pending.events.push(event);
    }

    pub fn buffer_now(&mut self, turn: u64, kind: impl Into<String>, at: ExactTimestamp) {
        self.buffer(self.event(turn, kind, at));
    }

    pub fn commit_turn(&mut self, turn: u64) -> anyhow::Result<()> {
        let Some(pending) = self.pending_turn.take() else {
            return Ok(());
        };
        if pending.turn != turn {
            self.pending_turn = Some(pending);
            return Ok(());
        }
        for event in pending.events {
            self.emit(event)?;
        }
        Ok(())
    }

    pub fn discard_turn(&mut self, turn: u64) {
        if self
            .pending_turn
            .as_ref()
            .is_some_and(|pending| pending.turn == turn)
        {
            self.pending_turn = None;
        }
    }

    pub fn begin_suppression(
        &mut self,
        turn: u64,
        started_at: ExactTimestamp,
        expected_until: ExactTimestamp,
    ) -> anyhow::Result<()> {
        let mut event = self.event(turn, "self_hearing_suppression_started", started_at);
        event.expected_until_unix_ns = Some(unix_nanos_u64(expected_until));
        self.emit(event)?;
        self.pending_suppression = Some(PendingSuppression {
            turn,
            expected_until,
        });
        Ok(())
    }

    pub fn maybe_end_suppression(&mut self, observed_at: ExactTimestamp) -> anyhow::Result<()> {
        let Some(pending) = self.pending_suppression else {
            return Ok(());
        };
        if observed_at.unix_nanos < pending.expected_until.unix_nanos {
            return Ok(());
        }
        self.pending_suppression = None;
        self.emit_now(
            pending.turn,
            "self_hearing_suppression_ended",
            pending.expected_until,
        )
    }
}

/// A sink that emits to two downstream sinks simultaneously.
pub struct TeeSink<A, B>(pub A, pub B);

impl<A, B> LiveTraceSink for TeeSink<A, B>
where
    A: LiveTraceSink,
    B: LiveTraceSink,
{
    fn emit(&mut self, event: LiveTraceEvent) -> anyhow::Result<()> {
        self.0.emit(event.clone())?;
        self.1.emit(event)?;
        Ok(())
    }
}

/// Broadcasts live trace events to subscribed SSE clients.
///
/// Clone is cheap – all clones share the same sender list.
#[derive(Clone, Debug)]
pub struct SseBroadcaster {
    senders: Arc<Mutex<Vec<crossbeam_channel::Sender<LiveTraceEvent>>>>,
    history: Arc<Mutex<Vec<LiveTraceEvent>>>,
}

impl SseBroadcaster {
    pub fn new() -> Self {
        Self {
            senders: Arc::new(Mutex::new(Vec::new())),
            history: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Create a new receiver that will receive the session history and future events.
    pub fn subscribe(&self) -> crossbeam_channel::Receiver<LiveTraceEvent> {
        let (tx, rx) = crossbeam_channel::unbounded();
        match self.history.lock() {
            Ok(history) => {
                for event in history.iter().cloned() {
                    if tx.send(event).is_err() {
                        return rx;
                    }
                }
            }
            Err(error) => {
                tracing::error!(
                    "SseBroadcaster history mutex poisoned in subscribe; receiver will not get replayed events: {error}"
                );
            }
        }
        match self.senders.lock() {
            Ok(mut senders) => {
                senders.push(tx);
            }
            Err(error) => {
                tracing::error!(
                    "SseBroadcaster senders mutex poisoned in subscribe; receiver will not get events: {error}"
                );
            }
        }
        rx
    }
}

impl Default for SseBroadcaster {
    fn default() -> Self {
        Self::new()
    }
}

impl LiveTraceSink for SseBroadcaster {
    fn emit(&mut self, event: LiveTraceEvent) -> anyhow::Result<()> {
        match self.history.lock() {
            Ok(mut history) => {
                history.push(event.clone());
            }
            Err(error) => {
                tracing::error!(
                    "SseBroadcaster history mutex poisoned in emit; event will not be replayed to future subscribers: {error}"
                );
            }
        }
        match self.senders.lock() {
            Ok(mut senders) => {
                senders.retain(|tx| tx.send(event.clone()).is_ok());
            }
            Err(error) => {
                tracing::error!(
                    "SseBroadcaster senders mutex poisoned in emit; dropping event broadcast: {error}"
                );
            }
        }
        Ok(())
    }
}

pub fn trace_path_looks_like_jsonl(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .is_some_and(|extension| extension.eq_ignore_ascii_case("jsonl"))
}

fn trace_session_metadata_path(path: &Path) -> Option<PathBuf> {
    if path.file_name() == Some(OsStr::new(TRACE_SESSION_METADATA_FILE)) {
        return Some(path.to_path_buf());
    }
    if trace_path_looks_like_jsonl(path) || path.is_file() {
        return None;
    }
    Some(path.join(TRACE_SESSION_METADATA_FILE))
}

pub fn read_trace_session_metadata(path: &Path) -> anyhow::Result<TraceSessionMetadata> {
    let metadata_path = trace_session_metadata_path(path)
        .ok_or_else(|| anyhow::anyhow!("{} is not a trace session path", path.display()))?;
    let raw = std::fs::read(&metadata_path)
        .with_context(|| format!("read trace session metadata {}", metadata_path.display()))?;
    serde_json::from_slice(&raw)
        .with_context(|| format!("parse trace session metadata {}", metadata_path.display()))
}

pub fn read_live_trace_events(path: &Path) -> anyhow::Result<Vec<LiveTraceEvent>> {
    let events_path = if let Some(metadata_path) = trace_session_metadata_path(path) {
        let metadata = read_trace_session_metadata(&metadata_path)?;
        metadata_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(metadata.events_path)
    } else {
        path.to_path_buf()
    };
    let input = File::open(&events_path)
        .with_context(|| format!("open live trace events at {}", events_path.display()))?;
    let reader = std::io::BufReader::new(input);
    let mut events = Vec::new();
    for (line_index, line) in std::io::BufRead::lines(reader).enumerate() {
        let line = line.with_context(|| format!("read JSONL line {}", line_index + 1))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let event = serde_json::from_str(line).with_context(|| {
            format!(
                "parse live trace JSONL line {} in {}",
                line_index + 1,
                events_path.display()
            )
        })?;
        events.push(event);
    }
    Ok(events)
}

pub fn read_trace_jsonl(path: &Path) -> anyhow::Result<String> {
    let events_path = if let Some(metadata_path) = trace_session_metadata_path(path) {
        let metadata = read_trace_session_metadata(&metadata_path)?;
        metadata_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(metadata.events_path)
    } else {
        path.to_path_buf()
    };
    std::fs::read_to_string(&events_path)
        .with_context(|| format!("read trace from {}", events_path.display()))
}

pub fn read_trace_session(path: &Path) -> anyhow::Result<TraceSessionEnvelope> {
    let events = read_live_trace_events(path)?;
    let metadata = if trace_session_metadata_path(path).is_some() {
        read_trace_session_metadata(path)?
    } else {
        synthesize_trace_session_metadata(path, &events)
    };
    Ok(TraceSessionEnvelope { metadata, events })
}

fn synthesize_trace_session_metadata(
    path: &Path,
    events: &[LiveTraceEvent],
) -> TraceSessionMetadata {
    let session_id = events
        .iter()
        .find_map(|event| event.session_id)
        .unwrap_or_default();
    let session_started_at_unix_ns = events
        .iter()
        .map(|event| {
            event
                .t_unix_ns
                .saturating_sub(event.elapsed_ms.saturating_mul(1_000_000))
        })
        .min()
        .unwrap_or_default();
    let mut runtime = TraceRuntimeMetadata::new("imported-jsonl");
    runtime.configuration.insert(
        "source_path".to_string(),
        Value::String(path.display().to_string()),
    );
    runtime.configuration.insert(
        "source_kind".to_string(),
        Value::String("jsonl".to_string()),
    );
    TraceSessionMetadata {
        format: TRACE_SESSION_FORMAT.to_string(),
        session_id,
        session_started_at_unix_ns,
        recorded_at_unix_ns: session_started_at_unix_ns,
        events_path: path
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or(TRACE_SESSION_EVENTS_FILE)
            .to_string(),
        runtime,
    }
}

fn unix_nanos_u64(timestamp: ExactTimestamp) -> u64 {
    match u64::try_from(timestamp.unix_nanos) {
        Ok(value) => value,
        Err(_) => {
            tracing::warn!(
                unix_nanos = timestamp.unix_nanos.to_string(),
                "live trace timestamp exceeded u64 range"
            );
            debug_assert!(false, "live trace timestamp exceeded u64 range");
            u64::MAX
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn ts(ms: u64) -> ExactTimestamp {
        ExactTimestamp {
            unix_nanos: u128::from(ms) * 1_000_000,
        }
    }

    #[test]
    fn jsonl_writer_creates_parent_dirs_and_serializes_event() {
        let path = std::env::temp_dir().join(format!(
            "listenbury-live-trace-{}/nested/trace.jsonl",
            uuid::Uuid::new_v4()
        ));
        let mut writer = JsonlTraceWriter::create(&path).unwrap();
        let mut event =
            LiveTraceEvent::new(SessionId::new(), 1, "asr_finished", ts(1_250), ts(1_000));
        event.text = Some("hello".to_string());
        writer.emit(event).unwrap();

        let raw = std::fs::read_to_string(&path).unwrap();
        let line: Value = serde_json::from_str(raw.trim()).unwrap();
        assert_eq!(line["turn"], 1);
        assert!(line["session_id"].is_string());
        assert_eq!(line["turn_id"], 1);
        assert_eq!(line["kind"], "asr_finished");
        assert_eq!(line["elapsed_ms"], 250);
        assert_eq!(line["text"], "hello");

        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap()).unwrap();
    }

    #[test]
    fn trace_session_writer_persists_metadata_and_events() {
        let root =
            std::env::temp_dir().join(format!("listenbury-live-session-{}", uuid::Uuid::new_v4()));
        let session_id = SessionId::new();
        let metadata = TraceSessionMetadata::new(
            session_id,
            ts(1_000),
            TraceRuntimeMetadata {
                command: "listen".to_string(),
                mode: Some("half_duplex".to_string()),
                configuration: Map::from_iter([(
                    "vad".to_string(),
                    Value::String("webrtc".to_string()),
                )]),
            },
        );
        let mut writer = TraceSessionWriter::create(&root, metadata.clone()).unwrap();
        let mut event = LiveTraceEvent::new(session_id, 1, "transcript", ts(1_250), ts(1_000));
        event.text = Some("hello".to_string());
        writer.emit(event).unwrap();

        let saved_metadata: TraceSessionMetadata =
            serde_json::from_str(&std::fs::read_to_string(root.join("metadata.json")).unwrap())
                .unwrap();
        assert_eq!(saved_metadata, metadata);

        let envelope = read_trace_session(&root).unwrap();
        assert_eq!(envelope.metadata, metadata);
        assert_eq!(envelope.events.len(), 1);
        assert_eq!(envelope.events[0].text.as_deref(), Some("hello"));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn read_trace_session_synthesizes_metadata_for_jsonl_input() {
        let path = std::env::temp_dir().join(format!(
            "listenbury-live-trace-{}.jsonl",
            uuid::Uuid::new_v4()
        ));
        let session_id = SessionId::new();
        let mut writer = JsonlTraceWriter::create(&path).unwrap();
        writer
            .emit(LiveTraceEvent::new(
                session_id,
                2,
                "playback_started",
                ts(1_500),
                ts(1_000),
            ))
            .unwrap();

        let envelope = read_trace_session(&path).unwrap();
        assert_eq!(envelope.metadata.session_id, session_id);
        assert_eq!(envelope.metadata.runtime.command, "imported-jsonl");
        assert_eq!(
            envelope.metadata.events_path,
            path.file_name().unwrap().to_string_lossy()
        );
        assert_eq!(envelope.events[0].kind, "playback_started");

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn sse_broadcaster_replays_history_to_new_subscribers() {
        let mut broadcaster = SseBroadcaster::new();
        broadcaster
            .emit(LiveTraceEvent::new(
                SessionId::new(),
                0,
                "capture_started",
                ts(1_000),
                ts(1_000),
            ))
            .unwrap();

        let rx = broadcaster.subscribe();
        let replayed = rx.try_recv().expect("subscriber should receive history");
        assert_eq!(replayed.kind, "capture_started");

        broadcaster
            .emit(LiveTraceEvent::new(
                SessionId::new(),
                1,
                "speech_started",
                ts(1_100),
                ts(1_000),
            ))
            .unwrap();
        let live = rx
            .try_recv()
            .expect("subscriber should receive future event");
        assert_eq!(live.kind, "speech_started");
    }

    #[test]
    fn synthetic_turn_timeline_flushes_committed_events_and_discards_empty_turns() {
        let mut trace = LiveTraceRecorder::new(ts(1_000), Vec::new());

        trace.emit_now(0, "capture_started", ts(1_000)).unwrap();
        trace.buffer_now(1, "speech_started", ts(1_100));
        let mut opened = trace.event(1, "breath_group_opened", ts(1_120));
        opened.group_id = Some("group-1".to_string());
        trace.buffer(opened);
        let mut closed = trace.event(1, "breath_group_closed", ts(1_400));
        closed.group_id = Some("group-1".to_string());
        closed.reason = Some("silence".to_string());
        trace.buffer(closed);
        trace.buffer_now(1, "asr_started", ts(1_410));
        trace.buffer_now(1, "asr_finished", ts(1_470));
        let mut transcript = trace.event(1, "transcript", ts(1_470));
        transcript.text = Some("hello there".to_string());
        trace.buffer(transcript);
        trace.commit_turn(1).unwrap();
        trace
            .emit_now(1, "llm_generation_started", ts(1_500))
            .unwrap();
        trace.emit_now(1, "first_llm_token", ts(1_560)).unwrap();
        let mut speech_unit = trace.event(1, "first_safe_speech_unit_emitted", ts(1_610));
        speech_unit.text = Some("Hi.".to_string());
        speech_unit.unit_kind = Some("complete_sentence".to_string());
        trace.emit(speech_unit).unwrap();
        trace
            .emit_now(1, "first_tts_audio_frame_available", ts(1_690))
            .unwrap();
        trace.emit_now(1, "playback_started", ts(1_700)).unwrap();
        trace.begin_suppression(1, ts(1_690), ts(2_050)).unwrap();
        trace.maybe_end_suppression(ts(2_100)).unwrap();
        trace.emit_now(1, "playback_finished", ts(1_980)).unwrap();

        trace.buffer_now(2, "speech_started", ts(3_000));
        trace.buffer_now(2, "asr_started", ts(3_050));
        trace.buffer_now(2, "asr_finished", ts(3_090));
        trace.discard_turn(2);

        let events = trace.into_sink();
        let kinds = events
            .iter()
            .map(|event| event.kind.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            kinds,
            vec![
                "capture_started",
                "speech_started",
                "breath_group_opened",
                "breath_group_closed",
                "asr_started",
                "asr_finished",
                "transcript",
                "llm_generation_started",
                "first_llm_token",
                "first_safe_speech_unit_emitted",
                "first_tts_audio_frame_available",
                "playback_started",
                "self_hearing_suppression_started",
                "self_hearing_suppression_ended",
                "playback_finished",
            ]
        );
        assert_eq!(events[3].elapsed_ms, 400);
        assert_eq!(events[6].text.as_deref(), Some("hello there"));
        assert_eq!(events[9].unit_kind.as_deref(), Some("complete_sentence"));
        assert_eq!(events[12].expected_until_unix_ns, Some(2_050_000_000));
        assert!(events.iter().all(|event| event.turn != 2));
    }
}
