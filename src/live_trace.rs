use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::time::ExactTimestamp;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LiveTraceEvent {
    pub turn: u64,
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
        turn: u64,
        kind: impl Into<String>,
        at: ExactTimestamp,
        session_started_at: ExactTimestamp,
    ) -> Self {
        let t_unix_ns = unix_nanos_u64(at);
        let started_unix_ns = unix_nanos_u64(session_started_at);
        Self {
            turn,
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
    session_started_at: ExactTimestamp,
    sink: S,
    pending_turn: Option<PendingTurn>,
    pending_suppression: Option<PendingSuppression>,
}

impl<S> LiveTraceRecorder<S>
where
    S: LiveTraceSink,
{
    pub fn new(session_started_at: ExactTimestamp, sink: S) -> Self {
        Self {
            session_started_at,
            sink,
            pending_turn: None,
            pending_suppression: None,
        }
    }

    pub fn into_sink(self) -> S {
        self.sink
    }

    pub fn event(&self, turn: u64, kind: impl Into<String>, at: ExactTimestamp) -> LiveTraceEvent {
        LiveTraceEvent::new(turn, kind, at, self.session_started_at)
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
        let mut event = LiveTraceEvent::new(1, "asr_finished", ts(1_250), ts(1_000));
        event.text = Some("hello".to_string());
        writer.emit(event).unwrap();

        let raw = std::fs::read_to_string(&path).unwrap();
        let line: Value = serde_json::from_str(raw.trim()).unwrap();
        assert_eq!(line["turn"], 1);
        assert_eq!(line["kind"], "asr_finished");
        assert_eq!(line["elapsed_ms"], 250);
        assert_eq!(line["text"], "hello");

        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap()).unwrap();
    }

    #[test]
    fn sse_broadcaster_replays_history_to_new_subscribers() {
        let mut broadcaster = SseBroadcaster::new();
        broadcaster
            .emit(LiveTraceEvent::new(
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
