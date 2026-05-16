use std::collections::{BTreeMap, HashMap};
use std::io::BufRead;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::live_trace::LiveTraceEvent;
use crate::word::{
    BoundarySource, TextSpan, TimedWordStream, WordCommitment, WordId, WordNode, WordStreamId,
    WordStreamSource, WordTiming,
};

const DEFAULT_WORD_SLOT_MS: u64 = 240;
const DEFAULT_LANE_TAIL_BUFFER_MS: u64 = 200;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewerPayload {
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<ViewerAudio>,
    pub streams: Vec<ViewerWordLane>,
    #[serde(default)]
    pub events: Vec<ViewerEvent>,
    #[serde(default)]
    pub markers: Vec<ViewerMarker>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewerAudio {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewerWordLane {
    pub label: String,
    pub stream: TimedWordStream,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewerEvent {
    pub lane: String,
    pub kind: String,
    pub start_ms: u64,
    pub end_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewerMarker {
    pub lane: String,
    pub kind: String,
    pub at_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

pub fn live_trace_jsonl_to_viewer_payload(input: &str) -> Result<ViewerPayload> {
    let mut events = Vec::new();
    for (line_index, line) in input.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let event: LiveTraceEvent = serde_json::from_str(line).with_context(|| {
            format!(
                "parse live trace JSONL line {} into LiveTraceEvent",
                line_index + 1
            )
        })?;
        events.push(event);
    }
    Ok(live_trace_events_to_viewer_payload(&events))
}

pub fn live_trace_jsonl_reader_to_viewer_payload<R: BufRead>(reader: R) -> Result<ViewerPayload> {
    let mut events = Vec::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("read JSONL line {}", line_index + 1))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let event: LiveTraceEvent = serde_json::from_str(line).with_context(|| {
            format!(
                "parse live trace JSONL line {} into LiveTraceEvent",
                line_index + 1
            )
        })?;
        events.push(event);
    }
    Ok(live_trace_events_to_viewer_payload(&events))
}

pub fn live_trace_events_to_viewer_payload(events: &[LiveTraceEvent]) -> ViewerPayload {
    let mut sorted = events.to_vec();
    sorted.sort_by_key(|event| (event.elapsed_ms, event.t_unix_ns, event.turn));

    let text_lanes = collect_text_lanes(&sorted);
    let (viewer_events, viewer_markers) = collect_event_lanes(&sorted);

    ViewerPayload {
        title: "Listenbury Live Trace".to_string(),
        audio: None,
        streams: text_lanes,
        events: viewer_events,
        markers: viewer_markers,
    }
}

#[derive(Clone, Copy)]
enum TextLaneKind {
    UserTranscript,
    PeteIntendedSpeech,
}

fn collect_text_lanes(events: &[LiveTraceEvent]) -> Vec<ViewerWordLane> {
    let mut lane_events: BTreeMap<&'static str, Vec<(u64, String)>> = BTreeMap::new();
    for event in events {
        let Some(text) = event.text.as_ref().map(|text| text.trim()) else {
            continue;
        };
        if text.is_empty() {
            continue;
        }
        let Some(kind) = text_lane_kind_for_event(&event.kind) else {
            continue;
        };

        let key = match kind {
            TextLaneKind::UserTranscript => "User transcript",
            TextLaneKind::PeteIntendedSpeech => "Pete intended speech",
        };
        lane_events
            .entry(key)
            .or_default()
            .push((event.elapsed_ms, text.to_string()));
    }

    lane_events
        .into_iter()
        .enumerate()
        .map(|(stream_index, (label, snippets))| {
            let source = if label == "User transcript" {
                WordStreamSource::RecordedAudio
            } else {
                WordStreamSource::GeneratedText
            };
            let commitment = if label == "User transcript" {
                WordCommitment::Final
            } else {
                WordCommitment::StableText
            };
            let boundary_source = if label == "User transcript" {
                BoundarySource::Whisper
            } else {
                BoundarySource::Predicted
            };

            let mut words = Vec::new();
            let mut next_word_id = 1u64;

            for (snippet_index, (start_ms, text)) in snippets.iter().enumerate() {
                let end_ms = snippets
                    .get(snippet_index + 1)
                    .map(|(next_ms, _)| *next_ms)
                    .unwrap_or_else(|| {
                        start_ms.saturating_add(
                            DEFAULT_WORD_SLOT_MS * text.split_whitespace().count() as u64
                                + DEFAULT_LANE_TAIL_BUFFER_MS,
                        )
                    });
                let end_ms = end_ms.max(start_ms.saturating_add(1));

                let tokens = split_words_with_spans(text);
                if tokens.is_empty() {
                    continue;
                }
                let token_count = tokens.len();

                let lane_duration = end_ms.saturating_sub(*start_ms).max(token_count as u64);
                let slot_ms = (lane_duration / token_count as u64).max(1);
                for (token_index, (token, span)) in tokens.into_iter().enumerate() {
                    let token_start =
                        start_ms.saturating_add(slot_ms.saturating_mul(token_index as u64));
                    let token_end = if token_index + 1 == token_count {
                        end_ms
                    } else {
                        token_start.saturating_add(slot_ms)
                    };
                    let timing = WordTiming::new(token_start, token_end).unwrap_or(WordTiming {
                        start_ms: token_start,
                        end_ms: token_start,
                    });

                    words.push(WordNode {
                        id: WordId(next_word_id),
                        text: token,
                        lexical_span: Some(span),
                        timing: Some(timing),
                        timing_confidence: None,
                        commitment,
                        boundary_source,
                        audio_ref: None,
                    });
                    next_word_id = next_word_id.saturating_add(1);
                }
            }

            ViewerWordLane {
                label: label.to_string(),
                stream: TimedWordStream {
                    id: WordStreamId(stream_index as u64 + 1),
                    source,
                    words,
                },
            }
        })
        .collect()
}

fn split_words_with_spans(text: &str) -> Vec<(String, TextSpan)> {
    let mut out = Vec::new();
    let mut in_token = false;
    let mut token_start = 0usize;
    for (index, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if in_token {
                let token = &text[token_start..index];
                out.push((
                    token.to_string(),
                    TextSpan {
                        start: token_start,
                        end: index,
                    },
                ));
                in_token = false;
            }
        } else if !in_token {
            token_start = index;
            in_token = true;
        }
    }
    if in_token {
        out.push((
            text[token_start..].to_string(),
            TextSpan {
                start: token_start,
                end: text.len(),
            },
        ));
    }
    out
}

fn collect_event_lanes(events: &[LiveTraceEvent]) -> (Vec<ViewerEvent>, Vec<ViewerMarker>) {
    let mut spans = Vec::new();
    let mut markers = Vec::new();
    let mut pending_starts: HashMap<(u64, String), (u64, Option<Value>)> = HashMap::new();

    for event in events {
        let lane = lane_for_kind(&event.kind).to_string();
        let metadata = Some(metadata_from_event(event));

        if let Some((base_kind, _end_kind)) = start_to_end_kind(&event.kind) {
            if event.kind == "self_hearing_suppression_started" {
                if let Some(until_unix_ns) = event.expected_until_unix_ns {
                    let until_ms = until_unix_ns / 1_000_000;
                    let end_ms = until_ms.max(event.elapsed_ms);
                    spans.push(ViewerEvent {
                        lane,
                        kind: "self_hearing_suppression".to_string(),
                        start_ms: event.elapsed_ms,
                        end_ms: Some(end_ms),
                        label: Some(humanize_kind("self_hearing_suppression")),
                        metadata,
                    });
                    continue;
                }
            }

            pending_starts.insert(
                (event.turn, base_kind.to_string()),
                (event.elapsed_ms, metadata.clone()),
            );
            markers.push(ViewerMarker {
                lane,
                kind: event.kind.clone(),
                at_ms: event.elapsed_ms,
                label: Some(humanize_kind(&event.kind)),
                metadata,
            });
            continue;
        }

        if let Some(base_kind) = end_kind_to_base_kind(&event.kind) {
            if let Some((start_ms, start_metadata)) =
                pending_starts.remove(&(event.turn, base_kind.to_string()))
            {
                spans.push(ViewerEvent {
                    lane,
                    kind: base_kind.to_string(),
                    start_ms,
                    end_ms: Some(event.elapsed_ms.max(start_ms)),
                    label: Some(humanize_kind(base_kind)),
                    metadata: merge_metadata(start_metadata, metadata),
                });
                continue;
            }
        }

        markers.push(ViewerMarker {
            lane,
            kind: event.kind.clone(),
            at_ms: event.elapsed_ms,
            label: Some(humanize_kind(&event.kind)),
            metadata,
        });
    }

    spans.sort_by_key(|event| (event.start_ms, event.end_ms.unwrap_or(event.start_ms)));
    markers.sort_by_key(|marker| marker.at_ms);
    (spans, markers)
}

fn merge_metadata(first: Option<Value>, second: Option<Value>) -> Option<Value> {
    match (first, second) {
        (Some(Value::Object(mut left)), Some(Value::Object(right))) => {
            for (key, value) in right {
                left.insert(key, value);
            }
            Some(Value::Object(left))
        }
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (Some(left), Some(_)) => Some(left),
        (None, None) => None,
    }
}

fn start_to_end_kind(kind: &str) -> Option<(&str, &str)> {
    if let Some(base) = kind.strip_suffix("_started") {
        return Some((base, "ended"));
    }
    None
}

fn end_kind_to_base_kind(kind: &str) -> Option<&str> {
    kind.strip_suffix("_ended")
        .or_else(|| kind.strip_suffix("_finished"))
        .or_else(|| kind.strip_suffix("_stopped"))
}

fn text_lane_kind_for_event(kind: &str) -> Option<TextLaneKind> {
    match kind {
        "transcript" => Some(TextLaneKind::UserTranscript),
        "first_safe_speech_unit_emitted" => Some(TextLaneKind::PeteIntendedSpeech),
        _ => None,
    }
}

fn lane_for_kind(kind: &str) -> &'static str {
    if kind.contains("overlap") {
        "Overlap"
    } else if kind.contains("interrupt") || kind.contains("yield") {
        "Interruption"
    } else if kind.contains("environment") {
        "Environmental"
    } else if kind.contains("asr")
        || kind.contains("llm")
        || kind.contains("tts")
        || kind.contains("playback")
        || kind.contains("capture")
        || kind.contains("suppression")
    {
        "Latency"
    } else {
        "Runtime"
    }
}

fn metadata_from_event(event: &LiveTraceEvent) -> Value {
    let mut metadata = Map::new();
    metadata.insert("turn".to_string(), Value::from(event.turn));
    metadata.insert("t_unix_ns".to_string(), Value::from(event.t_unix_ns));
    if let Some(text) = event.text.as_ref() {
        metadata.insert("text".to_string(), Value::from(text.clone()));
    }
    if let Some(confidence) = event.confidence {
        metadata.insert("confidence".to_string(), Value::from(confidence));
    }
    if let Some(group_id) = event.group_id.as_ref() {
        metadata.insert("group_id".to_string(), Value::from(group_id.clone()));
    }
    if let Some(reason) = event.reason.as_ref() {
        metadata.insert("reason".to_string(), Value::from(reason.clone()));
    }
    if let Some(face) = event.face.as_ref() {
        metadata.insert("face".to_string(), Value::from(face.clone()));
    }
    if let Some(unit_kind) = event.unit_kind.as_ref() {
        metadata.insert("unit_kind".to_string(), Value::from(unit_kind.clone()));
    }
    if let Some(expected_until_unix_ns) = event.expected_until_unix_ns {
        metadata.insert(
            "expected_until_unix_ns".to_string(),
            Value::from(expected_until_unix_ns),
        );
    }
    Value::Object(metadata)
}

fn humanize_kind(kind: &str) -> String {
    kind.split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            let Some(first) = chars.next() else {
                return String::new();
            };
            first.to_ascii_uppercase().to_string() + chars.as_str()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(turn: u64, kind: &str, elapsed_ms: u64) -> LiveTraceEvent {
        LiveTraceEvent {
            turn,
            kind: kind.to_string(),
            t_unix_ns: elapsed_ms * 1_000_000,
            elapsed_ms,
            text: None,
            confidence: None,
            group_id: None,
            reason: None,
            face: None,
            unit_kind: None,
            expected_until_unix_ns: None,
        }
    }

    #[test]
    fn converts_live_trace_into_word_and_event_lanes() {
        let mut transcript = event(1, "transcript", 420);
        transcript.text = Some("hello there".to_string());
        let mut speech = event(1, "first_safe_speech_unit_emitted", 690);
        speech.text = Some("hi back".to_string());
        speech.unit_kind = Some("complete_sentence".to_string());

        let overlap_started = event(1, "overlap_started", 730);
        let overlap_ended = event(1, "overlap_ended", 890);
        let llm_marker = event(1, "first_llm_token", 560);
        let playback_marker = event(1, "playback_started", 900);
        let interruption = event(1, "interruption_detected", 910);

        let payload = live_trace_events_to_viewer_payload(&[
            transcript,
            speech,
            overlap_started,
            overlap_ended,
            llm_marker,
            playback_marker,
            interruption,
        ]);

        assert_eq!(payload.title, "Listenbury Live Trace");
        assert!(
            payload
                .streams
                .iter()
                .any(|lane| lane.label == "User transcript"),
            "user transcript lane should be present"
        );
        assert!(
            payload
                .streams
                .iter()
                .any(|lane| lane.label == "Pete intended speech"),
            "intended speech lane should be present"
        );
        assert!(
            payload
                .events
                .iter()
                .any(|event| event.lane == "Overlap" && event.kind == "overlap"),
            "overlap span should be captured as an event"
        );
        assert!(
            payload
                .markers
                .iter()
                .any(|marker| marker.lane == "Latency" && marker.kind == "first_llm_token"),
            "latency markers should be present"
        );
        assert!(
            payload
                .markers
                .iter()
                .any(|marker| marker.lane == "Interruption"
                    && marker.kind == "interruption_detected"),
            "interruption markers should be visible"
        );
    }

    #[test]
    fn parses_jsonl_lines_to_payload() {
        let jsonl = r#"
{"turn":1,"kind":"transcript","t_unix_ns":1000000000,"elapsed_ms":1000,"text":"hello there"}
{"turn":1,"kind":"first_llm_token","t_unix_ns":1200000000,"elapsed_ms":1200}
{"turn":1,"kind":"playback_started","t_unix_ns":1400000000,"elapsed_ms":1400}
"#;
        let payload = live_trace_jsonl_to_viewer_payload(jsonl).expect("jsonl should parse");
        assert!(!payload.streams.is_empty());
        assert!(!payload.markers.is_empty());
    }
}
