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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_ref: Option<ViewerClipAudioRef>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_ref: Option<ViewerClipAudioRef>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewerClipAudioRef {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_ms: Option<u64>,
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
    let mut live_asr_streams = Vec::<(u64, TimedWordStream)>::new();
    let mut live_tts_revision_streams = Vec::<(u64, TimedWordStream)>::new();
    for event in events {
        if let Some(stream) = live_asr_timed_word_stream_from_event(event) {
            live_asr_streams.push((event.elapsed_ms, stream));
            continue;
        }
        if let Some(stream) = live_tts_timed_word_stream_from_event(event) {
            live_tts_revision_streams.push((event.elapsed_ms, stream));
            continue;
        }
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
    if !live_asr_streams.is_empty() {
        lane_events.entry("User transcript").or_default();
    }
    if !live_tts_revision_streams.is_empty() {
        lane_events.entry("Pete intended speech").or_default();
    }

    lane_events
        .into_iter()
        .enumerate()
        .map(|(stream_index, (label, snippets))| {
            if label == "User transcript" && !live_asr_streams.is_empty() {
                let mut streams = live_asr_streams.iter().collect::<Vec<_>>();
                streams.sort_by_key(|(elapsed_ms, _)| *elapsed_ms);
                let mut words = Vec::new();
                let mut next_word_id = 1u64;
                for (_elapsed_ms, stream) in streams {
                    for mut word in stream.words.iter().cloned() {
                        word.id = WordId(next_word_id);
                        next_word_id = next_word_id.saturating_add(1);
                        words.push(word);
                    }
                }
                return ViewerWordLane {
                    label: label.to_string(),
                    stream: TimedWordStream {
                        id: WordStreamId(stream_index as u64 + 1),
                        source: WordStreamSource::LiveAsr,
                        words,
                    },
                };
            }
            if label == "Pete intended speech" && !live_tts_revision_streams.is_empty() {
                let mut streams = live_tts_revision_streams.iter().collect::<Vec<_>>();
                streams.sort_by_key(|(elapsed_ms, _)| *elapsed_ms);
                let mut words = Vec::new();
                let mut next_word_id = 1u64;
                for (_elapsed_ms, stream) in streams {
                    for mut word in stream.words.iter().cloned() {
                        word.id = WordId(next_word_id);
                        next_word_id = next_word_id.saturating_add(1);
                        words.push(word);
                    }
                }
                return ViewerWordLane {
                    label: label.to_string(),
                    stream: TimedWordStream {
                        id: WordStreamId(stream_index as u64 + 1),
                        source: WordStreamSource::SyntheticSpeech,
                        words,
                    },
                };
            }

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
    let mut pending_starts: HashMap<
        (u64, String),
        (u64, Option<Value>, Option<ViewerClipAudioRef>),
    > = HashMap::new();

    for event in events {
        if event.kind == "asr_timed_word_stream" || event.kind == "tts_timed_word_stream_revision" {
            continue;
        }
        let lane = lane_for_kind(&event.kind).to_string();
        let metadata = Some(metadata_from_event(event));
        let audio_ref = event_audio_ref(event);

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
                        audio_ref: audio_ref.clone(),
                    });
                    continue;
                }
            }

            pending_starts.insert(
                (event.turn, base_kind.to_string()),
                (event.elapsed_ms, metadata.clone(), audio_ref.clone()),
            );
            markers.push(ViewerMarker {
                lane,
                kind: event.kind.clone(),
                at_ms: event.elapsed_ms,
                label: Some(humanize_kind(&event.kind)),
                metadata,
                audio_ref,
            });
            continue;
        }

        if let Some(base_kind) = end_kind_to_base_kind(&event.kind) {
            if let Some((start_ms, start_metadata, start_audio_ref)) =
                pending_starts.remove(&(event.turn, base_kind.to_string()))
            {
                spans.push(ViewerEvent {
                    lane,
                    kind: base_kind.to_string(),
                    start_ms,
                    end_ms: Some(event.elapsed_ms.max(start_ms)),
                    label: Some(humanize_kind(base_kind)),
                    metadata: merge_metadata(start_metadata, metadata),
                    audio_ref: start_audio_ref.or(audio_ref.clone()),
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
            audio_ref,
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
    if let Some(artifact) = event.artifact.as_ref() {
        metadata.insert("artifact".to_string(), artifact.clone());
    }
    Value::Object(metadata)
}

fn live_asr_timed_word_stream_from_event(event: &LiveTraceEvent) -> Option<TimedWordStream> {
    if event.kind != "asr_timed_word_stream" {
        return None;
    }
    let artifact = event.artifact.as_ref()?;
    serde_json::from_value(artifact.clone()).ok()
}

fn live_tts_timed_word_stream_from_event(event: &LiveTraceEvent) -> Option<TimedWordStream> {
    if event.kind != "tts_timed_word_stream_revision" {
        return None;
    }
    let artifact = event.artifact.as_ref()?;
    serde_json::from_value(artifact.clone()).ok()
}

fn event_audio_ref(event: &LiveTraceEvent) -> Option<ViewerClipAudioRef> {
    clip_audio_ref_from_value(event.artifact.as_ref()?)
}

fn clip_audio_ref_from_value(value: &Value) -> Option<ViewerClipAudioRef> {
    let Value::Object(map) = value else {
        return None;
    };
    if let Some(nested) = map
        .get("audio_ref")
        .or_else(|| map.get("audio"))
        .or_else(|| map.get("clip"))
    {
        if let Some(audio_ref) = clip_audio_ref_from_value(nested) {
            return Some(audio_ref);
        }
    }
    let url = map
        .get("url")
        .or_else(|| map.get("audio_url"))
        .or_else(|| map.get("path"))
        .or_else(|| map.get("audio_path"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())?
        .to_string();
    Some(ViewerClipAudioRef {
        url,
        start_ms: clip_audio_ref_ms(
            map.get("start_ms")
                .or_else(|| map.get("clip_start_ms"))
                .or_else(|| map.get("at_ms")),
        ),
        end_ms: clip_audio_ref_ms(map.get("end_ms").or_else(|| map.get("clip_end_ms"))),
    })
}

fn clip_audio_ref_ms(value: Option<&Value>) -> Option<u64> {
    value.and_then(|value| match value {
        Value::Number(number) => {
            if let Some(unsigned) = number.as_u64() {
                Some(unsigned)
            } else {
                number
                    .as_i64()
                    .and_then(|signed| u64::try_from(signed).ok())
            }
        }
        _ => None,
    })
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
            artifact: None,
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

    #[test]
    fn uses_live_asr_timed_word_stream_artifact_for_user_lane() {
        let mut stream_event = event(1, "asr_timed_word_stream", 300);
        let artifact_stream = TimedWordStream {
            id: WordStreamId(9),
            source: WordStreamSource::LiveAsr,
            words: vec![WordNode {
                id: WordId(1),
                text: "hello".to_string(),
                lexical_span: Some(TextSpan { start: 0, end: 5 }),
                timing: Some(WordTiming {
                    start_ms: 100,
                    end_ms: 260,
                }),
                timing_confidence: Some(0.91),
                commitment: WordCommitment::Final,
                boundary_source: BoundarySource::Whisper,
                audio_ref: None,
            }],
        };
        stream_event.artifact = Some(serde_json::to_value(artifact_stream).unwrap());
        let mut transcript = event(1, "transcript", 300);
        transcript.text = Some("hello".to_string());

        let payload = live_trace_events_to_viewer_payload(&[transcript, stream_event]);
        let lane = payload
            .streams
            .iter()
            .find(|lane| lane.label == "User transcript")
            .expect("user transcript lane should be present");
        assert_eq!(lane.stream.source, WordStreamSource::LiveAsr);
        assert_eq!(lane.stream.words.len(), 1);
        assert_eq!(lane.stream.words[0].text, "hello");
        assert_eq!(
            lane.stream.words[0].timing,
            Some(WordTiming {
                start_ms: 100,
                end_ms: 260
            })
        );
        assert_eq!(lane.stream.words[0].timing_confidence, Some(0.91));
    }

    #[test]
    fn uses_live_tts_timed_word_stream_revisions_for_pete_lane() {
        let mut provisional = event(1, "tts_timed_word_stream_revision", 300);
        let provisional_stream = TimedWordStream {
            id: WordStreamId(10),
            source: WordStreamSource::SyntheticSpeech,
            words: vec![
                WordNode {
                    id: WordId(1),
                    text: "sure".to_string(),
                    lexical_span: Some(TextSpan { start: 0, end: 4 }),
                    timing: None,
                    timing_confidence: None,
                    commitment: WordCommitment::Hypothetical,
                    boundary_source: BoundarySource::Predicted,
                    audio_ref: None,
                },
                WordNode {
                    id: WordId(2),
                    text: "wait".to_string(),
                    lexical_span: Some(TextSpan { start: 5, end: 9 }),
                    timing: None,
                    timing_confidence: None,
                    commitment: WordCommitment::Cancelled,
                    boundary_source: BoundarySource::Predicted,
                    audio_ref: None,
                },
            ],
        };
        provisional.artifact = Some(serde_json::to_value(provisional_stream).unwrap());

        let mut committed = event(1, "tts_timed_word_stream_revision", 320);
        let committed_stream = TimedWordStream {
            id: WordStreamId(10),
            source: WordStreamSource::SyntheticSpeech,
            words: vec![WordNode {
                id: WordId(1),
                text: "sure".to_string(),
                lexical_span: Some(TextSpan { start: 0, end: 4 }),
                timing: None,
                timing_confidence: None,
                commitment: WordCommitment::Final,
                boundary_source: BoundarySource::Predicted,
                audio_ref: None,
            }],
        };
        committed.artifact = Some(serde_json::to_value(committed_stream).unwrap());

        let payload = live_trace_events_to_viewer_payload(&[provisional, committed]);
        let lane = payload
            .streams
            .iter()
            .find(|lane| lane.label == "Pete intended speech")
            .expect("pete intended speech lane should be present");
        assert_eq!(lane.stream.source, WordStreamSource::SyntheticSpeech);
        assert!(
            lane.stream
                .words
                .iter()
                .any(|word| word.commitment == WordCommitment::Hypothetical)
        );
        assert!(
            lane.stream
                .words
                .iter()
                .any(|word| word.commitment == WordCommitment::Cancelled)
        );
        assert!(
            lane.stream
                .words
                .iter()
                .any(|word| word.commitment == WordCommitment::Final)
        );
    }

    #[test]
    fn exposes_event_audio_refs_from_artifacts() {
        let mut playback_started = event(1, "playback_started", 900);
        playback_started.artifact = Some(serde_json::json!({
            "url": "clips/turn-001-playback.wav",
            "start_ms": 120,
            "end_ms": 460
        }));
        let mut overlap_started = event(1, "overlap_started", 930);
        overlap_started.artifact = Some(serde_json::json!({
            "audio": {
                "path": "clips/turn-001-overlap.wav",
                "clip_start_ms": 0,
                "clip_end_ms": 210
            }
        }));
        let overlap_ended = event(1, "overlap_ended", 1140);

        let payload = live_trace_events_to_viewer_payload(&[
            playback_started,
            overlap_started,
            overlap_ended,
        ]);

        let playback_marker = payload
            .markers
            .iter()
            .find(|marker| marker.kind == "playback_started")
            .expect("playback marker should exist");
        assert_eq!(
            playback_marker.audio_ref,
            Some(ViewerClipAudioRef {
                url: "clips/turn-001-playback.wav".to_string(),
                start_ms: Some(120),
                end_ms: Some(460),
            })
        );

        let overlap_event = payload
            .events
            .iter()
            .find(|event| event.kind == "overlap")
            .expect("overlap span should exist");
        assert_eq!(
            overlap_event.audio_ref,
            Some(ViewerClipAudioRef {
                url: "clips/turn-001-overlap.wav".to_string(),
                start_ms: Some(0),
                end_ms: Some(210),
            })
        );
    }
}
