use std::collections::{BTreeMap, HashMap};
use std::io::BufRead;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::live_trace::LiveTraceEvent;
use crate::live_trace::{TraceSessionAudioArtifact, TraceSessionEnvelope};
use crate::span::{
    Alignment, AlignmentGraph, AlignmentKind, Cursor, Modality, Span as RuntimeSpan,
    SpanId as RuntimeSpanId, SpanRevision, SpanState, TextId,
};
use crate::speech::prosody_timing::ProsodyTimingPlan;
use crate::trace::timestamp_sanity::normalize_live_trace_events;
use crate::word::{
    BoundarySource, TextSpan, TimedWordStream, WordCommitment, WordId, WordNode, WordStreamId,
    WordStreamSource, WordTiming, attach_cmudict_pronunciations,
};

const DEFAULT_WORD_SLOT_MS: u64 = 240;
const DEFAULT_LANE_TAIL_BUFFER_MS: u64 = 200;
const RUNTIME_TRACE_TEXT_ID: TextId = TextId(1);
const USER_TRANSCRIPT_LANE: &str = "User transcript";
const PETE_INTENDED_SPEECH_LANE: &str = "Pete intended speech";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewerPayload {
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<ViewerAudio>,
    pub streams: Vec<ViewerWordLane>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span_graph: Option<ViewerSpanGraph>,
    #[serde(default)]
    pub events: Vec<ViewerEvent>,
    #[serde(default)]
    pub markers: Vec<ViewerMarker>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewerAudio {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acoustic_analysis_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate_hz: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channels: Option<u16>,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewerSpanGraph {
    pub text_id: TextId,
    pub spans: Vec<RuntimeSpan<String>>,
    #[serde(default)]
    pub alignments: Vec<Alignment>,
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
    let sorted = normalize_live_trace_events(events);

    let text_lanes = collect_text_lanes(&sorted);
    let (viewer_events, viewer_markers) = collect_event_lanes(&sorted);

    let mut payload = ViewerPayload {
        title: "Listenbury Live Trace".to_string(),
        audio: None,
        streams: text_lanes,
        span_graph: None,
        events: viewer_events,
        markers: viewer_markers,
    };
    enrich_payload_pronunciations(&mut payload);
    payload.span_graph = build_runtime_span_graph(&payload);
    payload
}

pub fn trace_session_to_viewer_payload(session: &TraceSessionEnvelope) -> ViewerPayload {
    let mut payload = live_trace_events_to_viewer_payload(&session.events);
    payload.audio = canonical_session_audio(&session.metadata.audio_artifacts);
    payload
}

fn canonical_session_audio(artifacts: &[TraceSessionAudioArtifact]) -> Option<ViewerAudio> {
    let artifact = artifacts.first()?;
    Some(ViewerAudio {
        url: format!("/api/session-audio/{}", artifact.artifact_id),
        acoustic_analysis_url: artifact
            .acoustic_analysis_path
            .as_ref()
            .map(|_| format!("/api/session-audio/{}/acoustic.json", artifact.artifact_id)),
        duration_ms: Some(artifact.duration_ms),
        artifact_id: Some(artifact.artifact_id.clone()),
        sample_rate_hz: Some(artifact.sample_rate_hz),
        channels: Some(artifact.channels),
    })
}

fn enrich_payload_pronunciations(payload: &mut ViewerPayload) {
    for lane in &mut payload.streams {
        attach_cmudict_pronunciations(&mut lane.stream);
    }
}

fn build_runtime_span_graph(payload: &ViewerPayload) -> Option<ViewerSpanGraph> {
    let text_id = RUNTIME_TRACE_TEXT_ID;
    let mut next_span_id = 1u64;
    let mut spans = Vec::<RuntimeSpan<String>>::new();
    let mut timed_word_spans = Vec::<(RuntimeSpanId, String, u64, u64)>::new();
    let mut playback_spans = Vec::<(RuntimeSpanId, u64, u64)>::new();

    for lane in &payload.streams {
        for word in &lane.stream.words {
            let Some(timing) = word.timing else {
                continue;
            };
            let span_id = take_next_runtime_span_id(&mut next_span_id);
            let state = span_state_for_word_commitment(word.commitment);
            spans.push(RuntimeSpan {
                id: span_id,
                text_id,
                modality: Modality::Word,
                state,
                start: Cursor(timing.start_ms),
                end: Some(Cursor(timing.end_ms)),
                contents: word.text.clone(),
                confidence: word.timing_confidence.unwrap_or(0.5),
                stability: span_state_stability(state),
                needs_review: false,
                repair_attempts: 0,
                revisions: Vec::<SpanRevision<String>>::new(),
            });
            timed_word_spans.push((span_id, lane.label.clone(), timing.start_ms, timing.end_ms));
        }
    }

    for event in &payload.events {
        let Some(end_ms) = event.end_ms else {
            continue;
        };
        let span_id = take_next_runtime_span_id(&mut next_span_id);
        spans.push(RuntimeSpan {
            id: span_id,
            text_id,
            modality: event_modality(&event.kind),
            state: SpanState::Committed,
            start: Cursor(event.start_ms),
            end: Some(Cursor(end_ms)),
            contents: event.kind.clone(),
            confidence: 1.0,
            stability: 1.0,
            needs_review: false,
            repair_attempts: 0,
            revisions: Vec::<SpanRevision<String>>::new(),
        });
        if event.kind == "playback" {
            playback_spans.push((span_id, event.start_ms, end_ms));
        }
    }

    if spans.is_empty() {
        return None;
    }

    let mut graph = AlignmentGraph::new();
    for (playback_span_id, playback_start_ms, playback_end_ms) in playback_spans {
        for (word_span_id, lane_label, word_start_ms, word_end_ms) in &timed_word_spans {
            if lane_label != PETE_INTENDED_SPEECH_LANE {
                continue;
            }
            if !ranges_overlap(
                playback_start_ms,
                playback_end_ms,
                *word_start_ms,
                *word_end_ms,
            ) {
                continue;
            }
            let relation = if playback_start_ms <= *word_start_ms && playback_end_ms >= *word_end_ms
            {
                AlignmentKind::Contains
            } else {
                AlignmentKind::Overlaps
            };
            graph.add_alignment(Alignment::new(
                playback_span_id,
                *word_span_id,
                1.0,
                relation,
            ));
        }
    }

    Some(ViewerSpanGraph {
        text_id,
        spans,
        alignments: graph.alignments,
    })
}

fn span_state_for_word_commitment(commitment: WordCommitment) -> SpanState {
    match commitment {
        WordCommitment::Hypothetical => SpanState::Hypothesis,
        WordCommitment::StableText
        | WordCommitment::Prepared
        | WordCommitment::Playable
        | WordCommitment::Played => SpanState::Stable,
        WordCommitment::Final | WordCommitment::Confirmed => SpanState::Committed,
        WordCommitment::Cancelled => SpanState::Deprecated,
    }
}

fn take_next_runtime_span_id(next_span_id: &mut u64) -> RuntimeSpanId {
    let span_id = RuntimeSpanId(*next_span_id);
    *next_span_id = next_span_id.saturating_add(1);
    span_id
}

fn span_state_stability(state: SpanState) -> f32 {
    match state {
        SpanState::Hypothesis => 0.25,
        SpanState::Stable => 0.7,
        SpanState::Committed => 1.0,
        SpanState::Revised => 0.8,
        SpanState::Deprecated => 0.0,
    }
}

fn event_modality(kind: &str) -> Modality {
    match kind {
        "playback" | "speech" | "asr" | "capture" | "self_hearing_suppression" => Modality::Audio,
        "tts" | "llm" => Modality::Clause,
        "overlap" => Modality::BreathGroup,
        _ if kind.contains("speech")
            || kind.contains("playback")
            || kind.contains("asr")
            || kind.contains("capture")
            || kind.contains("suppression") =>
        {
            Modality::Audio
        }
        _ if kind.contains("tts") || kind.contains("llm") => Modality::Clause,
        _ if kind.contains("overlap") => Modality::BreathGroup,
        _ => Modality::Semantic,
    }
}

/// Returns true when two inclusive millisecond ranges overlap.
///
/// Boundary-touching ranges are treated as overlapping (`a_end == b_start`).
fn ranges_overlap(a_start: u64, a_end: u64, b_start: u64, b_end: u64) -> bool {
    a_start <= b_end && a_end >= b_start
}

#[derive(Clone, Copy)]
enum TextLaneKind {
    UnknownVoice,
    PeteVoice,
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
            TextLaneKind::UnknownVoice => USER_TRANSCRIPT_LANE,
            TextLaneKind::PeteVoice => PETE_INTENDED_SPEECH_LANE,
        };
        lane_events
            .entry(key)
            .or_default()
            .push((event.elapsed_ms, text.to_string()));
    }
    if !live_asr_streams.is_empty() {
        lane_events.entry(USER_TRANSCRIPT_LANE).or_default();
    }
    if !live_tts_revision_streams.is_empty() {
        lane_events.entry(PETE_INTENDED_SPEECH_LANE).or_default();
    }

    lane_events
        .into_iter()
        .enumerate()
        .map(|(stream_index, (label, snippets))| {
            if label == USER_TRANSCRIPT_LANE && !live_asr_streams.is_empty() {
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
            if label == PETE_INTENDED_SPEECH_LANE && !live_tts_revision_streams.is_empty() {
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

            let source = if label == USER_TRANSCRIPT_LANE {
                WordStreamSource::RecordedAudio
            } else {
                WordStreamSource::GeneratedText
            };
            let commitment = if label == USER_TRANSCRIPT_LANE {
                WordCommitment::Final
            } else {
                WordCommitment::StableText
            };
            let boundary_source = if label == USER_TRANSCRIPT_LANE {
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
                        pronunciation: None,
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
    type PendingStartKey = (u64, String);
    type PendingStartValue = (u64, Option<Value>, Option<ViewerClipAudioRef>);

    let mut spans = Vec::new();
    let mut markers = Vec::new();
    let mut pending_starts: HashMap<PendingStartKey, PendingStartValue> = HashMap::new();

    for event in events {
        if event.kind == "asr_timed_word_stream" || event.kind == "tts_timed_word_stream_revision" {
            continue;
        }
        if event.kind == "prosody_plan" {
            append_prosody_plan_lanes(event, &mut spans, &mut markers);
            continue;
        }
        let lane = lane_for_kind(&event.kind).to_string();
        let metadata = Some(metadata_from_event(event));
        let audio_ref = event_audio_ref(event);

        if let Some((base_kind, _end_kind)) = start_to_end_kind(&event.kind) {
            if event.kind == "self_hearing_suppression_started"
                && let Some(until_unix_ns) = event.expected_until_unix_ns
            {
                // `expected_until_unix_ns` is an absolute Unix timestamp in
                // nanoseconds.  Convert to session-relative elapsed_ms by
                // subtracting the session start time inferred from this
                // event's own (t_unix_ns, elapsed_ms) pair.
                let session_start_ns = event
                    .t_unix_ns
                    .saturating_sub(event.elapsed_ms.saturating_mul(1_000_000));
                let until_relative_ns = until_unix_ns.saturating_sub(session_start_ns);
                let until_ms = until_relative_ns / 1_000_000;
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

        if let Some(base_kind) = end_kind_to_base_kind(&event.kind)
            && let Some((start_ms, start_metadata, start_audio_ref)) =
                pending_starts.remove(&(event.turn, base_kind.to_string()))
        {
            spans.push(ViewerEvent {
                lane,
                kind: base_kind.to_string(),
                start_ms,
                end_ms: Some(event.elapsed_ms.max(start_ms)),
                label: Some(humanize_kind(base_kind)),
                metadata: merge_span_metadata(start_metadata, metadata, start_ms, event.elapsed_ms),
                audio_ref: start_audio_ref.or(audio_ref.clone()),
            });
            continue;
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

    apply_causal_clamps(&mut spans);
    spans.sort_by_key(|event| (event.start_ms, event.end_ms.unwrap_or(event.start_ms)));
    markers.sort_by_key(|marker| marker.at_ms);
    (spans, markers)
}

fn append_prosody_plan_lanes(
    event: &LiveTraceEvent,
    spans: &mut Vec<ViewerEvent>,
    markers: &mut Vec<ViewerMarker>,
) {
    let Some(plan) = prosody_plan_from_event(event) else {
        return;
    };
    let base_metadata = metadata_from_event(event);
    let audio_ref = event_audio_ref(event);

    for group in &plan.breath_groups {
        let start_ms = seconds_to_ms(group.t0);
        let end_ms = seconds_to_ms(group.t1).max(start_ms);
        spans.push(ViewerEvent {
            lane: "Breath Groups".to_string(),
            kind: "breath_group".to_string(),
            start_ms,
            end_ms: Some(end_ms),
            label: Some("breath group".to_string()),
            metadata: Some(prosody_child_metadata(
                &base_metadata,
                serde_json::json!({
                    "utterance_id": plan.utterance_id,
                    "t0": group.t0,
                    "t1": group.t1,
                }),
            )),
            audio_ref: audio_ref.clone(),
        });
    }

    for (word_index, segment) in plan.segments.iter().enumerate() {
        for (phone_index, phone) in segment.phones.iter().enumerate() {
            let start_ms = seconds_to_ms(phone.t0);
            let end_ms = seconds_to_ms(phone.t1).max(start_ms);
            spans.push(ViewerEvent {
                lane: "Phones".to_string(),
                kind: "phone".to_string(),
                start_ms,
                end_ms: Some(end_ms),
                label: Some(phone.p.clone()),
                metadata: Some(prosody_child_metadata(
                    &base_metadata,
                    serde_json::json!({
                        "utterance_id": plan.utterance_id,
                        "word": segment.word,
                        "word_index": word_index,
                        "phone_index": phone_index,
                        "phone": phone.p,
                        "nucleus": phone.nucleus,
                        "pace_target_ms": phone.pace_target_ms,
                        "t0": phone.t0,
                        "t1": phone.t1,
                    }),
                )),
                audio_ref: audio_ref.clone(),
            });
        }

        if let Some(break_ms) = segment.break_hint_ms {
            markers.push(ViewerMarker {
                lane: "Breaks".to_string(),
                kind: "break".to_string(),
                at_ms: seconds_to_ms(segment.t1),
                label: Some(format!("{break_ms}ms")),
                metadata: Some(prosody_child_metadata(
                    &base_metadata,
                    serde_json::json!({
                        "utterance_id": plan.utterance_id,
                        "word": segment.word,
                        "word_index": word_index,
                        "break_hint_ms": break_ms,
                        "break_reason": segment.break_reason,
                    }),
                )),
                audio_ref: audio_ref.clone(),
            });
        }
    }
}

fn prosody_child_metadata(base_metadata: &Value, child: Value) -> Value {
    let mut metadata = match base_metadata {
        Value::Object(map) => map.clone(),
        value => {
            let mut map = Map::new();
            map.insert("parent".to_string(), value.clone());
            map
        }
    };
    metadata.insert("prosody".to_string(), child);
    Value::Object(metadata)
}

fn apply_causal_clamps(spans: &mut [ViewerEvent]) {
    let mut parent_end_by_span_id = HashMap::<String, u64>::new();
    for span in spans.iter() {
        let Some(span_id) = metadata_string(&span.metadata, "span_id") else {
            continue;
        };
        let end_ms = span.end_ms.unwrap_or(span.start_ms).max(span.start_ms);
        parent_end_by_span_id.insert(span_id, end_ms);
    }

    for span in spans {
        let Some(parent_span_id) = metadata_parent_span_id(&span.metadata) else {
            continue;
        };
        let Some(parent_end_ms) = parent_end_by_span_id.get(&parent_span_id).copied() else {
            continue;
        };
        if span.start_ms < parent_end_ms {
            span.start_ms = parent_end_ms;
            if let Some(end_ms) = span.end_ms.as_mut() {
                *end_ms = (*end_ms).max(parent_end_ms);
            }
        }
    }
}

fn metadata_string(metadata: &Option<Value>, key: &str) -> Option<String> {
    metadata
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|map| map.get(key))
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn metadata_parent_span_id(metadata: &Option<Value>) -> Option<String> {
    metadata
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|map| map.get("cause"))
        .and_then(Value::as_object)
        .and_then(|map| map.get("parent_span_id"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
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

fn merge_span_metadata(
    first: Option<Value>,
    second: Option<Value>,
    start_event_ms: u64,
    end_event_ms: u64,
) -> Option<Value> {
    let mut metadata = match merge_metadata(first, second) {
        Some(Value::Object(map)) => map,
        Some(value) => {
            let mut map = Map::new();
            map.insert("value".to_string(), value);
            map
        }
        None => Map::new(),
    };
    metadata.insert("start_event_ms".to_string(), Value::from(start_event_ms));
    metadata.insert("end_event_ms".to_string(), Value::from(end_event_ms));
    Some(Value::Object(metadata))
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
        "transcript" => Some(TextLaneKind::UnknownVoice),
        "first_safe_speech_unit_emitted" => Some(TextLaneKind::PeteVoice),
        _ => None,
    }
}

fn lane_for_kind(kind: &str) -> &'static str {
    if kind.contains("overlap") {
        "Overlap"
    } else if kind.contains("interrupt") || kind.contains("yield") {
        "Interruption"
    } else if kind.contains("visual_speech") {
        "Visual speech"
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
    metadata.insert("elapsed_ms".to_string(), Value::from(event.elapsed_ms));
    metadata.insert("t_unix_ns".to_string(), Value::from(event.t_unix_ns));
    if let Some(emitter_id) = event.emitter_id.as_ref() {
        metadata.insert("emitter_id".to_string(), Value::from(emitter_id.clone()));
    }
    if let Some(seq_id) = event.seq_id {
        metadata.insert("seq_id".to_string(), Value::from(seq_id));
    }
    if let Some(observed_elapsed_ms) = event.observed_elapsed_ms {
        metadata.insert(
            "observed_elapsed_ms".to_string(),
            Value::from(observed_elapsed_ms),
        );
    }
    if let Some(observed_unix_ns) = event.observed_unix_ns {
        metadata.insert(
            "observed_unix_ns".to_string(),
            Value::from(observed_unix_ns),
        );
    }
    if let Some(ts_local_monotonic) = event.ts_local_monotonic {
        metadata.insert(
            "ts_local_monotonic".to_string(),
            Value::from(ts_local_monotonic),
        );
    }
    if let Some(ts_wall_approx_unix_ns) = event.ts_wall_approx_unix_ns {
        metadata.insert(
            "ts_wall_approx_unix_ns".to_string(),
            Value::from(ts_wall_approx_unix_ns),
        );
    }
    if let Some(cause) = event.cause.as_ref()
        && let Ok(value) = serde_json::to_value(cause)
    {
        metadata.insert("cause".to_string(), value);
    }
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
    if let Some(session_id) = event.session_id {
        metadata.insert(
            "session_id".to_string(),
            Value::from(session_id.0.to_string()),
        );
    }
    if let Some(turn_id) = event.turn_id {
        metadata.insert("turn_id".to_string(), Value::from(turn_id.0));
    }
    if let Some(utterance_id) = event.utterance_id {
        metadata.insert(
            "utterance_id".to_string(),
            Value::from(utterance_id.0.to_string()),
        );
    }
    if let Some(speech_unit_id) = event.speech_unit_id {
        metadata.insert("speech_unit_id".to_string(), Value::from(speech_unit_id.0));
    }
    if let Some(transcript_revision_id) = event.transcript_revision_id {
        metadata.insert(
            "transcript_revision_id".to_string(),
            Value::from(transcript_revision_id.0),
        );
    }
    if let Some(span_id) = event.span_id {
        metadata.insert("span_id".to_string(), Value::from(span_id.0.to_string()));
    }
    if let Some(audio_clip_id) = event.audio_clip_id {
        metadata.insert(
            "audio_clip_id".to_string(),
            Value::from(audio_clip_id.0.to_string()),
        );
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

fn prosody_plan_from_event(event: &LiveTraceEvent) -> Option<ProsodyTimingPlan> {
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
        && let Some(audio_ref) = clip_audio_ref_from_value(nested)
    {
        return Some(audio_ref);
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

fn seconds_to_ms(seconds: f64) -> u64 {
    if !seconds.is_finite() || seconds <= 0.0 {
        return 0;
    }
    (seconds * 1000.0).round() as u64
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
            soundscape_id: None,
            voice_id: None,
            voice_label: None,
            voice_attributions: Vec::new(),
            session_id: None,
            turn_id: None,
            utterance_id: None,
            speech_unit_id: None,
            transcript_revision_id: None,
            span_id: None,
            audio_clip_id: None,
            kind: kind.to_string(),
            source: Some("test".to_string()),
            emitter_id: Some("test".to_string()),
            seq_id: None,
            t_unix_ns: elapsed_ms * 1_000_000,
            elapsed_ms,
            normalized_elapsed_ms: Some(elapsed_ms),
            normalized_unix_ns: Some(elapsed_ms * 1_000_000),
            observed_elapsed_ms: None,
            observed_unix_ns: None,
            ts_local_monotonic: Some(elapsed_ms),
            ts_wall_approx_unix_ns: Some(elapsed_ms * 1_000_000),
            clock_sync: None,
            cause: None,
            text: None,
            confidence: None,
            group_id: None,
            reason: None,
            face: None,
            unit_kind: None,
            expected_until_unix_ns: None,
            artifact: None,
            runtime_event: None,
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
                .any(|lane| lane.label == USER_TRANSCRIPT_LANE),
            "unknown voice lane should be present"
        );
        assert!(
            payload
                .streams
                .iter()
                .any(|lane| lane.label == PETE_INTENDED_SPEECH_LANE),
            "pete lane should be present"
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
                pronunciation: None,
            }],
        };
        stream_event.artifact = Some(serde_json::to_value(artifact_stream).unwrap());
        let mut transcript = event(1, "transcript", 300);
        transcript.text = Some("hello".to_string());

        let payload = live_trace_events_to_viewer_payload(&[transcript, stream_event]);
        let lane = payload
            .streams
            .iter()
            .find(|lane| lane.label == USER_TRANSCRIPT_LANE)
            .expect("unknown voice lane should be present");
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
        assert_eq!(
            lane.stream.words[0]
                .pronunciation
                .as_ref()
                .map(|pronunciation| pronunciation.phonemes.as_slice()),
            Some(
                &[
                    "HH".to_string(),
                    "AH0".to_string(),
                    "L".to_string(),
                    "OW1".to_string()
                ][..]
            )
        );
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
                    pronunciation: None,
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
                    pronunciation: None,
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
                pronunciation: None,
            }],
        };
        committed.artifact = Some(serde_json::to_value(committed_stream).unwrap());

        let payload = live_trace_events_to_viewer_payload(&[provisional, committed]);
        let lane = payload
            .streams
            .iter()
            .find(|lane| lane.label == PETE_INTENDED_SPEECH_LANE)
            .expect("pete lane should be present");
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
    fn builds_playback_to_word_alignments_in_span_graph() {
        let mut playback_started = event(1, "playback_started", 900);
        playback_started.text = Some("playback start".to_string());
        let playback_finished = event(1, "playback_finished", 1420);

        let mut tts_revision = event(1, "tts_timed_word_stream_revision", 920);
        let stream = TimedWordStream {
            id: WordStreamId(10),
            source: WordStreamSource::SyntheticSpeech,
            words: vec![
                WordNode {
                    id: WordId(1),
                    text: "sure".to_string(),
                    lexical_span: Some(TextSpan { start: 0, end: 4 }),
                    timing: Some(WordTiming {
                        start_ms: 980,
                        end_ms: 1120,
                    }),
                    timing_confidence: Some(0.88),
                    commitment: WordCommitment::Final,
                    boundary_source: BoundarySource::Predicted,
                    audio_ref: None,
                    pronunciation: None,
                },
                WordNode {
                    id: WordId(2),
                    text: "thing".to_string(),
                    lexical_span: Some(TextSpan { start: 5, end: 10 }),
                    timing: Some(WordTiming {
                        start_ms: 1130,
                        end_ms: 1350,
                    }),
                    timing_confidence: Some(0.9),
                    commitment: WordCommitment::Final,
                    boundary_source: BoundarySource::Predicted,
                    audio_ref: None,
                    pronunciation: None,
                },
            ],
        };
        tts_revision.artifact = Some(serde_json::to_value(stream).expect("serialize stream"));

        let payload = live_trace_events_to_viewer_payload(&[
            playback_started,
            tts_revision,
            playback_finished,
        ]);
        let span_graph = payload
            .span_graph
            .as_ref()
            .expect("span graph should be emitted");

        let playback_span = span_graph
            .spans
            .iter()
            .find(|span| span.contents == "playback")
            .expect("playback span should exist");
        let sure_span = span_graph
            .spans
            .iter()
            .find(|span| span.contents == "sure")
            .expect("sure word span should exist");
        let thing_span = span_graph
            .spans
            .iter()
            .find(|span| span.contents == "thing")
            .expect("thing word span should exist");

        assert!(span_graph.alignments.iter().any(|alignment| {
            alignment.source == playback_span.id
                && alignment.target == sure_span.id
                && alignment.relation == AlignmentKind::Contains
        }));
        assert!(span_graph.alignments.iter().any(|alignment| {
            alignment.source == playback_span.id
                && alignment.target == thing_span.id
                && alignment.relation == AlignmentKind::Contains
        }));
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

    #[test]
    fn suppression_span_end_uses_session_relative_elapsed_ms() {
        // `expected_until_unix_ns` is an absolute Unix timestamp in
        // nanoseconds.  The span end_ms must be session-relative, not an
        // absolute Unix millisecond value.
        let mut suppression_started = event(1, "self_hearing_suppression_started", 1_000);
        // Session start is at unix_ns=0, elapsed_ms=1000 → unix_ns=1_000_000_000.
        // expected_until = session-relative 3_500ms = unix_ns 3_500_000_000.
        suppression_started.expected_until_unix_ns =
            Some(suppression_started.elapsed_ms.saturating_mul(1_000_000) + 2_500_000_000);

        let payload = live_trace_events_to_viewer_payload(&[suppression_started]);

        let suppression_event = payload
            .events
            .iter()
            .find(|event| event.kind == "self_hearing_suppression")
            .expect("self_hearing_suppression span should be present");

        assert_eq!(
            suppression_event.start_ms, 1_000,
            "start_ms must be elapsed_ms"
        );
        assert_eq!(
            suppression_event.end_ms,
            Some(3_500),
            "end_ms must be session-relative elapsed ms, not absolute Unix ms"
        );
    }

    #[test]
    fn viewer_uses_normalized_elapsed_ms_and_keeps_observed_metadata() {
        let mut marker = event(1, "first_llm_token", 200);
        marker.normalized_elapsed_ms = Some(120);
        marker.normalized_unix_ns = Some(120_000_000);

        let payload = live_trace_events_to_viewer_payload(&[marker]);
        let marker = payload
            .markers
            .iter()
            .find(|marker| marker.kind == "first_llm_token")
            .expect("marker should be present");

        assert_eq!(marker.at_ms, 120);
        assert_eq!(
            marker
                .metadata
                .as_ref()
                .and_then(Value::as_object)
                .and_then(|metadata| metadata.get("observed_elapsed_ms"))
                .and_then(Value::as_u64),
            Some(200)
        );
    }

    #[test]
    fn causal_child_span_start_is_clamped_to_parent_end() {
        let parent_id = crate::speech_timeline::SpanId::new();
        let child_id = crate::speech_timeline::SpanId::new();

        let mut parent_started = event(1, "playback_started", 100);
        parent_started.span_id = Some(parent_id);
        let mut parent_ended = event(1, "playback_ended", 150);
        parent_ended.span_id = Some(parent_id);

        let mut child_started = event(1, "overlap_started", 120);
        child_started.span_id = Some(child_id);
        child_started.cause = Some(crate::live_trace::TraceCause {
            parent_span_id: Some(parent_id),
            msg_id: None,
            edge_kind: Some("recv".to_string()),
        });
        let mut child_ended = event(1, "overlap_ended", 180);
        child_ended.span_id = Some(child_id);

        let payload = live_trace_events_to_viewer_payload(&[
            child_started,
            parent_started,
            child_ended,
            parent_ended,
        ]);
        let child = payload
            .events
            .iter()
            .find(|event| event.kind == "overlap")
            .expect("child span should be present");

        assert_eq!(child.start_ms, 150);
        assert_eq!(child.end_ms, Some(180));
    }

    #[test]
    fn trace_session_payload_exposes_canonical_session_audio() {
        let session_id = crate::live_trace::SessionId::new();
        let session = TraceSessionEnvelope {
            metadata: crate::live_trace::TraceSessionMetadata {
                format: crate::live_trace::TRACE_SESSION_FORMAT.to_string(),
                session_id,
                session_started_at_unix_ns: 1_000,
                recorded_at_unix_ns: 2_000,
                events_path: crate::live_trace::TRACE_SESSION_EVENTS_FILE.to_string(),
                audio_artifacts: vec![TraceSessionAudioArtifact {
                    session_id,
                    artifact_id: "session-audio".to_string(),
                    path: "audio/session.wav".to_string(),
                    acoustic_analysis_path: Some("audio/session.acoustic.json".to_string()),
                    duration_ms: 2_500,
                    sample_rate_hz: 16_000,
                    channels: 1,
                    created_at_unix_ns: 2_000,
                }],
                runtime: crate::live_trace::TraceRuntimeMetadata::new("test"),
            },
            events: vec![],
        };

        let payload = trace_session_to_viewer_payload(&session);
        let audio = payload.audio.expect("session audio should be exposed");

        assert_eq!(audio.url, "/api/session-audio/session-audio");
        assert_eq!(
            audio.acoustic_analysis_url.as_deref(),
            Some("/api/session-audio/session-audio/acoustic.json")
        );
        assert_eq!(audio.duration_ms, Some(2_500));
        assert_eq!(audio.sample_rate_hz, Some(16_000));
        assert_eq!(audio.channels, Some(1));
    }
}
