use serde_json::{Map, Value, json};

use crate::memory::trace::{MemoryTrace, SpeakerRole};

/// A logical graph node derived from a [`MemoryTrace`].
#[derive(Debug, Clone, PartialEq)]
pub struct Neo4jNode {
    pub logical_id: String,
    pub label: String,
    pub properties: Map<String, Value>,
}

/// A logical graph relationship derived from a [`MemoryTrace`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Neo4jRelationship {
    pub from_logical_id: String,
    pub to_logical_id: String,
    pub kind: String,
}

/// The graph write generated for a single memory trace.
#[derive(Debug, Clone, PartialEq)]
pub struct Neo4jTraceWrite {
    pub primary_node: Neo4jNode,
    pub related_nodes: Vec<Neo4jNode>,
    pub relationships: Vec<Neo4jRelationship>,
}

/// Result returned by a Neo4j-backed graph store after persisting a trace.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Neo4jWriteResult {
    pub primary_node_id: Option<String>,
    pub related_node_ids: Vec<String>,
}

/// Writes memory-derived graph documents into Neo4j or a compatible mock.
pub trait Neo4jStore: Send + Sync {
    fn store_trace(&self, write: Neo4jTraceWrite) -> anyhow::Result<Neo4jWriteResult>;
}

/// Convert a runtime [`MemoryTrace`] into a graph-oriented write payload.
pub fn trace_write_for(trace: &MemoryTrace, sequence: u64) -> Neo4jTraceWrite {
    let provenance = provenance_node(trace, sequence);
    let mut related_nodes = vec![provenance.clone()];
    let mut relationships = Vec::new();

    let primary_node = match trace {
        MemoryTrace::ConversationTurnFinalized {
            speaker,
            text,
            occurred_at,
        } => Neo4jNode {
            logical_id: format!("conversation_turn:{sequence}"),
            label: "ConversationTurn".to_string(),
            properties: props([
                ("speaker", json!(speaker_role_name(speaker))),
                ("text", json!(text)),
                ("occurred_at", json!(occurred_at)),
            ]),
        },
        MemoryTrace::TimedWordStreamFinalized {
            stream_id,
            summary,
            occurred_at,
        } => {
            let stream_node = Neo4jNode {
                logical_id: format!("timed_word_stream:{stream_id}:{sequence}"),
                label: "TimedWordStream".to_string(),
                properties: props([
                    ("stream_id", json!(stream_id)),
                    ("summary", json!(summary)),
                    ("occurred_at", json!(occurred_at)),
                ]),
            };
            let summary_node = Neo4jNode {
                logical_id: format!("timed_word_stream_summary:{stream_id}:{sequence}"),
                label: "MemorySummary".to_string(),
                properties: props([
                    ("headline", json!(summary)),
                    ("text", json!(summary)),
                    ("occurred_at", json!(occurred_at)),
                ]),
            };
            relationships.push(Neo4jRelationship {
                from_logical_id: summary_node.logical_id.clone(),
                to_logical_id: stream_node.logical_id.clone(),
                kind: "DERIVED_FROM".to_string(),
            });
            related_nodes.push(summary_node);
            stream_node
        }
        MemoryTrace::MouthPlaybackStarted {
            utterance_id,
            text,
            occurred_at,
        } => playback_node(
            "MouthPlaybackStarted",
            sequence,
            *utterance_id,
            text,
            *occurred_at,
        ),
        MemoryTrace::MouthPlaybackCompleted {
            utterance_id,
            text,
            occurred_at,
        } => playback_node(
            "MouthPlaybackCompleted",
            sequence,
            *utterance_id,
            text,
            *occurred_at,
        ),
        MemoryTrace::AuditorySceneObservation {
            description,
            salience,
            occurred_at,
        } => Neo4jNode {
            logical_id: format!("auditory_observation:{sequence}"),
            label: "AuditoryObservation".to_string(),
            properties: props([
                ("description", json!(description)),
                ("salience", json!(salience)),
                ("occurred_at", json!(occurred_at)),
            ]),
        },
        MemoryTrace::OverlapDetected {
            description,
            occurred_at,
        } => Neo4jNode {
            logical_id: format!("overlap_detected:{sequence}"),
            label: "OverlapEvent".to_string(),
            properties: props([
                ("description", json!(description)),
                ("occurred_at", json!(occurred_at)),
            ]),
        },
        MemoryTrace::RecallResultUsed {
            query,
            result_summary,
            occurred_at,
        } => Neo4jNode {
            logical_id: format!("recall_result:{sequence}"),
            label: "RecallResult".to_string(),
            properties: props([
                ("query", json!(query)),
                ("result_summary", json!(result_summary)),
                ("occurred_at", json!(occurred_at)),
            ]),
        },
    };

    relationships.push(Neo4jRelationship {
        from_logical_id: primary_node.logical_id.clone(),
        to_logical_id: provenance.logical_id,
        kind: "OBSERVED_FROM".to_string(),
    });

    Neo4jTraceWrite {
        primary_node,
        related_nodes,
        relationships,
    }
}

fn provenance_node(trace: &MemoryTrace, sequence: u64) -> Neo4jNode {
    Neo4jNode {
        logical_id: format!("memory_trace_event:{sequence}"),
        label: "MemoryTraceEvent".to_string(),
        properties: props([
            ("kind", json!(trace.kind_name())),
            ("sequence", json!(sequence)),
            ("occurred_at", json!(trace.occurred_at())),
        ]),
    }
}

fn playback_node(
    label: &str,
    sequence: u64,
    utterance_id: u64,
    text: &str,
    occurred_at: crate::time::ExactTimestamp,
) -> Neo4jNode {
    Neo4jNode {
        logical_id: format!("mouth_playback:{utterance_id}:{sequence}:{label}"),
        label: label.to_string(),
        properties: props([
            ("utterance_id", json!(utterance_id)),
            ("text", json!(text)),
            ("occurred_at", json!(occurred_at)),
        ]),
    }
}

fn speaker_role_name(role: &SpeakerRole) -> String {
    match role {
        SpeakerRole::Pete => "pete".to_string(),
        SpeakerRole::Named(name) => name.trim().to_lowercase(),
        SpeakerRole::UnknownVoice { ordinal } => format!("unknown_voice_{ordinal}"),
        SpeakerRole::BackgroundVoice => "background_voice".to_string(),
        SpeakerRole::Environment => "environment".to_string(),
    }
}

fn props<const N: usize>(pairs: [(&str, Value); N]) -> Map<String, Value> {
    let mut properties = Map::with_capacity(N);
    for (key, value) in pairs {
        properties.insert(key.to_string(), value);
    }
    properties
}
