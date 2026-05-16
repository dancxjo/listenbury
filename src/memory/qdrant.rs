use serde_json::{Map, Value, json};

use crate::memory::neo4j::Neo4jWriteResult;
use crate::memory::trace::MemoryTrace;

pub const DEFAULT_QDRANT_COLLECTION: &str = "listenbury_memory";

/// A vector document prepared for Qdrant insertion.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorDocument {
    pub point_id: String,
    pub text: String,
    pub payload: Map<String, Value>,
}

impl VectorDocument {
    pub fn into_point(self, vector: Vec<f32>) -> QdrantPoint {
        QdrantPoint {
            id: self.point_id,
            vector,
            payload: self.payload,
        }
    }
}

/// A Qdrant point upsert payload.
#[derive(Debug, Clone, PartialEq)]
pub struct QdrantPoint {
    pub id: String,
    pub vector: Vec<f32>,
    pub payload: Map<String, Value>,
}

/// A Qdrant search result payload.
#[derive(Debug, Clone, PartialEq)]
pub struct QdrantSearchHit {
    pub id: String,
    pub score: f32,
    pub payload: Map<String, Value>,
}

/// Writes and queries cold-memory vectors in Qdrant or a compatible mock.
pub trait QdrantStore: Send + Sync {
    fn upsert_points(&self, collection: &str, points: &[QdrantPoint]) -> anyhow::Result<()>;

    fn search(
        &self,
        collection: &str,
        query_vector: &[f32],
        limit: usize,
    ) -> anyhow::Result<Vec<QdrantSearchHit>>;
}

/// Convert a trace into vectorisable documents for Qdrant.
pub fn vector_documents_for_trace(
    trace: &MemoryTrace,
    sequence: u64,
    graph_result: &Neo4jWriteResult,
) -> Vec<VectorDocument> {
    match trace {
        MemoryTrace::ConversationTurnFinalized { text, .. } => vec![vector_document(
            format!("conversation_turn_vector:{sequence}"),
            text.clone(),
            [
                ("kind", json!("conversation_turn")),
                ("headline", json!(text)),
                ("text", json!(text)),
                (
                    "neo4j_node_id",
                    optional(graph_result.primary_node_id.clone()),
                ),
            ],
        )],
        MemoryTrace::TimedWordStreamFinalized {
            summary, stream_id, ..
        } => vec![vector_document(
            format!("timed_word_stream_summary_vector:{stream_id}:{sequence}"),
            summary.clone(),
            [
                ("kind", json!("summary")),
                ("headline", json!(summary)),
                ("text", json!(summary)),
                ("stream_id", json!(stream_id)),
                (
                    "neo4j_node_id",
                    optional(graph_result.primary_node_id.clone()),
                ),
            ],
        )],
        MemoryTrace::AuditorySceneObservation {
            description,
            salience,
            ..
        } => vec![vector_document(
            format!("auditory_observation_vector:{sequence}"),
            description.clone(),
            [
                ("kind", json!("auditory_observation")),
                ("headline", json!(description)),
                ("text", json!(description)),
                ("salience", json!(salience)),
                (
                    "neo4j_node_id",
                    optional(graph_result.primary_node_id.clone()),
                ),
            ],
        )],
        _ => Vec::new(),
    }
}

fn vector_document<const N: usize>(
    point_id: String,
    text: String,
    pairs: [(&str, Value); N],
) -> VectorDocument {
    VectorDocument {
        point_id,
        text,
        payload: payload_map(pairs),
    }
}

fn payload_map<const N: usize>(pairs: [(&str, Value); N]) -> Map<String, Value> {
    let mut payload = Map::with_capacity(N);
    for (key, value) in pairs {
        payload.insert(key.to_string(), value);
    }
    payload
}

fn optional(value: Option<String>) -> Value {
    match value {
        Some(value) => json!(value),
        None => Value::Null,
    }
}
