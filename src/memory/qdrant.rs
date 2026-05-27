use serde_json::{Map, Value, json};

use crate::memory::neo4j::{Neo4jTraceWrite, Neo4jWriteResult, trace_write_for};
use crate::memory::trace::MemoryTrace;

pub const DEFAULT_QDRANT_COLLECTION: &str = "listenbury_memory";
pub const PICTURE_QDRANT_COLLECTION: &str = "listenbury_picture_vectors";
pub const VOICE_QDRANT_COLLECTION: &str = "listenbury_voice_vectors";

/// A vector document prepared for Qdrant insertion.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorDocument {
    pub point_id: String,
    pub text: String,
    pub payload: Map<String, Value>,
    pub vector: Option<Vec<f32>>,
    pub collection: Option<String>,
}

impl VectorDocument {
    pub fn into_point(self, vector: Vec<f32>) -> QdrantPoint {
        QdrantPoint {
            id: self.point_id,
            vector,
            payload: self.payload,
        }
    }

    pub fn into_point_with_direct_vector(self) -> Option<QdrantPoint> {
        self.vector.map(|vector| QdrantPoint {
            id: self.point_id,
            vector,
            payload: self.payload,
        })
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
    let artifact_node_id = graph_result.primary_node_id.clone();
    let related_graph_node_ids = graph_result.related_node_ids.clone();
    let mut documents = match trace {
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
                ("graph_node_id", optional(artifact_node_id.clone())),
                ("vector_target", json!("artifact")),
                ("artifact_node_id", optional(artifact_node_id.clone())),
                ("referent_node_id", Value::Null),
                ("related_graph_node_ids", json!(related_graph_node_ids)),
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
                ("graph_node_id", optional(artifact_node_id.clone())),
                ("vector_target", json!("artifact")),
                ("artifact_node_id", optional(artifact_node_id.clone())),
                ("referent_node_id", Value::Null),
                ("related_graph_node_ids", json!(related_graph_node_ids)),
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
                ("graph_node_id", optional(artifact_node_id.clone())),
                ("vector_target", json!("artifact")),
                ("artifact_node_id", optional(artifact_node_id.clone())),
                ("referent_node_id", Value::Null),
                ("related_graph_node_ids", json!(related_graph_node_ids)),
            ],
        )],
        MemoryTrace::EntityExtractionPerformed {
            source_text,
            entities,
            ..
        } => entities
            .iter()
            .enumerate()
            .map(|(index, entity)| {
                let text = format!(
                    "{} ({}) mentioned in: {}",
                    entity.label, entity.kind, source_text
                );
                vector_document(
                    format!(
                        "entity_mention_vector:{sequence}:{index}:{}",
                        vector_safe_id(&entity.node_id)
                    ),
                    text.clone(),
                    [
                        ("kind", json!("entity_mention")),
                        ("headline", json!(entity.label.as_str())),
                        ("text", json!(text)),
                        ("source_text", json!(source_text.as_str())),
                        ("entity_kind", json!(entity.kind.as_str())),
                        ("confidence", json!(entity.confidence)),
                        ("span_start", json!(entity.span_start)),
                        ("span_end", json!(entity.span_end)),
                        ("neo4j_node_id", json!(entity.node_id.as_str())),
                        ("graph_node_id", json!(entity.node_id.as_str())),
                        ("vector_target", json!("referent")),
                        ("artifact_node_id", optional(artifact_node_id.clone())),
                        ("referent_node_id", json!(entity.node_id.as_str())),
                        (
                            "related_graph_node_ids",
                            json!(related_graph_node_ids_with_referent(
                                &related_graph_node_ids,
                                &entity.node_id
                            )),
                        ),
                    ],
                )
            })
            .collect(),
        MemoryTrace::ImageVectorCaptured { image, .. } => {
            let graph_node_id = image
                .content_node_id
                .clone()
                .unwrap_or_else(|| image.image_id.clone());
            vec![direct_vector_document(
                format!("image_vector:{}", vector_safe_id(&image.image_id)),
                format!(
                    "image vector {} from {} ({}x{})",
                    image.image_id, image.source, image.width, image.height
                ),
                image.vector.clone(),
                PICTURE_QDRANT_COLLECTION,
                [
                    ("kind", json!("image_vector")),
                    ("headline", json!(image.image_id.as_str())),
                    (
                        "text",
                        json!(format!(
                            "image vector {} from {}",
                            image.image_id, image.source
                        )),
                    ),
                    ("image_id", json!(image.image_id.as_str())),
                    ("source", json!(image.source.as_str())),
                    ("width", json!(image.width)),
                    ("height", json!(image.height)),
                    ("image_retained", json!(image.retained_image)),
                    ("neo4j_node_id", json!(graph_node_id.as_str())),
                    ("graph_node_id", json!(graph_node_id.as_str())),
                    ("vector_target", json!("referent_and_artifact")),
                    ("artifact_node_id", json!(image.image_id.as_str())),
                    (
                        "referent_node_id",
                        image
                            .content_node_id
                            .as_ref()
                            .map_or(Value::Null, |id| json!(id)),
                    ),
                    (
                        "related_graph_node_ids",
                        json!(image_related_ids(
                            &image.image_id,
                            image.content_node_id.as_deref()
                        )),
                    ),
                ],
            )]
        }
        MemoryTrace::VoiceVectorCaptured { voice, .. } => vec![direct_vector_document(
            format!("voice_vector:{}", vector_safe_id(&voice.voice_signature_id)),
            format!("voice vector {}", voice.voice_signature_id),
            voice.vector.clone(),
            VOICE_QDRANT_COLLECTION,
            [
                ("kind", json!("voice_vector")),
                ("headline", json!(voice.voice_signature_id.as_str())),
                (
                    "text",
                    json!(format!("voice vector {}", voice.voice_signature_id)),
                ),
                (
                    "voice_signature_id",
                    json!(voice.voice_signature_id.as_str()),
                ),
                ("voice_node_id", json!(voice.voice_node_id.as_str())),
                ("source", json!(voice.source.as_str())),
                ("span_id", optional_u64(voice.span_id)),
                ("confidence", json!(voice.confidence)),
                ("neo4j_node_id", json!(voice.voice_node_id.as_str())),
                ("graph_node_id", json!(voice.voice_node_id.as_str())),
                ("vector_target", json!("referent")),
                ("artifact_node_id", json!(voice.voice_signature_id.as_str())),
                ("referent_node_id", json!(voice.voice_node_id.as_str())),
                (
                    "related_graph_node_ids",
                    json!([
                        voice.voice_signature_id.as_str(),
                        voice.voice_node_id.as_str()
                    ]),
                ),
            ],
        )],
        _ => Vec::new(),
    };
    documents.extend(graph_node_description_documents_for_trace(
        trace,
        sequence,
        graph_result,
    ));
    documents
}

fn graph_node_description_documents_for_trace(
    trace: &MemoryTrace,
    sequence: u64,
    graph_result: &Neo4jWriteResult,
) -> Vec<VectorDocument> {
    let write = trace_write_for(trace, sequence);
    graph_node_description_documents_for_write(write, sequence, graph_result)
}

fn graph_node_description_documents_for_write(
    write: Neo4jTraceWrite,
    sequence: u64,
    graph_result: &Neo4jWriteResult,
) -> Vec<VectorDocument> {
    let mut seen = Vec::<String>::new();
    std::iter::once((write.primary_node, true))
        .chain(write.related_nodes.into_iter().map(|node| (node, false)))
        .filter_map(|(node, is_primary)| {
            let graph_node_id = if is_primary {
                graph_result
                    .primary_node_id
                    .clone()
                    .unwrap_or_else(|| node.logical_id.clone())
            } else {
                node.logical_id.clone()
            };
            if seen.iter().any(|seen_id| seen_id == &graph_node_id) {
                return None;
            }
            seen.push(graph_node_id.clone());
            let description = node
                .properties
                .get("description")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|description| !description.is_empty())?
                .to_string();
            Some(vector_document(
                format!(
                    "graph_node_description_vector:{sequence}:{}",
                    vector_safe_id(&graph_node_id)
                ),
                description.clone(),
                [
                    ("kind", json!("graph_node_description")),
                    ("headline", json!(description.as_str())),
                    ("text", json!(description.as_str())),
                    ("description", json!(description.as_str())),
                    ("graph_node_id", json!(graph_node_id.as_str())),
                    ("neo4j_node_id", json!(graph_node_id.as_str())),
                    ("vector_target", json!("graph_node_description")),
                    (
                        "artifact_node_id",
                        optional(graph_result.primary_node_id.clone()),
                    ),
                    ("referent_node_id", json!(graph_node_id.as_str())),
                    (
                        "related_graph_node_ids",
                        json!(related_graph_node_ids_with_referent(
                            &graph_result.related_node_ids,
                            &graph_node_id
                        )),
                    ),
                ],
            ))
        })
        .collect()
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
        vector: None,
        collection: None,
    }
}

fn direct_vector_document<const N: usize>(
    point_id: String,
    text: String,
    vector: Vec<f32>,
    collection: &str,
    pairs: [(&str, Value); N],
) -> VectorDocument {
    VectorDocument {
        point_id,
        text,
        payload: payload_map(pairs),
        vector: Some(vector),
        collection: Some(collection.to_string()),
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

fn optional_u64(value: Option<u64>) -> Value {
    value.map_or(Value::Null, |value| json!(value))
}

fn image_related_ids(image_id: &str, content_node_id: Option<&str>) -> Vec<String> {
    let mut ids = vec![image_id.to_string()];
    if let Some(content_node_id) = content_node_id
        && content_node_id != image_id
    {
        ids.push(content_node_id.to_string());
    }
    ids
}

fn vector_safe_id(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == ':' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn related_graph_node_ids_with_referent(existing: &[String], referent: &str) -> Vec<String> {
    let mut ids = existing.to_vec();
    if !ids.iter().any(|id| id == referent) {
        ids.push(referent.to_string());
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::trace::{MemoryEntityMention, MemoryImageVector, MemoryVoiceVector};
    use crate::time::ExactTimestamp;

    #[test]
    fn entity_vectors_point_at_referent_and_keep_artifact_link() {
        let trace = MemoryTrace::EntityExtractionPerformed {
            source_text: "My name is Travis".to_string(),
            entities: vec![MemoryEntityMention {
                node_id: "person:travis".to_string(),
                label: "Travis".to_string(),
                kind: "person".to_string(),
                confidence: 0.96,
                span_start: 11,
                span_end: 17,
            }],
            occurred_at: ExactTimestamp::now(),
        };
        let graph_result = Neo4jWriteResult {
            primary_node_id: Some("neo4j::entity_extraction:2".to_string()),
            related_node_ids: vec![
                "neo4j::memory_trace_event:2".to_string(),
                "person:travis".to_string(),
            ],
        };

        let documents = vector_documents_for_trace(&trace, 2, &graph_result);

        let payload = &documents
            .iter()
            .find(|document| {
                document.payload.get("kind").and_then(Value::as_str) == Some("entity_mention")
            })
            .expect("entity mention vector document")
            .payload;
        assert_eq!(
            payload.get("graph_node_id").and_then(Value::as_str),
            Some("person:travis")
        );
        assert_eq!(
            payload.get("referent_node_id").and_then(Value::as_str),
            Some("person:travis")
        );
        assert_eq!(
            payload.get("artifact_node_id").and_then(Value::as_str),
            Some("neo4j::entity_extraction:2")
        );
        assert_eq!(
            payload.get("vector_target").and_then(Value::as_str),
            Some("referent")
        );
        let description_payload = &documents
            .iter()
            .find(|document| {
                document.payload.get("kind").and_then(Value::as_str)
                    == Some("graph_node_description")
                    && document
                        .payload
                        .get("graph_node_id")
                        .and_then(Value::as_str)
                        == Some("person:travis")
            })
            .expect("entity graph node description vector")
            .payload;
        assert_eq!(
            description_payload
                .get("description")
                .and_then(Value::as_str),
            Some("person named Travis")
        );
        assert_eq!(
            description_payload
                .get("referent_node_id")
                .and_then(Value::as_str),
            Some("person:travis")
        );
    }

    #[test]
    fn artifact_vectors_include_source_link_fields() {
        let trace = MemoryTrace::ConversationTurnFinalized {
            speaker: crate::memory::trace::SpeakerRole::Named("Travis".to_string()),
            text: "hello".to_string(),
            occurred_at: ExactTimestamp::now(),
        };
        let graph_result = Neo4jWriteResult {
            primary_node_id: Some("neo4j::conversation_turn:0".to_string()),
            related_node_ids: vec!["neo4j::memory_trace_event:0".to_string()],
        };

        let documents = vector_documents_for_trace(&trace, 0, &graph_result);
        let payload = &documents[0].payload;

        assert_eq!(
            payload.get("graph_node_id").and_then(Value::as_str),
            Some("neo4j::conversation_turn:0")
        );
        assert_eq!(
            payload.get("artifact_node_id").and_then(Value::as_str),
            Some("neo4j::conversation_turn:0")
        );
        assert_eq!(payload.get("referent_node_id"), Some(&Value::Null));
        assert_eq!(
            payload.get("vector_target").and_then(Value::as_str),
            Some("artifact")
        );
    }

    #[test]
    fn image_vectors_use_picture_collection_and_direct_vector() {
        let trace = MemoryTrace::ImageVectorCaptured {
            image: MemoryImageVector {
                image_id: "image:abc".to_string(),
                source: "linux_v4l2:/dev/video0".to_string(),
                width: 320,
                height: 240,
                vector: vec![0.1, 0.2, 0.3],
                content_node_id: Some("object:mug".to_string()),
                retained_image: false,
            },
            captured_at: ExactTimestamp::now(),
        };

        let documents = vector_documents_for_trace(&trace, 4, &Neo4jWriteResult::default());

        let image_document = documents
            .iter()
            .find(|document| document.collection.as_deref() == Some(PICTURE_QDRANT_COLLECTION))
            .expect("image vector document");
        assert_eq!(
            image_document.collection.as_deref(),
            Some(PICTURE_QDRANT_COLLECTION)
        );
        assert_eq!(image_document.vector, Some(vec![0.1, 0.2, 0.3]));
        assert_eq!(
            image_document
                .payload
                .get("graph_node_id")
                .and_then(Value::as_str),
            Some("object:mug")
        );
        assert_eq!(
            image_document
                .payload
                .get("artifact_node_id")
                .and_then(Value::as_str),
            Some("image:abc")
        );
    }

    #[test]
    fn voice_vectors_use_voice_collection_and_referent_payload() {
        let trace = MemoryTrace::VoiceVectorCaptured {
            voice: MemoryVoiceVector {
                voice_signature_id: "sig-1".to_string(),
                voice_node_id: "voice:sig-1".to_string(),
                source: "native_mic".to_string(),
                span_id: Some(7),
                vector: vec![0.4, 0.5],
                confidence: 0.8,
            },
            captured_at: ExactTimestamp::now(),
        };

        let documents = vector_documents_for_trace(&trace, 5, &Neo4jWriteResult::default());

        let voice_document = documents
            .iter()
            .find(|document| document.collection.as_deref() == Some(VOICE_QDRANT_COLLECTION))
            .expect("voice vector document");
        assert_eq!(
            voice_document.collection.as_deref(),
            Some(VOICE_QDRANT_COLLECTION)
        );
        assert_eq!(voice_document.vector, Some(vec![0.4, 0.5]));
        assert_eq!(
            voice_document
                .payload
                .get("referent_node_id")
                .and_then(Value::as_str),
            Some("voice:sig-1")
        );
    }
}
