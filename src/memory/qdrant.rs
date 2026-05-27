use std::time::Duration;

use anyhow::{Context, bail};
use reqwest::{StatusCode, Url};
use serde_json::{Map, Value, json};
use uuid::Uuid;

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

const QDRANT_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// Qdrant REST client using the same JSON API shape Daringsby uses.
#[derive(Debug, Clone)]
pub struct QdrantHttpStore {
    pub url: String,
}

impl QdrantHttpStore {
    pub fn new(url: impl Into<String>) -> Self {
        Self { url: url.into() }
    }

    pub fn from_env() -> Self {
        Self::new(std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".into()))
    }

    fn endpoint(&self, path: &str) -> anyhow::Result<Url> {
        let base = self.url.trim_end_matches('/');
        Url::parse(&format!("{base}/{}", path.trim_start_matches('/')))
            .with_context(|| format!("invalid Qdrant URL {}", self.url))
    }

    fn ensure_collection(&self, collection: &str, vector_size: usize) -> anyhow::Result<()> {
        let client = reqwest::blocking::Client::new();
        let url = self.endpoint(&format!("collections/{collection}"))?;
        let response = client
            .get(url.clone())
            .timeout(QDRANT_REQUEST_TIMEOUT)
            .send()
            .with_context(|| format!("failed to inspect Qdrant collection {collection}"))?;

        if response.status().is_success() {
            let body: Value = response
                .json()
                .with_context(|| format!("failed to decode Qdrant collection {collection}"))?;
            let existing_size = qdrant_collection_vector_size(&body).with_context(|| {
                format!("Qdrant collection {collection} did not report a vector size")
            })?;
            if existing_size != vector_size {
                tracing::warn!(
                    target: "qdrant",
                    collection,
                    existing_size,
                    vector_size,
                    "recreating Qdrant collection with incompatible vector dimension"
                );
                self.recreate_collection(collection, vector_size)?;
            }
            return Ok(());
        }

        if response.status() != StatusCode::NOT_FOUND {
            return Err(unexpected_qdrant_response(
                response,
                &format!("inspecting collection {collection}"),
            ));
        }

        self.create_collection(&client, url, collection, vector_size)
    }

    fn recreate_collection(&self, collection: &str, vector_size: usize) -> anyhow::Result<()> {
        let client = reqwest::blocking::Client::new();
        let url = self.endpoint(&format!("collections/{collection}"))?;
        let response = client
            .delete(url.clone())
            .timeout(QDRANT_REQUEST_TIMEOUT)
            .send()
            .with_context(|| format!("failed to delete Qdrant collection {collection}"))?;

        if !response.status().is_success() && response.status() != StatusCode::NOT_FOUND {
            return Err(unexpected_qdrant_response(
                response,
                &format!("deleting collection {collection}"),
            ));
        }

        self.create_collection(&client, url, collection, vector_size)
    }

    fn create_collection(
        &self,
        client: &reqwest::blocking::Client,
        url: Url,
        collection: &str,
        vector_size: usize,
    ) -> anyhow::Result<()> {
        let response = client
            .put(url)
            .json(&json!({
                "vectors": {
                    "size": vector_size,
                    "distance": "Cosine",
                }
            }))
            .timeout(QDRANT_REQUEST_TIMEOUT)
            .send()
            .with_context(|| format!("failed to create Qdrant collection {collection}"))?;

        if response.status().is_success() || response.status() == StatusCode::CONFLICT {
            Ok(())
        } else {
            Err(unexpected_qdrant_response(
                response,
                &format!("creating collection {collection}"),
            ))
        }
    }
}

impl QdrantStore for QdrantHttpStore {
    fn upsert_points(&self, collection: &str, points: &[QdrantPoint]) -> anyhow::Result<()> {
        if points.is_empty() {
            return Ok(());
        }
        let vector_size = points
            .first()
            .map(|point| point.vector.len())
            .filter(|len| *len > 0)
            .with_context(|| format!("refusing to upsert empty vector into {collection}"))?;
        if points.iter().any(|point| point.vector.len() != vector_size) {
            bail!("Qdrant upsert batch for {collection} contains mixed vector sizes");
        }
        self.ensure_collection(collection, vector_size)?;

        let qdrant_points = points
            .iter()
            .map(|point| {
                let mut payload = point.payload.clone();
                payload
                    .entry("listenbury_point_id".to_string())
                    .or_insert_with(|| json!(point.id.as_str()));
                json!({
                    "id": Uuid::new_v4().to_string(),
                    "vector": point.vector,
                    "payload": payload,
                })
            })
            .collect::<Vec<_>>();

        let response = reqwest::blocking::Client::new()
            .put(self.endpoint(&format!("collections/{collection}/points?wait=true"))?)
            .json(&json!({ "points": qdrant_points }))
            .timeout(QDRANT_REQUEST_TIMEOUT)
            .send()
            .with_context(|| {
                format!("failed to upsert points into Qdrant collection {collection}")
            })?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(unexpected_qdrant_response(
                response,
                &format!("upserting points into collection {collection}"),
            ))
        }
    }

    fn search(
        &self,
        collection: &str,
        query_vector: &[f32],
        limit: usize,
    ) -> anyhow::Result<Vec<QdrantSearchHit>> {
        if query_vector.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }

        let response = reqwest::blocking::Client::new()
            .post(self.endpoint(&format!("collections/{collection}/points/search"))?)
            .json(&json!({
                "vector": query_vector,
                "limit": limit.max(1),
                "with_payload": true,
            }))
            .timeout(QDRANT_REQUEST_TIMEOUT)
            .send()
            .with_context(|| format!("failed to search Qdrant collection {collection}"))?;

        if response.status() == StatusCode::NOT_FOUND {
            return Ok(Vec::new());
        }
        if !response.status().is_success() {
            return Err(unexpected_qdrant_response(
                response,
                &format!("searching collection {collection}"),
            ));
        }

        let body: Value = response
            .json()
            .with_context(|| format!("failed to decode Qdrant search response for {collection}"))?;
        qdrant_search_hits(&body)
            .with_context(|| format!("Qdrant search response for {collection} was invalid"))
    }
}

fn unexpected_qdrant_response(
    response: reqwest::blocking::Response,
    action: &str,
) -> anyhow::Error {
    let status = response.status();
    let body = response.text().unwrap_or_default();
    anyhow::anyhow!("Qdrant returned {status} while {action}: {body}")
}

fn qdrant_collection_vector_size(collection: &Value) -> Option<usize> {
    let vectors = collection.pointer("/result/config/params/vectors")?;
    if let Some(size) = vectors.get("size").and_then(Value::as_u64) {
        return usize::try_from(size).ok();
    }
    vectors
        .as_object()?
        .values()
        .find_map(|vector| vector.get("size").and_then(Value::as_u64))
        .and_then(|size| usize::try_from(size).ok())
}

fn qdrant_search_hits(response: &Value) -> anyhow::Result<Vec<QdrantSearchHit>> {
    response
        .pointer("/result")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(qdrant_search_hit)
        .collect()
}

fn qdrant_search_hit(value: Value) -> anyhow::Result<QdrantSearchHit> {
    let object = value
        .as_object()
        .context("Qdrant search result was not an object")?;
    let payload = object
        .get("payload")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let id = payload
        .get("listenbury_point_id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| qdrant_point_id(object.get("id")).unwrap_or_default());
    let score = object
        .get("score")
        .and_then(Value::as_f64)
        .context("Qdrant search result is missing numeric score")? as f32;
    Ok(QdrantSearchHit { id, score, payload })
}

fn qdrant_point_id(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(id) => Some(id.clone()),
        Value::Number(number) => Some(number.to_string()),
        _ => None,
    }
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
        MemoryTrace::AssistantAnalysisCaptured { text, scene, .. } => vec![vector_document(
            format!("assistant_analysis_vector:{sequence}"),
            text.clone(),
            [
                ("kind", json!("assistant_analysis")),
                ("headline", json!(scene.summary.as_str())),
                ("text", json!(text)),
                ("scene_node_id", json!(scene.node_id.as_str())),
                ("scene_description", json!(scene.description.as_str())),
                ("scene_summary", json!(scene.summary.as_str())),
                (
                    "neo4j_node_id",
                    optional(graph_result.primary_node_id.clone()),
                ),
                ("graph_node_id", optional(artifact_node_id.clone())),
                ("vector_target", json!("artifact")),
                ("artifact_node_id", optional(artifact_node_id.clone())),
                ("referent_node_id", json!(scene.node_id.as_str())),
                (
                    "related_graph_node_ids",
                    json!(related_graph_node_ids_with_referent(
                        &related_graph_node_ids,
                        &scene.node_id
                    )),
                ),
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
    use crate::memory::trace::{
        MemoryEntityMention, MemoryImageVector, MemorySceneRef, MemoryVoiceVector,
    };
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
    fn assistant_analysis_vectors_point_at_current_scene() {
        let trace = MemoryTrace::AssistantAnalysisCaptured {
            text: "Pete should use the source listing before speaking.".to_string(),
            scene: MemorySceneRef {
                node_id: "scene:source-review".to_string(),
                description: "Setting: live coding session. Action: Pete reviews source files."
                    .to_string(),
                summary: "Pete reviews source files.".to_string(),
            },
            occurred_at: ExactTimestamp::now(),
        };
        let graph_result = Neo4jWriteResult {
            primary_node_id: Some("neo4j::assistant_analysis:3".to_string()),
            related_node_ids: vec![
                "neo4j::memory_trace_event:3".to_string(),
                "scene:source-review".to_string(),
            ],
        };

        let documents = vector_documents_for_trace(&trace, 3, &graph_result);
        let payload = &documents
            .iter()
            .find(|document| {
                document.payload.get("kind").and_then(Value::as_str)
                    == Some("assistant_analysis")
            })
            .expect("assistant analysis vector document")
            .payload;

        assert_eq!(
            payload.get("referent_node_id").and_then(Value::as_str),
            Some("scene:source-review")
        );
        assert_eq!(
            payload.get("scene_node_id").and_then(Value::as_str),
            Some("scene:source-review")
        );
        assert_eq!(
            payload
                .get("related_graph_node_ids")
                .and_then(Value::as_array)
                .expect("related ids")
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>(),
            vec!["neo4j::memory_trace_event:3", "scene:source-review"]
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
