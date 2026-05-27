use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use anyhow::{Context, bail};
use reqwest::Url;
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

const NEO4J_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// Neo4j HTTP transaction client using Daringsby's `bolt://` to HTTP endpoint
/// conversion and JSON Cypher commit path.
#[derive(Debug, Clone)]
pub struct Neo4jHttpStore {
    pub uri: String,
    pub user: String,
    pub pass: String,
    constraint_ensured: Arc<AtomicBool>,
}

impl Neo4jHttpStore {
    pub fn new(uri: impl Into<String>, user: impl Into<String>, pass: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            user: user.into(),
            pass: pass.into(),
            constraint_ensured: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn from_env() -> Self {
        Self::new(
            std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".into()),
            std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into()),
            std::env::var("NEO4J_PASS")
                .or_else(|_| std::env::var("NEO4J_PASSWORD"))
                .unwrap_or_else(|_| "password".into()),
        )
    }

    fn http_endpoint(&self) -> anyhow::Result<Url> {
        let parsed =
            Url::parse(&self.uri).with_context(|| format!("invalid Neo4j URI {}", self.uri))?;
        let mut url = match parsed.scheme() {
            "http" | "https" => parsed,
            "bolt" | "neo4j" => neo4j_http_url(&parsed, "http", 7474)?,
            "bolt+s" | "neo4j+s" => neo4j_http_url(&parsed, "https", 7473)?,
            scheme => bail!("unsupported Neo4j URI scheme {scheme}"),
        };
        url.set_path("/db/neo4j/tx/commit");
        url.set_query(None);
        url.set_fragment(None);
        Ok(url)
    }

    fn ensure_constraint(
        &self,
        client: &reqwest::blocking::Client,
        endpoint: &Url,
    ) -> anyhow::Result<()> {
        if self.constraint_ensured.load(Ordering::SeqCst) {
            return Ok(());
        }
        commit_neo4j_statements(
            client,
            endpoint,
            &self.user,
            &self.pass,
            &[CypherStatement {
                statement:
                    "CREATE CONSTRAINT listenbury_graph_node_id IF NOT EXISTS FOR (n:GraphNode) REQUIRE n.id IS UNIQUE"
                        .into(),
                parameters: json!({}),
            }],
            "ensuring graph node constraint",
        )?;
        self.constraint_ensured.store(true, Ordering::SeqCst);
        Ok(())
    }
}

impl Neo4jStore for Neo4jHttpStore {
    fn store_trace(&self, write: Neo4jTraceWrite) -> anyhow::Result<Neo4jWriteResult> {
        let endpoint = self.http_endpoint()?;
        let client = reqwest::blocking::Client::new();
        self.ensure_constraint(&client, &endpoint)?;
        let statements = trace_write_statements(&write)?;
        commit_neo4j_statements(
            &client,
            &endpoint,
            &self.user,
            &self.pass,
            &statements,
            "committing Listenbury memory trace",
        )?;
        Ok(Neo4jWriteResult {
            primary_node_id: Some(format!("neo4j::{}", write.primary_node.logical_id)),
            related_node_ids: write
                .related_nodes
                .iter()
                .map(|node| format!("neo4j::{}", node.logical_id))
                .collect(),
        })
    }
}

#[derive(Debug, Clone)]
struct CypherStatement {
    statement: String,
    parameters: Value,
}

fn trace_write_statements(write: &Neo4jTraceWrite) -> anyhow::Result<Vec<CypherStatement>> {
    let mut statements = Vec::with_capacity(
        1usize
            .saturating_add(write.related_nodes.len())
            .saturating_add(write.relationships.len()),
    );
    statements.push(node_statement(&write.primary_node)?);
    for node in &write.related_nodes {
        statements.push(node_statement(node)?);
    }
    for relationship in &write.relationships {
        statements.push(relationship_statement(relationship)?);
    }
    Ok(statements)
}

fn node_statement(node: &Neo4jNode) -> anyhow::Result<CypherStatement> {
    validate_graph_name(&node.label, "label")?;
    let mut props = graph_property_map(&node.properties);
    props.insert("id".to_string(), json!(node.logical_id.as_str()));
    props.insert("logical_id".to_string(), json!(node.logical_id.as_str()));
    Ok(CypherStatement {
        statement: format!(
            "MERGE (n:GraphNode {{id: $id}}) SET n += $props SET n:`{}`",
            node.label
        ),
        parameters: json!({
            "id": node.logical_id,
            "props": props,
        }),
    })
}

fn relationship_statement(relationship: &Neo4jRelationship) -> anyhow::Result<CypherStatement> {
    validate_graph_name(&relationship.kind, "relationship type")?;
    Ok(CypherStatement {
        statement: format!(
            "MATCH (from:GraphNode {{id: $from}}), (to:GraphNode {{id: $to}}) MERGE (from)-[r:`{}`]->(to)",
            relationship.kind
        ),
        parameters: json!({
            "from": relationship.from_logical_id,
            "to": relationship.to_logical_id,
        }),
    })
}

fn graph_property_map(properties: &Map<String, Value>) -> Map<String, Value> {
    properties
        .iter()
        .filter_map(|(key, value)| graph_property(value).map(|value| (key.clone(), value)))
        .collect()
}

fn graph_property(value: &Value) -> Option<Value> {
    match value {
        Value::Null | Value::Object(_) => None,
        Value::Array(items) => Some(Value::Array(
            items.iter().filter_map(graph_property).collect::<Vec<_>>(),
        )),
        Value::String(_) | Value::Bool(_) | Value::Number(_) => Some(value.clone()),
    }
}

fn validate_graph_name(name: &str, kind: &str) -> anyhow::Result<()> {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        bail!("empty Neo4j {kind}");
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        bail!("invalid Neo4j {kind}: {name}");
    }
    if chars.any(|c| !(c == '_' || c.is_ascii_alphanumeric())) {
        bail!("invalid Neo4j {kind}: {name}");
    }
    Ok(())
}

fn commit_neo4j_statements(
    client: &reqwest::blocking::Client,
    endpoint: &Url,
    user: &str,
    pass: &str,
    statements: &[CypherStatement],
    action: &str,
) -> anyhow::Result<()> {
    let response = client
        .post(endpoint.clone())
        .basic_auth(user, Some(pass))
        .json(&json!({
            "statements": statements.iter().map(|statement| {
                json!({
                    "statement": statement.statement,
                    "parameters": statement.parameters,
                })
            }).collect::<Vec<_>>()
        }))
        .timeout(NEO4J_REQUEST_TIMEOUT)
        .send()
        .with_context(|| format!("failed while {action} at {endpoint}"))?;
    if !response.status().is_success() {
        return Err(unexpected_neo4j_response(response, action));
    }
    let body: Value = response
        .json()
        .with_context(|| format!("failed to decode Neo4j response while {action}"))?;
    if let Some(errors) = body.get("errors").and_then(Value::as_array) {
        if !errors.is_empty() {
            bail!("Neo4j returned errors while {action}: {errors:?}");
        }
    }
    Ok(())
}

fn neo4j_http_url(source: &Url, scheme: &str, default_port: u16) -> anyhow::Result<Url> {
    let host = source
        .host_str()
        .with_context(|| format!("Neo4j URI {} is missing a host", source.as_str()))?;
    let port = match source.port() {
        Some(7687) | None => default_port,
        Some(port) => port,
    };
    Url::parse(&format!("{scheme}://{host}:{port}"))
        .with_context(|| format!("failed to convert {} to {scheme}", source.as_str()))
}

fn unexpected_neo4j_response(response: reqwest::blocking::Response, action: &str) -> anyhow::Error {
    let status = response.status();
    let body = response.text().unwrap_or_default();
    anyhow::anyhow!("Neo4j returned {status} while {action}: {body}")
}

/// Convert a runtime [`MemoryTrace`] into a graph-oriented write payload.
pub fn trace_write_for(trace: &MemoryTrace, sequence: u64) -> Neo4jTraceWrite {
    let provenance = provenance_node(trace, sequence);
    let mut related_nodes = vec![provenance.clone()];
    let mut relationships = Vec::new();

    let mut primary_node = match trace {
        MemoryTrace::ConversationTurnFinalized {
            speaker,
            text,
            occurred_at,
        } => Neo4jNode {
            logical_id: format!("conversation_turn:{sequence}"),
            label: "ConversationTurn".to_string(),
            properties: props([
                ("speaker", json!(voice_label_name(speaker))),
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
        MemoryTrace::EntityExtractionPerformed {
            source_text,
            entities,
            occurred_at,
        } => {
            let extraction_node = Neo4jNode {
                logical_id: format!("entity_extraction:{sequence}"),
                label: "EntityExtraction".to_string(),
                properties: props([
                    ("source_text", json!(source_text)),
                    ("entity_count", json!(entities.len())),
                    ("occurred_at", json!(occurred_at)),
                ]),
            };
            for entity in entities {
                let entity_node = Neo4jNode {
                    logical_id: entity.node_id.clone(),
                    label: entity_node_label(&entity.kind).to_string(),
                    properties: props([
                        ("label", json!(entity.label.as_str())),
                        ("entity_kind", json!(entity.kind.as_str())),
                        ("confidence", json!(entity.confidence)),
                        ("span_start", json!(entity.span_start)),
                        ("span_end", json!(entity.span_end)),
                        ("last_observed_at", json!(occurred_at)),
                    ]),
                };
                relationships.push(Neo4jRelationship {
                    from_logical_id: extraction_node.logical_id.clone(),
                    to_logical_id: entity_node.logical_id.clone(),
                    kind: "EXTRACTED_ENTITY".to_string(),
                });
                related_nodes.push(entity_node);
            }
            extraction_node
        }
        MemoryTrace::GraphNodeFieldsUpdated {
            update,
            occurred_at,
        } => {
            let update_node = Neo4jNode {
                logical_id: format!(
                    "graph_node_field_update:{}:{sequence}",
                    vector_safe_id(&update.node_id)
                ),
                label: "GraphNodeFieldUpdate".to_string(),
                properties: props([
                    ("node_id", json!(update.node_id.as_str())),
                    ("label", optional_string(update.label.as_deref())),
                    ("fields", json!(update.fields)),
                    (
                        "source_text",
                        optional_string(update.source_text.as_deref()),
                    ),
                    ("confidence", json!(update.confidence)),
                    ("occurred_at", json!(occurred_at)),
                ]),
            };
            let mut target_properties = update.fields.clone();
            if let Some(label) = update.label.as_deref() {
                target_properties.insert("label".to_string(), json!(label));
            }
            target_properties.insert("last_updated_at".to_string(), json!(occurred_at));
            target_properties.insert(
                "last_update_confidence".to_string(),
                json!(update.confidence),
            );
            let target_node = Neo4jNode {
                logical_id: update.node_id.clone(),
                label: "GraphNode".to_string(),
                properties: target_properties,
            };
            relationships.push(Neo4jRelationship {
                from_logical_id: update_node.logical_id.clone(),
                to_logical_id: target_node.logical_id.clone(),
                kind: "UPDATES_NODE".to_string(),
            });
            related_nodes.push(target_node);
            update_node
        }
        MemoryTrace::ImageVectorCaptured { image, captured_at } => {
            let observation_node = Neo4jNode {
                logical_id: format!("image_observation:{sequence}"),
                label: "ImageObservation".to_string(),
                properties: props([
                    ("image_id", json!(image.image_id.as_str())),
                    ("source", json!(image.source.as_str())),
                    ("width", json!(image.width)),
                    ("height", json!(image.height)),
                    ("retained_image", json!(image.retained_image)),
                    ("captured_at", json!(captured_at)),
                ]),
            };
            let image_node = Neo4jNode {
                logical_id: image.image_id.clone(),
                label: "ImageArtifact".to_string(),
                properties: props([
                    ("image_id", json!(image.image_id.as_str())),
                    ("source", json!(image.source.as_str())),
                    ("width", json!(image.width)),
                    ("height", json!(image.height)),
                    ("retained_image", json!(image.retained_image)),
                    ("captured_at", json!(captured_at)),
                ]),
            };
            relationships.push(Neo4jRelationship {
                from_logical_id: observation_node.logical_id.clone(),
                to_logical_id: image_node.logical_id.clone(),
                kind: "OBSERVED".to_string(),
            });
            related_nodes.push(image_node);
            if let Some(content_node_id) = &image.content_node_id {
                let content_node = Neo4jNode {
                    logical_id: content_node_id.clone(),
                    label: "VisualReferent".to_string(),
                    properties: props([
                        ("referent_node_id", json!(content_node_id.as_str())),
                        ("last_observed_at", json!(captured_at)),
                    ]),
                };
                relationships.push(Neo4jRelationship {
                    from_logical_id: image.image_id.clone(),
                    to_logical_id: content_node.logical_id.clone(),
                    kind: "DEPICTS".to_string(),
                });
                related_nodes.push(content_node);
            }
            observation_node
        }
        MemoryTrace::VoiceVectorCaptured { voice, captured_at } => {
            let signature_node = Neo4jNode {
                logical_id: voice.voice_signature_id.clone(),
                label: "VoiceSignature".to_string(),
                properties: props([
                    (
                        "voice_signature_id",
                        json!(voice.voice_signature_id.as_str()),
                    ),
                    ("voice_node_id", json!(voice.voice_node_id.as_str())),
                    ("source", json!(voice.source.as_str())),
                    ("span_id", optional_u64(voice.span_id)),
                    ("confidence", json!(voice.confidence)),
                    ("captured_at", json!(captured_at)),
                ]),
            };
            let voice_node = Neo4jNode {
                logical_id: voice.voice_node_id.clone(),
                label: "Voice".to_string(),
                properties: props([
                    ("voice_node_id", json!(voice.voice_node_id.as_str())),
                    (
                        "last_signature_id",
                        json!(voice.voice_signature_id.as_str()),
                    ),
                    ("last_observed_at", json!(captured_at)),
                ]),
            };
            relationships.push(Neo4jRelationship {
                from_logical_id: signature_node.logical_id.clone(),
                to_logical_id: voice_node.logical_id.clone(),
                kind: "SIGNATURE_OF".to_string(),
            });
            related_nodes.push(voice_node);
            signature_node
        }
    };

    ensure_node_description(&mut primary_node);
    for node in &mut related_nodes {
        ensure_node_description(node);
    }

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

fn ensure_node_description(node: &mut Neo4jNode) {
    if node
        .properties
        .get("description")
        .and_then(Value::as_str)
        .is_some_and(|description| !description.trim().is_empty())
    {
        return;
    }
    node.properties
        .insert("description".to_string(), json!(node_description(node)));
}

fn node_description(node: &Neo4jNode) -> String {
    let label = node.properties.get("label").and_then(Value::as_str);
    match node.label.as_str() {
        "Person" => named_description("person", label, &node.logical_id),
        "Place" => named_description("place", label, &node.logical_id),
        "Topic" => named_description("topic", label, &node.logical_id),
        "Organization" => named_description("organization", label, &node.logical_id),
        "Object" => named_description("object", label, &node.logical_id),
        "Task" => named_description("task", label, &node.logical_id),
        "Entity" | "GraphNode" => named_description("graph node", label, &node.logical_id),
        "ConversationTurn" => node
            .properties
            .get("speaker")
            .and_then(Value::as_str)
            .map(|speaker| format!("conversation turn from {speaker}"))
            .unwrap_or_else(|| "conversation turn".to_string()),
        "TimedWordStream" => "timed word stream".to_string(),
        "MemorySummary" => "memory summary".to_string(),
        "MouthPlaybackStarted" => "mouth playback start event".to_string(),
        "MouthPlaybackCompleted" => "mouth playback completion event".to_string(),
        "AuditoryObservation" => "auditory scene observation".to_string(),
        "OverlapEvent" => "overlapping speech event".to_string(),
        "RecallResult" => "memory recall result".to_string(),
        "EntityExtraction" => "entity extraction event".to_string(),
        "GraphNodeFieldUpdate" => "graph node field update event".to_string(),
        "ImageObservation" => "image observation".to_string(),
        "ImageArtifact" => "image artifact".to_string(),
        "VisualReferent" => "visual referent".to_string(),
        "VoiceSignature" => "voice signature".to_string(),
        "Voice" => "heard voice".to_string(),
        "MemoryTraceEvent" => "memory trace event".to_string(),
        other => named_description(&other.to_ascii_lowercase(), label, &node.logical_id),
    }
}

fn named_description(kind: &str, label: Option<&str>, fallback: &str) -> String {
    label
        .map(str::trim)
        .filter(|label| !label.is_empty())
        .map(|label| format!("{kind} named {label}"))
        .unwrap_or_else(|| format!("{kind} {fallback}"))
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

fn voice_label_name(role: &SpeakerRole) -> String {
    match role {
        SpeakerRole::Pete => "pete".to_string(),
        SpeakerRole::Named(name) => name.trim().to_lowercase(),
        SpeakerRole::UnknownVoice { ordinal } => format!("unknown_voice_{ordinal}"),
        SpeakerRole::BackgroundVoice => "background_voice".to_string(),
        SpeakerRole::Environment => "environment".to_string(),
    }
}

fn entity_node_label(kind: &str) -> &'static str {
    match kind {
        "person" => "Person",
        "place" => "Place",
        "topic" => "Topic",
        "org" | "organization" => "Organization",
        "object" => "Object",
        "task" => "Task",
        _ => "Entity",
    }
}

fn props<const N: usize>(pairs: [(&str, Value); N]) -> Map<String, Value> {
    let mut properties = Map::with_capacity(N);
    for (key, value) in pairs {
        properties.insert(key.to_string(), value);
    }
    properties
}

fn optional_u64(value: Option<u64>) -> Value {
    value.map_or(Value::Null, |value| json!(value))
}

fn optional_string(value: Option<&str>) -> Value {
    value.map_or(Value::Null, |value| json!(value))
}

fn vector_safe_id(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | ':') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::trace::MemoryEntityMention;
    use crate::time::ExactTimestamp;

    #[test]
    fn entity_extraction_write_uses_central_referent_nodes() {
        let trace = MemoryTrace::EntityExtractionPerformed {
            source_text: "My name is Travis".to_string(),
            entities: vec![MemoryEntityMention {
                node_id: "person:travis".to_string(),
                label: "Travis".to_string(),
                kind: "person".to_string(),
                confidence: 0.92,
                span_start: 11,
                span_end: 17,
            }],
            occurred_at: ExactTimestamp::now(),
        };

        let write = trace_write_for(&trace, 7);

        assert_eq!(write.primary_node.logical_id, "entity_extraction:7");
        assert_eq!(write.primary_node.label, "EntityExtraction");
        let travis = write
            .related_nodes
            .iter()
            .find(|node| node.logical_id == "person:travis")
            .expect("central person node should be related");
        assert_eq!(travis.label, "Person");
        assert_eq!(
            travis.properties.get("label").and_then(Value::as_str),
            Some("Travis")
        );
        assert!(write.relationships.iter().any(|relationship| {
            relationship.from_logical_id == "entity_extraction:7"
                && relationship.to_logical_id == "person:travis"
                && relationship.kind == "EXTRACTED_ENTITY"
        }));
        for node in std::iter::once(&write.primary_node).chain(write.related_nodes.iter()) {
            assert!(
                node.properties
                    .get("description")
                    .and_then(Value::as_str)
                    .is_some_and(|description| !description.trim().is_empty()),
                "node {} should have description",
                node.logical_id
            );
        }
    }

    #[test]
    fn graph_node_field_update_write_targets_updated_node_properties() {
        let trace = MemoryTrace::GraphNodeFieldsUpdated {
            update: crate::memory::trace::MemoryGraphNodeFieldUpdate {
                node_id: "person:travis".to_string(),
                label: Some("Travis".to_string()),
                fields: props([
                    ("preferred_name", json!("Trav")),
                    ("timezone", json!("America/Los_Angeles")),
                ]),
                source_text: Some("Pete command".to_string()),
                confidence: 0.97,
            },
            occurred_at: ExactTimestamp::now(),
        };

        let write = trace_write_for(&trace, 4);
        let target = write
            .related_nodes
            .iter()
            .find(|node| node.logical_id == "person:travis")
            .expect("updated graph node should be related");

        assert_eq!(write.primary_node.label, "GraphNodeFieldUpdate");
        assert_eq!(
            target
                .properties
                .get("preferred_name")
                .and_then(Value::as_str),
            Some("Trav")
        );
        assert_eq!(
            target.properties.get("timezone").and_then(Value::as_str),
            Some("America/Los_Angeles")
        );
        assert_eq!(
            target.properties.get("description").and_then(Value::as_str),
            Some("graph node named Travis")
        );
        assert!(write.relationships.iter().any(|relationship| {
            relationship.kind == "UPDATES_NODE" && relationship.to_logical_id == "person:travis"
        }));
    }
}
