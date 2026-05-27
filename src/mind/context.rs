use std::cmp::Ordering;
use std::sync::Arc;

use anyhow::Context as _;
use serde_json::Value;

use crate::memory::{EmbeddingProvider, QdrantSearchHit, QdrantStore};
use crate::mind::controller::ConversationRole;

pub const DEFAULT_CONTEXT_MAX_CHARS: usize = 1_024;
pub const DEFAULT_SELF_NODE_ID: &str = "pete:self";
pub const DEFAULT_SELF_NODE_LABEL: &str = "Pete Listenbury";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphNodeRef {
    pub id: String,
    pub label: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextBudget {
    pub max_chars: usize,
}

impl Default for ContextBudget {
    fn default() -> Self {
        Self {
            max_chars: DEFAULT_CONTEXT_MAX_CHARS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationTurn {
    pub role: ConversationRole,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextNodeRole {
    SelfIdentity,
    CurrentUser,
    RecentMention,
    RetrievedMemory,
    ActiveTopic,
    Place,
    Object,
    Task,
}

impl ContextNodeRole {
    pub fn as_str(self) -> &'static str {
        match self {
            ContextNodeRole::SelfIdentity => "SelfIdentity",
            ContextNodeRole::CurrentUser => "CurrentUser",
            ContextNodeRole::RecentMention => "RecentMention",
            ContextNodeRole::RetrievedMemory => "RetrievedMemory",
            ContextNodeRole::ActiveTopic => "ActiveTopic",
            ContextNodeRole::Place => "Place",
            ContextNodeRole::Object => "Object",
            ContextNodeRole::Task => "Task",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContextNode {
    pub node: GraphNodeRef,
    pub role: ContextNodeRole,
    pub relevance: f32,
    pub reason: String,
    pub summary: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConversationContext {
    pub system_prompt: String,
    pub self_node: GraphNodeRef,
    pub selected_nodes: Vec<ContextNode>,
    pub conversation_tail: Vec<ConversationTurn>,
    pub budget: ContextBudget,
}

impl ConversationContext {
    pub fn render_compact_nodes(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "- [{}] {} ({}) — {}",
            ContextNodeRole::SelfIdentity.as_str(),
            self.self_node.label,
            self.self_node.id,
            "Pete's persistent self node"
        ));

        let mut used_chars = lines.iter().map(String::len).sum::<usize>();
        let mut omitted = 0usize;

        let mut selected_nodes = self.selected_nodes.iter().collect::<Vec<_>>();
        selected_nodes.sort_by(|left, right| right.relevance.total_cmp(&left.relevance));

        for selected in selected_nodes {
            let line = format!(
                "- [{}] {} ({}) rel={:.2} reason={} summary={}",
                selected.role.as_str(),
                selected.node.label,
                selected.node.id,
                selected.relevance,
                selected.reason.trim(),
                selected.summary.trim()
            );
            if used_chars + line.len() > self.budget.max_chars {
                omitted += 1;
                continue;
            }
            used_chars += line.len();
            lines.push(line);
        }

        if omitted > 0 {
            lines.push(format!(
                "- [Budget] Omitted {omitted} lower-priority node(s) due to context budget"
            ));
        }

        lines.join("\n")
    }

    pub fn debug_nodes(&self) -> String {
        let selected = if self.selected_nodes.is_empty() {
            "none".to_string()
        } else {
            self.selected_nodes
                .iter()
                .map(|node| {
                    format!(
                        "{}:{}({:.2})",
                        node.role.as_str(),
                        node.node.id,
                        node.relevance
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!("self={} selected=[{}]", self.self_node.id, selected)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecallQuery {
    pub text: String,
    pub limit: usize,
    pub min_score: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecallSource {
    VectorStore {
        collection: String,
        point_id: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecallHit {
    pub node: GraphNodeRef,
    pub score: f32,
    pub source: RecallSource,
    pub reason: String,
    pub summary: Option<String>,
}

pub trait EmbeddingRecall: Send + Sync {
    fn recall(&self, query: RecallQuery) -> anyhow::Result<Vec<RecallHit>>;
}

#[derive(Clone)]
pub struct QdrantEmbeddingRecall {
    qdrant: Arc<dyn QdrantStore>,
    embeddings: Arc<dyn EmbeddingProvider>,
    collection: String,
}

impl QdrantEmbeddingRecall {
    pub fn new(
        qdrant: Arc<dyn QdrantStore>,
        embeddings: Arc<dyn EmbeddingProvider>,
        collection: impl Into<String>,
    ) -> Self {
        Self {
            qdrant,
            embeddings,
            collection: collection.into(),
        }
    }
}

impl EmbeddingRecall for QdrantEmbeddingRecall {
    fn recall(&self, query: RecallQuery) -> anyhow::Result<Vec<RecallHit>> {
        let text = query.text.trim();
        if text.is_empty() || query.limit == 0 {
            return Ok(Vec::new());
        }
        let vector = self
            .embeddings
            .embed(text)
            .with_context(|| format!("embed recall query text ({text})"))?;
        let mut hits = self.qdrant.search(&self.collection, &vector, query.limit)?;
        hits.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
        });
        Ok(hits
            .into_iter()
            .filter_map(|hit| recall_hit_from_qdrant_hit(hit, &self.collection, query.min_score))
            .collect())
    }
}

#[derive(Clone)]
pub struct EmbeddingRecallProvider {
    self_node: GraphNodeRef,
    recall: Option<Arc<dyn EmbeddingRecall>>,
    recall_limit: usize,
    min_score: Option<f32>,
    conversation_tail_limit: usize,
}

impl EmbeddingRecallProvider {
    pub fn new(self_node: GraphNodeRef) -> Self {
        Self {
            self_node,
            recall: None,
            recall_limit: 8,
            min_score: None,
            conversation_tail_limit: 6,
        }
    }

    pub fn with_recall(mut self, recall: Arc<dyn EmbeddingRecall>) -> Self {
        self.recall = Some(recall);
        self
    }

    pub fn with_recall_limit(mut self, recall_limit: usize) -> Self {
        self.recall_limit = recall_limit;
        self
    }

    pub fn with_min_score(mut self, min_score: Option<f32>) -> Self {
        self.min_score = min_score;
        self
    }

    pub fn with_conversation_tail_limit(mut self, conversation_tail_limit: usize) -> Self {
        self.conversation_tail_limit = conversation_tail_limit;
        self
    }
}

pub trait ContextProvider {
    fn self_node(&self) -> GraphNodeRef;
    fn selected_nodes(
        &self,
        _utterance: &str,
        _conversation_tail: &[ConversationTurn],
        _budget: &ContextBudget,
    ) -> Vec<ContextNode> {
        Vec::new()
    }
}

impl ContextProvider for EmbeddingRecallProvider {
    fn self_node(&self) -> GraphNodeRef {
        self.self_node.clone()
    }

    fn selected_nodes(
        &self,
        utterance: &str,
        conversation_tail: &[ConversationTurn],
        _budget: &ContextBudget,
    ) -> Vec<ContextNode> {
        let Some(recall) = self.recall.as_ref() else {
            return Vec::new();
        };
        let query_text =
            recall_query_text(utterance, conversation_tail, self.conversation_tail_limit);
        if query_text.trim().is_empty() {
            return Vec::new();
        }
        let recall_query = RecallQuery {
            text: query_text,
            limit: self.recall_limit,
            min_score: self.min_score,
        };
        let mut hits = match recall.recall(recall_query) {
            Ok(hits) => hits,
            Err(error) => {
                tracing::warn!("embedding recall failed: {error:#}");
                return Vec::new();
            }
        };
        hits.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
        });
        let selected = hits
            .into_iter()
            .map(|hit| ContextNode {
                node: hit.node,
                role: ContextNodeRole::RetrievedMemory,
                relevance: hit.score,
                reason: hit.reason,
                summary: hit
                    .summary
                    .unwrap_or_else(|| "Retrieved from embedding recall".to_string()),
            })
            .collect::<Vec<_>>();
        let seeds = selected
            .iter()
            .map(|node| format!("{}:{:.3}", node.node.id, node.relevance))
            .collect::<Vec<_>>();
        tracing::debug!(?seeds, "embedding recall selected context seeds");
        selected
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StubContextProvider {
    self_node: GraphNodeRef,
}

fn recall_hit_from_qdrant_hit(
    hit: QdrantSearchHit,
    collection: &str,
    min_score: Option<f32>,
) -> Option<RecallHit> {
    if min_score.is_some_and(|minimum| hit.score < minimum) {
        return None;
    }
    let node_id = payload_string(&hit.payload, &["neo4j_node_id", "graph_node_id"])?;
    let node_label = payload_string(&hit.payload, &["headline", "node_label", "text"])
        .unwrap_or_else(|| format!("Graph node {node_id}"));
    let kind = payload_string(&hit.payload, &["kind"]).unwrap_or_else(|| "memory".to_string());
    let reason = format!(
        "vector similarity {:.3} from {} point {} ({kind})",
        hit.score, collection, hit.id
    );
    let summary = payload_string(&hit.payload, &["text", "headline"]);
    Some(RecallHit {
        node: GraphNodeRef {
            id: node_id,
            label: node_label,
        },
        score: hit.score,
        source: RecallSource::VectorStore {
            collection: collection.to_string(),
            point_id: hit.id,
        },
        reason,
        summary,
    })
}

fn payload_string(payload: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| payload.get(*key).and_then(Value::as_str))
        .map(str::trim)
        .find(|value| !value.is_empty())
        .map(str::to_string)
}

fn recall_query_text(
    utterance: &str,
    conversation_tail: &[ConversationTurn],
    conversation_tail_limit: usize,
) -> String {
    let mut sections = Vec::new();
    let utterance = utterance.trim();
    if !utterance.is_empty() {
        sections.push(format!("Utterance: {utterance}"));
    }
    if conversation_tail_limit > 0 && !conversation_tail.is_empty() {
        let start = conversation_tail
            .len()
            .saturating_sub(conversation_tail_limit);
        let tail = conversation_tail[start..]
            .iter()
            .map(|turn| format!("{}: {}", turn.role.label(), turn.text.trim()))
            .filter(|line| !line.ends_with(": "))
            .collect::<Vec<_>>();
        if !tail.is_empty() {
            sections.push(format!("Conversation tail:\n{}", tail.join("\n")));
        }
    }
    sections.join("\n\n")
}

impl StubContextProvider {
    pub fn new(self_node: GraphNodeRef) -> Self {
        Self { self_node }
    }
}

impl Default for StubContextProvider {
    fn default() -> Self {
        Self::new(GraphNodeRef {
            id: DEFAULT_SELF_NODE_ID.to_string(),
            label: DEFAULT_SELF_NODE_LABEL.to_string(),
        })
    }
}

impl ContextProvider for StubContextProvider {
    fn self_node(&self) -> GraphNodeRef {
        self.self_node.clone()
    }
}

pub fn build_conversation_context(
    provider: &dyn ContextProvider,
    system_prompt: impl Into<String>,
    utterance: &str,
    conversation_tail: Vec<ConversationTurn>,
    budget: ContextBudget,
) -> ConversationContext {
    let self_node = provider.self_node();
    let selected_nodes = provider.selected_nodes(utterance, &conversation_tail, &budget);
    ConversationContext {
        system_prompt: system_prompt.into(),
        self_node,
        selected_nodes,
        conversation_tail,
        budget,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::*;
    use crate::memory::{EmbeddingProvider, QdrantPoint, QdrantSearchHit, QdrantStore};

    #[test]
    fn stub_context_always_contains_self_node() {
        let provider = StubContextProvider::default();
        let context = build_conversation_context(
            &provider,
            "system",
            "hello",
            Vec::new(),
            ContextBudget { max_chars: 200 },
        );
        assert_eq!(context.self_node.id, DEFAULT_SELF_NODE_ID);
        assert!(
            context
                .render_compact_nodes()
                .contains(DEFAULT_SELF_NODE_ID)
        );
    }

    #[test]
    fn compact_nodes_respect_budget() {
        let context = ConversationContext {
            system_prompt: "system".to_string(),
            self_node: GraphNodeRef {
                id: DEFAULT_SELF_NODE_ID.to_string(),
                label: DEFAULT_SELF_NODE_LABEL.to_string(),
            },
            selected_nodes: vec![
                ContextNode {
                    node: GraphNodeRef {
                        id: "topic:latency".to_string(),
                        label: "Latency".to_string(),
                    },
                    role: ContextNodeRole::ActiveTopic,
                    relevance: 0.95,
                    reason: "recently mentioned".to_string(),
                    summary: "User asked about latency in the latest turn.".to_string(),
                },
                ContextNode {
                    node: GraphNodeRef {
                        id: "memory:old".to_string(),
                        label: "Old memory".to_string(),
                    },
                    role: ContextNodeRole::RetrievedMemory,
                    relevance: 0.25,
                    reason: "older mention".to_string(),
                    summary: "Low-priority memory node that should be omitted.".to_string(),
                },
            ],
            conversation_tail: Vec::new(),
            budget: ContextBudget { max_chars: 260 },
        };

        let rendered = context.render_compact_nodes();
        assert!(rendered.contains(DEFAULT_SELF_NODE_ID));
        assert!(rendered.contains("topic:latency"));
        assert!(!rendered.contains("memory:old"));
        assert!(rendered.contains("Omitted 1 lower-priority node"));
    }

    #[test]
    fn embedding_recall_provider_without_vector_backend_is_graceful() {
        let provider = EmbeddingRecallProvider::new(GraphNodeRef {
            id: DEFAULT_SELF_NODE_ID.to_string(),
            label: DEFAULT_SELF_NODE_LABEL.to_string(),
        });
        let selected = provider.selected_nodes(
            "hello",
            &[ConversationTurn {
                role: ConversationRole::User,
                text: "Can you remember my latency question?".to_string(),
            }],
            &ContextBudget::default(),
        );
        assert!(selected.is_empty());
    }

    #[test]
    fn qdrant_recall_orders_hits_by_relevance() {
        let qdrant = Arc::new(StaticQdrantStore {
            hits: vec![
                QdrantSearchHit {
                    id: "point:low".to_string(),
                    score: 0.33,
                    payload: serde_json::from_value(json!({
                        "neo4j_node_id": "node:low",
                        "headline": "Older topic",
                        "text": "Low relevance topic",
                        "kind": "conversation_turn"
                    }))
                    .expect("valid payload"),
                },
                QdrantSearchHit {
                    id: "point:high".to_string(),
                    score: 0.95,
                    payload: serde_json::from_value(json!({
                        "neo4j_node_id": "node:high",
                        "headline": "Current topic",
                        "text": "Most relevant topic",
                        "kind": "summary"
                    }))
                    .expect("valid payload"),
                },
            ],
        });
        let embeddings = Arc::new(StaticEmbeddingProvider);
        let recall = QdrantEmbeddingRecall::new(qdrant, embeddings, "listenbury_memory");

        let hits = recall
            .recall(RecallQuery {
                text: "latest topic".to_string(),
                limit: 8,
                min_score: Some(0.2),
            })
            .expect("recall should succeed");

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].node.id, "node:high");
        assert_eq!(hits[1].node.id, "node:low");
        assert!(hits[0].score >= hits[1].score);
    }

    #[derive(Clone)]
    struct StaticQdrantStore {
        hits: Vec<QdrantSearchHit>,
    }

    impl QdrantStore for StaticQdrantStore {
        fn upsert_points(&self, _collection: &str, _points: &[QdrantPoint]) -> anyhow::Result<()> {
            Ok(())
        }

        fn search(
            &self,
            _collection: &str,
            _query_vector: &[f32],
            limit: usize,
        ) -> anyhow::Result<Vec<QdrantSearchHit>> {
            Ok(self.hits.iter().take(limit).cloned().collect())
        }
    }

    struct StaticEmbeddingProvider;

    impl EmbeddingProvider for StaticEmbeddingProvider {
        fn embed(&self, _text: &str) -> anyhow::Result<Vec<f32>> {
            Ok(vec![0.1, 0.2, 0.3])
        }
    }
}
