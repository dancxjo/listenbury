use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use anyhow::Context as _;
use serde_json::Value;

use crate::memory::{EmbeddingProvider, QdrantSearchHit, QdrantStore};
use crate::mind::controller::ConversationRole;
use crate::mind::entity::{EntityExtractor, resolve_entities};

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
    Organization,
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
            ContextNodeRole::Organization => "Organization",
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

    pub fn graph_expansion_request(
        &self,
        max_depth: usize,
        max_nodes: usize,
        max_edges: usize,
    ) -> GraphExpansionRequest {
        let mut seen = HashSet::new();
        let mut roots = Vec::new();
        roots.push(self.self_node.clone());
        seen.insert(self.self_node.id.clone());
        for seed in self
            .selected_nodes
            .iter()
            .filter(|node| node.role == ContextNodeRole::RetrievedMemory)
            .map(|node| node.node.clone())
        {
            if seen.insert(seed.id.clone()) {
                roots.push(seed);
            }
        }
        GraphExpansionRequest {
            roots,
            max_depth,
            max_nodes,
            max_edges,
        }
    }

    pub fn expand_graph(
        &self,
        graph: &dyn ContextGraph,
        mut request: GraphExpansionRequest,
    ) -> ExpandedContextGraph {
        if request
            .roots
            .iter()
            .all(|root| root.id != self.self_node.id)
        {
            request.roots.insert(0, self.self_node.clone());
        }
        expand_context_graph(graph, request)
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

pub type GraphNodeId = String;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphTraversalEdge {
    pub from: GraphNodeId,
    pub to: GraphNodeId,
    pub kind: String,
}

pub trait ContextGraph: Send + Sync {
    fn node(&self, node_id: &str) -> Option<GraphNodeRef>;
    fn outgoing(&self, node_id: &str) -> Vec<GraphTraversalEdge>;
    fn incoming(&self, node_id: &str) -> Vec<GraphTraversalEdge>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphExpansionRequest {
    pub roots: Vec<GraphNodeRef>,
    pub max_depth: usize,
    pub max_nodes: usize,
    pub max_edges: usize,
}

impl Default for GraphExpansionRequest {
    fn default() -> Self {
        Self {
            roots: Vec::new(),
            max_depth: 2,
            max_nodes: 64,
            max_edges: 128,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalDirection {
    Outgoing,
    Incoming,
}

impl TraversalDirection {
    pub fn as_str(self) -> &'static str {
        match self {
            TraversalDirection::Outgoing => "outgoing",
            TraversalDirection::Incoming => "incoming",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraversalPathEdge {
    pub from: GraphNodeId,
    pub to: GraphNodeId,
    pub kind: String,
    pub direction: TraversalDirection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraversalProvenance {
    pub seed: GraphNodeRef,
    pub depth: usize,
    pub path: Vec<TraversalPathEdge>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpandedNode {
    pub node: GraphNodeRef,
    pub min_depth: usize,
    pub provenance: Vec<TraversalProvenance>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpandedEdge {
    pub edge: GraphTraversalEdge,
    pub min_depth: usize,
    pub provenance: Vec<TraversalProvenance>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpandedContextGraph {
    pub nodes: Vec<ExpandedNode>,
    pub edges: Vec<ExpandedEdge>,
}

impl ExpandedContextGraph {
    pub fn debug_view(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "nodes={} edges={}",
            self.nodes.len(),
            self.edges.len()
        ));
        for node in &self.nodes {
            let seeds = node
                .provenance
                .iter()
                .map(|entry| format!("{}@{}", entry.seed.id, entry.depth))
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!(
                "node {} depth={} seeds=[{}]",
                node.node.id, node.min_depth, seeds
            ));
        }
        for edge in &self.edges {
            let seeds = edge
                .provenance
                .iter()
                .map(|entry| format!("{}@{}", entry.seed.id, entry.depth))
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!(
                "edge {} -{}-> {} depth={} seeds=[{}]",
                edge.edge.from, edge.edge.kind, edge.edge.to, edge.min_depth, seeds
            ));
        }
        lines.join("\n")
    }
}

pub trait EmbeddingRecall: Send + Sync {
    fn recall(&self, query: RecallQuery) -> anyhow::Result<Vec<RecallHit>>;
}

pub fn expand_context_graph(
    graph: &dyn ContextGraph,
    request: GraphExpansionRequest,
) -> ExpandedContextGraph {
    let mut roots = Vec::new();
    let mut seen_roots = HashSet::new();
    for root in request.roots {
        if seen_roots.insert(root.id.clone()) {
            roots.push(root);
        }
    }
    let mut nodes_by_id = HashMap::<GraphNodeId, ExpandedNode>::new();
    let mut edges_by_key = HashMap::<(GraphNodeId, GraphNodeId, String), ExpandedEdge>::new();
    let mut queue = VecDeque::new();
    let mut visited_by_seed = HashMap::<GraphNodeId, HashSet<GraphNodeId>>::new();
    let max_nodes = request.max_nodes.max(1);

    for seed in roots {
        if nodes_by_id.len() >= max_nodes {
            break;
        }
        let node = graph.node(&seed.id).unwrap_or_else(|| seed.clone());
        let provenance = TraversalProvenance {
            seed: seed.clone(),
            depth: 0,
            path: Vec::new(),
        };
        upsert_node(&mut nodes_by_id, node.clone(), 0, provenance);
        visited_by_seed
            .entry(seed.id.clone())
            .or_default()
            .insert(seed.id.clone());
        queue.push_back(BfsState {
            node_id: seed.id.clone(),
            seed: seed.clone(),
            depth: 0,
            path: Vec::new(),
        });
    }

    while let Some(state) = queue.pop_front() {
        if state.depth >= request.max_depth {
            continue;
        }
        let next_depth = state.depth + 1;
        let mut traversals = Vec::new();
        traversals.extend(graph.outgoing(&state.node_id).into_iter().map(|edge| {
            (
                TraversalDirection::Outgoing,
                edge.to.clone(),
                TraversalPathEdge {
                    from: edge.from.clone(),
                    to: edge.to.clone(),
                    kind: edge.kind.clone(),
                    direction: TraversalDirection::Outgoing,
                },
                edge,
            )
        }));
        traversals.extend(graph.incoming(&state.node_id).into_iter().map(|edge| {
            (
                TraversalDirection::Incoming,
                edge.from.clone(),
                TraversalPathEdge {
                    from: edge.from.clone(),
                    to: edge.to.clone(),
                    kind: edge.kind.clone(),
                    direction: TraversalDirection::Incoming,
                },
                edge,
            )
        }));
        traversals.sort_by(|left, right| {
            let left_key = (
                &left.1,
                &left.3.from,
                &left.3.to,
                &left.3.kind,
                left.0.as_str(),
            );
            let right_key = (
                &right.1,
                &right.3.from,
                &right.3.to,
                &right.3.kind,
                right.0.as_str(),
            );
            left_key.cmp(&right_key)
        });

        for (_, neighbor_id, path_edge, edge) in traversals {
            let visited = visited_by_seed.entry(state.seed.id.clone()).or_default();
            if visited.contains(&neighbor_id) {
                continue;
            }

            let is_new_node = !nodes_by_id.contains_key(&neighbor_id);
            if is_new_node && nodes_by_id.len() >= max_nodes {
                continue;
            }

            let edge_key = (edge.from.clone(), edge.to.clone(), edge.kind.clone());
            let is_new_edge = !edges_by_key.contains_key(&edge_key);
            if is_new_edge && edges_by_key.len() >= request.max_edges {
                continue;
            }

            let mut next_path = state.path.clone();
            next_path.push(path_edge);
            let provenance = TraversalProvenance {
                seed: state.seed.clone(),
                depth: next_depth,
                path: next_path.clone(),
            };

            let neighbor = graph.node(&neighbor_id).unwrap_or_else(|| GraphNodeRef {
                id: neighbor_id.clone(),
                label: format!("Graph node {neighbor_id}"),
            });
            upsert_node(&mut nodes_by_id, neighbor, next_depth, provenance.clone());
            upsert_edge(&mut edges_by_key, edge, next_depth, provenance);

            visited.insert(neighbor_id.clone());
            queue.push_back(BfsState {
                node_id: neighbor_id,
                seed: state.seed.clone(),
                depth: next_depth,
                path: next_path,
            });
        }
    }

    let mut nodes = nodes_by_id.into_values().collect::<Vec<_>>();
    nodes.sort_by(|left, right| left.node.id.cmp(&right.node.id));
    let mut edges = edges_by_key.into_values().collect::<Vec<_>>();
    edges.sort_by(|left, right| {
        (
            &left.edge.from,
            &left.edge.to,
            &left.edge.kind,
            left.min_depth,
        )
            .cmp(&(
                &right.edge.from,
                &right.edge.to,
                &right.edge.kind,
                right.min_depth,
            ))
    });
    ExpandedContextGraph { nodes, edges }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BfsState {
    node_id: GraphNodeId,
    seed: GraphNodeRef,
    depth: usize,
    path: Vec<TraversalPathEdge>,
}

fn upsert_node(
    nodes_by_id: &mut HashMap<GraphNodeId, ExpandedNode>,
    node: GraphNodeRef,
    depth: usize,
    provenance: TraversalProvenance,
) {
    let entry = nodes_by_id
        .entry(node.id.clone())
        .or_insert_with(|| ExpandedNode {
            node: node.clone(),
            min_depth: depth,
            provenance: Vec::new(),
        });
    if depth < entry.min_depth {
        entry.min_depth = depth;
    }
    if !entry.provenance.contains(&provenance) {
        entry.provenance.push(provenance);
        entry.provenance.sort_by(|left, right| {
            (left.depth, &left.seed.id, left.path.len()).cmp(&(
                right.depth,
                &right.seed.id,
                right.path.len(),
            ))
        });
    }
}

fn upsert_edge(
    edges_by_key: &mut HashMap<(GraphNodeId, GraphNodeId, String), ExpandedEdge>,
    edge: GraphTraversalEdge,
    depth: usize,
    provenance: TraversalProvenance,
) {
    let key = (edge.from.clone(), edge.to.clone(), edge.kind.clone());
    let entry = edges_by_key.entry(key).or_insert_with(|| ExpandedEdge {
        edge: edge.clone(),
        min_depth: depth,
        provenance: Vec::new(),
    });
    if depth < entry.min_depth {
        entry.min_depth = depth;
    }
    if !entry.provenance.contains(&provenance) {
        entry.provenance.push(provenance);
        entry.provenance.sort_by(|left, right| {
            (left.depth, &left.seed.id, left.path.len()).cmp(&(
                right.depth,
                &right.seed.id,
                right.path.len(),
            ))
        });
    }
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
        hits.retain(|hit| hit.score.is_finite());
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
    entity_extractor: Option<Arc<dyn EntityExtractor>>,
}

impl EmbeddingRecallProvider {
    pub fn new(self_node: GraphNodeRef) -> Self {
        Self {
            self_node,
            recall: None,
            recall_limit: 8,
            min_score: None,
            conversation_tail_limit: 6,
            entity_extractor: None,
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

    /// Attach an entity extractor.  Extracted entities will be added to
    /// `selected_nodes` as provisional or resolved `ContextNode`s and a
    /// `tracing::debug!` line will log them for every turn.
    pub fn with_entity_extractor(mut self, extractor: Arc<dyn EntityExtractor>) -> Self {
        self.entity_extractor = Some(extractor);
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
        let mut selected = Vec::new();

        // Entity extraction — runs regardless of whether embedding recall is
        // configured so that entity anchors are always available.
        if let Some(extractor) = &self.entity_extractor {
            let extracted = extractor.extract(utterance);
            tracing::debug!(
                turn_entities = ?extracted
                    .iter()
                    .map(|e| format!("{}({:?}, {:.2})", e.text, e.kind, e.confidence))
                    .collect::<Vec<_>>(),
                "extracted entities from utterance"
            );
            let entity_nodes = resolve_entities(&extracted, &|_| None);
            selected.extend(entity_nodes);
        }

        // Embedding recall nodes.
        if let Some(recall) = self.recall.as_ref() {
            let query_text =
                recall_query_text(utterance, conversation_tail, self.conversation_tail_limit);
            if !query_text.trim().is_empty() {
                let recall_query = RecallQuery {
                    text: query_text,
                    limit: self.recall_limit,
                    min_score: self.min_score,
                };
                match recall.recall(recall_query) {
                    Ok(hits) => {
                        let recall_nodes: Vec<ContextNode> = hits
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
                            .collect();
                        let seeds: Vec<String> = recall_nodes
                            .iter()
                            .map(|node| format!("{}:{:.3}", node.node.id, node.relevance))
                            .collect();
                        tracing::debug!(?seeds, "embedding recall selected context seeds");
                        selected.extend(recall_nodes);
                    }
                    Err(error) => {
                        tracing::warn!("embedding recall failed: {error:#}");
                    }
                }
            }
        }

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
    // Keep `neo4j_node_id` fallback for old payloads while `graph_node_id`
    // is the preferred database-agnostic key going forward.
    let node_id = payload_string(&hit.payload, &GRAPH_NODE_ID_KEYS)?;
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

const GRAPH_NODE_ID_KEYS: [&str; 2] = ["graph_node_id", "neo4j_node_id"];

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
            .filter_map(|turn| {
                let text = turn.text.trim();
                if text.is_empty() {
                    return None;
                }
                Some(format!("{}: {text}", turn.role.label()))
            })
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
    use std::collections::HashMap;
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

    #[test]
    fn payload_string_uses_first_non_empty_key() {
        let payload = serde_json::from_value(json!({
            "primary": "  ",
            "secondary": "fallback",
            "ignored": 42
        }))
        .expect("valid payload");
        assert_eq!(
            payload_string(&payload, &["primary", "secondary", "ignored"]),
            Some("fallback".to_string())
        );
        assert_eq!(payload_string(&payload, &["missing"]), None);
    }

    #[test]
    fn recall_query_text_combines_utterance_and_tail() {
        let text = recall_query_text(
            "  hello  ",
            &[
                ConversationTurn {
                    role: ConversationRole::User,
                    text: "first".to_string(),
                },
                ConversationTurn {
                    role: ConversationRole::Pete,
                    text: "second".to_string(),
                },
                ConversationTurn {
                    role: ConversationRole::User,
                    text: "   ".to_string(),
                },
            ],
            2,
        );
        assert!(text.contains("Utterance: hello"));
        assert!(text.contains("Pete: second"));
        assert!(!text.contains("User:    "));

        let only_utterance = recall_query_text("hello", &[], 0);
        assert_eq!(only_utterance, "Utterance: hello");
    }

    #[test]
    fn graph_expansion_traverses_multiple_hops_incoming_and_outgoing() {
        let context = ConversationContext {
            system_prompt: "system".to_string(),
            self_node: GraphNodeRef {
                id: DEFAULT_SELF_NODE_ID.to_string(),
                label: DEFAULT_SELF_NODE_LABEL.to_string(),
            },
            selected_nodes: vec![ContextNode {
                node: GraphNodeRef {
                    id: "topic:latency".to_string(),
                    label: "Latency".to_string(),
                },
                role: ContextNodeRole::RetrievedMemory,
                relevance: 0.9,
                reason: "vector recall".to_string(),
                summary: "Latency topic".to_string(),
            }],
            conversation_tail: Vec::new(),
            budget: ContextBudget::default(),
        };

        let graph = StaticContextGraph::new(
            vec![
                GraphNodeRef {
                    id: DEFAULT_SELF_NODE_ID.to_string(),
                    label: DEFAULT_SELF_NODE_LABEL.to_string(),
                },
                GraphNodeRef {
                    id: "user:dan".to_string(),
                    label: "Dan".to_string(),
                },
                GraphNodeRef {
                    id: "episode:1".to_string(),
                    label: "Episode 1".to_string(),
                },
                GraphNodeRef {
                    id: "topic:latency".to_string(),
                    label: "Latency".to_string(),
                },
            ],
            vec![
                GraphTraversalEdge {
                    from: DEFAULT_SELF_NODE_ID.to_string(),
                    to: "user:dan".to_string(),
                    kind: "KNOWS".to_string(),
                },
                GraphTraversalEdge {
                    from: "user:dan".to_string(),
                    to: "episode:1".to_string(),
                    kind: "SPOKE_IN".to_string(),
                },
                GraphTraversalEdge {
                    from: "episode:1".to_string(),
                    to: "topic:latency".to_string(),
                    kind: "ABOUT".to_string(),
                },
            ],
        );

        let expanded = context.expand_graph(
            &graph,
            context.graph_expansion_request(
                2,  // multi-hop
                32, // node budget
                64, // edge budget
            ),
        );

        assert!(
            expanded
                .nodes
                .iter()
                .any(|node| node.node.id == DEFAULT_SELF_NODE_ID)
        );
        assert!(
            expanded
                .nodes
                .iter()
                .any(|node| node.node.id == "topic:latency")
        );
        assert!(
            expanded
                .nodes
                .iter()
                .any(|node| node.node.id == "episode:1")
        );
        assert!(expanded.nodes.iter().any(|node| node.node.id == "user:dan"));
        assert!(
            expanded
                .edges
                .iter()
                .any(|edge| edge.edge.from == "episode:1" && edge.edge.to == "topic:latency")
        );

        let user = expanded
            .nodes
            .iter()
            .find(|node| node.node.id == "user:dan")
            .expect("user should be expanded");
        assert!(user.provenance.iter().any(|provenance| {
            provenance.seed.id == "topic:latency"
                && provenance.depth == 2
                && provenance.path.len() == 2
        }));
    }

    #[test]
    fn graph_expansion_respects_node_and_edge_limits() {
        let context = ConversationContext {
            system_prompt: "system".to_string(),
            self_node: GraphNodeRef {
                id: DEFAULT_SELF_NODE_ID.to_string(),
                label: DEFAULT_SELF_NODE_LABEL.to_string(),
            },
            selected_nodes: vec![ContextNode {
                node: GraphNodeRef {
                    id: "topic:seed".to_string(),
                    label: "Seed".to_string(),
                },
                role: ContextNodeRole::RetrievedMemory,
                relevance: 0.9,
                reason: "vector recall".to_string(),
                summary: "seed topic".to_string(),
            }],
            conversation_tail: Vec::new(),
            budget: ContextBudget::default(),
        };

        let graph = StaticContextGraph::new(
            vec![
                GraphNodeRef {
                    id: DEFAULT_SELF_NODE_ID.to_string(),
                    label: DEFAULT_SELF_NODE_LABEL.to_string(),
                },
                GraphNodeRef {
                    id: "topic:seed".to_string(),
                    label: "Seed".to_string(),
                },
                GraphNodeRef {
                    id: "node:a".to_string(),
                    label: "A".to_string(),
                },
                GraphNodeRef {
                    id: "node:b".to_string(),
                    label: "B".to_string(),
                },
                GraphNodeRef {
                    id: "node:c".to_string(),
                    label: "C".to_string(),
                },
            ],
            vec![
                GraphTraversalEdge {
                    from: "topic:seed".to_string(),
                    to: "node:a".to_string(),
                    kind: "REL".to_string(),
                },
                GraphTraversalEdge {
                    from: "topic:seed".to_string(),
                    to: "node:b".to_string(),
                    kind: "REL".to_string(),
                },
                GraphTraversalEdge {
                    from: "topic:seed".to_string(),
                    to: "node:c".to_string(),
                    kind: "REL".to_string(),
                },
            ],
        );

        let expanded = context.expand_graph(
            &graph,
            GraphExpansionRequest {
                roots: vec![GraphNodeRef {
                    id: "topic:seed".to_string(),
                    label: "Seed".to_string(),
                }],
                max_depth: 3,
                max_nodes: 3,
                max_edges: 2,
            },
        );

        assert!(expanded.nodes.len() <= 3);
        assert!(expanded.edges.len() <= 2);
        assert!(
            expanded
                .nodes
                .iter()
                .any(|node| node.node.id == DEFAULT_SELF_NODE_ID)
        );
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

    #[derive(Clone)]
    struct StaticContextGraph {
        nodes: HashMap<String, GraphNodeRef>,
        edges: Vec<GraphTraversalEdge>,
    }

    impl StaticContextGraph {
        fn new(nodes: Vec<GraphNodeRef>, edges: Vec<GraphTraversalEdge>) -> Self {
            Self {
                nodes: nodes
                    .into_iter()
                    .map(|node| (node.id.clone(), node))
                    .collect(),
                edges,
            }
        }
    }

    impl ContextGraph for StaticContextGraph {
        fn node(&self, node_id: &str) -> Option<GraphNodeRef> {
            self.nodes.get(node_id).cloned()
        }

        fn outgoing(&self, node_id: &str) -> Vec<GraphTraversalEdge> {
            self.edges
                .iter()
                .filter(|edge| edge.from == node_id)
                .cloned()
                .collect()
        }

        fn incoming(&self, node_id: &str) -> Vec<GraphTraversalEdge> {
            self.edges
                .iter()
                .filter(|edge| edge.to == node_id)
                .cloned()
                .collect()
        }
    }
}
