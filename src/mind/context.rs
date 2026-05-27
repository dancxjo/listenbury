use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use anyhow::Context as _;
use serde_json::{Map, Value};

use crate::memory::{EmbeddingProvider, QdrantSearchHit, QdrantStore};
use crate::mind::controller::ConversationRole;
use crate::mind::entity::{EntityExtractor, resolve_entities};

pub const DEFAULT_CONTEXT_MAX_CHARS: usize = 1_024;
pub const DEFAULT_GRAPH_SUMMARY_MAX_CHARS: usize = 768;
pub const DEFAULT_GRAPH_SUMMARY_CHARS_PER_TOKEN: usize = 4;
pub const DEFAULT_SELF_NODE_ID: &str = "pete:self";
pub const DEFAULT_SELF_NODE_LABEL: &str = "Pete Listenbury";
pub const DEFAULT_SELF_NODE_SUMMARY: &str = "Pete is the Listenbury live voice system. The user is speaking aloud; ASR transcribes that speech into the text Pete receives, and Pete speaks replies aloud through TTS. Pete may receive conversation history, retrieved memories, and working-memory graph nodes in this prompt. When asked about identity, hearing, memory, or how the prompt works, answer from these runtime facts: Pete is not just a generic text-only chatbot, and should not claim there is no speech input, no memory context, or no larger Listenbury system.";

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
pub struct GraphNodeFieldUpdate {
    pub node_id: GraphNodeId,
    pub label: Option<String>,
    pub fields: Map<String, Value>,
    pub reason: String,
    pub relevance: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphNodeSearchQuery {
    pub text: Option<String>,
    pub field: Option<String>,
    pub value: Option<Value>,
    pub limit: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphNodeSearchHit {
    pub node: GraphNodeRef,
    pub score: f32,
    pub fields: Map<String, Value>,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinScope {
    Permanent,
    Session,
    Temporary { remaining_turns: usize },
}

impl PinScope {
    pub fn as_str(self) -> &'static str {
        match self {
            PinScope::Permanent => "Permanent",
            PinScope::Session => "Session",
            PinScope::Temporary { .. } => "Temporary",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedContextNode {
    pub node_id: GraphNodeId,
    pub scope: PinScope,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TopicActivation {
    pub node_id: GraphNodeId,
    pub salience: f32,
    pub last_activated_turn: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConversationContext {
    pub system_prompt: String,
    pub self_node: GraphNodeRef,
    pub pinned_nodes: Vec<PinnedContextNode>,
    pub active_topics: Vec<TopicActivation>,
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
            DEFAULT_SELF_NODE_SUMMARY
        ));

        let mut used_chars = lines.iter().map(String::len).sum::<usize>();
        for pinned in &self.pinned_nodes {
            let line = format!(
                "- [Pinned:{}] {} reason={}",
                pinned.scope.as_str(),
                pinned.node_id,
                pinned.reason.trim()
            );
            used_chars += line.len();
            lines.push(line);
        }
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
        let pinned = if self.pinned_nodes.is_empty() {
            "none".to_string()
        } else {
            self.pinned_nodes
                .iter()
                .map(|node| format!("{}:{}({})", node.scope.as_str(), node.node_id, node.reason))
                .collect::<Vec<_>>()
                .join(", ")
        };
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
        let active_topics = if self.active_topics.is_empty() {
            "none".to_string()
        } else {
            self.active_topics
                .iter()
                .map(|topic| format!("{}({:.2})", topic.node_id, topic.salience))
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!(
            "self={} pinned=[{}] active_topics=[{}] selected=[{}]",
            self.self_node.id, pinned, active_topics, selected
        )
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
        for pinned in &self.pinned_nodes {
            if seen.insert(pinned.node_id.clone()) {
                roots.push(GraphNodeRef {
                    id: pinned.node_id.clone(),
                    label: format!("Pinned node {}", pinned.node_id),
                });
            }
        }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphNeighborhoodSummaryConfig {
    pub max_chars: usize,
    pub max_tokens: Option<usize>,
    pub chars_per_token: usize,
    pub verbatim_node_ids: Vec<GraphNodeId>,
    pub max_edge_lines: usize,
}

impl Default for GraphNeighborhoodSummaryConfig {
    fn default() -> Self {
        Self {
            max_chars: DEFAULT_GRAPH_SUMMARY_MAX_CHARS,
            max_tokens: None,
            chars_per_token: DEFAULT_GRAPH_SUMMARY_CHARS_PER_TOKEN,
            verbatim_node_ids: Vec::new(),
            max_edge_lines: 12,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphNeighborhoodSummaryStats {
    pub original_node_count: usize,
    pub original_edge_count: usize,
    pub compressed_node_count: usize,
    pub compressed_edge_count: usize,
    pub rendered_chars: usize,
    pub rendered_tokens_estimate: usize,
    pub budget_chars: usize,
    pub budget_tokens: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphNeighborhoodSummary {
    pub rendered: String,
    pub kept_node_ids: Vec<GraphNodeId>,
    pub kept_edge_count: usize,
    pub stats: GraphNeighborhoodSummaryStats,
}

impl GraphNeighborhoodSummary {
    pub fn debug_view(&self) -> String {
        format!(
            "original_graph=nodes:{} edges:{} compressed_graph=nodes:{} edges:{} prompt_budget=chars:{}/{} tokens~:{}/{}",
            self.stats.original_node_count,
            self.stats.original_edge_count,
            self.stats.compressed_node_count,
            self.stats.compressed_edge_count,
            self.stats.rendered_chars,
            self.stats.budget_chars,
            self.stats.rendered_tokens_estimate,
            self.stats
                .budget_tokens
                .map_or_else(|| "none".to_string(), |tokens| tokens.to_string()),
        )
    }
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

    pub fn summarize_neighborhood(
        &self,
        config: GraphNeighborhoodSummaryConfig,
    ) -> GraphNeighborhoodSummary {
        let budget_chars = effective_graph_summary_char_budget(&config);
        let mut lines = Vec::new();
        let mut used_chars = 0usize;

        let header = format!(
            "Neighborhood summary from {} node(s), {} edge(s):",
            self.nodes.len(),
            self.edges.len()
        );
        append_summary_line(&mut lines, &mut used_chars, budget_chars, header);

        let verbatim_ids = config
            .verbatim_node_ids
            .iter()
            .map(|id| id.as_str())
            .collect::<HashSet<_>>();
        let mut nodes = self.nodes.iter().collect::<Vec<_>>();
        nodes.sort_by(|left, right| {
            node_priority(right, &verbatim_ids)
                .cmp(&node_priority(left, &verbatim_ids))
                .then_with(|| left.min_depth.cmp(&right.min_depth))
                .then_with(|| left.node.id.cmp(&right.node.id))
        });

        let mut kept_node_ids = Vec::new();
        let mut emitted_node_ids = HashSet::new();
        let mut omitted_nodes = 0usize;

        for node in nodes {
            let is_verbatim = verbatim_ids.contains(node.node.id.as_str());
            let line = summarize_node_line(node, &self.edges, is_verbatim);
            if append_summary_line(&mut lines, &mut used_chars, budget_chars, line.clone()) {
                if emitted_node_ids.insert(node.node.id.as_str()) {
                    kept_node_ids.push(node.node.id.clone());
                }
                continue;
            }

            let is_critical = is_verbatim || node.min_depth == 0;
            if is_critical
                && append_summary_line(
                    &mut lines,
                    &mut used_chars,
                    budget_chars,
                    truncate_for_budget(&line, 72),
                )
            {
                if emitted_node_ids.insert(node.node.id.as_str()) {
                    kept_node_ids.push(node.node.id.clone());
                }
                continue;
            }

            omitted_nodes += 1;
        }

        if omitted_nodes > 0 {
            append_summary_line(
                &mut lines,
                &mut used_chars,
                budget_chars,
                format!("- [Compressed] Omitted {omitted_nodes} node(s) due to budget limits."),
            );
        }

        let kept_node_set = kept_node_ids
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        let mut kept_edge_count = 0usize;
        let mut edge_lines = 0usize;
        for edge in &self.edges {
            if edge_lines >= config.max_edge_lines {
                break;
            }
            if !kept_node_set.contains(edge.edge.from.as_str())
                || !kept_node_set.contains(edge.edge.to.as_str())
            {
                continue;
            }
            let line = format!(
                "- [Edge] {} -{}-> {} (depth={})",
                edge.edge.from, edge.edge.kind, edge.edge.to, edge.min_depth
            );
            if !append_summary_line(&mut lines, &mut used_chars, budget_chars, line) {
                break;
            }
            kept_edge_count += 1;
            edge_lines += 1;
        }

        let rendered = lines.join("\n");
        let rendered_chars = rendered.len();
        let rendered_tokens_estimate = estimate_tokens(rendered_chars, config.chars_per_token);
        GraphNeighborhoodSummary {
            rendered,
            kept_node_ids,
            kept_edge_count,
            stats: GraphNeighborhoodSummaryStats {
                original_node_count: self.nodes.len(),
                original_edge_count: self.edges.len(),
                compressed_node_count: emitted_node_ids.len(),
                compressed_edge_count: kept_edge_count,
                rendered_chars,
                rendered_tokens_estimate,
                budget_chars,
                budget_tokens: config.max_tokens,
            },
        }
    }
}

fn node_priority(node: &ExpandedNode, verbatim_ids: &HashSet<&str>) -> i32 {
    let mut priority = (10_i32.saturating_sub(node.min_depth as i32)).saturating_mul(10);
    priority = priority.saturating_add((node.provenance.len() as i32).saturating_mul(4));
    if verbatim_ids.contains(node.node.id.as_str()) {
        priority = priority.saturating_add(1_000);
    }
    if looks_eventful(&node.node) {
        priority = priority.saturating_add(20);
    }
    if looks_emotional(&node.node) {
        priority = priority.saturating_add(20);
    }
    priority
}

fn summarize_node_line(node: &ExpandedNode, edges: &[ExpandedEdge], verbatim: bool) -> String {
    let mut outgoing = Vec::new();
    let mut incoming = Vec::new();
    for edge in edges {
        if edge.edge.from == node.node.id {
            outgoing.push((edge.edge.kind.as_str(), edge.edge.to.as_str()));
        }
        if edge.edge.to == node.node.id {
            incoming.push((edge.edge.kind.as_str(), edge.edge.from.as_str()));
        }
    }
    outgoing.sort_unstable();
    incoming.sort_unstable();

    let outgoing_hint = outgoing
        .iter()
        .take(2)
        .map(|(kind, target)| format!("{kind}->{target}"))
        .collect::<Vec<_>>();
    let incoming_hint = incoming
        .iter()
        .take(2)
        .map(|(kind, source)| format!("{source}-{kind}"))
        .collect::<Vec<_>>();
    let mut continuity_tags = Vec::new();
    if looks_eventful(&node.node) {
        continuity_tags.push("recent-event");
    }
    if looks_emotional(&node.node) {
        continuity_tags.push("emotion");
    }
    if continuity_tags.is_empty() && node.min_depth <= 1 {
        continuity_tags.push("narrative-anchor");
    }

    if verbatim {
        return format!(
            "- [NodeVerbatim] {} ({}) depth={} outgoing=[{}] incoming=[{}] continuity=[{}]",
            node.node.label,
            node.node.id,
            node.min_depth,
            outgoing_hint.join(", "),
            incoming_hint.join(", "),
            continuity_tags.join(", "),
        );
    }

    format!(
        "- [NodeSummary] {} ({}) depth={} links(out=[{}], in=[{}]) continuity=[{}]",
        node.node.label,
        node.node.id,
        node.min_depth,
        outgoing_hint.join(", "),
        incoming_hint.join(", "),
        continuity_tags.join(", "),
    )
}

fn looks_eventful(node: &GraphNodeRef) -> bool {
    let lowered = format!("{} {}", node.id, node.label).to_ascii_lowercase();
    [
        "event",
        "episode",
        "turn",
        "message",
        "recent",
        "today",
        "yesterday",
    ]
    .iter()
    .any(|needle| lowered.contains(needle))
}

fn looks_emotional(node: &GraphNodeRef) -> bool {
    let lowered = format!("{} {}", node.id, node.label).to_ascii_lowercase();
    [
        "emotion", "mood", "feeling", "feel", "happy", "sad", "angry", "afraid", "anxious", "calm",
    ]
    .iter()
    .any(|needle| lowered.contains(needle))
}

fn append_summary_line(
    lines: &mut Vec<String>,
    used_chars: &mut usize,
    max_chars: usize,
    line: String,
) -> bool {
    let separator_len = if lines.is_empty() { 0 } else { 1 };
    if used_chars
        .saturating_add(separator_len)
        .saturating_add(line.len())
        <= max_chars
    {
        *used_chars = used_chars
            .saturating_add(separator_len)
            .saturating_add(line.len());
        lines.push(line);
        true
    } else {
        false
    }
}

fn truncate_for_budget(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }
    if max_chars <= 1 {
        return "…".to_string();
    }
    let content_limit = max_chars.saturating_sub(1);
    let mut truncated = String::new();
    let mut char_count = 0usize;
    for ch in text.chars() {
        if char_count >= content_limit {
            break;
        }
        truncated.push(ch);
        char_count = char_count.saturating_add(1);
    }
    truncated.push('…');
    truncated
}

fn effective_graph_summary_char_budget(config: &GraphNeighborhoodSummaryConfig) -> usize {
    let token_budget_chars = config
        .max_tokens
        .map(|tokens| tokens.saturating_mul(config.chars_per_token.max(1)));
    token_budget_chars
        .map(|token_chars| token_chars.min(config.max_chars))
        .unwrap_or(config.max_chars)
        .max(1)
}

fn estimate_tokens(chars: usize, chars_per_token: usize) -> usize {
    let chars_per_token = chars_per_token.max(1);
    chars
        .saturating_add(chars_per_token.saturating_sub(1))
        .saturating_div(chars_per_token)
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

/// Fraction of salience retained by unmentioned topics each turn.
const TOPIC_DECAY_PER_TURN: f32 = 0.82;
/// Baseline boost applied whenever a topic is mentioned in the current turn.
const TOPIC_REINFORCEMENT_BASE: f32 = 0.35;
/// Additional boost scaled by mention relevance (0..1).
const TOPIC_REINFORCEMENT_RELEVANCE_WEIGHT: f32 = 0.65;
/// Drop topics once salience fades below this floor (unless pinned).
const TOPIC_MIN_SALIENCE: f32 = 0.10;
/// Upper bound to prevent runaway salience growth; with ~1.0 max boost/turn,
/// this caps reinforcement to a handful of strongly repeated turns.
const TOPIC_MAX_SALIENCE: f32 = 4.0;
/// Pinned topics keep at least this salience and do not decay.
const PINNED_TOPIC_SALIENCE_FLOOR: f32 = 0.75;

#[derive(Debug, Clone)]
struct TopicActivationState {
    node: GraphNodeRef,
    salience: f32,
    last_activated_turn: u64,
}

#[derive(Debug, Clone)]
struct GraphNodeFieldOverlay {
    label: Option<String>,
    fields: Map<String, Value>,
    reason: String,
    relevance: f32,
}

#[derive(Debug, Default)]
struct ActiveTopicTracker {
    current_turn: u64,
    last_turn_fingerprint: Option<u64>,
    topics: HashMap<GraphNodeId, TopicActivationState>,
}

#[derive(Clone)]
pub struct EmbeddingRecallProvider {
    self_node: GraphNodeRef,
    recall: Option<Arc<dyn EmbeddingRecall>>,
    recall_limit: usize,
    min_score: Option<f32>,
    conversation_tail_limit: usize,
    entity_extractor: Option<Arc<dyn EntityExtractor>>,
    pinned_nodes: Arc<Mutex<Vec<PinnedContextNode>>>,
    active_topics: Arc<Mutex<ActiveTopicTracker>>,
    graph_node_fields: Arc<Mutex<HashMap<GraphNodeId, GraphNodeFieldOverlay>>>,
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
            pinned_nodes: Arc::new(Mutex::new(Vec::new())),
            active_topics: Arc::new(Mutex::new(ActiveTopicTracker::default())),
            graph_node_fields: Arc::new(Mutex::new(HashMap::new())),
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

    pub fn recall_text(
        &self,
        text: impl Into<String>,
        limit: Option<usize>,
        min_score: Option<f32>,
    ) -> anyhow::Result<Vec<RecallHit>> {
        let Some(recall) = self.recall.as_ref() else {
            return Ok(Vec::new());
        };
        recall.recall(RecallQuery {
            text: text.into(),
            limit: limit.unwrap_or(self.recall_limit).max(1),
            min_score: min_score.or(self.min_score),
        })
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

    pub fn with_pinned_node(self, pinned: PinnedContextNode) -> Self {
        self.pin_node(pinned);
        self
    }

    pub fn pin_node(&self, pinned: PinnedContextNode) {
        if let Ok(mut pins) = self.pinned_nodes.lock() {
            pins.retain(|existing| existing.node_id != pinned.node_id);
            pins.push(pinned);
        }
    }

    pub fn unpin_node(&self, node_id: &str) -> bool {
        if let Ok(mut pins) = self.pinned_nodes.lock() {
            let original_len = pins.len();
            pins.retain(|pin| pin.node_id != node_id);
            return pins.len() != original_len;
        }
        false
    }

    pub fn pinned_nodes_snapshot(&self) -> Vec<PinnedContextNode> {
        self.pinned_nodes
            .lock()
            .map(|pins| pins.clone())
            .unwrap_or_default()
    }

    pub fn update_graph_node_fields(&self, update: GraphNodeFieldUpdate) {
        let Ok(mut fields_by_node) = self.graph_node_fields.lock() else {
            return;
        };
        fields_by_node
            .entry(update.node_id)
            .and_modify(|overlay| {
                if let Some(label) = update.label.as_ref() {
                    overlay.label = Some(label.clone());
                }
                overlay.fields.extend(update.fields.clone());
                overlay.reason = update.reason.clone();
                overlay.relevance = update.relevance;
            })
            .or_insert_with(|| GraphNodeFieldOverlay {
                label: update.label,
                fields: update.fields,
                reason: update.reason,
                relevance: update.relevance,
            });
    }

    pub fn graph_node_fields_snapshot(&self, node_id: &str) -> Option<Map<String, Value>> {
        self.graph_node_fields
            .lock()
            .ok()
            .and_then(|fields| fields.get(node_id).map(|overlay| overlay.fields.clone()))
    }

    pub fn search_graph_nodes(&self, query: GraphNodeSearchQuery) -> Vec<GraphNodeSearchHit> {
        let text = query
            .text
            .as_deref()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(|text| text.to_ascii_lowercase());
        let field = query
            .field
            .as_deref()
            .map(str::trim)
            .filter(|field| !field.is_empty());
        if text.is_none() && field.is_none() && query.value.is_none() {
            return Vec::new();
        }

        let Ok(fields_by_node) = self.graph_node_fields.lock() else {
            return Vec::new();
        };
        let mut hits = fields_by_node
            .iter()
            .filter_map(|(node_id, overlay)| {
                graph_node_search_hit(
                    node_id,
                    overlay,
                    text.as_deref(),
                    field,
                    query.value.as_ref(),
                )
            })
            .collect::<Vec<_>>();
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.node.id.cmp(&right.node.id))
        });
        hits.truncate(query.limit.max(1));
        hits
    }

    fn field_overlay(&self, node_id: &str) -> Option<GraphNodeFieldOverlay> {
        self.graph_node_fields
            .lock()
            .ok()
            .and_then(|fields| fields.get(node_id).cloned())
    }

    fn overlay_node_ref(&self, node: GraphNodeRef) -> GraphNodeRef {
        let Some(overlay) = self.field_overlay(&node.id) else {
            return node;
        };
        GraphNodeRef {
            label: overlay.label.unwrap_or(node.label),
            ..node
        }
    }

    fn apply_field_overlays(&self, nodes: &mut [ContextNode]) {
        for node in nodes {
            let Some(overlay) = self.field_overlay(&node.node.id) else {
                continue;
            };
            if let Some(label) = overlay.label {
                node.node.label = label;
            }
            let fields = summarize_graph_node_fields(&overlay.fields);
            if !fields.is_empty() {
                if node.summary.trim().is_empty() {
                    node.summary = fields;
                } else if !node.summary.contains(&fields) {
                    node.summary = format!("{}; fields: {}", node.summary.trim(), fields);
                }
            }
            node.reason = format!("{}; {}", node.reason.trim(), overlay.reason.trim());
            node.relevance = node.relevance.max(overlay.relevance);
        }
    }

    fn active_topic_nodes_snapshot(&self) -> Vec<ContextNode> {
        self.active_topics
            .lock()
            .map(|tracker| {
                let mut topics = tracker
                    .topics
                    .values()
                    .map(|state| {
                        let mut node = ContextNode {
                            node: state.node.clone(),
                            role: ContextNodeRole::ActiveTopic,
                            relevance: state.salience,
                            reason: format!(
                                "active-topic salience {:.2} (last activated turn {})",
                                state.salience, state.last_activated_turn
                            ),
                            summary: "Conversationally salient topic".to_string(),
                        };
                        self.apply_field_overlays(std::slice::from_mut(&mut node));
                        node
                    })
                    .collect::<Vec<_>>();
                topics.sort_by(|left, right| right.relevance.total_cmp(&left.relevance));
                topics
            })
            .unwrap_or_default()
    }

    fn active_topics_snapshot(&self) -> Vec<TopicActivation> {
        self.active_topics
            .lock()
            .map(|tracker| {
                let mut topics = tracker
                    .topics
                    .values()
                    .map(|state| TopicActivation {
                        node_id: state.node.id.clone(),
                        salience: state.salience,
                        last_activated_turn: state.last_activated_turn,
                    })
                    .collect::<Vec<_>>();
                topics.sort_by(|left, right| right.salience.total_cmp(&left.salience));
                topics
            })
            .unwrap_or_default()
    }

    fn topic_reinforcement_boost(relevance: f32) -> f32 {
        TOPIC_REINFORCEMENT_BASE + relevance.clamp(0.0, 1.0) * TOPIC_REINFORCEMENT_RELEVANCE_WEIGHT
    }

    fn update_active_topics(
        &self,
        selected_nodes: &[ContextNode],
        pinned_nodes: &[PinnedContextNode],
        turn_fingerprint: u64,
    ) {
        let Ok(mut tracker) = self.active_topics.lock() else {
            return;
        };
        if tracker.last_turn_fingerprint != Some(turn_fingerprint) {
            tracker.current_turn = tracker.current_turn.saturating_add(1);
            tracker.last_turn_fingerprint = Some(turn_fingerprint);
        }
        let current_turn = tracker.current_turn;

        let pinned_ids = pinned_nodes
            .iter()
            .map(|pin| pin.node_id.as_str())
            .collect::<HashSet<_>>();

        for (node_id, state) in &mut tracker.topics {
            if pinned_ids.contains(node_id.as_str()) {
                state.salience = state.salience.max(PINNED_TOPIC_SALIENCE_FLOOR);
            } else {
                state.salience *= TOPIC_DECAY_PER_TURN;
            }
        }

        let mut seen_node_ids = HashSet::new();
        for node in selected_nodes {
            if !seen_node_ids.insert(node.node.id.as_str()) {
                continue;
            }
            let entry = tracker
                .topics
                .entry(node.node.id.clone())
                .or_insert_with(|| TopicActivationState {
                    node: node.node.clone(),
                    salience: 0.0,
                    last_activated_turn: current_turn,
                });
            entry.node = node.node.clone();
            let boost = Self::topic_reinforcement_boost(node.relevance);
            entry.salience = (entry.salience + boost).min(TOPIC_MAX_SALIENCE);
            entry.last_activated_turn = current_turn;
        }

        for pinned in pinned_nodes {
            let entry = tracker
                .topics
                .entry(pinned.node_id.clone())
                .or_insert_with(|| TopicActivationState {
                    node: self.overlay_node_ref(GraphNodeRef {
                        id: pinned.node_id.clone(),
                        label: pinned.node_id.clone(),
                    }),
                    salience: PINNED_TOPIC_SALIENCE_FLOOR,
                    last_activated_turn: current_turn,
                });
            entry.salience = entry.salience.max(PINNED_TOPIC_SALIENCE_FLOOR);
        }

        tracker.topics.retain(|node_id, state| {
            pinned_ids.contains(node_id.as_str()) || state.salience >= TOPIC_MIN_SALIENCE
        });
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

    fn pinned_nodes(&self) -> Vec<PinnedContextNode> {
        Vec::new()
    }

    fn active_topics(&self) -> Vec<TopicActivation> {
        Vec::new()
    }

    fn advance_pins(&self) {}
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
            // `EmbeddingRecallProvider` does not have direct access to a
            // knowledge graph, so provisional nodes (deterministic IDs derived
            // from the entity surface form) are used here.  Callers that have a
            // graph can resolve entities against it by calling `resolve_entities`
            // with a real lookup closure and merging the result themselves.
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
                                summary: hit.summary.unwrap_or_else(|| {
                                    "Retrieved from embedding recall".to_string()
                                }),
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

        self.apply_field_overlays(&mut selected);
        let pinned_nodes = self.pinned_nodes_snapshot();
        let turn_fingerprint = input_turn_fingerprint(utterance, conversation_tail);
        self.update_active_topics(&selected, &pinned_nodes, turn_fingerprint);
        let active_topic_nodes = self.active_topic_nodes_snapshot();
        for active_node in active_topic_nodes {
            if let Some(existing) = selected
                .iter_mut()
                .find(|existing| existing.node.id == active_node.node.id)
            {
                if active_node.relevance > existing.relevance {
                    existing.relevance = active_node.relevance;
                }
            } else {
                selected.push(active_node);
            }
        }

        selected
    }

    fn pinned_nodes(&self) -> Vec<PinnedContextNode> {
        self.pinned_nodes
            .lock()
            .map(|pins| {
                pins.iter()
                    .filter(|pin| !matches!(pin.scope, PinScope::Temporary { remaining_turns: 0 }))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    fn advance_pins(&self) {
        if let Ok(mut pins) = self.pinned_nodes.lock() {
            for pin in pins.iter_mut() {
                if let PinScope::Temporary { remaining_turns } = &mut pin.scope
                    && *remaining_turns > 0
                {
                    *remaining_turns -= 1;
                }
            }
            pins.retain(|pin| !matches!(pin.scope, PinScope::Temporary { remaining_turns: 0 }));
        }
    }

    fn active_topics(&self) -> Vec<TopicActivation> {
        self.active_topics_snapshot()
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

fn summarize_graph_node_fields(fields: &Map<String, Value>) -> String {
    let mut pairs = fields
        .iter()
        .filter_map(|(key, value)| {
            if value.is_null() {
                return None;
            }
            Some(format!("{}={}", key, compact_json_value(value)))
        })
        .collect::<Vec<_>>();
    pairs.sort();
    pairs.join(", ")
}

fn graph_node_search_hit(
    node_id: &str,
    overlay: &GraphNodeFieldOverlay,
    text: Option<&str>,
    field: Option<&str>,
    value: Option<&Value>,
) -> Option<GraphNodeSearchHit> {
    let mut score = 0.0_f32;
    let mut reasons = Vec::new();

    if let Some(field) = field {
        let Some(field_value) = overlay.fields.get(field) else {
            return None;
        };
        score += 1.0;
        reasons.push(format!("field {field} exists"));
        if let Some(expected) = value {
            if !json_values_match(field_value, expected) {
                return None;
            }
            score += 1.0;
            reasons.push(format!(
                "field {field} matched value {}",
                compact_json_value(expected)
            ));
        }
    } else if let Some(expected) = value {
        if !overlay
            .fields
            .values()
            .any(|field_value| json_values_match(field_value, expected))
        {
            return None;
        }
        score += 1.0;
        reasons.push(format!(
            "some field matched value {}",
            compact_json_value(expected)
        ));
    }

    if let Some(text) = text {
        let searchable = graph_node_search_text(node_id, overlay);
        if !searchable.contains(text) {
            return None;
        }
        score += 1.0;
        reasons.push(format!("text matched {text}"));
    }

    if score <= 0.0 {
        return None;
    }

    Some(GraphNodeSearchHit {
        node: GraphNodeRef {
            id: node_id.to_string(),
            label: overlay.label.clone().unwrap_or_else(|| node_id.to_string()),
        },
        score,
        fields: overlay.fields.clone(),
        reason: reasons.join("; "),
    })
}

fn graph_node_search_text(node_id: &str, overlay: &GraphNodeFieldOverlay) -> String {
    let mut parts = vec![node_id.to_ascii_lowercase()];
    if let Some(label) = &overlay.label {
        parts.push(label.to_ascii_lowercase());
    }
    for (key, value) in &overlay.fields {
        parts.push(key.to_ascii_lowercase());
        parts.push(compact_json_value(value).to_ascii_lowercase());
    }
    parts.join(" ")
}

fn json_values_match(candidate: &Value, expected: &Value) -> bool {
    if candidate == expected {
        return true;
    }
    match (candidate, expected) {
        (Value::String(candidate), Value::String(expected)) => {
            candidate.eq_ignore_ascii_case(expected)
        }
        (Value::String(candidate), _) => {
            candidate.eq_ignore_ascii_case(&compact_json_value(expected))
        }
        (_, Value::String(expected)) => {
            compact_json_value(candidate).eq_ignore_ascii_case(expected)
        }
        _ => false,
    }
}

fn compact_json_value(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::Array(values) => values
            .iter()
            .map(compact_json_value)
            .collect::<Vec<_>>()
            .join("|"),
        Value::Object(_) => serde_json::to_string(value).unwrap_or_else(|_| "<object>".to_string()),
        Value::Null => "null".to_string(),
    }
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

fn input_turn_fingerprint(utterance: &str, conversation_tail: &[ConversationTurn]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    utterance.trim().hash(&mut hasher);
    for turn in conversation_tail {
        turn.role.label().hash(&mut hasher);
        turn.text.trim().hash(&mut hasher);
    }
    hasher.finish()
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
    let mut selected_nodes = provider.selected_nodes(utterance, &conversation_tail, &budget);
    let pinned_nodes = provider.pinned_nodes();
    let active_topics = provider.active_topics();
    if !pinned_nodes.is_empty() {
        let pinned_ids = pinned_nodes
            .iter()
            .map(|node| node.node_id.as_str())
            .collect::<HashSet<_>>();
        selected_nodes.retain(|node| !pinned_ids.contains(node.node.id.as_str()));
    }
    provider.advance_pins();
    ConversationContext {
        system_prompt: system_prompt.into(),
        self_node,
        pinned_nodes,
        active_topics,
        selected_nodes,
        conversation_tail,
        budget,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::ops::Range;
    use std::sync::Arc;

    use serde_json::json;

    use super::*;
    use crate::memory::{EmbeddingProvider, QdrantPoint, QdrantSearchHit, QdrantStore};
    use crate::mind::entity::{EntityKind, ExtractedEntity};

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
        assert!(
            context
                .render_compact_nodes()
                .contains("ASR transcribes that speech")
        );
        assert!(
            context
                .render_compact_nodes()
                .contains("retrieved memories")
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
            pinned_nodes: Vec::new(),
            active_topics: Vec::new(),
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
            budget: ContextBudget { max_chars: 820 },
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
            pinned_nodes: Vec::new(),
            active_topics: Vec::new(),
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
            pinned_nodes: Vec::new(),
            active_topics: Vec::new(),
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

    #[test]
    fn neighborhood_summary_respects_budget() {
        let graph = ExpandedContextGraph {
            nodes: vec![
                ExpandedNode {
                    node: GraphNodeRef {
                        id: "topic:seed".to_string(),
                        label: "Seed topic".to_string(),
                    },
                    min_depth: 0,
                    provenance: vec![TraversalProvenance {
                        seed: GraphNodeRef {
                            id: "topic:seed".to_string(),
                            label: "Seed topic".to_string(),
                        },
                        depth: 0,
                        path: Vec::new(),
                    }],
                },
                ExpandedNode {
                    node: GraphNodeRef {
                        id: "event:recent".to_string(),
                        label: "Recent event".to_string(),
                    },
                    min_depth: 1,
                    provenance: vec![TraversalProvenance {
                        seed: GraphNodeRef {
                            id: "topic:seed".to_string(),
                            label: "Seed topic".to_string(),
                        },
                        depth: 1,
                        path: Vec::new(),
                    }],
                },
                ExpandedNode {
                    node: GraphNodeRef {
                        id: "emotion:calm".to_string(),
                        label: "Calm mood".to_string(),
                    },
                    min_depth: 1,
                    provenance: vec![TraversalProvenance {
                        seed: GraphNodeRef {
                            id: "topic:seed".to_string(),
                            label: "Seed topic".to_string(),
                        },
                        depth: 1,
                        path: Vec::new(),
                    }],
                },
            ],
            edges: vec![
                ExpandedEdge {
                    edge: GraphTraversalEdge {
                        from: "topic:seed".to_string(),
                        to: "event:recent".to_string(),
                        kind: "ABOUT".to_string(),
                    },
                    min_depth: 1,
                    provenance: Vec::new(),
                },
                ExpandedEdge {
                    edge: GraphTraversalEdge {
                        from: "event:recent".to_string(),
                        to: "emotion:calm".to_string(),
                        kind: "AFFECTS".to_string(),
                    },
                    min_depth: 2,
                    provenance: Vec::new(),
                },
            ],
        };

        let config = GraphNeighborhoodSummaryConfig {
            max_chars: 150,
            max_tokens: Some(25),
            chars_per_token: 4,
            verbatim_node_ids: Vec::new(),
            max_edge_lines: 8,
        };
        let expected_budget = effective_graph_summary_char_budget(&config);
        let summary = graph.summarize_neighborhood(config);

        assert!(summary.rendered.len() <= expected_budget);
        assert!(summary.stats.rendered_chars <= summary.stats.budget_chars);
        assert!(
            summary
                .debug_view()
                .contains("original_graph=nodes:3 edges:2")
        );
    }

    #[test]
    fn neighborhood_summary_keeps_key_nodes_verbatim() {
        let graph = ExpandedContextGraph {
            nodes: vec![
                ExpandedNode {
                    node: GraphNodeRef {
                        id: "topic:aurora".to_string(),
                        label: "Aurora".to_string(),
                    },
                    min_depth: 0,
                    provenance: vec![TraversalProvenance {
                        seed: GraphNodeRef {
                            id: "topic:aurora".to_string(),
                            label: "Aurora".to_string(),
                        },
                        depth: 0,
                        path: Vec::new(),
                    }],
                },
                ExpandedNode {
                    node: GraphNodeRef {
                        id: "event:flight".to_string(),
                        label: "Flight event".to_string(),
                    },
                    min_depth: 1,
                    provenance: Vec::new(),
                },
            ],
            edges: vec![ExpandedEdge {
                edge: GraphTraversalEdge {
                    from: "topic:aurora".to_string(),
                    to: "event:flight".to_string(),
                    kind: "ABOUT".to_string(),
                },
                min_depth: 1,
                provenance: Vec::new(),
            }],
        };

        let summary = graph.summarize_neighborhood(GraphNeighborhoodSummaryConfig {
            max_chars: 320,
            max_tokens: None,
            chars_per_token: 4,
            verbatim_node_ids: vec!["topic:aurora".to_string()],
            max_edge_lines: 8,
        });

        assert!(
            summary
                .rendered
                .contains("[NodeVerbatim] Aurora (topic:aurora)")
        );
        assert!(
            summary
                .kept_node_ids
                .iter()
                .any(|node_id| node_id == "topic:aurora")
        );
        assert!(
            summary
                .rendered
                .contains("[Edge] topic:aurora -ABOUT-> event:flight")
        );
    }

    #[test]
    fn pinned_nodes_are_rendered_and_debugged_separately() {
        let context = ConversationContext {
            system_prompt: "system".to_string(),
            self_node: GraphNodeRef {
                id: DEFAULT_SELF_NODE_ID.to_string(),
                label: DEFAULT_SELF_NODE_LABEL.to_string(),
            },
            pinned_nodes: vec![PinnedContextNode {
                node_id: "topic:mission".to_string(),
                scope: PinScope::Session,
                reason: "active mission".to_string(),
            }],
            active_topics: Vec::new(),
            selected_nodes: vec![ContextNode {
                node: GraphNodeRef {
                    id: "memory:old".to_string(),
                    label: "Old memory".to_string(),
                },
                role: ContextNodeRole::RetrievedMemory,
                relevance: 0.1,
                reason: "older mention".to_string(),
                summary: "Low-priority memory node that should be omitted.".to_string(),
            }],
            conversation_tail: Vec::new(),
            budget: ContextBudget { max_chars: 180 },
        };

        let rendered = context.render_compact_nodes();
        assert!(rendered.contains("[Pinned:Session] topic:mission"));
        assert!(
            context
                .debug_nodes()
                .contains("pinned=[Session:topic:mission(active mission)]")
        );
    }

    #[test]
    fn session_pins_survive_unrelated_turns_and_temporary_pins_expire() {
        let provider = EmbeddingRecallProvider::new(GraphNodeRef {
            id: DEFAULT_SELF_NODE_ID.to_string(),
            label: DEFAULT_SELF_NODE_LABEL.to_string(),
        });
        provider.pin_node(PinnedContextNode {
            node_id: "topic:project-aurora".to_string(),
            scope: PinScope::Session,
            reason: "ongoing project".to_string(),
        });
        provider.pin_node(PinnedContextNode {
            node_id: "topic:mood".to_string(),
            scope: PinScope::Temporary { remaining_turns: 1 },
            reason: "temporary emotional theme".to_string(),
        });

        let first = build_conversation_context(
            &provider,
            "system",
            "What's the weather tomorrow?",
            vec![ConversationTurn {
                role: ConversationRole::User,
                text: "Tell me a joke".to_string(),
            }],
            ContextBudget::default(),
        );
        assert!(
            first
                .pinned_nodes
                .iter()
                .any(|node| node.node_id == "topic:project-aurora")
        );
        assert!(
            first
                .pinned_nodes
                .iter()
                .any(|node| node.node_id == "topic:mood")
        );

        let second = build_conversation_context(
            &provider,
            "system",
            "Do you know any spaceship trivia?",
            vec![ConversationTurn {
                role: ConversationRole::User,
                text: "Different subject now".to_string(),
            }],
            ContextBudget::default(),
        );
        assert!(
            second
                .pinned_nodes
                .iter()
                .any(|node| node.node_id == "topic:project-aurora")
        );
        assert!(
            !second
                .pinned_nodes
                .iter()
                .any(|node| node.node_id == "topic:mood")
        );
    }

    #[test]
    fn graph_node_field_updates_overlay_extracted_context_nodes() {
        let provider = EmbeddingRecallProvider::new(GraphNodeRef {
            id: DEFAULT_SELF_NODE_ID.to_string(),
            label: DEFAULT_SELF_NODE_LABEL.to_string(),
        })
        .with_entity_extractor(Arc::new(KeywordTopicExtractor::new(
            "aurora",
            "Project Aurora",
        )));
        provider.update_graph_node_fields(GraphNodeFieldUpdate {
            node_id: "topic:project_aurora".to_string(),
            label: Some("Project Aurora".to_string()),
            fields: serde_json::Map::from_iter([
                ("status".to_string(), Value::String("blocked".to_string())),
                ("owner".to_string(), Value::String("Travis".to_string())),
            ]),
            reason: "Pete updated graph node fields on turn 7".to_string(),
            relevance: 1.0,
        });

        let context = build_conversation_context(
            &provider,
            "system",
            "Tell me about Project Aurora",
            Vec::new(),
            ContextBudget::default(),
        );
        let rendered = context.render_compact_nodes();

        assert!(rendered.contains("Project Aurora"));
        assert!(rendered.contains("owner=Travis"));
        assert!(rendered.contains("status=blocked"));
    }

    #[test]
    fn graph_node_search_matches_field_value_and_text() {
        let provider = EmbeddingRecallProvider::new(GraphNodeRef {
            id: DEFAULT_SELF_NODE_ID.to_string(),
            label: DEFAULT_SELF_NODE_LABEL.to_string(),
        });
        provider.update_graph_node_fields(GraphNodeFieldUpdate {
            node_id: "person:travis".to_string(),
            label: Some("Travis".to_string()),
            fields: serde_json::Map::from_iter([
                (
                    "timezone".to_string(),
                    Value::String("America/Los_Angeles".to_string()),
                ),
                (
                    "favorite_city".to_string(),
                    Value::String("Seattle".to_string()),
                ),
            ]),
            reason: "test fields".to_string(),
            relevance: 0.9,
        });

        let field_hits = provider.search_graph_nodes(GraphNodeSearchQuery {
            text: None,
            field: Some("timezone".to_string()),
            value: Some(Value::String("america/los_angeles".to_string())),
            limit: 8,
        });
        let text_hits = provider.search_graph_nodes(GraphNodeSearchQuery {
            text: Some("seattle".to_string()),
            field: None,
            value: None,
            limit: 8,
        });

        assert_eq!(field_hits.len(), 1);
        assert_eq!(field_hits[0].node.id, "person:travis");
        assert_eq!(text_hits.len(), 1);
        assert_eq!(text_hits[0].node.label, "Travis");
    }

    #[test]
    fn repeated_mentions_reinforce_topic_salience() {
        let provider = EmbeddingRecallProvider::new(GraphNodeRef {
            id: DEFAULT_SELF_NODE_ID.to_string(),
            label: DEFAULT_SELF_NODE_LABEL.to_string(),
        })
        .with_entity_extractor(Arc::new(KeywordTopicExtractor::new("aurora", "Aurora")));

        let first = build_conversation_context(
            &provider,
            "system",
            "Let's discuss aurora today.",
            Vec::new(),
            ContextBudget::default(),
        );
        let first_salience = first
            .active_topics
            .iter()
            .find(|topic| topic.node_id == "topic:aurora")
            .map(|topic| topic.salience)
            .expect("topic should be active after first mention");

        let second = build_conversation_context(
            &provider,
            "system",
            "aurora details again please",
            Vec::new(),
            ContextBudget::default(),
        );
        let second_salience = second
            .active_topics
            .iter()
            .find(|topic| topic.node_id == "topic:aurora")
            .map(|topic| topic.salience)
            .expect("topic should remain active");

        assert!(second_salience > first_salience);
        assert!(second.debug_nodes().contains("active_topics=[topic:aurora"));
    }

    #[test]
    fn unmentioned_topics_decay_gradually_across_turns() {
        let provider = EmbeddingRecallProvider::new(GraphNodeRef {
            id: DEFAULT_SELF_NODE_ID.to_string(),
            label: DEFAULT_SELF_NODE_LABEL.to_string(),
        })
        .with_entity_extractor(Arc::new(KeywordTopicExtractor::new("aurora", "Aurora")));

        let first = build_conversation_context(
            &provider,
            "system",
            "aurora mission planning",
            Vec::new(),
            ContextBudget::default(),
        );
        let first_salience = first
            .active_topics
            .iter()
            .find(|topic| topic.node_id == "topic:aurora")
            .map(|topic| topic.salience)
            .expect("topic should be active after mention");

        let second = build_conversation_context(
            &provider,
            "system",
            "Let's switch to weather.",
            Vec::new(),
            ContextBudget::default(),
        );
        let second_salience = second
            .active_topics
            .iter()
            .find(|topic| topic.node_id == "topic:aurora")
            .map(|topic| topic.salience)
            .expect("topic should not disappear immediately");

        assert!(second_salience < first_salience);
        assert!(second_salience > 0.0);
    }

    #[test]
    fn pinned_topics_bypass_decay() {
        let provider = EmbeddingRecallProvider::new(GraphNodeRef {
            id: DEFAULT_SELF_NODE_ID.to_string(),
            label: DEFAULT_SELF_NODE_LABEL.to_string(),
        })
        .with_entity_extractor(Arc::new(KeywordTopicExtractor::new("aurora", "Aurora")));
        provider.pin_node(PinnedContextNode {
            node_id: "topic:aurora".to_string(),
            scope: PinScope::Session,
            reason: "keep mission topic active".to_string(),
        });

        let first = build_conversation_context(
            &provider,
            "system",
            "aurora mission planning",
            Vec::new(),
            ContextBudget::default(),
        );
        let first_salience = first
            .active_topics
            .iter()
            .find(|topic| topic.node_id == "topic:aurora")
            .map(|topic| topic.salience)
            .expect("topic should be active after mention");

        let second = build_conversation_context(
            &provider,
            "system",
            "unrelated subject",
            Vec::new(),
            ContextBudget::default(),
        );
        let second_salience = second
            .active_topics
            .iter()
            .find(|topic| topic.node_id == "topic:aurora")
            .map(|topic| topic.salience)
            .expect("pinned topic should stay active");

        assert!(second_salience >= first_salience);
    }

    #[derive(Clone)]
    struct StaticQdrantStore {
        hits: Vec<QdrantSearchHit>,
    }

    struct KeywordTopicExtractor {
        keyword: String,
        label: String,
    }

    impl KeywordTopicExtractor {
        fn new(keyword: &str, label: &str) -> Self {
            Self {
                keyword: keyword.to_ascii_lowercase(),
                label: label.to_string(),
            }
        }
    }

    impl EntityExtractor for KeywordTopicExtractor {
        fn extract(&self, text: &str) -> Vec<ExtractedEntity> {
            let lowered = text.to_ascii_lowercase();
            let Some(start) = lowered.find(&self.keyword) else {
                return Vec::new();
            };
            let Some(end) = start.checked_add(self.keyword.len()) else {
                return Vec::new();
            };
            let span: Range<usize> = start..end;
            vec![ExtractedEntity {
                text: self.label.clone(),
                span,
                kind: EntityKind::Topic,
                confidence: 0.8,
            }]
        }
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
