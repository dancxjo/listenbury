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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StubContextProvider {
    self_node: GraphNodeRef,
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
    use super::*;

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
}
