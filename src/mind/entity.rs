use std::collections::HashSet;
use std::ops::Range;

use crate::mind::context::{ContextNode, ContextNodeRole, GraphNodeRef};

/// The semantic category of an extracted entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityKind {
    Person,
    Place,
    Topic,
    Organization,
    Object,
    Task,
}

impl EntityKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EntityKind::Person => "person",
            EntityKind::Place => "place",
            EntityKind::Topic => "topic",
            EntityKind::Organization => "org",
            EntityKind::Object => "object",
            EntityKind::Task => "task",
        }
    }

    /// Map this entity kind to the closest `ContextNodeRole`.
    pub fn to_context_node_role(self) -> ContextNodeRole {
        match self {
            EntityKind::Person => ContextNodeRole::RecentMention,
            EntityKind::Place => ContextNodeRole::Place,
            EntityKind::Topic => ContextNodeRole::ActiveTopic,
            EntityKind::Organization => ContextNodeRole::Organization,
            EntityKind::Object => ContextNodeRole::Object,
            EntityKind::Task => ContextNodeRole::Task,
        }
    }
}

/// A candidate entity extracted from a user utterance.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedEntity {
    /// The surface text of the entity as it appears in the utterance.
    pub text: String,
    /// Byte span of the entity within the source utterance.
    pub span: Range<usize>,
    /// Semantic category.
    pub kind: EntityKind,
    /// Confidence score in \[0, 1\].
    pub confidence: f32,
}

impl ExtractedEntity {
    /// Returns a deterministic graph node ID for this entity.
    ///
    /// Uses the pattern `kind:normalized_label`, e.g. `person:sylvia`.
    /// Repeated references to the same surface form always produce the same ID,
    /// enabling stable graph anchoring across turns.
    pub fn provisional_node_id(&self) -> String {
        let normalized = self
            .text
            .split_whitespace()
            .map(|w| {
                w.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_ascii_lowercase()
            })
            .filter(|w| !w.is_empty())
            .collect::<Vec<_>>()
            .join("_");
        format!("{}:{}", self.kind.as_str(), normalized)
    }

    /// Builds a provisional `GraphNodeRef` for this entity.
    ///
    /// Used when no existing graph node matches the entity.
    pub fn provisional_node_ref(&self) -> GraphNodeRef {
        GraphNodeRef {
            id: self.provisional_node_id(),
            label: self.text.clone(),
        }
    }

    /// Converts this entity into a `ContextNode` using the provided (or provisional) node.
    pub fn to_context_node(&self, node: GraphNodeRef) -> ContextNode {
        ContextNode {
            node,
            role: self.kind.to_context_node_role(),
            relevance: self.confidence,
            reason: format!(
                "extracted entity ({}) from utterance span {}..{}",
                self.kind.as_str(),
                self.span.start,
                self.span.end,
            ),
            summary: format!("{} ({})", self.text, self.kind.as_str()),
        }
    }
}

/// Trait for extracting named entities from free text.
pub trait EntityExtractor: Send + Sync {
    fn extract(&self, text: &str) -> Vec<ExtractedEntity>;
}

/// Resolves a slice of extracted entities to `ContextNode`s.
///
/// For each entity the `lookup` closure is called with the entity's provisional
/// node ID.  When a matching node exists it is used; otherwise a new provisional
/// `GraphNodeRef` is created.  Entities that map to the same node ID are
/// deduplicated so that repeated references produce a single `ContextNode`.
pub fn resolve_entities(
    entities: &[ExtractedEntity],
    lookup: &dyn Fn(&str) -> Option<GraphNodeRef>,
) -> Vec<ContextNode> {
    let mut seen_ids: HashSet<String> = HashSet::new();
    let mut nodes = Vec::new();

    for entity in entities {
        let node_id = entity.provisional_node_id();
        if !seen_ids.insert(node_id.clone()) {
            continue;
        }
        let node = lookup(&node_id).unwrap_or_else(|| entity.provisional_node_ref());
        nodes.push(entity.to_context_node(node));
    }

    nodes
}

// ── Heuristic extractor ───────────────────────────────────────────────────

/// A lightweight heuristic entity extractor.
///
/// Identifies candidate entities using capitalization patterns, contextual
/// keyword cues, and a built-in place-name list.  Requires no external models
/// and is suitable as a default that can later be swapped for an LLM-assisted
/// implementation.
pub struct HeuristicEntityExtractor;

impl EntityExtractor for HeuristicEntityExtractor {
    fn extract(&self, text: &str) -> Vec<ExtractedEntity> {
        extract_heuristic(text)
    }
}

// ── Heuristic implementation details ─────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntityContext {
    Person,
    Place,
    Topic,
    Task,
    None,
}

/// Verbs of social interaction that strongly signal a following proper noun is a
/// Person.
const STRONG_PERSON_VERBS: &[&str] = &[
    "talked",
    "talk",
    "spoke",
    "speak",
    "met",
    "meet",
    "called",
    "call",
    "emailed",
    "email",
    "messaged",
    "message",
    "asked",
    "ask",
    "told",
    "tell",
    "mentioned",
    "mention",
    "introduced",
    "introduce",
    "hired",
    "hire",
    "invited",
    "invite",
    "knows",
    "know",
    "saw",
    "see",
    "heard",
    "hear",
    "contacted",
    "contact",
    "visited",
    "visit",
];

/// Prepositions/words that appear directly before a Person name.
const PERSON_PREPOSITIONS: &[&str] = &["with", "by", "for", "from"];

/// Prepositions that signal a following proper noun is a Place.
const PLACE_CUES: &[&str] = &[
    "in", "at", "near", "around", "outside", "inside", "within", "toward",
    "towards", "visiting",
];

/// Words that signal a following proper noun is a Topic or concept.
const TOPIC_CUES: &[&str] = &[
    "about", "regarding", "on", "concerning", "discussing", "discussed", "using",
    "via", "through", "like",
];

/// Words that signal a following proper noun is a Task or project.
const TASK_CUES: &[&str] = &[
    "fix", "create", "build", "implement", "write", "design", "review", "update",
    "migrate", "deploy", "refactor", "test", "debug", "task", "project", "issue",
    "ticket",
];

/// Words that are never entities on their own even when capitalised.
const COMMON_WORDS: &[&str] = &[
    "i",
    "a",
    "an",
    "the",
    "is",
    "it",
    "he",
    "she",
    "we",
    "they",
    "you",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "our",
    "their",
    "who",
    "what",
    "where",
    "when",
    "why",
    "how",
    "which",
    "ok",
    "yes",
    "no",
    "so",
    "but",
    "and",
    "or",
    "if",
    "to",
    "of",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

/// Well-known place names (lower-case).
const PLACE_NAMES: &[&str] = &[
    "seattle",
    "new york",
    "san francisco",
    "los angeles",
    "chicago",
    "boston",
    "austin",
    "denver",
    "portland",
    "miami",
    "dallas",
    "london",
    "paris",
    "berlin",
    "tokyo",
    "beijing",
    "shanghai",
    "sydney",
    "toronto",
    "montreal",
    "vancouver",
    "amsterdam",
    "barcelona",
    "madrid",
    "rome",
    "moscow",
    "dubai",
    "singapore",
    "america",
    "canada",
    "mexico",
    "england",
    "scotland",
    "france",
    "germany",
    "italy",
    "spain",
    "china",
    "japan",
    "india",
    "australia",
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "texas",
    "utah",
    "virginia",
    "washington",
    "wisconsin",
    "wyoming",
];

/// Common suffixes that indicate an Organization name.
const ORG_SUFFIXES: &[&str] = &[
    "inc",
    "corp",
    "llc",
    "ltd",
    "co",
    "company",
    "labs",
    "systems",
    "technologies",
    "solutions",
    "group",
    "foundation",
    "institute",
];

fn strip_punct(s: &str) -> &str {
    s.trim_matches(|c: char| !c.is_alphanumeric())
}

fn is_common_word(word: &str) -> bool {
    COMMON_WORDS.contains(&word.to_ascii_lowercase().as_str())
}

fn is_place_name(text: &str) -> bool {
    PLACE_NAMES.contains(&text.to_ascii_lowercase().as_str())
}

fn has_org_suffix(text: &str) -> bool {
    let last = text
        .split_whitespace()
        .next_back()
        .unwrap_or("")
        .to_ascii_lowercase();
    ORG_SUFFIXES.contains(&last.as_str())
}

fn is_acronym(text: &str) -> bool {
    text.len() >= 2 && text.chars().all(|c| c.is_uppercase() || c.is_ascii_digit())
}

fn starts_uppercase(s: &str) -> bool {
    s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
}

/// Determine the entity context from the words that appear before the candidate.
///
/// `words_before` is ordered left-to-right; we look at the *last* few words.
fn context_from_preceding(words_before: &[&str]) -> EntityContext {
    // Strong person verbs anywhere in the last 3 words win first.
    for &w in words_before.iter().rev().take(3) {
        let lower = w.to_ascii_lowercase();
        let alpha = lower.trim_matches(|c: char| !c.is_alphabetic());
        if STRONG_PERSON_VERBS.contains(&alpha) {
            return EntityContext::Person;
        }
    }

    // Now check the immediately preceding word.
    if let Some(&last) = words_before.last() {
        let lower = last.to_ascii_lowercase();
        let alpha = lower.trim_matches(|c: char| !c.is_alphabetic());
        if PERSON_PREPOSITIONS.contains(&alpha) {
            return EntityContext::Person;
        }
        if PLACE_CUES.contains(&alpha) {
            return EntityContext::Place;
        }
        if TOPIC_CUES.contains(&alpha) {
            return EntityContext::Topic;
        }
        if TASK_CUES.contains(&alpha) {
            return EntityContext::Task;
        }
    }

    EntityContext::None
}

struct WordToken<'a> {
    raw: &'a str,
    stripped: &'a str,
    byte_start: usize,
    byte_end: usize,
}

fn tokenize(text: &str) -> Vec<WordToken<'_>> {
    let mut tokens = Vec::new();
    let mut start: Option<usize> = None;

    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = start.take() {
                let raw = &text[s..i];
                tokens.push(WordToken {
                    raw,
                    stripped: strip_punct(raw),
                    byte_start: s,
                    byte_end: i,
                });
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        let raw = &text[s..];
        tokens.push(WordToken {
            raw,
            stripped: strip_punct(raw),
            byte_start: s,
            byte_end: text.len(),
        });
    }
    tokens
}

fn extract_heuristic(text: &str) -> Vec<ExtractedEntity> {
    let tokens = tokenize(text);
    if tokens.is_empty() {
        return Vec::new();
    }

    let mut entities = Vec::new();
    let mut i = 0;

    while i < tokens.len() {
        let stripped = tokens[i].stripped;

        if stripped.is_empty() || is_common_word(stripped) || !starts_uppercase(stripped) {
            i += 1;
            continue;
        }

        // Greedily collect consecutive capitalised tokens into a multi-word span.
        let byte_start = tokens[i].byte_start;
        let mut byte_end = tokens[i].byte_end;
        let mut span_words = vec![stripped.to_string()];
        let mut j = i + 1;

        while j < tokens.len() {
            let next = tokens[j].stripped;
            if !next.is_empty() && starts_uppercase(next) && !is_common_word(next) {
                span_words.push(next.to_string());
                byte_end = tokens[j].byte_end;
                j += 1;
            } else {
                break;
            }
        }

        let span_text = span_words.join(" ");

        // Gather the alpha-only surface form of preceding tokens for context.
        let preceding: Vec<&str> = tokens[..i]
            .iter()
            .map(|t| t.stripped)
            .filter(|w| !w.is_empty())
            .collect();

        let ctx = context_from_preceding(&preceding);

        let kind = match ctx {
            EntityContext::Person => EntityKind::Person,
            EntityContext::Place => EntityKind::Place,
            EntityContext::Topic => EntityKind::Topic,
            EntityContext::Task => EntityKind::Task,
            EntityContext::None => {
                if is_place_name(&span_text) {
                    EntityKind::Place
                } else if has_org_suffix(&span_text) {
                    EntityKind::Organization
                } else if is_acronym(&span_text) {
                    EntityKind::Topic
                } else {
                    // Sentence-start capitals are ambiguous; lower confidence.
                    EntityKind::Person
                }
            }
        };

        let confidence = if ctx != EntityContext::None {
            0.80
        } else if is_place_name(&span_text) || has_org_suffix(&span_text) || is_acronym(&span_text) {
            0.75
        } else if i == 0 {
            0.40 // sentence-start capital — weakly assumed Person
        } else {
            0.65
        };

        entities.push(ExtractedEntity {
            text: span_text,
            span: byte_start..byte_end,
            kind,
            confidence,
        });

        i = j;
    }

    entities
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn extractor() -> HeuristicEntityExtractor {
        HeuristicEntityExtractor
    }

    #[test]
    fn extracts_person_place_and_topic_from_example_utterance() {
        let entities = extractor().extract("I talked to Sylvia about Qdrant in Seattle");

        let person = entities
            .iter()
            .find(|e| e.text == "Sylvia")
            .expect("Sylvia should be extracted");
        assert_eq!(person.kind, EntityKind::Person);

        let topic = entities
            .iter()
            .find(|e| e.text == "Qdrant")
            .expect("Qdrant should be extracted");
        assert_eq!(topic.kind, EntityKind::Topic);

        let place = entities
            .iter()
            .find(|e| e.text == "Seattle")
            .expect("Seattle should be extracted");
        assert_eq!(place.kind, EntityKind::Place);
    }

    #[test]
    fn span_bytes_match_source_text() {
        let text = "I talked to Sylvia about Qdrant in Seattle";
        let entities = extractor().extract(text);
        for entity in &entities {
            assert_eq!(&text[entity.span.clone()], entity.text.as_str());
        }
    }

    #[test]
    fn repeated_references_map_to_same_node_id() {
        let first = extractor().extract("I talked to Sylvia yesterday");
        let second = extractor().extract("Sylvia sent me a message");

        let id1 = first
            .iter()
            .find(|e| e.text == "Sylvia")
            .expect("first mention")
            .provisional_node_id();
        let id2 = second
            .iter()
            .find(|e| e.text == "Sylvia")
            .expect("second mention")
            .provisional_node_id();

        assert_eq!(id1, id2, "both mentions should resolve to the same node id");
        assert_eq!(id1, "person:sylvia");
    }

    #[test]
    fn resolve_entities_uses_existing_graph_node_when_available() {
        let existing = GraphNodeRef {
            id: "person:sylvia".to_string(),
            label: "Sylvia Briggs".to_string(),
        };
        let entities = extractor().extract("I talked to Sylvia about the project");
        let nodes = resolve_entities(&entities, &|node_id| {
            if node_id == "person:sylvia" {
                Some(existing.clone())
            } else {
                None
            }
        });

        let sylvia_node = nodes
            .iter()
            .find(|n| n.node.id == "person:sylvia")
            .expect("Sylvia node should be present");
        assert_eq!(
            sylvia_node.node.label, "Sylvia Briggs",
            "existing node label should be used"
        );
    }

    #[test]
    fn resolve_entities_creates_provisional_node_when_none_exists() {
        let entities = extractor().extract("I talked to Sylvia");
        let nodes = resolve_entities(&entities, &|_| None);

        let sylvia_node = nodes
            .iter()
            .find(|n| n.node.id == "person:sylvia")
            .expect("provisional Sylvia node should be created");
        assert_eq!(sylvia_node.node.label, "Sylvia");
    }

    #[test]
    fn resolve_entities_deduplicates_repeated_mentions() {
        let entities = vec![
            ExtractedEntity {
                text: "Sylvia".to_string(),
                span: 0..6,
                kind: EntityKind::Person,
                confidence: 0.8,
            },
            ExtractedEntity {
                text: "Sylvia".to_string(),
                span: 20..26,
                kind: EntityKind::Person,
                confidence: 0.8,
            },
        ];
        let nodes = resolve_entities(&entities, &|_| None);
        assert_eq!(nodes.len(), 1, "duplicate entity should be deduplicated");
        assert_eq!(nodes[0].node.id, "person:sylvia");
    }

    #[test]
    fn place_name_without_cue_is_still_classified_as_place() {
        let entities = extractor().extract("Seattle is a great city");
        let place = entities
            .iter()
            .find(|e| e.text == "Seattle")
            .expect("Seattle should be extracted");
        assert_eq!(place.kind, EntityKind::Place);
    }

    #[test]
    fn acronym_without_cue_is_classified_as_topic() {
        let entities = extractor().extract("We deployed LLVM yesterday");
        let topic = entities
            .iter()
            .find(|e| e.text == "LLVM")
            .expect("LLVM should be extracted");
        assert_eq!(topic.kind, EntityKind::Topic);
    }

    #[test]
    fn empty_text_returns_no_entities() {
        let entities = extractor().extract("");
        assert!(entities.is_empty());
    }

    #[test]
    fn all_lowercase_text_returns_no_entities() {
        let entities = extractor().extract("i talked to someone about something");
        assert!(entities.is_empty());
    }

    #[test]
    fn provisional_node_id_is_stable_across_case_variations() {
        let entity = ExtractedEntity {
            text: "New York".to_string(),
            span: 0..8,
            kind: EntityKind::Place,
            confidence: 0.9,
        };
        assert_eq!(entity.provisional_node_id(), "place:new_york");
    }
}
