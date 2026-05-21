use serde::{Deserialize, Serialize};

use crate::mouth::riper::espeak_ng_rules::english_to_rule_descriptor;
use crate::mouth::riper::text::{NormalizedText, NormalizedToken, detect_vocative_spans};

pub type WordIndex = usize;

const INFINITIVAL_MARKER_CONFIDENCE: f32 = 0.92;
const WEAK_FUNCTION_CANDIDATE_CONFIDENCE: f32 = 0.88;
const DETERMINER_LINK_CONFIDENCE: f32 = 0.83;
const AUXILIARY_LINK_CONFIDENCE: f32 = 0.82;
const MODIFIER_LINK_CONFIDENCE: f32 = 0.72;
const CONTRAST_PAIR_CONFIDENCE: f32 = 0.91;
const VOCATIVE_LINK_CONFIDENCE: f32 = 0.86;
const APPOSITION_LINK_CONFIDENCE: f32 = 0.8;
const PARENTHETICAL_LINK_CONFIDENCE: f32 = 0.79;
const AMBIGUOUS_NOUN_ATTACHMENT_CONFIDENCE: f32 = 0.46;
const AMBIGUOUS_VERB_ATTACHMENT_CONFIDENCE: f32 = 0.44;

const COMMON_LINK_ADJECTIVES: &[&str] = &[
    "small", "big", "good", "bad", "bright", "dark", "quick", "slow", "new", "old", "young",
];

#[derive(Debug, Clone, PartialEq)]
pub struct SentenceAnalysis {
    pub tokens: Vec<TokenAnalysis>,
    pub link_parses: Vec<SyntacticLinkParse>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenAnalysis {
    pub token_index: usize,
    pub word_index: Option<WordIndex>,
    pub text: String,
    pub pos: PartOfSpeech,
    pub syntactic_role: Option<SyntacticRole>,
    pub prosodic_role: ProsodicRole,
    pub reduction: ReductionClass,
    pub reduction_diagnostic: Option<ReductionDiagnostic>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntacticLink {
    pub left: WordIndex,
    pub right: WordIndex,
    pub kind: SyntacticLinkKind,
    pub confidence: f32,
    pub source: AnalysisSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum SyntacticLinkKind {
    Subject,
    Object,
    Complement,
    InfinitivalMarker,
    Modifier,
    Determiner,
    Auxiliary,
    Coordination,
    ContrastPair,
    Vocative,
    Apposition,
    Parenthetical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisSource {
    HeuristicGrammarIsland,
    AmbiguityVariant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisClaimKind {
    InfinitivalMarker,
    WeakFunctionCandidate,
    ContrastPair,
    VocativeBoundary,
    ParentheticalBoundary,
    AppositionBoundary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisClaim {
    pub word_indices: Vec<WordIndex>,
    pub kind: AnalysisClaimKind,
    pub confidence: f32,
    pub source: AnalysisSource,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntacticLinkParse {
    pub links: Vec<SyntacticLink>,
    pub claims: Vec<AnalysisClaim>,
    pub rank: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvironmentPattern {
    pub predicates: Vec<ContextPredicate>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextPredicate {
    SyntacticLink(SyntacticLinkKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartOfSpeech {
    Noun,
    Verb,
    Auxiliary,
    Determiner,
    Preposition,
    Pronoun,
    Adverb,
    Adjective,
    Conjunction,
    Particle,
    ProperName,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyntacticRole {
    InfinitivalMarker,
    PrepositionalObjectLink,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProsodicRole {
    Content,
    FunctionWeak,
    FunctionStrong,
    Contrastive,
    Focus,
    DirectAddress,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReductionClass {
    None,
    WeakFunctionWord,
    CliticLike,
    Contracted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReductionStatus {
    Applied,
    Blocked,
    Provisional,
    NotApplicable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReductionDiagnostic {
    pub word: String,
    pub word_index: usize,
    pub rule: String,
    pub source: String,
    pub source_file: String,
    pub source_license: String,
    pub citation: String,
    pub realized: String,
    pub reason: String,
    pub status: ReductionStatus,
}

pub trait SentenceAnalyzer {
    fn analyze(&self, source_text: &str, normalized: &NormalizedText) -> SentenceAnalysis;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicSentenceAnalyzer;

impl SentenceAnalyzer for HeuristicSentenceAnalyzer {
    fn analyze(&self, source_text: &str, normalized: &NormalizedText) -> SentenceAnalysis {
        let word_slots = normalized
            .tokens
            .iter()
            .enumerate()
            .filter_map(|(token_index, token)| match token {
                NormalizedToken::Word(word) => Some((token_index, word.clone())),
                NormalizedToken::Initial(initial) => Some((token_index, initial.to_string())),
                NormalizedToken::PhraseBreak => None,
            })
            .collect::<Vec<_>>();
        let source_words = word_slots
            .iter()
            .map(|(token_index, _)| {
                normalized
                    .token_spans
                    .get(*token_index)
                    .and_then(|span| source_text.get(span.clone()))
                    .unwrap_or_default()
                    .to_string()
            })
            .collect::<Vec<_>>();
        let token_to_word_index = word_slots
            .iter()
            .enumerate()
            .map(|(word_index, (token_index, _))| (*token_index, word_index))
            .collect::<std::collections::HashMap<_, _>>();

        let tokens = normalized
            .tokens
            .iter()
            .enumerate()
            .map(|(token_index, token)| {
                let (token_text, base_pos) = match token {
                    NormalizedToken::Word(word) => (word.clone(), base_pos(word)),
                    NormalizedToken::Initial(initial) => (
                        initial.to_ascii_lowercase().to_string(),
                        PartOfSpeech::ProperName,
                    ),
                    NormalizedToken::PhraseBreak => ("|".to_string(), PartOfSpeech::Unknown),
                };
                let Some(word_index) = token_to_word_index.get(&token_index).copied() else {
                    return TokenAnalysis {
                        token_index,
                        word_index: None,
                        text: token_text,
                        pos: PartOfSpeech::Unknown,
                        syntactic_role: None,
                        prosodic_role: ProsodicRole::Content,
                        reduction: ReductionClass::None,
                        reduction_diagnostic: None,
                    };
                };

                if token_text != "to" {
                    let prosodic_role = if is_function_word(&token_text) {
                        ProsodicRole::FunctionWeak
                    } else {
                        ProsodicRole::Content
                    };
                    return TokenAnalysis {
                        token_index,
                        word_index: Some(word_index),
                        text: token_text,
                        pos: base_pos,
                        syntactic_role: None,
                        prosodic_role,
                        reduction: ReductionClass::None,
                        reduction_diagnostic: None,
                    };
                }

                let raw_token = normalized
                    .token_spans
                    .get(token_index)
                    .and_then(|span| source_text.get(span.clone()))
                    .unwrap_or("to");
                let prev = word_index
                    .checked_sub(1)
                    .and_then(|idx| word_slots.get(idx))
                    .map(|(_, text)| text.as_str());
                let prev_prev = word_index
                    .checked_sub(2)
                    .and_then(|idx| word_slots.get(idx))
                    .map(|(_, text)| text.as_str());
                let next = word_slots
                    .get(word_index + 1)
                    .map(|(_, text)| text.as_str());

                let (pos, syntactic_role, prosodic_role, reduction, diagnostic) =
                    classify_to_token(raw_token, word_index, prev_prev, prev, next);

                TokenAnalysis {
                    token_index,
                    word_index: Some(word_index),
                    text: token_text,
                    pos,
                    syntactic_role,
                    prosodic_role,
                    reduction,
                    reduction_diagnostic: Some(diagnostic),
                }
            })
            .collect();
        let link_parses = build_link_parses(source_text, normalized, &word_slots, &source_words);

        SentenceAnalysis {
            tokens,
            link_parses,
        }
    }
}

impl SentenceAnalysis {
    pub fn claims(&self) -> Vec<AnalysisClaim> {
        self.link_parses
            .iter()
            .flat_map(|parse| parse.claims.iter().cloned())
            .collect()
    }

    pub fn environment_patterns(&self) -> Vec<EnvironmentPattern> {
        self.link_parses
            .iter()
            .map(SyntacticLinkParse::as_environment_pattern)
            .collect()
    }
}

impl SyntacticLinkParse {
    pub fn as_environment_pattern(&self) -> EnvironmentPattern {
        let mut seen = std::collections::HashSet::new();
        let predicates = self
            .links
            .iter()
            .filter_map(|link| {
                if seen.insert(link.kind) {
                    Some(ContextPredicate::SyntacticLink(link.kind))
                } else {
                    None
                }
            })
            .collect();
        EnvironmentPattern { predicates }
    }
}

fn build_link_parses(
    source_text: &str,
    normalized: &NormalizedText,
    word_slots: &[(usize, String)],
    source_words: &[String],
) -> Vec<SyntacticLinkParse> {
    let words = word_slots
        .iter()
        .map(|(_, word)| word.as_str())
        .collect::<Vec<_>>();
    let token_to_word_index = word_slots
        .iter()
        .enumerate()
        .map(|(word_index, (token_index, _))| (*token_index, word_index))
        .collect::<std::collections::HashMap<_, _>>();
    let word_spans = word_slots
        .iter()
        .map(|(token_index, _)| {
            normalized
                .token_spans
                .get(*token_index)
                .cloned()
                .unwrap_or(0..0)
        })
        .collect::<Vec<_>>();

    let mut links = Vec::new();
    let mut claims = Vec::new();

    for (idx, window) in words.windows(2).enumerate() {
        let left = window[0];
        let right = window[1];
        if left == "to" && is_likely_verb(right) {
            push_link(
                &mut links,
                SyntacticLink {
                    left: idx,
                    right: idx + 1,
                    kind: SyntacticLinkKind::InfinitivalMarker,
                    confidence: INFINITIVAL_MARKER_CONFIDENCE,
                    source: AnalysisSource::HeuristicGrammarIsland,
                },
            );
            claims.push(AnalysisClaim {
                word_indices: vec![idx],
                kind: AnalysisClaimKind::InfinitivalMarker,
                confidence: INFINITIVAL_MARKER_CONFIDENCE,
                source: AnalysisSource::HeuristicGrammarIsland,
            });
            claims.push(AnalysisClaim {
                word_indices: vec![idx],
                kind: AnalysisClaimKind::WeakFunctionCandidate,
                confidence: WEAK_FUNCTION_CANDIDATE_CONFIDENCE,
                source: AnalysisSource::HeuristicGrammarIsland,
            });
        }

        if is_determiner(left) && is_likely_nominal(right) {
            push_link(
                &mut links,
                SyntacticLink {
                    left: idx,
                    right: idx + 1,
                    kind: SyntacticLinkKind::Determiner,
                    confidence: DETERMINER_LINK_CONFIDENCE,
                    source: AnalysisSource::HeuristicGrammarIsland,
                },
            );
        }

        if is_auxiliary(left) && is_likely_verb(right) {
            push_link(
                &mut links,
                SyntacticLink {
                    left: idx,
                    right: idx + 1,
                    kind: SyntacticLinkKind::Auxiliary,
                    confidence: AUXILIARY_LINK_CONFIDENCE,
                    source: AnalysisSource::HeuristicGrammarIsland,
                },
            );
        }

        if is_modifier_pair(left, right) {
            push_link(
                &mut links,
                SyntacticLink {
                    left: idx,
                    right: idx + 1,
                    kind: SyntacticLinkKind::Modifier,
                    confidence: MODIFIER_LINK_CONFIDENCE,
                    source: AnalysisSource::HeuristicGrammarIsland,
                },
            );
        }
    }

    for (left, right) in detect_contrast_pairs(&words, source_words) {
        push_link(
            &mut links,
            SyntacticLink {
                left,
                right,
                kind: SyntacticLinkKind::ContrastPair,
                confidence: CONTRAST_PAIR_CONFIDENCE,
                source: AnalysisSource::HeuristicGrammarIsland,
            },
        );
        claims.push(AnalysisClaim {
            word_indices: vec![left, right],
            kind: AnalysisClaimKind::ContrastPair,
            confidence: CONTRAST_PAIR_CONFIDENCE,
            source: AnalysisSource::HeuristicGrammarIsland,
        });
    }

    let vocative_spans = detect_vocative_spans(source_text);
    for span in vocative_spans {
        let targets = word_spans
            .iter()
            .enumerate()
            .filter_map(|(word_index, word_span)| {
                (word_span.start < span.end && word_span.end > span.start).then_some(word_index)
            })
            .collect::<Vec<_>>();
        if let Some(&first_target) = targets.first() {
            let anchor = first_target.saturating_sub(1);
            push_link(
                &mut links,
                SyntacticLink {
                    left: anchor,
                    right: first_target,
                    kind: SyntacticLinkKind::Vocative,
                    confidence: VOCATIVE_LINK_CONFIDENCE,
                    source: AnalysisSource::HeuristicGrammarIsland,
                },
            );
            claims.push(AnalysisClaim {
                word_indices: vec![first_target],
                kind: AnalysisClaimKind::VocativeBoundary,
                confidence: VOCATIVE_LINK_CONFIDENCE,
                source: AnalysisSource::HeuristicGrammarIsland,
            });
        }
    }

    let comma_breaks = normalized
        .tokens
        .iter()
        .enumerate()
        .filter_map(|(token_index, token)| {
            if !matches!(token, NormalizedToken::PhraseBreak) {
                return None;
            }
            let span = normalized.token_spans.get(token_index)?;
            let mark = source_text.get(span.clone())?;
            if mark != "," {
                return None;
            }
            let left_word = (0..token_index)
                .rev()
                .find_map(|idx| token_to_word_index.get(&idx).copied());
            let right_word = ((token_index + 1)..normalized.tokens.len())
                .find_map(|idx| token_to_word_index.get(&idx).copied());
            Some((span.clone(), left_word, right_word))
        })
        .collect::<Vec<_>>();

    for pair in comma_breaks.windows(2) {
        let left_break = &pair[0];
        let right_break = &pair[1];
        let between = word_spans
            .iter()
            .enumerate()
            .filter_map(|(word_index, span)| {
                (span.start >= left_break.0.end && span.end <= right_break.0.start)
                    .then_some(word_index)
            })
            .collect::<Vec<_>>();
        if between.is_empty() {
            continue;
        }
        let Some(left_anchor) = left_break.1 else {
            continue;
        };
        let is_apposition = between
            .first()
            .and_then(|idx| words.get(*idx).copied())
            .is_some_and(|word| matches!(word, "who" | "which" | "that" | "whom"));
        if is_apposition {
            let target = between[0];
            push_link(
                &mut links,
                SyntacticLink {
                    left: left_anchor,
                    right: target,
                    kind: SyntacticLinkKind::Apposition,
                    confidence: APPOSITION_LINK_CONFIDENCE,
                    source: AnalysisSource::HeuristicGrammarIsland,
                },
            );
            claims.push(AnalysisClaim {
                word_indices: between.clone(),
                kind: AnalysisClaimKind::AppositionBoundary,
                confidence: APPOSITION_LINK_CONFIDENCE,
                source: AnalysisSource::HeuristicGrammarIsland,
            });
            continue;
        }
        if let Some(right_anchor) = right_break.2 {
            push_link(
                &mut links,
                SyntacticLink {
                    left: left_anchor,
                    right: right_anchor,
                    kind: SyntacticLinkKind::Parenthetical,
                    confidence: PARENTHETICAL_LINK_CONFIDENCE,
                    source: AnalysisSource::HeuristicGrammarIsland,
                },
            );
            claims.push(AnalysisClaim {
                word_indices: between.clone(),
                kind: AnalysisClaimKind::ParentheticalBoundary,
                confidence: PARENTHETICAL_LINK_CONFIDENCE,
                source: AnalysisSource::HeuristicGrammarIsland,
            });
        }
    }

    let primary_parse = SyntacticLinkParse {
        links: links.clone(),
        claims: claims.clone(),
        rank: 1.0,
    };
    if let Some((verb_anchor, noun_anchor, object_index)) = detect_with_attachment_ambiguity(&words)
    {
        let mut noun_parse = primary_parse.clone();
        noun_parse.rank = 0.6;
        push_link(
            &mut noun_parse.links,
            SyntacticLink {
                left: noun_anchor,
                right: object_index,
                kind: SyntacticLinkKind::Modifier,
                confidence: AMBIGUOUS_NOUN_ATTACHMENT_CONFIDENCE,
                source: AnalysisSource::AmbiguityVariant,
            },
        );
        let mut verb_parse = primary_parse;
        verb_parse.rank = 0.55;
        push_link(
            &mut verb_parse.links,
            SyntacticLink {
                left: verb_anchor,
                right: object_index,
                kind: SyntacticLinkKind::Complement,
                confidence: AMBIGUOUS_VERB_ATTACHMENT_CONFIDENCE,
                source: AnalysisSource::AmbiguityVariant,
            },
        );
        return vec![noun_parse, verb_parse];
    }

    vec![primary_parse]
}

fn push_link(links: &mut Vec<SyntacticLink>, candidate: SyntacticLink) {
    if links.iter().any(|existing| {
        existing.left == candidate.left
            && existing.right == candidate.right
            && existing.kind == candidate.kind
    }) {
        return;
    }
    links.push(candidate);
}

fn detect_contrast_pairs(words: &[&str], source_words: &[String]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for index in 0..words.len() {
        if words[index] == "not" && index + 3 < words.len() && words[index + 2] == "but" {
            pairs.push((index + 1, index + 3));
            continue;
        }
        if words[index] == "not" && index > 0 && index + 1 < words.len() {
            pairs.push((index - 1, index + 1));
            continue;
        }
        if index + 2 < words.len()
            && words[index + 1] == "not"
            && source_words
                .get(index)
                .is_some_and(|word| is_all_caps_token(word))
            && source_words
                .get(index + 2)
                .is_some_and(|word| is_all_caps_token(word))
        {
            pairs.push((index, index + 2));
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

fn is_all_caps_token(word: &str) -> bool {
    let mut has_alpha = false;
    for ch in word.chars() {
        if ch.is_ascii_alphabetic() {
            has_alpha = true;
            if !ch.is_ascii_uppercase() {
                return false;
            }
        }
    }
    has_alpha
}

fn detect_with_attachment_ambiguity(words: &[&str]) -> Option<(usize, usize, usize)> {
    for with_index in 1..words.len() {
        if words[with_index] != "with" || with_index + 2 >= words.len() {
            continue;
        }
        if !is_determiner(words[with_index + 1]) || !is_likely_nominal(words[with_index + 2]) {
            continue;
        }
        let noun_anchor = with_index.checked_sub(1)?;
        if !is_likely_nominal(words[noun_anchor]) {
            continue;
        }
        let verb_anchor = (0..noun_anchor)
            .rev()
            .find(|index| is_likely_verb(words[*index]))?;
        return Some((verb_anchor, noun_anchor, with_index + 2));
    }
    None
}

fn is_likely_nominal(word: &str) -> bool {
    matches!(
        base_pos(word),
        PartOfSpeech::Noun | PartOfSpeech::Pronoun | PartOfSpeech::ProperName
    ) && !is_likely_verb(word)
}

fn is_modifier_pair(left: &str, right: &str) -> bool {
    let adverb = left.ends_with("ly");
    let adjective = COMMON_LINK_ADJECTIVES.contains(&left)
        || left.ends_with("ous")
        || left.ends_with("ive")
        || left.ends_with("al");
    (adverb && is_likely_verb(right)) || (adjective && is_likely_nominal(right))
}

fn classify_to_token(
    raw_token: &str,
    word_index: usize,
    prev_prev: Option<&str>,
    prev: Option<&str>,
    next: Option<&str>,
) -> (
    PartOfSpeech,
    Option<SyntacticRole>,
    ProsodicRole,
    ReductionClass,
    ReductionDiagnostic,
) {
    let resolve_rule = |rule_id: &str| -> ToRuleDescriptorFallback {
        english_to_rule_descriptor(rule_id)
            .map(Into::into)
            .unwrap_or_else(|| {
                let output_transformation = if rule_id == "weak_form_to_before_verb" {
                    "T AH0"
                } else {
                    "T UW1"
                };
                ToRuleDescriptorFallback {
                    rule_id: rule_id.to_string(),
                    source: "espeak-ng-derived".to_string(),
                    source_file: "dictsource/en_rules".to_string(),
                    source_license: "GPL-3.0-or-later".to_string(),
                    citation_form: "T UW1".to_string(),
                    output_transformation: output_transformation.to_string(),
                }
            })
    };
    let weak_before_verb = resolve_rule("weak_form_to_before_verb");
    let phrase_final = resolve_rule("weak_form_to_phrase_final_provisional");
    let contrastive = resolve_rule("strong_to_contrastive_uppercase");
    let explicit_override = resolve_rule("strong_to_explicit_phonetic_override");
    let citation_initial = resolve_rule("strong_to_citation_phrase_initial");
    let quotation_citation = resolve_rule("strong_to_quotation_or_citation");
    let prepositional = resolve_rule("strong_to_prepositional");

    let diagnostic = |rule: &ToRuleDescriptorFallback, realized: &str, reason: &str, status| {
        ReductionDiagnostic {
            word: "to".to_string(),
            word_index,
            rule: rule.rule_id.clone(),
            source: rule.source.clone(),
            source_file: rule.source_file.clone(),
            source_license: rule.source_license.clone(),
            citation: rule.citation_form.clone(),
            realized: realized.to_string(),
            reason: reason.to_string(),
            status,
        }
    };

    if raw_token.chars().all(|ch| ch.is_ascii_uppercase()) && raw_token.len() > 1 {
        return (
            PartOfSpeech::Preposition,
            Some(SyntacticRole::PrepositionalObjectLink),
            ProsodicRole::Contrastive,
            ReductionClass::None,
            diagnostic(
                &contrastive,
                &contrastive.output_transformation,
                "contrastive_emphasis",
                ReductionStatus::Blocked,
            ),
        );
    }

    if raw_token.contains('/') || raw_token.contains('@') {
        return (
            PartOfSpeech::Preposition,
            Some(SyntacticRole::PrepositionalObjectLink),
            ProsodicRole::FunctionStrong,
            ReductionClass::None,
            diagnostic(
                &explicit_override,
                &explicit_override.output_transformation,
                "explicit_phonetic_override",
                ReductionStatus::Blocked,
            ),
        );
    }

    if next.is_none() {
        return (
            PartOfSpeech::Particle,
            Some(SyntacticRole::InfinitivalMarker),
            ProsodicRole::FunctionWeak,
            ReductionClass::WeakFunctionWord,
            diagnostic(
                &phrase_final,
                &phrase_final.output_transformation,
                "phrase_final_uncertainty",
                ReductionStatus::Provisional,
            ),
        );
    }

    if next == Some("be") && prev.is_none() {
        return (
            PartOfSpeech::Particle,
            Some(SyntacticRole::InfinitivalMarker),
            ProsodicRole::FunctionStrong,
            ReductionClass::None,
            diagnostic(
                &citation_initial,
                &citation_initial.output_transformation,
                "citation_form_phrase_initial",
                ReductionStatus::Blocked,
            ),
        );
    }

    if prev_prev == Some("or") && prev == Some("not") && next == Some("be") {
        return (
            PartOfSpeech::Particle,
            Some(SyntacticRole::InfinitivalMarker),
            ProsodicRole::FunctionStrong,
            ReductionClass::None,
            diagnostic(
                &quotation_citation,
                &quotation_citation.output_transformation,
                "quotation_or_citation_form",
                ReductionStatus::Blocked,
            ),
        );
    }

    if next.is_some_and(is_likely_verb) {
        return (
            PartOfSpeech::Particle,
            Some(SyntacticRole::InfinitivalMarker),
            ProsodicRole::FunctionWeak,
            ReductionClass::WeakFunctionWord,
            diagnostic(
                &weak_before_verb,
                &weak_before_verb.output_transformation,
                "unstressed_function_word_before_verb",
                ReductionStatus::Applied,
            ),
        );
    }

    (
        PartOfSpeech::Preposition,
        Some(SyntacticRole::PrepositionalObjectLink),
        ProsodicRole::FunctionStrong,
        ReductionClass::None,
        diagnostic(
            &prepositional,
            &prepositional.output_transformation,
            "prepositional_to",
            ReductionStatus::Blocked,
        ),
    )
}

struct ToRuleDescriptorFallback {
    rule_id: String,
    source: String,
    source_file: String,
    source_license: String,
    citation_form: String,
    output_transformation: String,
}

impl From<crate::mouth::riper::espeak_ng_rules::ToRuleDescriptor> for ToRuleDescriptorFallback {
    fn from(value: crate::mouth::riper::espeak_ng_rules::ToRuleDescriptor) -> Self {
        Self {
            rule_id: value.rule_id,
            source: value.provenance.source,
            source_file: value.provenance.source_file,
            source_license: value.provenance.source_license,
            citation_form: value.citation_form,
            output_transformation: value.output_transformation,
        }
    }
}

fn base_pos(word: &str) -> PartOfSpeech {
    if matches!(word, "to") {
        return PartOfSpeech::Preposition;
    }
    if is_pronoun(word) {
        return PartOfSpeech::Pronoun;
    }
    if is_determiner(word) {
        return PartOfSpeech::Determiner;
    }
    if is_conjunction(word) {
        return PartOfSpeech::Conjunction;
    }
    if is_auxiliary(word) {
        return PartOfSpeech::Auxiliary;
    }
    if is_likely_verb(word) {
        return PartOfSpeech::Verb;
    }
    PartOfSpeech::Noun
}

fn is_function_word(word: &str) -> bool {
    is_pronoun(word)
        || is_determiner(word)
        || is_conjunction(word)
        || is_auxiliary(word)
        || matches!(word, "to" | "for" | "of" | "and")
}

fn is_pronoun(word: &str) -> bool {
    matches!(
        word,
        "i" | "you" | "he" | "she" | "it" | "we" | "they" | "me" | "him" | "her" | "us" | "them"
    )
}

fn is_determiner(word: &str) -> bool {
    matches!(
        word,
        "a" | "an" | "the" | "this" | "that" | "these" | "those"
    )
}

fn is_conjunction(word: &str) -> bool {
    matches!(word, "and" | "or" | "but" | "not")
}

fn is_auxiliary(word: &str) -> bool {
    matches!(
        word,
        "be" | "am"
            | "is"
            | "are"
            | "was"
            | "were"
            | "been"
            | "do"
            | "does"
            | "did"
            | "have"
            | "has"
            | "had"
            | "will"
            | "would"
            | "should"
            | "could"
            | "may"
            | "might"
            | "must"
            | "can"
    )
}

fn is_likely_verb(word: &str) -> bool {
    matches!(
        word,
        "go" | "leave"
            | "remember"
            | "see"
            | "stay"
            | "be"
            | "try"
            | "need"
            | "want"
            | "make"
            | "take"
            | "get"
            | "keep"
            | "let"
            | "tell"
            | "call"
            | "put"
            | "ask"
    ) || has_likely_verb_suffix(word)
}

fn has_likely_verb_suffix(word: &str) -> bool {
    const COMMON_NON_VERB_ING: &[&str] = &["thing", "king", "morning", "ceiling"];
    const COMMON_NON_VERB_ED: &[&str] = &["red", "bed", "sled"];
    (word.len() >= 5 && word.ends_with("ing") && !COMMON_NON_VERB_ING.contains(&word))
        || (word.len() >= 4 && word.ends_with("ed") && !COMMON_NON_VERB_ED.contains(&word))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mouth::riper::text::TextNormalizer;

    fn analyze(text: &str) -> SentenceAnalysis {
        let normalized = TextNormalizer
            .normalize(text)
            .expect("text should normalize");
        HeuristicSentenceAnalyzer.analyze(text, &normalized)
    }

    fn word_index(analysis: &SentenceAnalysis, word: &str) -> usize {
        analysis
            .tokens
            .iter()
            .find(|token| token.word_index.is_some() && token.text == word)
            .and_then(|token| token.word_index)
            .expect("word should exist")
    }

    fn has_link(
        parse: &SyntacticLinkParse,
        left: usize,
        right: usize,
        kind: SyntacticLinkKind,
    ) -> bool {
        parse
            .links
            .iter()
            .any(|link| link.left == left && link.right == right && link.kind == kind)
    }

    #[test]
    fn fixture_links_infinitival_to_and_claims() {
        let analysis = analyze("I want to go.");
        let parse = analysis.link_parses.first().expect("link parse");
        let to = word_index(&analysis, "to");
        let go = word_index(&analysis, "go");
        assert!(has_link(
            parse,
            to,
            go,
            SyntacticLinkKind::InfinitivalMarker
        ));
        assert!(
            parse
                .claims
                .iter()
                .any(|claim| claim.kind == AnalysisClaimKind::InfinitivalMarker
                    && claim.word_indices == vec![to])
        );
        assert!(parse.claims.iter().any(|claim| claim.kind
            == AnalysisClaimKind::WeakFunctionCandidate
            && claim.word_indices == vec![to]));
        assert!(
            analysis
                .environment_patterns()
                .iter()
                .any(|pattern| pattern
                    .predicates
                    .contains(&ContextPredicate::SyntacticLink(
                        SyntacticLinkKind::InfinitivalMarker
                    )))
        );
    }

    #[test]
    fn fixture_links_contrast_pair() {
        let analysis = analyze("I said TO, not FROM.");
        let parse = analysis.link_parses.first().expect("link parse");
        let to = word_index(&analysis, "to");
        let from = word_index(&analysis, "from");
        assert!(has_link(parse, to, from, SyntacticLinkKind::ContrastPair));
        assert!(parse.claims.iter().any(|claim| {
            claim.kind == AnalysisClaimKind::ContrastPair && claim.word_indices == vec![to, from]
        }));
    }

    #[test]
    fn fixture_links_vocative_boundary() {
        let analysis = analyze("Thank you, Dave.");
        let parse = analysis.link_parses.first().expect("link parse");
        let you = word_index(&analysis, "you");
        let dave = word_index(&analysis, "dave");
        assert!(has_link(parse, you, dave, SyntacticLinkKind::Vocative));
        assert!(
            parse
                .claims
                .iter()
                .any(|claim| claim.kind == AnalysisClaimKind::VocativeBoundary
                    && claim.word_indices == vec![dave])
        );
    }

    #[test]
    fn detects_vocative_span_boundaries() {
        let spans = detect_vocative_spans("Thank you, Dave.");
        assert_eq!(spans.len(), 1);
        assert_eq!("Dave", &"Thank you, Dave."[spans[0].clone()]);
    }

    #[test]
    fn fixture_links_parenthetical_and_apposition() {
        let parenthetical = analyze("The machine, unfortunately, exploded.");
        let parse = parenthetical.link_parses.first().expect("link parse");
        let machine = word_index(&parenthetical, "machine");
        let exploded = word_index(&parenthetical, "exploded");
        assert!(has_link(
            parse,
            machine,
            exploded,
            SyntacticLinkKind::Parenthetical
        ));
        assert!(
            parse
                .claims
                .iter()
                .any(|claim| claim.kind == AnalysisClaimKind::ParentheticalBoundary)
        );

        let apposition = analyze("My brother, who lives in Tacoma, arrived.");
        let apposition_parse = apposition.link_parses.first().expect("link parse");
        let brother = word_index(&apposition, "brother");
        let who = word_index(&apposition, "who");
        assert!(has_link(
            apposition_parse,
            brother,
            who,
            SyntacticLinkKind::Apposition
        ));
        assert!(
            apposition_parse
                .claims
                .iter()
                .any(|claim| claim.kind == AnalysisClaimKind::AppositionBoundary)
        );
    }

    #[test]
    fn preserves_ambiguous_with_attachment_as_alternative_parses() {
        let analysis = analyze("I saw the man with the telescope.");
        assert_eq!(analysis.link_parses.len(), 2);
        let saw = word_index(&analysis, "saw");
        let man = word_index(&analysis, "man");
        let telescope = word_index(&analysis, "telescope");
        assert!(analysis.link_parses.iter().any(|parse| {
            has_link(parse, man, telescope, SyntacticLinkKind::Modifier)
                && parse.rank >= 0.5
                && parse.links.iter().any(|link| {
                    link.source == AnalysisSource::AmbiguityVariant
                        || link.kind == SyntacticLinkKind::Determiner
                })
        }));
        assert!(analysis.link_parses.iter().any(|parse| {
            has_link(parse, saw, telescope, SyntacticLinkKind::Complement)
                && parse.rank >= 0.5
                && parse
                    .links
                    .iter()
                    .any(|link| link.source == AnalysisSource::AmbiguityVariant)
        }));
    }

    #[test]
    fn keeps_single_parse_for_non_ambiguous_sentence() {
        let analysis = analyze("I saw the man.");
        assert_eq!(analysis.link_parses.len(), 1);
    }

    #[test]
    fn detects_additional_contrast_patterns() {
        let adjacent = analyze("TO not FROM");
        let adjacent_parse = adjacent.link_parses.first().expect("link parse");
        let to = word_index(&adjacent, "to");
        let from = word_index(&adjacent, "from");
        assert!(has_link(
            adjacent_parse,
            to,
            from,
            SyntacticLinkKind::ContrastPair
        ));

        let but_pattern = analyze("not red but blue");
        let but_parse = but_pattern.link_parses.first().expect("link parse");
        let red = word_index(&but_pattern, "red");
        let blue = word_index(&but_pattern, "blue");
        assert!(has_link(
            but_parse,
            red,
            blue,
            SyntacticLinkKind::ContrastPair
        ));
    }

    #[test]
    fn creates_modifier_links_for_adjective_and_adverb_pairs() {
        let adjective = analyze("The bright machine exploded.");
        let adjective_parse = adjective.link_parses.first().expect("link parse");
        let bright = word_index(&adjective, "bright");
        let machine = word_index(&adjective, "machine");
        assert!(has_link(
            adjective_parse,
            bright,
            machine,
            SyntacticLinkKind::Modifier
        ));

        let adverb = analyze("They quickly leave.");
        let adverb_parse = adverb.link_parses.first().expect("link parse");
        let quickly = word_index(&adverb, "quickly");
        let leave = word_index(&adverb, "leave");
        assert!(has_link(
            adverb_parse,
            quickly,
            leave,
            SyntacticLinkKind::Modifier
        ));
    }
}
