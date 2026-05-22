use serde::{Deserialize, Serialize};

use crate::mouth::riper::espeak_ng_rules::{
    LexicalProsodyFlagFact, english_lexical_flag_facts_for_rule, english_to_rule_descriptor,
};
use crate::mouth::riper::evidence::{
    AnalysisClaim, AnalysisSourceKind, AnalysisTarget, ClaimKind, ClaimValue,
};
use crate::mouth::riper::prosody_audit::PhraseBoundaryKind;
use crate::mouth::riper::text::{NormalizedText, NormalizedToken, detect_vocative_spans};

pub type WordIndex = usize;

const INFINITIVAL_MARKER_CONFIDENCE: f32 = 0.92;
const WEAK_FUNCTION_CANDIDATE_CONFIDENCE: f32 = 0.88;
const DETERMINER_LINK_CONFIDENCE: f32 = 0.83;
const AUXILIARY_LINK_CONFIDENCE: f32 = 0.82;
const PREPOSITION_LINK_CONFIDENCE: f32 = 0.8;
const SUBJECT_LINK_CONFIDENCE: f32 = 0.8;
const OBJECT_LINK_CONFIDENCE: f32 = 0.78;
const COMPLEMENT_LINK_CONFIDENCE: f32 = 0.76;
const COORDINATION_LINK_CONFIDENCE: f32 = 0.74;
const MODIFIER_LINK_CONFIDENCE: f32 = 0.72;
const NOUN_COMPOUND_LINK_CONFIDENCE: f32 = 0.78;
const CONTRAST_PAIR_CONFIDENCE: f32 = 0.91;
const VOCATIVE_LINK_CONFIDENCE: f32 = 0.86;
const APPOSITION_LINK_CONFIDENCE: f32 = 0.8;
const PARENTHETICAL_LINK_CONFIDENCE: f32 = 0.79;
const CORE_CLAUSE_CLAIM_CONFIDENCE: f32 = 0.76;
const COORDINATION_CLAIM_CONFIDENCE: f32 = 0.72;
const CONTRASTIVE_FOCUS_CLAIM_CONFIDENCE: f32 = 0.89;
const COMMA_BEHAVIOR_CLAIM_CONFIDENCE: f32 = 0.84;
const ARTICLE_HOOK_CLAIM_CONFIDENCE: f32 = 0.85;
const AMBIGUOUS_NOUN_ATTACHMENT_CONFIDENCE: f32 = 0.46;
const AMBIGUOUS_VERB_ATTACHMENT_CONFIDENCE: f32 = 0.44;

const COMMON_LINK_ADJECTIVES: &[&str] = &[
    "small", "big", "good", "bad", "bright", "dark", "quick", "slow", "new", "old", "young",
];
const PARENTHETICAL_CUE_WORDS: &[&str] = &[
    "actually",
    "basically",
    "frankly",
    "honestly",
    "however",
    "meanwhile",
    "nevertheless",
    "unfortunately",
];
const US_STATE_NAMES: &[&str] = &[
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
    "new hampshire",
    "new jersey",
    "new mexico",
    "new york",
    "north carolina",
    "north dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhode island",
    "south carolina",
    "south dakota",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west virginia",
    "wisconsin",
    "wyoming",
];
const PLACE_APPOSITION_WORDS: &[&str] = &[
    "america",
    "canada",
    "mexico",
    "england",
    "scotland",
    "wales",
    "ireland",
    "france",
    "germany",
    "italy",
    "spain",
    "china",
    "japan",
    "korea",
    "india",
    "australia",
    "zealand",
];

#[derive(Debug, Clone, PartialEq)]
pub struct SentenceAnalysis {
    pub tokens: Vec<TokenAnalysis>,
    pub link_parses: Vec<SyntacticLinkParse>,
    pub terminal_boundary_kind: PhraseBoundaryKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenAnalysis {
    pub token_index: usize,
    pub word_index: Option<WordIndex>,
    pub text: String,
    pub pos: PartOfSpeech,
    pub syntactic_role: Option<SyntacticRole>,
    pub prosodic_role: ProsodicRole,
    pub orthographic_emphasis: OrthographicEmphasisKind,
    pub reduction: ReductionClass,
    pub reduction_diagnostic: Option<ReductionDiagnostic>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProsodyEnvironmentFacts {
    pub word_index: usize,
    pub pos: PartOfSpeech,
    pub prosodic_role: ProsodicRole,
    pub orthographic_emphasis: OrthographicEmphasisKind,
    pub phrase_boundary_after: PhraseBoundaryKind,
    pub syntactic_links: Vec<SyntacticLinkKind>,
    pub lexical_flags: Vec<LexicalProsodyFlagFact>,
    pub confidence: f32,
    pub conservative: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntacticLink {
    pub left: WordIndex,
    pub right: WordIndex,
    pub kind: SyntacticLinkKind,
    pub confidence: f32,
    pub source: SyntacticLinkSource,
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
    Preposition,
    Coordination,
    ContrastPair,
    NounCompound,
    Vocative,
    Apposition,
    Parenthetical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyntacticLinkSource {
    /// Produced by the primary heuristic grammar-island analyser.
    HeuristicGrammarIsland,
    /// Produced as an alternative parse for an attachment ambiguity.
    AmbiguityVariant,
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
pub enum OrthographicEmphasisKind {
    None,
    CapitalizedName,
    AllCapsEmphasis,
    Abbreviation,
    Acronym,
    ExplicitCitationForm,
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
                        orthographic_emphasis: OrthographicEmphasisKind::None,
                        reduction: ReductionClass::None,
                        reduction_diagnostic: None,
                    };
                };

                if token_text != "to" {
                    let raw_token = normalized
                        .token_spans
                        .get(token_index)
                        .and_then(|span| source_text.get(span.clone()))
                        .unwrap_or(token_text.as_str());
                    let orthographic_emphasis =
                        classify_orthographic_emphasis(raw_token, &token_text, base_pos);
                    if is_function_word(&token_text)
                        && matches!(
                            orthographic_emphasis,
                            OrthographicEmphasisKind::AllCapsEmphasis
                                | OrthographicEmphasisKind::Abbreviation
                                | OrthographicEmphasisKind::Acronym
                        )
                    {
                        if matches!(
                            orthographic_emphasis,
                            OrthographicEmphasisKind::Abbreviation
                                | OrthographicEmphasisKind::Acronym
                        ) {
                            return TokenAnalysis {
                                token_index,
                                word_index: Some(word_index),
                                text: token_text,
                                pos: PartOfSpeech::ProperName,
                                syntactic_role: None,
                                prosodic_role: ProsodicRole::Content,
                                orthographic_emphasis,
                                reduction: ReductionClass::None,
                                reduction_diagnostic: None,
                            };
                        }
                        return TokenAnalysis {
                            token_index,
                            word_index: Some(word_index),
                            text: token_text,
                            pos: base_pos,
                            syntactic_role: None,
                            prosodic_role: ProsodicRole::Contrastive,
                            orthographic_emphasis,
                            reduction: ReductionClass::None,
                            reduction_diagnostic: None,
                        };
                    }
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
                        orthographic_emphasis,
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

                let (
                    pos,
                    syntactic_role,
                    prosodic_role,
                    orthographic_emphasis,
                    reduction,
                    diagnostic,
                ) = classify_to_token(raw_token, word_index, prev_prev, prev, next);

                TokenAnalysis {
                    token_index,
                    word_index: Some(word_index),
                    text: token_text,
                    pos,
                    syntactic_role,
                    prosodic_role,
                    orthographic_emphasis,
                    reduction,
                    reduction_diagnostic: Some(diagnostic),
                }
            })
            .collect();
        let link_parses = build_link_parses(source_text, normalized, &word_slots, &source_words);

        SentenceAnalysis {
            tokens,
            link_parses,
            terminal_boundary_kind: normalized.boundary_kind,
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

    pub fn prosody_environment_facts(&self) -> Vec<ProsodyEnvironmentFacts> {
        let Some(primary_parse) = self.link_parses.first() else {
            return Vec::new();
        };
        let conservative_ambiguity = self.has_low_confidence_ambiguity();
        let parse_rank_confidence = primary_parse.rank.clamp(0.0, 1.0);
        let last_word_index = self
            .tokens
            .iter()
            .filter_map(|token| token.word_index)
            .max();

        self.tokens
            .iter()
            .filter_map(|token| {
                let word_index = token.word_index?;
                let mut syntactic_links = primary_parse
                    .links
                    .iter()
                    .filter_map(|link| {
                        (link.left == word_index || link.right == word_index).then_some(link.kind)
                    })
                    .collect::<Vec<_>>();
                syntactic_links.sort_unstable_by_key(|kind| *kind as u8);
                syntactic_links.dedup();

                let claim_confidence = primary_parse
                    .claims
                    .iter()
                    .filter(|claim| targets_word(claim, word_index))
                    .map(|claim| claim.confidence)
                    .fold(0.0_f32, f32::max);
                let link_confidence = primary_parse
                    .links
                    .iter()
                    .filter(|link| link.left == word_index || link.right == word_index)
                    .map(|link| link.confidence)
                    .fold(0.0_f32, f32::max);
                let confidence = parse_rank_confidence.max(claim_confidence.max(link_confidence));

                let prosodic_role = if conservative_ambiguity {
                    token.prosodic_role
                } else {
                    claim_prosodic_role(primary_parse, word_index).unwrap_or(token.prosodic_role)
                };

                let phrase_boundary_after = if conservative_ambiguity {
                    PhraseBoundaryKind::None
                } else {
                    boundary_after_word(primary_parse, word_index)
                };
                let phrase_boundary_after = if !conservative_ambiguity
                    && Some(word_index) == last_word_index
                    && matches!(phrase_boundary_after, PhraseBoundaryKind::None)
                {
                    self.terminal_boundary_kind
                } else {
                    phrase_boundary_after
                };
                let mut lexical_flags = token
                    .reduction_diagnostic
                    .as_ref()
                    .map(|diagnostic| english_lexical_flag_facts_for_rule(&diagnostic.rule))
                    .unwrap_or_default();
                if Some(word_index) == last_word_index {
                    lexical_flags.extend(boundary_flag_facts_for_kind(phrase_boundary_after));
                }
                lexical_flags.sort_unstable_by(|left, right| {
                    left.source_rule_id
                        .cmp(&right.source_rule_id)
                        .then((left.flag as u8).cmp(&(right.flag as u8)))
                });
                lexical_flags.dedup_by(|left, right| {
                    left.source_rule_id == right.source_rule_id && left.flag == right.flag
                });

                Some(ProsodyEnvironmentFacts {
                    word_index,
                    pos: token.pos,
                    prosodic_role,
                    orthographic_emphasis: token.orthographic_emphasis,
                    phrase_boundary_after,
                    syntactic_links,
                    lexical_flags,
                    confidence,
                    conservative: conservative_ambiguity,
                })
            })
            .collect()
    }

    pub fn prosody_environment_facts_for_word(
        &self,
        word_index: usize,
    ) -> Option<ProsodyEnvironmentFacts> {
        self.prosody_environment_facts()
            .into_iter()
            .find(|facts| facts.word_index == word_index)
    }

    fn has_low_confidence_ambiguity(&self) -> bool {
        if self.link_parses.len() < 2 {
            return false;
        }
        let best = self.link_parses[0].rank;
        let second = self.link_parses[1].rank;
        (best - second).abs() <= 0.1 || best < 0.7
    }
}

fn targets_word(claim: &AnalysisClaim, word_index: usize) -> bool {
    match &claim.target {
        AnalysisTarget::WordIndex(index) => *index == word_index,
        AnalysisTarget::WordRange(range) => range.contains(&word_index),
        AnalysisTarget::Boundary { left_word, .. } => {
            left_word.is_some_and(|left| left == word_index)
        }
        _ => false,
    }
}

fn claim_prosodic_role(parse: &SyntacticLinkParse, word_index: usize) -> Option<ProsodicRole> {
    parse
        .claims
        .iter()
        .filter(|claim| {
            claim.kind == ClaimKind::ProsodicRole
                && claim.target == AnalysisTarget::WordIndex(word_index)
        })
        .max_by(|left, right| {
            left.confidence
                .partial_cmp(&right.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .and_then(|claim| match &claim.value {
            ClaimValue::ProsodicRole(role) => parse_prosodic_role(role),
            _ => None,
        })
}

fn parse_prosodic_role(role: &str) -> Option<ProsodicRole> {
    match role {
        "Content" => Some(ProsodicRole::Content),
        "FunctionWeak" => Some(ProsodicRole::FunctionWeak),
        "FunctionStrong" => Some(ProsodicRole::FunctionStrong),
        "Contrastive" => Some(ProsodicRole::Contrastive),
        "Focus" => Some(ProsodicRole::Focus),
        "DirectAddress" => Some(ProsodicRole::DirectAddress),
        _ => None,
    }
}

fn boundary_after_word(parse: &SyntacticLinkParse, word_index: usize) -> PhraseBoundaryKind {
    let claimed = parse
        .claims
        .iter()
        .filter_map(|claim| match (&claim.target, &claim.value) {
            (
                AnalysisTarget::Boundary {
                    left_word: Some(left),
                    ..
                },
                ClaimValue::BoundaryKind(boundary),
            ) if *left == word_index => Some((claim.confidence, boundary.as_str())),
            _ => None,
        })
        .max_by(|left, right| {
            left.0
                .partial_cmp(&right.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .and_then(|(_, boundary)| match boundary {
            "VocativeCommaPauseSuppressed" => Some(PhraseBoundaryKind::Vocative),
            "AppositivePhrase" => Some(PhraseBoundaryKind::MinorPhrase),
            "Coordination" => Some(PhraseBoundaryKind::MinorPhrase),
            "PrepositionalPhrase" => Some(PhraseBoundaryKind::MinorPhrase),
            "MajorPhrasePause" => Some(PhraseBoundaryKind::MajorPhrase),
            "MinorPhrasePause" => Some(PhraseBoundaryKind::MinorPhrase),
            _ => None,
        });
    if let Some(boundary) = claimed {
        return boundary;
    }
    if parse.links.iter().any(|link| {
        (link.left == word_index || link.right == word_index)
            && matches!(link.kind, SyntacticLinkKind::Parenthetical)
    }) {
        return PhraseBoundaryKind::Parenthetical;
    }
    if parse.links.iter().any(|link| {
        (link.left == word_index || link.right == word_index)
            && matches!(link.kind, SyntacticLinkKind::Apposition)
    }) {
        return PhraseBoundaryKind::MinorPhrase;
    }
    PhraseBoundaryKind::None
}

fn boundary_flag_facts_for_kind(kind: PhraseBoundaryKind) -> Vec<LexicalProsodyFlagFact> {
    match kind {
        PhraseBoundaryKind::Exclamation => {
            english_lexical_flag_facts_for_rule("punctuation_exclamation_boundary")
        }
        PhraseBoundaryKind::FinalRising => {
            english_lexical_flag_facts_for_rule("punctuation_question_rising_boundary")
        }
        _ => Vec::new(),
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
        if !word_slots_are_phrase_adjacent(normalized, word_slots[idx].0, word_slots[idx + 1].0) {
            continue;
        }

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
                    source: SyntacticLinkSource::HeuristicGrammarIsland,
                },
            );
            claims.push(AnalysisClaim::new(
                AnalysisTarget::WordIndex(idx),
                ClaimKind::InfinitivalMarker,
                ClaimValue::Syntactic,
                AnalysisSourceKind::SyntaxRule,
                INFINITIVAL_MARKER_CONFIDENCE,
                "to before likely verb is an infinitival marker",
            ));
            claims.push(AnalysisClaim::new(
                AnalysisTarget::WordIndex(idx),
                ClaimKind::WeakFunctionCandidate,
                ClaimValue::Syntactic,
                AnalysisSourceKind::SyntaxRule,
                WEAK_FUNCTION_CANDIDATE_CONFIDENCE,
                "infinitival to is a weak function word candidate",
            ));
        }

        if is_determiner(left) && is_likely_nominal(right) {
            push_link(
                &mut links,
                SyntacticLink {
                    left: idx,
                    right: idx + 1,
                    kind: SyntacticLinkKind::Determiner,
                    confidence: DETERMINER_LINK_CONFIDENCE,
                    source: SyntacticLinkSource::HeuristicGrammarIsland,
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
                    source: SyntacticLinkSource::HeuristicGrammarIsland,
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
                    source: SyntacticLinkSource::HeuristicGrammarIsland,
                },
            );
        }

        if is_noun_compound_pair(left, right) {
            push_link(
                &mut links,
                SyntacticLink {
                    left: idx,
                    right: idx + 1,
                    kind: SyntacticLinkKind::NounCompound,
                    confidence: NOUN_COMPOUND_LINK_CONFIDENCE,
                    source: SyntacticLinkSource::HeuristicGrammarIsland,
                },
            );
        }
    }

    push_determiner_phrase_links(normalized, word_slots, &words, &mut links, &mut claims);
    push_auxiliary_phrase_links(normalized, word_slots, &words, &mut links, &mut claims);
    push_prepositional_links(normalized, word_slots, &words, &mut links, &mut claims);
    push_core_clause_links(normalized, word_slots, &words, &mut links, &mut claims);
    push_coordination_links(normalized, word_slots, &words, &mut links, &mut claims);

    for (left, right) in detect_contrast_pairs(&words, source_words) {
        push_link(
            &mut links,
            SyntacticLink {
                left,
                right,
                kind: SyntacticLinkKind::ContrastPair,
                confidence: CONTRAST_PAIR_CONFIDENCE,
                source: SyntacticLinkSource::HeuristicGrammarIsland,
            },
        );
        claims.push(AnalysisClaim::new(
            AnalysisTarget::WordRange(vec![left, right]),
            ClaimKind::ContrastPair,
            ClaimValue::Syntactic,
            AnalysisSourceKind::SyntaxRule,
            CONTRAST_PAIR_CONFIDENCE,
            "contrastive negation pattern detected",
        ));
        for contrast_word in [left, right] {
            claims.push(AnalysisClaim::new(
                AnalysisTarget::WordIndex(contrast_word),
                ClaimKind::ProsodicRole,
                ClaimValue::ProsodicRole("Contrastive".to_string()),
                AnalysisSourceKind::SyntaxRule,
                CONTRASTIVE_FOCUS_CLAIM_CONFIDENCE,
                "contrast pair marks focused item",
            ));
            claims.push(AnalysisClaim::new(
                AnalysisTarget::WordIndex(contrast_word),
                ClaimKind::Reduction,
                ClaimValue::Reduction("WeakFormSuppressed".to_string()),
                AnalysisSourceKind::SyntaxRule,
                CONTRASTIVE_FOCUS_CLAIM_CONFIDENCE,
                "contrastive focus suppresses weak-form reduction",
            ));
        }
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
            let last_target = targets.last().copied().unwrap_or(first_target);
            let (link_left, link_right, boundary_left, boundary_right) = if first_target == 0 {
                let anchor = (last_target + 1 < words.len()).then_some(last_target + 1);
                (
                    first_target,
                    anchor.unwrap_or(last_target),
                    Some(last_target),
                    anchor,
                )
            } else {
                let anchor = first_target - 1;
                (anchor, first_target, Some(anchor), Some(first_target))
            };
            push_link(
                &mut links,
                SyntacticLink {
                    left: link_left,
                    right: link_right,
                    kind: SyntacticLinkKind::Vocative,
                    confidence: VOCATIVE_LINK_CONFIDENCE,
                    source: SyntacticLinkSource::HeuristicGrammarIsland,
                },
            );
            claims.push(AnalysisClaim::new(
                AnalysisTarget::WordIndex(first_target),
                ClaimKind::VocativeBoundary,
                ClaimValue::Syntactic,
                AnalysisSourceKind::SyntaxRule,
                VOCATIVE_LINK_CONFIDENCE,
                "comma-delimited proper name after verb is a vocative boundary",
            ));
            claims.push(AnalysisClaim::new(
                AnalysisTarget::WordIndex(first_target),
                ClaimKind::ProsodicRole,
                ClaimValue::ProsodicRole("DirectAddress".to_string()),
                AnalysisSourceKind::SyntaxRule,
                VOCATIVE_LINK_CONFIDENCE,
                "vocative addressee receives direct-address prosodic role",
            ));
            claims.push(AnalysisClaim::new(
                AnalysisTarget::Boundary {
                    left_word: boundary_left,
                    right_word: boundary_right,
                },
                ClaimKind::BoundaryKind,
                ClaimValue::BoundaryKind("VocativeCommaPauseSuppressed".to_string()),
                AnalysisSourceKind::SyntaxRule,
                COMMA_BEHAVIOR_CLAIM_CONFIDENCE,
                "vocative comma prefers reduced pause behavior",
            ));
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
        let is_place_apposition =
            is_place_name_apposition(&words, source_words, left_anchor, &between);
        if is_apposition || is_place_apposition {
            let target = between[0];
            push_link(
                &mut links,
                SyntacticLink {
                    left: left_anchor,
                    right: target,
                    kind: SyntacticLinkKind::Apposition,
                    confidence: APPOSITION_LINK_CONFIDENCE,
                    source: SyntacticLinkSource::HeuristicGrammarIsland,
                },
            );
            claims.push(AnalysisClaim::new(
                AnalysisTarget::WordRange(between.clone()),
                ClaimKind::AppositionBoundary,
                ClaimValue::Syntactic,
                AnalysisSourceKind::SyntaxRule,
                APPOSITION_LINK_CONFIDENCE,
                if is_place_apposition {
                    "place-name apposition between commas"
                } else {
                    "relative clause introduced by who/which/that/whom"
                },
            ));
            claims.push(AnalysisClaim::new(
                AnalysisTarget::Boundary {
                    left_word: left_break.1,
                    right_word: Some(target),
                },
                ClaimKind::BoundaryKind,
                ClaimValue::BoundaryKind("AppositivePhrase".to_string()),
                AnalysisSourceKind::SyntaxRule,
                APPOSITION_LINK_CONFIDENCE,
                "appositive comma uses phrase timing without parenthetical de-emphasis",
            ));
            if let Some(&last_between) = between.last() {
                claims.push(AnalysisClaim::new(
                    AnalysisTarget::Boundary {
                        left_word: Some(last_between),
                        right_word: right_break.2,
                    },
                    ClaimKind::BoundaryKind,
                    ClaimValue::BoundaryKind("AppositivePhrase".to_string()),
                    AnalysisSourceKind::SyntaxRule,
                    APPOSITION_LINK_CONFIDENCE,
                    "appositive comma uses phrase timing without parenthetical de-emphasis",
                ));
            }
            continue;
        }
        if let Some(right_anchor) = right_break.2
            && is_parenthetical_comma_island(&words, &between)
        {
            push_link(
                &mut links,
                SyntacticLink {
                    left: left_anchor,
                    right: right_anchor,
                    kind: SyntacticLinkKind::Parenthetical,
                    confidence: PARENTHETICAL_LINK_CONFIDENCE,
                    source: SyntacticLinkSource::HeuristicGrammarIsland,
                },
            );
            claims.push(AnalysisClaim::new(
                AnalysisTarget::WordRange(between.clone()),
                ClaimKind::ParentheticalBoundary,
                ClaimValue::Syntactic,
                AnalysisSourceKind::SyntaxRule,
                PARENTHETICAL_LINK_CONFIDENCE,
                "parenthetical phrase between commas",
            ));
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
                source: SyntacticLinkSource::AmbiguityVariant,
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
                source: SyntacticLinkSource::AmbiguityVariant,
            },
        );
        return vec![noun_parse, verb_parse];
    }

    vec![primary_parse]
}

fn push_core_clause_links(
    normalized: &NormalizedText,
    word_slots: &[(usize, String)],
    words: &[&str],
    links: &mut Vec<SyntacticLink>,
    claims: &mut Vec<AnalysisClaim>,
) {
    for predicate_index in 0..words.len() {
        let word = words[predicate_index];
        let predicate_can_take_subject =
            is_likely_verb(word) || is_auxiliary_predicate_head(words, predicate_index);
        if predicate_can_take_subject
            && let Some(subject_index) =
                find_subject_before_predicate(normalized, word_slots, words, predicate_index)
        {
            push_link(
                links,
                SyntacticLink {
                    left: subject_index,
                    right: predicate_index,
                    kind: SyntacticLinkKind::Subject,
                    confidence: SUBJECT_LINK_CONFIDENCE,
                    source: SyntacticLinkSource::HeuristicGrammarIsland,
                },
            );
            claims.push(AnalysisClaim::new(
                AnalysisTarget::WordIndex(subject_index),
                ClaimKind::ProsodicRole,
                ClaimValue::ProsodicRole("Content".to_string()),
                AnalysisSourceKind::SyntaxRule,
                CORE_CLAUSE_CLAIM_CONFIDENCE,
                "nominal before predicate linked as clause subject",
            ));
        }

        if is_likely_verb(word)
            && let Some(object_index) =
                find_object_after_verb(normalized, word_slots, words, predicate_index)
        {
            push_link(
                links,
                SyntacticLink {
                    left: predicate_index,
                    right: object_index,
                    kind: SyntacticLinkKind::Object,
                    confidence: OBJECT_LINK_CONFIDENCE,
                    source: SyntacticLinkSource::HeuristicGrammarIsland,
                },
            );
            claims.push(AnalysisClaim::new(
                AnalysisTarget::WordIndex(object_index),
                ClaimKind::ProsodicRole,
                ClaimValue::ProsodicRole("Focus".to_string()),
                AnalysisSourceKind::SyntaxRule,
                CORE_CLAUSE_CLAIM_CONFIDENCE,
                "nominal after verb linked as likely object/focus",
            ));
        }

        if let Some(complement_index) =
            find_complement_after_predicate(normalized, word_slots, words, predicate_index)
        {
            push_link(
                links,
                SyntacticLink {
                    left: predicate_index,
                    right: complement_index,
                    kind: SyntacticLinkKind::Complement,
                    confidence: COMPLEMENT_LINK_CONFIDENCE,
                    source: SyntacticLinkSource::HeuristicGrammarIsland,
                },
            );
            claims.push(AnalysisClaim::new(
                AnalysisTarget::WordIndex(complement_index),
                ClaimKind::ProsodicRole,
                ClaimValue::ProsodicRole("Focus".to_string()),
                AnalysisSourceKind::SyntaxRule,
                CORE_CLAUSE_CLAIM_CONFIDENCE,
                "predicate complement linked for focus planning",
            ));
        }
    }
}

fn push_coordination_links(
    normalized: &NormalizedText,
    word_slots: &[(usize, String)],
    words: &[&str],
    links: &mut Vec<SyntacticLink>,
    claims: &mut Vec<AnalysisClaim>,
) {
    for conjunction_index in 1..words.len().saturating_sub(1) {
        if !is_coordination_conjunction(words[conjunction_index]) {
            continue;
        }
        if !word_indices_are_phrase_adjacent(
            normalized,
            word_slots,
            conjunction_index - 1,
            conjunction_index,
        ) || !word_indices_are_phrase_adjacent(
            normalized,
            word_slots,
            conjunction_index,
            conjunction_index + 1,
        ) {
            continue;
        }
        let Some(left) = find_coordination_item_left(words, conjunction_index) else {
            continue;
        };
        let Some(right) = find_coordination_item_right(words, conjunction_index) else {
            continue;
        };
        if base_pos(words[left]) != base_pos(words[right])
            && !(is_likely_nominal(words[left]) && is_likely_nominal(words[right]))
            && !(is_likely_verb(words[left]) && is_likely_verb(words[right]))
        {
            continue;
        }
        push_link(
            links,
            SyntacticLink {
                left,
                right,
                kind: SyntacticLinkKind::Coordination,
                confidence: COORDINATION_LINK_CONFIDENCE,
                source: SyntacticLinkSource::HeuristicGrammarIsland,
            },
        );
        claims.push(AnalysisClaim::new(
            AnalysisTarget::Boundary {
                left_word: Some(left),
                right_word: Some(right),
            },
            ClaimKind::BoundaryKind,
            ClaimValue::BoundaryKind("Coordination".to_string()),
            AnalysisSourceKind::SyntaxRule,
            COORDINATION_CLAIM_CONFIDENCE,
            "coordinated word pair linked across conjunction",
        ));
    }
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

fn is_parenthetical_comma_island(words: &[&str], between: &[usize]) -> bool {
    let Some(&first) = between.first() else {
        return false;
    };
    let Some(first_word) = words.get(first).copied() else {
        return false;
    };
    PARENTHETICAL_CUE_WORDS.contains(&first_word)
}

fn is_place_name_apposition(
    words: &[&str],
    source_words: &[String],
    left_anchor: usize,
    between: &[usize],
) -> bool {
    if between.is_empty() || !source_word_looks_like_proper_name(source_words, left_anchor) {
        return false;
    }
    let place_words = between
        .iter()
        .filter_map(|idx| words.get(*idx).copied())
        .collect::<Vec<_>>();
    if place_words.is_empty() || place_words.len() > 3 {
        return false;
    }
    let joined = place_words.join(" ");
    US_STATE_NAMES.contains(&joined.as_str())
        || PLACE_APPOSITION_WORDS.contains(&joined.as_str())
        || place_words
            .iter()
            .all(|word| PLACE_APPOSITION_WORDS.contains(word))
}

fn source_word_looks_like_proper_name(source_words: &[String], word_index: usize) -> bool {
    source_words.get(word_index).is_some_and(|word| {
        word.chars()
            .find(|ch| ch.is_ascii_alphabetic())
            .is_some_and(|ch| ch.is_ascii_uppercase())
    })
}

fn push_determiner_phrase_links(
    normalized: &NormalizedText,
    word_slots: &[(usize, String)],
    words: &[&str],
    links: &mut Vec<SyntacticLink>,
    claims: &mut Vec<AnalysisClaim>,
) {
    for determiner_index in 0..words.len() {
        let determiner = words[determiner_index];
        if !is_determiner(determiner) {
            continue;
        }
        let Some(head_index) =
            find_nominal_head_after_determiner(normalized, word_slots, words, determiner_index)
        else {
            continue;
        };
        push_link(
            links,
            SyntacticLink {
                left: determiner_index,
                right: head_index,
                kind: SyntacticLinkKind::Determiner,
                confidence: DETERMINER_LINK_CONFIDENCE,
                source: SyntacticLinkSource::HeuristicGrammarIsland,
            },
        );
        push_determiner_claims(
            claims,
            determiner,
            determiner_index,
            head_index,
            words[head_index],
        );
    }
}

fn push_determiner_claims(
    claims: &mut Vec<AnalysisClaim>,
    determiner: &str,
    determiner_index: usize,
    head_index: usize,
    head: &str,
) {
    if !matches!(determiner, "a" | "an" | "the") {
        return;
    }
    let article_noun_range = vec![determiner_index, head_index];
    let onset = if has_graphemic_vowel_onset(head) {
        "vowel"
    } else {
        "consonant"
    };
    claims.push(AnalysisClaim::new(
        AnalysisTarget::WordRange(article_noun_range.clone()),
        ClaimKind::MorphologicalForm,
        ClaimValue::MorphologicalForm(format!(
            "article_phonetic_agreement:{determiner}_before_{onset}"
        )),
        AnalysisSourceKind::SyntaxRule,
        ARTICLE_HOOK_CLAIM_CONFIDENCE,
        "article+noun pair emitted for phonetic agreement/allomorph selection hooks",
    ));
    if determiner == "the" {
        let realization = if onset == "vowel" {
            "the_before_vowel:DH_IY0"
        } else {
            "the_before_consonant:DH_AH0"
        };
        claims.push(AnalysisClaim::new(
            AnalysisTarget::WordRange(article_noun_range),
            ClaimKind::PhonemeRealization,
            ClaimValue::PhonemeRealization(realization.to_string()),
            AnalysisSourceKind::SyntaxRule,
            ARTICLE_HOOK_CLAIM_CONFIDENCE,
            "the+noun pair emitted for weak/strong allomorph selection hooks",
        ));
    }
    claims.push(AnalysisClaim::new(
        AnalysisTarget::WordIndex(determiner_index),
        ClaimKind::WeakFunctionCandidate,
        ClaimValue::Syntactic,
        AnalysisSourceKind::SyntaxRule,
        WEAK_FUNCTION_CANDIDATE_CONFIDENCE,
        "article is a weak-form candidate unless contrastively focused",
    ));
}

fn push_auxiliary_phrase_links(
    normalized: &NormalizedText,
    word_slots: &[(usize, String)],
    words: &[&str],
    links: &mut Vec<SyntacticLink>,
    claims: &mut Vec<AnalysisClaim>,
) {
    for auxiliary_index in 0..words.len() {
        if !is_auxiliary(words[auxiliary_index]) {
            continue;
        }
        let Some(verb_index) =
            find_verb_head_after_auxiliary(normalized, word_slots, words, auxiliary_index)
        else {
            continue;
        };
        push_link(
            links,
            SyntacticLink {
                left: auxiliary_index,
                right: verb_index,
                kind: SyntacticLinkKind::Auxiliary,
                confidence: AUXILIARY_LINK_CONFIDENCE,
                source: SyntacticLinkSource::HeuristicGrammarIsland,
            },
        );
        push_auxiliary_claims(claims, auxiliary_index);
    }
}

fn push_auxiliary_claims(claims: &mut Vec<AnalysisClaim>, auxiliary_index: usize) {
    claims.push(AnalysisClaim::new(
        AnalysisTarget::WordIndex(auxiliary_index),
        ClaimKind::ProsodicRole,
        ClaimValue::ProsodicRole("FunctionWeak".to_string()),
        AnalysisSourceKind::SyntaxRule,
        AUXILIARY_LINK_CONFIDENCE,
        "auxiliary linked to predicate head for de-emphasis planning",
    ));
    claims.push(AnalysisClaim::new(
        AnalysisTarget::WordIndex(auxiliary_index),
        ClaimKind::WeakFunctionCandidate,
        ClaimValue::Syntactic,
        AnalysisSourceKind::SyntaxRule,
        WEAK_FUNCTION_CANDIDATE_CONFIDENCE,
        "auxiliary is a weak-form candidate unless contrastively focused",
    ));
}

fn push_prepositional_links(
    normalized: &NormalizedText,
    word_slots: &[(usize, String)],
    words: &[&str],
    links: &mut Vec<SyntacticLink>,
    claims: &mut Vec<AnalysisClaim>,
) {
    for preposition_index in 0..words.len() {
        let preposition = words[preposition_index];
        if !is_preposition(preposition) {
            continue;
        }
        let Some(object_index) =
            find_prepositional_object(normalized, word_slots, words, preposition_index)
        else {
            continue;
        };
        push_link(
            links,
            SyntacticLink {
                left: preposition_index,
                right: object_index,
                kind: SyntacticLinkKind::Preposition,
                confidence: PREPOSITION_LINK_CONFIDENCE,
                source: SyntacticLinkSource::HeuristicGrammarIsland,
            },
        );
        claims.push(AnalysisClaim::new(
            AnalysisTarget::WordIndex(preposition_index),
            ClaimKind::WeakFunctionCandidate,
            ClaimValue::Syntactic,
            AnalysisSourceKind::SyntaxRule,
            WEAK_FUNCTION_CANDIDATE_CONFIDENCE,
            "preposition is a weak-function candidate before its object",
        ));
        claims.push(AnalysisClaim::new(
            AnalysisTarget::Boundary {
                left_word: Some(preposition_index),
                right_word: Some(object_index),
            },
            ClaimKind::BoundaryKind,
            ClaimValue::BoundaryKind("PrepositionalPhrase".to_string()),
            AnalysisSourceKind::SyntaxRule,
            PREPOSITION_LINK_CONFIDENCE,
            "preposition linked to object for phrase grouping",
        ));
    }
}

fn find_nominal_head_after_determiner(
    normalized: &NormalizedText,
    word_slots: &[(usize, String)],
    words: &[&str],
    determiner_index: usize,
) -> Option<usize> {
    let mut head = None;
    for (index, &word) in words.iter().enumerate().skip(determiner_index + 1) {
        if !word_indices_are_in_same_phrase(normalized, word_slots, determiner_index, index) {
            break;
        }
        if is_adjective(word) {
            continue;
        }
        if is_likely_nominal(word) {
            head = Some(index);
            continue;
        }
        break;
    }
    head
}

fn find_verb_head_after_auxiliary(
    normalized: &NormalizedText,
    word_slots: &[(usize, String)],
    words: &[&str],
    auxiliary_index: usize,
) -> Option<usize> {
    for (index, &word) in words.iter().enumerate().skip(auxiliary_index + 1) {
        if !word_indices_are_in_same_phrase(normalized, word_slots, auxiliary_index, index) {
            break;
        }
        if word == "not" || is_adverb(word) {
            continue;
        }
        if is_likely_verb(word) {
            return Some(index);
        }
        break;
    }
    None
}

fn find_prepositional_object(
    normalized: &NormalizedText,
    word_slots: &[(usize, String)],
    words: &[&str],
    preposition_index: usize,
) -> Option<usize> {
    let mut object_head = None;
    for (index, &word) in words.iter().enumerate().skip(preposition_index + 1) {
        if !word_indices_are_in_same_phrase(normalized, word_slots, preposition_index, index) {
            break;
        }
        if is_determiner(word) || is_adjective(word) {
            continue;
        }
        if is_likely_nominal(word) {
            object_head = Some(index);
            continue;
        }
        break;
    }
    object_head
}

fn word_indices_are_phrase_adjacent(
    normalized: &NormalizedText,
    word_slots: &[(usize, String)],
    left_word_index: usize,
    right_word_index: usize,
) -> bool {
    word_slots
        .get(left_word_index)
        .zip(word_slots.get(right_word_index))
        .is_some_and(|((left_token_index, _), (right_token_index, _))| {
            word_slots_are_phrase_adjacent(normalized, *left_token_index, *right_token_index)
        })
}

fn word_indices_are_in_same_phrase(
    normalized: &NormalizedText,
    word_slots: &[(usize, String)],
    left_word_index: usize,
    right_word_index: usize,
) -> bool {
    let (left, right) = if left_word_index <= right_word_index {
        (left_word_index, right_word_index)
    } else {
        (right_word_index, left_word_index)
    };
    word_indices_are_phrase_adjacent(normalized, word_slots, left, right)
}

fn word_slots_are_phrase_adjacent(
    normalized: &NormalizedText,
    left_token_index: usize,
    right_token_index: usize,
) -> bool {
    normalized.tokens[left_token_index + 1..right_token_index]
        .iter()
        .all(|token| !matches!(token, NormalizedToken::PhraseBreak))
}

fn find_subject_before_predicate(
    normalized: &NormalizedText,
    word_slots: &[(usize, String)],
    words: &[&str],
    predicate_index: usize,
) -> Option<usize> {
    if predicate_index > 0
        && is_likely_verb(words[predicate_index])
        && is_auxiliary(words[predicate_index - 1])
        && word_indices_are_phrase_adjacent(
            normalized,
            word_slots,
            predicate_index - 1,
            predicate_index,
        )
    {
        return None;
    }

    for index in (0..predicate_index).rev() {
        if !word_indices_are_in_same_phrase(normalized, word_slots, index, predicate_index) {
            break;
        }
        let word = words[index];
        if word == "to" || is_preposition(word) || is_coordination_conjunction(word) {
            break;
        }
        if is_likely_nominal(word) {
            return Some(index);
        }
        if is_likely_verb(word) || is_auxiliary(word) {
            break;
        }
    }
    None
}

fn find_object_after_verb(
    normalized: &NormalizedText,
    word_slots: &[(usize, String)],
    words: &[&str],
    verb_index: usize,
) -> Option<usize> {
    let mut object_head = None;
    let mut saw_noun_phrase_material = false;
    for (index, &word) in words.iter().enumerate().skip(verb_index + 1) {
        if !word_indices_are_in_same_phrase(normalized, word_slots, verb_index, index) {
            break;
        }
        if word == "to" || is_preposition(word) || is_conjunction(word) {
            break;
        }
        if is_determiner(word) || is_adjective(word) {
            saw_noun_phrase_material = true;
            continue;
        }
        if is_likely_nominal(word) {
            object_head = Some(index);
            saw_noun_phrase_material = true;
            continue;
        }
        if saw_noun_phrase_material {
            break;
        }
    }
    object_head
}

fn find_complement_after_predicate(
    normalized: &NormalizedText,
    word_slots: &[(usize, String)],
    words: &[&str],
    predicate_index: usize,
) -> Option<usize> {
    if is_copular_auxiliary(words[predicate_index]) {
        let next_index = predicate_index + 1;
        if words
            .get(next_index)
            .is_some_and(|word| is_likely_verb(word))
            && word_indices_are_phrase_adjacent(normalized, word_slots, predicate_index, next_index)
        {
            return None;
        }
        return find_predicate_complement_head(normalized, word_slots, words, predicate_index);
    }

    if is_likely_verb(words[predicate_index])
        && predicate_index + 2 < words.len()
        && words[predicate_index + 1] == "to"
        && is_likely_verb(words[predicate_index + 2])
        && word_indices_are_phrase_adjacent(
            normalized,
            word_slots,
            predicate_index,
            predicate_index + 1,
        )
        && word_indices_are_phrase_adjacent(
            normalized,
            word_slots,
            predicate_index + 1,
            predicate_index + 2,
        )
    {
        return Some(predicate_index + 2);
    }

    None
}

fn find_predicate_complement_head(
    normalized: &NormalizedText,
    word_slots: &[(usize, String)],
    words: &[&str],
    predicate_index: usize,
) -> Option<usize> {
    let mut complement_head = None;
    for (index, &word) in words.iter().enumerate().skip(predicate_index + 1) {
        if !word_indices_are_in_same_phrase(normalized, word_slots, predicate_index, index) {
            break;
        }
        if is_preposition(word) || is_conjunction(word) || word == "to" {
            break;
        }
        if is_determiner(word) {
            continue;
        }
        if is_adjective(word) {
            return Some(index);
        }
        if is_likely_nominal(word) {
            complement_head = Some(index);
            continue;
        }
        if complement_head.is_some() {
            break;
        }
    }
    complement_head
}

fn find_coordination_item_left(words: &[&str], conjunction_index: usize) -> Option<usize> {
    (0..conjunction_index)
        .rev()
        .find(|index| is_coordination_item(words[*index]))
}

fn find_coordination_item_right(words: &[&str], conjunction_index: usize) -> Option<usize> {
    ((conjunction_index + 1)..words.len()).find(|index| is_coordination_item(words[*index]))
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

fn is_all_caps_abbreviation(raw_token: &str, normalized_token: &str) -> bool {
    is_all_caps_token(raw_token) && matches!(normalized_token, "us" | "uk" | "eu" | "un")
}

fn is_all_caps_acronym(raw_token: &str, normalized_token: &str) -> bool {
    is_all_caps_token(raw_token) && matches!(normalized_token, "usa" | "nato" | "fbi" | "cia")
}

fn is_capitalized_name_token(raw_token: &str) -> bool {
    let mut chars = raw_token.chars().filter(|ch| ch.is_ascii_alphabetic());
    let Some(first) = chars.next() else {
        return false;
    };
    first.is_ascii_uppercase() && chars.all(|ch| ch.is_ascii_lowercase())
}

fn classify_orthographic_emphasis(
    raw_token: &str,
    normalized_token: &str,
    base_pos: PartOfSpeech,
) -> OrthographicEmphasisKind {
    if raw_token.contains('/') || raw_token.contains('@') {
        return OrthographicEmphasisKind::ExplicitCitationForm;
    }
    if is_all_caps_abbreviation(raw_token, normalized_token) {
        return OrthographicEmphasisKind::Abbreviation;
    }
    if is_all_caps_acronym(raw_token, normalized_token) {
        return OrthographicEmphasisKind::Acronym;
    }
    if is_all_caps_token(raw_token) && raw_token.len() > 1 {
        return OrthographicEmphasisKind::AllCapsEmphasis;
    }
    if matches!(base_pos, PartOfSpeech::ProperName) && is_capitalized_name_token(raw_token) {
        return OrthographicEmphasisKind::CapitalizedName;
    }
    OrthographicEmphasisKind::None
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

fn is_noun_compound_pair(left: &str, right: &str) -> bool {
    matches!(
        base_pos(left),
        PartOfSpeech::Noun | PartOfSpeech::ProperName
    ) && matches!(
        base_pos(right),
        PartOfSpeech::Noun | PartOfSpeech::ProperName
    )
}

fn is_modifier_pair(left: &str, right: &str) -> bool {
    let left_pos = base_pos(left);
    (matches!(left_pos, PartOfSpeech::Adverb) && is_likely_verb(right))
        || (matches!(left_pos, PartOfSpeech::Adjective) && is_likely_nominal(right))
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
    OrthographicEmphasisKind,
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
            OrthographicEmphasisKind::AllCapsEmphasis,
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
            OrthographicEmphasisKind::ExplicitCitationForm,
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
            OrthographicEmphasisKind::None,
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
            OrthographicEmphasisKind::None,
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
            OrthographicEmphasisKind::None,
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
            OrthographicEmphasisKind::None,
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
        OrthographicEmphasisKind::None,
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
    if is_preposition(word) {
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
    if is_adverb(word) {
        return PartOfSpeech::Adverb;
    }
    if is_adjective(word) {
        return PartOfSpeech::Adjective;
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
        || is_preposition(word)
}

fn is_pronoun(word: &str) -> bool {
    matches!(
        word,
        "i" | "i'm"
            | "you"
            | "he"
            | "she"
            | "it"
            | "we"
            | "they"
            | "me"
            | "him"
            | "her"
            | "us"
            | "them"
    )
}

fn is_determiner(word: &str) -> bool {
    matches!(
        word,
        "a" | "an"
            | "the"
            | "this"
            | "that"
            | "these"
            | "those"
            | "my"
            | "your"
            | "his"
            | "her"
            | "its"
            | "our"
            | "their"
            | "whose"
            | "which"
            | "what"
    )
}

fn is_conjunction(word: &str) -> bool {
    matches!(word, "and" | "or" | "but" | "not")
}

fn is_coordination_conjunction(word: &str) -> bool {
    matches!(word, "and" | "or" | "but")
}

fn is_preposition(word: &str) -> bool {
    matches!(
        word,
        "to" | "for"
            | "from"
            | "of"
            | "with"
            | "by"
            | "as"
            | "in"
            | "on"
            | "at"
            | "about"
            | "before"
            | "after"
            | "through"
            | "over"
            | "under"
            | "into"
            | "onto"
            | "around"
            | "near"
            | "between"
            | "during"
            | "without"
            | "within"
    )
}

fn is_adverb(word: &str) -> bool {
    matches!(word, "then") || word.ends_with("ly")
}

fn is_adjective(word: &str) -> bool {
    matches!(word, "long" | "other")
        || COMMON_LINK_ADJECTIVES.contains(&word)
        || word.ends_with("ous")
        || word.ends_with("ive")
        || word.ends_with("al")
        || word.ends_with("ic")
        || word.ends_with("ated")
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

fn is_auxiliary_predicate_head(words: &[&str], index: usize) -> bool {
    is_auxiliary(words[index])
        && words.get(index + 1).is_some_and(|next| {
            is_likely_verb(next) || is_adjective(next) || is_likely_nominal(next)
        })
}

fn is_copular_auxiliary(word: &str) -> bool {
    matches!(word, "be" | "am" | "is" | "are" | "was" | "were" | "been")
}

fn is_coordination_item(word: &str) -> bool {
    !is_determiner(word)
        && !is_preposition(word)
        && !is_conjunction(word)
        && (is_likely_nominal(word) || is_likely_verb(word) || is_adjective(word))
}

fn is_likely_verb(word: &str) -> bool {
    matches!(
        word,
        "go" | "leave"
            | "remember"
            | "see"
            | "saw"
            | "stay"
            | "be"
            | "try"
            | "need"
            | "want"
            | "make"
            | "take"
            | "distinguish"
            | "add"
            | "adjust"
            | "change"
            | "follow"
            | "know"
            | "say"
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
    const COMMON_NON_VERB_ING: &[&str] = &["thing", "king", "morning", "ceiling", "timing"];
    const COMMON_NON_VERB_ED: &[&str] = &["red", "bed", "sled", "unpunctuated"];
    (word.len() >= 5 && word.ends_with("ing") && !COMMON_NON_VERB_ING.contains(&word))
        || (word.len() >= 4 && word.ends_with("ed") && !COMMON_NON_VERB_ED.contains(&word))
}

fn has_graphemic_vowel_onset(word: &str) -> bool {
    word.chars()
        .next()
        .is_some_and(|ch| matches!(ch.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u'))
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
                .any(|claim| claim.kind == ClaimKind::InfinitivalMarker
                    && claim.target == AnalysisTarget::WordIndex(to))
        );
        assert!(
            parse
                .claims
                .iter()
                .any(|claim| claim.kind == ClaimKind::WeakFunctionCandidate
                    && claim.target == AnalysisTarget::WordIndex(to))
        );
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
            claim.kind == ClaimKind::ContrastPair
                && claim.target == AnalysisTarget::WordRange(vec![to, from])
        }));
        assert!(parse.claims.iter().any(|claim| {
            claim.kind == ClaimKind::ProsodicRole
                && claim.target == AnalysisTarget::WordIndex(to)
                && claim.value == ClaimValue::ProsodicRole("Contrastive".to_string())
        }));
        assert!(parse.claims.iter().any(|claim| {
            claim.kind == ClaimKind::Reduction
                && claim.target == AnalysisTarget::WordIndex(from)
                && claim.value == ClaimValue::Reduction("WeakFormSuppressed".to_string())
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
                .any(|claim| claim.kind == ClaimKind::VocativeBoundary
                    && claim.target == AnalysisTarget::WordIndex(dave))
        );
        assert!(parse.claims.iter().any(|claim| {
            claim.kind == ClaimKind::ProsodicRole
                && claim.target == AnalysisTarget::WordIndex(dave)
                && claim.value == ClaimValue::ProsodicRole("DirectAddress".to_string())
        }));
        assert!(parse.claims.iter().any(|claim| {
            claim.kind == ClaimKind::BoundaryKind
                && claim.value
                    == ClaimValue::BoundaryKind("VocativeCommaPauseSuppressed".to_string())
        }));
    }

    #[test]
    fn fixture_links_initial_direct_address_boundary() {
        let analysis = analyze("Dave, thank you.");
        let parse = analysis.link_parses.first().expect("link parse");
        let dave = word_index(&analysis, "dave");
        let thank = word_index(&analysis, "thank");
        assert!(has_link(parse, dave, thank, SyntacticLinkKind::Vocative));
        assert!(parse.claims.iter().any(|claim| {
            claim.kind == ClaimKind::VocativeBoundary
                && claim.target == AnalysisTarget::WordIndex(dave)
        }));
        assert!(parse.claims.iter().any(|claim| {
            claim.kind == ClaimKind::BoundaryKind
                && claim.target
                    == (AnalysisTarget::Boundary {
                        left_word: Some(dave),
                        right_word: Some(thank),
                    })
                && claim.value
                    == ClaimValue::BoundaryKind("VocativeCommaPauseSuppressed".to_string())
        }));
    }

    #[test]
    fn fixture_links_comma_surrounded_direct_address_without_parenthetical() {
        let analysis = analyze("Thank you, Dave, I appreciate it.");
        let parse = analysis.link_parses.first().expect("link parse");
        let you = word_index(&analysis, "you");
        let dave = word_index(&analysis, "dave");
        assert!(has_link(parse, you, dave, SyntacticLinkKind::Vocative));
        assert!(
            !parse
                .links
                .iter()
                .any(|link| link.kind == SyntacticLinkKind::Parenthetical)
        );
        let facts = analysis
            .prosody_environment_facts_for_word(you)
            .expect("you facts");
        assert_eq!(facts.phrase_boundary_after, PhraseBoundaryKind::Vocative);
    }

    #[test]
    fn detects_vocative_span_boundaries() {
        let spans = detect_vocative_spans("Thank you, Dave.");
        assert_eq!(spans.len(), 1);
        assert_eq!("Dave", &"Thank you, Dave."[spans[0].clone()]);

        let initial_spans = detect_vocative_spans("Dave, thank you.");
        assert_eq!(initial_spans.len(), 1);
        assert_eq!("Dave", &"Dave, thank you."[initial_spans[0].clone()]);

        let medial_spans = detect_vocative_spans("Listen, Dave, this matters.");
        assert_eq!(medial_spans.len(), 1);
        assert_eq!(
            "Dave",
            &"Listen, Dave, this matters."[medial_spans[0].clone()]
        );

        let greeting_spans = detect_vocative_spans("Hello, Dave.");
        assert_eq!(greeting_spans.len(), 1);
        assert_eq!("Dave", &"Hello, Dave."[greeting_spans[0].clone()]);

        let greeting_pair_spans = detect_vocative_spans("Hey, Dave, listen.");
        assert_eq!(greeting_pair_spans.len(), 1);
        assert_eq!(
            "Dave",
            &"Hey, Dave, listen."[greeting_pair_spans[0].clone()]
        );
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
                .any(|claim| claim.kind == ClaimKind::ParentheticalBoundary)
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
                .any(|claim| claim.kind == ClaimKind::AppositionBoundary)
        );
    }

    #[test]
    fn fixture_keeps_place_apposition_out_of_parenthetical_prosody() {
        let analysis = analyze("Seattle, Washington, a great city");
        let parse = analysis.link_parses.first().expect("link parse");
        let seattle = word_index(&analysis, "seattle");
        let washington = word_index(&analysis, "washington");
        assert!(has_link(
            parse,
            seattle,
            washington,
            SyntacticLinkKind::Apposition
        ));
        assert!(
            !parse
                .links
                .iter()
                .any(|link| link.kind == SyntacticLinkKind::Parenthetical)
        );
        let seattle_facts = analysis
            .prosody_environment_facts_for_word(seattle)
            .expect("seattle facts");
        let washington_facts = analysis
            .prosody_environment_facts_for_word(washington)
            .expect("washington facts");
        assert_eq!(
            seattle_facts.phrase_boundary_after,
            PhraseBoundaryKind::MinorPhrase
        );
        assert_eq!(
            washington_facts.phrase_boundary_after,
            PhraseBoundaryKind::MinorPhrase
        );
    }

    #[test]
    fn fixture_emits_article_phonetic_agreement_hooks() {
        let a_dog = analyze("a dog");
        let a_dog_parse = a_dog.link_parses.first().expect("link parse");
        let a = word_index(&a_dog, "a");
        let dog = word_index(&a_dog, "dog");
        assert!(has_link(a_dog_parse, a, dog, SyntacticLinkKind::Determiner));
        assert!(a_dog_parse.claims.iter().any(|claim| {
            claim.kind == ClaimKind::MorphologicalForm
                && claim.target == AnalysisTarget::WordRange(vec![a, dog])
                && claim.value
                    == ClaimValue::MorphologicalForm(
                        "article_phonetic_agreement:a_before_consonant".to_string(),
                    )
        }));

        let an_owl = analyze("an owl");
        let an_owl_parse = an_owl.link_parses.first().expect("link parse");
        let an = word_index(&an_owl, "an");
        let owl = word_index(&an_owl, "owl");
        assert!(has_link(
            an_owl_parse,
            an,
            owl,
            SyntacticLinkKind::Determiner
        ));
        assert!(an_owl_parse.claims.iter().any(|claim| {
            claim.kind == ClaimKind::MorphologicalForm
                && claim.target == AnalysisTarget::WordRange(vec![an, owl])
                && claim.value
                    == ClaimValue::MorphologicalForm(
                        "article_phonetic_agreement:an_before_vowel".to_string(),
                    )
        }));

        let the_owl = analyze("the owl");
        let the_owl_parse = the_owl.link_parses.first().expect("link parse");
        let the = word_index(&the_owl, "the");
        assert!(the_owl_parse.claims.iter().any(|claim| {
            claim.kind == ClaimKind::WeakFunctionCandidate
                && claim.target == AnalysisTarget::WordIndex(the)
        }));

        let the_door = analyze("the door");
        let the_door_parse = the_door.link_parses.first().expect("link parse");
        let the_door_idx = word_index(&the_door, "the");
        let door = word_index(&the_door, "door");
        assert!(has_link(
            the_door_parse,
            the_door_idx,
            door,
            SyntacticLinkKind::Determiner
        ));
        assert!(
            the_door
                .environment_patterns()
                .iter()
                .any(|pattern| pattern
                    .predicates
                    .contains(&ContextPredicate::SyntacticLink(
                        SyntacticLinkKind::Determiner
                    )))
        );
    }

    #[test]
    fn fixture_covers_remaining_infinitival_and_vocative_examples_with_diagnostics() {
        for text in ["We need to leave.", "Try to remember."] {
            let analysis = analyze(text);
            let parse = analysis.link_parses.first().expect("link parse");
            let to = word_index(&analysis, "to");
            assert!(parse.links.iter().any(|link| {
                link.left == to && link.kind == SyntacticLinkKind::InfinitivalMarker
            }));
            assert!(parse.claims.iter().any(|claim| {
                claim.kind == ClaimKind::WeakFunctionCandidate
                    && claim.target == AnalysisTarget::WordIndex(to)
                    && !claim.rationale.is_empty()
            }));
        }

        let vocative = analyze("Listen, professor, this matters.");
        let parse = vocative.link_parses.first().expect("link parse");
        let professor = word_index(&vocative, "professor");
        assert!(
            parse.links.iter().any(|link| {
                link.right == professor && link.kind == SyntacticLinkKind::Vocative
            })
        );
        assert!(parse.claims.iter().any(|claim| {
            claim.kind == ClaimKind::ProsodicRole
                && claim.target == AnalysisTarget::WordIndex(professor)
                && claim.value == ClaimValue::ProsodicRole("DirectAddress".to_string())
                && !claim.rationale.is_empty()
        }));
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
                    link.source == SyntacticLinkSource::AmbiguityVariant
                        || link.kind == SyntacticLinkKind::Determiner
                })
        }));
        assert!(analysis.link_parses.iter().any(|parse| {
            has_link(parse, saw, telescope, SyntacticLinkKind::Complement)
                && parse.rank >= 0.5
                && parse
                    .links
                    .iter()
                    .any(|link| link.source == SyntacticLinkSource::AmbiguityVariant)
        }));
    }

    #[test]
    fn bridges_link_claims_into_prosody_environment_facts() {
        let analysis = analyze("I saw the bright machine.");
        let machine = word_index(&analysis, "machine");
        let facts = analysis
            .prosody_environment_facts_for_word(machine)
            .expect("machine facts");
        assert_eq!(facts.word_index, machine);
        assert_eq!(facts.prosodic_role, ProsodicRole::Focus);
        assert!(facts.syntactic_links.contains(&SyntacticLinkKind::Object));
        assert!(facts.confidence >= 0.7);
        assert!(!facts.conservative);

        let vocative = analyze("Thank you, Dave.");
        let you = word_index(&vocative, "you");
        let you_facts = vocative
            .prosody_environment_facts_for_word(you)
            .expect("you facts");
        assert_eq!(
            you_facts.phrase_boundary_after,
            PhraseBoundaryKind::Vocative
        );
    }

    #[test]
    fn imports_unstressed_dictionary_flags_into_word_environment_facts() {
        let analysis = analyze("I want to go.");
        let to = word_index(&analysis, "to");
        let facts = analysis
            .prosody_environment_facts_for_word(to)
            .expect("to facts");
        let unstressed = facts
            .lexical_flags
            .iter()
            .find(|fact| fact.flag == crate::mouth::riper::LexicalProsodyFlag::Unstressed)
            .expect("unstressed lexical flag");
        assert_eq!(unstressed.source_rule_id, "weak_form_to_before_verb");
        assert_eq!(unstressed.provenance.source, "espeak-ng-derived");
    }

    #[test]
    fn imports_pause_break_flags_for_terminal_exclamation_word() {
        let analysis = analyze("Go now!");
        let now = word_index(&analysis, "now");
        let facts = analysis
            .prosody_environment_facts_for_word(now)
            .expect("now facts");
        assert_eq!(facts.phrase_boundary_after, PhraseBoundaryKind::Exclamation);
        assert!(facts.lexical_flags.iter().any(|fact| {
            fact.flag == crate::mouth::riper::LexicalProsodyFlag::BreakAfter
                && fact.source_rule_id == "punctuation_exclamation_boundary"
        }));
        assert!(facts.lexical_flags.iter().any(|fact| {
            fact.flag == crate::mouth::riper::LexicalProsodyFlag::PauseAfter
                && fact.source_rule_id == "punctuation_exclamation_boundary"
        }));
    }

    #[test]
    fn distinguishes_all_caps_abbreviation_from_contrastive_emphasis() {
        let analysis = analyze("US policy changed.");
        let us = analysis
            .tokens
            .iter()
            .find(|token| token.text == "us")
            .expect("US token");
        assert_eq!(us.pos, PartOfSpeech::ProperName);
        assert_eq!(us.prosodic_role, ProsodicRole::Content);
        assert_eq!(
            us.orthographic_emphasis,
            OrthographicEmphasisKind::Abbreviation
        );
        assert_eq!(us.reduction, ReductionClass::None);
        assert!(us.reduction_diagnostic.is_none());
    }

    #[test]
    fn marks_emphatic_all_caps_function_word_as_contrastive() {
        let analysis = analyze("FOR now.");
        let for_token = analysis
            .tokens
            .iter()
            .find(|token| token.text == "for")
            .expect("FOR token");
        assert_eq!(
            for_token.orthographic_emphasis,
            OrthographicEmphasisKind::AllCapsEmphasis
        );
        assert_eq!(for_token.prosodic_role, ProsodicRole::Contrastive);
    }

    #[test]
    fn marks_initial_as_capitalized_name_orthography() {
        let analysis = analyze("F. Scott Fitzgerald wrote.");
        let initial = analysis
            .tokens
            .iter()
            .find(|token| token.text == "f")
            .expect("initial token");
        assert_eq!(
            initial.orthographic_emphasis,
            OrthographicEmphasisKind::CapitalizedName
        );
    }

    #[test]
    fn ambiguous_attachment_facts_stay_conservative() {
        let analysis = analyze("I saw the man with the telescope.");
        let telescope = word_index(&analysis, "telescope");
        let facts = analysis
            .prosody_environment_facts_for_word(telescope)
            .expect("telescope facts");
        assert!(facts.conservative);
        assert_eq!(facts.phrase_boundary_after, PhraseBoundaryKind::None);
        assert_eq!(facts.prosodic_role, ProsodicRole::Content);
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

    #[test]
    fn links_subjects_to_predicates() {
        let analysis = analyze("The bright machine exploded.");
        let parse = analysis.link_parses.first().expect("link parse");
        let machine = word_index(&analysis, "machine");
        let exploded = word_index(&analysis, "exploded");
        assert!(has_link(
            parse,
            machine,
            exploded,
            SyntacticLinkKind::Subject
        ));
        assert!(parse.claims.iter().any(|claim| {
            claim.kind == ClaimKind::ProsodicRole
                && claim.target == AnalysisTarget::WordIndex(machine)
                && claim.value == ClaimValue::ProsodicRole("Content".to_string())
        }));

        let auxiliary = analyze("They will leave.");
        let auxiliary_parse = auxiliary.link_parses.first().expect("link parse");
        let they = word_index(&auxiliary, "they");
        let will = word_index(&auxiliary, "will");
        let leave = word_index(&auxiliary, "leave");
        assert!(has_link(
            auxiliary_parse,
            they,
            will,
            SyntacticLinkKind::Subject
        ));
        assert!(!has_link(
            auxiliary_parse,
            they,
            leave,
            SyntacticLinkKind::Subject
        ));
    }

    #[test]
    fn links_objects_and_complements() {
        let object = analyze("I saw the bright machine.");
        let object_parse = object.link_parses.first().expect("link parse");
        let saw = word_index(&object, "saw");
        let machine = word_index(&object, "machine");
        assert!(has_link(
            object_parse,
            saw,
            machine,
            SyntacticLinkKind::Object
        ));
        assert!(object_parse.claims.iter().any(|claim| {
            claim.kind == ClaimKind::ProsodicRole
                && claim.target == AnalysisTarget::WordIndex(machine)
                && claim.value == ClaimValue::ProsodicRole("Focus".to_string())
        }));

        let copular = analyze("The machine is bright.");
        let copular_parse = copular.link_parses.first().expect("link parse");
        let is = word_index(&copular, "is");
        let bright = word_index(&copular, "bright");
        assert!(has_link(
            copular_parse,
            is,
            bright,
            SyntacticLinkKind::Complement
        ));

        let infinitival = analyze("I want to go.");
        let infinitival_parse = infinitival.link_parses.first().expect("link parse");
        let want = word_index(&infinitival, "want");
        let go = word_index(&infinitival, "go");
        assert!(has_link(
            infinitival_parse,
            want,
            go,
            SyntacticLinkKind::Complement
        ));
    }

    #[test]
    fn links_coordinated_items() {
        let analysis = analyze("I saw dogs and cats.");
        let parse = analysis.link_parses.first().expect("link parse");
        let dogs = word_index(&analysis, "dogs");
        let cats = word_index(&analysis, "cats");
        assert!(has_link(parse, dogs, cats, SyntacticLinkKind::Coordination));
        assert!(parse.claims.iter().any(|claim| {
            claim.kind == ClaimKind::BoundaryKind
                && claim.target
                    == AnalysisTarget::Boundary {
                        left_word: Some(dogs),
                        right_word: Some(cats),
                    }
                && claim.value == ClaimValue::BoundaryKind("Coordination".to_string())
        }));

        let verbs = analyze("Try to ask and remember.");
        let verbs_parse = verbs.link_parses.first().expect("link parse");
        let ask = word_index(&verbs, "ask");
        let remember = word_index(&verbs, "remember");
        assert!(has_link(
            verbs_parse,
            ask,
            remember,
            SyntacticLinkKind::Coordination
        ));
    }

    #[test]
    fn links_extended_english_function_word_patterns() {
        let determiner_stack = analyze("The bright timing model exploded.");
        let determiner_parse = determiner_stack.link_parses.first().expect("link parse");
        let the = word_index(&determiner_stack, "the");
        let model = word_index(&determiner_stack, "model");
        assert!(has_link(
            determiner_parse,
            the,
            model,
            SyntacticLinkKind::Determiner
        ));
        assert!(determiner_parse.claims.iter().any(|claim| {
            claim.kind == ClaimKind::PhonemeRealization
                && claim.target == AnalysisTarget::WordRange(vec![the, model])
                && claim.value
                    == ClaimValue::PhonemeRealization("the_before_consonant:DH_AH0".to_string())
        }));

        let possessive = analyze("I follow your instructions.");
        let possessive_parse = possessive.link_parses.first().expect("link parse");
        let your = word_index(&possessive, "your");
        let instructions = word_index(&possessive, "instructions");
        assert!(has_link(
            possessive_parse,
            your,
            instructions,
            SyntacticLinkKind::Determiner
        ));

        let auxiliary = analyze("They will quickly leave.");
        let auxiliary_parse = auxiliary.link_parses.first().expect("link parse");
        let will = word_index(&auxiliary, "will");
        let leave = word_index(&auxiliary, "leave");
        assert!(has_link(
            auxiliary_parse,
            will,
            leave,
            SyntacticLinkKind::Auxiliary
        ));
        assert!(auxiliary_parse.claims.iter().any(|claim| {
            claim.kind == ClaimKind::WeakFunctionCandidate
                && claim.target == AnalysisTarget::WordIndex(will)
        }));

        let prepositional = analyze("The screenplay is about my life.");
        let prepositional_parse = prepositional.link_parses.first().expect("link parse");
        let about = word_index(&prepositional, "about");
        let life = word_index(&prepositional, "life");
        assert!(has_link(
            prepositional_parse,
            about,
            life,
            SyntacticLinkKind::Preposition
        ));
        assert!(prepositional.environment_patterns().iter().any(|pattern| {
            pattern
                .predicates
                .contains(&ContextPredicate::SyntacticLink(
                    SyntacticLinkKind::Preposition,
                ))
        }));

        let prepositional_to = analyze("This is addressed to you.");
        let prepositional_to_parse = prepositional_to.link_parses.first().expect("link parse");
        let to = word_index(&prepositional_to, "to");
        let you = word_index(&prepositional_to, "you");
        assert!(has_link(
            prepositional_to_parse,
            to,
            you,
            SyntacticLinkKind::Preposition
        ));
    }

    #[test]
    fn links_breath_break_as_noun_compound_in_sample_sentence() {
        let analysis = analyze(
            "I’m going to make the timing model distinguish vowel nuclei from other phones, then add a periodic breath break for long unpunctuated runs.",
        );
        let parse = analysis.link_parses.first().expect("link parse");
        let breath = word_index(&analysis, "breath");
        let break_word = word_index(&analysis, "break");
        let breath_token = analysis
            .tokens
            .iter()
            .find(|token| token.word_index == Some(breath))
            .expect("breath token");
        let break_token = analysis
            .tokens
            .iter()
            .find(|token| token.word_index == Some(break_word))
            .expect("break token");

        assert_eq!(breath_token.pos, PartOfSpeech::Noun);
        assert_eq!(break_token.pos, PartOfSpeech::Noun);
        assert!(has_link(
            parse,
            breath,
            break_word,
            SyntacticLinkKind::NounCompound
        ));
        assert!(has_link(
            parse,
            word_index(&analysis, "timing"),
            word_index(&analysis, "model"),
            SyntacticLinkKind::NounCompound
        ));
        assert!(has_link(
            parse,
            word_index(&analysis, "periodic"),
            breath,
            SyntacticLinkKind::Modifier
        ));
        assert!(
            !has_link(
                parse,
                word_index(&analysis, "phones"),
                word_index(&analysis, "then"),
                SyntacticLinkKind::NounCompound
            ),
            "immediate links should not cross the comma phrase break"
        );
        assert!(
            !has_link(
                parse,
                word_index(&analysis, "nuclei"),
                word_index(&analysis, "from"),
                SyntacticLinkKind::NounCompound
            ),
            "prepositional phrase starts should not be noun compounds"
        );
        assert!(
            !has_link(
                parse,
                break_word,
                word_index(&analysis, "for"),
                SyntacticLinkKind::NounCompound
            ),
            "for should attach as a preposition, not a noun-compound head"
        );
    }
}
