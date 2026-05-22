use serde::{Deserialize, Serialize};

use crate::linguistic::environment::*;
use crate::linguistic::inventory::{MajorClass, PhonemeClass, PhonemeSchema, WordPosition};
use crate::linguistic::phone::{Phone, PhoneStatus, PhoneString, Stress};
use crate::linguistic::phoneme::Phoneme;

/// Legacy descriptive allophone rule metadata.
///
/// Runtime allophone matching is performed by private declarative rules with
/// structured [`EnvironmentPattern`] values. This type is kept for callers that
/// still serialize the older hint-oriented shape; it is not the canonical rule
/// matcher input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LegacyAllophoneRule {
    pub id: String,
    pub applies_to_symbols: Vec<String>,
    pub output_phone_string: PhoneString,
    pub environment_hint: String,
}

#[deprecated(
    since = "0.1.0",
    note = "use LegacyAllophoneRule for the compatibility shape; runtime matching uses structured EnvironmentPattern rules"
)]
pub type AllophoneRule = LegacyAllophoneRule;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealizationMethod {
    Default,
    AllophoneRule,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Realization {
    pub phone_string: PhoneString,
    pub ipa: String,
    pub method: RealizationMethod,
    pub rule: Option<String>,
    pub environment: Option<Environment>,
    pub environment_match: Option<EnvironmentMatch>,
}
#[derive(Debug, Clone, PartialEq)]
pub struct RealizationConfig {
    pub enable_allophone_rules: bool,
    pub language: String,
    pub dialect: String,
    pub now_index: Option<usize>,
    pub span_state: SpanState,
    pub confidence: f32,
    pub phrase_boundary: Option<PhraseBoundaryKind>,
    pub prosodic_role: Option<ProsodicRole>,
    pub morpheme_kind: Option<MorphemeKind>,
    pub part_of_speech: Option<PartOfSpeech>,
    pub careful_style: bool,
}

impl Default for RealizationConfig {
    fn default() -> Self {
        Self {
            enable_allophone_rules: true,
            language: "en".to_string(),
            dialect: "american_english".to_string(),
            now_index: None,
            span_state: SpanState::Candidate,
            confidence: 1.0,
            phrase_boundary: None,
            prosodic_role: None,
            morpheme_kind: None,
            part_of_speech: None,
            careful_style: false,
        }
    }
}
pub fn realize_sequence(sequence: &[Phoneme], config: &RealizationConfig) -> Vec<Phoneme> {
    if !config.enable_allophone_rules {
        return sequence.to_vec();
    }

    let mut realized = sequence.to_vec();
    let rules = declarative_environment_rules(config);
    for i in 0..realized.len() {
        let context = RuleMatchContext::from_sequence(&realized, i, config);
        for rule in &rules {
            if let Some(mut environment_match) = match_environment_pattern(rule, &context) {
                if rule.id == "american_english_intervocalic_flapping" {
                    if config.careful_style
                        || matches!(config.phrase_boundary, Some(PhraseBoundaryKind::Major))
                    {
                        continue;
                    }
                    environment_match.matched_environment.stress_context =
                        Some("between stressed vowel and unstressed vowel".to_string());
                }

                realized[i].realization =
                    realization_from_rule_output(rule, &realized[i], environment_match);
                break;
            }
        }
    }
    realized
}

pub fn realize_sequence_as_schema(
    sequence: &[Phoneme],
    config: &RealizationConfig,
    schema: PhonemeSchema,
) -> Vec<String> {
    realize_sequence(sequence, config)
        .iter()
        .flat_map(|phoneme| phoneme.symbols_in_schema(schema))
        .collect()
}
#[derive(Debug, Clone)]
struct DeclarativeAllophoneRule {
    id: String,
    output_phone_string: PhoneString,
    pattern: EnvironmentPattern,
}

fn declarative_environment_rules(config: &RealizationConfig) -> Vec<DeclarativeAllophoneRule> {
    vec![
        DeclarativeAllophoneRule {
            id: "american_english_intervocalic_flapping".to_string(),
            output_phone_string: mapped_phone_string(&["ɾ"]),
            pattern: EnvironmentPattern {
                target: TargetPattern::Symbol("T".to_string()),
                left: vec![
                    ContextPredicate::PhonemeClass(PhonemeClass::Vowel),
                    ContextPredicate::Stress(StressPattern::Stressed),
                ],
                right: vec![
                    ContextPredicate::PhonemeClass(PhonemeClass::Vowel),
                    ContextPredicate::Stress(StressPattern::Unstressed),
                ],
                contains: Vec::new(),
                overlaps: Vec::new(),
                word_position: Some(PositionPattern::Medial),
                syllable_position: None,
                phrase_position: None,
                stress: None,
                language: Some(config.language.clone()),
                variety: Some(config.dialect.clone()),
                timing: vec![TimingPredicate::AtOrBeforeNow],
            },
        },
        DeclarativeAllophoneRule {
            id: "alveolar_nasal_velar_assimilation".to_string(),
            output_phone_string: mapped_phone_string(&["ŋ"]),
            pattern: EnvironmentPattern {
                target: TargetPattern::PhonemeClass(PhonemeClass::AlveolarNasal),
                left: Vec::new(),
                right: vec![ContextPredicate::PhonemeClass(PhonemeClass::VelarStop)],
                contains: Vec::new(),
                overlaps: Vec::new(),
                word_position: None,
                syllable_position: None,
                phrase_position: None,
                stress: None,
                language: Some(config.language.clone()),
                variety: Some(config.dialect.clone()),
                timing: vec![TimingPredicate::AtOrBeforeNow],
            },
        },
    ]
}

fn mapped_phone_string(ipa_segments: &[&str]) -> PhoneString {
    PhoneString {
        phones: ipa_segments
            .iter()
            .map(|ipa| Phone {
                ipa: (*ipa).to_string(),
                source_symbol: None,
                status: PhoneStatus::Mapped,
            })
            .collect(),
    }
}

fn realization_from_rule_output(
    rule: &DeclarativeAllophoneRule,
    target: &Phoneme,
    environment_match: EnvironmentMatch,
) -> Realization {
    let phone_string = rule_phone_string_for_target(rule, target);
    Realization {
        ipa: phone_string.to_ipa(),
        phone_string,
        method: RealizationMethod::AllophoneRule,
        rule: Some(rule.id.clone()),
        environment: Some(environment_match.matched_environment.clone()),
        environment_match: Some(environment_match),
    }
}

fn rule_phone_string_for_target(rule: &DeclarativeAllophoneRule, target: &Phoneme) -> PhoneString {
    PhoneString {
        phones: rule
            .output_phone_string
            .phones
            .iter()
            .cloned()
            .map(|mut phone| {
                phone.source_symbol = Some(target.source_symbol.clone());
                phone
            })
            .collect(),
    }
}

fn match_environment_pattern(
    rule: &DeclarativeAllophoneRule,
    context: &RuleMatchContext<'_>,
) -> Option<EnvironmentMatch> {
    let mut diagnostics = Vec::new();
    let target = context.target();
    if !target_pattern_matches(&rule.pattern.target, target) {
        return None;
    }
    diagnostics.push(PredicateDiagnostic {
        side: ContextSide::Target,
        predicate: format!("{:?}", rule.pattern.target),
    });

    for predicate in &rule.pattern.left {
        if !context_predicate_matches(predicate, context.left(), context) {
            return None;
        }
        diagnostics.push(PredicateDiagnostic {
            side: ContextSide::Left,
            predicate: format!("{:?}", predicate),
        });
    }

    for predicate in &rule.pattern.right {
        if !context_predicate_matches(predicate, context.right(), context) {
            return None;
        }
        diagnostics.push(PredicateDiagnostic {
            side: ContextSide::Right,
            predicate: format!("{:?}", predicate),
        });
    }

    for predicate in &rule.pattern.contains {
        if !context_predicate_matches(predicate, Some(target), context) {
            return None;
        }
        diagnostics.push(PredicateDiagnostic {
            side: ContextSide::Contains,
            predicate: format!("{:?}", predicate),
        });
    }

    for predicate in &rule.pattern.overlaps {
        if !context_predicate_matches(predicate, Some(target), context) {
            return None;
        }
        diagnostics.push(PredicateDiagnostic {
            side: ContextSide::Overlaps,
            predicate: format!("{:?}", predicate),
        });
    }

    if let Some(position) = rule.pattern.word_position {
        if !position_pattern_matches(
            position,
            word_position(context.index, context.sequence.len()),
        ) {
            return None;
        }
        diagnostics.push(PredicateDiagnostic {
            side: ContextSide::Target,
            predicate: format!("word_position:{position:?}"),
        });
    }

    if let Some(stress_pattern) = rule.pattern.stress {
        if !stress_pattern_matches(stress_pattern, target.stress) {
            return None;
        }
        diagnostics.push(PredicateDiagnostic {
            side: ContextSide::Target,
            predicate: format!("stress:{stress_pattern:?}"),
        });
    }

    if let Some(language) = &rule.pattern.language
        && language != &context.language
    {
        return None;
    }
    if let Some(variety) = &rule.pattern.variety
        && variety != &context.variety
    {
        return None;
    }

    for timing in &rule.pattern.timing {
        if !timing_predicate_matches(timing, context) {
            return None;
        }
        diagnostics.push(PredicateDiagnostic {
            side: ContextSide::Timing,
            predicate: format!("{:?}", timing),
        });
    }

    let left = context.left();
    let right = context.right();
    let matched_environment = Environment {
        left_phone: left.map(|phone| phone.realization.ipa.clone()),
        right_phone: right.map(|phone| phone.realization.ipa.clone()),
        left_class: left.map(phoneme_class_label),
        right_class: right.map(phoneme_class_label),
        word_position: Some(word_position(context.index, context.sequence.len())),
        syllable_position: None,
        stress_context: None,
        phrase_position: context
            .phrase_boundary
            .map(|boundary| format!("{boundary:?}").to_lowercase()),
        language: Some(context.language.clone()),
        dialect: Some(context.variety.clone()),
    };

    Some(EnvironmentMatch {
        rule: rule.id.clone(),
        target: target.symbol.clone(),
        matched_environment,
        matched_predicates: diagnostics,
        commitment: context.commitment(),
        result: rule.output_phone_string.to_ipa(),
    })
}

fn target_pattern_matches(pattern: &TargetPattern, target: &Phoneme) -> bool {
    match pattern {
        TargetPattern::Symbol(symbol) => &target.symbol == symbol,
        TargetPattern::Symbols(symbols) => symbols.iter().any(|symbol| symbol == &target.symbol),
        TargetPattern::PhonemeClass(class) => phoneme_class_matches(*class, target),
    }
}

fn context_predicate_matches(
    predicate: &ContextPredicate,
    phone: Option<&Phoneme>,
    context: &RuleMatchContext<'_>,
) -> bool {
    match predicate {
        ContextPredicate::Symbol(symbol) => phone.is_some_and(|entry| entry.symbol == *symbol),
        ContextPredicate::PhoneIpa(ipa) => phone.is_some_and(|entry| entry.realization.ipa == *ipa),
        ContextPredicate::PhonemeClass(class) => {
            phone.is_some_and(|entry| phoneme_class_matches(*class, entry))
        }
        ContextPredicate::Stress(pattern) => {
            phone.is_some_and(|entry| stress_pattern_matches(*pattern, entry.stress))
        }
        ContextPredicate::MorphemeKind(kind) => context.morphology == Some(*kind),
        ContextPredicate::WordText(text) => context
            .parent_word_span
            .as_ref()
            .and_then(|span| span.text.as_ref())
            .is_some_and(|surface| surface == text),
        ContextPredicate::Pos(pos) => context.part_of_speech == Some(*pos),
        ContextPredicate::ProsodicRole(role) => context.prosodic_role == Some(*role),
        ContextPredicate::BoundaryKind(boundary) => context.phrase_boundary == Some(*boundary),
        ContextPredicate::SpanState(state) => context.span_state == *state,
        ContextPredicate::ConfidenceAtLeast(minimum) => context.confidence >= *minimum,
    }
}

fn timing_predicate_matches(predicate: &TimingPredicate, context: &RuleMatchContext<'_>) -> bool {
    match predicate {
        TimingPredicate::AtOrBeforeNow => context
            .now
            .now_index
            .is_none_or(|now_index| context.index <= now_index),
        TimingPredicate::AtNow => context.now.now_index == Some(context.index),
        TimingPredicate::SpanState(state) => &context.span_state == state,
        TimingPredicate::ConfidenceAtLeast(minimum) => context.confidence >= *minimum,
    }
}

fn position_pattern_matches(pattern: PositionPattern, position: WordPosition) -> bool {
    matches!(
        (pattern, position),
        (PositionPattern::Any, _)
            | (PositionPattern::Singleton, WordPosition::Singleton)
            | (PositionPattern::Initial, WordPosition::WordInitial)
            | (PositionPattern::Medial, WordPosition::WordMedial)
            | (PositionPattern::Final, WordPosition::WordFinal)
    )
}

fn stress_pattern_matches(pattern: StressPattern, stress: Option<Stress>) -> bool {
    match pattern {
        StressPattern::Any => true,
        StressPattern::Stressed => matches!(stress, Some(Stress::Primary | Stress::Secondary)),
        StressPattern::Unstressed => matches!(stress, Some(Stress::Unstressed)),
        StressPattern::Primary => matches!(stress, Some(Stress::Primary)),
        StressPattern::Secondary => matches!(stress, Some(Stress::Secondary)),
    }
}

fn phoneme_class_name(class: PhonemeClass) -> &'static str {
    match class {
        PhonemeClass::Vowel => "vowel",
        PhonemeClass::Consonant => "consonant",
        PhonemeClass::AlveolarStop => "alveolar_stop",
        PhonemeClass::AlveolarNasal => "alveolar_nasal",
        PhonemeClass::VelarStop => "velar_stop",
        PhonemeClass::VelarConsonant => "velar_consonant",
        PhonemeClass::Sonorant => "sonorant",
        PhonemeClass::Obstruent => "obstruent",
        PhonemeClass::Continuant => "continuant",
        PhonemeClass::Coronal => "coronal",
        PhonemeClass::Dorsal => "dorsal",
        PhonemeClass::Labial => "labial",
        PhonemeClass::Nasal => "nasal",
        PhonemeClass::Liquid => "liquid",
        PhonemeClass::Glide => "glide",
        PhonemeClass::Sibilant => "sibilant",
        PhonemeClass::HighVowel => "high_vowel",
        PhonemeClass::UnstressedVowel => "unstressed_vowel",
    }
}

fn phoneme_class_label(phoneme: &Phoneme) -> String {
    phoneme_class_name(primary_phoneme_class(phoneme)).to_string()
}

pub(crate) fn phoneme_class_matches(class: PhonemeClass, phoneme: &Phoneme) -> bool {
    phoneme.features.matches_class(class, phoneme.stress)
}

fn primary_phoneme_class(phoneme: &Phoneme) -> PhonemeClass {
    if phoneme.features.major == MajorClass::Vowel {
        PhonemeClass::Vowel
    } else if phoneme
        .features
        .matches_class(PhonemeClass::AlveolarNasal, phoneme.stress)
    {
        PhonemeClass::AlveolarNasal
    } else if phoneme
        .features
        .matches_class(PhonemeClass::AlveolarStop, phoneme.stress)
    {
        PhonemeClass::AlveolarStop
    } else if phoneme
        .features
        .matches_class(PhonemeClass::VelarStop, phoneme.stress)
    {
        PhonemeClass::VelarStop
    } else if phoneme
        .features
        .matches_class(PhonemeClass::VelarConsonant, phoneme.stress)
    {
        PhonemeClass::VelarConsonant
    } else {
        PhonemeClass::Consonant
    }
}

fn word_position(index: usize, len: usize) -> WordPosition {
    if len <= 1 {
        WordPosition::Singleton
    } else if index == 0 {
        WordPosition::WordInitial
    } else if index == len - 1 {
        WordPosition::WordFinal
    } else {
        WordPosition::WordMedial
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linguistic::arpabet::phoneme_from_arpabet;

    #[test]
    fn rule_output_can_realize_multiple_structural_phones() {
        let target = phoneme_from_arpabet("T", "cmudict");
        let rule = DeclarativeAllophoneRule {
            id: "test_affrication".to_string(),
            output_phone_string: mapped_phone_string(&["t", "s"]),
            pattern: EnvironmentPattern {
                target: TargetPattern::Symbol("T".to_string()),
                left: Vec::new(),
                right: Vec::new(),
                contains: Vec::new(),
                overlaps: Vec::new(),
                word_position: None,
                syllable_position: None,
                phrase_position: None,
                stress: None,
                language: None,
                variety: None,
                timing: Vec::new(),
            },
        };
        let environment_match = EnvironmentMatch {
            rule: rule.id.clone(),
            target: target.symbol.clone(),
            matched_environment: Environment::default(),
            matched_predicates: Vec::new(),
            commitment: MatchCommitment::Committed,
            result: rule.output_phone_string.to_ipa(),
        };

        let realization = realization_from_rule_output(&rule, &target, environment_match);

        assert_eq!(realization.ipa, "ts");
        assert_eq!(realization.phone_string.ipa_segments(), vec!["t", "s"]);
        assert_eq!(
            realization
                .phone_string
                .phones
                .iter()
                .map(|phone| phone.source_symbol.as_deref())
                .collect::<Vec<_>>(),
            vec![Some("T"), Some("T")]
        );
    }
}
