use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

const ESPEAK_NG_SEED_RULES_JSON: &str = include_str!("data/espeak_ng_seed_rules.json");

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleProvenance {
    pub source: String,
    pub source_file: String,
    pub source_license: String,
    pub imported_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleContextConstraint {
    pub previous_words: Vec<String>,
    pub next_words: Vec<String>,
    pub next_pos: Vec<String>,
    pub disallow_all_caps: bool,
    pub allow_phrase_final: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeakFormRule {
    pub rule_id: String,
    pub match_pattern: String,
    pub context: RuleContextConstraint,
    pub citation_form: String,
    pub output_transformation: String,
    pub confidence: u8,
    pub priority: i32,
    pub provenance: RuleProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StressRule {
    pub rule_id: String,
    pub match_pattern: String,
    pub context: RuleContextConstraint,
    pub output_transformation: String,
    pub confidence: u8,
    pub priority: i32,
    pub provenance: RuleProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PronunciationOverrideRule {
    pub rule_id: String,
    pub match_pattern: String,
    pub context: RuleContextConstraint,
    pub citation_form: String,
    pub output_transformation: String,
    pub confidence: u8,
    pub priority: i32,
    pub provenance: RuleProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PunctuationProsodyRule {
    pub rule_id: String,
    pub match_pattern: String,
    pub context: RuleContextConstraint,
    pub output_transformation: String,
    pub confidence: u8,
    pub priority: i32,
    pub provenance: RuleProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoiceVariantRule {
    pub rule_id: String,
    pub match_pattern: String,
    pub context: RuleContextConstraint,
    pub output_transformation: String,
    pub confidence: u8,
    pub priority: i32,
    pub provenance: RuleProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhonemeMappingRule {
    pub rule_id: String,
    pub match_pattern: String,
    pub context: RuleContextConstraint,
    pub output_transformation: String,
    pub confidence: u8,
    pub priority: i32,
    pub provenance: RuleProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinguisticVarieties {
    pub roots: Vec<LinguisticVarietyRuleTable>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinguisticVarietyRuleTable {
    pub tag: String,
    pub name: String,
    pub children: Vec<LinguisticVarietyRuleTable>,
    pub weak_form_rules: Vec<WeakFormRule>,
    pub stress_rules: Vec<StressRule>,
    pub pronunciation_override_rules: Vec<PronunciationOverrideRule>,
    pub punctuation_prosody_rules: Vec<PunctuationProsodyRule>,
    pub voice_variant_rules: Vec<VoiceVariantRule>,
    pub phoneme_mapping_rules: Vec<PhonemeMappingRule>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EspeakNgSeedRuleTable {
    pub source: String,
    pub imported_at: String,
    pub source_license: String,
    pub linguistic_varieties: LinguisticVarieties,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToRuleDescriptor {
    pub rule_id: String,
    pub citation_form: String,
    pub output_transformation: String,
    pub provenance: RuleProvenance,
}

impl EspeakNgSeedRuleTable {
    pub fn find_variety(&self, tag: &str) -> Option<&LinguisticVarietyRuleTable> {
        self.linguistic_varieties
            .roots
            .iter()
            .find_map(|root| root.find_nested(tag))
    }
}

impl LinguisticVarietyRuleTable {
    fn find_nested(&self, tag: &str) -> Option<&Self> {
        if self.tag == tag {
            return Some(self);
        }
        self.children
            .iter()
            .find_map(|child| child.find_nested(tag))
    }
}

pub fn import_rule_table_from_str(input: &str) -> Result<EspeakNgSeedRuleTable, serde_json::Error> {
    serde_json::from_str(input)
}

pub fn export_rule_table_to_json(
    table: &EspeakNgSeedRuleTable,
) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(table)
}

pub fn load_seed_rule_table() -> &'static EspeakNgSeedRuleTable {
    static TABLE: OnceLock<EspeakNgSeedRuleTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        import_rule_table_from_str(ESPEAK_NG_SEED_RULES_JSON)
            .expect("bundled eSpeak-ng seed rules JSON should parse")
    })
}

pub fn english_seed_variety() -> &'static LinguisticVarietyRuleTable {
    let table = load_seed_rule_table();
    table
        .find_variety("en-us-general")
        .or_else(|| table.find_variety("en-us"))
        .or_else(|| table.find_variety("en"))
        .expect("bundled eSpeak-ng seed rules should include an English variety")
}

pub fn english_to_rule_descriptor(rule_id: &str) -> Option<ToRuleDescriptor> {
    let variety = english_seed_variety();
    if let Some(rule) = variety
        .weak_form_rules
        .iter()
        .find(|rule| rule.rule_id == rule_id)
    {
        return Some(ToRuleDescriptor {
            rule_id: rule.rule_id.clone(),
            citation_form: rule.citation_form.clone(),
            output_transformation: rule.output_transformation.clone(),
            provenance: rule.provenance.clone(),
        });
    }
    variety
        .pronunciation_override_rules
        .iter()
        .find(|rule| rule.rule_id == rule_id)
        .map(|rule| ToRuleDescriptor {
            rule_id: rule.rule_id.clone(),
            citation_form: rule.citation_form.clone(),
            output_transformation: rule.output_transformation.clone(),
            provenance: rule.provenance.clone(),
        })
}

pub fn english_punctuation_rule(
    terminal_punctuation: char,
) -> Option<&'static PunctuationProsodyRule> {
    let pattern = terminal_punctuation.to_string();
    english_seed_variety()
        .punctuation_prosody_rules
        .iter()
        .find(|rule| rule.match_pattern == pattern)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bundled_seed_rule_table_and_supports_nested_varieties() {
        let table = load_seed_rule_table();
        assert_eq!(table.source, "espeak-ng-derived");
        assert!(table.find_variety("en-us-general").is_some());
        assert!(table.find_variety("fr-fr").is_some());
        assert!(table.find_variety("de-de").is_some());
    }

    #[test]
    fn converter_round_trip_is_deterministic() {
        let parsed = import_rule_table_from_str(ESPEAK_NG_SEED_RULES_JSON).expect("parse");
        let emitted = export_rule_table_to_json(&parsed).expect("export");
        let reparsed = import_rule_table_from_str(&emitted).expect("reparse");
        assert_eq!(parsed, reparsed);
    }

    #[test]
    fn english_to_rule_descriptors_include_provenance() {
        let weak = english_to_rule_descriptor("weak_form_to_before_verb").expect("weak rule");
        assert_eq!(weak.citation_form, "T UW1");
        assert_eq!(weak.output_transformation, "T AH0");
        assert_eq!(weak.provenance.source, "espeak-ng-derived");
        assert!(
            weak.provenance.source_file.contains("en_rules"),
            "expected source file metadata"
        );

        let punctuation = english_punctuation_rule('!').expect("exclamation rule");
        assert_eq!(punctuation.output_transformation, "boundary:exclamation");
        assert_eq!(punctuation.provenance.source_license, "GPL-3.0-or-later");
    }
}
