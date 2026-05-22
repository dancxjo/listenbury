use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::linguistic::arpabet::default_phone_string_for_arpabet;
use crate::linguistic::environment as ling_env;
use crate::linguistic::inventory::VarietyImplementationStatus;
use crate::linguistic::phone::PhoneString;
use crate::mouth::riper::{NormalizedText, SentenceAnalysis};

const ESPEAK_NG_SEED_RULES_JSON: &str = include_str!("data/espeak_ng_seed_rules.json");

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleProvenance {
    pub source: String,
    pub source_file: String,
    pub source_license: String,
    pub imported_at: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "snake_case")]
pub enum LexicalProsodyFlag {
    Unstressed,
    PauseBefore,
    PauseAfter,
    BreakAfter,
    ClauseFinalStress,
    Abbreviation,
    CapitalSensitive,
    AllCapsEmphasis,
    LikelyVerbContext,
    LikelyNounContext,
    LikelyPastContext,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LexicalProsodyFlagFact {
    pub source_rule_id: String,
    pub flag: LexicalProsodyFlag,
    pub confidence: f32,
    pub provenance: RuleProvenance,
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
    #[serde(default)]
    pub dictionary_flags: Vec<LexicalProsodyFlag>,
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
    #[serde(default)]
    pub dictionary_flags: Vec<LexicalProsodyFlag>,
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
    #[serde(default)]
    pub dictionary_flags: Vec<LexicalProsodyFlag>,
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
    #[serde(default)]
    pub dictionary_flags: Vec<LexicalProsodyFlag>,
    pub provenance: RuleProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiWordSeedRule {
    pub rule_id: String,
    pub words: Vec<String>,
    pub context: RuleContextConstraint,
    pub output_transformation: String,
    pub confidence: u8,
    pub priority: i32,
    #[serde(default)]
    pub required_links: Vec<String>,
    #[serde(default)]
    pub dictionary_flags: Vec<LexicalProsodyFlag>,
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
    #[serde(default)]
    pub dictionary_flags: Vec<LexicalProsodyFlag>,
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
    #[serde(default)]
    pub dictionary_flags: Vec<LexicalProsodyFlag>,
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
    #[serde(default)]
    pub multi_word_rules: Vec<MultiWordSeedRule>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct ToRuleDescriptor {
    pub rule_id: String,
    pub citation_form: String,
    pub output_transformation: String,
    pub lexical_flags: Vec<LexicalProsodyFlagFact>,
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
        let lexical_flags = lexical_flag_facts_for_weak_form_rule(rule);
        return Some(ToRuleDescriptor {
            rule_id: rule.rule_id.clone(),
            citation_form: rule.citation_form.clone(),
            output_transformation: rule.output_transformation.clone(),
            lexical_flags,
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
            lexical_flags: lexical_flag_facts_for_pronunciation_override_rule(rule),
            provenance: rule.provenance.clone(),
        })
}

pub fn english_lexical_flag_facts_for_rule(rule_id: &str) -> Vec<LexicalProsodyFlagFact> {
    let variety = english_seed_variety();
    if let Some(rule) = variety
        .weak_form_rules
        .iter()
        .find(|rule| rule.rule_id == rule_id)
    {
        return lexical_flag_facts_for_weak_form_rule(rule);
    }
    if let Some(rule) = variety
        .stress_rules
        .iter()
        .find(|rule| rule.rule_id == rule_id)
    {
        return lexical_flag_facts_for_stress_rule(rule);
    }
    if let Some(rule) = variety
        .pronunciation_override_rules
        .iter()
        .find(|rule| rule.rule_id == rule_id)
    {
        return lexical_flag_facts_for_pronunciation_override_rule(rule);
    }
    if let Some(rule) = variety
        .punctuation_prosody_rules
        .iter()
        .find(|rule| rule.rule_id == rule_id)
    {
        return lexical_flag_facts_for_punctuation_rule(rule);
    }
    Vec::new()
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

// ---------------------------------------------------------------------------
// Morphophonology rule types
// ---------------------------------------------------------------------------

/// A spelling repair that must be applied to a stripped surface form before
/// looking up the stem in the lexicon.
///
/// These three repairs encode the eSpeak rule-file wisdom about how English
/// orthography mutates at suffix boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpellingRepairHint {
    /// Restore a trailing `e` that was dropped before the suffix
    /// (e.g. `"liked"` → stem candidate `"like"` before `-ed` lookup).
    RestoreTrailingE,
    /// Undo consonant doubling at the suffix boundary
    /// (e.g. `"running"` → stem candidate `"run"` before `-ing` lookup).
    RemoveDoubledConsonant,
    /// Reverse a `y` → `i` change at the stem/suffix boundary
    /// (e.g. `"happily"` → stem candidate `"happy"` before `-ly` lookup).
    IToY,
}

/// Describes how to extract and look up the stem once the surface affix has
/// been stripped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StemRetranslationPolicy {
    /// Strip the affix and look up the bare stem directly.
    DirectStripAndLookup,
    /// Try the listed spelling repairs (in order) as additional stem
    /// candidates before falling back to the bare form.
    SpellingRepair(Vec<SpellingRepairHint>),
}

/// Specifies what this morphophonology rule contributes to the output once
/// the stem pronunciation has been resolved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MorphophonologyOutput {
    /// Append a space-separated ARPAbet phone string to the stem's phones.
    AppendArpabet(String),
    /// Preserve the stem's pronunciation without appending anything.
    /// Used for prefix-only rules where the prefix phones are prepended
    /// externally.
    PreserveStemPronunciation,
}

/// A native morphophonology rule encoding eSpeak-style affix and
/// retranslation behaviour.
///
/// These rules represent the linguistic knowledge that eSpeak's rule files
/// encode (prefix/suffix removal, spelling repairs, POS hints) in a structured
/// format that any downstream component can inspect without re-parsing
/// eSpeak's proprietary rule syntax.
///
/// A rule carries full [`RuleProvenance`] so diagnostics can always trace
/// where a particular pronunciation decision originated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MorphophonologyRule {
    /// Unique stable identifier for this rule.
    pub id: String,
    /// The orthographic affix string (e.g. `"ed"`, `"ing"`, `"un"`).
    pub affix: String,
    /// Whether the affix is a prefix or a suffix.
    pub morpheme_kind: ling_env::MorphemeKind,
    /// How to derive the stem form for pronunciation lookup.
    pub stem_policy: StemRetranslationPolicy,
    /// What the rule produces once the stem is resolved.
    pub output_policy: MorphophonologyOutput,
    /// Where this rule's knowledge was originally encoded.
    pub provenance: RuleProvenance,
}

fn espeak_provenance() -> RuleProvenance {
    RuleProvenance {
        source: "espeak-ng-derived".to_string(),
        source_file: "dictsource/en_rules".to_string(),
        source_license: "GPL-3.0-or-later".to_string(),
        imported_at: "2026-05-21T00:23:41Z".to_string(),
    }
}

/// Return the bundled set of English native morphophonology rules derived
/// from eSpeak affix and retranslation wisdom.
///
/// These rules cover the most productive English suffixes (`-ed`, `-ing`,
/// `-s`, `-ly`) and common prefixes (`un-`, `re-`, `di-`).  Every rule
/// carries [`RuleProvenance`] so callers can surface attribution in
/// diagnostics.
pub fn english_native_morphophonology_rules() -> Vec<MorphophonologyRule> {
    let prov = espeak_provenance();
    vec![
        // --- Suffixes ---
        MorphophonologyRule {
            id: "suffix_ed_attachment".to_string(),
            affix: "ed".to_string(),
            morpheme_kind: ling_env::MorphemeKind::Suffix,
            stem_policy: StemRetranslationPolicy::SpellingRepair(vec![
                SpellingRepairHint::RestoreTrailingE,
                SpellingRepairHint::RemoveDoubledConsonant,
                SpellingRepairHint::IToY,
            ]),
            output_policy: MorphophonologyOutput::AppendArpabet(
                "IH0 D".to_string(), // underlying /ɪd/; allomorph selected at realisation
            ),
            provenance: prov.clone(),
        },
        MorphophonologyRule {
            id: "suffix_ing_attachment".to_string(),
            affix: "ing".to_string(),
            morpheme_kind: ling_env::MorphemeKind::Suffix,
            stem_policy: StemRetranslationPolicy::SpellingRepair(vec![
                SpellingRepairHint::RestoreTrailingE,
                SpellingRepairHint::RemoveDoubledConsonant,
            ]),
            output_policy: MorphophonologyOutput::AppendArpabet("IH0 NG".to_string()),
            provenance: prov.clone(),
        },
        MorphophonologyRule {
            id: "suffix_s_attachment".to_string(),
            affix: "s".to_string(),
            morpheme_kind: ling_env::MorphemeKind::Suffix,
            stem_policy: StemRetranslationPolicy::DirectStripAndLookup,
            output_policy: MorphophonologyOutput::AppendArpabet(
                "Z".to_string(), // underlying /z/; allomorph selected at realisation
            ),
            provenance: prov.clone(),
        },
        MorphophonologyRule {
            id: "suffix_ly_attachment".to_string(),
            affix: "ly".to_string(),
            morpheme_kind: ling_env::MorphemeKind::Suffix,
            stem_policy: StemRetranslationPolicy::SpellingRepair(vec![
                SpellingRepairHint::IToY,
            ]),
            output_policy: MorphophonologyOutput::AppendArpabet("L IY0".to_string()),
            provenance: prov.clone(),
        },
        // --- Prefixes ---
        MorphophonologyRule {
            id: "prefix_un_attachment".to_string(),
            affix: "un".to_string(),
            morpheme_kind: ling_env::MorphemeKind::Prefix,
            stem_policy: StemRetranslationPolicy::DirectStripAndLookup,
            output_policy: MorphophonologyOutput::PreserveStemPronunciation,
            provenance: prov.clone(),
        },
        MorphophonologyRule {
            id: "prefix_re_attachment".to_string(),
            affix: "re".to_string(),
            morpheme_kind: ling_env::MorphemeKind::Prefix,
            stem_policy: StemRetranslationPolicy::DirectStripAndLookup,
            output_policy: MorphophonologyOutput::PreserveStemPronunciation,
            provenance: prov.clone(),
        },
        MorphophonologyRule {
            id: "prefix_di_attachment".to_string(),
            affix: "di".to_string(),
            morpheme_kind: ling_env::MorphemeKind::Prefix,
            stem_policy: StemRetranslationPolicy::DirectStripAndLookup,
            output_policy: MorphophonologyOutput::PreserveStemPronunciation,
            provenance: prov,
        },
    ]
}

// ---------------------------------------------------------------------------
// Native rule types
// ---------------------------------------------------------------------------

/// Backend-neutral output produced by a converted eSpeak-derived rule.
///
/// Unlike backend-specific outputs (e.g. `PiperPhonemeSequence`), these variants
/// describe the linguistic intent — phoneme replacement or prosodic boundary — in
/// terms that any downstream renderer can interpret.
#[derive(Debug, Clone, PartialEq)]
pub enum RuleOutput {
    /// Replace the target phoneme(s) with this IPA phone string.
    PhoneString(PhoneString),
    /// Annotate a phrase boundary with the given kind and an optional prosodic
    /// contour label (e.g. `"exclamation"`, `"final_rising"`).
    ProsodyBoundary {
        boundary: ling_env::PhraseBoundaryKind,
        contour: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum MultiWordRuleOutput {
    /// Replace the matched phrase with an explicit phone sequence.
    PhoneString(PhoneString),
    /// Keep the phrase contiguous (suppress auto-inserted internal break/breath split).
    NoBreak,
    /// Prefer citation forms over weak reductions in this phrase.
    CitationFormSelection,
    /// Prefer weak forms inside this phrase.
    WeakFormSelection,
    /// Override the phrase boundary associated with this phrase.
    PhraseBoundary {
        boundary: ling_env::PhraseBoundaryKind,
        contour: Option<String>,
    },
}

pub const META_ALLOW_STUB_GA_INHERITANCE: &str = "meta:allow_stub_ga_inheritance";
pub const META_VOICE_RENDER_ONLY: &str = "meta:voice_render_only";
const VARIETY_ID_EN_US_GA: &str = "en-US-GA";
const VARIETY_ID_AMERICAN_ENGLISH: &str = "american_english";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RuleSelectionScope {
    #[default]
    Phonological,
    VoiceRender,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VarietyRuleCondition {
    pub language: String,
    pub variety: String,
    pub voice_profile: Option<String>,
    #[serde(default)]
    pub enabled_flags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ActiveVarietyProfile {
    pub language: String,
    pub variety: String,
    pub voice_profile: Option<String>,
    #[serde(default)]
    pub enabled_flags: Vec<String>,
    pub implementation_status: Option<VarietyImplementationStatus>,
    #[serde(default)]
    pub selection_scope: RuleSelectionScope,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct RuleConditionDiagnostics {
    pub applies: bool,
    pub reasons: Vec<String>,
}

/// An eSpeak-ng seed rule translated into a Listenbury-native rule descriptor.
///
/// The [`ling_env::EnvironmentPattern`] encodes the linguistic conditions under
/// which this rule applies (POS, prosodic role, phrase boundary, confidence,
/// language/variety).  Provenance fields are preserved verbatim so diagnostics
/// can trace every rule back to its eSpeak-ng origin.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedEnvironmentRule {
    /// Unique rule identifier, copied from the seed rule's `rule_id`.
    pub id: String,
    /// Trace back to the originating eSpeak-ng source file and license.
    pub provenance: RuleProvenance,
    /// Higher values take precedence when multiple rules match.
    pub priority: i32,
    /// Normalised confidence in `[0, 1]`.
    pub confidence: f32,
    /// Native lexical/prosody dictionary flags associated with this rule.
    pub lexical_flags: Vec<LexicalProsodyFlagFact>,
    /// Variety/profile conditions imported from eSpeak dialect/voice rules.
    pub conditions: Vec<VarietyRuleCondition>,
    /// Phonological/prosodic conditions that must hold for the rule to fire.
    pub pattern: ling_env::EnvironmentPattern,
    /// What the rule produces when it fires.
    pub output: RuleOutput,
}

/// Phrase-level pronunciation/prosody rule imported from an eSpeak multi-word seed entry.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiWordPronunciationRule {
    pub id: String,
    pub words: Vec<String>,
    pub conditions: Vec<VarietyRuleCondition>,
    pub pattern: ling_env::EnvironmentPattern,
    pub output: MultiWordRuleOutput,
    pub provenance: RuleProvenance,
    pub priority: i32,
    pub confidence: f32,
    pub required_links: Vec<String>,
    pub lexical_flags: Vec<LexicalProsodyFlagFact>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchedWordSpan {
    pub words: Vec<String>,
    pub word_range: std::ops::Range<usize>,
    pub token_range: std::ops::Range<usize>,
    pub source_span: std::ops::Range<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiWordRuleMatch {
    pub rule_id: String,
    pub matched_word_span: MatchedWordSpan,
    pub provenance: RuleProvenance,
}

// ---------------------------------------------------------------------------
// Conversion helpers (private)
// ---------------------------------------------------------------------------

/// Strip the ARPAbet stress-digit suffix (`0`, `1`, `2`) from a symbol.
fn arpabet_base(symbol: &str) -> &str {
    symbol.trim_end_matches(|c: char| c.is_ascii_digit())
}

/// Parse a space-separated ARPAbet string (e.g. `"T AH0"`) into a [`PhoneString`].
fn arpabet_to_phone_string(arpabet_str: &str) -> PhoneString {
    let phones = arpabet_str
        .split_whitespace()
        .flat_map(|token| {
            // `base` is the stress-stripped symbol (e.g. "AH" from "AH0") used as the
            // ARPAbet lookup key, while `token` is passed as the full original token so
            // the underlying helper can use the stress digit for tie-breaking if needed.
            let base = arpabet_base(token);
            default_phone_string_for_arpabet(base, token).phones
        })
        .collect();
    PhoneString { phones }
}

/// Parse the `output_transformation` field of a [`PunctuationProsodyRule`] into a
/// `(PhraseBoundaryKind, contour_label)` pair.
///
/// Expected format: `"boundary:<label>"` where `<label>` is a lower-case contour
/// name such as `"exclamation"` or `"final_rising"`.  Any unrecognised payload
/// maps to [`ling_env::PhraseBoundaryKind::Major`].
fn parse_boundary_output(output: &str) -> (ling_env::PhraseBoundaryKind, Option<String>) {
    if let Some(label) = output.strip_prefix("boundary:") {
        let boundary = match label {
            "none" => ling_env::PhraseBoundaryKind::None,
            "minor" => ling_env::PhraseBoundaryKind::Minor,
            _ => ling_env::PhraseBoundaryKind::Major,
        };
        (boundary, Some(label.to_string()))
    } else {
        (ling_env::PhraseBoundaryKind::Major, None)
    }
}

/// Seed rule `confidence` values are stored as integers in the range 0–100.
/// Divide by this factor to normalise them to the [0.0, 1.0] range expected by
/// [`ImportedEnvironmentRule::confidence`].
const CONFIDENCE_SCALE_FACTOR: f32 = 100.0;

fn confidence_from_seed(raw: u8) -> f32 {
    raw as f32 / CONFIDENCE_SCALE_FACTOR
}

fn push_flag_once(flags: &mut Vec<LexicalProsodyFlag>, flag: LexicalProsodyFlag) {
    if !flags.contains(&flag) {
        flags.push(flag);
    }
}

fn contextual_flags(context: &RuleContextConstraint) -> Vec<LexicalProsodyFlag> {
    let mut flags = Vec::new();
    if context.disallow_all_caps {
        push_flag_once(&mut flags, LexicalProsodyFlag::CapitalSensitive);
    }
    if !context.allow_phrase_final {
        push_flag_once(&mut flags, LexicalProsodyFlag::ClauseFinalStress);
    }
    if context.next_pos.iter().any(|pos| pos == "verb") {
        push_flag_once(&mut flags, LexicalProsodyFlag::LikelyVerbContext);
    }
    if context
        .next_pos
        .iter()
        .any(|pos| matches!(pos.as_str(), "noun" | "pronoun"))
    {
        push_flag_once(&mut flags, LexicalProsodyFlag::LikelyNounContext);
    }
    if context.next_pos.iter().any(|pos| pos == "past") {
        push_flag_once(&mut flags, LexicalProsodyFlag::LikelyPastContext);
    }
    flags
}

fn lexical_flag_facts(
    rule_id: &str,
    confidence: f32,
    provenance: &RuleProvenance,
    mut flags: Vec<LexicalProsodyFlag>,
) -> Vec<LexicalProsodyFlagFact> {
    flags.sort_unstable_by_key(|flag| *flag as u8);
    flags.dedup();
    flags
        .into_iter()
        .map(|flag| LexicalProsodyFlagFact {
            source_rule_id: rule_id.to_string(),
            flag,
            confidence,
            provenance: provenance.clone(),
        })
        .collect()
}

fn lexical_flag_facts_for_weak_form_rule(rule: &WeakFormRule) -> Vec<LexicalProsodyFlagFact> {
    let mut flags = rule.dictionary_flags.clone();
    push_flag_once(&mut flags, LexicalProsodyFlag::Unstressed);
    flags.extend(contextual_flags(&rule.context));
    lexical_flag_facts(
        &rule.rule_id,
        confidence_from_seed(rule.confidence),
        &rule.provenance,
        flags,
    )
}

fn lexical_flag_facts_for_stress_rule(rule: &StressRule) -> Vec<LexicalProsodyFlagFact> {
    let mut flags = rule.dictionary_flags.clone();
    if rule
        .output_transformation
        .eq_ignore_ascii_case("unstressed")
    {
        push_flag_once(&mut flags, LexicalProsodyFlag::Unstressed);
    }
    flags.extend(contextual_flags(&rule.context));
    lexical_flag_facts(
        &rule.rule_id,
        confidence_from_seed(rule.confidence),
        &rule.provenance,
        flags,
    )
}

fn lexical_flag_facts_for_pronunciation_override_rule(
    rule: &PronunciationOverrideRule,
) -> Vec<LexicalProsodyFlagFact> {
    let mut flags = rule.dictionary_flags.clone();
    flags.extend(contextual_flags(&rule.context));
    if rule
        .match_pattern
        .chars()
        .any(|ch| ch.is_ascii_alphabetic() && ch.is_ascii_uppercase())
        && rule
            .match_pattern
            .chars()
            .all(|ch| !ch.is_ascii_alphabetic() || ch.is_ascii_uppercase())
    {
        push_flag_once(&mut flags, LexicalProsodyFlag::AllCapsEmphasis);
    }
    lexical_flag_facts(
        &rule.rule_id,
        confidence_from_seed(rule.confidence),
        &rule.provenance,
        flags,
    )
}

fn lexical_flag_facts_for_punctuation_rule(
    rule: &PunctuationProsodyRule,
) -> Vec<LexicalProsodyFlagFact> {
    let mut flags = rule.dictionary_flags.clone();
    push_flag_once(&mut flags, LexicalProsodyFlag::PauseAfter);
    push_flag_once(&mut flags, LexicalProsodyFlag::BreakAfter);
    flags.extend(contextual_flags(&rule.context));
    lexical_flag_facts(
        &rule.rule_id,
        confidence_from_seed(rule.confidence),
        &rule.provenance,
        flags,
    )
}

fn lexical_flag_facts_for_multi_word_rule(rule: &MultiWordSeedRule) -> Vec<LexicalProsodyFlagFact> {
    let mut flags = rule.dictionary_flags.clone();
    if rule.output_transformation == "no_break" {
        push_flag_once(&mut flags, LexicalProsodyFlag::PauseAfter);
    }
    flags.extend(contextual_flags(&rule.context));
    lexical_flag_facts(
        &rule.rule_id,
        confidence_from_seed(rule.confidence),
        &rule.provenance,
        flags,
    )
}

fn variety_condition(language: &str, variety: &str) -> VarietyRuleCondition {
    VarietyRuleCondition {
        language: language.to_string(),
        variety: variety.to_string(),
        voice_profile: None,
        enabled_flags: Vec::new(),
    }
}

fn condition_is_stub_ga_allowed(condition: &VarietyRuleCondition) -> bool {
    condition.enabled_flags.iter().any(|flag| flag == META_ALLOW_STUB_GA_INHERITANCE)
}

fn runtime_flag_is_enabled(active: &ActiveVarietyProfile, flag: &str) -> bool {
    active.enabled_flags.iter().any(|enabled| enabled == flag)
}

fn is_meta_flag(flag: &str) -> bool {
    flag.starts_with("meta:")
}

fn is_stub_derived_from_ga(status: Option<&VarietyImplementationStatus>) -> bool {
    matches!(
        status,
        Some(VarietyImplementationStatus::StubDerivedFrom(source))
            if *source == crate::linguistic::inventory::VarietyId::new(VARIETY_ID_EN_US_GA)
    )
}

fn condition_matches_active_profile(
    condition: &VarietyRuleCondition,
    active: &ActiveVarietyProfile,
) -> RuleConditionDiagnostics {
    let mut reasons = Vec::new();
    if condition.language != active.language {
        reasons.push(format!(
            "language mismatch: rule={}, active={}",
            condition.language, active.language
        ));
        return RuleConditionDiagnostics { applies: false, reasons };
    }
    if condition.variety != active.variety {
        let allow_stub_ga = condition_is_stub_ga_allowed(condition)
            && is_stub_derived_from_ga(active.implementation_status.as_ref())
            && condition.variety == VARIETY_ID_AMERICAN_ENGLISH;
        if !allow_stub_ga {
            reasons.push(format!(
                "variety mismatch: rule={}, active={}",
                condition.variety, active.variety
            ));
            return RuleConditionDiagnostics { applies: false, reasons };
        }
        reasons.push(format!(
            "matched via explicit GA stub inheritance metadata ({})",
            META_ALLOW_STUB_GA_INHERITANCE
        ));
    }
    if let Some(required_voice_profile) = &condition.voice_profile {
        if active.voice_profile.as_ref() != Some(required_voice_profile) {
            reasons.push(format!(
                "voice profile mismatch: rule={}, active={}",
                required_voice_profile,
                active.voice_profile.as_deref().unwrap_or("<none>")
            ));
            return RuleConditionDiagnostics { applies: false, reasons };
        }
    }
    if condition.enabled_flags.iter().any(|flag| flag == META_VOICE_RENDER_ONLY)
        && active.selection_scope != RuleSelectionScope::VoiceRender
    {
        reasons.push("voice/render-only condition skipped during phonological rule selection".to_string());
        return RuleConditionDiagnostics { applies: false, reasons };
    }
    for flag in condition.enabled_flags.iter().filter(|flag| !is_meta_flag(flag)) {
        if !runtime_flag_is_enabled(active, flag) {
            reasons.push(format!("missing enabled flag: {flag}"));
            return RuleConditionDiagnostics { applies: false, reasons };
        }
    }
    reasons.push("condition matched active profile".to_string());
    RuleConditionDiagnostics { applies: true, reasons }
}

pub fn evaluate_rule_conditions(
    rule: &ImportedEnvironmentRule,
    active: &ActiveVarietyProfile,
) -> RuleConditionDiagnostics {
    if rule.conditions.is_empty() {
        return RuleConditionDiagnostics {
            applies: true,
            reasons: vec!["rule has no explicit variety/profile conditions".to_string()],
        };
    }
    let mut rejection_reasons = Vec::new();
    for condition in &rule.conditions {
        let evaluated = condition_matches_active_profile(condition, active);
        if evaluated.applies {
            return evaluated;
        }
        rejection_reasons.extend(evaluated.reasons);
    }
    RuleConditionDiagnostics {
        applies: false,
        reasons: rejection_reasons,
    }
}

pub fn english_voice_render_conditions() -> Vec<VarietyRuleCondition> {
    english_seed_variety()
        .voice_variant_rules
        .iter()
        .map(|rule| VarietyRuleCondition {
            language: "en".to_string(),
            variety: VARIETY_ID_AMERICAN_ENGLISH.to_string(),
            voice_profile: Some(rule.match_pattern.clone()),
            enabled_flags: vec![
                META_VOICE_RENDER_ONLY.to_string(),
                format!("voice_style:{}", rule.output_transformation),
            ],
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Public conversion functions
// ---------------------------------------------------------------------------

/// Convert a [`WeakFormRule`] into a native [`ImportedEnvironmentRule`].
///
/// The resulting rule fires when the phonological context indicates a function
/// word in a weak prosodic position ([`ling_env::ProsodicRole::FunctionWeak`]).
///
/// The seed rule's `next_pos` constraint (e.g. "fire before a verb") is *not*
/// mapped to a [`ling_env::ContextPredicate::Pos`] predicate because it refers to
/// the *next* word's POS, which the current `EnvironmentPattern` engine does not
/// model directly.  The `ProsodicRole::FunctionWeak` predicate captures the same
/// linguistic insight at a higher level of abstraction.
pub fn convert_weak_form_rule(
    rule: &WeakFormRule,
    language: &str,
    variety: &str,
) -> ImportedEnvironmentRule {
    let output_phones = arpabet_to_phone_string(&rule.output_transformation);
    let confidence = confidence_from_seed(rule.confidence);

    // Prosodic role: weak form words are always function words in weak position.
    // Note: the seed rule's `next_pos` constraint ("fire when the next word is a
    // verb/noun/…") cannot be expressed as a `ContextPredicate::Pos` on the
    // *current* word — the native engine does not yet model "next-word POS"
    // directly.  The `ProsodicRole::FunctionWeak` predicate captures the same
    // semantic intent: "to" before a verb is always a weakly-stressed function word.
    let contains: Vec<ling_env::ContextPredicate> = vec![ling_env::ContextPredicate::ProsodicRole(
        ling_env::ProsodicRole::FunctionWeak,
    )];

    // Build a target pattern from the citation form's base ARPAbet symbols.
    let citation_symbols: Vec<String> = rule
        .citation_form
        .split_whitespace()
        .map(|t| arpabet_base(t).to_string())
        .collect();
    let target = if citation_symbols.len() == 1 {
        ling_env::TargetPattern::Symbol(citation_symbols.into_iter().next().unwrap())
    } else {
        ling_env::TargetPattern::Symbols(citation_symbols)
    };

    ImportedEnvironmentRule {
        id: rule.rule_id.clone(),
        provenance: rule.provenance.clone(),
        priority: rule.priority,
        confidence,
        lexical_flags: lexical_flag_facts_for_weak_form_rule(rule),
        conditions: vec![variety_condition(language, variety)],
        pattern: ling_env::EnvironmentPattern {
            target,
            left: Vec::new(),
            right: Vec::new(),
            contains,
            overlaps: Vec::new(),
            word_position: None,
            syllable_position: None,
            phrase_position: None,
            stress: None,
            language: Some(language.to_string()),
            variety: Some(variety.to_string()),
            timing: Vec::new(),
        },
        output: RuleOutput::PhoneString(output_phones),
    }
}

/// Convert a [`PunctuationProsodyRule`] into a native [`ImportedEnvironmentRule`].
///
/// The output is a [`RuleOutput::ProsodyBoundary`] whose `contour` label is
/// derived from the seed rule's `output_transformation` string
/// (`"boundary:<label>"`).  The target pattern carries the literal punctuation
/// character so callers can identify which surface form triggers this rule.
pub fn convert_punctuation_prosody_rule(
    rule: &PunctuationProsodyRule,
    language: &str,
    variety: &str,
) -> ImportedEnvironmentRule {
    let confidence = confidence_from_seed(rule.confidence);
    let (boundary, contour) = parse_boundary_output(&rule.output_transformation);

    ImportedEnvironmentRule {
        id: rule.rule_id.clone(),
        provenance: rule.provenance.clone(),
        priority: rule.priority,
        confidence,
        lexical_flags: lexical_flag_facts_for_punctuation_rule(rule),
        conditions: vec![variety_condition(language, variety)],
        pattern: ling_env::EnvironmentPattern {
            target: ling_env::TargetPattern::Symbol(rule.match_pattern.clone()),
            left: Vec::new(),
            right: Vec::new(),
            contains: Vec::new(),
            overlaps: Vec::new(),
            word_position: None,
            syllable_position: None,
            phrase_position: None,
            stress: None,
            language: Some(language.to_string()),
            variety: Some(variety.to_string()),
            timing: Vec::new(),
        },
        output: RuleOutput::ProsodyBoundary { boundary, contour },
    }
}

fn parse_multi_word_output(output: &str) -> MultiWordRuleOutput {
    if output.eq_ignore_ascii_case("no_break") {
        return MultiWordRuleOutput::NoBreak;
    }
    if output.eq_ignore_ascii_case("citation_form") {
        return MultiWordRuleOutput::CitationFormSelection;
    }
    if output.eq_ignore_ascii_case("weak_form") {
        return MultiWordRuleOutput::WeakFormSelection;
    }
    if let Some(arpabet) = output.strip_prefix("phones:") {
        return MultiWordRuleOutput::PhoneString(arpabet_to_phone_string(arpabet));
    }
    if output.starts_with("boundary:") {
        let (boundary, contour) = parse_boundary_output(output);
        return MultiWordRuleOutput::PhraseBoundary { boundary, contour };
    }
    MultiWordRuleOutput::PhoneString(arpabet_to_phone_string(output))
}

/// Convert a [`MultiWordSeedRule`] into a native phrase-level
/// [`MultiWordPronunciationRule`].
pub fn convert_multi_word_rule(
    rule: &MultiWordSeedRule,
    language: &str,
    variety: &str,
) -> MultiWordPronunciationRule {
    MultiWordPronunciationRule {
        id: rule.rule_id.clone(),
        words: rule.words.clone(),
        conditions: vec![variety_condition(language, variety)],
        pattern: ling_env::EnvironmentPattern {
            target: ling_env::TargetPattern::Symbols(rule.words.clone()),
            left: Vec::new(),
            right: Vec::new(),
            contains: Vec::new(),
            overlaps: Vec::new(),
            word_position: None,
            syllable_position: None,
            phrase_position: None,
            stress: None,
            language: Some(language.to_string()),
            variety: Some(variety.to_string()),
            timing: Vec::new(),
        },
        output: parse_multi_word_output(&rule.output_transformation),
        provenance: rule.provenance.clone(),
        priority: rule.priority,
        confidence: confidence_from_seed(rule.confidence),
        required_links: rule.required_links.clone(),
        lexical_flags: lexical_flag_facts_for_multi_word_rule(rule),
    }
}

// ---------------------------------------------------------------------------
// Structured boundary prosody rule types
// ---------------------------------------------------------------------------

/// The pattern that triggers a [`BoundaryProsodyRule`] — either a specific
/// punctuation character or a clause/utterance position.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryPattern {
    /// A specific punctuation character (e.g. `'!'`, `'?'`, `','`).
    Punctuation(char),
    /// The end of a complete clause or sentence (any final-boundary position).
    ClauseEnd,
    /// The start of a clause or sentence.
    ClauseStart,
    /// The very end of the utterance string — stronger than `ClauseEnd`.
    UtteranceEnd,
    /// Any major phrase boundary, regardless of punctuation.
    AnyMajorBoundary,
    /// Any minor phrase boundary, regardless of punctuation.
    AnyMinorBoundary,
}

/// Phrase boundary strength for a [`BoundaryProsodyRule`] output.
///
/// This is a prosody-level classification and is intentionally kept separate
/// from [`ling_env::PhraseBoundaryKind`] (which covers only `None/Minor/Major`)
/// to allow richer downstream discrimination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryKind {
    None,
    Minor,
    Major,
}

/// Pitch movement direction at a prosodic boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PitchDirection {
    /// Falling pitch at the end of a declarative sentence.
    FinalFalling,
    /// Rising pitch at the end of a question.
    FinalRising,
    /// High/expanded pitch associated with exclamations.
    Exclamation,
    /// Level or neutral pitch — continuation or parenthetical.
    Level,
}

/// Stress effect at a clause boundary position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StressEffect {
    /// The final stressed syllable before the boundary receives extra prominence.
    ClauseFinalStress,
    /// Stress is reduced or relaxed — typical for parentheticals or appositives.
    StressRelaxation,
    /// No special stress adjustment.
    Neutral,
}

/// Context evidence that can suppress or soften a comma pause.
///
/// When any of these conditions hold at the comma site, the minor-phrase pause
/// should be shortened or omitted, matching eSpeak-ng's vocative/appositive
/// comma suppression behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuppressibleBy {
    /// A vocative (direct-address) construction is present at the comma site.
    VocativeContext,
    /// A non-restrictive appositive phrase flanks the comma.
    AppositiveContext,
    /// A parenthetical aside is delimited by the comma.
    ParentheticalContext,
    /// A tight syntactic dependency (e.g. determiner–noun, auxiliary–verb)
    /// crosses the comma, indicating the pause is prosodically inappropriate.
    TightSyntacticLink,
}

/// Structured prosodic effect produced when a [`BoundaryProsodyRule`] fires.
///
/// All fields are optional; `None` means "do not modify the default".  A rule
/// may influence only the boundary strength while leaving pitch and stress to
/// downstream defaults, for example.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryProsodyEffect {
    /// The phrase boundary strength that applies at this position.
    pub boundary_kind: BoundaryKind,
    /// The pitch contour direction at this boundary, if specified.
    pub pitch_direction: Option<PitchDirection>,
    /// Additional pause duration in milliseconds (relative to the baseline
    /// pause for this boundary kind).  Positive values extend the pause;
    /// negative values shorten it.
    pub pause_delta_ms: Option<i32>,
    /// Pitch range multiplier: `1.0` is neutral, `> 1.0` widens the range
    /// (excited/exclamation), `< 1.0` narrows it.
    pub pitch_range_factor: Option<f32>,
    /// Clause-boundary stress adjustment.
    pub stress_effect: StressEffect,
    /// Evidence kinds that can suppress or soften this boundary's pause.
    /// Empty means the pause is unconditional.
    pub suppressible_by: Vec<SuppressibleBy>,
}

/// A structured boundary/prosody rule derived from eSpeak-ng clause-position
/// and punctuation prosody rules.
///
/// Unlike [`ImportedEnvironmentRule`], which stores the output as an opaque
/// `contour: Option<String>`, `BoundaryProsodyRule` encodes every prosodic
/// dimension as a typed field so that downstream renderers can act on them
/// without parsing strings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryProsodyRule {
    /// Unique identifier, copied from the originating seed rule's `rule_id`.
    pub id: String,
    /// What position or punctuation character triggers this rule.
    pub boundary_pattern: BoundaryPattern,
    /// The structured prosodic effect when the rule fires.
    pub output: BoundaryProsodyEffect,
    /// Higher values take precedence when multiple rules match.
    pub priority: i32,
    /// Normalised confidence in `[0, 1]`.
    pub confidence: f32,
    /// Trace back to the originating eSpeak-ng source file and license.
    pub provenance: RuleProvenance,
}

// ---------------------------------------------------------------------------
// Conversion: PunctuationProsodyRule → BoundaryProsodyRule
// ---------------------------------------------------------------------------

/// Intermediate result of parsing an `output_transformation` string into
/// structured prosody effect fields.  Used only inside
/// [`structured_boundary_output`] to avoid returning a raw 4-tuple.
struct ParsedBoundaryOutput {
    boundary_kind: BoundaryKind,
    pitch_direction: Option<PitchDirection>,
    pitch_range_factor: Option<f32>,
    stress_effect: StressEffect,
}

/// Derive the structured prosody effect fields from an `output_transformation`
/// string such as `"boundary:exclamation"`.
fn structured_boundary_output(output: &str) -> ParsedBoundaryOutput {
    if let Some(label) = output.strip_prefix("boundary:") {
        match label {
            "none" => ParsedBoundaryOutput {
                boundary_kind: BoundaryKind::None,
                pitch_direction: None,
                pitch_range_factor: None,
                stress_effect: StressEffect::Neutral,
            },
            "minor" => ParsedBoundaryOutput {
                boundary_kind: BoundaryKind::Minor,
                pitch_direction: Some(PitchDirection::Level),
                pitch_range_factor: None,
                stress_effect: StressEffect::Neutral,
            },
            "major" => ParsedBoundaryOutput {
                boundary_kind: BoundaryKind::Major,
                pitch_direction: Some(PitchDirection::Level),
                pitch_range_factor: None,
                stress_effect: StressEffect::ClauseFinalStress,
            },
            "final_falling" => ParsedBoundaryOutput {
                boundary_kind: BoundaryKind::Major,
                pitch_direction: Some(PitchDirection::FinalFalling),
                pitch_range_factor: None,
                stress_effect: StressEffect::ClauseFinalStress,
            },
            "final_rising" => ParsedBoundaryOutput {
                boundary_kind: BoundaryKind::Major,
                pitch_direction: Some(PitchDirection::FinalRising),
                pitch_range_factor: None,
                stress_effect: StressEffect::ClauseFinalStress,
            },
            "exclamation" => ParsedBoundaryOutput {
                boundary_kind: BoundaryKind::Major,
                pitch_direction: Some(PitchDirection::Exclamation),
                pitch_range_factor: Some(1.25),
                stress_effect: StressEffect::ClauseFinalStress,
            },
            _ => ParsedBoundaryOutput {
                boundary_kind: BoundaryKind::Major,
                pitch_direction: None,
                pitch_range_factor: None,
                stress_effect: StressEffect::Neutral,
            },
        }
    } else {
        ParsedBoundaryOutput {
            boundary_kind: BoundaryKind::Major,
            pitch_direction: None,
            pitch_range_factor: None,
            stress_effect: StressEffect::Neutral,
        }
    }
}

/// Derive which context evidence can suppress this punctuation's pause.
///
/// Comma pauses can be suppressed by vocative/appositive/parenthetical
/// evidence; sentence-final punctuation (`!`, `?`, `.`) is never suppressible.
fn suppressible_by_for_punctuation(match_pattern: &str) -> Vec<SuppressibleBy> {
    match match_pattern {
        "," => vec![
            SuppressibleBy::VocativeContext,
            SuppressibleBy::AppositiveContext,
            SuppressibleBy::ParentheticalContext,
            SuppressibleBy::TightSyntacticLink,
        ],
        // Semicolons and colons can be softened when a tight link is present
        // (e.g. a colon introducing a list where the first item follows immediately).
        ";" | ":" => vec![SuppressibleBy::TightSyntacticLink],
        _ => vec![],
    }
}

/// Convert a [`PunctuationProsodyRule`] into a structured [`BoundaryProsodyRule`].
///
/// Unlike [`convert_punctuation_prosody_rule`] which produces an
/// [`ImportedEnvironmentRule`] with a string `contour` field, this function
/// produces fully typed output that does not depend on string parsing downstream.
///
/// `match_pattern` is expected to be a single Unicode scalar value (punctuation
/// character).  If it is empty the pattern falls back to
/// [`BoundaryPattern::AnyMajorBoundary`].
pub fn convert_to_boundary_prosody_rule(rule: &PunctuationProsodyRule) -> BoundaryProsodyRule {
    let confidence = confidence_from_seed(rule.confidence);
    // match_pattern is always a single punctuation character in the seed data.
    // chars().next() safely handles the empty-string edge case without panicking.
    let boundary_pattern = match rule.match_pattern.chars().next() {
        Some(ch) => BoundaryPattern::Punctuation(ch),
        None => BoundaryPattern::AnyMajorBoundary,
    };
    let parsed = structured_boundary_output(&rule.output_transformation);
    let suppressible_by = suppressible_by_for_punctuation(&rule.match_pattern);

    BoundaryProsodyRule {
        id: rule.rule_id.clone(),
        boundary_pattern,
        output: BoundaryProsodyEffect {
            boundary_kind: parsed.boundary_kind,
            pitch_direction: parsed.pitch_direction,
            pause_delta_ms: None,
            pitch_range_factor: parsed.pitch_range_factor,
            stress_effect: parsed.stress_effect,
            suppressible_by,
        },
        priority: rule.priority,
        confidence,
        provenance: rule.provenance.clone(),
    }
}

/// Return whether a [`BoundaryProsodyRule`]'s comma pause is suppressed by the
/// given context evidence.
///
/// Returns `true` if at least one of the rule's [`SuppressibleBy`] kinds is
/// present in `active_suppressors`.  A `false` result means the pause should
/// proceed at full strength.
pub fn is_comma_pause_suppressed(
    rule: &BoundaryProsodyRule,
    active_suppressors: &[SuppressibleBy],
) -> bool {
    rule.output
        .suppressible_by
        .iter()
        .any(|kind| active_suppressors.contains(kind))
}

// ---------------------------------------------------------------------------
// Clause-position boundary rules
// ---------------------------------------------------------------------------

/// Build a hardcoded set of clause-position [`BoundaryProsodyRule`]s that
/// capture start-of-clause, end-of-clause, and end-of-utterance behaviour
/// independently of any specific punctuation character.
///
/// These supplement the punctuation-derived rules with position-based prosodic
/// effects such as utterance-final falling cadence and clause-start freshness.
pub fn clause_position_boundary_rules() -> Vec<BoundaryProsodyRule> {
    let provenance = RuleProvenance {
        source: "espeak-ng-derived".to_string(),
        source_file: "dictsource/en_rules, phsource/prosody".to_string(),
        source_license: "GPL-3.0-or-later".to_string(),
        imported_at: "2026-05-21T00:23:41Z".to_string(),
    };
    vec![
        BoundaryProsodyRule {
            id: "clause_utterance_end_falling".to_string(),
            boundary_pattern: BoundaryPattern::UtteranceEnd,
            output: BoundaryProsodyEffect {
                boundary_kind: BoundaryKind::Major,
                pitch_direction: Some(PitchDirection::FinalFalling),
                pause_delta_ms: None,
                pitch_range_factor: None,
                stress_effect: StressEffect::ClauseFinalStress,
                suppressible_by: vec![],
            },
            priority: 110,
            confidence: 0.92,
            provenance: provenance.clone(),
        },
        BoundaryProsodyRule {
            id: "clause_end_falling".to_string(),
            boundary_pattern: BoundaryPattern::ClauseEnd,
            output: BoundaryProsodyEffect {
                boundary_kind: BoundaryKind::Major,
                pitch_direction: Some(PitchDirection::FinalFalling),
                pause_delta_ms: None,
                pitch_range_factor: None,
                stress_effect: StressEffect::ClauseFinalStress,
                suppressible_by: vec![],
            },
            priority: 100,
            confidence: 0.85,
            provenance: provenance.clone(),
        },
        BoundaryProsodyRule {
            id: "clause_start_level".to_string(),
            boundary_pattern: BoundaryPattern::ClauseStart,
            output: BoundaryProsodyEffect {
                boundary_kind: BoundaryKind::None,
                pitch_direction: Some(PitchDirection::Level),
                pause_delta_ms: None,
                pitch_range_factor: None,
                stress_effect: StressEffect::Neutral,
                suppressible_by: vec![],
            },
            priority: 90,
            confidence: 0.80,
            provenance,
        },
    ]
}

// ---------------------------------------------------------------------------
// Bulk converters for the bundled English variety
// ---------------------------------------------------------------------------

/// Return native [`ImportedEnvironmentRule`] descriptors for all weak-form rules
/// in the bundled English (US, General American) seed variety.
pub fn english_imported_weak_form_rules() -> Vec<ImportedEnvironmentRule> {
    english_seed_variety()
        .weak_form_rules
        .iter()
        .map(|r| convert_weak_form_rule(r, "en", "american_english"))
        .collect()
}

/// Return native [`ImportedEnvironmentRule`] descriptors for all punctuation
/// prosody rules in the bundled English (US, General American) seed variety.
pub fn english_imported_punctuation_rules() -> Vec<ImportedEnvironmentRule> {
    english_seed_variety()
        .punctuation_prosody_rules
        .iter()
        .map(|r| convert_punctuation_prosody_rule(r, "en", "american_english"))
        .collect()
}

/// Return all structured [`BoundaryProsodyRule`]s for the bundled English
/// (US, General American) variety.
///
/// The result combines:
/// 1. Punctuation-derived rules converted from the seed JSON via
///    [`convert_to_boundary_prosody_rule`].
/// 2. Clause-position rules from [`clause_position_boundary_rules`].
///
/// Rules are returned sorted by descending priority so callers may apply them
/// in order, stopping at the first match.
pub fn english_boundary_prosody_rules() -> Vec<BoundaryProsodyRule> {
    let mut rules: Vec<BoundaryProsodyRule> = english_seed_variety()
        .punctuation_prosody_rules
        .iter()
        .map(convert_to_boundary_prosody_rule)
        .collect();
    rules.extend(clause_position_boundary_rules());
    rules.sort_unstable_by(|a, b| b.priority.cmp(&a.priority));
    rules
}

/// Return native phrase-level descriptors for all imported multi-word rules in
/// the bundled English (US, General American) seed variety.
pub fn english_imported_multi_word_rules() -> Vec<MultiWordPronunciationRule> {
    english_seed_variety()
        .multi_word_rules
        .iter()
        .map(|r| convert_multi_word_rule(r, "en", "american_english"))
        .collect()
}

// ---------------------------------------------------------------------------
// Context matching
// ---------------------------------------------------------------------------

/// Check whether a converted rule's environment constraints are satisfied by
/// the given [`ling_env::RuleMatchContext`].
///
/// This covers the word-level predicates relevant to weak-form and
/// punctuation-prosody rules: [`ling_env::ContextPredicate::ProsodicRole`],
/// [`ling_env::ContextPredicate::Pos`], [`ling_env::ContextPredicate::BoundaryKind`],
/// [`ling_env::ContextPredicate::ConfidenceAtLeast`],
/// [`ling_env::ContextPredicate::SpanState`], and
/// [`ling_env::ContextPredicate::MorphemeKind`], plus language/variety.
/// Phoneme-level predicates (`Symbol`, `PhoneIpa`, `PhonemeClass`, `Stress`) are
/// out of scope for word-level imported rules and are treated as always-satisfied.
pub fn rule_matches_context(
    rule: &ImportedEnvironmentRule,
    context: &ling_env::RuleMatchContext<'_>,
) -> bool {
    let active_profile = ActiveVarietyProfile {
        language: context.language.clone(),
        variety: context.variety.clone(),
        voice_profile: None,
        enabled_flags: Vec::new(),
        implementation_status: None,
        selection_scope: RuleSelectionScope::Phonological,
    };
    if !evaluate_rule_conditions(rule, &active_profile).applies {
        return false;
    }
    if let Some(lang) = &rule.pattern.language {
        if lang != &context.language {
            return false;
        }
    }
    if let Some(variety) = &rule.pattern.variety {
        if variety != &context.variety {
            return false;
        }
    }
    rule.pattern
        .contains
        .iter()
        .all(|predicate| env_predicate_matches(predicate, context))
}

fn env_predicate_matches(
    predicate: &ling_env::ContextPredicate,
    context: &ling_env::RuleMatchContext<'_>,
) -> bool {
    match predicate {
        ling_env::ContextPredicate::ProsodicRole(role) => context.prosodic_role == Some(*role),
        ling_env::ContextPredicate::Pos(pos) => context.part_of_speech == Some(*pos),
        ling_env::ContextPredicate::BoundaryKind(boundary) => {
            context.phrase_boundary == Some(*boundary)
        }
        ling_env::ContextPredicate::ConfidenceAtLeast(min) => context.confidence >= *min,
        ling_env::ContextPredicate::SpanState(state) => context.span_state == *state,
        ling_env::ContextPredicate::MorphemeKind(kind) => context.morphology == Some(*kind),
        // Phoneme-level predicates are out of scope for word-level imported rules.
        _ => true,
    }
}

fn canonical_link_label(label: &str) -> String {
    label
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

fn link_kind_matches_label(
    kind: crate::mouth::riper::SyntacticLinkKind,
    required_label: &str,
) -> bool {
    canonical_link_label(&format!("{kind:?}")) == canonical_link_label(required_label)
}

fn required_links_satisfied(
    required_links: &[String],
    sentence_analysis: &SentenceAnalysis,
    span_start_word: usize,
    span_end_word_exclusive: usize,
) -> bool {
    if required_links.is_empty() {
        return true;
    }
    let Some(primary_parse) = sentence_analysis.link_parses.first() else {
        return false;
    };
    required_links.iter().all(|required_label| {
        primary_parse.links.iter().any(|link| {
            let (left, right) = if link.left <= link.right {
                (link.left, link.right)
            } else {
                (link.right, link.left)
            };
            left >= span_start_word
                && right < span_end_word_exclusive
                && link_kind_matches_label(link.kind, required_label)
        })
    })
}

/// Match a phrase-level rule against normalized words and return all matched
/// spans, preserving both word-index and source-token ranges for diagnostics.
pub fn match_multi_word_rule(
    rule: &MultiWordPronunciationRule,
    normalized: &NormalizedText,
    sentence_analysis: &SentenceAnalysis,
) -> Vec<MultiWordRuleMatch> {
    if rule.words.is_empty() {
        return Vec::new();
    }
    let mut words_with_tokens = sentence_analysis
        .tokens
        .iter()
        .filter_map(|token| {
            token
                .word_index
                .map(|word_index| (word_index, token.token_index, token.text.as_str()))
        })
        .collect::<Vec<_>>();
    words_with_tokens.sort_unstable_by_key(|(word_index, _, _)| *word_index);

    let phrase_len = rule.words.len();
    words_with_tokens
        .windows(phrase_len)
        .filter_map(|window| {
            let words_match = window
                .iter()
                .zip(rule.words.iter())
                .all(|((_, _, token_word), rule_word)| token_word.eq_ignore_ascii_case(rule_word));
            if !words_match {
                return None;
            }

            let span_start_word = window.first()?.0;
            let span_end_word_exclusive = span_start_word + phrase_len;
            if !required_links_satisfied(
                &rule.required_links,
                sentence_analysis,
                span_start_word,
                span_end_word_exclusive,
            ) {
                return None;
            }

            let token_start = window.first()?.1;
            let token_end_inclusive = window.last()?.1;
            let source_start = normalized.token_spans.get(token_start)?.start;
            let source_end = normalized.token_spans.get(token_end_inclusive)?.end;

            Some(MultiWordRuleMatch {
                rule_id: rule.id.clone(),
                matched_word_span: MatchedWordSpan {
                    words: window
                        .iter()
                        .map(|(_, _, word)| (*word).to_string())
                        .collect(),
                    word_range: span_start_word..span_end_word_exclusive,
                    token_range: token_start..(token_end_inclusive + 1),
                    source_span: source_start..source_end,
                },
                provenance: rule.provenance.clone(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linguistic::arpabet::phoneme_from_arpabet;
    use crate::linguistic::environment as ling_env;
    use crate::linguistic::inventory::VarietyId;
    use crate::linguistic::realization::RealizationConfig;
    use crate::mouth::riper::{HeuristicSentenceAnalyzer, SentenceAnalyzer, TextNormalizer};

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
            weak.lexical_flags
                .iter()
                .any(|fact| fact.flag == LexicalProsodyFlag::Unstressed),
            "weak-form descriptor should expose imported unstressed flag facts"
        );
        assert!(
            weak.provenance.source_file.contains("en_rules"),
            "expected source file metadata"
        );

        let punctuation = english_punctuation_rule('!').expect("exclamation rule");
        assert_eq!(punctuation.output_transformation, "boundary:exclamation");
        assert_eq!(punctuation.provenance.source_license, "GPL-3.0-or-later");
    }

    // --- Converted native rule tests ---

    #[test]
    fn converted_weak_form_rule_preserves_provenance() {
        let seed_rule = english_seed_variety()
            .weak_form_rules
            .iter()
            .find(|r| r.rule_id == "weak_form_to_before_verb")
            .expect("seed rule must exist");

        let native = convert_weak_form_rule(seed_rule, "en", "american_english");

        assert_eq!(native.id, "weak_form_to_before_verb");
        assert_eq!(native.provenance.source, "espeak-ng-derived");
        assert!(
            native
                .lexical_flags
                .iter()
                .any(|fact| fact.flag == LexicalProsodyFlag::Unstressed),
            "weak-form conversion should preserve unstressed dictionary flag"
        );
        assert!(
            native.provenance.source_file.contains("en_rules"),
            "provenance source file should survive conversion"
        );
    }

    #[test]
    fn converted_weak_form_rule_matches_function_weak_prosodic_role() {
        let seed_rule = english_seed_variety()
            .weak_form_rules
            .iter()
            .find(|r| r.rule_id == "weak_form_to_before_verb")
            .expect("seed rule must exist");

        let native = convert_weak_form_rule(seed_rule, "en", "american_english");

        let sequence = vec![
            phoneme_from_arpabet("T", "cmudict"),
            phoneme_from_arpabet("UW1", "cmudict"),
        ];
        let config = RealizationConfig {
            language: "en".to_string(),
            dialect: "american_english".to_string(),
            prosodic_role: Some(ling_env::ProsodicRole::FunctionWeak),
            ..Default::default()
        };
        let context = ling_env::RuleMatchContext::from_sequence(&sequence, 0, &config);

        assert!(
            rule_matches_context(&native, &context),
            "rule should match when word is in FunctionWeak prosodic role"
        );
    }

    #[test]
    fn converted_weak_form_rule_rejects_content_word_prosodic_role() {
        let seed_rule = english_seed_variety()
            .weak_form_rules
            .iter()
            .find(|r| r.rule_id == "weak_form_to_before_verb")
            .expect("seed rule must exist");

        let native = convert_weak_form_rule(seed_rule, "en", "american_english");

        let sequence = vec![phoneme_from_arpabet("T", "cmudict")];
        let config = RealizationConfig {
            language: "en".to_string(),
            dialect: "american_english".to_string(),
            prosodic_role: Some(ling_env::ProsodicRole::Content),
            ..Default::default()
        };
        let context = ling_env::RuleMatchContext::from_sequence(&sequence, 0, &config);

        assert!(
            !rule_matches_context(&native, &context),
            "rule should not match when word is a content word"
        );
    }

    #[test]
    fn converted_weak_form_rule_rejects_wrong_language() {
        let seed_rule = english_seed_variety()
            .weak_form_rules
            .iter()
            .find(|r| r.rule_id == "weak_form_to_before_verb")
            .expect("seed rule must exist");

        let native = convert_weak_form_rule(seed_rule, "en", "american_english");

        let sequence = vec![phoneme_from_arpabet("T", "cmudict")];
        let config = RealizationConfig {
            language: "fr".to_string(),
            dialect: "standard_french".to_string(),
            prosodic_role: Some(ling_env::ProsodicRole::FunctionWeak),
            ..Default::default()
        };
        let context = ling_env::RuleMatchContext::from_sequence(&sequence, 0, &config);

        assert!(
            !rule_matches_context(&native, &context),
            "English rule should not match a French context"
        );
    }

    #[test]
    fn imported_rule_conditions_support_profile_flag_enable_disable() {
        let seed_rule = english_seed_variety()
            .weak_form_rules
            .iter()
            .find(|r| r.rule_id == "weak_form_to_before_verb")
            .expect("seed rule must exist");
        let mut native = convert_weak_form_rule(seed_rule, "en", VARIETY_ID_AMERICAN_ENGLISH);
        native.conditions = vec![VarietyRuleCondition {
            language: "en".to_string(),
            variety: VARIETY_ID_AMERICAN_ENGLISH.to_string(),
            voice_profile: None,
            enabled_flags: vec!["profile:fast_reduction".to_string()],
        }];

        let disabled = evaluate_rule_conditions(
            &native,
            &ActiveVarietyProfile {
                language: "en".to_string(),
                variety: VARIETY_ID_AMERICAN_ENGLISH.to_string(),
                ..Default::default()
            },
        );
        assert!(!disabled.applies);
        assert!(
            disabled.reasons.iter().any(|reason| reason.contains("missing enabled flag")),
            "expected missing-flag diagnostic"
        );

        let enabled = evaluate_rule_conditions(
            &native,
            &ActiveVarietyProfile {
                language: "en".to_string(),
                variety: VARIETY_ID_AMERICAN_ENGLISH.to_string(),
                enabled_flags: vec!["profile:fast_reduction".to_string()],
                ..Default::default()
            },
        );
        assert!(enabled.applies);
    }

    #[test]
    fn ga_stub_inheritance_requires_explicit_condition_metadata() {
        let seed_rule = english_seed_variety()
            .weak_form_rules
            .iter()
            .find(|r| r.rule_id == "weak_form_to_before_verb")
            .expect("seed rule must exist");
        let mut native = convert_weak_form_rule(seed_rule, "en", VARIETY_ID_AMERICAN_ENGLISH);
        let active_stub_profile = ActiveVarietyProfile {
            language: "en".to_string(),
            variety: "received_pronunciation".to_string(),
            implementation_status: Some(VarietyImplementationStatus::StubDerivedFrom(VarietyId::new("en-US-GA"))),
            ..Default::default()
        };

        let blocked = evaluate_rule_conditions(&native, &active_stub_profile);
        assert!(!blocked.applies);

        native.conditions = vec![VarietyRuleCondition {
            language: "en".to_string(),
            variety: VARIETY_ID_AMERICAN_ENGLISH.to_string(),
            voice_profile: None,
            enabled_flags: vec![META_ALLOW_STUB_GA_INHERITANCE.to_string()],
        }];
        let allowed = evaluate_rule_conditions(&native, &active_stub_profile);
        assert!(allowed.applies);
        assert!(
            allowed.reasons.iter().any(|reason| reason.contains("stub inheritance")),
            "expected explicit-stub-inheritance diagnostic"
        );
    }

    #[test]
    fn voice_render_conditions_do_not_apply_to_phonological_selection() {
        let seed_rule = english_seed_variety()
            .weak_form_rules
            .iter()
            .find(|r| r.rule_id == "weak_form_to_before_verb")
            .expect("seed rule must exist");
        let mut native = convert_weak_form_rule(seed_rule, "en", VARIETY_ID_AMERICAN_ENGLISH);
        native.conditions = vec![VarietyRuleCondition {
            language: "en".to_string(),
            variety: VARIETY_ID_AMERICAN_ENGLISH.to_string(),
            voice_profile: Some("default".to_string()),
            enabled_flags: vec![META_VOICE_RENDER_ONLY.to_string()],
        }];

        let phonological = evaluate_rule_conditions(
            &native,
            &ActiveVarietyProfile {
                language: "en".to_string(),
                variety: VARIETY_ID_AMERICAN_ENGLISH.to_string(),
                voice_profile: Some("default".to_string()),
                selection_scope: RuleSelectionScope::Phonological,
                ..Default::default()
            },
        );
        assert!(!phonological.applies);
        assert!(
            phonological.reasons.iter().any(|reason| reason.contains("voice/render-only")),
            "expected voice/render leakage diagnostic"
        );

        let voice = evaluate_rule_conditions(
            &native,
            &ActiveVarietyProfile {
                language: "en".to_string(),
                variety: VARIETY_ID_AMERICAN_ENGLISH.to_string(),
                voice_profile: Some("default".to_string()),
                selection_scope: RuleSelectionScope::VoiceRender,
                ..Default::default()
            },
        );
        assert!(voice.applies);
    }

    #[test]
    fn voice_render_conditions_are_imported_from_espeak_voice_variants() {
        let conditions = english_voice_render_conditions();
        assert!(
            !conditions.is_empty(),
            "expected at least one imported voice/render condition"
        );
        assert!(conditions.iter().any(|condition| {
            condition.voice_profile.as_deref() == Some("default")
                && condition.enabled_flags.iter().any(|flag| flag == META_VOICE_RENDER_ONLY)
        }));
    }

    #[test]
    fn converted_weak_form_rule_output_is_phone_string_not_backend_specific() {
        let seed_rule = english_seed_variety()
            .weak_form_rules
            .iter()
            .find(|r| r.rule_id == "weak_form_to_before_verb")
            .expect("seed rule must exist");

        let native = convert_weak_form_rule(seed_rule, "en", "american_english");

        assert!(
            matches!(&native.output, RuleOutput::PhoneString(_)),
            "weak form output must be a PhoneString, not a backend-specific sequence"
        );
        if let RuleOutput::PhoneString(ps) = &native.output {
            // IPA phones should not retain raw ARPAbet tokens (non-empty all-caps+digits)
            for phone in &ps.phones {
                let s = phone.ipa.as_str();
                let looks_like_arpabet = !s.is_empty()
                    && s.chars()
                        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit());
                assert!(
                    !looks_like_arpabet,
                    "IPA output '{}' looks like a raw ARPAbet token",
                    phone.ipa
                );
            }
        }
    }

    #[test]
    fn converted_punctuation_prosody_rule_has_major_boundary_output() {
        let seed_rule = english_seed_variety()
            .punctuation_prosody_rules
            .iter()
            .find(|r| r.match_pattern == "!")
            .expect("exclamation rule must exist");

        let native = convert_punctuation_prosody_rule(seed_rule, "en", "american_english");

        assert_eq!(native.id, "punctuation_exclamation_boundary");
        assert!(
            matches!(&native.output, RuleOutput::ProsodyBoundary { .. }),
            "punctuation output must be a ProsodyBoundary"
        );
        assert!(
            native
                .lexical_flags
                .iter()
                .any(|fact| fact.flag == LexicalProsodyFlag::BreakAfter),
            "punctuation conversion should preserve break-after dictionary flag"
        );
        if let RuleOutput::ProsodyBoundary { boundary, contour } = &native.output {
            assert_eq!(*boundary, ling_env::PhraseBoundaryKind::Major);
            assert_eq!(contour.as_deref(), Some("exclamation"));
        }
    }

    #[test]
    fn english_rule_flag_lookup_covers_multiple_native_flag_kinds() {
        let mut flags = english_lexical_flag_facts_for_rule("weak_form_to_before_verb")
            .into_iter()
            .map(|fact| fact.flag)
            .collect::<Vec<_>>();
        flags.extend(
            english_lexical_flag_facts_for_rule("strong_to_contrastive_uppercase")
                .into_iter()
                .map(|fact| fact.flag),
        );
        flags.extend(
            english_lexical_flag_facts_for_rule("punctuation_exclamation_boundary")
                .into_iter()
                .map(|fact| fact.flag),
        );
        flags.sort_unstable_by_key(|flag| *flag as u8);
        flags.dedup();
        assert!(
            flags.len() >= 5,
            "expected at least five distinct native lexical/prosody flags from imported rules"
        );
    }

    #[test]
    fn converted_punctuation_prosody_rule_preserves_provenance() {
        let seed_rule = english_seed_variety()
            .punctuation_prosody_rules
            .iter()
            .find(|r| r.match_pattern == "!")
            .expect("exclamation rule must exist");

        let native = convert_punctuation_prosody_rule(seed_rule, "en", "american_english");

        assert_eq!(native.provenance.source, "espeak-ng-derived");
        assert_eq!(native.provenance.source_license, "GPL-3.0-or-later");
    }

    #[test]
    fn question_mark_rule_has_final_rising_contour() {
        let seed_rule = english_seed_variety()
            .punctuation_prosody_rules
            .iter()
            .find(|r| r.match_pattern == "?")
            .expect("question mark rule must exist");

        let native = convert_punctuation_prosody_rule(seed_rule, "en", "american_english");

        if let RuleOutput::ProsodyBoundary { boundary, contour } = &native.output {
            assert_eq!(*boundary, ling_env::PhraseBoundaryKind::Major);
            assert_eq!(contour.as_deref(), Some("final_rising"));
        } else {
            panic!("expected ProsodyBoundary output");
        }
    }

    #[test]
    fn bulk_english_weak_form_rules_are_non_empty_and_all_have_provenance() {
        let rules = english_imported_weak_form_rules();
        assert!(
            !rules.is_empty(),
            "should have at least one English weak form rule"
        );
        for rule in &rules {
            assert_eq!(
                rule.provenance.source, "espeak-ng-derived",
                "rule {} provenance should survive bulk conversion",
                rule.id
            );
            assert!(
                matches!(&rule.output, RuleOutput::PhoneString(_)),
                "weak form rule {} output should be a PhoneString",
                rule.id
            );
        }
    }

    #[test]
    fn bulk_english_punctuation_rules_are_non_empty_and_all_have_boundary_output() {
        let rules = english_imported_punctuation_rules();
        assert!(
            !rules.is_empty(),
            "should have at least one English punctuation rule"
        );
        for rule in &rules {
            assert_eq!(
                rule.provenance.source, "espeak-ng-derived",
                "rule {} provenance should survive bulk conversion",
                rule.id
            );
            assert!(
                matches!(&rule.output, RuleOutput::ProsodyBoundary { .. }),
                "punctuation rule {} output should be a ProsodyBoundary",
                rule.id
            );
        }
    }

    // ---------------------------------------------------------------------------
    // BoundaryProsodyRule tests
    // ---------------------------------------------------------------------------

    #[test]
    fn boundary_prosody_rule_exclamation_has_structured_effect() {
        let seed_rule = english_seed_variety()
            .punctuation_prosody_rules
            .iter()
            .find(|r| r.match_pattern == "!")
            .expect("exclamation rule must exist");

        let rule = convert_to_boundary_prosody_rule(seed_rule);

        assert_eq!(rule.id, "punctuation_exclamation_boundary");
        assert_eq!(rule.boundary_pattern, BoundaryPattern::Punctuation('!'));
        assert_eq!(rule.output.boundary_kind, BoundaryKind::Major);
        assert_eq!(rule.output.pitch_direction, Some(PitchDirection::Exclamation));
        assert_eq!(rule.output.stress_effect, StressEffect::ClauseFinalStress);
        assert!(
            rule.output.pitch_range_factor.map_or(false, |f| f > 1.0),
            "exclamation should have expanded pitch range"
        );
        assert!(
            rule.output.suppressible_by.is_empty(),
            "exclamation pause must not be suppressible"
        );
        assert_eq!(rule.provenance.source, "espeak-ng-derived");
    }

    #[test]
    fn boundary_prosody_rule_question_has_final_rising_direction() {
        let seed_rule = english_seed_variety()
            .punctuation_prosody_rules
            .iter()
            .find(|r| r.match_pattern == "?")
            .expect("question mark rule must exist");

        let rule = convert_to_boundary_prosody_rule(seed_rule);

        assert_eq!(rule.id, "punctuation_question_rising_boundary");
        assert_eq!(rule.boundary_pattern, BoundaryPattern::Punctuation('?'));
        assert_eq!(rule.output.boundary_kind, BoundaryKind::Major);
        assert_eq!(rule.output.pitch_direction, Some(PitchDirection::FinalRising));
        assert_eq!(rule.output.stress_effect, StressEffect::ClauseFinalStress);
        assert!(
            rule.output.suppressible_by.is_empty(),
            "question-mark pause must not be suppressible"
        );
    }
    // --- MorphophonologyRule tests ---

    #[test]
    fn english_native_morphophonology_rules_are_non_empty() {
        let rules = english_native_morphophonology_rules();
        assert!(
            !rules.is_empty(),
            "should have at least one English native morphophonology rule"
        );
    }

    #[test]
    fn bulk_english_multi_word_rules_are_imported_as_phrase_level_rules() {
        let rules = english_imported_multi_word_rules();
        assert!(
            !rules.is_empty(),
            "expected at least one imported multi-word phrase rule"
        );
        assert!(
            rules.iter().any(|rule| rule.words == vec!["kind", "of"]),
            "expected function-word phrase seed entry"
        );
        assert!(
            rules.iter().any(|rule| rule.words == vec!["to", "go"]
                && matches!(rule.output, MultiWordRuleOutput::NoBreak)),
            "expected no-break phrase seed entry"
        );
    }

    #[test]
    fn boundary_prosody_rule_period_has_final_falling_direction() {
        let seed_rule = english_seed_variety()
            .punctuation_prosody_rules
            .iter()
            .find(|r| r.match_pattern == ".")
            .expect("period rule must exist");

        let rule = convert_to_boundary_prosody_rule(seed_rule);

        assert_eq!(rule.id, "punctuation_period_final_boundary");
        assert_eq!(rule.boundary_pattern, BoundaryPattern::Punctuation('.'));
        assert_eq!(rule.output.boundary_kind, BoundaryKind::Major);
        assert_eq!(rule.output.pitch_direction, Some(PitchDirection::FinalFalling));
        assert_eq!(rule.output.stress_effect, StressEffect::ClauseFinalStress);
        assert!(
            rule.output.suppressible_by.is_empty(),
            "period pause must not be suppressible"
        );
    }

    #[test]
    fn boundary_prosody_rule_comma_is_minor_and_suppressible() {
        let seed_rule = english_seed_variety()
            .punctuation_prosody_rules
            .iter()
            .find(|r| r.match_pattern == ",")
            .expect("comma rule must exist");

        let rule = convert_to_boundary_prosody_rule(seed_rule);

        assert_eq!(rule.id, "punctuation_comma_minor_boundary");
        assert_eq!(rule.boundary_pattern, BoundaryPattern::Punctuation(','));
        assert_eq!(rule.output.boundary_kind, BoundaryKind::Minor);
        assert!(
            rule.output.suppressible_by.contains(&SuppressibleBy::VocativeContext),
            "comma should be suppressible by vocative context"
        );
        assert!(
            rule.output.suppressible_by.contains(&SuppressibleBy::AppositiveContext),
            "comma should be suppressible by appositive context"
        );
        assert!(
            rule.output.suppressible_by.contains(&SuppressibleBy::ParentheticalContext),
            "comma should be suppressible by parenthetical context"
        );
        assert!(
            rule.output.suppressible_by.contains(&SuppressibleBy::TightSyntacticLink),
            "comma should be suppressible by tight syntactic link"
        );
    }

    #[test]
    fn comma_pause_suppression_by_vocative_evidence() {
        let seed_rule = english_seed_variety()
            .punctuation_prosody_rules
            .iter()
            .find(|r| r.match_pattern == ",")
            .expect("comma rule must exist");

        let rule = convert_to_boundary_prosody_rule(seed_rule);

        // Vocative context suppresses the pause.
        assert!(
            is_comma_pause_suppressed(&rule, &[SuppressibleBy::VocativeContext]),
            "vocative context should suppress comma pause"
        );
        // Appositive context also suppresses the pause.
        assert!(
            is_comma_pause_suppressed(&rule, &[SuppressibleBy::AppositiveContext]),
            "appositive context should suppress comma pause"
        );
        // No active suppressors → pause proceeds.
        assert!(
            !is_comma_pause_suppressed(&rule, &[]),
            "comma pause should not be suppressed without context evidence"
        );
    }

    #[test]
    fn exclamation_pause_is_never_suppressed() {
        let seed_rule = english_seed_variety()
            .punctuation_prosody_rules
            .iter()
            .find(|r| r.match_pattern == "!")
            .expect("exclamation rule must exist");

        let rule = convert_to_boundary_prosody_rule(seed_rule);

        // Even with every possible suppressor, exclamation is never suppressed.
        assert!(
            !is_comma_pause_suppressed(
                &rule,
                &[
                    SuppressibleBy::VocativeContext,
                    SuppressibleBy::AppositiveContext,
                    SuppressibleBy::ParentheticalContext,
                    SuppressibleBy::TightSyntacticLink,
                ]
            ),
            "exclamation pause must never be suppressed"
        );
    }

    #[test]
    fn clause_position_rules_cover_utterance_end_clause_end_and_start() {
        let rules = clause_position_boundary_rules();

        let utterance_end = rules
            .iter()
            .find(|r| r.boundary_pattern == BoundaryPattern::UtteranceEnd)
            .expect("utterance-end clause rule must exist");
        assert_eq!(utterance_end.output.boundary_kind, BoundaryKind::Major);
        assert_eq!(utterance_end.output.pitch_direction, Some(PitchDirection::FinalFalling));
        assert_eq!(utterance_end.output.stress_effect, StressEffect::ClauseFinalStress);

        let clause_end = rules
            .iter()
            .find(|r| r.boundary_pattern == BoundaryPattern::ClauseEnd)
            .expect("clause-end rule must exist");
        assert_eq!(clause_end.output.boundary_kind, BoundaryKind::Major);
        assert_eq!(clause_end.output.pitch_direction, Some(PitchDirection::FinalFalling));

        let clause_start = rules
            .iter()
            .find(|r| r.boundary_pattern == BoundaryPattern::ClauseStart)
            .expect("clause-start rule must exist");
        assert_eq!(clause_start.output.boundary_kind, BoundaryKind::None);
        assert_eq!(clause_start.output.stress_effect, StressEffect::Neutral);
    }

    #[test]
    fn english_boundary_prosody_rules_covers_all_seed_punctuation_and_clause_positions() {
        let rules = english_boundary_prosody_rules();

        assert!(!rules.is_empty(), "bulk rules must not be empty");

        // All seed punctuation rules appear.
        for ch in ['!', '?', '.', ',', ';', ':'] {
            assert!(
                rules
                    .iter()
                    .any(|r| r.boundary_pattern == BoundaryPattern::Punctuation(ch)),
                "bulk rules must include punctuation rule for '{ch}'"
            );
        }

        // Clause-position rules are present.
        assert!(
            rules.iter().any(|r| r.boundary_pattern == BoundaryPattern::UtteranceEnd),
            "bulk rules must include utterance-end clause rule"
        );
        assert!(
            rules.iter().any(|r| r.boundary_pattern == BoundaryPattern::ClauseEnd),
            "bulk rules must include clause-end rule"
        );

        // All rules have espeak provenance.
        for rule in &rules {
            assert_eq!(
                rule.provenance.source, "espeak-ng-derived",
                "rule {} provenance should be espeak-ng-derived",
                rule.id
            );
        }

        // Rules are sorted by descending priority.
        let priorities: Vec<i32> = rules.iter().map(|r| r.priority).collect();
        let mut sorted = priorities.clone();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        assert_eq!(
            priorities, sorted,
            "english_boundary_prosody_rules() must return rules sorted by descending priority"
        );
    }

    #[test]
    fn boundary_prosody_effects_are_structurally_typed_not_strings() {
        // Every BoundaryProsodyRule output uses typed enums, not opaque strings.
        // This is the core acceptance criterion: effects are structural, not string hacks.
        for rule in english_boundary_prosody_rules() {
            // boundary_kind is a typed enum variant — not a String
            let _ = match rule.output.boundary_kind {
                BoundaryKind::None | BoundaryKind::Minor | BoundaryKind::Major => true,
            };
            // pitch_direction is a typed Option<PitchDirection>
            if let Some(dir) = rule.output.pitch_direction {
                let _ = match dir {
                    PitchDirection::FinalFalling
                    | PitchDirection::FinalRising
                    | PitchDirection::Exclamation
                    | PitchDirection::Level => true,
                };
            }
            // stress_effect is a typed enum variant
            let _ = match rule.output.stress_effect {
                StressEffect::ClauseFinalStress
                | StressEffect::StressRelaxation
                | StressEffect::Neutral => true,
            };
        }
    }

    #[test]
    fn morphophonology_rules_cover_required_affixes() {
        let rules = english_native_morphophonology_rules();
        let ids: Vec<&str> = rules.iter().map(|r| r.id.as_str()).collect();
        // At least one suffix and one prefix must be present.
        let has_suffix = rules
            .iter()
            .any(|r| r.morpheme_kind == ling_env::MorphemeKind::Suffix);
        let has_prefix = rules
            .iter()
            .any(|r| r.morpheme_kind == ling_env::MorphemeKind::Prefix);
        assert!(has_suffix, "expected at least one suffix rule; ids: {ids:?}");
        assert!(has_prefix, "expected at least one prefix rule; ids: {ids:?}");
    }

    #[test]
    fn morphophonology_rules_all_have_espeak_provenance() {
        let rules = english_native_morphophonology_rules();
        for rule in &rules {
            assert_eq!(
                rule.provenance.source, "espeak-ng-derived",
                "rule {} should carry eSpeak provenance",
                rule.id
            );
            assert!(
                !rule.provenance.source_license.is_empty(),
                "rule {} should have a non-empty source_license",
                rule.id
            );
        }
    }

    #[test]
    fn morphophonology_rule_ly_has_ito_y_spelling_repair() {
        let rules = english_native_morphophonology_rules();
        let ly_rule = rules
            .iter()
            .find(|r| r.id == "suffix_ly_attachment")
            .expect("suffix_ly_attachment rule must exist");
        assert_eq!(ly_rule.morpheme_kind, ling_env::MorphemeKind::Suffix);
        match &ly_rule.stem_policy {
            StemRetranslationPolicy::SpellingRepair(hints) => {
                assert!(
                    hints.contains(&SpellingRepairHint::IToY),
                    "-ly rule must include IToY spelling repair hint"
                );
            }
            StemRetranslationPolicy::DirectStripAndLookup => {
                panic!("-ly rule must have SpellingRepair policy, not DirectStripAndLookup");
            }
        }
    }

    #[test]
    fn morphophonology_rule_ing_has_doubled_consonant_repair() {
        let rules = english_native_morphophonology_rules();
        let ing_rule = rules
            .iter()
            .find(|r| r.id == "suffix_ing_attachment")
            .expect("suffix_ing_attachment rule must exist");
        match &ing_rule.stem_policy {
            StemRetranslationPolicy::SpellingRepair(hints) => {
                assert!(
                    hints.contains(&SpellingRepairHint::RemoveDoubledConsonant),
                    "-ing rule must include RemoveDoubledConsonant hint"
                );
            }
            StemRetranslationPolicy::DirectStripAndLookup => {
                panic!("-ing rule must have SpellingRepair policy");
            }
        }
    }

    #[test]
    fn morphophonology_rule_ed_covers_all_three_spelling_repairs() {
        let rules = english_native_morphophonology_rules();
        let ed_rule = rules
            .iter()
            .find(|r| r.id == "suffix_ed_attachment")
            .expect("suffix_ed_attachment rule must exist");
        match &ed_rule.stem_policy {
            StemRetranslationPolicy::SpellingRepair(hints) => {
                assert!(hints.contains(&SpellingRepairHint::RestoreTrailingE));
                assert!(hints.contains(&SpellingRepairHint::RemoveDoubledConsonant));
                assert!(hints.contains(&SpellingRepairHint::IToY));
            }
            StemRetranslationPolicy::DirectStripAndLookup => {
                panic!("-ed rule must have SpellingRepair policy");
            }
        }
    }

    #[test]
    fn multi_word_rule_matches_normalized_span_and_preserves_source_spans() {
        let normalizer = TextNormalizer::default();
        let source = "kind of odd";
        let normalized = normalizer.normalize(source).expect("normalize");
        let analysis = HeuristicSentenceAnalyzer.analyze(source, &normalized);
        let rule = english_imported_multi_word_rules()
            .into_iter()
            .find(|rule| rule.id == "phrase_kind_of_reduction")
            .expect("kind-of rule should exist");
        let matches = match_multi_word_rule(&rule, &normalized, &analysis);
        let matched = matches.first().expect("kind-of phrase should match");

        assert_eq!(matched.matched_word_span.word_range, 0..2);
        assert_eq!(matched.matched_word_span.token_range, 0..2);
        assert_eq!(matched.matched_word_span.source_span, 0..7);
        assert_eq!(matched.matched_word_span.words, vec!["kind", "of"]);
        assert_eq!(matched.provenance.source, "espeak-ng-derived");
    }

    #[test]
    fn multi_word_link_requirements_gate_break_suppression_matches() {
        let normalizer = TextNormalizer::default();
        let source = "to go now";
        let normalized = normalizer.normalize(source).expect("normalize");
        let analysis = HeuristicSentenceAnalyzer.analyze(source, &normalized);
        let rule = english_imported_multi_word_rules()
            .into_iter()
            .find(|rule| rule.id == "phrase_to_go_no_break")
            .expect("to-go no-break rule should exist");

        let matches = match_multi_word_rule(&rule, &normalized, &analysis);
        assert_eq!(
            matches.len(),
            1,
            "expected infinitival phrase to match once"
        );

        let mut wrong_link_rule = rule.clone();
        wrong_link_rule.required_links = vec!["Determiner".to_string()];
        assert!(
            match_multi_word_rule(&wrong_link_rule, &normalized, &analysis).is_empty(),
            "link-constrained matching should fail when required link kind is absent"
        );
    }
}
