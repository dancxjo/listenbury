use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::linguistic::cmudict::{self, CmuPhoneme, Stress as CmuStress};
use crate::linguistic::language_pack_rules::{
    MorphophonologyOutput, MorphophonologyRule, SourceProvenance, SpellingRepairHint,
    StemRetranslationPolicy, english_native_morphophonology_rules,
};
use crate::linguistic::orthography::OrthographicWord;
use crate::linguistic::pronounce::OrthographyToPhonemes;
use crate::linguistic::sound_it_out::{SoundItOutPronouncer, SoundItOutRules};
use crate::linguistic::variety::{LinguisticVariety, Phonology};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisSource {
    ExactLexicalEntry,
    KnownDerivedEntry,
    ProductiveMorphology,
    SpellingToSoundFallback,
    UnknownWordSafeFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MorphemeKind {
    Prefix,
    Stem,
    Suffix,
    Clitic,
    CompoundMember,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MorphemeFeatures {
    pub tags: Vec<String>,
    pub meaning: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhonologicalStress {
    Primary,
    Secondary,
    Unstressed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UnderlyingPhonologicalForm {
    pub symbols: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RealizedPhoneSequence {
    pub symbols: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StressPattern {
    pub levels_by_phone: Vec<Option<PhonologicalStress>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MorphemeBoundary {
    pub phone_index: usize,
    pub label: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisplayNotation {
    Ipa,
    Arpabet,
    EspeakLike,
    SampaLike,
    PiperIds,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhonologicalForm {
    pub underlying: UnderlyingPhonologicalForm,
    pub realized: RealizedPhoneSequence,
    pub stress_pattern: StressPattern,
    pub boundaries: Vec<MorphemeBoundary>,
}

impl PhonologicalForm {
    pub fn display(&self, notation: DisplayNotation) -> String {
        encode_symbols(&self.realized.symbols, notation)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MorphemeAnalysis {
    pub surface: String,
    pub kind: MorphemeKind,
    pub lemma: Option<String>,
    pub features: MorphemeFeatures,
    pub phonology: Option<PhonologicalForm>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MorphologicalAnalysis {
    pub surface: String,
    pub morphemes: Vec<MorphemeAnalysis>,
    pub confidence: f32,
    pub source: AnalysisSource,
    pub phonology: Option<PhonologicalForm>,
    pub rules: Vec<String>,
    pub pipeline: Vec<String>,
    pub parser_spike_path: String,
    /// Provenance records for the morphophonology rules that fired.
    /// Empty for plain lexical lookups; populated for derived forms.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rule_provenance: Vec<SourceProvenance>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordPronunciation {
    pub symbols: Vec<String>,
    pub stress_by_phone: Vec<Option<PhonologicalStress>>,
}

#[derive(Debug, Clone)]
struct AffixPronunciation {
    text: &'static str,
    surface: &'static str,
    lemma: &'static str,
    kind: MorphemeKind,
    tags: &'static [&'static str],
    meaning: Option<&'static str>,
    rule: &'static str,
    pronunciation: WordPronunciation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MorphophonologyResult {
    pub analysis: MorphologicalAnalysis,
    pub pronunciation: WordPronunciation,
}

pub fn analyze_word(surface: &str) -> MorphophonologyResult {
    if let Some(known) = known_derived(surface) {
        return known;
    }
    if let Some(productive) = productive_morphology(surface) {
        return productive;
    }
    if let Some(exact) = exact_lexical(surface) {
        return exact;
    }
    if let Some(spelling) = spelling_fallback(surface) {
        return spelling;
    }
    safe_unknown(surface)
}

fn exact_lexical(surface: &str) -> Option<MorphophonologyResult> {
    let stem = lexicon_pronunciation(surface)?;
    let phonology = phonology_form(
        stem.symbols.clone(),
        stem.symbols.clone(),
        stem.stress_by_phone.clone(),
        Vec::new(),
    );
    Some(MorphophonologyResult {
        analysis: MorphologicalAnalysis {
            surface: surface.to_string(),
            morphemes: vec![MorphemeAnalysis {
                surface: surface.to_string(),
                kind: MorphemeKind::Stem,
                lemma: Some(surface.to_string()),
                features: MorphemeFeatures::default(),
                phonology: Some(phonology.clone()),
            }],
            confidence: 1.0,
            source: AnalysisSource::ExactLexicalEntry,
            phonology: Some(phonology),
            rules: vec![
                "stem_lookup_exact".to_string(),
                "stress_assignment".to_string(),
            ],
            pipeline: default_pipeline(),
            parser_spike_path: parser_spike_path(),
            rule_provenance: vec![],
        },
        pronunciation: stem,
    })
}

fn known_derived(surface: &str) -> Option<MorphophonologyResult> {
    if surface.eq_ignore_ascii_case("unpunctuated") {
        return known_unpunctuated(surface);
    }
    None
}

fn known_unpunctuated(surface: &str) -> Option<MorphophonologyResult> {
    let stem = lexicon_pronunciation("punctuate")?;
    let prefix_symbols = vec!["AH1".to_string(), "N".to_string()];
    let prefix_stress = vec![Some(PhonologicalStress::Primary), None];
    let ed = ed_suffix_from_stem(&stem.symbols)?;

    let mut realized = prefix_symbols.clone();
    realized.extend(stem.symbols.clone());
    realized.extend(ed.realized.clone());

    let mut underlying = prefix_symbols;
    underlying.extend(stem.symbols.clone());
    underlying.extend(["EH0".to_string(), "D".to_string()]);

    let mut stress = prefix_stress;
    stress.extend(stem.stress_by_phone.clone());
    stress.extend(ed.stress.clone());

    let boundaries = vec![
        MorphemeBoundary {
            phone_index: 2,
            label: "un-".to_string(),
        },
        MorphemeBoundary {
            phone_index: 2 + stem.symbols.len(),
            label: "-ed".to_string(),
        },
    ];

    let phonology = phonology_form(
        underlying,
        realized.clone(),
        stress.clone(),
        boundaries.clone(),
    );

    Some(MorphophonologyResult {
        analysis: MorphologicalAnalysis {
            surface: surface.to_string(),
            morphemes: vec![
                MorphemeAnalysis {
                    surface: "un-".to_string(),
                    kind: MorphemeKind::Prefix,
                    lemma: Some("un".to_string()),
                    features: MorphemeFeatures {
                        tags: vec!["negative_reversive".to_string()],
                        meaning: Some("negative/reversive".to_string()),
                    },
                    phonology: None,
                },
                MorphemeAnalysis {
                    surface: "punctuate".to_string(),
                    kind: MorphemeKind::Stem,
                    lemma: Some("punctuate".to_string()),
                    features: MorphemeFeatures::default(),
                    phonology: None,
                },
                MorphemeAnalysis {
                    surface: "-ed".to_string(),
                    kind: MorphemeKind::Suffix,
                    lemma: Some("ed".to_string()),
                    features: MorphemeFeatures {
                        tags: vec!["past_participle".to_string()],
                        meaning: None,
                    },
                    phonology: None,
                },
            ],
            confidence: 0.98,
            source: AnalysisSource::KnownDerivedEntry,
            phonology: Some(phonology.clone()),
            rules: vec![
                "prefix_un_attachment".to_string(),
                "stem_lookup_punctuate".to_string(),
                format!("{}", ed.rule),
                "stress_assignment".to_string(),
            ],
            pipeline: default_pipeline(),
            parser_spike_path: parser_spike_path(),
            rule_provenance: vec![],
        },
        pronunciation: WordPronunciation {
            symbols: realized,
            stress_by_phone: stress,
        },
    })
}

fn productive_morphology(surface: &str) -> Option<MorphophonologyResult> {
    if surface.len() <= 2 {
        return None;
    }

    if let Some(prefixed) = analyze_known_prefix(surface) {
        return Some(prefixed);
    }
    if let Some(mixed) = analyze_un_plus_stem_plus_ed(surface) {
        return Some(mixed);
    }
    if let Some(un_prefixed) = analyze_un_prefix(surface) {
        return Some(un_prefixed);
    }
    if let Some(ed_word) = analyze_ed_suffix(surface) {
        return Some(ed_word);
    }
    if let Some(ing_word) = analyze_ing_suffix(surface) {
        return Some(ing_word);
    }
    if let Some(ly_word) = analyze_ly_suffix(surface) {
        return Some(ly_word);
    }
    if let Some(possessive_word) = analyze_possessive_suffix(surface) {
        return Some(possessive_word);
    }
    if let Some(s_word) = analyze_s_suffix(surface) {
        return Some(s_word);
    }
    None
}

fn analyze_known_prefix(surface: &str) -> Option<MorphophonologyResult> {
    if lexicon_pronunciation(surface).is_some() {
        return None;
    }

    let prefix = known_prefixes()
        .into_iter()
        .find(|prefix| surface_starts_with_affix(surface, prefix.text))?;
    let stem_text = &surface[prefix.text.len()..];
    if stem_text.is_empty() {
        return None;
    }

    let stem = lexicon_pronunciation(stem_text)?;

    let mut realized = prefix.pronunciation.symbols.clone();
    realized.extend(stem.symbols.clone());

    let mut stress = prefix.pronunciation.stress_by_phone.clone();
    stress.extend(stem.stress_by_phone.clone());

    let boundaries = vec![MorphemeBoundary {
        phone_index: prefix.pronunciation.symbols.len(),
        label: prefix.surface.to_string(),
    }];

    let mut underlying = prefix.pronunciation.symbols.clone();
    underlying.extend(stem.symbols.clone());

    let phonology = phonology_form(underlying, realized.clone(), stress.clone(), boundaries);

    Some(MorphophonologyResult {
        analysis: MorphologicalAnalysis {
            surface: surface.to_string(),
            morphemes: vec![
                MorphemeAnalysis {
                    surface: prefix.surface.to_string(),
                    kind: prefix.kind,
                    lemma: Some(prefix.lemma.to_string()),
                    features: MorphemeFeatures {
                        tags: prefix.tags.iter().map(|tag| (*tag).to_string()).collect(),
                        meaning: prefix.meaning.map(str::to_string),
                    },
                    phonology: None,
                },
                MorphemeAnalysis {
                    surface: stem_text.to_string(),
                    kind: MorphemeKind::Stem,
                    lemma: Some(stem_text.to_string()),
                    features: MorphemeFeatures::default(),
                    phonology: None,
                },
            ],
            confidence: 0.84,
            source: AnalysisSource::ProductiveMorphology,
            phonology: Some(phonology),
            rules: vec![
                prefix.rule.to_string(),
                "affix_lookup_cmudict_variants".to_string(),
                "stem_lookup_or_fallback".to_string(),
                "stress_assignment".to_string(),
            ],
            pipeline: default_pipeline(),
            parser_spike_path: parser_spike_path(),
            rule_provenance: vec![],
        },
        pronunciation: WordPronunciation {
            symbols: realized,
            stress_by_phone: stress,
        },
    })
}

fn analyze_un_plus_stem_plus_ed(surface: &str) -> Option<MorphophonologyResult> {
    let lower = surface.to_ascii_lowercase();
    if !lower.starts_with("un") || !lower.ends_with("ed") {
        return None;
    }
    if lexicon_pronunciation(surface).is_some() {
        return None;
    }
    let inner = &surface[2..surface.len().saturating_sub(2)];
    if inner.is_empty() {
        return None;
    }

    let candidates = stem_candidates_for_ed_base(inner);
    let (stem_text, stem) = candidates
        .iter()
        .find_map(|candidate| lexicon_pronunciation(candidate).map(|p| (candidate.clone(), p)))?;
    let ed = ed_suffix_from_stem(&stem.symbols)?;

    let prefix_symbols = vec!["AH0".to_string(), "N".to_string()];
    let prefix_stress = vec![Some(PhonologicalStress::Unstressed), None];

    let mut realized = prefix_symbols.clone();
    realized.extend(stem.symbols.clone());
    realized.extend(ed.realized.clone());

    let mut underlying = prefix_symbols;
    underlying.extend(stem.symbols.clone());
    underlying.extend(["EH0".to_string(), "D".to_string()]);

    let mut stress = prefix_stress;
    stress.extend(stem.stress_by_phone.clone());
    stress.extend(ed.stress.clone());

    let boundaries = vec![
        MorphemeBoundary {
            phone_index: 2,
            label: "un-".to_string(),
        },
        MorphemeBoundary {
            phone_index: 2 + stem.symbols.len(),
            label: "-ed".to_string(),
        },
    ];

    let phonology = phonology_form(underlying, realized.clone(), stress.clone(), boundaries);

    Some(MorphophonologyResult {
        analysis: MorphologicalAnalysis {
            surface: surface.to_string(),
            morphemes: vec![
                MorphemeAnalysis {
                    surface: "un-".to_string(),
                    kind: MorphemeKind::Prefix,
                    lemma: Some("un".to_string()),
                    features: MorphemeFeatures {
                        tags: vec!["negative_reversive".to_string()],
                        meaning: Some("negative/reversive".to_string()),
                    },
                    phonology: None,
                },
                MorphemeAnalysis {
                    surface: stem_text,
                    kind: MorphemeKind::Stem,
                    lemma: None,
                    features: MorphemeFeatures::default(),
                    phonology: None,
                },
                MorphemeAnalysis {
                    surface: "-ed".to_string(),
                    kind: MorphemeKind::Suffix,
                    lemma: Some("ed".to_string()),
                    features: MorphemeFeatures {
                        tags: vec!["past_participle".to_string()],
                        meaning: None,
                    },
                    phonology: None,
                },
            ],
            confidence: 0.82,
            source: AnalysisSource::ProductiveMorphology,
            phonology: Some(phonology),
            rules: vec![
                "prefix_un_attachment".to_string(),
                "stem_lookup_or_fallback".to_string(),
                ed.rule,
                "stress_assignment".to_string(),
            ],
            pipeline: default_pipeline(),
            parser_spike_path: parser_spike_path(),
            rule_provenance: vec![],
        },
        pronunciation: WordPronunciation {
            symbols: realized,
            stress_by_phone: stress,
        },
    })
}

fn analyze_un_prefix(surface: &str) -> Option<MorphophonologyResult> {
    if !surface
        .get(0..2)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("un"))
    {
        return None;
    }
    let stem_text = &surface[2..];
    if stem_text.is_empty() {
        return None;
    }

    let stem = lexicon_pronunciation(stem_text)?;
    let prefix_symbols = vec!["AH0".to_string(), "N".to_string()];
    let prefix_stress = vec![Some(PhonologicalStress::Unstressed), None];

    let mut realized = prefix_symbols.clone();
    realized.extend(stem.symbols.clone());

    let mut stress = prefix_stress;
    stress.extend(stem.stress_by_phone.clone());

    let boundaries = vec![MorphemeBoundary {
        phone_index: 2,
        label: "un-".to_string(),
    }];

    let mut underlying = prefix_symbols;
    underlying.extend(stem.symbols.clone());

    let phonology = phonology_form(underlying, realized.clone(), stress.clone(), boundaries);

    Some(MorphophonologyResult {
        analysis: MorphologicalAnalysis {
            surface: surface.to_string(),
            morphemes: vec![
                MorphemeAnalysis {
                    surface: "un-".to_string(),
                    kind: MorphemeKind::Prefix,
                    lemma: Some("un".to_string()),
                    features: MorphemeFeatures {
                        tags: vec!["negative_reversive".to_string()],
                        meaning: Some("negative/reversive".to_string()),
                    },
                    phonology: None,
                },
                MorphemeAnalysis {
                    surface: stem_text.to_string(),
                    kind: MorphemeKind::Stem,
                    lemma: None,
                    features: MorphemeFeatures::default(),
                    phonology: None,
                },
            ],
            confidence: 0.78,
            source: AnalysisSource::ProductiveMorphology,
            phonology: Some(phonology),
            rules: vec![
                "prefix_un_attachment".to_string(),
                "stem_lookup_or_fallback".to_string(),
                "stress_assignment".to_string(),
            ],
            pipeline: default_pipeline(),
            parser_spike_path: parser_spike_path(),
            rule_provenance: vec![],
        },
        pronunciation: WordPronunciation {
            symbols: realized,
            stress_by_phone: stress,
        },
    })
}

fn analyze_ed_suffix(surface: &str) -> Option<MorphophonologyResult> {
    if surface.len() <= 2
        || !surface
            .get(surface.len().saturating_sub(2)..)
            .is_some_and(|suffix| suffix.eq_ignore_ascii_case("ed"))
    {
        return None;
    }
    if surface.eq_ignore_ascii_case("developped") {
        return None;
    }
    if lexicon_pronunciation(surface).is_some() {
        return None;
    }
    let stem_text = &surface[..surface.len().saturating_sub(2)];
    let candidates = stem_candidates_for_ed_base(stem_text);
    let (resolved_stem_text, stem) = candidates
        .iter()
        .find_map(|candidate| lexicon_pronunciation(candidate).map(|p| (candidate.clone(), p)))?;
    let ed = ed_suffix_from_stem(&stem.symbols)?;

    let mut realized = stem.symbols.clone();
    realized.extend(ed.realized.clone());

    let mut stress = stem.stress_by_phone.clone();
    stress.extend(ed.stress.clone());

    let mut underlying = stem.symbols.clone();
    underlying.extend(["EH0".to_string(), "D".to_string()]);

    let boundaries = vec![MorphemeBoundary {
        phone_index: stem.symbols.len(),
        label: "-ed".to_string(),
    }];

    let phonology = phonology_form(underlying, realized.clone(), stress.clone(), boundaries);

    Some(MorphophonologyResult {
        analysis: MorphologicalAnalysis {
            surface: surface.to_string(),
            morphemes: vec![
                MorphemeAnalysis {
                    surface: resolved_stem_text,
                    kind: MorphemeKind::Stem,
                    lemma: None,
                    features: MorphemeFeatures::default(),
                    phonology: None,
                },
                MorphemeAnalysis {
                    surface: "-ed".to_string(),
                    kind: MorphemeKind::Suffix,
                    lemma: Some("ed".to_string()),
                    features: MorphemeFeatures {
                        tags: vec!["past_participle".to_string()],
                        meaning: None,
                    },
                    phonology: None,
                },
            ],
            confidence: 0.8,
            source: AnalysisSource::ProductiveMorphology,
            phonology: Some(phonology),
            rules: vec![
                "stem_lookup_or_fallback".to_string(),
                ed.rule,
                "stress_assignment".to_string(),
            ],
            pipeline: default_pipeline(),
            parser_spike_path: parser_spike_path(),
            rule_provenance: vec![],
        },
        pronunciation: WordPronunciation {
            symbols: realized,
            stress_by_phone: stress,
        },
    })
}

// ---------------------------------------------------------------------------
// MorphophonologyRule-driven analysis helpers
// ---------------------------------------------------------------------------

/// Return the bundled English morphophonology rules (cached).
fn native_morphophonology_rules() -> &'static [MorphophonologyRule] {
    static RULES: OnceLock<Vec<MorphophonologyRule>> = OnceLock::new();
    RULES.get_or_init(english_native_morphophonology_rules)
}

/// Returns `true` when `ch` is an English vowel letter.
fn is_vowel(ch: char) -> bool {
    matches!(ch, 'a' | 'e' | 'i' | 'o' | 'u')
}

/// Apply the spelling repairs encoded in `stem_policy` to `base` and return
/// the full list of stem candidates to try (most specific first).
fn stem_candidates_with_policy(base: &str, stem_policy: &StemRetranslationPolicy) -> Vec<String> {
    let mut candidates = vec![base.to_string()];
    if let StemRetranslationPolicy::SpellingRepair(hints) = stem_policy {
        for hint in hints {
            match hint {
                SpellingRepairHint::RestoreTrailingE => {
                    candidates.push(format!("{base}e"));
                }
                SpellingRepairHint::RemoveDoubledConsonant => {
                    let chars: Vec<char> = base.chars().collect();
                    if chars.len() >= 2 {
                        let last = chars[chars.len() - 1];
                        let prev = chars[chars.len() - 2];
                        if last == prev && !is_vowel(last) {
                            candidates.push(chars[..chars.len() - 1].iter().collect());
                        }
                    }
                }
                SpellingRepairHint::IToY => {
                    if base.ends_with('i') {
                        let mut y_stem = base.to_string();
                        y_stem.pop();
                        y_stem.push('y');
                        candidates.push(y_stem);
                    }
                }
            }
        }
    }
    candidates.dedup();
    candidates
}

/// Parse a space-separated ARPAbet string into symbol/stress pairs, mirroring
/// the conventions used by the rest of this module.
fn arpabet_str_to_phones(arpabet: &str) -> WordPronunciation {
    let raw_symbols: Vec<&str> = arpabet.split_whitespace().collect();
    let symbols: Vec<String> = raw_symbols.iter().map(|s| (*s).to_string()).collect();
    let stress_by_phone: Vec<Option<PhonologicalStress>> = raw_symbols
        .iter()
        .map(|s| {
            if s.ends_with('1') {
                Some(PhonologicalStress::Primary)
            } else if s.ends_with('2') {
                Some(PhonologicalStress::Secondary)
            } else if s.ends_with('0') {
                Some(PhonologicalStress::Unstressed)
            } else {
                None
            }
        })
        .collect();
    WordPronunciation {
        symbols,
        stress_by_phone,
    }
}

/// Analyze a word as *stem + `-ing`*, backed by the `suffix_ing_attachment`
/// native morphophonology rule.
fn analyze_ing_suffix(surface: &str) -> Option<MorphophonologyResult> {
    let lower = surface.to_ascii_lowercase();
    if !lower.ends_with("ing") || lower.len() <= 3 {
        return None;
    }
    // Skip if the whole word is already in the lexicon.
    if lexicon_pronunciation(surface).is_some() {
        return None;
    }
    let rule = native_morphophonology_rules()
        .iter()
        .find(|r| r.id == "suffix_ing_attachment")?;

    let base = &lower[..lower.len() - 3];
    if base.is_empty() {
        return None;
    }

    let MorphophonologyOutput::AppendArpabet(ref affix_arpabet) = rule.output_policy else {
        return None;
    };
    let affix_phones = arpabet_str_to_phones(affix_arpabet);

    let candidates = stem_candidates_with_policy(base, &rule.stem_policy);
    let (stem_text, stem) = candidates
        .iter()
        .find_map(|c| lexicon_pronunciation(c).map(|p| (c.clone(), p)))?;

    let mut realized = stem.symbols.clone();
    realized.extend(affix_phones.symbols.clone());

    let mut stress = stem.stress_by_phone.clone();
    stress.extend(affix_phones.stress_by_phone.clone());

    let mut underlying = stem.symbols.clone();
    underlying.extend(affix_phones.symbols.clone());

    let boundaries = vec![MorphemeBoundary {
        phone_index: stem.symbols.len(),
        label: "-ing".to_string(),
    }];

    let phonology = phonology_form(underlying, realized.clone(), stress.clone(), boundaries);

    Some(MorphophonologyResult {
        analysis: MorphologicalAnalysis {
            surface: surface.to_string(),
            morphemes: vec![
                MorphemeAnalysis {
                    surface: stem_text,
                    kind: MorphemeKind::Stem,
                    lemma: None,
                    features: MorphemeFeatures::default(),
                    phonology: None,
                },
                MorphemeAnalysis {
                    surface: "-ing".to_string(),
                    kind: MorphemeKind::Suffix,
                    lemma: Some("ing".to_string()),
                    features: MorphemeFeatures {
                        tags: vec!["progressive".to_string(), "gerund".to_string()],
                        meaning: None,
                    },
                    phonology: None,
                },
            ],
            confidence: 0.82,
            source: AnalysisSource::ProductiveMorphology,
            phonology: Some(phonology),
            rules: vec![
                rule.id.clone(),
                "stem_lookup_or_fallback".to_string(),
                "stress_assignment".to_string(),
            ],
            pipeline: default_pipeline(),
            parser_spike_path: parser_spike_path(),
            rule_provenance: vec![rule.provenance.clone()],
        },
        pronunciation: WordPronunciation {
            symbols: realized,
            stress_by_phone: stress,
        },
    })
}

/// Analyze a word as *stem + `-ly`*, backed by the `suffix_ly_attachment`
/// native morphophonology rule.  Handles `y → i` spelling repair (e.g.
/// `"happily"` → `"happy"` + `"-ly"`).
fn analyze_ly_suffix(surface: &str) -> Option<MorphophonologyResult> {
    let lower = surface.to_ascii_lowercase();
    if !lower.ends_with("ly") || lower.len() <= 2 {
        return None;
    }
    if lexicon_pronunciation(surface).is_some() {
        return None;
    }
    let rule = native_morphophonology_rules()
        .iter()
        .find(|r| r.id == "suffix_ly_attachment")?;

    let base = &lower[..lower.len() - 2];
    if base.is_empty() {
        return None;
    }

    let MorphophonologyOutput::AppendArpabet(ref affix_arpabet) = rule.output_policy else {
        return None;
    };
    let affix_phones = arpabet_str_to_phones(affix_arpabet);

    let candidates = stem_candidates_with_policy(base, &rule.stem_policy);
    let (stem_text, stem) = candidates
        .iter()
        .find_map(|c| lexicon_pronunciation(c).map(|p| (c.clone(), p)))?;

    let mut realized = stem.symbols.clone();
    realized.extend(affix_phones.symbols.clone());

    let mut stress = stem.stress_by_phone.clone();
    stress.extend(affix_phones.stress_by_phone.clone());

    let mut underlying = stem.symbols.clone();
    underlying.extend(affix_phones.symbols.clone());

    let boundaries = vec![MorphemeBoundary {
        phone_index: stem.symbols.len(),
        label: "-ly".to_string(),
    }];

    let phonology = phonology_form(underlying, realized.clone(), stress.clone(), boundaries);

    Some(MorphophonologyResult {
        analysis: MorphologicalAnalysis {
            surface: surface.to_string(),
            morphemes: vec![
                MorphemeAnalysis {
                    surface: stem_text,
                    kind: MorphemeKind::Stem,
                    lemma: None,
                    features: MorphemeFeatures::default(),
                    phonology: None,
                },
                MorphemeAnalysis {
                    surface: "-ly".to_string(),
                    kind: MorphemeKind::Suffix,
                    lemma: Some("ly".to_string()),
                    features: MorphemeFeatures {
                        tags: vec!["adverb_forming".to_string()],
                        meaning: None,
                    },
                    phonology: None,
                },
            ],
            confidence: 0.80,
            source: AnalysisSource::ProductiveMorphology,
            phonology: Some(phonology),
            rules: vec![
                rule.id.clone(),
                "stem_lookup_or_fallback".to_string(),
                "stress_assignment".to_string(),
            ],
            pipeline: default_pipeline(),
            parser_spike_path: parser_spike_path(),
            rule_provenance: vec![rule.provenance.clone()],
        },
        pronunciation: WordPronunciation {
            symbols: realized,
            stress_by_phone: stress,
        },
    })
}

/// Analyze a word as *stem + `-s`*, backed by the `suffix_s_attachment`
/// native morphophonology rule.
fn analyze_s_suffix(surface: &str) -> Option<MorphophonologyResult> {
    let lower = surface.to_ascii_lowercase();
    if !lower.ends_with('s') || lower.len() <= 1 {
        return None;
    }
    // Skip if the whole word is already in the lexicon.
    if lexicon_pronunciation(surface).is_some() {
        return None;
    }
    let rule = native_morphophonology_rules()
        .iter()
        .find(|r| r.id == "suffix_s_attachment")?;

    let base = &lower[..lower.len() - 1];
    if base.is_empty() {
        return None;
    }

    let MorphophonologyOutput::AppendArpabet(ref affix_arpabet) = rule.output_policy else {
        return None;
    };

    let candidates = stem_candidates_with_policy(base, &rule.stem_policy);
    let (stem_text, stem) = candidates
        .iter()
        .find_map(|c| lexicon_pronunciation(c).map(|p| (c.clone(), p)))?;

    // Select voiced/voiceless allomorph based on stem-final phone.
    let s_allomorph = s_allomorph_from_stem(&stem.symbols);
    let affix_phones = if s_allomorph != affix_arpabet.as_str() {
        arpabet_str_to_phones(s_allomorph)
    } else {
        arpabet_str_to_phones(affix_arpabet)
    };

    let mut realized = stem.symbols.clone();
    realized.extend(affix_phones.symbols.clone());

    let mut stress = stem.stress_by_phone.clone();
    stress.extend(affix_phones.stress_by_phone.clone());

    let mut underlying = stem.symbols.clone();
    // The underlying form is the canonical /z/ from the rule's output policy;
    // the realized form uses the allomorph selected by s_allomorph_from_stem.
    let canonical_affix = arpabet_str_to_phones(affix_arpabet);
    underlying.extend(canonical_affix.symbols);

    let boundaries = vec![MorphemeBoundary {
        phone_index: stem.symbols.len(),
        label: "-s".to_string(),
    }];

    let phonology = phonology_form(underlying, realized.clone(), stress.clone(), boundaries);

    let allomorph_rule = format!("s_suffix_realization_{}", s_allomorph.to_ascii_lowercase());

    Some(MorphophonologyResult {
        analysis: MorphologicalAnalysis {
            surface: surface.to_string(),
            morphemes: vec![
                MorphemeAnalysis {
                    surface: stem_text,
                    kind: MorphemeKind::Stem,
                    lemma: None,
                    features: MorphemeFeatures::default(),
                    phonology: None,
                },
                MorphemeAnalysis {
                    surface: "-s".to_string(),
                    kind: MorphemeKind::Suffix,
                    lemma: Some("s".to_string()),
                    features: MorphemeFeatures {
                        tags: vec!["plural".to_string(), "third_person_singular".to_string()],
                        meaning: None,
                    },
                    phonology: None,
                },
            ],
            confidence: 0.78,
            source: AnalysisSource::ProductiveMorphology,
            phonology: Some(phonology),
            rules: vec![
                rule.id.clone(),
                allomorph_rule,
                "stem_lookup_or_fallback".to_string(),
                "stress_assignment".to_string(),
            ],
            pipeline: default_pipeline(),
            parser_spike_path: parser_spike_path(),
            rule_provenance: vec![rule.provenance.clone()],
        },
        pronunciation: WordPronunciation {
            symbols: realized,
            stress_by_phone: stress,
        },
    })
}

fn analyze_possessive_suffix(surface: &str) -> Option<MorphophonologyResult> {
    if lexicon_pronunciation(surface).is_some() {
        return None;
    }

    if let Some(stem_text) = possessive_apostrophe_s_stem(surface) {
        return analyze_apostrophe_s_possessive(surface, stem_text);
    }

    if let Some(stem_text) = possessive_s_apostrophe_stem(surface) {
        return analyze_s_apostrophe_possessive(surface, stem_text);
    }

    None
}

fn possessive_apostrophe_s_stem(surface: &str) -> Option<String> {
    let lower = surface.to_ascii_lowercase();
    lower
        .strip_suffix("'s")
        .or_else(|| lower.strip_suffix("’s"))
        .filter(|stem| !stem.is_empty())
        .map(str::to_string)
}

fn possessive_s_apostrophe_stem(surface: &str) -> Option<String> {
    let lower = surface.to_ascii_lowercase();
    lower
        .strip_suffix('\'')
        .or_else(|| lower.strip_suffix('’'))
        .filter(|stem| stem.len() > 1 && stem.ends_with('s'))
        .map(str::to_string)
}

fn analyze_apostrophe_s_possessive(
    surface: &str,
    stem_text: String,
) -> Option<MorphophonologyResult> {
    let stem = lexicon_pronunciation(&stem_text)?;
    let rule = native_morphophonology_rules()
        .iter()
        .find(|r| r.id == "suffix_s_attachment")?;
    let MorphophonologyOutput::AppendArpabet(ref affix_arpabet) = rule.output_policy else {
        return None;
    };

    let s_allomorph = s_allomorph_from_stem(&stem.symbols);
    let affix_phones = arpabet_str_to_phones(s_allomorph);

    let mut realized = stem.symbols.clone();
    realized.extend(affix_phones.symbols.clone());

    let mut stress = stem.stress_by_phone.clone();
    stress.extend(affix_phones.stress_by_phone.clone());

    let mut underlying = stem.symbols.clone();
    let canonical_affix = arpabet_str_to_phones(affix_arpabet);
    underlying.extend(canonical_affix.symbols);

    let boundaries = vec![MorphemeBoundary {
        phone_index: stem.symbols.len(),
        label: "'s".to_string(),
    }];
    let phonology = phonology_form(underlying, realized.clone(), stress.clone(), boundaries);
    let allomorph_rule = format!("s_suffix_realization_{}", s_allomorph.to_ascii_lowercase());

    Some(MorphophonologyResult {
        analysis: MorphologicalAnalysis {
            surface: surface.to_string(),
            morphemes: vec![
                MorphemeAnalysis {
                    surface: stem_text,
                    kind: MorphemeKind::Stem,
                    lemma: None,
                    features: MorphemeFeatures::default(),
                    phonology: None,
                },
                MorphemeAnalysis {
                    surface: "'s".to_string(),
                    kind: MorphemeKind::Clitic,
                    lemma: Some("possessive_s".to_string()),
                    features: MorphemeFeatures {
                        tags: vec!["possessive".to_string(), "clitic_s".to_string()],
                        meaning: Some("possessive".to_string()),
                    },
                    phonology: None,
                },
            ],
            confidence: 0.82,
            source: AnalysisSource::ProductiveMorphology,
            phonology: Some(phonology),
            rules: vec![
                "possessive_s_attachment".to_string(),
                rule.id.clone(),
                allomorph_rule,
                "stem_lookup_or_fallback".to_string(),
                "stress_assignment".to_string(),
            ],
            pipeline: default_pipeline(),
            parser_spike_path: parser_spike_path(),
            rule_provenance: vec![rule.provenance.clone()],
        },
        pronunciation: WordPronunciation {
            symbols: realized,
            stress_by_phone: stress,
        },
    })
}

fn analyze_s_apostrophe_possessive(
    surface: &str,
    stem_text: String,
) -> Option<MorphophonologyResult> {
    let stem_result = exact_lexical(&stem_text).or_else(|| analyze_s_suffix(&stem_text))?;
    let mut morphemes = stem_result.analysis.morphemes.clone();
    morphemes.push(MorphemeAnalysis {
        surface: "'".to_string(),
        kind: MorphemeKind::Clitic,
        lemma: Some("possessive_apostrophe".to_string()),
        features: MorphemeFeatures {
            tags: vec![
                "possessive".to_string(),
                "zero_possessive_after_s".to_string(),
            ],
            meaning: Some("possessive".to_string()),
        },
        phonology: None,
    });

    let mut phonology = stem_result.analysis.phonology.clone();
    if let Some(form) = phonology.as_mut() {
        form.boundaries.push(MorphemeBoundary {
            phone_index: stem_result.pronunciation.symbols.len(),
            label: "s'".to_string(),
        });
    }

    let mut rules = stem_result.analysis.rules.clone();
    rules.insert(0, "possessive_apostrophe_after_s".to_string());

    Some(MorphophonologyResult {
        analysis: MorphologicalAnalysis {
            surface: surface.to_string(),
            morphemes,
            confidence: 0.80,
            source: AnalysisSource::ProductiveMorphology,
            phonology,
            rules,
            pipeline: default_pipeline(),
            parser_spike_path: parser_spike_path(),
            rule_provenance: stem_result.analysis.rule_provenance,
        },
        pronunciation: stem_result.pronunciation,
    })
}

/// Select the correct surface allomorph for `-s` based on the stem-final phone.
/// Returns `"Z"` (voiced), `"S"` (voiceless), or `"IH0 Z"` (after sibilants).
fn s_allomorph_from_stem(stem_symbols: &[String]) -> &'static str {
    let last = stem_symbols
        .last()
        .map(|s| s.trim_end_matches(|c: char| c.is_ascii_digit()))
        .unwrap_or("");
    if matches!(last, "S" | "Z" | "SH" | "ZH" | "CH" | "JH") {
        return "IH0 Z";
    }
    if is_voiceless(last) {
        return "S";
    }
    "Z"
}

#[derive(Debug, Clone)]
struct EdAllomorph {
    realized: Vec<String>,
    stress: Vec<Option<PhonologicalStress>>,
    rule: String,
}

fn ed_suffix_from_stem(stem_symbols: &[String]) -> Option<EdAllomorph> {
    let last = stem_symbols
        .last()?
        .trim_end_matches(|ch: char| ch.is_ascii_digit());
    if matches!(last, "T" | "D") {
        return Some(EdAllomorph {
            realized: vec!["IH0".to_string(), "D".to_string()],
            stress: vec![Some(PhonologicalStress::Unstressed), None],
            rule: "ed_suffix_realization_id_after_t_or_d".to_string(),
        });
    }
    if is_voiceless(last) {
        return Some(EdAllomorph {
            realized: vec!["T".to_string()],
            stress: vec![None],
            rule: "ed_suffix_realization_t_after_voiceless".to_string(),
        });
    }
    Some(EdAllomorph {
        realized: vec!["D".to_string()],
        stress: vec![None],
        rule: "ed_suffix_realization_d_after_voiced".to_string(),
    })
}

fn is_voiceless(symbol: &str) -> bool {
    matches!(symbol, "P" | "T" | "K" | "F" | "S" | "SH" | "CH" | "TH")
}

fn spelling_fallback(surface: &str) -> Option<MorphophonologyResult> {
    let pronunciation = fallback_pronunciation(surface)?;
    let phonology = phonology_form(
        pronunciation.symbols.clone(),
        pronunciation.symbols.clone(),
        pronunciation.stress_by_phone.clone(),
        Vec::new(),
    );
    Some(MorphophonologyResult {
        analysis: MorphologicalAnalysis {
            surface: surface.to_string(),
            morphemes: vec![MorphemeAnalysis {
                surface: surface.to_string(),
                kind: MorphemeKind::Stem,
                lemma: None,
                features: MorphemeFeatures {
                    tags: vec!["spelling_fallback".to_string()],
                    meaning: None,
                },
                phonology: None,
            }],
            confidence: 0.45,
            source: AnalysisSource::SpellingToSoundFallback,
            phonology: Some(phonology),
            rules: vec![
                "spelling_to_sound_fallback".to_string(),
                "stress_assignment".to_string(),
            ],
            pipeline: default_pipeline(),
            parser_spike_path: parser_spike_path(),
            rule_provenance: vec![],
        },
        pronunciation,
    })
}

fn safe_unknown(surface: &str) -> MorphophonologyResult {
    let symbols = vec!["AH0".to_string()];
    let stress = vec![Some(PhonologicalStress::Unstressed)];
    let phonology = phonology_form(symbols.clone(), symbols.clone(), stress.clone(), Vec::new());
    MorphophonologyResult {
        analysis: MorphologicalAnalysis {
            surface: surface.to_string(),
            morphemes: vec![MorphemeAnalysis {
                surface: surface.to_string(),
                kind: MorphemeKind::Stem,
                lemma: None,
                features: MorphemeFeatures {
                    tags: vec!["safe_unknown_fallback".to_string()],
                    meaning: None,
                },
                phonology: None,
            }],
            confidence: 0.15,
            source: AnalysisSource::UnknownWordSafeFallback,
            phonology: Some(phonology),
            rules: vec!["unknown_word_safe_fallback".to_string()],
            pipeline: default_pipeline(),
            parser_spike_path: parser_spike_path(),
            rule_provenance: vec![],
        },
        pronunciation: WordPronunciation {
            symbols,
            stress_by_phone: stress,
        },
    }
}

fn lexicon_pronunciation(surface: &str) -> Option<WordPronunciation> {
    if let Some(pronunciation) = lexical_override(surface) {
        return Some(pronunciation);
    }
    let phones = cmudict::bundled().lookup(surface)?;
    Some(cmu_phones_to_symbols(phones))
}

fn known_prefixes() -> Vec<AffixPronunciation> {
    let mut prefixes = Vec::new();
    if let Some(pronunciation) = affix_pronunciation("di", AffixVariantPreference::PreferAy) {
        prefixes.push(AffixPronunciation {
            text: "di",
            surface: "di-",
            lemma: "di",
            kind: MorphemeKind::Prefix,
            tags: &["technical_prefix"],
            meaning: Some("two"),
            rule: "prefix_di_attachment",
            pronunciation,
        });
    }
    prefixes
}

fn surface_starts_with_affix(surface: &str, affix: &str) -> bool {
    surface
        .get(..affix.len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(affix))
}

#[derive(Debug, Clone, Copy)]
enum AffixVariantPreference {
    PreferAy,
}

fn affix_pronunciation(
    surface: &str,
    preference: AffixVariantPreference,
) -> Option<WordPronunciation> {
    let variants = affix_pronunciations(surface)?;
    match preference {
        AffixVariantPreference::PreferAy => variants
            .iter()
            .find(|pronunciation| {
                pronunciation
                    .symbols
                    .iter()
                    .any(|symbol| symbol.trim_end_matches(|ch: char| ch.is_ascii_digit()) == "AY")
            })
            .cloned()
            .or_else(|| variants.into_iter().next()),
    }
}

fn affix_pronunciations(surface: &str) -> Option<Vec<WordPronunciation>> {
    cmudict::bundled().lookup_all(surface).map(|variants| {
        variants
            .iter()
            .map(|phones| WordPronunciation {
                symbols: phones.iter().map(cmu_phone_source_symbol).collect(),
                stress_by_phone: phones
                    .iter()
                    .map(|phone| cmu_stress_level(phone.stress))
                    .collect(),
            })
            .collect()
    })
}

fn lexical_override(surface: &str) -> Option<WordPronunciation> {
    type LexicalOverrideEntry<'a> = (&'a str, &'a [(&'a str, Option<PhonologicalStress>)]);

    let entries: &[LexicalOverrideEntry<'_>] = &[
        (
            "model",
            &[
                ("M", None),
                ("AA1", Some(PhonologicalStress::Primary)),
                ("D", None),
                ("AH0", Some(PhonologicalStress::Unstressed)),
                ("L", None),
            ],
        ),
        (
            "nuclei",
            &[
                ("N", None),
                ("UW1", Some(PhonologicalStress::Primary)),
                ("K", None),
                ("L", None),
                ("IY0", Some(PhonologicalStress::Unstressed)),
                ("AY2", Some(PhonologicalStress::Secondary)),
            ],
        ),
        (
            "periodic",
            &[
                ("P", None),
                ("IH2", Some(PhonologicalStress::Secondary)),
                ("R", None),
                ("IY0", Some(PhonologicalStress::Unstressed)),
                ("AA1", Some(PhonologicalStress::Primary)),
                ("D", None),
                ("IH0", Some(PhonologicalStress::Unstressed)),
                ("K", None),
            ],
        ),
        (
            "punctuate",
            &[
                ("P", None),
                ("AH1", Some(PhonologicalStress::Primary)),
                ("NG", None),
                ("K", None),
                ("CH", None),
                ("UW", Some(PhonologicalStress::Unstressed)),
                ("EY2", Some(PhonologicalStress::Secondary)),
                ("T", None),
            ],
        ),
        (
            "embody",
            &[
                ("IH0", Some(PhonologicalStress::Unstressed)),
                ("M", None),
                ("B", None),
                ("AA1", Some(PhonologicalStress::Primary)),
                ("D", None),
                ("IY0", Some(PhonologicalStress::Unstressed)),
            ],
        ),
        (
            "embodied",
            &[
                ("IH0", Some(PhonologicalStress::Unstressed)),
                ("M", None),
                ("B", None),
                ("AA1", Some(PhonologicalStress::Primary)),
                ("D", None),
                ("IY0", Some(PhonologicalStress::Unstressed)),
                ("D", None),
            ],
        ),
    ];

    let lower = surface.to_ascii_lowercase();
    let phones = entries
        .iter()
        .find_map(|(word, phones)| (*word == lower).then_some(*phones))?;
    Some(WordPronunciation {
        symbols: phones
            .iter()
            .map(|(symbol, _)| (*symbol).to_string())
            .collect(),
        stress_by_phone: phones.iter().map(|(_, stress)| *stress).collect(),
    })
}

fn fallback_pronunciation(surface: &str) -> Option<WordPronunciation> {
    let ortho = OrthographicWord::new(surface);
    let variety = LinguisticVariety::untagged("en-US-fallback", Phonology::new("English fallback"));
    fallback_english_pronouncer()
        .realize_word(&variety, &ortho)
        .ok()
        .map(|seq| {
            let symbols = seq
                .phonemes
                .into_iter()
                .map(|phoneme| phoneme.symbol)
                .collect::<Vec<_>>();
            let stress_by_phone = vec![None; symbols.len()];
            WordPronunciation {
                symbols,
                stress_by_phone,
            }
        })
        .filter(|pronunciation| !pronunciation.symbols.is_empty())
}

fn fallback_english_pronouncer() -> &'static SoundItOutPronouncer {
    static FALLBACK: OnceLock<SoundItOutPronouncer> = OnceLock::new();
    FALLBACK.get_or_init(|| SoundItOutPronouncer::new(SoundItOutRules::english_arpabet_fallback()))
}

fn cmu_phones_to_symbols(phones: &[CmuPhoneme]) -> WordPronunciation {
    let symbols = phones
        .iter()
        .map(|source| {
            if source.base == "AH" {
                cmu_phone_source_symbol(source)
            } else {
                source.base.clone()
            }
        })
        .collect::<Vec<_>>();

    let stress_by_phone = phones
        .iter()
        .map(|phone| cmu_stress_level(phone.stress))
        .collect();

    WordPronunciation {
        symbols,
        stress_by_phone,
    }
}

fn cmu_phone_source_symbol(phone: &CmuPhoneme) -> String {
    match phone.stress {
        Some(stress) => format!("{}{}", phone.base, cmu_stress_digit(stress)),
        None => phone.base.clone(),
    }
}

fn cmu_stress_digit(stress: CmuStress) -> char {
    match stress {
        CmuStress::Primary => '1',
        CmuStress::Secondary => '2',
        CmuStress::Unstressed => '0',
    }
}

fn cmu_stress_level(stress: Option<CmuStress>) -> Option<PhonologicalStress> {
    match stress {
        Some(CmuStress::Primary) => Some(PhonologicalStress::Primary),
        Some(CmuStress::Secondary) => Some(PhonologicalStress::Secondary),
        Some(CmuStress::Unstressed) => Some(PhonologicalStress::Unstressed),
        None => None,
    }
}

fn phonology_form(
    underlying: Vec<String>,
    realized: Vec<String>,
    stress_by_phone: Vec<Option<PhonologicalStress>>,
    boundaries: Vec<MorphemeBoundary>,
) -> PhonologicalForm {
    PhonologicalForm {
        underlying: UnderlyingPhonologicalForm {
            symbols: underlying,
        },
        realized: RealizedPhoneSequence { symbols: realized },
        stress_pattern: StressPattern {
            levels_by_phone: stress_by_phone,
        },
        boundaries,
    }
}

fn default_pipeline() -> Vec<String> {
    vec![
        "orthographic_token".to_string(),
        "morphological_segmentation".to_string(),
        "lexical_stem_lookup".to_string(),
        "morpheme_features".to_string(),
        "phonological_representation".to_string(),
        "stress_assignment".to_string(),
        "allomorphy_morphophonemic_rules".to_string(),
        "connected_speech_reduction".to_string(),
        "riper_phone_sequence".to_string(),
        "prosody_diagnostics".to_string(),
    ]
}

fn parser_spike_path() -> String {
    "custom_rule_engine_with_treebender_evaluation_path".to_string()
}

fn stem_candidates_for_ed_base(base: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    candidates.push(base.to_string());
    candidates.push(format!("{base}e"));

    let chars: Vec<char> = base.chars().collect();
    if chars.len() >= 2 {
        let last = chars[chars.len() - 1];
        let prev = chars[chars.len() - 2];
        if last == prev && !matches!(last, 'a' | 'e' | 'i' | 'o' | 'u') {
            let stem = chars[..chars.len() - 1].iter().collect::<String>();
            candidates.push(stem);
        }
    }

    if base.ends_with('i') {
        let mut y_stem = base.to_string();
        y_stem.pop();
        y_stem.push('y');
        candidates.push(y_stem);
    }

    candidates.dedup();
    candidates
}

fn encode_symbols(symbols: &[String], notation: DisplayNotation) -> String {
    match notation {
        DisplayNotation::Arpabet => symbols.join(" "),
        DisplayNotation::Ipa => symbols
            .iter()
            .map(|symbol| arpabet_to_ipa(symbol))
            .collect::<Vec<_>>()
            .join(""),
        DisplayNotation::EspeakLike => symbols
            .iter()
            .map(|symbol| arpabet_to_espeak_like(symbol))
            .collect::<Vec<_>>()
            .join(" "),
        DisplayNotation::SampaLike => symbols
            .iter()
            .map(|symbol| arpabet_to_sampa_like(symbol))
            .collect::<Vec<_>>()
            .join(" "),
        DisplayNotation::PiperIds => symbols
            .iter()
            .map(|symbol| format!("piper:{symbol}"))
            .collect::<Vec<_>>()
            .join(" "),
    }
}

fn arpabet_to_ipa(symbol: &str) -> String {
    let stress = symbol.chars().next_back();
    let marker = match stress {
        Some('1') => "ˈ",
        Some('2') => "ˌ",
        _ => "",
    };
    let base = symbol.trim_end_matches(|ch: char| ch.is_ascii_digit());
    let ipa = match base {
        "AA" => "ɑ",
        "AE" => "æ",
        "AH" => {
            if matches!(stress, Some('0')) {
                "ə"
            } else {
                "ʌ"
            }
        }
        "AO" => "ɔ",
        "AW" => "aʊ",
        "AY" => "aɪ",
        "B" => "b",
        "CH" => "tʃ",
        "D" => "d",
        "DH" => "ð",
        "DX" => "ɾ",
        "EH" => "ɛ",
        "ER" => "ɚ",
        "EY" => "eɪ",
        "F" => "f",
        "G" => "ɡ",
        "HH" => "h",
        "IH" => "ɪ",
        "IY" => "i",
        "JH" => "dʒ",
        "K" => "k",
        "L" => "l",
        "M" => "m",
        "N" => "n",
        "NG" => "ŋ",
        "OW" => "oʊ",
        "OY" => "ɔɪ",
        "P" => "p",
        "R" => "ɹ",
        "S" => "s",
        "SH" => "ʃ",
        "T" => "t",
        "TH" => "θ",
        "UH" => "ʊ",
        "UW" => "u",
        "V" => "v",
        "W" => "w",
        "Y" => "j",
        "Z" => "z",
        "ZH" => "ʒ",
        "ɾ" => "ɾ",
        other => other,
    };
    format!("{marker}{ipa}")
}

fn arpabet_to_espeak_like(symbol: &str) -> String {
    match symbol.trim_end_matches(|ch: char| ch.is_ascii_digit()) {
        "AH" => "@".to_string(),
        "ER" => "3".to_string(),
        "CH" => "tS".to_string(),
        "SH" => "S".to_string(),
        "ZH" => "Z".to_string(),
        "TH" => "T".to_string(),
        "DH" => "D".to_string(),
        "NG" => "N".to_string(),
        other => other.to_ascii_lowercase(),
    }
}

fn arpabet_to_sampa_like(symbol: &str) -> String {
    match symbol.trim_end_matches(|ch: char| ch.is_ascii_digit()) {
        "AH" => "V".to_string(),
        "ER" => "3`".to_string(),
        "CH" => "tS".to_string(),
        "SH" => "S".to_string(),
        "ZH" => "Z".to_string(),
        "TH" => "T".to_string(),
        "DH" => "D".to_string(),
        "NG" => "N".to_string(),
        other => other.to_ascii_lowercase(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symbols_of(word: &str) -> Vec<String> {
        analyze_word(word).pronunciation.symbols
    }

    #[test]
    fn analyzes_seed_unpunctuated_with_morpheme_diagnostics() {
        let result = analyze_word("unpunctuated");
        assert_eq!(
            result
                .analysis
                .morphemes
                .iter()
                .map(|m| m.surface.as_str())
                .collect::<Vec<_>>(),
            vec!["un-", "punctuate", "-ed"]
        );
        assert!(
            result
                .analysis
                .rules
                .iter()
                .any(|rule| rule.starts_with("ed_suffix_realization_"))
        );
        let ipa = result
            .analysis
            .phonology
            .as_ref()
            .expect("phonology")
            .display(DisplayNotation::Ipa);
        assert!(ipa.contains("ˈ"), "expected stress mark in IPA: {ipa}");
        assert_eq!(
            result.pronunciation.symbols,
            vec![
                "AH1", "N", "P", "AH1", "NG", "K", "CH", "UW", "EY2", "T", "IH0", "D"
            ],
            "morphophonology should keep the phonemic /t/ before G2P surface realization"
        );
    }

    #[test]
    fn analyzes_diphones_as_di_plus_phones() {
        let result = analyze_word("diphones");
        assert_eq!(
            result
                .analysis
                .morphemes
                .iter()
                .map(|m| m.surface.as_str())
                .collect::<Vec<_>>(),
            vec!["di-", "phones"]
        );
        assert!(matches!(
            result.analysis.source,
            AnalysisSource::ProductiveMorphology
        ));
        assert_eq!(
            result.pronunciation.symbols,
            vec!["D", "AY1", "F", "OW", "N", "Z"]
        );
        assert!(
            result
                .analysis
                .rules
                .iter()
                .any(|rule| rule == "prefix_di_attachment")
        );
    }

    #[test]
    fn pulls_di_prefix_variants_from_cmudict() {
        let variants = affix_pronunciations("di").expect("di prefix variants");
        let symbols = variants
            .iter()
            .map(|pronunciation| pronunciation.symbols.clone())
            .collect::<Vec<_>>();
        assert!(symbols.contains(&vec!["D".to_string(), "IY1".to_string()]));
        assert!(symbols.contains(&vec!["D".to_string(), "AY1".to_string()]));
        assert_eq!(
            affix_pronunciation("di", AffixVariantPreference::PreferAy)
                .expect("preferred di prefix")
                .symbols,
            vec!["D", "AY1"]
        );
    }

    #[test]
    fn realizes_ed_allomorphy() {
        let walked_stem = lexicon_pronunciation("walk").expect("walk stem");
        let played_stem = lexicon_pronunciation("play").expect("play stem");
        let wanted_stem = lexicon_pronunciation("want").expect("want stem");
        let needed_stem = lexicon_pronunciation("need").expect("need stem");

        assert_eq!(
            ed_suffix_from_stem(&walked_stem.symbols)
                .expect("walk + ed")
                .realized,
            vec!["T"]
        );
        assert_eq!(
            ed_suffix_from_stem(&played_stem.symbols)
                .expect("play + ed")
                .realized,
            vec!["D"]
        );
        assert_eq!(
            ed_suffix_from_stem(&wanted_stem.symbols)
                .expect("want + ed")
                .realized,
            vec!["IH0", "D"]
        );
        assert_eq!(
            ed_suffix_from_stem(&needed_stem.symbols)
                .expect("need + ed")
                .realized,
            vec!["IH0", "D"]
        );
    }

    #[test]
    fn analyzes_apostrophe_s_possessive_with_s_allomorphy() {
        let result = analyze_word("twilight's");
        assert_eq!(
            result
                .analysis
                .morphemes
                .iter()
                .map(|m| m.surface.as_str())
                .collect::<Vec<_>>(),
            vec!["twilight", "'s"]
        );
        assert_eq!(
            result.pronunciation.symbols,
            vec!["T", "W", "AY", "L", "AY", "T", "S"]
        );
        assert!(
            result
                .analysis
                .rules
                .iter()
                .any(|rule| rule == "possessive_s_attachment")
        );
    }

    #[test]
    fn analyzes_s_apostrophe_possessive_without_extra_phone() {
        let result = analyze_word("twilights'");
        assert_eq!(
            result.pronunciation.symbols,
            vec!["T", "W", "AY", "L", "AY", "T", "S"]
        );
        assert!(
            result
                .analysis
                .rules
                .iter()
                .any(|rule| rule == "possessive_apostrophe_after_s")
        );
    }

    #[test]
    fn applies_sample_sentence_lexical_overrides() {
        assert_eq!(symbols_of("model"), vec!["M", "AA1", "D", "AH0", "L"]);
        assert_eq!(
            symbols_of("nuclei"),
            vec!["N", "UW1", "K", "L", "IY0", "AY2"]
        );
        assert_eq!(
            symbols_of("periodic"),
            vec!["P", "IH2", "R", "IY0", "AA1", "D", "IH0", "K"]
        );
        assert_eq!(
            symbols_of("punctuate"),
            vec!["P", "AH1", "NG", "K", "CH", "UW", "EY2", "T"]
        );
        assert_eq!(
            symbols_of("embodied"),
            vec!["IH0", "M", "B", "AA1", "D", "IY0", "D"]
        );
    }

    #[test]
    fn keeps_embodied_as_exact_lexical_ir() {
        let result = analyze_word("embodied");
        assert_eq!(result.analysis.source, AnalysisSource::ExactLexicalEntry);
        assert_eq!(
            result
                .analysis
                .morphemes
                .iter()
                .map(|m| m.surface.as_str())
                .collect::<Vec<_>>(),
            vec!["embodied"]
        );
        assert_eq!(
            result
                .analysis
                .phonology
                .as_ref()
                .expect("phonology")
                .underlying
                .symbols,
            vec!["IH0", "M", "B", "AA1", "D", "IY0", "D"]
        );
    }

    #[test]
    fn analyzes_un_prefix_words() {
        let unhappy = analyze_word("unhappy");
        assert_eq!(unhappy.analysis.morphemes[0].surface, "un-");
        assert_eq!(unhappy.analysis.morphemes[1].surface, "happy");

        let unfair = analyze_word("unfair");
        assert_eq!(unfair.analysis.morphemes[0].surface, "un-");
        assert_eq!(unfair.analysis.morphemes[1].surface, "fair");
    }

    #[test]
    fn keeps_notation_views_separate() {
        let result = analyze_word("unpunctuated");
        let phonology = result.analysis.phonology.as_ref().expect("phonology");
        let arpabet = phonology.display(DisplayNotation::Arpabet);
        let ipa = phonology.display(DisplayNotation::Ipa);
        let piper = phonology.display(DisplayNotation::PiperIds);
        assert_ne!(arpabet, ipa);
        assert_ne!(ipa, piper);
    }

    #[test]
    fn falls_back_for_unknown_derived_looking_words() {
        let result = analyze_word("unblorfed");
        assert!(!result.pronunciation.symbols.is_empty());
        assert!(matches!(
            result.analysis.source,
            AnalysisSource::ProductiveMorphology
                | AnalysisSource::SpellingToSoundFallback
                | AnalysisSource::UnknownWordSafeFallback
        ));
    }

    // --- Native MorphophonologyRule-driven suffix tests ---

    /// `"crispily"` is not in CMUdict; the `-ly` rule must find stem `"crispy"`
    /// via the `IToY` spelling repair and append the `-ly` phones.
    #[test]
    fn analyzes_crispily_via_ito_y_repair_and_ly_rule() {
        let result = analyze_word("crispily");
        let morphemes: Vec<&str> = result
            .analysis
            .morphemes
            .iter()
            .map(|m| m.surface.as_str())
            .collect();
        // Stem resolved via y→i repair; "-ly" appended.
        assert_eq!(morphemes, vec!["crispy", "-ly"]);
        assert_eq!(result.analysis.morphemes[0].kind, MorphemeKind::Stem);
        assert_eq!(result.analysis.morphemes[1].kind, MorphemeKind::Suffix);
        assert!(matches!(
            result.analysis.source,
            AnalysisSource::ProductiveMorphology
        ));
        // Phonology should end with L IY0.
        assert!(
            result
                .pronunciation
                .symbols
                .ends_with(&["L".to_string(), "IY0".to_string()]),
            "expected -ly phones at end: {:?}",
            result.pronunciation.symbols
        );
        // Stem's primary stress must be preserved in the derived form.
        let has_primary = result
            .pronunciation
            .stress_by_phone
            .iter()
            .any(|s| *s == Some(PhonologicalStress::Primary));
        assert!(
            has_primary,
            "primary stress from stem must survive in derived form"
        );
    }

    /// `"thriftily"` is not in CMUdict; the `-ly` rule must resolve stem
    /// `"thrifty"` via `IToY` repair.
    #[test]
    fn analyzes_thriftily_with_y_to_i_spelling_repair() {
        let result = analyze_word("thriftily");
        let morphemes: Vec<&str> = result
            .analysis
            .morphemes
            .iter()
            .map(|m| m.surface.as_str())
            .collect();
        assert_eq!(morphemes, vec!["thrifty", "-ly"]);
        // Rule name from the native rule must appear in diagnostics.
        assert!(
            result
                .analysis
                .rules
                .iter()
                .any(|r| r == "suffix_ly_attachment"),
            "ly rule id must appear in rules: {:?}",
            result.analysis.rules
        );
        // Provenance must be present (eSpeak-derived).
        assert!(
            !result.analysis.rule_provenance.is_empty(),
            "rule_provenance should be non-empty for ly-derived words"
        );
        assert_eq!(
            result.analysis.rule_provenance[0].source,
            "espeak-ng-derived"
        );
    }

    /// `"plopping"` is not in CMUdict; the `-ing` rule must find stem `"plop"`
    /// via the `RemoveDoubledConsonant` spelling repair.
    #[test]
    fn analyzes_plopping_with_doubled_consonant_repair() {
        let result = analyze_word("plopping");
        let morphemes: Vec<&str> = result
            .analysis
            .morphemes
            .iter()
            .map(|m| m.surface.as_str())
            .collect();
        assert_eq!(morphemes, vec!["plop", "-ing"]);
        // Phonology should end with IH0 NG.
        assert!(
            result
                .pronunciation
                .symbols
                .ends_with(&["IH0".to_string(), "NG".to_string()]),
            "expected -ing phones at end: {:?}",
            result.pronunciation.symbols
        );
        // Stem's primary stress must be preserved in the derived form.
        let has_primary = result
            .pronunciation
            .stress_by_phone
            .iter()
            .any(|s| *s == Some(PhonologicalStress::Primary));
        assert!(
            has_primary,
            "primary stress from stem must survive in -ing derived form"
        );
        // Provenance must carry eSpeak attribution.
        assert!(
            result
                .analysis
                .rule_provenance
                .iter()
                .any(|p| p.source == "espeak-ng-derived"),
            "rule_provenance should contain eSpeak provenance"
        );
    }

    /// Verify that `"-ing"` morpheme exposes the expected POS-hint tags.
    #[test]
    fn ing_morpheme_exposes_pos_hint_tags() {
        let result = analyze_word("plopping");
        let ing_morpheme = result
            .analysis
            .morphemes
            .iter()
            .find(|m| m.surface == "-ing")
            .expect("-ing morpheme must be present");
        assert!(
            ing_morpheme
                .features
                .tags
                .iter()
                .any(|t| t == "progressive" || t == "gerund"),
            "expected progressive/gerund tag on -ing morpheme"
        );
    }

    /// `"woodenly"` is not in CMUdict; the `-ly` rule resolves stem `"wooden"`
    /// directly (no spelling repair needed) and preserves its stress.
    #[test]
    fn analyzes_woodenly_with_stress_preservation() {
        let result = analyze_word("woodenly");
        let morphemes: Vec<&str> = result
            .analysis
            .morphemes
            .iter()
            .map(|m| m.surface.as_str())
            .collect();
        assert_eq!(morphemes, vec!["wooden", "-ly"]);

        // Stem "wooden" = W UH1 D AH0 N — primary stress on UH1.
        // After appending L IY0 the primary stress index must still refer to UH1.
        let stem_primary_idx = result
            .pronunciation
            .stress_by_phone
            .iter()
            .position(|s| *s == Some(PhonologicalStress::Primary))
            .expect("derived form must have a primary-stress phone");

        // The suffix phones (L IY0) are appended after the stem, so the primary
        // stress must fall within the stem portion.
        let stem_len = result.pronunciation.symbols.len().saturating_sub(2); // L IY0
        assert!(
            stem_primary_idx < stem_len,
            "primary stress should be in stem, not suffix: idx={stem_primary_idx}, stem_len={stem_len}"
        );
    }
}
