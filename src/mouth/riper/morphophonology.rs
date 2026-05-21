use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::linguistic::cmudict::{self, CmuPhoneme, Stress as CmuStress};
use crate::linguistic::orthography::OrthographicWord;
use crate::linguistic::phonology::{
    RealizationConfig, RealizationMethod, phoneme_from_arpabet, realize_sequence,
};
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordPronunciation {
    pub symbols: Vec<String>,
    pub stress_by_phone: Vec<Option<PhonologicalStress>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MorphophonologyResult {
    pub analysis: MorphologicalAnalysis,
    pub pronunciation: WordPronunciation,
}

pub fn analyze_word(surface: &str) -> MorphophonologyResult {
    if let Some(exact) = exact_lexical(surface) {
        return exact;
    }
    if let Some(known) = known_derived(surface) {
        return known;
    }
    if let Some(productive) = productive_morphology(surface) {
        return productive;
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
        },
        pronunciation: stem,
    })
}

fn known_derived(surface: &str) -> Option<MorphophonologyResult> {
    if !surface.eq_ignore_ascii_case("unpunctuated") {
        return None;
    }
    let stem = lexicon_pronunciation("punctuate")?;
    let prefix_symbols = vec!["AH0".to_string(), "N".to_string()];
    let prefix_stress = vec![Some(PhonologicalStress::Unstressed), None];
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

    if let Some(mixed) = analyze_un_plus_stem_plus_ed(surface) {
        return Some(mixed);
    }
    if let Some(un_prefixed) = analyze_un_prefix(surface) {
        return Some(un_prefixed);
    }
    if let Some(ed_word) = analyze_ed_suffix(surface) {
        return Some(ed_word);
    }
    None
}

fn analyze_un_plus_stem_plus_ed(surface: &str) -> Option<MorphophonologyResult> {
    if !surface.to_ascii_lowercase().starts_with("un")
        || !surface.to_ascii_lowercase().ends_with("ed")
    {
        return None;
    }
    let inner = &surface[2..surface.len().saturating_sub(2)];
    if inner.is_empty() {
        return None;
    }

    let candidates = [inner.to_string(), format!("{inner}e")];
    let (stem_text, stem) = candidates
        .iter()
        .find_map(|candidate| lookup_any(candidate).map(|p| (candidate.clone(), p)))?;
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
                        tags: vec!["past_tense_or_participle".to_string()],
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
        },
        pronunciation: WordPronunciation {
            symbols: realized,
            stress_by_phone: stress,
        },
    })
}

fn analyze_un_prefix(surface: &str) -> Option<MorphophonologyResult> {
    if !surface.to_ascii_lowercase().starts_with("un") {
        return None;
    }
    let stem_text = &surface[2..];
    if stem_text.is_empty() {
        return None;
    }

    let stem = lookup_any(stem_text)?;
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
        },
        pronunciation: WordPronunciation {
            symbols: realized,
            stress_by_phone: stress,
        },
    })
}

fn analyze_ed_suffix(surface: &str) -> Option<MorphophonologyResult> {
    if !surface.to_ascii_lowercase().ends_with("ed") || surface.len() <= 2 {
        return None;
    }
    let stem_text = &surface[..surface.len() - 2];
    let stem = lookup_any(stem_text)?;
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
                    surface: stem_text.to_string(),
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
                        tags: vec!["past_tense_or_participle".to_string()],
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
        },
        pronunciation: WordPronunciation {
            symbols: realized,
            stress_by_phone: stress,
        },
    })
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
        },
        pronunciation: WordPronunciation {
            symbols,
            stress_by_phone: stress,
        },
    }
}

fn lookup_any(surface: &str) -> Option<WordPronunciation> {
    lexicon_pronunciation(surface).or_else(|| fallback_pronunciation(surface))
}

fn lexicon_pronunciation(surface: &str) -> Option<WordPronunciation> {
    let phones = cmudict::bundled().lookup(surface)?;
    Some(cmu_phones_to_symbols(phones))
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
    let phonology_sequence = phones
        .iter()
        .map(|phone| phoneme_from_arpabet(&cmu_phone_source_symbol(phone), "cmudict"))
        .collect::<Vec<_>>();
    let realized = realize_sequence(
        &phonology_sequence,
        &RealizationConfig {
            enable_allophone_rules: true,
            ..RealizationConfig::default()
        },
    );

    let symbols = phones
        .iter()
        .zip(realized.iter())
        .map(|(source, realized)| {
            if matches!(
                realized.realization.method,
                RealizationMethod::AllophoneRule
            ) && realized.realization.ipa == "ɾ"
            {
                "ɾ".to_string()
            } else if source.base == "AH" {
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
    "custom_rule_engine_now__treebender_spike_path".to_string()
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
    }

    #[test]
    fn realizes_ed_allomorphy() {
        assert_eq!(symbols_of("walked"), vec!["W", "AO", "K", "T"]);
        assert_eq!(symbols_of("played"), vec!["P", "L", "EY", "D"]);
        assert_eq!(symbols_of("wanted"), vec!["W", "AA", "N", "T", "IH0", "D"]);
        assert_eq!(symbols_of("needed"), vec!["N", "IY", "D", "IH0", "D"]);
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
}
