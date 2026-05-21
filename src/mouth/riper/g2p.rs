use std::sync::OnceLock;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::linguistic::cmudict;
use crate::linguistic::cmudict::{CmuPhoneme, Stress as CmuStress};
use crate::linguistic::orthography::OrthographicWord;
use crate::linguistic::phoneme::{Phoneme, PhonemeSeq, PhonemeText, PhonemeTextUnit};
use crate::linguistic::phonology::{
    RealizationConfig, RealizationMethod, phoneme_from_arpabet, realize_sequence,
};
use crate::linguistic::pronounce::{OrthographyToPhonemes, PhonologyError};
use crate::linguistic::sound_it_out::{SoundItOutPronouncer, SoundItOutRules};
use crate::linguistic::variety::{LinguisticVariety, Phonology};
use crate::mouth::riper::phoneme::{PiperPhoneme, PiperPhonemeSequence};
use crate::mouth::riper::prosody_audit::{
    PhraseBoundaryKind, ProminenceClass, Stress, WordProsodyInfo,
};
use crate::mouth::riper::sentence_analysis::{
    HeuristicSentenceAnalyzer, ReductionStatus, SentenceAnalysis, SentenceAnalyzer,
};
use crate::mouth::riper::text::{
    NormalizedToken, ProsodyBoundaryHint, ProsodyCommitment, PunctuationCommitmentState,
    TextNormalizationError, TextNormalizer,
};
use crate::text_stability::stable_prefix_len;

const WORD_SEPARATOR_SYMBOL: &str = " ";
const PHRASE_BREAK_SYMBOL: &str = "|";
const BREATH_BREAK_WORD_INTERVAL: usize = 9;
const BREATH_BREAK_MIN_WORDS_AFTER: usize = 4;

pub trait GraphemeToPhoneme {
    fn phonemize(&self, text: &str) -> Result<PiperPhonemeSequence>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhonemizedUnit {
    pub phonemes: PiperPhonemeSequence,
    pub length_hints: Vec<PhoneLengthHint>,
    pub word_targets: Vec<WordProsodyTarget>,
    pub phoneme_to_word: Vec<Option<usize>>,
    pub lexical_stress: Vec<LexicalStressTarget>,
    pub sentence_analysis: SentenceAnalysis,
    pub boundary: ProsodyBoundaryHint,
    pub boundary_kind: PhraseBoundaryKind,
    pub commitment: ProsodyCommitment,
    pub punctuation_commitment: PunctuationCommitmentState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhoneLengthClass {
    Short,
    Medium,
    Long,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhoneLengthHint {
    pub symbol: String,
    pub class: PhoneLengthClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpeechCandidateId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimingHintSource {
    HeuristicFromPhonemeClass,
    HeuristicFromWordLength,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhoneTimingHint {
    pub phoneme_index: usize,
    pub approximate_duration_ms: Option<u64>,
    pub source: TimingHintSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordTimingHint {
    pub word_index: usize,
    pub approximate_duration_ms: Option<u64>,
    pub source: TimingHintSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LexicalStressLevel {
    Primary,
    Secondary,
    Unstressed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LexicalStressSource {
    Cmudict,
    HeuristicFallback,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexicalStressTarget {
    pub word_index: usize,
    pub phoneme_index: usize,
    pub stress: LexicalStressLevel,
    pub source: LexicalStressSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordProsodyTarget {
    pub word_index: usize,
    pub text_range: std::ops::Range<usize>,
    pub phoneme_range: std::ops::Range<usize>,
    pub normalized_text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhonemeProsodyCandidate {
    pub id: SpeechCandidateId,
    pub text: String,
    pub phonemes: PiperPhonemeSequence,
    pub phone_hints: Vec<PhoneTimingHint>,
    pub word_hints: Vec<WordTimingHint>,
    pub word_targets: Vec<WordProsodyTarget>,
    pub phoneme_to_word: Vec<Option<usize>>,
    pub lexical_stress: Vec<LexicalStressTarget>,
    pub sentence_analysis: SentenceAnalysis,
    pub boundary_hint: ProsodyBoundaryHint,
    pub boundary_kind: PhraseBoundaryKind,
    pub commitment: ProsodyCommitment,
    pub punctuation_commitment: PunctuationCommitmentState,
    pub stable_prefix_len: usize,
}

impl PhonemeProsodyCandidate {
    pub fn mark_prepared(&mut self) {
        if !matches!(self.commitment, ProsodyCommitment::Cancelled) {
            self.commitment = ProsodyCommitment::Prepared;
        }
    }

    pub fn mark_playable(&mut self) {
        if !matches!(self.commitment, ProsodyCommitment::Cancelled) {
            self.commitment = ProsodyCommitment::Playable;
        }
    }

    pub fn mark_committed(&mut self) {
        if !matches!(self.commitment, ProsodyCommitment::Cancelled) {
            self.commitment = ProsodyCommitment::Committed;
            self.punctuation_commitment = PunctuationCommitmentState::FinalCadence;
            if matches!(self.boundary_hint, ProsodyBoundaryHint::PossibleSentenceEnd) {
                self.boundary_hint = ProsodyBoundaryHint::FinalSentenceEnd;
            }
        }
    }

    pub fn cancel(&mut self) {
        self.commitment = ProsodyCommitment::Cancelled;
    }

    pub fn word_prosody_info(&self) -> Vec<WordProsodyInfo> {
        self.word_targets
            .iter()
            .map(|target| {
                let lexical_stress = self
                    .lexical_stress
                    .iter()
                    .filter(|stress| {
                        stress.phoneme_index >= target.phoneme_range.start
                            && stress.phoneme_index < target.phoneme_range.end
                    })
                    .map(|stress| map_lexical_stress(stress.stress))
                    .collect::<Vec<_>>();
                WordProsodyInfo {
                    word_index: target.word_index,
                    text_range: target.text_range.clone(),
                    phoneme_range: target.phoneme_range.clone(),
                    lexical_stress,
                    prominence_class: if is_default_function_word(&target.normalized_text) {
                        ProminenceClass::Weak
                    } else {
                        ProminenceClass::Content
                    },
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PhonemeProsodyCandidateEvent {
    CandidateStarted {
        id: SpeechCandidateId,
    },
    CandidateUpdated {
        candidate: PhonemeProsodyCandidate,
    },
    CandidateReplaced {
        old: SpeechCandidateId,
        new: SpeechCandidateId,
        stable_prefix_len: usize,
    },
    CandidateCancelled {
        id: SpeechCandidateId,
    },
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum G2pError {
    #[error(transparent)]
    Normalization(#[from] TextNormalizationError),
    #[error("unsupported word `{word}` for Riper simple English G2P")]
    UnsupportedWord { word: String },
    #[error("unsupported initial `{initial}` for Riper simple English G2P")]
    UnsupportedInitial { initial: char },
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SimpleEnglishG2p {
    normalizer: TextNormalizer,
}

impl SimpleEnglishG2p {
    pub fn phonemize_unit(&self, text: &str) -> std::result::Result<PhonemizedUnit, G2pError> {
        let normalized = self.normalizer.normalize(text)?;
        let sentence_analysis = HeuristicSentenceAnalyzer.analyze(text, &normalized);
        let mut symbols = Vec::new();
        let mut word_targets = Vec::new();
        let mut phoneme_to_word = Vec::new();
        let mut lexical_stress = Vec::new();
        let mut word_index = 0usize;

        let pronounceable_count = normalized
            .tokens
            .iter()
            .filter(|token| !matches!(token, NormalizedToken::PhraseBreak))
            .count();
        let mut emitted_pronounceable = 0usize;

        for (token_index, (token, token_span)) in normalized
            .tokens
            .iter()
            .zip(normalized.token_spans.iter())
            .enumerate()
        {
            match token {
                NormalizedToken::Word(word) => {
                    let word_realization = word_to_phones_with_metadata(word)
                        .ok_or_else(|| G2pError::UnsupportedWord { word: word.clone() })?;
                    let analyzed_token = sentence_analysis
                        .tokens
                        .iter()
                        .find(|analysis| analysis.word_index == Some(word_index));
                    let reduced_symbols = analyzed_token
                        .and_then(|analysis| analysis.reduction_diagnostic.as_ref())
                        .and_then(|diagnostic| {
                            if matches!(diagnostic.status, ReductionStatus::Applied) {
                                Some(
                                    diagnostic
                                        .realized
                                        .split_ascii_whitespace()
                                        .map(str::to_string)
                                        .collect::<Vec<_>>(),
                                )
                            } else {
                                None
                            }
                        });
                    let emitted_symbols =
                        reduced_symbols.unwrap_or_else(|| word_realization.symbols.clone());
                    let emitted_stress_by_phone = if analyzed_token.is_some_and(|analysis| {
                        analysis
                            .reduction_diagnostic
                            .as_ref()
                            .is_some_and(|diagnostic| {
                                matches!(diagnostic.status, ReductionStatus::Applied)
                            })
                    }) {
                        emitted_symbols
                            .iter()
                            .map(|symbol| stress_level_from_symbol(symbol))
                            .collect::<Vec<_>>()
                    } else {
                        word_realization.stress_by_phone.clone()
                    };
                    let start = symbols.len();
                    symbols.extend(emitted_symbols.iter().cloned());
                    let end = symbols.len();
                    phoneme_to_word.extend(std::iter::repeat(Some(word_index)).take(end - start));
                    word_targets.push(WordProsodyTarget {
                        word_index,
                        text_range: token_span.clone(),
                        phoneme_range: start..end,
                        normalized_text: word.clone(),
                    });
                    lexical_stress.extend(emitted_stress_by_phone.iter().enumerate().filter_map(
                        |(offset, stress)| {
                            stress.map(|stress| LexicalStressTarget {
                                word_index,
                                phoneme_index: start + offset,
                                stress,
                                source: word_realization.stress_source,
                            })
                        },
                    ));
                    emitted_pronounceable += 1;
                    word_index += 1;
                    if emitted_pronounceable < pronounceable_count {
                        symbols.push(inter_word_boundary_symbol(
                            emitted_pronounceable,
                            pronounceable_count,
                            normalized.tokens.get(token_index + 1),
                        ));
                        phoneme_to_word.push(None);
                    }
                }
                NormalizedToken::Initial(initial) => {
                    let initial_symbols = initial_to_phones(*initial)
                        .ok_or(G2pError::UnsupportedInitial { initial: *initial })?;
                    let start = symbols.len();
                    symbols.extend(initial_symbols.iter().copied().map(String::from));
                    let end = symbols.len();
                    phoneme_to_word.extend(std::iter::repeat(Some(word_index)).take(end - start));
                    word_targets.push(WordProsodyTarget {
                        word_index,
                        text_range: token_span.clone(),
                        phoneme_range: start..end,
                        normalized_text: initial.to_ascii_lowercase().to_string(),
                    });
                    emitted_pronounceable += 1;
                    word_index += 1;
                    if emitted_pronounceable < pronounceable_count {
                        symbols.push(inter_word_boundary_symbol(
                            emitted_pronounceable,
                            pronounceable_count,
                            normalized.tokens.get(token_index + 1),
                        ));
                        phoneme_to_word.push(None);
                    }
                }
                NormalizedToken::PhraseBreak => {
                    if !matches!(symbols.last(), Some(last) if last == PHRASE_BREAK_SYMBOL) {
                        symbols.push(PHRASE_BREAK_SYMBOL.to_string());
                        phoneme_to_word.push(None);
                    }
                }
            }
        }

        if matches!(
            normalized.boundary,
            ProsodyBoundaryHint::PossibleSentenceEnd
        ) && !matches!(symbols.last(), Some(last) if last == PHRASE_BREAK_SYMBOL)
        {
            symbols.push(PHRASE_BREAK_SYMBOL.to_string());
            phoneme_to_word.push(None);
        }

        let length_hints = symbols
            .iter()
            .map(|symbol| PhoneLengthHint {
                symbol: symbol.clone(),
                class: if symbol == WORD_SEPARATOR_SYMBOL || symbol == PHRASE_BREAK_SYMBOL {
                    PhoneLengthClass::Short
                } else if is_nucleus_symbol(symbol) {
                    PhoneLengthClass::Long
                } else {
                    PhoneLengthClass::Medium
                },
            })
            .collect();

        Ok(PhonemizedUnit {
            phonemes: PiperPhonemeSequence {
                phonemes: symbols.into_iter().map(PiperPhoneme).collect(),
            },
            length_hints,
            word_targets,
            phoneme_to_word,
            lexical_stress,
            sentence_analysis,
            boundary: normalized.boundary,
            boundary_kind: normalized.boundary_kind,
            commitment: normalized.commitment,
            punctuation_commitment: normalized.punctuation_commitment,
        })
    }
}

impl GraphemeToPhoneme for SimpleEnglishG2p {
    fn phonemize(&self, text: &str) -> Result<PiperPhonemeSequence> {
        Ok(self.phonemize_unit(text)?.phonemes)
    }
}

impl OrthographyToPhonemes for SimpleEnglishG2p {
    fn realize_word(
        &self,
        _variety: &LinguisticVariety,
        word: &OrthographicWord,
    ) -> Result<PhonemeSeq, PhonologyError> {
        let phones = word_to_phones(&word.text).ok_or_else(|| PhonologyError::UnsupportedWord {
            word: word.text.clone(),
        })?;
        Ok(PhonemeSeq::new(
            phones.into_iter().map(Phoneme::new).collect(),
        ))
    }

    fn realize_text(
        &self,
        variety: &LinguisticVariety,
        text: &str,
    ) -> Result<PhonemeText, PhonologyError> {
        let normalized = self
            .normalizer
            .normalize(text)
            .map_err(|e| PhonologyError::Message {
                message: e.to_string(),
            })?;

        let mut units: Vec<PhonemeTextUnit> = Vec::new();
        let mut pending_word_boundary = false;

        for token in &normalized.tokens {
            match token {
                NormalizedToken::Word(word) => {
                    let ortho = OrthographicWord::new(word.as_str());
                    let seq = self.realize_word(variety, &ortho)?;
                    if pending_word_boundary {
                        units.push(PhonemeTextUnit::WordBoundary);
                    }
                    units.push(PhonemeTextUnit::Word {
                        orthography: ortho,
                        phonemes: seq,
                    });
                    pending_word_boundary = true;
                }
                NormalizedToken::Initial(initial) => {
                    let initial_phones =
                        initial_to_phones(*initial).ok_or_else(|| PhonologyError::Message {
                            message: format!("unsupported initial '{initial}'"),
                        })?;
                    let ortho = OrthographicWord::new(&initial.to_string());
                    let seq =
                        PhonemeSeq::new(initial_phones.iter().copied().map(Phoneme::new).collect());
                    if pending_word_boundary {
                        units.push(PhonemeTextUnit::WordBoundary);
                    }
                    units.push(PhonemeTextUnit::Word {
                        orthography: ortho,
                        phonemes: seq,
                    });
                    pending_word_boundary = true;
                }
                NormalizedToken::PhraseBreak => {
                    units.push(PhonemeTextUnit::PhraseBoundary);
                    pending_word_boundary = false;
                }
            }
        }

        if matches!(
            normalized.boundary,
            ProsodyBoundaryHint::PossibleSentenceEnd
        ) && !matches!(units.last(), Some(PhonemeTextUnit::PhraseBoundary))
        {
            units.push(PhonemeTextUnit::PhraseBoundary);
        }

        Ok(PhonemeText::new(units))
    }
}

pub trait PhonemeProsodyPhonemizer {
    fn phonemize_unit(&self, text: &str) -> std::result::Result<PhonemizedUnit, G2pError>;
}

impl PhonemeProsodyPhonemizer for SimpleEnglishG2p {
    fn phonemize_unit(&self, text: &str) -> std::result::Result<PhonemizedUnit, G2pError> {
        Self::phonemize_unit(self, text)
    }
}

#[derive(Debug)]
pub struct PhonemeProsodyCandidateTracker<P = SimpleEnglishG2p> {
    next_id: u64,
    active: Option<PhonemeProsodyCandidate>,
    phonemizer: P,
}

impl<P: PhonemeProsodyPhonemizer + Default> Default for PhonemeProsodyCandidateTracker<P> {
    fn default() -> Self {
        Self {
            next_id: 0,
            active: None,
            phonemizer: P::default(),
        }
    }
}

impl<P: PhonemeProsodyPhonemizer> PhonemeProsodyCandidateTracker<P> {
    pub fn new(phonemizer: P) -> Self {
        Self {
            next_id: 0,
            active: None,
            phonemizer,
        }
    }

    pub fn active(&self) -> Option<&PhonemeProsodyCandidate> {
        self.active.as_ref()
    }

    pub fn ingest_text(
        &mut self,
        text: impl Into<String>,
    ) -> std::result::Result<Vec<PhonemeProsodyCandidateEvent>, G2pError> {
        let text = text.into();
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let mut events = Vec::new();
        let (id, stable_prefix_len) = if let Some(active) = self.active.as_ref() {
            let stable = stable_prefix_len(&active.text, &text);
            if stable < active.text.len() {
                let old = active.id;
                events.push(PhonemeProsodyCandidateEvent::CandidateCancelled { id: old });
                let new = self.next_id();
                events.push(PhonemeProsodyCandidateEvent::CandidateReplaced {
                    old,
                    new,
                    stable_prefix_len: stable,
                });
                (new, stable)
            } else {
                (active.id, stable)
            }
        } else {
            let id = self.next_id();
            events.push(PhonemeProsodyCandidateEvent::CandidateStarted { id });
            (id, 0)
        };

        let phonemized = self.phonemizer.phonemize_unit(&text)?;
        let candidate = build_candidate(id, text, stable_prefix_len, phonemized);
        self.active = Some(candidate.clone());
        events.push(PhonemeProsodyCandidateEvent::CandidateUpdated { candidate });
        Ok(events)
    }
}

impl<P> PhonemeProsodyCandidateTracker<P> {
    fn next_id(&mut self) -> SpeechCandidateId {
        // IDs intentionally start at 1 to align with existing candidate trackers.
        self.next_id = self
            .next_id
            .checked_add(1)
            .expect("speech candidate id space exhausted");
        SpeechCandidateId(self.next_id)
    }
}

fn build_candidate(
    id: SpeechCandidateId,
    text: String,
    stable_prefix_len: usize,
    phonemized: PhonemizedUnit,
) -> PhonemeProsodyCandidate {
    let phone_hints = phonemized
        .length_hints
        .iter()
        .enumerate()
        .map(|(phoneme_index, hint)| PhoneTimingHint {
            phoneme_index,
            approximate_duration_ms: Some(match hint.class {
                PhoneLengthClass::Short => 70,
                PhoneLengthClass::Medium => 120,
                PhoneLengthClass::Long => 145,
            }),
            source: TimingHintSource::HeuristicFromPhonemeClass,
        })
        .collect();

    let word_hints = phonemized
        .word_targets
        .iter()
        .map(|target| WordTimingHint {
            word_index: target.word_index,
            approximate_duration_ms: Some(
                (target
                    .text_range
                    .end
                    .saturating_sub(target.text_range.start) as u64)
                    .saturating_mul(90),
            ),
            source: TimingHintSource::HeuristicFromWordLength,
        })
        .collect();

    PhonemeProsodyCandidate {
        id,
        text,
        phonemes: phonemized.phonemes,
        phone_hints,
        word_hints,
        word_targets: phonemized.word_targets,
        phoneme_to_word: phonemized.phoneme_to_word,
        lexical_stress: phonemized.lexical_stress,
        sentence_analysis: phonemized.sentence_analysis,
        boundary_hint: phonemized.boundary,
        boundary_kind: phonemized.boundary_kind,
        commitment: ProsodyCommitment::Provisional,
        punctuation_commitment: phonemized.punctuation_commitment,
        stable_prefix_len,
    }
}

#[derive(Debug, Clone)]
struct WordPhoneRealization {
    symbols: Vec<String>,
    stress_by_phone: Vec<Option<LexicalStressLevel>>,
    stress_source: LexicalStressSource,
}

fn word_to_phones_with_metadata(word: &str) -> Option<WordPhoneRealization> {
    if let Some(phones) = cmudict::bundled().lookup(word) {
        return Some(cmu_phones_to_riper_symbols(phones));
    }

    let ortho = OrthographicWord::new(word);
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
            WordPhoneRealization {
                symbols,
                stress_by_phone,
                stress_source: LexicalStressSource::HeuristicFallback,
            }
        })
        .filter(|realization| !realization.symbols.is_empty())
}

fn word_to_phones(word: &str) -> Option<Vec<String>> {
    word_to_phones_with_metadata(word).map(|realization| realization.symbols)
}

fn inter_word_boundary_symbol(
    emitted_pronounceable: usize,
    pronounceable_count: usize,
    next_token: Option<&NormalizedToken>,
) -> String {
    if should_insert_breath_break(emitted_pronounceable, pronounceable_count, next_token) {
        PHRASE_BREAK_SYMBOL
    } else {
        WORD_SEPARATOR_SYMBOL
    }
    .to_string()
}

fn should_insert_breath_break(
    emitted_pronounceable: usize,
    pronounceable_count: usize,
    next_token: Option<&NormalizedToken>,
) -> bool {
    emitted_pronounceable >= BREATH_BREAK_WORD_INTERVAL
        && emitted_pronounceable % BREATH_BREAK_WORD_INTERVAL == 0
        && pronounceable_count.saturating_sub(emitted_pronounceable) >= BREATH_BREAK_MIN_WORDS_AFTER
        && !matches!(next_token, Some(NormalizedToken::PhraseBreak))
}

fn is_nucleus_symbol(symbol: &str) -> bool {
    let base = symbol.trim_end_matches(|ch: char| ch.is_ascii_digit());
    matches!(
        base,
        "AA" | "AE"
            | "AH"
            | "AO"
            | "AW"
            | "AY"
            | "EH"
            | "ER"
            | "EY"
            | "IH"
            | "IY"
            | "OW"
            | "OY"
            | "UH"
            | "UW"
            | "a"
            | "e"
            | "i"
            | "o"
            | "u"
            | "æ"
            | "ɑ"
            | "ɔ"
            | "ə"
            | "ɛ"
            | "ɚ"
            | "ɪ"
            | "ʊ"
            | "ʌ"
    )
}

fn fallback_english_pronouncer() -> &'static SoundItOutPronouncer {
    static FALLBACK: OnceLock<SoundItOutPronouncer> = OnceLock::new();
    FALLBACK.get_or_init(|| SoundItOutPronouncer::new(SoundItOutRules::english_arpabet_fallback()))
}

fn cmu_phones_to_riper_symbols(phones: &[CmuPhoneme]) -> WordPhoneRealization {
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
    WordPhoneRealization {
        symbols,
        stress_by_phone,
        stress_source: LexicalStressSource::Cmudict,
    }
}

fn cmu_phone_source_symbol(phone: &CmuPhoneme) -> String {
    match phone.stress {
        Some(stress) => format!("{}{}", phone.base, cmu_stress_digit(stress)),
        None => phone.base.clone(),
    }
}

fn cmu_stress_digit(stress: crate::linguistic::cmudict::Stress) -> char {
    match stress {
        crate::linguistic::cmudict::Stress::Primary => '1',
        crate::linguistic::cmudict::Stress::Secondary => '2',
        crate::linguistic::cmudict::Stress::Unstressed => '0',
    }
}

fn cmu_stress_level(stress: Option<CmuStress>) -> Option<LexicalStressLevel> {
    match stress {
        Some(CmuStress::Primary) => Some(LexicalStressLevel::Primary),
        Some(CmuStress::Secondary) => Some(LexicalStressLevel::Secondary),
        Some(CmuStress::Unstressed) => Some(LexicalStressLevel::Unstressed),
        None => None,
    }
}

fn map_lexical_stress(stress: LexicalStressLevel) -> Stress {
    match stress {
        LexicalStressLevel::Primary => Stress::Primary,
        LexicalStressLevel::Secondary => Stress::Secondary,
        LexicalStressLevel::Unstressed => Stress::Reduced,
    }
}

fn stress_level_from_symbol(symbol: &str) -> Option<LexicalStressLevel> {
    match symbol.chars().next_back() {
        Some('1') => Some(LexicalStressLevel::Primary),
        Some('2') => Some(LexicalStressLevel::Secondary),
        Some('0') => Some(LexicalStressLevel::Unstressed),
        _ => None,
    }
}

fn is_default_function_word(word: &str) -> bool {
    matches!(
        word,
        "the"
            | "a"
            | "an"
            | "if"
            | "then"
            | "than"
            | "of"
            | "to"
            | "for"
            | "from"
            | "with"
            | "by"
            | "as"
            | "in"
            | "on"
            | "at"
            | "are"
            | "is"
            | "was"
            | "were"
            | "be"
            | "been"
            | "am"
            | "it"
            | "this"
            | "that"
            | "these"
            | "those"
            | "he"
            | "she"
            | "they"
            | "we"
            | "you"
            | "i"
            | "me"
            | "my"
            | "your"
            | "our"
            | "their"
            | "because"
            | "and"
            | "or"
            | "but"
    )
}

fn initial_to_phones(initial: char) -> Option<&'static [&'static str]> {
    match initial.to_ascii_lowercase() {
        'f' => Some(&["EH", "F"]),
        'j' => Some(&["JH", "EY"]),
        'r' => Some(&["AA", "R"]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symbols(sequence: &PiperPhonemeSequence) -> Vec<String> {
        sequence.phonemes.iter().map(|p| p.0.clone()).collect()
    }

    fn to_token_analyses(unit: &PhonemizedUnit) -> Vec<&crate::mouth::riper::TokenAnalysis> {
        unit.sentence_analysis
            .tokens
            .iter()
            .filter(|token| token.text == "to")
            .collect()
    }

    #[test]
    fn phonemizes_okay_sentence() {
        let g2p = SimpleEnglishG2p::default();
        let unit = g2p.phonemize_unit("Okay.").expect("phonemize");
        assert_eq!(symbols(&unit.phonemes), vec!["OW", "K", "EY", "|"]);
        assert_eq!(unit.boundary, ProsodyBoundaryHint::PossibleSentenceEnd);
        assert_eq!(unit.commitment, ProsodyCommitment::Provisional);
        assert_eq!(
            unit.punctuation_commitment,
            PunctuationCommitmentState::SafeToPlay
        );
    }

    #[test]
    fn phonemizes_i_see_sentence() {
        let g2p = SimpleEnglishG2p::default();
        let unit = g2p.phonemize_unit("I see.").expect("phonemize");
        assert_eq!(symbols(&unit.phonemes), vec!["AY", " ", "S", "IY", "|"]);
    }

    #[test]
    fn phonemizes_honorific_word() {
        let g2p = SimpleEnglishG2p::default();
        let unit = g2p.phonemize_unit("Dr. King").expect("phonemize");
        assert_eq!(
            symbols(&unit.phonemes),
            vec!["D", "AA", "K", "T", "ER", " ", "K", "IH", "NG"]
        );
        assert_eq!(unit.boundary, ProsodyBoundaryHint::None);
    }

    #[test]
    fn phonemizes_initials_and_words() {
        let g2p = SimpleEnglishG2p::default();
        let unit = g2p
            .phonemize_unit("F. Scott Fitzgerald")
            .expect("phonemize");
        assert_eq!(
            symbols(&unit.phonemes),
            vec![
                "EH", "F", " ", "S", "K", "AA", "T", " ", "F", "IH", "T", "S", "JH", "EH", "R",
                "AH0", "L", "D"
            ]
        );
    }

    #[test]
    fn phonemizes_xylophone() {
        let g2p = SimpleEnglishG2p::default();
        let unit = g2p.phonemize_unit("xylophone").expect("phonemize");
        assert_eq!(
            symbols(&unit.phonemes),
            vec!["Z", "AY", "L", "AH0", "F", "OW", "N"]
        );
    }

    #[test]
    fn vowel_nuclei_get_longer_timing_hints() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let events = tracker.ingest_text("see").expect("candidate");
        let candidate = match events.last().expect("events") {
            PhonemeProsodyCandidateEvent::CandidateUpdated { candidate } => candidate,
            other => panic!("unexpected event: {other:?}"),
        };

        let vowel_hint = candidate
            .phone_hints
            .iter()
            .find(|hint| candidate.phonemes.phonemes[hint.phoneme_index].0 == "IY")
            .expect("IY nucleus timing hint");
        let consonant_hint = candidate
            .phone_hints
            .iter()
            .find(|hint| candidate.phonemes.phonemes[hint.phoneme_index].0 == "S")
            .expect("S timing hint");

        assert!(
            vowel_hint.approximate_duration_ms > consonant_hint.approximate_duration_ms,
            "vowel nuclei should be held longer than neighboring consonants"
        );
        assert_eq!(vowel_hint.approximate_duration_ms, Some(145));
    }

    #[test]
    fn long_runs_insert_periodic_breath_breaks() {
        let g2p = SimpleEnglishG2p::default();
        let unit = g2p
            .phonemize_unit(
                "We represent the lollipop guild because the machine needs another minute before returning today.",
            )
            .expect("phonemize");
        let needs = unit
            .word_targets
            .iter()
            .find(|target| target.normalized_text == "needs")
            .expect("needs target");

        assert_eq!(
            unit.phonemes.phonemes[needs.phoneme_range.end].0, "|",
            "long unpunctuated runs should take a breath after the ninth word"
        );
        assert!(
            symbols(&unit.phonemes)
                .iter()
                .filter(|symbol| symbol.as_str() == "|")
                .count()
                >= 2,
            "breath break should coexist with the final sentence break"
        );
    }

    #[test]
    fn preserves_cmudict_ah_stress_for_improbable() {
        let g2p = SimpleEnglishG2p::default();
        let unit = g2p.phonemize_unit("improbable").expect("phonemize");
        assert_eq!(
            symbols(&unit.phonemes),
            vec!["IH", "M", "P", "R", "AA", "B", "AH0", "B", "AH0", "L"]
        );
    }

    #[test]
    fn applies_intervocalic_flap_for_riper() {
        let g2p = SimpleEnglishG2p::default();
        let unit = g2p.phonemize_unit("bottle").expect("phonemize");
        assert_eq!(symbols(&unit.phonemes), vec!["B", "AA", "ɾ", "AH0", "L"]);
    }

    #[test]
    fn applies_intervocalic_d_flap_for_already() {
        let g2p = SimpleEnglishG2p::default();
        let unit = g2p.phonemize_unit("already").expect("phonemize");
        assert_eq!(
            symbols(&unit.phonemes),
            vec!["AO", "L", "R", "EH", "ɾ", "IY"]
        );
    }

    #[test]
    fn does_not_flap_t_in_politics() {
        let g2p = SimpleEnglishG2p::default();
        let unit = g2p.phonemize_unit("politics").expect("phonemize");
        assert_eq!(
            symbols(&unit.phonemes),
            vec!["P", "AA", "L", "AH0", "T", "IH", "K", "S"]
        );
        assert!(unit.lexical_stress.iter().any(|stress| {
            stress.phoneme_index == 3 && stress.stress == LexicalStressLevel::Unstressed
        }));
        assert!(unit.lexical_stress.iter().any(|stress| {
            stress.phoneme_index == 5 && stress.stress == LexicalStressLevel::Secondary
        }));
        assert_eq!(unit.boundary_kind, PhraseBoundaryKind::None);
    }

    #[test]
    fn does_not_flap_without_stress_context() {
        let g2p = SimpleEnglishG2p::default();
        let unit = g2p.phonemize_unit("represent").expect("phonemize");
        assert_eq!(
            symbols(&unit.phonemes),
            vec!["R", "EH", "P", "R", "IH", "Z", "EH", "N", "T"]
        );
    }

    #[test]
    fn phonemizes_unknown_words_with_english_fallback() {
        let g2p = SimpleEnglishG2p::default();
        let unit = g2p.phonemize_unit("MBROLA developped").expect("phonemize");
        assert_eq!(
            symbols(&unit.phonemes),
            vec![
                "M", "B", "R", "OW", "L", "AH", " ", "D", "EH", "V", "EH", "L", "OW", "P", "P",
                "EH", "D"
            ]
        );
    }

    #[test]
    fn phonemizes_contractions_and_unknown_words_together() {
        let g2p = SimpleEnglishG2p::default();
        let unit = g2p
            .phonemize_unit("MBROLA was developped. It's ready.")
            .expect("phonemize");
        assert!(symbols(&unit.phonemes).iter().any(|symbol| symbol == "S"));
        assert_eq!(unit.boundary, ProsodyBoundaryHint::PossibleSentenceEnd);
    }

    #[test]
    fn phonemizes_piper_compare_sample_with_fallbacks() {
        let g2p = SimpleEnglishG2p::default();
        g2p.phonemize_unit(
            "Yo ho ho and a bottle of rum. I am a computer voice. MBROLA was developped by Thierry Dutoit. It's a speech synthesizer based on the concatenation of diphones. It takes a list of phonemes as input, together with prosodic information, and produces speech at the sampling frequency of the diphone database.",
        )
        .expect("sample text should phonemize");
    }

    #[test]
    fn phonemizes_lollipop_guild_sentence() {
        let g2p = SimpleEnglishG2p::default();
        let unit = g2p
            .phonemize_unit("We represent the lollipop guild.")
            .expect("phonemize");
        assert_eq!(
            symbols(&unit.phonemes),
            vec![
                "W", "IY", " ", "R", "EH", "P", "R", "IH", "Z", "EH", "N", "T", " ", "DH", "AH0",
                " ", "L", "AA", "L", "IY", "P", "AA", "P", " ", "G", "IH", "L", "D", "|"
            ]
        );
    }

    #[test]
    fn reduces_to_before_infinitive_verbs() {
        let g2p = SimpleEnglishG2p::default();
        for text in ["I want to go.", "We need to leave.", "Try to remember."] {
            let unit = g2p.phonemize_unit(text).expect("phonemize");
            let analyses = to_token_analyses(&unit);
            let analysis = analyses.first().expect("to analysis");
            assert_eq!(analysis.pos, crate::mouth::riper::PartOfSpeech::Particle);
            assert_eq!(
                analysis.prosodic_role,
                crate::mouth::riper::ProsodicRole::FunctionWeak
            );
            assert_eq!(
                analysis.reduction,
                crate::mouth::riper::ReductionClass::WeakFunctionWord
            );
            let diagnostic = analysis
                .reduction_diagnostic
                .as_ref()
                .expect("reduction diagnostic");
            assert_eq!(
                diagnostic.reason,
                "unstressed_function_word_before_verb".to_string()
            );
            assert_eq!(
                diagnostic.status,
                crate::mouth::riper::ReductionStatus::Applied
            );
            assert_eq!(diagnostic.rule, "weak_form_to_before_verb");
            assert_eq!(diagnostic.source, "espeak-ng-derived");
            assert_eq!(diagnostic.citation, "T UW1");
            assert_eq!(diagnostic.realized, "T AH0");
            assert!(
                symbols(&unit.phonemes)
                    .windows(2)
                    .any(|phones| phones[0] == "T" && phones[1] == "AH0"),
                "expected reduced /tə/ phones in `{text}`"
            );
        }
    }

    #[test]
    fn mixed_contrast_sentence_keeps_to_reduction_in_non_contrast_positions() {
        let g2p = SimpleEnglishG2p::default();
        let unit = g2p
            .phonemize_unit("I said to go, not to stay.")
            .expect("phonemize");
        let analyses = to_token_analyses(&unit);
        assert_eq!(analyses.len(), 2);
        for analysis in analyses {
            let diagnostic = analysis
                .reduction_diagnostic
                .as_ref()
                .expect("reduction diagnostic");
            assert_eq!(
                diagnostic.reason,
                "unstressed_function_word_before_verb".to_string()
            );
            assert_eq!(
                diagnostic.status,
                crate::mouth::riper::ReductionStatus::Applied
            );
        }
        let reduced_count = symbols(&unit.phonemes)
            .windows(2)
            .filter(|phones| phones[0] == "T" && phones[1] == "AH0")
            .count();
        assert!(reduced_count >= 2, "expected both `to` tokens to reduce");
    }

    #[test]
    fn keeps_contrastive_to_strong() {
        let g2p = SimpleEnglishG2p::default();
        let unit = g2p
            .phonemize_unit("I said TO, not FROM.")
            .expect("phonemize");
        let analysis = to_token_analyses(&unit)
            .into_iter()
            .next()
            .expect("to analysis");
        assert_eq!(analysis.pos, crate::mouth::riper::PartOfSpeech::Preposition);
        assert_eq!(
            analysis.prosodic_role,
            crate::mouth::riper::ProsodicRole::Contrastive
        );
        assert_eq!(
            analysis.reduction,
            crate::mouth::riper::ReductionClass::None
        );
        let diagnostic = analysis
            .reduction_diagnostic
            .as_ref()
            .expect("reduction diagnostic");
        assert_eq!(diagnostic.rule, "strong_to_contrastive_uppercase");
        assert_eq!(diagnostic.source, "espeak-ng-derived");
        assert_eq!(diagnostic.reason, "contrastive_emphasis".to_string());
        assert_eq!(
            diagnostic.status,
            crate::mouth::riper::ReductionStatus::Blocked
        );
        assert!(
            symbols(&unit.phonemes)
                .windows(2)
                .any(|phones| phones[0] == "T" && phones[1].starts_with("UW")),
            "contrastive TO should remain unreduced"
        );
    }

    #[test]
    fn keeps_citation_or_prepositional_to_strong() {
        let g2p = SimpleEnglishG2p::default();
        let to_be = g2p
            .phonemize_unit("To be, or not to be.")
            .expect("phonemize");
        let to_be_analyses = to_token_analyses(&to_be);
        assert_eq!(to_be_analyses.len(), 2);
        assert_eq!(
            to_be_analyses[0]
                .reduction_diagnostic
                .as_ref()
                .expect("diagnostic")
                .reason,
            "citation_form_phrase_initial".to_string()
        );
        assert_eq!(
            to_be_analyses[1]
                .reduction_diagnostic
                .as_ref()
                .expect("diagnostic")
                .reason,
            "quotation_or_citation_form".to_string()
        );
        assert!(
            !symbols(&to_be.phonemes)
                .windows(2)
                .any(|phones| phones[0] == "T" && phones[1] == "AH0"),
            "citation-form example should avoid weak reduction"
        );

        let addressed = g2p
            .phonemize_unit("This is addressed to you.")
            .expect("phonemize");
        let addressed_to = to_token_analyses(&addressed)
            .into_iter()
            .next()
            .expect("to analysis");
        assert_eq!(
            addressed_to.pos,
            crate::mouth::riper::PartOfSpeech::Preposition
        );
        assert_eq!(
            addressed_to.prosodic_role,
            crate::mouth::riper::ProsodicRole::FunctionStrong
        );
        let diagnostic = addressed_to
            .reduction_diagnostic
            .as_ref()
            .expect("reduction diagnostic");
        assert_eq!(diagnostic.reason, "prepositional_to".to_string());
        assert_eq!(
            diagnostic.status,
            crate::mouth::riper::ReductionStatus::Blocked
        );
    }

    #[test]
    fn keeps_phrase_final_to_provisional_for_incremental_revision() {
        let g2p = SimpleEnglishG2p::default();
        let provisional = g2p.phonemize_unit("I want to").expect("phonemize");
        let analysis = to_token_analyses(&provisional)
            .into_iter()
            .next()
            .expect("to analysis");
        let diagnostic = analysis
            .reduction_diagnostic
            .as_ref()
            .expect("reduction diagnostic");
        assert_eq!(diagnostic.reason, "phrase_final_uncertainty".to_string());
        assert_eq!(
            diagnostic.status,
            crate::mouth::riper::ReductionStatus::Provisional
        );

        let confirmed = g2p.phonemize_unit("I want to go").expect("phonemize");
        let confirmed_diag = to_token_analyses(&confirmed)[0]
            .reduction_diagnostic
            .as_ref()
            .expect("reduction diagnostic");
        assert_eq!(
            confirmed_diag.status,
            crate::mouth::riper::ReductionStatus::Applied
        );
    }

    #[test]
    fn candidate_complete_sentence_stays_provisional_until_committed() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let events = tracker.ingest_text("Okay.").expect("candidate");
        let candidate = match events.last().expect("events") {
            PhonemeProsodyCandidateEvent::CandidateUpdated { candidate } => candidate.clone(),
            other => panic!("unexpected event: {other:?}"),
        };

        assert_eq!(candidate.commitment, ProsodyCommitment::Provisional);
        assert_eq!(
            candidate.boundary_hint,
            ProsodyBoundaryHint::PossibleSentenceEnd
        );
        assert_eq!(candidate.stable_prefix_len, 0);
        assert!(!candidate.phone_hints.is_empty());
        assert_eq!(candidate.word_hints.len(), 1);
        assert_eq!(candidate.word_targets.len(), 1);
        assert_eq!(candidate.word_targets[0].text_range, 0..5);
        assert_eq!(candidate.boundary_kind, PhraseBoundaryKind::FinalFalling);
        assert!(!candidate.lexical_stress.is_empty());
        assert_eq!(candidate.phoneme_to_word[0], Some(0));
        assert_eq!(
            candidate.phoneme_to_word.last(),
            Some(&None),
            "sentence boundary marker should not map to a word"
        );

        let mut committed = candidate.clone();
        committed.mark_prepared();
        committed.mark_playable();
        committed.mark_committed();
        assert_eq!(committed.commitment, ProsodyCommitment::Committed);
        assert_eq!(
            committed.boundary_hint,
            ProsodyBoundaryHint::FinalSentenceEnd
        );
        assert_eq!(
            committed.punctuation_commitment,
            PunctuationCommitmentState::FinalCadence
        );
    }

    #[test]
    fn exposes_word_prosody_mapping_metadata() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let events = tracker
            .ingest_text(
                "University politics are vicious precisely because the stakes are so small.",
            )
            .expect("candidate");
        let candidate = match events.last().expect("events") {
            PhonemeProsodyCandidateEvent::CandidateUpdated { candidate } => candidate,
            other => panic!("unexpected event: {other:?}"),
        };
        let infos = candidate.word_prosody_info();
        let because = infos
            .iter()
            .find(|info| info.word_index == 5)
            .expect("because info");
        assert_eq!(because.prominence_class, ProminenceClass::Weak);
        assert!(
            infos
                .iter()
                .any(|info| info.lexical_stress.contains(&Stress::Primary)),
            "at least one word should carry lexical stress metadata"
        );
    }

    #[test]
    fn candidate_delays_commitment_for_f_scott_fitzgerald() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let first = tracker.ingest_text("F.").expect("candidate");
        let first_candidate = match first.last().expect("events") {
            PhonemeProsodyCandidateEvent::CandidateUpdated { candidate } => candidate.clone(),
            other => panic!("unexpected event: {other:?}"),
        };
        assert_eq!(first_candidate.boundary_hint, ProsodyBoundaryHint::None);
        assert_eq!(first_candidate.commitment, ProsodyCommitment::Provisional);
        assert_eq!(
            first_candidate.punctuation_commitment,
            PunctuationCommitmentState::SafeToPrepare
        );

        let second = tracker
            .ingest_text("F. Scott Fitzgerald.")
            .expect("candidate");
        let second_candidate = match second.last().expect("events") {
            PhonemeProsodyCandidateEvent::CandidateUpdated { candidate } => candidate.clone(),
            other => panic!("unexpected event: {other:?}"),
        };
        assert_eq!(second_candidate.id, first_candidate.id);
        assert_eq!(
            second_candidate.boundary_hint,
            ProsodyBoundaryHint::PossibleSentenceEnd
        );
        assert_eq!(
            second_candidate.punctuation_commitment,
            PunctuationCommitmentState::SafeToPlay
        );
    }

    #[test]
    fn candidate_delays_commitment_for_dr_king() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let first = tracker.ingest_text("Dr.").expect("candidate");
        let first_candidate = match first.last().expect("events") {
            PhonemeProsodyCandidateEvent::CandidateUpdated { candidate } => candidate.clone(),
            other => panic!("unexpected event: {other:?}"),
        };
        assert_eq!(first_candidate.boundary_hint, ProsodyBoundaryHint::None);

        let second = tracker.ingest_text("Dr. King.").expect("candidate");
        let second_candidate = match second.last().expect("events") {
            PhonemeProsodyCandidateEvent::CandidateUpdated { candidate } => candidate.clone(),
            other => panic!("unexpected event: {other:?}"),
        };
        assert_eq!(second_candidate.id, first_candidate.id);
        assert_eq!(
            second_candidate.boundary_hint,
            ProsodyBoundaryHint::PossibleSentenceEnd
        );
    }

    #[test]
    fn candidate_extension_preserves_stable_prefix() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let first = tracker.ingest_text("I see.").expect("candidate");
        let first_candidate = match first.last().expect("events") {
            PhonemeProsodyCandidateEvent::CandidateUpdated { candidate } => candidate.clone(),
            other => panic!("unexpected event: {other:?}"),
        };

        let second = tracker.ingest_text("I see. Okay.").expect("candidate");
        let second_candidate = match second.last().expect("events") {
            PhonemeProsodyCandidateEvent::CandidateUpdated { candidate } => candidate.clone(),
            other => panic!("unexpected event: {other:?}"),
        };

        assert_eq!(second_candidate.id, first_candidate.id);
        assert_eq!(second_candidate.stable_prefix_len, "I see.".len());
    }

    #[test]
    fn candidate_tracks_word_and_phoneme_index_mappings() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let events = tracker.ingest_text(" I see, okay ").expect("candidate");
        let candidate = match events.last().expect("events") {
            PhonemeProsodyCandidateEvent::CandidateUpdated { candidate } => candidate.clone(),
            other => panic!("unexpected event: {other:?}"),
        };

        assert_eq!(candidate.word_targets.len(), 3);
        assert_eq!(candidate.word_targets[0].text_range, 1..2);
        assert_eq!(candidate.word_targets[1].text_range, 3..6);
        assert_eq!(candidate.word_targets[2].text_range, 8..12);
        for target in &candidate.word_targets {
            for idx in target.phoneme_range.clone() {
                assert_eq!(candidate.phoneme_to_word[idx], Some(target.word_index));
            }
        }
    }

    #[test]
    fn candidate_preserves_cmudict_lexical_stress_metadata() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let events = tracker.ingest_text("xylophone").expect("candidate");
        let candidate = match events.last().expect("events") {
            PhonemeProsodyCandidateEvent::CandidateUpdated { candidate } => candidate.clone(),
            other => panic!("unexpected event: {other:?}"),
        };

        assert!(
            candidate
                .lexical_stress
                .iter()
                .any(|stress| stress.stress == LexicalStressLevel::Primary),
            "expected at least one primary lexical stress target"
        );
        assert!(
            candidate
                .lexical_stress
                .iter()
                .any(|stress| stress.stress == LexicalStressLevel::Secondary),
            "expected at least one secondary lexical stress target"
        );
    }

    #[test]
    fn candidate_head_change_cancels_and_replaces() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let first = tracker.ingest_text("Okay.").expect("candidate");
        let first_id = match first.last().expect("events") {
            PhonemeProsodyCandidateEvent::CandidateUpdated { candidate } => candidate.id,
            other => panic!("unexpected event: {other:?}"),
        };

        let second = tracker.ingest_text("I see.").expect("candidate");
        assert!(matches!(
            second.first(),
            Some(PhonemeProsodyCandidateEvent::CandidateCancelled { id }) if *id == first_id
        ));
        let replacement = second
            .iter()
            .find_map(|event| match event {
                PhonemeProsodyCandidateEvent::CandidateReplaced {
                    old,
                    new,
                    stable_prefix_len,
                } => Some((*old, *new, *stable_prefix_len)),
                _ => None,
            })
            .expect("replacement event");
        assert_eq!(replacement.0, first_id);
        assert_eq!(replacement.2, 0);
    }

    mod realize_text {
        use super::*;
        use crate::linguistic::phoneme::PhonemeTextUnit;
        use crate::linguistic::variety::{LinguisticVariety, Phonology, VarietyTag};
        use crate::mouth::riper::encoder::PiperEncoder;

        fn en_us() -> LinguisticVariety {
            LinguisticVariety::tagged(
                VarietyTag::new("en_US"),
                "English (US)",
                Phonology::new("General American"),
            )
        }

        fn piper_seq(text: &str) -> PiperPhonemeSequence {
            let g2p = SimpleEnglishG2p::default();
            let phoneme_text = g2p.realize_text(&en_us(), text).expect("realize_text");
            PiperEncoder.encode(&phoneme_text)
        }

        fn sym(seq: &PiperPhonemeSequence) -> Vec<&str> {
            seq.phonemes.iter().map(|p| p.0.as_str()).collect()
        }

        #[test]
        fn realize_text_okay_matches_phonemize_unit() {
            let seq = piper_seq("Okay.");
            assert_eq!(sym(&seq), vec!["OW", "K", "EY", "|"]);
        }

        #[test]
        fn realize_text_i_see_matches_phonemize_unit() {
            let seq = piper_seq("I see.");
            assert_eq!(sym(&seq), vec!["AY", " ", "S", "IY", "|"]);
        }

        #[test]
        fn realize_text_honorific_matches_phonemize_unit() {
            let seq = piper_seq("Dr. King");
            assert_eq!(
                sym(&seq),
                vec!["D", "AA", "K", "T", "ER", " ", "K", "IH", "NG"]
            );
        }

        #[test]
        fn realize_text_initials_and_words_matches_phonemize_unit() {
            let seq = piper_seq("F. Scott Fitzgerald");
            assert_eq!(
                sym(&seq),
                vec![
                    "EH", "F", " ", "S", "K", "AA", "T", " ", "F", "IH", "T", "S", "JH", "EH", "R",
                    "AH0", "L", "D"
                ]
            );
        }

        #[test]
        fn realize_text_phrase_boundary_unit_present() {
            let g2p = SimpleEnglishG2p::default();
            let phoneme_text = g2p.realize_text(&en_us(), "Okay.").expect("realize_text");
            assert!(
                phoneme_text.units.last() == Some(&PhonemeTextUnit::PhraseBoundary),
                "trailing PhraseBoundary expected for sentence-ending text"
            );
        }

        #[test]
        fn realize_text_no_trailing_boundary_for_incomplete_phrase() {
            let g2p = SimpleEnglishG2p::default();
            let phoneme_text = g2p
                .realize_text(&en_us(), "Dr. King")
                .expect("realize_text");
            assert!(
                !matches!(
                    phoneme_text.units.last(),
                    Some(PhonemeTextUnit::PhraseBoundary)
                ),
                "no trailing PhraseBoundary expected for non-sentence-ending text"
            );
        }
    }
}
