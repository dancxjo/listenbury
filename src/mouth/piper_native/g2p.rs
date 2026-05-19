use anyhow::Result;
use thiserror::Error;

use crate::linguistic::cmudict;
use crate::linguistic::orthography::OrthographicWord;
use crate::linguistic::phoneme::{Phoneme, PhonemeSeq, PhonemeText, PhonemeTextUnit};
use crate::linguistic::pronounce::{OrthographyToPhonemes, PhonologyError};
use crate::linguistic::variety::LinguisticVariety;
use crate::mouth::piper_native::phoneme::{PiperPhoneme, PiperPhonemeSequence};
use crate::mouth::piper_native::text::{
    NormalizedToken, ProsodyBoundaryHint, ProsodyCommitment, PunctuationCommitmentState,
    TextNormalizationError, TextNormalizer,
};
use crate::text_stability::stable_prefix_len;

const WORD_SEPARATOR_SYMBOL: &str = " ";
const PHRASE_BREAK_SYMBOL: &str = "|";

pub trait GraphemeToPhoneme {
    fn phonemize(&self, text: &str) -> Result<PiperPhonemeSequence>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhonemizedUnit {
    pub phonemes: PiperPhonemeSequence,
    pub length_hints: Vec<PhoneLengthHint>,
    pub boundary: ProsodyBoundaryHint,
    pub commitment: ProsodyCommitment,
    pub punctuation_commitment: PunctuationCommitmentState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhoneLengthClass {
    Short,
    Medium,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhoneLengthHint {
    pub symbol: String,
    pub class: PhoneLengthClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhonemeProsodyCandidate {
    pub id: SpeechCandidateId,
    pub text: String,
    pub phonemes: PiperPhonemeSequence,
    pub phone_hints: Vec<PhoneTimingHint>,
    pub word_hints: Vec<WordTimingHint>,
    pub boundary_hint: ProsodyBoundaryHint,
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    #[error("unsupported word `{word}` for native Piper simple English G2P")]
    UnsupportedWord { word: String },
    #[error("unsupported initial `{initial}` for native Piper simple English G2P")]
    UnsupportedInitial { initial: char },
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SimpleEnglishG2p {
    normalizer: TextNormalizer,
}

impl SimpleEnglishG2p {
    pub fn phonemize_unit(&self, text: &str) -> std::result::Result<PhonemizedUnit, G2pError> {
        let normalized = self.normalizer.normalize(text)?;
        let mut symbols = Vec::new();

        let pronounceable_count = normalized
            .tokens
            .iter()
            .filter(|token| !matches!(token, NormalizedToken::PhraseBreak))
            .count();
        let mut emitted_pronounceable = 0usize;

        for token in &normalized.tokens {
            match token {
                NormalizedToken::Word(word) => {
                    let word_symbols = word_to_phones(word)
                        .ok_or_else(|| G2pError::UnsupportedWord { word: word.clone() })?;
                    symbols.extend(word_symbols);
                    emitted_pronounceable += 1;
                    if emitted_pronounceable < pronounceable_count {
                        symbols.push(WORD_SEPARATOR_SYMBOL.to_string());
                    }
                }
                NormalizedToken::Initial(initial) => {
                    let initial_symbols = initial_to_phones(*initial)
                        .ok_or(G2pError::UnsupportedInitial { initial: *initial })?;
                    symbols.extend(initial_symbols.iter().copied().map(String::from));
                    emitted_pronounceable += 1;
                    if emitted_pronounceable < pronounceable_count {
                        symbols.push(WORD_SEPARATOR_SYMBOL.to_string());
                    }
                }
                NormalizedToken::PhraseBreak => {
                    if !matches!(symbols.last(), Some(last) if last == PHRASE_BREAK_SYMBOL) {
                        symbols.push(PHRASE_BREAK_SYMBOL.to_string());
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
        }

        let length_hints = symbols
            .iter()
            .map(|symbol| PhoneLengthHint {
                symbol: symbol.clone(),
                class: if symbol == WORD_SEPARATOR_SYMBOL || symbol == PHRASE_BREAK_SYMBOL {
                    PhoneLengthClass::Short
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
            boundary: normalized.boundary,
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
        let phones = word_to_phones(&word.text)
            .ok_or_else(|| PhonologyError::UnsupportedWord { word: word.text.clone() })?;
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
            .map_err(|e| PhonologyError::Message { message: e.to_string() })?;

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
                    let initial_phones = initial_to_phones(*initial).ok_or_else(|| {
                        PhonologyError::Message {
                            message: format!("unsupported initial '{initial}'"),
                        }
                    })?;
                    let ortho = OrthographicWord::new(&initial.to_string());
                    let seq = PhonemeSeq::new(
                        initial_phones
                            .iter()
                            .copied()
                            .map(Phoneme::new)
                            .collect(),
                    );
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

        if matches!(normalized.boundary, ProsodyBoundaryHint::PossibleSentenceEnd)
            && !matches!(units.last(), Some(PhonemeTextUnit::PhraseBoundary))
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
            }),
            source: TimingHintSource::HeuristicFromPhonemeClass,
        })
        .collect();

    let word_hints = text
        .split_ascii_whitespace()
        .enumerate()
        .map(|(word_index, word)| WordTimingHint {
            word_index,
            approximate_duration_ms: Some((word.len() as u64).saturating_mul(90)),
            source: TimingHintSource::HeuristicFromWordLength,
        })
        .collect();

    PhonemeProsodyCandidate {
        id,
        text,
        phonemes: phonemized.phonemes,
        phone_hints,
        word_hints,
        boundary_hint: phonemized.boundary,
        commitment: ProsodyCommitment::Provisional,
        punctuation_commitment: phonemized.punctuation_commitment,
        stable_prefix_len,
    }
}

fn word_to_phones(word: &str) -> Option<Vec<String>> {
    let phones = cmudict::bundled().lookup(word)?;
    Some(phones.iter().map(|p| p.base.clone()).collect())
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
                "EH", "F", " ", "S", "K", "AA", "T", " ", "F", "IH", "TS", "JH", "EH", "R", "AH",
                "L", "D"
            ]
        );
    }

    #[test]
    fn phonemizes_xylophone() {
        let g2p = SimpleEnglishG2p::default();
        let unit = g2p.phonemize_unit("xylophone").expect("phonemize");
        assert_eq!(
            symbols(&unit.phonemes),
            vec!["Z", "AY", "L", "AH", "F", "OW", "N"]
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
        use crate::mouth::piper_native::encoder::PiperEncoder;

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
                    "EH", "F", " ", "S", "K", "AA", "T", " ", "F", "IH", "TS", "JH", "EH", "R",
                    "AH", "L", "D"
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
            let phoneme_text = g2p.realize_text(&en_us(), "Dr. King").expect("realize_text");
            assert!(
                !matches!(phoneme_text.units.last(), Some(PhonemeTextUnit::PhraseBoundary)),
                "no trailing PhraseBoundary expected for non-sentence-ending text"
            );
        }
    }
}
