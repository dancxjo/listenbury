use crate::linguistic::orthography::OrthographicWord;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Phoneme {
    pub symbol: String,
}

impl Phoneme {
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PhonemeSeq {
    pub phonemes: Vec<Phoneme>,
}

impl PhonemeSeq {
    pub fn new(phonemes: Vec<Phoneme>) -> Self {
        Self { phonemes }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PhonemeText {
    pub units: Vec<PhonemeTextUnit>,
}

impl PhonemeText {
    pub fn new(units: Vec<PhonemeTextUnit>) -> Self {
        Self { units }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhonemeTextUnit {
    Word {
        orthography: OrthographicWord,
        phonemes: PhonemeSeq,
    },
    WordBoundary,
    PhraseBoundary,
}
