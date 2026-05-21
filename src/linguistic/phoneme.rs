use crate::linguistic::orthography::OrthographicWord;

pub use crate::linguistic::phonology::Phoneme;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct PhonemeSeq {
    pub phonemes: Vec<Phoneme>,
}

impl PhonemeSeq {
    pub fn new(phonemes: Vec<Phoneme>) -> Self {
        Self { phonemes }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct PhonemeText {
    pub units: Vec<PhonemeTextUnit>,
}

impl PhonemeText {
    pub fn new(units: Vec<PhonemeTextUnit>) -> Self {
        Self { units }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PhonemeTextUnit {
    Word {
        orthography: OrthographicWord,
        phonemes: PhonemeSeq,
    },
    WordBoundary,
    PhraseBoundary,
}
