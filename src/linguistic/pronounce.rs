use thiserror::Error;

use crate::linguistic::{
    orthography::OrthographicWord,
    phoneme::{PhonemeSeq, PhonemeText},
    variety::LinguisticVariety,
};

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PhonologyError {
    #[error("unsupported orthographic word `{word}`")]
    UnsupportedWord { word: String },
    #[error("{message}")]
    Message { message: String },
}

/// General-purpose interface for realizing orthography into phonemic units.
pub trait OrthographyToPhonemes {
    /// Realize one orthographic word into a phoneme sequence.
    fn realize_word(
        &self,
        variety: &LinguisticVariety,
        word: &OrthographicWord,
    ) -> Result<PhonemeSeq, PhonologyError>;

    /// Realize free-form text into phoneme text units with explicit boundaries.
    fn realize_text(
        &self,
        variety: &LinguisticVariety,
        text: &str,
    ) -> Result<PhonemeText, PhonologyError>;
}
