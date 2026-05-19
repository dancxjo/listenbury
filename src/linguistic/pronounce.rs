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

pub trait OrthographyToPhonemes {
    fn realize_word(
        &self,
        variety: &LinguisticVariety,
        word: &OrthographicWord,
    ) -> Result<PhonemeSeq, PhonologyError>;

    fn realize_text(
        &self,
        variety: &LinguisticVariety,
        text: &str,
    ) -> Result<PhonemeText, PhonologyError>;
}
