use serde::{Deserialize, Serialize};

use crate::linguistic::arpabet::phoneme_from_arpabet;
use crate::linguistic::inventory::FeatureBundle;
use crate::linguistic::inventory::PhonemeSchema;
use crate::linguistic::orthography::OrthographicWord;
use crate::linguistic::phone::{PhoneString, Stress};
use crate::linguistic::realization::{Realization, RealizationMethod};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Phoneme {
    pub symbol: String,
    pub source_symbol: String,
    pub source: String,
    pub stress: Option<Stress>,
    pub features: FeatureBundle,
    pub default_phone_string: PhoneString,
    pub realization: Realization,
}

impl Phoneme {
    pub fn new(symbol: impl Into<String>) -> Self {
        let symbol = symbol.into();
        phoneme_from_arpabet(&symbol, "manual")
    }

    pub fn symbols_in_schema(&self, schema: PhonemeSchema) -> Vec<String> {
        match schema {
            PhonemeSchema::Arpabet => vec![self.source_symbol.clone()],
            PhonemeSchema::Cmudict => vec![self.source_symbol.clone()],
            PhonemeSchema::ArpabetSurface => {
                if self.is_realized_american_english_tap() {
                    vec!["DX".to_string()]
                } else {
                    vec![self.source_symbol.clone()]
                }
            }
            PhonemeSchema::Ipa => vec![self.realization.ipa.clone()],
        }
    }

    pub fn symbol_in_schema(&self, schema: PhonemeSchema) -> String {
        self.symbols_in_schema(schema).join(" ")
    }

    fn is_realized_american_english_tap(&self) -> bool {
        matches!(self.realization.method, RealizationMethod::AllophoneRule)
            && self.symbol == "T"
            && self.realization.ipa == "ɾ"
    }
}

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
