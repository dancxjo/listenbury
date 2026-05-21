use super::inventory::general_american_english;
use super::phonology::{PhoneComparisonMode, PhoneEqualityOptions, PhonemicInventory};

pub use super::phonology::VarietyId;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VarietyTag(pub String);

impl VarietyTag {
    pub fn new(tag: impl Into<String>) -> Self {
        Self(tag.into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Pronunciation system associated with a linguistic variety.
pub struct Phonology {
    pub name: String,
}

impl Phonology {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Optional lexical resource associated with a linguistic variety.
pub struct Lexicon {
    pub name: String,
}

impl Lexicon {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LinguisticRuntimeProfile {
    #[default]
    Realtime,
    Batch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Language variety context used by orthography-to-phoneme realizers.
pub struct LinguisticVariety {
    pub tag: Option<VarietyTag>,
    pub name: String,
    pub phonology: Phonology,
    pub lexicon: Option<Lexicon>,
    pub runtime: LinguisticRuntimeProfile,
}

impl LinguisticVariety {
    pub fn tagged(tag: VarietyTag, name: impl Into<String>, phonology: Phonology) -> Self {
        Self {
            tag: Some(tag),
            name: name.into(),
            phonology,
            lexicon: None,
            runtime: LinguisticRuntimeProfile::default(),
        }
    }

    pub fn untagged(name: impl Into<String>, phonology: Phonology) -> Self {
        Self {
            tag: None,
            name: name.into(),
            phonology,
            lexicon: None,
            runtime: LinguisticRuntimeProfile::default(),
        }
    }
}

/// Which English phonological variety drives phonotactics and inventory policy.
///
/// Only [`GeneralAmerican`][EnglishVariety::GeneralAmerican] is a full
/// production implementation. The others intentionally use GA as a labelled
/// stub for future differentiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnglishVariety {
    /// General American English (default).
    GeneralAmerican,
    /// Received Pronunciation / Southern British English.
    ReceivedPronunciation,
    /// Scottish Standard English.
    ScottishEnglish,
    /// African American English.
    AfricanAmericanEnglish,
    /// Deliberately permissive profile for singing or poetic scansion where
    /// normal phonotactic constraints are relaxed.
    PermissiveSinging,
}

impl EnglishVariety {
    /// Construct the [`PhonemicInventory`] for this variety.
    ///
    /// Currently only [`GeneralAmerican`][EnglishVariety::GeneralAmerican] has
    /// a complete inventory. All other variants use the GA inventory with
    /// distinct identifiers as clearly labelled stubs.
    pub fn phonemic_inventory(self) -> PhonemicInventory {
        match self {
            EnglishVariety::GeneralAmerican => general_american_english(),
            EnglishVariety::ReceivedPronunciation => {
                stub_inventory("en-GB-RP", "Received Pronunciation (stub)")
            }
            EnglishVariety::ScottishEnglish => {
                stub_inventory("en-GB-ScotE", "Scottish English (stub)")
            }
            EnglishVariety::AfricanAmericanEnglish => {
                stub_inventory("en-US-AAE", "African American English (stub)")
            }
            EnglishVariety::PermissiveSinging => {
                // Same inventory as GA but with a broad comparison mode so
                // aspiration and other diacritics don't interfere.
                let mut inv = general_american_english();
                inv.id = VarietyId::new("en-US-singing");
                inv.label = "Permissive Singing Profile".into();
                inv.phone_equality = PhoneEqualityOptions {
                    mode: PhoneComparisonMode::Broad,
                    ignore_diacritics: true,
                    ignore_length: true,
                    ..Default::default()
                };
                inv
            }
        }
    }
}

fn stub_inventory(id: &str, label: &str) -> PhonemicInventory {
    // TODO: differentiate these varieties from GA
    let mut inv = general_american_english();
    inv.id = VarietyId::new(id);
    inv.label = label.into();
    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_varieties_have_distinct_ids() {
        let ga = EnglishVariety::GeneralAmerican.phonemic_inventory();
        let rp = EnglishVariety::ReceivedPronunciation.phonemic_inventory();
        assert_ne!(ga.id, rp.id);
    }
}
