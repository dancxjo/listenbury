use super::phonology::PhonemicInventory;
use super::rule_registry::RuleRegistry;

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
/// production implementation. The others intentionally use GA as a labeled
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
    pub fn rule_id(self) -> &'static str {
        match self {
            EnglishVariety::GeneralAmerican => "en-US-GA",
            EnglishVariety::ReceivedPronunciation => "en-GB-RP",
            EnglishVariety::ScottishEnglish => "en-GB-ScotE",
            EnglishVariety::AfricanAmericanEnglish => "en-US-AAE",
            EnglishVariety::PermissiveSinging => "en-US-singing",
        }
    }

    /// Construct the [`PhonemicInventory`] for this variety.
    ///
    /// Currently only [`GeneralAmerican`][EnglishVariety::GeneralAmerican] has
    /// a complete inventory. All other variants use the GA inventory with
    /// distinct identifiers as clearly labeled stubs.
    pub fn phonemic_inventory(self) -> PhonemicInventory {
        RuleRegistry::builtin()
            .inventory(self.rule_id())
            .expect("built-in registry should include English variety profile")
    }
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
