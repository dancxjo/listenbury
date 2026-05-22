use super::phonology::{PhonemicInventory, VarietyImplementationStatus};
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
    ///
    /// Use [`implementation_status`][EnglishVariety::implementation_status] to
    /// check whether this variety is a complete implementation, a GA-derived
    /// stub, or an intentionally permissive profile.
    pub fn phonemic_inventory(self) -> PhonemicInventory {
        RuleRegistry::builtin()
            .inventory(self.rule_id())
            .expect("built-in registry should include English variety profile")
    }

    /// Return the implementation status of this variety's phonological profile.
    ///
    /// This makes stub status explicit so that metadata consumers can detect
    /// when a variety has a distinct label and ID but not yet a distinct
    /// phonological implementation.  The status is embedded in the
    /// [`PhonemicInventory`] returned by
    /// [`phonemic_inventory`][EnglishVariety::phonemic_inventory].
    ///
    /// Contributors advancing a variety beyond stub status should add real
    /// phonological differences — inventory extensions, allophone rules, or
    /// distinct phonotactic tables — and change the corresponding
    /// [`ImplementationStatusSpec`][crate::linguistic::rule_registry] entry to
    /// [`VarietyImplementationStatus::Complete`].
    pub fn implementation_status(self) -> VarietyImplementationStatus {
        self.phonemic_inventory().implementation_status
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

    #[test]
    fn all_non_ga_varieties_have_distinct_ids_from_ga() {
        let ga_id = EnglishVariety::GeneralAmerican.phonemic_inventory().id;
        for variety in [
            EnglishVariety::ReceivedPronunciation,
            EnglishVariety::ScottishEnglish,
            EnglishVariety::AfricanAmericanEnglish,
            EnglishVariety::PermissiveSinging,
        ] {
            assert_ne!(
                variety.phonemic_inventory().id,
                ga_id,
                "{variety:?} should have a distinct ID from GA",
            );
        }
    }

    #[test]
    fn general_american_is_complete() {
        assert_eq!(
            EnglishVariety::GeneralAmerican.implementation_status(),
            VarietyImplementationStatus::Complete,
        );
    }

    #[test]
    fn rp_is_ga_derived_stub() {
        assert_eq!(
            EnglishVariety::ReceivedPronunciation.implementation_status(),
            VarietyImplementationStatus::StubDerivedFrom(VarietyId::new("en-US-GA")),
        );
    }

    #[test]
    fn scottish_english_is_ga_derived_stub() {
        assert_eq!(
            EnglishVariety::ScottishEnglish.implementation_status(),
            VarietyImplementationStatus::StubDerivedFrom(VarietyId::new("en-US-GA")),
        );
    }

    #[test]
    fn aae_is_ga_derived_stub() {
        assert_eq!(
            EnglishVariety::AfricanAmericanEnglish.implementation_status(),
            VarietyImplementationStatus::StubDerivedFrom(VarietyId::new("en-US-GA")),
        );
    }

    #[test]
    fn permissive_singing_is_permissive_profile() {
        assert_eq!(
            EnglishVariety::PermissiveSinging.implementation_status(),
            VarietyImplementationStatus::PermissiveProfile,
        );
    }

    #[test]
    fn implementation_status_is_embedded_in_phonemic_inventory() {
        let inv = EnglishVariety::ReceivedPronunciation.phonemic_inventory();
        assert_eq!(
            inv.implementation_status,
            VarietyImplementationStatus::StubDerivedFrom(VarietyId::new("en-US-GA")),
        );
    }
}
