use std::collections::{HashMap, HashSet};

use super::inventory::english_phoneme_table;
use super::phonology::{
    Phone, PhoneComparisonMode, PhoneEqualityOptions, PhoneStatus, PhoneString, PhonemeClass,
    PhonemeDefinition, PhonemeId, PhonemeSchema, PhonemicInventory, SourceSymbol, VarietyId,
    VarietyImplementationStatus, feature_bundle_for_arpabet,
};
use crate::prosody::phonotactics::tables::{
    illegal_single_onsets, legal_coda_clusters, legal_onset_clusters,
    permissive_singing_onset_additions,
};

#[derive(Debug, Clone, PartialEq)]
pub struct InventoryData {
    pub phonemes: Vec<PhonemeDefinition>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhonotacticData {
    pub illegal_single_onsets: Vec<Phone>,
    pub legal_onset_clusters: Vec<PhoneString>,
    pub legal_coda_clusters: Vec<PhoneString>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleFragment {
    pub includes: Vec<String>,
    pub inventory: Option<String>,
    pub phonotactics: Vec<String>,
    pub phone_equality: Option<String>,
    pub legal_onset_additions: Vec<PhoneString>,
    pub legal_coda_additions: Vec<PhoneString>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarietyRuleData {
    pub id: String,
    pub language: String,
    pub label: String,
    pub implementation_status: VarietyImplementationStatus,
    pub includes: Vec<String>,
    pub inventory: Option<String>,
    pub phonotactics: Vec<String>,
    pub phone_equality: Option<String>,
    pub legal_onset_additions: Vec<PhoneString>,
    pub legal_coda_additions: Vec<PhoneString>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuleProfile {
    pub id: String,
    pub language: String,
    pub label: String,
    pub implementation_status: VarietyImplementationStatus,
    pub inventory: PhonemicInventory,
    pub illegal_single_onsets: Vec<Phone>,
    pub legal_onset_clusters: Vec<PhoneString>,
    pub legal_coda_clusters: Vec<PhoneString>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleRegistryError {
    MissingRule(String),
    MissingFragment(String),
    MissingInventory(String),
    MissingPhonotactics(String),
    MissingPhoneEquality(String),
    CyclicInclude(String),
}

impl std::fmt::Display for RuleRegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuleRegistryError::MissingRule(id) => write!(f, "missing rule `{id}`"),
            RuleRegistryError::MissingFragment(id) => write!(f, "missing fragment `{id}`"),
            RuleRegistryError::MissingInventory(id) => write!(f, "missing inventory `{id}`"),
            RuleRegistryError::MissingPhonotactics(id) => {
                write!(f, "missing phonotactics pack `{id}`")
            }
            RuleRegistryError::MissingPhoneEquality(id) => {
                write!(f, "missing phone equality policy `{id}`")
            }
            RuleRegistryError::CyclicInclude(id) => write!(f, "cyclic fragment include at `{id}`"),
        }
    }
}

impl std::error::Error for RuleRegistryError {}

#[derive(Debug, Clone)]
pub struct RuleRegistry {
    pub inventories: HashMap<String, InventoryData>,
    pub phonotactics: HashMap<String, PhonotacticData>,
    pub phone_equality: HashMap<String, PhoneEqualityOptions>,
    pub fragments: HashMap<String, RuleFragment>,
    rules: HashMap<String, VarietyRuleData>,
}

impl Default for RuleRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

impl RuleRegistry {
    pub fn builtin() -> Self {
        let inventories = INVENTORIES
            .iter()
            .map(|spec| (spec.id.to_string(), spec.to_data()))
            .collect();

        let phonotactics = PHONOTACTIC_PACKS
            .iter()
            .map(|spec| (spec.id.to_string(), spec.to_data()))
            .collect();

        let mut phone_equality = HashMap::new();
        phone_equality.insert("default".into(), PhoneEqualityOptions::default());
        phone_equality.insert(
            "english/permissive_singing".into(),
            PhoneEqualityOptions {
                mode: PhoneComparisonMode::Broad,
                ignore_diacritics: true,
                ignore_length: true,
                ..Default::default()
            },
        );

        let fragments = RULE_FRAGMENTS
            .iter()
            .map(|spec| (spec.id.to_string(), spec.to_fragment()))
            .collect();

        let rules = VARIETY_RULES
            .iter()
            .map(|spec| (spec.id.to_string(), spec.to_rule()))
            .collect();

        Self {
            inventories,
            phonotactics,
            phone_equality,
            fragments,
            rules,
        }
    }

    pub fn rules(&self) -> &HashMap<String, VarietyRuleData> {
        &self.rules
    }

    pub fn profile(&self, id: &str) -> Result<RuleProfile, RuleRegistryError> {
        let rule = self
            .rules
            .get(id)
            .ok_or_else(|| RuleRegistryError::MissingRule(id.to_string()))?;
        let resolved = self.resolve_rule(id, &mut HashSet::new())?;
        let inventory_id = resolved
            .inventory
            .ok_or_else(|| RuleRegistryError::MissingInventory(format!("inventory for `{id}`")))?;
        let inventory_data = self
            .inventories
            .get(&inventory_id)
            .ok_or_else(|| RuleRegistryError::MissingInventory(inventory_id.clone()))?;
        let phone_equality_id = resolved.phone_equality.unwrap_or_else(|| "default".into());
        let phone_equality = self
            .phone_equality
            .get(&phone_equality_id)
            .ok_or_else(|| RuleRegistryError::MissingPhoneEquality(phone_equality_id.clone()))?
            .clone();

        let mut illegal_single_onsets = Vec::new();
        let mut legal_onset_clusters = Vec::new();
        let mut legal_coda_clusters = Vec::new();
        for phonotactic_ref in resolved.phonotactics {
            let pack = self
                .phonotactics
                .get(&phonotactic_ref)
                .ok_or_else(|| RuleRegistryError::MissingPhonotactics(phonotactic_ref.clone()))?;
            illegal_single_onsets.extend(pack.illegal_single_onsets.iter().cloned());
            legal_onset_clusters.extend(pack.legal_onset_clusters.iter().cloned());
            legal_coda_clusters.extend(pack.legal_coda_clusters.iter().cloned());
        }
        legal_onset_clusters.extend(resolved.legal_onset_additions);
        legal_coda_clusters.extend(resolved.legal_coda_additions);
        dedup_phones(&mut illegal_single_onsets);
        dedup_phone_strings(&mut legal_onset_clusters);
        dedup_phone_strings(&mut legal_coda_clusters);

        let inventory = PhonemicInventory {
            id: VarietyId::new(rule.id.clone()),
            implementation_status: rule.implementation_status.clone(),
            language: rule.language.clone(),
            label: rule.label.clone(),
            phonemes: inventory_data.phonemes.clone(),
            phone_equality,
        };

        Ok(RuleProfile {
            id: rule.id.clone(),
            language: rule.language.clone(),
            label: rule.label.clone(),
            implementation_status: rule.implementation_status.clone(),
            inventory,
            illegal_single_onsets,
            legal_onset_clusters,
            legal_coda_clusters,
        })
    }

    pub fn inventory(&self, id: &str) -> Result<PhonemicInventory, RuleRegistryError> {
        self.profile(id).map(|p| p.inventory)
    }

    fn resolve_rule(
        &self,
        id: &str,
        visiting: &mut HashSet<String>,
    ) -> Result<ResolvedRule, RuleRegistryError> {
        let rule = self
            .rules
            .get(id)
            .ok_or_else(|| RuleRegistryError::MissingRule(id.to_string()))?;
        let mut resolved = ResolvedRule::default();
        for include in &rule.includes {
            self.apply_fragment(include, visiting, &mut resolved)?;
        }
        resolved.apply_entry(
            rule.inventory.clone(),
            rule.phonotactics.clone(),
            rule.phone_equality.clone(),
            rule.legal_onset_additions.clone(),
            rule.legal_coda_additions.clone(),
        );
        Ok(resolved)
    }

    fn apply_fragment(
        &self,
        id: &str,
        visiting: &mut HashSet<String>,
        resolved: &mut ResolvedRule,
    ) -> Result<(), RuleRegistryError> {
        if !visiting.insert(id.to_string()) {
            return Err(RuleRegistryError::CyclicInclude(id.to_string()));
        }
        let fragment = self
            .fragments
            .get(id)
            .ok_or_else(|| RuleRegistryError::MissingFragment(id.to_string()))?;
        for include in &fragment.includes {
            self.apply_fragment(include, visiting, resolved)?;
        }
        resolved.apply_entry(
            fragment.inventory.clone(),
            fragment.phonotactics.clone(),
            fragment.phone_equality.clone(),
            fragment.legal_onset_additions.clone(),
            fragment.legal_coda_additions.clone(),
        );
        visiting.remove(id);
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
struct ResolvedRule {
    inventory: Option<String>,
    phonotactics: Vec<String>,
    phone_equality: Option<String>,
    legal_onset_additions: Vec<PhoneString>,
    legal_coda_additions: Vec<PhoneString>,
}

impl ResolvedRule {
    fn apply_entry(
        &mut self,
        inventory: Option<String>,
        mut phonotactics: Vec<String>,
        phone_equality: Option<String>,
        mut legal_onset_additions: Vec<PhoneString>,
        mut legal_coda_additions: Vec<PhoneString>,
    ) {
        if let Some(inventory) = inventory {
            self.inventory = Some(inventory);
        }
        self.phonotactics.append(&mut phonotactics);
        if let Some(phone_equality) = phone_equality {
            self.phone_equality = Some(phone_equality);
        }
        self.legal_onset_additions
            .append(&mut legal_onset_additions);
        self.legal_coda_additions.append(&mut legal_coda_additions);
    }
}

fn dedup_phones(phones: &mut Vec<Phone>) {
    let mut seen = HashSet::new();
    phones.retain(|phone| seen.insert(phone.ipa.clone()));
}

fn dedup_phone_strings(phone_strings: &mut Vec<PhoneString>) {
    let mut seen = HashSet::new();
    phone_strings.retain(|cluster| {
        let key: String = cluster.phones.iter().map(|p| p.ipa.as_str()).collect();
        seen.insert(key)
    });
}

fn phone_string(symbols: &[&str]) -> PhoneString {
    PhoneString {
        phones: symbols.iter().map(|s| Phone::mapped(*s)).collect(),
    }
}

struct InventorySpec {
    id: &'static str,
    source: InventorySource,
}

enum InventorySource {
    Builder(fn() -> Vec<PhonemeDefinition>),
    Rows(&'static [PhonemeRowSpec]),
}

struct PhonemeRowSpec {
    symbol: &'static str,
    ipa: &'static str,
    classes: &'static [PhonemeClass],
}

struct PhonotacticSpec {
    id: &'static str,
    illegal_single_onsets: PhonotacticRows,
    legal_onset_clusters: PhonotacticRows,
    legal_coda_clusters: PhonotacticRows,
}

enum PhonotacticRows {
    Builder(fn() -> Vec<PhoneString>),
    Static(&'static [&'static [&'static str]]),
}

struct RuleFragmentSpec {
    id: &'static str,
    includes: &'static [&'static str],
    inventory: Option<&'static str>,
    phonotactics: &'static [&'static str],
    phone_equality: Option<&'static str>,
    legal_onset_additions: RulePhoneStringRows,
    legal_coda_additions: RulePhoneStringRows,
}

struct VarietyRuleSpec {
    id: &'static str,
    language: &'static str,
    label: &'static str,
    implementation_status: VarietyImplementationStatusSpec,
    includes: &'static [&'static str],
    inventory: Option<&'static str>,
    phonotactics: &'static [&'static str],
    phone_equality: Option<&'static str>,
    legal_onset_additions: RulePhoneStringRows,
    legal_coda_additions: RulePhoneStringRows,
}

enum RulePhoneStringRows {
    Builder(fn() -> Vec<PhoneString>),
    Static(&'static [&'static [&'static str]]),
}

#[derive(Clone, Copy)]
enum VarietyImplementationStatusSpec {
    Complete,
    StubDerivedFrom(&'static str),
    PermissiveProfile,
}

impl InventorySpec {
    fn to_data(&self) -> InventoryData {
        let phonemes = match &self.source {
            InventorySource::Builder(builder) => builder(),
            InventorySource::Rows(rows) => rows.iter().map(PhonemeRowSpec::to_definition).collect(),
        };
        InventoryData { phonemes }
    }
}

impl PhonemeRowSpec {
    fn to_definition(&self) -> PhonemeDefinition {
        PhonemeDefinition {
            id: PhonemeId::new(self.symbol.to_lowercase()),
            ipa: self.ipa.to_string(),
            source_symbols: vec![
                SourceSymbol {
                    schema: PhonemeSchema::Ipa,
                    symbol: self.ipa.to_string(),
                },
                SourceSymbol {
                    schema: PhonemeSchema::Arpabet,
                    symbol: self.symbol.to_string(),
                },
            ],
            default_phone_string: PhoneString {
                phones: vec![Phone {
                    ipa: self.ipa.to_string(),
                    source_symbol: Some(self.symbol.to_string()),
                    status: PhoneStatus::Mapped,
                }],
            },
            classes: self.classes.to_vec(),
            features: feature_bundle_for_arpabet(self.symbol),
        }
    }
}

impl PhonotacticSpec {
    fn to_data(&self) -> PhonotacticData {
        PhonotacticData {
            illegal_single_onsets: self
                .illegal_single_onsets
                .to_phone_strings()
                .into_iter()
                .flat_map(|phone_string| phone_string.phones)
                .collect(),
            legal_onset_clusters: self.legal_onset_clusters.to_phone_strings(),
            legal_coda_clusters: self.legal_coda_clusters.to_phone_strings(),
        }
    }
}

impl PhonotacticRows {
    fn to_phone_strings(&self) -> Vec<PhoneString> {
        match self {
            PhonotacticRows::Builder(builder) => builder(),
            PhonotacticRows::Static(rows) => rows.iter().map(|row| phone_string(row)).collect(),
        }
    }
}

impl RuleFragmentSpec {
    fn to_fragment(&self) -> RuleFragment {
        RuleFragment {
            includes: strings(self.includes),
            inventory: self.inventory.map(str::to_string),
            phonotactics: strings(self.phonotactics),
            phone_equality: self.phone_equality.map(str::to_string),
            legal_onset_additions: self.legal_onset_additions.to_phone_strings(),
            legal_coda_additions: self.legal_coda_additions.to_phone_strings(),
        }
    }
}

impl VarietyRuleSpec {
    fn to_rule(&self) -> VarietyRuleData {
        VarietyRuleData {
            id: self.id.to_string(),
            language: self.language.to_string(),
            label: self.label.to_string(),
            implementation_status: self.implementation_status.to_status(),
            includes: strings(self.includes),
            inventory: self.inventory.map(str::to_string),
            phonotactics: strings(self.phonotactics),
            phone_equality: self.phone_equality.map(str::to_string),
            legal_onset_additions: self.legal_onset_additions.to_phone_strings(),
            legal_coda_additions: self.legal_coda_additions.to_phone_strings(),
        }
    }
}

impl VarietyImplementationStatusSpec {
    fn to_status(self) -> VarietyImplementationStatus {
        match self {
            VarietyImplementationStatusSpec::Complete => VarietyImplementationStatus::Complete,
            VarietyImplementationStatusSpec::StubDerivedFrom(id) => {
                VarietyImplementationStatus::StubDerivedFrom(VarietyId::new(id))
            }
            VarietyImplementationStatusSpec::PermissiveProfile => {
                VarietyImplementationStatus::PermissiveProfile
            }
        }
    }
}

impl RulePhoneStringRows {
    fn to_phone_strings(&self) -> Vec<PhoneString> {
        match self {
            RulePhoneStringRows::Builder(builder) => builder(),
            RulePhoneStringRows::Static(rows) => rows.iter().map(|row| phone_string(row)).collect(),
        }
    }
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}

static INVENTORIES: &[InventorySpec] = &[
    InventorySpec {
        id: "english/base_inventory",
        source: InventorySource::Builder(english_phoneme_table),
    },
    InventorySpec {
        id: "eo/base_inventory",
        source: InventorySource::Rows(ESPERANTO_PHONEMES),
    },
];

static ESPERANTO_PHONEMES: &[PhonemeRowSpec] = &[
    PhonemeRowSpec {
        symbol: "A",
        ipa: "a",
        classes: &[PhonemeClass::Vowel],
    },
    PhonemeRowSpec {
        symbol: "E",
        ipa: "e",
        classes: &[PhonemeClass::Vowel],
    },
    PhonemeRowSpec {
        symbol: "I",
        ipa: "i",
        classes: &[PhonemeClass::Vowel],
    },
    PhonemeRowSpec {
        symbol: "O",
        ipa: "o",
        classes: &[PhonemeClass::Vowel],
    },
    PhonemeRowSpec {
        symbol: "U",
        ipa: "u",
        classes: &[PhonemeClass::Vowel],
    },
    PhonemeRowSpec {
        symbol: "P",
        ipa: "p",
        classes: &[PhonemeClass::Consonant],
    },
    PhonemeRowSpec {
        symbol: "L",
        ipa: "l",
        classes: &[PhonemeClass::Consonant],
    },
    PhonemeRowSpec {
        symbol: "R",
        ipa: "r",
        classes: &[PhonemeClass::Consonant],
    },
    PhonemeRowSpec {
        symbol: "S",
        ipa: "s",
        classes: &[PhonemeClass::Consonant],
    },
    PhonemeRowSpec {
        symbol: "N",
        ipa: "n",
        classes: &[PhonemeClass::Consonant],
    },
    PhonemeRowSpec {
        symbol: "M",
        ipa: "m",
        classes: &[PhonemeClass::Consonant],
    },
    PhonemeRowSpec {
        symbol: "T",
        ipa: "t",
        classes: &[PhonemeClass::Consonant],
    },
    PhonemeRowSpec {
        symbol: "K",
        ipa: "k",
        classes: &[PhonemeClass::Consonant],
    },
];

static PHONOTACTIC_PACKS: &[PhonotacticSpec] = &[
    PhonotacticSpec {
        id: "english/base_onsets",
        illegal_single_onsets: PhonotacticRows::Builder(illegal_single_onsets_as_strings),
        legal_onset_clusters: PhonotacticRows::Builder(legal_onset_clusters),
        legal_coda_clusters: PhonotacticRows::Static(&[]),
    },
    PhonotacticSpec {
        id: "english/base_codas",
        illegal_single_onsets: PhonotacticRows::Static(&[]),
        legal_onset_clusters: PhonotacticRows::Static(&[]),
        legal_coda_clusters: PhonotacticRows::Builder(legal_coda_clusters),
    },
    PhonotacticSpec {
        id: "eo/base_phonotactics",
        illegal_single_onsets: PhonotacticRows::Static(&[]),
        legal_onset_clusters: PhonotacticRows::Static(&[&["p", "l"], &["p", "r"]]),
        legal_coda_clusters: PhonotacticRows::Static(&[]),
    },
];

static RULE_FRAGMENTS: &[RuleFragmentSpec] = &[
    RuleFragmentSpec {
        id: "english/base_inventory",
        includes: &[],
        inventory: Some("english/base_inventory"),
        phonotactics: &[],
        phone_equality: None,
        legal_onset_additions: RulePhoneStringRows::Static(&[]),
        legal_coda_additions: RulePhoneStringRows::Static(&[]),
    },
    RuleFragmentSpec {
        id: "english/base_onsets",
        includes: &[],
        inventory: None,
        phonotactics: &["english/base_onsets"],
        phone_equality: None,
        legal_onset_additions: RulePhoneStringRows::Static(&[]),
        legal_coda_additions: RulePhoneStringRows::Static(&[]),
    },
    RuleFragmentSpec {
        id: "english/base_codas",
        includes: &[],
        inventory: None,
        phonotactics: &["english/base_codas"],
        phone_equality: None,
        legal_onset_additions: RulePhoneStringRows::Static(&[]),
        legal_coda_additions: RulePhoneStringRows::Static(&[]),
    },
    RuleFragmentSpec {
        id: "english/rhotic",
        includes: &[],
        inventory: None,
        phonotactics: &[],
        phone_equality: None,
        legal_onset_additions: RulePhoneStringRows::Static(&[]),
        legal_coda_additions: RulePhoneStringRows::Static(&[]),
    },
    RuleFragmentSpec {
        id: "english/non_rhotic",
        includes: &[],
        inventory: None,
        phonotactics: &[],
        phone_equality: None,
        legal_onset_additions: RulePhoneStringRows::Static(&[]),
        legal_coda_additions: RulePhoneStringRows::Static(&[]),
    },
    RuleFragmentSpec {
        id: "english/yod_dropping",
        includes: &[],
        inventory: None,
        phonotactics: &[],
        phone_equality: None,
        legal_onset_additions: RulePhoneStringRows::Static(&[]),
        legal_coda_additions: RulePhoneStringRows::Static(&[]),
    },
    RuleFragmentSpec {
        id: "english/permissive_singing",
        includes: &[],
        inventory: None,
        phonotactics: &[],
        phone_equality: Some("english/permissive_singing"),
        legal_onset_additions: RulePhoneStringRows::Builder(permissive_singing_onset_additions),
        legal_coda_additions: RulePhoneStringRows::Static(&[]),
    },
    RuleFragmentSpec {
        id: "eo/base_inventory",
        includes: &[],
        inventory: Some("eo/base_inventory"),
        phonotactics: &[],
        phone_equality: None,
        legal_onset_additions: RulePhoneStringRows::Static(&[]),
        legal_coda_additions: RulePhoneStringRows::Static(&[]),
    },
    RuleFragmentSpec {
        id: "eo/base_phonotactics",
        includes: &[],
        inventory: None,
        phonotactics: &["eo/base_phonotactics"],
        phone_equality: None,
        legal_onset_additions: RulePhoneStringRows::Static(&[]),
        legal_coda_additions: RulePhoneStringRows::Static(&[]),
    },
];

static VARIETY_RULES: &[VarietyRuleSpec] = &[
    VarietyRuleSpec {
        id: "en-US-GA",
        language: "en",
        label: "General American English",
        implementation_status: VarietyImplementationStatusSpec::Complete,
        includes: &[
            "english/base_inventory",
            "english/base_onsets",
            "english/base_codas",
            "english/rhotic",
        ],
        inventory: None,
        phonotactics: &[],
        phone_equality: None,
        legal_onset_additions: RulePhoneStringRows::Static(&[]),
        legal_coda_additions: RulePhoneStringRows::Static(&[]),
    },
    VarietyRuleSpec {
        id: "en-US-singing",
        language: "en",
        label: "Permissive Singing Profile",
        implementation_status: VarietyImplementationStatusSpec::PermissiveProfile,
        includes: &[
            "english/base_inventory",
            "english/base_onsets",
            "english/base_codas",
            "english/rhotic",
            "english/permissive_singing",
        ],
        inventory: None,
        phonotactics: &[],
        phone_equality: None,
        legal_onset_additions: RulePhoneStringRows::Static(&[]),
        legal_coda_additions: RulePhoneStringRows::Static(&[]),
    },
    VarietyRuleSpec {
        id: "en-GB-RP",
        language: "en",
        label: "Received Pronunciation (stub)",
        implementation_status: VarietyImplementationStatusSpec::StubDerivedFrom("en-US-GA"),
        includes: &[
            "english/base_inventory",
            "english/base_onsets",
            "english/base_codas",
            "english/non_rhotic",
        ],
        inventory: None,
        phonotactics: &[],
        phone_equality: None,
        legal_onset_additions: RulePhoneStringRows::Static(&[]),
        legal_coda_additions: RulePhoneStringRows::Static(&[]),
    },
    VarietyRuleSpec {
        id: "en-GB-ScotE",
        language: "en",
        label: "Scottish English (stub)",
        implementation_status: VarietyImplementationStatusSpec::StubDerivedFrom("en-US-GA"),
        includes: &[
            "english/base_inventory",
            "english/base_onsets",
            "english/base_codas",
            "english/rhotic",
        ],
        inventory: None,
        phonotactics: &[],
        phone_equality: None,
        legal_onset_additions: RulePhoneStringRows::Static(&[]),
        legal_coda_additions: RulePhoneStringRows::Static(&[]),
    },
    VarietyRuleSpec {
        id: "en-US-AAE",
        language: "en",
        label: "African American English (stub)",
        implementation_status: VarietyImplementationStatusSpec::StubDerivedFrom("en-US-GA"),
        includes: &[
            "english/base_inventory",
            "english/base_onsets",
            "english/base_codas",
            "english/rhotic",
        ],
        inventory: None,
        phonotactics: &[],
        phone_equality: None,
        legal_onset_additions: RulePhoneStringRows::Static(&[]),
        legal_coda_additions: RulePhoneStringRows::Static(&[]),
    },
    VarietyRuleSpec {
        id: "eo",
        language: "eo",
        label: "Esperanto (sample)",
        implementation_status: VarietyImplementationStatusSpec::Complete,
        includes: &["eo/base_inventory", "eo/base_phonotactics"],
        inventory: None,
        phonotactics: &[],
        phone_equality: None,
        legal_onset_additions: RulePhoneStringRows::Static(&[]),
        legal_coda_additions: RulePhoneStringRows::Static(&[]),
    },
];

fn illegal_single_onsets_as_strings() -> Vec<PhoneString> {
    illegal_single_onsets()
        .into_iter()
        .map(|phone| PhoneString {
            phones: vec![phone],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contains_cluster(clusters: &[PhoneString], expected: &[&str]) -> bool {
        clusters.iter().any(|cluster| {
            cluster.phones.len() == expected.len()
                && cluster
                    .phones
                    .iter()
                    .zip(expected.iter())
                    .all(|(phone, expected)| phone.ipa == *expected)
        })
    }

    #[test]
    fn registry_profile_lookup_for_general_american() {
        let registry = RuleRegistry::builtin();
        let profile = registry
            .profile("en-US-GA")
            .expect("GA profile should exist");
        assert_eq!(profile.inventory.id, VarietyId::new("en-US-GA"));
        assert!(contains_cluster(
            &profile.legal_onset_clusters,
            &["s", "t", "ɹ"]
        ));
        assert!(!contains_cluster(
            &profile.legal_onset_clusters,
            &["t", "l"]
        ));
    }

    #[test]
    fn registry_profile_lookup_for_esperanto_sample() {
        let registry = RuleRegistry::builtin();
        let profile = registry
            .profile("eo")
            .expect("Esperanto profile should exist");
        assert_eq!(profile.inventory.id, VarietyId::new("eo"));
        assert_eq!(profile.inventory.language, "eo");
        assert!(profile.inventory.find_by_ipa("a").is_some());
    }

    #[test]
    fn registry_fragments_allow_shared_base_with_override() {
        let registry = RuleRegistry::builtin();
        let ga = registry
            .profile("en-US-GA")
            .expect("GA profile should exist");
        let singing = registry
            .profile("en-US-singing")
            .expect("singing profile should exist");
        assert!(contains_cluster(&ga.legal_onset_clusters, &["s", "t", "ɹ"]));
        assert!(contains_cluster(
            &singing.legal_onset_clusters,
            &["s", "t", "ɹ"]
        ));
        assert!(!contains_cluster(&ga.legal_onset_clusters, &["t", "l"]));
        assert!(contains_cluster(&singing.legal_onset_clusters, &["t", "l"]));
        assert_ne!(
            ga.inventory.phone_equality,
            singing.inventory.phone_equality
        );
    }
}
