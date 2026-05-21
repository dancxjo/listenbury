use std::collections::{HashMap, HashSet};

use super::inventory::english_phoneme_table;
use super::phonology::{
    Phone, PhoneComparisonMode, PhoneEqualityOptions, PhoneStatus, PhoneString, PhonemeClass,
    PhonemeDefinition, PhonemeId, PhonemeSchema, PhonemicInventory, SourceSymbol, VarietyId,
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
        let mut inventories = HashMap::new();
        inventories.insert(
            "english/base_inventory".into(),
            InventoryData {
                phonemes: english_phoneme_table(),
            },
        );
        inventories.insert(
            "eo/base_inventory".into(),
            InventoryData {
                phonemes: esperanto_phonemes(),
            },
        );

        let mut phonotactics = HashMap::new();
        phonotactics.insert(
            "english/base_onsets".into(),
            PhonotacticData {
                illegal_single_onsets: illegal_single_onsets(),
                legal_onset_clusters: legal_onset_clusters(),
                legal_coda_clusters: vec![],
            },
        );
        phonotactics.insert(
            "english/base_codas".into(),
            PhonotacticData {
                illegal_single_onsets: vec![],
                legal_onset_clusters: vec![],
                legal_coda_clusters: legal_coda_clusters(),
            },
        );
        phonotactics.insert(
            "eo/base_phonotactics".into(),
            PhonotacticData {
                illegal_single_onsets: vec![],
                // Intentionally tiny proof-of-shape sample for this ticket; not
                // a full Esperanto phonotactic model.
                legal_onset_clusters: vec![phone_string(&["p", "l"]), phone_string(&["p", "r"])],
                legal_coda_clusters: vec![],
            },
        );

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

        let mut fragments = HashMap::new();
        fragments.insert(
            "english/base_inventory".into(),
            RuleFragment {
                includes: vec![],
                inventory: Some("english/base_inventory".into()),
                phonotactics: vec![],
                phone_equality: None,
                legal_onset_additions: vec![],
                legal_coda_additions: vec![],
            },
        );
        fragments.insert(
            "english/base_onsets".into(),
            RuleFragment {
                includes: vec![],
                inventory: None,
                phonotactics: vec!["english/base_onsets".into()],
                phone_equality: None,
                legal_onset_additions: vec![],
                legal_coda_additions: vec![],
            },
        );
        fragments.insert(
            "english/base_codas".into(),
            RuleFragment {
                includes: vec![],
                inventory: None,
                phonotactics: vec!["english/base_codas".into()],
                phone_equality: None,
                legal_onset_additions: vec![],
                legal_coda_additions: vec![],
            },
        );
        fragments.insert(
            "english/rhotic".into(),
            RuleFragment {
                includes: vec![],
                inventory: None,
                phonotactics: vec![],
                phone_equality: None,
                legal_onset_additions: vec![],
                legal_coda_additions: vec![],
            },
        );
        fragments.insert(
            "english/non_rhotic".into(),
            RuleFragment {
                includes: vec![],
                inventory: None,
                phonotactics: vec![],
                phone_equality: None,
                legal_onset_additions: vec![],
                legal_coda_additions: vec![],
            },
        );
        fragments.insert(
            "english/yod_dropping".into(),
            RuleFragment {
                includes: vec![],
                inventory: None,
                phonotactics: vec![],
                phone_equality: None,
                legal_onset_additions: vec![],
                legal_coda_additions: vec![],
            },
        );
        fragments.insert(
            "english/permissive_singing".into(),
            RuleFragment {
                includes: vec![],
                inventory: None,
                phonotactics: vec![],
                phone_equality: Some("english/permissive_singing".into()),
                legal_onset_additions: permissive_singing_onset_additions(),
                legal_coda_additions: vec![],
            },
        );
        fragments.insert(
            "eo/base_inventory".into(),
            RuleFragment {
                includes: vec![],
                inventory: Some("eo/base_inventory".into()),
                phonotactics: vec![],
                phone_equality: None,
                legal_onset_additions: vec![],
                legal_coda_additions: vec![],
            },
        );
        fragments.insert(
            "eo/base_phonotactics".into(),
            RuleFragment {
                includes: vec![],
                inventory: None,
                phonotactics: vec!["eo/base_phonotactics".into()],
                phone_equality: None,
                legal_onset_additions: vec![],
                legal_coda_additions: vec![],
            },
        );

        let mut rules = HashMap::new();
        rules.insert(
            "en-US-GA".into(),
            VarietyRuleData {
                id: "en-US-GA".into(),
                language: "en".into(),
                label: "General American English".into(),
                includes: vec![
                    "english/base_inventory".into(),
                    "english/base_onsets".into(),
                    "english/base_codas".into(),
                    "english/rhotic".into(),
                ],
                inventory: None,
                phonotactics: vec![],
                phone_equality: None,
                legal_onset_additions: vec![],
                legal_coda_additions: vec![],
            },
        );
        rules.insert(
            "en-US-singing".into(),
            VarietyRuleData {
                id: "en-US-singing".into(),
                language: "en".into(),
                label: "Permissive Singing Profile".into(),
                includes: vec![
                    "english/base_inventory".into(),
                    "english/base_onsets".into(),
                    "english/base_codas".into(),
                    "english/rhotic".into(),
                    "english/permissive_singing".into(),
                ],
                inventory: None,
                phonotactics: vec![],
                phone_equality: None,
                legal_onset_additions: vec![],
                legal_coda_additions: vec![],
            },
        );
        rules.insert(
            "en-GB-RP".into(),
            VarietyRuleData {
                id: "en-GB-RP".into(),
                language: "en".into(),
                label: "Received Pronunciation (stub)".into(),
                includes: vec![
                    "english/base_inventory".into(),
                    "english/base_onsets".into(),
                    "english/base_codas".into(),
                    "english/non_rhotic".into(),
                ],
                inventory: None,
                phonotactics: vec![],
                phone_equality: None,
                legal_onset_additions: vec![],
                legal_coda_additions: vec![],
            },
        );
        rules.insert(
            "en-GB-ScotE".into(),
            VarietyRuleData {
                id: "en-GB-ScotE".into(),
                language: "en".into(),
                label: "Scottish English (stub)".into(),
                includes: vec![
                    "english/base_inventory".into(),
                    "english/base_onsets".into(),
                    "english/base_codas".into(),
                    "english/rhotic".into(),
                ],
                inventory: None,
                phonotactics: vec![],
                phone_equality: None,
                legal_onset_additions: vec![],
                legal_coda_additions: vec![],
            },
        );
        rules.insert(
            "en-US-AAE".into(),
            VarietyRuleData {
                id: "en-US-AAE".into(),
                language: "en".into(),
                label: "African American English (stub)".into(),
                includes: vec![
                    "english/base_inventory".into(),
                    "english/base_onsets".into(),
                    "english/base_codas".into(),
                    "english/rhotic".into(),
                ],
                inventory: None,
                phonotactics: vec![],
                phone_equality: None,
                legal_onset_additions: vec![],
                legal_coda_additions: vec![],
            },
        );
        rules.insert(
            "eo".into(),
            VarietyRuleData {
                id: "eo".into(),
                language: "eo".into(),
                label: "Esperanto (sample)".into(),
                includes: vec!["eo/base_inventory".into(), "eo/base_phonotactics".into()],
                inventory: None,
                phonotactics: vec![],
                phone_equality: None,
                legal_onset_additions: vec![],
                legal_coda_additions: vec![],
            },
        );

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
            language: rule.language.clone(),
            label: rule.label.clone(),
            phonemes: inventory_data.phonemes.clone(),
            phone_equality,
        };

        Ok(RuleProfile {
            id: rule.id.clone(),
            language: rule.language.clone(),
            label: rule.label.clone(),
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

fn esperanto_phonemes() -> Vec<PhonemeDefinition> {
    // (phoneme_symbol, ipa, is_vowel)
    // Intentionally minimal sample inventory proving non-English registry
    // shape; this is not a complete Esperanto phoneme list.
    let rows: &[(&str, &str, bool)] = &[
        ("A", "a", true),
        ("E", "e", true),
        ("I", "i", true),
        ("O", "o", true),
        ("U", "u", true),
        ("P", "p", false),
        ("L", "l", false),
        ("R", "r", false),
        ("S", "s", false),
        ("N", "n", false),
        ("M", "m", false),
        ("T", "t", false),
        ("K", "k", false),
    ];

    rows.iter()
        .map(|(symbol, ipa, is_vowel)| {
            let id = PhonemeId::new(symbol.to_lowercase());
            let classes = if *is_vowel {
                vec![PhonemeClass::Vowel]
            } else {
                vec![PhonemeClass::Consonant]
            };
            PhonemeDefinition {
                id,
                ipa: ipa.to_string(),
                source_symbols: vec![
                    SourceSymbol {
                        schema: PhonemeSchema::Ipa,
                        symbol: ipa.to_string(),
                    },
                    SourceSymbol {
                        schema: PhonemeSchema::Arpabet,
                        symbol: symbol.to_string(),
                    },
                ],
                default_phone_string: PhoneString {
                    phones: vec![Phone {
                        ipa: ipa.to_string(),
                        source_symbol: Some(symbol.to_string()),
                        status: PhoneStatus::Mapped,
                    }],
                },
                classes,
            }
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
