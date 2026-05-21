use crate::linguistic::phonology::{
    Phone, PhoneString, PhonemeClass, PhonemicInventory, phones_equivalent,
};
use crate::prosody::phonotactics::EnglishVariety;
use crate::prosody::phonotactics::profile::{OnsetVerdict, PhonotacticProfile};
use crate::prosody::phonotactics::tables::{
    illegal_single_onsets, legal_coda_clusters, legal_onset_clusters,
};

/// English phonotactic profile, operating on [`Phone`] objects via
/// [`PhonemicInventory`]-driven comparison.
///
/// Construct with [`EnglishPhonotactics::for_variety`].
///
/// ### Internal representation
///
/// Nuclei, illegal onsets, and cluster lists are stored as [`Phone`] /
/// [`PhoneString`] values.  Lookup is done by iterating and calling
/// `phone_matches` (which uses the inventory's [`PhoneEqualityOptions`]),
/// so a received `[tʰ]` can match a table entry `/t/` when the variety's
/// policy says aspiration is non-contrastive.
pub struct EnglishPhonotactics {
    variety: EnglishVariety,
    inventory: PhonemicInventory,
    nuclei: Vec<Phone>,
    illegal_single_onsets: Vec<Phone>,
    legal_onset_clusters: Vec<PhoneString>,
    legal_coda_clusters: Vec<PhoneString>,
}

impl EnglishPhonotactics {
    /// Construct an English phonotactic profile for `variety`.
    ///
    /// # Example
    ///
    /// ```
    /// use listenbury::prosody::phonotactics::{EnglishPhonotactics, EnglishVariety, PhonotacticProfile};
    /// use listenbury::linguistic::phonology::Phone;
    ///
    /// let ga = EnglishPhonotactics::for_variety(EnglishVariety::GeneralAmerican);
    /// let pr = [Phone::mapped("p"), Phone::mapped("ɹ")];
    /// assert!(ga.is_legal_onset(&pr.iter().collect::<Vec<_>>()));  // /pɹ/ "pretty"
    /// let tl = [Phone::mapped("t"), Phone::mapped("l")];
    /// assert!(!ga.is_legal_onset(&tl.iter().collect::<Vec<_>>()));  // /tl/ illegal in GA
    /// ```
    pub fn for_variety(variety: EnglishVariety) -> Self {
        let inventory = variety.phonemic_inventory();

        // ── Nuclei ────────────────────────────────────────────────────────────
        // Derived dynamically from the inventory's vowel phonemes so that the
        // phonotactic profile stays in sync with the phonemic inventory and
        // duplicated vowel knowledge is avoided.
        let nuclei: Vec<Phone> = inventory
            .phonemes_of_class(PhonemeClass::Vowel)
            .into_iter()
            .flat_map(|def| def.default_phone_string.phones.iter().cloned())
            .collect();

        Self {
            variety,
            inventory,
            nuclei,
            illegal_single_onsets: illegal_single_onsets(),
            legal_onset_clusters: legal_onset_clusters(variety),
            legal_coda_clusters: legal_coda_clusters(),
        }
    }

    /// Return the active [`EnglishVariety`].
    pub fn variety(&self) -> EnglishVariety {
        self.variety
    }

    /// Return the [`PhonemicInventory`] backing this profile.
    pub fn inventory(&self) -> &PhonemicInventory {
        &self.inventory
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn phone_matches(&self, left: &Phone, right: &Phone) -> bool {
        phones_equivalent(left, right, &self.inventory.phone_equality)
    }

    fn cluster_matches(&self, actual: &[&Phone], expected: &PhoneString) -> bool {
        actual.len() == expected.phones.len()
            && actual
                .iter()
                .zip(expected.phones.iter())
                .all(|(a, e)| self.phone_matches(a, e))
    }
}

impl PhonotacticProfile for EnglishPhonotactics {
    fn variety_name(&self) -> &str {
        &self.inventory.label
    }

    fn is_nucleus(&self, phone: &Phone) -> bool {
        self.nuclei.iter().any(|n| self.phone_matches(phone, n))
    }

    fn onset_verdict(&self, cluster: &[&Phone]) -> OnsetVerdict {
        let phones: Vec<Phone> = cluster.iter().map(|p| (*p).clone()).collect();

        if cluster.is_empty() {
            return OnsetVerdict {
                cluster: phones,
                is_legal: true,
                reason: "null onset is always legal".into(),
            };
        }

        if cluster.len() == 1 {
            let phone = cluster[0];
            if self
                .illegal_single_onsets
                .iter()
                .any(|p| self.phone_matches(phone, p))
            {
                return OnsetVerdict {
                    cluster: phones,
                    is_legal: false,
                    reason: format!(
                        "/{ipa}/ is not a legal syllable onset in {variety}",
                        ipa = phone.ipa,
                        variety = self.variety_name()
                    ),
                };
            }
            if self.nuclei.iter().any(|n| self.phone_matches(phone, n)) {
                return OnsetVerdict {
                    cluster: phones,
                    is_legal: false,
                    reason: format!(
                        "/{ipa}/ is a nucleus, not an onset consonant",
                        ipa = phone.ipa
                    ),
                };
            }
            return OnsetVerdict {
                cluster: phones,
                is_legal: true,
                reason: format!("/{ipa}/ is a legal simple onset", ipa = phone.ipa),
            };
        }

        // Multi-phone cluster: check legal onset cluster table.
        let is_legal = self
            .legal_onset_clusters
            .iter()
            .any(|ps| self.cluster_matches(cluster, ps));
        let ipa_cluster: String = cluster.iter().map(|p| p.ipa.as_str()).collect();
        let slash_cluster = format!("/{ipa_cluster}/");
        let reason = if is_legal {
            format!(
                "{slash_cluster} is a legal onset cluster in {}",
                self.variety_name()
            )
        } else {
            format!(
                "{slash_cluster} is not a legal onset cluster in {}",
                self.variety_name()
            )
        };
        OnsetVerdict {
            cluster: phones,
            is_legal,
            reason,
        }
    }

    fn is_legal_coda(&self, cluster: &[&Phone]) -> bool {
        if cluster.is_empty() {
            return true;
        }
        if cluster.len() == 1 {
            // Any single non-nucleus phone can be a simple coda in English.
            return !self
                .nuclei
                .iter()
                .any(|n| self.phone_matches(cluster[0], n));
        }
        self.legal_coda_clusters
            .iter()
            .any(|ps| self.cluster_matches(cluster, ps))
    }
}
