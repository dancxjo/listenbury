//! Phonotactic profiles for IPA-based syllabification.
//!
//! A [`PhonotacticProfile`] encapsulates the rules that determine which
//! [`Phone`] sequences are legal onsets or codas in a given linguistic variety.
//! The syllabifier queries a profile rather than embedding hard-coded rules,
//! so phonotactic assumptions are separable from the algorithm and can be
//! swapped for a different English variety or a completely different language.
//!
//! All profile methods receive **[`Phone`] references** from the phonology
//! layer; the `phone.ipa` field carries the IPA surface form, and phone
//! comparisons respect the variety's [`PhoneEqualityOptions`] so that
//! aspiration and other diacritic detail can be treated as non-contrastive.
//!
//! The primary English implementation is [`EnglishPhonotactics`], constructed
//! with [`EnglishPhonotactics::for_variety`].  A maximally permissive
//! [`PermissiveProfile`] is also provided for testing.
//!
//! # Example
//!
//! ```
//! use listenbury::prosody::phonotactics::{EnglishPhonotactics, EnglishVariety, PhonotacticProfile};
//! use listenbury::linguistic::phonology::Phone;
//!
//! let profile = EnglishPhonotactics::for_variety(EnglishVariety::GeneralAmerican);
//!
//! // /stɹ/ is a legal English onset (e.g. "strong", "stream")
//! let stɹ = [Phone::mapped("s"), Phone::mapped("t"), Phone::mapped("ɹ")];
//! assert!(profile.is_legal_onset(&stɹ.iter().collect::<Vec<_>>()));
//!
//! // /tl/ is not a legal onset in General American English
//! let tl = [Phone::mapped("t"), Phone::mapped("l")];
//! assert!(!profile.is_legal_onset(&tl.iter().collect::<Vec<_>>()));
//!
//! // Vowels are nuclei
//! assert!(profile.is_nucleus(&Phone::mapped("ɛ")));
//! assert!(!profile.is_nucleus(&Phone::mapped("t")));
//! ```

use crate::linguistic::phonology::{
    Phone, PhoneString, PhonemicInventory, phones_equivalent,
};
use crate::prosody::syllable::{DiagnosticKind, SyllableDiagnostic};

pub use crate::linguistic::inventory::general_american_english;
pub use crate::linguistic::variety::EnglishVariety;

// ─── Profile trait ────────────────────────────────────────────────────────────

/// Verdict returned when the profile evaluates an onset cluster.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnsetVerdict {
    /// The phones that were evaluated (cloned from the input references).
    pub cluster: Vec<Phone>,
    /// Whether this cluster is a legal onset under the active profile.
    pub is_legal: bool,
    /// Human-readable explanation of the decision, using IPA.
    pub reason: String,
}

impl OnsetVerdict {
    /// Convert this verdict into a [`SyllableDiagnostic`].
    pub fn as_diagnostic(&self) -> SyllableDiagnostic {
        let kind = if self.is_legal {
            DiagnosticKind::LegalOnset
        } else {
            DiagnosticKind::RejectedOnset
        };
        SyllableDiagnostic::new(kind, self.reason.clone())
    }

    /// Concatenate the IPA strings of all phones in this verdict's cluster.
    pub fn cluster_ipa(&self) -> String {
        self.cluster.iter().map(|p| p.ipa.as_str()).collect()
    }
}

/// A phonotactic profile that the syllabifier consults to determine the
/// legality of onset and coda clusters.
///
/// Methods take slices of [`Phone`] references (`&[&Phone]`) so that the
/// syllabifier can build candidate clusters without heap-allocating clones.
///
/// Implement this trait to add new variety profiles.
pub trait PhonotacticProfile {
    /// Display name of this variety/profile, e.g. `"General American English"`.
    fn variety_name(&self) -> &str;

    /// Returns `true` if `phone` is a nucleus (vowel or syllabic consonant)
    /// in this variety.
    fn is_nucleus(&self, phone: &Phone) -> bool;

    /// Returns a detailed verdict for whether `cluster` is a legal onset.
    ///
    /// An empty cluster is always legal (null onset).
    fn onset_verdict(&self, cluster: &[&Phone]) -> OnsetVerdict;

    /// Returns `true` if `cluster` is a legal onset.
    ///
    /// Convenience wrapper around [`onset_verdict`][Self::onset_verdict].
    fn is_legal_onset(&self, cluster: &[&Phone]) -> bool {
        cluster.is_empty() || self.onset_verdict(cluster).is_legal
    }

    /// Returns `true` if `cluster` is a legal coda.
    ///
    /// An empty cluster is always legal.
    fn is_legal_coda(&self, cluster: &[&Phone]) -> bool;
}

// ─── English phonotactics ─────────────────────────────────────────────────────

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
        // IPA values from default_phone_for_arpabet in linguistic/phonology.rs.
        let nuclei: Vec<Phone> = [
            "ɑ", "æ", "ʌ", "ɔ", "ɛ", "ɝ", "ɪ", "iː", "ʊ", "uː",
            "aʊ", "aɪ", "eɪ", "oʊ", "ɔɪ",
        ]
        .iter()
        .map(|s| Phone::mapped(*s))
        .collect();

        // ── Illegal simple onsets ─────────────────────────────────────────────
        let illegal_single_onsets: Vec<Phone> = ["ŋ"].iter().map(|s| Phone::mapped(*s)).collect();

        // ── Legal multi-phone onset clusters ──────────────────────────────────
        let ps = |syms: &[&str]| -> PhoneString {
            PhoneString {
                phones: syms.iter().map(|s| Phone::mapped(*s)).collect(),
            }
        };

        let mut legal_onset_clusters: Vec<PhoneString> = vec![
            // ── Stop / fricative + lateral ───────────────────────────────────
            ps(&["p", "l"]),
            ps(&["b", "l"]),
            ps(&["k", "l"]),
            ps(&["ɡ", "l"]),
            ps(&["f", "l"]),
            // ── Stop / fricative + rhotic ────────────────────────────────────
            ps(&["p", "ɹ"]),
            ps(&["b", "ɹ"]),
            ps(&["t", "ɹ"]),
            ps(&["d", "ɹ"]),
            ps(&["k", "ɹ"]),
            ps(&["ɡ", "ɹ"]),
            ps(&["f", "ɹ"]),
            ps(&["θ", "ɹ"]),
            ps(&["ʃ", "ɹ"]),
            // ── /s/ + obstruent / sonorant ────────────────────────────────────
            ps(&["s", "p"]),
            ps(&["s", "t"]),
            ps(&["s", "k"]),
            ps(&["s", "l"]),
            ps(&["s", "m"]),
            ps(&["s", "n"]),
            ps(&["s", "w"]),
            ps(&["s", "f"]),
            // ── Stop / fricative + glide /w/ ──────────────────────────────────
            ps(&["t", "w"]),
            ps(&["k", "w"]),
            ps(&["ɡ", "w"]),
            ps(&["d", "w"]),
            ps(&["ʃ", "w"]),
            ps(&["θ", "w"]),
            // ── Three-phone clusters ──────────────────────────────────────────
            ps(&["s", "p", "l"]),
            ps(&["s", "p", "ɹ"]),
            ps(&["s", "t", "ɹ"]),
            ps(&["s", "k", "ɹ"]),
            ps(&["s", "k", "w"]),
            ps(&["s", "t", "w"]),
        ];

        // ── Variety-specific additions ────────────────────────────────────────
        if variety == EnglishVariety::PermissiveSinging {
            legal_onset_clusters.extend([
                ps(&["t", "l"]),
                ps(&["d", "l"]),
                ps(&["v", "ɹ"]),
                ps(&["v", "l"]),
                ps(&["z", "w"]),
            ]);
        }

        // ── Legal coda clusters ───────────────────────────────────────────────
        let legal_coda_clusters: Vec<PhoneString> = vec![
            // Two-phone
            ps(&["n", "d"]),
            ps(&["n", "t"]),
            ps(&["n", "z"]),
            ps(&["ŋ", "k"]),
            ps(&["ŋ", "z"]),
            ps(&["m", "p"]),
            ps(&["m", "z"]),
            ps(&["l", "d"]),
            ps(&["l", "t"]),
            ps(&["l", "k"]),
            ps(&["l", "p"]),
            ps(&["l", "f"]),
            ps(&["l", "m"]),
            ps(&["l", "n"]),
            ps(&["l", "z"]),
            ps(&["s", "t"]),
            ps(&["s", "k"]),
            ps(&["s", "p"]),
            ps(&["f", "t"]),
            ps(&["k", "t"]),
            ps(&["k", "s"]),
            ps(&["p", "t"]),
            ps(&["p", "s"]),
            ps(&["t", "s"]),
            ps(&["d", "z"]),
            ps(&["ɹ", "d"]),
            ps(&["ɹ", "t"]),
            ps(&["ɹ", "k"]),
            ps(&["ɹ", "n"]),
            ps(&["ɹ", "m"]),
            ps(&["ɹ", "z"]),
            ps(&["ɹ", "p"]),
            ps(&["ɹ", "f"]),
            ps(&["n", "tʃ"]),
            ps(&["n", "dʒ"]),
            ps(&["l", "tʃ"]),
            ps(&["ɹ", "tʃ"]),
            // Three-phone
            ps(&["n", "d", "z"]),
            ps(&["n", "t", "s"]),
            ps(&["ŋ", "k", "s"]),
            ps(&["l", "d", "z"]),
            ps(&["l", "t", "s"]),
            ps(&["l", "k", "s"]),
            ps(&["m", "p", "t"]),
            ps(&["m", "p", "s"]),
            ps(&["s", "t", "s"]),
            ps(&["k", "t", "s"]),
            // -ngths (e.g. "lengths", "strengths")
            ps(&["ŋ", "θ", "s"]),
            ps(&["ŋ", "k", "θ", "s"]),
        ];

        Self {
            variety,
            inventory,
            nuclei,
            illegal_single_onsets,
            legal_onset_clusters,
            legal_coda_clusters,
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
            if self.illegal_single_onsets.iter().any(|p| self.phone_matches(phone, p)) {
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
                    reason: format!("/{ipa}/ is a nucleus, not an onset consonant", ipa = phone.ipa),
                };
            }
            return OnsetVerdict {
                cluster: phones,
                is_legal: true,
                reason: format!("/{ipa}/ is a legal simple onset", ipa = phone.ipa),
            };
        }

        // Multi-phone cluster: check legal onset cluster table.
        let is_legal = self.legal_onset_clusters.iter().any(|ps| self.cluster_matches(cluster, ps));
        let ipa_cluster: String = cluster.iter().map(|p| p.ipa.as_str()).collect();
        let slash_cluster = format!("/{ipa_cluster}/");
        let reason = if is_legal {
            format!("{slash_cluster} is a legal onset cluster in {}", self.variety_name())
        } else {
            format!("{slash_cluster} is not a legal onset cluster in {}", self.variety_name())
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
            return !self.nuclei.iter().any(|n| self.phone_matches(cluster[0], n));
        }
        self.legal_coda_clusters.iter().any(|ps| self.cluster_matches(cluster, ps))
    }
}

// ─── Permissive mock profile ──────────────────────────────────────────────────

/// A maximally permissive phonotactic profile that accepts every non-empty
/// onset and coda cluster.
///
/// Useful in unit tests to exercise the syllabifier algorithm independently
/// of phonotactic constraints.
pub struct PermissiveProfile;

impl PhonotacticProfile for PermissiveProfile {
    fn variety_name(&self) -> &str {
        "Permissive (test)"
    }

    fn is_nucleus(&self, phone: &Phone) -> bool {
        matches!(
            phone.ipa.as_str(),
            "ɑ" | "æ"
                | "ʌ"
                | "ɔ"
                | "aʊ"
                | "aɪ"
                | "ɛ"
                | "ɝ"
                | "eɪ"
                | "ɪ"
                | "iː"
                | "oʊ"
                | "ɔɪ"
                | "ʊ"
                | "uː"
        )
    }

    fn onset_verdict(&self, cluster: &[&Phone]) -> OnsetVerdict {
        OnsetVerdict {
            cluster: cluster.iter().map(|p| (*p).clone()).collect(),
            is_legal: true,
            reason: "permissive profile accepts all onsets".into(),
        }
    }

    fn is_legal_coda(&self, _cluster: &[&Phone]) -> bool {
        true
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linguistic::phonology::{PhoneComparisonMode, PhoneEqualityOptions, phones_equivalent};

    fn ga() -> EnglishPhonotactics {
        EnglishPhonotactics::for_variety(EnglishVariety::GeneralAmerican)
    }

    fn singing() -> EnglishPhonotactics {
        EnglishPhonotactics::for_variety(EnglishVariety::PermissiveSinging)
    }

    /// Build a `Vec<Phone>` from IPA strings — for collecting into `&[&Phone]`.
    fn phones(ipas: &[&str]) -> Vec<Phone> {
        ipas.iter().map(|s| Phone::mapped(*s)).collect()
    }

    /// Collect owned phones into a `Vec<&Phone>` slice for trait calls.
    fn refs(v: &[Phone]) -> Vec<&Phone> {
        v.iter().collect()
    }

    // ── Nucleus detection ────────────────────────────────────────────────────

    #[test]
    fn ipa_vowels_are_nuclei() {
        let p = ga();
        for v in &["ɑ", "æ", "ʌ", "ɔ", "ɛ", "ɝ", "ɪ", "iː", "ʊ", "uː"] {
            assert!(p.is_nucleus(&Phone::mapped(*v)), "/{v}/ should be a nucleus");
        }
    }

    #[test]
    fn ipa_diphthongs_are_nuclei() {
        let p = ga();
        for v in &["aʊ", "aɪ", "eɪ", "oʊ", "ɔɪ"] {
            assert!(p.is_nucleus(&Phone::mapped(*v)), "/{v}/ should be a nucleus");
        }
    }

    #[test]
    fn ipa_consonants_are_not_nuclei() {
        let p = ga();
        for c in &["t", "k", "s", "n", "m", "l", "ɹ", "p", "b", "ŋ"] {
            assert!(!p.is_nucleus(&Phone::mapped(*c)), "/{c}/ should not be a nucleus");
        }
    }

    // ── Null onset ───────────────────────────────────────────────────────────

    #[test]
    fn null_onset_is_always_legal() {
        assert!(ga().is_legal_onset(&[]));
    }

    // ── Simple legal onsets ──────────────────────────────────────────────────

    #[test]
    fn simple_consonants_are_legal_onsets() {
        let p = ga();
        for c in &["t", "s", "p", "b", "k", "ɡ", "m", "n", "l", "ɹ", "f", "v"] {
            let v = phones(&[c]);
            assert!(p.is_legal_onset(&refs(&v)), "/{c}/ should be a legal onset");
        }
    }

    #[test]
    fn affricates_are_legal_simple_onsets() {
        let p = ga();
        let tsh = phones(&["tʃ"]);
        let dzh = phones(&["dʒ"]);
        assert!(p.is_legal_onset(&refs(&tsh)), "/tʃ/ should be legal");
        assert!(p.is_legal_onset(&refs(&dzh)), "/dʒ/ should be legal");
    }

    // ── Illegal simple onsets ────────────────────────────────────────────────

    #[test]
    fn velar_nasal_is_not_a_legal_onset() {
        let ng = phones(&["ŋ"]);
        assert!(!ga().is_legal_onset(&refs(&ng)));
    }

    // ── Two-phone legal clusters ─────────────────────────────────────────────

    #[test]
    fn stop_lateral_clusters_are_legal() {
        let p = ga();
        for pair in &[["p", "l"], ["b", "l"], ["k", "l"], ["ɡ", "l"], ["f", "l"]] {
            let v = phones(pair);
            assert!(p.is_legal_onset(&refs(&v)), "/{}/{} should be legal", pair[0], pair[1]);
        }
    }

    #[test]
    fn stop_rhotic_clusters_are_legal() {
        let p = ga();
        for pair in &[
            ["p", "ɹ"], ["b", "ɹ"], ["t", "ɹ"], ["d", "ɹ"],
            ["k", "ɹ"], ["ɡ", "ɹ"], ["f", "ɹ"], ["θ", "ɹ"], ["ʃ", "ɹ"],
        ] {
            let v = phones(pair);
            assert!(p.is_legal_onset(&refs(&v)), "/{}/{} should be legal", pair[0], pair[1]);
        }
    }

    #[test]
    fn s_clusters_are_legal_onsets() {
        let p = ga();
        for pair in &[["s", "p"], ["s", "t"], ["s", "k"], ["s", "l"], ["s", "m"], ["s", "n"], ["s", "w"]] {
            let v = phones(pair);
            assert!(p.is_legal_onset(&refs(&v)));
        }
    }

    #[test]
    fn tw_and_kw_are_legal_onsets() {
        let p = ga();
        let tw = phones(&["t", "w"]);
        let kw = phones(&["k", "w"]);
        assert!(p.is_legal_onset(&refs(&tw)));
        assert!(p.is_legal_onset(&refs(&kw)));
    }

    // ── Illegal two-phone clusters ───────────────────────────────────────────

    #[test]
    fn tl_is_not_legal_in_general_american() {
        let tl = phones(&["t", "l"]);
        assert!(!ga().is_legal_onset(&refs(&tl)));
    }

    #[test]
    fn dl_is_not_legal_in_general_american() {
        let dl = phones(&["d", "l"]);
        assert!(!ga().is_legal_onset(&refs(&dl)));
    }

    // ── Three-phone clusters ─────────────────────────────────────────────────

    #[test]
    fn three_phone_s_clusters_are_legal() {
        let p = ga();
        for triple in &[
            ["s", "t", "ɹ"], ["s", "p", "ɹ"], ["s", "k", "ɹ"],
            ["s", "p", "l"], ["s", "k", "w"],
        ] {
            let v = phones(triple);
            assert!(p.is_legal_onset(&refs(&v)), "/{}{}{} should be legal", triple[0], triple[1], triple[2]);
        }
    }

    // ── Variety-specific differences ─────────────────────────────────────────

    #[test]
    fn permissive_singing_allows_tl() {
        let tl = phones(&["t", "l"]);
        assert!(singing().is_legal_onset(&refs(&tl)));
    }

    #[test]
    fn permissive_singing_allows_dl() {
        let dl = phones(&["d", "l"]);
        assert!(singing().is_legal_onset(&refs(&dl)));
    }

    #[test]
    fn general_american_rejects_what_singing_allows() {
        let tl = phones(&["t", "l"]);
        assert!(!ga().is_legal_onset(&refs(&tl)));
        assert!(singing().is_legal_onset(&refs(&tl)));
    }

    // ── Onset verdict diagnostics ────────────────────────────────────────────

    #[test]
    fn rejected_verdict_message_cites_ipa_cluster() {
        let tl = phones(&["t", "l"]);
        let v = ga().onset_verdict(&refs(&tl));
        assert!(!v.is_legal);
        assert!(v.reason.contains("tl"), "expected /tl/ in reason, got: {}", v.reason);
    }

    #[test]
    fn accepted_verdict_message_cites_ipa_cluster() {
        let str = phones(&["s", "t", "ɹ"]);
        let v = ga().onset_verdict(&refs(&str));
        assert!(v.is_legal);
        assert!(v.reason.contains("stɹ"), "expected /stɹ/ in reason, got: {}", v.reason);
    }

    #[test]
    fn verdict_as_diagnostic_uses_correct_kind() {
        let tl = phones(&["t", "l"]);
        let v = ga().onset_verdict(&refs(&tl));
        let d = v.as_diagnostic();
        assert_eq!(d.kind, crate::prosody::syllable::DiagnosticKind::RejectedOnset);
    }

    #[test]
    fn cluster_ipa_concatenates_phone_ipas() {
        let str = phones(&["s", "t", "ɹ"]);
        let v = ga().onset_verdict(&refs(&str));
        assert_eq!(v.cluster_ipa(), "stɹ");
    }

    // ── Coda legality ────────────────────────────────────────────────────────

    #[test]
    fn empty_coda_is_legal() {
        assert!(ga().is_legal_coda(&[]));
    }

    #[test]
    fn single_consonant_coda_is_legal() {
        for c in &["k", "n", "ŋ"] {
            let v = phones(&[c]);
            assert!(ga().is_legal_coda(&refs(&v)));
        }
    }

    #[test]
    fn known_coda_clusters_are_legal() {
        let p = ga();
        let nd = phones(&["n", "d"]);
        let ngk = phones(&["ŋ", "k"]);
        let ngths = phones(&["ŋ", "θ", "s"]);
        assert!(p.is_legal_coda(&refs(&nd)));
        assert!(p.is_legal_coda(&refs(&ngk)));
        assert!(p.is_legal_coda(&refs(&ngths)));
    }

    // ── Phone equality ───────────────────────────────────────────────────────

    #[test]
    fn exact_phone_equality_distinguishes_aspiration() {
        let t  = Phone::mapped("t");
        let th = Phone::mapped("tʰ");
        assert!(!phones_equivalent(&t, &th, &PhoneEqualityOptions::default()));
    }

    #[test]
    fn broad_phone_equality_ignores_aspiration() {
        let t  = Phone::mapped("t");
        let th = Phone::mapped("tʰ");
        let broad = PhoneEqualityOptions {
            mode: PhoneComparisonMode::Broad,
            ignore_diacritics: true,
            ..Default::default()
        };
        assert!(phones_equivalent(&t, &th, &broad));
    }

    #[test]
    fn aspirated_t_r_counts_as_legal_tr_onset_in_broad_profile() {
        // PermissiveSinging uses broad comparison with ignore_diacritics.
        let singing = EnglishPhonotactics::for_variety(EnglishVariety::PermissiveSinging);
        let t_asp = Phone::mapped("tʰ");
        let r     = Phone::mapped("ɹ");
        assert!(singing.is_legal_onset(&[&t_asp, &r]));
    }

    #[test]
    fn tap_is_not_t_by_broad_equality() {
        // /ɾ/ is an allophone of /t/ but is a different segment — not just a
        // diacritic variant — so broad equality must NOT conflate them.
        let t   = Phone::mapped("t");
        let tap = Phone::mapped("ɾ");
        let broad = PhoneEqualityOptions {
            mode: PhoneComparisonMode::Broad,
            ignore_diacritics: true,
            ..Default::default()
        };
        assert!(!phones_equivalent(&t, &tap, &broad));
    }

    #[test]
    fn exact_mode_ignores_no_flags() {
        // Even with all flags set, ExactIpa ignores them all.
        let t  = Phone::mapped("t");
        let th = Phone::mapped("tʰ");
        let opts = PhoneEqualityOptions {
            mode: PhoneComparisonMode::ExactIpa,
            ignore_diacritics: true,
            ignore_length: true,
            ..Default::default()
        };
        assert!(!phones_equivalent(&t, &th, &opts));
    }

    // ── Permissive profile ───────────────────────────────────────────────────

    #[test]
    fn permissive_profile_accepts_any_onset() {
        let p = PermissiveProfile;
        let tl = phones(&["t", "l"]);
        let ng = phones(&["ŋ"]);
        assert!(p.is_legal_onset(&refs(&tl)));
        assert!(p.is_legal_onset(&refs(&ng)));
    }

    #[test]
    fn permissive_profile_accepts_any_coda() {
        let tlk = phones(&["t", "l", "k"]);
        assert!(PermissiveProfile.is_legal_coda(&refs(&tlk)));
    }
}
