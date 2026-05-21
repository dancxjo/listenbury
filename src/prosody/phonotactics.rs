//! Phonotactic profiles for IPA-based syllabification.
//!
//! A [`PhonotacticProfile`] encapsulates the rules that determine which IPA
//! phone sequences are legal onsets or codas in a given linguistic variety.
//! The syllabifier queries a profile rather than embedding hard-coded rules,
//! so that the phonotactic assumptions are separable from the algorithm and
//! can be swapped for a different English variety or a completely different
//! language.
//!
//! All phone symbols throughout this module are **IPA strings** as produced
//! by [`crate::linguistic::phonology::Phoneme::realization`] — not ARPABET
//! mnemonics.
//!
//! The primary English implementation is [`EnglishPhonotactics`], constructed
//! with [`EnglishPhonotactics::for_variety`].  A maximally permissive
//! [`PermissiveProfile`] is also provided for testing.
//!
//! # Example
//!
//! ```
//! use listenbury::prosody::phonotactics::{EnglishPhonotactics, EnglishVariety, PhonotacticProfile};
//!
//! let profile = EnglishPhonotactics::for_variety(EnglishVariety::GeneralAmerican);
//!
//! // /stɹ/ is a legal English onset (e.g. "strong", "stream")
//! assert!(profile.is_legal_onset(&["s", "t", "ɹ"]));
//!
//! // /tl/ is not a legal onset in General American English
//! assert!(!profile.is_legal_onset(&["t", "l"]));
//!
//! // /ŋ/ is not a legal word-initial consonant in English
//! assert!(!profile.is_legal_onset(&["ŋ"]));
//!
//! // Vowels are nuclei
//! assert!(profile.is_nucleus("ɛ"));
//! assert!(!profile.is_nucleus("t"));
//! ```

use std::collections::HashSet;

use crate::prosody::syllable::{DiagnosticKind, SyllableDiagnostic};

// ─── Profile trait ────────────────────────────────────────────────────────────

/// Verdict returned when the profile evaluates an onset cluster.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnsetVerdict {
    /// The cluster that was evaluated (IPA strings, one per phone).
    pub cluster: Vec<String>,
    /// Whether this cluster is a legal onset under the active profile.
    pub is_legal: bool,
    /// Human-readable explanation of the decision.
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
}

/// A phonotactic profile that the syllabifier consults to determine the
/// legality of onset and coda clusters.
///
/// Implement this trait to add new variety profiles (Scottish English, learner
/// interlanguage, permissive singing mode, …).
///
/// All `symbol` / `cluster` arguments use **IPA strings**.
pub trait PhonotacticProfile {
    /// Display name of this variety/profile, e.g. `"General American English"`.
    fn variety_name(&self) -> &str;

    /// Returns `true` if `symbol` is a nucleus phone (vowel or syllabic
    /// consonant) in this variety.
    fn is_nucleus(&self, symbol: &str) -> bool;

    /// Returns a detailed verdict for whether `cluster` is a legal onset.
    ///
    /// An empty cluster is always legal (null onset).
    fn onset_verdict(&self, cluster: &[&str]) -> OnsetVerdict;

    /// Returns `true` if `cluster` is a legal onset.
    ///
    /// Convenience wrapper around [`onset_verdict`][Self::onset_verdict].
    fn is_legal_onset(&self, cluster: &[&str]) -> bool {
        cluster.is_empty() || self.onset_verdict(cluster).is_legal
    }

    /// Returns `true` if `cluster` is a legal coda.
    ///
    /// An empty cluster is always legal.
    fn is_legal_coda(&self, cluster: &[&str]) -> bool;
}

// ─── English variety ──────────────────────────────────────────────────────────

/// Which English phonological variety drives the phonotactic decisions.
///
/// Only [`GeneralAmerican`][EnglishVariety::GeneralAmerican] is a full
/// production implementation. The others are listed so that the API makes
/// variety selection explicit and future profiles can be added without an
/// API break.
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

// ─── English phonotactics ─────────────────────────────────────────────────────

/// English phonotactic profile, operating entirely on IPA phone strings.
///
/// Construct with [`EnglishPhonotactics::for_variety`].
///
/// ### IPA symbol conventions
///
/// Phone symbols match the output of
/// [`phoneme_from_arpabet`][crate::linguistic::phonology::phoneme_from_arpabet]:
///
/// | IPA | Segment |
/// |-----|---------|
/// | `ɹ` | rhotic (English /r/) |
/// | `ɡ` | voiced velar stop (U+0261) |
/// | `ŋ` | velar nasal |
/// | `ʃ` | postalveolar fricative |
/// | `ʒ` | voiced postalveolar fricative |
/// | `θ` | voiceless dental fricative |
/// | `ð` | voiced dental fricative |
/// | `tʃ` | affricate (treated as a single segment) |
/// | `dʒ` | affricate (treated as a single segment) |
/// | `ɪ`, `iː`, `ɛ`, `æ`, `ʌ`, `ɑ`, `ɔ`, … | vowels / nuclei |
/// | `aɪ`, `aʊ`, `eɪ`, `oʊ`, `ɔɪ` | diphthongs (single nucleus entry) |
///
/// ### Legal onset clusters (General American, shared base)
///
/// | Two-phone | Three-phone |
/// |-----------|-------------|
/// | pl pɹ bl bɹ tɹ dɹ kl kɹ ɡl ɡɹ | spl spɹ stɹ skɹ skw stw |
/// | fl fɹ θɹ ʃɹ | |
/// | sp st sk sl sm sn sw sf | |
/// | tw kw ɡw dw ʃw θw | |
///
/// `tl` and `dl` are **not** legal onsets in General American English.
/// `ŋ` is not a legal onset in any variety (English has no word-initial /ŋ/).
///
/// `PermissiveSinging` additionally permits `tl`, `dl`, `vɹ`, `vl`, `zw`.
pub struct EnglishPhonotactics {
    variety: EnglishVariety,
    nuclei: HashSet<&'static str>,
    illegal_single_onsets: HashSet<&'static str>,
    legal_onset_clusters: HashSet<Vec<&'static str>>,
    legal_coda_clusters: HashSet<Vec<&'static str>>,
}

impl EnglishPhonotactics {
    /// Construct an English phonotactic profile for `variety`.
    ///
    /// # Example
    ///
    /// ```
    /// use listenbury::prosody::phonotactics::{EnglishPhonotactics, EnglishVariety, PhonotacticProfile};
    ///
    /// let ga = EnglishPhonotactics::for_variety(EnglishVariety::GeneralAmerican);
    /// assert!(ga.is_legal_onset(&["p", "ɹ"]));   // /pɹ/ as in "pretty"
    /// assert!(!ga.is_legal_onset(&["t", "l"]));   // /tl/ illegal in GA
    /// ```
    pub fn for_variety(variety: EnglishVariety) -> Self {
        // ── Nuclei ────────────────────────────────────────────────────────────
        // IPA values from default_phone_for_arpabet in linguistic/phonology.rs.
        let nuclei: HashSet<&'static str> = [
            // Monophthongs
            "ɑ",  // AA
            "æ",  // AE
            "ʌ",  // AH (also reduced ə in many analyses; same symbol here)
            "ɔ",  // AO
            "ɛ",  // EH
            "ɝ",  // ER (rhotacised mid central)
            "ɪ",  // IH
            "iː", // IY
            "ʊ",  // UH
            "uː", // UW
            // Diphthongs (each encoded as a single IPA string)
            "aʊ", // AW
            "aɪ", // AY
            "eɪ", // EY
            "oʊ", // OW
            "ɔɪ", // OY
        ]
        .into_iter()
        .collect();

        // ── Illegal simple onsets ─────────────────────────────────────────────
        // ŋ does not begin a syllable in any standard English variety.
        let illegal_single_onsets: HashSet<&'static str> = ["ŋ"].into_iter().collect();

        // ── Legal multi-phone onset clusters ──────────────────────────────────
        let mut legal_onset_clusters: HashSet<Vec<&'static str>> = [
            // ── Stop / fricative + lateral ──────────────────────────────────
            vec!["p", "l"],
            vec!["b", "l"],
            vec!["k", "l"],
            vec!["ɡ", "l"],
            vec!["f", "l"],
            // ── Stop / fricative + rhotic ────────────────────────────────────
            vec!["p", "ɹ"],
            vec!["b", "ɹ"],
            vec!["t", "ɹ"],
            vec!["d", "ɹ"],
            vec!["k", "ɹ"],
            vec!["ɡ", "ɹ"],
            vec!["f", "ɹ"],
            vec!["θ", "ɹ"],
            vec!["ʃ", "ɹ"],
            // ── /s/ + obstruent / sonorant ───────────────────────────────────
            vec!["s", "p"],
            vec!["s", "t"],
            vec!["s", "k"],
            vec!["s", "l"],
            vec!["s", "m"],
            vec!["s", "n"],
            vec!["s", "w"],
            vec!["s", "f"],
            // ── Stop / fricative + glide /w/ ─────────────────────────────────
            vec!["t", "w"],
            vec!["k", "w"],
            vec!["ɡ", "w"],
            vec!["d", "w"],
            vec!["ʃ", "w"],
            vec!["θ", "w"],
            // ── Three-phone clusters ─────────────────────────────────────────
            vec!["s", "p", "l"],
            vec!["s", "p", "ɹ"],
            vec!["s", "t", "ɹ"],
            vec!["s", "k", "ɹ"],
            vec!["s", "k", "w"],
            vec!["s", "t", "w"],
        ]
        .into_iter()
        .collect();

        // ── Variety-specific additions ────────────────────────────────────────
        if variety == EnglishVariety::PermissiveSinging {
            // Relax phonotactics for sung/poetic contexts.
            legal_onset_clusters.extend([
                vec!["t", "l"],
                vec!["d", "l"],
                vec!["v", "ɹ"],
                vec!["v", "l"],
                vec!["z", "w"],
            ]);
        }

        // ── Legal coda clusters ───────────────────────────────────────────────
        // A representative set sufficient to prevent absurd re-syllabification.
        // Single consonants are always legal as simple codas; only multi-phone
        // clusters need to be enumerated.
        let legal_coda_clusters: HashSet<Vec<&'static str>> = [
            // Two-phone
            vec!["n", "d"],
            vec!["n", "t"],
            vec!["n", "z"],
            vec!["ŋ", "k"],
            vec!["ŋ", "z"],
            vec!["m", "p"],
            vec!["m", "z"],
            vec!["l", "d"],
            vec!["l", "t"],
            vec!["l", "k"],
            vec!["l", "p"],
            vec!["l", "f"],
            vec!["l", "m"],
            vec!["l", "n"],
            vec!["l", "z"],
            vec!["s", "t"],
            vec!["s", "k"],
            vec!["s", "p"],
            vec!["f", "t"],
            vec!["k", "t"],
            vec!["k", "s"],
            vec!["p", "t"],
            vec!["p", "s"],
            vec!["t", "s"],
            vec!["d", "z"],
            vec!["ɹ", "d"],
            vec!["ɹ", "t"],
            vec!["ɹ", "k"],
            vec!["ɹ", "n"],
            vec!["ɹ", "m"],
            vec!["ɹ", "z"],
            vec!["ɹ", "p"],
            vec!["ɹ", "f"],
            vec!["n", "tʃ"],
            vec!["n", "dʒ"],
            vec!["l", "tʃ"],
            vec!["ɹ", "tʃ"],
            // Three-phone
            vec!["n", "d", "z"],
            vec!["n", "t", "s"],
            vec!["ŋ", "k", "s"],
            vec!["l", "d", "z"],
            vec!["l", "t", "s"],
            vec!["l", "k", "s"],
            vec!["m", "p", "t"],
            vec!["m", "p", "s"],
            vec!["s", "t", "s"],
            vec!["k", "t", "s"],
            // -ngths (e.g. "lengths", "strengths")
            vec!["ŋ", "θ", "s"],
            vec!["ŋ", "k", "θ", "s"],
        ]
        .into_iter()
        .collect();

        Self {
            variety,
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
}

impl PhonotacticProfile for EnglishPhonotactics {
    fn variety_name(&self) -> &str {
        match self.variety {
            EnglishVariety::GeneralAmerican => "General American English",
            EnglishVariety::ReceivedPronunciation => "Received Pronunciation",
            EnglishVariety::ScottishEnglish => "Scottish English",
            EnglishVariety::AfricanAmericanEnglish => "African American English",
            EnglishVariety::PermissiveSinging => "Permissive Singing Profile",
        }
    }

    fn is_nucleus(&self, symbol: &str) -> bool {
        self.nuclei.contains(symbol)
    }

    fn onset_verdict(&self, cluster: &[&str]) -> OnsetVerdict {
        let cluster_owned: Vec<String> = cluster.iter().map(|s| s.to_string()).collect();

        if cluster.is_empty() {
            return OnsetVerdict {
                cluster: cluster_owned,
                is_legal: true,
                reason: "null onset is always legal".into(),
            };
        }

        if cluster.len() == 1 {
            let sym = cluster[0];
            if self.illegal_single_onsets.contains(sym) {
                return OnsetVerdict {
                    cluster: cluster_owned,
                    is_legal: false,
                    reason: format!(
                        "/{sym}/ is not a legal syllable onset in {}",
                        self.variety_name()
                    ),
                };
            }
            if self.nuclei.contains(sym) {
                return OnsetVerdict {
                    cluster: cluster_owned,
                    is_legal: false,
                    reason: format!("/{sym}/ is a nucleus, not an onset consonant"),
                };
            }
            return OnsetVerdict {
                cluster: cluster_owned,
                is_legal: true,
                reason: format!("/{sym}/ is a legal simple onset"),
            };
        }

        // Multi-phone cluster: consult the legal cluster table.
        let is_legal = self.legal_onset_clusters.contains(cluster);
        let slash_cluster = format!("/{}/", cluster.join(""));
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
            cluster: cluster_owned,
            is_legal,
            reason,
        }
    }

    fn is_legal_coda(&self, cluster: &[&str]) -> bool {
        if cluster.is_empty() {
            return true;
        }
        if cluster.len() == 1 {
            // Any consonant can be a simple coda in English.
            return !self.nuclei.contains(cluster[0]);
        }
        self.legal_coda_clusters.contains(cluster)
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

    fn is_nucleus(&self, symbol: &str) -> bool {
        matches!(
            symbol,
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

    fn onset_verdict(&self, cluster: &[&str]) -> OnsetVerdict {
        OnsetVerdict {
            cluster: cluster.iter().map(|s| s.to_string()).collect(),
            is_legal: true,
            reason: "permissive profile accepts all onsets".into(),
        }
    }

    fn is_legal_coda(&self, _cluster: &[&str]) -> bool {
        true
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ga() -> EnglishPhonotactics {
        EnglishPhonotactics::for_variety(EnglishVariety::GeneralAmerican)
    }

    fn singing() -> EnglishPhonotactics {
        EnglishPhonotactics::for_variety(EnglishVariety::PermissiveSinging)
    }

    // ── Nucleus detection ────────────────────────────────────────────────────

    #[test]
    fn ipa_vowels_are_nuclei() {
        let p = ga();
        for v in &["ɑ", "æ", "ʌ", "ɔ", "ɛ", "ɝ", "ɪ", "iː", "ʊ", "uː"] {
            assert!(p.is_nucleus(v), "/{v}/ should be a nucleus");
        }
    }

    #[test]
    fn ipa_diphthongs_are_nuclei() {
        let p = ga();
        for v in &["aʊ", "aɪ", "eɪ", "oʊ", "ɔɪ"] {
            assert!(p.is_nucleus(v), "/{v}/ should be a nucleus");
        }
    }

    #[test]
    fn ipa_consonants_are_not_nuclei() {
        let p = ga();
        for c in &["t", "k", "s", "n", "m", "l", "ɹ", "p", "b", "ŋ"] {
            assert!(!p.is_nucleus(c), "/{c}/ should not be a nucleus");
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
            assert!(p.is_legal_onset(&[c]), "/{c}/ should be a legal onset");
        }
    }

    #[test]
    fn affricates_are_legal_simple_onsets() {
        let p = ga();
        assert!(p.is_legal_onset(&["tʃ"]), "/tʃ/ should be legal");
        assert!(p.is_legal_onset(&["dʒ"]), "/dʒ/ should be legal");
    }

    // ── Illegal simple onsets ────────────────────────────────────────────────

    #[test]
    fn velar_nasal_is_not_a_legal_onset() {
        assert!(!ga().is_legal_onset(&["ŋ"]));
    }

    // ── Two-phone legal clusters ─────────────────────────────────────────────

    #[test]
    fn stop_lateral_clusters_are_legal() {
        let p = ga();
        assert!(p.is_legal_onset(&["p", "l"]));  // /pl/ as in "play"
        assert!(p.is_legal_onset(&["b", "l"]));  // /bl/ as in "blue"
        assert!(p.is_legal_onset(&["k", "l"]));  // /kl/ as in "clay"
        assert!(p.is_legal_onset(&["ɡ", "l"]));  // /ɡl/ as in "glad"
        assert!(p.is_legal_onset(&["f", "l"]));  // /fl/ as in "fly"
    }

    #[test]
    fn stop_rhotic_clusters_are_legal() {
        let p = ga();
        assert!(p.is_legal_onset(&["p", "ɹ"]));  // /pɹ/ as in "pray"
        assert!(p.is_legal_onset(&["b", "ɹ"]));  // /bɹ/ as in "bring"
        assert!(p.is_legal_onset(&["t", "ɹ"]));  // /tɹ/ as in "tree"
        assert!(p.is_legal_onset(&["d", "ɹ"]));  // /dɹ/ as in "draw"
        assert!(p.is_legal_onset(&["k", "ɹ"]));  // /kɹ/ as in "cry"
        assert!(p.is_legal_onset(&["ɡ", "ɹ"]));  // /ɡɹ/ as in "grow"
        assert!(p.is_legal_onset(&["f", "ɹ"]));  // /fɹ/ as in "free"
        assert!(p.is_legal_onset(&["θ", "ɹ"]));  // /θɹ/ as in "three"
        assert!(p.is_legal_onset(&["ʃ", "ɹ"]));  // /ʃɹ/ as in "shred"
    }

    #[test]
    fn s_clusters_are_legal_onsets() {
        let p = ga();
        assert!(p.is_legal_onset(&["s", "p"]));
        assert!(p.is_legal_onset(&["s", "t"]));
        assert!(p.is_legal_onset(&["s", "k"]));
        assert!(p.is_legal_onset(&["s", "l"]));
        assert!(p.is_legal_onset(&["s", "m"]));
        assert!(p.is_legal_onset(&["s", "n"]));
        assert!(p.is_legal_onset(&["s", "w"]));
    }

    #[test]
    fn tw_and_kw_are_legal_onsets() {
        let p = ga();
        assert!(p.is_legal_onset(&["t", "w"]));
        assert!(p.is_legal_onset(&["k", "w"]));
    }

    // ── Illegal two-phone clusters ───────────────────────────────────────────

    #[test]
    fn tl_is_not_legal_in_general_american() {
        assert!(!ga().is_legal_onset(&["t", "l"]));
    }

    #[test]
    fn dl_is_not_legal_in_general_american() {
        assert!(!ga().is_legal_onset(&["d", "l"]));
    }

    // ── Three-phone clusters ─────────────────────────────────────────────────

    #[test]
    fn three_phone_s_clusters_are_legal() {
        let p = ga();
        assert!(p.is_legal_onset(&["s", "t", "ɹ"]));  // /stɹ/ as in "strong"
        assert!(p.is_legal_onset(&["s", "p", "ɹ"]));  // /spɹ/ as in "spring"
        assert!(p.is_legal_onset(&["s", "k", "ɹ"]));  // /skɹ/ as in "scream"
        assert!(p.is_legal_onset(&["s", "p", "l"]));  // /spl/ as in "split"
        assert!(p.is_legal_onset(&["s", "k", "w"]));  // /skw/ as in "squeal"
    }

    // ── Variety-specific differences ─────────────────────────────────────────

    #[test]
    fn permissive_singing_allows_tl() {
        assert!(singing().is_legal_onset(&["t", "l"]));
    }

    #[test]
    fn permissive_singing_allows_dl() {
        assert!(singing().is_legal_onset(&["d", "l"]));
    }

    #[test]
    fn general_american_rejects_what_singing_allows() {
        // tl is rejected by GA but accepted by PermissiveSinging.
        assert!(!ga().is_legal_onset(&["t", "l"]));
        assert!(singing().is_legal_onset(&["t", "l"]));
    }

    // ── Onset verdict diagnostics ────────────────────────────────────────────

    #[test]
    fn rejected_verdict_message_cites_ipa_cluster() {
        let v = ga().onset_verdict(&["t", "l"]);
        assert!(!v.is_legal);
        assert!(
            v.reason.contains("tl"),
            "expected /tl/ in reason, got: {}",
            v.reason
        );
    }

    #[test]
    fn accepted_verdict_message_cites_ipa_cluster() {
        let v = ga().onset_verdict(&["s", "t", "ɹ"]);
        assert!(v.is_legal);
        assert!(
            v.reason.contains("stɹ"),
            "expected /stɹ/ in reason, got: {}",
            v.reason
        );
    }

    // ── Coda legality ────────────────────────────────────────────────────────

    #[test]
    fn empty_coda_is_legal() {
        assert!(ga().is_legal_coda(&[]));
    }

    #[test]
    fn simple_coda_consonants_are_legal() {
        let p = ga();
        assert!(p.is_legal_coda(&["t"]));
        assert!(p.is_legal_coda(&["k"]));
        assert!(p.is_legal_coda(&["s"]));
        assert!(p.is_legal_coda(&["ŋ"])); // word-final ŋ is fine (e.g. "ring")
    }

    // ── Variety names ────────────────────────────────────────────────────────

    #[test]
    fn variety_names_are_human_readable() {
        assert_eq!(ga().variety_name(), "General American English");
        assert_eq!(singing().variety_name(), "Permissive Singing Profile");
        assert_eq!(
            EnglishPhonotactics::for_variety(EnglishVariety::ReceivedPronunciation).variety_name(),
            "Received Pronunciation"
        );
    }

    // ── PermissiveProfile ─────────────────────────────────────────────────────

    #[test]
    fn permissive_profile_accepts_any_onset() {
        let p = PermissiveProfile;
        assert!(p.is_legal_onset(&["t", "l"]));
        assert!(p.is_legal_onset(&["ŋ"]));
        assert!(p.is_legal_onset(&["d", "l", "k"]));
    }

    #[test]
    fn permissive_profile_recognises_ipa_vowels() {
        let p = PermissiveProfile;
        assert!(p.is_nucleus("ɛ"));
        assert!(p.is_nucleus("eɪ"));
        assert!(!p.is_nucleus("t"));
    }
}
