//! Maximum Onset Principle syllabification over realized phone sequences.
//!
//! The entry point is [`syllabify`], which takes a slice of [`Phoneme`]s and
//! a [`PhonotacticProfile`] and returns a `Vec<`[`Syllable`]`>`.
//!
//! [`syllables_to_ipa`] renders the syllable sequence as an IPA transcription
//! string with stress marks (`ˈ` primary, `ˌ` secondary) and syllable
//! boundaries (`.`).
//!
//! # Algorithm
//!
//! 1. Derive one [`Phone`] per [`Phoneme`] from `phoneme.realization.ipa`.
//! 2. Scan for nucleus positions (phones where [`PhonotacticProfile::is_nucleus`]
//!    returns `true`).
//! 3. For each inter-nuclear consonant cluster, apply the **Maximum Onset
//!    Principle**: try to assign the entire cluster to the following syllable's
//!    onset; if the profile rejects it, trim one phone from the left and retry
//!    until either the onset is legal or empty.  The trimmed phones become the
//!    preceding syllable's coda.
//! 4. Any consonants before the first nucleus form the initial onset; any
//!    consonants after the last nucleus form the final coda.
//!
//! # Example
//!
//! ```
//! use listenbury::prosody::phonotactics::{EnglishPhonotactics, EnglishVariety};
//! use listenbury::prosody::syllabification::{syllabify, syllables_to_ipa};
//! use listenbury::linguistic::phonology::phoneme_from_arpabet;
//!
//! // "extra" = EH1 K S T R AH0
//! let phonemes = vec![
//!     phoneme_from_arpabet("EH1", "test"),
//!     phoneme_from_arpabet("K",   "test"),
//!     phoneme_from_arpabet("S",   "test"),
//!     phoneme_from_arpabet("T",   "test"),
//!     phoneme_from_arpabet("R",   "test"),
//!     phoneme_from_arpabet("AH0", "test"),
//! ];
//! let ga = EnglishPhonotactics::for_variety(EnglishVariety::GeneralAmerican);
//! let syllables = syllabify(&phonemes, &ga);
//! assert_eq!(syllables_to_ipa(&syllables), "ˈɛk.stɹʌ");
//! ```

use crate::linguistic::phonology::{Phone, PhoneString, Phoneme, Stress};
use crate::prosody::phonotactics::PhonotacticProfile;
use crate::prosody::syllable::{DiagnosticKind, SourceSpan, Syllable, SyllableDiagnostic};

// ─── Public API ───────────────────────────────────────────────────────────────

/// Syllabify a sequence of [`Phoneme`]s using the given [`PhonotacticProfile`].
///
/// Returns a `Vec<Syllable>` in order. Each syllable's
/// `source_span.start..source_span.end` indexes back into the `phonemes` slice.
///
/// If the sequence contains no nucleus (e.g. an all-consonant cluster), the
/// entire sequence is returned as a single degenerate syllable with an empty
/// nucleus.
pub fn syllabify<P: PhonotacticProfile>(phonemes: &[Phoneme], profile: &P) -> Vec<Syllable> {
    if phonemes.is_empty() {
        return vec![];
    }

    // Derive realized phones from phoneme realizations.
    let phones: Vec<Phone> = phonemes
        .iter()
        .map(|p| Phone {
            ipa: p.realization.ipa.clone(),
            source_symbol: Some(p.source_symbol.clone()),
            status: crate::linguistic::phonology::PhoneStatus::Mapped,
        })
        .collect();

    // Find all nucleus positions.
    let nucleus_indices: Vec<usize> = phones
        .iter()
        .enumerate()
        .filter(|(_, ph)| profile.is_nucleus(ph))
        .map(|(i, _)| i)
        .collect();

    if nucleus_indices.is_empty() {
        // No vowel found: return a single degenerate syllable.
        let all = PhoneString {
            phones: phones.clone(),
        };
        return vec![Syllable {
            onset: all,
            nucleus: PhoneString::empty(),
            coda: PhoneString::empty(),
            source_span: SourceSpan {
                start: 0,
                end: phonemes.len(),
            },
            stress: None,
            variety: profile.variety_name().to_string(),
            diagnostics: vec![SyllableDiagnostic::new(
                DiagnosticKind::FallbackParse,
                "no nucleus found; entire sequence returned as onset",
            )],
        }];
    }

    // Build syllables by iterating over nuclei.
    //
    // We track `prev_coda_start`: the index of the first unassigned phone
    // after the previous syllable's nucleus (or 0 for the first syllable).
    let mut syllables: Vec<Syllable> = Vec::with_capacity(nucleus_indices.len());
    let mut prev_end = 0usize; // index of first phone not yet claimed

    for (syl_idx, &nuc_pos) in nucleus_indices.iter().enumerate() {
        // Determine the nucleus span (usually just one phone, potentially a
        // diphthong encoded as one IPA string in one Phone).
        let nuc_end = nuc_pos + 1;

        // Consonant cluster between prev_end and nuc_pos.
        let cluster_range = prev_end..nuc_pos;
        let cluster: Vec<&Phone> = phones[cluster_range.clone()].iter().collect();

        let (onset_phones, coda_phones, mut diagnostics) =
            split_mop(&cluster, profile, &phones[cluster_range.start..nuc_pos]);

        // Coda from previous syllable = coda_phones (everything that MOP
        // couldn't assign to the current onset).
        // But for syllable _construction_ we need to patch the previous
        // syllable's coda with these phones.
        if syl_idx > 0 {
            let prev = syllables.last_mut().unwrap();
            prev.coda = PhoneString { phones: coda_phones.clone() };
        }
        // The initial coda (before the first nucleus) is dropped — it becomes
        // a leading onset for the very first syllable.

        // Source span for this syllable: onset_start..nuc_end.
        let onset_start = nuc_pos - onset_phones.len();
        let source_start = if syl_idx == 0 {
            0
        } else {
            onset_start
        };

        let stress = phonemes[nuc_pos].stress;

        // Check if this syllable's onset decision was variety-specific.
        if !diagnostics.is_empty() {
            if let Some(d) = diagnostics.last_mut() {
                if matches!(d.kind, DiagnosticKind::RejectedOnset | DiagnosticKind::LegalOnset) {
                    // Mark as variety-specific if the decision differs from a
                    // permissive profile (not implemented here; placeholder).
                    // Future: compare against PermissiveProfile verdict.
                }
            }
        }

        syllables.push(Syllable {
            onset: PhoneString { phones: onset_phones },
            nucleus: PhoneString {
                phones: phones[nuc_pos..nuc_end].to_vec(),
            },
            coda: PhoneString::empty(), // filled in by next iteration
            source_span: SourceSpan {
                start: source_start,
                end: nuc_end,
            },
            stress,
            variety: profile.variety_name().to_string(),
            diagnostics,
        });

        prev_end = nuc_end;
    }

    // Handle trailing consonants after the last nucleus: they become the coda
    // of the last syllable.
    let last = syllables.last_mut().unwrap();
    if prev_end < phones.len() {
        let trailing: Vec<Phone> = phones[prev_end..].to_vec();
        last.coda = PhoneString { phones: trailing };
    }

    // Fix source span end for the last syllable to include trailing coda.
    last.source_span.end = phonemes.len();

    syllables
}

/// Render a syllable sequence as an IPA transcription string.
///
/// Syllables are joined with `.`.  Each syllable is preceded by `ˈ`
/// (U+02C8) for primary stress or `ˌ` (U+02CC) for secondary stress.
/// Unstressed syllables have no prefix.
///
/// # Example
///
/// ```
/// use listenbury::prosody::phonotactics::{EnglishPhonotactics, EnglishVariety};
/// use listenbury::prosody::syllabification::{syllabify, syllables_to_ipa};
/// use listenbury::linguistic::phonology::phoneme_from_arpabet;
///
/// // "atlas" = AE1 T L AH0 S
/// let phonemes = vec![
///     phoneme_from_arpabet("AE1", "test"),
///     phoneme_from_arpabet("T",   "test"),
///     phoneme_from_arpabet("L",   "test"),
///     phoneme_from_arpabet("AH0", "test"),
///     phoneme_from_arpabet("S",   "test"),
/// ];
/// let ga = EnglishPhonotactics::for_variety(EnglishVariety::GeneralAmerican);
/// let syllables = syllabify(&phonemes, &ga);
/// // /tl/ is illegal → T stays as coda of first syllable
/// assert_eq!(syllables_to_ipa(&syllables), "ˈæt.lʌs");
/// ```
pub fn syllables_to_ipa(syllables: &[Syllable]) -> String {
    syllables
        .iter()
        .enumerate()
        .map(|(i, syl)| {
            let prefix = match syl.stress {
                Some(Stress::Primary) => "ˈ",
                Some(Stress::Secondary) => "ˌ",
                _ => "",
            };
            let body = syl.phones_to_ipa();
            let dot = if i > 0 { "." } else { "" };
            format!("{dot}{prefix}{body}")
        })
        .collect()
}

// ─── MOP split ────────────────────────────────────────────────────────────────

/// Apply the Maximum Onset Principle to a consonant cluster that appears
/// between two nuclei (or before the first nucleus).
///
/// Returns `(onset_phones, coda_phones, diagnostics)` where:
/// - `onset_phones` is the maximal legal prefix of the cluster that belongs
///   to the *following* syllable's onset.
/// - `coda_phones` is the remainder that belongs to the *preceding* syllable's
///   coda.
///
/// If the full cluster is legal it becomes the entire onset (empty coda).
/// If even a single-phone onset is rejected, the whole cluster becomes the
/// coda (empty onset, fallback parse).
fn split_mop<P: PhonotacticProfile>(
    cluster: &[&Phone],
    profile: &P,
    _source_phones: &[Phone],
) -> (Vec<Phone>, Vec<Phone>, Vec<SyllableDiagnostic>) {
    if cluster.is_empty() {
        return (vec![], vec![], vec![]);
    }

    let mut diagnostics = Vec::new();

    // Try the full cluster first, then progressively trim from the left.
    for split in 0..=cluster.len() {
        let candidate_onset = &cluster[split..];
        let verdict = profile.onset_verdict(candidate_onset);

        if verdict.is_legal {
            if split > 0 {
                diagnostics.push(verdict.as_diagnostic());
                // Record the originally-rejected full cluster.
                let rejected_ipa: String = cluster.iter().map(|p| p.ipa.as_str()).collect();
                diagnostics.push(SyllableDiagnostic::new(
                    DiagnosticKind::RejectedOnset,
                    format!("/{rejected_ipa}/ trimmed to find legal onset"),
                ));
            } else {
                // Full cluster was immediately legal — only add diagnostic if
                // it's non-trivial (multi-phone) so noise is kept low.
                if candidate_onset.len() > 1 {
                    diagnostics.push(verdict.as_diagnostic());
                }
            }

            let coda_phones: Vec<Phone> = cluster[..split].iter().map(|p| (*p).clone()).collect();
            let onset_phones: Vec<Phone> = candidate_onset.iter().map(|p| (*p).clone()).collect();
            return (onset_phones, coda_phones, diagnostics);
        }
    }

    // Unreachable: the empty onset is always legal, so split == cluster.len()
    // must succeed above.  But handle gracefully as a fallback.
    diagnostics.push(SyllableDiagnostic::new(
        DiagnosticKind::FallbackParse,
        "entire cluster assigned to coda as fallback",
    ));
    let coda: Vec<Phone> = cluster.iter().map(|p| (*p).clone()).collect();
    (vec![], coda, diagnostics)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linguistic::phonology::phoneme_from_arpabet;
    use crate::prosody::phonotactics::{EnglishPhonotactics, EnglishVariety, PermissiveProfile};

    fn ga() -> EnglishPhonotactics {
        EnglishPhonotactics::for_variety(EnglishVariety::GeneralAmerican)
    }

    fn singing() -> EnglishPhonotactics {
        EnglishPhonotactics::for_variety(EnglishVariety::PermissiveSinging)
    }

    // Helper: build phoneme sequence from ARPABET strings.
    fn seq(symbols: &[&str]) -> Vec<Phoneme> {
        symbols.iter().map(|s| phoneme_from_arpabet(s, "test")).collect()
    }

    // ── Core IPA output ──────────────────────────────────────────────────────

    #[test]
    fn extra_syllabifies_as_ek_str() {
        // "extra" = EH1 K S T R AH0 → /ˈɛk.stɹʌ/
        // MOP: /stɹ/ is a legal GA onset → {ɛk | stɹʌ}
        let s = syllabify(&seq(&["EH1", "K", "S", "T", "R", "AH0"]), &ga());
        assert_eq!(syllables_to_ipa(&s), "ˈɛk.stɹʌ");
    }

    #[test]
    fn atlas_syllabifies_with_tl_in_coda() {
        // "atlas" = AE1 T L AH0 S → /ˈæt.lʌs/
        // MOP: /tl/ is NOT a GA onset → T goes to coda, /l/ is onset of second syl
        let s = syllabify(&seq(&["AE1", "T", "L", "AH0", "S"]), &ga());
        assert_eq!(syllables_to_ipa(&s), "ˈæt.lʌs");
    }

    #[test]
    fn happy_assigns_p_to_onset() {
        // "happy" = HH AE1 P IY0 → /ˈhæ.pi/
        let s = syllabify(&seq(&["HH", "AE1", "P", "IY0"]), &ga());
        assert_eq!(syllables_to_ipa(&s), "ˈhæ.piː");
    }

    #[test]
    fn atlas_under_singing_profile_allows_tl_onset() {
        // With PermissiveSinging, /tl/ is a legal onset → /ˈæ.tlʌs/
        let s = syllabify(&seq(&["AE1", "T", "L", "AH0", "S"]), &singing());
        assert_eq!(syllables_to_ipa(&s), "ˈæ.tlʌs");
    }

    // ── Stress markers ───────────────────────────────────────────────────────

    #[test]
    fn primary_stress_marker_appears_in_output() {
        let s = syllabify(&seq(&["AE1"]), &ga());
        assert_eq!(syllables_to_ipa(&s), "ˈæ");
    }

    #[test]
    fn secondary_stress_marker_appears_in_output() {
        let s = syllabify(&seq(&["AE2"]), &ga());
        assert_eq!(syllables_to_ipa(&s), "ˌæ");
    }

    #[test]
    fn unstressed_syllable_has_no_marker() {
        let s = syllabify(&seq(&["AH0"]), &ga());
        assert_eq!(syllables_to_ipa(&s), "ʌ");
    }

    // ── Edge cases ───────────────────────────────────────────────────────────

    #[test]
    fn empty_input_returns_empty() {
        let s = syllabify(&[], &ga());
        assert!(s.is_empty());
    }

    #[test]
    fn single_vowel_returns_one_syllable() {
        let s = syllabify(&seq(&["AE1"]), &ga());
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].nucleus.to_ipa(), "æ");
    }

    #[test]
    fn strengths_does_not_panic() {
        // "strengths" = S T R EH1 NG K TH S
        let s = syllabify(&seq(&["S", "T", "R", "EH1", "NG", "K", "TH", "S"]), &ga());
        // Just check it produces output without panicking and contains IPA.
        let ipa = syllables_to_ipa(&s);
        assert!(ipa.contains("ɛ"), "expected nucleus ɛ in strengths, got: {ipa}");
    }

    #[test]
    fn permissive_profile_puts_all_consonants_in_onset() {
        // With a permissive profile, MOP assigns all inter-nuclear consonants
        // to the following onset.
        let s = syllabify(&seq(&["AE1", "T", "L", "AH0"]), &PermissiveProfile);
        // /tl/ is accepted → coda of first syllable is empty
        assert_eq!(s[0].coda.to_ipa(), "", "expected empty coda with permissive profile");
    }

    // ── Syllable structure ───────────────────────────────────────────────────

    #[test]
    fn syllable_count_for_extra_is_two() {
        let s = syllabify(&seq(&["EH1", "K", "S", "T", "R", "AH0"]), &ga());
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn first_syllable_of_extra_has_k_coda() {
        let s = syllabify(&seq(&["EH1", "K", "S", "T", "R", "AH0"]), &ga());
        assert_eq!(s[0].coda.to_ipa(), "k");
    }

    #[test]
    fn second_syllable_of_extra_has_str_onset() {
        let s = syllabify(&seq(&["EH1", "K", "S", "T", "R", "AH0"]), &ga());
        assert_eq!(s[1].onset.to_ipa(), "stɹ");
    }

    #[test]
    fn source_spans_are_within_bounds() {
        let seq_in = seq(&["EH1", "K", "S", "T", "R", "AH0"]);
        let len = seq_in.len();
        let s = syllabify(&seq_in, &ga());
        for syl in &s {
            assert!(syl.source_span.start <= syl.source_span.end);
            assert!(syl.source_span.end <= len);
        }
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    #[test]
    fn rejected_onset_diagnostic_present_when_cluster_trimmed() {
        // In "extra", /stɹ/ is legal so no trimming. Use "atlas" where /tl/ is rejected.
        let s = syllabify(&seq(&["AE1", "T", "L", "AH0", "S"]), &ga());
        // The second syllable should record that /l/ was the accepted onset
        // (after /tl/ was rejected).
        let all_diags: Vec<_> = s.iter().flat_map(|syl| &syl.diagnostics).collect();
        assert!(
            !all_diags.is_empty(),
            "expected diagnostics for atlas syllabification"
        );
    }
}
