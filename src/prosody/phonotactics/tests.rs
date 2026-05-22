use super::*;
use crate::linguistic::phonology::{
    Phone, PhoneComparisonMode, PhoneEqualityOptions, phones_equivalent,
};

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
        assert!(
            p.is_nucleus(&Phone::mapped(*v)),
            "/{v}/ should be a nucleus"
        );
    }
}

#[test]
fn ipa_diphthongs_are_nuclei() {
    let p = ga();
    for v in &["aʊ", "aɪ", "eɪ", "oʊ", "ɔɪ"] {
        assert!(
            p.is_nucleus(&Phone::mapped(*v)),
            "/{v}/ should be a nucleus"
        );
    }
}

#[test]
fn ipa_consonants_are_not_nuclei() {
    let p = ga();
    for c in &["t", "k", "s", "n", "m", "l", "ɹ", "p", "b", "ŋ"] {
        assert!(
            !p.is_nucleus(&Phone::mapped(*c)),
            "/{c}/ should not be a nucleus"
        );
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
        assert!(
            p.is_legal_onset(&refs(&v)),
            "/{}/{} should be legal",
            pair[0],
            pair[1]
        );
    }
}

#[test]
fn stop_rhotic_clusters_are_legal() {
    let p = ga();
    for pair in &[
        ["p", "ɹ"],
        ["b", "ɹ"],
        ["t", "ɹ"],
        ["d", "ɹ"],
        ["k", "ɹ"],
        ["ɡ", "ɹ"],
        ["f", "ɹ"],
        ["θ", "ɹ"],
        ["ʃ", "ɹ"],
    ] {
        let v = phones(pair);
        assert!(
            p.is_legal_onset(&refs(&v)),
            "/{}/{} should be legal",
            pair[0],
            pair[1]
        );
    }
}

#[test]
fn s_clusters_are_legal_onsets() {
    let p = ga();
    for pair in &[
        ["s", "p"],
        ["s", "t"],
        ["s", "k"],
        ["s", "l"],
        ["s", "m"],
        ["s", "n"],
        ["s", "w"],
    ] {
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
        ["s", "t", "ɹ"],
        ["s", "p", "ɹ"],
        ["s", "k", "ɹ"],
        ["s", "p", "l"],
        ["s", "k", "w"],
    ] {
        let v = phones(triple);
        assert!(
            p.is_legal_onset(&refs(&v)),
            "/{}{}{} should be legal",
            triple[0],
            triple[1],
            triple[2]
        );
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
    assert!(
        v.reason.contains("tl"),
        "expected /tl/ in reason, got: {}",
        v.reason
    );
}

#[test]
fn accepted_verdict_message_cites_ipa_cluster() {
    let str = phones(&["s", "t", "ɹ"]);
    let v = ga().onset_verdict(&refs(&str));
    assert!(v.is_legal);
    assert!(
        v.reason.contains("stɹ"),
        "expected /stɹ/ in reason, got: {}",
        v.reason
    );
}

#[test]
fn verdict_as_diagnostic_uses_correct_kind() {
    let tl = phones(&["t", "l"]);
    let v = ga().onset_verdict(&refs(&tl));
    let d = v.as_diagnostic();
    assert_eq!(
        d.kind,
        crate::prosody::syllable::DiagnosticKind::RejectedOnset
    );
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
    let ntsh = phones(&["n", "tʃ"]);
    let ndzh = phones(&["n", "dʒ"]);
    assert!(p.is_legal_coda(&refs(&nd)));
    assert!(p.is_legal_coda(&refs(&ngk)));
    assert!(p.is_legal_coda(&refs(&ngths)));
    assert!(p.is_legal_coda(&refs(&ntsh)));
    assert!(p.is_legal_coda(&refs(&ndzh)));
}

// ── Phone equality ───────────────────────────────────────────────────────

#[test]
fn exact_phone_equality_distinguishes_aspiration() {
    let t = Phone::mapped("t");
    let th = Phone::mapped("tʰ");
    assert!(!phones_equivalent(
        &t,
        &th,
        &PhoneEqualityOptions::default()
    ));
}

#[test]
fn broad_phone_equality_ignores_aspiration() {
    let t = Phone::mapped("t");
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
    let r = Phone::mapped("ɹ");
    assert!(singing.is_legal_onset(&[&t_asp, &r]));
}

#[test]
fn tap_is_not_t_by_broad_equality() {
    // /ɾ/ is an allophone of /t/ but is a different segment — not just a
    // diacritic variant — so broad equality must NOT conflate them.
    let t = Phone::mapped("t");
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
    let t = Phone::mapped("t");
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
