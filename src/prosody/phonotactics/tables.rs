use crate::linguistic::phonology::{Phone, PhoneString};

pub(crate) fn illegal_single_onsets() -> Vec<Phone> {
    ["ŋ"].iter().map(|s| Phone::mapped(*s)).collect()
}

pub(crate) fn legal_onset_clusters() -> Vec<PhoneString> {
    let ps = |syms: &[&str]| -> PhoneString {
        PhoneString {
            phones: syms.iter().map(|s| Phone::mapped(*s)).collect(),
        }
    };

    vec![
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
    ]
}

pub(crate) fn permissive_singing_onset_additions() -> Vec<PhoneString> {
    let ps = |syms: &[&str]| -> PhoneString {
        PhoneString {
            phones: syms.iter().map(|s| Phone::mapped(*s)).collect(),
        }
    };
    vec![
        ps(&["t", "l"]),
        ps(&["d", "l"]),
        ps(&["v", "ɹ"]),
        ps(&["v", "l"]),
        ps(&["z", "w"]),
    ]
}

pub(crate) fn legal_coda_clusters() -> Vec<PhoneString> {
    let ps = |syms: &[&str]| -> PhoneString {
        PhoneString {
            phones: syms.iter().map(|s| Phone::mapped(*s)).collect(),
        }
    };

    vec![
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
    ]
}
