use crate::linguistic::phoneme::Phoneme;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stress {
    Primary,
    Secondary,
    Unstressed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Phone {
    pub ipa: String,
    pub source_symbol: Option<String>,
    pub status: PhoneStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhoneStatus {
    Mapped,
    UnknownSymbol,
}

/// Controls how phonemic phones are exposed to downstream timing/rendering code.
///
/// Broad speech planning can keep English diphthongs as one phonemic phone
/// (`/oʊ/`), while singing and low-level acoustic renderers can ask for the
/// internal vowel targets (`[o, ʊ]`) when they need to shape the transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhoneDecompositionPolicy {
    /// Preserve broad phonemic phones: `/oʊ/` remains one phone.
    KeepPhonemic,
    /// Split singable diphthong nuclei into stable vowel + release glide.
    SplitForSinging,
    /// Split renderer-friendly composite targets, including affricates.
    SplitForAcoustics,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhoneString {
    pub phones: Vec<Phone>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RealizedPhone {
    pub phone: Phone,
    pub source_phoneme_index: usize,
    pub source_symbol: String,
    pub stress: Option<Stress>,
}

impl Phone {
    /// Construct a [`Phone`] directly from an IPA string with no source symbol.
    ///
    /// Useful for constructing phones outside the ARPABET lookup path, e.g.
    /// in tests or when working with raw IPA input.
    pub fn new_ipa(ipa: impl Into<String>) -> Self {
        Self {
            ipa: ipa.into(),
            source_symbol: None,
            status: PhoneStatus::Mapped,
        }
    }

    /// Construct a [`Phone`] from a mapped IPA string.  Alias for [`new_ipa`].
    ///
    /// Preferred in phonotactic table construction and test assertions where
    /// the name `mapped` emphasises that the phone has a known IPA symbol.
    pub fn mapped(ipa: impl Into<String>) -> Self {
        Self::new_ipa(ipa)
    }
}

impl PhoneString {
    /// Construct an empty [`PhoneString`].
    pub fn empty() -> Self {
        Self { phones: vec![] }
    }

    /// Concatenate the IPA strings of all contained phones into a single
    /// `String`.
    ///
    /// # Example
    ///
    /// ```
    /// use listenbury::linguistic::phonology::{Phone, PhoneString};
    ///
    /// let ps = PhoneString { phones: vec![
    ///     Phone::new_ipa("s"),
    ///     Phone::new_ipa("t"),
    ///     Phone::new_ipa("ɹ"),
    ///     Phone::new_ipa("ʌ"),
    /// ]};
    /// assert_eq!(ps.to_ipa(), "stɹʌ");
    /// ```
    pub fn to_ipa(&self) -> String {
        self.phones.iter().map(|p| p.ipa.as_str()).collect()
    }

    /// Return the IPA string for each contained phone as a `Vec<&str>`.
    ///
    /// Useful for phonotactic table lookup and diagnostics without allocating
    /// a concatenated string.
    pub fn ipa_segments(&self) -> Vec<&str> {
        self.phones.iter().map(|p| p.ipa.as_str()).collect()
    }

    /// Build a [`PhoneString`] from one [`Phoneme`] by cloning its current
    /// structural realization.
    ///
    /// For a slice of phonemes, use [`from_phoneme_slice`].
    ///
    pub fn from_realized(phoneme: &Phoneme) -> Self {
        phoneme.realization.phone_string.clone()
    }

    /// Build a [`PhoneString`] from a slice of [`Phoneme`]s by expanding each
    /// phoneme's current structural realization.
    ///
    /// This is the primary bridge from a phoneme sequence to the phone
    /// representation used by the syllabifier.
    pub fn from_phoneme_slice(phonemes: &[Phoneme]) -> Self {
        Self {
            phones: RealizedPhone::from_phoneme_slice(phonemes)
                .into_iter()
                .map(|realized| realized.phone)
                .collect(),
        }
    }

    /// Build a [`PhoneString`] from phonemes using an explicit decomposition
    /// policy.
    pub fn from_phoneme_slice_with_policy(
        phonemes: &[Phoneme],
        policy: PhoneDecompositionPolicy,
    ) -> Self {
        Self {
            phones: RealizedPhone::from_phoneme_slice_with_policy(phonemes, policy)
                .into_iter()
                .map(|realized| realized.phone)
                .collect(),
        }
    }
}

impl RealizedPhone {
    /// Build realized phone tokens from one [`Phoneme`], preserving the source
    /// phoneme metadata that would be lost in a bare [`PhoneString`].
    pub fn from_phoneme(index: usize, phoneme: &Phoneme) -> Vec<Self> {
        phoneme
            .realization
            .phone_string
            .phones
            .iter()
            .cloned()
            .map(|phone| Self {
                phone,
                source_phoneme_index: index,
                source_symbol: phoneme.source_symbol.clone(),
                stress: phoneme.stress,
            })
            .collect()
    }

    /// Build realized phone tokens from one [`Phoneme`] using an explicit
    /// decomposition policy.
    pub fn from_phoneme_with_policy(
        index: usize,
        phoneme: &Phoneme,
        policy: PhoneDecompositionPolicy,
    ) -> Vec<Self> {
        let source_symbol = Some(phoneme.source_symbol.clone());
        let phones = match policy {
            PhoneDecompositionPolicy::KeepPhonemic => vec![Phone {
                ipa: phoneme.realization.ipa.clone(),
                source_symbol,
                status: phoneme
                    .realization
                    .phone_string
                    .phones
                    .first()
                    .map(|phone| phone.status)
                    .unwrap_or(PhoneStatus::Mapped),
            }],
            PhoneDecompositionPolicy::SplitForSinging
            | PhoneDecompositionPolicy::SplitForAcoustics => phoneme
                .realization
                .phone_string
                .phones
                .iter()
                .flat_map(|phone| decompose_phone(phone, policy))
                .collect(),
        };

        phones
            .into_iter()
            .map(|phone| Self {
                phone,
                source_phoneme_index: index,
                source_symbol: phoneme.source_symbol.clone(),
                stress: phoneme.stress,
            })
            .collect()
    }

    /// Build realized phone tokens from a phoneme slice.
    pub fn from_phoneme_slice(phonemes: &[Phoneme]) -> Vec<Self> {
        phonemes
            .iter()
            .enumerate()
            .flat_map(|(index, phoneme)| Self::from_phoneme(index, phoneme))
            .collect()
    }

    /// Build realized phone tokens from a phoneme slice using an explicit
    /// decomposition policy.
    pub fn from_phoneme_slice_with_policy(
        phonemes: &[Phoneme],
        policy: PhoneDecompositionPolicy,
    ) -> Vec<Self> {
        phonemes
            .iter()
            .enumerate()
            .flat_map(|(index, phoneme)| Self::from_phoneme_with_policy(index, phoneme, policy))
            .collect()
    }
}

/// Decompose a single phone according to `policy`.
pub fn decompose_phone(phone: &Phone, policy: PhoneDecompositionPolicy) -> Vec<Phone> {
    let segments = match policy {
        PhoneDecompositionPolicy::KeepPhonemic => None,
        PhoneDecompositionPolicy::SplitForSinging => singing_decomposition(phone.ipa.as_str()),
        PhoneDecompositionPolicy::SplitForAcoustics => singing_decomposition(phone.ipa.as_str())
            .or_else(|| acoustic_decomposition(phone.ipa.as_str())),
    };

    match segments {
        Some(segments) => segments
            .iter()
            .map(|ipa| Phone {
                ipa: (*ipa).to_string(),
                source_symbol: phone.source_symbol.clone(),
                status: phone.status,
            })
            .collect(),
        None => vec![phone.clone()],
    }
}

fn singing_decomposition(ipa: &str) -> Option<&'static [&'static str]> {
    match ipa {
        "aʊ" => Some(&["a", "ʊ"]),
        "aɪ" => Some(&["a", "ɪ"]),
        "eɪ" => Some(&["e", "ɪ"]),
        "oʊ" => Some(&["o", "ʊ"]),
        "ɔɪ" => Some(&["ɔ", "ɪ"]),
        _ => None,
    }
}

fn acoustic_decomposition(ipa: &str) -> Option<&'static [&'static str]> {
    match ipa {
        "tʃ" => Some(&["t", "ʃ"]),
        "dʒ" => Some(&["d", "ʒ"]),
        _ => None,
    }
}

// ─── Phone comparison ─────────────────────────────────────────────────────────

/// Controls how two [`Phone`]s are compared for equality.
///
/// `ExactIpa` is the default: two phones match only if their `ipa` strings are
/// identical.  `Broad` applies the active [`PhoneEqualityOptions`] flags to
/// strip length marks, tie bars, and/or diacritics before comparing, allowing
/// `[tʰ]` to match `/t/` when aspiration is considered non-contrastive.
///
/// `Segmental` is reserved for a future feature-bundle comparison layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhoneComparisonMode {
    /// Two phones are equal only when their IPA strings are byte-identical.
    ExactIpa,
    /// Apply the [`PhoneEqualityOptions`] normalization flags before comparing.
    Broad,
    /// Reserved: future feature-bundle / distinctive-feature comparison.
    Segmental,
}

/// Policy for comparing two [`Phone`]s for "same segment" equality.
///
/// The default compares IPA strings exactly.  Set flags to strip phonetic
/// detail that the calling context does not care about.
///
/// # Example
///
/// ```
/// use listenbury::linguistic::phonology::{Phone, PhoneComparisonMode, PhoneEqualityOptions, phones_equivalent};
///
/// let t    = Phone::mapped("t");
/// let t_h  = Phone::mapped("tʰ");
///
/// // Exact mode: aspirated /tʰ/ ≠ plain /t/
/// assert!(!phones_equivalent(&t, &t_h, &PhoneEqualityOptions::default()));
///
/// // Broad mode + ignore_diacritics: /tʰ/ ≈ /t/
/// let broad = PhoneEqualityOptions {
///     mode: PhoneComparisonMode::Broad,
///     ignore_diacritics: true,
///     ..Default::default()
/// };
/// assert!(phones_equivalent(&t, &t_h, &broad));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhoneEqualityOptions {
    /// Which comparison algorithm to use.
    pub mode: PhoneComparisonMode,
    /// Strip IPA stress diacritics (ˈ ˌ) before comparing.
    pub ignore_stress: bool,
    /// Strip length marks (ː ˑ) before comparing.
    pub ignore_length: bool,
    /// Strip tie bars (͡ ͜) before comparing.
    pub ignore_tie_bars: bool,
    /// Strip superscript modifier diacritics (aspiration ʰ, labialization ʷ,
    /// palatalization ʲ, etc.) and combining marks before comparing.
    pub ignore_diacritics: bool,
}

impl Default for PhoneEqualityOptions {
    fn default() -> Self {
        Self {
            mode: PhoneComparisonMode::ExactIpa,
            ignore_stress: false,
            ignore_length: false,
            ignore_tie_bars: false,
            ignore_diacritics: false,
        }
    }
}

/// Normalize a phone's IPA string according to `options`, producing a
/// comparison key.
pub fn phone_comparison_key(phone: &Phone, options: &PhoneEqualityOptions) -> String {
    if options.mode == PhoneComparisonMode::ExactIpa {
        return phone.ipa.clone();
    }
    let mut s = phone.ipa.clone();
    if options.ignore_stress {
        // ˈ U+02C8, ˌ U+02CC
        s = s.replace('\u{02C8}', "").replace('\u{02CC}', "");
    }
    if options.ignore_length {
        // ː U+02D0, ˑ U+02D1
        s = s.replace('\u{02D0}', "").replace('\u{02D1}', "");
    }
    if options.ignore_tie_bars {
        // ͡ U+0361, ͜ U+035C
        s = s.replace('\u{0361}', "").replace('\u{035C}', "");
    }
    if options.ignore_diacritics {
        // Strip superscript modifier letters (U+02B0..=U+02FF) and combining
        // diacritical marks (U+0300..=U+036F), but only those that represent
        // phonetic detail rather than base segment identity.
        s = s
            .chars()
            .filter(|c| {
                let cp = *c as u32;
                // Keep: regular IPA base characters, length (already stripped above),
                // stress (already stripped above).
                // Remove: superscript modifiers (aspiration, labialization, …) and
                // combining diacritics (dental, nasalization, …).
                // Spacing modifier letters (02B0–02FF) excluding stress marks
                // ˈ (02C8) and ˌ (02CC), plus combining diacritics (0300–036F).
                !matches!(cp, 0x02B0..=0x02C7 | 0x02C9..=0x02CB | 0x02CD..=0x02FF | 0x0300..=0x036F)
            })
            .collect();
    }
    s
}

/// Return `true` if `left` and `right` represent the same phone segment under
/// the given comparison `options`.
///
/// This is the canonical comparison function for phonotactic lookup: it lets
/// `[tʰ]` match `/t/` when `ignore_diacritics` is set, while keeping `/t/`
/// distinct from `/ɾ/` (a different segment, not a diacritic variant).
pub fn phones_equivalent(left: &Phone, right: &Phone, options: &PhoneEqualityOptions) -> bool {
    phone_comparison_key(left, options) == phone_comparison_key(right, options)
}
