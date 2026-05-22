//! Render targets: phone-level synthesis parameters and default acoustic
//! targets for English phones.
//!
//! This module provides:
//!
//! - [`PhoneRenderTarget`] — per-phone synthesis parameters for the Klatt
//!   renderer.
//! - [`GlottalSourceTarget`] and [`VocalTractFilterTarget`] — explicit source
//!   and filter parameters for synthesis.
//! - [`PhoneAcousticTarget`] — combined acoustic description of a phone used
//!   to build render targets.
//! - [`default_english_phone_targets`] — a data table of default acoustic
//!   parameters for core English phones (not a pile of `fn english_*()` fns).
//! - Builder functions that convert [`PhoneString`], [`Syllable`], and
//!   [`SungSyllable`] into `Vec<PhoneRenderTarget>`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::linguistic::phonology::{Phone, PhoneString};
use crate::prosody::syllable::{SungSyllable, Syllable};

// ---------------------------------------------------------------------------
// Synthesis parameter types
// ---------------------------------------------------------------------------

/// Glottal source parameters for synthesis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlottalSourceTarget {
    /// Proportion of aspiration noise to mix with the periodic source
    /// (0.0 = fully modal, 1.0 = fully aspirated/whispered).
    pub breathiness: f32,
    /// Open quotient (0.0–1.0).  0.5 is typical modal phonation.
    pub open_quotient: f32,
    /// Spectral tilt of the source in dB/octave (typically −6 to −3).
    pub spectral_tilt_db_per_octave: f32,
}

/// Vocal-tract formant filter parameters for synthesis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VocalTractFilterTarget {
    /// F1 centre frequency in Hz.
    pub f1_hz: f32,
    /// F1 bandwidth (−3 dB) in Hz.
    pub f1_bw_hz: f32,
    /// F1 relative amplitude offset in dB.
    pub f1_amp_db: f32,
    /// F2 centre frequency in Hz.
    pub f2_hz: f32,
    /// F2 bandwidth in Hz.
    pub f2_bw_hz: f32,
    /// F2 relative amplitude offset in dB.
    pub f2_amp_db: f32,
    /// F3 centre frequency in Hz.
    pub f3_hz: f32,
    /// F3 bandwidth in Hz.
    pub f3_bw_hz: f32,
    /// F3 relative amplitude offset in dB.
    pub f3_amp_db: f32,
    /// Optional F4 centre frequency in Hz.
    pub f4_hz: Option<f32>,
    /// Optional F4 bandwidth in Hz.
    pub f4_bw_hz: Option<f32>,
    /// Optional F4 amplitude offset in dB (0.0 if not set).
    pub f4_amp_db: Option<f32>,
}

/// Complete render target for a single phone segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhoneRenderTarget {
    /// The phone to render.
    pub phone: Phone,
    /// Requested duration in milliseconds.
    pub duration_ms: u64,
    /// Fundamental frequency in Hz.  `None` for unvoiced phones.
    pub f0_hz: Option<f32>,
    /// Overall amplitude (linear 0.0–1.0).
    pub amplitude: f32,
    /// Optional explicit glottal source parameters (overrides defaults).
    pub source: Option<GlottalSourceTarget>,
    /// Optional explicit vocal-tract filter parameters (overrides defaults).
    pub filter: Option<VocalTractFilterTarget>,
}

// ---------------------------------------------------------------------------
// Acoustic target descriptor
// ---------------------------------------------------------------------------

/// Combined acoustic description of a single phone, used as the data-table
/// entry for [`default_english_phone_targets`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhoneAcousticTarget {
    /// IPA symbol.
    pub ipa: String,
    /// `true` if this phone is typically voiced.
    pub voiced: bool,
    /// `true` for vowels.
    pub is_vowel: bool,
    /// `true` for fricatives.
    pub is_fricative: bool,
    /// `true` for stops / affricates.
    pub is_stop: bool,
    /// `true` for nasals.
    pub is_nasal: bool,
    /// `true` for approximants / liquids / glides.
    pub is_approximant: bool,
    /// Default vocal-tract filter targets.  `None` for stops (burst only).
    pub filter: Option<VocalTractFilterTarget>,
    /// Default glottal source parameters.
    pub source: GlottalSourceTarget,
    /// Default duration in milliseconds.
    pub default_duration_ms: u64,
}

// ---------------------------------------------------------------------------
// Default English phone table
// ---------------------------------------------------------------------------

/// Return a table of default acoustic targets for core English phones.
///
/// Keys are IPA symbols.  The table covers:
/// - Core English vowels `/i ɪ e ɛ æ ə ʌ ɑ ɔ o ʊ u/`
/// - Sonorants `/m n ŋ l ɹ j w/`
/// - Fricatives `/s z ʃ ʒ f v θ ð h/`
/// - Stops `/p b t d k ɡ/`
///
/// Formant values are representative General-American targets.  They are
/// deliberately approximate — the goal is testable synthesis, not perfect
/// quality.
pub fn default_english_phone_targets() -> HashMap<String, PhoneAcousticTarget> {
    let mut map: HashMap<String, PhoneAcousticTarget> = HashMap::new();

    // --- Helpers ------------------------------------------------------------

    fn vowel(ipa: &str, f1: f32, f2: f32, f3: f32, dur_ms: u64) -> PhoneAcousticTarget {
        PhoneAcousticTarget {
            ipa: ipa.to_string(),
            voiced: true,
            is_vowel: true,
            is_fricative: false,
            is_stop: false,
            is_nasal: false,
            is_approximant: false,
            filter: Some(VocalTractFilterTarget {
                f1_hz: f1,
                f1_bw_hz: 80.0,
                f1_amp_db: 0.0,
                f2_hz: f2,
                f2_bw_hz: 100.0,
                f2_amp_db: -3.0,
                f3_hz: f3,
                f3_bw_hz: 150.0,
                f3_amp_db: -6.0,
                f4_hz: Some(3500.0),
                f4_bw_hz: Some(200.0),
                f4_amp_db: Some(-9.0),
            }),
            source: GlottalSourceTarget {
                breathiness: 0.05,
                open_quotient: 0.5,
                spectral_tilt_db_per_octave: -6.0,
            },
            default_duration_ms: dur_ms,
        }
    }

    fn nasal(ipa: &str, f2: f32) -> PhoneAcousticTarget {
        PhoneAcousticTarget {
            ipa: ipa.to_string(),
            voiced: true,
            is_vowel: false,
            is_fricative: false,
            is_stop: false,
            is_nasal: true,
            is_approximant: false,
            filter: Some(VocalTractFilterTarget {
                f1_hz: 280.0,
                f1_bw_hz: 100.0,
                f1_amp_db: -6.0,
                f2_hz: f2,
                f2_bw_hz: 150.0,
                f2_amp_db: -9.0,
                f3_hz: 2500.0,
                f3_bw_hz: 200.0,
                f3_amp_db: -12.0,
                f4_hz: None,
                f4_bw_hz: None,
                f4_amp_db: None,
            }),
            source: GlottalSourceTarget {
                breathiness: 0.0,
                open_quotient: 0.5,
                spectral_tilt_db_per_octave: -9.0,
            },
            default_duration_ms: 70,
        }
    }

    fn approximant(ipa: &str, f1: f32, f2: f32, f3: f32) -> PhoneAcousticTarget {
        PhoneAcousticTarget {
            ipa: ipa.to_string(),
            voiced: true,
            is_vowel: false,
            is_fricative: false,
            is_stop: false,
            is_nasal: false,
            is_approximant: true,
            filter: Some(VocalTractFilterTarget {
                f1_hz: f1,
                f1_bw_hz: 100.0,
                f1_amp_db: -3.0,
                f2_hz: f2,
                f2_bw_hz: 120.0,
                f2_amp_db: -3.0,
                f3_hz: f3,
                f3_bw_hz: 200.0,
                f3_amp_db: -6.0,
                f4_hz: None,
                f4_bw_hz: None,
                f4_amp_db: None,
            }),
            source: GlottalSourceTarget {
                breathiness: 0.05,
                open_quotient: 0.5,
                spectral_tilt_db_per_octave: -6.0,
            },
            default_duration_ms: 80,
        }
    }

    fn fricative(ipa: &str, voiced: bool, f2_noise_hz: Option<f32>) -> PhoneAcousticTarget {
        PhoneAcousticTarget {
            ipa: ipa.to_string(),
            voiced,
            is_vowel: false,
            is_fricative: true,
            is_stop: false,
            is_nasal: false,
            is_approximant: false,
            filter: f2_noise_hz.map(|f2| VocalTractFilterTarget {
                f1_hz: 400.0,
                f1_bw_hz: 200.0,
                f1_amp_db: -12.0,
                f2_hz: f2,
                f2_bw_hz: 500.0,
                f2_amp_db: -6.0,
                f3_hz: 3200.0,
                f3_bw_hz: 600.0,
                f3_amp_db: -6.0,
                f4_hz: None,
                f4_bw_hz: None,
                f4_amp_db: None,
            }),
            source: GlottalSourceTarget {
                breathiness: if voiced { 0.3 } else { 0.95 },
                open_quotient: if voiced { 0.5 } else { 0.0 },
                spectral_tilt_db_per_octave: -3.0,
            },
            default_duration_ms: 70,
        }
    }

    fn stop(ipa: &str, voiced: bool) -> PhoneAcousticTarget {
        PhoneAcousticTarget {
            ipa: ipa.to_string(),
            voiced,
            is_vowel: false,
            is_fricative: false,
            is_stop: true,
            is_nasal: false,
            is_approximant: false,
            filter: None,
            source: GlottalSourceTarget {
                breathiness: if voiced { 0.1 } else { 0.85 },
                open_quotient: if voiced { 0.5 } else { 0.0 },
                spectral_tilt_db_per_octave: -3.0,
            },
            default_duration_ms: 60,
        }
    }

    // --- Vowels (General American targets) ----------------------------------

    for t in [
        vowel("i", 280.0, 2250.0, 2900.0, 90),  // heed
        vowel("ɪ", 400.0, 1920.0, 2550.0, 80),  // hid
        vowel("e", 370.0, 2080.0, 2750.0, 90),  // hay (monophthong)
        vowel("ɛ", 580.0, 1820.0, 2650.0, 85),  // head
        vowel("æ", 700.0, 1660.0, 2430.0, 100), // had
        vowel("ə", 500.0, 1500.0, 2500.0, 60),  // schwa
        vowel("ʌ", 640.0, 1200.0, 2400.0, 80),  // hud
        vowel("ɑ", 730.0, 1090.0, 2440.0, 100), // hot / father
        vowel("ɔ", 570.0, 840.0, 2410.0, 90),   // saw
        vowel("o", 360.0, 640.0, 2500.0, 90),   // hoe (monophthong)
        vowel("ʊ", 440.0, 1020.0, 2240.0, 80),  // hood
        vowel("u", 310.0, 870.0, 2250.0, 90),   // hoot
    ] {
        map.insert(t.ipa.clone(), t);
    }

    // --- Sonorants ----------------------------------------------------------

    for t in [
        nasal("m", 1100.0),
        nasal("n", 1700.0),
        nasal("ŋ", 2300.0),
        approximant("l", 360.0, 1100.0, 2600.0),
        approximant("ɹ", 460.0, 1060.0, 1600.0), // rhotic
        approximant("j", 320.0, 2100.0, 2900.0), // yod
        approximant("w", 300.0, 610.0, 2200.0),  // labial-velar
    ] {
        map.insert(t.ipa.clone(), t);
    }

    // --- Fricatives ---------------------------------------------------------

    for t in [
        fricative("s", false, Some(7000.0)),
        fricative("z", true, Some(7000.0)),
        fricative("ʃ", false, Some(2500.0)),
        fricative("ʒ", true, Some(2500.0)),
        fricative("f", false, None),
        fricative("v", true, None),
        fricative("θ", false, None),
        fricative("ð", true, None),
        fricative("h", false, Some(1600.0)), // glottal fricative
    ] {
        map.insert(t.ipa.clone(), t);
    }

    // --- Stops / affricates (burst + transition placeholder) ----------------

    for t in [
        stop("p", false),
        stop("b", true),
        stop("t", false),
        stop("d", true),
        stop("k", false),
        stop("ɡ", true),
    ] {
        map.insert(t.ipa.clone(), t);
    }

    map
}

// ---------------------------------------------------------------------------
// Builder functions
// ---------------------------------------------------------------------------

/// Build a list of [`PhoneRenderTarget`] from a [`PhoneString`].
///
/// * `f0_hz`     — global F0 to apply to voiced phones (`None` for speech at
///   natural pitch from the table default).
/// * `amplitude` — overall linear gain (0.0–1.0).
/// * `targets`   — acoustic-target table (from [`default_english_phone_targets`]).
pub fn phone_render_targets_from_string(
    phone_string: &PhoneString,
    f0_hz: Option<f32>,
    amplitude: f32,
    targets: &HashMap<String, PhoneAcousticTarget>,
) -> Vec<PhoneRenderTarget> {
    phone_string
        .phones
        .iter()
        .map(|phone| build_render_target(phone, f0_hz, amplitude, targets))
        .collect()
}

/// Build render targets from a [`Syllable`] (onset → nucleus → coda order).
pub fn render_targets_from_syllable(
    syllable: &Syllable,
    f0_hz: Option<f32>,
    amplitude: f32,
    targets: &HashMap<String, PhoneAcousticTarget>,
) -> Vec<PhoneRenderTarget> {
    let ps = PhoneString {
        phones: syllable.phones().cloned().collect(),
    };
    phone_render_targets_from_string(&ps, f0_hz, amplitude, targets)
}

/// Build render targets from a [`SungSyllable`].
///
/// The F0 is derived from the syllable's [`PitchCurve`] first anchor point
/// if present, or from the [`NoteTarget`] if no explicit curve was attached.
pub fn render_targets_from_sung_syllable(
    syllable: &SungSyllable,
    amplitude: f32,
    targets: &HashMap<String, PhoneAcousticTarget>,
) -> Vec<PhoneRenderTarget> {
    // Prefer the continuous pitch curve's first anchor, fall back to note target
    let f0_hz: Option<f32> = syllable
        .pitch_curve
        .as_ref()
        .and_then(|pc| pc.points.first().map(|p| p.hz))
        .or_else(|| {
            syllable
                .note
                .as_ref()
                .map(|n| n.pitch.frequency_hz() as f32)
        });

    let phones: Vec<Phone> = syllable
        .phones
        .iter()
        .map(|tpr| tpr.phone.clone())
        .collect();
    let ps = PhoneString { phones };
    phone_render_targets_from_string(&ps, f0_hz, amplitude, targets)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn build_render_target(
    phone: &Phone,
    f0_hz: Option<f32>,
    amplitude: f32,
    targets: &HashMap<String, PhoneAcousticTarget>,
) -> PhoneRenderTarget {
    let entry = targets.get(phone.ipa.as_str());
    let effective_f0 = match entry {
        Some(t) if !t.voiced => None,
        _ => f0_hz,
    };
    let duration_ms = entry.map(|t| t.default_duration_ms).unwrap_or(80);
    PhoneRenderTarget {
        phone: phone.clone(),
        duration_ms,
        f0_hz: effective_f0,
        amplitude,
        source: entry.map(|t| t.source.clone()),
        filter: entry.and_then(|t| t.filter.clone()),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linguistic::phonology::Phone;

    #[test]
    fn default_table_covers_core_vowels() {
        let table = default_english_phone_targets();
        for ipa in ["i", "ɪ", "ɛ", "æ", "ə", "ʌ", "ɑ", "u"] {
            assert!(table.contains_key(ipa), "expected table to contain /{ipa}/");
        }
    }

    #[test]
    fn default_table_covers_core_consonants() {
        let table = default_english_phone_targets();
        for ipa in ["m", "n", "ŋ", "s", "z", "p", "b", "t", "d", "k"] {
            assert!(table.contains_key(ipa), "expected table to contain /{ipa}/");
        }
    }

    #[test]
    fn unvoiced_phones_get_no_f0() {
        let table = default_english_phone_targets();
        let phone = Phone::new_ipa("s");
        let targets = phone_render_targets_from_string(
            &PhoneString {
                phones: vec![phone],
            },
            Some(120.0),
            0.7,
            &table,
        );
        assert_eq!(targets.len(), 1);
        assert!(targets[0].f0_hz.is_none(), "/s/ should be unvoiced");
    }

    #[test]
    fn voiced_phone_keeps_f0() {
        let table = default_english_phone_targets();
        let phone = Phone::new_ipa("i");
        let targets = phone_render_targets_from_string(
            &PhoneString {
                phones: vec![phone],
            },
            Some(220.0),
            0.8,
            &table,
        );
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].f0_hz, Some(220.0));
    }

    #[test]
    fn phone_string_targets_preserve_order() {
        let table = default_english_phone_targets();
        let ps = PhoneString {
            phones: vec![
                Phone::new_ipa("s"),
                Phone::new_ipa("ɪ"),
                Phone::new_ipa("t"),
            ],
        };
        let targets = phone_render_targets_from_string(&ps, Some(150.0), 0.7, &table);
        assert_eq!(targets.len(), 3);
        assert_eq!(targets[0].phone.ipa, "s");
        assert_eq!(targets[1].phone.ipa, "ɪ");
        assert_eq!(targets[2].phone.ipa, "t");
    }

    #[test]
    fn syllable_targets_include_onset_nucleus_coda() {
        use crate::prosody::syllable::{SourceSpan, Syllable};

        let table = default_english_phone_targets();
        let syl = Syllable {
            onset: PhoneString {
                phones: vec![Phone::new_ipa("s")],
            },
            nucleus: PhoneString {
                phones: vec![Phone::new_ipa("ɪ")],
            },
            coda: PhoneString {
                phones: vec![Phone::new_ipa("t")],
            },
            source_span: SourceSpan { start: 0, end: 3 },
            stress: None,
            variety: "test".into(),
            diagnostics: vec![],
        };
        let targets = render_targets_from_syllable(&syl, Some(150.0), 0.7, &table);
        assert_eq!(targets.len(), 3);
    }

    #[test]
    fn render_target_json_round_trips() {
        let table = default_english_phone_targets();
        let phone = Phone::new_ipa("i");
        let targets = phone_render_targets_from_string(
            &PhoneString {
                phones: vec![phone],
            },
            Some(220.0),
            0.8,
            &table,
        );
        let json = serde_json::to_string(&targets[0]).unwrap();
        let back: PhoneRenderTarget = serde_json::from_str(&json).unwrap();
        assert_eq!(targets[0], back);
    }
}
