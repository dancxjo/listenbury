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

use crate::linguistic::phonology::{
    FeatureBundle, MajorClass, Manner, Phone, PhoneString, PhonemicInventory, Place, Roundedness,
    Voicing, VowelBackness, VowelHeight, general_american_english,
};
use crate::prosody::syllable::{SungSyllable, Syllable};
use crate::prosody::vibrato::Vibrato;

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
    /// Optional vibrato modulation to apply over this phone's F0.
    pub vibrato: Option<Vibrato>,
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
    /// Frication/noise amount for this phone (0.0-1.0).
    #[serde(default)]
    pub frication_level: f32,
    /// Aspiration amount for this phone (0.0-1.0).
    #[serde(default)]
    pub aspiration_level: f32,
    /// Optional burst spectral center hint in Hz for stops/affricates.
    #[serde(default)]
    pub burst_hz_hint: Option<f32>,
    /// Optional closure duration hint in milliseconds for stops/affricates.
    #[serde(default)]
    pub closure_ms_hint: Option<u64>,
    /// Optional release duration hint in milliseconds for stops/affricates.
    #[serde(default)]
    pub release_ms_hint: Option<u64>,
    /// Optional nasal pole frequency hint in Hz.
    #[serde(default)]
    pub nasal_pole_hz: Option<f32>,
    /// Optional nasal zero frequency hint in Hz.
    #[serde(default)]
    pub nasal_zero_hz: Option<f32>,
    /// Coarticulation resistance / transition stiffness (0.0-1.0).
    #[serde(default)]
    pub transition_stiffness: f32,
}

// ---------------------------------------------------------------------------
// Default English phone table
// ---------------------------------------------------------------------------

/// Return a table of default acoustic targets for core English phones.
///
/// Keys are IPA symbols.  The table covers:
/// - Core English vowels `/i ɪ e ɛ æ ə ʌ ɑ ɔ o ʊ u/`
/// - Common English diphthongs `/aɪ ɑɪ aʊ eɪ oʊ ɔɪ/`
/// - Sonorants `/m n ŋ l ɹ j w/`
/// - Fricatives `/s z ʃ ʒ f v θ ð h/`
/// - Stops `/p b t d k ɡ/`
/// - Affricates `/tʃ dʒ/`
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
            frication_level: 0.0,
            aspiration_level: 0.05,
            burst_hz_hint: None,
            closure_ms_hint: None,
            release_ms_hint: None,
            nasal_pole_hz: None,
            nasal_zero_hz: None,
            transition_stiffness: 0.35,
        }
    }

    fn diphthong(ipa: &str, f1: f32, f2: f32, f3: f32) -> PhoneAcousticTarget {
        let mut target = vowel(ipa, f1, f2, f3, 140);
        target.filter = target.filter.map(|mut filter| {
            filter.f1_bw_hz = 100.0;
            filter.f2_bw_hz = 140.0;
            filter
        });
        target
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
            frication_level: 0.0,
            aspiration_level: 0.0,
            burst_hz_hint: None,
            closure_ms_hint: None,
            release_ms_hint: None,
            nasal_pole_hz: Some(300.0),
            nasal_zero_hz: Some(900.0),
            transition_stiffness: 0.55,
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
            frication_level: 0.0,
            aspiration_level: 0.03,
            burst_hz_hint: None,
            closure_ms_hint: None,
            release_ms_hint: None,
            nasal_pole_hz: None,
            nasal_zero_hz: None,
            transition_stiffness: 0.6,
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
            frication_level: if voiced { 0.7 } else { 0.95 },
            aspiration_level: if voiced { 0.2 } else { 0.5 },
            burst_hz_hint: None,
            closure_ms_hint: None,
            release_ms_hint: None,
            nasal_pole_hz: None,
            nasal_zero_hz: None,
            transition_stiffness: 0.7,
        }
    }

    fn stop(ipa: &str, voiced: bool, burst_hz: f32) -> PhoneAcousticTarget {
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
            frication_level: 0.0,
            aspiration_level: if voiced { 0.05 } else { 0.4 },
            burst_hz_hint: Some(burst_hz),
            closure_ms_hint: Some(45),
            release_ms_hint: Some(15),
            nasal_pole_hz: None,
            nasal_zero_hz: None,
            transition_stiffness: 0.85,
        }
    }

    fn affricate(ipa: &str, voiced: bool) -> PhoneAcousticTarget {
        PhoneAcousticTarget {
            ipa: ipa.to_string(),
            voiced,
            is_vowel: false,
            is_fricative: true,
            is_stop: true,
            is_nasal: false,
            is_approximant: false,
            filter: Some(VocalTractFilterTarget {
                f1_hz: 420.0,
                f1_bw_hz: 220.0,
                f1_amp_db: -10.0,
                f2_hz: 2500.0,
                f2_bw_hz: 520.0,
                f2_amp_db: -4.0,
                f3_hz: 3400.0,
                f3_bw_hz: 620.0,
                f3_amp_db: -5.0,
                f4_hz: None,
                f4_bw_hz: None,
                f4_amp_db: None,
            }),
            source: GlottalSourceTarget {
                breathiness: if voiced { 0.55 } else { 0.98 },
                open_quotient: if voiced { 0.5 } else { 0.0 },
                spectral_tilt_db_per_octave: -3.0,
            },
            default_duration_ms: 85,
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

    // --- Diphthongs ---------------------------------------------------------
    //
    // These are steady acoustic approximations of the vowel-to-glide movement.
    // The phone-timed Klatt path keeps diphthongs as one nucleus phone, so the
    // targets sit between the start vowel and offglide until the renderer grows
    // time-varying formant trajectories.
    for t in [
        diphthong("aɪ", 560.0, 1650.0, 2500.0), // price, broad IPA
        diphthong("ɑɪ", 560.0, 1500.0, 2480.0), // price, GA-flavored onset
        diphthong("aʊ", 610.0, 1120.0, 2350.0), // mouth
        diphthong("eɪ", 350.0, 2200.0, 2800.0), // face
        diphthong("oʊ", 380.0, 760.0, 2380.0),  // goat
        diphthong("ɔɪ", 500.0, 1450.0, 2450.0), // choice
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
        stop("p", false, 900.0),
        stop("b", true, 900.0),
        stop("t", false, 3000.0),
        stop("d", true, 3000.0),
        stop("k", false, 1900.0),
        stop("ɡ", true, 1900.0),
        stop("p", false),
        stop("b", true),
        stop("t", false),
        stop("d", true),
        stop("k", false),
        stop("ɡ", true),
        affricate("tʃ", false),
        affricate("dʒ", true),
    ] {
        map.insert(t.ipa.clone(), t);
    }

    map
}

fn stop_burst_hint(place: Option<Place>) -> f32 {
    match place {
        Some(Place::Bilabial | Place::Labiodental) => 900.0,
        Some(Place::Dental | Place::Alveolar | Place::Postalveolar) => 3000.0,
        Some(Place::Palatal | Place::Velar) => 1900.0,
        Some(Place::Glottal) | None => 1600.0,
    }
}

const MIN_VOWEL_F2_HZ: f32 = 500.0;
const MIN_VOWEL_F3_HZ: f32 = 1300.0;

fn fricative_noise_center(place: Option<Place>) -> f32 {
    match place {
        Some(Place::Bilabial | Place::Labiodental) => 1300.0,
        Some(Place::Dental) => 1800.0,
        Some(Place::Alveolar | Place::Postalveolar) => 4200.0,
        Some(Place::Palatal) => 3200.0,
        Some(Place::Velar) => 2200.0,
        Some(Place::Glottal) | None => 1600.0,
    }
}

fn consonant_formant_anchor(place: Option<Place>) -> (f32, f32, f32) {
    match place {
        Some(Place::Bilabial | Place::Labiodental) => (360.0, 1100.0, 2400.0),
        Some(Place::Dental | Place::Alveolar | Place::Postalveolar) => (380.0, 1700.0, 2600.0),
        Some(Place::Palatal) => (320.0, 2100.0, 2800.0),
        Some(Place::Velar) => (340.0, 1500.0, 2500.0),
        Some(Place::Glottal) | None => (400.0, 1600.0, 2500.0),
    }
}

fn vowel_formants_from_features(features: &FeatureBundle) -> (f32, f32, f32) {
    let mut f1: f32 = match features.vowel_height {
        Some(VowelHeight::High) => 320.0,
        Some(VowelHeight::Mid) => 500.0,
        Some(VowelHeight::Low) => 700.0,
        Some(VowelHeight::Rhotic) => 460.0,
        None => 520.0,
    };
    let mut f2: f32 = match features.vowel_backness {
        Some(VowelBackness::Front) => 2150.0,
        Some(VowelBackness::Central) => 1500.0,
        Some(VowelBackness::Back) => 950.0,
        None => 1500.0,
    };
    let mut f3: f32 = if features.vowel_height == Some(VowelHeight::Rhotic) {
        1750.0
    } else {
        2600.0
    };

    if features.roundedness == Some(Roundedness::Rounded) {
        f2 -= 180.0;
        f3 -= 220.0;
        f1 -= 20.0;
    }

    // Keep feature-derived vowels inside a physically plausible region so
    // extreme feature combinations cannot collapse resonances.
    (f1, f2.max(MIN_VOWEL_F2_HZ), f3.max(MIN_VOWEL_F3_HZ))
}

/// Derive a Klatt acoustic target from structured phonological features.
pub fn klatt_targets_from_features(
    phone: &Phone,
    features: &FeatureBundle,
    _variety: &PhonemicInventory,
) -> PhoneAcousticTarget {
    if features.major == MajorClass::Vowel || features.syllabic {
        let (f1, f2, f3) = vowel_formants_from_features(features);
        return PhoneAcousticTarget {
            ipa: phone.ipa.clone(),
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
            default_duration_ms: if features.vowel_height == Some(VowelHeight::Rhotic) {
                110
            } else {
                90
            },
            frication_level: 0.0,
            aspiration_level: 0.05,
            burst_hz_hint: None,
            closure_ms_hint: None,
            release_ms_hint: None,
            nasal_pole_hz: None,
            nasal_zero_hz: None,
            transition_stiffness: 0.35,
        };
    }

    let voiced = !matches!(features.voicing, Some(Voicing::Voiceless));
    // Unknown consonant manner defaults to stop-like behavior so unknown
    // symbols remain bounded (no accidental broadband frication) and can still
    // carry conservative closure/release hints.
    let manner = features.manner.unwrap_or(Manner::Stop);

    match manner {
        Manner::Nasal => {
            let (_, f2, _) = consonant_formant_anchor(features.place);
            PhoneAcousticTarget {
                ipa: phone.ipa.clone(),
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
                frication_level: 0.0,
                aspiration_level: 0.0,
                burst_hz_hint: None,
                closure_ms_hint: None,
                release_ms_hint: None,
                nasal_pole_hz: Some(300.0),
                nasal_zero_hz: Some(900.0),
                transition_stiffness: 0.55,
            }
        }
        Manner::Fricative | Manner::Affricate => {
            let noise_center = fricative_noise_center(features.place);
            PhoneAcousticTarget {
                ipa: phone.ipa.clone(),
                voiced,
                is_vowel: false,
                is_fricative: true,
                is_stop: manner == Manner::Affricate,
                is_nasal: false,
                is_approximant: false,
                filter: Some(VocalTractFilterTarget {
                    f1_hz: 400.0,
                    f1_bw_hz: 200.0,
                    f1_amp_db: -12.0,
                    f2_hz: noise_center,
                    f2_bw_hz: 500.0,
                    f2_amp_db: -6.0,
                    f3_hz: (noise_center + 1200.0).max(2600.0),
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
                default_duration_ms: if manner == Manner::Affricate { 85 } else { 70 },
                frication_level: if voiced { 0.7 } else { 0.95 },
                aspiration_level: if voiced { 0.2 } else { 0.5 },
                burst_hz_hint: if manner == Manner::Affricate {
                    Some(stop_burst_hint(features.place))
                } else {
                    None
                },
                closure_ms_hint: if manner == Manner::Affricate {
                    Some(35)
                } else {
                    None
                },
                release_ms_hint: if manner == Manner::Affricate {
                    Some(20)
                } else {
                    None
                },
                nasal_pole_hz: None,
                nasal_zero_hz: None,
                transition_stiffness: 0.75,
            }
        }
        Manner::Liquid | Manner::Glide => {
            let (f1, f2, mut f3) = consonant_formant_anchor(features.place);
            if features.place == Some(Place::Postalveolar) {
                f3 = 1700.0;
            }
            PhoneAcousticTarget {
                ipa: phone.ipa.clone(),
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
                frication_level: 0.0,
                aspiration_level: 0.03,
                burst_hz_hint: None,
                closure_ms_hint: None,
                release_ms_hint: None,
                nasal_pole_hz: None,
                nasal_zero_hz: None,
                transition_stiffness: 0.65,
            }
        }
        Manner::Stop => PhoneAcousticTarget {
            ipa: phone.ipa.clone(),
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
            frication_level: 0.0,
            aspiration_level: if voiced { 0.05 } else { 0.4 },
            burst_hz_hint: Some(stop_burst_hint(features.place)),
            closure_ms_hint: Some(45),
            release_ms_hint: Some(15),
            nasal_pole_hz: None,
            nasal_zero_hz: None,
            transition_stiffness: 0.85,
        },
        Manner::Vowel => {
            let (f1, f2, f3) = vowel_formants_from_features(features);
            PhoneAcousticTarget {
                ipa: phone.ipa.clone(),
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
                default_duration_ms: 90,
                frication_level: 0.0,
                aspiration_level: 0.05,
                burst_hz_hint: None,
                closure_ms_hint: None,
                release_ms_hint: None,
                nasal_pole_hz: None,
                nasal_zero_hz: None,
                transition_stiffness: 0.35,
            }
        }
    }
}

fn merge_feature_target_with_symbol_override(
    mut base: PhoneAcousticTarget,
    symbol_override: &PhoneAcousticTarget,
) -> PhoneAcousticTarget {
    base.filter = symbol_override.filter.clone();
    base.source = symbol_override.source.clone();
    base.default_duration_ms = symbol_override.default_duration_ms;
    base.frication_level = symbol_override.frication_level;
    base.aspiration_level = symbol_override.aspiration_level;
    base.burst_hz_hint = symbol_override.burst_hz_hint;
    base.closure_ms_hint = symbol_override.closure_ms_hint;
    base.release_ms_hint = symbol_override.release_ms_hint;
    base.nasal_pole_hz = symbol_override.nasal_pole_hz;
    base.nasal_zero_hz = symbol_override.nasal_zero_hz;
    base.transition_stiffness = symbol_override.transition_stiffness;
    base
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
    let inventory = general_american_english();
    phone_render_targets_from_string_with_inventory(
        phone_string,
        f0_hz,
        amplitude,
        targets,
        &inventory,
    )
}

/// Build a list of [`PhoneRenderTarget`] from a [`PhoneString`] using a
/// specific phonemic inventory for feature derivation.
pub fn phone_render_targets_from_string_with_inventory(
    phone_string: &PhoneString,
    f0_hz: Option<f32>,
    amplitude: f32,
    targets: &HashMap<String, PhoneAcousticTarget>,
    inventory: &PhonemicInventory,
) -> Vec<PhoneRenderTarget> {
    phone_string
        .phones
        .iter()
        .map(|phone| build_render_target(phone, f0_hz, amplitude, targets, inventory))
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
    let mut rendered = phone_render_targets_from_string(&ps, f0_hz, amplitude, targets);
    if let Some(vibrato) = syllable.vibrato {
        for idx in syllable.nucleus.start..syllable.nucleus.end {
            if let Some(target) = rendered.get_mut(idx) {
                target.vibrato = Some(vibrato);
            }
        }
    }
    rendered
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn build_render_target(
    phone: &Phone,
    f0_hz: Option<f32>,
    amplitude: f32,
    targets: &HashMap<String, PhoneAcousticTarget>,
    inventory: &PhonemicInventory,
) -> PhoneRenderTarget {
    let features = inventory.features_for_phone(phone);
    let mut target = klatt_targets_from_features(phone, &features, inventory);
    if let Some(symbol_override) = targets.get(phone.ipa.as_str()) {
        target = merge_feature_target_with_symbol_override(target, symbol_override);
    }
    let effective_f0 = if target.voiced { f0_hz } else { None };
    let duration_ms = target.default_duration_ms;
    PhoneRenderTarget {
        phone: phone.clone(),
        duration_ms,
        f0_hz: effective_f0,
        amplitude,
        vibrato: None,
        source: Some(target.source),
        filter: target.filter,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linguistic::phonology::Phone;

    fn target_from_features(phone_ipa: &str, features: FeatureBundle) -> PhoneAcousticTarget {
        let inventory = general_american_english();
        klatt_targets_from_features(&Phone::new_ipa(phone_ipa), &features, &inventory)
    }

    #[test]
    fn default_table_covers_core_vowels() {
        let table = default_english_phone_targets();
        for ipa in ["i", "ɪ", "ɛ", "æ", "ə", "ʌ", "ɑ", "u"] {
            assert!(table.contains_key(ipa), "expected table to contain /{ipa}/");
        }
    }

    #[test]
    fn default_table_covers_common_english_diphthongs() {
        let table = default_english_phone_targets();
        for ipa in ["aɪ", "ɑɪ", "aʊ", "eɪ", "oʊ", "ɔɪ"] {
            let target = table
                .get(ipa)
                .unwrap_or_else(|| panic!("expected table to contain /{ipa}/"));
            assert!(target.voiced, "/{ipa}/ should be voiced");
            assert!(target.is_vowel, "/{ipa}/ should be treated as vowelic");
            assert!(
                target.filter.is_some(),
                "/{ipa}/ should have formant targets"
            );
        }
    }

    #[test]
    fn default_table_covers_core_consonants() {
        let table = default_english_phone_targets();
        for ipa in ["m", "n", "ŋ", "s", "z", "p", "b", "t", "d", "k", "tʃ", "dʒ"] {
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

    #[test]
    fn vowel_height_backness_and_rounding_shift_formants() {
        let high_front_unrounded = target_from_features(
            "u_test_1",
            FeatureBundle {
                major: MajorClass::Vowel,
                place: None,
                vowel_height: Some(VowelHeight::High),
                vowel_backness: Some(VowelBackness::Front),
                roundedness: Some(Roundedness::Unrounded),
                manner: Some(Manner::Vowel),
                voicing: Some(Voicing::Voiced),
                syllabic: true,
            },
        );
        let low_back_rounded = target_from_features(
            "u_test_2",
            FeatureBundle {
                major: MajorClass::Vowel,
                place: None,
                vowel_height: Some(VowelHeight::Low),
                vowel_backness: Some(VowelBackness::Back),
                roundedness: Some(Roundedness::Rounded),
                manner: Some(Manner::Vowel),
                voicing: Some(Voicing::Voiced),
                syllabic: true,
            },
        );

        let high_filter = high_front_unrounded.filter.unwrap();
        let low_filter = low_back_rounded.filter.unwrap();
        assert!(
            low_filter.f1_hz > high_filter.f1_hz,
            "low vowels should raise F1"
        );
        assert!(
            low_filter.f2_hz < high_filter.f2_hz,
            "back rounded vowels should lower F2"
        );
        assert!(
            low_filter.f3_hz < high_filter.f3_hz,
            "rounding should lower F3"
        );
    }

    #[test]
    fn rhotic_vowels_lower_f3() {
        let non_rhotic = target_from_features(
            "v_test_1",
            FeatureBundle {
                major: MajorClass::Vowel,
                place: None,
                vowel_height: Some(VowelHeight::Mid),
                vowel_backness: Some(VowelBackness::Central),
                roundedness: Some(Roundedness::Unrounded),
                manner: Some(Manner::Vowel),
                voicing: Some(Voicing::Voiced),
                syllabic: true,
            },
        );
        let rhotic = target_from_features(
            "v_test_2",
            FeatureBundle {
                major: MajorClass::Vowel,
                place: None,
                vowel_height: Some(VowelHeight::Rhotic),
                vowel_backness: Some(VowelBackness::Central),
                roundedness: Some(Roundedness::Unrounded),
                manner: Some(Manner::Vowel),
                voicing: Some(Voicing::Voiced),
                syllabic: true,
            },
        );
        assert!(
            rhotic.filter.unwrap().f3_hz < non_rhotic.filter.unwrap().f3_hz,
            "rhotic vowels should lower F3"
        );
    }

    #[test]
    fn voiced_and_voiceless_consonants_change_source_behavior() {
        let voiced = target_from_features(
            "cons_v",
            FeatureBundle {
                major: MajorClass::Consonant,
                place: Some(Place::Alveolar),
                vowel_height: None,
                vowel_backness: None,
                roundedness: None,
                manner: Some(Manner::Fricative),
                voicing: Some(Voicing::Voiced),
                syllabic: false,
            },
        );
        let voiceless = target_from_features(
            "cons_vl",
            FeatureBundle {
                voicing: Some(Voicing::Voiceless),
                ..FeatureBundle {
                    major: MajorClass::Consonant,
                    place: Some(Place::Alveolar),
                    vowel_height: None,
                    vowel_backness: None,
                    roundedness: None,
                    manner: Some(Manner::Fricative),
                    voicing: Some(Voicing::Voiced),
                    syllabic: false,
                }
            },
        );
        assert!(voiced.voiced);
        assert!(!voiceless.voiced);
        assert!(
            voiced.source.open_quotient > voiceless.source.open_quotient,
            "voiceless fricatives should reduce periodic source"
        );
        assert!(
            voiced.source.breathiness < voiceless.source.breathiness,
            "voiceless fricatives should increase aspiration/noise"
        );
    }

    #[test]
    fn stops_emit_closure_release_and_burst_hints() {
        let stop = target_from_features(
            "stop_t",
            FeatureBundle {
                major: MajorClass::Consonant,
                place: Some(Place::Alveolar),
                vowel_height: None,
                vowel_backness: None,
                roundedness: None,
                manner: Some(Manner::Stop),
                voicing: Some(Voicing::Voiceless),
                syllabic: false,
            },
        );
        assert!(stop.is_stop);
        assert!(stop.closure_ms_hint.is_some());
        assert!(stop.release_ms_hint.is_some());
        assert!(stop.burst_hz_hint.is_some());
    }

    #[test]
    fn nasals_emit_nasal_coloring_hints() {
        let nasal = target_from_features(
            "nasal_n",
            FeatureBundle {
                major: MajorClass::Consonant,
                place: Some(Place::Alveolar),
                vowel_height: None,
                vowel_backness: None,
                roundedness: None,
                manner: Some(Manner::Nasal),
                voicing: Some(Voicing::Voiced),
                syllabic: false,
            },
        );
        assert!(nasal.is_nasal);
        assert!(nasal.nasal_pole_hz.is_some());
        assert!(nasal.nasal_zero_hz.is_some());
    }

    #[test]
    fn same_features_are_stable_across_symbol_spellings() {
        let features = FeatureBundle {
            major: MajorClass::Consonant,
            place: Some(Place::Labiodental),
            vowel_height: None,
            vowel_backness: None,
            roundedness: None,
            manner: Some(Manner::Fricative),
            voicing: Some(Voicing::Voiceless),
            syllabic: false,
        };
        let a = target_from_features("spell_a", features);
        let b = target_from_features("spell_b", features);
        assert_eq!(a.voiced, b.voiced);
        assert_eq!(a.frication_level, b.frication_level);
        assert_eq!(a.transition_stiffness, b.transition_stiffness);
        assert_eq!(a.filter, b.filter);
        assert_eq!(a.source, b.source);
    }

    #[test]
    fn symbol_override_can_tune_targets_without_changing_feature_logic() {
        let default_table = default_english_phone_targets();
        let mut tuned_table = default_table.clone();
        let original = default_table.get("i").unwrap().clone();
        let mut tuned = original.clone();
        tuned.transition_stiffness = original.transition_stiffness + 0.2;
        tuned.default_duration_ms = original.default_duration_ms + 25;
        tuned_table.insert("i".to_string(), tuned);

        let ps = PhoneString {
            phones: vec![Phone::new_ipa("i")],
        };
        let baseline = phone_render_targets_from_string(&ps, Some(150.0), 0.7, &default_table);
        let tuned_render = phone_render_targets_from_string(&ps, Some(150.0), 0.7, &tuned_table);

        assert_eq!(baseline[0].filter, tuned_render[0].filter);
        assert_eq!(baseline[0].source, tuned_render[0].source);
        assert_ne!(baseline[0].duration_ms, tuned_render[0].duration_ms);
    }
}
