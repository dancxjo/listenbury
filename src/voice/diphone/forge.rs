//! Neural diphone forging from the Riper/Piper ONNX backend.
//!
//! The forge synthesizes carefully-designed carrier phoneme sequences, extracts
//! the target diphone region, and normalizes the result into a [`DiphoneUnit`].
//!
//! # Segmentation caveats
//!
//! Piper ONNX does not expose per-phoneme boundary markers.  The forge uses a
//! simple energy-proportional heuristic to locate the transition midpoint.
//! Segmentation confidence is reported honestly; the renderer should treat
//! low-confidence units with the same caution as MBROLA boundary-fallback units.
//!
//! # Licensing
//!
//! Generated diphone units inherit the license constraints of the source ONNX
//! model.  Do not redistribute cache entries without checking whether the model
//! license permits redistribution of derived audio.

use std::io::Read as _;
use std::path::Path;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

use crate::mouth::riper::phoneme::{PiperPhoneme, PiperPhonemeSequence};
use crate::voice::mbrola::diphone_provider::{
    DiphoneKey, DiphoneUnit, DiphoneUnitMetadata, DiphoneUnitSource, ForgeProvenance,
};

use super::normalize::normalize_diphone;

/// Version tag for the carrier-selection strategy.
///
/// Increment this when the carrier logic changes so that old cache entries are
/// invalidated automatically.
pub const CARRIER_STRATEGY_VERSION: &str = "v1";

/// Version tag for the overall forge pipeline settings.
///
/// Increment this when noise/length/noise_w defaults change.
pub const FORGE_SETTINGS_VERSION: &str = "v1";

/// Version tag for the normalization algorithm.
pub const NORMALIZATION_VERSION: &str = "v1";

/// Broad phoneme class used to select a carrier context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhoneClass {
    Vowel,
    StopConsonant,
    FricativeConsonant,
    NasalConsonant,
    Liquid,
    Glide,
    Silence,
    Unknown,
}

impl PhoneClass {
    /// Classify an IPA/espeak phone symbol.
    pub fn of(symbol: &str) -> Self {
        match symbol {
            "_" | "#" | "pau" => Self::Silence,
            "i" | "ɪ" | "e" | "ɛ" | "æ" | "ɑ" | "ɔ" | "ʌ" | "ə" | "ɚ" | "u" | "ʊ" | "o" | "a"
            | "ø" | "y" | "ɐ" | "ɜ" | "ɞ" | "@" => Self::Vowel,
            "p" | "b" | "t" | "d" | "k" | "ɡ" | "g" | "ʔ" => Self::StopConsonant,
            "f" | "v" | "θ" | "ð" | "s" | "z" | "ʃ" | "ʒ" | "h" | "x" | "ç" => {
                Self::FricativeConsonant
            }
            "m" | "n" | "ŋ" | "ɲ" => Self::NasalConsonant,
            "l" | "ɫ" | "r" | "ɾ" | "ɹ" | "ɻ" | "ʀ" => Self::Liquid,
            "j" | "w" | "ɥ" => Self::Glide,
            _ => Self::Unknown,
        }
    }

    /// Return a neutral vowel carrier appropriate for this class.
    fn neutral_carrier(self) -> &'static str {
        match self {
            // Stops and fricatives need sonorant padding to avoid clipping
            Self::StopConsonant | Self::FricativeConsonant => "ə",
            Self::NasalConsonant => "ə",
            // Liquids and glides get a mid vowel
            Self::Liquid | Self::Glide => "ə",
            // Vowels and unknowns use schwa too
            _ => "ə",
        }
    }
}

/// Settings controlling how the forge synthesizes and extracts diphones.
#[derive(Debug, Clone, PartialEq)]
pub struct ForgeSettings {
    /// Minimum number of samples required for a valid extraction.
    pub min_samples: usize,
    /// Fraction of the total output used as a guard region before the target.
    pub guard_fraction: f32,
}

impl Default for ForgeSettings {
    fn default() -> Self {
        Self {
            min_samples: 32,
            guard_fraction: 0.25,
        }
    }
}

/// A forged diphone unit with its segmentation metadata.
#[derive(Debug)]
pub struct ForgedUnit {
    pub unit: DiphoneUnit,
    pub carrier_sequence: Vec<String>,
    pub segmentation_confidence: f32,
}

/// Synthesize a diphone `(left, right)` from the Riper backend.
///
/// The forge:
/// 1. Builds a carrier sequence: `_ / carrier_vowel / left / right / carrier_vowel / _`
/// 2. Converts the carrier to Piper IDs.
/// 3. Runs ONNX inference.
/// 4. Extracts and normalizes the transition region.
///
/// # Feature gate
///
/// This function is only available when the `tts-riper` Cargo feature is enabled.
#[cfg(feature = "tts-riper")]
pub fn forge_diphone(
    backend: &mut crate::mouth::riper::backend::RiperBackend,
    left: &str,
    right: &str,
    settings: &ForgeSettings,
) -> Result<ForgedUnit> {
    let carrier = PhoneClass::of(left).neutral_carrier();
    let carrier_sequence: Vec<String> = vec![
        "_".into(),
        carrier.into(),
        left.into(),
        right.into(),
        carrier.into(),
        "_".into(),
    ];

    let phonemes: Vec<PiperPhoneme> = carrier_sequence
        .iter()
        .map(|s| PiperPhoneme(s.clone()))
        .collect();
    let sequence = PiperPhonemeSequence { phonemes };
    let ids = sequence
        .to_piper_ids_compatible(backend.config())
        .with_context(|| {
            format!(
                "failed to map carrier sequence {carrier_sequence:?} to Piper IDs for diphone {left}-{right}"
            )
        })?;

    let pcm = backend.synthesize_ids(&ids).with_context(|| {
        format!("Riper ONNX synthesis failed for carrier diphone {left}-{right}")
    })?;

    if pcm.samples.len() < settings.min_samples {
        bail!(
            "Riper produced only {} samples for diphone {left}-{right}; expected at least {}",
            pcm.samples.len(),
            settings.min_samples
        );
    }

    let model_fingerprint = fingerprint_path(backend.model_path());
    let config_fingerprint = fingerprint_config(backend.config());

    let (samples, halfseg_samples, confidence) =
        segment_diphone(&pcm.samples, left, right, settings);

    let mut samples = samples;
    normalize_diphone(&mut samples);

    let provenance = ForgeProvenance {
        model_fingerprint,
        config_fingerprint,
        carrier_sequence: carrier_sequence.clone(),
        segmentation_confidence: confidence,
        generated_at: now_iso8601(),
    };

    let unit = DiphoneUnit {
        key: DiphoneKey::new(left, right),
        samples,
        sample_rate_hz: pcm.sample_rate_hz,
        halfseg_samples,
        frame_center_samples: Vec::new(),
        source: DiphoneUnitSource::NeuralGenerated,
        metadata: DiphoneUnitMetadata {
            requested_key: None,
            warning: if confidence < 0.5 {
                Some(format!(
                    "low segmentation confidence {confidence:.2} for diphone {left}-{right}"
                ))
            } else {
                None
            },
            forge_provenance: Some(provenance),
        },
    };

    Ok(ForgedUnit {
        unit,
        carrier_sequence,
        segmentation_confidence: confidence,
    })
}

/// Segment the target diphone region from a synthesized carrier waveform.
///
/// Returns `(samples, halfseg_samples, confidence)`.
///
/// The segmentation is based on proportional position:  the carrier has the
/// shape `[_ carrier left right carrier _]` so we expect the target
/// transition to be in the middle half of the waveform.  Energy changes
/// within the target window refine the midpoint estimate.
fn segment_diphone(
    all_samples: &[f32],
    _left: &str,
    _right: &str,
    settings: &ForgeSettings,
) -> (Vec<f32>, usize, f32) {
    let total = all_samples.len();
    // Guard: skip the leading and trailing `guard_fraction` of the signal.
    let guard = ((total as f32 * settings.guard_fraction) as usize).max(1);
    let window_start = guard;
    let window_end = total.saturating_sub(guard).max(window_start + 1);

    let window = &all_samples[window_start..window_end];
    if window.len() < settings.min_samples {
        // Fallback: use the whole signal centre
        let half = total / 2;
        return (all_samples.to_vec(), half, 0.3);
    }

    // Estimate the join midpoint as the energy minimum within the window.
    let midpoint_in_window = energy_min_index(window, 8);

    // Confidence: ratio of the energy drop at the midpoint vs average energy.
    let confidence = energy_confidence(window, midpoint_in_window, 8);

    (window.to_vec(), midpoint_in_window, confidence)
}

/// Find the index of minimum local energy (using a small frame) in `samples`.
fn energy_min_index(samples: &[f32], frame_size: usize) -> usize {
    let n = samples.len();
    if n <= frame_size {
        return n / 2;
    }
    let mut min_energy = f32::MAX;
    let mut min_idx = n / 2;
    for i in 0..=(n - frame_size) {
        let e: f32 = samples[i..i + frame_size].iter().map(|s| s * s).sum();
        if e < min_energy {
            min_energy = e;
            min_idx = i + frame_size / 2;
        }
    }
    min_idx
}

/// Compute a confidence score based on how much the energy dips at `midpoint`.
fn energy_confidence(samples: &[f32], midpoint: usize, frame_size: usize) -> f32 {
    let avg_energy = {
        let e: f32 = samples.iter().map(|s| s * s).sum();
        e / samples.len() as f32
    };
    if avg_energy < 1e-10 {
        return 0.0;
    }
    let half_frame = (frame_size / 2).max(1);
    let start = midpoint.saturating_sub(half_frame);
    let end = (midpoint + half_frame).min(samples.len());
    let local_e: f32 = if start < end {
        samples[start..end].iter().map(|s| s * s).sum::<f32>() / (end - start) as f32
    } else {
        avg_energy
    };
    let ratio = 1.0 - (local_e / avg_energy).min(1.0);
    ratio.clamp(0.0, 1.0)
}

/// Fingerprint a model path as a short hex string (deterministic for the same path).
pub fn fingerprint_path(path: &Path) -> String {
    const LARGE_MODEL_THRESHOLD_BYTES: u64 = 128 * 1024 * 1024;

    let mut hasher = Sha256::new();
    let meta = std::fs::metadata(path);
    match meta {
        Ok(meta) if meta.len() <= LARGE_MODEL_THRESHOLD_BYTES => match std::fs::File::open(path) {
            Ok(mut f) => {
                let mut buf = [0_u8; 64 * 1024];
                loop {
                    match f.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => hasher.update(&buf[..n]),
                        Err(_) => break,
                    }
                }
            }
            Err(_) => {
                hasher.update(path.to_string_lossy().as_bytes());
            }
        },
        Ok(meta) => {
            // Cheaper fallback for very large models.
            hasher.update(path.to_string_lossy().as_bytes());
            let model_len_bytes: [u8; 8] = meta.len().to_le_bytes();
            hasher.update(model_len_bytes);
            if let Ok(modified) = meta.modified()
                && let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH)
            {
                hasher.update(duration.as_secs().to_le_bytes());
                hasher.update(duration.subsec_nanos().to_le_bytes());
            }
        }
        Err(_) => {
            hasher.update(path.to_string_lossy().as_bytes());
        }
    }
    hex_sha256_digest(hasher.finalize())
}

/// Fingerprint the voice config via its phoneme map size and sample rate.
pub fn fingerprint_config(config: &crate::mouth::riper::config::PiperVoiceConfig) -> String {
    let mut hasher = Sha256::new();
    hasher.update(config.sample_rate_hz.to_le_bytes());
    hasher.update(config.phoneme_id_map.len().to_le_bytes());

    // Hash sorted phoneme IDs for deterministic output.
    let mut phonemes: Vec<(&str, &Vec<i64>)> = config
        .phoneme_id_map
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect();
    phonemes.sort_by_key(|(k, _)| *k);
    for (k, v) in phonemes {
        hasher.update(k.as_bytes());
        hasher.update([0_u8]);
        hasher.update(v.len().to_le_bytes());
        for id in v {
            hasher.update(id.to_le_bytes());
        }
    }
    hex_sha256_digest(hasher.finalize())
}

fn now_iso8601() -> String {
    // Use a simple RFC 3339 format.  chrono is already a dependency.
    chrono::Utc::now().to_rfc3339()
}

fn hex_sha256_digest(digest: impl AsRef<[u8]>) -> String {
    let bytes = digest.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phone_class_classifies_correctly() {
        assert_eq!(PhoneClass::of("ə"), PhoneClass::Vowel);
        assert_eq!(PhoneClass::of("p"), PhoneClass::StopConsonant);
        assert_eq!(PhoneClass::of("s"), PhoneClass::FricativeConsonant);
        assert_eq!(PhoneClass::of("m"), PhoneClass::NasalConsonant);
        assert_eq!(PhoneClass::of("l"), PhoneClass::Liquid);
        assert_eq!(PhoneClass::of("j"), PhoneClass::Glide);
        assert_eq!(PhoneClass::of("_"), PhoneClass::Silence);
    }

    #[test]
    fn segment_diphone_returns_non_empty() {
        let samples: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let settings = ForgeSettings::default();
        let (extracted, halfseg, confidence) = segment_diphone(&samples, "p", "ae", &settings);
        assert!(!extracted.is_empty());
        assert!(halfseg <= extracted.len());
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }

    #[test]
    fn segment_diphone_rejects_too_short() {
        // When the window is too short, falls back to center with low confidence
        let samples = vec![0.0_f32; 8];
        let settings = ForgeSettings {
            min_samples: 64,
            ..Default::default()
        };
        let (extracted, _halfseg, confidence) = segment_diphone(&samples, "h", "@", &settings);
        assert!(!extracted.is_empty());
        assert!(confidence < 0.5);
    }

    #[test]
    fn fingerprint_path_is_stable() {
        let path = Path::new("/tmp/model.onnx");
        assert_eq!(fingerprint_path(path), fingerprint_path(path));
    }

    #[test]
    fn fingerprint_path_differs_for_different_paths() {
        let a = fingerprint_path(Path::new("/a/model.onnx"));
        let b = fingerprint_path(Path::new("/b/model.onnx"));
        assert_ne!(a, b);
    }
}
