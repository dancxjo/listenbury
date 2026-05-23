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

use crate::voice::mbrola::diphone_provider::{
    DiphoneKey, DiphoneUnit, DiphoneUnitMetadata, DiphoneUnitSource, ForgeProvenance,
};

use super::normalize::{NormalizationReport, normalize_diphone};

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

/// An ordered list of phoneme symbols fed to the neural model as a carrier context.
///
/// The carrier wraps the target diphone pair with vowel padding to ensure stable
/// synthesis of boundary material.  Typical shape: `[_, vowel, left, right, vowel, _]`.
pub type CarrierSequence = Vec<String>;

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

/// Report from the diphone boundary segmentation step.
///
/// Provides an honest account of what the segmentation heuristic found so
/// downstream consumers (renderers, cache writers, CLI tools) can make
/// informed decisions about unit quality.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentationReport {
    /// Confidence score in [0.0, 1.0].
    ///
    /// Below 0.5 indicates unreliable boundary detection; the unit should be
    /// used with caution or regenerated with different carrier parameters.
    pub confidence: f32,
    /// Human-readable warnings produced during segmentation (empty if clean).
    pub warnings: Vec<String>,
    /// Start sample (inclusive) of the extracted window within the full carrier output.
    pub source_start_sample: usize,
    /// End sample (exclusive) of the extracted window within the full carrier output.
    pub source_end_sample: usize,
    /// Estimated join/half-segment point within the *extracted* window (not carrier).
    pub halfseg_samples: usize,
}

/// A forged diphone unit with attached segmentation and normalization reports.
#[derive(Debug)]
pub struct ForgedUnit {
    pub unit: DiphoneUnit,
    pub carrier_sequence: CarrierSequence,
    pub segmentation: SegmentationReport,
    pub normalization: NormalizationReport,
    /// Convenience accessor: segmentation confidence (same as `segmentation.confidence`).
    pub segmentation_confidence: f32,
}

/// Build the carrier phoneme sequence for the given `(left, right)` diphone.
///
/// The sequence has the shape `[_, vowel, left, right, vowel, _]` where the
/// vowel is chosen based on the phone class of `left`.  This is intentionally
/// conservative: all classes currently use schwa, but the function is
/// structured so phone-class-specific carriers can be added.
///
/// This function is pure (no synthesis) and is fully testable without ONNX.
pub fn build_carrier_sequence(left: &str, right: &str) -> CarrierSequence {
    let carrier = PhoneClass::of(left).neutral_carrier();
    vec![
        "_".into(),
        carrier.into(),
        left.into(),
        right.into(),
        carrier.into(),
        "_".into(),
    ]
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
    use crate::mouth::riper::phoneme::{PiperPhoneme, PiperPhonemeSequence};

    let carrier_sequence = build_carrier_sequence(left, right);

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

    forge_from_samples(
        left,
        right,
        &pcm.samples,
        pcm.sample_rate_hz,
        &carrier_sequence,
        &model_fingerprint,
        &config_fingerprint,
        settings,
    )
}

/// Forge a diphone unit from pre-synthesized PCM samples.
///
/// This is the core extraction pipeline: it runs segmentation, normalization,
/// and assembles the [`ForgedUnit`].  Because it accepts raw PCM it is fully
/// testable without a real ONNX model.
///
/// # Arguments
///
/// * `left`, `right` – phone symbols for the diphone key.
/// * `samples` – raw PCM from the carrier synthesis (f32 normalized to [-1, 1]).
/// * `sample_rate_hz` – sample rate of the synthesis output.
/// * `carrier_sequence` – the carrier symbol list that produced `samples`.
/// * `model_fingerprint` – hex fingerprint of the source model (use empty string for tests).
/// * `config_fingerprint` – hex fingerprint of the voice config (use empty string for tests).
/// * `settings` – forge settings.
pub fn forge_from_samples(
    left: &str,
    right: &str,
    samples: &[f32],
    sample_rate_hz: u32,
    carrier_sequence: &[String],
    model_fingerprint: &str,
    config_fingerprint: &str,
    settings: &ForgeSettings,
) -> Result<ForgedUnit> {
    if samples.len() < settings.min_samples {
        bail!(
            "input has only {} samples for diphone {left}-{right}; expected at least {}",
            samples.len(),
            settings.min_samples
        );
    }

    let (mut extracted, seg_report) = segment_diphone(samples, left, right, settings);

    let norm_report = normalize_diphone(&mut extracted);

    let confidence = seg_report.confidence;
    let halfseg_samples = seg_report.halfseg_samples;

    let provenance = ForgeProvenance {
        model_fingerprint: model_fingerprint.to_string(),
        config_fingerprint: config_fingerprint.to_string(),
        carrier_sequence: carrier_sequence.to_vec(),
        segmentation_confidence: confidence,
        generated_at: now_iso8601(),
    };

    let unit = DiphoneUnit {
        key: DiphoneKey::new(left, right),
        samples: extracted,
        sample_rate_hz,
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
        carrier_sequence: carrier_sequence.to_vec(),
        segmentation_confidence: confidence,
        segmentation: seg_report,
        normalization: norm_report,
    })
}

/// Segment the target diphone region from a synthesized carrier waveform.
///
/// Returns `(extracted_samples, SegmentationReport)`.
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
) -> (Vec<f32>, SegmentationReport) {
    let total = all_samples.len();
    // Guard: skip the leading and trailing `guard_fraction` of the signal.
    let guard = ((total as f32 * settings.guard_fraction) as usize).max(1);
    let window_start = guard;
    let window_end = total.saturating_sub(guard).max(window_start + 1);

    let window = &all_samples[window_start..window_end];
    if window.len() < settings.min_samples {
        // Fallback: use the whole signal centre
        let half = total / 2;
        let report = SegmentationReport {
            confidence: 0.3,
            warnings: vec![format!(
                "window too short ({} < {}); fell back to full signal",
                window.len(),
                settings.min_samples
            )],
            source_start_sample: 0,
            source_end_sample: total,
            halfseg_samples: half,
        };
        return (all_samples.to_vec(), report);
    }

    // Estimate the join midpoint as the energy minimum within the window.
    let midpoint_in_window = energy_min_index(window, 8);

    // Confidence: ratio of the energy drop at the midpoint vs average energy.
    let confidence = energy_confidence(window, midpoint_in_window, 8);

    let mut warnings = Vec::new();
    if confidence < 0.5 {
        warnings.push(format!(
            "low boundary confidence {confidence:.2}; segmentation heuristic may be unreliable"
        ));
    }

    let report = SegmentationReport {
        confidence,
        warnings,
        source_start_sample: window_start,
        source_end_sample: window_end,
        halfseg_samples: midpoint_in_window,
    };

    (window.to_vec(), report)
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
#[cfg(feature = "tts-riper")]
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
    fn build_carrier_sequence_has_correct_shape() {
        let seq = build_carrier_sequence("p", "ae");
        assert_eq!(seq.len(), 6);
        assert_eq!(seq[0], "_");
        assert_eq!(seq[2], "p");
        assert_eq!(seq[3], "ae");
        assert_eq!(seq[5], "_");
    }

    #[test]
    fn build_carrier_sequence_uses_schwa_for_stops() {
        let seq = build_carrier_sequence("p", "ae");
        assert_eq!(seq[1], "ə");
        assert_eq!(seq[4], "ə");
    }

    #[test]
    fn build_carrier_sequence_uses_schwa_for_fricatives() {
        let seq = build_carrier_sequence("s", "i");
        assert_eq!(seq[1], "ə");
        assert_eq!(seq[4], "ə");
    }

    #[test]
    fn segment_diphone_returns_non_empty() {
        let samples: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let settings = ForgeSettings::default();
        let (extracted, report) = segment_diphone(&samples, "p", "ae", &settings);
        assert!(!extracted.is_empty());
        assert!(report.halfseg_samples <= extracted.len());
        assert!(report.confidence >= 0.0 && report.confidence <= 1.0);
        assert!(report.source_end_sample > report.source_start_sample);
    }

    #[test]
    fn segment_diphone_rejects_too_short() {
        // When the window is too short, falls back to center with low confidence
        let samples = vec![0.0_f32; 8];
        let settings = ForgeSettings {
            min_samples: 64,
            ..Default::default()
        };
        let (extracted, report) = segment_diphone(&samples, "h", "@", &settings);
        assert!(!extracted.is_empty());
        assert!(report.confidence < 0.5);
        assert!(!report.warnings.is_empty());
    }

    #[test]
    fn forge_from_samples_produces_valid_unit() {
        // Test the full pipeline without ONNX using synthetic PCM.
        let samples: Vec<f32> = (0..512).map(|i| (i as f32 * 0.05).sin()).collect();
        let carrier = build_carrier_sequence("p", "ae");
        let result = forge_from_samples(
            "p",
            "ae",
            &samples,
            22050,
            &carrier,
            "test_model_fp",
            "test_config_fp",
            &ForgeSettings::default(),
        );
        let forged = result.expect("forge_from_samples should succeed");
        assert_eq!(forged.unit.key.left, "p");
        assert_eq!(forged.unit.key.right, "ae");
        assert_eq!(forged.unit.sample_rate_hz, 22050);
        assert_eq!(forged.unit.source, DiphoneUnitSource::NeuralGenerated);
        assert!(!forged.unit.samples.is_empty());
        assert!(forged.unit.halfseg_samples <= forged.unit.samples.len());
        assert!(forged.segmentation.confidence >= 0.0);
        assert!(forged.segmentation.source_end_sample > forged.segmentation.source_start_sample);
    }

    #[test]
    fn forge_from_samples_rejects_too_short() {
        let samples = vec![0.0_f32; 8];
        let carrier = build_carrier_sequence("p", "ae");
        let result = forge_from_samples(
            "p",
            "ae",
            &samples,
            22050,
            &carrier,
            "",
            "",
            &ForgeSettings {
                min_samples: 64,
                ..Default::default()
            },
        );
        assert!(result.is_err(), "should reject too-short input");
    }

    #[test]
    fn forge_from_samples_rejects_all_silence() {
        // All-silence input → energy_confidence returns 0.0 (below threshold)
        let samples = vec![0.0_f32; 512];
        let carrier = build_carrier_sequence("p", "ae");
        let forged = forge_from_samples(
            "p",
            "ae",
            &samples,
            22050,
            &carrier,
            "",
            "",
            &ForgeSettings::default(),
        )
        .expect("pipeline itself succeeds");
        // All-silence → confidence should be 0; warning should be set on unit
        assert_eq!(forged.segmentation.confidence, 0.0);
        assert!(forged.unit.metadata.warning.is_some());
    }

    #[test]
    fn forge_from_samples_unit_has_provenance() {
        let samples: Vec<f32> = (0..512).map(|i| (i as f32 * 0.05).sin()).collect();
        let carrier = build_carrier_sequence("h", "@");
        let forged = forge_from_samples(
            "h",
            "@",
            &samples,
            22050,
            &carrier,
            "myfp",
            "mycfp",
            &ForgeSettings::default(),
        )
        .expect("forge should succeed");
        let prov = forged
            .unit
            .metadata
            .forge_provenance
            .expect("provenance must be set");
        assert_eq!(prov.model_fingerprint, "myfp");
        assert_eq!(prov.config_fingerprint, "mycfp");
        assert_eq!(prov.carrier_sequence, carrier);
        assert!(!prov.generated_at.is_empty());
    }

    #[test]
    fn forge_from_samples_normalization_removes_dc() {
        // Build samples with a significant DC offset
        let samples: Vec<f32> = (0..512).map(|i| 2.0 + (i as f32 * 0.05).sin()).collect();
        let carrier = build_carrier_sequence("m", "ae");
        let forged = forge_from_samples(
            "m",
            "ae",
            &samples,
            22050,
            &carrier,
            "",
            "",
            &ForgeSettings::default(),
        )
        .expect("forge should succeed");
        // DC offset should have been reported
        assert!(forged.normalization.dc_offset_removed.abs() > 0.5);
        // Final mean should be near zero
        let mean: f32 = forged.unit.samples.iter().sum::<f32>() / forged.unit.samples.len() as f32;
        assert!(
            mean.abs() < 0.05,
            "mean should be near zero after DC removal, got {mean}"
        );
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
