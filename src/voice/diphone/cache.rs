//! Disk-backed cache for forged neural diphone units.
//!
//! Cache entries are stored as pairs:
//! - `<hex_key>.json` – provenance metadata (human-readable JSON)
//! - `<hex_key>.pcm`  – raw little-endian f32 samples
//!
//! The cache is intentionally not committed to version control by default.
//! Cache metadata records model/config provenance so stale entries can be
//! detected when the model or forge settings change.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::voice::mbrola::diphone_provider::{
    DiphoneKey, DiphoneUnit, DiphoneUnitMetadata, DiphoneUnitSource, ForgeProvenance,
};

/// All parameters that affect what a cached diphone unit sounds like.
///
/// A change in any field produces a different cache key, invalidating the
/// entry for the old parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheKey {
    /// Hex-encoded SHA-256 of the model file, or a stable model path tag.
    pub model_fingerprint: String,
    /// Hex-encoded SHA-256 of the voice config JSON.
    pub config_fingerprint: String,
    /// Speaker identifier (empty string for single-speaker models).
    pub speaker_id: String,
    /// Left phone symbol of the diphone.
    pub left: String,
    /// Right phone symbol of the diphone.
    pub right: String,
    /// Opaque version string that changes when the carrier strategy logic changes.
    pub carrier_strategy_version: String,
    /// Opaque version string that changes when the forge pipeline changes.
    pub forge_settings_version: String,
    /// Sample rate of the synthesized audio in Hz.
    pub sample_rate_hz: u32,
    /// Opaque version string that changes when the normalization algorithm changes.
    pub normalization_version: String,
}

impl CacheKey {
    /// Produce the SHA-256 filename stem used for this key's cache files.
    pub fn filename_stem(&self) -> String {
        let bytes = serde_json::to_vec(self).unwrap_or_else(|_| {
            format!(
                "{}|{}|{}|{}|{}|{}|{}|{}|{}",
                self.model_fingerprint,
                self.config_fingerprint,
                self.speaker_id,
                self.left,
                self.right,
                self.carrier_strategy_version,
                self.forge_settings_version,
                self.sample_rate_hz,
                self.normalization_version
            )
            .into_bytes()
        });
        hex_sha256(&bytes)
    }
}

/// Provenance metadata stored alongside each cached diphone unit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheEntryMetadata {
    /// The full cache key for this entry.
    pub key: CacheKey,
    /// ISO-8601 UTC timestamp when this entry was generated.
    pub generated_at: String,
    /// Carrier phoneme symbols fed to the neural model.
    pub carrier_sequence: Vec<String>,
    /// Half-segment boundary in samples (join midpoint).
    pub halfseg_samples: usize,
    /// Confidence score in [0.0, 1.0] from the boundary segmentation step.
    pub segmentation_confidence: f32,
    /// Number of samples in the extracted unit.
    pub sample_count: usize,
    /// Start (inclusive) of extracted sample range in the synthesized carrier.
    pub extraction_start_sample: usize,
    /// End (exclusive) of extracted sample range in the synthesized carrier.
    pub extraction_end_sample: usize,
    /// Model/source license identifier when known; `unknown` otherwise.
    pub model_license: String,
    /// A human-readable note about model/license constraints.
    pub provenance_note: String,
}

/// A disk-backed cache for generated neural diphone units.
///
/// The cache directory is created on first use.  No entry is committed to git
/// by default: add `diphone-cache/` to `.gitignore` at the project root.
#[derive(Debug, Clone)]
pub struct DiphoneCache {
    dir: PathBuf,
}

/// Result of attempting to read a cache entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheLookup {
    Hit,
    Miss,
    Corrupt { reason: String },
}

impl DiphoneCache {
    /// Open (or create) a cache rooted at `dir`.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir).with_context(|| {
            format!("failed to create diphone cache directory {}", dir.display())
        })?;
        Ok(Self { dir })
    }

    /// Look up a cached diphone unit for `key`.
    ///
    /// Returns `None` if no entry exists or the files are corrupt.
    pub fn get(&self, key: &CacheKey) -> Option<DiphoneUnit> {
        self.read(key).ok().flatten()
    }

    /// Read and validate a cached diphone unit.
    ///
    /// Missing entries return `Ok(None)`. Corrupt entries return `Err` with
    /// a clear reason describing the failed validation.
    pub fn read(&self, key: &CacheKey) -> Result<Option<DiphoneUnit>> {
        let stem = key.filename_stem();
        let meta_path = self.dir.join(format!("{stem}.json"));
        let pcm_path = self.dir.join(format!("{stem}.pcm"));

        if !meta_path.is_file() || !pcm_path.is_file() {
            return Ok(None);
        }

        let meta_bytes = std::fs::read(&meta_path)
            .with_context(|| format!("failed to read cache metadata {}", meta_path.display()))?;
        let meta: CacheEntryMetadata = serde_json::from_slice(&meta_bytes)
            .with_context(|| format!("invalid cache metadata JSON {}", meta_path.display()))?;
        if &meta.key != key {
            anyhow::bail!("cache metadata key mismatch for {}", stem);
        }

        let pcm_bytes = std::fs::read(&pcm_path)
            .with_context(|| format!("failed to read cache PCM {}", pcm_path.display()))?;
        let samples = bytes_to_f32_samples(&pcm_bytes)
            .with_context(|| format!("invalid cache PCM bytes {}", pcm_path.display()))?;

        if samples.len() != meta.sample_count {
            anyhow::bail!(
                "sample_count mismatch for {}: metadata={}, pcm={}",
                stem,
                meta.sample_count,
                samples.len()
            );
        }
        if meta.halfseg_samples > samples.len() {
            anyhow::bail!(
                "halfseg out of bounds for {}: halfseg={}, samples={}",
                stem,
                meta.halfseg_samples,
                samples.len()
            );
        }
        if meta.extraction_start_sample > meta.extraction_end_sample {
            anyhow::bail!(
                "invalid extraction range for {}: start={} > end={}",
                stem,
                meta.extraction_start_sample,
                meta.extraction_end_sample
            );
        }
        if meta.extraction_end_sample < meta.extraction_start_sample + samples.len() {
            anyhow::bail!(
                "extraction range too short for {}: start={}, end={}, sample_count={}",
                stem,
                meta.extraction_start_sample,
                meta.extraction_end_sample,
                samples.len()
            );
        }

        Ok(Some(DiphoneUnit {
            key: DiphoneKey::new(&meta.key.left, &meta.key.right),
            samples,
            sample_rate_hz: meta.key.sample_rate_hz,
            halfseg_samples: meta.halfseg_samples,
            frame_center_samples: Vec::new(),
            source: DiphoneUnitSource::CacheHit,
            metadata: DiphoneUnitMetadata {
                requested_key: None,
                warning: None,
                forge_provenance: Some(ForgeProvenance {
                    model_fingerprint: meta.key.model_fingerprint.clone(),
                    config_fingerprint: meta.key.config_fingerprint.clone(),
                    carrier_sequence: meta.carrier_sequence.clone(),
                    segmentation_confidence: meta.segmentation_confidence,
                    generated_at: meta.generated_at.clone(),
                }),
            },
        }))
    }

    /// Look up cache state with explicit diagnostics.
    pub fn lookup_state(&self, key: &CacheKey) -> CacheLookup {
        let stem = key.filename_stem();
        let meta_path = self.dir.join(format!("{stem}.json"));
        let pcm_path = self.dir.join(format!("{stem}.pcm"));
        if !meta_path.is_file() || !pcm_path.is_file() {
            return CacheLookup::Miss;
        }
        match self.read(key) {
            Ok(Some(_)) => CacheLookup::Hit,
            Ok(None) => CacheLookup::Miss,
            Err(err) => CacheLookup::Corrupt {
                reason: err.to_string(),
            },
        }
    }

    /// Store a diphone unit in the cache, writing both `.json` and `.pcm` files.
    pub fn store(
        &self,
        key: &CacheKey,
        unit: &DiphoneUnit,
        meta: CacheEntryMetadata,
    ) -> Result<()> {
        if &meta.key != key {
            anyhow::bail!("metadata key does not match provided cache key");
        }
        if unit.sample_rate_hz != key.sample_rate_hz {
            anyhow::bail!(
                "unit sample rate {} does not match key sample rate {}",
                unit.sample_rate_hz,
                key.sample_rate_hz
            );
        }
        if unit.key.left != key.left || unit.key.right != key.right {
            anyhow::bail!(
                "unit phone key {}-{} does not match cache key {}-{}",
                unit.key.left,
                unit.key.right,
                key.left,
                key.right
            );
        }
        if unit.samples.len() != meta.sample_count {
            anyhow::bail!(
                "metadata sample_count {} does not match unit sample count {}",
                meta.sample_count,
                unit.samples.len()
            );
        }
        if meta.extraction_start_sample > meta.extraction_end_sample {
            anyhow::bail!(
                "invalid extraction range: start={} > end={}",
                meta.extraction_start_sample,
                meta.extraction_end_sample
            );
        }
        if meta.model_license.trim().is_empty() {
            anyhow::bail!("model_license must be set (use \"unknown\" if unavailable)");
        }
        let stem = key.filename_stem();
        let meta_path = self.dir.join(format!("{stem}.json"));
        let pcm_path = self.dir.join(format!("{stem}.pcm"));

        let meta_json =
            serde_json::to_string_pretty(&meta).context("failed to serialize cache metadata")?;
        std::fs::write(&meta_path, meta_json.as_bytes()).with_context(|| {
            format!("failed to write cache metadata to {}", meta_path.display())
        })?;

        let pcm_bytes = f32_samples_to_bytes(&unit.samples);
        std::fs::write(&pcm_path, &pcm_bytes)
            .with_context(|| format!("failed to write cache PCM to {}", pcm_path.display()))?;

        Ok(())
    }

    /// Return the directory this cache is rooted at.
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn f32_samples_to_bytes(samples: &[f32]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

fn bytes_to_f32_samples(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    let samples = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Some(samples)
}

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key(left: &str, right: &str) -> CacheKey {
        CacheKey {
            model_fingerprint: "abc123".to_string(),
            config_fingerprint: "def456".to_string(),
            speaker_id: String::new(),
            left: left.to_string(),
            right: right.to_string(),
            carrier_strategy_version: "v1".to_string(),
            forge_settings_version: "v1".to_string(),
            sample_rate_hz: 22050,
            normalization_version: "v1".to_string(),
        }
    }

    #[test]
    fn cache_key_stem_is_stable() {
        let key = test_key("h", "@");
        let stem1 = key.filename_stem();
        let stem2 = key.filename_stem();
        assert_eq!(stem1, stem2, "cache key stem must be deterministic");
    }

    #[test]
    fn cache_key_changes_when_model_changes() {
        let key1 = test_key("h", "@");
        let mut key2 = key1.clone();
        key2.model_fingerprint = "different_model".to_string();
        assert_ne!(
            key1.filename_stem(),
            key2.filename_stem(),
            "cache key must change when model fingerprint changes"
        );
    }

    #[test]
    fn cache_key_changes_when_phones_change() {
        let key1 = test_key("h", "@");
        let key2 = test_key("h", "i");
        assert_ne!(
            key1.filename_stem(),
            key2.filename_stem(),
            "cache key must change when phones change"
        );
    }

    #[test]
    fn cache_roundtrip() {
        let dir =
            std::env::temp_dir().join(format!("listenbury_diphone_test_{}", std::process::id()));
        let cache = DiphoneCache::open(&dir).expect("open cache");
        let key = test_key("p", "ae");

        let unit = DiphoneUnit {
            key: DiphoneKey::new("p", "ae"),
            samples: vec![0.1, -0.2, 0.05, 0.0],
            sample_rate_hz: 22050,
            halfseg_samples: 2,
            frame_center_samples: Vec::new(),
            source: DiphoneUnitSource::NeuralGenerated,
            metadata: DiphoneUnitMetadata::default(),
        };

        let meta = CacheEntryMetadata {
            key: key.clone(),
            generated_at: "2026-01-01T00:00:00Z".to_string(),
            carrier_sequence: vec![
                "_".into(),
                "ax".into(),
                "p".into(),
                "ae".into(),
                "ax".into(),
                "_".into(),
            ],
            halfseg_samples: 2,
            segmentation_confidence: 0.75,
            sample_count: unit.samples.len(),
            extraction_start_sample: 64,
            extraction_end_sample: 64 + unit.samples.len(),
            model_license: "unknown".to_string(),
            provenance_note: "test".to_string(),
        };

        cache.store(&key, &unit, meta).expect("store unit");
        let retrieved = cache.read(&key).expect("read unit").expect("get unit");

        assert_eq!(retrieved.samples, unit.samples);
        assert_eq!(retrieved.halfseg_samples, unit.halfseg_samples);
        assert_eq!(retrieved.source, DiphoneUnitSource::CacheHit);
    }

    #[test]
    fn cache_miss_returns_none() {
        let dir = std::env::temp_dir().join(format!(
            "listenbury_diphone_test_miss_{}",
            std::process::id()
        ));
        let cache = DiphoneCache::open(&dir).expect("open cache");
        let key = test_key("z", "z");
        assert!(cache.read(&key).expect("read").is_none());
    }

    #[test]
    fn f32_roundtrip() {
        let original = vec![0.0_f32, 0.5, -0.5, 1.0, -1.0];
        let bytes = f32_samples_to_bytes(&original);
        let recovered = bytes_to_f32_samples(&bytes).expect("roundtrip");
        assert_eq!(original, recovered);
    }

    #[test]
    fn cache_key_changes_when_config_changes() {
        let key1 = test_key("h", "@");
        let mut key2 = key1.clone();
        key2.config_fingerprint = "different_config".to_string();
        assert_ne!(key1.filename_stem(), key2.filename_stem());
    }

    #[test]
    fn cache_key_changes_when_versions_change() {
        let key1 = test_key("h", "@");
        let mut key2 = key1.clone();
        key2.normalization_version = "v2".to_string();
        assert_ne!(key1.filename_stem(), key2.filename_stem());

        let mut key3 = key1.clone();
        key3.forge_settings_version = "v2".to_string();
        assert_ne!(key1.filename_stem(), key3.filename_stem());
    }

    #[test]
    fn cache_rejects_mismatched_metadata_key() {
        let dir = std::env::temp_dir().join(format!(
            "listenbury_diphone_test_mismatch_{}",
            std::process::id()
        ));
        let cache = DiphoneCache::open(&dir).expect("open cache");
        let key = test_key("p", "ae");
        let unit = DiphoneUnit {
            key: DiphoneKey::new("p", "ae"),
            samples: vec![0.1, -0.2, 0.05, 0.0],
            sample_rate_hz: 22050,
            halfseg_samples: 2,
            frame_center_samples: Vec::new(),
            source: DiphoneUnitSource::NeuralGenerated,
            metadata: DiphoneUnitMetadata::default(),
        };
        let mut bad_meta_key = key.clone();
        bad_meta_key.right = "ih".to_string();
        let meta = CacheEntryMetadata {
            key: bad_meta_key,
            generated_at: "2026-01-01T00:00:00Z".to_string(),
            carrier_sequence: vec![
                "_".into(),
                "ax".into(),
                "p".into(),
                "ae".into(),
                "ax".into(),
                "_".into(),
            ],
            halfseg_samples: 2,
            segmentation_confidence: 0.75,
            sample_count: unit.samples.len(),
            extraction_start_sample: 0,
            extraction_end_sample: unit.samples.len(),
            model_license: "unknown".to_string(),
            provenance_note: "test".to_string(),
        };

        cache
            .store(&key, &unit, meta)
            .expect_err("must reject key mismatch");
    }

    #[test]
    fn cache_rejects_corrupt_pcm() {
        let dir = std::env::temp_dir().join(format!(
            "listenbury_diphone_test_corrupt_{}",
            std::process::id()
        ));
        let cache = DiphoneCache::open(&dir).expect("open cache");
        let key = test_key("p", "ae");
        let unit = DiphoneUnit {
            key: DiphoneKey::new("p", "ae"),
            samples: vec![0.1, -0.2, 0.05, 0.0],
            sample_rate_hz: 22050,
            halfseg_samples: 2,
            frame_center_samples: Vec::new(),
            source: DiphoneUnitSource::NeuralGenerated,
            metadata: DiphoneUnitMetadata::default(),
        };
        let meta = CacheEntryMetadata {
            key: key.clone(),
            generated_at: "2026-01-01T00:00:00Z".to_string(),
            carrier_sequence: vec![
                "_".into(),
                "ax".into(),
                "p".into(),
                "ae".into(),
                "ax".into(),
                "_".into(),
            ],
            halfseg_samples: 2,
            segmentation_confidence: 0.75,
            sample_count: unit.samples.len(),
            extraction_start_sample: 0,
            extraction_end_sample: unit.samples.len(),
            model_license: "unknown".to_string(),
            provenance_note: "test".to_string(),
        };
        cache.store(&key, &unit, meta).expect("store unit");

        let stem = key.filename_stem();
        let pcm_path = cache.dir().join(format!("{stem}.pcm"));
        std::fs::write(&pcm_path, [0_u8, 1_u8, 2_u8]).expect("corrupt pcm");

        let state = cache.lookup_state(&key);
        assert!(matches!(state, CacheLookup::Corrupt { .. }));
    }

    #[test]
    fn cache_missing_sidecar_is_miss() {
        let dir = std::env::temp_dir().join(format!(
            "listenbury_diphone_test_missing_sidecar_{}",
            std::process::id()
        ));
        let cache = DiphoneCache::open(&dir).expect("open cache");
        let key = test_key("p", "ae");
        let stem = key.filename_stem();
        let pcm_path = cache.dir().join(format!("{stem}.pcm"));
        let samples = vec![0.1_f32, -0.2, 0.05, 0.0];
        std::fs::write(&pcm_path, f32_samples_to_bytes(&samples)).expect("write pcm only");

        let state = cache.lookup_state(&key);
        assert!(matches!(state, CacheLookup::Miss));
    }
}
