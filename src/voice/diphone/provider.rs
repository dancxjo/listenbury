//! [`DiphoneProvider`] implementations that serve neural and cached diphones.
//!
//! # Providers
//!
//! - [`NeuralDiphoneProvider`] – checks the on-disk cache first; on a miss it
//!   invokes the [`DiphoneForge`] to synthesize a fresh unit, stores it, and
//!   returns it.  Requires the `tts-riper` feature.
//! - [`FallbackDiphoneProvider`] – wraps a *primary* provider and a *secondary*
//!   provider.  On a primary miss it tries the secondary.  This lets you layer
//!   "MBROLA database" over "neural generated" so that MBROLA exact units are
//!   used when available and neural units fill the gaps.

use anyhow::Result;

use crate::voice::mbrola::diphone_provider::{DiphoneLookup, DiphoneProvider};

use super::cache::{CacheEntryMetadata, CacheKey, DiphoneCache};
use super::forge::{
    CARRIER_STRATEGY_VERSION, FORGE_SETTINGS_VERSION, ForgeSettings, NORMALIZATION_VERSION,
    fingerprint_config, fingerprint_path,
};

// ── NeuralDiphoneProvider ─────────────────────────────────────────────────────

/// A [`DiphoneProvider`] that serves neural cached/generated diphone units.
///
/// On each `get_diphone` call it:
/// 1. Constructs a [`CacheKey`] for the requested diphone.
/// 2. Tries to load an existing cache entry.
/// 3. On a cache miss, forges the diphone via the Riper backend, normalizes it,
///    writes it to the cache, and returns it with source `NeuralGenerated`.
/// 4. On a cache hit, returns the loaded unit with source `CacheHit`.
///
/// # Feature gate
///
/// The `tts-riper` Cargo feature must be enabled to compile this type.
#[cfg(feature = "tts-riper")]
pub struct NeuralDiphoneProvider {
    backend: crate::mouth::riper::backend::RiperBackend,
    cache: DiphoneCache,
    settings: ForgeSettings,
    speaker_id: String,
}

#[cfg(feature = "tts-riper")]
impl NeuralDiphoneProvider {
    /// Create a new provider backed by `backend` and `cache`.
    pub fn new(backend: crate::mouth::riper::backend::RiperBackend, cache: DiphoneCache) -> Self {
        Self {
            backend,
            cache,
            settings: ForgeSettings::default(),
            speaker_id: String::new(),
        }
    }

    /// Override the default [`ForgeSettings`].
    pub fn with_settings(mut self, settings: ForgeSettings) -> Self {
        self.settings = settings;
        self
    }

    /// Set the speaker identifier embedded in cache keys.
    pub fn with_speaker_id(mut self, speaker_id: impl Into<String>) -> Self {
        self.speaker_id = speaker_id.into();
        self
    }

    fn cache_key(&self, left: &str, right: &str) -> CacheKey {
        CacheKey {
            model_fingerprint: fingerprint_path(self.backend.model_path()),
            config_fingerprint: fingerprint_config(self.backend.config()),
            speaker_id: self.speaker_id.clone(),
            left: left.to_string(),
            right: right.to_string(),
            carrier_strategy_version: CARRIER_STRATEGY_VERSION.to_string(),
            forge_settings_version: FORGE_SETTINGS_VERSION.to_string(),
            sample_rate_hz: self.backend.config().sample_rate_hz,
            normalization_version: NORMALIZATION_VERSION.to_string(),
        }
    }
}

#[cfg(feature = "tts-riper")]
impl DiphoneProvider for NeuralDiphoneProvider {
    fn get_diphone(&mut self, left: &str, right: &str) -> Result<DiphoneLookup> {
        let key = self.cache_key(left, right);

        // Cache hit
        if let Some(unit) = self.cache.get(&key) {
            return Ok(DiphoneLookup { unit });
        }

        // Cache miss – forge a new unit
        let forged = super::forge::forge_diphone(&mut self.backend, left, right, &self.settings)
            .with_context(|| format!("failed to forge diphone {left}-{right}"))?;

        let meta = CacheEntryMetadata {
            key: key.clone(),
            generated_at: forged
                .unit
                .metadata
                .forge_provenance
                .as_ref()
                .map(|p| p.generated_at.clone())
                .unwrap_or_default(),
            carrier_sequence: forged.carrier_sequence.clone(),
            halfseg_samples: forged.unit.halfseg_samples,
            segmentation_confidence: forged.segmentation_confidence,
            sample_count: forged.unit.samples.len(),
            extraction_start_sample: forged.segmentation.source_start_sample,
            extraction_end_sample: forged.segmentation.source_end_sample,
            model_license: "unknown".to_string(),
            provenance_note: concat!(
                "Generated locally from a Piper ONNX model. ",
                "Do not redistribute without checking the model license."
            )
            .to_string(),
        };

        self.cache
            .store(&key, &forged.unit, meta)
            .with_context(|| format!("failed to cache forged diphone {left}-{right}"))?;

        Ok(DiphoneLookup { unit: forged.unit })
    }
}

#[cfg(feature = "tts-riper")]
use anyhow::Context as _;

// ── FallbackDiphoneProvider ───────────────────────────────────────────────────

/// A [`DiphoneProvider`] that tries a primary provider and falls back to a
/// secondary when the primary returns an error.
///
/// Typical usage: wrap an [`MbrolaDiphoneProvider`] as primary and a
/// [`NeuralDiphoneProvider`] as secondary so that MBROLA exact units are used
/// when available and neural units fill missing entries.
///
/// [`MbrolaDiphoneProvider`]: crate::voice::mbrola::diphone_provider::MbrolaDiphoneProvider
pub struct FallbackDiphoneProvider<P, F> {
    primary: P,
    fallback: F,
}

impl<P, F> FallbackDiphoneProvider<P, F>
where
    P: DiphoneProvider,
    F: DiphoneProvider,
{
    /// Create a new fallback provider.
    pub fn new(primary: P, fallback: F) -> Self {
        Self { primary, fallback }
    }
}

impl<P, F> DiphoneProvider for FallbackDiphoneProvider<P, F>
where
    P: DiphoneProvider,
    F: DiphoneProvider,
{
    fn get_diphone(&mut self, left: &str, right: &str) -> Result<DiphoneLookup> {
        self.primary
            .get_diphone(left, right)
            .or_else(|_| self.fallback.get_diphone(left, right))
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use anyhow::anyhow;

    use crate::voice::mbrola::diphone_provider::{
        DiphoneKey, DiphoneLookup, DiphoneProvider, DiphoneUnit, DiphoneUnitMetadata,
        DiphoneUnitSource,
    };

    use super::FallbackDiphoneProvider;

    // ── helpers ───────────────────────────────────────────────────────────────

    struct MapProvider(BTreeMap<(String, String), DiphoneUnit>);

    impl DiphoneProvider for MapProvider {
        fn get_diphone(&mut self, left: &str, right: &str) -> anyhow::Result<DiphoneLookup> {
            self.0
                .get(&(left.to_string(), right.to_string()))
                .cloned()
                .map(|unit| DiphoneLookup { unit })
                .ok_or_else(|| anyhow!("missing diphone {left}-{right}"))
        }
    }

    fn make_unit(left: &str, right: &str, source: DiphoneUnitSource) -> DiphoneUnit {
        DiphoneUnit {
            key: DiphoneKey::new(left, right),
            samples: vec![0.1, 0.2, 0.3],
            sample_rate_hz: 16_000,
            halfseg_samples: 1,
            frame_center_samples: Vec::new(),
            source,
            metadata: DiphoneUnitMetadata::default(),
        }
    }

    // ── FallbackDiphoneProvider tests ─────────────────────────────────────────

    #[test]
    fn fallback_uses_primary_when_available() {
        let mut primary = BTreeMap::new();
        primary.insert(
            ("h".into(), "@".into()),
            make_unit("h", "@", DiphoneUnitSource::MbrolaExact),
        );
        let secondary = BTreeMap::new();
        let mut provider =
            FallbackDiphoneProvider::new(MapProvider(primary), MapProvider(secondary));
        let lookup = provider.get_diphone("h", "@").expect("should succeed");
        assert_eq!(lookup.unit.source, DiphoneUnitSource::MbrolaExact);
    }

    #[test]
    fn fallback_falls_back_to_secondary() {
        let primary = BTreeMap::new();
        let mut secondary = BTreeMap::new();
        secondary.insert(
            ("h".into(), "@".into()),
            make_unit("h", "@", DiphoneUnitSource::NeuralGenerated),
        );
        let mut provider =
            FallbackDiphoneProvider::new(MapProvider(primary), MapProvider(secondary));
        let lookup = provider.get_diphone("h", "@").expect("should succeed");
        assert_eq!(lookup.unit.source, DiphoneUnitSource::NeuralGenerated);
    }

    #[test]
    fn fallback_errors_when_both_miss() {
        let primary = BTreeMap::new();
        let secondary = BTreeMap::new();
        let mut provider =
            FallbackDiphoneProvider::new(MapProvider(primary), MapProvider(secondary));
        assert!(provider.get_diphone("z", "z").is_err());
    }

    #[test]
    fn fallback_reports_primary_source_not_secondary() {
        // Even when both have the diphone, primary should win.
        let mut primary = BTreeMap::new();
        primary.insert(
            ("p".into(), "ae".into()),
            make_unit("p", "ae", DiphoneUnitSource::MbrolaExact),
        );
        let mut secondary = BTreeMap::new();
        secondary.insert(
            ("p".into(), "ae".into()),
            make_unit("p", "ae", DiphoneUnitSource::NeuralGenerated),
        );
        let mut provider =
            FallbackDiphoneProvider::new(MapProvider(primary), MapProvider(secondary));
        let lookup = provider.get_diphone("p", "ae").expect("should succeed");
        assert_eq!(lookup.unit.source, DiphoneUnitSource::MbrolaExact);
    }

    // ── Cache-backed forge round-trip (no ONNX needed) ────────────────────────

    /// Verify that a unit forged via `forge_from_samples` can be stored in the
    /// cache and retrieved as a cache hit with matching samples and metadata.
    #[test]
    fn cache_backed_forge_roundtrip_without_onnx() {
        use crate::voice::diphone::cache::{CacheEntryMetadata, CacheKey, DiphoneCache};
        use crate::voice::diphone::forge::{
            CARRIER_STRATEGY_VERSION, FORGE_SETTINGS_VERSION, NORMALIZATION_VERSION,
            ForgeSettings, build_carrier_sequence, forge_from_samples,
        };

        let dir = std::env::temp_dir().join(format!(
            "listenbury_provider_test_roundtrip_{}",
            std::process::id()
        ));
        // Ensure cleanup on test exit.
        struct CleanupGuard(std::path::PathBuf);
        impl Drop for CleanupGuard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _guard = CleanupGuard(dir.clone());
        let cache = DiphoneCache::open(&dir).expect("open cache");

        let key = CacheKey {
            model_fingerprint: "test_model".to_string(),
            config_fingerprint: "test_config".to_string(),
            speaker_id: String::new(),
            left: "p".to_string(),
            right: "ae".to_string(),
            carrier_strategy_version: CARRIER_STRATEGY_VERSION.to_string(),
            forge_settings_version: FORGE_SETTINGS_VERSION.to_string(),
            sample_rate_hz: 22050,
            normalization_version: NORMALIZATION_VERSION.to_string(),
        };

        // Forge without ONNX
        let synthetic_pcm: Vec<f32> =
            (0..512).map(|i| (i as f32 * 0.05_f32).sin()).collect();
        let carrier = build_carrier_sequence("p", "ae");
        let forged = forge_from_samples(
            "p",
            "ae",
            &synthetic_pcm,
            22050,
            &carrier,
            "test_model",
            "test_config",
            &ForgeSettings::default(),
        )
        .expect("forge_from_samples should succeed");

        // Should be a cache miss before storing
        assert!(cache.get(&key).is_none(), "should be a miss before storing");

        let meta = CacheEntryMetadata {
            key: key.clone(),
            generated_at: "2026-01-01T00:00:00Z".to_string(),
            carrier_sequence: forged.carrier_sequence.clone(),
            halfseg_samples: forged.unit.halfseg_samples,
            segmentation_confidence: forged.segmentation_confidence,
            sample_count: forged.unit.samples.len(),
            extraction_start_sample: forged.segmentation.source_start_sample,
            extraction_end_sample: forged.segmentation.source_end_sample,
            model_license: "unknown".to_string(),
            provenance_note: "test".to_string(),
        };
        cache
            .store(&key, &forged.unit, meta)
            .expect("cache store should succeed");

        // Should be a cache hit after storing
        let retrieved = cache.get(&key).expect("should be a cache hit after storing");
        assert_eq!(retrieved.samples, forged.unit.samples);
        assert_eq!(retrieved.halfseg_samples, forged.unit.halfseg_samples);
        assert_eq!(retrieved.source, DiphoneUnitSource::CacheHit);
        assert_eq!(retrieved.key.left, "p");
        assert_eq!(retrieved.key.right, "ae");
        assert_eq!(retrieved.sample_rate_hz, 22050);
    }
}
