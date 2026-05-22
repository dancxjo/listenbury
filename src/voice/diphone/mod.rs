//! Neural diphone generation and caching.
//!
//! This module provides a forge pipeline that synthesizes diphone units from a
//! Piper/Riper ONNX model, caches the results on disk, and exposes them
//! through the standard [`DiphoneProvider`] trait so the PSOLA renderer can
//! consume them alongside (or instead of) MBROLA database units.
//!
//! # Key types
//!
//! | Type | Description |
//! |------|-------------|
//! | [`DiphoneCache`] | Disk-backed cache of forged units |
//! | [`CacheKey`] | Stable hash key that includes model/config/phones/versions |
//! | [`CacheEntryMetadata`] | Provenance JSON stored alongside each PCM file |
//! | [`FallbackDiphoneProvider`] | Primary + secondary provider chain |
//! | [`NeuralDiphoneProvider`] | Cache-first; forges on miss (requires `tts-riper`) |
//! | [`ForgeSettings`] | Tuning parameters for the carrier synthesis pipeline |
//!
//! # Licensing note
//!
//! Generated diphone units inherit the license of the source ONNX model.
//! Cache entries **must not** be committed to version control or redistributed
//! without verifying that the model license permits redistribution of derived
//! audio.  Add `diphone-cache/` (or whatever path you choose) to `.gitignore`.
//! See `docs/architecture/diphone-cache.md` for local-only cache expectations.

pub mod cache;
pub mod forge;
pub mod normalize;
pub mod provider;

pub use cache::{CacheEntryMetadata, CacheKey, CacheLookup, DiphoneCache};
pub use forge::{
    CARRIER_STRATEGY_VERSION, CarrierSequence, FORGE_SETTINGS_VERSION, ForgeSettings,
    ForgedUnit, NORMALIZATION_VERSION, PhoneClass, SegmentationReport, build_carrier_sequence,
    forge_from_samples,
};
pub use normalize::NormalizationReport;
pub use provider::FallbackDiphoneProvider;

#[cfg(feature = "tts-riper")]
pub use provider::NeuralDiphoneProvider;
