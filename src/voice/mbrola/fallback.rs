//! Ranked fallback strategies for missing MBROLA diphones.
//!
//! When the exact `(left, right)` diphone is not present in a voice database,
//! the renderer needs a graceful degradation path.  This module defines the
//! ranked strategy chain and produces explicit diagnostics so callers can
//! surface every non-exact substitution in a render report.

use super::diphone_provider::{
    DiphoneKey, DiphoneLookup, DiphoneProvider, DiphoneUnit, DiphoneUnitMetadata, DiphoneUnitSource,
};

/// Describes why a particular diphone unit was chosen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FallbackReason {
    /// The exact `(left, right)` diphone was found in the database.
    Exact,
    /// A voice-level alias (stored in the database as a redirect entry) was used.
    VoiceAlias,
    /// A different phoneme of the same broad class was substituted.
    SameClassSubstitute {
        /// The canonical phoneme that was substituted for the requested one.
        substitute: String,
    },
    /// The `(left, _)` or `(_, right)` boundary diphone was used instead of
    /// the unavailable inner diphone.
    BoundaryHalf,
    /// No usable diphone material was found; silence was synthesised.
    SyntheticSilence,
}

impl FallbackReason {
    /// Returns `true` when this reason represents a completely faithful lookup.
    pub fn is_exact(&self) -> bool {
        matches!(self, Self::Exact | Self::VoiceAlias)
    }

    /// Human-readable description suitable for a warning message.
    pub fn description(&self) -> String {
        match self {
            Self::Exact => "exact diphone lookup".to_string(),
            Self::VoiceAlias => "voice-level alias".to_string(),
            Self::SameClassSubstitute { substitute } => {
                format!("same-class substitute `{substitute}`")
            }
            Self::BoundaryHalf => "boundary-half fallback".to_string(),
            Self::SyntheticSilence => "synthetic silence (no diphone material found)".to_string(),
        }
    }
}

/// The result of a fallback resolution attempt.
#[derive(Debug, Clone, PartialEq)]
pub struct FallbackResult {
    /// The diphone unit that was resolved, regardless of strategy.
    pub lookup: DiphoneLookup,
    /// The strategy that produced this result.
    pub reason: FallbackReason,
}

/// Look up the right half of a phone context `(left, phone)` using the ranked
/// fallback chain.
///
/// The right half of the phone is the second half of the `(left, phone)` diphone,
/// covering from the phone midpoint outward toward the left context.
///
/// Fallback order:
/// 1. Exact `(left, phone)` lookup
/// 2. Boundary-half `(_, phone)`
/// 3. Synthetic silence
pub fn resolve_right_half(
    provider: &mut impl DiphoneProvider,
    left: &str,
    phone: &str,
    sample_rate_hz: u32,
    period_samples: usize,
) -> FallbackResult {
    // 1. Exact
    if let Ok(lookup) = provider.get_diphone(left, phone) {
        return FallbackResult {
            lookup,
            reason: FallbackReason::Exact,
        };
    }

    // 2. Boundary half: (_, phone)
    if left != "_" && phone != "_" {
        if let Ok(lookup) = provider.get_diphone("_", phone) {
            let unit = boundary_fallback_unit(lookup.unit, left, phone);
            return FallbackResult {
                lookup: DiphoneLookup { unit },
                reason: FallbackReason::BoundaryHalf,
            };
        }
    }

    // 3. Synthetic silence
    let silence = synthetic_silence_unit(left, phone, sample_rate_hz, period_samples);
    FallbackResult {
        lookup: DiphoneLookup { unit: silence },
        reason: FallbackReason::SyntheticSilence,
    }
}

/// Look up the left half of a phone context `(phone, right)` using the ranked
/// fallback chain.
///
/// The left half covers from the phone midpoint outward toward the right context.
///
/// Fallback order:
/// 1. Exact `(phone, right)` lookup
/// 2. Boundary-half `(phone, _)`
/// 3. Synthetic silence
pub fn resolve_left_half(
    provider: &mut impl DiphoneProvider,
    phone: &str,
    right: &str,
    sample_rate_hz: u32,
    period_samples: usize,
) -> FallbackResult {
    // 1. Exact
    if let Ok(lookup) = provider.get_diphone(phone, right) {
        return FallbackResult {
            lookup,
            reason: FallbackReason::Exact,
        };
    }

    // 2. Boundary half: (phone, _)
    if phone != "_" && right != "_" {
        if let Ok(lookup) = provider.get_diphone(phone, "_") {
            let unit = boundary_fallback_unit(lookup.unit, phone, right);
            return FallbackResult {
                lookup: DiphoneLookup { unit },
                reason: FallbackReason::BoundaryHalf,
            };
        }
    }

    // 3. Synthetic silence
    let silence = synthetic_silence_unit(phone, right, sample_rate_hz, period_samples);
    FallbackResult {
        lookup: DiphoneLookup { unit: silence },
        reason: FallbackReason::SyntheticSilence,
    }
}

/// Build a warning message for any non-exact fallback result.
pub fn fallback_warning(
    requested_left: &str,
    requested_right: &str,
    reason: &FallbackReason,
) -> Option<String> {
    if reason.is_exact() {
        return None;
    }
    Some(format!(
        "diphone {requested_left}-{requested_right}: {}",
        reason.description()
    ))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn boundary_fallback_unit(
    mut unit: DiphoneUnit,
    requested_left: &str,
    requested_right: &str,
) -> DiphoneUnit {
    let warning = format!(
        "used boundary fallback diphone {}-{} for requested {}-{}",
        unit.key.left, unit.key.right, requested_left, requested_right
    );
    unit.source = DiphoneUnitSource::MbrolaBoundaryFallback;
    unit.metadata = DiphoneUnitMetadata {
        requested_key: Some(DiphoneKey::new(requested_left, requested_right)),
        warning: Some(warning),
    };
    unit
}

fn synthetic_silence_unit(
    left: &str,
    right: &str,
    sample_rate_hz: u32,
    period_samples: usize,
) -> DiphoneUnit {
    let silence_len = (period_samples * 2).max(16);
    let warning = format!("synthesised silence for missing diphone {left}-{right}");
    DiphoneUnit {
        key: DiphoneKey::new(left, right),
        samples: vec![0.0_f32; silence_len],
        sample_rate_hz,
        halfseg_samples: silence_len / 2,
        frame_center_samples: Vec::new(),
        source: DiphoneUnitSource::SyntheticSilence,
        metadata: DiphoneUnitMetadata {
            requested_key: Some(DiphoneKey::new(left, right)),
            warning: Some(warning),
        },
    }
}

#[cfg(test)]
mod tests {
    use anyhow::anyhow;
    use anyhow::Result;
    use std::collections::BTreeMap;

    use super::*;

    #[derive(Default)]
    struct MapProvider {
        units: BTreeMap<(String, String), DiphoneUnit>,
    }

    impl MapProvider {
        fn insert(mut self, left: &str, right: &str, halfseg: usize) -> Self {
            self.units.insert(
                (left.to_string(), right.to_string()),
                DiphoneUnit {
                    key: DiphoneKey::new(left, right),
                    samples: vec![0.1_f32; 8],
                    sample_rate_hz: 16_000,
                    halfseg_samples: halfseg,
                    frame_center_samples: vec![2, 6],
                    source: DiphoneUnitSource::MbrolaExact,
                    metadata: DiphoneUnitMetadata::default(),
                },
            );
            self
        }
    }

    impl DiphoneProvider for MapProvider {
        fn get_diphone(&mut self, left: &str, right: &str) -> Result<DiphoneLookup> {
            self.units
                .get(&(left.to_string(), right.to_string()))
                .cloned()
                .map(|unit| DiphoneLookup { unit })
                .ok_or_else(|| anyhow!("missing {left}-{right}"))
        }
    }

    #[test]
    fn resolve_right_half_exact_when_present() {
        let mut p = MapProvider::default().insert("a", "b", 4);
        let result = resolve_right_half(&mut p, "a", "b", 16_000, 80);
        assert_eq!(result.reason, FallbackReason::Exact);
        assert_eq!(result.lookup.unit.source, DiphoneUnitSource::MbrolaExact);
    }

    #[test]
    fn resolve_right_half_uses_boundary_when_exact_missing() {
        let mut p = MapProvider::default().insert("_", "b", 4);
        let result = resolve_right_half(&mut p, "a", "b", 16_000, 80);
        assert_eq!(result.reason, FallbackReason::BoundaryHalf);
        assert_eq!(
            result.lookup.unit.source,
            DiphoneUnitSource::MbrolaBoundaryFallback
        );
    }

    #[test]
    fn resolve_right_half_falls_back_to_silence_when_nothing_available() {
        let mut p = MapProvider::default();
        let result = resolve_right_half(&mut p, "a", "b", 16_000, 80);
        assert_eq!(result.reason, FallbackReason::SyntheticSilence);
        assert_eq!(
            result.lookup.unit.source,
            DiphoneUnitSource::SyntheticSilence
        );
        assert!(!result.lookup.unit.samples.is_empty());
    }

    #[test]
    fn resolve_left_half_exact_when_present() {
        let mut p = MapProvider::default().insert("b", "c", 4);
        let result = resolve_left_half(&mut p, "b", "c", 16_000, 80);
        assert_eq!(result.reason, FallbackReason::Exact);
    }

    #[test]
    fn resolve_left_half_uses_boundary_when_exact_missing() {
        let mut p = MapProvider::default().insert("b", "_", 4);
        let result = resolve_left_half(&mut p, "b", "c", 16_000, 80);
        assert_eq!(result.reason, FallbackReason::BoundaryHalf);
    }

    #[test]
    fn fallback_warning_none_for_exact() {
        let w = fallback_warning("a", "b", &FallbackReason::Exact);
        assert!(w.is_none());
    }

    #[test]
    fn fallback_warning_present_for_boundary() {
        let w = fallback_warning("a", "b", &FallbackReason::BoundaryHalf);
        assert!(w.is_some());
        let msg = w.unwrap();
        assert!(msg.contains("a-b"), "expected 'a-b' in '{msg}'");
        assert!(msg.contains("boundary"), "expected 'boundary' in '{msg}'");
    }

    #[test]
    fn fallback_warning_present_for_silence() {
        let w = fallback_warning("x", "y", &FallbackReason::SyntheticSilence);
        assert!(w.is_some());
        assert!(w.unwrap().contains("synthetic silence"));
    }
}
