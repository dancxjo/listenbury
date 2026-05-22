//! Signal normalization for extracted diphone units.
//!
//! Provides DC-offset removal, RMS normalization, and short boundary fades.
//! These are applied to raw PCM extracted from neural synthesis before the
//! unit is stored in the diphone cache.

/// Report describing what normalization operations were applied to a diphone unit.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizationReport {
    /// DC offset that was removed (mean before removal).
    pub dc_offset_removed: f32,
    /// Number of samples used for each boundary fade.
    pub fade_samples_applied: usize,
    /// RMS level before normalization.
    pub rms_before: Option<f32>,
    /// RMS level after normalization (target).
    pub rms_after: Option<f32>,
}

/// Remove the mean (DC offset) from `samples` in place.
pub fn remove_dc_offset(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    for s in samples.iter_mut() {
        *s -= mean;
    }
}

/// Apply short linear fades at both ends of `samples` in place.
///
/// `fade_samples` is clamped to at most half the buffer length so that
/// overlapping fades cannot invert the signal.
pub fn apply_boundary_fades(samples: &mut [f32], fade_samples: usize) {
    if samples.is_empty() || fade_samples == 0 {
        return;
    }
    let fade = fade_samples.min(samples.len() / 2);
    for i in 0..fade {
        let gain = i as f32 / fade as f32;
        samples[i] *= gain;
        samples[samples.len() - 1 - i] *= gain;
    }
}

/// Compute the RMS of `samples`, returning `None` if the slice is empty.
pub fn rms(samples: &[f32]) -> Option<f32> {
    if samples.is_empty() {
        return None;
    }
    let mean_sq = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
    Some(mean_sq.sqrt())
}

/// Scale `samples` so their RMS equals `target_rms`.
///
/// If the current RMS is zero (silence) the buffer is left unchanged.
pub fn normalize_rms(samples: &mut [f32], target_rms: f32) {
    if let Some(current) = rms(samples) {
        if current > 1e-9 {
            let gain = target_rms / current;
            for s in samples.iter_mut() {
                *s *= gain;
            }
        }
    }
}

/// Apply the standard normalization pipeline to a diphone unit buffer in place.
///
/// Steps:
/// 1. Remove DC offset.
/// 2. Apply short boundary fades (32 samples by default).
/// 3. Normalize RMS to 0.1.
///
/// Returns a [`NormalizationReport`] describing what was applied.
pub fn normalize_diphone(samples: &mut Vec<f32>) -> NormalizationReport {
    let dc_offset = if samples.is_empty() {
        0.0
    } else {
        samples.iter().sum::<f32>() / samples.len() as f32
    };
    remove_dc_offset(samples);

    const FADE: usize = 32;
    let actual_fade = FADE.min(samples.len() / 2);
    apply_boundary_fades(samples, FADE);

    let rms_before = rms(samples);
    normalize_rms(samples, 0.1);
    let rms_after = rms(samples);

    NormalizationReport {
        dc_offset_removed: dc_offset,
        fade_samples_applied: actual_fade,
        rms_before,
        rms_after,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remove_dc_shifts_to_zero_mean() {
        let mut samples = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        remove_dc_offset(&mut samples);
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(mean.abs() < 1e-6, "mean should be near zero, got {mean}");
    }

    #[test]
    fn remove_dc_preserves_length() {
        let mut samples = vec![0.5_f32; 100];
        remove_dc_offset(&mut samples);
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn apply_fades_tapers_boundaries() {
        let mut samples = vec![1.0_f32; 16];
        apply_boundary_fades(&mut samples, 4);
        assert_eq!(samples[0], 0.0);
        assert!(samples[3] < 1.0);
        assert_eq!(samples[4], 1.0);
        assert_eq!(samples[15], 0.0);
        assert!(samples[12] < 1.0);
    }

    #[test]
    fn apply_fades_clamps_to_half_length() {
        let mut samples = vec![1.0_f32; 8];
        // fade_samples = 100 > len/2 = 4, should be clamped
        apply_boundary_fades(&mut samples, 100);
        // must not panic; first sample should be 0
        assert_eq!(samples[0], 0.0);
    }

    #[test]
    fn normalize_rms_scales_correctly() {
        let mut samples = vec![0.5_f32, 0.5, 0.5, 0.5];
        normalize_rms(&mut samples, 0.1);
        let r = rms(&samples).unwrap();
        assert!((r - 0.1).abs() < 1e-6, "rms should be 0.1, got {r}");
    }

    #[test]
    fn normalize_diphone_preserves_length() {
        let mut samples: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let original_len = samples.len();
        normalize_diphone(&mut samples);
        assert_eq!(samples.len(), original_len);
    }

    #[test]
    fn normalize_diphone_removes_dc() {
        let mut samples = vec![2.0_f32; 64];
        normalize_diphone(&mut samples);
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(mean.abs() < 1e-6);
    }

    #[test]
    fn normalize_diphone_report_has_dc_offset() {
        let mut samples = vec![2.0_f32; 64];
        let report = normalize_diphone(&mut samples);
        assert!((report.dc_offset_removed - 2.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_diphone_report_rms_after_near_target() {
        let mut samples: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();
        let report = normalize_diphone(&mut samples);
        if let Some(rms_after) = report.rms_after {
            assert!((rms_after - 0.1).abs() < 1e-4, "rms_after should be ~0.1, got {rms_after}");
        }
    }
}
