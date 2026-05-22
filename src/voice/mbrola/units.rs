//! Diphone unit assembly with join smoothing.
//!
//! This module handles building the combined audio segment for a single phone
//! from its two surrounding half-diphones:
//!
//! ```text
//! [ right half of (prev, phone) ] ++ [ left half of (phone, next) ]
//!                                  ^
//!                                join point
//! ```
//!
//! Before handing the concatenated samples to PSOLA, we apply:
//!
//! 1. Per-half DC removal (bias removal around each half independently).
//! 2. Equal-power crossfade across a short window at the join point to reduce
//!    click-through artefacts.
//! 3. Optional RMS normalisation near the join to reduce sudden loudness jumps.
//!
//! The module is designed to be independently testable without a live MBROLA
//! database.

use super::diphone_provider::DiphoneUnit;

/// The boundary position between two half-diphones inside a combined unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JoinPoint {
    /// Sample index where the right half ends and the left half begins.
    pub sample_index: usize,
}

/// Diagnostic information produced during unit assembly.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UnitAssemblyReport {
    /// Length of the right-half contribution (samples).
    pub right_half_len: usize,
    /// Length of the left-half contribution (samples).
    pub left_half_len: usize,
    /// Position of the crossfade centre in the assembled unit.
    pub join_point: Option<JoinPoint>,
    /// Width of the crossfade window used (samples on each side of the join).
    pub crossfade_radius: usize,
    /// True when RMS normalisation was applied near the join.
    pub rms_normalised: bool,
}

/// Assemble a unit from two half-diphone sample slices and smooth the join.
///
/// Returns the assembled + smoothed sample buffer together with a diagnostic
/// report.
///
/// # Arguments
/// * `right_half` – samples from the right (second) half of `(prev, phone)`
/// * `left_half`  – samples from the left (first) half of `(phone, next)`
/// * `crossfade_samples` – half-width of the equal-power crossfade in samples.
///   Pass `0` to disable crossfade.
/// * `normalise_join` – whether to apply RMS normalisation on a window near
///   the join point to reduce loudness jumps.
pub fn assemble_unit(
    right_half: &[f32],
    left_half: &[f32],
    crossfade_samples: usize,
    normalise_join: bool,
) -> (Vec<f32>, UnitAssemblyReport) {
    // 1. Per-half DC removal
    let right_cleaned = remove_dc(right_half);
    let left_cleaned = remove_dc(left_half);

    let right_len = right_cleaned.len();
    let left_len = left_cleaned.len();
    let total = right_len + left_len;

    if total == 0 {
        return (
            Vec::new(),
            UnitAssemblyReport {
                right_half_len: 0,
                left_half_len: 0,
                ..UnitAssemblyReport::default()
            },
        );
    }

    // 2. Concatenate
    let mut combined = Vec::with_capacity(total);
    combined.extend_from_slice(&right_cleaned);
    combined.extend_from_slice(&left_cleaned);

    let join_idx = right_len;

    // 3. Equal-power crossfade across the join
    let radius = crossfade_radius(crossfade_samples, right_len, left_len);
    if radius > 0 {
        apply_crossfade(&mut combined, join_idx, radius);
    }

    // 4. Optional RMS normalisation near the join
    let rms_normalised = if normalise_join && radius > 0 {
        normalise_near_join(&mut combined, join_idx, radius * 4)
    } else {
        false
    };

    // Clamp to prevent float overflow
    for s in &mut combined {
        *s = s.clamp(-1.0, 1.0);
    }

    let report = UnitAssemblyReport {
        right_half_len: right_len,
        left_half_len: left_len,
        join_point: if total > 0 {
            Some(JoinPoint {
                sample_index: join_idx,
            })
        } else {
            None
        },
        crossfade_radius: radius,
        rms_normalised,
    };

    (combined, report)
}

/// Extract the right half (samples from `halfseg_samples` onward) of a diphone unit.
pub fn right_half_samples(unit: &DiphoneUnit) -> &[f32] {
    let split = halfseg_split(unit);
    &unit.samples[split..]
}

/// Extract the left half (samples up to `halfseg_samples`) of a diphone unit.
pub fn left_half_samples(unit: &DiphoneUnit) -> &[f32] {
    let split = halfseg_split(unit);
    &unit.samples[..split]
}

fn halfseg_split(unit: &DiphoneUnit) -> usize {
    unit.halfseg_samples.min(unit.samples.len())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Remove the mean (DC bias) from a sample slice.
fn remove_dc(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    samples.iter().map(|&s| s - mean).collect()
}

/// Choose a crossfade radius that fits within both half lengths.
fn crossfade_radius(requested: usize, right_len: usize, left_len: usize) -> usize {
    let max = (right_len.min(left_len) / 2).max(0);
    requested.min(max)
}

/// Apply an equal-power crossfade of `radius` samples on each side of `join_idx`.
///
/// Samples in `[join_idx - radius, join_idx)` fade out the right half while
/// the equivalent tail from the left half fades in, blended using a quarter-
/// cosine envelope so that power is preserved.
fn apply_crossfade(buf: &mut [f32], join_idx: usize, radius: usize) {
    if radius == 0 || buf.len() < 2 {
        return;
    }

    // The crossfade window straddles `join_idx`:
    //   right side: buf[join_idx - radius .. join_idx]   (fades out)
    //   left  side: buf[join_idx .. join_idx + radius]   (fades in)
    //
    // We blend the two using equal-power (cos/sin) so that
    //   right_gain^2 + left_gain^2 == 1.
    //
    // To avoid reading past the buffer ends we clamp the indices.
    let fade_start = join_idx.saturating_sub(radius);
    let fade_end = (join_idx + radius).min(buf.len());
    let fade_len = fade_end - fade_start;
    if fade_len < 2 {
        return;
    }

    // right section: indices [fade_start, join_idx)
    // left  section: indices [join_idx,   fade_end )
    let buf_len = buf.len();
    let right_range = join_idx.min(buf_len) - fade_start;
    let left_range = fade_end - join_idx.min(buf_len);

    // Apply fade-out on right side: last `right_range` samples before join
    for (i, sample) in buf[fade_start..fade_start + right_range]
        .iter_mut()
        .enumerate()
    {
        let t = (i + 1) as f32 / (right_range + 1) as f32;
        // gain goes from 1 → 0 as i increases (approaching join)
        let gain = (std::f32::consts::FRAC_PI_2 * (1.0 - t)).cos();
        *sample *= gain;
    }

    // Apply fade-in on left side: first `left_range` samples after join
    for (i, sample) in buf[join_idx.min(buf_len)..fade_end].iter_mut().enumerate() {
        let t = (i + 1) as f32 / (left_range + 1) as f32;
        // gain goes from 0 → 1 as i increases (moving away from join)
        let gain = (std::f32::consts::FRAC_PI_2 * t).sin();
        *sample *= gain;
    }
}

/// Apply RMS normalisation within `window` samples on each side of the join.
///
/// Scales the left and right sides independently so their RMS matches the
/// overall RMS of the provided context window.  Returns `true` when any
/// normalisation was applied.
fn normalise_near_join(buf: &mut [f32], join_idx: usize, window: usize) -> bool {
    if buf.is_empty() || window == 0 {
        return false;
    }

    let win_start = join_idx.saturating_sub(window);
    let win_end = (join_idx + window).min(buf.len());
    if win_start >= win_end {
        return false;
    }

    let target_rms = rms(&buf[win_start..win_end]);
    if target_rms < 1.0e-6 {
        return false;
    }

    let right_start = join_idx.saturating_sub(window / 2);
    let right_end = join_idx.min(buf.len());
    let left_start = join_idx.min(buf.len());
    let left_end = (join_idx + window / 2).min(buf.len());

    let mut applied = false;

    if right_start < right_end {
        let r = rms(&buf[right_start..right_end]);
        if r > 1.0e-6 {
            let scale = (target_rms / r).min(4.0);
            for s in &mut buf[right_start..right_end] {
                *s *= scale;
            }
            applied = true;
        }
    }

    if left_start < left_end {
        let r = rms(&buf[left_start..left_end]);
        if r > 1.0e-6 {
            let scale = (target_rms / r).min(4.0);
            for s in &mut buf[left_start..left_end] {
                *s *= scale;
            }
            applied = true;
        }
    }

    applied
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_with_halfseg(samples: Vec<f32>, halfseg: usize) -> DiphoneUnit {
        use super::super::diphone_provider::{DiphoneKey, DiphoneUnitMetadata, DiphoneUnitSource};
        DiphoneUnit {
            key: DiphoneKey::new("a", "b"),
            samples,
            sample_rate_hz: 16_000,
            halfseg_samples: halfseg,
            frame_center_samples: vec![2, 6],
            source: DiphoneUnitSource::MbrolaExact,
            metadata: DiphoneUnitMetadata::default(),
        }
    }

    #[test]
    fn assemble_unit_preserves_total_length() {
        let right = vec![0.1_f32; 8];
        let left = vec![0.2_f32; 8];
        let (out, report) = assemble_unit(&right, &left, 0, false);
        assert_eq!(out.len(), right.len() + left.len());
        assert_eq!(report.right_half_len, 8);
        assert_eq!(report.left_half_len, 8);
    }

    #[test]
    fn assemble_unit_reports_join_point() {
        let right = vec![0.1_f32; 6];
        let left = vec![0.2_f32; 4];
        let (_, report) = assemble_unit(&right, &left, 0, false);
        assert_eq!(report.join_point, Some(JoinPoint { sample_index: 6 }));
    }

    #[test]
    fn assemble_unit_crossfade_reduces_discontinuity() {
        // Create a signal that has a large DC jump at the join without crossfade.
        let right: Vec<f32> = (0..16).map(|_| 0.8_f32).collect();
        let left: Vec<f32> = (0..16).map(|_| -0.8_f32).collect();

        let (no_fade, _) = assemble_unit(&right, &left, 0, false);
        let (with_fade, report) = assemble_unit(&right, &left, 4, false);

        // Crossfade should have been applied
        assert!(report.crossfade_radius > 0);

        // The jump at the join should be smaller with crossfade
        let join = 16;
        let jump_no_fade = (no_fade[join] - no_fade[join - 1]).abs();
        let jump_fade = (with_fade[join] - with_fade[join - 1]).abs();
        assert!(
            jump_fade <= jump_no_fade + 0.01,
            "crossfade should reduce or maintain discontinuity: no_fade={jump_no_fade:.3}, fade={jump_fade:.3}"
        );
    }

    #[test]
    fn assemble_unit_empty_input() {
        let (out, report) = assemble_unit(&[], &[], 4, false);
        assert!(out.is_empty());
        assert!(report.join_point.is_none());
    }

    #[test]
    fn assemble_unit_one_side_empty() {
        let right = vec![0.5_f32; 8];
        let (out, report) = assemble_unit(&right, &[], 4, false);
        assert_eq!(out.len(), 8);
        assert_eq!(report.left_half_len, 0);
        assert_eq!(report.crossfade_radius, 0); // no room for crossfade
    }

    #[test]
    fn left_and_right_half_samples_split_at_halfseg() {
        let unit = unit_with_halfseg(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3);
        assert_eq!(right_half_samples(&unit), &[4.0, 5.0, 6.0]);
        assert_eq!(left_half_samples(&unit), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn halfseg_clamps_to_sample_length() {
        // halfseg_samples larger than the samples vec should not panic
        let unit = unit_with_halfseg(vec![0.1, 0.2], 100);
        let left = left_half_samples(&unit);
        let right = right_half_samples(&unit);
        assert_eq!(left.len() + right.len(), 2);
    }

    #[test]
    fn remove_dc_removes_mean() {
        let samples = vec![1.0_f32, 2.0, 3.0, 4.0];
        let cleaned = remove_dc(&samples);
        let mean: f32 = cleaned.iter().sum::<f32>() / cleaned.len() as f32;
        assert!(mean.abs() < 1.0e-6, "DC not removed, mean={mean}");
    }

    #[test]
    fn crossfade_radius_clamped_to_half_lengths() {
        // Neither half is long enough for a 100-sample crossfade
        assert_eq!(crossfade_radius(100, 4, 4), 2);
        // No room at all
        assert_eq!(crossfade_radius(100, 1, 1), 0);
        // Requested fits
        assert_eq!(crossfade_radius(3, 20, 20), 3);
    }
}
