use std::time::Duration;

use crate::voice::tract::targets::{GlottalSourceTarget, PhoneRenderTarget};

use super::params::{interpolate, KlattFrameParams};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TrajectoryConfig {
    pub frame_ms: u64,
    pub blend_ms: u64,
}

impl Default for TrajectoryConfig {
    fn default() -> Self {
        Self {
            frame_ms: 5,
            blend_ms: 15,
        }
    }
}

pub(crate) fn trajectory_targets_from_phones(
    targets: &[PhoneRenderTarget],
    config: TrajectoryConfig,
) -> Vec<PhoneRenderTarget> {
    if targets.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    let frame_ms = config.frame_ms.max(1);
    for (idx, target) in targets.iter().enumerate() {
        let chunks = duration_chunks(target.duration_ms.max(1), frame_ms);
        let blend_chunks = ((config.blend_ms + frame_ms - 1) / frame_ms) as usize;
        let left = KlattFrameParams::from_target(target);
        let right = targets.get(idx + 1).map(KlattFrameParams::from_target);
        let blend_allowed = right
            .as_ref()
            .is_some_and(|_| should_blend_phone_boundary(target, &targets[idx + 1]));
        let mut t_ms = 0u64;
        for (chunk_idx, duration_ms) in chunks.iter().copied().enumerate() {
            let mut params = left.clone();
            if let Some(next) = right.as_ref() {
                if blend_allowed {
                    let blend_start = chunks.len().saturating_sub(blend_chunks.max(1));
                    if chunk_idx >= blend_start {
                        let rel = (chunk_idx - blend_start + 1) as f32;
                        let den = (chunks.len() - blend_start + 1) as f32;
                        let alpha = smoothstep((rel / den).clamp(0.0, 1.0));
                        params = interpolate(&left, next, alpha);
                    }
                }
            }
            if let (Some(base_f0), Some(vibrato)) = (params.f0_hz, params.vibrato) {
                params.f0_hz = Some(vibrato.apply_to_hz(base_f0, Duration::from_millis(t_ms)));
            }
            out.push(to_target(target, params, duration_ms));
            t_ms = t_ms.saturating_add(duration_ms);
        }
    }
    out
}

fn to_target(
    base: &PhoneRenderTarget,
    params: KlattFrameParams,
    duration_ms: u64,
) -> PhoneRenderTarget {
    PhoneRenderTarget {
        phone: base.phone.clone(),
        duration_ms,
        f0_hz: params.f0_hz,
        amplitude: params.amplitude,
        vibrato: params.vibrato,
        source: Some(GlottalSourceTarget {
            breathiness: params.breathiness,
            open_quotient: base.source.as_ref().map(|s| s.open_quotient).unwrap_or(0.5),
            spectral_tilt_db_per_octave: params.spectral_tilt_db_per_octave,
        }),
        filter: params.filter,
    }
}

fn should_blend_phone_boundary(left: &PhoneRenderTarget, right: &PhoneRenderTarget) -> bool {
    !(is_stop_like(left) || is_stop_like(right))
}

fn is_stop_like(target: &PhoneRenderTarget) -> bool {
    matches!(
        target.phone.ipa.as_str(),
        "p" | "b" | "t" | "d" | "k" | "ɡ" | "tʃ" | "dʒ"
    ) || target.filter.is_none()
}

fn duration_chunks(total_ms: u64, frame_ms: u64) -> Vec<u64> {
    let mut chunks = Vec::new();
    let mut remaining = total_ms;
    while remaining >= frame_ms {
        chunks.push(frame_ms);
        remaining -= frame_ms;
    }
    if remaining > 0 {
        chunks.push(remaining);
    }
    if chunks.is_empty() {
        chunks.push(1);
    }
    chunks
}

fn smoothstep(alpha: f32) -> f32 {
    alpha * alpha * (3.0 - 2.0 * alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linguistic::phonology::{Phone, PhoneString};
    use crate::prosody::vibrato::Vibrato;
    use crate::voice::tract::targets::{
        default_english_phone_targets, phone_render_targets_from_string,
    };

    #[test]
    fn interpolation_creates_intermediate_f0_near_boundary() {
        let table = default_english_phone_targets();
        let left = PhoneString {
            phones: vec![Phone::new_ipa("i"), Phone::new_ipa("e")],
        };
        let mut targets = phone_render_targets_from_string(&left, Some(220.0), 0.8, &table);
        targets[1].f0_hz = Some(330.0);
        targets[0].duration_ms = 100;
        targets[1].duration_ms = 100;
        let traj = trajectory_targets_from_phones(
            &targets,
            TrajectoryConfig {
                frame_ms: 10,
                blend_ms: 20,
            },
        );
        assert!(
            traj.iter()
                .any(|t| t.f0_hz.is_some_and(|f0| f0 > 220.0 && f0 < 330.0)),
            "expected interpolated F0 values in blended trajectory"
        );
    }

    #[test]
    fn vibrato_modulates_sustained_phone_f0() {
        let table = default_english_phone_targets();
        let mut targets = phone_render_targets_from_string(
            &PhoneString {
                phones: vec![Phone::new_ipa("æ")],
            },
            Some(220.0),
            0.8,
            &table,
        );
        targets[0].duration_ms = 400;
        targets[0].vibrato = Some(Vibrato::new(5.0, 30.0, Duration::ZERO, Duration::ZERO, 0.0));
        let traj = trajectory_targets_from_phones(&targets, TrajectoryConfig::default());
        let min = traj
            .iter()
            .filter_map(|target| target.f0_hz)
            .fold(f32::INFINITY, f32::min);
        let max = traj
            .iter()
            .filter_map(|target| target.f0_hz)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max > min,
            "vibrato should modulate F0 over trajectory frames"
        );
    }
}
