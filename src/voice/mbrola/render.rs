use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::audio::{frame::AudioFrame, write_wav};
use crate::time::ExactTimestamp;

use super::database::{MbrolaDatabase, MbrolaDatabaseError};
use super::pho::{MbrolaPitchTarget, PhoneTimedPlan, write_pho_file};
use super::voice::MbrolaVoice;

const MIN_PSOLA_GRAINS: usize = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MbrolaRendererConfig {
    pub voice: MbrolaVoice,
    pub keep_pho: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderReport {
    pub backend: String,
    pub voice_name: String,
    pub voice_path: PathBuf,
    pub out_wav: PathBuf,
    pub phone_count: usize,
    pub duration_ms: u64,
    pub pho_path: Option<PathBuf>,
}

pub trait PhoneTimedRenderer {
    fn render_phone_plan(&self, plan: &PhoneTimedPlan, out_wav: &Path) -> Result<RenderReport>;
}

#[derive(Debug, Clone)]
pub struct MbrolaRenderer {
    config: MbrolaRendererConfig,
    database: MbrolaDatabase,
}

impl MbrolaRenderer {
    pub fn new(config: MbrolaRendererConfig) -> Self {
        let database = MbrolaDatabase::load(&config.voice.path)
            .expect("MBROLA database should load after MbrolaVoice validation");
        Self { config, database }
    }

    pub fn from_voice_path(
        _executable: Option<PathBuf>,
        voice_path: impl Into<PathBuf>,
    ) -> Result<Self> {
        let voice = MbrolaVoice::load(voice_path)?;
        Ok(Self::new(MbrolaRendererConfig {
            voice,
            keep_pho: None,
        }))
    }

    pub fn voice(&self) -> &MbrolaVoice {
        &self.config.voice
    }

    pub fn render_phone_plan_to_frames(&self, plan: &PhoneTimedPlan) -> Result<Vec<AudioFrame>> {
        if !self.config.voice.path.is_file() {
            bail!(
                "MBROLA voice database not found at {}",
                self.config.voice.path.display()
            );
        }
        if plan.phones.is_empty() {
            bail!("cannot render an empty MBROLA phone plan");
        }
        render_native_diphone_frames(plan, &self.database)
    }
}

impl PhoneTimedRenderer for MbrolaRenderer {
    fn render_phone_plan(&self, plan: &PhoneTimedPlan, out_wav: &Path) -> Result<RenderReport> {
        if !self.config.voice.path.is_file() {
            bail!(
                "MBROLA voice database not found at {}",
                self.config.voice.path.display()
            );
        }
        if plan.phones.is_empty() {
            bail!("cannot render an empty MBROLA phone plan");
        }

        if let Some(parent) = out_wav
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create MBROLA output directory {}",
                    parent.display()
                )
            })?;
        }

        let temp_pho = self
            .config
            .keep_pho
            .clone()
            .unwrap_or_else(|| out_wav.with_extension("pho"));
        write_pho_file(&temp_pho, plan)?;
        let frames = self.render_phone_plan_to_frames(plan)?;
        write_wav(out_wav, &frames).with_context(|| {
            format!(
                "failed to write native MBROLA PSOLA WAV {}",
                out_wav.display()
            )
        })?;

        if !out_wav.is_file() {
            bail!(
                "MBROLA reported success but did not create {}",
                out_wav.display()
            );
        }

        Ok(RenderReport {
            backend: "mbrola-native-psola".to_string(),
            voice_name: self.config.voice.name.clone(),
            voice_path: self.config.voice.path.clone(),
            out_wav: out_wav.to_path_buf(),
            phone_count: plan.phones.len(),
            duration_ms: plan.total_duration_ms(),
            pho_path: Some(temp_pho),
        })
    }
}

pub fn render_raw_pho(
    executable: Option<PathBuf>,
    voice_path: impl Into<PathBuf>,
    pho_path: &Path,
    out_wav: &Path,
) -> Result<RenderReport> {
    let _ = executable;
    let voice = MbrolaVoice::load(voice_path)?;
    if !pho_path.is_file() {
        bail!("MBROLA .pho input not found at {}", pho_path.display());
    }
    if let Some(parent) = out_wav
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create MBROLA output directory {}",
                parent.display()
            )
        })?;
    }

    let plan = super::pho::read_pho_file(pho_path)?;
    let database = MbrolaDatabase::load(&voice.path)?;
    let frames = render_native_diphone_frames(&plan, &database)?;
    write_wav(out_wav, &frames).with_context(|| {
        format!(
            "failed to write native MBROLA PSOLA WAV {}",
            out_wav.display()
        )
    })?;

    Ok(RenderReport {
        backend: "mbrola-native-psola".to_string(),
        voice_name: voice.name,
        voice_path: voice.path,
        out_wav: out_wav.to_path_buf(),
        phone_count: plan.phones.len(),
        duration_ms: plan.total_duration_ms(),
        pho_path: Some(pho_path.to_path_buf()),
    })
}

fn render_native_diphone_frames(
    plan: &PhoneTimedPlan,
    database: &MbrolaDatabase,
) -> Result<Vec<AudioFrame>> {
    let mut samples = Vec::new();
    for (index, phone) in plan.phones.iter().enumerate() {
        if phone.symbol == "_" {
            let silence_len = duration_samples(phone.duration_ms, database.sample_rate_hz).max(1);
            samples.extend(std::iter::repeat_n(0.0, silence_len));
            continue;
        }
        let prev = previous_symbol(plan, index).unwrap_or("_");
        let next = next_symbol(plan, index).unwrap_or("_");
        let mut unit = Vec::new();
        unit.extend(diphone_right_half(database, prev, &phone.symbol)?);
        unit.extend(diphone_left_half(database, &phone.symbol, next)?);
        if unit.is_empty() {
            return Err(anyhow::anyhow!(
                "MBROLA diphone material for phone `{}` was empty",
                phone.symbol
            ));
        }
        remove_dc(&mut unit);
        samples.extend(psola_synthesize(
            &unit,
            phone.duration_ms,
            &phone.pitch_targets,
            database.sample_rate_hz,
            database.mbr_period,
        ));
    }
    Ok(vec![AudioFrame {
        captured_at: ExactTimestamp::now(),
        sample_rate_hz: database.sample_rate_hz,
        channels: 1,
        samples,
        voice_signatures: Vec::new(),
    }])
}

fn previous_symbol(plan: &PhoneTimedPlan, index: usize) -> Option<&str> {
    index
        .checked_sub(1)
        .and_then(|previous| plan.phones.get(previous))
        .map(|phone| phone.symbol.as_str())
        .or(Some("_"))
}

fn next_symbol(plan: &PhoneTimedPlan, index: usize) -> Option<&str> {
    plan.phones
        .get(index + 1)
        .map(|phone| phone.symbol.as_str())
        .or(Some("_"))
}

fn diphone_left_half(database: &MbrolaDatabase, left: &str, right: &str) -> Result<Vec<f32>> {
    let diphone =
        database
            .diphone(left, right)
            .ok_or_else(|| MbrolaDatabaseError::MissingDiphone {
                left: left.to_string(),
                right: right.to_string(),
            })?;
    let samples = database.samples_for_diphone(diphone)?;
    let split = diphone.halfseg_samples.min(samples.len());
    Ok(samples[..split].to_vec())
}

fn diphone_right_half(database: &MbrolaDatabase, left: &str, right: &str) -> Result<Vec<f32>> {
    let diphone =
        database
            .diphone(left, right)
            .ok_or_else(|| MbrolaDatabaseError::MissingDiphone {
                left: left.to_string(),
                right: right.to_string(),
            })?;
    let samples = database.samples_for_diphone(diphone)?;
    let split = diphone.halfseg_samples.min(samples.len());
    Ok(samples[split..].to_vec())
}

fn duration_samples(duration_ms: u32, sample_rate_hz: u32) -> usize {
    (u64::from(duration_ms) * u64::from(sample_rate_hz) / 1000) as usize
}

fn resample_linear(input: &[f32], output_len: usize) -> Vec<f32> {
    if input.is_empty() || output_len == 0 {
        return Vec::new();
    }
    if input.len() == output_len {
        return input.to_vec();
    }
    if output_len == 1 {
        return vec![input[0]];
    }
    let scale = (input.len() - 1) as f32 / (output_len - 1) as f32;
    (0..output_len)
        .map(|idx| {
            let pos = idx as f32 * scale;
            let left = pos.floor() as usize;
            let right = (left + 1).min(input.len() - 1);
            let frac = pos - left as f32;
            input[left] * (1.0 - frac) + input[right] * frac
        })
        .collect()
}

fn psola_synthesize(
    input: &[f32],
    duration_ms: u32,
    pitch_targets: &[MbrolaPitchTarget],
    sample_rate_hz: u32,
    source_period_samples: usize,
) -> Vec<f32> {
    let output_len = duration_samples(duration_ms, sample_rate_hz).max(1);
    if input.is_empty() {
        return vec![0.0; output_len];
    }

    let source_period_samples = source_period_samples.max(1);
    let grain_len = (source_period_samples * 2).max(4);
    let source_marks = pitch_marks_for_len(input.len(), source_period_samples);
    if input.len() < grain_len || source_marks.len() < MIN_PSOLA_GRAINS {
        return resample_linear(input, output_len);
    }

    let neutral_hz = sample_rate_hz as f32 / source_period_samples as f32;
    let pitch_curve = PitchTargetCurve::new(pitch_targets, neutral_hz);
    let mut output = vec![0.0; output_len];
    let mut weights = vec![0.0; output_len];
    let window = hann_window(grain_len);
    let half_grain = grain_len / 2;
    let stretch = input.len() as f32 / output_len as f32;

    let mut dst_center = 0.5 * target_period_at(&pitch_curve, 0, output_len, sample_rate_hz);
    while dst_center < output_len as f32 + half_grain as f32 {
        let src_pos = (dst_center * stretch).clamp(0.0, input.len().saturating_sub(1) as f32);
        let src_center = nearest_mark(&source_marks, src_pos);
        overlap_add_grain(
            input,
            &window,
            src_center,
            dst_center.round() as isize,
            half_grain,
            &mut output,
            &mut weights,
        );

        let period = target_period_at(
            &pitch_curve,
            dst_center.max(0.0).round() as usize,
            output_len,
            sample_rate_hz,
        );
        dst_center += period.max(1.0);
    }

    for (sample, weight) in output.iter_mut().zip(weights) {
        if weight > 1.0e-6 {
            *sample = (*sample / weight).clamp(-1.0, 1.0);
        }
    }
    output
}

fn pitch_marks_for_len(sample_len: usize, period: usize) -> Vec<usize> {
    if sample_len == 0 {
        return Vec::new();
    }
    let mut marks = Vec::new();
    let mut center = period / 2;
    while center < sample_len {
        marks.push(center);
        center += period;
    }
    if marks.last().copied() != Some(sample_len - 1) {
        marks.push(sample_len - 1);
    }
    marks
}

fn nearest_mark(marks: &[usize], target: f32) -> usize {
    let idx = marks.partition_point(|mark| (*mark as f32) < target);
    match (idx.checked_sub(1), marks.get(idx)) {
        (Some(left), Some(&right)) => {
            let left = marks[left];
            if target - left as f32 <= right as f32 - target {
                left
            } else {
                right
            }
        }
        (Some(left), None) => marks[left],
        (None, Some(&right)) => right,
        (None, None) => 0,
    }
}

fn overlap_add_grain(
    input: &[f32],
    window: &[f32],
    src_center: usize,
    dst_center: isize,
    half_grain: usize,
    output: &mut [f32],
    weights: &mut [f32],
) {
    for (win_idx, &weight) in window.iter().enumerate() {
        let src_idx = src_center as isize + win_idx as isize - half_grain as isize;
        let dst_idx = dst_center + win_idx as isize - half_grain as isize;
        if src_idx < 0 || dst_idx < 0 {
            continue;
        }
        let src_idx = src_idx as usize;
        let dst_idx = dst_idx as usize;
        if src_idx >= input.len() || dst_idx >= output.len() {
            continue;
        }
        output[dst_idx] += input[src_idx] * weight;
        weights[dst_idx] += weight;
    }
}

fn hann_window(len: usize) -> Vec<f32> {
    if len <= 1 {
        return vec![1.0; len];
    }
    (0..len)
        .map(|idx| {
            let phase = std::f32::consts::TAU * idx as f32 / (len - 1) as f32;
            0.5 - 0.5 * phase.cos()
        })
        .collect()
}

fn target_period_at(
    pitch_curve: &PitchTargetCurve,
    sample_index: usize,
    output_len: usize,
    sample_rate_hz: u32,
) -> f32 {
    let hz = pitch_curve
        .hz_at(sample_index, output_len)
        .clamp(40.0, sample_rate_hz as f32 / 2.0);
    sample_rate_hz as f32 / hz
}

#[derive(Debug, Clone)]
struct PitchTargetCurve {
    neutral_hz: f32,
    targets: Vec<MbrolaPitchTarget>,
}

impl PitchTargetCurve {
    fn new(targets: &[MbrolaPitchTarget], neutral_hz: f32) -> Self {
        let mut targets = targets
            .iter()
            .copied()
            .filter(|target| target.hz.is_finite() && target.hz > 0.0)
            .collect::<Vec<_>>();
        targets.sort_by_key(|target| target.percent.min(100));
        Self {
            neutral_hz: neutral_hz.max(1.0),
            targets,
        }
    }

    fn hz_at(&self, sample_index: usize, output_len: usize) -> f32 {
        if self.targets.is_empty() {
            return self.neutral_hz;
        }
        if self.targets.len() == 1 || output_len <= 1 {
            return self.targets[0].hz;
        }

        let percent = (sample_index as f32 * 100.0 / (output_len - 1) as f32).clamp(0.0, 100.0);
        if percent <= self.targets[0].percent.min(100) as f32 {
            return self.targets[0].hz;
        }

        for pair in self.targets.windows(2) {
            let left = pair[0];
            let right = pair[1];
            let left_percent = left.percent.min(100) as f32;
            let right_percent = right.percent.min(100) as f32;
            if percent <= right_percent {
                let span = (right_percent - left_percent).max(f32::EPSILON);
                let frac = ((percent - left_percent) / span).clamp(0.0, 1.0);
                return left.hz * (1.0 - frac) + right.hz * frac;
            }
        }

        self.targets
            .last()
            .map(|target| target.hz)
            .unwrap_or(self.neutral_hz)
    }
}

fn remove_dc(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    for sample in samples {
        *sample = (*sample - mean).clamp(-1.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psola_synthesize_matches_requested_duration() {
        let input = sine(220.0, 1600, 16_000);
        let output = psola_synthesize(&input, 250, &[], 16_000, 80);

        assert_eq!(output.len(), 4000);
        assert!(output.iter().any(|sample| sample.abs() > 0.01));
    }

    #[test]
    fn psola_uses_pitch_targets_for_period_spacing() {
        let input = sine(180.0, 3200, 16_000);
        let low = psola_synthesize(
            &input,
            200,
            &[MbrolaPitchTarget {
                percent: 0,
                hz: 100.0,
            }],
            16_000,
            80,
        );
        let high = psola_synthesize(
            &input,
            200,
            &[MbrolaPitchTarget {
                percent: 0,
                hz: 220.0,
            }],
            16_000,
            80,
        );

        assert_eq!(low.len(), high.len());
        assert_ne!(zero_crossings(&low), zero_crossings(&high));
    }

    #[test]
    fn pitch_curve_interpolates_pho_targets() {
        let curve = PitchTargetCurve::new(
            &[
                MbrolaPitchTarget {
                    percent: 0,
                    hz: 100.0,
                },
                MbrolaPitchTarget {
                    percent: 100,
                    hz: 200.0,
                },
            ],
            150.0,
        );

        assert!((curve.hz_at(50, 101) - 150.0).abs() < 0.01);
    }

    fn sine(hz: f32, len: usize, sample_rate_hz: u32) -> Vec<f32> {
        (0..len)
            .map(|idx| {
                let phase = std::f32::consts::TAU * hz * idx as f32 / sample_rate_hz as f32;
                phase.sin() * 0.4
            })
            .collect()
    }

    fn zero_crossings(samples: &[f32]) -> usize {
        samples
            .windows(2)
            .filter(|pair| (pair[0] < 0.0 && pair[1] >= 0.0) || (pair[0] >= 0.0 && pair[1] < 0.0))
            .count()
    }
}
