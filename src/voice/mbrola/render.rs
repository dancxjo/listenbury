use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::audio::{frame::AudioFrame, write_wav};
use crate::time::ExactTimestamp;

use super::database::{MbrolaDatabase, MbrolaDatabaseError};
use super::pho::{PhoneTimedPlan, write_pho_file};
use super::voice::MbrolaVoice;

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
        render_native_probe_frames(plan, &self.database)
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
                "failed to write native MBROLA probe WAV {}",
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
            backend: "mbrola-native-probe".to_string(),
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
    let frames = render_native_probe_frames(&plan, &database)?;
    write_wav(out_wav, &frames).with_context(|| {
        format!(
            "failed to write native MBROLA probe WAV {}",
            out_wav.display()
        )
    })?;

    Ok(RenderReport {
        backend: "mbrola-native-probe".to_string(),
        voice_name: voice.name,
        voice_path: voice.path,
        out_wav: out_wav.to_path_buf(),
        phone_count: plan.phones.len(),
        duration_ms: plan.total_duration_ms(),
        pho_path: Some(pho_path.to_path_buf()),
    })
}

fn render_native_probe_frames(
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
        let target_len = duration_samples(phone.duration_ms, database.sample_rate_hz).max(1);
        samples.extend(resample_linear(&unit, target_len));
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

fn remove_dc(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    for sample in samples {
        *sample = (*sample - mean).clamp(-1.0, 1.0);
    }
}
