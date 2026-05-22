use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::audio::{frame::AudioFrame, write_wav};
use crate::time::ExactTimestamp;

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
}

impl MbrolaRenderer {
    pub fn new(config: MbrolaRendererConfig) -> Self {
        Self { config }
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
        Ok(render_native_probe_frames(
            plan,
            self.config.voice.sample_rate.unwrap_or(16_000),
        ))
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
    let frames = render_native_probe_frames(&plan, voice.sample_rate.unwrap_or(16_000));
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

fn render_native_probe_frames(plan: &PhoneTimedPlan, sample_rate_hz: u32) -> Vec<AudioFrame> {
    let mut samples = Vec::new();
    let mut phase = 0.0_f32;
    for phone in &plan.phones {
        let sample_count =
            (u64::from(phone.duration_ms) * u64::from(sample_rate_hz) / 1000) as usize;
        let f0 = phone
            .pitch_targets
            .first()
            .map(|target| target.hz)
            .unwrap_or(120.0)
            .max(40.0);
        let voiced = !phone.pitch_targets.is_empty() && phone.symbol != "_";
        for idx in 0..sample_count {
            let t = if sample_count > 0 {
                idx as f32 / sample_count as f32
            } else {
                0.0
            };
            let envelope = (t.min(1.0 - t) * 10.0).clamp(0.0, 1.0);
            let sample = if voiced {
                phase += std::f32::consts::TAU * f0 / sample_rate_hz as f32;
                phase.sin() * 0.18 * envelope
            } else {
                0.0
            };
            samples.push(sample);
        }
    }
    vec![AudioFrame {
        captured_at: ExactTimestamp::now(),
        sample_rate_hz,
        channels: 1,
        samples,
        voice_signatures: Vec::new(),
    }]
}
