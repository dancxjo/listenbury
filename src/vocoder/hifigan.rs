use std::f32::consts::TAU;
#[cfg(feature = "tts-riper")]
use std::fs;
#[cfg(feature = "tts-riper")]
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
#[cfg(feature = "tts-riper")]
use ort::session::{Session, builder::GraphOptimizationLevel};
#[cfg(feature = "tts-riper")]
use ort::value::{DynTensorValueType, Tensor, TensorElementType};

use crate::audio::frame::AudioFrame;
#[cfg(feature = "tts-riper")]
use crate::mouth::riper::backend::initialize_ort_runtime;
use crate::time::ExactTimestamp;
use crate::vocoder::{
    BackendCapabilities, BackendFamily, MelConfig, MelFrame, MelScale, MelSpectrogram,
    MelTensorLayout, VocoderBackend, VocoderDescriptor, VocoderInput,
};

pub struct HifiganBackend {
    #[cfg(feature = "tts-riper")]
    model_path: Option<PathBuf>,
    #[cfg(feature = "tts-riper")]
    onnx_config: Option<HifiganOnnxConfig>,
    #[cfg(feature = "tts-riper")]
    session: Option<Session>,
}

const SAMPLE_RATE_HZ: u32 = 22_050;
const SPEECHT5_HIFIGAN_SAMPLE_RATE_HZ: u32 = 16_000;
const HOP_SAMPLES: usize = 256;
const MIN_F0_HZ: f32 = 55.0;
const MAX_F0_HZ: f32 = 1_200.0;
const NOISE_GAIN: f32 = 0.018;
const MIN_NORMALIZABLE_PEAK: f32 = 1.0e-4;

#[cfg(feature = "tts-riper")]
#[derive(Debug, Clone)]
struct HifiganOnnxConfig {
    mel: MelConfig,
    input_name: Option<String>,
    output_name: Option<String>,
    input_layout: Option<MelTensorLayout>,
}

impl HifiganBackend {
    pub fn default_mel_config() -> MelConfig {
        MelConfig {
            sample_rate_hz: SPEECHT5_HIFIGAN_SAMPLE_RATE_HZ,
            hop_samples: HOP_SAMPLES,
            n_fft: 1024,
            win_length: 1024,
            n_mels: 80,
            f_min_hz: 0.0,
            f_max_hz: Some(8_000.0),
            scale: MelScale::NaturalLogEnergy,
        }
    }

    pub fn expected_mel_config(&self) -> MelConfig {
        #[cfg(feature = "tts-riper")]
        if let Some(config) = &self.onnx_config {
            return config.mel.clone();
        }
        MelConfig::test_default(80)
    }

    pub fn deterministic() -> Self {
        Self {
            #[cfg(feature = "tts-riper")]
            model_path: None,
            #[cfg(feature = "tts-riper")]
            onnx_config: None,
            #[cfg(feature = "tts-riper")]
            session: None,
        }
    }

    #[cfg(feature = "tts-riper")]
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();
        ensure!(
            model_path.is_file(),
            "HiFi-GAN ONNX model file not found at {}",
            model_path.display()
        );

        let onnx_config = load_hifigan_model_config(&model_path)?;
        initialize_ort_runtime()?;
        let session = Session::builder()
            .context("failed to create HiFi-GAN ONNX session builder")?
            .with_intra_threads(1)
            .map_err(|error| {
                anyhow::anyhow!("failed to configure HiFi-GAN ONNX intra-op threads: {error}")
            })?
            .with_inter_threads(1)
            .map_err(|error| {
                anyhow::anyhow!("failed to configure HiFi-GAN ONNX inter-op threads: {error}")
            })?
            .with_intra_op_spinning(false)
            .map_err(|error| {
                anyhow::anyhow!("failed to configure HiFi-GAN ONNX intra-op spinning: {error}")
            })?
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .map_err(|error| {
                anyhow::anyhow!("failed to configure HiFi-GAN ONNX optimization level: {error}")
            })?
            .commit_from_file(&model_path)
            .with_context(|| {
                format!(
                    "failed to load HiFi-GAN ONNX model from {}",
                    model_path.display()
                )
            })?;

        Ok(Self {
            model_path: Some(model_path),
            onnx_config: Some(onnx_config),
            session: Some(session),
        })
    }

    pub fn descriptor() -> VocoderDescriptor {
        VocoderDescriptor {
            id: "hifigan",
            family: BackendFamily::NeuralVocoder,
            capabilities: BackendCapabilities {
                accepts_phone_timed: false,
                accepts_partial_prosody: false,
                accepts_coarse_text: false,
                accepts_mel: true,
                accepts_mel_f0: true,
                honors_explicit_duration: false,
                honors_explicit_f0: false,
                honors_vibrato: false,
                streaming_safe: false,
            },
            sample_rate_hz: SAMPLE_RATE_HZ,
            backend_kind: None,
            detail: None,
            notes: &[
                "Runs a real HiFi-GAN-compatible ONNX mel vocoder when loaded with a model.",
                "Consumes model-specific mel spectrograms (not text, phones, or language-side features).",
                "The deterministic local renderer is retained only as a compile-safe fallback for tests and non-ONNX builds.",
            ],
        }
    }

    fn render_mel(
        &mut self,
        mel: &MelSpectrogram,
        f0_hz: Option<&[f32]>,
        voiced: Option<&[bool]>,
    ) -> Result<Vec<AudioFrame>> {
        #[cfg(feature = "tts-riper")]
        if self.session.is_some() {
            return self.render_mel_onnx(mel);
        }

        Self::render_mel_deterministic(mel, f0_hz, voiced)
    }

    fn render_mel_deterministic(
        mel: &MelSpectrogram,
        f0_hz: Option<&[f32]>,
        voiced: Option<&[bool]>,
    ) -> Result<Vec<AudioFrame>> {
        let frames = &mel.frames;
        ensure!(
            !frames.is_empty(),
            "hifigan backend received empty mel input"
        );
        if let Some(f0_hz) = f0_hz {
            ensure!(
                f0_hz.len() == frames.len(),
                "hifigan backend received {} F0 values for {} mel frames",
                f0_hz.len(),
                frames.len()
            );
        }
        if let Some(voiced) = voiced {
            ensure!(
                voiced.len() == frames.len(),
                "hifigan backend received {} voiced flags for {} mel frames",
                voiced.len(),
                frames.len()
            );
        }
        ensure!(
            frames
                .iter()
                .all(|frame| frame.bins.iter().all(|bin| bin.is_finite())),
            "hifigan backend requires finite mel bins"
        );

        let mut phase = 0.0f32;
        let mut noise_state = 0x4d59_4446u32;
        let mut samples = Vec::with_capacity(frames.len() * HOP_SAMPLES);

        for (frame_index, frame) in frames.iter().enumerate() {
            let next_frame = frames.get(frame_index + 1).unwrap_or(frame);
            let f0_start = f0_for_frame(frame, f0_hz.map(|values| values[frame_index]));
            let f0_end = f0_for_frame(
                next_frame,
                f0_hz.map(|values| {
                    values
                        .get(frame_index + 1)
                        .copied()
                        .unwrap_or(values[frame_index])
                }),
            );
            let voiced_start = voiced.map(|values| values[frame_index]).unwrap_or(true);
            let voiced_end = voiced
                .map(|values| values.get(frame_index + 1).copied().unwrap_or(voiced_start))
                .unwrap_or(voiced_start);
            let amp_start = amplitude_for_frame(frame);
            let amp_end = amplitude_for_frame(next_frame);
            let brightness_start = brightness_for_frame(frame);
            let brightness_end = brightness_for_frame(next_frame);

            for sample_index in 0..HOP_SAMPLES {
                let t = sample_index as f32 / HOP_SAMPLES as f32;
                let amp = lerp(amp_start, amp_end, t);
                let brightness = lerp(brightness_start, brightness_end, t);
                let frame_f0 = lerp(f0_start, f0_end, t);
                let is_voiced = if t < 0.5 { voiced_start } else { voiced_end };

                let value = if is_voiced {
                    phase = (phase + TAU * frame_f0 / SAMPLE_RATE_HZ as f32) % TAU;
                    let harmonic_mix = 0.18 + brightness * 0.32;
                    let source = phase.sin()
                        + harmonic_mix * (phase * 2.0).sin()
                        + (harmonic_mix * 0.45) * (phase * 3.0).sin();
                    source * amp
                } else {
                    (next_noise_sample(&mut noise_state) * 2.0 - 1.0) * amp * NOISE_GAIN
                };
                samples.push(value.clamp(-1.0, 1.0));
            }
        }

        ensure!(!samples.is_empty(), "hifigan backend produced no audio");
        normalize_peak(&mut samples, 0.92);

        Ok(vec![AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: SAMPLE_RATE_HZ,
            channels: 1,
            samples,
            voice_signatures: Vec::new(),
        }])
    }

    #[cfg(feature = "tts-riper")]
    fn render_mel_onnx(&mut self, mel: &MelSpectrogram) -> Result<Vec<AudioFrame>> {
        ensure!(
            !mel.frames.is_empty(),
            "hifigan backend received empty mel input"
        );
        ensure!(
            mel.frames
                .iter()
                .all(|frame| frame.bins.iter().all(|bin| bin.is_finite())),
            "hifigan backend requires finite mel bins"
        );

        let model_path = self
            .model_path
            .as_ref()
            .context("HiFi-GAN ONNX model path is not loaded")?
            .clone();
        let session = self
            .session
            .as_mut()
            .context("HiFi-GAN ONNX session has not been loaded")?;
        let onnx_config = self
            .onnx_config
            .as_ref()
            .context("HiFi-GAN ONNX config was not loaded")?;
        validate_mel_compatibility(mel, &onnx_config.mel, &model_path)?;
        let (input_name, input_layout) = resolve_hifigan_input(session, onnx_config, &model_path)?;
        let output_name = resolve_hifigan_output_name(session, onnx_config, &model_path)?;
        let (shape, values) =
            mel_values_for_layout(&mel.frames, mel.config.n_mels, input_layout).with_context(
                || {
                    format!(
                        "failed to convert mel frames to HiFi-GAN tensor layout {:?} for model {} input `{input_name}`",
                        input_layout,
                        model_path.display()
                    )
                },
            )?;

        let tensor = Tensor::from_array((shape, values))
            .with_context(|| format!("failed to build HiFi-GAN ONNX `{input_name}` tensor"))?
            .upcast();
        let outputs = session
            .run(vec![(input_name.clone(), tensor)])
            .with_context(|| {
                format!(
                    "failed to run HiFi-GAN ONNX inference for model {}",
                    model_path.display()
                )
            })?;
        let output = outputs.get(output_name.as_str()).with_context(|| {
            format!("HiFi-GAN ONNX inference did not return expected output `{output_name}`")
        })?;
        let output = output
            .downcast_ref::<DynTensorValueType>()
            .with_context(|| format!("HiFi-GAN ONNX output `{output_name}` is not a tensor"))?;
        let (_, samples) = output.try_extract_tensor::<f32>().with_context(|| {
            format!("HiFi-GAN ONNX output `{output_name}` is not an f32 tensor")
        })?;
        ensure!(
            !samples.is_empty(),
            "HiFi-GAN ONNX inference returned an empty waveform output"
        );

        let mut samples = samples.to_vec();
        normalize_peak(&mut samples, 0.92);
        Ok(vec![AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: SPEECHT5_HIFIGAN_SAMPLE_RATE_HZ,
            channels: 1,
            samples,
            voice_signatures: Vec::new(),
        }])
    }
}

impl Default for HifiganBackend {
    fn default() -> Self {
        Self::deterministic()
    }
}

impl VocoderBackend for HifiganBackend {
    fn id(&self) -> &'static str {
        Self::descriptor().id
    }

    fn descriptor(&self) -> VocoderDescriptor {
        Self::descriptor()
    }

    fn render(&mut self, input: VocoderInput<'_>) -> Result<Vec<AudioFrame>> {
        match input {
            VocoderInput::Mel(mel) => self.render_mel(mel, None, None),
            VocoderInput::MelF0 { mel, f0_hz, voiced } => {
                self.render_mel(mel, Some(f0_hz), Some(voiced))
            }
            _ => bail!("hifigan backend requires Mel or MelF0 input"),
        }
    }
}

#[cfg(feature = "tts-riper")]
fn resolve_hifigan_input(
    session: &Session,
    onnx_config: &HifiganOnnxConfig,
    model_path: &Path,
) -> Result<(String, MelTensorLayout)> {
    let input = if let Some(name) = onnx_config.input_name.as_deref() {
        session
            .inputs()
            .iter()
            .find(|input| input.name() == name)
            .with_context(|| {
                format!(
                    "HiFi-GAN ONNX model `{}` does not expose configured input `{name}`",
                    model_path.display()
                )
            })?
    } else {
        let candidates = ["spectrogram", "input", "mel", "mel_spectrogram", "logmel"];
        if let Some(found) = candidates.iter().find_map(|candidate| {
            session
                .inputs()
                .iter()
                .find(|input| input.name() == *candidate)
        }) {
            found
        } else {
            session.inputs().first().with_context(|| {
                format!(
                    "HiFi-GAN ONNX model `{}` exposes no inputs",
                    model_path.display()
                )
            })?
        }
    };

    ensure!(
        input.dtype().tensor_type() == Some(TensorElementType::Float32),
        "HiFi-GAN ONNX input `{}` in `{}` is not f32",
        input.name(),
        model_path.display()
    );
    let shape = input.dtype().tensor_shape();
    let shape_values = shape.map(|dims| dims.iter().copied().collect::<Vec<_>>());
    let layout = if let Some(layout) = onnx_config.input_layout {
        validate_layout_matches_shape(
            layout,
            shape_values.as_deref(),
            onnx_config.mel.n_mels,
            model_path,
            input.name(),
        )?;
        layout
    } else {
        detect_layout_from_shape(shape_values.as_deref(), onnx_config.mel.n_mels).with_context(|| {
            format!(
                "HiFi-GAN ONNX model `{}` input `{}` has dynamic or ambiguous dimensions {:?}; set `input_layout` in the model config",
                model_path.display(),
                input.name(),
                shape
            )
        })?
    };
    Ok((input.name().to_string(), layout))
}

#[cfg(feature = "tts-riper")]
fn resolve_hifigan_output_name(
    session: &Session,
    onnx_config: &HifiganOnnxConfig,
    model_path: &Path,
) -> Result<String> {
    let output = if let Some(name) = onnx_config.output_name.as_deref() {
        session
            .outputs()
            .iter()
            .find(|output| output.name() == name)
            .with_context(|| {
                format!(
                    "HiFi-GAN ONNX model `{}` does not expose configured output `{name}`",
                    model_path.display()
                )
            })?
    } else {
        let candidates = ["waveform", "output", "audio", "y"];
        if let Some(found) = candidates.iter().find_map(|candidate| {
            session
                .outputs()
                .iter()
                .find(|output| output.name() == *candidate)
        }) {
            found
        } else {
            session.outputs().first().with_context(|| {
                format!(
                    "HiFi-GAN ONNX model `{}` exposes no outputs",
                    model_path.display()
                )
            })?
        }
    };

    ensure!(
        output.dtype().tensor_type() == Some(TensorElementType::Float32),
        "HiFi-GAN ONNX output `{}` in `{}` is not f32",
        output.name(),
        model_path.display()
    );
    Ok(output.name().to_string())
}

#[cfg(feature = "tts-riper")]
fn validate_mel_compatibility(
    mel: &MelSpectrogram,
    expected: &MelConfig,
    model_path: &Path,
) -> Result<()> {
    ensure!(
        mel.config.n_mels == expected.n_mels,
        "HiFi-GAN ONNX model `{}` expects n_mels={}, but received n_mels={}",
        model_path.display(),
        expected.n_mels,
        mel.config.n_mels
    );
    ensure!(
        mel.config.sample_rate_hz == expected.sample_rate_hz,
        "HiFi-GAN ONNX model `{}` expects sample_rate_hz={}, but received sample_rate_hz={}",
        model_path.display(),
        expected.sample_rate_hz,
        mel.config.sample_rate_hz
    );
    ensure!(
        mel.config.hop_samples == expected.hop_samples,
        "HiFi-GAN ONNX model `{}` expects hop_samples={}, but received hop_samples={}",
        model_path.display(),
        expected.hop_samples,
        mel.config.hop_samples
    );
    ensure!(
        mel.config.scale == expected.scale,
        "HiFi-GAN ONNX model `{}` expects mel scale {:?}, but received {:?}",
        model_path.display(),
        expected.scale,
        mel.config.scale
    );
    for (frame_index, frame) in mel.frames.iter().enumerate() {
        ensure!(
            frame.bins.len() == expected.n_mels,
            "HiFi-GAN ONNX model `{}` expects {} mel bins per frame, but frame {} has {} bins",
            model_path.display(),
            expected.n_mels,
            frame_index,
            frame.bins.len()
        );
    }
    Ok(())
}

#[cfg(feature = "tts-riper")]
fn mel_values_for_layout(
    mel: &[MelFrame],
    n_mels: usize,
    layout: MelTensorLayout,
) -> Result<(Vec<i64>, Vec<f32>)> {
    let frames = i64::try_from(mel.len()).context("HiFi-GAN mel sequence is too long")?;
    let bins = i64::try_from(n_mels).context("HiFi-GAN mel bin count is invalid")?;
    let mut values = Vec::with_capacity(mel.len() * n_mels);
    match layout {
        MelTensorLayout::FramesBins | MelTensorLayout::BatchFramesBins => {
            for frame in mel {
                values.extend(frame.bins.iter().copied());
            }
        }
        MelTensorLayout::BinsFrames | MelTensorLayout::BatchBinsFrames => {
            for bin_index in 0..n_mels {
                for frame in mel {
                    values.push(frame.bins[bin_index]);
                }
            }
        }
    }
    let shape = match layout {
        MelTensorLayout::FramesBins => vec![frames, bins],
        MelTensorLayout::BinsFrames => vec![bins, frames],
        MelTensorLayout::BatchFramesBins => vec![1_i64, frames, bins],
        MelTensorLayout::BatchBinsFrames => vec![1_i64, bins, frames],
    };
    Ok((shape, values))
}

#[cfg(feature = "tts-riper")]
fn detect_layout_from_shape(shape: Option<&[i64]>, n_mels: usize) -> Option<MelTensorLayout> {
    let shape = shape?;
    let n_mels = i64::try_from(n_mels).ok()?;
    match shape {
        [a, b] => match (*a, *b) {
            (x, y) if x == n_mels && y != n_mels => Some(MelTensorLayout::BinsFrames),
            (x, y) if y == n_mels && x != n_mels => Some(MelTensorLayout::FramesBins),
            _ => None,
        },
        [1, a, b] => match (*a, *b) {
            (x, y) if x == n_mels && y != n_mels => Some(MelTensorLayout::BatchBinsFrames),
            (x, y) if y == n_mels && x != n_mels => Some(MelTensorLayout::BatchFramesBins),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(feature = "tts-riper")]
fn validate_layout_matches_shape(
    layout: MelTensorLayout,
    shape: Option<&[i64]>,
    n_mels: usize,
    model_path: &Path,
    input_name: &str,
) -> Result<()> {
    let shape = shape.with_context(|| {
        format!(
            "HiFi-GAN ONNX model `{}` input `{}` has unknown shape; cannot validate configured layout {:?}",
            model_path.display(),
            input_name,
            layout
        )
    })?;
    let n_mels = i64::try_from(n_mels).context("n_mels does not fit in i64")?;
    let valid = match (layout, shape) {
        (MelTensorLayout::FramesBins, [_, bins]) => *bins == n_mels,
        (MelTensorLayout::BinsFrames, [bins, _]) => *bins == n_mels,
        (MelTensorLayout::BatchFramesBins, [1, _, bins]) => *bins == n_mels,
        (MelTensorLayout::BatchBinsFrames, [1, bins, _]) => *bins == n_mels,
        _ => false,
    };
    ensure!(
        valid,
        "HiFi-GAN ONNX model `{}` input `{}` shape {:?} does not match configured layout {:?} with n_mels={}",
        model_path.display(),
        input_name,
        shape,
        layout,
        n_mels
    );
    Ok(())
}

#[cfg(feature = "tts-riper")]
fn load_hifigan_model_config(model_path: &Path) -> Result<HifiganOnnxConfig> {
    let default = HifiganOnnxConfig {
        mel: HifiganBackend::default_mel_config(),
        input_name: None,
        output_name: None,
        input_layout: None,
    };
    let config_path = model_path.with_extension("config.json");
    if !config_path.is_file() {
        return Ok(default);
    }

    let raw = fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read HiFi-GAN config {}", config_path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse HiFi-GAN config {}", config_path.display()))?;
    let n_mels = value_usize(&json, &["num_mel_bins", "n_mels"]).unwrap_or(default.mel.n_mels);
    Ok(HifiganOnnxConfig {
        mel: MelConfig {
            sample_rate_hz: value_u32(&json, &["sampling_rate", "sample_rate", "sample_rate_hz"])
                .unwrap_or(default.mel.sample_rate_hz),
            hop_samples: value_usize(&json, &["hop_length", "hop_samples"])
                .unwrap_or(default.mel.hop_samples),
            n_fft: value_usize(&json, &["n_fft"]).unwrap_or(default.mel.n_fft),
            win_length: value_usize(&json, &["win_length"]).unwrap_or(default.mel.win_length),
            n_mels,
            f_min_hz: value_f32(&json, &["f_min", "fmin"]).unwrap_or(default.mel.f_min_hz),
            f_max_hz: value_f32(&json, &["f_max", "fmax"]).or(default.mel.f_max_hz),
            scale: parse_mel_scale(value_string(&json, &["mel_scale", "scale"]))
                .unwrap_or(default.mel.scale),
        },
        input_name: value_string(&json, &["input_name", "onnx_input_name"]),
        output_name: value_string(&json, &["output_name", "onnx_output_name"]),
        input_layout: parse_mel_tensor_layout(value_string(&json, &["input_layout"])),
    })
}

#[cfg(feature = "tts-riper")]
fn parse_mel_scale(value: Option<String>) -> Option<MelScale> {
    match value?.to_ascii_lowercase().as_str() {
        "linearenergy" | "linear_energy" | "linear" => Some(MelScale::LinearEnergy),
        "naturallogenergy" | "natural_log_energy" | "ln" | "log" => {
            Some(MelScale::NaturalLogEnergy)
        }
        "log10energy" | "log10_energy" | "log10" => Some(MelScale::Log10Energy),
        "dynamicrangecompressed" | "dynamic_range_compressed" | "drc" => {
            Some(MelScale::DynamicRangeCompressed)
        }
        other => Some(MelScale::ModelSpecific(other.to_string())),
    }
}

#[cfg(feature = "tts-riper")]
fn parse_mel_tensor_layout(value: Option<String>) -> Option<MelTensorLayout> {
    match value?.to_ascii_lowercase().as_str() {
        "framesbins" | "frames_bins" | "t_m" => Some(MelTensorLayout::FramesBins),
        "binsframes" | "bins_frames" | "m_t" => Some(MelTensorLayout::BinsFrames),
        "batchframesbins" | "batch_frames_bins" | "1_t_m" => Some(MelTensorLayout::BatchFramesBins),
        "batchbinsframes" | "batch_bins_frames" | "1_m_t" => Some(MelTensorLayout::BatchBinsFrames),
        _ => None,
    }
}

#[cfg(feature = "tts-riper")]
fn value_at_path<'a>(json: &'a serde_json::Value, path: &[&str]) -> Option<&'a serde_json::Value> {
    path.iter().find_map(|key| json.get(*key))
}

#[cfg(feature = "tts-riper")]
fn value_u32(json: &serde_json::Value, path: &[&str]) -> Option<u32> {
    value_at_path(json, path)?
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
}

#[cfg(feature = "tts-riper")]
fn value_usize(json: &serde_json::Value, path: &[&str]) -> Option<usize> {
    value_at_path(json, path)?
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
}

#[cfg(feature = "tts-riper")]
fn value_f32(json: &serde_json::Value, path: &[&str]) -> Option<f32> {
    value_at_path(json, path)?
        .as_f64()
        .map(|value| value as f32)
}

#[cfg(feature = "tts-riper")]
fn value_string(json: &serde_json::Value, path: &[&str]) -> Option<String> {
    value_at_path(json, path)?.as_str().map(ToOwned::to_owned)
}

fn amplitude_for_frame(frame: &MelFrame) -> f32 {
    if frame.bins.is_empty() {
        return 0.0;
    }
    let mean_abs = frame.bins.iter().map(|bin| bin.abs()).sum::<f32>() / frame.bins.len() as f32;
    let mean_positive =
        frame.bins.iter().map(|bin| bin.max(0.0)).sum::<f32>() / frame.bins.len() as f32;
    let level = if mean_abs > 2.0 {
        10.0f32.powf((mean_abs - 80.0) / 40.0)
    } else {
        mean_positive.max(mean_abs * 0.25)
    };
    level.sqrt().clamp(0.0, 0.35)
}

fn brightness_for_frame(frame: &MelFrame) -> f32 {
    if frame.bins.is_empty() {
        return 0.0;
    }
    let mut weighted = 0.0f32;
    let mut total = 0.0f32;
    let max_index = (frame.bins.len() - 1).max(1) as f32;
    for (index, bin) in frame.bins.iter().enumerate() {
        let energy = bin.max(0.0).abs();
        weighted += energy * (index as f32 / max_index);
        total += energy;
    }
    if total <= f32::EPSILON {
        0.0
    } else {
        (weighted / total).clamp(0.0, 1.0)
    }
}

fn f0_for_frame(frame: &MelFrame, explicit_f0: Option<f32>) -> f32 {
    explicit_f0
        .filter(|hz| hz.is_finite() && *hz > 0.0)
        .unwrap_or_else(|| 90.0 + brightness_for_frame(frame).powf(1.4) * 410.0)
        .clamp(MIN_F0_HZ, MAX_F0_HZ)
}

fn lerp(start: f32, end: f32, t: f32) -> f32 {
    start + (end - start) * t.clamp(0.0, 1.0)
}

fn next_noise_sample(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    ((*state >> 8) as f32) / ((u32::MAX >> 8) as f32)
}

fn normalize_peak(samples: &mut [f32], target_peak: f32) {
    let peak = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0f32, f32::max);
    if peak >= MIN_NORMALIZABLE_PEAK && peak.is_finite() && target_peak.is_finite() {
        let gain = target_peak / peak;
        for sample in samples {
            *sample *= gain;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_spectrogram(n_mels: usize) -> MelSpectrogram {
        MelSpectrogram {
            config: MelConfig::test_default(n_mels),
            frames: vec![
                MelFrame {
                    bins: (0..n_mels).map(|idx| idx as f32).collect(),
                },
                MelFrame {
                    bins: (0..n_mels).map(|idx| (idx as f32) + 10.0).collect(),
                },
            ],
        }
    }

    #[test]
    fn normalize_peak_lifts_quiet_vocoder_output() {
        let mut samples = vec![0.0, 0.01, -0.02, 0.005];

        normalize_peak(&mut samples, 0.92);

        let peak = samples
            .iter()
            .map(|sample| sample.abs())
            .fold(0.0f32, f32::max);
        assert!((peak - 0.92).abs() < 1.0e-6);
    }

    #[test]
    fn normalize_peak_keeps_near_silence_silent() {
        let mut samples = vec![0.0, 0.000_001, -0.000_002];

        normalize_peak(&mut samples, 0.92);

        assert_eq!(samples, vec![0.0, 0.000_001, -0.000_002]);
    }

    #[cfg(feature = "tts-riper")]
    #[test]
    fn mel_layout_conversion_supports_all_layouts() {
        let mel = test_spectrogram(3);

        let (shape, values) =
            mel_values_for_layout(&mel.frames, mel.config.n_mels, MelTensorLayout::FramesBins)
                .expect("frames-bins");
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(values, vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0]);

        let (shape, values) =
            mel_values_for_layout(&mel.frames, mel.config.n_mels, MelTensorLayout::BinsFrames)
                .expect("bins-frames");
        assert_eq!(shape, vec![3, 2]);
        assert_eq!(values, vec![0.0, 10.0, 1.0, 11.0, 2.0, 12.0]);

        let (shape, values) = mel_values_for_layout(
            &mel.frames,
            mel.config.n_mels,
            MelTensorLayout::BatchFramesBins,
        )
        .expect("batch-frames-bins");
        assert_eq!(shape, vec![1, 2, 3]);
        assert_eq!(values, vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0]);

        let (shape, values) = mel_values_for_layout(
            &mel.frames,
            mel.config.n_mels,
            MelTensorLayout::BatchBinsFrames,
        )
        .expect("batch-bins-frames");
        assert_eq!(shape, vec![1, 3, 2]);
        assert_eq!(values, vec![0.0, 10.0, 1.0, 11.0, 2.0, 12.0]);
    }

    #[cfg(feature = "tts-riper")]
    #[test]
    fn mel_compatibility_validation_is_explicit() {
        let expected = HifiganBackend::default_mel_config();
        let mut mel = test_spectrogram(expected.n_mels);
        mel.config.sample_rate_hz = expected.sample_rate_hz + 1;

        let err = validate_mel_compatibility(&mel, &expected, Path::new("model.onnx"))
            .expect_err("sample rate mismatch should fail");
        assert!(err.to_string().contains("expects sample_rate_hz"));
    }

    #[cfg(feature = "tts-riper")]
    #[test]
    fn layout_detection_handles_common_hifigan_shapes() {
        assert_eq!(
            detect_layout_from_shape(Some(&[1, 80, 12]), 80),
            Some(MelTensorLayout::BatchBinsFrames)
        );
        assert_eq!(
            detect_layout_from_shape(Some(&[1, 12, 80]), 80),
            Some(MelTensorLayout::BatchFramesBins)
        );
        assert_eq!(
            detect_layout_from_shape(Some(&[80, 12]), 80),
            Some(MelTensorLayout::BinsFrames)
        );
        assert_eq!(
            detect_layout_from_shape(Some(&[12, 80]), 80),
            Some(MelTensorLayout::FramesBins)
        );
    }
}
