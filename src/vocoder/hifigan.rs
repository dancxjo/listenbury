use std::f32::consts::TAU;
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
    BackendCapabilities, BackendFamily, MelFrame, VocoderBackend, VocoderDescriptor, VocoderInput,
};

pub struct HifiganBackend {
    #[cfg(feature = "tts-riper")]
    model_path: Option<PathBuf>,
    #[cfg(feature = "tts-riper")]
    session: Option<Session>,
}

const SAMPLE_RATE_HZ: u32 = 16_000;
const HOP_SAMPLES: usize = 256;
const MIN_F0_HZ: f32 = 55.0;
const MAX_F0_HZ: f32 = 1_200.0;
const NOISE_GAIN: f32 = 0.018;
const MODEL_MEL_BINS: usize = 80;
const LOG_MEL_MIN: f32 = -8.0;
const LOG_MEL_MAX: f32 = 2.0;
const MIN_NORMALIZABLE_PEAK: f32 = 1.0e-4;
const CLIP_THRESHOLD: f32 = 0.99;
const SILENCE_THRESHOLD: f32 = 1.0e-5;
const MODEL_N_FFT: usize = 1_024;
const MODEL_WIN_LENGTH: usize = 1_024;
const MODEL_FMIN_HZ: f32 = 0.0;
const MODEL_FMAX_HZ: f32 = 8_000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelScale {
    Htk,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelNormalization {
    NaturalLogClamped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelTensorLayout {
    FramesBins,
    BinsFrames,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MelContract {
    pub sample_rate_hz: u32,
    pub hop_samples: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub mel_bins: usize,
    pub f_min_hz: f32,
    pub f_max_hz: f32,
    pub scale: MelScale,
    pub normalization: MelNormalization,
    pub log_floor: f32,
    pub log_ceiling: f32,
}

pub const SPEECHT5_HIFIGAN_MEL_CONTRACT: MelContract = MelContract {
    sample_rate_hz: SAMPLE_RATE_HZ,
    hop_samples: HOP_SAMPLES,
    n_fft: MODEL_N_FFT,
    win_length: MODEL_WIN_LENGTH,
    mel_bins: MODEL_MEL_BINS,
    f_min_hz: MODEL_FMIN_HZ,
    f_max_hz: MODEL_FMAX_HZ,
    scale: MelScale::Htk,
    normalization: MelNormalization::NaturalLogClamped,
    log_floor: LOG_MEL_MIN,
    log_ceiling: LOG_MEL_MAX,
};

impl HifiganBackend {
    pub fn validate_acoustic_contract(sample_rate_hz: u32, hop_samples: usize) -> Result<()> {
        let contract = SPEECHT5_HIFIGAN_MEL_CONTRACT;
        ensure!(
            sample_rate_hz == contract.sample_rate_hz,
            "hifigan backend requires {} Hz acoustic input sample rate, got {} Hz",
            contract.sample_rate_hz,
            sample_rate_hz
        );
        ensure!(
            hop_samples == contract.hop_samples,
            "hifigan backend requires {}-sample acoustic hop length, got {} samples",
            contract.hop_samples,
            hop_samples
        );
        Ok(())
    }

    pub fn deterministic() -> Self {
        Self {
            #[cfg(feature = "tts-riper")]
            model_path: None,
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
                "Expects SpeechT5 HiFi-GAN mel frames: 80-bin natural-log mel, clamped to [-8, 2], 16 kHz, 256-sample hop.",
                "Duration control belongs to an upstream acoustic model that lays out mel/F0 frames.",
                "Runs a real HiFi-GAN-compatible ONNX mel vocoder when loaded with a model; the deterministic renderer is a compile-safe fallback.",
            ],
        }
    }

    fn render_mel(
        &mut self,
        mel: &[MelFrame],
        f0_hz: Option<&[f32]>,
        voiced: Option<&[bool]>,
    ) -> Result<Vec<AudioFrame>> {
        validate_mel_f0_tracks(mel, f0_hz, voiced)?;
        validate_speecht5_hifigan_mel(mel)?;
        log_hifigan_mel_summary(mel);

        #[cfg(feature = "tts-riper")]
        if self.session.is_some() {
            return self.render_mel_onnx(mel);
        }

        let frames = Self::render_mel_deterministic(mel, f0_hz, voiced)?;
        log_hifigan_waveform_summary("deterministic", &frames);
        Ok(frames)
    }

    fn render_mel_deterministic(
        mel: &[MelFrame],
        f0_hz: Option<&[f32]>,
        voiced: Option<&[bool]>,
    ) -> Result<Vec<AudioFrame>> {
        let mut phase = 0.0f32;
        let mut noise_state = 0x4d59_4446u32;
        let mut samples = Vec::with_capacity(mel.len() * HOP_SAMPLES);

        for (frame_index, frame) in mel.iter().enumerate() {
            let next_frame = mel.get(frame_index + 1).unwrap_or(frame);
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
        normalize_loudness(&mut samples, 0.075, 0.92);

        Ok(vec![AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: SAMPLE_RATE_HZ,
            channels: 1,
            samples,
            voice_signatures: Vec::new(),
        }])
    }

    #[cfg(feature = "tts-riper")]
    fn render_mel_onnx(&mut self, mel: &[MelFrame]) -> Result<Vec<AudioFrame>> {
        let model_path = self
            .model_path
            .as_ref()
            .context("HiFi-GAN ONNX model path is not loaded")?
            .clone();
        let session = self
            .session
            .as_mut()
            .context("HiFi-GAN ONNX session has not been loaded")?;
        let (input_name, input_shape, layout) = resolve_hifigan_input(session, &model_path)?;
        let output_name = resolve_hifigan_output_name(session, &model_path)?;
        let frames = i64::try_from(mel.len()).context("HiFi-GAN mel sequence is too long")?;
        let bins = i64::try_from(MODEL_MEL_BINS).context("HiFi-GAN mel bin count is invalid")?;
        let values = flatten_contract_mel(mel, layout);
        let shape = match input_shape.as_deref().map(|shape| shape.len()) {
            Some(2) => match layout {
                MelTensorLayout::FramesBins => vec![frames, bins],
                MelTensorLayout::BinsFrames => vec![bins, frames],
            },
            Some(3) | None => match layout {
                MelTensorLayout::FramesBins => vec![1_i64, frames, bins],
                MelTensorLayout::BinsFrames => vec![1_i64, bins, frames],
            },
            Some(rank) => bail!(
                "HiFi-GAN ONNX model `{}` expects rank-{rank} input `{input_name}`, but Listenbury can provide rank-2 or rank-3 mel tensors",
                model_path.display()
            ),
        };
        tracing::debug!(
            model = %model_path.display(),
            input = %input_name,
            tensor_layout = ?layout,
            model_input_shape = ?input_shape,
            requested_shape = ?shape,
            mel_frames = mel.len(),
            mel_bins = MODEL_MEL_BINS,
            "hifigan onnx input contract"
        );

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
        normalize_loudness(&mut samples, 0.075, 0.92);
        let frames = vec![AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: SAMPLE_RATE_HZ,
            channels: 1,
            samples,
            voice_signatures: Vec::new(),
        }];
        log_hifigan_waveform_summary("onnx", &frames);
        Ok(frames)
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
            _ => bail!("hifigan backend requires Mel or MelF0 input from an acoustic model"),
        }
    }
}

#[cfg(feature = "tts-riper")]
fn resolve_hifigan_input(
    session: &Session,
    model_path: &Path,
) -> Result<(String, Option<Vec<i64>>, MelTensorLayout)> {
    let candidates = ["spectrogram", "input", "mel", "mel_spectrogram", "logmel"];
    for candidate in candidates {
        if let Some(input) = session
            .inputs()
            .iter()
            .find(|input| input.name() == candidate)
        {
            ensure!(
                input.dtype().tensor_type() == Some(TensorElementType::Float32),
                "HiFi-GAN ONNX input `{candidate}` in `{}` is not f32",
                model_path.display()
            );
            let shape = input.dtype().tensor_shape().map(|shape| shape.to_vec());
            let layout = infer_mel_tensor_layout(shape.as_deref());
            return Ok((candidate.to_string(), shape, layout));
        }
    }
    let input = session.inputs().first().with_context(|| {
        format!(
            "HiFi-GAN ONNX model `{}` exposes no inputs",
            model_path.display()
        )
    })?;
    ensure!(
        input.dtype().tensor_type() == Some(TensorElementType::Float32),
        "HiFi-GAN ONNX input `{}` in `{}` is not f32",
        input.name(),
        model_path.display()
    );
    let shape = input.dtype().tensor_shape().map(|shape| shape.to_vec());
    let layout = infer_mel_tensor_layout(shape.as_deref());
    Ok((input.name().to_string(), shape, layout))
}

#[cfg(feature = "tts-riper")]
fn resolve_hifigan_output_name(session: &Session, model_path: &Path) -> Result<String> {
    let candidates = ["waveform", "output", "audio", "y"];
    for candidate in candidates {
        if let Some(output) = session
            .outputs()
            .iter()
            .find(|output| output.name() == candidate)
        {
            ensure!(
                output.dtype().tensor_type() == Some(TensorElementType::Float32),
                "HiFi-GAN ONNX output `{candidate}` in `{}` is not f32",
                model_path.display()
            );
            return Ok(candidate.to_string());
        }
    }
    let output = session.outputs().first().with_context(|| {
        format!(
            "HiFi-GAN ONNX model `{}` exposes no outputs",
            model_path.display()
        )
    })?;
    ensure!(
        output.dtype().tensor_type() == Some(TensorElementType::Float32),
        "HiFi-GAN ONNX output `{}` in `{}` is not f32",
        output.name(),
        model_path.display()
    );
    Ok(output.name().to_string())
}

fn validate_mel_f0_tracks(
    mel: &[MelFrame],
    f0_hz: Option<&[f32]>,
    voiced: Option<&[bool]>,
) -> Result<()> {
    ensure!(!mel.is_empty(), "hifigan backend received empty mel input");
    if let Some(f0_hz) = f0_hz {
        ensure!(
            f0_hz.len() == mel.len(),
            "hifigan backend received {} F0 values for {} mel frames",
            f0_hz.len(),
            mel.len()
        );
    }
    if let Some(voiced) = voiced {
        ensure!(
            voiced.len() == mel.len(),
            "hifigan backend received {} voiced flags for {} mel frames",
            voiced.len(),
            mel.len()
        );
    }
    Ok(())
}

fn validate_speecht5_hifigan_mel(mel: &[MelFrame]) -> Result<()> {
    let contract = SPEECHT5_HIFIGAN_MEL_CONTRACT;
    ensure!(!mel.is_empty(), "hifigan backend received empty mel input");
    for (frame_index, frame) in mel.iter().enumerate() {
        ensure!(
            frame.bins.len() == contract.mel_bins,
            "hifigan backend requires {} mel bins per frame for the SpeechT5 HiFi-GAN contract; frame {} has {}",
            contract.mel_bins,
            frame_index,
            frame.bins.len()
        );
        for (bin_index, bin) in frame.bins.iter().enumerate() {
            ensure!(
                bin.is_finite(),
                "hifigan backend requires finite mel bins; frame {frame_index} bin {bin_index} is {bin}"
            );
            ensure!(
                *bin >= contract.log_floor && *bin <= contract.log_ceiling,
                "hifigan backend requires natural-log mel bins in [{}, {}]; frame {} bin {} is {}",
                contract.log_floor,
                contract.log_ceiling,
                frame_index,
                bin_index,
                bin
            );
        }
    }
    Ok(())
}

fn log_hifigan_mel_summary(mel: &[MelFrame]) {
    let stats = summarize_mel_values(mel);
    let ranges = summarize_per_band_ranges(mel);
    tracing::debug!(
        contract_sample_rate_hz = SPEECHT5_HIFIGAN_MEL_CONTRACT.sample_rate_hz,
        contract_hop_samples = SPEECHT5_HIFIGAN_MEL_CONTRACT.hop_samples,
        contract_n_fft = SPEECHT5_HIFIGAN_MEL_CONTRACT.n_fft,
        contract_win_length = SPEECHT5_HIFIGAN_MEL_CONTRACT.win_length,
        contract_mel_bins = SPEECHT5_HIFIGAN_MEL_CONTRACT.mel_bins,
        contract_f_min_hz = SPEECHT5_HIFIGAN_MEL_CONTRACT.f_min_hz,
        contract_f_max_hz = SPEECHT5_HIFIGAN_MEL_CONTRACT.f_max_hz,
        contract_scale = ?SPEECHT5_HIFIGAN_MEL_CONTRACT.scale,
        contract_normalization = ?SPEECHT5_HIFIGAN_MEL_CONTRACT.normalization,
        tensor_layout = ?MelTensorLayout::FramesBins,
        frame_count = mel.len(),
        value_count = stats.count,
        min = stats.min,
        max = stats.max,
        mean = stats.mean,
        rms = stats.rms,
        nan_count = stats.nan_count,
        inf_count = stats.inf_count,
        per_band_minmax = ?ranges,
        "hifigan mel contract summary"
    );
}

fn log_hifigan_waveform_summary(renderer: &str, frames: &[AudioFrame]) {
    let stats = summarize_waveform_values(frames);
    tracing::debug!(
        renderer,
        frame_count = frames.len(),
        sample_rate_hz = frames
            .first()
            .map(|frame| frame.sample_rate_hz)
            .unwrap_or(0),
        samples = stats.count,
        min = stats.min,
        max = stats.max,
        mean = stats.mean,
        rms = stats.rms,
        clip_count = stats.clip_count,
        silence_count = stats.silence_count,
        nan_count = stats.nan_count,
        inf_count = stats.inf_count,
        "hifigan waveform summary"
    );
}

#[derive(Debug, Clone, Copy)]
struct ScalarStats {
    min: f32,
    max: f32,
    mean: f32,
    rms: f32,
    count: usize,
    nan_count: usize,
    inf_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct WaveformStats {
    min: f32,
    max: f32,
    mean: f32,
    rms: f32,
    count: usize,
    clip_count: usize,
    silence_count: usize,
    nan_count: usize,
    inf_count: usize,
}

fn summarize_mel_values(mel: &[MelFrame]) -> ScalarStats {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    let mut count = 0usize;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    for frame in mel {
        for value in &frame.bins {
            if value.is_nan() {
                nan_count += 1;
                continue;
            }
            if value.is_infinite() {
                inf_count += 1;
                continue;
            }
            min = min.min(*value);
            max = max.max(*value);
            sum += *value;
            sum_sq += value * value;
            count += 1;
        }
    }

    if count == 0 {
        return ScalarStats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            rms: 0.0,
            count,
            nan_count,
            inf_count,
        };
    }

    ScalarStats {
        min,
        max,
        mean: sum / count as f32,
        rms: (sum_sq / count as f32).sqrt(),
        count,
        nan_count,
        inf_count,
    }
}

fn summarize_per_band_ranges(mel: &[MelFrame]) -> Vec<(f32, f32)> {
    let bins = SPEECHT5_HIFIGAN_MEL_CONTRACT.mel_bins;
    let mut mins = vec![f32::INFINITY; bins];
    let mut maxes = vec![f32::NEG_INFINITY; bins];
    for frame in mel {
        for (index, value) in frame.bins.iter().enumerate() {
            mins[index] = mins[index].min(*value);
            maxes[index] = maxes[index].max(*value);
        }
    }
    mins.into_iter()
        .zip(maxes)
        .map(|(min, max)| {
            if min.is_infinite() || max.is_infinite() {
                (0.0, 0.0)
            } else {
                (min, max)
            }
        })
        .collect()
}

fn summarize_waveform_values(frames: &[AudioFrame]) -> WaveformStats {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    let mut count = 0usize;
    let mut clip_count = 0usize;
    let mut silence_count = 0usize;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    for frame in frames {
        for sample in &frame.samples {
            if sample.is_nan() {
                nan_count += 1;
                continue;
            }
            if sample.is_infinite() {
                inf_count += 1;
                continue;
            }
            min = min.min(*sample);
            max = max.max(*sample);
            sum += *sample;
            sum_sq += sample * sample;
            if sample.abs() >= CLIP_THRESHOLD {
                clip_count += 1;
            }
            if sample.abs() <= SILENCE_THRESHOLD {
                silence_count += 1;
            }
            count += 1;
        }
    }

    if count == 0 {
        return WaveformStats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            rms: 0.0,
            count,
            clip_count,
            silence_count,
            nan_count,
            inf_count,
        };
    }

    WaveformStats {
        min,
        max,
        mean: sum / count as f32,
        rms: (sum_sq / count as f32).sqrt(),
        count,
        clip_count,
        silence_count,
        nan_count,
        inf_count,
    }
}

#[cfg(feature = "tts-riper")]
fn flatten_contract_mel(mel: &[MelFrame], layout: MelTensorLayout) -> Vec<f32> {
    let mut values = Vec::with_capacity(mel.len() * SPEECHT5_HIFIGAN_MEL_CONTRACT.mel_bins);
    match layout {
        MelTensorLayout::FramesBins => {
            for frame in mel {
                values.extend_from_slice(&frame.bins);
            }
        }
        MelTensorLayout::BinsFrames => {
            for bin_index in 0..SPEECHT5_HIFIGAN_MEL_CONTRACT.mel_bins {
                for frame in mel {
                    values.push(frame.bins[bin_index]);
                }
            }
        }
    }
    values
}

fn infer_mel_tensor_layout(shape: Option<&[i64]>) -> MelTensorLayout {
    let Some(shape) = shape else {
        return MelTensorLayout::FramesBins;
    };
    match shape {
        [left, right] => match (
            *left == MODEL_MEL_BINS as i64,
            *right == MODEL_MEL_BINS as i64,
        ) {
            (true, false) => MelTensorLayout::BinsFrames,
            (false, true) => MelTensorLayout::FramesBins,
            _ => MelTensorLayout::FramesBins,
        },
        [_, middle, right] => match (
            *middle == MODEL_MEL_BINS as i64,
            *right == MODEL_MEL_BINS as i64,
        ) {
            (true, false) => MelTensorLayout::BinsFrames,
            (false, true) => MelTensorLayout::FramesBins,
            _ => MelTensorLayout::FramesBins,
        },
        _ => MelTensorLayout::FramesBins,
    }
}

fn amplitude_for_frame(frame: &MelFrame) -> f32 {
    if frame.bins.is_empty() {
        return 0.0;
    }
    let level = frame
        .bins
        .iter()
        .map(|bin| mel_bin_energy(*bin))
        .sum::<f32>()
        / frame.bins.len() as f32;
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
        let energy = mel_bin_energy(*bin);
        weighted += energy * (index as f32 / max_index);
        total += energy;
    }
    if total <= f32::EPSILON {
        0.0
    } else {
        (weighted / total).clamp(0.0, 1.0)
    }
}

fn mel_bin_energy(bin: f32) -> f32 {
    if (LOG_MEL_MIN..=LOG_MEL_MAX).contains(&bin) {
        bin.exp()
    } else {
        bin.max(0.0)
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

fn normalize_loudness(samples: &mut [f32], target_rms: f32, ceiling: f32) {
    if samples.is_empty() || !target_rms.is_finite() || !ceiling.is_finite() {
        return;
    }

    let rms =
        (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt();
    if rms >= MIN_NORMALIZABLE_PEAK && rms.is_finite() {
        let gain = (target_rms / rms).clamp(0.25, 16.0);
        for sample in samples.iter_mut() {
            *sample *= gain;
        }
    }

    let limit = ceiling.abs().max(MIN_NORMALIZABLE_PEAK);
    let knee = limit * 0.86;
    for sample in samples.iter_mut() {
        *sample = soft_limit(*sample, knee, limit);
    }
}

fn soft_limit(sample: f32, knee: f32, limit: f32) -> f32 {
    let sign = sample.signum();
    let magnitude = sample.abs();
    if magnitude <= knee {
        return sample;
    }

    let headroom = (limit - knee).max(MIN_NORMALIZABLE_PEAK);
    let curved = knee + (1.0 - (-(magnitude - knee) / headroom).exp()) * headroom;
    sign * curved.min(limit)
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    fn synthetic_mel_frames() -> Vec<MelFrame> {
        (0..6)
            .map(|frame_index| MelFrame {
                bins: (0..MODEL_MEL_BINS)
                    .map(|bin_index| {
                        let envelope = 1.0 - (bin_index as f32 / MODEL_MEL_BINS as f32);
                        ((0.12 + frame_index as f32 * 0.01) * envelope.max(0.05)).ln()
                    })
                    .collect(),
            })
            .collect()
    }

    #[test]
    fn infers_bins_frames_layout_from_3d_shape() {
        assert_eq!(
            infer_mel_tensor_layout(Some(&[1, MODEL_MEL_BINS as i64, -1])),
            MelTensorLayout::BinsFrames
        );
        assert_eq!(
            infer_mel_tensor_layout(Some(&[1, -1, MODEL_MEL_BINS as i64])),
            MelTensorLayout::FramesBins
        );
        assert_eq!(
            infer_mel_tensor_layout(Some(&[1, MODEL_MEL_BINS as i64, MODEL_MEL_BINS as i64])),
            MelTensorLayout::FramesBins
        );
    }

    #[cfg(feature = "tts-riper")]
    #[test]
    fn flatten_contract_mel_transposes_for_bins_frames_layout() {
        let mel = synthetic_mel_frames();
        let frames_bins = flatten_contract_mel(&mel, MelTensorLayout::FramesBins);
        let bins_frames = flatten_contract_mel(&mel, MelTensorLayout::BinsFrames);
        assert_eq!(frames_bins.len(), bins_frames.len());
        assert_eq!(frames_bins[0], mel[0].bins[0]);
        assert_eq!(bins_frames[0], mel[0].bins[0]);
        assert_eq!(bins_frames[1], mel[1].bins[0]);
        assert_eq!(bins_frames[mel.len()], mel[0].bins[1]);
    }

    #[test]
    fn validates_acoustic_contract_metadata() {
        HifiganBackend::validate_acoustic_contract(
            SPEECHT5_HIFIGAN_MEL_CONTRACT.sample_rate_hz,
            SPEECHT5_HIFIGAN_MEL_CONTRACT.hop_samples,
        )
        .expect("contract metadata should validate");
        let err = HifiganBackend::validate_acoustic_contract(
            SPEECHT5_HIFIGAN_MEL_CONTRACT.sample_rate_hz,
            SPEECHT5_HIFIGAN_MEL_CONTRACT.hop_samples + 1,
        )
        .expect_err("mismatched hop should fail");
        assert!(err.to_string().contains("hop length"));
    }

    #[test]
    fn synthetic_mel_distribution_matches_contract_bounds() {
        let mel = synthetic_mel_frames();
        validate_speecht5_hifigan_mel(&mel).expect("synthetic fixture should satisfy contract");
        let stats = summarize_mel_values(&mel);
        assert!(stats.min >= SPEECHT5_HIFIGAN_MEL_CONTRACT.log_floor);
        assert!(stats.max <= SPEECHT5_HIFIGAN_MEL_CONTRACT.log_ceiling);
        assert!(stats.rms > 0.0);
    }
}

#[cfg(test)]
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
    use crate::acoustic::{AcousticInput, AcousticModelBackend, SourceFilterAcousticModel};
    use crate::linguistic::phonology::Phone;
    use crate::voice::articulator::PhoneTimedRenderTarget;

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

    #[test]
    fn rejects_phone_timed_targets_without_acoustic_model() {
        let targets = vec![
            PhoneTimedRenderTarget {
                phone: Phone::new_ipa("h"),
                duration_ms: 60,
                f0_hz: None,
                amplitude: 0.7,
                vibrato: None,
            },
            PhoneTimedRenderTarget {
                phone: Phone::new_ipa("ɑ"),
                duration_ms: 140,
                f0_hz: Some(150.0),
                amplitude: 0.7,
                vibrato: None,
            },
        ];
        let mut backend = HifiganBackend::deterministic();

        let err = backend
            .render(VocoderInput::PhoneTimed(&targets))
            .expect_err("HiFi-GAN should require acoustic frames");

        assert!(err.to_string().contains("acoustic model"));
    }

    #[test]
    fn renders_acoustic_model_mel_f0_track() {
        let targets = vec![PhoneTimedRenderTarget {
            phone: Phone::new_ipa("ɑ"),
            duration_ms: 96,
            f0_hz: Some(150.0),
            amplitude: 0.7,
            vibrato: None,
        }];
        let mut acoustic = SourceFilterAcousticModel;
        let track = acoustic
            .generate(AcousticInput::PhoneTimed(&targets))
            .expect("acoustic track");
        let mut backend = HifiganBackend::deterministic();

        let frames = backend
            .render(VocoderInput::MelF0 {
                mel: &track.mel,
                f0_hz: &track.f0_hz,
                voiced: &track.voiced,
            })
            .expect("acoustic track HiFi-GAN render");

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].channels, 1);
        assert!(frames[0].samples.iter().any(|sample| sample.abs() > 0.0));
    }
}
