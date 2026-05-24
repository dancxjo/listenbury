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
use crate::vocoder::source_filter_mel::{
    phone_timed_to_source_filter_track, source_filter_track_to_mel_f0,
};
use crate::vocoder::{
    BackendCapabilities, BackendFamily, MelFrame, VocoderBackend, VocoderDescriptor, VocoderInput,
};
use crate::voice::articulator::{PhoneTimedRenderTarget, RenderPlan};
use crate::voice::tract::SourceFilterTrack;

pub struct HifiganBackend {
    #[cfg(feature = "tts-riper")]
    model_path: Option<PathBuf>,
    #[cfg(feature = "tts-riper")]
    session: Option<Session>,
}

const SAMPLE_RATE_HZ: u32 = 22_050;
const SPEECHT5_HIFIGAN_SAMPLE_RATE_HZ: u32 = 16_000;
const HOP_SAMPLES: usize = 256;
const MIN_F0_HZ: f32 = 55.0;
const MAX_F0_HZ: f32 = 1_200.0;
const NOISE_GAIN: f32 = 0.018;
const MODEL_MEL_BINS: usize = 80;
const MIN_NORMALIZABLE_PEAK: f32 = 1.0e-4;

impl HifiganBackend {
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
                "Runs a real HiFi-GAN-compatible ONNX mel vocoder when loaded with a model.",
                "The deterministic local renderer is retained only as a compile-safe fallback for tests and non-ONNX builds.",
            ],
        }
    }

    fn render_mel(
        &mut self,
        mel: &[MelFrame],
        f0_hz: Option<&[f32]>,
        voiced: Option<&[bool]>,
    ) -> Result<Vec<AudioFrame>> {
        #[cfg(feature = "tts-riper")]
        if self.session.is_some() {
            return self.render_mel_onnx(mel);
        }

        Self::render_mel_deterministic(mel, f0_hz, voiced)
    }

    fn render_phone_timed(
        &mut self,
        targets: &[PhoneTimedRenderTarget],
    ) -> Result<Vec<AudioFrame>> {
        let source_filter = phone_timed_to_source_filter_track(targets)?;
        self.render_source_filter_track(&source_filter)
    }

    fn render_source_filter_track(&mut self, track: &SourceFilterTrack) -> Result<Vec<AudioFrame>> {
        let mel_f0 = source_filter_track_to_mel_f0(track)?;
        self.render_mel(&mel_f0.mel, Some(&mel_f0.f0_hz), Some(&mel_f0.voiced))
    }

    fn render_mel_deterministic(
        mel: &[MelFrame],
        f0_hz: Option<&[f32]>,
        voiced: Option<&[bool]>,
    ) -> Result<Vec<AudioFrame>> {
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
        ensure!(
            mel.iter()
                .all(|frame| frame.bins.iter().all(|bin| bin.is_finite())),
            "hifigan backend requires finite mel bins"
        );

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
    fn render_mel_onnx(&mut self, mel: &[MelFrame]) -> Result<Vec<AudioFrame>> {
        ensure!(!mel.is_empty(), "hifigan backend received empty mel input");
        ensure!(
            mel.iter()
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
        let (input_name, input_rank) = resolve_hifigan_input(session, &model_path)?;
        let output_name = resolve_hifigan_output_name(session, &model_path)?;
        let frames = i64::try_from(mel.len()).context("HiFi-GAN mel sequence is too long")?;
        let bins = i64::try_from(MODEL_MEL_BINS).context("HiFi-GAN mel bin count is invalid")?;
        let values = normalize_mel_for_onnx(mel, MODEL_MEL_BINS);
        let shape = match input_rank {
            Some(2) => vec![frames, bins],
            Some(3) | None => vec![1_i64, frames, bins],
            Some(rank) => bail!(
                "HiFi-GAN ONNX model `{}` expects rank-{rank} input `{input_name}`, but Listenbury can provide rank-2 or rank-3 mel tensors",
                model_path.display()
            ),
        };

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
            VocoderInput::RenderPlan(RenderPlan::PhoneTimed(targets)) => {
                self.render_phone_timed(targets)
            }
            VocoderInput::RenderPlan(_) => {
                bail!("hifigan backend requires a phone-timed render plan or mel input")
            }
            VocoderInput::PhoneTimed(targets) => self.render_phone_timed(targets),
            VocoderInput::SourceFilterTrack(track) => self.render_source_filter_track(track),
            VocoderInput::Mel(mel) => self.render_mel(mel, None, None),
            VocoderInput::MelF0 { mel, f0_hz, voiced } => {
                self.render_mel(mel, Some(f0_hz), Some(voiced))
            }
            _ => bail!("hifigan backend requires PhoneTimed, Mel, or MelF0 input"),
        }
    }
}

#[cfg(feature = "tts-riper")]
fn resolve_hifigan_input(session: &Session, model_path: &Path) -> Result<(String, Option<usize>)> {
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
            return Ok((
                candidate.to_string(),
                input.dtype().tensor_shape().map(|shape| shape.len()),
            ));
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
    Ok((
        input.name().to_string(),
        input.dtype().tensor_shape().map(|shape| shape.len()),
    ))
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

#[cfg(feature = "tts-riper")]
fn normalize_mel_for_onnx(mel: &[MelFrame], target_bins: usize) -> Vec<f32> {
    if mel.is_empty() || target_bins == 0 {
        return Vec::new();
    }

    let mut values = Vec::with_capacity(mel.len() * target_bins);
    for frame in mel {
        let frame_peak = (0..target_bins)
            .map(|index| resample_bin(&frame.bins, index, target_bins))
            .fold(0.0_f32, f32::max);
        let adaptive_floor = (frame_peak * 0.006).max(1.0e-5);
        for index in 0..target_bins {
            let source = resample_bin(&frame.bins, index, target_bins);
            values.push(source.max(adaptive_floor).ln().clamp(-8.0, 2.0));
        }
    }
    smooth_normalized_mel_frames(&mut values, target_bins);
    values
}

#[cfg(feature = "tts-riper")]
fn smooth_normalized_mel_frames(values: &mut [f32], bins: usize) {
    if bins == 0 || values.len() < bins * 3 {
        return;
    }
    let original = values.to_vec();
    let frames = values.len() / bins;
    for frame_index in 1..frames - 1 {
        for bin in 0..bins {
            let index = frame_index * bins + bin;
            let neighbor_mean = (original[index - bins] + original[index + bins]) * 0.5;
            values[index] = original[index] * 0.82 + neighbor_mean * 0.18;
        }
    }
}

#[cfg(feature = "tts-riper")]
fn resample_bin(bins: &[f32], target_index: usize, target_bins: usize) -> f32 {
    if bins.is_empty() {
        return 0.0;
    }
    if bins.len() == 1 || target_bins <= 1 {
        return bins[0];
    }
    let source_position =
        target_index as f32 * (bins.len() - 1) as f32 / (target_bins - 1).max(1) as f32;
    let left = source_position.floor() as usize;
    let right = source_position.ceil() as usize;
    if left == right {
        bins[left]
    } else {
        lerp(bins[left], bins[right], source_position - left as f32)
    }
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
    use crate::linguistic::phonology::Phone;

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
    fn renders_phone_timed_targets_through_mel_adapter() {
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

        let frames = backend
            .render(VocoderInput::PhoneTimed(&targets))
            .expect("phone-timed HiFi-GAN render");

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].sample_rate_hz, SAMPLE_RATE_HZ);
        assert_eq!(frames[0].channels, 1);
        assert!(frames[0].samples.len() >= HOP_SAMPLES * 2);
        assert!(frames[0].samples.iter().any(|sample| sample.abs() > 0.0));
    }

    #[test]
    fn renders_source_filter_track_through_mel_adapter() {
        let targets = vec![PhoneTimedRenderTarget {
            phone: Phone::new_ipa("ɑ"),
            duration_ms: 96,
            f0_hz: Some(150.0),
            amplitude: 0.7,
            vibrato: None,
        }];
        let track = phone_timed_to_source_filter_track(&targets).expect("source-filter track");
        let mut backend = HifiganBackend::deterministic();

        let frames = backend
            .render(VocoderInput::SourceFilterTrack(&track))
            .expect("source-filter HiFi-GAN render");

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].channels, 1);
        assert!(frames[0].samples.iter().any(|sample| sample.abs() > 0.0));
    }
}
