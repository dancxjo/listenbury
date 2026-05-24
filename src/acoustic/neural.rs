use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail, ensure};

use crate::acoustic::{
    AcousticFrameTrack, AcousticInput, AcousticModelBackend, MelFrame,
    registry::AcousticModelDescriptor,
};
use crate::voice::articulator::PhoneTimedRenderTarget;

#[cfg(feature = "tts-onnx")]
use ort::session::{Session, builder::GraphOptimizationLevel};
#[cfg(feature = "tts-onnx")]
use ort::value::{DynTensorValueType, Tensor, TensorElementType};

#[cfg(feature = "tts-onnx")]
use crate::mouth::riper::backend::initialize_ort_runtime;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuralAcousticModelKind {
    FastSpeech2,
    Matcha,
    VitsPiper,
    SpeechT5,
}

impl NeuralAcousticModelKind {
    pub const fn id(self) -> &'static str {
        match self {
            Self::FastSpeech2 => "fastspeech2",
            Self::Matcha => "matcha",
            Self::VitsPiper => "vits-piper",
            Self::SpeechT5 => "speecht5",
        }
    }

    pub const fn display_name(self) -> &'static str {
        match self {
            Self::FastSpeech2 => "FastSpeech2",
            Self::Matcha => "Matcha",
            Self::VitsPiper => "VITS/Piper",
            Self::SpeechT5 => "SpeechT5",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuralMelOutputLayout {
    Auto,
    FramesBins,
    BinsFrames,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeuralAcousticTensorNames {
    pub token_ids: String,
    pub token_lengths: Option<String>,
    pub durations: Option<String>,
    pub f0_hz: Option<String>,
    pub output_mel: String,
    pub output_f0_hz: Option<String>,
    pub output_voiced: Option<String>,
}

impl Default for NeuralAcousticTensorNames {
    fn default() -> Self {
        Self {
            token_ids: "input_ids".to_string(),
            token_lengths: Some("input_lengths".to_string()),
            durations: None,
            f0_hz: None,
            output_mel: "mel".to_string(),
            output_f0_hz: None,
            output_voiced: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeuralAcousticTrackContract {
    pub sample_rate_hz: u32,
    pub hop_samples: usize,
    pub mel_bins: usize,
}

impl Default for NeuralAcousticTrackContract {
    fn default() -> Self {
        Self {
            sample_rate_hz: 16_000,
            hop_samples: 256,
            mel_bins: 80,
        }
    }
}

pub type NeuralPhoneIdMap = BTreeMap<String, i64>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeuralAcousticOnnxConfig {
    pub kind: NeuralAcousticModelKind,
    pub model_path: PathBuf,
    pub tensor_names: NeuralAcousticTensorNames,
    pub track_contract: NeuralAcousticTrackContract,
    pub phone_ids: NeuralPhoneIdMap,
    pub mel_output_layout: NeuralMelOutputLayout,
}

pub struct NeuralAcousticModel {
    kind: NeuralAcousticModelKind,
    runtime: NeuralAcousticRuntime,
}

enum NeuralAcousticRuntime {
    Unloaded,
    #[cfg(feature = "tts-onnx")]
    Onnx(OnnxAcousticRuntime),
}

#[cfg(feature = "tts-onnx")]
struct OnnxAcousticRuntime {
    session: Session,
    config: NeuralAcousticOnnxConfig,
}

pub type FastSpeech2AcousticModel = NeuralAcousticModel;
pub type MatchaAcousticModel = NeuralAcousticModel;
pub type VitsPiperAcousticModel = NeuralAcousticModel;
pub type SpeechT5AcousticModel = NeuralAcousticModel;

impl NeuralAcousticModel {
    pub const fn new(kind: NeuralAcousticModelKind) -> Self {
        Self {
            kind,
            runtime: NeuralAcousticRuntime::Unloaded,
        }
    }

    pub fn descriptor_for(kind: NeuralAcousticModelKind) -> AcousticModelDescriptor {
        AcousticModelDescriptor {
            id: kind.id(),
            notes: &[
                "Neural acoustic backend slot that produces AcousticFrameTrack mel/F0 output.",
                "Use NeuralAcousticModel::load_onnx with a model path, tensor names, and model-specific phone/token IDs to run an actual acoustic model.",
            ],
        }
    }

    #[cfg(feature = "tts-onnx")]
    pub fn load_onnx(config: NeuralAcousticOnnxConfig) -> Result<Self> {
        ensure!(
            config.model_path.is_file(),
            "{} acoustic ONNX model file not found at {}",
            config.kind.display_name(),
            config.model_path.display()
        );
        ensure!(
            config.track_contract.sample_rate_hz > 0,
            "neural acoustic track contract requires a non-zero sample rate"
        );
        ensure!(
            config.track_contract.hop_samples > 0,
            "neural acoustic track contract requires a non-zero hop size"
        );
        ensure!(
            config.track_contract.mel_bins > 0,
            "neural acoustic track contract requires at least one mel bin"
        );

        initialize_ort_runtime()?;
        let session = Session::builder()
            .context("failed to create neural acoustic ONNX session builder")?
            .with_intra_threads(1)
            .map_err(|error| {
                anyhow::anyhow!(
                    "failed to configure neural acoustic ONNX intra-op threads: {error}"
                )
            })?
            .with_inter_threads(1)
            .map_err(|error| {
                anyhow::anyhow!(
                    "failed to configure neural acoustic ONNX inter-op threads: {error}"
                )
            })?
            .with_intra_op_spinning(false)
            .map_err(|error| {
                anyhow::anyhow!(
                    "failed to configure neural acoustic ONNX intra-op spinning: {error}"
                )
            })?
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .map_err(|error| {
                anyhow::anyhow!(
                    "failed to configure neural acoustic ONNX optimization level: {error}"
                )
            })?
            .commit_from_file(&config.model_path)
            .with_context(|| {
                format!(
                    "failed to load {} acoustic ONNX model from {}",
                    config.kind.display_name(),
                    config.model_path.display()
                )
            })?;

        validate_session_contract(&session, &config)?;

        Ok(Self {
            kind: config.kind,
            runtime: NeuralAcousticRuntime::Onnx(OnnxAcousticRuntime { session, config }),
        })
    }

    #[cfg(not(feature = "tts-onnx"))]
    pub fn load_onnx(config: NeuralAcousticOnnxConfig) -> Result<Self> {
        let _ = config;
        bail!("neural acoustic ONNX backends require the `tts-onnx` feature")
    }
}

impl AcousticModelBackend for NeuralAcousticModel {
    fn id(&self) -> &'static str {
        self.kind.id()
    }

    fn generate(&mut self, input: AcousticInput<'_>) -> Result<AcousticFrameTrack> {
        match &mut self.runtime {
            NeuralAcousticRuntime::Unloaded => bail!(
                "{} acoustic backend is registered but no model is loaded; construct it with NeuralAcousticModel::load_onnx",
                self.kind.display_name()
            ),
            #[cfg(feature = "tts-onnx")]
            NeuralAcousticRuntime::Onnx(runtime) => runtime.generate(input),
        }
    }
}

#[cfg(feature = "tts-onnx")]
impl OnnxAcousticRuntime {
    fn generate(&mut self, input: AcousticInput<'_>) -> Result<AcousticFrameTrack> {
        let token_ids = token_ids_for_input(input, &self.config.phone_ids)?;
        ensure!(
            !token_ids.is_empty(),
            "neural acoustic backend received no token IDs"
        );
        let token_len = i64::try_from(token_ids.len()).context("token sequence is too long")?;
        let mut inputs = Vec::new();

        let ids_tensor = Tensor::from_array((vec![1_i64, token_len], token_ids))
            .with_context(|| {
                format!(
                    "failed to build neural acoustic `{}` tensor",
                    self.config.tensor_names.token_ids
                )
            })?
            .upcast();
        inputs.push((self.config.tensor_names.token_ids.clone(), ids_tensor));

        if let Some(name) = &self.config.tensor_names.token_lengths {
            let len_tensor = Tensor::from_array((vec![1_i64], vec![token_len]))
                .with_context(|| format!("failed to build neural acoustic `{name}` tensor"))?
                .upcast();
            inputs.push((name.clone(), len_tensor));
        }

        if let Some(name) = &self.config.tensor_names.durations {
            let durations = frame_durations_for_input(input, &self.config.track_contract)?;
            let duration_len =
                i64::try_from(durations.len()).context("duration sequence is too long")?;
            let tensor = Tensor::from_array((vec![1_i64, duration_len], durations))
                .with_context(|| format!("failed to build neural acoustic `{name}` tensor"))?
                .upcast();
            inputs.push((name.clone(), tensor));
        }

        if let Some(name) = &self.config.tensor_names.f0_hz {
            let f0_hz = phone_f0_for_input(input)?;
            let f0_len = i64::try_from(f0_hz.len()).context("F0 sequence is too long")?;
            let tensor = Tensor::from_array((vec![1_i64, f0_len], f0_hz))
                .with_context(|| format!("failed to build neural acoustic `{name}` tensor"))?
                .upcast();
            inputs.push((name.clone(), tensor));
        }

        let outputs = self.session.run(inputs).with_context(|| {
            format!(
                "failed to run {} acoustic ONNX inference for model {}",
                self.config.kind.display_name(),
                self.config.model_path.display()
            )
        })?;

        let mel_output = outputs
            .get(self.config.tensor_names.output_mel.as_str())
            .with_context(|| {
                format!(
                    "neural acoustic inference did not return expected mel output `{}`",
                    self.config.tensor_names.output_mel
                )
            })?;
        let mel_output = mel_output
            .downcast_ref::<DynTensorValueType>()
            .with_context(|| {
                format!(
                    "neural acoustic output `{}` is not a tensor",
                    self.config.tensor_names.output_mel
                )
            })?;
        let (mel_shape, mel_values) =
            mel_output.try_extract_tensor::<f32>().with_context(|| {
                format!(
                    "neural acoustic output `{}` is not an f32 tensor",
                    self.config.tensor_names.output_mel
                )
            })?;
        let mel = mel_frames_from_tensor(
            mel_shape,
            mel_values,
            self.config.track_contract.mel_bins,
            self.config.mel_output_layout,
        )?;

        let frame_count = mel.len();
        let f0_hz = optional_f32_output(
            &outputs,
            self.config.tensor_names.output_f0_hz.as_deref(),
            frame_count,
            "F0",
        )?
        .unwrap_or_else(|| fallback_f0_track(input, frame_count, &self.config.track_contract));
        let voiced = optional_f32_output(
            &outputs,
            self.config.tensor_names.output_voiced.as_deref(),
            frame_count,
            "voiced",
        )?
        .map(|values| values.into_iter().map(|value| value > 0.5).collect())
        .unwrap_or_else(|| f0_hz.iter().map(|value| *value > 0.0).collect());

        Ok(AcousticFrameTrack {
            mel,
            f0_hz,
            voiced,
            sample_rate_hz: self.config.track_contract.sample_rate_hz,
            hop_samples: self.config.track_contract.hop_samples,
        })
    }
}

#[cfg(feature = "tts-onnx")]
fn validate_session_contract(session: &Session, config: &NeuralAcousticOnnxConfig) -> Result<()> {
    validate_input_name(session, &config.tensor_names.token_ids)?;
    if let Some(name) = &config.tensor_names.token_lengths {
        validate_input_name(session, name)?;
    }
    if let Some(name) = &config.tensor_names.durations {
        validate_input_name(session, name)?;
    }
    if let Some(name) = &config.tensor_names.f0_hz {
        validate_input_name(session, name)?;
    }
    validate_output_name(session, &config.tensor_names.output_mel)?;
    if let Some(name) = &config.tensor_names.output_f0_hz {
        validate_output_name(session, name)?;
    }
    if let Some(name) = &config.tensor_names.output_voiced {
        validate_output_name(session, name)?;
    }
    Ok(())
}

#[cfg(feature = "tts-onnx")]
fn validate_input_name(session: &Session, name: &str) -> Result<()> {
    let input = session
        .inputs()
        .iter()
        .find(|input| input.name() == name)
        .with_context(|| format!("neural acoustic ONNX model exposes no input named `{name}`"))?;
    ensure!(
        matches!(
            input.dtype().tensor_type(),
            Some(TensorElementType::Int64) | Some(TensorElementType::Float32)
        ),
        "neural acoustic ONNX input `{name}` must be int64 or f32"
    );
    Ok(())
}

#[cfg(feature = "tts-onnx")]
fn validate_output_name(session: &Session, name: &str) -> Result<()> {
    let output = session
        .outputs()
        .iter()
        .find(|output| output.name() == name)
        .with_context(|| format!("neural acoustic ONNX model exposes no output named `{name}`"))?;
    ensure!(
        output.dtype().tensor_type() == Some(TensorElementType::Float32),
        "neural acoustic ONNX output `{name}` must be f32"
    );
    Ok(())
}

fn token_ids_for_input(input: AcousticInput<'_>, phone_ids: &NeuralPhoneIdMap) -> Result<Vec<i64>> {
    match input {
        AcousticInput::TokenIds(ids) => Ok(ids.to_vec()),
        AcousticInput::PhoneTimed(targets) => phone_timed_to_token_ids(targets, phone_ids),
        AcousticInput::Singing(_) | AcousticInput::SourceFilterTrack(_) => bail!(
            "neural acoustic backend requires token IDs or phone-timed input with a model-specific phone ID map"
        ),
    }
}

fn phone_timed_to_token_ids(
    targets: &[PhoneTimedRenderTarget],
    phone_ids: &NeuralPhoneIdMap,
) -> Result<Vec<i64>> {
    ensure!(
        !phone_ids.is_empty(),
        "phone-timed neural acoustic input requires a model-specific phone ID map"
    );
    targets
        .iter()
        .map(|target| {
            let symbol = target.phone.ipa.as_str();
            phone_ids.get(symbol).copied().with_context(|| {
                format!("neural acoustic phone ID map has no entry for phone `{symbol}`")
            })
        })
        .collect()
}

fn frame_durations_for_input(
    input: AcousticInput<'_>,
    contract: &NeuralAcousticTrackContract,
) -> Result<Vec<i64>> {
    match input {
        AcousticInput::PhoneTimed(targets) => Ok(targets
            .iter()
            .map(|target| duration_ms_to_frames(target.duration_ms, contract))
            .collect()),
        AcousticInput::TokenIds(ids) => Ok(vec![1_i64; ids.len()]),
        AcousticInput::Singing(_) | AcousticInput::SourceFilterTrack(_) => {
            bail!("neural acoustic duration tensor requires token IDs or phone-timed input")
        }
    }
}

fn phone_f0_for_input(input: AcousticInput<'_>) -> Result<Vec<f32>> {
    match input {
        AcousticInput::PhoneTimed(targets) => Ok(targets
            .iter()
            .map(|target| target.f0_hz.unwrap_or(0.0))
            .collect()),
        AcousticInput::TokenIds(ids) => Ok(vec![0.0; ids.len()]),
        AcousticInput::Singing(_) | AcousticInput::SourceFilterTrack(_) => {
            bail!("neural acoustic F0 tensor requires token IDs or phone-timed input")
        }
    }
}

fn duration_ms_to_frames(duration_ms: u64, contract: &NeuralAcousticTrackContract) -> i64 {
    let frames = (duration_ms as f32 * contract.sample_rate_hz as f32
        / 1_000.0
        / contract.hop_samples as f32)
        .round()
        .max(1.0);
    frames as i64
}

#[cfg(feature = "tts-onnx")]
fn mel_frames_from_tensor(
    shape: &[i64],
    values: &[f32],
    mel_bins: usize,
    layout: NeuralMelOutputLayout,
) -> Result<Vec<MelFrame>> {
    ensure!(
        !values.is_empty(),
        "neural acoustic inference returned empty mel output"
    );
    ensure!(
        values.len() % mel_bins == 0,
        "neural acoustic mel output has {} values, not divisible by {mel_bins} mel bins",
        values.len()
    );
    let layout = match layout {
        NeuralMelOutputLayout::Auto => infer_mel_output_layout(shape, mel_bins),
        other => other,
    };
    let frames = values.len() / mel_bins;
    let mut mel = Vec::with_capacity(frames);
    match layout {
        NeuralMelOutputLayout::FramesBins | NeuralMelOutputLayout::Auto => {
            for frame_values in values.chunks_exact(mel_bins) {
                mel.push(MelFrame {
                    bins: frame_values.to_vec(),
                });
            }
        }
        NeuralMelOutputLayout::BinsFrames => {
            for frame_index in 0..frames {
                let bins = (0..mel_bins)
                    .map(|bin_index| values[bin_index * frames + frame_index])
                    .collect::<Vec<_>>();
                mel.push(MelFrame { bins });
            }
        }
    }
    ensure!(
        mel.iter()
            .flat_map(|frame| &frame.bins)
            .all(|bin| bin.is_finite()),
        "neural acoustic mel output contains non-finite values"
    );
    Ok(mel)
}

#[cfg(feature = "tts-onnx")]
fn infer_mel_output_layout(shape: &[i64], mel_bins: usize) -> NeuralMelOutputLayout {
    let mel_bins = mel_bins as i64;
    match shape {
        [left, right] => match (*left == mel_bins, *right == mel_bins) {
            (true, false) => NeuralMelOutputLayout::BinsFrames,
            (false, true) => NeuralMelOutputLayout::FramesBins,
            _ => NeuralMelOutputLayout::FramesBins,
        },
        [_, middle, right] => match (*middle == mel_bins, *right == mel_bins) {
            (true, false) => NeuralMelOutputLayout::BinsFrames,
            (false, true) => NeuralMelOutputLayout::FramesBins,
            _ => NeuralMelOutputLayout::FramesBins,
        },
        _ => NeuralMelOutputLayout::FramesBins,
    }
}

#[cfg(feature = "tts-onnx")]
fn optional_f32_output(
    outputs: &ort::session::SessionOutputs<'_>,
    name: Option<&str>,
    frame_count: usize,
    label: &str,
) -> Result<Option<Vec<f32>>> {
    let Some(name) = name else {
        return Ok(None);
    };
    let output = outputs.get(name).with_context(|| {
        format!("neural acoustic inference did not return {label} output `{name}`")
    })?;
    let output = output
        .downcast_ref::<DynTensorValueType>()
        .with_context(|| format!("neural acoustic {label} output `{name}` is not a tensor"))?;
    let (_, values) = output
        .try_extract_tensor::<f32>()
        .with_context(|| format!("neural acoustic {label} output `{name}` is not f32"))?;
    ensure!(
        values.len() == frame_count,
        "neural acoustic {label} output `{name}` has {} values for {frame_count} mel frames",
        values.len()
    );
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "neural acoustic {label} output `{name}` contains non-finite values"
    );
    Ok(Some(values.to_vec()))
}

fn fallback_f0_track(
    input: AcousticInput<'_>,
    frame_count: usize,
    contract: &NeuralAcousticTrackContract,
) -> Vec<f32> {
    match input {
        AcousticInput::PhoneTimed(targets) => expand_phone_f0(targets, frame_count, contract),
        AcousticInput::TokenIds(_)
        | AcousticInput::Singing(_)
        | AcousticInput::SourceFilterTrack(_) => {
            vec![0.0; frame_count]
        }
    }
}

fn expand_phone_f0(
    targets: &[PhoneTimedRenderTarget],
    frame_count: usize,
    contract: &NeuralAcousticTrackContract,
) -> Vec<f32> {
    let mut f0 = Vec::with_capacity(frame_count);
    for target in targets {
        let frames = duration_ms_to_frames(target.duration_ms, contract).max(1) as usize;
        f0.extend(std::iter::repeat_n(target.f0_hz.unwrap_or(0.0), frames));
    }
    if f0.is_empty() {
        f0.resize(frame_count, 0.0);
    }
    let last = *f0.last().unwrap_or(&0.0);
    f0.resize(frame_count, last);
    f0.truncate(frame_count);
    f0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linguistic::phonology::Phone;

    #[test]
    fn phone_timed_input_maps_through_model_specific_phone_ids() {
        let mut ids = NeuralPhoneIdMap::new();
        ids.insert("h".to_string(), 12);
        ids.insert("ɑ".to_string(), 42);
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
                amplitude: 0.8,
                vibrato: None,
            },
        ];

        let token_ids =
            token_ids_for_input(AcousticInput::PhoneTimed(&targets), &ids).expect("token IDs");

        assert_eq!(token_ids, vec![12, 42]);
    }

    #[test]
    fn phone_timed_f0_expands_to_frame_track() {
        let targets = vec![
            PhoneTimedRenderTarget {
                phone: Phone::new_ipa("h"),
                duration_ms: 16,
                f0_hz: None,
                amplitude: 0.7,
                vibrato: None,
            },
            PhoneTimedRenderTarget {
                phone: Phone::new_ipa("ɑ"),
                duration_ms: 32,
                f0_hz: Some(150.0),
                amplitude: 0.8,
                vibrato: None,
            },
        ];

        let f0 = expand_phone_f0(
            &targets,
            4,
            &NeuralAcousticTrackContract {
                sample_rate_hz: 16_000,
                hop_samples: 256,
                mel_bins: 80,
            },
        );

        assert_eq!(f0, vec![0.0, 150.0, 150.0, 150.0]);
    }

    #[cfg(feature = "tts-onnx")]
    #[test]
    fn mel_tensor_layout_transposes_bins_frames_outputs() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mel = mel_frames_from_tensor(&[1, 3, 2], &values, 3, NeuralMelOutputLayout::Auto)
            .expect("mel frames");

        assert_eq!(
            mel,
            vec![
                MelFrame {
                    bins: vec![1.0, 3.0, 5.0]
                },
                MelFrame {
                    bins: vec![2.0, 4.0, 6.0]
                }
            ]
        );
    }
}
