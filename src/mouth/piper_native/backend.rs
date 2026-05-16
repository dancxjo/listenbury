use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use ort::session::Session;
use ort::value::{DynTensorValueType, Tensor, TensorElementType};

use crate::audio::frame::AudioFrame;
use crate::mouth::backend::TtsBackend;

use super::{PiperIdSequence, PiperVoiceConfig};

const NATIVE_PIPER_FRAME_SAMPLES: usize = 1024;
// Piper ONNX vits output is a single waveform tensor for one speaker stream.
const NATIVE_PIPER_CHANNELS: u16 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PiperModelContract {
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativePiperPcm {
    pub sample_rate_hz: u32,
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
struct PiperTensorSpec {
    name: String,
    tensor_type: Option<TensorElementType>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PiperInferenceContract {
    id_input: String,
    id_lengths_input: String,
    scales_input: Option<String>,
    noise_scale_input: Option<String>,
    length_scale_input: Option<String>,
    noise_w_input: Option<String>,
    speaker_input: Option<String>,
    output_audio: String,
}

#[derive(Debug)]
pub struct NativePiperBackend {
    config: PiperVoiceConfig,
    model_path: PathBuf,
    session: Option<Session>,
}

impl NativePiperBackend {
    pub fn load(model_path: impl AsRef<Path>, config: PiperVoiceConfig) -> Result<Self> {
        validate_config(&config)?;

        let model_path = model_path.as_ref().to_path_buf();
        if !model_path.is_file() {
            bail!(
                "Piper ONNX model file not found at {}",
                model_path.display()
            );
        }

        let session = Session::builder()
            .context("failed to create Piper ONNX session builder")?
            .commit_from_file(&model_path)
            .with_context(|| {
                format!(
                    "failed to load Piper ONNX model from {}",
                    model_path.display()
                )
            })?;

        Ok(Self {
            config,
            model_path,
            session: Some(session),
        })
    }

    pub fn validate_model_contract(&self) -> Result<PiperModelContract> {
        let session = self
            .session
            .as_ref()
            .context("Piper ONNX session has not been loaded")?;

        let input_names = session
            .inputs()
            .iter()
            .map(|input| input.name().to_string())
            .collect::<Vec<_>>();
        let output_names = session
            .outputs()
            .iter()
            .map(|output| output.name().to_string())
            .collect::<Vec<_>>();

        ensure!(
            !input_names.is_empty(),
            "Piper ONNX model `{}` exposes no inputs",
            self.model_path.display()
        );
        ensure!(
            !output_names.is_empty(),
            "Piper ONNX model `{}` exposes no outputs",
            self.model_path.display()
        );
        ensure!(
            input_names.iter().all(|name| !name.trim().is_empty()),
            "Piper ONNX model `{}` has an unnamed input",
            self.model_path.display()
        );
        ensure!(
            output_names.iter().all(|name| !name.trim().is_empty()),
            "Piper ONNX model `{}` has an unnamed output",
            self.model_path.display()
        );

        Ok(PiperModelContract {
            input_names,
            output_names,
        })
    }

    pub fn config(&self) -> &PiperVoiceConfig {
        &self.config
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn synthesize_ids(&mut self, ids: &PiperIdSequence) -> Result<NativePiperPcm> {
        ensure!(
            !ids.ids.is_empty(),
            "Piper ID sequence cannot be empty for ONNX synthesis"
        );

        let sample_rate_hz = self.config.sample_rate_hz;
        let config = &self.config;
        let model_path = self.model_path.clone();
        let session = self
            .session
            .as_mut()
            .context("Piper ONNX session has not been loaded")?;

        let input_specs = session
            .inputs()
            .iter()
            .map(|input| PiperTensorSpec {
                name: input.name().to_string(),
                tensor_type: input.dtype().tensor_type(),
            })
            .collect::<Vec<_>>();
        let output_specs = session
            .outputs()
            .iter()
            .map(|output| PiperTensorSpec {
                name: output.name().to_string(),
                tensor_type: output.dtype().tensor_type(),
            })
            .collect::<Vec<_>>();
        let contract =
            resolve_inference_contract(&input_specs, &output_specs, config, &model_path)?;

        let ids_len = i64::try_from(ids.ids.len()).context("Piper ID sequence is too long")?;
        let mut inputs = Vec::with_capacity(6);

        let ids_tensor = Tensor::from_array((vec![1_i64, ids_len], ids.ids.clone()))
            .context("failed to build Piper ONNX `input` tensor from IDs")?
            .upcast();
        inputs.push((contract.id_input.clone(), ids_tensor));

        let ids_len_tensor = Tensor::from_array((vec![1_i64], vec![ids_len]))
            .context("failed to build Piper ONNX `input_lengths` tensor")?
            .upcast();
        inputs.push((contract.id_lengths_input.clone(), ids_len_tensor));

        let scales = inference_scales(config);
        if let Some(name) = &contract.scales_input {
            let scales_tensor = Tensor::from_array((vec![3_i64], scales.to_vec()))
                .with_context(|| format!("failed to build Piper ONNX `{name}` tensor"))?
                .upcast();
            inputs.push((name.clone(), scales_tensor));
        }
        if let Some(name) = &contract.noise_scale_input {
            let noise_scale_tensor = Tensor::from_array((vec![1_i64], vec![scales[0]]))
                .with_context(|| format!("failed to build Piper ONNX `{name}` tensor"))?
                .upcast();
            inputs.push((name.clone(), noise_scale_tensor));
        }
        if let Some(name) = &contract.length_scale_input {
            let length_scale_tensor = Tensor::from_array((vec![1_i64], vec![scales[1]]))
                .with_context(|| format!("failed to build Piper ONNX `{name}` tensor"))?
                .upcast();
            inputs.push((name.clone(), length_scale_tensor));
        }
        if let Some(name) = &contract.noise_w_input {
            let noise_w_tensor = Tensor::from_array((vec![1_i64], vec![scales[2]]))
                .with_context(|| format!("failed to build Piper ONNX `{name}` tensor"))?
                .upcast();
            inputs.push((name.clone(), noise_w_tensor));
        }
        if let Some(name) = &contract.speaker_input {
            let speaker_id_tensor = Tensor::from_array((vec![1_i64], vec![0_i64]))
                .with_context(|| format!("failed to build Piper ONNX `{name}` tensor"))?
                .upcast();
            inputs.push((name.clone(), speaker_id_tensor));
        }

        let outputs = session.run(inputs).with_context(|| {
            format!(
                "failed to run Piper ONNX inference for model {}",
                model_path.display()
            )
        })?;
        let output = outputs
            .get(contract.output_audio.as_str())
            .with_context(|| {
                format!(
                    "Piper ONNX inference did not return expected output `{}`",
                    contract.output_audio
                )
            })?;
        let output = output
            .downcast_ref::<DynTensorValueType>()
            .with_context(|| {
                format!(
                    "Piper ONNX output `{}` is not a tensor",
                    contract.output_audio
                )
            })?;
        let (_, samples) = output.try_extract_tensor::<f32>().with_context(|| {
            format!(
                "Piper ONNX output `{}` is not an f32 tensor",
                contract.output_audio
            )
        })?;
        ensure!(
            !samples.is_empty(),
            "Piper ONNX inference returned an empty waveform output"
        );

        Ok(NativePiperPcm {
            sample_rate_hz,
            samples: samples.to_vec(),
        })
    }

    pub fn synthesize_id_frames(&mut self, ids: &PiperIdSequence) -> Result<Vec<AudioFrame>> {
        let pcm = self.synthesize_ids(ids)?;
        Ok(native_pcm_to_audio_frames(pcm, NATIVE_PIPER_FRAME_SAMPLES))
    }

    #[cfg(test)]
    fn unloaded_for_tests(model_path: PathBuf, config: PiperVoiceConfig) -> Self {
        Self {
            config,
            model_path,
            session: None,
        }
    }
}

impl TtsBackend for NativePiperBackend {
    fn synthesize(&mut self, _text: &str) -> Result<Vec<AudioFrame>> {
        bail!(
            "Native Piper synthesis is not implemented yet for {}",
            self.model_path.display()
        );
    }
}

fn validate_config(config: &PiperVoiceConfig) -> Result<()> {
    ensure!(
        config.sample_rate_hz > 0,
        "missing required Piper voice config field `audio.sample_rate`"
    );
    ensure!(
        !config.phoneme_id_map.is_empty(),
        "missing required Piper voice config field `phoneme_id_map`"
    );
    if let Some(num_speakers) = config.num_speakers {
        ensure!(
            num_speakers > 0,
            "invalid Piper voice config field `num_speakers`: expected a value greater than zero"
        );
    }
    Ok(())
}

fn resolve_inference_contract(
    input_specs: &[PiperTensorSpec],
    output_specs: &[PiperTensorSpec],
    config: &PiperVoiceConfig,
    model_path: &Path,
) -> Result<PiperInferenceContract> {
    ensure!(
        !input_specs.is_empty(),
        "Piper ONNX model `{}` exposes no inputs",
        model_path.display()
    );
    ensure!(
        !output_specs.is_empty(),
        "Piper ONNX model `{}` exposes no outputs",
        model_path.display()
    );

    let available_input_names = input_specs
        .iter()
        .map(|spec| spec.name.clone())
        .collect::<Vec<_>>();
    let available_output_names = output_specs
        .iter()
        .map(|spec| spec.name.clone())
        .collect::<Vec<_>>();

    let id_input = resolve_required_tensor_input(
        input_specs,
        &["input", "input_ids", "phoneme_ids", "ids"],
        TensorElementType::Int64,
        "phoneme ID input tensor",
        model_path,
    )?;
    let id_lengths_input = resolve_required_tensor_input(
        input_specs,
        &["input_lengths", "lengths", "input_lengths_tensor"],
        TensorElementType::Int64,
        "phoneme length input tensor",
        model_path,
    )?;

    let scales_input =
        resolve_optional_tensor_input(input_specs, &["scales"], TensorElementType::Float32)?;
    let noise_scale_input =
        resolve_optional_tensor_input(input_specs, &["noise_scale"], TensorElementType::Float32)?;
    let length_scale_input =
        resolve_optional_tensor_input(input_specs, &["length_scale"], TensorElementType::Float32)?;
    let noise_w_input =
        resolve_optional_tensor_input(input_specs, &["noise_w"], TensorElementType::Float32)?;
    let speaker_input = resolve_optional_tensor_input(
        input_specs,
        &["sid", "speaker_id"],
        TensorElementType::Int64,
    )?;

    let speaker_count = match config.num_speakers {
        Some(num_speakers) => num_speakers,
        None => u32::try_from(config.speaker_id_map.len()).with_context(|| {
            format!(
                "invalid Piper voice config for `{}`: `speaker_id_map` size exceeds u32",
                model_path.display()
            )
        })?,
    };
    if speaker_count > 1 {
        bail!(
            "Piper ONNX multi-speaker inference is not supported yet for `{}`: config reports {} speakers; available inputs: {}",
            model_path.display(),
            speaker_count,
            available_input_names.join(", ")
        );
    }

    let mut supported_input_names = vec![id_input.clone(), id_lengths_input.clone()];
    supported_input_names.extend(scales_input.iter().cloned());
    supported_input_names.extend(noise_scale_input.iter().cloned());
    supported_input_names.extend(length_scale_input.iter().cloned());
    supported_input_names.extend(noise_w_input.iter().cloned());
    supported_input_names.extend(speaker_input.iter().cloned());
    for input in input_specs {
        if !supported_input_names.iter().any(|name| name == &input.name) {
            bail!(
                "Unsupported Piper ONNX input `{}` for model `{}`; supported inputs are explicit phoneme IDs, lengths, scales/noise controls, and optional speaker ID. Model inputs: {}",
                input.name,
                model_path.display(),
                available_input_names.join(", ")
            );
        }
    }

    let output_audio = resolve_required_tensor_output(
        output_specs,
        &["output", "audio", "waveform"],
        TensorElementType::Float32,
        "audio output tensor",
        model_path,
    )?;

    if output_specs.iter().any(|spec| {
        spec.name != output_audio && spec.tensor_type == Some(TensorElementType::Float32)
    }) {
        bail!(
            "Unsupported Piper ONNX model `{}` contract: multiple f32 outputs detected ({})",
            model_path.display(),
            available_output_names.join(", ")
        );
    }

    Ok(PiperInferenceContract {
        id_input,
        id_lengths_input,
        scales_input,
        noise_scale_input,
        length_scale_input,
        noise_w_input,
        speaker_input,
        output_audio,
    })
}

fn resolve_required_tensor_input(
    inputs: &[PiperTensorSpec],
    aliases: &[&str],
    expected_type: TensorElementType,
    label: &str,
    model_path: &Path,
) -> Result<String> {
    let available = inputs
        .iter()
        .map(|spec| spec.name.clone())
        .collect::<Vec<_>>()
        .join(", ");
    let input = resolve_tensor_by_alias(inputs, aliases)
        .with_context(|| {
            format!(
                "unsupported Piper ONNX model contract for `{}`: missing {} (expected one of: {}; model inputs: {})",
                model_path.display(),
                label,
                aliases.join(", "),
                available
            )
        })?;
    ensure!(
        input.tensor_type == Some(expected_type),
        "unsupported Piper ONNX model contract for `{}`: input `{}` expected type {:?}, got {:?}",
        model_path.display(),
        input.name,
        expected_type,
        input.tensor_type
    );
    Ok(input.name.clone())
}

fn resolve_optional_tensor_input(
    inputs: &[PiperTensorSpec],
    aliases: &[&str],
    expected_type: TensorElementType,
) -> Result<Option<String>> {
    let Some(input) = resolve_tensor_by_alias(inputs, aliases) else {
        return Ok(None);
    };
    ensure!(
        input.tensor_type == Some(expected_type),
        "unsupported Piper ONNX model contract: input `{}` expected type {:?}, got {:?}",
        input.name,
        expected_type,
        input.tensor_type
    );
    Ok(Some(input.name.clone()))
}

fn resolve_required_tensor_output(
    outputs: &[PiperTensorSpec],
    aliases: &[&str],
    expected_type: TensorElementType,
    label: &str,
    model_path: &Path,
) -> Result<String> {
    let available = outputs
        .iter()
        .map(|spec| spec.name.clone())
        .collect::<Vec<_>>()
        .join(", ");
    let output = resolve_tensor_by_alias(outputs, aliases).or_else(|| {
        outputs
            .iter()
            .find(|spec| spec.tensor_type == Some(expected_type))
    });
    let Some(output) = output else {
        bail!(
            "unsupported Piper ONNX model contract for `{}`: missing {} (expected one of: {}; model outputs: {})",
            model_path.display(),
            label,
            aliases.join(", "),
            available
        );
    };
    ensure!(
        output.tensor_type == Some(expected_type),
        "unsupported Piper ONNX model contract for `{}`: output `{}` expected type {:?}, got {:?}",
        model_path.display(),
        output.name,
        expected_type,
        output.tensor_type
    );
    Ok(output.name.clone())
}

fn resolve_tensor_by_alias<'a>(
    specs: &'a [PiperTensorSpec],
    aliases: &[&str],
) -> Option<&'a PiperTensorSpec> {
    aliases
        .iter()
        .find_map(|alias| specs.iter().find(|spec| spec.name == *alias))
}

fn inference_scales(config: &PiperVoiceConfig) -> [f32; 3] {
    [
        config.noise_scale.unwrap_or(0.667),
        config.length_scale.unwrap_or(1.0),
        config.noise_w.unwrap_or(0.8),
    ]
}

fn native_pcm_to_audio_frames(pcm: NativePiperPcm, frame_samples: usize) -> Vec<AudioFrame> {
    assert!(frame_samples > 0, "frame_samples must be greater than zero");
    if pcm.samples.is_empty() {
        return Vec::new();
    }

    pcm.samples
        .chunks(frame_samples)
        .map(|chunk| AudioFrame {
            captured_at: crate::time::ExactTimestamp::now(),
            sample_rate_hz: pcm.sample_rate_hz,
            channels: NATIVE_PIPER_CHANNELS,
            samples: chunk
                .iter()
                .map(|sample| if sample.is_finite() { *sample } else { 0.0 })
                .collect(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;

    fn voice_config() -> PiperVoiceConfig {
        PiperVoiceConfig {
            sample_rate_hz: 22_050,
            phoneme_id_map: HashMap::from([("a".to_string(), vec![1])]),
            num_speakers: None,
            speaker_id_map: HashMap::new(),
            length_scale: None,
            noise_scale: None,
            noise_w: None,
            model_metadata: HashMap::new(),
        }
    }

    fn unique_path(label: &str) -> PathBuf {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should advance")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "listenbury-native-piper-{label}-{}-{ts}.onnx",
            std::process::id()
        ))
    }

    #[test]
    fn load_returns_clear_error_for_missing_model_file() {
        let model_path = unique_path("missing-model");
        let error = NativePiperBackend::load(&model_path, voice_config())
            .expect_err("missing model should fail");

        assert_eq!(
            error.to_string(),
            format!(
                "Piper ONNX model file not found at {}",
                model_path.display()
            )
        );
    }

    #[test]
    fn load_rejects_missing_phoneme_map_before_session_creation() {
        let model_path = unique_path("config-error");
        fs::write(&model_path, b"placeholder").expect("placeholder model file");

        let mut config = voice_config();
        config.phoneme_id_map.clear();

        let error = NativePiperBackend::load(&model_path, config)
            .expect_err("empty phoneme map should fail");
        assert_eq!(
            error.to_string(),
            "missing required Piper voice config field `phoneme_id_map`"
        );

        let _ = fs::remove_file(model_path);
    }

    #[test]
    fn synthesize_returns_clear_unimplemented_error() {
        let model_path = unique_path("unimplemented");
        let mut backend =
            NativePiperBackend::unloaded_for_tests(model_path.clone(), voice_config());

        let error = backend
            .synthesize("hello")
            .expect_err("native synthesis is not implemented");
        assert_eq!(
            error.to_string(),
            format!(
                "Native Piper synthesis is not implemented yet for {}",
                model_path.display()
            )
        );
    }

    fn input(name: &str, tensor_type: TensorElementType) -> PiperTensorSpec {
        PiperTensorSpec {
            name: name.to_string(),
            tensor_type: Some(tensor_type),
        }
    }

    fn output(name: &str, tensor_type: TensorElementType) -> PiperTensorSpec {
        PiperTensorSpec {
            name: name.to_string(),
            tensor_type: Some(tensor_type),
        }
    }

    #[test]
    fn synthesize_ids_requires_loaded_session() {
        let model_path = unique_path("unloaded-session");
        let mut backend = NativePiperBackend::unloaded_for_tests(model_path, voice_config());

        let error = backend
            .synthesize_ids(&PiperIdSequence { ids: vec![1, 2, 3] })
            .expect_err("unloaded session should fail");
        assert_eq!(error.to_string(), "Piper ONNX session has not been loaded");
    }

    #[test]
    fn synthesize_ids_rejects_empty_id_sequence() {
        let model_path = unique_path("empty-ids");
        let mut backend = NativePiperBackend::unloaded_for_tests(model_path, voice_config());

        let error = backend
            .synthesize_ids(&PiperIdSequence { ids: Vec::new() })
            .expect_err("empty IDs should fail");
        assert_eq!(
            error.to_string(),
            "Piper ID sequence cannot be empty for ONNX synthesis"
        );
    }

    #[test]
    fn synthesize_id_frames_requires_loaded_session() {
        let model_path = unique_path("unloaded-session-frames");
        let mut backend = NativePiperBackend::unloaded_for_tests(model_path, voice_config());

        let error = backend
            .synthesize_id_frames(&PiperIdSequence { ids: vec![1, 2, 3] })
            .expect_err("unloaded session should fail");
        assert_eq!(error.to_string(), "Piper ONNX session has not been loaded");
    }

    #[test]
    fn resolve_contract_rejects_unknown_required_inputs() {
        let error = resolve_inference_contract(
            &[input("tokens", TensorElementType::Int64)],
            &[output("output", TensorElementType::Float32)],
            &voice_config(),
            Path::new("test.onnx"),
        )
        .expect_err("missing input_lengths should fail");

        assert!(
            error
                .to_string()
                .contains("missing phoneme ID input tensor"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn resolve_contract_rejects_unsupported_extra_inputs() {
        let error = resolve_inference_contract(
            &[
                input("input", TensorElementType::Int64),
                input("input_lengths", TensorElementType::Int64),
                input("temperature", TensorElementType::Float32),
            ],
            &[output("output", TensorElementType::Float32)],
            &voice_config(),
            Path::new("test.onnx"),
        )
        .expect_err("unsupported extra input should fail");

        assert!(
            error
                .to_string()
                .contains("Unsupported Piper ONNX input `temperature`"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn resolve_contract_rejects_multi_speaker_configs() {
        let mut config = voice_config();
        config.num_speakers = Some(2);

        let error = resolve_inference_contract(
            &[
                input("input", TensorElementType::Int64),
                input("input_lengths", TensorElementType::Int64),
                input("sid", TensorElementType::Int64),
            ],
            &[output("output", TensorElementType::Float32)],
            &config,
            Path::new("test.onnx"),
        )
        .expect_err("multi-speaker should fail clearly");

        assert!(
            error
                .to_string()
                .contains("multi-speaker inference is not supported yet"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn native_pcm_to_audio_frames_returns_empty_for_empty_pcm() {
        let frames = native_pcm_to_audio_frames(
            NativePiperPcm {
                sample_rate_hz: 22_050,
                samples: Vec::new(),
            },
            1024,
        );

        assert!(frames.is_empty(), "expected empty frame list for empty PCM");
    }

    #[test]
    fn native_pcm_to_audio_frames_coerces_non_finite_samples() {
        let frames = native_pcm_to_audio_frames(
            NativePiperPcm {
                sample_rate_hz: 22_050,
                samples: vec![0.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.2],
            },
            16,
        );

        assert_eq!(frames.len(), 1);
        assert!(frames[0].samples.iter().all(|sample| sample.is_finite()));
        assert_eq!(frames[0].samples, vec![0.1, 0.0, 0.0, 0.0, -0.2]);
    }

    #[test]
    fn native_pcm_to_audio_frames_preserves_sample_rate_and_mono_channel() {
        let frames = native_pcm_to_audio_frames(
            NativePiperPcm {
                sample_rate_hz: 16_000,
                samples: vec![0.1, 0.2, 0.3],
            },
            2,
        );

        assert_eq!(frames.len(), 2);
        assert!(frames.iter().all(|frame| frame.sample_rate_hz == 16_000));
        assert!(frames.iter().all(|frame| frame.channels == 1));
    }

    #[test]
    fn native_pcm_to_audio_frames_chunks_using_requested_frame_size() {
        let frames = native_pcm_to_audio_frames(
            NativePiperPcm {
                sample_rate_hz: 22_050,
                samples: vec![0.0, 0.1, 0.2, 0.3, 0.4],
            },
            2,
        );

        let chunk_sizes = frames
            .iter()
            .map(|frame| frame.samples.len())
            .collect::<Vec<_>>();
        assert_eq!(chunk_sizes, vec![2, 2, 1]);
    }
}
