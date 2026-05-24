use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::{DynTensorValueType, Tensor, TensorElementType};

use crate::audio::frame::AudioFrame;
use crate::mouth::backend::TtsBackend;

use super::{
    PiperIdSequence, PiperVoiceConfig, SimpleEnglishG2p,
    prosody_controls::{
        ControlStatusEntry, PiperProsodyControls, PiperSynthesisDiagnostics, ProsodyControlStatus,
    },
};

const RIPER_FRAME_SAMPLES: usize = 1024;
// Piper ONNX vits output is a single waveform tensor for one speaker stream.
const RIPER_CHANNELS: u16 = 1;
const RIPER_DEFAULT_LENGTH_SCALE_MULTIPLIER: f32 = 1.12;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PiperModelContract {
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RiperPcm {
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
pub struct RiperBackend {
    config: PiperVoiceConfig,
    model_path: PathBuf,
    session: Option<Session>,
}

impl RiperBackend {
    pub fn load(model_path: impl AsRef<Path>, config: PiperVoiceConfig) -> Result<Self> {
        validate_config(&config)?;

        let model_path = model_path.as_ref().to_path_buf();
        if !model_path.is_file() {
            bail!(
                "Piper ONNX model file not found at {}",
                model_path.display()
            );
        }

        initialize_ort_runtime()?;

        let session = Session::builder()
            .context("failed to create Piper ONNX session builder")?
            .with_intra_threads(1)
            .map_err(|error| {
                anyhow::anyhow!("failed to configure Piper ONNX intra-op threads: {error}")
            })?
            .with_inter_threads(1)
            .map_err(|error| {
                anyhow::anyhow!("failed to configure Piper ONNX inter-op threads: {error}")
            })?
            .with_intra_op_spinning(false)
            .map_err(|error| {
                anyhow::anyhow!("failed to configure Piper ONNX intra-op spinning: {error}")
            })?
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .map_err(|error| {
                anyhow::anyhow!("failed to configure Piper ONNX optimization level: {error}")
            })?
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

    pub fn synthesize_ids(&mut self, ids: &PiperIdSequence) -> Result<RiperPcm> {
        let scales = inference_scales(&self.config);
        self.synthesize_ids_with_scales(ids, scales)
    }

    /// Synthesize phoneme IDs with optional prosody controls and return both
    /// the PCM output and diagnostics describing how each control was handled.
    ///
    /// When `controls` is `None` this is equivalent to [`Self::synthesize_ids`],
    /// producing empty diagnostics and default acoustic parameters.
    ///
    /// # Control handling
    /// - `length_scale` / `noise_scale` / `noise_w` overrides → [`ProsodyControlStatus::Realized`]
    ///   (values passed directly as ONNX tensor inputs where the model exposes them).
    /// - `pause_overrides` → [`ProsodyControlStatus::Approximated`]
    ///   (silence samples appended to the output PCM).
    /// - `phoneme_duration_overrides` / `boundary_overrides` →
    ///   [`ProsodyControlStatus::AdvisoryOnly`] (recorded in diagnostics only;
    ///   no per-phoneme or boundary knob is available in the ONNX path).
    pub fn synthesize_ids_with_controls(
        &mut self,
        ids: &PiperIdSequence,
        controls: Option<&PiperProsodyControls>,
    ) -> Result<(RiperPcm, PiperSynthesisDiagnostics)> {
        let config_scales = inference_scales(&self.config);

        let (effective_scales, mut control_statuses) = match controls {
            Some(controls) => compute_controlled_scales(config_scales, controls),
            None => (config_scales, Vec::new()),
        };

        let mut pcm = self.synthesize_ids_with_scales(ids, effective_scales)?;

        let mut inserted_pause_ms = 0u64;
        if let Some(controls) = controls {
            for pause in &controls.pause_overrides {
                let silence_samples = compute_silence_samples(pause.millis, pcm.sample_rate_hz)?;
                pcm.samples
                    .extend(std::iter::repeat_n(0.0_f32, silence_samples));
                inserted_pause_ms = inserted_pause_ms.saturating_add(pause.millis);
                control_statuses.push(ControlStatusEntry {
                    name: format!("pause_override[{}]", pause.label),
                    status: ProsodyControlStatus::Approximated,
                    detail: format!(
                        "silence of {} ms appended to PCM ({} samples at {} Hz)",
                        pause.millis, silence_samples, pcm.sample_rate_hz
                    ),
                });
            }

            for (i, ovr) in controls.phoneme_duration_overrides.iter().enumerate() {
                control_statuses.push(ControlStatusEntry {
                    name: format!("phoneme_duration_override[{i}]"),
                    status: ProsodyControlStatus::AdvisoryOnly,
                    detail: format!(
                        "per-phoneme duration hint for phoneme index {} ({} ms) is advisory only; \
                         no per-phoneme timing control is available in the current ONNX path",
                        ovr.phoneme_index, ovr.millis
                    ),
                });
            }

            for (i, ovr) in controls.boundary_overrides.iter().enumerate() {
                control_statuses.push(ControlStatusEntry {
                    name: format!("boundary_override[{i}]"),
                    status: ProsodyControlStatus::AdvisoryOnly,
                    detail: format!(
                        "boundary hint after index {} ({}) is advisory only; \
                         no boundary control is available in the current ONNX path",
                        ovr.after_index,
                        if ovr.strong { "strong" } else { "weak" }
                    ),
                });
            }
        }

        let pcm_duration_ms = pcm_duration_ms(&pcm);
        let diagnostics = PiperSynthesisDiagnostics {
            input_phoneme_ids: ids.ids.clone(),
            applied_length_scale: effective_scales[1],
            applied_noise_scale: effective_scales[0],
            applied_noise_w: effective_scales[2],
            inserted_pause_ms,
            pcm_duration_ms,
            control_statuses,
        };

        Ok((pcm, diagnostics))
    }

    pub fn synthesize_id_frames(&mut self, ids: &PiperIdSequence) -> Result<Vec<AudioFrame>> {
        let pcm = self.synthesize_ids(ids)?;
        Ok(riper_pcm_to_audio_frames(pcm, RIPER_FRAME_SAMPLES))
    }

    /// Synthesize phoneme IDs with optional prosody controls and return
    /// [`AudioFrame`]s alongside diagnostics.
    ///
    /// Equivalent to calling [`Self::synthesize_ids_with_controls`] and then
    /// converting the resulting PCM into audio frames.
    pub fn synthesize_id_frames_with_controls(
        &mut self,
        ids: &PiperIdSequence,
        controls: Option<&PiperProsodyControls>,
    ) -> Result<(Vec<AudioFrame>, PiperSynthesisDiagnostics)> {
        let (pcm, diagnostics) = self.synthesize_ids_with_controls(ids, controls)?;
        Ok((
            riper_pcm_to_audio_frames(pcm, RIPER_FRAME_SAMPLES),
            diagnostics,
        ))
    }

    // Private helper: run ONNX inference with explicitly provided scale values.
    // scales = [noise_scale, length_scale, noise_w]
    fn synthesize_ids_with_scales(
        &mut self,
        ids: &PiperIdSequence,
        scales: [f32; 3],
    ) -> Result<RiperPcm> {
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

        Ok(RiperPcm {
            sample_rate_hz,
            samples: samples.to_vec(),
        })
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

pub(crate) fn initialize_ort_runtime() -> Result<()> {
    if let Some(path) = std::env::var_os("ORT_DYLIB_PATH").filter(|value| !value.is_empty()) {
        let path = PathBuf::from(path);
        ensure!(
            path.is_file(),
            "ORT_DYLIB_PATH points to {}, but that file does not exist",
            path.display()
        );
        initialize_ort_runtime_from(&path)?;
        return Ok(());
    }

    if let Some(path) = find_onnxruntime_dylib() {
        initialize_ort_runtime_from(&path)?;
    } else {
        bail!(
            "Riper requires an ONNX Runtime shared library, but none was found. \
             Install ONNX Runtime or set ORT_DYLIB_PATH to libonnxruntime.so \
             (for example, a Python onnxruntime package copy under site-packages/onnxruntime/capi)."
        );
    }

    Ok(())
}

fn initialize_ort_runtime_from(path: &Path) -> Result<()> {
    ort::init_from(path)
        .map_err(|error| {
            anyhow::anyhow!(
                "failed to load ONNX Runtime dynamic library from {}: {error}",
                path.display()
            )
        })?
        .commit();
    Ok(())
}

fn find_onnxruntime_dylib() -> Option<PathBuf> {
    find_home_onnxruntime_dylib().or_else(find_linker_onnxruntime_dylib)
}

fn find_home_onnxruntime_dylib() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    find_onnxruntime_dylib_in_dirs(home_onnxruntime_search_dirs(&home))
}

fn home_onnxruntime_search_dirs(home: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let local_lib = home.join(".local/lib");
    if let Ok(entries) = std::fs::read_dir(local_lib) {
        dirs.extend(entries.flatten().filter_map(|entry| {
            let file_name = entry.file_name();
            file_name
                .to_string_lossy()
                .starts_with("python")
                .then(|| entry.path().join("site-packages/onnxruntime/capi"))
        }));
    }

    for extensions_dir in [
        home.join(".vscode/extensions"),
        home.join(".vscode-server/extensions"),
    ] {
        if let Ok(entries) = std::fs::read_dir(extensions_dir) {
            dirs.extend(entries.flatten().filter_map(|entry| {
                let file_name = entry.file_name();
                file_name
                    .to_string_lossy()
                    .contains("windows-ai-studio")
                    .then(|| entry.path().join("bin"))
            }));
        }
    }
    dirs
}

fn find_linker_onnxruntime_dylib() -> Option<PathBuf> {
    let mut search_dirs = Vec::new();
    if let Some(paths) = std::env::var_os("LD_LIBRARY_PATH") {
        search_dirs.extend(std::env::split_paths(&paths));
    }
    search_dirs.extend([
        PathBuf::from("/usr/local/lib"),
        PathBuf::from("/usr/local/lib64"),
        PathBuf::from("/usr/lib"),
        PathBuf::from("/usr/lib64"),
        PathBuf::from("/usr/lib/x86_64-linux-gnu"),
        PathBuf::from("/lib/x86_64-linux-gnu"),
    ]);
    find_onnxruntime_dylib_in_dirs(search_dirs)
}

fn find_onnxruntime_dylib_in_dirs(dirs: impl IntoIterator<Item = PathBuf>) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        candidates.extend(entries.flatten().filter_map(|entry| {
            let name = entry.file_name();
            is_onnxruntime_dylib_name(&name.to_string_lossy()).then(|| entry.path())
        }));
    }
    candidates.sort();
    candidates.pop()
}

fn is_onnxruntime_dylib_name(name: &str) -> bool {
    name == "libonnxruntime.so" || name.starts_with("libonnxruntime.so.")
}

impl TtsBackend for RiperBackend {
    fn synthesize(&mut self, text: &str) -> Result<Vec<AudioFrame>> {
        let phonemes = SimpleEnglishG2p::default()
            .phonemize_unit(text)
            .with_context(|| format!("failed to realize Riper phonemes for text `{text}`"))?
            .phonemes;
        let ids = phonemes
            .to_piper_text_ids_compatible(&self.config)
            .with_context(|| {
                format!(
                    "failed to map phonemes to IDs for Riper model {}",
                    self.model_path.display()
                )
            })?;
        self.synthesize_id_frames(&ids).with_context(|| {
            format!(
                "Riper ONNX synthesis failed for model {}",
                self.model_path.display()
            )
        })
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
        config.length_scale.unwrap_or(1.0) * RIPER_DEFAULT_LENGTH_SCALE_MULTIPLIER,
        config.noise_w.unwrap_or(0.8),
    ]
}

/// Compute effective scale values by applying any overrides from `controls` on
/// top of the configuration-derived defaults.  Returns the effective scales and
/// a list of [`ControlStatusEntry`] values describing each applied override.
///
/// `scales` layout: `[noise_scale, length_scale, noise_w]` (matching
/// [`inference_scales`]).
fn compute_controlled_scales(
    config_scales: [f32; 3],
    controls: &PiperProsodyControls,
) -> ([f32; 3], Vec<ControlStatusEntry>) {
    let mut scales = config_scales;
    let mut statuses = Vec::new();

    if let Some(noise_scale) = controls.noise_scale {
        statuses.push(ControlStatusEntry {
            name: "noise_scale".to_string(),
            status: ProsodyControlStatus::Realized,
            detail: format!(
                "noise_scale overridden from {:.3} to {:.3}",
                config_scales[0], noise_scale
            ),
        });
        scales[0] = noise_scale;
    }
    if let Some(length_scale) = controls.length_scale {
        statuses.push(ControlStatusEntry {
            name: "length_scale".to_string(),
            status: ProsodyControlStatus::Realized,
            detail: format!(
                "length_scale overridden from {:.3} to {:.3}",
                config_scales[1], length_scale
            ),
        });
        scales[1] = length_scale;
    }
    if let Some(noise_w) = controls.noise_w {
        statuses.push(ControlStatusEntry {
            name: "noise_w".to_string(),
            status: ProsodyControlStatus::Realized,
            detail: format!(
                "noise_w overridden from {:.3} to {:.3}",
                config_scales[2], noise_w
            ),
        });
        scales[2] = noise_w;
    }

    (scales, statuses)
}

fn pcm_duration_ms(pcm: &RiperPcm) -> u64 {
    if pcm.sample_rate_hz == 0 {
        return 0;
    }
    // usize fits within u64 on all supported platforms; saturate to avoid overflow in
    // pathological cases rather than silently wrapping.
    let samples = pcm.samples.len().min(u64::MAX as usize) as u64;
    samples * 1000 / u64::from(pcm.sample_rate_hz)
}

/// Compute the number of silence samples needed for a pause of `millis` ms at
/// `sample_rate_hz` Hz.  Returns an error if the resulting count would exceed
/// `usize::MAX` (which would indicate an unreasonably long pause).
fn compute_silence_samples(millis: u64, sample_rate_hz: u32) -> Result<usize> {
    let sample_count = millis
        .checked_mul(u64::from(sample_rate_hz))
        .map(|n| n / 1000)
        .with_context(|| {
            format!(
                "pause duration {} ms overflows when computing silence samples at {} Hz",
                millis, sample_rate_hz
            )
        })?;
    usize::try_from(sample_count).with_context(|| {
        format!(
            "pause of {} ms at {} Hz requires {} samples which exceeds usize::MAX",
            millis, sample_rate_hz, sample_count
        )
    })
}

fn riper_pcm_to_audio_frames(pcm: RiperPcm, frame_samples: usize) -> Vec<AudioFrame> {
    assert!(frame_samples > 0, "frame_samples must be greater than zero");
    if pcm.samples.is_empty() {
        return Vec::new();
    }

    pcm.samples
        .chunks(frame_samples)
        .map(|chunk| AudioFrame {
            captured_at: crate::time::ExactTimestamp::now(),
            sample_rate_hz: pcm.sample_rate_hz,
            channels: RIPER_CHANNELS,
            samples: chunk
                .iter()
                .map(|sample| if sample.is_finite() { *sample } else { 0.0 })
                .collect(),
            voice_signatures: Vec::new(),
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
            "listenbury-riper-{label}-{}-{ts}.onnx",
            std::process::id()
        ))
    }

    #[test]
    fn onnxruntime_dylib_name_matches_unversioned_and_versioned_library() {
        assert!(is_onnxruntime_dylib_name("libonnxruntime.so"));
        assert!(is_onnxruntime_dylib_name("libonnxruntime.so.1.23.0"));
        assert!(!is_onnxruntime_dylib_name(
            "libonnxruntime_providers_cuda.so"
        ));
        assert!(!is_onnxruntime_dylib_name("onnxruntime.dll"));
    }

    #[test]
    fn find_onnxruntime_dylib_in_dirs_prefers_matching_library_files() {
        let dir =
            std::env::temp_dir().join(format!("listenbury-riper-ort-find-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp dir");
        fs::write(dir.join("libonnxruntime_providers_cuda.so"), b"provider")
            .expect("write provider");
        fs::write(dir.join("libonnxruntime.so.1"), b"runtime").expect("write runtime");

        let found = find_onnxruntime_dylib_in_dirs([dir.clone()]);
        assert_eq!(found, Some(dir.join("libonnxruntime.so.1")));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn home_onnxruntime_search_dirs_include_python_and_vscode_locations() {
        let home =
            std::env::temp_dir().join(format!("listenbury-riper-ort-home-{}", std::process::id()));
        let _ = fs::remove_dir_all(&home);
        fs::create_dir_all(home.join(".local/lib/python3.12")).expect("create python lib dir");
        fs::create_dir_all(
            home.join(".vscode/extensions")
                .join("ms-windows-ai-studio.windows-ai-studio-1.2.1-linux-x64"),
        )
        .expect("create vscode extension dir");

        let dirs = home_onnxruntime_search_dirs(&home);

        assert!(
            dirs.contains(
                &home
                    .join(".local/lib/python3.12")
                    .join("site-packages/onnxruntime/capi")
            )
        );
        assert!(
            dirs.contains(
                &home
                    .join(".vscode/extensions")
                    .join("ms-windows-ai-studio.windows-ai-studio-1.2.1-linux-x64")
                    .join("bin")
            )
        );

        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn load_returns_clear_error_for_missing_model_file() {
        let model_path = unique_path("missing-model");
        let error =
            RiperBackend::load(&model_path, voice_config()).expect_err("missing model should fail");

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

        let error =
            RiperBackend::load(&model_path, config).expect_err("empty phoneme map should fail");
        assert_eq!(
            error.to_string(),
            "missing required Piper voice config field `phoneme_id_map`"
        );

        let _ = fs::remove_file(model_path);
    }

    #[test]
    fn synthesize_surfaces_clear_g2p_error_for_unsupported_text() {
        let model_path = unique_path("unimplemented");
        let mut backend = RiperBackend::unloaded_for_tests(model_path.clone(), voice_config());

        let error = backend
            .synthesize("#")
            .expect_err("unsupported text should fail before ONNX inference");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("failed to realize Riper phonemes for text `#`"),
            "expected phonemize context, got: {rendered}"
        );
        assert!(
            rendered.contains("unsupported character"),
            "expected unsupported-text detail, got: {rendered}"
        );
        assert!(
            !rendered.contains(model_path.to_string_lossy().as_ref()),
            "unsupported text path should fail before model access"
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
        let mut backend = RiperBackend::unloaded_for_tests(model_path, voice_config());

        let error = backend
            .synthesize_ids(&PiperIdSequence { ids: vec![1, 2, 3] })
            .expect_err("unloaded session should fail");
        assert_eq!(error.to_string(), "Piper ONNX session has not been loaded");
    }

    #[test]
    fn synthesize_ids_rejects_empty_id_sequence() {
        let model_path = unique_path("empty-ids");
        let mut backend = RiperBackend::unloaded_for_tests(model_path, voice_config());

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
        let mut backend = RiperBackend::unloaded_for_tests(model_path, voice_config());

        let error = backend
            .synthesize_id_frames(&PiperIdSequence { ids: vec![1, 2, 3] })
            .expect_err("unloaded session should fail");
        assert_eq!(error.to_string(), "Piper ONNX session has not been loaded");
    }

    #[test]
    fn inference_scales_slow_default_rate_slightly() {
        let scales = inference_scales(&voice_config());
        assert!((scales[1] - 1.12).abs() < f32::EPSILON);
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
    fn riper_pcm_to_audio_frames_returns_empty_for_empty_pcm() {
        let frames = riper_pcm_to_audio_frames(
            RiperPcm {
                sample_rate_hz: 22_050,
                samples: Vec::new(),
            },
            1024,
        );

        assert!(frames.is_empty(), "expected empty frame list for empty PCM");
    }

    #[test]
    fn riper_pcm_to_audio_frames_coerces_non_finite_samples() {
        let frames = riper_pcm_to_audio_frames(
            RiperPcm {
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
    fn riper_pcm_to_audio_frames_preserves_sample_rate_and_mono_channel() {
        let frames = riper_pcm_to_audio_frames(
            RiperPcm {
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
    fn riper_pcm_to_audio_frames_chunks_using_requested_frame_size() {
        let frames = riper_pcm_to_audio_frames(
            RiperPcm {
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

    // --- compute_controlled_scales tests ---

    #[test]
    fn compute_controlled_scales_uses_config_defaults_when_no_overrides() {
        let config_scales = [0.667_f32, 1.0, 0.8];
        let controls = PiperProsodyControls::default();
        let (scales, statuses) = compute_controlled_scales(config_scales, &controls);
        assert_eq!(
            scales, config_scales,
            "no overrides should leave scales unchanged"
        );
        assert!(
            statuses.is_empty(),
            "no overrides should produce no status entries"
        );
    }

    #[test]
    fn compute_controlled_scales_overrides_length_scale() {
        let config_scales = [0.667_f32, 1.0, 0.8];
        let controls = PiperProsodyControls {
            length_scale: Some(1.5),
            ..Default::default()
        };
        let (scales, statuses) = compute_controlled_scales(config_scales, &controls);
        assert!(
            (scales[1] - 1.5).abs() < f32::EPSILON,
            "length_scale should be overridden"
        );
        assert_eq!(scales[0], 0.667);
        assert_eq!(scales[2], 0.8);
        assert_eq!(statuses.len(), 1);
        assert_eq!(statuses[0].name, "length_scale");
        assert_eq!(statuses[0].status, ProsodyControlStatus::Realized);
        assert!(
            statuses[0].detail.contains("1.500"),
            "detail should mention new value"
        );
    }

    #[test]
    fn compute_controlled_scales_overrides_noise_scale() {
        let config_scales = [0.667_f32, 1.0, 0.8];
        let controls = PiperProsodyControls {
            noise_scale: Some(0.3),
            ..Default::default()
        };
        let (scales, statuses) = compute_controlled_scales(config_scales, &controls);
        assert!((scales[0] - 0.3).abs() < f32::EPSILON);
        assert_eq!(statuses[0].name, "noise_scale");
        assert_eq!(statuses[0].status, ProsodyControlStatus::Realized);
    }

    #[test]
    fn compute_controlled_scales_overrides_noise_w() {
        let config_scales = [0.667_f32, 1.0, 0.8];
        let controls = PiperProsodyControls {
            noise_w: Some(0.5),
            ..Default::default()
        };
        let (scales, statuses) = compute_controlled_scales(config_scales, &controls);
        assert!((scales[2] - 0.5).abs() < f32::EPSILON);
        assert_eq!(statuses[0].name, "noise_w");
        assert_eq!(statuses[0].status, ProsodyControlStatus::Realized);
    }

    #[test]
    fn compute_controlled_scales_overrides_all_three_scales() {
        let config_scales = [0.667_f32, 1.0, 0.8];
        let controls = PiperProsodyControls {
            noise_scale: Some(0.4),
            length_scale: Some(1.2),
            noise_w: Some(0.6),
            ..Default::default()
        };
        let (scales, statuses) = compute_controlled_scales(config_scales, &controls);
        assert!(
            (scales[0] - 0.4).abs() < f32::EPSILON,
            "noise_scale override"
        );
        assert!(
            (scales[1] - 1.2).abs() < f32::EPSILON,
            "length_scale override"
        );
        assert!((scales[2] - 0.6).abs() < f32::EPSILON, "noise_w override");
        assert_eq!(statuses.len(), 3);
        let names: Vec<_> = statuses.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"noise_scale"));
        assert!(names.contains(&"length_scale"));
        assert!(names.contains(&"noise_w"));
        assert!(
            statuses
                .iter()
                .all(|s| s.status == ProsodyControlStatus::Realized)
        );
    }

    // --- pcm_duration_ms tests ---

    #[test]
    fn pcm_duration_ms_is_zero_for_empty_samples() {
        let pcm = RiperPcm {
            sample_rate_hz: 22_050,
            samples: Vec::new(),
        };
        assert_eq!(pcm_duration_ms(&pcm), 0);
    }

    #[test]
    fn pcm_duration_ms_computes_correct_duration() {
        // 22050 samples at 22050 Hz = 1000 ms
        let pcm = RiperPcm {
            sample_rate_hz: 22_050,
            samples: vec![0.0; 22_050],
        };
        assert_eq!(pcm_duration_ms(&pcm), 1000);
    }

    #[test]
    fn pcm_duration_ms_handles_partial_second() {
        // 11025 samples at 22050 Hz = 500 ms
        let pcm = RiperPcm {
            sample_rate_hz: 22_050,
            samples: vec![0.0; 11_025],
        };
        assert_eq!(pcm_duration_ms(&pcm), 500);
    }

    #[test]
    fn pcm_duration_ms_is_zero_for_zero_sample_rate() {
        let pcm = RiperPcm {
            sample_rate_hz: 0,
            samples: vec![0.0; 100],
        };
        assert_eq!(pcm_duration_ms(&pcm), 0);
    }

    // --- synthesize_ids_with_controls error propagation ---

    #[test]
    fn synthesize_ids_with_controls_fails_when_session_not_loaded() {
        let model_path = unique_path("controls-no-session");
        let mut backend = RiperBackend::unloaded_for_tests(model_path, voice_config());
        let ids = PiperIdSequence { ids: vec![1, 2] };
        let error = backend
            .synthesize_ids_with_controls(&ids, None)
            .expect_err("unloaded session should fail");
        assert_eq!(error.to_string(), "Piper ONNX session has not been loaded");
    }

    #[test]
    fn synthesize_ids_with_controls_fails_on_empty_ids() {
        let model_path = unique_path("controls-empty-ids");
        let mut backend = RiperBackend::unloaded_for_tests(model_path, voice_config());
        let ids = PiperIdSequence { ids: Vec::new() };
        let error = backend
            .synthesize_ids_with_controls(&ids, None)
            .expect_err("empty IDs should fail");
        assert_eq!(
            error.to_string(),
            "Piper ID sequence cannot be empty for ONNX synthesis"
        );
    }

    // --- advisory and approximated control status tests (using synthesize_ids_with_controls
    //     with a mock PCM; we test the post-synthesis diagnostics path by verifying the
    //     control statuses that would be built for various controls configurations) ---

    fn mock_pcm(sample_rate_hz: u32, samples: Vec<f32>) -> RiperPcm {
        RiperPcm {
            sample_rate_hz,
            samples,
        }
    }

    /// Build diagnostics from a mock PCM and a set of controls, simulating what
    /// `synthesize_ids_with_controls` does after inference succeeds.
    fn build_diagnostics_from_controls(
        config_scales: [f32; 3],
        ids: &[i64],
        mut pcm: RiperPcm,
        controls: &PiperProsodyControls,
    ) -> PiperSynthesisDiagnostics {
        let (effective_scales, mut statuses) = compute_controlled_scales(config_scales, controls);

        let mut inserted_pause_ms = 0u64;
        for pause in &controls.pause_overrides {
            let silence_samples = compute_silence_samples(pause.millis, pcm.sample_rate_hz)
                .expect("test pause duration should be reasonable");
            pcm.samples
                .extend(std::iter::repeat_n(0.0_f32, silence_samples));
            inserted_pause_ms = inserted_pause_ms.saturating_add(pause.millis);
            statuses.push(ControlStatusEntry {
                name: format!("pause_override[{}]", pause.label),
                status: ProsodyControlStatus::Approximated,
                detail: format!(
                    "silence of {} ms appended to PCM ({} samples at {} Hz)",
                    pause.millis, silence_samples, pcm.sample_rate_hz
                ),
            });
        }
        for (i, ovr) in controls.phoneme_duration_overrides.iter().enumerate() {
            statuses.push(ControlStatusEntry {
                name: format!("phoneme_duration_override[{i}]"),
                status: ProsodyControlStatus::AdvisoryOnly,
                detail: format!(
                    "per-phoneme duration hint for phoneme index {} ({} ms) is advisory only; \
                     no per-phoneme timing control is available in the current ONNX path",
                    ovr.phoneme_index, ovr.millis
                ),
            });
        }
        for (i, ovr) in controls.boundary_overrides.iter().enumerate() {
            statuses.push(ControlStatusEntry {
                name: format!("boundary_override[{i}]"),
                status: ProsodyControlStatus::AdvisoryOnly,
                detail: format!(
                    "boundary hint after index {} ({}) is advisory only; \
                     no boundary control is available in the current ONNX path",
                    ovr.after_index,
                    if ovr.strong { "strong" } else { "weak" }
                ),
            });
        }

        PiperSynthesisDiagnostics {
            input_phoneme_ids: ids.to_vec(),
            applied_length_scale: effective_scales[1],
            applied_noise_scale: effective_scales[0],
            applied_noise_w: effective_scales[2],
            inserted_pause_ms,
            pcm_duration_ms: pcm_duration_ms(&pcm),
            control_statuses: statuses,
        }
    }

    #[test]
    fn diagnostics_records_pause_as_approximated() {
        let controls = PiperProsodyControls {
            pause_overrides: vec![super::super::prosody_controls::PiperPauseOverride {
                millis: 200,
                label: "after sentence".to_string(),
            }],
            ..Default::default()
        };
        let pcm = mock_pcm(22_050, vec![0.0; 22_050]); // 1 second of audio
        let diag = build_diagnostics_from_controls([0.667, 1.0, 0.8], &[1, 2], pcm, &controls);
        assert_eq!(diag.inserted_pause_ms, 200);
        assert_eq!(diag.control_statuses.len(), 1);
        assert_eq!(
            diag.control_statuses[0].status,
            ProsodyControlStatus::Approximated
        );
        assert!(diag.control_statuses[0].name.contains("after sentence"));
        // 1000 ms (original) + 200 ms (pause) = 1200 ms
        assert_eq!(diag.pcm_duration_ms, 1200);
    }

    #[test]
    fn diagnostics_records_multiple_pauses() {
        let controls = PiperProsodyControls {
            pause_overrides: vec![
                super::super::prosody_controls::PiperPauseOverride {
                    millis: 100,
                    label: "first".to_string(),
                },
                super::super::prosody_controls::PiperPauseOverride {
                    millis: 150,
                    label: "second".to_string(),
                },
            ],
            ..Default::default()
        };
        let pcm = mock_pcm(22_050, vec![0.0; 22_050]); // 1 second
        let diag = build_diagnostics_from_controls([0.667, 1.0, 0.8], &[1], pcm, &controls);
        assert_eq!(diag.inserted_pause_ms, 250);
        assert_eq!(diag.control_statuses.len(), 2);
        assert!(
            diag.control_statuses
                .iter()
                .all(|s| s.status == ProsodyControlStatus::Approximated)
        );
    }

    #[test]
    fn diagnostics_records_phoneme_duration_override_as_advisory() {
        let controls = PiperProsodyControls {
            phoneme_duration_overrides: vec![
                super::super::prosody_controls::PiperPhonemeDurationOverride {
                    phoneme_index: 2,
                    millis: 80,
                },
            ],
            ..Default::default()
        };
        let pcm = mock_pcm(22_050, vec![0.0; 100]);
        let diag = build_diagnostics_from_controls([0.667, 1.0, 0.8], &[1, 2, 3], pcm, &controls);
        assert_eq!(diag.control_statuses.len(), 1);
        assert_eq!(
            diag.control_statuses[0].status,
            ProsodyControlStatus::AdvisoryOnly
        );
        assert!(
            diag.control_statuses[0]
                .name
                .contains("phoneme_duration_override")
        );
        assert!(diag.control_statuses[0].detail.contains("phoneme index 2"));
    }

    #[test]
    fn diagnostics_records_boundary_override_as_advisory() {
        let controls = PiperProsodyControls {
            boundary_overrides: vec![super::super::prosody_controls::PiperBoundaryOverride {
                after_index: 4,
                strong: true,
            }],
            ..Default::default()
        };
        let pcm = mock_pcm(22_050, vec![0.0; 100]);
        let diag = build_diagnostics_from_controls([0.667, 1.0, 0.8], &[1], pcm, &controls);
        assert_eq!(diag.control_statuses.len(), 1);
        assert_eq!(
            diag.control_statuses[0].status,
            ProsodyControlStatus::AdvisoryOnly
        );
        assert!(diag.control_statuses[0].detail.contains("strong"));
    }

    #[test]
    fn diagnostics_records_weak_boundary_override_detail() {
        let controls = PiperProsodyControls {
            boundary_overrides: vec![super::super::prosody_controls::PiperBoundaryOverride {
                after_index: 1,
                strong: false,
            }],
            ..Default::default()
        };
        let pcm = mock_pcm(22_050, vec![0.0; 100]);
        let diag = build_diagnostics_from_controls([0.667, 1.0, 0.8], &[1], pcm, &controls);
        assert!(diag.control_statuses[0].detail.contains("weak"));
    }

    #[test]
    fn diagnostics_records_scale_overrides_as_realized_alongside_advisory_controls() {
        let controls = PiperProsodyControls {
            length_scale: Some(1.3),
            phoneme_duration_overrides: vec![
                super::super::prosody_controls::PiperPhonemeDurationOverride {
                    phoneme_index: 0,
                    millis: 60,
                },
            ],
            boundary_overrides: vec![super::super::prosody_controls::PiperBoundaryOverride {
                after_index: 0,
                strong: false,
            }],
            ..Default::default()
        };
        let pcm = mock_pcm(22_050, vec![0.0; 100]);
        let diag = build_diagnostics_from_controls([0.667, 1.0, 0.8], &[1], pcm, &controls);
        // length_scale (Realized) + phoneme_duration_override (Advisory) + boundary_override (Advisory)
        assert_eq!(diag.control_statuses.len(), 3);
        let realized: Vec<_> = diag
            .control_statuses
            .iter()
            .filter(|s| s.status == ProsodyControlStatus::Realized)
            .collect();
        assert_eq!(realized.len(), 1);
        assert_eq!(realized[0].name, "length_scale");
        assert!((diag.applied_length_scale - 1.3).abs() < 0.001);
    }

    #[test]
    fn diagnostics_stores_input_phoneme_ids() {
        let controls = PiperProsodyControls::default();
        let pcm = mock_pcm(22_050, vec![0.0; 100]);
        let ids = &[10_i64, 20, 30];
        let diag = build_diagnostics_from_controls([0.667, 1.0, 0.8], ids, pcm, &controls);
        assert_eq!(diag.input_phoneme_ids, vec![10, 20, 30]);
    }
}
