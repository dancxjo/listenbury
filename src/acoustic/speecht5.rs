use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

use crate::acoustic::{AcousticFrameTrack, MelFrame};

#[cfg(feature = "tts-onnx")]
use ort::session::{Session, builder::GraphOptimizationLevel};
#[cfg(feature = "tts-onnx")]
use ort::value::{DynTensorValueType, Tensor};

#[cfg(feature = "tts-onnx")]
use crate::mouth::riper::backend::initialize_ort_runtime;

const SAMPLE_RATE_HZ: u32 = 16_000;
const HOP_SAMPLES: usize = 256;
const MEL_BINS: usize = 80;
const HIDDEN_SIZE: usize = 768;
const SPEAKER_EMBEDDING_DIM: usize = 512;
const DECODER_LAYERS: usize = 6;
const DECODER_HEADS: usize = 12;
const DECODER_HEAD_DIM: usize = 64;
const REDUCTION_FACTOR: usize = 2;
const STOP_THRESHOLD: f32 = 0.5;
const MINLEN_RATIO: f32 = 0.0;
const MAXLEN_RATIO: f32 = 20.0;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechT5OnnxPaths {
    pub encoder_model: PathBuf,
    pub decoder_model: PathBuf,
    pub tokenizer_json: PathBuf,
    pub speaker_embeddings: PathBuf,
}

impl SpeechT5OnnxPaths {
    pub fn from_dir(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        Self {
            encoder_model: dir.join("encoder_model_quantized.onnx"),
            decoder_model: dir.join("decoder_model_merged_quantized.onnx"),
            tokenizer_json: dir.join("tokenizer.json"),
            speaker_embeddings: dir.join("speaker_embeddings.bin"),
        }
    }
}

pub struct SpeechT5OnnxAcousticGenerator {
    tokenizer: SpeechT5Tokenizer,
    speaker_embeddings: Vec<f32>,
    #[cfg(feature = "tts-onnx")]
    encoder: Session,
    #[cfg(feature = "tts-onnx")]
    decoder: Session,
}

impl SpeechT5OnnxAcousticGenerator {
    #[cfg(feature = "tts-onnx")]
    pub fn load(paths: SpeechT5OnnxPaths) -> Result<Self> {
        ensure!(
            paths.encoder_model.is_file(),
            "SpeechT5 encoder ONNX model file not found at {}",
            paths.encoder_model.display()
        );
        ensure!(
            paths.decoder_model.is_file(),
            "SpeechT5 decoder ONNX model file not found at {}",
            paths.decoder_model.display()
        );

        initialize_ort_runtime()?;
        let encoder = load_session(&paths.encoder_model, "SpeechT5 encoder")?;
        let decoder = load_session(&paths.decoder_model, "SpeechT5 decoder")?;
        Ok(Self {
            tokenizer: SpeechT5Tokenizer::from_file(&paths.tokenizer_json)?,
            speaker_embeddings: read_speaker_embeddings(&paths.speaker_embeddings)?,
            encoder,
            decoder,
        })
    }

    #[cfg(not(feature = "tts-onnx"))]
    pub fn load(paths: SpeechT5OnnxPaths) -> Result<Self> {
        let _ = paths;
        anyhow::bail!("SpeechT5 ONNX acoustic generation requires the `tts-onnx` feature")
    }

    #[cfg(feature = "tts-onnx")]
    pub fn generate_text(&mut self, text: &str) -> Result<AcousticFrameTrack> {
        let token_ids = self.tokenizer.encode(text)?;
        ensure!(
            !token_ids.is_empty(),
            "SpeechT5 tokenizer produced no token IDs"
        );
        let token_count = token_ids.len();
        let token_len = i64::try_from(token_ids.len()).context("SpeechT5 text is too long")?;
        let encoder_input = Tensor::from_array((vec![1_i64, token_len], token_ids))
            .context("failed to build SpeechT5 encoder input_ids tensor")?
            .upcast();
        let (encoder_sequence_len, encoder_hidden_states, encoder_attention_mask) = {
            let encoder_outputs = self
                .encoder
                .run(vec![("input_ids".to_string(), encoder_input)])
                .context("failed to run SpeechT5 encoder ONNX inference")?;
            let (encoder_shape, encoder_hidden_states) =
                extract_f32_tensor(&encoder_outputs, "encoder_outputs")?;
            ensure!(
                encoder_shape.len() == 3
                    && encoder_shape[0] == 1
                    && encoder_shape[2] == HIDDEN_SIZE as i64,
                "SpeechT5 encoder output has unexpected shape {:?}",
                encoder_shape
            );
            let encoder_sequence_len = usize::try_from(encoder_shape[1])
                .context("invalid SpeechT5 encoder sequence length")?;
            let (_, encoder_attention_mask) =
                extract_i64_tensor(&encoder_outputs, "encoder_attention_mask")?;
            (
                encoder_sequence_len,
                encoder_hidden_states,
                encoder_attention_mask,
            )
        };

        let heuristic_min_len = token_count.saturating_sub(1).saturating_mul(2);
        let max_len = ((encoder_sequence_len as f32 / REDUCTION_FACTOR as f32) * MAXLEN_RATIO)
            .floor()
            .max(heuristic_min_len.max(1) as f32) as usize;
        let min_len = ((encoder_sequence_len as f32 / REDUCTION_FACTOR as f32) * MINLEN_RATIO)
            .floor()
            .max(heuristic_min_len as f32) as usize;

        let mut output_sequence = TensorData {
            shape: vec![1, 1, MEL_BINS as i64],
            values: vec![0.0; MEL_BINS],
        };
        let mut past = SpeechT5DecoderCache::empty();
        let mut has_decoder_outputs = false;
        let mut mel = Vec::new();

        for step in 1..=max_len {
            let decoder_outputs = self.run_decoder(
                has_decoder_outputs,
                &output_sequence,
                &encoder_hidden_states,
                encoder_sequence_len,
                &encoder_attention_mask,
                &past,
            )?;
            has_decoder_outputs = true;
            output_sequence = decoder_outputs.output_sequence;
            past.update_from_present(decoder_outputs.present);

            for frame_values in decoder_outputs.spectrum.values.chunks_exact(MEL_BINS) {
                mel.push(MelFrame {
                    bins: frame_values.to_vec(),
                });
            }

            if step >= min_len
                && decoder_outputs
                    .prob
                    .iter()
                    .any(|probability| *probability >= STOP_THRESHOLD)
            {
                break;
            }
        }

        ensure!(
            !mel.is_empty(),
            "SpeechT5 acoustic decoder produced no mel frames"
        );
        let f0_hz = vec![0.0; mel.len()];
        let voiced = vec![false; mel.len()];
        Ok(AcousticFrameTrack {
            mel,
            f0_hz,
            voiced,
            sample_rate_hz: SAMPLE_RATE_HZ,
            hop_samples: HOP_SAMPLES,
        })
    }

    #[cfg(not(feature = "tts-onnx"))]
    pub fn generate_text(&mut self, text: &str) -> Result<AcousticFrameTrack> {
        let _ = text;
        anyhow::bail!("SpeechT5 ONNX acoustic generation requires the `tts-onnx` feature")
    }

    #[cfg(feature = "tts-onnx")]
    fn run_decoder(
        &mut self,
        use_cache_branch: bool,
        output_sequence: &TensorData,
        encoder_hidden_states: &[f32],
        encoder_sequence_len: usize,
        encoder_attention_mask: &[i64],
        past: &SpeechT5DecoderCache,
    ) -> Result<SpeechT5DecoderStep> {
        let encoder_sequence_len =
            i64::try_from(encoder_sequence_len).context("SpeechT5 encoder sequence too long")?;
        let output_sequence_tensor = Tensor::from_array((
            output_sequence.shape.clone(),
            output_sequence.values.clone(),
        ))
        .context("failed to build SpeechT5 decoder output_sequence tensor")?
        .upcast();
        let speaker_tensor = Tensor::from_array((
            vec![1_i64, SPEAKER_EMBEDDING_DIM as i64],
            self.speaker_embeddings.clone(),
        ))
        .context("failed to build SpeechT5 speaker_embeddings tensor")?
        .upcast();
        let encoder_tensor = Tensor::from_array((
            vec![1_i64, encoder_sequence_len, HIDDEN_SIZE as i64],
            encoder_hidden_states.to_vec(),
        ))
        .context("failed to build SpeechT5 encoder_hidden_states tensor")?
        .upcast();
        let attention_tensor = Tensor::from_array((
            vec![1_i64, encoder_sequence_len],
            encoder_attention_mask.to_vec(),
        ))
        .context("failed to build SpeechT5 encoder_attention_mask tensor")?
        .upcast();
        let use_cache_tensor = Tensor::from_array((vec![1_i64], vec![use_cache_branch]))
            .context("failed to build SpeechT5 use_cache_branch tensor")?
            .upcast();

        let mut inputs: Vec<(String, ort::value::DynValue)> = vec![
            ("speaker_embeddings".to_string(), speaker_tensor),
            ("encoder_hidden_states".to_string(), encoder_tensor),
            ("output_sequence".to_string(), output_sequence_tensor),
            ("encoder_attention_mask".to_string(), attention_tensor),
            ("use_cache_branch".to_string(), use_cache_tensor),
        ]
        .into_iter()
        .map(|(name, value)| (name, value.into()))
        .collect();
        past.push_inputs(&mut inputs)?;

        let outputs = self
            .decoder
            .run(inputs)
            .context("failed to run SpeechT5 decoder ONNX inference")?;
        let (output_shape, output_values) = extract_f32_tensor(&outputs, "output_sequence_out")?;
        let (spectrum_shape, spectrum_values) = extract_f32_tensor(&outputs, "spectrum")?;
        ensure!(
            spectrum_shape.len() == 2 && spectrum_shape[1] == MEL_BINS as i64,
            "SpeechT5 decoder spectrum has unexpected shape {:?}",
            spectrum_shape
        );
        let (_, prob) = extract_f32_tensor(&outputs, "prob")?;
        let present = SpeechT5DecoderCache::from_outputs(&outputs)?;

        Ok(SpeechT5DecoderStep {
            output_sequence: TensorData {
                shape: output_shape,
                values: output_values,
            },
            spectrum: TensorData {
                shape: spectrum_shape,
                values: spectrum_values,
            },
            prob,
            present,
        })
    }
}

#[cfg(feature = "tts-onnx")]
fn load_session(path: &Path, label: &str) -> Result<Session> {
    Session::builder()
        .with_context(|| format!("failed to create {label} ONNX session builder"))?
        .with_intra_threads(1)
        .map_err(|error| anyhow::anyhow!("failed to configure {label} intra-op threads: {error}"))?
        .with_inter_threads(1)
        .map_err(|error| anyhow::anyhow!("failed to configure {label} inter-op threads: {error}"))?
        .with_intra_op_spinning(false)
        .map_err(|error| anyhow::anyhow!("failed to configure {label} intra-op spinning: {error}"))?
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .map_err(|error| anyhow::anyhow!("failed to configure {label} optimization: {error}"))?
        .commit_from_file(path)
        .with_context(|| format!("failed to load {label} ONNX model from {}", path.display()))
}

#[cfg(feature = "tts-onnx")]
#[derive(Debug, Clone)]
struct TensorData {
    shape: Vec<i64>,
    values: Vec<f32>,
}

#[cfg(feature = "tts-onnx")]
struct SpeechT5DecoderStep {
    output_sequence: TensorData,
    spectrum: TensorData,
    prob: Vec<f32>,
    present: SpeechT5DecoderCache,
}

#[cfg(feature = "tts-onnx")]
#[derive(Debug, Clone)]
struct SpeechT5DecoderCache {
    entries: HashMap<String, TensorData>,
}

#[cfg(feature = "tts-onnx")]
impl SpeechT5DecoderCache {
    fn empty() -> Self {
        let mut entries = HashMap::new();
        for layer in 0..DECODER_LAYERS {
            for branch in ["decoder", "encoder"] {
                for kind in ["key", "value"] {
                    entries.insert(
                        format!("past_key_values.{layer}.{branch}.{kind}"),
                        TensorData {
                            shape: vec![1, DECODER_HEADS as i64, 1, DECODER_HEAD_DIM as i64],
                            values: vec![0.0; DECODER_HEADS * DECODER_HEAD_DIM],
                        },
                    );
                }
            }
        }
        Self { entries }
    }

    fn push_inputs(&self, inputs: &mut Vec<(String, ort::value::DynValue)>) -> Result<()> {
        for layer in 0..DECODER_LAYERS {
            for branch in ["decoder", "encoder"] {
                for kind in ["key", "value"] {
                    let name = format!("past_key_values.{layer}.{branch}.{kind}");
                    let data = self
                        .entries
                        .get(&name)
                        .with_context(|| format!("missing SpeechT5 decoder cache `{name}`"))?;
                    let tensor = Tensor::from_array((data.shape.clone(), data.values.clone()))
                        .with_context(|| {
                            format!("failed to build SpeechT5 decoder cache tensor `{name}`")
                        })?
                        .upcast();
                    inputs.push((name, tensor.into()));
                }
            }
        }
        Ok(())
    }

    fn from_outputs(outputs: &ort::session::SessionOutputs<'_>) -> Result<Self> {
        let mut entries = HashMap::new();
        for layer in 0..DECODER_LAYERS {
            for branch in ["decoder", "encoder"] {
                for kind in ["key", "value"] {
                    let output_name = format!("present.{layer}.{branch}.{kind}");
                    let input_name = format!("past_key_values.{layer}.{branch}.{kind}");
                    let (shape, values) = extract_f32_tensor(outputs, &output_name)?;
                    entries.insert(input_name, TensorData { shape, values });
                }
            }
        }
        Ok(Self { entries })
    }

    fn update_from_present(&mut self, present: Self) {
        for (name, value) in present.entries {
            let current_encoder_cache_is_real = name.contains(".encoder.")
                && self
                    .entries
                    .get(&name)
                    .and_then(|entry| entry.shape.get(2))
                    .is_some_and(|len| *len > 1);
            if current_encoder_cache_is_real {
                continue;
            }
            self.entries.insert(name, value);
        }
    }
}

#[cfg(feature = "tts-onnx")]
fn extract_f32_tensor(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<(Vec<i64>, Vec<f32>)> {
    let output = outputs
        .get(name)
        .with_context(|| format!("SpeechT5 ONNX inference did not return `{name}`"))?;
    let output = output
        .downcast_ref::<DynTensorValueType>()
        .with_context(|| format!("SpeechT5 ONNX output `{name}` is not a tensor"))?;
    let (shape, values) = output
        .try_extract_tensor::<f32>()
        .with_context(|| format!("SpeechT5 ONNX output `{name}` is not f32"))?;
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "SpeechT5 ONNX output `{name}` contains non-finite values"
    );
    Ok((shape.to_vec(), values.to_vec()))
}

#[cfg(feature = "tts-onnx")]
fn extract_i64_tensor(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<(Vec<i64>, Vec<i64>)> {
    let output = outputs
        .get(name)
        .with_context(|| format!("SpeechT5 ONNX inference did not return `{name}`"))?;
    let output = output
        .downcast_ref::<DynTensorValueType>()
        .with_context(|| format!("SpeechT5 ONNX output `{name}` is not a tensor"))?;
    let (shape, values) = output
        .try_extract_tensor::<i64>()
        .with_context(|| format!("SpeechT5 ONNX output `{name}` is not i64"))?;
    Ok((shape.to_vec(), values.to_vec()))
}

#[derive(Debug, Deserialize)]
struct TokenizerJson {
    model: TokenizerModel,
}

#[derive(Debug, Deserialize)]
struct TokenizerModel {
    vocab: HashMap<String, i64>,
}

#[derive(Debug, Clone)]
struct SpeechT5Tokenizer {
    vocab: HashMap<String, i64>,
}

impl SpeechT5Tokenizer {
    fn from_file(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read SpeechT5 tokenizer at {}", path.display()))?;
        let tokenizer: TokenizerJson = serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse SpeechT5 tokenizer at {}", path.display()))?;
        Ok(Self {
            vocab: tokenizer.model.vocab,
        })
    }

    fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let mut ids = Vec::new();
        for word in text.split_whitespace() {
            self.push_token("▁", &mut ids)?;
            for ch in word.chars() {
                self.push_token(&ch.to_string(), &mut ids)?;
            }
        }
        ids.push(self.token_id("</s>")?);
        Ok(ids)
    }

    fn push_token(&self, token: &str, ids: &mut Vec<i64>) -> Result<()> {
        ids.push(
            self.vocab
                .get(token)
                .copied()
                .or_else(|| self.vocab.get("<unk>").copied())
                .with_context(|| format!("SpeechT5 tokenizer has no token `{token}` or <unk>"))?,
        );
        Ok(())
    }

    fn token_id(&self, token: &str) -> Result<i64> {
        self.vocab
            .get(token)
            .copied()
            .with_context(|| format!("SpeechT5 tokenizer has no token `{token}`"))
    }
}

fn read_speaker_embeddings(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| {
        format!(
            "failed to read SpeechT5 speaker embeddings at {}",
            path.display()
        )
    })?;
    ensure!(
        bytes.len() == SPEAKER_EMBEDDING_DIM * std::mem::size_of::<f32>(),
        "SpeechT5 speaker embeddings must contain {SPEAKER_EMBEDDING_DIM} f32 values; {} has {} bytes",
        path.display(),
        bytes.len()
    );
    let values = bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect::<Vec<_>>();
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "SpeechT5 speaker embeddings contain non-finite values"
    );
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::SpeechT5Tokenizer;

    #[test]
    fn tokenizer_uses_metaspace_char_tokens_and_eos() {
        let mut vocab = std::collections::HashMap::new();
        vocab.insert("▁".to_string(), 4);
        vocab.insert("H".to_string(), 35);
        vocab.insert("i".to_string(), 10);
        vocab.insert("</s>".to_string(), 2);
        vocab.insert("<unk>".to_string(), 3);
        let tokenizer = SpeechT5Tokenizer { vocab };

        assert_eq!(tokenizer.encode("Hi").unwrap(), vec![4, 35, 10, 2]);
        assert_eq!(tokenizer.encode("Hi 🙂").unwrap(), vec![4, 35, 10, 4, 3, 2]);
    }
}
