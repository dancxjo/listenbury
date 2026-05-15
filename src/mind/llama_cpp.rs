use std::collections::HashMap;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result, bail};
use crossbeam_channel::{Receiver, unbounded};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{LogOptions, send_logs_to_tracing};
use uuid::Uuid;

use crate::mind::llm::{GenerationId, GenerationRequest, LlmEngine, LlmEvent};
use crate::runtime::developer_diagnostics_enabled;

static LLAMA_BACKEND: OnceLock<Arc<LlamaBackend>> = OnceLock::new();

#[derive(Debug, Clone)]
pub struct LlamaCppConfig {
    pub model_path: PathBuf,
    pub gpu_layers: Option<u32>,
    pub context_size: u32,
    pub max_tokens: usize,
    pub threads: usize,
    pub temperature: f32,
    pub top_p: f32,
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            gpu_layers: None,
            context_size: 2048,
            max_tokens: 128,
            threads: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(4),
            temperature: 0.8,
            top_p: 0.95,
        }
    }
}

#[derive(Debug)]
pub struct LlamaCppEngine {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
    config: LlamaCppConfig,
    active: HashMap<GenerationId, ActiveGeneration>,
}

#[derive(Debug)]
struct ActiveGeneration {
    events: Receiver<LlmEvent>,
    cancel: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl LlamaCppEngine {
    pub fn new(config: LlamaCppConfig) -> Result<Self> {
        if config.model_path.as_os_str().is_empty() {
            bail!("llama.cpp model_path is required");
        }
        if config.context_size == 0 {
            bail!("llama.cpp context_size must be greater than zero");
        }
        if config.max_tokens == 0 {
            bail!("llama.cpp max_tokens must be greater than zero");
        }
        if config.threads == 0 {
            bail!("llama.cpp threads must be greater than zero");
        }

        let backend = llama_backend()?;
        let mut model_params = LlamaModelParams::default();
        if let Some(gpu_layers) = config.gpu_layers {
            model_params = model_params.with_n_gpu_layers(gpu_layers);
        }
        let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)
            .with_context(|| {
                format!(
                    "failed to load llama.cpp model at {}",
                    config.model_path.display()
                )
            })?;

        Ok(Self {
            backend,
            model: Arc::new(model),
            config,
            active: HashMap::new(),
        })
    }
}

impl LlmEngine for LlamaCppEngine {
    fn start(&mut self, request: GenerationRequest) -> Result<GenerationId> {
        let id = GenerationId(Uuid::new_v4());
        let (sender, receiver) = unbounded();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker = LlamaGenerationWorker {
            backend: Arc::clone(&self.backend),
            model: Arc::clone(&self.model),
            config: self.config.clone(),
            request,
            cancel: Arc::clone(&cancel),
        };

        let handle = thread::Builder::new()
            .name(format!("llama-cpp-generation-{}", id.0))
            .spawn(move || {
                let event = match worker.run(&sender) {
                    Ok(GenerationOutcome::Completed) => LlmEvent::Completed,
                    Ok(GenerationOutcome::Cancelled) => LlmEvent::Cancelled,
                    Err(error) => LlmEvent::Error {
                        message: error.to_string(),
                    },
                };
                let _ = sender.send(event);
            })
            .context("failed to spawn llama.cpp generation worker")?;

        self.active.insert(
            id,
            ActiveGeneration {
                events: receiver,
                cancel,
                handle: Some(handle),
            },
        );
        Ok(id)
    }

    fn poll(&mut self, id: GenerationId) -> Result<Vec<LlmEvent>> {
        let Some(active) = self.active.get_mut(&id) else {
            return Ok(vec![LlmEvent::Error {
                message: "generation not found".to_string(),
            }]);
        };

        let events = active.events.try_iter().collect::<Vec<_>>();
        if events.iter().any(is_terminal_event) {
            if let Some(mut active) = self.active.remove(&id) {
                if let Some(handle) = active.handle.take() {
                    let _ = handle.join();
                }
            }
        }

        Ok(events)
    }

    fn cancel(&mut self, id: GenerationId) -> Result<()> {
        let Some(active) = self.active.get(&id) else {
            bail!("generation not found");
        };
        active.cancel.store(true, Ordering::Relaxed);
        Ok(())
    }
}

impl Drop for LlamaCppEngine {
    fn drop(&mut self) {
        for active in self.active.values() {
            active.cancel.store(true, Ordering::Relaxed);
        }
        for active in self.active.values_mut() {
            if let Some(handle) = active.handle.take() {
                let _ = handle.join();
            }
        }
    }
}

#[derive(Debug)]
struct LlamaGenerationWorker {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
    config: LlamaCppConfig,
    request: GenerationRequest,
    cancel: Arc<AtomicBool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GenerationOutcome {
    Completed,
    Cancelled,
}

impl LlamaGenerationWorker {
    fn run(self, sender: &crossbeam_channel::Sender<LlmEvent>) -> Result<GenerationOutcome> {
        let context_size = NonZeroU32::new(self.config.context_size)
            .context("llama.cpp context_size must be greater than zero")?;
        let max_tokens = self.request.max_tokens.unwrap_or(self.config.max_tokens);
        let max_total_tokens = checked_total_tokens(&self.request.prompt, &self.model, max_tokens)?;

        let thread_count =
            i32::try_from(self.config.threads).context("threads exceeds i32::MAX")?;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(context_size))
            .with_n_threads(thread_count)
            .with_n_threads_batch(thread_count);
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .context("failed to create llama.cpp context")?;

        let prompt_tokens = self
            .model
            .str_to_token(&self.request.prompt, AddBos::Always)
            .context("failed to tokenize prompt")?;
        if prompt_tokens.is_empty() {
            bail!("prompt produced no tokens");
        }

        let n_ctx = ctx.n_ctx() as usize;
        if max_total_tokens > n_ctx {
            bail!(
                "generation needs {max_total_tokens} context tokens, but context_size is {n_ctx}"
            );
        }

        let mut batch = LlamaBatch::new(prompt_tokens.len().max(1), 1);
        let last_index = prompt_tokens.len() - 1;
        for (index, token) in prompt_tokens.into_iter().enumerate() {
            let position =
                i32::try_from(index).context("prompt token position exceeds i32::MAX")?;
            batch.add(token, position, &[0], index == last_index)?;
        }
        ctx.decode(&mut batch)
            .context("failed to decode prompt with llama.cpp")?;

        let mut n_cur = batch.n_tokens();
        let mut sampler = build_sampler(self.config.temperature, self.config.top_p);
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut stop_detector = StopDetector::new(self.request.stop);

        while (n_cur as usize) < max_total_tokens {
            if self.cancel.load(Ordering::Relaxed) {
                return Ok(GenerationOutcome::Cancelled);
            }

            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if self.model.is_eog_token(token) {
                return Ok(GenerationOutcome::Completed);
            }

            let text = self
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .context("failed to decode llama.cpp token")?;
            if !text.is_empty() {
                let outcome = stop_detector.push(&text);
                if !outcome.text.is_empty()
                    && sender.send(LlmEvent::Token { text: outcome.text }).is_err()
                {
                    return Ok(GenerationOutcome::Cancelled);
                }
                if outcome.stopped {
                    return Ok(GenerationOutcome::Completed);
                }
            }

            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .context("failed to add sampled token to llama.cpp batch")?;
            n_cur += 1;
            ctx.decode(&mut batch)
                .context("failed to decode sampled token with llama.cpp")?;
        }

        let trailing = stop_detector.finish();
        if !trailing.is_empty() && sender.send(LlmEvent::Token { text: trailing }).is_err() {
            return Ok(GenerationOutcome::Cancelled);
        }

        Ok(GenerationOutcome::Completed)
    }
}

#[derive(Debug, Default)]
struct StopDetector {
    stops: Vec<String>,
    pending: String,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct StopDetection {
    text: String,
    stopped: bool,
}

impl StopDetector {
    fn new(stops: Vec<String>) -> Self {
        Self {
            stops: stops.into_iter().filter(|stop| !stop.is_empty()).collect(),
            pending: String::new(),
        }
    }

    fn push(&mut self, text: &str) -> StopDetection {
        if self.stops.is_empty() {
            return StopDetection {
                text: text.to_string(),
                stopped: false,
            };
        }

        self.pending.push_str(text);
        if let Some(stop_index) = self.find_earliest_stop() {
            let output = self.pending[..stop_index].to_string();
            self.pending.clear();
            return StopDetection {
                text: output,
                stopped: true,
            };
        }

        let keep = self.longest_stop_prefix_suffix_len();
        let emit_len = self.pending.len() - keep;
        let output = self.pending[..emit_len].to_string();
        self.pending = self.pending[emit_len..].to_string();
        StopDetection {
            text: output,
            stopped: false,
        }
    }

    fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    fn find_earliest_stop(&self) -> Option<usize> {
        self.stops
            .iter()
            .filter_map(|stop| self.pending.find(stop))
            .min()
    }

    fn longest_stop_prefix_suffix_len(&self) -> usize {
        self.stops
            .iter()
            .flat_map(|stop| {
                stop.char_indices()
                    .skip(1)
                    .map(|(index, _)| index)
                    .chain(std::iter::once(stop.len()))
                    .filter(|&len| len <= self.pending.len())
                    .filter(|&len| self.pending.ends_with(&stop[..len]))
            })
            .max()
            .unwrap_or(0)
    }
}

fn llama_backend() -> Result<Arc<LlamaBackend>> {
    if let Some(backend) = LLAMA_BACKEND.get() {
        return Ok(Arc::clone(backend));
    }

    send_logs_to_tracing(LogOptions::default().with_logs_enabled(developer_diagnostics_enabled()));
    let backend = Arc::new(LlamaBackend::init().context("failed to initialize llama.cpp backend")?);
    let _ = LLAMA_BACKEND.set(Arc::clone(&backend));
    Ok(backend)
}

fn checked_total_tokens(prompt: &str, model: &LlamaModel, max_tokens: usize) -> Result<usize> {
    if max_tokens == 0 {
        bail!("max_tokens must be greater than zero");
    }
    let prompt_tokens = model
        .str_to_token(prompt, AddBos::Always)
        .context("failed to tokenize prompt")?
        .len();
    prompt_tokens
        .checked_add(max_tokens)
        .context("prompt plus max_tokens overflowed usize")
}

fn build_sampler(temperature: f32, top_p: f32) -> LlamaSampler {
    if temperature <= 0.0 {
        return LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    }

    let clamped_top_p = top_p.clamp(0.0, 1.0);
    LlamaSampler::chain_simple([
        LlamaSampler::top_p(clamped_top_p, 1),
        LlamaSampler::temp(temperature),
        LlamaSampler::dist(1234),
    ])
}

fn is_terminal_event(event: &LlmEvent) -> bool {
    matches!(
        event,
        LlmEvent::Completed | LlmEvent::Cancelled | LlmEvent::Error { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_detector_stops_before_marker_in_single_token() {
        let mut detector = StopDetector::new(vec!["\nUser:".to_string()]);

        assert_eq!(
            detector.push("Yes.\nUser: Again"),
            StopDetection {
                text: "Yes.".to_string(),
                stopped: true,
            }
        );
    }

    #[test]
    fn stop_detector_holds_split_marker_prefix() {
        let mut detector = StopDetector::new(vec!["\nUser:".to_string()]);

        assert_eq!(
            detector.push("Yes.\nUs"),
            StopDetection {
                text: "Yes.".to_string(),
                stopped: false,
            }
        );
        assert_eq!(
            detector.push("er: Again"),
            StopDetection {
                text: String::new(),
                stopped: true,
            }
        );
    }

    #[test]
    fn stop_detector_holds_marker_prefix_when_it_is_the_whole_token() {
        let mut detector = StopDetector::new(vec!["\nUser:".to_string()]);

        assert_eq!(
            detector.push("\nUs"),
            StopDetection {
                text: String::new(),
                stopped: false,
            }
        );
        assert_eq!(
            detector.push("er: Again"),
            StopDetection {
                text: String::new(),
                stopped: true,
            }
        );
    }

    #[test]
    fn stop_detector_flushes_unmatched_pending_text() {
        let mut detector = StopDetector::new(vec!["\nUser:".to_string()]);

        assert_eq!(
            detector.push("Yes.\nUsual"),
            StopDetection {
                text: "Yes.\nUsual".to_string(),
                stopped: false,
            }
        );
        assert_eq!(detector.finish(), "");
    }
}
