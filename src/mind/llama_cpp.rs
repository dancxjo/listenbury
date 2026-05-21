use std::collections::HashMap;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result, bail};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError, unbounded};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{LogOptions, send_logs_to_tracing};
use uuid::Uuid;

use crate::diagnostics::developer_diagnostics_enabled;
use crate::mind::llm::{GenerationId, GenerationRequest, LlmEngine, LlmEvent};

static LLAMA_BACKEND: OnceLock<Arc<LlamaBackend>> = OnceLock::new();

#[derive(Debug, Clone)]
pub struct LlamaCppConfig {
    pub model_path: PathBuf,
    pub gpu_layers: Option<u32>,
    pub cpu_only: bool,
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
            cpu_only: false,
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
    controls: Sender<GenerationControl>,
    cancel: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

#[derive(Debug)]
enum GenerationControl {
    AppendPrompt { text: String },
    SetPaused { paused: bool },
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
        ensure_model_runtime_supported(&config.model_path)?;

        let backend = llama_backend()?;
        let mut model_params = LlamaModelParams::default();
        if config.cpu_only {
            model_params = model_params.with_n_gpu_layers(0);
        } else if let Some(gpu_layers) = config.gpu_layers {
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

fn ensure_model_runtime_supported(model_path: &std::path::Path) -> Result<()> {
    let filename = model_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    if filename.contains("gemma-4") {
        bail!(
            "llama.cpp runtime is incompatible with this LLM: {}. Select a supported model with `cargo run -- models use llm llama`, or pass `--llm-model <path>`.",
            model_path.display()
        );
    }

    Ok(())
}

impl LlmEngine for LlamaCppEngine {
    fn start(&mut self, request: GenerationRequest) -> Result<GenerationId> {
        let id = GenerationId(Uuid::new_v4());
        let (sender, receiver) = unbounded();
        let (control_sender, control_receiver) = unbounded();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker = LlamaGenerationWorker {
            backend: Arc::clone(&self.backend),
            model: Arc::clone(&self.model),
            config: self.config.clone(),
            request,
            controls: control_receiver,
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
                controls: control_sender,
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
        if events.iter().any(is_terminal_event)
            && let Some(mut active) = self.active.remove(&id)
            && let Some(handle) = active.handle.take()
        {
            let _ = handle.join();
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

    fn append_prompt(&mut self, id: GenerationId, text: String) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }

        let Some(active) = self.active.get(&id) else {
            bail!("generation not found");
        };
        active
            .controls
            .send(GenerationControl::AppendPrompt { text })
            .context("generation is no longer accepting prompt appends")
    }
}

impl LlamaCppEngine {
    pub fn set_paused(&mut self, id: GenerationId, paused: bool) -> Result<()> {
        let Some(active) = self.active.get(&id) else {
            bail!("generation not found");
        };
        active
            .controls
            .send(GenerationControl::SetPaused { paused })
            .context("generation is no longer accepting pause controls")
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
    controls: Receiver<GenerationControl>,
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
        let max_total_tokens =
            checked_total_tokens(&self.request.prompt, &self.model, self.request.max_tokens)?;

        let thread_count =
            i32::try_from(self.config.threads).context("threads exceeds i32::MAX")?;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(context_size))
            .with_n_threads(thread_count)
            .with_n_threads_batch(thread_count);
        let ctx_params = if self.config.cpu_only {
            ctx_params
                .with_offload_kqv(false)
                .with_op_offload(false)
                .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_DISABLED)
        } else {
            ctx_params
        };
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

        let mut batch = LlamaBatch::new(n_ctx, 1);
        let mut n_cur = 0;
        decode_prompt_tokens(
            &mut ctx,
            &mut batch,
            &prompt_tokens,
            &mut n_cur,
            n_ctx,
            "prompt",
        )?;
        let mut generated_tokens = 0usize;
        let mut sampler = build_sampler(self.config.temperature, self.config.top_p);
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut stop_detector = StopDetector::new(self.request.stop);
        let mut paused = false;

        while within_generation_limit(generated_tokens, self.request.max_tokens)
            && (n_cur as usize) < n_ctx
        {
            if self.cancel.load(Ordering::Relaxed) {
                return Ok(GenerationOutcome::Cancelled);
            }
            // Apply append-only live input at token boundaries. This preserves a single KV
            // context: pending appends are decoded after all prior prompt/generated tokens and
            // before the next assistant token is sampled.
            drain_generation_controls(
                &self.model,
                &mut ctx,
                &mut batch,
                &mut n_cur,
                n_ctx,
                &self.controls,
                &mut paused,
            )?;
            wait_while_paused(
                &self.model,
                &mut ctx,
                &mut batch,
                &mut n_cur,
                n_ctx,
                &self.controls,
                &self.cancel,
                &mut paused,
            )?;
            if self.cancel.load(Ordering::Relaxed) {
                return Ok(GenerationOutcome::Cancelled);
            }
            if (n_cur as usize) >= n_ctx {
                break;
            }

            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if self.model.is_eog_token(token) {
                if self.request.max_tokens.is_some() {
                    return Ok(GenerationOutcome::Completed);
                }
                commit_sampled_token(&mut ctx, &mut batch, token, &mut n_cur)?;
                generated_tokens += 1;
                continue;
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

            commit_sampled_token(&mut ctx, &mut batch, token, &mut n_cur)?;
            generated_tokens += 1;
        }

        let trailing = stop_detector.finish();
        if !trailing.is_empty() && sender.send(LlmEvent::Token { text: trailing }).is_err() {
            return Ok(GenerationOutcome::Cancelled);
        }

        Ok(GenerationOutcome::Completed)
    }
}

fn within_generation_limit(generated_tokens: usize, max_tokens: Option<usize>) -> bool {
    max_tokens.is_none_or(|max_tokens| generated_tokens < max_tokens)
}

fn commit_sampled_token(
    ctx: &mut LlamaContext<'_>,
    batch: &mut LlamaBatch<'_>,
    token: llama_cpp_2::token::LlamaToken,
    n_cur: &mut i32,
) -> Result<()> {
    batch.clear();
    batch
        .add(token, *n_cur, &[0], true)
        .context("failed to add sampled token to llama.cpp batch")?;
    *n_cur += 1;
    ctx.decode(batch)
        .context("failed to decode sampled token with llama.cpp")?;
    Ok(())
}

fn drain_generation_controls(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    batch: &mut LlamaBatch<'_>,
    n_cur: &mut i32,
    n_ctx: usize,
    controls: &Receiver<GenerationControl>,
    paused: &mut bool,
) -> Result<()> {
    loop {
        match controls.try_recv() {
            Ok(GenerationControl::AppendPrompt { text }) => {
                decode_appended_prompt(model, ctx, batch, n_cur, n_ctx, &text)?;
            }
            Ok(GenerationControl::SetPaused { paused: next }) => {
                *paused = next;
            }
            Err(TryRecvError::Empty) => return Ok(()),
            Err(TryRecvError::Disconnected) => return Ok(()),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn wait_while_paused(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    batch: &mut LlamaBatch<'_>,
    n_cur: &mut i32,
    n_ctx: usize,
    controls: &Receiver<GenerationControl>,
    cancel: &AtomicBool,
    paused: &mut bool,
) -> Result<()> {
    while *paused {
        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }
        match controls.recv_timeout(std::time::Duration::from_millis(10)) {
            Ok(GenerationControl::AppendPrompt { text }) => {
                decode_appended_prompt(model, ctx, batch, n_cur, n_ctx, &text)?;
            }
            Ok(GenerationControl::SetPaused { paused: next }) => {
                *paused = next;
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return Ok(()),
        }
    }
    Ok(())
}

fn decode_appended_prompt(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    batch: &mut LlamaBatch<'_>,
    n_cur: &mut i32,
    n_ctx: usize,
    text: &str,
) -> Result<()> {
    let tokens = model
        .str_to_token(text, AddBos::Never)
        .context("failed to tokenize appended prompt")?;
    if tokens.is_empty() {
        return Ok(());
    }

    let required_tokens = (*n_cur as usize)
        .checked_add(tokens.len())
        .context("context token count overflowed usize")?;
    if required_tokens > n_ctx {
        bail!(
            "appended prompt needs {required_tokens} context tokens, but context_size is {n_ctx}"
        );
    }

    decode_prompt_tokens(ctx, batch, &tokens, n_cur, n_ctx, "appended prompt")?;

    Ok(())
}

fn decode_prompt_tokens(
    ctx: &mut LlamaContext<'_>,
    batch: &mut LlamaBatch<'_>,
    tokens: &[llama_cpp_2::token::LlamaToken],
    n_cur: &mut i32,
    n_ctx: usize,
    label: &str,
) -> Result<()> {
    if tokens.is_empty() {
        return Ok(());
    }

    let max_decode_tokens =
        usize::try_from(ctx.n_batch()).context("llama.cpp n_batch does not fit usize")?;
    anyhow::ensure!(
        max_decode_tokens > 0,
        "llama.cpp n_batch must be greater than zero"
    );

    batch.clear();
    let last_index = tokens.len() - 1;
    for chunk_start in (0..tokens.len()).step_by(max_decode_tokens) {
        batch.clear();
        let chunk_end = chunk_start
            .saturating_add(max_decode_tokens)
            .min(tokens.len());
        for (index, token) in tokens[chunk_start..chunk_end].iter().copied().enumerate() {
            let global_index = chunk_start + index;
            let position = (*n_cur)
                .checked_add(
                    i32::try_from(index).context("prompt chunk position exceeds i32::MAX")?,
                )
                .context("prompt token position exceeds i32::MAX")?;
            batch
                .add(token, position, &[0], global_index == last_index)
                .with_context(|| format!("failed to add {label} token to llama.cpp batch"))?;
        }

        ctx.decode(batch)
            .with_context(|| format!("failed to decode {label} with llama.cpp"))?;
        *n_cur = (*n_cur)
            .checked_add(batch.n_tokens())
            .context("prompt token count exceeds i32::MAX")?;
        if (*n_cur as usize) > n_ctx {
            bail!("{label} exceeded context_size {n_ctx} while decoding");
        }
    }

    Ok(())
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

fn checked_total_tokens(
    prompt: &str,
    model: &LlamaModel,
    max_tokens: Option<usize>,
) -> Result<usize> {
    let prompt_tokens = model
        .str_to_token(prompt, AddBos::Always)
        .context("failed to tokenize prompt")?
        .len();
    match max_tokens {
        Some(max_tokens) => {
            if max_tokens == 0 {
                bail!("max_tokens must be greater than zero");
            }
            prompt_tokens
                .checked_add(max_tokens)
                .context("prompt plus max_tokens overflowed usize")
        }
        None => Ok(prompt_tokens),
    }
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
