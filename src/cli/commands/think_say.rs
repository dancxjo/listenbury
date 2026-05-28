use super::llama::build_prompt;
use crate::cli::{PromptMode, model_paths};
use crate::cli::{SayCommand, ThinkSayCommand, ThinkSayMouthOption};
use anyhow::{Context, Result};
use listenbury::LlmEngine;
use listenbury::mind::llm::{GenerationRequest, LlmEvent, MockLlmEngine};
use listenbury::{LlamaCppConfig, LlamaCppEngine};
use serde_json::json;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

const DEFAULT_BREATH_GROUP_CHARS: usize = 120;
const MOCK_LLM_TOKENS: [&str; 3] = ["Hello,", " there.", " I am Listenbury."];

pub(crate) fn run_think_say(command: ThinkSayCommand) -> Result<()> {
    let args = ThinkSayArgs::from_command(command)?;
    let mut trace = TraceWriter::create(args.dump.as_deref())?;

    if args.mock_mouth {
        let mut mouth = MockThinkSayMouth::default();
        let mut runner = ThinkSayRunner::new(&mut mouth, &mut trace, args.chunk_threshold);
        run_llm(&args, &mut runner)
    } else {
        let mut mouth = SayThinkSayMouth { mouth: args.mouth };
        let mut runner = ThinkSayRunner::new(&mut mouth, &mut trace, args.chunk_threshold);
        run_llm(&args, &mut runner)
    }
}

fn run_llm<M: ThinkSayMouth>(
    args: &ThinkSayArgs,
    runner: &mut ThinkSayRunner<'_, M>,
) -> Result<()> {
    runner.trace_event(
        "llm.request_start",
        json!({
            "prompt": args.prompt,
            "mock": args.mock_llm,
            "max_tokens": args.max_tokens,
            "llm_model": args.llm_model.as_ref().map(|path| path.display().to_string()),
            "llm_gpu_layers": args.llm_gpu_layers,
        }),
    )?;

    if args.mock_llm {
        let mut llm = MockLlmEngine::with_response(
            MOCK_LLM_TOKENS
                .iter()
                .map(|token| (*token).to_string())
                .collect(),
        );
        let id = llm.start(GenerationRequest {
            prompt: args.prompt.clone(),
            max_tokens: Some(args.max_tokens),
            stop: Vec::new(),
        })?;

        loop {
            let events = llm.poll(id)?;
            let done = runner.handle_events(&events)?;
            if done {
                break;
            }
        }
        runner.finish()
    } else {
        run_real_llm(args, runner)
    }
}

fn run_real_llm<M: ThinkSayMouth>(
    args: &ThinkSayArgs,
    runner: &mut ThinkSayRunner<'_, M>,
) -> Result<()> {
    let model_path = model_paths::resolve_llm_model(args.llm_model.clone())?;
    let llm_placement = model_paths::llm_runtime_placement(&model_path, args.llm_gpu_layers, None)?;
    let config = LlamaCppConfig {
        model_path,
        gpu_layers: llm_placement.gpu_layers,
        cpu_only: llm_placement.cpu_only,
        ..Default::default()
    };
    let mut llm = LlamaCppEngine::new(config).context("failed to initialize llama.cpp engine")?;
    let (prompt, stop) = build_prompt(PromptMode::Spoken, &args.prompt);
    let id = llm
        .start(GenerationRequest {
            prompt,
            max_tokens: Some(args.max_tokens),
            stop,
        })
        .context("failed to start llama.cpp generation")?;

    loop {
        let events = llm.poll(id)?;
        if events.is_empty() {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        }
        let done = runner.handle_events(&events)?;
        if done {
            break;
        }
    }
    runner.finish()
}

#[derive(Debug)]
struct ThinkSayArgs {
    prompt: String,
    mock_llm: bool,
    mock_mouth: bool,
    mouth: ThinkSayMouthOption,
    llm_model: Option<PathBuf>,
    llm_gpu_layers: Option<u32>,
    max_tokens: usize,
    chunk_threshold: Option<usize>,
    dump: Option<PathBuf>,
}

impl ThinkSayArgs {
    fn from_command(command: ThinkSayCommand) -> Result<Self> {
        let max_tokens =
            usize::try_from(command.max_tokens).context("max_tokens does not fit in usize")?;
        anyhow::ensure!(max_tokens > 0, "max_tokens must be greater than zero");

        let prompt = command.prompt.join(" ");
        anyhow::ensure!(
            !prompt.trim().is_empty(),
            "missing prompt; try `debug think-say \"Say hello.\"`"
        );

        let chunk_threshold = (!command.sentence_by_sentence).then_some(DEFAULT_BREATH_GROUP_CHARS);

        Ok(Self {
            prompt,
            mock_llm: command.mock_llm,
            mock_mouth: command.mock_mouth,
            mouth: command.mouth,
            llm_model: command.llm_model,
            llm_gpu_layers: command.llm_gpu_layers,
            max_tokens,
            chunk_threshold,
            dump: command.dump,
        })
    }
}

struct ThinkSayRunner<'a, M: ThinkSayMouth> {
    chunker: SpeakableChunker,
    mouth: &'a mut M,
    trace: &'a mut TraceWriter,
    saw_first_token: bool,
}

impl<'a, M: ThinkSayMouth> ThinkSayRunner<'a, M> {
    fn new(mouth: &'a mut M, trace: &'a mut TraceWriter, chunk_threshold: Option<usize>) -> Self {
        Self {
            chunker: SpeakableChunker::new(chunk_threshold),
            mouth,
            trace,
            saw_first_token: false,
        }
    }

    fn handle_events(&mut self, events: &[LlmEvent]) -> Result<bool> {
        let mut done = false;
        for event in events {
            match event {
                LlmEvent::Token { text } => {
                    if !self.saw_first_token {
                        self.saw_first_token = true;
                        self.trace_event("llm.first_token", json!({ "text": text }))?;
                    }
                    terminal_event("LLM_TOKEN", text)?;
                    self.trace_event("llm.token", json!({ "text": text }))?;
                    for chunk in self.chunker.push(text) {
                        self.queue_chunk(chunk)?;
                    }
                }
                LlmEvent::Completed | LlmEvent::Cancelled => {
                    done = true;
                    for chunk in self.chunker.finish() {
                        self.queue_chunk(chunk)?;
                    }
                }
                LlmEvent::Error { message } => {
                    self.trace_event("llm.error", json!({ "message": message }))?;
                    anyhow::bail!("LLM generation failed: {message}");
                }
            }
        }
        Ok(done)
    }

    fn finish(&mut self) -> Result<()> {
        self.trace_event("command.done", json!({}))?;
        self.trace.flush()
    }

    fn queue_chunk(&mut self, chunk: String) -> Result<()> {
        terminal_event("TEXT_CHUNK", &chunk)?;
        self.trace_event("text.chunk_ready", json!({ "text": chunk }))?;
        terminal_event("MOUTH_QUEUE", &chunk)?;
        self.trace_event("mouth.queue", json!({ "text": chunk }))?;
        self.mouth.queue(&chunk)?;
        terminal_event("MOUTH_DONE", &chunk)?;
        self.trace_event("mouth.done", json!({ "text": chunk }))?;
        Ok(())
    }

    fn trace_event(&mut self, kind: &str, fields: serde_json::Value) -> Result<()> {
        self.trace.write(kind, fields)
    }
}

trait ThinkSayMouth {
    fn queue(&mut self, text: &str) -> Result<()>;
}

#[derive(Debug, Default)]
struct MockThinkSayMouth {
    chunks: Vec<String>,
}

impl ThinkSayMouth for MockThinkSayMouth {
    fn queue(&mut self, text: &str) -> Result<()> {
        if !text.trim().is_empty() {
            self.chunks.push(text.to_string());
        }
        Ok(())
    }
}

struct SayThinkSayMouth {
    mouth: ThinkSayMouthOption,
}

impl ThinkSayMouth for SayThinkSayMouth {
    fn queue(&mut self, text: &str) -> Result<()> {
        super::run_say(say_command_for_text(text, self.mouth))
    }
}

fn say_command_for_text(text: &str, mouth: ThinkSayMouthOption) -> SayCommand {
    SayCommand {
        piper: matches!(mouth, ThinkSayMouthOption::Piper),
        riper: matches!(mouth, ThinkSayMouthOption::Current),
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: matches!(mouth, ThinkSayMouthOption::Klatt),
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: matches!(mouth, ThinkSayMouthOption::Diphone),
        mbrola_voice: None,
        words: vec![text.to_string()],
    }
}

#[derive(Debug)]
struct SpeakableChunker {
    buffer: String,
    threshold: Option<usize>,
}

impl SpeakableChunker {
    fn new(threshold: Option<usize>) -> Self {
        Self {
            buffer: String::new(),
            threshold,
        }
    }

    fn push(&mut self, token: &str) -> Vec<String> {
        self.buffer.push_str(token);
        let mut chunks = Vec::new();

        loop {
            let Some(end) = self.next_boundary() else {
                break;
            };
            if let Some(chunk) = self.drain_chunk(end) {
                chunks.push(chunk);
            }
        }

        if let Some(end) = self.threshold_boundary()
            && let Some(chunk) = self.drain_chunk(end)
        {
            chunks.push(chunk);
        }

        chunks
    }

    fn finish(&mut self) -> Vec<String> {
        let Some(chunk) = self.drain_chunk(self.buffer.len()) else {
            return Vec::new();
        };
        vec![chunk]
    }

    fn next_boundary(&self) -> Option<usize> {
        self.buffer.char_indices().find_map(|(index, ch)| {
            matches!(ch, '.' | '?' | '!' | '\n').then_some(index + ch.len_utf8())
        })
    }

    fn threshold_boundary(&self) -> Option<usize> {
        let threshold = self.threshold?;
        let leading_len = self.buffer.len() - self.buffer.trim_start().len();
        let rest = &self.buffer[leading_len..];
        if rest.chars().count() < threshold {
            return None;
        }

        let hard_end = nth_char_boundary(rest, threshold).unwrap_or(rest.len());
        let cut = rest[..hard_end]
            .char_indices()
            .rev()
            .find_map(|(index, ch)| ch.is_whitespace().then_some(index))
            .filter(|index| *index > 0)
            .unwrap_or(hard_end);
        Some(leading_len + cut)
    }

    fn drain_chunk(&mut self, end: usize) -> Option<String> {
        let chunk = self.buffer[..end].trim().to_string();
        self.buffer.drain(..end);
        (!chunk.is_empty()).then_some(chunk)
    }
}

fn nth_char_boundary(text: &str, count: usize) -> Option<usize> {
    if count == 0 {
        return Some(0);
    }
    text.char_indices().nth(count).map(|(index, _)| index)
}

struct TraceWriter {
    writer: Option<BufWriter<File>>,
}

impl TraceWriter {
    fn create(path: Option<&Path>) -> Result<Self> {
        let writer = if let Some(path) = path {
            if let Some(parent) = path.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create {}", parent.display()))?;
            }
            Some(BufWriter::new(File::create(path).with_context(|| {
                format!("failed to create {}", path.display())
            })?))
        } else {
            None
        };
        Ok(Self { writer })
    }

    fn write(&mut self, kind: &str, fields: serde_json::Value) -> Result<()> {
        let Some(writer) = self.writer.as_mut() else {
            return Ok(());
        };
        let record = json!({
            "kind": kind,
            "fields": fields,
        });
        serde_json::to_writer(&mut *writer, &record)?;
        writeln!(writer)?;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if let Some(writer) = self.writer.as_mut() {
            writer.flush()?;
        }
        Ok(())
    }
}

fn terminal_event(kind: &str, text: &str) -> Result<()> {
    println!("{kind} {}", text.replace('\n', "\\n"));
    std::io::stdout().flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_mock_tokens(tokens: &[&str], threshold: Option<usize>) -> Vec<String> {
        let mut mouth = MockThinkSayMouth::default();
        let mut trace = TraceWriter { writer: None };
        let mut runner = ThinkSayRunner::new(&mut mouth, &mut trace, threshold);
        for token in tokens {
            runner
                .handle_events(&[LlmEvent::Token {
                    text: (*token).to_string(),
                }])
                .unwrap();
        }
        runner.handle_events(&[LlmEvent::Completed]).unwrap();
        mouth.chunks
    }

    #[test]
    fn mock_llm_one_sentence_queues_one_mouth_chunk() {
        let chunks = run_mock_tokens(&["Hello", " there."], Some(DEFAULT_BREATH_GROUP_CHARS));
        assert_eq!(chunks, ["Hello there."]);
    }

    #[test]
    fn mock_llm_two_sentences_queues_two_mouth_chunks() {
        let chunks = run_mock_tokens(
            &["Hello there. I am Listenbury."],
            Some(DEFAULT_BREATH_GROUP_CHARS),
        );
        assert_eq!(chunks, ["Hello there.", "I am Listenbury."]);
    }

    #[test]
    fn text_without_punctuation_flushes_by_length_threshold() {
        let chunks = run_mock_tokens(
            &["alpha beta gamma delta epsilon zeta"],
            Some("alpha beta gamma".len()),
        );
        assert_eq!(chunks[0], "alpha beta");
    }

    #[test]
    fn empty_and_whitespace_chunks_are_ignored() {
        let chunks = run_mock_tokens(
            &["   ", "\n", "Hello.", "   "],
            Some(DEFAULT_BREATH_GROUP_CHARS),
        );
        assert_eq!(chunks, ["Hello."]);
    }

    #[test]
    fn mock_mouth_records_chunks_without_audio() {
        let mut mouth = MockThinkSayMouth::default();
        mouth.queue("Hello.").unwrap();
        mouth.queue("   ").unwrap();
        mouth.queue("Again.").unwrap();
        assert_eq!(mouth.chunks, ["Hello.", "Again."]);
    }
}
