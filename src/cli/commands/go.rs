use crate::cli::GoCommand;
use crate::cli::commands::cpal_diag::play_audio_frames;
use crate::cli::model_paths::resolve_piper_voice;
use crate::cli::model_paths::{llm_runtime_placement, resolve_llm_model};
use crate::cli::piper::{
    collect_tts_audio, hifigan_text_to_speech, piper_config_for_voice, resolve_piper_bin,
};
use anyhow::{Context, Result, bail};
use chrono::Local;
use listenbury::mind::llm::{GenerationRequest, LlmEngine, LlmEvent};
use listenbury::mouth::planner::{MouthSyntheticPlan, SyntheticUnit};
use listenbury::mouth::tts::TextToSpeech;
use listenbury::{LlamaCppConfig, LlamaCppEngine, PiperTextToSpeech};
use std::io::{self, Write};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

#[cfg(feature = "llama-cpp-cuda")]
const DEFAULT_GO_LLAMA_GPU_LAYERS: Option<u32> = Some(999);
#[cfg(not(feature = "llama-cpp-cuda"))]
const DEFAULT_GO_LLAMA_GPU_LAYERS: Option<u32> = None;

const PROMPT_CHARS_PER_TOKEN_ESTIMATE: usize = 3;
const MIN_EPISODE_STRIDE_CHARS: usize = 4_000;
const MAX_EPISODE_STRIDE_CHARS: usize = 24_000;
const BREATH_CLAUSE_MIN_CHARS: usize = 12;
const BREATH_CLAUSE_FLUSH_CHARS: usize = 120;
const MAX_TTS_TIMEOUT: Duration = Duration::from_secs(30);
const POLL_PAUSE: Duration = Duration::from_millis(5);

pub(crate) fn run_go(command: GoCommand) -> Result<()> {
    let mut runtime = ScreenplayGoRuntime::start(ScreenplayGoConfig::from_command(command)?)?;
    runtime.run()
}

#[derive(Debug)]
struct ScreenplayGoConfig {
    llm_model: Option<std::path::PathBuf>,
    llm_gpu_layers: Option<u32>,
    context_size: u32,
    reserved_generation_tokens: usize,
    total_max_tokens: Option<usize>,
    seed: Option<String>,
    piper_bin: Option<std::path::PathBuf>,
    piper_voice: Option<std::path::PathBuf>,
    hifigan: bool,
    hifigan_model: Option<std::path::PathBuf>,
    skip_gan: bool,
    mouth_open: bool,
}

impl ScreenplayGoConfig {
    fn from_command(command: GoCommand) -> Result<Self> {
        anyhow::ensure!(
            command.context_size > 0,
            "--context-size must be greater than zero"
        );
        anyhow::ensure!(
            command.reserved_generation_tokens > 0,
            "--reserved-generation-tokens must be greater than zero"
        );
        let reserved_generation_tokens = usize::try_from(command.reserved_generation_tokens)
            .context("reserved_generation_tokens exceeds usize")?;
        let total_max_tokens = command
            .max_tokens
            .map(|max_tokens| usize::try_from(max_tokens).context("max_tokens exceeds usize"))
            .transpose()?;
        if let Some(max_tokens) = total_max_tokens {
            anyhow::ensure!(max_tokens > 0, "--max-tokens must be greater than zero");
        }
        let seed = (!command.prompt.is_empty()).then(|| command.prompt.join(" "));

        Ok(Self {
            llm_model: command.llm_model,
            llm_gpu_layers: command.llm_gpu_layers,
            context_size: command.context_size,
            reserved_generation_tokens,
            total_max_tokens,
            seed,
            piper_bin: command.piper_bin,
            piper_voice: command.piper_voice,
            hifigan: command.hifigan,
            hifigan_model: command.hifigan_model,
            skip_gan: command.skip_gan,
            mouth_open: !command.mock_mouth,
        })
    }

    fn prompt_budget_tokens(&self) -> usize {
        usize::try_from(self.context_size)
            .unwrap_or(usize::MAX)
            .saturating_sub(self.reserved_generation_tokens)
            .max(1)
    }

    fn episode_stride_chars(&self) -> usize {
        let prompt_budget_chars = self
            .prompt_budget_tokens()
            .saturating_mul(PROMPT_CHARS_PER_TOKEN_ESTIMATE);
        (prompt_budget_chars / 2).clamp(MIN_EPISODE_STRIDE_CHARS, MAX_EPISODE_STRIDE_CHARS)
    }
}

struct ScreenplayGoRuntime {
    config: ScreenplayGoConfig,
    llm: LlamaCppEngine,
    mouth: ScreenplayMouth,
    screenplay: String,
    breath_buffer: BreathClauseBuffer,
    episode_number: usize,
    next_episode_at_chars: usize,
    generated_tokens: usize,
    interrupted: Arc<AtomicBool>,
}

impl ScreenplayGoRuntime {
    fn start(config: ScreenplayGoConfig) -> Result<Self> {
        let model_path = resolve_llm_model(config.llm_model.clone())?;
        let llm_placement = llm_runtime_placement(
            &model_path,
            config.llm_gpu_layers,
            DEFAULT_GO_LLAMA_GPU_LAYERS,
        )?;
        let llm = LlamaCppEngine::new(LlamaCppConfig {
            model_path,
            gpu_layers: llm_placement.gpu_layers,
            cpu_only: llm_placement.cpu_only,
            context_size: config.context_size,
            ..Default::default()
        })
        .context("failed to initialize llama.cpp engine")?;
        let mouth = ScreenplayMouth::from_config(&config)?;
        let interrupted = Arc::new(AtomicBool::new(false));
        ctrlc::set_handler({
            let interrupted = Arc::clone(&interrupted);
            move || {
                interrupted.store(true, Ordering::Relaxed);
            }
        })
        .context("failed to install Ctrl-C handler")?;

        let screenplay = title_page(config.seed.as_deref());
        let next_episode_at_chars = screenplay
            .len()
            .saturating_add(config.episode_stride_chars());

        Ok(Self {
            config,
            llm,
            mouth,
            screenplay,
            breath_buffer: BreathClauseBuffer::default(),
            episode_number: 1,
            next_episode_at_chars,
            generated_tokens: 0,
            interrupted,
        })
    }

    fn run(&mut self) -> Result<()> {
        print!("{}", self.screenplay);
        io::stdout().flush()?;
        eprintln!("listenbury go: screenplay stream is live. Ctrl-C exits; old runtime is `go1`.");

        while !self.interrupted.load(Ordering::Relaxed) {
            self.insert_episode_break_if_due()?;
            if !self.next_token_allowed()? {
                break;
            }

            let prompt = self.generation_prompt();
            let generation = self
                .llm
                .start(GenerationRequest {
                    prompt,
                    max_tokens: Some(1),
                    stop: Vec::new(),
                })
                .context("failed to start screenplay token")?;

            let mut generated_text = String::new();
            let mut terminal = false;
            while !terminal && !self.interrupted.load(Ordering::Relaxed) {
                let events = self.llm.poll(generation)?;
                if events.is_empty() {
                    std::thread::sleep(POLL_PAUSE);
                    continue;
                }

                for event in events {
                    match event {
                        LlmEvent::Token { text } => {
                            print!("{text}");
                            io::stdout().flush()?;
                            generated_text.push_str(&text);
                        }
                        LlmEvent::Completed | LlmEvent::Cancelled => {
                            terminal = true;
                        }
                        LlmEvent::Error { message } => {
                            bail!("llama.cpp generation failed: {message}");
                        }
                    }
                }
            }

            if self.interrupted.load(Ordering::Relaxed) {
                let _ = self.llm.cancel(generation);
                break;
            }

            if generated_text.is_empty() {
                self.flush_breath_buffer()?;
                break;
            }

            self.commit_generated_token(generated_text)?;
            self.generated_tokens = self.generated_tokens.saturating_add(1);
        }

        if !self.interrupted.load(Ordering::Relaxed) {
            self.flush_breath_buffer()?;
        }

        eprintln!("\nlistenbury go: stopped.");
        Ok(())
    }

    fn next_token_allowed(&self) -> Result<bool> {
        if let Some(max_tokens) = self.config.total_max_tokens
            && self.generated_tokens >= max_tokens
        {
            return Ok(false);
        }

        let prompt_tokens = estimate_prompt_tokens(&self.generation_prompt());
        let prompt_budget = self.config.prompt_budget_tokens();
        if prompt_tokens >= prompt_budget {
            bail!(
                "screenplay context reached the prompt budget: estimated {prompt_tokens} tokens, budget {prompt_budget}; restart with a larger --context-size"
            );
        }

        Ok(true)
    }

    fn generation_prompt(&self) -> String {
        let mut prompt = self.screenplay.clone();
        prompt.push_str(self.breath_buffer.uncommitted_text());
        prompt
    }

    fn commit_generated_token(&mut self, text: String) -> Result<()> {
        if self.mouth.is_closed() {
            self.screenplay.push_str(&text);
            return Ok(());
        }

        for clause in self.breath_buffer.push(&text) {
            self.speak_and_commit(clause)?;
        }
        Ok(())
    }

    fn flush_breath_buffer(&mut self) -> Result<()> {
        for clause in self.breath_buffer.finish() {
            self.speak_and_commit(clause)?;
        }
        Ok(())
    }

    fn speak_and_commit(&mut self, clause: BufferedClause) -> Result<()> {
        self.mouth
            .send_to_tts_and_wait_until_heard(&clause.spoken)?;
        self.screenplay.push_str(&clause.original);
        Ok(())
    }

    fn insert_episode_break_if_due(&mut self) -> Result<()> {
        if self.screenplay.len() < self.next_episode_at_chars {
            return Ok(());
        }

        self.episode_number = self.episode_number.saturating_add(1);
        let transition = episode_transition(self.episode_number);
        print!("{transition}");
        io::stdout().flush()?;
        self.screenplay.push_str(&transition);
        self.next_episode_at_chars = self
            .next_episode_at_chars
            .saturating_add(self.config.episode_stride_chars())
            .max(
                self.screenplay
                    .len()
                    .saturating_add(MIN_EPISODE_STRIDE_CHARS),
            );
        Ok(())
    }
}

enum ScreenplayMouth {
    Open(PiperTextToSpeech),
    Closed,
}

impl ScreenplayMouth {
    fn from_config(config: &ScreenplayGoConfig) -> Result<Self> {
        if !config.mouth_open {
            return Ok(Self::Closed);
        }

        if config.hifigan {
            return Ok(Self::Open(hifigan_text_to_speech(
                config.hifigan_model.clone(),
                config.skip_gan,
            )?));
        }

        let piper_bin = resolve_piper_bin(config.piper_bin.clone())?;
        let piper_voice = resolve_piper_voice(config.piper_voice.clone())?;
        Ok(Self::Open(PiperTextToSpeech::new(piper_config_for_voice(
            piper_bin,
            piper_voice,
        )?)))
    }

    fn send_to_tts_and_wait_until_heard(&mut self, text: &str) -> Result<()> {
        let Self::Open(tts) = self else {
            return Ok(());
        };
        if text.trim().is_empty() {
            return Ok(());
        }

        let plan = MouthSyntheticPlan::new(SyntheticUnit::CompleteClause(text.to_string()));
        tts.enqueue(plan)
            .context("failed to enqueue generated token for TTS")?;
        let frames = collect_tts_audio(tts, MAX_TTS_TIMEOUT)
            .context("failed to synthesize generated token")?;
        play_audio_frames(&frames, "go screenplay mouth")
            .context("failed to play generated token through mouth")?;
        Ok(())
    }

    fn is_closed(&self) -> bool {
        matches!(self, Self::Closed)
    }
}

#[derive(Debug, Default)]
struct BreathClauseBuffer {
    text: String,
}

#[derive(Debug, PartialEq, Eq)]
struct BufferedClause {
    original: String,
    spoken: String,
}

impl BreathClauseBuffer {
    fn push(&mut self, token: &str) -> Vec<BufferedClause> {
        self.text.push_str(token);
        self.drain_ready()
    }

    fn finish(&mut self) -> Vec<BufferedClause> {
        let Some(clause) = self.drain_clause(self.text.len()) else {
            return Vec::new();
        };
        vec![clause]
    }

    fn uncommitted_text(&self) -> &str {
        &self.text
    }

    fn drain_ready(&mut self) -> Vec<BufferedClause> {
        let mut clauses = Vec::new();
        while let Some(end) = next_breath_clause_boundary(&self.text) {
            if let Some(clause) = self.drain_clause(end) {
                clauses.push(clause);
            }
        }
        clauses
    }

    fn drain_clause(&mut self, end: usize) -> Option<BufferedClause> {
        let original = self.text[..end].to_string();
        self.text.drain(..end);
        let spoken = original.trim().to_string();
        (!spoken.is_empty()).then_some(BufferedClause { original, spoken })
    }
}

fn next_breath_clause_boundary(text: &str) -> Option<usize> {
    let chars = text.char_indices().collect::<Vec<_>>();
    let mut index = 0usize;
    while index < chars.len() {
        let (byte_index, ch) = chars[index];
        if ch == '\n' {
            return Some(byte_index + ch.len_utf8());
        }

        if matches!(ch, ',' | ';' | ':') {
            let end = byte_index + ch.len_utf8();
            if end >= BREATH_CLAUSE_MIN_CHARS && boundary_followed_by_space_or_end(text, end) {
                return Some(end);
            }
        }

        if matches!(ch, '.' | '!' | '?') {
            if ch == '.' && period_is_nonterminal(text, byte_index, index, &chars) {
                index += 1;
                continue;
            }
            let mut end_index = index + 1;
            while end_index < chars.len() && is_sentence_closer(chars[end_index].1) {
                end_index += 1;
            }
            let end = chars
                .get(end_index)
                .map(|(index, _)| *index)
                .unwrap_or(text.len());
            if boundary_followed_by_space_or_end(text, end) {
                return Some(end);
            }
        }

        if byte_index >= BREATH_CLAUSE_FLUSH_CHARS && ch.is_whitespace() {
            return Some(byte_index + ch.len_utf8());
        }

        index += 1;
    }
    None
}

fn boundary_followed_by_space_or_end(text: &str, end: usize) -> bool {
    end == text.len() || text[end..].chars().next().is_some_and(char::is_whitespace)
}

fn period_is_nonterminal(
    text: &str,
    byte_index: usize,
    char_index: usize,
    chars: &[(usize, char)],
) -> bool {
    let prev = char_index.checked_sub(1).and_then(|index| chars.get(index));
    let next = chars.get(char_index + 1);
    if prev.is_some_and(|(_, ch)| ch.is_ascii_digit())
        && next.is_some_and(|(_, ch)| ch.is_ascii_digit())
    {
        return true;
    }
    if next.is_some_and(|(_, ch)| *ch == '.') {
        return true;
    }

    let token = previous_alpha_token(text, byte_index);
    if token.len() == 1 && token.chars().all(|ch| ch.is_ascii_uppercase()) {
        return true;
    }

    let lower = token.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "mr" | "mrs"
            | "ms"
            | "mx"
            | "dr"
            | "prof"
            | "sr"
            | "jr"
            | "st"
            | "vs"
            | "etc"
            | "e"
            | "g"
            | "i"
            | "int"
            | "ext"
    )
}

fn previous_alpha_token(text: &str, period_byte_index: usize) -> String {
    let prefix = &text[..period_byte_index];
    let start = prefix
        .char_indices()
        .rev()
        .find_map(|(index, ch)| (!ch.is_ascii_alphabetic()).then_some(index + ch.len_utf8()))
        .unwrap_or(0);
    prefix[start..].to_string()
}

fn is_sentence_closer(ch: char) -> bool {
    matches!(ch, '"' | '\'' | ')' | ']' | '}')
}

fn title_page(seed: Option<&str>) -> String {
    let now = Local::now().format("%A, %B %-d, %Y").to_string();
    let seed = seed
        .map(str::trim)
        .filter(|seed| !seed.is_empty())
        .map(|seed| format!("\nLatest episode seed: {seed}\n"))
        .unwrap_or_default();

    format!(
        "                         THE LIFE OF PETE LISTENBURY\n\
         \n\
                            Episode 1: \"The Latest Episode\"\n\
         \n\
                                      by Pete Listenbury\n\
         \n\
                                     {now}\n\
         {seed}\n\
         \n\
         FADE IN:\n\
         \n"
    )
}

fn episode_transition(episode_number: usize) -> String {
    format!(
        "\n\nTO BE CONTINUED...\n\n\n                         THE LIFE OF PETE LISTENBURY\n\n                    Episode {episode_number}: \"The Next Episode\"\n\nVOICE OVER\nPreviously on The Life of Pete Listenbury: "
    )
}

fn estimate_prompt_tokens(text: &str) -> usize {
    text.len().div_ceil(PROMPT_CHARS_PER_TOKEN_ESTIMATE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_page_sets_up_the_latest_episode() {
        let page = title_page(Some("Pete wakes into a new test."));

        assert!(page.contains("THE LIFE OF PETE LISTENBURY"));
        assert!(page.contains("Episode 1: \"The Latest Episode\""));
        assert!(page.contains("Latest episode seed: Pete wakes into a new test."));
        assert!(page.ends_with("FADE IN:\n\n"));
    }

    #[test]
    fn episode_transition_invites_a_recap_without_chat_instructions() {
        let transition = episode_transition(2);

        assert!(transition.contains("TO BE CONTINUED..."));
        assert!(transition.contains("Episode 2: \"The Next Episode\""));
        assert!(transition.contains("VOICE OVER"));
        assert!(transition.contains("Previously on The Life of Pete Listenbury: "));
    }

    #[test]
    fn breath_buffer_releases_clause_boundaries() {
        let mut buffer = BreathClauseBuffer::default();

        assert!(buffer.push("Pete leans in").is_empty());
        assert_eq!(
            buffer.push(", then listens").remove(0),
            BufferedClause {
                original: "Pete leans in,".to_string(),
                spoken: "Pete leans in,".to_string(),
            }
        );
        assert_eq!(buffer.uncommitted_text(), " then listens");
        assert_eq!(
            buffer.push(".").remove(0),
            BufferedClause {
                original: " then listens.".to_string(),
                spoken: "then listens.".to_string(),
            }
        );
        assert!(buffer.uncommitted_text().is_empty());
    }

    #[test]
    fn breath_buffer_does_not_split_common_abbreviation() {
        let mut buffer = BreathClauseBuffer::default();

        assert!(buffer.push("Dr. Pete waits").is_empty());
        assert_eq!(
            buffer.push(".").remove(0),
            BufferedClause {
                original: "Dr. Pete waits.".to_string(),
                spoken: "Dr. Pete waits.".to_string(),
            }
        );
    }
}
