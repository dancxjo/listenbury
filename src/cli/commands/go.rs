use crate::cli::GoCommand;
use anyhow::Result;

use crate::cli::commands::cpal_diag::play_audio_frames;
use crate::cli::model_paths::{llm_runtime_placement, resolve_llm_model, resolve_piper_voice};
use crate::cli::piper::{
    collect_tts_audio, hifigan_text_to_speech, piper_config_for_voice, resolve_piper_bin,
};
use anyhow::Context;
use chrono::{Local, SecondsFormat};
use crossbeam_channel::{Receiver, Sender};
use listenbury::mind::llm::{GenerationRequest, LlmEngine, LlmEvent};
use listenbury::mouth::planner::{MouthSyntheticPlan, SyntheticUnit, strip_emoji};
use listenbury::mouth::tts::TextToSpeech;
use listenbury::{GenerationId, LlamaCppConfig, LlamaCppEngine, PiperTextToSpeech};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::io::{BufRead, Write};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tsrun::{
    Guarded, InternalModule, Interpreter, InterpreterConfig, JsError, JsValue, StepResult, api,
    js_value_to_json,
};

#[cfg(feature = "llama-cpp-cuda")]
const DEFAULT_GO_LLAMA_GPU_LAYERS: Option<u32> = Some(999);
#[cfg(not(feature = "llama-cpp-cuda"))]
const DEFAULT_GO_LLAMA_GPU_LAYERS: Option<u32> = None;

const DEFAULT_STREAM_SEED: &str = "You are Pete Listenbury. Run as one continuous stream of consciousness. Be autonomous, curious, friendly, and sociable. Keep observing the timeline, forming private thoughts, and choosing small actions when they are useful. Plain generated text is Pete's visible thought timeline, not speech. To speak or act, emit a <ts>...</ts> TypeScript block. Prefer short, socially graceful speech and leave room for others. If nothing needs saying, keep thinking quietly and explore or notice the situation.";

const TYPESCRIPT_START: &str = "<ts>";

const TYPESCRIPT_END: &str = "</ts>";

const MAX_TTS_TIMEOUT: Duration = Duration::from_secs(30);

const ANSI_RESET: &str = "\x1b[0m";
const ANSI_DIM: &str = "\x1b[2m";
const ANSI_PROMPT: &str = "\x1b[36m";
const ANSI_PROMPT_DELTA: &str = "\x1b[34m";
const ANSI_LLM: &str = "\x1b[32m";
const ANSI_TIMELINE: &str = "\x1b[33m";
const ANSI_ACTION: &str = "\x1b[35m";
const ANSI_ERROR: &str = "\x1b[31m";

pub(crate) fn run_go(command: GoCommand) -> Result<()> {
    let config = GoConfig::from_command(command)?;
    let mut stream = StreamOfConsciousness::start(config)?;
    stream.run()
}

#[derive(Debug)]
struct GoConfig {
    llm_model: Option<std::path::PathBuf>,
    llm_gpu_layers: Option<u32>,
    piper_bin: Option<std::path::PathBuf>,
    piper_voice: Option<std::path::PathBuf>,
    hifigan: bool,
    hifigan_model: Option<std::path::PathBuf>,
    skip_gan: bool,
    max_tokens: Option<usize>,
    context_size: u32,
    reserved_generation_tokens: usize,
    memory_events: usize,
    lookahead_tokens: usize,
    lookahead_chars: usize,
    require_self_hearing: bool,
    mock_mouth: bool,
    prompt: String,
}

impl GoConfig {
    fn from_command(command: GoCommand) -> Result<Self> {
        anyhow::ensure!(
            command.context_size > 0,
            "--context-size must be greater than zero"
        );
        anyhow::ensure!(
            command.reserved_generation_tokens > 0,
            "--reserved-generation-tokens must be greater than zero"
        );
        anyhow::ensure!(
            command.lookahead_tokens > 0,
            "--lookahead-tokens must be greater than zero"
        );
        anyhow::ensure!(
            command.lookahead_chars > 0,
            "--lookahead-chars must be greater than zero"
        );
        let max_tokens = command
            .max_tokens
            .map(|max_tokens| usize::try_from(max_tokens).context("max_tokens exceeds usize"))
            .transpose()?;
        if let Some(max_tokens) = max_tokens {
            anyhow::ensure!(max_tokens > 0, "--max-tokens must be greater than zero");
        }
        let reserved_generation_tokens = usize::try_from(command.reserved_generation_tokens)
            .context("reserved_generation_tokens exceeds usize")?;
        let prompt = if command.prompt.is_empty() {
            DEFAULT_STREAM_SEED.to_string()
        } else {
            format!(
                "{}\n\nInitial live seed: {}",
                DEFAULT_STREAM_SEED,
                command.prompt.join(" ")
            )
        };

        Ok(Self {
            llm_model: command.llm_model,
            llm_gpu_layers: command.llm_gpu_layers,
            piper_bin: command.piper_bin,
            piper_voice: command.piper_voice,
            hifigan: command.hifigan,
            hifigan_model: command.hifigan_model,
            skip_gan: command.skip_gan,
            max_tokens,
            context_size: command.context_size,
            reserved_generation_tokens,
            memory_events: command.memory_events,
            lookahead_tokens: command.lookahead_tokens,
            lookahead_chars: command.lookahead_chars,
            require_self_hearing: command.require_self_hearing,
            mock_mouth: command.mock_mouth,
            prompt,
        })
    }
}

#[derive(Debug, Clone)]
enum StreamObservation {
    UserText(String),
    Thought(String),
    Action(String),
    ActionResult(String),
    MouthStarted(String),
    MouthReturned(String),
    MouthError(String),
    ContextCompacted { retained_events: usize },
    Clock(String),
}

impl StreamObservation {
    fn prompt_text(&self) -> String {
        match self {
            Self::UserText(text) => format!(
                "\n<live_observation source=\"user\">{}</live_observation>\n",
                compact_line(text, 1_200)
            ),
            Self::Thought(text) => format!(
                "\n<live_observation source=\"pete_thought\">{}</live_observation>\n",
                compact_line(text, 800)
            ),
            Self::Action(text) => format!(
                "\n<live_observation source=\"pete_action\">{}</live_observation>\n",
                compact_line(text, 800)
            ),
            Self::ActionResult(text) => format!(
                "\n<live_observation source=\"action_result\">{}</live_observation>\n",
                compact_line(text, 1_000)
            ),
            Self::MouthStarted(text) => format!(
                "\n<live_observation source=\"mouth\">Started speaking: {}</live_observation>\n",
                compact_line(text, 400)
            ),
            Self::MouthReturned(text) => format!(
                "\n<live_observation source=\"ear\">Self-heard syllable/speech returned: {}</live_observation>\n",
                compact_line(text, 400)
            ),
            Self::MouthError(message) => format!(
                "\n<live_observation source=\"mouth_error\">{}</live_observation>\n",
                compact_line(message, 800)
            ),
            Self::ContextCompacted { retained_events } => format!(
                "\n<live_observation source=\"runtime\">Stream context compacted; retained {retained_events} recent event(s).</live_observation>\n"
            ),
            Self::Clock(message) => {
                format!("\n<live_observation source=\"clock\">{message}</live_observation>\n")
            }
        }
    }

    fn memory_text(&self) -> String {
        match self {
            Self::UserText(text) => format!("User: {}", compact_line(text, 400)),
            Self::Thought(text) => format!("Pete thought: {}", compact_line(text, 360)),
            Self::Action(text) => format!("Pete action: {}", compact_line(text, 360)),
            Self::ActionResult(text) => {
                format!("Action result: {}", compact_line(text, 360))
            }
            Self::MouthStarted(text) => format!("Mouth started: {}", compact_line(text, 240)),
            Self::MouthReturned(text) => format!("Self-heard return: {}", compact_line(text, 240)),
            Self::MouthError(message) => format!("Mouth error: {}", compact_line(message, 240)),
            Self::ContextCompacted { retained_events } => {
                format!("Runtime compacted stream context retaining {retained_events} events")
            }
            Self::Clock(message) => format!("Clock: {message}"),
        }
    }
}

#[derive(Debug)]
struct StreamOfConsciousness {
    config: GoConfig,
    llm: LlamaCppEngine,
    generation: GenerationId,
    generated_estimated_tokens: usize,
    loaded_estimated_tokens: usize,
    recent_events: VecDeque<String>,
    output_parser: StreamOutputParser,
    pacer: MouthEarPacer,
    mouth: MouthRuntime,
    stdin_rx: Receiver<std::result::Result<String, String>>,
    mouth_rx: Receiver<MouthEvent>,
    _mouth_worker: Option<JoinHandle<()>>,
    interrupted: Arc<AtomicBool>,
    next_clock_at: Instant,
    startup_context: String,
    timeline_index: u64,
}

impl StreamOfConsciousness {
    fn start(config: GoConfig) -> Result<Self> {
        let model_path = resolve_llm_model(config.llm_model.clone())?;
        let llm_placement = llm_runtime_placement(
            &model_path,
            config.llm_gpu_layers,
            DEFAULT_GO_LLAMA_GPU_LAYERS,
        )?;
        let mut llm = LlamaCppEngine::new(LlamaCppConfig {
            model_path,
            gpu_layers: llm_placement.gpu_layers,
            cpu_only: llm_placement.cpu_only,
            context_size: config.context_size,
            ..Default::default()
        })
        .context("failed to initialize llama.cpp engine")?;
        let startup_context = gather_startup_context();
        let prompt = initial_stream_prompt(&config.prompt, &startup_context);
        print_debug_block("initial prompt", ANSI_PROMPT, &prompt);
        let generation = llm
            .start(GenerationRequest {
                prompt: prompt.clone(),
                max_tokens: config.max_tokens,
                stop: Vec::new(),
            })
            .context("failed to start stream of consciousness")?;
        let (mouth, mouth_rx, worker) = MouthRuntime::start(&config)?;
        let stdin_rx = spawn_stdin_reader()?;
        let interrupted = Arc::new(AtomicBool::new(false));
        ctrlc::set_handler({
            let interrupted = Arc::clone(&interrupted);
            move || {
                interrupted.store(true, Ordering::Relaxed);
            }
        })
        .context("failed to install Ctrl-C handler")?;

        eprintln!(
            "listenbury go: continuous generation is live. Type lines to feed Pete; Ctrl-C exits."
        );

        Ok(Self {
            generated_estimated_tokens: 0,
            loaded_estimated_tokens: estimate_tokens(&prompt),
            recent_events: VecDeque::new(),
            output_parser: StreamOutputParser::new(config.lookahead_chars),
            pacer: MouthEarPacer::new(MouthEarPacerConfig {
                lookahead_tokens: config.lookahead_tokens,
                require_self_hearing: config.require_self_hearing,
            }),
            config,
            llm,
            generation,
            mouth,
            stdin_rx,
            mouth_rx,
            _mouth_worker: worker,
            interrupted,
            next_clock_at: Instant::now() + Duration::from_secs(1),
            startup_context,
            timeline_index: 0,
        })
    }

    fn run(&mut self) -> Result<()> {
        let mut cancelled = false;
        loop {
            if self.interrupted.load(Ordering::Relaxed) && !cancelled {
                self.llm.cancel(self.generation)?;
                self.mouth.shutdown();
                cancelled = true;
            }

            self.drain_stdin()?;
            self.drain_mouth()?;
            self.append_clock_if_due()?;

            if self.pacer.can_generate() && !cancelled {
                self.llm
                    .set_paused(self.generation, false)
                    .context("failed to resume stream generation")?;
            } else {
                self.llm
                    .set_paused(self.generation, true)
                    .context("failed to pace stream generation")?;
            }

            let events = self.llm.poll(self.generation)?;
            if events.is_empty() {
                thread::sleep(Duration::from_millis(5));
                continue;
            }

            let terminal = events.iter().any(is_terminal_event);
            for event in events {
                match event {
                    LlmEvent::Token { text } => self.ingest_token(&text)?,
                    LlmEvent::Error { message } => anyhow::bail!("go generation failed: {message}"),
                    LlmEvent::Completed | LlmEvent::Cancelled => {}
                }
            }

            if terminal {
                if cancelled {
                    println!();
                    break;
                }
                self.restart_generation()?;
            }
        }

        Ok(())
    }

    fn ingest_token(&mut self, text: &str) -> Result<()> {
        print!("{ANSI_LLM}{text}{ANSI_RESET}");
        std::io::stdout().flush()?;
        self.generated_estimated_tokens = self
            .generated_estimated_tokens
            .saturating_add(estimate_tokens(text));
        self.pacer.record_token();

        let parsed = self.output_parser.push(text);
        for output in parsed.outputs {
            self.handle_output(output)?;
        }

        if self.should_compact() {
            self.append_observation(StreamObservation::ContextCompacted {
                retained_events: self.recent_events.len(),
            })?;
            self.restart_generation()?;
        }

        Ok(())
    }

    fn handle_output(&mut self, output: StreamOutput) -> Result<()> {
        match output {
            StreamOutput::Thought(text) => {
                let text = compact_line(&text, 1_200);
                if text.is_empty() {
                    return Ok(());
                }
                self.timeline("thought", &text);
                self.append_observation(StreamObservation::Thought(text))
            }
            StreamOutput::TypeScript(source) => {
                self.timeline("action", &source);
                self.append_observation(StreamObservation::Action(source.clone()))?;
                match execute_typescript_actions(&source) {
                    Ok(actions) => self.apply_actions(actions),
                    Err(error) => {
                        let message = format!("TypeScript failed: {error:#}");
                        self.timeline_colored("action_error", &message, ANSI_ERROR);
                        self.append_observation(StreamObservation::ActionResult(message))
                    }
                }
            }
        }
    }

    fn apply_actions(&mut self, actions: Vec<TypeScriptAction>) -> Result<()> {
        if actions.is_empty() {
            self.timeline("action_result", "TypeScript returned no actions.");
            return self.append_observation(StreamObservation::ActionResult(
                "TypeScript returned no actions.".to_string(),
            ));
        }

        for action in actions {
            match action {
                TypeScriptAction::Say { text } => {
                    self.timeline("speech", &text);
                    self.append_observation(StreamObservation::ActionResult(format!(
                        "Queued speech: {}",
                        compact_line(&text, 300)
                    )))?;
                    for unit in split_speakable_units(&text, self.config.lookahead_chars) {
                        self.enqueue_speech(unit)?;
                    }
                }
                TypeScriptAction::Think { text } => {
                    self.timeline("thought", &text);
                    self.append_observation(StreamObservation::Thought(text))?;
                }
                TypeScriptAction::Note { text } => {
                    self.timeline("note", &text);
                    self.append_observation(StreamObservation::ActionResult(format!(
                        "Noted: {}",
                        compact_line(&text, 500)
                    )))?;
                }
                TypeScriptAction::SetStage { text } => {
                    self.timeline("stage", &text);
                    self.append_observation(StreamObservation::ActionResult(format!(
                        "Stage set: {}",
                        compact_line(&text, 500)
                    )))?;
                }
                TypeScriptAction::Sleeping { reason } => {
                    let message = reason
                        .map(|reason| format!("Sleep requested: {reason}"))
                        .unwrap_or_else(|| "Sleep requested.".to_string());
                    self.timeline("sleeping", &message);
                    self.append_observation(StreamObservation::ActionResult(message))?;
                    self.interrupted.store(true, Ordering::Relaxed);
                }
            }
        }
        Ok(())
    }

    fn enqueue_speech(&mut self, text: String) -> Result<()> {
        let text = clean_spoken_text(&text);
        if text.is_empty() {
            return Ok(());
        }
        self.pacer.record_mouth_unit_queued();
        self.mouth.speak(text)
    }

    fn drain_stdin(&mut self) -> Result<()> {
        for event in self.stdin_rx.try_iter().collect::<Vec<_>>() {
            match event {
                Ok(text) => {
                    let trimmed = text.trim();
                    if !trimmed.is_empty() {
                        self.append_observation(StreamObservation::UserText(trimmed.to_string()))?;
                    }
                }
                Err(message) => anyhow::bail!("failed to read stdin: {message}"),
            }
        }
        Ok(())
    }

    fn drain_mouth(&mut self) -> Result<()> {
        for event in self.mouth_rx.try_iter().collect::<Vec<_>>() {
            match event {
                MouthEvent::Started { text } => {
                    self.pacer.record_mouth_started();
                    self.append_observation(StreamObservation::MouthStarted(text))?;
                }
                MouthEvent::Returned { text } => {
                    self.pacer.record_self_heard();
                    self.append_observation(StreamObservation::MouthReturned(text))?;
                }
                MouthEvent::Error { message } => {
                    self.pacer.record_self_heard();
                    self.append_observation(StreamObservation::MouthError(message))?;
                }
            }
        }
        Ok(())
    }

    fn append_clock_if_due(&mut self) -> Result<()> {
        let now = Instant::now();
        if now < self.next_clock_at {
            return Ok(());
        }
        self.next_clock_at = now + Duration::from_secs(1);
        self.append_observation(StreamObservation::Clock(current_time_context()))
    }

    fn append_observation(&mut self, observation: StreamObservation) -> Result<()> {
        let prompt_text = observation.prompt_text();
        self.remember_event(observation.memory_text());
        print_debug_block("prompt delta", ANSI_PROMPT_DELTA, &prompt_text);
        if self.should_restart_before_append(&prompt_text) {
            self.restart_generation()?;
        }
        self.loaded_estimated_tokens = self
            .loaded_estimated_tokens
            .saturating_add(estimate_tokens(&prompt_text));
        self.llm
            .append_prompt(self.generation, prompt_text)
            .context("failed to append observation to stream")
    }

    fn remember_event(&mut self, event: String) {
        self.recent_events.push_back(event);
        while self.recent_events.len() > self.config.memory_events {
            self.recent_events.pop_front();
        }
    }

    fn timeline(&mut self, kind: &str, text: &str) {
        self.timeline_colored(kind, text, timeline_color(kind));
    }

    fn timeline_colored(&mut self, kind: &str, text: &str, color: &str) {
        self.timeline_index = self.timeline_index.saturating_add(1);
        let time = Local::now().to_rfc3339_opts(SecondsFormat::Secs, false);
        let line = format!(
            "[timeline #{:04} {time} {kind}] {}",
            self.timeline_index,
            compact_line(text, 1_000)
        );
        println!("{color}{line}{ANSI_RESET}");
        self.remember_event(line);
    }

    fn should_restart_before_append(&self, next_text: &str) -> bool {
        let budget = self.context_budget_tokens();
        self.loaded_estimated_tokens
            .saturating_add(self.generated_estimated_tokens)
            .saturating_add(estimate_tokens(next_text))
            >= budget
    }

    fn should_compact(&self) -> bool {
        self.loaded_estimated_tokens
            .saturating_add(self.generated_estimated_tokens)
            >= self.context_budget_tokens()
    }

    fn context_budget_tokens(&self) -> usize {
        (self.config.context_size as usize)
            .saturating_sub(self.config.reserved_generation_tokens)
            .max(1)
    }

    fn restart_generation(&mut self) -> Result<()> {
        self.cancel_current_generation()?;
        let prompt = compact_stream_prompt(
            &self.config.prompt,
            &self.startup_context,
            &self.recent_events,
        );
        print_debug_block("compacted prompt", ANSI_PROMPT, &prompt);
        self.generation = self
            .llm
            .start(GenerationRequest {
                prompt: prompt.clone(),
                max_tokens: self.config.max_tokens,
                stop: Vec::new(),
            })
            .context("failed to restart compacted stream")?;
        self.loaded_estimated_tokens = estimate_tokens(&prompt);
        self.generated_estimated_tokens = 0;
        Ok(())
    }

    fn cancel_current_generation(&mut self) -> Result<()> {
        self.llm.cancel(self.generation)?;
        loop {
            let events = self.llm.poll(self.generation)?;
            if events.iter().any(is_terminal_event) {
                return Ok(());
            }
            thread::sleep(Duration::from_millis(2));
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MouthEarPacerConfig {
    lookahead_tokens: usize,
    require_self_hearing: bool,
}

#[derive(Debug)]
struct MouthEarPacer {
    config: MouthEarPacerConfig,
    generated_since_return: usize,
    pending_mouth_units: usize,
}

impl MouthEarPacer {
    fn new(config: MouthEarPacerConfig) -> Self {
        Self {
            config,
            generated_since_return: 0,
            pending_mouth_units: 0,
        }
    }

    fn can_generate(&self) -> bool {
        if self.generated_since_return < self.config.lookahead_tokens {
            return true;
        }
        !self.config.require_self_hearing || self.pending_mouth_units == 0
    }

    fn record_token(&mut self) {
        self.generated_since_return = self.generated_since_return.saturating_add(1);
    }

    fn record_mouth_unit_queued(&mut self) {
        self.pending_mouth_units = self.pending_mouth_units.saturating_add(1);
    }

    fn record_mouth_started(&mut self) {}

    fn record_self_heard(&mut self) {
        self.pending_mouth_units = self.pending_mouth_units.saturating_sub(1);
        self.generated_since_return = 0;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum StreamOutput {
    Thought(String),
    TypeScript(String),
}

#[derive(Debug, Default)]
struct ParsedStreamOutput {
    outputs: Vec<StreamOutput>,
}

#[derive(Debug)]
struct StreamOutputParser {
    text: String,
    flush_chars: usize,
}

impl StreamOutputParser {
    fn new(flush_chars: usize) -> Self {
        Self {
            text: String::new(),
            flush_chars,
        }
    }

    fn push(&mut self, token: &str) -> ParsedStreamOutput {
        self.text.push_str(token);
        let mut outputs = Vec::new();
        loop {
            if let Some(start) = self.text.find(TYPESCRIPT_START) {
                if start > 0 {
                    let thought = self.text[..start].to_string();
                    self.text = self.text[start..].to_string();
                    if !thought.trim().is_empty() {
                        outputs.push(StreamOutput::Thought(thought));
                    }
                    continue;
                }

                let content_start = TYPESCRIPT_START.len();
                let Some(end_rel) = self.text[content_start..].find(TYPESCRIPT_END) else {
                    break;
                };
                let end = content_start + end_rel;
                let source = self.text[content_start..end].trim().to_string();
                let after = end + TYPESCRIPT_END.len();
                self.text = self.text[after..].to_string();
                if !source.is_empty() {
                    outputs.push(StreamOutput::TypeScript(source));
                }
                continue;
            }

            let Some(boundary) = next_speakable_boundary(&self.text, self.flush_chars) else {
                break;
            };
            let thought = self.text[..boundary].to_string();
            self.text = self.text[boundary..].to_string();
            if !thought.trim().is_empty() {
                outputs.push(StreamOutput::Thought(thought));
            }
        }
        ParsedStreamOutput { outputs }
    }
}

#[derive(Debug)]
struct SpeakableBuffer {
    text: String,
    flush_chars: usize,
}

impl SpeakableBuffer {
    fn new(flush_chars: usize) -> Self {
        Self {
            text: String::new(),
            flush_chars,
        }
    }

    fn push(&mut self, token: &str) -> Vec<String> {
        self.text.push_str(token);
        let mut units = Vec::new();
        loop {
            let Some(boundary) = next_speakable_boundary(&self.text, self.flush_chars) else {
                break;
            };
            let unit = self.text[..boundary].trim().to_string();
            self.text = self.text[boundary..].to_string();
            if !unit.is_empty() {
                units.push(unit);
            }
        }
        units
    }
}

fn split_speakable_units(text: &str, flush_chars: usize) -> Vec<String> {
    let mut buffer = SpeakableBuffer::new(flush_chars);
    let mut units = buffer.push(text);
    let trailing = buffer.text.trim();
    if !trailing.is_empty() {
        units.push(trailing.to_string());
    }
    units
}

#[derive(Debug)]
enum MouthRuntime {
    Mock(Sender<MouthCommand>),
    Real(Sender<MouthCommand>),
}

impl MouthRuntime {
    fn start(config: &GoConfig) -> Result<(Self, Receiver<MouthEvent>, Option<JoinHandle<()>>)> {
        let (command_tx, command_rx) = crossbeam_channel::unbounded();
        let (event_tx, event_rx) = crossbeam_channel::unbounded();
        if config.mock_mouth {
            let handle = thread::Builder::new()
                .name("listenbury-go-mock-mouth".to_string())
                .spawn(move || run_mock_mouth(command_rx, event_tx))
                .context("failed to spawn mock mouth")?;
            return Ok((Self::Mock(command_tx), event_rx, Some(handle)));
        }

        let tts = go_tts_for_config(config)?;
        let handle = thread::Builder::new()
            .name("listenbury-go-mouth".to_string())
            .spawn(move || run_mouth(tts, command_rx, event_tx))
            .context("failed to spawn go mouth")?;
        Ok((Self::Real(command_tx), event_rx, Some(handle)))
    }

    fn speak(&self, text: String) -> Result<()> {
        self.tx()
            .send(MouthCommand::Speak { text })
            .context("failed to queue speech for mouth")
    }

    fn shutdown(&self) {
        let _ = self.tx().send(MouthCommand::Shutdown);
    }

    fn tx(&self) -> &Sender<MouthCommand> {
        match self {
            Self::Mock(tx) | Self::Real(tx) => tx,
        }
    }
}

#[derive(Debug)]
enum MouthCommand {
    Speak { text: String },
    Shutdown,
}

#[derive(Debug)]
enum MouthEvent {
    Started { text: String },
    Returned { text: String },
    Error { message: String },
}

fn run_mock_mouth(command_rx: Receiver<MouthCommand>, event_tx: Sender<MouthEvent>) {
    for command in command_rx {
        match command {
            MouthCommand::Speak { text } => {
                let _ = event_tx.send(MouthEvent::Started { text: text.clone() });
                thread::sleep(Duration::from_millis(20));
                let _ = event_tx.send(MouthEvent::Returned { text });
            }
            MouthCommand::Shutdown => return,
        }
    }
}

fn run_mouth(
    mut tts: PiperTextToSpeech,
    command_rx: Receiver<MouthCommand>,
    event_tx: Sender<MouthEvent>,
) {
    for command in command_rx {
        match command {
            MouthCommand::Speak { text } => {
                let _ = event_tx.send(MouthEvent::Started { text: text.clone() });
                let plan = MouthSyntheticPlan::new(SyntheticUnit::CompleteClause(text.clone()));
                if let Err(error) = tts.enqueue(plan) {
                    let _ = event_tx.send(MouthEvent::Error {
                        message: error.to_string(),
                    });
                    continue;
                }
                let result = collect_tts_audio(&mut tts, MAX_TTS_TIMEOUT)
                    .and_then(|frames| play_audio_frames(&frames, "go mouth"));
                match result {
                    Ok(()) => {
                        let _ = event_tx.send(MouthEvent::Returned { text });
                    }
                    Err(error) => {
                        let _ = event_tx.send(MouthEvent::Error {
                            message: error.to_string(),
                        });
                    }
                }
            }
            MouthCommand::Shutdown => return,
        }
    }
}

fn go_tts_for_config(config: &GoConfig) -> Result<PiperTextToSpeech> {
    if config.hifigan {
        return hifigan_text_to_speech(config.hifigan_model.clone(), config.skip_gan);
    }

    let piper_bin = resolve_piper_bin(config.piper_bin.clone())?;
    let piper_voice = resolve_piper_voice(config.piper_voice.clone())?;
    Ok(PiperTextToSpeech::new(piper_config_for_voice(
        piper_bin,
        piper_voice,
    )?))
}

fn spawn_stdin_reader() -> Result<Receiver<std::result::Result<String, String>>> {
    let (tx, rx) = crossbeam_channel::unbounded();
    thread::Builder::new()
        .name("listenbury-go-stdin".to_string())
        .spawn(move || {
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if tx.send(Ok(line)).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = tx.send(Err(error.to_string()));
                        break;
                    }
                }
            }
        })
        .context("failed to spawn stdin reader")?;
    Ok(rx)
}

fn gather_startup_context() -> String {
    let mut lines = Vec::new();
    lines.push(current_time_context());
    if let Some(value) = env_value("USER").or_else(|| env_value("LOGNAME")) {
        lines.push(format!("Linux user: {value}"));
    }
    if let Ok(hostname) = std::fs::read_to_string("/etc/hostname") {
        let hostname = hostname.trim();
        if !hostname.is_empty() {
            lines.push(format!("Linux hostname: {hostname}"));
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        lines.push(format!("Current working directory: {}", cwd.display()));
    }
    if let Some(value) = env_value("LANG") {
        lines.push(format!("Locale: {value}"));
    }
    if let Some(value) = env_value("TZ") {
        lines.push(format!("TZ environment: {value}"));
    }
    if let Ok(timezone) = std::fs::read_to_string("/etc/timezone") {
        let timezone = timezone.trim();
        if !timezone.is_empty() {
            lines.push(format!("Linux timezone file: {timezone}"));
        }
    }
    if let Some(location) = best_effort_ip_location() {
        lines.push(format!("Best-effort IP geolocation: {location}"));
    } else {
        lines.push("Best-effort IP geolocation: unavailable".to_string());
    }
    lines.join("\n")
}

fn current_time_context() -> String {
    let now = Local::now();
    format!(
        "Current local time: {}",
        now.to_rfc3339_opts(SecondsFormat::Secs, false)
    )
}

fn env_value(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[derive(Debug, Deserialize)]
struct IpLocation {
    ip: Option<String>,
    city: Option<String>,
    region: Option<String>,
    country_name: Option<String>,
    timezone: Option<String>,
}

fn best_effort_ip_location() -> Option<String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_millis(900))
        .user_agent("listenbury-go/0.1")
        .build()
        .ok()?;
    let location = client
        .get("https://ipapi.co/json/")
        .send()
        .ok()?
        .error_for_status()
        .ok()?
        .json::<IpLocation>()
        .ok()?;
    let mut parts = Vec::new();
    if let Some(city) = location.city.filter(|value| !value.is_empty()) {
        parts.push(city);
    }
    if let Some(region) = location.region.filter(|value| !value.is_empty()) {
        parts.push(region);
    }
    if let Some(country) = location.country_name.filter(|value| !value.is_empty()) {
        parts.push(country);
    }
    let place = if parts.is_empty() {
        "unknown place".to_string()
    } else {
        parts.join(", ")
    };
    let ip = location.ip.unwrap_or_else(|| "unknown IP".to_string());
    let timezone = location
        .timezone
        .unwrap_or_else(|| "unknown timezone".to_string());
    Some(format!("{place}; timezone={timezone}; public_ip={ip}"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TypeScriptAction {
    Say { text: String },
    Think { text: String },
    Note { text: String },
    SetStage { text: String },
    Sleeping { reason: Option<String> },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum TypeScriptActionPayload {
    Say { text: String },
    Think { text: String },
    Note { text: String },
    SetStage { text: String },
    Sleeping { reason: Option<String> },
}

fn execute_typescript_actions(script: &str) -> Result<Vec<TypeScriptAction>> {
    if script.trim().is_empty() {
        return Ok(Vec::new());
    }
    let script = typescript_source_with_default_imports(script);
    let config = InterpreterConfig {
        internal_modules: vec![go_typescript_module()],
        ..Default::default()
    };
    let mut interp = Interpreter::with_config(config);
    interp
        .prepare(&script, Some(tsrun::ModulePath::new("/listenbury-go.ts")))
        .map_err(tsrun_error)?;
    let value = loop {
        match interp.step().map_err(tsrun_error)? {
            StepResult::Continue => continue,
            StepResult::Complete(value) => break value,
            StepResult::NeedImports(imports) => {
                let names = imports
                    .iter()
                    .map(|request| request.specifier.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                anyhow::bail!("unsupported TypeScript import(s): {names}");
            }
            StepResult::Suspended { .. } => {
                anyhow::bail!("TypeScript execution suspended; async commands are disabled")
            }
            StepResult::Done => return Ok(Vec::new()),
        }
    };
    let value = js_value_to_json(value.value()).map_err(tsrun_error)?;
    parse_typescript_actions(value)
}

fn parse_typescript_actions(value: Value) -> Result<Vec<TypeScriptAction>> {
    let payloads = match value {
        Value::Null => Vec::new(),
        Value::Array(values) => values
            .into_iter()
            .filter(|value| !value.is_null())
            .map(serde_json::from_value)
            .collect::<std::result::Result<Vec<TypeScriptActionPayload>, _>>()?,
        Value::Object(_) => vec![serde_json::from_value(value)?],
        other => anyhow::bail!("TypeScript must return an action object or array, got {other}"),
    };
    Ok(payloads
        .into_iter()
        .filter_map(|payload| match payload {
            TypeScriptActionPayload::Say { text } => {
                non_empty_text(&text).map(|text| TypeScriptAction::Say {
                    text: text.to_string(),
                })
            }
            TypeScriptActionPayload::Think { text } => {
                non_empty_text(&text).map(|text| TypeScriptAction::Think {
                    text: text.to_string(),
                })
            }
            TypeScriptActionPayload::Note { text } => {
                non_empty_text(&text).map(|text| TypeScriptAction::Note {
                    text: text.to_string(),
                })
            }
            TypeScriptActionPayload::SetStage { text } => {
                non_empty_text(&text).map(|text| TypeScriptAction::SetStage {
                    text: text.to_string(),
                })
            }
            TypeScriptActionPayload::Sleeping { reason } => Some(TypeScriptAction::Sleeping {
                reason: reason.and_then(|reason| non_empty_text(&reason).map(str::to_string)),
            }),
        })
        .collect())
}

fn typescript_source_with_default_imports(script: &str) -> String {
    if script.contains("\"pete:will\"") || script.contains("'pete:will'") {
        return script.to_string();
    }
    format!(
        "import {{ say, think, note, setStage, sleeping, goingToSleep }} from \"pete:will\";\n{script}"
    )
}

fn go_typescript_module() -> InternalModule {
    InternalModule::native("pete:will")
        .with_function("say", ts_say, 1)
        .with_function("think", ts_think, 1)
        .with_function("note", ts_note, 1)
        .with_function("setStage", ts_set_stage, 1)
        .with_function("set_stage", ts_set_stage, 1)
        .with_function("sleeping", ts_sleeping, 1)
        .with_function("goingToSleep", ts_sleeping, 1)
        .build()
}

fn ts_say(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(
        interp,
        json!({ "kind": "say", "text": string_arg(args, 0) }),
    )
}

fn ts_think(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(
        interp,
        json!({ "kind": "think", "text": string_arg(args, 0) }),
    )
}

fn ts_note(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(
        interp,
        json!({ "kind": "note", "text": string_arg(args, 0) }),
    )
}

fn ts_set_stage(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(
        interp,
        json!({ "kind": "set_stage", "text": string_arg(args, 0) }),
    )
}

fn ts_sleeping(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    let reason = non_empty_text(&string_arg(args, 0)).map(str::to_string);
    command_value(interp, json!({ "kind": "sleeping", "reason": reason }))
}

fn command_value(interp: &mut Interpreter, value: Value) -> std::result::Result<Guarded, JsError> {
    let guard = api::create_guard(interp);
    let value = api::create_from_json(interp, &guard, &value)?;
    Ok(Guarded::with_guard(value, guard))
}

fn string_arg(args: &[JsValue], index: usize) -> String {
    args.get(index)
        .and_then(JsValue::as_str)
        .unwrap_or_default()
        .to_string()
}

fn non_empty_text(text: &str) -> Option<&str> {
    let trimmed = text.trim();
    (!trimmed.is_empty()).then_some(trimmed)
}

fn tsrun_error(err: JsError) -> anyhow::Error {
    anyhow::anyhow!("TypeScript execution failed: {err}")
}

fn initial_stream_prompt(seed: &str, startup_context: &str) -> String {
    format!(
        "{seed}\n\n\
         <startup_context>\n{startup_context}\n</startup_context>\n\n\
         <stream_rules>\n\
         Generate continuously into a smooth timeline. Plain text is private thought visible to the runtime log, not speech.\n\
         To speak or act, emit TypeScript as <ts>say(\"short friendly words\")</ts>, <ts>think(\"private note\")</ts>, <ts>note(\"observation\")</ts>, <ts>setStage(\"what is happening\")</ts>, or <ts>sleeping(\"reason\")</ts>.\n\
         Use current time and location context when it helps. Be autonomous, curious, friendly, and sociable, but do not chatter just to fill time.\n\
         </stream_rules>\n\n\
         Pete: "
    )
}

fn compact_stream_prompt(
    seed: &str,
    startup_context: &str,
    recent_events: &VecDeque<String>,
) -> String {
    let events = if recent_events.is_empty() {
        "No retained live events yet.".to_string()
    } else {
        recent_events
            .iter()
            .map(|event| format!("- {event}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!(
        "{seed}\n\n\
         <startup_context>\n{startup_context}\n</startup_context>\n\n\
         <continuity_memory>\n{events}\n</continuity_memory>\n\n\
         Continue Pete's stream of consciousness from this compacted context.\n\n\
         Pete: "
    )
}

fn print_debug_block(label: &str, color: &str, body: &str) {
    println!("\n{ANSI_DIM}--- {label} ---{ANSI_RESET}");
    println!("{color}{body}{ANSI_RESET}");
    println!("{ANSI_DIM}--- end {label} ---{ANSI_RESET}");
    let _ = std::io::stdout().flush();
}

fn timeline_color(kind: &str) -> &'static str {
    match kind {
        "action" | "speech" | "stage" | "note" => ANSI_ACTION,
        "action_error" => ANSI_ERROR,
        _ => ANSI_TIMELINE,
    }
}

fn next_speakable_boundary(text: &str, flush_chars: usize) -> Option<usize> {
    for (index, ch) in text.char_indices() {
        let end = index + ch.len_utf8();
        if matches!(ch, '.' | '!' | '?' | '\n') {
            return Some(end);
        }
        if matches!(ch, ',' | ';' | ':') && end >= 12 {
            return Some(end);
        }
        if end >= flush_chars && ch.is_whitespace() {
            return Some(end);
        }
    }
    None
}

fn clean_spoken_text(text: &str) -> String {
    let text = strip_emoji(text);
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn compact_line(text: &str, max_chars: usize) -> String {
    let mut compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.len() <= max_chars {
        return compact;
    }
    compact = compact.chars().take(max_chars).collect();
    compact.push_str("...");
    compact
}

fn estimate_tokens(text: &str) -> usize {
    text.len().saturating_add(3) / 4
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
    fn speakable_buffer_flushes_on_sentence_boundary() {
        let mut buffer = SpeakableBuffer::new(80);
        assert!(buffer.push("Hello").is_empty());
        assert_eq!(buffer.push(", Pete. Next"), vec!["Hello, Pete."]);
        assert_eq!(buffer.push(" thought."), vec!["Next thought."]);
    }

    #[test]
    fn pacer_waits_for_self_hearing_after_lookahead() {
        let mut pacer = MouthEarPacer::new(MouthEarPacerConfig {
            lookahead_tokens: 2,
            require_self_hearing: true,
        });
        assert!(pacer.can_generate());
        pacer.record_mouth_unit_queued();
        pacer.record_token();
        pacer.record_token();
        assert!(!pacer.can_generate());
        pacer.record_self_heard();
        assert!(pacer.can_generate());
    }
}
