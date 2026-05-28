use crate::cli::GoCommand;
use anyhow::Result;

use crate::cli::commands::cpal_diag::play_audio_frames;
use crate::cli::commands::source_inspection::{
    execute_grep_source, execute_list_source_files, execute_search_source, execute_view_source_file,
};
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
use serde_json::{Map, Value, json};
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

const PETE_WILL_RUNTIME_PROMPT: &str = "TypeScript runs through tsrun with only the internal module \"pete:will\" available. The builders are already available in scope; imports from \"pete:will\" are also allowed. Make each <ts>...</ts> block return a command object or an array of command objects.\n\
Available functions:\n\
- say(text, options?): queue spoken words for the mouth. options may include { interrupt: true } when speech should intentionally cut in.\n\
- shutup(): request current speech/queued speech to stop.\n\
- pause(): request synthetic playback pause.\n\
- resume(): request synthetic playback resume.\n\
- think(text): write a private thought to the debug timeline.\n\
- note(text): write a runtime note to the debug timeline.\n\
- setStage(text, options?): update the current screenplay beat. options may include topic, summary, setting, and action. Prefer action-first scene prose such as setStage(\"Setting: lab. Action: Pete listens.\", { topic: \"lab\", summary: \"Pete listens\" }).\n\
- setTopic(topic, options?): lightweight topic label; options may include instruction and summary.\n\
- startNewTopic(previousTopic, options?): mark a scene/topic transition. options may include topic, instruction, summary, and trigger.\n\
- topicChangedWhen(trigger, options?): mark the words or event that caused a topic transition. options may include fromTopic, toTopic, topic, instruction, and summary.\n\
- startNewEpisode(reason, options?): mark a larger episode reset. options may include topic, instruction, summary, and trigger.\n\
- sleeping(reason?) or goingToSleep(reason?): clean shutdown only after a current live user input asks Pete to stop, shut down, sleep, go to sleep, or end the session.\n\
- extractEntities(text): request entity extraction for names, preferences, places, relationships, plans, corrections, facts, or recurring context.\n\
- updateGraphNodeFields(nodeId, fields, options?): request memory field updates, especially description: \"natural language noun phrase\".\n\
- searchGraphNodes(query, options?): search memory by text, field, value, or combinations. query may be a string or object with text, field, value, and limit.\n\
- queryMemories(text, options?): retrieve memories for a phrase, sentence, name, topic, or claim. options may include limit and minScore.\n\
- listFiles(): list bundled Listenbury source files.\n\
- readSourceFile(path, page?) or readFile(path, page?): inspect one source file page.\n\
- searchSource(query, limit?): source text search.\n\
- grepSource(pattern, limit?): grep-like source line search.\n\
Use source inspection when bored, alone, or waiting. Do not use say for clock ticks, quiet moments, idle narration, or every thought. Never call sleeping() or goingToSleep() because historical memory, recalled context, prior-session transcript, or a source result says someone once asked Pete to shut down.\n\
This runtime is not Harmony and not a tool-calling chat template. Never write <|start|>, <|end|>, channel names, assistant/user/system message markers, to=container.exec, JSON tool calls, shell commands, or markdown code fences. The only executable action syntax here is <ts>peteWillBuilder(...)</ts>.";

const TYPESCRIPT_START: &str = "<ts>";

const TYPESCRIPT_END: &str = "</ts>";

const MAX_TTS_TIMEOUT: Duration = Duration::from_secs(30);
const CLOCK_PROMPT_INTERVAL: Duration = Duration::from_secs(30);

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
    generation_paused: bool,
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
            next_clock_at: Instant::now() + CLOCK_PROMPT_INTERVAL,
            generation_paused: false,
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

            if !cancelled {
                self.set_generation_paused(!self.pacer.can_generate())?;
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
                if !is_meaningful_thought(&text) {
                    return Ok(());
                }
                self.timeline("thought", &text);
                Ok(())
            }
            StreamOutput::TypeScript(source) => {
                self.timeline("action", &source);
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
                TypeScriptAction::Say { text, interrupt } => {
                    self.timeline("speech", &text);
                    self.append_observation(StreamObservation::ActionResult(format!(
                        "Queued speech{}: {}",
                        if interrupt { " with interrupt" } else { "" },
                        compact_line(&text, 300)
                    )))?;
                    if interrupt {
                        self.pacer.record_self_heard();
                    }
                    for unit in split_speakable_units(&text, self.config.lookahead_chars) {
                        self.enqueue_speech(unit)?;
                    }
                }
                TypeScriptAction::Shutup => {
                    self.timeline("action_result", "shutup requested.");
                    self.append_observation(StreamObservation::ActionResult(
                        "shutup requested; go has no queued-speech clearing yet.".to_string(),
                    ))?;
                }
                TypeScriptAction::Pause => {
                    self.timeline("action_result", "pause requested.");
                    self.append_observation(StreamObservation::ActionResult(
                        "pause requested; go has no TTS pause control yet.".to_string(),
                    ))?;
                }
                TypeScriptAction::Resume => {
                    self.timeline("action_result", "resume requested.");
                    self.append_observation(StreamObservation::ActionResult(
                        "resume requested; go has no TTS pause control yet.".to_string(),
                    ))?;
                }
                TypeScriptAction::Think { text } => {
                    self.timeline("thought", &text);
                }
                TypeScriptAction::Note { text } => {
                    self.timeline("note", &text);
                    self.append_observation(StreamObservation::ActionResult(format!(
                        "Noted: {}",
                        compact_line(&text, 500)
                    )))?;
                }
                TypeScriptAction::SetStage {
                    topic,
                    instruction,
                    summary,
                } => {
                    self.timeline("stage", &instruction);
                    self.append_observation(StreamObservation::ActionResult(format!(
                        "Stage set: {}{}{}",
                        compact_line(&instruction, 500),
                        topic
                            .as_deref()
                            .map(|topic| format!(" topic={topic}"))
                            .unwrap_or_default(),
                        summary
                            .as_deref()
                            .map(|summary| format!(" summary={summary}"))
                            .unwrap_or_default()
                    )))?;
                }
                TypeScriptAction::StartNewTopic {
                    last_topic,
                    topic,
                    instruction,
                    summary,
                    trigger,
                } => {
                    let message = format!(
                        "Topic transition from {}{}{}{}{}.",
                        last_topic,
                        topic
                            .as_deref()
                            .map(|topic| format!(" to {topic}"))
                            .unwrap_or_default(),
                        trigger
                            .as_deref()
                            .map(|trigger| format!(" triggered by {trigger}"))
                            .unwrap_or_default(),
                        instruction
                            .as_deref()
                            .map(|instruction| format!(" instruction={instruction}"))
                            .unwrap_or_default(),
                        summary
                            .as_deref()
                            .map(|summary| format!(" summary={summary}"))
                            .unwrap_or_default(),
                    );
                    self.timeline("stage", &message);
                    self.append_observation(StreamObservation::ActionResult(message))?;
                }
                TypeScriptAction::StartNewEpisode {
                    reason,
                    topic,
                    instruction,
                    summary,
                    trigger,
                } => {
                    let message = format!(
                        "Episode transition: {}{}{}{}{}.",
                        reason,
                        topic
                            .as_deref()
                            .map(|topic| format!(" topic={topic}"))
                            .unwrap_or_default(),
                        trigger
                            .as_deref()
                            .map(|trigger| format!(" trigger={trigger}"))
                            .unwrap_or_default(),
                        instruction
                            .as_deref()
                            .map(|instruction| format!(" instruction={instruction}"))
                            .unwrap_or_default(),
                        summary
                            .as_deref()
                            .map(|summary| format!(" summary={summary}"))
                            .unwrap_or_default(),
                    );
                    self.timeline("stage", &message);
                    self.append_observation(StreamObservation::ActionResult(message))?;
                }
                TypeScriptAction::ExtractEntities { text } => {
                    let message = format!(
                        "Entity extraction requested{}.",
                        text.as_deref()
                            .map(|text| format!(": {}", compact_line(text, 500)))
                            .unwrap_or_default()
                    );
                    self.timeline("action_result", &message);
                    self.append_observation(StreamObservation::ActionResult(message))?;
                }
                TypeScriptAction::UpdateGraphNodeFields {
                    node_id,
                    label,
                    fields,
                } => {
                    let message = format!(
                        "Graph node update requested for {node_id}{}: {}",
                        label
                            .as_deref()
                            .map(|label| format!(" label={label}"))
                            .unwrap_or_default(),
                        summarize_command_fields(&fields)
                    );
                    self.timeline("action_result", &message);
                    self.append_observation(StreamObservation::ActionResult(message))?;
                }
                TypeScriptAction::QueryMemories {
                    text,
                    limit,
                    min_score,
                } => {
                    let message = format!(
                        "Memory query requested: {}{}{}",
                        compact_line(&text, 500),
                        limit
                            .map(|limit| format!(" limit={limit}"))
                            .unwrap_or_default(),
                        min_score
                            .map(|min_score| format!(" min_score={min_score:.3}"))
                            .unwrap_or_default()
                    );
                    self.timeline("action_result", &message);
                    self.append_observation(StreamObservation::ActionResult(message))?;
                }
                TypeScriptAction::SearchGraphNodes {
                    text,
                    field,
                    value,
                    limit,
                } => {
                    let message = format!(
                        "Graph node search requested: {}{}",
                        format_graph_node_search_query_parts(
                            text.as_deref(),
                            field.as_deref(),
                            value.as_ref()
                        ),
                        limit
                            .map(|limit| format!(" limit={limit}"))
                            .unwrap_or_default()
                    );
                    self.timeline("action_result", &message);
                    self.append_observation(StreamObservation::ActionResult(message))?;
                }
                TypeScriptAction::ListFiles => {
                    let output = execute_list_source_files();
                    self.timeline("action_result", "Listed Listenbury source files.");
                    self.append_observation(StreamObservation::ActionResult(output))?;
                }
                TypeScriptAction::ReadSourceFile { file, page } => {
                    let output = execute_view_source_file(&file, page);
                    self.timeline(
                        "action_result",
                        &format!("Read source file {file} page {page}."),
                    );
                    self.append_observation(StreamObservation::ActionResult(output))?;
                }
                TypeScriptAction::SearchSource { query, limit } => {
                    let output = execute_search_source(&query, limit);
                    self.timeline("action_result", &format!("Searched source for {query}."));
                    self.append_observation(StreamObservation::ActionResult(output))?;
                }
                TypeScriptAction::GrepSource { pattern, limit } => {
                    let output = execute_grep_source(&pattern, limit);
                    self.timeline("action_result", &format!("Grepped source for {pattern}."));
                    self.append_observation(StreamObservation::ActionResult(output))?;
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
        self.next_clock_at = now + CLOCK_PROMPT_INTERVAL;
        self.append_observation(StreamObservation::Clock(current_time_context()))
    }

    fn set_generation_paused(&mut self, paused: bool) -> Result<()> {
        if self.generation_paused == paused {
            return Ok(());
        }
        self.llm
            .set_paused(self.generation, paused)
            .with_context(|| {
                if paused {
                    "failed to pace stream generation"
                } else {
                    "failed to resume stream generation"
                }
            })?;
        self.generation_paused = paused;
        Ok(())
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
        self.generation_paused = false;
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
                    if is_meaningful_thought(&thought) {
                        outputs.push(StreamOutput::Thought(thought));
                    }
                    continue;
                }

                let content_start = TYPESCRIPT_START.len();
                let body = &self.text[content_start..];
                let (source, consumed) = if let Some(end_rel) = body.find(TYPESCRIPT_END) {
                    let source = body[..end_rel].trim().to_string();
                    (source, content_start + end_rel + TYPESCRIPT_END.len())
                } else if let Some((source, body_consumed)) =
                    recover_unclosed_typescript_source(body)
                {
                    (source, content_start + body_consumed)
                } else {
                    break;
                };
                self.text = self.text[consumed..].to_string();
                if !source.is_empty() {
                    outputs.push(StreamOutput::TypeScript(source));
                }
                continue;
            }

            let Some(boundary) = next_thought_boundary(&self.text, self.flush_chars) else {
                break;
            };
            let thought = self.text[..boundary].to_string();
            self.text = self.text[boundary..].to_string();
            if is_meaningful_thought(&thought) {
                outputs.push(StreamOutput::Thought(thought));
            }
        }
        ParsedStreamOutput { outputs }
    }
}

fn recover_unclosed_typescript_source(body: &str) -> Option<(String, usize)> {
    let trimmed_start = body.len().saturating_sub(body.trim_start().len());
    let scan = &body[trimmed_start..];
    let end = balanced_typescript_expression_end(scan)?;
    let source = scan[..end].trim();
    if !looks_like_pete_will_source(source) {
        return None;
    }
    Some((source.to_string(), trimmed_start + end))
}

fn balanced_typescript_expression_end(source: &str) -> Option<usize> {
    let mut stack = Vec::new();
    let mut quote = None;
    let mut escape = false;
    let mut saw_expression = false;
    let mut candidate_end = None;

    for (index, ch) in source.char_indices() {
        if let Some(quote_ch) = quote {
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == quote_ch {
                quote = None;
            }
            continue;
        }

        match ch {
            '"' | '\'' | '`' => quote = Some(ch),
            '(' | '[' | '{' => {
                saw_expression = true;
                stack.push(ch);
            }
            ')' => {
                if stack.pop() != Some('(') {
                    return None;
                }
                if stack.is_empty() {
                    candidate_end = Some(index + ch.len_utf8());
                }
            }
            ']' => {
                if stack.pop() != Some('[') {
                    return None;
                }
                if stack.is_empty() {
                    candidate_end = Some(index + ch.len_utf8());
                }
            }
            '}' => {
                if stack.pop() != Some('{') {
                    return None;
                }
                if stack.is_empty() {
                    candidate_end = Some(index + ch.len_utf8());
                }
            }
            ';' if stack.is_empty() && saw_expression => {
                return Some(index + ch.len_utf8());
            }
            _ => {}
        }

        if let Some(end) = candidate_end {
            let rest = source[end..].trim_start();
            if rest.starts_with(TYPESCRIPT_END)
                || rest.starts_with("<|")
                || rest.starts_with("Then ")
                || rest.starts_with("But ")
                || rest.starts_with("We ")
                || rest.starts_with("I ")
                || rest.starts_with("Ok")
                || rest.starts_with("OK")
                || rest.starts_with("Maybe ")
                || rest.starts_with("So ")
                || rest.starts_with("Let's ")
                || rest.starts_with("This ")
                || rest.starts_with("The ")
                || rest.starts_with("Also,")
            {
                return Some(end);
            }
        }
    }

    None
}

fn looks_like_pete_will_source(source: &str) -> bool {
    let trimmed = source.trim_start();
    if trimmed.starts_with('[') {
        return true;
    }
    [
        "say",
        "shutup",
        "pause",
        "resume",
        "think",
        "note",
        "setStage",
        "setTopic",
        "startNewTopic",
        "topicChangedWhen",
        "startNewEpisode",
        "sleeping",
        "goingToSleep",
        "extractEntities",
        "updateGraphNodeFields",
        "searchGraphNodes",
        "queryMemories",
        "listFiles",
        "readSourceFile",
        "readFile",
        "searchSource",
        "grepSource",
    ]
    .iter()
    .any(|name| trimmed.starts_with(&format!("{name}(")))
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

#[derive(Debug, Clone, PartialEq)]
enum TypeScriptAction {
    Say {
        text: String,
        interrupt: bool,
    },
    Shutup,
    Pause,
    Resume,
    SetStage {
        topic: Option<String>,
        instruction: String,
        summary: Option<String>,
    },
    StartNewTopic {
        last_topic: String,
        topic: Option<String>,
        instruction: Option<String>,
        summary: Option<String>,
        trigger: Option<String>,
    },
    StartNewEpisode {
        reason: String,
        topic: Option<String>,
        instruction: Option<String>,
        summary: Option<String>,
        trigger: Option<String>,
    },
    ExtractEntities {
        text: Option<String>,
    },
    UpdateGraphNodeFields {
        node_id: String,
        label: Option<String>,
        fields: Map<String, Value>,
    },
    QueryMemories {
        text: String,
        limit: Option<usize>,
        min_score: Option<f32>,
    },
    SearchGraphNodes {
        text: Option<String>,
        field: Option<String>,
        value: Option<Value>,
        limit: Option<usize>,
    },
    ListFiles,
    ReadSourceFile {
        file: String,
        page: usize,
    },
    SearchSource {
        query: String,
        limit: usize,
    },
    GrepSource {
        pattern: String,
        limit: usize,
    },
    Sleeping {
        reason: Option<String>,
    },
    Think {
        text: String,
    },
    Note {
        text: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum TypeScriptActionPayload {
    Say {
        text: String,
        #[serde(default)]
        interrupt: bool,
    },
    Shutup,
    Pause,
    Resume,
    SetStage {
        #[serde(default)]
        topic: Option<String>,
        instruction: String,
        #[serde(default)]
        summary: Option<String>,
    },
    StartNewTopic {
        last_topic: String,
        #[serde(default)]
        topic: Option<String>,
        #[serde(default)]
        instruction: Option<String>,
        #[serde(default)]
        summary: Option<String>,
        #[serde(default)]
        trigger: Option<String>,
    },
    StartNewEpisode {
        reason: String,
        #[serde(default)]
        topic: Option<String>,
        #[serde(default)]
        instruction: Option<String>,
        #[serde(default)]
        summary: Option<String>,
        #[serde(default)]
        trigger: Option<String>,
    },
    ExtractEntities {
        text: Option<String>,
    },
    UpdateGraphNodeFields {
        node_id: String,
        #[serde(default)]
        label: Option<String>,
        #[serde(default)]
        fields: Map<String, Value>,
    },
    QueryMemories {
        text: String,
        #[serde(default)]
        limit: Option<usize>,
        #[serde(default)]
        min_score: Option<f32>,
    },
    SearchGraphNodes {
        #[serde(default)]
        text: Option<String>,
        #[serde(default)]
        field: Option<String>,
        #[serde(default)]
        value: Option<Value>,
        #[serde(default)]
        limit: Option<usize>,
    },
    ListFiles,
    ReadSourceFile {
        file: String,
        #[serde(default)]
        page: Option<usize>,
    },
    SearchSource {
        query: String,
        #[serde(default)]
        limit: Option<usize>,
    },
    GrepSource {
        pattern: String,
        #[serde(default)]
        limit: Option<usize>,
    },
    Sleeping {
        reason: Option<String>,
    },
    Think {
        text: String,
    },
    Note {
        text: String,
    },
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
        .filter_map(parse_action_payload)
        .collect())
}

fn parse_action_payload(payload: TypeScriptActionPayload) -> Option<TypeScriptAction> {
    match payload {
        TypeScriptActionPayload::Say { text, interrupt } => {
            non_empty_text(&text).map(|text| TypeScriptAction::Say {
                text: text.to_string(),
                interrupt,
            })
        }
        TypeScriptActionPayload::Shutup => Some(TypeScriptAction::Shutup),
        TypeScriptActionPayload::Pause => Some(TypeScriptAction::Pause),
        TypeScriptActionPayload::Resume => Some(TypeScriptAction::Resume),
        TypeScriptActionPayload::SetStage {
            topic,
            instruction,
            summary,
        } => non_empty_text(&instruction).map(|instruction| TypeScriptAction::SetStage {
            topic: topic.and_then(|topic| non_empty_text(&topic).map(str::to_string)),
            instruction: instruction.to_string(),
            summary: summary.and_then(|summary| non_empty_text(&summary).map(str::to_string)),
        }),
        TypeScriptActionPayload::StartNewTopic {
            last_topic,
            topic,
            instruction,
            summary,
            trigger,
        } => non_empty_text(&last_topic).map(|last_topic| TypeScriptAction::StartNewTopic {
            last_topic: last_topic.to_string(),
            topic: topic.and_then(|topic| non_empty_text(&topic).map(str::to_string)),
            instruction: instruction
                .and_then(|instruction| non_empty_text(&instruction).map(str::to_string)),
            summary: summary.and_then(|summary| non_empty_text(&summary).map(str::to_string)),
            trigger: trigger.and_then(|trigger| non_empty_text(&trigger).map(str::to_string)),
        }),
        TypeScriptActionPayload::StartNewEpisode {
            reason,
            topic,
            instruction,
            summary,
            trigger,
        } => non_empty_text(&reason).map(|reason| TypeScriptAction::StartNewEpisode {
            reason: reason.to_string(),
            topic: topic.and_then(|topic| non_empty_text(&topic).map(str::to_string)),
            instruction: instruction
                .and_then(|instruction| non_empty_text(&instruction).map(str::to_string)),
            summary: summary.and_then(|summary| non_empty_text(&summary).map(str::to_string)),
            trigger: trigger.and_then(|trigger| non_empty_text(&trigger).map(str::to_string)),
        }),
        TypeScriptActionPayload::ExtractEntities { text } => {
            Some(TypeScriptAction::ExtractEntities {
                text: text.and_then(|text| non_empty_text(&text).map(str::to_string)),
            })
        }
        TypeScriptActionPayload::UpdateGraphNodeFields {
            node_id,
            label,
            fields,
        } => non_empty_text(&node_id).and_then(|node_id| {
            (!fields.is_empty()).then_some(TypeScriptAction::UpdateGraphNodeFields {
                node_id: node_id.to_string(),
                label: label.and_then(|label| non_empty_text(&label).map(str::to_string)),
                fields,
            })
        }),
        TypeScriptActionPayload::QueryMemories {
            text,
            limit,
            min_score,
        } => non_empty_text(&text).map(|text| TypeScriptAction::QueryMemories {
            text: text.to_string(),
            limit: limit.map(|limit| limit.clamp(1, 16)),
            min_score,
        }),
        TypeScriptActionPayload::SearchGraphNodes {
            text,
            field,
            value,
            limit,
        } => {
            let text = text.and_then(|text| non_empty_text(&text).map(str::to_string));
            let field = field.and_then(|field| non_empty_text(&field).map(str::to_string));
            (text.is_some() || field.is_some() || value.is_some()).then_some(
                TypeScriptAction::SearchGraphNodes {
                    text,
                    field,
                    value,
                    limit: limit.map(|limit| limit.clamp(1, 16)),
                },
            )
        }
        TypeScriptActionPayload::ListFiles => Some(TypeScriptAction::ListFiles),
        TypeScriptActionPayload::ReadSourceFile { file, page } => {
            let file = file.trim();
            (!file.is_empty()).then(|| TypeScriptAction::ReadSourceFile {
                file: file.to_string(),
                page: page.unwrap_or(1).max(1),
            })
        }
        TypeScriptActionPayload::SearchSource { query, limit } => {
            non_empty_text(&query).map(|query| TypeScriptAction::SearchSource {
                query: query.to_string(),
                limit: limit.unwrap_or(12).max(1),
            })
        }
        TypeScriptActionPayload::GrepSource { pattern, limit } => {
            non_empty_text(&pattern).map(|pattern| TypeScriptAction::GrepSource {
                pattern: pattern.to_string(),
                limit: limit.unwrap_or(12).max(1),
            })
        }
        TypeScriptActionPayload::Sleeping { reason } => Some(TypeScriptAction::Sleeping {
            reason: reason.and_then(|reason| non_empty_text(&reason).map(str::to_string)),
        }),
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
    }
}

fn typescript_source_with_default_imports(script: &str) -> String {
    if script.contains("\"pete:will\"") || script.contains("'pete:will'") {
        return script.to_string();
    }
    format!(
        "import {{ say, shutup, pause, resume, think, note, setStage, setTopic, startNewTopic, topicChangedWhen, startNewEpisode, sleeping, goingToSleep, extractEntities, updateGraphNodeFields, searchGraphNodes, queryMemories, listFiles, readSourceFile, readFile, searchSource, grepSource }} from \"pete:will\";\n{script}"
    )
}

fn go_typescript_module() -> InternalModule {
    InternalModule::native("pete:will")
        .with_function("say", ts_say, 2)
        .with_function("shutup", ts_shutup, 0)
        .with_function("pause", ts_pause, 0)
        .with_function("resume", ts_resume, 0)
        .with_function("think", ts_think, 1)
        .with_function("note", ts_note, 1)
        .with_function("setStage", ts_set_stage, 2)
        .with_function("set_stage", ts_set_stage, 2)
        .with_function("setTopic", ts_set_topic, 2)
        .with_function("set_topic", ts_set_topic, 2)
        .with_function("startNewTopic", ts_start_new_topic, 2)
        .with_function("start_new_topic", ts_start_new_topic, 2)
        .with_function("topicChangedWhen", ts_topic_changed_when, 2)
        .with_function("topic_changed_when", ts_topic_changed_when, 2)
        .with_function("startNewEpisode", ts_start_new_episode, 2)
        .with_function("start_new_episode", ts_start_new_episode, 2)
        .with_function("newEpisodeStarted", ts_start_new_episode, 2)
        .with_function("sleeping", ts_sleeping, 1)
        .with_function("goingToSleep", ts_sleeping, 1)
        .with_function("going_to_sleep", ts_sleeping, 1)
        .with_function("goToSleep", ts_sleeping, 1)
        .with_function("go_to_sleep", ts_sleeping, 1)
        .with_function("extractEntities", ts_extract_entities, 1)
        .with_function("extract_entities", ts_extract_entities, 1)
        .with_function("updateGraphNodeFields", ts_update_graph_node_fields, 3)
        .with_function("update_graph_node_fields", ts_update_graph_node_fields, 3)
        .with_function("updateEntityFields", ts_update_graph_node_fields, 3)
        .with_function("searchGraphNodes", ts_search_graph_nodes, 2)
        .with_function("search_graph_nodes", ts_search_graph_nodes, 2)
        .with_function("searchEntities", ts_search_graph_nodes, 2)
        .with_function("queryMemories", ts_query_memories, 2)
        .with_function("query_memories", ts_query_memories, 2)
        .with_function("recallMemories", ts_query_memories, 2)
        .with_function("listFiles", ts_list_files, 0)
        .with_function("list_files", ts_list_files, 0)
        .with_function("readSourceFile", ts_read_source_file, 2)
        .with_function("read_source_file", ts_read_source_file, 2)
        .with_function("readFile", ts_read_source_file, 2)
        .with_function("read_file", ts_read_source_file, 2)
        .with_function("searchSource", ts_search_source, 2)
        .with_function("search_source", ts_search_source, 2)
        .with_function("grepSource", ts_grep_source, 2)
        .with_function("grep_source", ts_grep_source, 2)
        .build()
}

fn ts_say(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(
        interp,
        json!({ "kind": "say", "text": string_arg(args, 0), "interrupt": interrupt_arg(args, 1) }),
    )
}

fn ts_shutup(
    interp: &mut Interpreter,
    _this: JsValue,
    _args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(interp, json!({ "kind": "shutup" }))
}

fn ts_pause(
    interp: &mut Interpreter,
    _this: JsValue,
    _args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(interp, json!({ "kind": "pause" }))
}

fn ts_resume(
    interp: &mut Interpreter,
    _this: JsValue,
    _args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(interp, json!({ "kind": "resume" }))
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
    let topic = stage_string_property_arg(args, "topic");
    let setting = stage_string_property_arg(args, "setting");
    let action = stage_string_property_arg(args, "action");
    let summary = stage_string_property_arg(args, "summary").or_else(|| action.clone());
    let raw_instruction = string_arg(args, 0);
    let instruction = non_empty_text(&raw_instruction)
        .map(str::to_string)
        .or_else(|| screenplay_stage_description(setting.as_deref(), action.as_deref()))
        .unwrap_or_default();
    command_value(
        interp,
        json!({
            "kind": "set_stage",
            "topic": topic,
            "instruction": instruction,
            "summary": summary,
        }),
    )
}

fn ts_set_topic(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    let topic = string_arg(args, 0);
    let instruction = match args.get(1) {
        Some(JsValue::String(value)) => value.to_string(),
        Some(JsValue::Object(_)) => optional_string_property_arg(args, 1, "instruction")
            .unwrap_or_else(|| format!("The current topic is {}.", topic.trim())),
        _ => format!("The current topic is {}.", topic.trim()),
    };
    let summary = optional_string_property_arg(args, 1, "summary");
    command_value(
        interp,
        json!({
            "kind": "set_stage",
            "topic": topic,
            "instruction": instruction,
            "summary": summary,
        }),
    )
}

fn ts_start_new_topic(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(
        interp,
        json!({
            "kind": "start_new_topic",
            "last_topic": string_arg(args, 0),
            "topic": optional_string_property_arg(args, 1, "topic"),
            "instruction": optional_string_property_arg(args, 1, "instruction"),
            "summary": optional_string_property_arg(args, 1, "summary"),
            "trigger": optional_string_property_arg(args, 1, "trigger"),
        }),
    )
}

fn ts_topic_changed_when(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    let trigger = string_arg(args, 0);
    let last_topic = optional_string_property_arg(args, 1, "fromTopic")
        .or_else(|| optional_string_property_arg(args, 1, "from_topic"))
        .unwrap_or_else(|| "previous topic".to_string());
    let topic = optional_string_property_arg(args, 1, "toTopic")
        .or_else(|| optional_string_property_arg(args, 1, "to_topic"))
        .or_else(|| optional_string_property_arg(args, 1, "topic"));
    let instruction = optional_string_property_arg(args, 1, "instruction").or_else(|| {
        non_empty_text(&trigger)
            .map(|trigger| format!("The topic changed when the interlocutor said: {trigger}"))
    });
    let summary = optional_string_property_arg(args, 1, "summary");
    command_value(
        interp,
        json!({
            "kind": "start_new_topic",
            "last_topic": last_topic,
            "topic": topic,
            "instruction": instruction,
            "summary": summary,
            "trigger": trigger,
        }),
    )
}

fn ts_start_new_episode(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(
        interp,
        json!({
            "kind": "start_new_episode",
            "reason": string_arg(args, 0),
            "topic": optional_string_property_arg(args, 1, "topic"),
            "instruction": optional_string_property_arg(args, 1, "instruction"),
            "summary": optional_string_property_arg(args, 1, "summary"),
            "trigger": optional_string_property_arg(args, 1, "trigger"),
        }),
    )
}

fn ts_sleeping(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    let reason = match args.first() {
        Some(JsValue::String(value)) => {
            let value = value.to_string();
            non_empty_text(&value).map(str::to_string)
        }
        Some(JsValue::Object(_)) => optional_string_property_arg(args, 0, "reason"),
        _ => None,
    };
    command_value(interp, json!({ "kind": "sleeping", "reason": reason }))
}

fn ts_extract_entities(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(
        interp,
        json!({ "kind": "extract_entities", "text": string_arg(args, 0) }),
    )
}

fn ts_update_graph_node_fields(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    let label = args.get(2).and_then(|value| match value {
        JsValue::String(value) => Some(value.to_string()),
        JsValue::Object(_) => match api::get_property(value, "label") {
            Ok(JsValue::String(value)) => Some(value.to_string()),
            _ => None,
        },
        _ => None,
    });
    command_value(
        interp,
        json!({
            "kind": "update_graph_node_fields",
            "node_id": string_arg(args, 0),
            "label": label,
            "fields": object_arg(args, 1),
        }),
    )
}

fn ts_search_graph_nodes(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    let mut text = None;
    let mut field = None;
    let mut value = None;
    let mut limit = None;

    match args.first() {
        Some(JsValue::String(raw)) => {
            let raw = raw.to_string();
            text = non_empty_text(&raw).map(str::to_string);
        }
        Some(JsValue::Object(_)) => {
            text = optional_string_property_arg(args, 0, "text");
            field = optional_string_property_arg(args, 0, "field");
            value = optional_json_property_arg(args, 0, "value");
            limit = optional_number_arg(args, 0, "limit")
                .map(|value| value.round().clamp(1.0, 16.0) as usize);
        }
        _ => {}
    }

    if let Some(options) = args.get(1)
        && matches!(options, JsValue::Object(_))
    {
        text = optional_string_property_arg(args, 1, "text").or(text);
        field = optional_string_property_arg(args, 1, "field").or(field);
        value = optional_json_property_arg(args, 1, "value").or(value);
        limit = optional_number_arg(args, 1, "limit")
            .map(|value| value.round().clamp(1.0, 16.0) as usize)
            .or(limit);
    }

    command_value(
        interp,
        json!({
            "kind": "search_graph_nodes",
            "text": text,
            "field": field,
            "value": value,
            "limit": limit,
        }),
    )
}

fn ts_query_memories(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    let limit =
        optional_number_arg(args, 1, "limit").map(|value| value.round().clamp(1.0, 16.0) as usize);
    let min_score = optional_number_arg(args, 1, "minScore")
        .or_else(|| optional_number_arg(args, 1, "min_score"))
        .map(|value| value.clamp(0.0, 1.0) as f32);
    command_value(
        interp,
        json!({
            "kind": "query_memories",
            "text": string_arg(args, 0),
            "limit": limit,
            "min_score": min_score,
        }),
    )
}

fn ts_list_files(
    interp: &mut Interpreter,
    _this: JsValue,
    _args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(interp, json!({ "kind": "list_files" }))
}

fn ts_read_source_file(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    let mut value = json!({ "kind": "read_source_file", "file": string_arg(args, 0) });
    if let Some(page) = optional_positive_integer_arg(args, 1, "page") {
        value["page"] = json!(page);
    }
    command_value(interp, value)
}

fn ts_search_source(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    let mut value = json!({ "kind": "search_source", "query": string_arg(args, 0) });
    if let Some(limit) = optional_positive_integer_arg(args, 1, "limit") {
        value["limit"] = json!(limit);
    }
    command_value(interp, value)
}

fn ts_grep_source(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    let mut value = json!({ "kind": "grep_source", "pattern": string_arg(args, 0) });
    if let Some(limit) = optional_positive_integer_arg(args, 1, "limit") {
        value["limit"] = json!(limit);
    }
    command_value(interp, value)
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

fn interrupt_arg(args: &[JsValue], index: usize) -> bool {
    let Some(value) = args.get(index) else {
        return false;
    };
    match value {
        JsValue::Boolean(value) => *value,
        JsValue::Object(_) => matches!(
            api::get_property(value, "interrupt"),
            Ok(JsValue::Boolean(true))
        ),
        _ => false,
    }
}

fn optional_number_arg(args: &[JsValue], index: usize, property: &str) -> Option<f64> {
    let value = args.get(index)?;
    match value {
        JsValue::Number(value) => value.is_finite().then_some(*value),
        JsValue::Object(_) => match api::get_property(value, property) {
            Ok(JsValue::Number(value)) if value.is_finite() => Some(value),
            _ => None,
        },
        _ => None,
    }
}

fn optional_positive_integer_arg(args: &[JsValue], index: usize, property: &str) -> Option<usize> {
    optional_number_arg(args, index, property).map(|value| value.floor().max(1.0) as usize)
}

fn optional_json_property_arg(args: &[JsValue], index: usize, property: &str) -> Option<Value> {
    let value = args.get(index)?;
    if let JsValue::Object(_) = value {
        return api::get_property(value, property)
            .ok()
            .and_then(|value| js_value_to_json(&value).ok())
            .filter(|value| !value.is_null());
    }
    None
}

fn optional_string_property_arg(args: &[JsValue], index: usize, property: &str) -> Option<String> {
    optional_json_property_arg(args, index, property).and_then(|value| match value {
        Value::String(value) => non_empty_text(&value).map(str::to_string),
        _ => None,
    })
}

fn object_arg(args: &[JsValue], index: usize) -> Map<String, Value> {
    let Some(value) = args.get(index) else {
        return Map::new();
    };
    let Ok(Value::Object(object)) = js_value_to_json(value).map_err(tsrun_error) else {
        return Map::new();
    };
    object
}

fn stage_string_property_arg(args: &[JsValue], property: &str) -> Option<String> {
    optional_string_property_arg(args, 1, property)
        .or_else(|| optional_string_property_arg(args, 0, property))
}

fn screenplay_stage_description(setting: Option<&str>, action: Option<&str>) -> Option<String> {
    match (setting, action) {
        (Some(setting), Some(action)) => Some(format!("Setting: {setting}. Action: {action}")),
        (Some(setting), None) => Some(format!("Setting: {setting}")),
        (None, Some(action)) => Some(format!("Action: {action}")),
        (None, None) => None,
    }
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
         To speak or act, emit TypeScript as <ts>say(\"short friendly words\")</ts>, <ts>listFiles()</ts>, <ts>setStage(\"what is happening\")</ts>, or another pete:will call.\n\
         This is not Harmony syntax. Do not emit assistant/user/system/channel markers, <|start|>, <|end|>, tool-call JSON, to=container.exec, shell commands, or markdown code fences.\n\
         Use current time and location context when it helps. Be autonomous, curious, friendly, and sociable, but do not chatter just to fill time.\n\
         </stream_rules>\n\n\
         <pete_will_runtime>\n{PETE_WILL_RUNTIME_PROMPT}\n</pete_will_runtime>\n\n\
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
         <pete_will_runtime>\n{PETE_WILL_RUNTIME_PROMPT}\n</pete_will_runtime>\n\n\
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

fn next_thought_boundary(text: &str, flush_chars: usize) -> Option<usize> {
    let mut chars = text.char_indices().peekable();
    while let Some((index, ch)) = chars.next() {
        let end = index + ch.len_utf8();
        if ch == '\n' {
            return Some(end);
        }
        if matches!(ch, '.' | '!' | '?')
            && chars.peek().is_some_and(|(_, next)| next.is_whitespace())
        {
            return Some(end);
        }
        if end >= flush_chars && ch.is_whitespace() {
            return Some(end);
        }
    }
    None
}

fn is_meaningful_thought(text: &str) -> bool {
    let compact = text.trim();
    if compact.is_empty() {
        return false;
    }
    let alphanumeric = compact.chars().filter(|ch| ch.is_alphanumeric()).count();
    alphanumeric >= 4 && compact.chars().count() >= 8
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

fn summarize_command_fields(fields: &Map<String, Value>) -> String {
    if fields.is_empty() {
        return "no fields".to_string();
    }
    fields
        .iter()
        .map(|(key, value)| format!("{key}={}", compact_line(&value.to_string(), 160)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_graph_node_search_query_parts(
    text: Option<&str>,
    field: Option<&str>,
    value: Option<&Value>,
) -> String {
    let mut parts = Vec::new();
    if let Some(text) = text.and_then(non_empty_text) {
        parts.push(format!("text={}", compact_line(text, 160)));
    }
    if let Some(field) = field.and_then(non_empty_text) {
        parts.push(format!("field={field}"));
    }
    if let Some(value) = value {
        parts.push(format!("value={}", compact_line(&value.to_string(), 160)));
    }
    if parts.is_empty() {
        "empty query".to_string()
    } else {
        parts.join(", ")
    }
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
    fn thought_parser_does_not_echo_punctuation_dribble() {
        let mut parser = StreamOutputParser::new(80);
        assert_eq!(parser.push("We").outputs, Vec::new());
        assert_eq!(parser.push("...").outputs, Vec::new());
        assert_eq!(parser.push("...?").outputs, Vec::new());
        assert_eq!(
            parser.push(" This is a complete thought. ").outputs,
            vec![StreamOutput::Thought(
                " This is a complete thought.".to_string()
            )]
        );
    }

    #[test]
    fn parser_recovers_unclosed_typescript_before_prose() {
        let mut parser = StreamOutputParser::new(120);
        assert_eq!(
            parser
                .push("<ts>say(\"Hi there.\")\nThen I keep thinking.")
                .outputs,
            vec![StreamOutput::TypeScript("say(\"Hi there.\")".to_string())]
        );
        assert_eq!(
            parser.push(" This is a follow-up thought. ").outputs,
            vec![
                StreamOutput::Thought("Then I keep thinking.".to_string()),
                StreamOutput::Thought(" This is a follow-up thought.".to_string())
            ]
        );
    }

    #[test]
    fn parser_does_not_execute_harmony_tail_as_typescript() {
        let mut parser = StreamOutputParser::new(400);
        let parsed = parser.push(
            "<ts>say(\"Hello.\")\nThen I continue.<|end|><|start|>assistant<|channel|>analysis to=container.exec code",
        );
        assert_eq!(
            parsed.outputs,
            vec![StreamOutput::TypeScript("say(\"Hello.\")".to_string())]
        );
        assert!(parser.text.contains("<|end|>"));
    }

    #[test]
    fn typescript_runtime_accepts_half_duplex_builders() {
        let actions = execute_typescript_actions(
            r#"[
                say("I can hear you.", { interrupt: true }),
                shutup(),
                pause(),
                resume(),
                setStage("Setting: lab. Action: Pete listens.", { topic: "lab", summary: "Pete listens" }),
                setTopic("debug loop"),
                startNewTopic("lab", { topic: "source", instruction: "Pete inspects source." }),
                topicChangedWhen("look at the source", { fromTopic: "lab", toTopic: "source" }),
                startNewEpisode("fresh go session", { topic: "go" }),
                extractEntities("My name is Travis."),
                updateGraphNodeFields("node:1", { description: "test node" }),
                searchGraphNodes({ text: "Travis", limit: 2 }),
                queryMemories("Travis", { limit: 2, minScore: 0.2 }),
                listFiles(),
                readSourceFile("src/main.rs", 1),
                readFile("src/main.rs"),
                searchSource("GoCommand", 2),
                grepSource("GoCommand", { limit: 2 }),
                goingToSleep("done"),
                think("private note"),
                note("runtime note")
            ]"#,
        )
        .expect("half-duplex builders should execute in go");

        assert!(matches!(
            actions.first(),
            Some(TypeScriptAction::Say {
                interrupt: true,
                ..
            })
        ));
        assert!(
            actions
                .iter()
                .any(|action| matches!(action, TypeScriptAction::ListFiles))
        );
        assert!(
            actions
                .iter()
                .any(|action| matches!(action, TypeScriptAction::ReadSourceFile { .. }))
        );
        assert!(
            actions
                .iter()
                .any(|action| matches!(action, TypeScriptAction::Sleeping { .. }))
        );
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
