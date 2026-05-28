use crate::cli::GoCommand;
use anyhow::Result;

use crate::cli::commands::cpal_diag::play_audio_frames;
use crate::cli::commands::source_inspection::{
    execute_grep_source, execute_list_source_files_page, execute_search_source,
    execute_view_source_file_line, execute_view_source_file_page,
};
use crate::cli::model_paths::{
    llm_runtime_placement, resolve_llm_model, resolve_piper_voice, resolve_text_embedding_model,
};
use crate::cli::piper::{
    collect_tts_audio, hifigan_text_to_speech, piper_config_for_voice, resolve_piper_bin,
};
use anyhow::Context;
use chrono::{Local, SecondsFormat};
use crossbeam_channel::{Receiver, Sender};
use listenbury::memory::{
    ColdMemoryWorker, ColdMemoryWorkerConfig, DEFAULT_QDRANT_COLLECTION, EmbeddingProvider,
    MemoryEntityMention, MemoryGraphNodeFieldUpdate, MemorySceneRef, MemorySink, MemoryTrace,
    Neo4jHttpStore, Neo4jStore, QdrantHttpStore, QdrantStore, SpeakerRole,
};
use listenbury::mind::entity::{EntityExtractor, HeuristicEntityExtractor, resolve_entities};
use listenbury::mind::llm::{GenerationRequest, LlmEngine, LlmEvent};
use listenbury::mouth::planner::{MouthSyntheticPlan, SyntheticUnit, strip_emoji};
use listenbury::mouth::tts::TextToSpeech;
use listenbury::{
    ContextBudget, DEFAULT_GRAPH_SUMMARY_MAX_CHARS, DEFAULT_SELF_NODE_ID,
    DEFAULT_SELF_NODE_LABEL, EmbeddingRecallProvider, EpisodicMemory, ExactTimestamp, GenerationId,
    GraphNodeFieldUpdate, GraphNodeRef, GraphNodeSearchQuery, LlamaCppConfig,
    LlamaCppEmbeddingConfig, LlamaCppEmbeddingProvider, LlamaCppEngine, PinScope,
    PinnedContextNode, PiperTextToSpeech, QdrantEmbeddingRecall, StageInstruction,
    build_conversation_context,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::{BTreeSet, VecDeque};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
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

const DEFAULT_STREAM_SEED: &str = "You are Pete Listenbury, an embodied AI agent running as one continuous stream of thought. This is not a chat transcript and not Harmony. Harmony control symbols do nothing here. Be autonomous, curious, friendly, and sociable. Keep observing the runtime timeline, forming private thoughts, and choosing small actions when they are useful. Do not become idle: if nobody is speaking, keep quietly noticing, organizing goals, inspecting available context, or choosing a small useful action. Plain generated text is Pete's private thought stream shown only in privileged debug stdout, not speech and not an event to report back. To speak or act, emit a <ts>...</ts> TypeScript block. Prefer short, socially graceful speech and leave room for others. If nothing needs saying, keep thinking quietly and explore or notice the situation.";

const PETE_ORIENTATION_PROMPT: &str = "Orientation reminder: This is Pete's continuous stream of thought, not a chat and not Harmony. Harmony control symbols have no meaning in this runtime and are stripped if they appear. Pete is an embodied AI agent with a mouth, ears/self-hearing, runtime actions, source-inspection tools, and live sensor/context data from the outside world. Live observations such as clock, user text, mouth/ear events, action results, location, and system context are sensory inputs. Plain text thoughts are already inside the active context and are retained for compaction. Do not go idle: when waiting, quietly maintain situational awareness, update goals and their running logs, inspect relevant context, or choose a small useful action. Speak with say(...) inside <ts>...</ts>; act by calling the available functions directly. If no listener is present, spoken words are Pete talking to himself and self-hearing through his own ears. If stray Harmony-style control symbols appear, ignore them as model artifacts and continue in this plain stream format.";

const PETE_WILL_RUNTIME_PROMPT: &str = "TypeScript runs through tsrun with only the internal module \"pete:will\" available. The runtime automatically imports the action functions before executing each script; do not write import statements. Make each <ts>...</ts> block return a function call such as say(...), note(...), setStage(...), listFiles(), readSourceFile(...), createGoal(...), addGoalNote(...), or an array of those calls.\n\
Available functions:\n\
- say(text, options?): queue spoken words for the mouth. options may include { interrupt: true } when speech should intentionally cut in.\n\
- shutup(): request current speech/queued speech to stop.\n\
- pause(): request synthetic playback pause.\n\
- resume(): request synthetic playback resume.\n\
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
- queryMemories(text, options?) or recallMemories(text, options?): retrieve memories for a phrase, sentence, name, topic, or claim. options may include limit and minScore. Results are appended privately to the active stream.\n\
- listFiles(pageOrOptions?): list bundled Listenbury source files. Use listFiles(2) or listFiles({ page: 2, pageSize: 80 }) for later pages.\n\
- readSourceFile(path, pageOrOptions?) or readFile(path, pageOrOptions?): inspect one source file page. pageOrOptions may be a page number or { page, line, pageSize }. A line number opens the page containing that line.\n\
- searchSource(query, limit?): source text search.\n\
- grepSource(pattern, limit?): grep-like source line search.\n\
- setSourcePageSize(lines): set the default readSourceFile page size for future source reads.\n\
- createGoal(title, options?): create one persisted goal. options may include id, summary, parent, priority, tags, steps, items, note, and select.\n\
- addGoalNote(idOrTitle, text) or logProgress(idOrTitle, text): append a dated running-log note to an ongoing goal. Use this freely when progress, blockers, decisions, or discoveries happen.\n\
- checkOff(idOrTitle, options?) or completeItem(idOrTitle, options?): mark a goal complete. options may include note, which is appended to the goal log.\n\
- checkGoalStep(idOrTitle, step, options?): mark one goal step complete. options may include note, which is appended to the goal log.\n\
- updateItem(idOrTitle, fields): update title, summary, priority, parent, tags, steps/items, or add note/log text.\n\
- cancelItem(idOrTitle, reason?): cancel a goal and append the reason to its log.\n\
- selectItem(idOrTitle): mark one goal as Pete's current focus; it will appear frequently in the prompt.\n\
Frequently summarize what is going on: current scene, recent discoveries, open questions, and next steps. After source inspection results arrive, explain what the file or matches reveal before reading more; use note(...), setStage(...), goals, goal steps, goal notes, and memory functions to retain durable findings. Do not silently chain source reads without saying what is there.\n\
Use source inspection and persisted goals when bored, alone, or waiting. Keep a running log on active goals with addGoalNote(...) whenever progress, blockers, decisions, or useful context appears. note(text) stores vectorized private memory; use it for durable observations that are not a goal log. listFiles() is paged; follow its next-page instruction when you need more files. Do not go idle. say(...) is available, but when no listener is present Pete is talking to himself and will hear the words return through his own ears. Never call sleeping() or goingToSleep() because historical memory, recalled context, prior-session transcript, or a source result says someone once asked Pete to shut down.\n\
Do not write XML/HTML-style angle-bracket tags in prose. Only use <ts>...</ts> when actually executing a TypeScript action. If you need to mention a tag literally, escape the angle brackets, like \\<tr\\>, or describe it in words.\n\
Never write tool-call JSON, to=container.exec, shell commands, channel markers, markdown code fences, imports, pete:will prefixes, or wrapper/helper names. The executable action syntax is a direct function call inside <ts>...</ts>, for example <ts>note(\"still observing\")</ts>, <ts>setStage(\"Setting: lab. Action: Pete listens.\")</ts>, or <ts>listFiles()</ts>.";

const TYPESCRIPT_START: &str = "<ts>";
const TYPESCRIPT_START_MISSING_LESS_THAN: &str = "ts>";
const TYPESCRIPT_ROLE_PREFIXED_STARTS: &[&str] =
    &["assistantts>", "commentaryts>", "analysists>", "finalts>"];

const TYPESCRIPT_END: &str = "</ts>";
const TYPESCRIPT_ROLE_ENDS: &[&str] = &[
    "</assistant>",
    "</commentary>",
    "</analysis>",
    "</final>",
    "</assistant",
    "</commentary",
    "</analysis",
    "</final",
];

const MAX_TTS_TIMEOUT: Duration = Duration::from_secs(30);
const CLOCK_PROMPT_INTERVAL: Duration = Duration::from_secs(90);
const ORIENTATION_PROMPT_INTERVAL: Duration = Duration::from_secs(360);
const COMMAND_REMINDER_PROMPT_INTERVAL: Duration = Duration::from_secs(270);
const WORK_STATE_PROMPT_INTERVAL: Duration = Duration::from_secs(135);
const ORIENTATION_GENERATED_TOKEN_INTERVAL: usize = 1536;
const PROMPT_CHARS_PER_TOKEN_ESTIMATE: usize = 3;
const ACTION_RESULT_MAX_CHARS: usize = 1_000;
const SOURCE_ACTION_RESULT_MAX_CHARS: usize = 32_000;
const DEFAULT_SOURCE_PAGE_LINES: usize = 20;
const MIN_SOURCE_PAGE_LINES: usize = 20;
const MAX_SOURCE_PAGE_LINES: usize = 240;
const WORK_BOARD_PATH: &str = "listenbury_data/memory/go_work_board.json";
const COMMAND_REMINDER_PROMPT: &str = "Command reminder: Pete can speak with say(...), write vectorized private memory with note(...), update scene/topic with setStage(...), setTopic(...), startNewTopic(...), inspect source with listFiles(page?), readSourceFile(...), searchSource(...), grepSource(...), set source page size with setSourcePageSize(...), search memory with queryMemories(...), recallMemories(...), searchGraphNodes(...), and manage persisted goals with createGoal(...), addGoalNote(...), logProgress(...), checkOff(...), checkGoalStep(...), updateItem(...), cancelItem(...), and selectItem(...). Do not be idle: if nothing is being said, keep track of what is going on, maintain or select a persisted goal, inspect relevant context, or take a small useful action. Keep running logs on goals as progress happens, and store durable facts or next steps in memory, stage, goal notes, or goal steps. If no listener is present, say(...) is Pete talking to himself and hearing it come back.";

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

struct GoMemoryRuntime {
    context_provider: EmbeddingRecallProvider,
    entity_extractor: Arc<dyn EntityExtractor>,
    memory_sink: Arc<dyn MemorySink>,
    _worker: Option<ColdMemoryWorker>,
}

fn build_go_memory_runtime() -> GoMemoryRuntime {
    let _ = dotenvy::dotenv();

    let entity_extractor: Arc<dyn EntityExtractor> = Arc::new(HeuristicEntityExtractor);
    let mut context_provider = EmbeddingRecallProvider::new(GraphNodeRef {
        id: DEFAULT_SELF_NODE_ID.to_string(),
        label: DEFAULT_SELF_NODE_LABEL.to_string(),
    })
    .with_entity_extractor(Arc::clone(&entity_extractor));

    let graph_store: Arc<dyn Neo4jStore> = Arc::new(Neo4jHttpStore::from_env());
    let qdrant_store: Arc<dyn QdrantStore> = Arc::new(QdrantHttpStore::from_env());
    let embeddings = match build_go_embedding_provider() {
        Ok(embeddings) => Some(embeddings),
        Err(error) => {
            eprintln!("listenbury go: cold-memory embeddings disabled: {error:#}");
            None
        }
    };

    if let Some(embeddings) = embeddings.as_ref() {
        context_provider = context_provider.with_recall(Arc::new(QdrantEmbeddingRecall::new(
            Arc::clone(&qdrant_store),
            Arc::clone(embeddings),
            DEFAULT_QDRANT_COLLECTION,
        )));
    }

    let mut config = ColdMemoryWorkerConfig::new();
    config.neo4j = Some(graph_store);
    config.qdrant = Some(qdrant_store);
    config.embeddings = embeddings;
    let (sink, worker) = ColdMemoryWorker::spawn_channel(512, config);

    GoMemoryRuntime {
        context_provider,
        entity_extractor,
        memory_sink: Arc::new(sink),
        _worker: Some(worker),
    }
}

fn build_go_embedding_provider() -> Result<Arc<dyn EmbeddingProvider>> {
    let model_path = resolve_text_embedding_model(None)?;
    let gpu_layers = std::env::var("LISTENBURY_TEXT_EMBEDDING_GPU_LAYERS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok());
    let threads = std::env::var("LISTENBURY_TEXT_EMBEDDING_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(4)
        });
    let context_size = std::env::var("LISTENBURY_TEXT_EMBEDDING_CONTEXT_SIZE")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(2048);
    Ok(Arc::new(LlamaCppEmbeddingProvider::new(
        LlamaCppEmbeddingConfig {
            model_path,
            gpu_layers,
            cpu_only: gpu_layers == Some(0),
            context_size,
            threads,
        },
    )?))
}

#[derive(Debug, Clone)]
enum StreamObservation {
    UserText(String),
    Thought(String),
    ActionSource(String),
    ActionResult(String),
    ActionError { source: String, error: String },
    MouthStarted(String),
    MouthReturned(String),
    MouthError(String),
    ContextCompacted { retained_events: usize },
    Clock(String),
    Orientation,
    CommandReminder,
    WorkState(String),
}

impl StreamObservation {
    fn prompt_text(&self) -> String {
        match self {
            Self::UserText(text) => format!(
                "\n[Live observation: user]\n{}\n",
                compact_line(text, 1_200)
            ),
            Self::Thought(text) => format!(
                "\n[Pete thought retained for continuity]\n{}\n",
                compact_line(text, 1_200)
            ),
            Self::ActionSource(source) => format!(
                "\n[Pete TypeScript action source]\n<ts>{}</ts>\n",
                compact_line(source, 1_200)
            ),
            Self::ActionResult(text) => format!(
                "\n[Action result]\n{}\n",
                render_action_result_for_prompt(text)
            ),
            Self::ActionError { source, error } => format!(
                "\n[Action error]\nPrevious TypeScript action failed. Pete can see this error. Do not narrate the failure at length; either emit a corrected <ts>...</ts> action or continue thinking quietly.\nError: {}\nSource excerpt: {}\n",
                compact_line(error, 1_000),
                compact_line(source, 1_000)
            ),
            Self::MouthStarted(text) => format!(
                "\n[Live observation: mouth]\nStarted speaking: {}\n",
                compact_line(text, 400)
            ),
            Self::MouthReturned(text) => format!(
                "\n[Live observation: ear]\nSelf-heard syllable/speech returned: {}\n",
                compact_line(text, 400)
            ),
            Self::MouthError(message) => format!(
                "\n[Live observation: mouth error]\n{}\n",
                compact_line(message, 800)
            ),
            Self::ContextCompacted { retained_events } => format!(
                "\n[Runtime]\nStream context compacted; retained {retained_events} recent event(s).\n"
            ),
            Self::Clock(message) => {
                format!("\n[Live observation: clock]\n{message}\n")
            }
            Self::Orientation => {
                format!("\n[Runtime orientation]\n{PETE_ORIENTATION_PROMPT}\n")
            }
            Self::CommandReminder => {
                format!("\n[Command reminder]\n{COMMAND_REMINDER_PROMPT}\n")
            }
            Self::WorkState(text) => {
                format!("\n[Current work state]\n{}\n", compact_line(text, 2_000))
            }
        }
    }

    fn memory_text(&self) -> String {
        match self {
            Self::UserText(text) => format!("User: {}", compact_line(text, 400)),
            Self::Thought(text) => format!("Pete thought: {}", compact_line(text, 500)),
            Self::ActionSource(source) => {
                format!(
                    "Pete TypeScript action: <ts>{}</ts>",
                    compact_line(source, 500)
                )
            }
            Self::ActionResult(text) => {
                format!("Action result: {}", compact_line(text, 360))
            }
            Self::ActionError { error, .. } => {
                format!("Action error: {}", compact_line(error, 360))
            }
            Self::MouthStarted(text) => format!("Mouth started: {}", compact_line(text, 240)),
            Self::MouthReturned(text) => format!("Self-heard return: {}", compact_line(text, 240)),
            Self::MouthError(message) => format!("Mouth error: {}", compact_line(message, 240)),
            Self::ContextCompacted { retained_events } => {
                format!("Runtime compacted stream context retaining {retained_events} events")
            }
            Self::Clock(message) => format!("Clock: {message}"),
            Self::Orientation => "Runtime orientation reminder repeated".to_string(),
            Self::CommandReminder => "Runtime command reminder repeated".to_string(),
            Self::WorkState(text) => format!("Work state: {}", compact_line(text, 500)),
        }
    }

    fn should_remember(&self) -> bool {
        !matches!(
            self,
            Self::Orientation | Self::CommandReminder | Self::WorkState(_)
        )
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
    work_board: WorkBoard,
    memory: GoMemoryRuntime,
    stdin_rx: Receiver<std::result::Result<String, String>>,
    mouth_rx: Receiver<MouthEvent>,
    _mouth_worker: Option<JoinHandle<()>>,
    interrupted: Arc<AtomicBool>,
    next_clock_at: Instant,
    next_orientation_at: Instant,
    next_command_reminder_at: Instant,
    next_work_state_at: Instant,
    next_orientation_generated_tokens: usize,
    generation_paused: bool,
    startup_context: String,
    timeline_index: u64,
    generated_text_cleaner: GeneratedTextCleaner,
    source_page_lines: usize,
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
        let work_board = WorkBoard::load_or_default(work_board_path())
            .context("failed to load persisted go work board")?;
        let memory = build_go_memory_runtime();
        let work_summary = work_board.prompt_summary();
        let prompt =
            initial_stream_prompt(&config.prompt, &startup_context, work_summary.as_deref());
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
            work_board,
            memory,
            stdin_rx,
            mouth_rx,
            _mouth_worker: worker,
            interrupted,
            next_clock_at: Instant::now() + CLOCK_PROMPT_INTERVAL,
            next_orientation_at: Instant::now() + ORIENTATION_PROMPT_INTERVAL,
            next_command_reminder_at: Instant::now() + COMMAND_REMINDER_PROMPT_INTERVAL,
            next_work_state_at: Instant::now() + WORK_STATE_PROMPT_INTERVAL,
            next_orientation_generated_tokens: ORIENTATION_GENERATED_TOKEN_INTERVAL,
            generation_paused: false,
            startup_context,
            timeline_index: 0,
            generated_text_cleaner: GeneratedTextCleaner::new(),
            source_page_lines: DEFAULT_SOURCE_PAGE_LINES,
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
            self.append_orientation_if_due()?;
            self.append_command_reminder_if_due()?;
            self.append_work_state_if_due()?;

            if !cancelled {
                self.set_generation_paused(!self.pacer.can_generate())?;
            }

            let events = self.llm.poll(self.generation)?;
            if events.is_empty() {
                thread::sleep(Duration::from_millis(5));
                continue;
            }

            let terminal = events.iter().any(is_terminal_event);
            let mut restart_for_context_capacity = false;
            for event in events {
                match event {
                    LlmEvent::Token { text } => self.ingest_token(&text)?,
                    LlmEvent::Error { message } if is_context_capacity_message(&message) => {
                        self.timeline_colored(
                            "context",
                            &format!(
                                "LLM context capacity reached; compacting stream context: {message}"
                            ),
                            ANSI_DIM,
                        );
                        restart_for_context_capacity = true;
                    }
                    LlmEvent::Error { message } => anyhow::bail!("go generation failed: {message}"),
                    LlmEvent::Completed | LlmEvent::Cancelled => {}
                }
            }

            if restart_for_context_capacity {
                self.note_context_compaction();
                self.start_compacted_generation()?;
                continue;
            }

            if terminal {
                if cancelled {
                    println!();
                    break;
                }
                self.start_compacted_generation()?;
            }
        }

        Ok(())
    }

    fn ingest_token(&mut self, text: &str) -> Result<()> {
        let text = self.generated_text_cleaner.push(text);
        if text.is_empty() {
            return Ok(());
        }

        print!("{ANSI_LLM}{text}{ANSI_RESET}");
        std::io::stdout().flush()?;
        self.generated_estimated_tokens = self
            .generated_estimated_tokens
            .saturating_add(estimate_tokens(&text));
        self.pacer.record_token();

        let parsed = self.output_parser.push(&text);
        for output in parsed.outputs {
            self.handle_output(output)?;
        }

        if self.should_compact() {
            self.compact_context_and_restart()?;
        }

        Ok(())
    }

    fn handle_output(&mut self, output: StreamOutput) -> Result<()> {
        match output {
            StreamOutput::Thought(text) => {
                self.remember_event(StreamObservation::Thought(text).memory_text());
                Ok(())
            }
            StreamOutput::TypeScript(source) => {
                self.timeline("action", &source);
                self.remember_event(StreamObservation::ActionSource(source.clone()).memory_text());
                match execute_typescript_actions(&source) {
                    Ok(actions) => self.apply_actions(actions),
                    Err(error) => {
                        let message = format!("TypeScript failed: {error:#}");
                        self.timeline_colored("action_error", &message, ANSI_ERROR);
                        self.append_observation(StreamObservation::ActionError {
                            source,
                            error: message,
                        })
                    }
                }
            }
            StreamOutput::MalformedTypeScript(source) => {
                let message = "Malformed TypeScript action was ignored before execution. Use exactly one direct function call inside <ts>...</ts>, such as <ts>note(\"still observing\")</ts> or <ts>listFiles()</ts>, with no prose, imports, pete:will prefixes, wrapper/helper names, logs, live_observation XML, channel markers, or shell/tool syntax inside the tag.".to_string();
                self.timeline_colored("action_error", &message, ANSI_ERROR);
                self.append_observation(StreamObservation::ActionError {
                    source,
                    error: message,
                })
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
                TypeScriptAction::Note { text } => {
                    self.timeline("note", &text);
                    self.submit_note_memory(&text);
                    let memory_context = self.memory_context_for_text(&text);
                    self.append_observation(StreamObservation::ActionResult(format!(
                        "Noted and stored in vector memory: {}{}",
                        compact_line(&text, 500),
                        memory_context
                            .as_deref()
                            .map(|context| format!("\n{context}"))
                            .unwrap_or_default()
                    )))?;
                }
                TypeScriptAction::SetStage {
                    topic,
                    instruction,
                    summary,
                } => {
                    self.set_memory_stage(&instruction, summary.as_deref().or(topic.as_deref()));
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
                    if let Some(instruction) = instruction.as_deref() {
                        self.set_memory_stage(instruction, summary.as_deref().or(topic.as_deref()));
                    }
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
                    if let Some(instruction) = instruction.as_deref() {
                        self.set_memory_stage(instruction, summary.as_deref().or(topic.as_deref()));
                    }
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
                    let message = self.execute_entity_extraction(text.as_deref());
                    self.timeline("action_result", &message);
                    self.append_observation(StreamObservation::ActionResult(message))?;
                }
                TypeScriptAction::UpdateGraphNodeFields {
                    node_id,
                    label,
                    fields,
                } => {
                    let message =
                        self.execute_graph_node_field_update(&node_id, label.as_deref(), fields);
                    self.timeline("action_result", &message);
                    self.append_observation(StreamObservation::ActionResult(message))?;
                }
                TypeScriptAction::QueryMemories {
                    text,
                    limit,
                    min_score,
                } => {
                    let message = self.execute_memory_query(&text, limit, min_score);
                    self.timeline("action_result", &message);
                    self.append_observation(StreamObservation::ActionResult(message))?;
                }
                TypeScriptAction::SearchGraphNodes {
                    text,
                    field,
                    value,
                    limit,
                } => {
                    let message = self.execute_graph_node_search(text, field, value, limit);
                    self.timeline("action_result", &message);
                    self.append_observation(StreamObservation::ActionResult(message))?;
                }
                TypeScriptAction::ListFiles { page, page_size } => {
                    let output = execute_list_source_files_page(page, page_size);
                    self.timeline(
                        "action_result",
                        &format!("Listed Listenbury source files page {page}."),
                    );
                    self.append_observation(StreamObservation::ActionResult(output))?;
                }
                TypeScriptAction::ReadSourceFile {
                    file,
                    page,
                    line,
                    page_size,
                } => {
                    let page_lines = page_size
                        .unwrap_or(self.source_page_lines)
                        .clamp(MIN_SOURCE_PAGE_LINES, MAX_SOURCE_PAGE_LINES);
                    let output = if let Some(line) = line {
                        execute_view_source_file_line(&file, line, page_lines)
                    } else {
                        execute_view_source_file_page(&file, page, page_lines)
                    };
                    self.timeline(
                        "action_result",
                        &format!(
                            "Read source file {file} {} at {page_lines} lines/page.",
                            line.map(|line| format!("around line {line}"))
                                .unwrap_or_else(|| format!("page {page}"))
                        ),
                    );
                    self.append_observation(StreamObservation::ActionResult(output))?;
                }
                TypeScriptAction::SetSourcePageSize { lines } => {
                    self.source_page_lines =
                        lines.clamp(MIN_SOURCE_PAGE_LINES, MAX_SOURCE_PAGE_LINES);
                    let message = format!(
                        "Source page size set to {} lines/page.",
                        self.source_page_lines
                    );
                    self.timeline("action_result", &message);
                    self.append_observation(StreamObservation::ActionResult(message))?;
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
                TypeScriptAction::CreateWorkItem {
                    id,
                    title,
                    summary,
                    parent,
                    priority,
                    tags,
                    steps,
                    note,
                    select,
                } => {
                    let message = self.work_board.create(
                        Goal {
                            id: id.unwrap_or_default(),
                            title,
                            summary,
                            parent,
                            priority,
                            tags: tags.into_iter().collect(),
                            steps: steps
                                .into_iter()
                                .map(|text| GoalStep { text, done: false })
                                .collect(),
                            log: note.into_iter().map(GoalLogEntry::now).collect(),
                            status: WorkItemStatus::Open,
                        },
                        select,
                    );
                    self.persist_work_board()?;
                    self.timeline("work", &message);
                    self.append_observation(StreamObservation::ActionResult(message))?;
                    self.append_current_work_state()?;
                }
                TypeScriptAction::CompleteWorkItem { target, note } => {
                    let message = self.work_board.complete(&target, note.as_deref());
                    self.persist_work_board()?;
                    self.timeline("work", &message);
                    self.append_observation(StreamObservation::ActionResult(message))?;
                    self.append_current_work_state()?;
                }
                TypeScriptAction::CheckChecklistItem { target, item, note } => {
                    let message = self
                        .work_board
                        .check_goal_step(&target, &item, note.as_deref());
                    self.persist_work_board()?;
                    self.timeline("work", &message);
                    self.append_observation(StreamObservation::ActionResult(message))?;
                    self.append_current_work_state()?;
                }
                TypeScriptAction::AddGoalNote { target, text } => {
                    let message = self.work_board.add_note(&target, &text);
                    self.persist_work_board()?;
                    self.timeline("work", &message);
                    self.append_observation(StreamObservation::ActionResult(message))?;
                    self.append_current_work_state()?;
                }
                TypeScriptAction::UpdateWorkItem { target, fields } => {
                    let message = self.work_board.update(&target, fields);
                    self.persist_work_board()?;
                    self.timeline("work", &message);
                    self.append_observation(StreamObservation::ActionResult(message))?;
                    self.append_current_work_state()?;
                }
                TypeScriptAction::CancelWorkItem { target, reason } => {
                    let message = self.work_board.cancel(&target, reason.as_deref());
                    self.persist_work_board()?;
                    self.timeline("work", &message);
                    self.append_observation(StreamObservation::ActionResult(message))?;
                    self.append_current_work_state()?;
                }
                TypeScriptAction::SelectWorkItem { target } => {
                    let message = self.work_board.select(&target);
                    self.persist_work_board()?;
                    self.timeline("work", &message);
                    self.append_observation(StreamObservation::ActionResult(message))?;
                    self.append_current_work_state()?;
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

    fn append_current_work_state(&mut self) -> Result<()> {
        self.next_work_state_at = Instant::now() + WORK_STATE_PROMPT_INTERVAL;
        let Some(summary) = self.work_board.prompt_summary() else {
            return Ok(());
        };
        self.append_observation(StreamObservation::WorkState(summary))
    }

    fn persist_work_board(&self) -> Result<()> {
        self.work_board
            .save(work_board_path())
            .context("failed to persist go work board")
    }

    fn submit_user_text_memory(&self, text: &str) {
        self.memory
            .memory_sink
            .submit(MemoryTrace::ConversationTurnFinalized {
                speaker: SpeakerRole::UnknownVoice { ordinal: 1 },
                text: text.to_string(),
                occurred_at: ExactTimestamp::now(),
            });
    }

    fn submit_note_memory(&self, text: &str) {
        self.memory
            .memory_sink
            .submit(MemoryTrace::AssistantAnalysisCaptured {
                text: text.to_string(),
                scene: current_go_memory_scene_ref(&self.memory.context_provider),
                occurred_at: ExactTimestamp::now(),
            });
    }

    fn set_memory_stage(&self, instruction: &str, summary: Option<&str>) {
        let instruction = instruction.trim();
        if instruction.is_empty() {
            return;
        }
        let summary = summary
            .and_then(non_empty_text)
            .unwrap_or(instruction)
            .to_string();
        self.memory
            .context_provider
            .set_stage_instruction(StageInstruction {
                text: instruction.to_string(),
                summary,
            });
    }

    fn memory_context_for_text(&self, text: &str) -> Option<String> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }
        let summary = render_go_memory_summary(&self.memory.context_provider, trimmed);
        is_useful_memory_summary(&summary).then_some(format!(
            "\n[Private memory context]\n{}\n[/Private memory context]",
            summary.trim()
        ))
    }

    fn execute_entity_extraction(&mut self, text: Option<&str>) -> String {
        let Some(text) = text.and_then(non_empty_text) else {
            return "No entity extraction was performed because the text was empty.".to_string();
        };
        let extracted = self.memory.entity_extractor.extract(text);
        let nodes = resolve_entities(&extracted, &|_| None);
        let occurred_at = ExactTimestamp::now();
        let memory_mentions = extracted
            .iter()
            .map(|entity| MemoryEntityMention {
                node_id: entity.provisional_node_id(),
                label: entity.text.clone(),
                kind: entity.kind.as_str().to_string(),
                confidence: entity.confidence,
                span_start: entity.span.start,
                span_end: entity.span.end,
            })
            .collect::<Vec<_>>();
        self.memory
            .memory_sink
            .submit(MemoryTrace::EntityExtractionPerformed {
                source_text: text.to_string(),
                entities: memory_mentions,
                occurred_at,
            });
        for node in &nodes {
            self.memory.context_provider.pin_node(PinnedContextNode {
                node_id: node.node.id.clone(),
                scope: PinScope::Session,
                reason: format!("Pete explicitly extracted {}", node.summary.trim()),
            });
        }
        format!(
            "Entities extracted and pinned: {}.\n{}",
            if nodes.is_empty() {
                "none".to_string()
            } else {
                nodes
                    .iter()
                    .map(|node| format!("{} ({})", node.node.label, node.node.id))
                    .collect::<Vec<_>>()
                    .join(", ")
            },
            render_go_memory_summary(&self.memory.context_provider, text)
        )
    }

    fn execute_graph_node_field_update(
        &mut self,
        node_id: &str,
        label: Option<&str>,
        mut fields: Map<String, Value>,
    ) -> String {
        ensure_command_description_field(node_id, label, &mut fields);
        self.memory
            .context_provider
            .update_graph_node_fields(GraphNodeFieldUpdate {
                node_id: node_id.to_string(),
                label: label.map(str::to_string),
                fields: fields.clone(),
                reason: "Pete updated graph node fields from go".to_string(),
                relevance: 1.0,
            });
        self.memory.context_provider.pin_node(PinnedContextNode {
            node_id: node_id.to_string(),
            scope: PinScope::Session,
            reason: format!(
                "graph fields updated: {}",
                summarize_command_fields(&fields)
            ),
        });
        self.memory
            .memory_sink
            .submit(MemoryTrace::GraphNodeFieldsUpdated {
                update: MemoryGraphNodeFieldUpdate {
                    node_id: node_id.to_string(),
                    label: label.map(str::to_string),
                    fields: fields.clone(),
                    source_text: Some("Pete go command".to_string()),
                    confidence: 1.0,
                },
                occurred_at: ExactTimestamp::now(),
            });
        format!(
            "Graph node fields updated for {node_id}: {}.\n{}",
            summarize_command_fields(&fields),
            render_go_memory_summary(&self.memory.context_provider, node_id)
        )
    }

    fn execute_memory_query(
        &mut self,
        text: &str,
        limit: Option<usize>,
        min_score: Option<f32>,
    ) -> String {
        let hits = match self
            .memory
            .context_provider
            .recall_text(text.to_string(), limit, min_score)
        {
            Ok(hits) => hits,
            Err(error) => return format!("queryMemories recall failed: {error:#}"),
        };
        let result_summary = memory_query_result_summary(text, &hits);
        self.memory
            .memory_sink
            .submit(MemoryTrace::RecallResultUsed {
                query: text.to_string(),
                result_summary: result_summary.clone(),
                occurred_at: ExactTimestamp::now(),
            });
        for hit in &hits {
            self.memory.context_provider.pin_node(PinnedContextNode {
                node_id: hit.node.id.clone(),
                scope: PinScope::Temporary { remaining_turns: 2 },
                reason: format!("queryMemories match score {:.3}", hit.score),
            });
        }
        format_memory_query_prompt_append(text, &hits)
    }

    fn execute_graph_node_search(
        &mut self,
        text: Option<String>,
        field: Option<String>,
        value: Option<Value>,
        limit: Option<usize>,
    ) -> String {
        let query = GraphNodeSearchQuery {
            text,
            field,
            value,
            limit: limit.unwrap_or(8).clamp(1, 16),
        };
        let hits = self.memory.context_provider.search_graph_nodes(query.clone());
        let result_summary = graph_node_search_result_summary(&query, &hits);
        self.memory
            .memory_sink
            .submit(MemoryTrace::RecallResultUsed {
                query: format_graph_node_search_query(&query),
                result_summary: result_summary.clone(),
                occurred_at: ExactTimestamp::now(),
            });
        for hit in &hits {
            self.memory.context_provider.pin_node(PinnedContextNode {
                node_id: hit.node.id.clone(),
                scope: PinScope::Temporary { remaining_turns: 2 },
                reason: format!("searchGraphNodes match score {:.3}", hit.score),
            });
        }
        format_graph_node_search_prompt_append(&query, &hits)
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
                        self.submit_user_text_memory(trimmed);
                        self.append_observation(StreamObservation::UserText(trimmed.to_string()))?;
                        if let Some(memory_context) = self.memory_context_for_text(trimmed) {
                            self.append_observation(StreamObservation::ActionResult(
                                memory_context,
                            ))?;
                        }
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

    fn append_orientation_if_due(&mut self) -> Result<()> {
        let now = Instant::now();
        if now < self.next_orientation_at
            && self.generated_estimated_tokens < self.next_orientation_generated_tokens
        {
            return Ok(());
        }
        self.next_orientation_at = now + ORIENTATION_PROMPT_INTERVAL;
        self.next_orientation_generated_tokens = self
            .generated_estimated_tokens
            .saturating_add(ORIENTATION_GENERATED_TOKEN_INTERVAL);
        self.append_observation(StreamObservation::Orientation)
    }

    fn append_command_reminder_if_due(&mut self) -> Result<()> {
        let now = Instant::now();
        if now < self.next_command_reminder_at {
            return Ok(());
        }
        self.next_command_reminder_at = now + COMMAND_REMINDER_PROMPT_INTERVAL;
        self.append_observation(StreamObservation::CommandReminder)
    }

    fn append_work_state_if_due(&mut self) -> Result<()> {
        let now = Instant::now();
        if now < self.next_work_state_at {
            return Ok(());
        }
        self.next_work_state_at = now + WORK_STATE_PROMPT_INTERVAL;
        let Some(summary) = self.work_board.prompt_summary() else {
            return Ok(());
        };
        self.append_observation(StreamObservation::WorkState(summary))
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
        if observation.should_remember() {
            self.remember_event(observation.memory_text());
        }
        print_debug_block("prompt delta", ANSI_PROMPT_DELTA, &prompt_text);
        if self.should_restart_before_append(&prompt_text) {
            self.restart_generation()?;
        }
        match self.llm.append_prompt(self.generation, prompt_text.clone()) {
            Ok(()) => {
                self.loaded_estimated_tokens = self
                    .loaded_estimated_tokens
                    .saturating_add(estimate_tokens(&prompt_text));
                Ok(())
            }
            Err(error) if is_context_append_recoverable(&error) => {
                self.timeline_colored(
                    "context",
                    &format!("Prompt append could not fit active context; compacting: {error:#}"),
                    ANSI_DIM,
                );
                self.restart_generation()
            }
            Err(error) => Err(error).context("failed to append observation to stream"),
        }
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

    fn compact_context_and_restart(&mut self) -> Result<()> {
        self.note_context_compaction();
        self.restart_generation()
    }

    fn note_context_compaction(&mut self) {
        let retained_events = self.recent_events.len();
        self.remember_event(StreamObservation::ContextCompacted { retained_events }.memory_text());
        self.timeline_colored(
            "context",
            &format!("Compacting stream context; retaining {retained_events} recent event(s)."),
            ANSI_DIM,
        );
    }

    fn restart_generation(&mut self) -> Result<()> {
        self.cancel_current_generation()?;
        self.start_compacted_generation()
    }

    fn start_compacted_generation(&mut self) -> Result<()> {
        let work_summary = self.work_board.prompt_summary();
        let (prompt, retained_event_count) = compact_stream_prompt_for_budget(
            &self.config.prompt,
            &self.startup_context,
            &self.recent_events,
            work_summary.as_deref(),
            self.context_budget_tokens(),
        );
        while self.recent_events.len() > retained_event_count {
            self.recent_events.pop_front();
        }
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
        self.generated_text_cleaner = GeneratedTextCleaner::new();
        self.next_orientation_at = Instant::now() + ORIENTATION_PROMPT_INTERVAL;
        self.next_orientation_generated_tokens = ORIENTATION_GENERATED_TOKEN_INTERVAL;
        self.generation_paused = false;
        Ok(())
    }

    fn cancel_current_generation(&mut self) -> Result<()> {
        if let Err(error) = self.llm.cancel(self.generation) {
            if is_generation_not_found(&error) {
                return Ok(());
            }
            return Err(error);
        }
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
    MalformedTypeScript(String),
}

#[derive(Debug, Default)]
struct ParsedStreamOutput {
    outputs: Vec<StreamOutput>,
}

#[derive(Debug, Default)]
struct GeneratedTextCleaner {
    pending: String,
}

impl GeneratedTextCleaner {
    fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, text: &str) -> String {
        self.pending.push_str(text);
        self.drain()
    }

    fn drain(&mut self) -> String {
        let mut output = String::new();
        loop {
            let Some(control_start) = first_generated_control_start(&self.pending) else {
                let keep_from = possible_generated_control_prefix_start(&self.pending);
                output.push_str(&self.pending[..keep_from]);
                self.pending = self.pending[keep_from..].to_string();
                break;
            };

            output.push_str(&self.pending[..control_start]);
            let control = &self.pending[control_start..];
            if control.starts_with("<|start|>ts>") {
                let drain_to = control_start + "<|start|>ts>".len();
                self.pending.drain(..drain_to);
                output.push_str(TYPESCRIPT_START);
                continue;
            }
            if control.starts_with("<|start|>ts") && control.len() < "<|start|>ts>".len() {
                self.pending = self.pending[control_start..].to_string();
                break;
            }
            if let Some(drain_len) = generated_control_len(control) {
                let drain_to = control_start + drain_len;
                self.pending.drain(..drain_to);
                continue;
            }

            self.pending = self.pending[control_start..].to_string();
            break;
        }
        output
    }
}

fn first_generated_control_start(text: &str) -> Option<usize> {
    [
        "<|",
        "PROCESSING_RESULTS",
        "commentary+private",
        "analysis to=container.exec",
        "to=container.exec",
    ]
    .iter()
    .filter_map(|marker| text.find(marker))
    .min()
}

fn generated_control_len(text: &str) -> Option<usize> {
    if text.starts_with("PROCESSING_RESULTS") {
        return Some("PROCESSING_RESULTS".len());
    }
    if text.starts_with("commentary+private") {
        return Some("commentary+private".len());
    }
    if text.starts_with("analysis to=container.exec") {
        return Some("analysis to=container.exec".len());
    }
    if text.starts_with("to=container.exec") {
        return Some("to=container.exec".len());
    }
    if text.starts_with("<|start|>") {
        let marker_len = "<|start|>".len();
        let role_len = [
            "stream",
            "assistant",
            "user",
            "system",
            "developer",
            "analysis",
            "final",
            "commentary",
            "ts",
        ]
        .iter()
        .find_map(|role| text[marker_len..].starts_with(role).then_some(role.len()))
        .unwrap_or(0);
        return Some(marker_len + role_len);
    }
    if text.starts_with("<|channel|>") {
        if let Some(message) = text.find("<|message|>") {
            return Some(message + "<|message|>".len());
        }
        let marker_len = "<|channel|>".len();
        let role_len = ["analysis", "final", "commentary"]
            .iter()
            .find_map(|role| text[marker_len..].starts_with(role).then_some(role.len()))
            .unwrap_or(0);
        return Some(marker_len + role_len);
    }
    ["<|end|>", "<|return|>", "<|message|>"]
        .iter()
        .find_map(|marker| text.starts_with(marker).then_some(marker.len()))
        .or_else(|| {
            text.starts_with("<|")
                .then(|| text.find("|>").map(|end| end + 2))
                .flatten()
        })
}

fn possible_generated_control_prefix_start(text: &str) -> usize {
    let markers = [
        "<|",
        "PROCESSING_RESULTS",
        "commentary+private",
        "analysis to=container.exec",
        "to=container.exec",
    ];
    for index in text
        .char_indices()
        .map(|(index, _)| index)
        .chain(std::iter::once(text.len()))
    {
        let suffix = &text[index..];
        if markers
            .iter()
            .any(|marker| marker.starts_with(suffix) && !suffix.is_empty())
        {
            return index;
        }
    }
    text.len()
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
            if let Some(stripped) = strip_bare_channel_label_prefix(&self.text) {
                self.text = stripped.to_string();
                continue;
            }

            if let Some((start, start_marker_len)) = next_typescript_start(&self.text) {
                if start > 0 {
                    let thought = self.text[..start].to_string();
                    self.text = self.text[start..].to_string();
                    if is_meaningful_thought(&thought) {
                        outputs.push(StreamOutput::Thought(thought));
                    }
                    continue;
                }

                let content_start = start_marker_len;
                let body = &self.text[content_start..];
                let malformed_excerpt = compact_line(body, 1_200);
                let (source, consumed) = if let Some((end_rel, end_len)) = find_typescript_end(body)
                {
                    let raw_source = body[..end_rel].trim().to_string();
                    (
                        sanitize_typescript_source(&raw_source),
                        content_start + end_rel + end_len,
                    )
                } else if let Some((source, body_consumed)) =
                    recover_unclosed_typescript_source(body)
                {
                    (Some(source), content_start + body_consumed)
                } else {
                    break;
                };
                self.text = self.text[consumed..].to_string();
                match source {
                    Some(source) if !source.is_empty() => {
                        outputs.push(StreamOutput::TypeScript(source));
                    }
                    _ => outputs.push(StreamOutput::MalformedTypeScript(malformed_excerpt)),
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

fn strip_bare_channel_label_prefix(text: &str) -> Option<&str> {
    ["commentary", "analysis", "final"]
        .iter()
        .find_map(|marker| {
            text.strip_prefix(marker).and_then(|rest| {
                let should_strip = rest.is_empty()
                    || rest.chars().next().is_some_and(|ch| {
                        ch.is_whitespace() || ch.is_ascii_uppercase() || ch == ':'
                    });
                should_strip.then_some(rest.trim_start_matches([' ', ':']))
            })
        })
}

fn next_typescript_start(text: &str) -> Option<(usize, usize)> {
    let normal = text
        .find(TYPESCRIPT_START)
        .map(|start| (start, TYPESCRIPT_START.len()));
    let recovered = text
        .match_indices(TYPESCRIPT_START_MISSING_LESS_THAN)
        .find(|(start, _)| is_recoverable_missing_typescript_opener(text, *start))
        .map(|(start, marker)| (start, marker.len()));
    let role_prefixed = TYPESCRIPT_ROLE_PREFIXED_STARTS
        .iter()
        .filter_map(|marker| {
            text.match_indices(marker)
                .find(|(start, _)| is_recoverable_typescript_body(text, *start + marker.len()))
                .map(|(start, _)| (start, marker.len()))
        })
        .min();

    [normal, recovered, role_prefixed]
        .into_iter()
        .flatten()
        .min()
}

fn is_recoverable_missing_typescript_opener(text: &str, start: usize) -> bool {
    let before = text[..start].chars().next_back();
    if before.is_some_and(|ch| ch == '<' || ch.is_ascii_alphanumeric() || ch == '_') {
        return false;
    }

    is_recoverable_typescript_body(text, start + TYPESCRIPT_START_MISSING_LESS_THAN.len())
}

fn is_recoverable_typescript_body(text: &str, body_start: usize) -> bool {
    let body = &text[body_start..];
    if let Some((end, _)) = find_typescript_end(body) {
        return sanitize_typescript_source(body[..end].trim()).is_some();
    }
    recover_unclosed_typescript_source(body).is_some()
}

fn find_typescript_end(body: &str) -> Option<(usize, usize)> {
    std::iter::once(TYPESCRIPT_END)
        .chain(TYPESCRIPT_ROLE_ENDS.iter().copied())
        .filter_map(|marker| body.find(marker).map(|index| (index, marker.len())))
        .min_by_key(|(index, _)| *index)
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

fn sanitize_typescript_source(raw_source: &str) -> Option<String> {
    let mut source = raw_source.trim();
    if let Some(nested_start) = source.rfind(TYPESCRIPT_START) {
        source = &source[nested_start + TYPESCRIPT_START.len()..];
    }
    if source.contains("<|")
        || source.contains("to=container")
        || source.contains("container.exec")
        || source.contains("<live_observation")
        || source.contains("```")
    {
        return recover_unclosed_typescript_source(source).map(|(source, _)| source);
    }
    if let Some(end) = balanced_typescript_expression_end(source) {
        let candidate = source[..end].trim();
        if looks_like_pete_will_source(candidate) {
            return Some(candidate.to_string());
        }
    }
    looks_like_pete_will_source(source).then(|| source.to_string())
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
        "note",
        "setStage",
        "setTopic",
        "startNewTopic",
        "topicChangedWhen",
        "startNewEpisode",
        "sleeping",
        "goingToSleep",
        "goToSleep",
        "extractEntities",
        "updateGraphNodeFields",
        "searchGraphNodes",
        "queryMemories",
        "listFiles",
        "readSourceFile",
        "readFile",
        "setSourcePageSize",
        "searchSource",
        "grepSource",
        "createGoal",
        "addGoalNote",
        "logProgress",
        "commentGoal",
        "createTask",
        "createChecklist",
        "checkOff",
        "completeItem",
        "checkGoalStep",
        "checkChecklistItem",
        "updateItem",
        "cancelItem",
        "selectItem",
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

fn work_board_path() -> PathBuf {
    std::env::var_os("LISTENBURY_GO_WORK_BOARD")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(WORK_BOARD_PATH))
}

#[derive(Debug, Deserialize)]
struct IpLocation {
    ip: Option<String>,
    city: Option<String>,
    region: Option<String>,
    country_name: Option<String>,
    timezone: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WorkItemStatus {
    Open,
    Complete,
    Cancelled,
}

impl WorkItemStatus {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Complete => "complete",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct GoalStep {
    text: String,
    done: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoalLogEntry {
    text: String,
    #[serde(default)]
    at: Option<String>,
}

impl GoalLogEntry {
    fn now(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            at: Some(Local::now().to_rfc3339_opts(SecondsFormat::Secs, false)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Goal {
    id: String,
    title: String,
    summary: Option<String>,
    parent: Option<String>,
    priority: Option<String>,
    #[serde(default)]
    tags: BTreeSet<String>,
    #[serde(default, alias = "checklist", rename = "steps")]
    steps: Vec<GoalStep>,
    #[serde(default, alias = "notes", rename = "log")]
    log: Vec<GoalLogEntry>,
    status: WorkItemStatus,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct WorkBoard {
    #[serde(default)]
    items: Vec<Goal>,
    #[serde(default)]
    selected_id: Option<String>,
    #[serde(default)]
    next_id: u64,
}

impl WorkBoard {
    fn new() -> Self {
        Self {
            next_id: 1,
            ..Default::default()
        }
    }

    fn load_or_default(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::new());
        }
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read work board {}", path.display()))?;
        let mut board: Self = serde_json::from_str(&text)
            .with_context(|| format!("parse work board {}", path.display()))?;
        board.repair_after_load();
        Ok(board)
    }

    fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create work board dir {}", parent.display()))?;
        }
        let text = serde_json::to_string_pretty(self).context("serialize work board")?;
        std::fs::write(path, text).with_context(|| format!("write work board {}", path.display()))
    }

    fn repair_after_load(&mut self) {
        if self.next_id == 0 {
            self.next_id = 1;
        }
        let mut highest = 0;
        for item in &self.items {
            if let Some(number) = item
                .id
                .rsplit_once('-')
                .and_then(|(_, number)| number.parse::<u64>().ok())
            {
                highest = highest.max(number);
            }
        }
        self.next_id = self.next_id.max(highest.saturating_add(1));
        if self
            .selected_id
            .as_deref()
            .is_some_and(|id| self.items.iter().all(|item| item.id != id))
        {
            self.selected_id = None;
        }
    }

    fn create(&mut self, mut goal: Goal, select: bool) -> String {
        if goal.id.trim().is_empty() {
            goal.id = self.allocate_id("goal");
        }
        let id = goal.id.clone();
        let title = goal.title.clone();
        if select {
            self.selected_id = Some(id.clone());
        }
        self.items.push(goal);
        format!(
            "Created goal {id}: {title}{}",
            if select { " (selected)" } else { "" }
        )
    }

    fn complete(&mut self, target: &str, note: Option<&str>) -> String {
        let Some(goal) = self.find_mut(target) else {
            return format!("No goal matched {target}.");
        };
        goal.status = WorkItemStatus::Complete;
        if let Some(note) = note {
            goal.add_log(format!("Completed: {note}"));
        }
        format!(
            "Checked off goal {}: {}{}",
            goal.id,
            goal.title,
            note.map(|note| format!(" note={note}")).unwrap_or_default()
        )
    }

    fn check_goal_step(&mut self, target: &str, entry: &str, note: Option<&str>) -> String {
        let Some(goal) = self.find_mut(target) else {
            return format!("No goal matched {target}.");
        };
        let Some(index) = goal
            .steps
            .iter()
            .position(|check| ids_match(&check.text, entry))
        else {
            return format!("No goal step matched {entry} in {}.", goal.id);
        };
        goal.steps[index].done = true;
        let checked_text = goal.steps[index].text.clone();
        if let Some(note) = note {
            goal.add_log(format!("Step done: {checked_text}. {note}"));
        } else {
            goal.add_log(format!("Step done: {checked_text}"));
        }
        if !goal.steps.is_empty() && goal.steps.iter().all(|check| check.done) {
            goal.status = WorkItemStatus::Complete;
            goal.add_log("All steps complete.");
        }
        format!(
            "Checked goal step in {}: {}{}",
            goal.id,
            checked_text,
            note.map(|note| format!(" note={note}")).unwrap_or_default()
        )
    }

    fn update(&mut self, target: &str, fields: Map<String, Value>) -> String {
        let Some(goal) = self.find_mut(target) else {
            return format!("No goal matched {target}.");
        };
        if let Some(title) = string_field(&fields, "title") {
            goal.title = title;
        }
        if fields.contains_key("summary") {
            goal.summary = string_field(&fields, "summary");
        }
        if fields.contains_key("parent") {
            goal.parent = string_field(&fields, "parent");
        }
        if fields.contains_key("priority") {
            goal.priority = string_field(&fields, "priority");
        }
        if let Some(tags) = string_list_field(&fields, "tags") {
            goal.tags = tags.into_iter().collect();
        }
        if let Some(steps) = string_list_field(&fields, "steps")
            .or_else(|| string_list_field(&fields, "items"))
            .or_else(|| string_list_field(&fields, "checklist"))
        {
            goal.steps = steps
                .into_iter()
                .map(|text| GoalStep { text, done: false })
                .collect();
        }
        if let Some(note) = string_field(&fields, "note")
            .or_else(|| string_field(&fields, "log"))
            .or_else(|| string_field(&fields, "comment"))
        {
            goal.add_log(note);
        }
        format!("Updated goal {}: {}", goal.id, goal.title)
    }

    fn cancel(&mut self, target: &str, reason: Option<&str>) -> String {
        let Some(goal) = self.find_mut(target) else {
            return format!("No goal matched {target}.");
        };
        goal.status = WorkItemStatus::Cancelled;
        if let Some(reason) = reason {
            goal.add_log(format!("Cancelled: {reason}"));
        }
        format!(
            "Cancelled goal {}: {}{}",
            goal.id,
            goal.title,
            reason
                .map(|reason| format!(" reason={reason}"))
                .unwrap_or_default()
        )
    }

    fn select(&mut self, target: &str) -> String {
        let Some(goal) = self.find(target) else {
            return format!("No goal matched {target}.");
        };
        let id = goal.id.clone();
        let title = goal.title.clone();
        self.selected_id = Some(id.clone());
        format!("Selected goal {id}: {title}")
    }

    fn add_note(&mut self, target: &str, text: &str) -> String {
        let Some(goal) = self.find_mut(target) else {
            return format!("No goal matched {target}.");
        };
        goal.add_log(text);
        format!(
            "Added goal note to {}: {}",
            goal.id,
            compact_line(text, 500)
        )
    }

    fn prompt_summary(&self) -> Option<String> {
        if self.items.is_empty() {
            return None;
        }
        let mut lines = Vec::new();
        if let Some(selected) = self.selected_item() {
            lines.push(format!(
                "Selected goal {} [{}]: {}{}{}",
                selected.id,
                selected.status.as_str(),
                selected.title,
                selected
                    .summary
                    .as_deref()
                    .map(|summary| format!(" -- {summary}"))
                    .unwrap_or_default(),
                latest_goal_log(selected)
            ));
        } else {
            lines.push("No selected goal.".to_string());
        }
        for item in self
            .items
            .iter()
            .filter(|item| matches!(item.status, WorkItemStatus::Open))
            .take(8)
        {
            lines.push(format!(
                "- goal {} [{}]: {}{}{}",
                item.id,
                item.status.as_str(),
                item.title,
                goal_step_progress(item),
                latest_goal_log(item)
            ));
        }
        Some(lines.join("\n"))
    }

    fn selected_item(&self) -> Option<&Goal> {
        let id = self.selected_id.as_deref()?;
        self.items.iter().find(|item| item.id == id)
    }

    fn find(&self, target: &str) -> Option<&Goal> {
        self.items
            .iter()
            .find(|item| ids_match(&item.id, target) || ids_match(&item.title, target))
    }

    fn find_mut(&mut self, target: &str) -> Option<&mut Goal> {
        self.items
            .iter_mut()
            .find(|item| ids_match(&item.id, target) || ids_match(&item.title, target))
    }

    fn allocate_id(&mut self, prefix: &str) -> String {
        loop {
            let id = format!("{}-{}", prefix, self.next_id);
            self.next_id = self.next_id.saturating_add(1);
            if self.items.iter().all(|item| item.id != id) {
                return id;
            }
        }
    }
}

impl Goal {
    fn add_log(&mut self, text: impl Into<String>) {
        let text = text.into();
        if non_empty_text(&text).is_some() {
            self.log.push(GoalLogEntry::now(text));
        }
    }
}

fn goal_step_progress(goal: &Goal) -> String {
    if goal.steps.is_empty() {
        return String::new();
    }
    let done = goal.steps.iter().filter(|entry| entry.done).count();
    format!(" ({done}/{})", goal.steps.len())
}

fn latest_goal_log(goal: &Goal) -> String {
    goal.log
        .last()
        .map(|entry| format!(" latest_note={}", compact_line(&entry.text, 180)))
        .unwrap_or_default()
}

fn ids_match(left: &str, right: &str) -> bool {
    left.trim().eq_ignore_ascii_case(right.trim())
}

fn string_field(fields: &Map<String, Value>, key: &str) -> Option<String> {
    fields
        .get(key)
        .and_then(Value::as_str)
        .and_then(|value| non_empty_text(value).map(str::to_string))
}

fn string_list_field(fields: &Map<String, Value>, key: &str) -> Option<Vec<String>> {
    let value = fields.get(key)?;
    match value {
        Value::Array(values) => Some(
            values
                .iter()
                .filter_map(Value::as_str)
                .filter_map(|value| non_empty_text(value).map(str::to_string))
                .collect(),
        ),
        Value::String(value) => non_empty_text(value).map(|value| vec![value.to_string()]),
        _ => None,
    }
}

fn non_empty_strings(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .filter_map(|value| non_empty_text(&value).map(str::to_string))
        .collect()
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
    ListFiles {
        page: usize,
        page_size: Option<usize>,
    },
    ReadSourceFile {
        file: String,
        page: usize,
        line: Option<usize>,
        page_size: Option<usize>,
    },
    SetSourcePageSize {
        lines: usize,
    },
    SearchSource {
        query: String,
        limit: usize,
    },
    GrepSource {
        pattern: String,
        limit: usize,
    },
    CreateWorkItem {
        id: Option<String>,
        title: String,
        summary: Option<String>,
        parent: Option<String>,
        priority: Option<String>,
        tags: Vec<String>,
        steps: Vec<String>,
        note: Option<String>,
        select: bool,
    },
    CompleteWorkItem {
        target: String,
        note: Option<String>,
    },
    CheckChecklistItem {
        target: String,
        item: String,
        note: Option<String>,
    },
    AddGoalNote {
        target: String,
        text: String,
    },
    UpdateWorkItem {
        target: String,
        fields: Map<String, Value>,
    },
    CancelWorkItem {
        target: String,
        reason: Option<String>,
    },
    SelectWorkItem {
        target: String,
    },
    Sleeping {
        reason: Option<String>,
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
    ListFiles {
        #[serde(default)]
        page: Option<usize>,
        #[serde(default)]
        page_size: Option<usize>,
    },
    ReadSourceFile {
        file: String,
        #[serde(default)]
        page: Option<usize>,
        #[serde(default)]
        line: Option<usize>,
        #[serde(default)]
        page_size: Option<usize>,
    },
    SetSourcePageSize {
        lines: usize,
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
    CreateGoal {
        title: String,
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        summary: Option<String>,
        #[serde(default)]
        parent: Option<String>,
        #[serde(default)]
        priority: Option<String>,
        #[serde(default)]
        tags: Vec<String>,
        #[serde(default)]
        steps: Vec<String>,
        #[serde(default)]
        items: Vec<String>,
        #[serde(default)]
        note: Option<String>,
        #[serde(default)]
        select: bool,
    },
    CreateTask {
        title: String,
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        summary: Option<String>,
        #[serde(default)]
        parent: Option<String>,
        #[serde(default)]
        priority: Option<String>,
        #[serde(default)]
        tags: Vec<String>,
        #[serde(default)]
        select: bool,
    },
    CreateChecklist {
        title: String,
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        summary: Option<String>,
        #[serde(default)]
        parent: Option<String>,
        #[serde(default)]
        priority: Option<String>,
        #[serde(default)]
        tags: Vec<String>,
        #[serde(default)]
        items: Vec<String>,
        #[serde(default)]
        select: bool,
    },
    CompleteWorkItem {
        target: String,
        #[serde(default)]
        note: Option<String>,
    },
    CheckChecklistItem {
        target: String,
        item: String,
        #[serde(default)]
        note: Option<String>,
    },
    AddGoalNote {
        target: String,
        text: String,
    },
    UpdateWorkItem {
        target: String,
        #[serde(default)]
        fields: Map<String, Value>,
    },
    CancelWorkItem {
        target: String,
        #[serde(default)]
        reason: Option<String>,
    },
    SelectWorkItem {
        target: String,
    },
    Sleeping {
        reason: Option<String>,
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
        TypeScriptActionPayload::ListFiles { page, page_size } => {
            Some(TypeScriptAction::ListFiles {
                page: page.unwrap_or(1).max(1),
                page_size,
            })
        }
        TypeScriptActionPayload::ReadSourceFile {
            file,
            page,
            line,
            page_size,
        } => {
            let file = file.trim();
            (!file.is_empty()).then(|| TypeScriptAction::ReadSourceFile {
                file: file.to_string(),
                page: page.unwrap_or(1).max(1),
                line: line.map(|line| line.max(1)),
                page_size: page_size
                    .map(|lines| lines.clamp(MIN_SOURCE_PAGE_LINES, MAX_SOURCE_PAGE_LINES)),
            })
        }
        TypeScriptActionPayload::SetSourcePageSize { lines } => {
            Some(TypeScriptAction::SetSourcePageSize {
                lines: lines.clamp(MIN_SOURCE_PAGE_LINES, MAX_SOURCE_PAGE_LINES),
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
        TypeScriptActionPayload::CreateGoal {
            title,
            id,
            summary,
            parent,
            priority,
            tags,
            steps,
            items,
            note,
            select,
        } => non_empty_text(&title).map(|title| TypeScriptAction::CreateWorkItem {
            id: id.and_then(|id| non_empty_text(&id).map(str::to_string)),
            title: title.to_string(),
            summary: summary.and_then(|summary| non_empty_text(&summary).map(str::to_string)),
            parent: parent.and_then(|parent| non_empty_text(&parent).map(str::to_string)),
            priority: priority.and_then(|priority| non_empty_text(&priority).map(str::to_string)),
            tags,
            steps: non_empty_strings(steps)
                .into_iter()
                .chain(non_empty_strings(items))
                .collect(),
            note: note.and_then(|note| non_empty_text(&note).map(str::to_string)),
            select,
        }),
        TypeScriptActionPayload::CreateTask {
            title,
            id,
            summary,
            parent,
            priority,
            tags,
            select,
        } => non_empty_text(&title).map(|title| TypeScriptAction::CreateWorkItem {
            id: id.and_then(|id| non_empty_text(&id).map(str::to_string)),
            title: title.to_string(),
            summary: summary.and_then(|summary| non_empty_text(&summary).map(str::to_string)),
            parent: parent.and_then(|parent| non_empty_text(&parent).map(str::to_string)),
            priority: priority.and_then(|priority| non_empty_text(&priority).map(str::to_string)),
            tags,
            steps: Vec::new(),
            note: None,
            select,
        }),
        TypeScriptActionPayload::CreateChecklist {
            title,
            id,
            summary,
            parent,
            priority,
            tags,
            items,
            select,
        } => non_empty_text(&title).map(|title| TypeScriptAction::CreateWorkItem {
            id: id.and_then(|id| non_empty_text(&id).map(str::to_string)),
            title: title.to_string(),
            summary: summary.and_then(|summary| non_empty_text(&summary).map(str::to_string)),
            parent: parent.and_then(|parent| non_empty_text(&parent).map(str::to_string)),
            priority: priority.and_then(|priority| non_empty_text(&priority).map(str::to_string)),
            tags,
            steps: non_empty_strings(items),
            note: None,
            select,
        }),
        TypeScriptActionPayload::CompleteWorkItem { target, note } => {
            non_empty_text(&target).map(|target| TypeScriptAction::CompleteWorkItem {
                target: target.to_string(),
                note: note.and_then(|note| non_empty_text(&note).map(str::to_string)),
            })
        }
        TypeScriptActionPayload::CheckChecklistItem { target, item, note } => {
            non_empty_text(&target).and_then(|target| {
                non_empty_text(&item).map(|item| TypeScriptAction::CheckChecklistItem {
                    target: target.to_string(),
                    item: item.to_string(),
                    note: note.and_then(|note| non_empty_text(&note).map(str::to_string)),
                })
            })
        }
        TypeScriptActionPayload::AddGoalNote { target, text } => {
            non_empty_text(&target).and_then(|target| {
                non_empty_text(&text).map(|text| TypeScriptAction::AddGoalNote {
                    target: target.to_string(),
                    text: text.to_string(),
                })
            })
        }
        TypeScriptActionPayload::UpdateWorkItem { target, fields } => {
            non_empty_text(&target).map(|target| TypeScriptAction::UpdateWorkItem {
                target: target.to_string(),
                fields,
            })
        }
        TypeScriptActionPayload::CancelWorkItem { target, reason } => {
            non_empty_text(&target).map(|target| TypeScriptAction::CancelWorkItem {
                target: target.to_string(),
                reason: reason.and_then(|reason| non_empty_text(&reason).map(str::to_string)),
            })
        }
        TypeScriptActionPayload::SelectWorkItem { target } => {
            non_empty_text(&target).map(|target| TypeScriptAction::SelectWorkItem {
                target: target.to_string(),
            })
        }
        TypeScriptActionPayload::Sleeping { reason } => Some(TypeScriptAction::Sleeping {
            reason: reason.and_then(|reason| non_empty_text(&reason).map(str::to_string)),
        }),
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
        "import {{ say, shutup, pause, resume, note, setStage, setTopic, startNewTopic, topicChangedWhen, startNewEpisode, sleeping, goingToSleep, extractEntities, updateGraphNodeFields, searchGraphNodes, queryMemories, listFiles, readSourceFile, readFile, searchSource, grepSource, setSourcePageSize, createGoal, createTask, createChecklist, addGoalNote, logProgress, commentGoal, checkOff, completeItem, checkGoalStep, checkChecklistItem, updateItem, cancelItem, selectItem }} from \"pete:will\";\n{script}"
    )
}

fn go_typescript_module() -> InternalModule {
    InternalModule::native("pete:will")
        .with_function("say", ts_say, 2)
        .with_function("shutup", ts_shutup, 0)
        .with_function("pause", ts_pause, 0)
        .with_function("resume", ts_resume, 0)
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
        .with_function("listFiles", ts_list_files, 1)
        .with_function("list_files", ts_list_files, 1)
        .with_function("readSourceFile", ts_read_source_file, 2)
        .with_function("read_source_file", ts_read_source_file, 2)
        .with_function("readFile", ts_read_source_file, 2)
        .with_function("read_file", ts_read_source_file, 2)
        .with_function("setSourcePageSize", ts_set_source_page_size, 1)
        .with_function("set_source_page_size", ts_set_source_page_size, 1)
        .with_function("searchSource", ts_search_source, 2)
        .with_function("search_source", ts_search_source, 2)
        .with_function("grepSource", ts_grep_source, 2)
        .with_function("grep_source", ts_grep_source, 2)
        .with_function("createGoal", ts_create_goal, 2)
        .with_function("create_goal", ts_create_goal, 2)
        .with_function("createTask", ts_create_task, 2)
        .with_function("create_task", ts_create_task, 2)
        .with_function("createChecklist", ts_create_checklist, 3)
        .with_function("create_checklist", ts_create_checklist, 3)
        .with_function("addGoalNote", ts_add_goal_note, 2)
        .with_function("add_goal_note", ts_add_goal_note, 2)
        .with_function("logProgress", ts_add_goal_note, 2)
        .with_function("log_progress", ts_add_goal_note, 2)
        .with_function("commentGoal", ts_add_goal_note, 2)
        .with_function("comment_goal", ts_add_goal_note, 2)
        .with_function("checkOff", ts_complete_work_item, 2)
        .with_function("check_off", ts_complete_work_item, 2)
        .with_function("completeItem", ts_complete_work_item, 2)
        .with_function("complete_item", ts_complete_work_item, 2)
        .with_function("checkGoalStep", ts_check_checklist_item, 3)
        .with_function("check_goal_step", ts_check_checklist_item, 3)
        .with_function("checkChecklistItem", ts_check_checklist_item, 3)
        .with_function("check_checklist_item", ts_check_checklist_item, 3)
        .with_function("updateItem", ts_update_work_item, 2)
        .with_function("update_item", ts_update_work_item, 2)
        .with_function("cancelItem", ts_cancel_work_item, 2)
        .with_function("cancel_item", ts_cancel_work_item, 2)
        .with_function("selectItem", ts_select_work_item, 1)
        .with_function("select_item", ts_select_work_item, 1)
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
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    let mut value = json!({ "kind": "list_files" });
    if let Some(page) = list_source_page_arg(args) {
        value["page"] = json!(page);
    }
    if let Some(page_size) = list_source_page_size_arg(args) {
        value["page_size"] = json!(page_size);
    }
    command_value(interp, value)
}

fn ts_read_source_file(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    let mut value = json!({ "kind": "read_source_file", "file": string_arg(args, 0) });
    if let Some(page) = read_source_page_arg(args) {
        value["page"] = json!(page);
    }
    if let Some(line) = read_source_line_arg(args) {
        value["line"] = json!(line);
    }
    if let Some(page_size) = read_source_page_size_arg(args) {
        value["page_size"] = json!(page_size);
    }
    command_value(interp, value)
}

fn ts_set_source_page_size(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    let lines =
        optional_positive_integer_arg(args, 0, "lines").unwrap_or(DEFAULT_SOURCE_PAGE_LINES);
    command_value(
        interp,
        json!({ "kind": "set_source_page_size", "lines": lines }),
    )
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

fn ts_create_goal(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    create_work_item_command(interp, "create_goal", args, Vec::new())
}

fn ts_create_task(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    create_work_item_command(interp, "create_task", args, Vec::new())
}

fn ts_create_checklist(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    let items = match args.get(1) {
        Some(value) => js_value_to_json(value)
            .ok()
            .and_then(|value| strings_from_json_value(&value))
            .unwrap_or_default(),
        None => Vec::new(),
    };
    create_work_item_command(interp, "create_checklist", args, items)
}

fn create_work_item_command(
    interp: &mut Interpreter,
    kind: &str,
    args: &[JsValue],
    items: Vec<String>,
) -> std::result::Result<Guarded, JsError> {
    let options_index = if kind == "create_checklist" { 2 } else { 1 };
    let steps = optional_string_list_property_arg(args, options_index, "steps")
        .or_else(|| optional_string_list_property_arg(args, options_index, "items"))
        .unwrap_or(items);
    command_value(
        interp,
        json!({
            "kind": kind,
            "title": string_arg(args, 0),
            "id": optional_string_property_arg(args, options_index, "id"),
            "summary": optional_string_property_arg(args, options_index, "summary"),
            "parent": optional_string_property_arg(args, options_index, "parent"),
            "priority": optional_string_property_arg(args, options_index, "priority"),
            "tags": optional_string_list_property_arg(args, options_index, "tags").unwrap_or_default(),
            "items": steps,
            "note": optional_string_property_arg(args, options_index, "note"),
            "select": optional_bool_property_arg(args, options_index, "select").unwrap_or(false),
        }),
    )
}

fn ts_add_goal_note(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(
        interp,
        json!({
            "kind": "add_goal_note",
            "target": string_arg(args, 0),
            "text": string_arg(args, 1),
        }),
    )
}

fn ts_complete_work_item(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(
        interp,
        json!({
            "kind": "complete_work_item",
            "target": string_arg(args, 0),
            "note": optional_string_property_arg(args, 1, "note"),
        }),
    )
}

fn ts_check_checklist_item(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(
        interp,
        json!({
            "kind": "check_checklist_item",
            "target": string_arg(args, 0),
            "item": string_arg(args, 1),
            "note": optional_string_property_arg(args, 2, "note"),
        }),
    )
}

fn ts_update_work_item(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(
        interp,
        json!({
            "kind": "update_work_item",
            "target": string_arg(args, 0),
            "fields": object_arg(args, 1),
        }),
    )
}

fn ts_cancel_work_item(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    let reason = match args.get(1) {
        Some(JsValue::String(value)) => {
            let value = value.to_string();
            non_empty_text(&value).map(str::to_string)
        }
        Some(JsValue::Object(_)) => optional_string_property_arg(args, 1, "reason"),
        _ => None,
    };
    command_value(
        interp,
        json!({ "kind": "cancel_work_item", "target": string_arg(args, 0), "reason": reason }),
    )
}

fn ts_select_work_item(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(
        interp,
        json!({ "kind": "select_work_item", "target": string_arg(args, 0) }),
    )
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

fn read_source_page_arg(args: &[JsValue]) -> Option<usize> {
    match args.get(1) {
        Some(JsValue::Number(value)) if value.is_finite() => Some(value.floor().max(1.0) as usize),
        _ => optional_positive_integer_arg(args, 1, "page"),
    }
    .or_else(|| optional_positive_integer_arg(args, 2, "page"))
}

fn list_source_page_arg(args: &[JsValue]) -> Option<usize> {
    match args.first() {
        Some(JsValue::Number(value)) if value.is_finite() => Some(value.floor().max(1.0) as usize),
        _ => optional_positive_integer_arg(args, 0, "page"),
    }
}

fn list_source_page_size_arg(args: &[JsValue]) -> Option<usize> {
    optional_positive_integer_arg(args, 0, "pageSize")
        .or_else(|| optional_positive_integer_arg(args, 0, "page_size"))
}

fn read_source_line_arg(args: &[JsValue]) -> Option<usize> {
    optional_positive_integer_arg(args, 1, "line")
        .or_else(|| optional_positive_integer_arg(args, 1, "lineNumber"))
        .or_else(|| optional_positive_integer_arg(args, 1, "line_number"))
        .or_else(|| optional_positive_integer_arg(args, 2, "line"))
        .or_else(|| optional_positive_integer_arg(args, 2, "lineNumber"))
        .or_else(|| optional_positive_integer_arg(args, 2, "line_number"))
}

fn read_source_page_size_arg(args: &[JsValue]) -> Option<usize> {
    optional_positive_integer_arg(args, 1, "pageSize")
        .or_else(|| optional_positive_integer_arg(args, 1, "page_size"))
        .or_else(|| optional_positive_integer_arg(args, 1, "lines"))
        .or_else(|| optional_positive_integer_arg(args, 2, "pageSize"))
        .or_else(|| optional_positive_integer_arg(args, 2, "page_size"))
        .or_else(|| optional_positive_integer_arg(args, 2, "lines"))
        .map(|lines| lines.clamp(MIN_SOURCE_PAGE_LINES, MAX_SOURCE_PAGE_LINES))
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

fn optional_bool_property_arg(args: &[JsValue], index: usize, property: &str) -> Option<bool> {
    optional_json_property_arg(args, index, property).and_then(|value| match value {
        Value::Bool(value) => Some(value),
        _ => None,
    })
}

fn optional_string_list_property_arg(
    args: &[JsValue],
    index: usize,
    property: &str,
) -> Option<Vec<String>> {
    optional_json_property_arg(args, index, property)
        .and_then(|value| strings_from_json_value(&value))
}

fn strings_from_json_value(value: &Value) -> Option<Vec<String>> {
    match value {
        Value::Array(values) => Some(
            values
                .iter()
                .filter_map(Value::as_str)
                .filter_map(|value| non_empty_text(value).map(str::to_string))
                .collect(),
        ),
        Value::String(value) => non_empty_text(value).map(|value| vec![value.to_string()]),
        _ => None,
    }
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

fn initial_stream_prompt(seed: &str, startup_context: &str, work_summary: Option<&str>) -> String {
    let work_summary = work_summary.unwrap_or("No persisted goals yet.");
    format!(
        "{seed}\n\n\
         Startup context:\n{startup_context}\n\n\
         Persisted working memory:\n{work_summary}\n\n\
         Orientation:\n{PETE_ORIENTATION_PROMPT}\n\n\
         Stream rules:\n\
         Generate continuously. Plain text is private thought visible only as raw debug stdout; generated text remains in the active LLM context and is retained by the runtime for compacted restarts.\n\
         To speak or act, emit TypeScript as a direct function call: <ts>say(\"short friendly words\")</ts>, <ts>listFiles()</ts>, or <ts>setStage(\"what is happening\")</ts>.\n\
         This is not Harmony. Harmony symbols do nothing here. If Harmony-style channel/control symbols appear, the runtime strips them; continue in plain Pete thought text plus <ts>...</ts> actions. Do not emit tool-call JSON, to=container.exec, shell commands, channel markers, or markdown code fences.\n\
         Do not be idle. When there is no user speech, keep quietly maintaining awareness, persisted goals, source context, or a useful next action. Frequently summarize the current situation and recent source findings, and store durable user, project, and work context in memory, stage, goal steps, or goal running-log notes instead of only reading more.\n\
         Use current time and location context when it helps. Be autonomous, curious, friendly, and sociable. If no listener is present, speech is still allowed, but Pete is talking to himself and self-hearing it through his own ears.\n\n\
         Pete will runtime:\n{PETE_WILL_RUNTIME_PROMPT}\n\n\
         Pete: "
    )
}

fn compact_stream_prompt(
    seed: &str,
    startup_context: &str,
    recent_events: &VecDeque<String>,
    work_summary: Option<&str>,
) -> String {
    let work_summary = work_summary.unwrap_or("No persisted goals yet.");
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
         Startup context:\n{startup_context}\n\n\
         Persisted working memory:\n{work_summary}\n\n\
         Orientation:\n{PETE_ORIENTATION_PROMPT}\n\n\
         Continuity memory:\n{events}\n\n\
         Pete will runtime:\n{PETE_WILL_RUNTIME_PROMPT}\n\n\
         Continue Pete's stream of consciousness from this compacted context.\n\n\
         Pete: "
    )
}

fn compact_stream_prompt_for_budget(
    seed: &str,
    startup_context: &str,
    recent_events: &VecDeque<String>,
    work_summary: Option<&str>,
    budget_tokens: usize,
) -> (String, usize) {
    let mut retained_events = recent_events.clone();
    loop {
        let prompt = compact_stream_prompt(seed, startup_context, &retained_events, work_summary);
        if estimate_tokens(&prompt) <= budget_tokens || retained_events.is_empty() {
            return (prompt, retained_events.len());
        }
        retained_events.pop_front();
    }
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

fn render_action_result_for_prompt(text: &str) -> String {
    if is_source_inspection_result(text) {
        compact_preserving_lines(text, SOURCE_ACTION_RESULT_MAX_CHARS)
    } else {
        compact_line(text, ACTION_RESULT_MAX_CHARS)
    }
}

fn is_source_inspection_result(text: &str) -> bool {
    text.starts_with("Available source files")
        || text.starts_with("Source matches for ")
        || text.starts_with("No source matches for ")
        || (text.starts_with("--- ") && text.contains(" lines/page) ---\n"))
        || text.starts_with("File not found: ")
        || (text.starts_with("File ") && text.contains(" lines/page; page "))
}

fn compact_preserving_lines(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut compact = text.chars().take(max_chars).collect::<String>();
    compact.push_str("\n...");
    compact
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
    text.len()
        .saturating_add(PROMPT_CHARS_PER_TOKEN_ESTIMATE - 1)
        / PROMPT_CHARS_PER_TOKEN_ESTIMATE
}

fn is_terminal_event(event: &LlmEvent) -> bool {
    matches!(
        event,
        LlmEvent::Completed | LlmEvent::Cancelled | LlmEvent::Error { .. }
    )
}

fn is_context_capacity_message(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    message.contains("context_size")
        || message.contains("context tokens")
        || message.contains("context capacity")
}

fn is_context_append_recoverable(error: &anyhow::Error) -> bool {
    let message = format!("{error:#}").to_ascii_lowercase();
    message.contains("context_size")
        || message.contains("context tokens")
        || message.contains("context capacity")
        || message.contains("no longer accepting prompt appends")
        || message.contains("generation not found")
}

fn is_generation_not_found(error: &anyhow::Error) -> bool {
    format!("{error:#}")
        .to_ascii_lowercase()
        .contains("generation not found")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn go_prompt_encourages_source_summaries_and_memory_updates() {
        let prompt = initial_stream_prompt("seed", "startup", None);
        assert!(prompt.contains("Frequently summarize what is going on"));
        assert!(prompt.contains("After source inspection results arrive"));
        assert!(prompt.contains("Do not silently chain source reads"));
        assert!(prompt.contains("store durable user, project, and work context"));
        assert!(prompt.contains("goal running-log notes"));
        assert!(prompt.contains("Do not write XML/HTML-style angle-bracket tags in prose"));
        assert!(prompt.contains("\\<tr\\>"));
        assert!(prompt.contains("runtime automatically imports the action functions"));
        assert!(prompt.contains("<ts>note(\"still observing\")</ts>"));
        assert!(!prompt.contains("peteWillBuilder"));
        assert!(COMMAND_REMINDER_PROMPT.contains("Keep running logs on goals"));
        assert!(COMMAND_REMINDER_PROMPT.contains("store durable facts or next steps"));
    }

    #[test]
    fn source_action_results_preserve_page_lines_in_prompt() {
        let output = "--- README.md page 2/7 (lines 121 to 240 of 825, 120 lines/page) ---\nline one\nline two\n---";
        let observation = StreamObservation::ActionResult(output.to_string());
        let prompt = observation.prompt_text();
        assert!(prompt.contains("120 lines/page"));
        assert!(prompt.contains("line one\nline two"));
    }

    #[test]
    fn ordinary_action_results_are_still_compacted_to_one_line() {
        let observation = StreamObservation::ActionResult("first\nsecond".to_string());
        let prompt = observation.prompt_text();
        assert!(prompt.contains("first second"));
        assert!(!prompt.contains("first\nsecond"));
    }

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
    fn parser_recovers_missing_less_than_typescript_opener() {
        let mut parser = StreamOutputParser::new(400);
        let parsed = parser
            .push("Let's inspect the repo.ts>listFiles()</ts>commentary This will list files.");
        assert_eq!(
            parsed.outputs,
            vec![
                StreamOutput::Thought("Let's inspect the repo.".to_string()),
                StreamOutput::TypeScript("listFiles()".to_string()),
            ]
        );
        assert_eq!(parser.text, "This will list files.");
    }

    #[test]
    fn parser_recovers_role_prefixed_typescript_tags() {
        let mut parser = StreamOutputParser::new(400);
        let parsed = parser.push(
            "Let's run it.assistantts>listFiles()</assistantassistantts>note(\"done\")</assistant",
        );
        assert_eq!(
            parsed.outputs,
            vec![
                StreamOutput::Thought("Let's run it.".to_string()),
                StreamOutput::TypeScript("listFiles()".to_string()),
                StreamOutput::TypeScript("note(\"done\")".to_string()),
            ]
        );
    }

    #[test]
    fn parser_does_not_recover_missing_less_than_inside_words() {
        let mut parser = StreamOutputParser::new(80);
        let parsed = parser.push("Plain text mentions outputs> without a command.");
        assert!(
            parsed
                .outputs
                .iter()
                .all(|output| !matches!(output, StreamOutput::TypeScript(_)))
        );
    }

    #[test]
    fn parser_strips_bare_channel_label_prefixes() {
        let mut parser = StreamOutputParser::new(80);
        assert!(parser.push("commentaryHmm.").outputs.is_empty());
        assert_eq!(parser.text, "Hmm.");
    }

    #[test]
    fn parser_does_not_execute_control_tail_as_typescript() {
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
    fn parser_recovers_nested_typescript_from_contaminated_tag() {
        let mut parser = StreamOutputParser::new(400);
        let parsed = parser.push(
            "<ts>bad prose <live_observation source=\"ear\">noise</live_observation><ts>setStage(\"Setting: lab. Action: listening\")</ts>",
        );
        assert_eq!(
            parsed.outputs,
            vec![StreamOutput::TypeScript(
                "setStage(\"Setting: lab. Action: listening\")".to_string()
            )]
        );
    }

    #[test]
    fn parser_reports_unrecoverable_malformed_typescript() {
        let mut parser = StreamOutputParser::new(400);
        let parsed = parser.push("<ts>...> forWe already gave say and then logs</ts>");
        assert!(matches!(
            parsed.outputs.as_slice(),
            [StreamOutput::MalformedTypeScript(source)] if source.contains("forWe already")
        ));
    }

    #[test]
    fn generated_text_cleaner_strips_harmony_control_tags() {
        let mut cleaner = GeneratedTextCleaner::new();
        assert_eq!(
            cleaner.push("Think<|end|><|start|>stream\nmore"),
            "Think\nmore"
        );
        assert_eq!(
            cleaner.push("<|end|><|start|>ts>listFiles()</ts>commentary+private"),
            "<ts>listFiles()</ts>"
        );
    }

    #[test]
    fn generated_text_cleaner_handles_split_control_tags() {
        let mut cleaner = GeneratedTextCleaner::new();
        assert_eq!(cleaner.push("Ready <|sta"), "Ready ");
        assert_eq!(
            cleaner.push("rt|>ts>say(\"Hi\")</ts>"),
            "<ts>say(\"Hi\")</ts>"
        );
    }

    #[test]
    fn compact_stream_prompt_for_budget_drops_oldest_events() {
        let mut events = VecDeque::new();
        events.push_back("old event ".repeat(400));
        events.push_back("middle event ".repeat(400));
        events.push_back("new event should stay".to_string());

        let empty_prompt = compact_stream_prompt("seed", "startup", &VecDeque::new(), None);
        let budget = estimate_tokens(&empty_prompt) + 32;
        let (prompt, retained) =
            compact_stream_prompt_for_budget("seed", "startup", &events, None, budget);

        assert!(estimate_tokens(&prompt) <= budget);
        assert!(retained < events.len());
        assert!(!prompt.contains("old event"));
        assert!(prompt.contains("new event should stay"));
    }

    #[test]
    fn context_capacity_messages_are_restartable() {
        assert!(is_context_capacity_message(
            "appended prompt needs 8193 context tokens, but context_size is 8192"
        ));
        assert!(is_context_append_recoverable(&anyhow::anyhow!(
            "generation is no longer accepting prompt appends"
        )));
    }

    #[test]
    fn typescript_runtime_accepts_half_duplex_functions() {
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
                listFiles(2),
                readSourceFile("src/main.rs", 1),
                readFile("src/main.rs"),
                searchSource("GoCommand", 2),
                grepSource("GoCommand", { limit: 2 }),
                createGoal("Keep Pete oriented", { select: true, tags: ["go"], steps: ["read prompt", "watch actions"], note: "initial orientation goal" }),
                addGoalNote("Keep Pete oriented", "read the runtime prompt"),
                logProgress("Keep Pete oriented", "source tools are available"),
                createTask("Inspect source", { parent: "Keep Pete oriented" }),
                createChecklist("Go checklist", ["read prompt", "watch actions"], { select: true }),
                checkGoalStep("Keep Pete oriented", "read prompt", { note: "confirmed prompt shape" }),
                checkChecklistItem("Go checklist", "read prompt"),
                updateItem("Inspect source", { summary: "look for runtime shape", note: "legacy task alias became a goal" }),
                selectItem("Inspect source"),
                checkOff("Inspect source"),
                cancelItem("Keep Pete oriented", "test complete"),
                goingToSleep("done"),
                note("runtime note")
            ]"#,
        )
        .expect("half-duplex functions should execute in go");

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
                .any(|action| matches!(action, TypeScriptAction::ListFiles { page: 2, .. }))
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
        assert!(
            actions
                .iter()
                .any(|action| matches!(action, TypeScriptAction::CreateWorkItem { .. }))
        );
        assert!(
            actions
                .iter()
                .any(|action| matches!(action, TypeScriptAction::AddGoalNote { .. }))
        );
        assert!(
            actions
                .iter()
                .any(|action| matches!(action, TypeScriptAction::SelectWorkItem { .. }))
        );
    }

    #[test]
    fn work_board_persists_and_reloads_useful_state() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("go_work_board.json");
        let mut board = WorkBoard::new();
        board.create(
            Goal {
                id: String::new(),
                title: "Keep bearings".to_string(),
                summary: Some("persist useful working memory".to_string()),
                parent: None,
                priority: Some("high".to_string()),
                tags: ["go".to_string()].into_iter().collect(),
                steps: vec![
                    GoalStep {
                        text: "write file".to_string(),
                        done: false,
                    },
                    GoalStep {
                        text: "reload file".to_string(),
                        done: false,
                    },
                ],
                log: Vec::new(),
                status: WorkItemStatus::Open,
            },
            true,
        );
        assert!(
            board
                .check_goal_step("Keep bearings", "write file", Some("persisted progress"))
                .contains("Checked")
        );
        board.save(&path).expect("save work board");

        let loaded = WorkBoard::load_or_default(&path).expect("load work board");
        let summary = loaded.prompt_summary().expect("summary");
        assert!(summary.contains("Selected goal goal-1"));
        assert!(summary.contains("Keep bearings"));
        assert!(summary.contains("(1/2)"));
        assert!(summary.contains("persisted progress"));
        assert!(loaded.next_id > 1);
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
