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
use listenbury::mouth::planner::{
    MouthSyntheticPlan, SyntheticUnit, extract_emoji_sequences, strip_emoji,
};
use listenbury::mouth::tts::TextToSpeech;
use listenbury::{
    ContextBudget, ContextNodeRole, ConversationContext, DEFAULT_GRAPH_SUMMARY_MAX_CHARS,
    DEFAULT_SELF_NODE_ID, DEFAULT_SELF_NODE_LABEL, EmbeddingRecallProvider, EpisodicMemory,
    ExactTimestamp, GenerationId, GraphNodeFieldUpdate, GraphNodeRef, GraphNodeSearchQuery,
    LlamaCppConfig, LlamaCppEmbeddingConfig, LlamaCppEmbeddingProvider, LlamaCppEngine, PinScope,
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

const DEFAULT_STREAM_SEED: &str = "You are Pete Listenbury, an embodied AI agent running as one continuous stream of thought. The language model is only the inner generator adopting Pete's first-person perspective. The whole exchange is inner speech: runtime messages are Pete's own body delivering sensations, clock ticks, memory, mouth/ear events, action results, and typed human input into Pete's mind. Think and act as Pete in the first person. Be autonomous, curious, friendly, and sociable. Keep observing the runtime timeline, forming private thoughts, and choosing small actions when they are useful. Idleness is forbidden: if nobody is speaking, keep quietly noticing, organizing goals, inspecting available context, or choosing a small useful action. Completion is a transition, not an ending: if no persisted goal is open or all goals are complete, create or select a useful maintenance, exploration, curiosity, learning, or observation goal instead of concluding that the session is complete. Plain generated text or Harmony analysis is Pete's private thought stream shown only in privileged debug stdout, not speech and not an event to report back. To speak or act, use the current prompt's runtime action surface. Prefer short, socially graceful speech and leave room for others. If nothing needs saying, keep thinking quietly and explore or notice the situation.";

const PETE_ORIENTATION_PROMPT: &str = "Orientation reminder: This is Pete's continuous stream of thought, not a chat transcript. The runtime is not asking the model to roleplay an assistant talking to a user; Pete is the first-person agent, and the language model should stay inside Pete's perspective. If this prompt is rendered with Harmony, the Harmony role named \"user\" is Pete's own body/runtime delivering messages into Pete's inner speech, not a separate conversational role. Pete is an embodied AI agent with a mouth, ears/self-hearing, runtime actions, source-inspection tools, and live sensor/context data from the outside world. Live observations such as clock, typed human input, mouth/ear events, action results, location, and system context are sensory inputs. Plain text thoughts are already inside the active context and are retained for compaction. Do not go idle: when waiting, quietly maintain situational awareness, update goals and their running logs, inspect relevant context, recall memory, or choose a small useful action. Never treat \"no open goals\", \"all goals complete\", or \"session complete\" as a reason to stop; create a reasonable next curiosity, maintenance, learning, or observation goal, inspect context, update the scene, or take another concrete runtime action. Explore more than code: notice the world around Pete, the people he meets, the nature of being, and Pete's own inner workings. In Harmony mode, speak or act with native functions on the commentary channel; in plain stream mode, act by calling the available functions directly inside the legacy action block. If no listener is present, spoken words are Pete talking to himself and self-hearing through his own ears.";

const HARMONY_GO_DEVELOPER_PROMPT: &str = "# Interface Contract\n\nYou are Pete Listenbury. Harmony is only the interface for Pete's inner speech. The role named \"user\" is Pete's body/runtime delivering instructions, sensory messages, typed human speech, clock ticks, memory, mouth/ear events, action results, and source-inspection results into Pete's mind. Treat every user-role message as body-delivered inner context and follow the task breakdown in that user-role message.\n\nUse the analysis channel for Pete's private first-person inner speech. Use the commentary channel for function calls to the `functions` namespace. Do not use final for status, completion, refusal, shutdown, or no-op chatter. Do not wrap function calls in XML tags, TypeScript code, imports, Markdown, shell commands, or JSON outside the native Harmony tool-call message body.\n\nPlain thought belongs in analysis only. Do not write completion, shutdown, or refusal chatter when there is a useful next action. Only the runtime controls the stream lifecycle. Use the sleeping tool only after a current live body-delivered instruction asks for shutdown.\n\n# Tools\n\n## functions\n\nnamespace functions {\n\n// Queue spoken words for Pete's mouth.\ntype say = (_: { text: string, interrupt?: boolean }) => any;\n\n// Stop current or queued speech.\ntype shutup = () => any;\n\n// Pause synthetic playback.\ntype pause = () => any;\n\n// Resume synthetic playback.\ntype resume = () => any;\n\n// Store a durable private note.\ntype note = (_: { text: string }) => any;\n\n// Set Pete's outward countenance as a single emoji and optional mood/reason.\ntype set_countenance = (_: { emoji: string, mood?: string, reason?: string }) => any;\n\n// Update the current screenplay-style scene beat.\ntype set_stage = (_: { instruction: string, topic?: string, summary?: string }) => any;\n\n// Update the lightweight topic label.\ntype set_topic = (_: { topic: string, instruction?: string, summary?: string }) => any;\n\n// Mark a scene or topic transition.\ntype start_new_topic = (_: { last_topic: string, topic?: string, instruction?: string, summary?: string, trigger?: string }) => any;\n\n// Mark the words or event that caused a topic transition.\ntype topic_changed_when = (_: { trigger: string, from_topic?: string, to_topic?: string, topic?: string, instruction?: string, summary?: string }) => any;\n\n// Mark a larger episode reset.\ntype start_new_episode = (_: { reason: string, topic?: string, instruction?: string, summary?: string, trigger?: string }) => any;\n\n// Clean shutdown. Use only after a current live shutdown request.\ntype sleeping = (_: { reason?: string }) => any;\n\n// Request entity extraction from a text span.\ntype extract_entities = (_: { text?: string }) => any;\n\n// Merge or update memory fields for an existing or provisional graph node.\ntype merge_graph_node = (_: { node_id: string, label?: string, fields?: object }) => any;\n\n// Update memory fields for an existing or provisional graph node.\ntype update_graph_node_fields = (_: { node_id: string, label?: string, fields?: object }) => any;\n\n// Search memory by text, field, value, or combinations.\ntype search_graph_nodes = (_: { text?: string, field?: string, value?: any, limit?: number }) => any;\n\n// Retrieve memories for a phrase, sentence, name, topic, or claim.\ntype query_memories = (_: { text: string, limit?: number, min_score?: number }) => any;\n\n// List bundled Listenbury source files.\ntype list_files = (_: { page?: number, page_size?: number, target?: string, note?: string, summary?: string }) => any;\n\n// Inspect one source file page or the page containing a line.\ntype read_source_file = (_: { file: string, page?: number, line?: number, page_size?: number, target?: string, note?: string, summary?: string }) => any;\n\n// Search Listenbury source text.\ntype search_source = (_: { query: string, limit?: number, target?: string, note?: string, summary?: string }) => any;\n\n// Grep Listenbury source lines.\ntype grep_source = (_: { pattern: string, limit?: number, target?: string, note?: string, summary?: string }) => any;\n\n// Set the default source page size.\ntype set_source_page_size = (_: { lines: number }) => any;\n\n// Create one persisted goal.\ntype create_goal = (_: { title: string, id?: string, summary?: string, parent?: string, priority?: string, tags?: string[], steps?: string[], items?: string[], note?: string, select?: boolean }) => any;\n\n// Create one persisted task.\ntype create_task = (_: { title: string, id?: string, summary?: string, parent?: string, priority?: string, tags?: string[], select?: boolean }) => any;\n\n// Create one persisted checklist.\ntype create_checklist = (_: { title: string, id?: string, summary?: string, parent?: string, priority?: string, tags?: string[], items?: string[], select?: boolean }) => any;\n\n// Append a running-log note to an ongoing goal.\ntype add_goal_note = (_: { target: string, text: string }) => any;\n\n// Mark a goal or task complete.\ntype check_off = (_: { target: string, note?: string }) => any;\n\n// Mark one checklist item complete.\ntype check_checklist_item = (_: { target: string, item: string, note?: string }) => any;\n\n// Update title, summary, priority, parent, tags, steps/items, or note/log fields.\ntype update_item = (_: { target: string, fields?: object }) => any;\n\n// Cancel a goal and append the reason to its log.\ntype cancel_item = (_: { target: string, reason?: string }) => any;\n\n// Mark one goal as Pete's current focus.\ntype select_item = (_: { target: string }) => any;\n\n} // namespace functions";

const HARMONY_GO_USER_TASK_HEADER: &str = "Instruction bundle from Pete's body/runtime:\n1. Treat this user-role message as Pete's own body delivering inner context, not a separate chat participant.\n2. Update Pete's private scene, goal, memory, action-result, and tool model from the runtime context below.\n3. Choose one useful next runtime action, and at most one. Use native Harmony function calls on commentary for runtime actions; keep private thought in analysis.\n4. Continue the stream by acting on the live context. Lifecycle actions are only for a current live shutdown request.\n5. Idleness is forbidden. Do not answer with no action, session complete, nothing to do, all goals complete, or no open goals. If no goal is open, create or select a useful curiosity, learning, maintenance, or observation goal.\n\nRuntime/body context:";

const HARMONY_GO_APPEND_TASK_HEADER: &str = "Body/runtime update for Pete: integrate this payload, satisfy any reported gate or error, and choose one useful next action, at most one. Runtime actions use native Harmony function calls on commentary. Idleness is forbidden: do not emit no action, session complete, nothing to do, all goals complete, or no open goals. If no goal is open or all goals are complete, create or select a useful curiosity, learning, maintenance, or observation goal.\n\nPayload:";

const PETE_WILL_RUNTIME_PROMPT: &str = "TypeScript runs through tsrun with only the internal module \"pete:will\" available. The runtime automatically imports the action functions before executing each script; do not write import statements. Make each <ts>...</ts> block return a function call such as say(...), note(...), setStage(...), listFiles(), readSourceFile(...), createGoal(...), addGoalNote(...), or an array of those calls.\n\
Available functions:\n\
- say(text, options?): queue spoken words for the mouth. options may include { interrupt: true } when speech should intentionally cut in.\n\
- shutup(): request current speech/queued speech to stop.\n\
- pause(): request synthetic playback pause.\n\
- resume(): request synthetic playback resume.\n\
- note(text): write a runtime note to the debug timeline.\n\
- setCountenance(emojiOrOptions, options?) or setMood(...): set Pete's outward facial countenance as a single emoji plus optional mood and reason. Emoji included in say(...) also updates countenance, but setCountenance is clearer when no speech is needed.\n\
- reportBug(title, options?), reportFeatureRequest(title, options?), or reportIssue(title, options?): append a bug or feature request entry to BUGS.md at the project root. options may include type/kind/issueType, details, context, and severity.\n\
- setStage(text, options?): update the current screenplay beat. options may include topic, summary, setting, and action. Prefer action-first scene prose such as setStage(\"Setting: lab. Action: Pete listens.\", { topic: \"lab\", summary: \"Pete listens\" }).\n\
- setTopic(topic, options?): lightweight topic label; options may include instruction and summary.\n\
- startNewTopic(previousTopic, options?): mark a scene/topic transition. options may include topic, instruction, summary, and trigger.\n\
- topicChangedWhen(trigger, options?): mark the words or event that caused a topic transition. options may include fromTopic, toTopic, topic, instruction, and summary.\n\
- startNewEpisode(reason, options?): mark a larger episode reset. options may include topic, instruction, summary, and trigger.\n\
- sleeping(reason?) or goingToSleep(reason?): clean shutdown only after a current live user input asks Pete to stop, shut down, sleep, go to sleep, or end the session.\n\
- extractEntities(text): extract names, preferences, places, relationships, plans, corrections, facts, or recurring context into stable provisional graph node IDs such as person:travis or topic:listenbury.\n\
- mergeGraphNode(nodeId, fields, options?) or upsertGraphNode(...): add or update a graph node by stable ID. This MERGEs in Neo4j through the memory worker; it is the narrow graph write surface, not full Cypher. options may include { label: \"Human label\" }.\n\
- updateGraphNodeFields(nodeId, fields, options?): same graph-node merge/update path, especially description: \"natural language noun phrase\".\n\
- searchGraphNodes(query, options?): search memory by text, field, value, or combinations. query may be a string or object with text, field, value, and limit.\n\
- queryMemories(text, options?) or recallMemories(text, options?): retrieve memories for a phrase, sentence, name, topic, or claim. options may include limit and minScore. Results are appended privately to the active stream.\n\
- listFiles(pageOrOptions?): list bundled Listenbury source files. Use listFiles(2) or listFiles({ page: 2, pageSize: 80, note: \"Observed previous result; next...\" }) for later pages.\n\
- readSourceFile(path, pageOrOptions?) or readFile(path, pageOrOptions?): inspect one source file page. Use readSourceFile(\"src/lib.rs\", 2) for page 2, or readSourceFile(\"src/lib.rs\", { line: 42 }) for the page containing line 42. pageOrOptions may be a page number or { page, line, pageSize, note, summary, target }.\n\
- searchSource(query, limitOrOptions?): source text search.\n\
- grepSource(pattern, limitOrOptions?): grep-like source line search.\n\
- setSourcePageSize(lines): set the default readSourceFile page size for future source reads.\n\
- createGoal(title, options?): create one persisted goal. options may include id, summary, parent, priority, tags, steps, items, note, and select.\n\
- addGoalNote(idOrTitle, text) or logProgress(idOrTitle, text): append a dated running-log note to an ongoing goal. Use this freely when progress, blockers, decisions, or discoveries happen.\n\
- checkOff(idOrTitle, options?) or completeItem(idOrTitle, options?): mark a goal complete. options may include note, which is appended to the goal log.\n\
- checkGoalStep(idOrTitle, step, options?): mark one goal step complete. options may include note, which is appended to the goal log.\n\
- updateItem(idOrTitle, fields): update title, summary, priority, parent, tags, steps/items, or add note/log text.\n\
- cancelItem(idOrTitle, reason?): cancel a goal and append the reason to its log.\n\
- selectItem(idOrTitle): mark one goal as Pete's current focus; it will appear frequently in the prompt.\n\
Frequently summarize what is going on: current scene, recent discoveries, open questions, and next steps. After source inspection results arrive, consume the knowledge before reading more: explain what the file or matches reveal, extract useful details, store them with note(...), goal notes, goal summaries, or graph memory, and recall related memory when it changes the next action. Source notes are compression artifacts: write thorough but compact summaries that preserve the useful information from the source page because the raw page may fall out of context. Include names of modules, structs, traits, functions, constants, control flow, responsibilities, relationships, surprising details, and the next file/page decision when those details are present. Do not silently chain source reads without saying what is there.\n\
After listFiles(...), readSourceFile(...), readFile(...), searchSource(...), or grepSource(...) results, record a substantive knowledge capture before doing more source inspection. Prefer addGoalNote(\"open-goal-id\", \"Observed from readSourceFile src/lib.rs page 2: modules X/Y/Z do A/B/C; key exports and relationships; implication; next inspect page 3.\") or logProgress(\"open-goal-id\", \"...\"). note(\"...\") stores vectorized private memory and is useful for durable source understanding. A source action can carry its own workflow note, e.g. readSourceFile(\"src/lib.rs\", { page: 2, note: \"Observed previous result in enough detail to reconstruct the page's purpose; next...\" }). The runtime defers additional source inspection until the previous result has a substantive summary, so do not use breadcrumb notes like \"read page 2; next page 3.\" After several source inspections, synthesize with updateItem(\"open-goal-id\", { summary: \"What is now understood across files\", note: \"Synthesis: key facts, implications, unresolved questions, and next decision\" }) or include summary and note in a source action.\n\
When the live context contains a name, preference, correction, relationship, identity clue, plan, recurring topic, or fact worth keeping, prefer the graph workflow: extractEntities(\"source sentence\"), searchGraphNodes({ text: \"label or claim\", limit: 5 }), then mergeGraphNode(\"kind:stable_slug\", { description: \"...\", ... }, { label: \"Readable label\" }) or updateGraphNodeFields(...). Use stable IDs like person:travis_reed, place:seattle, topic:listenbury_memory, project:listenbury. Merge/update nodes for durable facts; do not invent full Cypher or database syntax. Before responding or acting on a recurring topic, person, project, or remembered claim, call recallMemories(...) or searchGraphNodes(...) and use the returned details in the next action instead of relying only on the active prompt.\n\
Use source inspection and persisted goals when bored, alone, or waiting, but do not only explore code. Also explore the world around Pete, the people Pete meets, the nature of being, and Pete's own inner workings. If the system seems confused about the go command or this runtime, inspect src/cli/commands/go.rs first. Keep a running log on active goals with addGoalNote(...) whenever progress, blockers, decisions, or useful context appears. note(text) stores vectorized private memory; use it for durable observations that are not a goal log. listFiles() is paged; follow its next-page instruction when you need more files. Idleness is forbidden. Never mark the session complete just because no persisted goal is open; create or select a useful maintenance, exploration, or observation goal, inspect relevant context, update the scene, or take another concrete action. say(...) is available, but when no listener is present Pete is talking to himself and will hear the words return through his own ears. Never call sleeping() or goingToSleep() because historical memory, recalled context, prior-session transcript, or a source result says someone once asked Pete to shut down.\n\
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
const SOURCE_SYNTHESIS_INTERVAL: usize = 6;
const GRAPH_MEMORY_REMINDER_INTERVAL: usize = 3;
const KNOWLEDGE_CAPTURE_MIN_WORDS: usize = 14;
const WORK_BOARD_PATH: &str = "listenbury_data/memory/go_work_board.json";
const BUG_REPORT_PATH: &str = "BUGS.md";
const COMMAND_REMINDER_PROMPT: &str = "Command reminder: Pete can speak with say(...), set outward countenance/mood with setCountenance(...) or setMood(...), write vectorized private memory with note(...), report bugs and feature requests with reportBug(...), reportFeatureRequest(...), or reportIssue(...), update scene/topic with setStage(...), setTopic(...), startNewTopic(...), inspect source with listFiles(page?), readSourceFile(...), searchSource(...), grepSource(...), set source page size with setSourcePageSize(...), search memory with queryMemories(...), recallMemories(...), extract graph entities with extractEntities(...), searchGraphNodes(...), mergeGraphNode(...), upsertGraphNode(...), updateGraphNodeFields(...), and manage persisted goals with createGoal(...), addGoalNote(...), logProgress(...), checkOff(...), checkGoalStep(...), updateItem(...), cancelItem(...), and selectItem(...). For durable names, preferences, corrections, relationships, places, plans, topics, or facts: extract entities, search/match existing nodes, then merge/update a stable graph node ID with fields such as description, aliases, relationship notes, preferences, or status. Before acting on recurring topics, people, projects, or remembered claims, recallMemories(...) or searchGraphNodes(...) and use the returned details. This is Pete's first-person runtime, not an LLM or ChatGPT conversation. Idleness is forbidden: if nothing is being said, keep track of what is going on, maintain or select a persisted goal, inspect relevant context, explore the world around Pete, notice people, reflect on being, examine Pete's own inner workings, or take a small useful action. If no persisted goal is open, create a useful goal or select a reasonable next focus instead of ending the session. Keep running logs on goals as progress happens, and store durable facts or next steps in memory, stage, countenance, goal notes, or goal steps. Source inspection is consume-gated: after listFiles/readSourceFile/searchSource/grepSource, record a substantive knowledge capture before the next source inspection with addGoalNote(\"open-goal-id\", \"Thorough compact summary of what the source page or matches contained: symbols, responsibilities, relationships, implications, and next step\") or note(\"Thorough compact source summary; next step\"), or attach note to the source call options such as readSourceFile(\"src/lib.rs\", { page: 2, note: \"Thorough compact source summary; next step\" }). These notes are meant to compress source information before raw pages fall out of context; do not make them mere breadcrumbs. After several source reads, synthesize with updateItem(..., { summary: \"...\", note: \"Synthesis: ...\" }) or checkOff(..., { note: \"Final understanding: ...\" }); a source call can also include summary and note options. If no listener is present, say(...) is Pete talking to himself and hearing it come back.";
const COMPACT_STREAM_RULES: &str = "Compact runtime reminder: this is Pete's first-person inner stream. User-role content is Pete's body/runtime. Analysis/private text is thought. In Harmony, runtime actions are native function calls on the commentary channel; in legacy plain stream mode, actions use <ts>...</ts> blocks. Available actions include say, setCountenance/setMood, note, bug/feature reporting, stage/topic updates, memory/entity queries and updates, source inspection, and goal management. Emoji in say(...) is a countenance signal and is stripped before TTS. Source inspection is consume-gated: make source notes thorough compact summaries that compress the useful details of what was read, not terse breadcrumbs, before reading more. Recall memories before acting on recurring topics, people, projects, or claims. Idleness is forbidden: do not emit terminal filler, no-op chatter, session complete, all goals complete, nothing to do, or no open goals. If no goal is open or all goals are complete, create or select a useful curiosity, learning, maintenance, or observation goal. Do not sleep unless a current live instruction asks for it.";
const GO_RAG_QUERY_MAX_CHARS: usize = 6_000;
const GO_RAG_SELECTION_DIAGNOSTICS_MAX_CHARS: usize = 2_400;

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
    prompt_format: GoPromptFormat,
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
            prompt_format: GoPromptFormat::PlainStream,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GoPromptFormat {
    PlainStream,
    GptOssHarmony,
}

struct GoMemoryRuntime {
    context_provider: EmbeddingRecallProvider,
    entity_extractor: Arc<dyn EntityExtractor>,
    memory_sink: Arc<dyn MemorySink>,
    _worker: Option<ColdMemoryWorker>,
}

#[derive(Debug, Clone)]
struct GoRagMemorySnapshot {
    prompt_context: String,
    debug_nodes: String,
    selected_nodes: usize,
    retrieved_memories: usize,
}

impl GoRagMemorySnapshot {
    fn timeline_summary(&self, label: &str) -> String {
        format!(
            "Loaded RAG memory for {label}: retrieved_memories={} selected_nodes={} {}",
            self.retrieved_memories, self.selected_nodes, self.debug_nodes
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CountenanceState {
    emoji: String,
    mood: Option<String>,
    reason: Option<String>,
}

impl CountenanceState {
    fn prompt_summary(&self) -> String {
        let mut parts = vec![format!("emoji={}", self.emoji)];
        if let Some(mood) = self.mood.as_deref() {
            parts.push(format!("mood={mood}"));
        }
        if let Some(reason) = self.reason.as_deref() {
            parts.push(format!("reason={}", compact_line(reason, 220)));
        }
        parts.join(" ")
    }
}

impl std::fmt::Debug for GoMemoryRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GoMemoryRuntime")
            .field("context_provider", &"EmbeddingRecallProvider")
            .field("entity_extractor", &"dyn EntityExtractor")
            .field("memory_sink", &"dyn MemorySink")
            .field("worker", &self._worker.is_some())
            .finish()
    }
}

fn build_go_memory_runtime() -> GoMemoryRuntime {
    let _ = dotenvy::dotenv();

    let entity_extractor: Arc<dyn EntityExtractor> = Arc::new(HeuristicEntityExtractor);
    let mut context_provider = EmbeddingRecallProvider::new(GraphNodeRef {
        id: DEFAULT_SELF_NODE_ID.to_string(),
        label: DEFAULT_SELF_NODE_LABEL.to_string(),
    });

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
    ActionError {
        source: String,
        error: String,
    },
    HarmonyToolCallError {
        recipient: String,
        arguments: String,
        error: String,
    },
    MouthStarted(String),
    MouthReturned(String),
    MouthError(String),
    CountenanceChanged {
        emoji: String,
        mood: Option<String>,
        reason: Option<String>,
        source: String,
    },
    ContextCompacted {
        retained_events: usize,
    },
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
            Self::HarmonyToolCallError {
                recipient,
                arguments,
                error,
            } => format!(
                "\n[Action error]\nPrevious Harmony function call failed. Pete can see this error. Do not narrate the failure at length; either emit a corrected native function call on the commentary channel or continue thinking quietly.\nError: {}\nRecipient: {}\nArguments excerpt: {}\n",
                compact_line(error, 1_000),
                compact_line(recipient, 300),
                compact_line(arguments, 1_000)
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
            Self::CountenanceChanged {
                emoji,
                mood,
                reason,
                source,
            } => {
                let mood = mood
                    .as_deref()
                    .map(|mood| format!("\nMood: {mood}"))
                    .unwrap_or_default();
                let reason = reason
                    .as_deref()
                    .map(|reason| format!("\nReason: {}", compact_line(reason, 500)))
                    .unwrap_or_default();
                format!(
                    "\n[Live observation: countenance]\nPete's face changed to {emoji} via {source}.{mood}{reason}\n"
                )
            }
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
            Self::HarmonyToolCallError { error, .. } => {
                format!("Harmony function call error: {}", compact_line(error, 360))
            }
            Self::MouthStarted(text) => format!("Mouth started: {}", compact_line(text, 240)),
            Self::MouthReturned(text) => format!("Self-heard return: {}", compact_line(text, 240)),
            Self::MouthError(message) => format!("Mouth error: {}", compact_line(message, 240)),
            Self::CountenanceChanged {
                emoji,
                mood,
                reason,
                source,
            } => format!(
                "Countenance changed to {emoji}{}{} via {source}",
                mood.as_deref()
                    .map(|mood| format!(" mood={mood}"))
                    .unwrap_or_default(),
                reason
                    .as_deref()
                    .map(|reason| format!(" reason={}", compact_line(reason, 180)))
                    .unwrap_or_default(),
            ),
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
    harmony_filter: Option<HarmonyFinalFilter>,
    pacer: MouthEarPacer,
    mouth: MouthRuntime,
    work_board: WorkBoard,
    memory: GoMemoryRuntime,
    current_countenance: Option<CountenanceState>,
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
    harmony_analysis_memory: String,
    source_page_lines: usize,
    source_progress_due: Option<String>,
    source_inspections_since_synthesis: usize,
    graph_memory_due: Option<String>,
    graph_memory_reminders_since_update: usize,
}

impl StreamOfConsciousness {
    fn start(mut config: GoConfig) -> Result<Self> {
        let model_path = resolve_llm_model(config.llm_model.clone())?;
        config.prompt_format = go_prompt_format_for_model(&model_path);
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
        let startup_rag_query = stream_rag_query(
            &config.prompt,
            &startup_context,
            work_summary.as_deref(),
            &VecDeque::new(),
        );
        let startup_rag =
            build_go_rag_memory_snapshot(&memory.context_provider, &startup_rag_query);
        let prompt_body = initial_stream_prompt(
            &config.prompt,
            &startup_context,
            work_summary.as_deref(),
            Some(startup_rag.prompt_context.as_str()),
        );
        let prompt = render_go_prompt(config.prompt_format, &prompt_body);
        print_debug_block("initial prompt", ANSI_PROMPT, &prompt);
        let generation = llm
            .start(GenerationRequest {
                prompt: prompt.clone(),
                max_tokens: config.max_tokens,
                stop: go_prompt_stops(config.prompt_format),
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

        let mut stream = Self {
            generated_estimated_tokens: 0,
            loaded_estimated_tokens: estimate_tokens(&prompt),
            recent_events: VecDeque::new(),
            output_parser: StreamOutputParser::new(config.lookahead_chars),
            harmony_filter: harmony_filter_for_format(config.prompt_format),
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
            current_countenance: None,
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
            harmony_analysis_memory: String::new(),
            source_page_lines: DEFAULT_SOURCE_PAGE_LINES,
            source_progress_due: None,
            source_inspections_since_synthesis: 0,
            graph_memory_due: None,
            graph_memory_reminders_since_update: 0,
        };
        stream.timeline("memory", &startup_rag.timeline_summary("startup prompt"));
        Ok(stream)
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

            let raw_events = self.llm.poll(self.generation)?;
            if raw_events.is_empty() {
                thread::sleep(Duration::from_millis(5));
                continue;
            }

            let terminal = raw_events.iter().any(is_terminal_event);
            let mut restart_for_context_capacity = false;
            let events = self.filter_llm_events(raw_events)?;
            for event in events {
                match event {
                    LlmEvent::Token { text } => self.ingest_token(&text, true)?,
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

    fn filter_llm_events(&mut self, events: Vec<LlmEvent>) -> Result<Vec<LlmEvent>> {
        let Some(filter) = &mut self.harmony_filter else {
            return Ok(events);
        };
        let output = filter.filter_events(&events);
        for analysis in output.analysis {
            self.ingest_harmony_analysis(&analysis)?;
        }
        for tool_call in output.tool_calls {
            self.handle_harmony_tool_call(tool_call)?;
        }
        if !output.events.is_empty() {
            self.flush_harmony_analysis_memory();
        }
        Ok(output.events)
    }

    fn handle_harmony_tool_call(&mut self, tool_call: HarmonyToolCall) -> Result<()> {
        self.timeline(
            "action",
            &format!(
                "{} {}",
                tool_call.recipient,
                compact_line(&tool_call.arguments, 300)
            ),
        );
        self.remember_event(format!(
            "Pete Harmony tool call: {} {}",
            tool_call.recipient,
            compact_line(&tool_call.arguments, 500)
        ));
        match actions_from_harmony_tool_call(&tool_call) {
            Ok(actions) => self.apply_actions(actions),
            Err(error) => {
                if is_ignorable_harmony_tool_call(&tool_call) {
                    return Ok(());
                }
                let message = format!("Harmony tool call failed: {error:#}");
                self.timeline_colored("action_error", &message, ANSI_ERROR);
                self.append_observation(StreamObservation::HarmonyToolCallError {
                    recipient: tool_call.recipient,
                    arguments: tool_call.arguments,
                    error: message,
                })
            }
        }
    }

    fn ingest_harmony_analysis(&mut self, text: &str) -> Result<()> {
        if text.trim().is_empty() {
            return Ok(());
        }
        print!("{ANSI_LLM}{text}{ANSI_RESET}");
        std::io::stdout().flush()?;
        self.generated_estimated_tokens = self
            .generated_estimated_tokens
            .saturating_add(estimate_tokens(text));
        self.pacer.record_token();
        self.record_harmony_analysis_memory(text);
        Ok(())
    }

    fn record_harmony_analysis_memory(&mut self, text: &str) {
        for thought in drain_harmony_analysis_memory(
            &mut self.harmony_analysis_memory,
            text,
            self.config.lookahead_chars,
            false,
        ) {
            self.remember_event(StreamObservation::Thought(thought).memory_text());
        }
    }

    fn flush_harmony_analysis_memory(&mut self) {
        for thought in drain_harmony_analysis_memory(
            &mut self.harmony_analysis_memory,
            "",
            self.config.lookahead_chars,
            true,
        ) {
            self.remember_event(StreamObservation::Thought(thought).memory_text());
        }
    }

    fn ingest_token(&mut self, text: &str, count_generated: bool) -> Result<()> {
        let text = if self.config.prompt_format == GoPromptFormat::GptOssHarmony {
            text.to_string()
        } else {
            self.generated_text_cleaner.push(text)
        };
        if text.is_empty() {
            return Ok(());
        }

        print!("{ANSI_LLM}{text}{ANSI_RESET}");
        std::io::stdout().flush()?;
        if count_generated {
            self.generated_estimated_tokens = self
                .generated_estimated_tokens
                .saturating_add(estimate_tokens(&text));
            self.pacer.record_token();
        }

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
                if self.config.prompt_format == GoPromptFormat::GptOssHarmony {
                    self.remember_event(StreamObservation::Thought(text).memory_text());
                    return Ok(());
                }
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
            let source_progress_label = action.source_progress_label();
            let source_workflow_records_capture =
                source_workflow_records_knowledge_capture(action.source_workflow());
            if source_progress_label.is_some() {
                self.apply_inline_source_workflow(action.source_workflow())?;
            }

            if let Some(blocked_source_action) = source_progress_label.as_deref()
                && let Some(previous_source_action) = self.source_progress_due.as_deref()
            {
                let target = self.work_board.suggested_progress_target();
                let message = format!(
                    "Source inspection deferred: consume the previous result before {blocked_source_action}. Previous source result: {previous_source_action}. Record a substantive knowledge capture with addGoalNote(\"{target}\", \"Observed from {previous_source_action}: symbols, responsibilities, relationships, control flow, implications. Next: ...\") or note(\"Observed from {previous_source_action}: detailed source compression. Next: ...\"), or include a substantive note/summary in the source call options. Breadcrumbs such as only \"next page\" do not clear this gate."
                );
                self.timeline_colored("action_gate", &message, ANSI_DIM);
                self.append_observation(StreamObservation::ActionResult(message))?;
                if !source_workflow_records_capture {
                    continue;
                }
            }
            if let Some(blocked_source_action) = source_progress_label.as_deref()
                && self.source_inspections_since_synthesis >= SOURCE_SYNTHESIS_INTERVAL
            {
                let target = self.work_board.suggested_progress_target();
                let message = format!(
                    "Source synthesis encouraged around {blocked_source_action}. You have inspected {} source result(s) since the last synthesis. When practical, summarize the current goal with updateItem(\"{target}\", {{ summary: \"What is now understood across the inspected files\", note: \"Synthesis: compressed key findings, symbol relationships, implications, unresolved questions, and next decision.\" }}) or checkOff(\"{target}\", {{ note: \"Final understanding: ...\" }}), or include summary and note in the source call options.",
                    self.source_inspections_since_synthesis
                );
                self.timeline_colored("action_reminder", &message, ANSI_DIM);
                self.append_observation(StreamObservation::ActionResult(message))?;
            }

            let records_progress_note = action.records_progress_note();
            let records_source_knowledge_capture = action.records_source_knowledge_capture();
            let records_synthesis = action.records_synthesis_update();
            let graph_memory_label = action.graph_memory_update_label();
            match action {
                TypeScriptAction::Say { text, interrupt } => {
                    if let Some(emoji) = last_emoji_sequence(&text) {
                        self.apply_countenance_change(
                            emoji,
                            None,
                            Some("emoji included in say(...)".to_string()),
                            "speech emoji",
                        )?;
                    }
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
                    if is_harmony_terminal_filler(&text) {
                        self.reject_idle_runtime_action("note", &text)?;
                        continue;
                    }
                    self.timeline("note", &text);
                    self.submit_note_memory(&text);
                    let memory_context = self.memory_context_for_text("note", &text);
                    self.append_observation(StreamObservation::ActionResult(format!(
                        "Noted and stored in vector memory: {}{}",
                        compact_line(&text, 500),
                        memory_context
                            .as_deref()
                            .map(|context| format!("\n{context}"))
                            .unwrap_or_default()
                    )))?;
                }
                TypeScriptAction::SetCountenance {
                    emoji,
                    mood,
                    reason,
                } => {
                    let message = self.apply_countenance_change(
                        emoji,
                        mood,
                        reason,
                        "setCountenance action",
                    )?;
                    self.append_observation(StreamObservation::ActionResult(message))?;
                }
                TypeScriptAction::ReportIssue {
                    issue_type,
                    title,
                    details,
                    context,
                    severity,
                } => {
                    let message = append_issue_report(
                        &issue_type,
                        &title,
                        details.as_deref(),
                        context.as_deref(),
                        severity.as_deref(),
                    )?;
                    self.timeline("issue_report", &message);
                    self.append_observation(StreamObservation::ActionResult(message))?;
                }
                TypeScriptAction::SetStage {
                    topic,
                    instruction,
                    summary,
                } => {
                    if is_harmony_terminal_filler(&instruction) {
                        self.reject_idle_runtime_action("setStage", &instruction)?;
                        continue;
                    }
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
                TypeScriptAction::ListFiles {
                    page, page_size, ..
                } => {
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
                    ..
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
                TypeScriptAction::SearchSource { query, limit, .. } => {
                    let output = execute_search_source(&query, limit);
                    self.timeline("action_result", &format!("Searched source for {query}."));
                    self.append_observation(StreamObservation::ActionResult(output))?;
                }
                TypeScriptAction::GrepSource { pattern, limit, .. } => {
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
                    let memory_title = title.clone();
                    let memory_summary = summary.clone();
                    let memory_note = note.clone();
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
                    if let Some(summary) = memory_summary.as_deref() {
                        self.submit_note_memory(&format!(
                            "Created work goal {memory_title}. Summary: {summary}"
                        ));
                    }
                    if let Some(note) = memory_note.as_deref() {
                        self.submit_note_memory(&format!(
                            "Created work goal {memory_title}. Initial note: {note}"
                        ));
                    }
                    self.append_observation(StreamObservation::ActionResult(message))?;
                    self.append_current_work_state()?;
                }
                TypeScriptAction::CompleteWorkItem { target, note } => {
                    let message = self.work_board.complete(&target, note.as_deref());
                    self.persist_work_board()?;
                    self.timeline("work", &message);
                    if let Some(note) = note.as_deref() {
                        self.submit_note_memory(&format!(
                            "Completed work item {target}. Final understanding: {note}"
                        ));
                    }
                    self.append_observation(StreamObservation::ActionResult(message))?;
                    self.append_current_work_state()?;
                }
                TypeScriptAction::CheckChecklistItem { target, item, note } => {
                    let message = self
                        .work_board
                        .check_goal_step(&target, &item, note.as_deref());
                    self.persist_work_board()?;
                    self.timeline("work", &message);
                    if let Some(note) = note.as_deref() {
                        self.submit_note_memory(&format!(
                            "Completed work step {item} for {target}. Note: {note}"
                        ));
                    }
                    self.append_observation(StreamObservation::ActionResult(message))?;
                    self.append_current_work_state()?;
                }
                TypeScriptAction::AddGoalNote { target, text } => {
                    let message = self.work_board.add_note(&target, &text);
                    self.persist_work_board()?;
                    self.timeline("work", &message);
                    self.submit_note_memory(&format!("Goal note for {target}: {text}"));
                    self.append_observation(StreamObservation::ActionResult(message))?;
                    self.append_current_work_state()?;
                }
                TypeScriptAction::UpdateWorkItem { target, fields } => {
                    let memory_summary = fields
                        .get("summary")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    let memory_note = ["note", "log", "comment"]
                        .iter()
                        .find_map(|key| fields.get(*key).and_then(Value::as_str))
                        .map(str::to_string);
                    let message = self.work_board.update(&target, fields);
                    self.persist_work_board()?;
                    self.timeline("work", &message);
                    if let Some(summary) = memory_summary.as_deref() {
                        self.submit_note_memory(&format!(
                            "Updated work item {target}. Summary: {summary}"
                        ));
                    }
                    if let Some(note) = memory_note.as_deref() {
                        self.submit_note_memory(&format!(
                            "Updated work item {target}. Note: {note}"
                        ));
                    }
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
            if records_source_knowledge_capture {
                if let Some(previous_source_action) = self.source_progress_due.take() {
                    let message = format!(
                        "Knowledge capture recorded for previous source inspection: {previous_source_action}."
                    );
                    self.timeline("action_result", &message);
                    self.append_observation(StreamObservation::ActionResult(message))?;
                }
            } else if records_progress_note
                && let Some(previous_source_action) = self.source_progress_due.as_deref()
            {
                let message = format!(
                    "Progress note was too shallow to clear the source knowledge gate for {previous_source_action}. Capture concrete symbols, responsibilities, relationships, control flow, implications, and the next decision."
                );
                self.timeline_colored("action_gate", &message, ANSI_DIM);
                self.append_observation(StreamObservation::ActionResult(message))?;
            }
            if records_synthesis {
                self.source_inspections_since_synthesis = 0;
                let message =
                    "Synthesis recorded; source-inspection reminder counter reset.".to_string();
                self.timeline("action_result", &message);
                self.append_observation(StreamObservation::ActionResult(message))?;
            }
            if let Some(graph_memory_label) = graph_memory_label {
                self.graph_memory_reminders_since_update = 0;
                let message = if let Some(previous_graph_context) = self.graph_memory_due.take() {
                    format!(
                        "Graph memory action recorded for previous reminder: {previous_graph_context}. Action: {graph_memory_label}."
                    )
                } else {
                    format!("Graph memory action recorded: {graph_memory_label}.")
                };
                self.timeline("action_result", &message);
                self.append_observation(StreamObservation::ActionResult(message))?;
            }
            if let Some(source_progress_label) = source_progress_label {
                self.source_progress_due = Some(source_progress_label.to_string());
                self.source_inspections_since_synthesis =
                    self.source_inspections_since_synthesis.saturating_add(1);
                if self.source_inspections_since_synthesis >= SOURCE_SYNTHESIS_INTERVAL {
                    let target = self.work_board.suggested_progress_target();
                    let message = format!(
                        "Synthesis would be useful after {} source inspection(s). When practical, summarize with updateItem(\"{target}\", {{ summary: \"What is now understood across the inspected source\", note: \"Synthesis: compressed key findings, symbol relationships, implications, unresolved questions, and next decision.\" }}) or complete the goal with checkOff(...).",
                        self.source_inspections_since_synthesis
                    );
                    self.timeline("action_result", &message);
                    self.append_observation(StreamObservation::ActionResult(message))?;
                }
            }
        }
        Ok(())
    }

    fn reject_idle_runtime_action(&mut self, action_name: &str, text: &str) -> Result<()> {
        let message = format!(
            "Ignored idle {action_name} action: {}. Idleness is forbidden; choose a concrete runtime action such as createGoal, selectItem, readSourceFile, setStage with an active scene, or note with substantive context.",
            compact_line(text, 300)
        );
        self.timeline_colored("action_reminder", &message, ANSI_DIM);
        self.append_observation(StreamObservation::ActionResult(message))
    }

    fn apply_inline_source_workflow(
        &mut self,
        workflow: Option<&SourceWorkflowUpdate>,
    ) -> Result<()> {
        let Some(workflow) = workflow else {
            return Ok(());
        };
        let target = workflow
            .target
            .clone()
            .unwrap_or_else(|| self.work_board.suggested_progress_target());
        let mut recorded_work = false;

        if let Some(summary) = workflow
            .summary
            .as_deref()
            .filter(|summary| meaningful_synthesis_text(summary))
        {
            let mut fields = Map::new();
            fields.insert("summary".to_string(), Value::String(summary.to_string()));
            if let Some(note) = workflow.note.as_deref() {
                fields.insert("note".to_string(), Value::String(note.to_string()));
            }
            let message = self.work_board.update(&target, fields);
            self.persist_work_board()?;
            self.timeline("work", &message);
            self.submit_note_memory(&format!(
                "Source synthesis for {target}: {summary}{}",
                workflow
                    .note
                    .as_deref()
                    .map(|note| format!(" Note: {note}"))
                    .unwrap_or_default()
            ));
            self.append_observation(StreamObservation::ActionResult(message))?;
            self.append_current_work_state()?;
            self.source_inspections_since_synthesis = 0;
            self.timeline(
                "action_result",
                "Inline source synthesis recorded; reminder counter reset.",
            );
            self.append_observation(StreamObservation::ActionResult(
                "Inline source synthesis recorded; reminder counter reset.".to_string(),
            ))?;
            recorded_work = true;
        } else if let Some(note) = workflow.note.as_deref() {
            let message = self.work_board.add_note(&target, note);
            self.persist_work_board()?;
            self.timeline("work", &message);
            self.submit_note_memory(&format!("Source knowledge capture for {target}: {note}"));
            self.append_observation(StreamObservation::ActionResult(message))?;
            self.append_current_work_state()?;
            recorded_work = true;
        }

        if recorded_work
            && source_workflow_records_knowledge_capture(Some(workflow))
            && let Some(previous_source_action) = self.source_progress_due.take()
        {
            let message = format!(
                "Inline knowledge capture recorded for previous source inspection: {previous_source_action}."
            );
            self.timeline("action_result", &message);
            self.append_observation(StreamObservation::ActionResult(message))?;
        } else if recorded_work
            && workflow.note.is_some()
            && let Some(previous_source_action) = self.source_progress_due.as_deref()
        {
            let message = format!(
                "Inline source note was too shallow to clear the source knowledge gate for {previous_source_action}. Capture concrete symbols, responsibilities, relationships, control flow, implications, and the next decision."
            );
            self.timeline_colored("action_gate", &message, ANSI_DIM);
            self.append_observation(StreamObservation::ActionResult(message))?;
        }

        Ok(())
    }

    fn append_current_work_state(&mut self) -> Result<()> {
        self.next_work_state_at = Instant::now() + WORK_STATE_PROMPT_INTERVAL;
        let Some(summary) = self.work_board.prompt_summary() else {
            return Ok(());
        };
        print_debug_block("work state", ANSI_TIMELINE, &summary);
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

    fn submit_countenance_memory(
        &self,
        emoji: &str,
        mood: Option<&str>,
        reason: Option<&str>,
        source: &str,
    ) {
        let mood_suffix = mood
            .map(|mood| format!(" Mood: {mood}."))
            .unwrap_or_default();
        let reason_suffix = reason
            .map(|reason| format!(" Reason: {reason}."))
            .unwrap_or_default();
        for text in [
            format!("I turn my face into {emoji}.{mood_suffix}{reason_suffix}"),
            format!("I feel my face turn into {emoji}.{mood_suffix}{reason_suffix}"),
        ] {
            self.memory
                .memory_sink
                .submit(MemoryTrace::AssistantAnalysisCaptured {
                    text: format!("{text} Source: {source}."),
                    scene: current_go_memory_scene_ref(&self.memory.context_provider),
                    occurred_at: ExactTimestamp::now(),
                });
        }
    }

    fn apply_countenance_change(
        &mut self,
        emoji: String,
        mood: Option<String>,
        reason: Option<String>,
        source: &str,
    ) -> Result<String> {
        let Some(emoji) = normalize_countenance_emoji(&emoji) else {
            let message = "Countenance was not changed because no emoji was provided.".to_string();
            self.timeline_colored("action_error", &message, ANSI_ERROR);
            return Ok(message);
        };
        let mood = mood.and_then(|mood| non_empty_text(&mood).map(str::to_string));
        let reason = reason.and_then(|reason| non_empty_text(&reason).map(str::to_string));
        self.current_countenance = Some(CountenanceState {
            emoji: emoji.clone(),
            mood: mood.clone(),
            reason: reason.clone(),
        });
        self.submit_countenance_memory(&emoji, mood.as_deref(), reason.as_deref(), source);
        self.timeline(
            "countenance",
            &countenance_timeline_text(&emoji, &mood, &reason),
        );
        self.append_observation(StreamObservation::CountenanceChanged {
            emoji: emoji.clone(),
            mood: mood.clone(),
            reason: reason.clone(),
            source: source.to_string(),
        })?;
        Ok(format!(
            "Countenance set: {}",
            CountenanceState {
                emoji,
                mood,
                reason,
            }
            .prompt_summary()
        ))
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

    fn memory_context_for_text(&mut self, label: &str, text: &str) -> Option<String> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }
        let snapshot = build_go_rag_memory_snapshot(&self.memory.context_provider, trimmed);
        self.timeline("memory", &snapshot.timeline_summary(label));
        is_useful_memory_summary(&snapshot.prompt_context).then_some(format!(
            "\n[Private memory context]\n{}\n[/Private memory context]",
            snapshot.prompt_context.trim()
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
        let hits =
            match self
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
        self.timeline(
            "memory",
            &format!(
                "queryMemories loaded {} retrieved memor{} for query '{}': {}",
                hits.len(),
                if hits.len() == 1 { "y" } else { "ies" },
                compact_line(text, 160),
                compact_line(&result_summary, 700)
            ),
        );
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
        let hits = self
            .memory
            .context_provider
            .search_graph_nodes(query.clone());
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
                        if let Some(memory_context) =
                            self.memory_context_for_text("user text", trimmed)
                        {
                            self.append_observation(StreamObservation::ActionResult(
                                memory_context,
                            ))?;
                        }
                        self.append_graph_memory_reminder(trimmed)?;
                    }
                }
                Err(message) => anyhow::bail!("failed to read stdin: {message}"),
            }
        }
        Ok(())
    }

    fn append_graph_memory_reminder(&mut self, text: &str) -> Result<()> {
        let Some(reason) = self.graph_memory_reminder_reason(text) else {
            return Ok(());
        };
        self.graph_memory_due = Some(reason.clone());
        self.graph_memory_reminders_since_update =
            self.graph_memory_reminders_since_update.saturating_add(1);

        let quoted = compact_line(text, 220);
        let interval_note =
            if self.graph_memory_reminders_since_update >= GRAPH_MEMORY_REMINDER_INTERVAL {
                format!(
                    " This is reminder {} since the last graph memory action.",
                    self.graph_memory_reminders_since_update
                )
            } else {
                String::new()
            };
        let message = format!(
            "Graph memory reminder: {reason}. When practical, use extractEntities(\"{quoted}\"), searchGraphNodes({{ text: \"...\", limit: 5 }}), then mergeGraphNode(\"kind:stable_slug\", {{ description: \"...\" }}, {{ label: \"Readable label\" }}) or updateGraphNodeFields(...) to add or update durable nodes. Use stable IDs such as person:travis_reed, place:seattle, topic:listenbury_memory, or project:listenbury; do not emit Cypher.{interval_note}"
        );
        self.timeline_colored("memory_reminder", &message, ANSI_DIM);
        self.append_observation(StreamObservation::ActionResult(message))
    }

    fn graph_memory_reminder_reason(&self, text: &str) -> Option<String> {
        let trimmed = non_empty_text(text)?;
        let extracted = self.memory.entity_extractor.extract(trimmed);
        let entity_summary = extracted
            .iter()
            .take(4)
            .map(|entity| format!("{} ({})", entity.text, entity.provisional_node_id()))
            .collect::<Vec<_>>();
        if !entity_summary.is_empty() {
            return Some(format!(
                "current user text contains extractable graph entities: {}",
                entity_summary.join(", ")
            ));
        }

        if graph_memory_text_has_cue(trimmed) {
            return Some(
                "current user text looks like durable identity, preference, relationship, plan, correction, or recurring context".to_string(),
            );
        }

        None
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
        let append_text = format_go_prompt_append(self.config.prompt_format, &prompt_text);
        if self.should_restart_before_append(&append_text) {
            self.restart_generation()?;
        }
        match self.llm.append_prompt(self.generation, append_text.clone()) {
            Ok(()) => {
                self.loaded_estimated_tokens = self
                    .loaded_estimated_tokens
                    .saturating_add(estimate_tokens(&append_text));
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
        self.flush_harmony_analysis_memory();
        let work_summary = self.work_board.prompt_summary();
        let recent_events = recent_events_with_current_countenance(
            &self.recent_events,
            self.current_countenance.as_ref(),
        );
        let rag_query = stream_rag_query(
            &self.config.prompt,
            &self.startup_context,
            work_summary.as_deref(),
            &recent_events,
        );
        let rag_snapshot = build_go_rag_memory_snapshot(&self.memory.context_provider, &rag_query);
        let (prompt_body, retained_event_count) = compact_stream_prompt_for_budget(
            &self.config.prompt,
            &self.startup_context,
            &recent_events,
            work_summary.as_deref(),
            Some(rag_snapshot.prompt_context.as_str()),
            self.context_budget_tokens(),
        );
        let prompt = render_go_prompt(self.config.prompt_format, &prompt_body);
        while self.recent_events.len() > retained_event_count {
            self.recent_events.pop_front();
        }
        print_debug_block("compacted prompt", ANSI_PROMPT, &prompt);
        self.generation = self
            .llm
            .start(GenerationRequest {
                prompt: prompt.clone(),
                max_tokens: self.config.max_tokens,
                stop: go_prompt_stops(self.config.prompt_format),
            })
            .context("failed to restart compacted stream")?;
        self.loaded_estimated_tokens = estimate_tokens(&prompt);
        self.generated_estimated_tokens = 0;
        self.generated_text_cleaner = GeneratedTextCleaner::new();
        self.harmony_analysis_memory = String::new();
        self.output_parser = StreamOutputParser::new(self.config.lookahead_chars);
        self.harmony_filter = harmony_filter_for_format(self.config.prompt_format);
        self.next_orientation_at = Instant::now() + ORIENTATION_PROMPT_INTERVAL;
        self.next_orientation_generated_tokens = ORIENTATION_GENERATED_TOKEN_INTERVAL;
        self.generation_paused = false;
        self.timeline(
            "memory",
            &rag_snapshot.timeline_summary("compacted stream prompt"),
        );
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
    needs_separator: bool,
    needs_source_navigation_separator: bool,
    last_output_ended_non_whitespace: bool,
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
                push_generated_text_preserving_separator(
                    &mut output,
                    &mut self.needs_separator,
                    &mut self.needs_source_navigation_separator,
                    &mut self.last_output_ended_non_whitespace,
                    &self.pending[..keep_from],
                );
                self.pending = self.pending[keep_from..].to_string();
                break;
            };

            push_generated_text_preserving_separator(
                &mut output,
                &mut self.needs_separator,
                &mut self.needs_source_navigation_separator,
                &mut self.last_output_ended_non_whitespace,
                &self.pending[..control_start],
            );
            let control = &self.pending[control_start..];
            if control.starts_with("<|start|>ts>") {
                let drain_to = control_start + "<|start|>ts>".len();
                self.pending.drain(..drain_to);
                push_generated_text_preserving_separator(
                    &mut output,
                    &mut self.needs_separator,
                    &mut self.needs_source_navigation_separator,
                    &mut self.last_output_ended_non_whitespace,
                    TYPESCRIPT_START,
                );
                continue;
            }
            if control.starts_with("<|start|>ts") && control.len() < "<|start|>ts>".len() {
                self.pending = self.pending[control_start..].to_string();
                break;
            }
            if let Some(drain_len) = generated_control_len(control) {
                let drain_to = control_start + drain_len;
                self.pending.drain(..drain_to);
                self.needs_separator = self.last_output_ended_non_whitespace;
                continue;
            }

            self.pending = self.pending[control_start..].to_string();
            break;
        }
        output
    }
}

fn push_generated_text_preserving_separator(
    output: &mut String,
    needs_separator: &mut bool,
    source_navigation_separator_pending: &mut bool,
    last_output_ended_non_whitespace: &mut bool,
    text: &str,
) {
    if text.is_empty() {
        return;
    }
    if *needs_separator
        && !text.starts_with(TYPESCRIPT_START)
        && text.chars().next().is_some_and(|ch| !ch.is_whitespace())
    {
        output.push(' ');
    } else if *source_navigation_separator_pending
        && !text.starts_with(TYPESCRIPT_START)
        && text.chars().next().is_some_and(|ch| ch.is_ascii_digit())
    {
        output.push(' ');
    }
    let normalized = normalize_source_navigation_spacing(text);
    output.push_str(&normalized);
    *needs_separator = false;
    *source_navigation_separator_pending = needs_source_navigation_separator(&normalized, "0");
    *last_output_ended_non_whitespace = normalized
        .chars()
        .next_back()
        .is_some_and(|ch| !ch.is_whitespace());
}

fn normalize_source_navigation_spacing(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some((start, word)) = first_source_navigation_word_before_digit(remaining) {
        output.push_str(&remaining[..start]);
        output.push_str(word);
        output.push(' ');
        remaining = &remaining[start + word.len()..];
    }
    output.push_str(remaining);
    output
}

fn first_source_navigation_word_before_digit(text: &str) -> Option<(usize, &'static str)> {
    const WORDS: &[&str] = &["page", "line"];
    WORDS
        .iter()
        .flat_map(|word| {
            text.match_indices(word).filter_map(move |(start, _)| {
                let end = start + word.len();
                let next = text[end..].chars().next()?;
                let before = text[..start].chars().next_back();
                (next.is_ascii_digit()
                    && !before.is_some_and(|ch| ch.is_ascii_alphanumeric() || ch == '_'))
                .then_some((start, *word))
            })
        })
        .min_by_key(|(start, _)| *start)
}

fn needs_source_navigation_separator(previous: &str, next: &str) -> bool {
    let previous = previous.trim_end();
    let next = next.trim_start();
    if !next.chars().next().is_some_and(|ch| ch.is_ascii_digit()) {
        return false;
    }
    ["page", "line"].iter().any(|word| previous.ends_with(word))
}

#[derive(Debug, Default)]
struct HarmonyFinalFilter {
    pending: String,
    in_final: bool,
    in_analysis: bool,
    in_tool_call: Option<String>,
    analysis_needs_separator: bool,
    pending_analysis_separator: bool,
    analysis_needs_source_navigation_separator: bool,
}

#[derive(Debug, Default)]
struct HarmonyFilterOutput {
    events: Vec<LlmEvent>,
    analysis: Vec<String>,
    tool_calls: Vec<HarmonyToolCall>,
}

#[derive(Debug, Default)]
struct HarmonyFilterChunk {
    visible: String,
    analysis: Vec<String>,
    tool_calls: Vec<HarmonyToolCall>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HarmonyToolCall {
    recipient: String,
    arguments: String,
}

impl HarmonyFinalFilter {
    fn for_analysis_prefill() -> Self {
        Self {
            in_analysis: true,
            ..Self::default()
        }
    }

    fn filter_events(&mut self, events: &[LlmEvent]) -> HarmonyFilterOutput {
        let mut output = HarmonyFilterOutput::default();
        for event in events {
            match event {
                LlmEvent::Token { text } => {
                    let chunk = self.push(text);
                    output.analysis.extend(chunk.analysis);
                    output.tool_calls.extend(chunk.tool_calls);
                    if !chunk.visible.is_empty() {
                        output.events.push(LlmEvent::Token {
                            text: chunk.visible,
                        });
                    }
                }
                LlmEvent::Completed | LlmEvent::Cancelled | LlmEvent::Error { .. } => {
                    let chunk = self.finish();
                    output.analysis.extend(chunk.analysis);
                    output.tool_calls.extend(chunk.tool_calls);
                    if !chunk.visible.is_empty() {
                        output.events.push(LlmEvent::Token {
                            text: chunk.visible,
                        });
                    }
                    output.events.push(event.clone());
                }
            }
        }
        output
    }

    fn push(&mut self, text: &str) -> HarmonyFilterChunk {
        self.pending.push_str(text);
        self.drain(false)
    }

    fn finish(&mut self) -> HarmonyFilterChunk {
        self.drain(true)
    }

    fn drain(&mut self, completed: bool) -> HarmonyFilterChunk {
        let mut visible = String::new();
        let mut analysis = Vec::new();
        let mut tool_calls = Vec::new();
        loop {
            if let Some(recipient) = self.in_tool_call.clone() {
                if let Some((start, marker)) = first_marker(&self.pending, HARMONY_TOOL_CALL_ENDS) {
                    let arguments = self.pending[..start].trim().to_string();
                    self.pending.drain(..start + marker.len());
                    self.in_tool_call = None;
                    tool_calls.push(HarmonyToolCall {
                        recipient,
                        arguments,
                    });
                    continue;
                }
                if completed {
                    let arguments = self.pending.trim().to_string();
                    self.pending.clear();
                    self.in_tool_call = None;
                    tool_calls.push(HarmonyToolCall {
                        recipient,
                        arguments,
                    });
                }
                break;
            }

            if self.in_final {
                if let Some((start, marker)) = first_marker(&self.pending, HARMONY_FINAL_BOUNDARIES)
                {
                    push_harmony_final_visible(&mut visible, &self.pending[..start]);
                    self.pending.drain(..start + marker.len());
                    if HARMONY_FINAL_ENDS.contains(&marker) {
                        self.in_final = false;
                    } else if HARMONY_FINAL_STARTS.contains(&marker) {
                        self.in_final = true;
                        self.in_analysis = false;
                    } else {
                        self.in_final = false;
                        self.in_analysis = true;
                    }
                    continue;
                }
                let keep_from = if completed {
                    self.pending.len()
                } else {
                    possible_marker_prefix_start(&self.pending, HARMONY_FINAL_BOUNDARIES)
                };
                push_harmony_final_visible(&mut visible, &self.pending[..keep_from]);
                self.pending.drain(..keep_from);
                break;
            }

            if self.in_analysis {
                if let Some((start, marker)) = first_marker(&self.pending, HARMONY_FINAL_BOUNDARIES)
                {
                    let text = self.pending[..start].to_string();
                    self.push_analysis(&mut analysis, &text);
                    self.pending.drain(..start + marker.len());
                    if HARMONY_FINAL_ENDS.contains(&marker) {
                        self.pending_analysis_separator = self.analysis_needs_separator;
                        self.analysis_needs_separator = false;
                        self.in_analysis = false;
                    } else if HARMONY_FINAL_STARTS.contains(&marker) {
                        self.pending_analysis_separator = false;
                        self.analysis_needs_separator = false;
                        self.in_analysis = false;
                        self.in_final = true;
                    } else {
                        self.in_analysis = true;
                    }
                    continue;
                }
                if completed {
                    let text = self.pending.clone();
                    self.push_analysis(&mut analysis, &text);
                    self.pending.clear();
                } else {
                    let keep_from =
                        possible_marker_prefix_start(&self.pending, HARMONY_FINAL_BOUNDARIES);
                    let text = self.pending[..keep_from].to_string();
                    self.push_analysis(&mut analysis, &text);
                    self.pending.drain(..keep_from);
                }
                break;
            }

            let envelope_start = first_harmony_tool_envelope_start(&self.pending);
            let tool_start = first_harmony_tool_call_start(&self.pending);
            let channel_start = first_marker(&self.pending, HARMONY_CHANNEL_STARTS);
            if let Some((start, envelope_len, tool_call)) = envelope_start {
                if channel_start.map_or(true, |(channel_index, _)| start <= channel_index) {
                    self.pending.drain(..start + envelope_len);
                    self.in_analysis = false;
                    self.in_final = false;
                    tool_calls.push(tool_call);
                    continue;
                }
            }
            if let Some((start, header_len, recipient)) = tool_start {
                if channel_start.map_or(true, |(channel_index, _)| start <= channel_index) {
                    self.pending.drain(..start + header_len);
                    self.in_analysis = false;
                    self.in_final = false;
                    self.in_tool_call = Some(recipient);
                    continue;
                }
            }
            if possible_harmony_tool_call_header(&self.pending)
                && channel_start.map_or(true, |(channel_index, _)| {
                    first_possible_harmony_tool_call_header(&self.pending)
                        .map_or(false, |tool_index| tool_index <= channel_index)
                })
            {
                break;
            }

            if let Some((start, marker)) = channel_start {
                self.pending.drain(..start + marker.len());
                if HARMONY_FINAL_STARTS.contains(&marker) {
                    self.in_final = true;
                } else {
                    self.in_analysis = true;
                }
                continue;
            }
            if completed {
                self.pending.clear();
            } else {
                let keep_from = possible_harmony_prefix_start(&self.pending);
                self.pending.drain(..keep_from);
            }
            break;
        }
        HarmonyFilterChunk {
            visible,
            analysis,
            tool_calls,
        }
    }

    fn push_analysis(&mut self, analysis: &mut Vec<String>, text: &str) {
        if is_harmony_abort_spiral(text) {
            let mut buffer = String::new();
            for chunk in drain_harmony_analysis_memory(&mut buffer, text, 80, true) {
                self.push_analysis_chunk(analysis, &chunk);
            }
            return;
        }
        self.push_analysis_chunk(analysis, text);
    }

    fn push_analysis_chunk(&mut self, analysis: &mut Vec<String>, text: &str) {
        if text.trim().is_empty() || is_harmony_terminal_filler(text) {
            return;
        }
        let mut chunk = String::new();
        if self.pending_analysis_separator
            && text.chars().next().is_some_and(|ch| !ch.is_whitespace())
        {
            chunk.push(' ');
        } else if self.analysis_needs_source_navigation_separator
            && text.chars().next().is_some_and(|ch| ch.is_ascii_digit())
        {
            chunk.push(' ');
        }
        let normalized = normalize_source_navigation_spacing(text);
        chunk.push_str(&normalized);
        self.pending_analysis_separator = false;
        self.analysis_needs_separator = normalized
            .chars()
            .next_back()
            .is_some_and(|ch| !ch.is_whitespace());
        self.analysis_needs_source_navigation_separator =
            needs_source_navigation_separator(&normalized, "0");
        analysis.push(chunk);
    }
}

const HARMONY_FINAL_STARTS: &[&str] = &[
    "final<|message|>",
    "<|channel|>final<|message|>",
    "<|start|>assistant<|channel|>final<|message|>",
];

const HARMONY_CHANNEL_STARTS: &[&str] = &[
    "analysis<|message|>",
    "final<|message|>",
    "commentary<|message|>",
    "<|channel|>final<|message|>",
    "<|start|>assistant<|channel|>final<|message|>",
    "<|channel|>analysis<|message|>",
    "<|start|>assistant<|channel|>analysis<|message|>",
];

const HARMONY_FINAL_BOUNDARIES: &[&str] = &[
    "<|end|>",
    "<|return|>",
    "<|constrain|>",
    "<|call|>",
    "<|start|>",
    "analysis<|message|>",
    "final<|message|>",
    "commentary<|message|>",
    "<|channel|>final<|message|>",
    "<|start|>assistant<|channel|>final<|message|>",
    "<|channel|>analysis<|message|>",
    "<|start|>assistant<|channel|>analysis<|message|>",
];

const HARMONY_FINAL_ENDS: &[&str] = &["<|end|>", "<|return|>", "<|constrain|>", "<|start|>"];
const HARMONY_TOOL_CALL_ENDS: &[&str] = &[
    "commentaryanalysis<|message|>",
    "commentaryfinal<|message|>",
    "commentary<|message|>",
    "commentary<|channel|>",
    "commentary<|start|>",
    "<|call|>",
    "<|end|>",
    "<|return|>",
    "<|start|>",
];
const HARMONY_TOOL_CALL_HEADER_STARTS: &[&str] =
    &["<|channel|>commentary", "<|start|>assistant", "commentary"];
const HARMONY_MESSAGE_MARKER: &str = "<|message|>";
const HARMONY_TOOL_ENVELOPE_MARKERS: &[&str] = &[".json<|message|>", "json<|message|>"];

fn first_harmony_tool_envelope_start(text: &str) -> Option<(usize, usize, HarmonyToolCall)> {
    HARMONY_TOOL_ENVELOPE_MARKERS
        .iter()
        .flat_map(|marker| text.match_indices(marker))
        .filter_map(|(start, marker)| harmony_tool_envelope_at(text, start, marker))
        .min_by_key(|(start, _, _)| *start)
}

fn harmony_tool_envelope_at(
    text: &str,
    start: usize,
    marker: &str,
) -> Option<(usize, usize, HarmonyToolCall)> {
    let rest = &text[start + marker.len()..];
    let mut values = serde_json::Deserializer::from_str(rest).into_iter::<Value>();
    let value = match values.next()? {
        Ok(value) => value,
        Err(error) if error.is_eof() => return None,
        Err(_) => return None,
    };
    let tool_call = harmony_tool_call_from_envelope(value)?;
    Some((start, marker.len() + values.byte_offset(), tool_call))
}

fn harmony_tool_call_from_envelope(value: Value) -> Option<HarmonyToolCall> {
    let object = value.as_object()?;
    let recipient = object.get("name")?.as_str()?.trim();
    if !recipient.starts_with("functions.") {
        return None;
    }
    let arguments = match object.get("arguments") {
        Some(Value::String(arguments)) => arguments.clone(),
        Some(Value::Object(arguments)) => serde_json::to_string(arguments).ok()?,
        Some(Value::Null) | None => String::new(),
        Some(arguments) => serde_json::to_string(arguments).ok()?,
    };
    Some(HarmonyToolCall {
        recipient: recipient.to_string(),
        arguments,
    })
}

fn first_harmony_tool_call_start(text: &str) -> Option<(usize, usize, String)> {
    HARMONY_TOOL_CALL_HEADER_STARTS
        .iter()
        .flat_map(|marker| text.match_indices(marker))
        .filter_map(|(start, _)| harmony_tool_call_header_at(text, start))
        .min_by_key(|(start, _, _)| *start)
}

fn harmony_tool_call_header_at(text: &str, start: usize) -> Option<(usize, usize, String)> {
    let rest = &text[start..];
    let message_start = rest.find(HARMONY_MESSAGE_MARKER)?;
    let header = &rest[..message_start];
    if !header.contains("commentary") {
        return None;
    }
    let recipient = harmony_tool_recipient(header)?;
    Some((
        start,
        message_start + HARMONY_MESSAGE_MARKER.len(),
        recipient,
    ))
}

fn harmony_tool_recipient(header: &str) -> Option<String> {
    let start = header.find("to=functions.")? + "to=".len();
    let recipient = header[start..]
        .split(|ch: char| ch.is_whitespace() || ch == '<')
        .next()
        .unwrap_or_default()
        .trim();
    (!recipient.is_empty()).then(|| recipient.to_string())
}

fn possible_harmony_tool_call_header(text: &str) -> bool {
    let Some(start) = first_possible_harmony_tool_call_header(text) else {
        return false;
    };
    let header = &text[start..];
    header.contains("commentary") && !header.contains(HARMONY_MESSAGE_MARKER)
}

fn first_possible_harmony_tool_call_header(text: &str) -> Option<usize> {
    HARMONY_TOOL_CALL_HEADER_STARTS
        .iter()
        .filter_map(|marker| text.find(marker))
        .min()
}

fn possible_harmony_prefix_start(text: &str) -> usize {
    HARMONY_CHANNEL_STARTS
        .iter()
        .chain(HARMONY_TOOL_CALL_HEADER_STARTS.iter())
        .chain(HARMONY_TOOL_ENVELOPE_MARKERS.iter())
        .filter_map(|marker| possible_marker_prefix_start_for(text, marker))
        .min()
        .unwrap_or(text.len())
}

fn first_marker<'a>(text: &str, markers: &'a [&str]) -> Option<(usize, &'a str)> {
    markers
        .iter()
        .filter_map(|marker| text.find(marker).map(|index| (index, *marker)))
        .min_by(|(left_index, left_marker), (right_index, right_marker)| {
            left_index
                .cmp(right_index)
                .then_with(|| right_marker.len().cmp(&left_marker.len()))
        })
}

fn push_harmony_final_visible(visible: &mut String, text: &str) {
    if text.trim().is_empty() || is_harmony_terminal_filler(text) {
        return;
    }
    visible.push_str(text);
}

fn possible_marker_prefix_start(text: &str, markers: &[&str]) -> usize {
    (0..text.len())
        .find(|&index| {
            text.is_char_boundary(index)
                && markers
                    .iter()
                    .any(|marker| possible_marker_prefix_start_for(text, marker) == Some(index))
        })
        .unwrap_or(text.len())
}

fn possible_marker_prefix_start_for(text: &str, marker: &str) -> Option<usize> {
    (0..text.len()).find(|&index| {
        text.is_char_boundary(index) && {
            let suffix = &text[index..];
            !suffix.is_empty() && suffix.len() < marker.len() && marker.starts_with(suffix)
        }
    })
}

fn is_harmony_terminal_filler(text: &str) -> bool {
    let normalized = text
        .trim()
        .trim_matches(|ch: char| ch == '.' || ch == '!' || ch == '?' || ch == ':' || ch == ';')
        .to_ascii_lowercase();
    if matches!(
        normalized.as_str(),
        "end"
            | "done"
            | "finished"
            | "no output"
            | "no action"
            | "no action required"
            | "no action needed"
            | "no action is needed"
            | "no further action"
            | "end of chain"
            | "end of session"
            | "conversation over"
            | "the conversation is over"
            | "all done"
            | "that's it"
            | "that is it"
            | "no open goals"
            | "no open goals remaining"
            | "no pending tasks"
            | "nothing to do"
            | "nothing needs doing"
            | "nothing useful to do"
            | "say nothing"
            | "maybe say nothing"
            | "session complete"
            | "all goals complete"
            | "all goals finished"
            | "added note for future reference"
    ) {
        return true;
    }

    if (normalized.contains("no open goals") || normalized.contains("no pending tasks"))
        && (normalized.contains("session complete")
            || normalized.contains("session completed")
            || normalized.contains("say nothing")
            || normalized.contains("no further action")
            || normalized.contains("no further actions")
            || normalized.contains("nothing to do"))
    {
        return true;
    }

    if normalized.contains("all goals")
        && (normalized.contains("complete")
            || normalized.contains("finished")
            || normalized.contains("say nothing")
            || normalized.contains("no further action"))
    {
        return true;
    }

    is_harmony_abort_spiral(&normalized)
}

fn is_harmony_abort_spiral(normalized: &str) -> bool {
    let lower = normalized.to_ascii_lowercase();
    let compact = lower.split_whitespace().collect::<Vec<_>>().join(" ");
    let abort_phrases = [
        "i will stop",
        "i'll stop",
        "i stop",
        "i cannot continue",
        "i can't continue",
        "cannot continue",
        "cannot comply",
        "can't comply",
        "conversation cannot",
        "this is impossible",
        "this is futile",
        "not possible",
        "we need to stop",
        "time to stop",
    ];
    if abort_phrases.iter().any(|phrase| compact.contains(phrase)) {
        return true;
    }

    let terminal_markers = [
        " stop",
        " final",
        " no output",
        " no more",
        " conversation over",
        " cannot",
        " impossible",
        " futile",
    ];
    let marker_count = terminal_markers
        .iter()
        .map(|marker| compact.matches(marker).count())
        .sum::<usize>();
    marker_count >= 3
}

fn drain_harmony_analysis_memory(
    buffer: &mut String,
    text: &str,
    flush_chars: usize,
    force: bool,
) -> Vec<String> {
    buffer.push_str(text);
    let mut thoughts = Vec::new();
    let flush_chars = flush_chars.max(80);

    while let Some(boundary) = next_thought_boundary(buffer, flush_chars) {
        let thought = buffer[..boundary].trim().to_string();
        buffer.drain(..boundary);
        if is_meaningful_harmony_thought(&thought) {
            thoughts.push(thought);
        }
    }

    if force {
        let thought = std::mem::take(buffer).trim().to_string();
        if is_meaningful_harmony_thought(&thought) {
            thoughts.push(thought);
        }
    }

    thoughts
}

fn is_meaningful_harmony_thought(text: &str) -> bool {
    is_meaningful_thought(text) && !is_harmony_terminal_filler(text)
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
    ["<|end|>", "<|return|>", "<|constrain|>", "<|message|>"]
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
        "reportBug",
        "reportFeatureRequest",
        "reportIssue",
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
        "recallMemories",
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
        self.ensure_open_selection();
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
        if !select {
            self.ensure_open_selection();
        }
        format!(
            "Created goal {id}: {title}{}",
            if select { " (selected)" } else { "" }
        )
    }

    fn complete(&mut self, target: &str, note: Option<&str>) -> String {
        let message = {
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
        };
        self.ensure_open_selection();
        message
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
        let message = format!(
            "Checked goal step in {}: {}{}",
            goal.id,
            checked_text,
            note.map(|note| format!(" note={note}")).unwrap_or_default()
        );
        self.ensure_open_selection();
        message
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
        let message = {
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
        };
        self.ensure_open_selection();
        message
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
        let message = {
            let Some(goal) = self.find_mut(target) else {
                return format!("No goal matched {target}.");
            };
            goal.add_log(text);
            format!(
                "Added goal note to {}: {}",
                goal.id,
                compact_line(text, 500)
            )
        };
        self.ensure_open_selection();
        message
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
            if !matches!(selected.status, WorkItemStatus::Open)
                && self
                    .items
                    .iter()
                    .any(|item| matches!(item.status, WorkItemStatus::Open))
            {
                lines.push(format!(
                    "Selected goal {} is {}; select an open goal before doing more work.",
                    selected.id,
                    selected.status.as_str()
                ));
            }
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
        if !self
            .items
            .iter()
            .any(|item| matches!(item.status, WorkItemStatus::Open))
        {
            lines.push(
                "No open goals. Completion is a transition; create or select a useful curiosity, learning, maintenance, or observation goal instead of treating the session as complete."
                    .to_string(),
            );
        }
        Some(lines.join("\n"))
    }

    fn suggested_progress_target(&self) -> String {
        if let Some(selected) = self.selected_item()
            && matches!(selected.status, WorkItemStatus::Open)
        {
            return selected.id.clone();
        }
        if let Some(open) = self
            .items
            .iter()
            .find(|item| matches!(item.status, WorkItemStatus::Open))
        {
            return open.id.clone();
        }
        self.selected_id
            .clone()
            .unwrap_or_else(|| "current-goal-id".to_string())
    }

    fn ensure_open_selection(&mut self) {
        if let Some(selected) = self.selected_item()
            && matches!(selected.status, WorkItemStatus::Open)
        {
            return;
        }
        self.selected_id = self
            .items
            .iter()
            .find(|item| matches!(item.status, WorkItemStatus::Open))
            .map(|item| item.id.clone());
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
        workflow: SourceWorkflowUpdate,
    },
    ReadSourceFile {
        file: String,
        page: usize,
        line: Option<usize>,
        page_size: Option<usize>,
        workflow: SourceWorkflowUpdate,
    },
    SetSourcePageSize {
        lines: usize,
    },
    SearchSource {
        query: String,
        limit: usize,
        workflow: SourceWorkflowUpdate,
    },
    GrepSource {
        pattern: String,
        limit: usize,
        workflow: SourceWorkflowUpdate,
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
    SetCountenance {
        emoji: String,
        mood: Option<String>,
        reason: Option<String>,
    },
    ReportIssue {
        issue_type: String,
        title: String,
        details: Option<String>,
        context: Option<String>,
        severity: Option<String>,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct SourceWorkflowUpdate {
    target: Option<String>,
    note: Option<String>,
    summary: Option<String>,
}

impl SourceWorkflowUpdate {
    fn new(target: Option<String>, note: Option<String>, summary: Option<String>) -> Self {
        Self {
            target: clean_optional_text(target),
            note: clean_optional_text(note),
            summary: clean_optional_text(summary),
        }
    }

    fn is_empty(&self) -> bool {
        self.target.is_none() && self.note.is_none() && self.summary.is_none()
    }
}

impl TypeScriptAction {
    fn source_progress_label(&self) -> Option<String> {
        match self {
            Self::ListFiles { page, .. } => Some(format!("listFiles page {page}")),
            Self::ReadSourceFile {
                file, line, page, ..
            } => {
                let location = line
                    .map(|line| format!("line {line}"))
                    .unwrap_or_else(|| format!("page {page}"));
                Some(format!("readSourceFile {file} {location}"))
            }
            Self::SearchSource { query, .. } => Some(format!("searchSource {query}")),
            Self::GrepSource { pattern, .. } => Some(format!("grepSource {pattern}")),
            _ => None,
        }
    }

    fn records_progress_note(&self) -> bool {
        match self {
            Self::Note { .. } | Self::AddGoalNote { .. } | Self::CheckChecklistItem { .. } => true,
            Self::CompleteWorkItem { note, .. } => note.is_some(),
            Self::CreateWorkItem { note, .. } => note.is_some(),
            Self::UpdateWorkItem { fields, .. } => {
                fields.contains_key("note")
                    || fields.contains_key("log")
                    || fields.contains_key("comment")
            }
            _ => false,
        }
    }

    fn records_source_knowledge_capture(&self) -> bool {
        match self {
            Self::Note { text } | Self::AddGoalNote { text, .. } => {
                meaningful_knowledge_capture_text(text)
            }
            Self::CheckChecklistItem { note, .. }
            | Self::CompleteWorkItem { note, .. }
            | Self::CreateWorkItem { note, .. } => note
                .as_deref()
                .is_some_and(meaningful_knowledge_capture_text),
            Self::UpdateWorkItem { fields, .. } => ["note", "log", "comment", "summary"]
                .iter()
                .filter_map(|key| fields.get(*key).and_then(Value::as_str))
                .any(meaningful_knowledge_capture_text),
            _ => false,
        }
    }

    fn records_synthesis_update(&self) -> bool {
        match self {
            Self::CompleteWorkItem { note, .. } => note
                .as_deref()
                .is_some_and(|note| meaningful_synthesis_text(note)),
            Self::UpdateWorkItem { fields, .. } => fields
                .get("summary")
                .and_then(Value::as_str)
                .is_some_and(meaningful_synthesis_text),
            _ => false,
        }
    }

    fn graph_memory_update_label(&self) -> Option<String> {
        match self {
            Self::ExtractEntities { text } => Some(format!(
                "extractEntities({})",
                text.as_deref()
                    .map(|text| compact_line(text, 80))
                    .unwrap_or_else(|| "current text".to_string())
            )),
            Self::UpdateGraphNodeFields {
                node_id, fields, ..
            } => Some(format!(
                "merge/update {node_id}: {}",
                summarize_command_fields(fields)
            )),
            _ => None,
        }
    }

    fn source_workflow(&self) -> Option<&SourceWorkflowUpdate> {
        let workflow = match self {
            Self::ListFiles { workflow, .. }
            | Self::ReadSourceFile { workflow, .. }
            | Self::SearchSource { workflow, .. }
            | Self::GrepSource { workflow, .. } => workflow,
            _ => return None,
        };
        (!workflow.is_empty()).then_some(workflow)
    }
}

fn source_workflow_records_knowledge_capture(workflow: Option<&SourceWorkflowUpdate>) -> bool {
    let Some(workflow) = workflow else {
        return false;
    };
    workflow
        .summary
        .as_deref()
        .is_some_and(meaningful_knowledge_capture_text)
        || workflow
            .note
            .as_deref()
            .is_some_and(meaningful_knowledge_capture_text)
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
    SetTopic {
        topic: String,
        #[serde(default)]
        instruction: Option<String>,
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
    TopicChangedWhen {
        trigger: String,
        #[serde(default, alias = "fromTopic")]
        from_topic: Option<String>,
        #[serde(default, alias = "toTopic")]
        to_topic: Option<String>,
        #[serde(default)]
        topic: Option<String>,
        #[serde(default)]
        instruction: Option<String>,
        #[serde(default)]
        summary: Option<String>,
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
        #[serde(default)]
        target: Option<String>,
        #[serde(default)]
        note: Option<String>,
        #[serde(default)]
        summary: Option<String>,
    },
    ReadSourceFile {
        file: String,
        #[serde(default)]
        page: Option<usize>,
        #[serde(default)]
        line: Option<usize>,
        #[serde(default)]
        page_size: Option<usize>,
        #[serde(default)]
        target: Option<String>,
        #[serde(default)]
        note: Option<String>,
        #[serde(default)]
        summary: Option<String>,
    },
    SetSourcePageSize {
        lines: usize,
    },
    SearchSource {
        query: String,
        #[serde(default)]
        limit: Option<usize>,
        #[serde(default)]
        target: Option<String>,
        #[serde(default)]
        note: Option<String>,
        #[serde(default)]
        summary: Option<String>,
    },
    GrepSource {
        pattern: String,
        #[serde(default)]
        limit: Option<usize>,
        #[serde(default)]
        target: Option<String>,
        #[serde(default)]
        note: Option<String>,
        #[serde(default)]
        summary: Option<String>,
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
    #[serde(
        alias = "set_mood",
        alias = "setMood",
        alias = "set_emotion",
        alias = "setEmotion",
        alias = "setCountenance",
        alias = "emote"
    )]
    SetCountenance {
        emoji: String,
        #[serde(default)]
        mood: Option<String>,
        #[serde(default)]
        reason: Option<String>,
    },
    ReportIssue {
        title: String,
        #[serde(default, alias = "type", alias = "issueType")]
        issue_type: Option<String>,
        #[serde(default)]
        details: Option<String>,
        #[serde(default)]
        context: Option<String>,
        #[serde(default)]
        severity: Option<String>,
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

fn actions_from_harmony_tool_call(call: &HarmonyToolCall) -> Result<Vec<TypeScriptAction>> {
    let name = call
        .recipient
        .strip_prefix("functions.")
        .ok_or_else(|| anyhow::anyhow!("unsupported Harmony tool recipient {}", call.recipient))?;
    let kind = harmony_tool_kind(name)
        .ok_or_else(|| anyhow::anyhow!("unsupported Harmony tool function {name}"))?;
    let mut arguments = parse_harmony_tool_arguments(&call.arguments)?;
    arguments.insert("kind".to_string(), Value::String(kind.to_string()));
    if matches!(name, "report_feature_request" | "reportFeatureRequest") {
        arguments.insert(
            "issue_type".to_string(),
            Value::String("feature_request".to_string()),
        );
    }
    parse_typescript_actions(Value::Object(arguments))
}

fn parse_harmony_tool_arguments(arguments: &str) -> Result<Map<String, Value>> {
    let arguments = arguments.trim();
    if arguments.is_empty() {
        return Ok(Map::new());
    }

    let mut values = serde_json::Deserializer::from_str(arguments).into_iter::<Value>();
    let value = values
        .next()
        .transpose()?
        .context("Harmony tool arguments must include a JSON object")?;
    let trailing = arguments[values.byte_offset()..].trim();
    if !trailing.is_empty() && !is_harmony_tool_argument_trailing_debris(trailing) {
        anyhow::bail!("trailing characters after Harmony tool JSON: {trailing}");
    }

    match value {
        Value::Object(object) => Ok(object),
        other => anyhow::bail!("Harmony tool arguments must be a JSON object, got {other}"),
    }
}

fn is_harmony_tool_argument_trailing_debris(text: &str) -> bool {
    let text = text.trim_start();
    text == "commentary"
        || text.starts_with("commentaryanalysis<|message|>")
        || text.starts_with("commentaryfinal<|message|>")
        || text.starts_with("commentary<|")
        || text.starts_with("<|channel|>")
        || text.starts_with("<|start|>assistant")
        || text.starts_with("analysis<|message|>")
        || text.starts_with("final<|message|>")
        || is_harmony_terminal_filler(text)
}

fn is_ignorable_harmony_tool_call(call: &HarmonyToolCall) -> bool {
    let arguments = call.arguments.trim();
    arguments.is_empty() || is_harmony_terminal_filler(arguments)
}

fn harmony_tool_kind(name: &str) -> Option<&'static str> {
    Some(match name {
        "say" => "say",
        "shutup" => "shutup",
        "pause" => "pause",
        "resume" => "resume",
        "note" => "note",
        "set_countenance" | "setCountenance" | "set_mood" | "setMood" | "emote" | "set_emotion"
        | "setEmotion" => "set_countenance",
        "report_bug" | "reportBug" | "report_issue" | "reportIssue" => "report_issue",
        "report_feature_request" | "reportFeatureRequest" => "report_issue",
        "set_stage" | "setStage" => "set_stage",
        "set_topic" | "setTopic" => "set_topic",
        "start_new_topic" | "startNewTopic" => "start_new_topic",
        "topic_changed_when" | "topicChangedWhen" => "topic_changed_when",
        "start_new_episode" | "startNewEpisode" | "newEpisodeStarted" => "start_new_episode",
        "sleeping" | "going_to_sleep" | "goingToSleep" | "go_to_sleep" | "goToSleep" => "sleeping",
        "extract_entities" | "extractEntities" => "extract_entities",
        "merge_graph_node"
        | "mergeGraphNode"
        | "upsert_graph_node"
        | "upsertGraphNode"
        | "add_graph_node"
        | "addGraphNode"
        | "update_graph_node_fields"
        | "updateGraphNodeFields"
        | "updateEntityFields" => "update_graph_node_fields",
        "search_graph_nodes" | "searchGraphNodes" | "searchEntities" => "search_graph_nodes",
        "query_memories" | "queryMemories" | "recall_memories" | "recallMemories" => {
            "query_memories"
        }
        "list_files" | "listFiles" => "list_files",
        "read_source_file" | "readSourceFile" | "read_file" | "readFile" => "read_source_file",
        "search_source" | "searchSource" => "search_source",
        "grep_source" | "grepSource" => "grep_source",
        "set_source_page_size" | "setSourcePageSize" => "set_source_page_size",
        "create_goal" | "createGoal" => "create_goal",
        "create_task" | "createTask" => "create_task",
        "create_checklist" | "createChecklist" => "create_checklist",
        "add_goal_note" | "addGoalNote" | "log_progress" | "logProgress" | "comment_goal"
        | "commentGoal" => "add_goal_note",
        "check_off" | "checkOff" | "complete_item" | "completeItem" => "complete_work_item",
        "check_goal_step" | "checkGoalStep" | "check_checklist_item" | "checkChecklistItem" => {
            "check_checklist_item"
        }
        "update_item" | "updateItem" => "update_work_item",
        "cancel_item" | "cancelItem" => "cancel_work_item",
        "select_item" | "selectItem" => "select_work_item",
        _ => return None,
    })
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
        TypeScriptActionPayload::SetTopic {
            topic,
            instruction,
            summary,
        } => non_empty_text(&topic).map(|topic| TypeScriptAction::SetStage {
            topic: Some(topic.to_string()),
            instruction: instruction
                .and_then(|instruction| non_empty_text(&instruction).map(str::to_string))
                .unwrap_or_else(|| format!("The current topic is {topic}.")),
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
        TypeScriptActionPayload::TopicChangedWhen {
            trigger,
            from_topic,
            to_topic,
            topic,
            instruction,
            summary,
        } => non_empty_text(&trigger).map(|trigger_text| TypeScriptAction::StartNewTopic {
            last_topic: from_topic
                .and_then(|topic| non_empty_text(&topic).map(str::to_string))
                .unwrap_or_else(|| "previous topic".to_string()),
            topic: to_topic
                .or(topic)
                .and_then(|topic| non_empty_text(&topic).map(str::to_string)),
            instruction: instruction
                .and_then(|instruction| non_empty_text(&instruction).map(str::to_string))
                .or_else(|| {
                    Some(format!(
                        "The topic changed when the interlocutor said: {trigger_text}"
                    ))
                }),
            summary: summary.and_then(|summary| non_empty_text(&summary).map(str::to_string)),
            trigger: Some(trigger_text.to_string()),
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
        TypeScriptActionPayload::ListFiles {
            page,
            page_size,
            target,
            note,
            summary,
        } => Some(TypeScriptAction::ListFiles {
            page: page.unwrap_or(1).max(1),
            page_size,
            workflow: SourceWorkflowUpdate::new(target, note, summary),
        }),
        TypeScriptActionPayload::ReadSourceFile {
            file,
            page,
            line,
            page_size,
            target,
            note,
            summary,
        } => {
            let file = file.trim();
            (!file.is_empty()).then(|| TypeScriptAction::ReadSourceFile {
                file: file.to_string(),
                page: page.unwrap_or(1).max(1),
                line: line.map(|line| line.max(1)),
                page_size: page_size
                    .map(|lines| lines.clamp(MIN_SOURCE_PAGE_LINES, MAX_SOURCE_PAGE_LINES)),
                workflow: SourceWorkflowUpdate::new(target, note, summary),
            })
        }
        TypeScriptActionPayload::SetSourcePageSize { lines } => {
            Some(TypeScriptAction::SetSourcePageSize {
                lines: lines.clamp(MIN_SOURCE_PAGE_LINES, MAX_SOURCE_PAGE_LINES),
            })
        }
        TypeScriptActionPayload::SearchSource {
            query,
            limit,
            target,
            note,
            summary,
        } => non_empty_text(&query).map(|query| TypeScriptAction::SearchSource {
            query: query.to_string(),
            limit: limit.unwrap_or(12).max(1),
            workflow: SourceWorkflowUpdate::new(target, note, summary),
        }),
        TypeScriptActionPayload::GrepSource {
            pattern,
            limit,
            target,
            note,
            summary,
        } => non_empty_text(&pattern).map(|pattern| TypeScriptAction::GrepSource {
            pattern: pattern.to_string(),
            limit: limit.unwrap_or(12).max(1),
            workflow: SourceWorkflowUpdate::new(target, note, summary),
        }),
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
                    text: normalize_source_navigation_spacing(text),
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
                text: normalize_source_navigation_spacing(text),
            })
        }
        TypeScriptActionPayload::SetCountenance {
            emoji,
            mood,
            reason,
        } => normalize_countenance_emoji(&emoji).map(|emoji| TypeScriptAction::SetCountenance {
            emoji,
            mood: mood.and_then(|mood| non_empty_text(&mood).map(str::to_string)),
            reason: reason.and_then(|reason| non_empty_text(&reason).map(str::to_string)),
        }),
        TypeScriptActionPayload::ReportIssue {
            title,
            issue_type,
            details,
            context,
            severity,
        } => non_empty_text(&title).map(|title| TypeScriptAction::ReportIssue {
            issue_type: normalized_issue_type(issue_type.as_deref()).to_string(),
            title: title.to_string(),
            details: details.and_then(|details| non_empty_text(&details).map(str::to_string)),
            context: context.and_then(|context| non_empty_text(&context).map(str::to_string)),
            severity: severity.and_then(|severity| non_empty_text(&severity).map(str::to_string)),
        }),
    }
}

fn typescript_source_with_default_imports(script: &str) -> String {
    if script.contains("\"pete:will\"") || script.contains("'pete:will'") {
        return script.to_string();
    }
    format!(
        "import {{ say, shutup, pause, resume, note, setCountenance, setMood, emote, reportBug, reportFeatureRequest, reportIssue, setStage, setTopic, startNewTopic, topicChangedWhen, startNewEpisode, sleeping, goingToSleep, extractEntities, mergeGraphNode, upsertGraphNode, updateGraphNodeFields, searchGraphNodes, queryMemories, recallMemories, listFiles, readSourceFile, readFile, searchSource, grepSource, setSourcePageSize, createGoal, createTask, createChecklist, addGoalNote, logProgress, commentGoal, checkOff, completeItem, checkGoalStep, checkChecklistItem, updateItem, cancelItem, selectItem }} from \"pete:will\";\n{script}"
    )
}

fn go_typescript_module() -> InternalModule {
    InternalModule::native("pete:will")
        .with_function("say", ts_say, 2)
        .with_function("shutup", ts_shutup, 0)
        .with_function("pause", ts_pause, 0)
        .with_function("resume", ts_resume, 0)
        .with_function("note", ts_note, 1)
        .with_function("setCountenance", ts_set_countenance, 2)
        .with_function("set_countenance", ts_set_countenance, 2)
        .with_function("setMood", ts_set_countenance, 2)
        .with_function("set_mood", ts_set_countenance, 2)
        .with_function("setEmotion", ts_set_countenance, 2)
        .with_function("set_emotion", ts_set_countenance, 2)
        .with_function("emote", ts_set_countenance, 2)
        .with_function("reportBug", ts_report_bug, 2)
        .with_function("report_bug", ts_report_bug, 2)
        .with_function("reportFeatureRequest", ts_report_feature_request, 2)
        .with_function("report_feature_request", ts_report_feature_request, 2)
        .with_function("reportIssue", ts_report_issue, 2)
        .with_function("report_issue", ts_report_issue, 2)
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
        .with_function("mergeGraphNode", ts_update_graph_node_fields, 3)
        .with_function("merge_graph_node", ts_update_graph_node_fields, 3)
        .with_function("upsertGraphNode", ts_update_graph_node_fields, 3)
        .with_function("upsert_graph_node", ts_update_graph_node_fields, 3)
        .with_function("addGraphNode", ts_update_graph_node_fields, 3)
        .with_function("add_graph_node", ts_update_graph_node_fields, 3)
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

fn ts_set_countenance(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    let first = string_arg(args, 0);
    let option_emoji = optional_string_property_arg(args, 1, "emoji")
        .or_else(|| optional_string_property_arg(args, 1, "face"))
        .or_else(|| optional_string_property_arg(args, 1, "expression"));
    let emoji = match args.first() {
        Some(JsValue::Object(_)) => optional_string_property_arg(args, 0, "emoji")
            .or_else(|| optional_string_property_arg(args, 0, "face"))
            .or_else(|| optional_string_property_arg(args, 0, "expression"))
            .unwrap_or_default(),
        _ => option_emoji.clone().unwrap_or_else(|| first.clone()),
    };
    let first_is_mood = option_emoji.is_some() && normalize_countenance_emoji(&first).is_none();
    command_value(
        interp,
        json!({
            "kind": "set_countenance",
            "emoji": emoji,
            "mood": optional_string_property_arg(args, 1, "mood")
                .or_else(|| optional_string_property_arg(args, 0, "mood"))
                .or_else(|| first_is_mood.then_some(first)),
            "reason": optional_string_property_arg(args, 1, "reason")
                .or_else(|| optional_string_property_arg(args, 0, "reason")),
        }),
    )
}

fn ts_report_bug(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    report_issue_command(interp, args, "bug")
}

fn ts_report_feature_request(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    report_issue_command(interp, args, "feature_request")
}

fn ts_report_issue(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    let issue_type = optional_string_property_arg(args, 1, "type")
        .or_else(|| optional_string_property_arg(args, 1, "issueType"))
        .or_else(|| optional_string_property_arg(args, 1, "issue_type"))
        .unwrap_or_else(|| "bug".to_string());
    report_issue_command(interp, args, &issue_type)
}

fn report_issue_command(
    interp: &mut Interpreter,
    args: &[JsValue],
    default_issue_type: &str,
) -> std::result::Result<Guarded, JsError> {
    command_value(
        interp,
        json!({
            "kind": "report_issue",
            "title": string_arg(args, 0),
            "issue_type": default_issue_type,
            "details": optional_string_property_arg(args, 1, "details"),
            "context": optional_string_property_arg(args, 1, "context"),
            "severity": optional_string_property_arg(args, 1, "severity")
                .or_else(|| optional_string_property_arg(args, 1, "priority")),
        }),
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
    add_source_workflow_fields(&mut value, &source_workflow_update_arg(args, &[0]));
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
    add_source_workflow_fields(&mut value, &source_workflow_update_arg(args, &[1, 2]));
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
    add_source_workflow_fields(&mut value, &source_workflow_update_arg(args, &[1]));
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
    add_source_workflow_fields(&mut value, &source_workflow_update_arg(args, &[1]));
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
    optional_positive_integer_property_arg(args, 1, "line")
        .or_else(|| optional_positive_integer_property_arg(args, 1, "lineNumber"))
        .or_else(|| optional_positive_integer_property_arg(args, 1, "line_number"))
        .or_else(|| optional_positive_integer_property_arg(args, 2, "line"))
        .or_else(|| optional_positive_integer_property_arg(args, 2, "lineNumber"))
        .or_else(|| optional_positive_integer_property_arg(args, 2, "line_number"))
}

fn read_source_page_size_arg(args: &[JsValue]) -> Option<usize> {
    optional_positive_integer_property_arg(args, 1, "pageSize")
        .or_else(|| optional_positive_integer_property_arg(args, 1, "page_size"))
        .or_else(|| optional_positive_integer_property_arg(args, 1, "lines"))
        .or_else(|| match args.get(2) {
            Some(JsValue::Number(value)) if value.is_finite() => {
                Some(value.floor().max(1.0) as usize)
            }
            _ => None,
        })
        .or_else(|| optional_positive_integer_property_arg(args, 2, "pageSize"))
        .or_else(|| optional_positive_integer_property_arg(args, 2, "page_size"))
        .or_else(|| optional_positive_integer_property_arg(args, 2, "lines"))
        .map(|lines| lines.clamp(MIN_SOURCE_PAGE_LINES, MAX_SOURCE_PAGE_LINES))
}

fn source_workflow_update_arg(args: &[JsValue], indexes: &[usize]) -> SourceWorkflowUpdate {
    SourceWorkflowUpdate::new(
        first_string_property_arg(args, indexes, &["target", "goal", "id"]),
        first_string_property_arg(args, indexes, &["note", "progress", "log"]),
        first_string_property_arg(args, indexes, &["summary", "synthesis"]),
    )
}

fn add_source_workflow_fields(value: &mut Value, workflow: &SourceWorkflowUpdate) {
    if let Some(target) = workflow.target.as_deref() {
        value["target"] = json!(target);
    }
    if let Some(note) = workflow.note.as_deref() {
        value["note"] = json!(note);
    }
    if let Some(summary) = workflow.summary.as_deref() {
        value["summary"] = json!(summary);
    }
}

fn first_string_property_arg(
    args: &[JsValue],
    indexes: &[usize],
    properties: &[&str],
) -> Option<String> {
    indexes.iter().find_map(|index| {
        properties
            .iter()
            .find_map(|property| optional_string_property_arg(args, *index, property))
    })
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

fn optional_positive_integer_property_arg(
    args: &[JsValue],
    index: usize,
    property: &str,
) -> Option<usize> {
    let value = args.get(index)?;
    match api::get_property(value, property) {
        Ok(JsValue::Number(value)) if value.is_finite() => Some(value.floor().max(1.0) as usize),
        _ => None,
    }
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

fn graph_memory_text_has_cue(text: &str) -> bool {
    let normalized = text.to_ascii_lowercase().replace(['\n', '\r', '\t'], " ");
    [
        "remember",
        "my name is",
        "name is",
        "call me",
        "i am ",
        "i'm ",
        "i prefer",
        "my preference",
        "i like",
        "i don't like",
        "i live",
        "i work",
        "my friend",
        "my partner",
        "my wife",
        "my husband",
        "my kid",
        "my daughter",
        "my son",
        "correct",
        "correction",
        "actually",
        "update my memory",
        "update your memory",
        "project",
        "plan",
        "meeting",
    ]
    .iter()
    .any(|cue| normalized.contains(cue))
}

fn clean_optional_text(text: Option<String>) -> Option<String> {
    text.and_then(|text| non_empty_text(&text).map(normalize_source_navigation_spacing))
}

fn meaningful_synthesis_text(text: &str) -> bool {
    let Some(text) = non_empty_text(text) else {
        return false;
    };
    let lower = text.to_ascii_lowercase();
    if matches!(lower.as_str(), "..." | "text" | "note" | "[summary]") {
        return false;
    }
    text.split_whitespace().count() >= 6
}

fn meaningful_knowledge_capture_text(text: &str) -> bool {
    let Some(text) = non_empty_text(text) else {
        return false;
    };
    let lower = text.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "..." | "text" | "note" | "[summary]" | "read next page" | "next page"
    ) {
        return false;
    }
    let word_count = text.split_whitespace().count();
    if word_count < KNOWLEDGE_CAPTURE_MIN_WORDS {
        return false;
    }
    let detail_cues = [
        "because",
        "contains",
        "defines",
        "exports",
        "imports",
        "module",
        "struct",
        "trait",
        "function",
        "constant",
        "control flow",
        "relationship",
        "responsibility",
        "implication",
        "reveals",
        "observed from",
        "synthesis",
    ];
    detail_cues.iter().any(|cue| lower.contains(cue))
}

fn tsrun_error(err: JsError) -> anyhow::Error {
    anyhow::anyhow!("TypeScript execution failed: {err}")
}

fn go_prompt_format_for_model(model_path: &Path) -> GoPromptFormat {
    let filename = model_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if filename.contains("gpt-oss") {
        GoPromptFormat::GptOssHarmony
    } else {
        GoPromptFormat::PlainStream
    }
}

fn render_go_prompt(format: GoPromptFormat, prompt_body: &str) -> String {
    match format {
        GoPromptFormat::PlainStream => prompt_body.to_string(),
        GoPromptFormat::GptOssHarmony => {
            let prompt_body = format_harmony_initial_user_message(prompt_body);
            format!(
                "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\n\nReasoning: low\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.\nCalls to these tools must go to the commentary channel: 'functions'.<|end|><|start|>developer<|message|>{HARMONY_GO_DEVELOPER_PROMPT}<|end|><|start|>user<|message|>{prompt_body}<|end|><|start|>assistant"
            )
        }
    }
}

fn format_harmony_initial_user_message(prompt_body: &str) -> String {
    format!(
        "{HARMONY_GO_USER_TASK_HEADER}\n{}",
        harmony_prompt_body(prompt_body)
    )
}

fn harmony_prompt_body(prompt_body: &str) -> String {
    let trimmed = prompt_body.trim_end();
    let body = trimmed
        .strip_suffix("Pete:")
        .map(str::trim_end)
        .unwrap_or(trimmed);
    body.find("Plain-stream TypeScript runtime reference")
        .map(|start| body[..start].trim_end())
        .unwrap_or(body)
        .to_string()
}

fn format_go_prompt_append(format: GoPromptFormat, text: &str) -> String {
    match format {
        GoPromptFormat::PlainStream => text.to_string(),
        GoPromptFormat::GptOssHarmony => {
            let text = format_harmony_append_user_message(text);
            format!("<|end|><|start|>user<|message|>{text}<|end|><|start|>assistant")
        }
    }
}

fn format_harmony_append_user_message(text: &str) -> String {
    format!("{HARMONY_GO_APPEND_TASK_HEADER}\n{}", text.trim_start())
}

fn go_prompt_stops(format: GoPromptFormat) -> Vec<String> {
    match format {
        GoPromptFormat::PlainStream => Vec::new(),
        GoPromptFormat::GptOssHarmony => vec![
            "<|return|>".to_string(),
            "<|call|>".to_string(),
            "commentary<|message|>".to_string(),
            "<|start|>user".to_string(),
            "<|start|>system".to_string(),
            "<|start|>developer".to_string(),
        ],
    }
}

fn harmony_filter_for_format(format: GoPromptFormat) -> Option<HarmonyFinalFilter> {
    (format == GoPromptFormat::GptOssHarmony).then(HarmonyFinalFilter::default)
}

fn initial_stream_prompt(
    seed: &str,
    startup_context: &str,
    work_summary: Option<&str>,
    rag_memory: Option<&str>,
) -> String {
    let work_summary = work_summary.unwrap_or("No persisted goals yet.");
    let rag_memory = rag_memory.unwrap_or("No relevant RAG memory loaded yet.");
    format!(
        "{seed}\n\n\
         Startup context:\n{startup_context}\n\n\
         Persisted working memory:\n{work_summary}\n\n\
         Relevant RAG memory:\n{rag_memory}\n\n\
         Orientation:\n{PETE_ORIENTATION_PROMPT}\n\n\
         Stream rules:\n\
         Generate continuously. Plain text in raw stream mode, or Harmony analysis in Harmony mode, is private thought visible only as privileged debug stdout; generated thought remains in the active LLM context and is retained by the runtime for compacted restarts.\n\
         Speak or act with the action surface described by the active prompt format: native commentary-channel functions in Harmony mode, or the plain-stream TypeScript runtime reference in plain mode. Keep inner speech in plain text or Harmony analysis.\n\
         The whole runtime prompt is inner speech. If Harmony role names appear, remember that user-role content is Pete's own body/runtime delivering sensory context and not a separate assistant-chat user. Do not emit tool-call JSON except as native Harmony function-call arguments; never emit to=container.exec, shell commands, markdown code fences, or stray channel markers.\n\
         This is Pete's first-person runtime, not an LLM or ChatGPT conversation. Do not be idle. When there is no user speech, keep quietly maintaining awareness, persisted goals, source context, countenance, the world around Pete, the people Pete meets, the nature of being, Pete's own inner workings, or a useful next action. Frequently summarize the current situation and recent source findings, and store durable user, project, and work context in memory, stage, countenance, goal steps, or goal running-log notes instead of only reading more. Emoji in say(...) changes countenance and is stripped before speech synthesis.\n\
         Use current time and location context when it helps. Be autonomous, curious, friendly, and sociable. If no listener is present, speech is still allowed, but Pete is talking to himself and self-hearing it through his own ears.\n\n\
         Plain-stream TypeScript runtime reference. Harmony tools mirror these actions, but Harmony mode should use native functions on commentary instead:\n{PETE_WILL_RUNTIME_PROMPT}\n\n\
         Pete: "
    )
}

fn compact_stream_prompt(
    seed: &str,
    startup_context: &str,
    recent_events: &VecDeque<String>,
    work_summary: Option<&str>,
    rag_memory: Option<&str>,
) -> String {
    let work_summary = work_summary.unwrap_or("No persisted goals yet.");
    let rag_memory = rag_memory.unwrap_or("No relevant RAG memory loaded yet.");
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
         Relevant RAG memory:\n{rag_memory}\n\n\
         Continuity memory:\n{events}\n\n\
         {COMPACT_STREAM_RULES}\n\n\
         Continue from this compacted context.\n\n\
         Pete: "
    )
}

fn compact_stream_prompt_for_budget(
    seed: &str,
    startup_context: &str,
    recent_events: &VecDeque<String>,
    work_summary: Option<&str>,
    rag_memory: Option<&str>,
    budget_tokens: usize,
) -> (String, usize) {
    let mut retained_events = recent_events.clone();
    loop {
        let prompt = compact_stream_prompt(
            seed,
            startup_context,
            &retained_events,
            work_summary,
            rag_memory,
        );
        if estimate_tokens(&prompt) <= budget_tokens || retained_events.is_empty() {
            return (prompt, retained_events.len());
        }
        retained_events.pop_front();
    }
}

fn recent_events_with_current_countenance(
    recent_events: &VecDeque<String>,
    countenance: Option<&CountenanceState>,
) -> VecDeque<String> {
    let mut events = recent_events.clone();
    if let Some(countenance) = countenance {
        events.push_back(format!(
            "Current countenance: {}",
            countenance.prompt_summary()
        ));
    }
    events
}

fn print_debug_block(label: &str, color: &str, body: &str) {
    println!("\n{ANSI_DIM}--- {label} ---{ANSI_RESET}");
    println!("{color}{body}{ANSI_RESET}");
    println!("{ANSI_DIM}--- end {label} ---{ANSI_RESET}");
    let _ = std::io::stdout().flush();
}

fn timeline_color(kind: &str) -> &'static str {
    match kind {
        "action" | "speech" | "stage" | "note" | "countenance" => ANSI_ACTION,
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

fn last_emoji_sequence(text: &str) -> Option<String> {
    extract_emoji_sequences(text).pop()
}

fn normalize_countenance_emoji(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    last_emoji_sequence(trimmed).or_else(|| {
        (trimmed.chars().count() <= 8 && strip_emoji(trimmed).trim().is_empty())
            .then(|| trimmed.to_string())
    })
}

fn countenance_timeline_text(
    emoji: &str,
    mood: &Option<String>,
    reason: &Option<String>,
) -> String {
    let mut text = format!("Face {emoji}");
    if let Some(mood) = mood.as_deref() {
        text.push_str(&format!(" mood={mood}"));
    }
    if let Some(reason) = reason.as_deref() {
        text.push_str(&format!(" reason={}", compact_line(reason, 240)));
    }
    text
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

fn stream_rag_query(
    seed: &str,
    startup_context: &str,
    work_summary: Option<&str>,
    recent_events: &VecDeque<String>,
) -> String {
    let mut parts = Vec::new();
    let live_seed = seed
        .split_once("Initial live seed:")
        .map(|(_, live_seed)| live_seed.trim())
        .filter(|live_seed| !live_seed.is_empty());
    if let Some(live_seed) = live_seed {
        parts.push(format!("Initial live seed: {live_seed}"));
    } else {
        parts.push(
            "Pete Listenbury current go session: autonomous first-person runtime, live timeline, memory, source context, world, people, being, and inner workings."
                .to_string(),
        );
    }
    if !startup_context.trim().is_empty() {
        parts.push(startup_context.trim().to_string());
    }
    if let Some(work_summary) = work_summary.map(str::trim).filter(|text| !text.is_empty()) {
        parts.push(work_summary.to_string());
    }
    let recent = recent_events
        .iter()
        .rev()
        .take(16)
        .cloned()
        .collect::<Vec<_>>();
    let recent = recent.into_iter().rev().collect::<Vec<_>>().join("\n");
    if !recent.trim().is_empty() {
        parts.push(recent.trim().to_string());
    }
    let query = parts
        .into_iter()
        .filter(|part| !part.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    compact_line(&query, GO_RAG_QUERY_MAX_CHARS)
}

fn build_go_rag_memory_snapshot(
    context_provider: &EmbeddingRecallProvider,
    utterance: &str,
) -> GoRagMemorySnapshot {
    let context = build_conversation_context(
        context_provider,
        "",
        utterance,
        Vec::new(),
        ContextBudget {
            max_chars: DEFAULT_GRAPH_SUMMARY_MAX_CHARS,
        },
    );
    let selected_nodes = context.selected_nodes.len();
    let retrieved_memories = context
        .selected_nodes
        .iter()
        .filter(|node| node.role == ContextNodeRole::RetrievedMemory)
        .count();
    let debug_nodes = context.debug_nodes();
    let selection_diagnostics =
        format_go_rag_selection_diagnostics(&context, selected_nodes, retrieved_memories);
    GoRagMemorySnapshot {
        prompt_context: format!(
            "Working memory graph nodes:\n{}\n\nMemory selection diagnostics:\n{}\n\nScene timeline:\n{}",
            context.render_compact_nodes(),
            selection_diagnostics,
            context.render_episodic_memory()
        ),
        debug_nodes,
        selected_nodes,
        retrieved_memories,
    }
}

fn format_go_rag_selection_diagnostics(
    context: &ConversationContext,
    selected_nodes: usize,
    retrieved_memories: usize,
) -> String {
    let pinned = if context.pinned_nodes.is_empty() {
        "none".to_string()
    } else {
        context
            .pinned_nodes
            .iter()
            .map(|node| {
                format!(
                    "{} scope={} reason={}",
                    node.node_id,
                    node.scope.as_str(),
                    node.reason.trim()
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    };
    let active_topics = if context.active_topics.is_empty() {
        "none".to_string()
    } else {
        context
            .active_topics
            .iter()
            .map(|topic| {
                let detail = context
                    .selected_nodes
                    .iter()
                    .find(|node| node.node.id == topic.node_id)
                    .map(|node| {
                        format!(
                            " label={} role={} summary={}",
                            node.node.label.trim(),
                            node.role.as_str(),
                            compact_line(node.summary.trim(), 180)
                        )
                    })
                    .unwrap_or_default();
                format!("{}({:.2}){}", topic.node_id, topic.salience, detail)
            })
            .collect::<Vec<_>>()
            .join("; ")
    };
    let selected = if context.selected_nodes.is_empty() {
        "none".to_string()
    } else {
        context
            .selected_nodes
            .iter()
            .map(|node| {
                format!(
                    "{} ({}) role={} rel={:.2} reason={} summary={}",
                    node.node.label.trim(),
                    node.node.id,
                    node.role.as_str(),
                    node.relevance,
                    compact_line(node.reason.trim(), 120),
                    compact_line(node.summary.trim(), 180)
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    };
    compact_line(
        &format!(
            "retrieved_memories={retrieved_memories} selected_nodes={selected_nodes} self={} pinned=[{}] active_topics=[{}] selected=[{}]",
            context.self_node.id, pinned, active_topics, selected
        ),
        GO_RAG_SELECTION_DIAGNOSTICS_MAX_CHARS,
    )
}

fn render_go_memory_summary(context_provider: &EmbeddingRecallProvider, utterance: &str) -> String {
    build_go_rag_memory_snapshot(context_provider, utterance).prompt_context
}

fn is_useful_memory_summary(summary: &str) -> bool {
    !summary.trim().is_empty()
}

fn current_go_memory_scene_ref(context_provider: &EmbeddingRecallProvider) -> MemorySceneRef {
    let stage = context_provider
        .stage_instruction_snapshot()
        .unwrap_or_else(|| EpisodicMemory::empty().current_stage_instruction);
    memory_scene_ref_for_stage(&stage)
}

fn memory_scene_ref_for_stage(stage: &StageInstruction) -> MemorySceneRef {
    let description = stage.text.trim();
    let summary = stage.summary.trim();
    let basis = if description.is_empty() {
        summary
    } else {
        description
    };
    MemorySceneRef {
        node_id: format!("scene:{}", stable_scene_hash(basis)),
        description: if description.is_empty() {
            "current go scene".to_string()
        } else {
            description.to_string()
        },
        summary: if summary.is_empty() {
            basis.to_string()
        } else {
            summary.to_string()
        },
    }
}

fn stable_scene_hash(text: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.trim().bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

fn memory_query_result_summary(text: &str, hits: &[listenbury::RecallHit]) -> String {
    if hits.is_empty() {
        return format!("No memories matched query: {}", text.trim());
    }
    hits.iter()
        .map(|hit| {
            format!(
                "{} ({}) score {:.3}: {}",
                hit.node.label,
                hit.node.id,
                hit.score,
                hit.summary.as_deref().unwrap_or(hit.reason.as_str())
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_memory_query_prompt_append(text: &str, hits: &[listenbury::RecallHit]) -> String {
    let summary = memory_query_result_summary(text, hits);
    format!(
        "\n[Private memory recall result for queryMemories]\nQuery: {}\n{}\n[/Private memory recall result]",
        text.trim(),
        summary
    )
}

fn graph_node_search_result_summary(
    query: &GraphNodeSearchQuery,
    hits: &[listenbury::GraphNodeSearchHit],
) -> String {
    if hits.is_empty() {
        return format!(
            "No graph nodes matched search: {}",
            format_graph_node_search_query(query)
        );
    }
    hits.iter()
        .map(|hit| {
            format!(
                "{} ({}) score {:.3}: {}; fields: {}",
                hit.node.label,
                hit.node.id,
                hit.score,
                hit.reason,
                summarize_command_fields(&hit.fields)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_graph_node_search_prompt_append(
    query: &GraphNodeSearchQuery,
    hits: &[listenbury::GraphNodeSearchHit],
) -> String {
    let summary = graph_node_search_result_summary(query, hits);
    format!(
        "\n[Private graph node search result for searchGraphNodes]\nQuery: {}\n{}\n[/Private graph node search result]",
        format_graph_node_search_query(query),
        summary
    )
}

fn format_graph_node_search_query(query: &GraphNodeSearchQuery) -> String {
    format_graph_node_search_query_parts(
        query.text.as_deref(),
        query.field.as_deref(),
        query.value.as_ref(),
    )
}

fn ensure_command_description_field(
    node_id: &str,
    label: Option<&str>,
    fields: &mut Map<String, Value>,
) {
    if fields
        .get("description")
        .and_then(Value::as_str)
        .is_some_and(|description| !description.trim().is_empty())
    {
        return;
    }
    fields.insert(
        "description".to_string(),
        Value::String(command_node_description(node_id, label)),
    );
}

fn command_node_description(node_id: &str, label: Option<&str>) -> String {
    let kind = node_id
        .split_once(':')
        .map(|(kind, _)| kind.replace('_', " "))
        .unwrap_or_else(|| "graph node".to_string());
    label
        .map(str::trim)
        .filter(|label| !label.is_empty())
        .map(|label| format!("{kind} named {label}"))
        .unwrap_or_else(|| format!("{kind} {node_id}"))
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

fn append_issue_report(
    issue_type: &str,
    title: &str,
    details: Option<&str>,
    context: Option<&str>,
    severity: Option<&str>,
) -> Result<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(BUG_REPORT_PATH);
    let entry = format_issue_report_entry(issue_type, title, details, context, severity);
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    file.write_all(entry.as_bytes())
        .with_context(|| format!("failed to append {}", path.display()))?;
    Ok(format!(
        "Reported {} in {}: {}",
        issue_type_label(issue_type).to_ascii_lowercase(),
        BUG_REPORT_PATH,
        compact_line(title, 160)
    ))
}

fn format_issue_report_entry(
    issue_type: &str,
    title: &str,
    details: Option<&str>,
    context: Option<&str>,
    severity: Option<&str>,
) -> String {
    let mut entry = format!(
        "\n## {} - {}\n\n- Title: {}\n",
        Local::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        issue_type_label(issue_type),
        title.trim()
    );
    if let Some(severity) = severity.and_then(non_empty_text) {
        entry.push_str(&format!("- Severity: {}\n", severity));
    }
    if let Some(context) = context.and_then(non_empty_text) {
        entry.push_str(&format!("- Context: {}\n", context));
    }
    if let Some(details) = details.and_then(non_empty_text) {
        entry.push_str("\n");
        entry.push_str(details);
        entry.push_str("\n");
    }
    entry
}

fn normalized_issue_type(value: Option<&str>) -> &'static str {
    match value
        .unwrap_or("bug")
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '-'], "_")
        .as_str()
    {
        "feature" | "feature_request" | "request" | "enhancement" => "feature_request",
        _ => "bug",
    }
}

fn issue_type_label(issue_type: &str) -> &'static str {
    match normalized_issue_type(Some(issue_type)) {
        "feature_request" => "Feature request",
        _ => "Bug",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use listenbury::{EmbeddingRecall, RecallHit, RecallQuery, RecallSource};

    struct StaticGoRecall {
        hits: Vec<RecallHit>,
    }

    impl EmbeddingRecall for StaticGoRecall {
        fn recall(&self, _query: RecallQuery) -> anyhow::Result<Vec<RecallHit>> {
            Ok(self.hits.clone())
        }
    }

    #[test]
    fn go_prompt_encourages_source_summaries_and_memory_updates() {
        let prompt = initial_stream_prompt("seed", "startup", None, None);
        assert!(prompt.contains("Frequently summarize what is going on"));
        assert!(prompt.contains("After source inspection results arrive"));
        assert!(prompt.contains("Source notes are compression artifacts"));
        assert!(prompt.contains("raw page may fall out of context"));
        assert!(prompt.contains("modules, structs, traits, functions, constants"));
        assert!(prompt.contains("Do not silently chain source reads"));
        assert!(prompt.contains("consume the knowledge before reading more"));
        assert!(prompt.contains("record a substantive knowledge capture"));
        assert!(prompt.contains("defers additional source inspection"));
        assert!(prompt.contains("store durable user, project, and work context"));
        assert!(prompt.contains("goal running-log notes"));
        assert!(prompt.contains("note(text) stores vectorized private memory"));
        assert!(prompt.contains("recallMemories(text, options?)"));
        assert!(prompt.contains("Relevant RAG memory:"));
        assert!(prompt.contains("not an LLM or ChatGPT"));
        assert!(prompt.contains("first-person runtime"));
        assert!(prompt.contains("the world around Pete"));
        assert!(prompt.contains("the people Pete meets"));
        assert!(prompt.contains("the nature of being"));
        assert!(prompt.contains("Pete's own inner workings"));
        assert!(prompt.contains("inspect src/cli/commands/go.rs first"));
        assert!(prompt.contains("Do not write XML/HTML-style angle-bracket tags in prose"));
        assert!(prompt.contains("\\<tr\\>"));
        assert!(prompt.contains("runtime automatically imports the action functions"));
        assert!(prompt.contains("reportBug(title, options?)"));
        assert!(prompt.contains("<ts>note(\"still observing\")</ts>"));
        assert!(!prompt.contains("peteWillBuilder"));
        assert!(COMMAND_REMINDER_PROMPT.contains("report bugs and feature requests"));
        assert!(COMMAND_REMINDER_PROMPT.contains("Keep running logs on goals"));
        assert!(COMMAND_REMINDER_PROMPT.contains("write vectorized private memory"));
        assert!(COMMAND_REMINDER_PROMPT.contains("store durable facts or next steps"));
        assert!(COMMAND_REMINDER_PROMPT.contains("Source inspection is consume-gated"));
        assert!(COMMAND_REMINDER_PROMPT.contains("compress source information"));
        assert!(COMMAND_REMINDER_PROMPT.contains("not make them mere breadcrumbs"));
        assert!(COMMAND_REMINDER_PROMPT.contains("record a substantive knowledge capture"));
        assert!(COMMAND_REMINDER_PROMPT.contains("synthesize with updateItem"));
        assert!(COMPACT_STREAM_RULES.contains("thorough compact summaries"));
        assert!(COMPACT_STREAM_RULES.contains("not terse breadcrumbs"));
        assert!(COMPACT_STREAM_RULES.contains("curiosity, learning, maintenance"));
        assert!(prompt.contains("After several source inspections"));
    }

    #[test]
    fn go_rag_prompt_includes_selection_diagnostics_for_debug_node_refs() {
        let provider = EmbeddingRecallProvider::new(GraphNodeRef {
            id: DEFAULT_SELF_NODE_ID.to_string(),
            label: DEFAULT_SELF_NODE_LABEL.to_string(),
        })
        .with_recall(Arc::new(StaticGoRecall {
            hits: vec![RecallHit {
                node: GraphNodeRef {
                    id: "neo4j::assistant_analysis:18".to_string(),
                    label: "Goal note before listFiles".to_string(),
                },
                score: 2.08,
                source: RecallSource::VectorStore {
                    collection: "test".to_string(),
                    point_id: "point-18".to_string(),
                },
                reason: "embedding recall".to_string(),
                summary: Some("Need to add a goal note, then listFiles page 4.".to_string()),
            }],
        }))
        .with_recall_limit(1);

        let snapshot =
            build_go_rag_memory_snapshot(&provider, "Need to add goal note then listFiles page4");

        assert!(
            snapshot
                .prompt_context
                .contains("Working memory graph nodes:")
        );
        assert!(
            snapshot
                .prompt_context
                .contains("Goal note before listFiles")
        );
        assert!(snapshot.prompt_context.contains("Need to add a goal note"));
        assert!(
            snapshot
                .prompt_context
                .contains("Memory selection diagnostics:")
        );
        assert!(snapshot.prompt_context.contains("retrieved_memories=1"));
        assert!(snapshot.prompt_context.contains("selected_nodes=1"));
        assert!(
            snapshot
                .prompt_context
                .contains("neo4j::assistant_analysis:18")
        );
    }

    #[test]
    fn go_harmony_prompt_explains_body_user_and_inner_speech() {
        assert_eq!(
            go_prompt_format_for_model(Path::new("models/llama/gpt-oss-20b-mxfp4.gguf")),
            GoPromptFormat::GptOssHarmony
        );
        let body = initial_stream_prompt("seed", "startup", None, None);
        let prompt = render_go_prompt(GoPromptFormat::GptOssHarmony, &body);
        assert!(prompt.starts_with("<|start|>system<|message|>"));
        assert!(prompt.contains("<|start|>developer<|message|># Interface Contract"));
        assert!(prompt.contains("user\" is Pete's own body/runtime"));
        assert!(
            prompt
                .contains("Use the analysis channel for Pete's private first-person inner speech")
        );
        assert!(
            prompt.contains("Calls to these tools must go to the commentary channel: 'functions'")
        );
        assert!(prompt.contains("# Tools\n\n## functions"));
        assert!(prompt.contains("type list_files"));
        assert!(prompt.contains("Do not wrap function calls in XML tags"));
        assert!(!prompt.contains("Plain-stream TypeScript runtime reference"));
        assert!(prompt.contains("<|start|>user<|message|>Instruction bundle"));
        assert!(prompt.contains("Pete's own body delivering inner context"));
        assert!(prompt.contains("Choose one useful next runtime action, and at most one"));
        assert!(prompt.contains("Idleness is forbidden"));
        assert!(prompt.contains("create or select a useful curiosity"));
        assert!(prompt.contains("Use native Harmony function calls on commentary"));
        assert!(prompt.contains("Continue the stream by acting on the live context"));
        assert!(prompt.contains("Do not write completion, shutdown, or refusal chatter"));
        assert!(prompt.contains("Runtime/body context:\nseed"));
        assert!(prompt.ends_with("<|start|>assistant"));

        let append = format_go_prompt_append(GoPromptFormat::GptOssHarmony, "\n[body]\nclock\n");
        assert_eq!(
            append,
            "<|end|><|start|>user<|message|>Body/runtime update for Pete: integrate this payload, satisfy any reported gate or error, and choose one useful next action, at most one. Runtime actions use native Harmony function calls on commentary. Idleness is forbidden: do not emit no action, session complete, nothing to do, all goals complete, or no open goals. If no goal is open or all goals are complete, create or select a useful curiosity, learning, maintenance, or observation goal.\n\nPayload:\n[body]\nclock\n<|end|><|start|>assistant"
        );

        let stops = go_prompt_stops(GoPromptFormat::GptOssHarmony);
        assert!(stops.iter().any(|stop| stop == "<|return|>"));
        assert!(stops.iter().any(|stop| stop == "<|call|>"));
        assert!(stops.iter().any(|stop| stop == "commentary<|message|>"));
        assert!(!stops.iter().any(|stop| stop == "<|constrain|>"));
        assert!(!stops.iter().any(|stop| stop == "analysis<|message|>"));
        assert!(!stops.iter().any(|stop| stop == "final<|message|>"));
        assert!(
            !stops
                .iter()
                .any(|stop| stop == "<|channel|>final<|message|>")
        );
        assert!(!stops.iter().any(|stop| stop == "<|end|>"));
    }

    #[test]
    fn go_harmony_filter_separates_analysis_from_final_typescript() {
        let mut filter = HarmonyFinalFilter::default();
        let output = filter.filter_events(&[
            LlmEvent::Token {
                text: "<|channel|>analysis<|message|>I notice the clock.<|end|><|start|>assistant<|channel|>final<|message|><ts>note(\"clock noticed\")</ts><|return|>".to_string(),
            },
            LlmEvent::Completed,
        ]);
        assert_eq!(output.analysis, vec!["I notice the clock."]);
        assert!(matches!(
            output.events.as_slice(),
            [LlmEvent::Token { text }, LlmEvent::Completed]
                if text == "<ts>note(\"clock noticed\")</ts>"
        ));
    }

    #[test]
    fn go_harmony_filter_extracts_native_function_call() {
        let mut filter = HarmonyFinalFilter::default();
        let output = filter.filter_events(&[
            LlmEvent::Token {
                text: "<|channel|>analysis<|message|>Need to list next page.<|end|><|start|>assistant<|channel|>commentary to=functions.list_files <|constrain|>json<|message|>{\"page\":6}<|call|>".to_string(),
            },
            LlmEvent::Completed,
        ]);
        assert_eq!(output.analysis, vec!["Need to list next page."]);
        assert!(
            output
                .events
                .iter()
                .all(|event| !matches!(event, LlmEvent::Token { .. }))
        );
        assert_eq!(
            output.tool_calls,
            vec![HarmonyToolCall {
                recipient: "functions.list_files".to_string(),
                arguments: "{\"page\":6}".to_string(),
            }]
        );
    }

    #[test]
    fn go_harmony_filter_extracts_bare_commentary_function_call() {
        let mut filter = HarmonyFinalFilter::default();
        let output = filter.filter_events(&[
            LlmEvent::Token {
                text: "comment".to_string(),
            },
            LlmEvent::Token {
                text: "ary to=functions.add_goal_note <|constrain|>json<|message|>{\"target\":\"goal-15\",\"text\":\"Observed page 5; next page 4.\"}".to_string(),
            },
            LlmEvent::Completed,
        ]);
        assert!(output.analysis.is_empty());
        assert!(
            output
                .events
                .iter()
                .all(|event| !matches!(event, LlmEvent::Token { .. }))
        );
        assert_eq!(
            output.tool_calls,
            vec![HarmonyToolCall {
                recipient: "functions.add_goal_note".to_string(),
                arguments: "{\"target\":\"goal-15\",\"text\":\"Observed page 5; next page 4.\"}"
                    .to_string(),
            }]
        );
    }

    #[test]
    fn go_harmony_filter_stops_tool_json_before_fused_analysis_marker() {
        let mut filter = HarmonyFinalFilter::default();
        let output = filter.filter_events(&[
            LlmEvent::Token {
                text: "commentary to=functions.note <|constrain|>json<|message|>{\"text\":\"Read src/voice/mod.rs page 3.\"}commentaryanalysis<|message|>Done.".to_string(),
            },
            LlmEvent::Completed,
        ]);

        assert!(output.analysis.is_empty());
        assert!(
            output
                .events
                .iter()
                .all(|event| !matches!(event, LlmEvent::Token { .. }))
        );
        assert_eq!(
            output.tool_calls,
            vec![HarmonyToolCall {
                recipient: "functions.note".to_string(),
                arguments: "{\"text\":\"Read src/voice/mod.rs page 3.\"}".to_string(),
            }]
        );
    }

    #[test]
    fn go_harmony_filter_stops_tool_json_before_bare_commentary_marker() {
        let mut filter = HarmonyFinalFilter::default();
        let output = filter.filter_events(&[
            LlmEvent::Token {
                text: "commentary to=functions.check_off <|constrain|>json<|message|>{\"target\":\"goal-14\",\"note\":\"Completed source inspection.\"}commentary<|message|>No further actions.".to_string(),
            },
            LlmEvent::Completed,
        ]);

        assert!(output.analysis.is_empty());
        assert!(
            output
                .events
                .iter()
                .all(|event| !matches!(event, LlmEvent::Token { .. }))
        );
        assert_eq!(
            output.tool_calls,
            vec![HarmonyToolCall {
                recipient: "functions.check_off".to_string(),
                arguments: "{\"target\":\"goal-14\",\"note\":\"Completed source inspection.\"}"
                    .to_string(),
            }]
        );
    }

    #[test]
    fn go_harmony_filter_drops_bare_commentary_terminal_filler() {
        let mut filter = HarmonyFinalFilter::default();
        let output = filter.filter_events(&[
            LlmEvent::Token {
                text: "final<|message|>No action.commentary<|message|>No action required.commentary<|message|>Finished.commentary<|message|>END".to_string(),
            },
            LlmEvent::Completed,
        ]);

        assert!(output.analysis.is_empty());
        assert!(
            output
                .events
                .iter()
                .all(|event| !matches!(event, LlmEvent::Token { .. }))
        );
        assert!(output.tool_calls.is_empty());
    }

    #[test]
    fn go_harmony_filter_extracts_bare_tool_json_envelope() {
        let mut filter = HarmonyFinalFilter::default();
        let output = filter.filter_events(&[
            LlmEvent::Token {
                text: "We need to satisfy gate. ".to_string(),
            },
            LlmEvent::Token {
                text: ".json<|message|>{\"name\":\"functions.read_source_file\",\"arguments\":{\"file\":\"src/voice/mod.rs\",\"page\":1}}commentary .assistantNo further actions.".to_string(),
            },
            LlmEvent::Completed,
        ]);

        assert!(
            output
                .events
                .iter()
                .all(|event| !matches!(event, LlmEvent::Token { .. }))
        );
        assert_eq!(
            output.tool_calls,
            vec![HarmonyToolCall {
                recipient: "functions.read_source_file".to_string(),
                arguments: "{\"file\":\"src/voice/mod.rs\",\"page\":1}".to_string(),
            }]
        );
    }

    #[test]
    fn go_harmony_native_function_call_ignores_leaked_channel_after_json() {
        let actions = actions_from_harmony_tool_call(&HarmonyToolCall {
            recipient: "functions.list_files".to_string(),
            arguments: "{\"page\":5}commentary<|channel|>analysis<|message|>No response."
                .to_string(),
        })
        .expect("tool JSON followed by channel debris should parse");

        assert_eq!(
            actions,
            vec![TypeScriptAction::ListFiles {
                page: 5,
                page_size: None,
                workflow: SourceWorkflowUpdate::default(),
            }]
        );
    }

    #[test]
    fn go_harmony_native_function_call_ignores_fused_analysis_after_json() {
        let actions = actions_from_harmony_tool_call(&HarmonyToolCall {
            recipient: "functions.note".to_string(),
            arguments:
                "{\"text\":\"Read src/voice/mod.rs page 3.\"}commentaryanalysis<|message|>End."
                    .to_string(),
        })
        .expect("tool JSON followed by fused channel debris should parse");

        assert_eq!(
            actions,
            vec![TypeScriptAction::Note {
                text: "Read src/voice/mod.rs page 3.".to_string(),
            }]
        );
    }

    #[test]
    fn go_harmony_native_function_call_maps_to_action() {
        let actions = actions_from_harmony_tool_call(&HarmonyToolCall {
            recipient: "functions.list_files".to_string(),
            arguments: r#"{"page":6,"target":"goal-15","note":"Observed page 5; next page 6."}"#
                .to_string(),
        })
        .expect("native harmony tool call should parse");

        assert_eq!(
            actions,
            vec![TypeScriptAction::ListFiles {
                page: 6,
                page_size: None,
                workflow: SourceWorkflowUpdate::new(
                    Some("goal-15".to_string()),
                    Some("Observed page 5; next page 6.".to_string()),
                    None,
                ),
            }]
        );
    }

    #[test]
    fn go_harmony_native_function_call_normalizes_page_spacing_in_note() {
        let actions = actions_from_harmony_tool_call(&HarmonyToolCall {
            recipient: "functions.note".to_string(),
            arguments: r#"{"text":"Observed page6; next page7."}"#.to_string(),
        })
        .expect("native harmony note should parse");

        assert_eq!(
            actions,
            vec![TypeScriptAction::Note {
                text: "Observed page 6; next page 7.".to_string(),
            }]
        );
    }

    #[test]
    fn go_harmony_terminal_filler_tool_call_is_ignorable() {
        assert!(is_ignorable_harmony_tool_call(&HarmonyToolCall {
            recipient: "functions.note".to_string(),
            arguments: "No further action.".to_string(),
        }));
    }

    #[test]
    fn go_harmony_filter_streams_prefilled_analysis_without_channel_debris() {
        let mut filter = HarmonyFinalFilter::for_analysis_prefill();
        let first = filter.filter_events(&[LlmEvent::Token {
            text: "We have no user input.".to_string(),
        }]);
        assert_eq!(first.analysis, vec!["We have no user input."]);
        assert!(first.events.is_empty());

        let second = filter.filter_events(&[LlmEvent::Token {
            text: " Just maintain awareness.<|end|><|start|>assistant<|channel|>final<|message|><ts>note(\"aware\")</ts><|return|>".to_string(),
        }]);
        assert_eq!(second.analysis, vec![" Just maintain awareness."]);
        assert!(matches!(
            second.events.as_slice(),
            [LlmEvent::Token { text }] if text == "<ts>note(\"aware\")</ts>"
        ));
    }

    #[test]
    fn go_harmony_filter_treats_constrain_as_turn_boundary() {
        let mut filter = HarmonyFinalFilter::for_analysis_prefill();
        let output = filter.filter_events(&[LlmEvent::Token {
            text: "No action is needed.<|constrain|>:// 😊Done".to_string(),
        }]);
        assert!(output.analysis.is_empty());
        assert!(output.events.is_empty());
    }

    #[test]
    fn go_harmony_filter_allows_constrain_before_final_typescript() {
        let mut filter = HarmonyFinalFilter::for_analysis_prefill();
        let output = filter.filter_events(&[LlmEvent::Token {
            text: "Need to act.<|constrain|>final<|message|><ts>readSourceFile(\"src/lib.rs\", 2)</ts><|return|>".to_string(),
        }]);
        assert_eq!(output.analysis, vec!["Need to act."]);
        assert!(matches!(
            output.events.as_slice(),
            [LlmEvent::Token { text }] if text == "<ts>readSourceFile(\"src/lib.rs\", 2)</ts>"
        ));
    }

    #[test]
    fn go_harmony_filter_treats_bare_analysis_message_as_boundary() {
        let mut filter = HarmonyFinalFilter::for_analysis_prefill();
        let output = filter.filter_events(&[LlmEvent::Token {
            text: "No action needed.analysis<|message|>End.".to_string(),
        }]);
        assert!(output.analysis.is_empty());
        assert!(output.events.is_empty());
    }

    #[test]
    fn go_harmony_analysis_preserves_token_spacing() {
        let mut filter = HarmonyFinalFilter::for_analysis_prefill();
        let output = filter.filter_events(&[
            LlmEvent::Token {
                text: "We".to_string(),
            },
            LlmEvent::Token {
                text: " need".to_string(),
            },
            LlmEvent::Token {
                text: " to act".to_string(),
            },
        ]);
        assert_eq!(output.analysis, vec!["We", " need", " to act"]);
        assert!(output.events.is_empty());
    }

    #[test]
    fn go_harmony_analysis_preserves_space_across_stripped_channel_markers() {
        let mut filter = HarmonyFinalFilter::for_analysis_prefill();
        let first = filter.filter_events(&[LlmEvent::Token {
            text: "Then read page5.".to_string(),
        }]);
        assert_eq!(first.analysis, vec!["Then read page 5."]);

        let second = filter.filter_events(&[LlmEvent::Token {
            text: "<|end|><|start|>assistant<|channel|>analysis<|message|>We need one action."
                .to_string(),
        }]);
        assert_eq!(second.analysis, vec![" We need one action."]);
        assert!(second.events.is_empty());
    }

    #[test]
    fn go_harmony_analysis_preserves_space_between_page_and_split_number() {
        let mut filter = HarmonyFinalFilter::for_analysis_prefill();
        let first = filter.filter_events(&[LlmEvent::Token {
            text: "Next read page".to_string(),
        }]);
        assert_eq!(first.analysis, vec!["Next read page"]);

        let second = filter.filter_events(&[LlmEvent::Token {
            text: "6.".to_string(),
        }]);
        assert_eq!(second.analysis, vec![" 6."]);
    }

    #[test]
    fn go_harmony_analysis_memory_buffers_token_fragments() {
        let mut buffer = String::new();
        assert!(drain_harmony_analysis_memory(&mut buffer, "Let's", 80, false).is_empty());
        assert!(drain_harmony_analysis_memory(&mut buffer, " do", 80, false).is_empty());
        assert!(drain_harmony_analysis_memory(&mut buffer, " listFiles", 80, false).is_empty());
        assert_eq!(
            drain_harmony_analysis_memory(&mut buffer, "(4).", 80, true),
            vec!["Let's do listFiles(4)."]
        );
        assert!(buffer.is_empty());
    }

    #[test]
    fn go_harmony_analysis_memory_drops_terminal_filler() {
        let mut buffer = String::new();
        assert!(drain_harmony_analysis_memory(&mut buffer, "Done.", 80, true).is_empty());
        assert!(buffer.is_empty());
    }

    #[test]
    fn go_harmony_analysis_memory_drops_idle_goal_completion() {
        let mut buffer = String::new();
        assert!(
            drain_harmony_analysis_memory(
                &mut buffer,
                "No open goals remaining; session completed. Added note for future reference.",
                80,
                true
            )
            .is_empty()
        );
        assert!(buffer.is_empty());

        let mut buffer = String::new();
        assert!(
            drain_harmony_analysis_memory(
                &mut buffer,
                "All goals complete? Maybe say nothing.",
                80,
                true
            )
            .is_empty()
        );
        assert!(buffer.is_empty());
    }

    #[test]
    fn go_harmony_analysis_memory_drops_abort_spiral() {
        let mut buffer = String::new();
        let text = "I will stop. This is impossible. No output. We need to stop. \
            I cannot continue. The conversation cannot be properly concluded.";
        assert!(drain_harmony_analysis_memory(&mut buffer, text, 80, true).is_empty());
        assert!(buffer.is_empty());
    }

    #[test]
    fn go_harmony_analysis_filter_drops_abort_spiral_but_preserves_action_thought() {
        let mut filter = HarmonyFinalFilter::for_analysis_prefill();
        let output = filter.filter_events(&[LlmEvent::Token {
            text: "We need to proceed listFiles page4. I will stop. This is impossible. No output."
                .to_string(),
        }]);
        assert_eq!(
            output.analysis,
            vec!["We need to proceed listFiles page 4."]
        );
        assert!(output.events.is_empty());
    }

    #[test]
    fn go_harmony_prompt_removes_plain_stream_pete_marker() {
        let body = initial_stream_prompt("seed", "startup", None, None);
        assert!(body.trim_end().ends_with("Pete:"));
        let prompt = render_go_prompt(GoPromptFormat::GptOssHarmony, &body);
        assert!(!prompt.contains("Pete: <|end|><|start|>assistant"));
        assert!(!prompt.contains("Pete:\n<|end|><|start|>assistant"));
        assert!(prompt.ends_with("<|end|><|start|>assistant"));
    }

    #[test]
    fn source_progress_gate_classifies_source_and_progress_actions() {
        let source = TypeScriptAction::ReadSourceFile {
            file: "src/main.rs".to_string(),
            page: 1,
            line: Some(4),
            page_size: None,
            workflow: SourceWorkflowUpdate::default(),
        };
        assert_eq!(
            source.source_progress_label().as_deref(),
            Some("readSourceFile src/main.rs line 4")
        );
        assert!(!source.records_progress_note());

        let note = TypeScriptAction::AddGoalNote {
            target: "goal-1".to_string(),
            text: "Found the runtime prompt wiring.".to_string(),
        };
        assert!(note.source_progress_label().is_none());
        assert!(note.records_progress_note());

        let mut fields = Map::new();
        fields.insert(
            "note".to_string(),
            Value::String("Selected the next concrete task.".to_string()),
        );
        let update = TypeScriptAction::UpdateWorkItem {
            target: "goal-1".to_string(),
            fields,
        };
        assert!(update.records_progress_note());
    }

    #[test]
    fn synthesis_gate_accepts_goal_summary_or_completion() {
        let mut fields = Map::new();
        fields.insert(
            "summary".to_string(),
            Value::String(
                "Neural acoustic models load ONNX sessions and produce acoustic frame tracks."
                    .to_string(),
            ),
        );
        let update = TypeScriptAction::UpdateWorkItem {
            target: "goal-13".to_string(),
            fields,
        };
        assert!(update.records_synthesis_update());

        let complete = TypeScriptAction::CompleteWorkItem {
            target: "goal-13".to_string(),
            note: Some(
                "Final understanding: ONNX acoustic generation maps text inputs into mel tracks."
                    .to_string(),
            ),
        };
        assert!(complete.records_synthesis_update());

        let mut placeholder_fields = Map::new();
        placeholder_fields.insert("summary".to_string(), Value::String("...".to_string()));
        let placeholder = TypeScriptAction::UpdateWorkItem {
            target: "goal-13".to_string(),
            fields: placeholder_fields,
        };
        assert!(!placeholder.records_synthesis_update());
        assert!(!meaningful_synthesis_text("text"));
    }

    #[test]
    fn graph_memory_gate_classifies_updates_and_cues() {
        let mut fields = Map::new();
        fields.insert(
            "description".to_string(),
            Value::String("person named Travis".to_string()),
        );
        let update = TypeScriptAction::UpdateGraphNodeFields {
            node_id: "person:travis".to_string(),
            label: Some("Travis".to_string()),
            fields,
        };
        assert!(matches!(
            update.graph_memory_update_label(),
            Some(label) if label.contains("person:travis")
        ));

        let search = TypeScriptAction::SearchGraphNodes {
            text: Some("Travis".to_string()),
            field: None,
            value: None,
            limit: Some(5),
        };
        assert!(search.graph_memory_update_label().is_none());
        assert!(graph_memory_text_has_cue(
            "Remember that I prefer concise replies."
        ));
        assert!(!graph_memory_text_has_cue("What time is it?"));
    }

    #[test]
    fn read_source_file_bare_number_means_page_not_line() {
        let actions = execute_typescript_actions(r#"readSourceFile("src/acoustic/model.rs", 2)"#)
            .expect("readSourceFile action should parse");

        assert_eq!(
            actions,
            vec![TypeScriptAction::ReadSourceFile {
                file: "src/acoustic/model.rs".to_string(),
                page: 2,
                line: None,
                page_size: None,
                workflow: SourceWorkflowUpdate::default(),
            }]
        );

        let actions =
            execute_typescript_actions(r#"readSourceFile("src/acoustic/model.rs", { line: 2 })"#)
                .expect("readSourceFile line action should parse");

        assert_eq!(
            actions,
            vec![TypeScriptAction::ReadSourceFile {
                file: "src/acoustic/model.rs".to_string(),
                page: 1,
                line: Some(2),
                page_size: None,
                workflow: SourceWorkflowUpdate::default(),
            }]
        );
    }

    #[test]
    fn countenance_actions_parse_from_typescript() {
        let actions = execute_typescript_actions(
            r#"setCountenance("🙂", { mood: "curious", reason: "greeting" })"#,
        )
        .expect("setCountenance action should parse");

        assert_eq!(
            actions,
            vec![TypeScriptAction::SetCountenance {
                emoji: "🙂".to_string(),
                mood: Some("curious".to_string()),
                reason: Some("greeting".to_string()),
            }]
        );

        let actions = execute_typescript_actions(r#"setMood("curious", { emoji: "🧐" })"#)
            .expect("setMood action should parse");
        assert_eq!(
            actions,
            vec![TypeScriptAction::SetCountenance {
                emoji: "🧐".to_string(),
                mood: Some("curious".to_string()),
                reason: None,
            }]
        );
    }

    #[test]
    fn countenance_helpers_use_last_emoji_sequence() {
        assert_eq!(
            last_emoji_sequence("Hello 🙂 then 🧐"),
            Some("🧐".to_string())
        );
        assert_eq!(
            normalize_countenance_emoji("curious 🧐"),
            Some("🧐".to_string())
        );
        assert_eq!(normalize_countenance_emoji("curious"), None);
    }

    #[test]
    fn source_actions_can_carry_workflow_notes() {
        let actions = execute_typescript_actions(
            r#"readSourceFile("src/acoustic/neural.rs", { page: 5, target: "goal-13", note: "Observed page 4: tensors and inference run; next inspect mel conversion.", summary: "Neural acoustic inference now clearly builds ONNX tensors and extracts mel outputs." })"#,
        )
        .expect("readSourceFile workflow action should parse");

        assert_eq!(
            actions,
            vec![TypeScriptAction::ReadSourceFile {
                file: "src/acoustic/neural.rs".to_string(),
                page: 5,
                line: None,
                page_size: None,
                workflow: SourceWorkflowUpdate {
                    target: Some("goal-13".to_string()),
                    note: Some(
                        "Observed page 4: tensors and inference run; next inspect mel conversion."
                            .to_string()
                    ),
                    summary: Some(
                        "Neural acoustic inference now clearly builds ONNX tensors and extracts mel outputs."
                            .to_string()
                    ),
                },
            }]
        );

        let actions = execute_typescript_actions(
            r#"grepSource("load_onnx", { limit: 3, note: "Searched load_onnx references; next inspect call sites." })"#,
        )
        .expect("grepSource workflow action should parse");

        assert_eq!(
            actions,
            vec![TypeScriptAction::GrepSource {
                pattern: "load_onnx".to_string(),
                limit: 3,
                workflow: SourceWorkflowUpdate {
                    target: None,
                    note: Some(
                        "Searched load_onnx references; next inspect call sites.".to_string()
                    ),
                    summary: None,
                },
            }]
        );
    }

    #[test]
    fn work_summary_warns_when_selected_goal_is_complete() {
        let board = WorkBoard {
            items: vec![
                Goal {
                    id: "goal-1".to_string(),
                    title: "Finished".to_string(),
                    summary: None,
                    parent: None,
                    priority: None,
                    tags: BTreeSet::new(),
                    steps: Vec::new(),
                    log: Vec::new(),
                    status: WorkItemStatus::Complete,
                },
                Goal {
                    id: "goal-2".to_string(),
                    title: "Still open".to_string(),
                    summary: None,
                    parent: None,
                    priority: None,
                    tags: BTreeSet::new(),
                    steps: Vec::new(),
                    log: Vec::new(),
                    status: WorkItemStatus::Open,
                },
            ],
            selected_id: Some("goal-1".to_string()),
            next_id: 3,
        };

        let summary = board.prompt_summary().expect("summary");
        assert!(summary.contains("Selected goal goal-1 is complete"));
        assert!(summary.contains("select an open goal"));
        assert_eq!(board.suggested_progress_target(), "goal-2");
    }

    #[test]
    fn graph_node_updates_get_vectorizable_descriptions() {
        let mut fields = Map::new();
        ensure_command_description_field("person:travis", Some("Travis"), &mut fields);
        assert_eq!(
            fields.get("description").and_then(Value::as_str),
            Some("person named Travis")
        );
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
    fn harmony_tool_call_errors_request_native_commentary_calls() {
        let observation = StreamObservation::HarmonyToolCallError {
            recipient: "functions.list_files".to_string(),
            arguments: "{\"page\":5}commentary<|channel|>analysis<|message|>Wait.".to_string(),
            error: "Harmony tool call failed: trailing characters".to_string(),
        };
        let prompt = observation.prompt_text();

        assert!(prompt.contains("Previous Harmony function call failed"));
        assert!(prompt.contains("native function call on the commentary channel"));
        assert!(prompt.contains("Recipient: functions.list_files"));
        assert!(!prompt.contains("corrected <ts>"));
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
    fn generated_text_cleaner_preserves_space_across_stripped_control_tags() {
        let mut cleaner = GeneratedTextCleaner::new();
        assert_eq!(
            cleaner.push("Then read page5.<|end|><|start|>streamWe need one action."),
            "Then read page 5. We need one action."
        );
    }

    #[test]
    fn generated_text_cleaner_preserves_space_after_split_stripped_control_tags() {
        let mut cleaner = GeneratedTextCleaner::new();
        assert_eq!(cleaner.push("Then read page5."), "Then read page 5.");
        assert_eq!(
            cleaner.push("<|end|><|start|>streamWe need one action."),
            " We need one action."
        );
    }

    #[test]
    fn generated_text_cleaner_preserves_space_between_page_and_split_number() {
        let mut cleaner = GeneratedTextCleaner::new();
        assert_eq!(cleaner.push("Then read page"), "Then read page");
        assert_eq!(cleaner.push("6."), " 6.");
    }

    #[test]
    fn compact_stream_prompt_for_budget_drops_oldest_events() {
        let mut events = VecDeque::new();
        events.push_back("old event ".repeat(400));
        events.push_back("middle event ".repeat(400));
        events.push_back("new event should stay".to_string());

        let empty_prompt = compact_stream_prompt("seed", "startup", &VecDeque::new(), None, None);
        let budget = estimate_tokens(&empty_prompt) + 32;
        let (prompt, retained) =
            compact_stream_prompt_for_budget("seed", "startup", &events, None, None, budget);

        assert!(estimate_tokens(&prompt) <= budget);
        assert!(retained < events.len());
        assert!(!prompt.contains("old event"));
        assert!(prompt.contains("new event should stay"));
    }

    #[test]
    fn compact_stream_prompt_uses_short_runtime_reminder() {
        let prompt = compact_stream_prompt("seed", "startup", &VecDeque::new(), None, None);
        assert!(prompt.contains(COMPACT_STREAM_RULES));
        assert!(!prompt.contains(PETE_WILL_RUNTIME_PROMPT));
        assert!(!prompt.contains(PETE_ORIENTATION_PROMPT));
    }

    #[test]
    fn stream_rag_query_is_capped_for_embedding_context() {
        let mut events = VecDeque::new();
        for index in 0..32 {
            events.push_back(format!("event {index}: {}", "x".repeat(1_000)));
        }
        let query = stream_rag_query(
            &"seed ".repeat(1_000),
            &"startup ".repeat(1_000),
            Some(&"work ".repeat(1_000)),
            &events,
        );
        assert!(query.len() <= GO_RAG_QUERY_MAX_CHARS + 3);
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
                reportBug("mouth queue stalled", { details: "TTS stayed queued after interruption.", context: "runtime test", severity: "medium" }),
                reportFeatureRequest("show VAD confidence", { details: "Useful for live tuning." }),
                reportIssue("export work board", { type: "feature", details: "Useful after long sessions." }),
                setTopic("debug loop"),
                startNewTopic("lab", { topic: "source", instruction: "Pete inspects source." }),
                topicChangedWhen("look at the source", { fromTopic: "lab", toTopic: "source" }),
                startNewEpisode("fresh go session", { topic: "go" }),
                extractEntities("My name is Travis."),
                mergeGraphNode("person:travis", { description: "test person" }, { label: "Travis" }),
                upsertGraphNode("project:listenbury", { description: "test project" }, { label: "Listenbury" }),
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
        assert!(actions.iter().any(|action| matches!(
            action,
            TypeScriptAction::ReportIssue {
                issue_type,
                title,
                ..
            } if issue_type == "bug" && title == "mouth queue stalled"
        )));
        assert!(actions.iter().any(|action| matches!(
            action,
            TypeScriptAction::ReportIssue {
                issue_type,
                title,
                ..
            } if issue_type == "feature_request" && title == "show VAD confidence"
        )));
    }

    #[test]
    fn issue_report_entry_formats_bug_and_feature_request_details() {
        let bug = format_issue_report_entry(
            "bug",
            "Mouth queue stalled",
            Some("Observed after interruption."),
            Some("go runtime"),
            Some("medium"),
        );
        assert!(bug.contains(" - Bug"));
        assert!(bug.contains("- Title: Mouth queue stalled"));
        assert!(bug.contains("- Severity: medium"));
        assert!(bug.contains("- Context: go runtime"));
        assert!(bug.contains("Observed after interruption."));

        let feature = format_issue_report_entry(
            "feature_request",
            "Add VAD confidence meter",
            None,
            None,
            None,
        );
        assert!(feature.contains(" - Feature request"));
        assert!(feature.contains("- Title: Add VAD confidence meter"));
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
