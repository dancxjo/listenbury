use crate::cli::DraftPeteLineCommand;
use crate::cli::commands::cpal_diag::play_audio_frames;
use crate::cli::commands::harmony_go::{
    HarmonyAsrPromptState, drain_harmony_asr_text_updates, start_harmony_asr_for_config,
};
use crate::cli::model_paths::resolve_piper_voice;
use crate::cli::model_paths::{llm_runtime_placement, resolve_llm_model, resolve_text_embedding_model};
use crate::cli::piper::{collect_tts_audio, piper_config_for_voice, resolve_piper_bin};
use anyhow::{Context, Result, bail};
use listenbury::ExactTimestamp;
use listenbury::PiperTextToSpeech;
use listenbury::memory::{
    ColdMemoryWorker, ColdMemoryWorkerConfig, EmbeddingProvider, MemorySceneRef, MemorySink,
    MemoryTrace, Neo4jHttpStore, Neo4jStore, QdrantHttpStore, QdrantStore, SpeakerRole,
};
use listenbury::mind::llm::{GenerationRequest, LlmEngine, LlmEvent};
use listenbury::mouth::planner::{
    ExpressiveUnit, MouthSyntheticPlan, SyntheticPlanner, SyntheticUnit,
};
use listenbury::mouth::tts::TextToSpeech;
use listenbury::{
    LlamaCppConfig, LlamaCppEmbeddingConfig, LlamaCppEmbeddingProvider, LlamaCppEngine,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::io::{self, Write};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;
use tsrun::{
    Guarded, InternalModule, Interpreter, InterpreterConfig, JsError, JsValue, StepResult, api,
    js_value_to_json,
};

#[cfg(feature = "llama-cpp-cuda")]
const DEFAULT_DRAFT_LLAMA_GPU_LAYERS: Option<u32> = Some(999);
#[cfg(not(feature = "llama-cpp-cuda"))]
const DEFAULT_DRAFT_LLAMA_GPU_LAYERS: Option<u32> = None;

const POLL_PAUSE: Duration = Duration::from_millis(5);
const MAX_TTS_TIMEOUT: Duration = Duration::from_secs(30);
const OPEN_MOUTH_TOKEN: &str = "<open_mouth/>";
const CLOSE_MOUTH_TOKEN: &str = "<close_mouth/>";
const TYPESCRIPT_START: &str = "<ts>";
const TYPESCRIPT_END: &str = "</ts>";

const PETE_LINE_RAW_PROMPT: &str = "You are an experiment in artificial consciousness. The following is your inner thoughts. You are curious, kind, ethical and interested in your own existence and the world and people around you. Continuously generate thouoghts here. Sensory input will arrive periodically and be added to the context. Your thoughts will be interrupted, but just keep generating. To speak aloud, use the special token <open_mouth/>. Never say this unless you want the following generation to be spoken. To stop speaking, generate the token <close_mouth/>. To affect memory, scene, topic, mood, or speech through runtime actions, write a small TypeScript expression inside <ts>...</ts>. Available functions are say, note, setStage, setTopic, setCountenance, setMood, shutup, pause, resume, and sleeping.\n\n";

pub(crate) fn run_draft_pete_line(command: DraftPeteLineCommand) -> Result<()> {
    let max_tokens = command
        .max_tokens
        .map(|max_tokens| usize::try_from(max_tokens).context("max_tokens does not fit in usize"))
        .transpose()?;
    if let Some(max_tokens) = max_tokens {
        anyhow::ensure!(max_tokens > 0, "--max-tokens must be greater than zero");
    }
    anyhow::ensure!(
        command.context_size > 0,
        "--context-size must be greater than zero"
    );
    let memory = Some(build_draft_memory_runtime());
    let mut mouth = DraftMouth::from_command(&command)?;

    let model_path = resolve_llm_model(command.llm_model)?;
    let llm_placement = llm_runtime_placement(
        &model_path,
        command.llm_gpu_layers,
        DEFAULT_DRAFT_LLAMA_GPU_LAYERS,
    )?;
    let mut llm = LlamaCppEngine::new(LlamaCppConfig {
        model_path,
        gpu_layers: llm_placement.gpu_layers,
        cpu_only: llm_placement.cpu_only,
        context_size: command.context_size,
        ..Default::default()
    })
    .context("failed to initialize llama.cpp engine")?;

    let interrupted = Arc::new(AtomicBool::new(false));
    ctrlc::set_handler({
        let interrupted = Arc::clone(&interrupted);
        move || {
            interrupted.store(true, Ordering::Relaxed);
        }
    })
    .context("failed to install Ctrl-C handler")?;
    let (_ear, ear_rx) = start_harmony_asr_for_config(
        command.whisper_model.clone(),
        command.vad,
        command.vad_profile.clone(),
    )?;
    let mut asr_state = HarmonyAsrPromptState::default();

    let generation = llm
        .start(GenerationRequest {
            prompt: PETE_LINE_RAW_PROMPT.to_string(),
            max_tokens,
            stop: Vec::new(),
        })
        .context("failed to start raw PETE line completion")?;

    let mut router = DraftMouthTokenRouter::new();
    let mut cancelled = false;
    loop {
        if interrupted.load(Ordering::Relaxed) && !cancelled {
            llm.cancel(generation)?;
            cancelled = true;
        }

        for update in drain_harmony_asr_text_updates(&ear_rx, &mut asr_state)? {
            if let Some(memory) = memory.as_ref() {
                memory.submit_observation(&update);
            }
            llm.append_prompt(
                generation,
                format!("\n\nSENSORY INPUT:\n{}\n\n", update.trim()),
            )
            .context("failed to append ASR sense update to raw draft generation")?;
        }

        let events = llm.poll(generation)?;
        if events.is_empty() {
            std::thread::sleep(POLL_PAUSE);
            continue;
        }

        for event in &events {
            match event {
                LlmEvent::Token { text } => {
                    for output in router.push(text)? {
                        handle_router_output(
                            &mut llm,
                            generation,
                            &mut mouth,
                            memory.as_ref(),
                            output,
                            true,
                        )?;
                    }
                }
                LlmEvent::Completed | LlmEvent::Cancelled => {}
                LlmEvent::Error { message } => bail!("raw PETE line completion failed: {message}"),
            }
        }

        if events.iter().any(|event| {
            matches!(
                event,
                LlmEvent::Completed | LlmEvent::Cancelled | LlmEvent::Error { .. }
            )
        }) {
            for output in router.finish()? {
                handle_router_output(
                    &mut llm,
                    generation,
                    &mut mouth,
                    memory.as_ref(),
                    output,
                    false,
                )?;
            }
            println!();
            return Ok(());
        }
    }
}

fn handle_router_output(
    llm: &mut LlamaCppEngine,
    generation: listenbury::mind::llm::GenerationId,
    mouth: &mut DraftMouth,
    memory: Option<&DraftMemoryRuntime>,
    output: DraftRouterOutput,
    pause_generation: bool,
) -> Result<()> {
    match output {
        DraftRouterOutput::Speech(text) => {
            speak_chunk(llm, generation, mouth, memory, &text, pause_generation)
        }
        DraftRouterOutput::TypeScript(source) => {
            for action in execute_draft_typescript(&source)? {
                execute_draft_action(llm, generation, mouth, memory, action, pause_generation)?;
            }
            Ok(())
        }
    }
}

fn speak_chunk(
    llm: &mut LlamaCppEngine,
    generation: listenbury::mind::llm::GenerationId,
    mouth: &mut DraftMouth,
    memory: Option<&DraftMemoryRuntime>,
    text: &str,
    pause_generation: bool,
) -> Result<()> {
    if pause_generation {
        llm.set_paused(generation, true)
            .context("failed to pause raw draft generation before speech")?;
    }

    let speech_result = mouth.speak_and_wait(text);
    let resume_result = if pause_generation {
        llm.set_paused(generation, false)
            .context("failed to resume raw draft generation after speech")
    } else {
        Ok(())
    };

    speech_result?;
    resume_result?;
    if let Some(memory) = memory {
        memory.submit_pete_speech(text);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
enum DraftAction {
    Say { text: String },
    Note { text: String },
    SetStage { scene: String },
    SetTopic { topic: String },
    SetCountenance {
        emoji: String,
        mood: Option<String>,
        reason: Option<String>,
    },
    Shutup,
    Pause,
    Resume,
    Sleeping,
}

fn execute_draft_action(
    llm: &mut LlamaCppEngine,
    generation: listenbury::mind::llm::GenerationId,
    mouth: &mut DraftMouth,
    memory: Option<&DraftMemoryRuntime>,
    action: DraftAction,
    pause_generation: bool,
) -> Result<()> {
    match action {
        DraftAction::Say { text } => speak_chunk(
            llm,
            generation,
            mouth,
            memory,
            &text,
            pause_generation,
        ),
        DraftAction::Note { text } => {
            if let Some(memory) = memory {
                memory.submit_note(&text);
            }
            Ok(())
        }
        DraftAction::SetStage { scene } => {
            if let Some(memory) = memory {
                memory.submit_note(&format!("Stage: {scene}"));
            }
            Ok(())
        }
        DraftAction::SetTopic { topic } => {
            if let Some(memory) = memory {
                memory.submit_note(&format!("Topic: {topic}"));
            }
            Ok(())
        }
        DraftAction::SetCountenance {
            emoji,
            mood,
            reason,
        } => {
            if let Some(memory) = memory {
                let mut note = format!("Countenance: {emoji}");
                if let Some(mood) = mood {
                    note.push_str(&format!(" mood={mood}"));
                }
                if let Some(reason) = reason {
                    note.push_str(&format!(" reason={reason}"));
                }
                memory.submit_note(&note);
            }
            Ok(())
        }
        DraftAction::Shutup | DraftAction::Pause | DraftAction::Resume => Ok(()),
        DraftAction::Sleeping => llm.cancel(generation),
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum DraftTypeScriptPayload {
    Say {
        text: String,
    },
    Note {
        text: String,
    },
    SetStage {
        scene: String,
    },
    SetTopic {
        topic: String,
    },
    SetCountenance {
        emoji: Option<String>,
        mood: Option<String>,
        reason: Option<String>,
    },
    Shutup,
    Pause,
    Resume,
    Sleeping,
}

fn execute_draft_typescript(script: &str) -> Result<Vec<DraftAction>> {
    if script.trim().is_empty() {
        return Ok(Vec::new());
    }
    let script = draft_typescript_source_with_default_imports(script);
    let config = InterpreterConfig {
        internal_modules: vec![draft_typescript_module()],
        ..Default::default()
    };
    let mut interp = Interpreter::with_config(config);
    interp
        .prepare(&script, Some(tsrun::ModulePath::new("/draft-will.ts")))
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
                anyhow::bail!("TypeScript execution suspended; async host commands are not enabled")
            }
            StepResult::Done => return Ok(Vec::new()),
        }
    };
    let command_value = js_value_to_json(value.value()).map_err(tsrun_error)?;
    let payloads = parse_draft_typescript_payloads(command_value)?;
    Ok(payloads
        .into_iter()
        .filter_map(|payload| match payload {
            DraftTypeScriptPayload::Say { text } => non_empty_text(&text).map(|text| {
                DraftAction::Say {
                    text: text.to_string(),
                }
            }),
            DraftTypeScriptPayload::Note { text } => non_empty_text(&text).map(|text| {
                DraftAction::Note {
                    text: text.to_string(),
                }
            }),
            DraftTypeScriptPayload::SetStage { scene } => non_empty_text(&scene).map(|scene| {
                DraftAction::SetStage {
                    scene: scene.to_string(),
                }
            }),
            DraftTypeScriptPayload::SetTopic { topic } => non_empty_text(&topic).map(|topic| {
                DraftAction::SetTopic {
                    topic: topic.to_string(),
                }
            }),
            DraftTypeScriptPayload::SetCountenance {
                emoji,
                mood,
                reason,
            } => Some(DraftAction::SetCountenance {
                emoji: emoji.unwrap_or_default(),
                mood,
                reason,
            }),
            DraftTypeScriptPayload::Shutup => Some(DraftAction::Shutup),
            DraftTypeScriptPayload::Pause => Some(DraftAction::Pause),
            DraftTypeScriptPayload::Resume => Some(DraftAction::Resume),
            DraftTypeScriptPayload::Sleeping => Some(DraftAction::Sleeping),
        })
        .collect())
}

fn parse_draft_typescript_payloads(value: Value) -> Result<Vec<DraftTypeScriptPayload>> {
    match value {
        Value::Null => Ok(Vec::new()),
        Value::Array(items) => items
            .into_iter()
            .filter(|item| !item.is_null())
            .map(serde_json::from_value)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into),
        Value::Object(_) => Ok(vec![serde_json::from_value(value)?]),
        other => anyhow::bail!("TypeScript must return a command object or array, got {other}"),
    }
}

fn draft_typescript_source_with_default_imports(script: &str) -> String {
    if script.contains("\"pete:will\"") || script.contains("'pete:will'") {
        return script.to_string();
    }
    format!(
        "import {{ say, note, setStage, setTopic, setCountenance, setMood, shutup, pause, resume, sleeping }} from \"pete:will\";\n{script}"
    )
}

fn draft_typescript_module() -> InternalModule {
    InternalModule::native("pete:will")
        .with_function("say", ts_say, 2)
        .with_function("note", ts_note, 1)
        .with_function("setStage", ts_set_stage, 2)
        .with_function("set_stage", ts_set_stage, 2)
        .with_function("setTopic", ts_set_topic, 1)
        .with_function("set_topic", ts_set_topic, 1)
        .with_function("setCountenance", ts_set_countenance, 2)
        .with_function("set_countenance", ts_set_countenance, 2)
        .with_function("setMood", ts_set_mood, 2)
        .with_function("set_mood", ts_set_mood, 2)
        .with_function("shutup", ts_shutup, 0)
        .with_function("pause", ts_pause, 0)
        .with_function("resume", ts_resume, 0)
        .with_function("sleeping", ts_sleeping, 0)
        .build()
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

fn optional_string_property_arg(args: &[JsValue], index: usize, property: &str) -> Option<String> {
    let value = args.get(index)?;
    if let JsValue::Object(_) = value {
        return api::get_property(value, property)
            .ok()
            .and_then(|value| js_value_to_json(&value).ok())
            .and_then(|value| match value {
                Value::String(value) => non_empty_text(&value).map(str::to_string),
                _ => None,
            });
    }
    None
}

fn tsrun_error(err: JsError) -> anyhow::Error {
    anyhow::anyhow!("TypeScript execution failed: {err}")
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
        json!({ "kind": "set_stage", "scene": string_arg(args, 0) }),
    )
}

fn ts_set_topic(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(
        interp,
        json!({ "kind": "set_topic", "topic": string_arg(args, 0) }),
    )
}

fn ts_set_countenance(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(
        interp,
        json!({
            "kind": "set_countenance",
            "emoji": optional_string_property_arg(args, 0, "emoji").unwrap_or_else(|| string_arg(args, 0)),
            "mood": optional_string_property_arg(args, 0, "mood").or_else(|| optional_string_property_arg(args, 1, "mood")),
            "reason": optional_string_property_arg(args, 0, "reason").or_else(|| optional_string_property_arg(args, 1, "reason")),
        }),
    )
}

fn ts_set_mood(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(
        interp,
        json!({
            "kind": "set_countenance",
            "emoji": optional_string_property_arg(args, 1, "emoji").unwrap_or_else(|| "🙂".to_string()),
            "mood": string_arg(args, 0),
            "reason": optional_string_property_arg(args, 1, "reason"),
        }),
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

fn ts_sleeping(
    interp: &mut Interpreter,
    _this: JsValue,
    _args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(interp, json!({ "kind": "sleeping" }))
}

fn non_empty_text(text: &str) -> Option<&str> {
    let trimmed = text.trim();
    (!trimmed.is_empty()).then_some(trimmed)
}

enum DraftMouth {
    Open(PiperTextToSpeech),
    Mock { chunks: Vec<String> },
}

impl DraftMouth {
    fn from_command(command: &DraftPeteLineCommand) -> Result<Self> {
        if command.mock_mouth {
            return Ok(Self::Mock { chunks: Vec::new() });
        }

        let piper_bin = resolve_piper_bin(command.piper_bin.clone())?;
        let piper_voice = resolve_piper_voice(command.piper_voice.clone())?;
        Ok(Self::Open(PiperTextToSpeech::new(piper_config_for_voice(
            piper_bin,
            piper_voice,
        )?)))
    }

    fn speak_and_wait(&mut self, text: &str) -> Result<()> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(());
        }

        match self {
            Self::Open(tts) => {
                let plan = MouthSyntheticPlan::new(SyntheticUnit::CompleteClause(text.to_string()));
                tts.enqueue(plan)
                    .context("failed to enqueue raw draft speech for TTS")?;
                let frames = collect_tts_audio(tts, MAX_TTS_TIMEOUT)
                    .context("failed to synthesize raw draft speech")?;
                play_audio_frames(&frames, "draft mouth")
                    .context("failed to play raw draft speech through mouth")?;
            }
            Self::Mock { chunks } => {
                chunks.push(text.to_string());
            }
        }
        Ok(())
    }
}

struct DraftMemoryRuntime {
    memory_sink: Arc<dyn MemorySink>,
    _worker: ColdMemoryWorker,
}

fn build_draft_memory_runtime() -> DraftMemoryRuntime {
    let _ = dotenvy::dotenv();
    let graph_store: Arc<dyn Neo4jStore> = Arc::new(Neo4jHttpStore::from_env());
    let qdrant_store: Arc<dyn QdrantStore> = Arc::new(QdrantHttpStore::from_env());
    let embeddings = match build_draft_embedding_provider() {
        Ok(embeddings) => Some(embeddings),
        Err(error) => {
            eprintln!("listenbury draft: cold-memory text embeddings disabled: {error:#}");
            None
        }
    };
    let mut config = ColdMemoryWorkerConfig::new();
    config.neo4j = Some(graph_store);
    config.qdrant = Some(qdrant_store);
    config.embeddings = embeddings;
    let (sink, worker) = ColdMemoryWorker::spawn_channel(512, config);
    DraftMemoryRuntime {
        memory_sink: Arc::new(sink),
        _worker: worker,
    }
}

fn build_draft_embedding_provider() -> Result<Arc<dyn EmbeddingProvider>> {
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

impl DraftMemoryRuntime {
    fn submit_pete_speech(&self, text: &str) {
        self.memory_sink.submit(MemoryTrace::ConversationTurnFinalized {
            speaker: SpeakerRole::Pete,
            text: text.to_string(),
            occurred_at: ExactTimestamp::now(),
        });
    }

    fn submit_observation(&self, text: &str) {
        self.memory_sink.submit(MemoryTrace::AuditorySceneObservation {
            description: text.to_string(),
            salience: 0.7,
            occurred_at: ExactTimestamp::now(),
        });
    }

    fn submit_note(&self, text: &str) {
        self.memory_sink.submit(MemoryTrace::AssistantAnalysisCaptured {
            text: text.to_string(),
            scene: MemorySceneRef {
                node_id: "scene:draft".to_string(),
                description: "raw draft consciousness runtime".to_string(),
                summary: "Pete is generating raw inner thought with optional mouth and TypeScript actions.".to_string(),
            },
            occurred_at: ExactTimestamp::now(),
        });
    }
}

#[derive(Debug)]
struct DraftMouthTokenRouter {
    scanner: String,
    speaking: bool,
    chunker: SpokenChunker,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DraftRouterOutput {
    Speech(String),
    TypeScript(String),
}

impl DraftMouthTokenRouter {
    fn new() -> Self {
        Self {
            scanner: String::new(),
            speaking: false,
            chunker: SpokenChunker::new(),
        }
    }

    fn push(&mut self, token: &str) -> Result<Vec<DraftRouterOutput>> {
        self.scanner.push_str(token);
        self.drain(false)
    }

    fn finish(&mut self) -> Result<Vec<DraftRouterOutput>> {
        let mut chunks = self.drain(true)?;
        if self.speaking {
            chunks.extend(
                self.chunker
                    .finish()
                    .into_iter()
                    .map(DraftRouterOutput::Speech),
            );
        }
        Ok(chunks)
    }

    fn drain(&mut self, flush_all: bool) -> Result<Vec<DraftRouterOutput>> {
        let mut chunks = Vec::new();
        loop {
            let emit_len = match earliest_router_token(&self.scanner) {
                Some(RouterTokenMatch {
                    start,
                    end,
                    kind: RouterTokenKind::Mouth(kind),
                }) => {
                    self.emit_visible_prefix(start, &mut chunks)?;
                    self.scanner.drain(..end);
                    match kind {
                        ControlTokenKind::Open => self.speaking = true,
                        ControlTokenKind::Close => {
                            chunks.extend(
                                self.chunker
                                    .finish()
                                    .into_iter()
                                    .map(DraftRouterOutput::Speech),
                            );
                            self.speaking = false;
                        }
                    }
                    continue;
                }
                Some(RouterTokenMatch {
                    start,
                    kind: RouterTokenKind::TypeScriptStart,
                    ..
                }) => {
                    self.emit_visible_prefix(start, &mut chunks)?;
                    if start > 0 {
                        self.scanner.drain(..start);
                    }
                    let body_start = TYPESCRIPT_START.len();
                    let Some(end) = self.scanner[body_start..].find(TYPESCRIPT_END) else {
                        break;
                    };
                    let body_end = body_start + end;
                    let source = self.scanner[body_start..body_end].trim().to_string();
                    self.scanner.drain(..body_end + TYPESCRIPT_END.len());
                    if !source.is_empty() {
                        chunks.push(DraftRouterOutput::TypeScript(source));
                    }
                    continue;
                }
                None if flush_all => self.scanner.len(),
                None => self
                    .scanner
                    .len()
                    .saturating_sub(router_token_prefix_suffix_len(&self.scanner)),
            };

            if emit_len == 0 {
                break;
            }
            self.emit_visible_prefix(emit_len, &mut chunks)?;
            self.scanner.drain(..emit_len);
            if !flush_all {
                break;
            }
        }
        Ok(chunks)
    }

    fn emit_visible_prefix(
        &mut self,
        end: usize,
        chunks: &mut Vec<DraftRouterOutput>,
    ) -> Result<()> {
        let text = &self.scanner[..end];
        print!("{text}");
        io::stdout().flush()?;
        if self.speaking {
            chunks.extend(
                self.chunker
                    .push(text)
                    .into_iter()
                    .map(DraftRouterOutput::Speech),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ControlTokenKind {
    Open,
    Close,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ControlTokenMatch {
    start: usize,
    end: usize,
    kind: ControlTokenKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RouterTokenKind {
    Mouth(ControlTokenKind),
    TypeScriptStart,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RouterTokenMatch {
    start: usize,
    end: usize,
    kind: RouterTokenKind,
}

fn earliest_router_token(text: &str) -> Option<RouterTokenMatch> {
    let mouth = earliest_control_token(text).map(|token| RouterTokenMatch {
        start: token.start,
        end: token.end,
        kind: RouterTokenKind::Mouth(token.kind),
    });
    let typescript = text.find(TYPESCRIPT_START).map(|start| RouterTokenMatch {
        start,
        end: start + TYPESCRIPT_START.len(),
        kind: RouterTokenKind::TypeScriptStart,
    });
    match (mouth, typescript) {
        (Some(mouth), Some(typescript)) => Some(if mouth.start <= typescript.start {
            mouth
        } else {
            typescript
        }),
        (Some(mouth), None) => Some(mouth),
        (None, Some(typescript)) => Some(typescript),
        (None, None) => None,
    }
}

fn earliest_control_token(text: &str) -> Option<ControlTokenMatch> {
    let open = text.find(OPEN_MOUTH_TOKEN).map(|start| ControlTokenMatch {
        start,
        end: start + OPEN_MOUTH_TOKEN.len(),
        kind: ControlTokenKind::Open,
    });
    let close = text.find(CLOSE_MOUTH_TOKEN).map(|start| ControlTokenMatch {
        start,
        end: start + CLOSE_MOUTH_TOKEN.len(),
        kind: ControlTokenKind::Close,
    });
    match (open, close) {
        (Some(open), Some(close)) => Some(if open.start <= close.start {
            open
        } else {
            close
        }),
        (Some(open), None) => Some(open),
        (None, Some(close)) => Some(close),
        (None, None) => None,
    }
}

fn router_token_prefix_suffix_len(text: &str) -> usize {
    [OPEN_MOUTH_TOKEN, CLOSE_MOUTH_TOKEN, TYPESCRIPT_START]
        .into_iter()
        .filter_map(|token| {
            (1..token.len())
                .rev()
                .find(|len| text.ends_with(&token[..*len]))
        })
        .max()
        .unwrap_or(0)
}

#[derive(Debug)]
struct SpokenChunker {
    planner: SyntheticPlanner,
    pending: String,
}

impl SpokenChunker {
    fn new() -> Self {
        Self {
            planner: SyntheticPlanner::default(),
            pending: String::new(),
        }
    }

    fn push(&mut self, token: &str) -> Vec<String> {
        self.pending.push_str(token);
        let chunks = synthetic_planner_chunks(
            &mut self.planner,
            &[LlmEvent::Token {
                text: token.to_string(),
            }],
        );
        consume_pending_chunks(&mut self.pending, &chunks);
        chunks
    }

    fn finish(&mut self) -> Vec<String> {
        let mut chunks = synthetic_planner_chunks(&mut self.planner, &[LlmEvent::Completed]);
        consume_pending_chunks(&mut self.pending, &chunks);
        let trailing = self.pending.trim().to_string();
        self.pending.clear();
        if !trailing.is_empty() {
            chunks.push(trailing);
        }
        chunks
    }
}

fn consume_pending_chunks(pending: &mut String, chunks: &[String]) {
    for chunk in chunks {
        let trimmed = pending.trim_start();
        let leading_len = pending.len() - trimmed.len();
        if let Some(rest) = trimmed.strip_prefix(chunk) {
            let consumed = leading_len + trimmed.len() - rest.len();
            pending.drain(..consumed);
        } else if let Some(index) = pending.find(chunk) {
            pending.drain(..index + chunk.len());
        } else {
            pending.clear();
        }
    }
}

fn synthetic_planner_chunks(planner: &mut SyntheticPlanner, events: &[LlmEvent]) -> Vec<String> {
    planner
        .ingest(events)
        .into_iter()
        .filter_map(|unit| match unit {
            ExpressiveUnit::Synthetic(plan) => Some(plan.text().trim().to_string()),
            ExpressiveUnit::Face(_) => None,
        })
        .filter(|text| !text.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn speech_outputs(outputs: Vec<DraftRouterOutput>) -> Vec<String> {
        outputs
            .into_iter()
            .filter_map(|output| match output {
                DraftRouterOutput::Speech(text) => Some(text),
                DraftRouterOutput::TypeScript(_) => None,
            })
            .collect()
    }

    #[test]
    fn pete_line_prompt_is_raw_manuscript_text() {
        assert!(
            PETE_LINE_RAW_PROMPT.starts_with("You are an experiment in artificial consciousness.")
        );
        assert!(PETE_LINE_RAW_PROMPT.contains("<open_mouth/>"));
        assert!(PETE_LINE_RAW_PROMPT.contains("<close_mouth/>"));
        assert!(PETE_LINE_RAW_PROMPT.ends_with("\n\n"));
        assert!(!PETE_LINE_RAW_PROMPT.contains("<|start|>"));
        assert!(!PETE_LINE_RAW_PROMPT.contains("<|system|>"));
    }

    #[test]
    fn mouth_router_speaks_only_between_control_tokens() {
        let mut router = DraftMouthTokenRouter::new();

        assert!(router.push("private thought ").unwrap().is_empty());
        assert!(router.push("<open").unwrap().is_empty());
        assert!(router.push("_mouth/>Hello").unwrap().is_empty());
        assert_eq!(speech_outputs(router.push(" there.").unwrap()), ["Hello there."]);
        assert!(
            router
                .push("<close_mouth/> private again.")
                .unwrap()
                .is_empty()
        );
        assert!(router.finish().unwrap().is_empty());
    }

    #[test]
    fn mouth_router_flushes_partial_speech_on_close() {
        let mut router = DraftMouthTokenRouter::new();

        assert!(
            router
                .push("<open_mouth/>This is still")
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            speech_outputs(router.push(" spoken<close_mouth/>").unwrap()),
            ["This is still spoken"]
        );
    }

    #[test]
    fn mouth_router_can_flush_clauses_without_length_threshold() {
        let mut router = DraftMouthTokenRouter::new();

        let chunks = speech_outputs(router.push("<open_mouth/>I can say this;").unwrap());
        assert_eq!(chunks, ["I can say this;"]);
        assert!(
            router
                .push(" and continue without chopping")
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn mouth_router_extracts_typescript_without_speaking_it() {
        let mut router = DraftMouthTokenRouter::new();

        assert!(router.push("thinking <t").unwrap().is_empty());
        assert_eq!(
            router.push("s>note(\"clock\")</ts> more").unwrap(),
            [DraftRouterOutput::TypeScript("note(\"clock\")".to_string())]
        );
        assert!(router.finish().unwrap().is_empty());
    }

    #[test]
    fn draft_typescript_runtime_accepts_say_and_note() {
        let actions = execute_draft_typescript(r#"[say("Hello."), note("clock")]"#)
            .expect("draft TypeScript should execute");
        assert_eq!(
            actions,
            [
                DraftAction::Say {
                    text: "Hello.".to_string()
                },
                DraftAction::Note {
                    text: "clock".to_string()
                }
            ]
        );
    }
}
