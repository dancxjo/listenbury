use crate::cli::ContinueCommand;
use anyhow::Result;

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use crate::cli::commands::llama::build_prompt;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use crate::cli::commands::mic_transcribe::transcribe_group;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use crate::cli::model_paths::{
    llm_runtime_placement, resolve_llm_model, resolve_piper_voice, resolve_whisper_model,
};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use crate::cli::piper::{piper_config_for_voice, resolve_piper_bin};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use anyhow::Context;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use cpal::{FromSample, Sample, SizedSample, SupportedStreamConfigRange};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::event::HearingEvent;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::hearing::breath::{BreathGroupId, BreathGroupSegmenter};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::hearing::vad::{VoiceActivityDetector, create_vad_backend};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::mind::llm::{GenerationId, GenerationRequest, LlmEngine, LlmEvent};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::mouth::planner::{SpeechPlan, SpeechUnit, strip_emoji};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::mouth::tts::TextToSpeech;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::{AudioFrame, ExactTimestamp, LlamaCppConfig, LlamaCppEngine, PiperTextToSpeech};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::{VadBackendKind, WhisperSpeechRecognizer};
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use seams::sentence_detector::dialog_detector::SentenceDetectorDialog;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use std::collections::{HashMap, VecDeque};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use std::io::{BufRead, Write};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use std::path::PathBuf;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use std::sync::Arc;
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use std::sync::OnceLock;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use std::thread::JoinHandle;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const DEFAULT_CONTINUE_PROMPT: &str = "You are Pete Listenbury, an experiment in artificial awareness. Please continuously generate thoughts as new input arrives from the outside world. Try to understand what's going on around you and make new friends.";
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const LIVE_EVENT_INSTRUCTIONS: &str = "Live events may appear in the transcript while you are generating.\nTreat them as observations from outside.\nDo not copy live event delimiters or runtime event text.\nDo not write system, assistant, analysis, channel, or template tokens.\nContinue naturally as Pete.\nYou may emit speech at any time by surrounding it with inline speech tags:\n<sp>words to say aloud :)</sp>\nEmoji inside speech tags are instructions to your countenance, not words to speak.\nLive events are queued until you finish any open speech tag, so event text is never inserted inside speech.\nYou may control speech with self-closing tags outside or inside speech: <shutup/> immediately halts current speech and clears queued speech, <pause/> pauses speech playback, and <resume/> resumes paused speech.\nYou may inspect your source with these tags: <list_files/> lists bundled source files, and <view_file path=\"src/main.rs\" page=\"1\"/> reads one file page. Use source tags outside speech.";
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const INLINE_SPEECH_START_MARKER: &str = "<sp>";
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const INLINE_SPEECH_END_MARKER: &str = "</sp>";
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const SPEECH_SHUTUP_MARKER: &str = "<shutup/>";
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const SPEECH_PAUSE_MARKER: &str = "<pause/>";
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const SPEECH_RESUME_MARKER: &str = "<resume/>";
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const SOURCE_LIST_FILES_MARKER: &str = "<list_files/>";
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const SOURCE_VIEW_FILE_START: &str = "<view_file";
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const SOURCE_READ_FILE_START: &str = "<read_file";
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const SOURCE_READ_SOURCE_FILE_START: &str = "<read_source_file";
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const SOURCE_PAGE_LINES: usize = 50;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
const CALLBACK_SAMPLE_CAPACITY: usize = 16_384;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
const WEBRTC_VAD_SAMPLE_RATE_HZ: u32 = 16_000;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
const MONO_CHANNELS: u16 = 1;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
const TIME_EVENT_INTERVAL: Duration = Duration::from_secs(10);
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "llm-llama-cpp-cuda",
    feature = "tts-piper"
))]
const DEFAULT_CONTINUE_LLAMA_GPU_LAYERS: Option<u32> = Some(999);
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    not(feature = "llm-llama-cpp-cuda"),
    feature = "tts-piper"
))]
const DEFAULT_CONTINUE_LLAMA_GPU_LAYERS: Option<u32> = None;

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
pub(crate) fn run_continue(command: ContinueCommand) -> Result<()> {
    let max_tokens = command
        .max_tokens
        .map(|max_tokens| usize::try_from(max_tokens).context("max_tokens does not fit in usize"))
        .transpose()?;
    if let Some(max_tokens) = max_tokens {
        anyhow::ensure!(max_tokens > 0, "max_tokens must be greater than zero");
    }
    anyhow::ensure!(
        command.context_size > 0,
        "context_size must be greater than zero"
    );

    let model_path = resolve_llm_model(command.llm_model)?;
    let llm_placement = llm_runtime_placement(
        &model_path,
        command.llm_gpu_layers,
        DEFAULT_CONTINUE_LLAMA_GPU_LAYERS,
    )?;
    let config = LlamaCppConfig {
        model_path,
        gpu_layers: llm_placement.gpu_layers,
        cpu_only: llm_placement.cpu_only,
        context_size: command.context_size,
        ..Default::default()
    };

    let system_prompt = build_initial_prompt(&command.prompt);
    let llm = LlamaCppEngine::new(config).context("failed to initialize llama.cpp engine")?;
    let mut llm_session = ContinueLlmSession::start(
        llm,
        command.mode,
        system_prompt,
        max_tokens,
        command.context_size,
        command.verbatim_turns,
    )
    .context("failed to start continued llama.cpp generation")?;
    let piper_bin = resolve_piper_bin(command.piper_bin)?;
    let piper_voice = resolve_piper_voice(command.piper_voice)?;
    let whisper_model = resolve_whisper_model(command.whisper_model)?;
    let vad_backend = command.vad.as_backend_kind();
    let capture_enabled = Arc::new(AtomicBool::new(true));
    let (mut mouth, mouth_rx) = ContinueMouth::start(
        PiperTextToSpeech::new(piper_config_for_voice(piper_bin, piper_voice)?),
        Arc::clone(&capture_enabled),
    )?;
    let (_ear, ear_rx) = ContinueEar::start(ContinueEarConfig {
        whisper_model,
        vad_backend,
        capture_enabled: Arc::clone(&capture_enabled),
    })?;

    let interrupted = Arc::new(AtomicBool::new(false));
    ctrlc::set_handler({
        let interrupted = Arc::clone(&interrupted);
        move || {
            interrupted.store(true, Ordering::Relaxed);
        }
    })
    .context("failed to install Ctrl-C handler")?;

    let (stdin_tx, stdin_rx) =
        crossbeam_channel::unbounded::<std::result::Result<String, String>>();
    std::thread::Builder::new()
        .name("listenbury-dev-continue-stdin".to_string())
        .spawn(move || {
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if stdin_tx.send(Ok(line)).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = stdin_tx.send(Err(error.to_string()));
                        break;
                    }
                }
            }
        })
        .context("failed to spawn stdin reader")?;

    eprintln!(
        "listenbury dev continue: streaming one generation; stdin lines, mic transcripts, and 10s time events append to the live context. Ctrl-C cancels."
    );

    let mut cancelled = false;
    let mut next_time_event_at = Instant::now() + TIME_EVENT_INTERVAL;
    let mut speech_events = SpeechEventDetector::default();
    let mut pending_mouth_utterances = 0usize;
    let mut llm_paused_for_mouth = false;
    let mut mouth_playback_paused = false;
    let mut deferred_live_events = VecDeque::<PromptPacket>::new();
    loop {
        if interrupted.load(Ordering::Relaxed) && !cancelled {
            llm_session.cancel()?;
            cancelled = true;
        }

        append_pending_live_events(
            &mut llm_session,
            &stdin_rx,
            &ear_rx,
            &mouth_rx,
            &mut pending_mouth_utterances,
            &mut mouth_playback_paused,
            &mut next_time_event_at,
            speech_events.defers_live_events(),
            &mut deferred_live_events,
        )?;

        if llm_paused_for_mouth && (pending_mouth_utterances == 0 || mouth_playback_paused) {
            llm_session
                .set_paused(false)
                .context("failed to resume continued llama.cpp generation")?;
            llm_paused_for_mouth = false;
        }

        if !llm_paused_for_mouth && pending_mouth_utterances > 0 && !mouth_playback_paused {
            llm_session
                .set_paused(true)
                .context("failed to throttle continued llama.cpp generation")?;
            llm_paused_for_mouth = true;
        }

        if llm_paused_for_mouth {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        }

        let events = llm_session.poll()?;
        if events.is_empty() {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        }

        let generation_terminal = events.iter().any(|event| {
            matches!(
                event,
                LlmEvent::Completed | LlmEvent::Cancelled | LlmEvent::Error { .. }
            )
        });

        for event in &events {
            match event {
                LlmEvent::Token { text } => {
                    print!("{text}");
                    std::io::stdout().flush()?;
                    llm_session.record_generated_text(text);
                    for speech_event in speech_events.ingest(text) {
                        if let ContinueRuntimeEvent::UtteranceCompleted { content, .. } =
                            &speech_event
                        {
                            llm_session.remember_spoken(content);
                        }
                        if let ContinueRuntimeEvent::SourceCommand { command } = &speech_event {
                            let source_result = execute_source_command(command);
                            if !generation_terminal {
                                append_or_defer_live_event(
                                    &mut llm_session,
                                    PromptPacket::source(source_result),
                                    speech_events.defers_live_events(),
                                    &mut deferred_live_events,
                                    "failed to append source event to live generation",
                                )?;
                            }
                        }
                        if mouth.enqueue_runtime_event(&speech_event)? {
                            pending_mouth_utterances += 1;
                        }
                    }
                }
                LlmEvent::Error { message } => {
                    anyhow::bail!("continued llama.cpp generation failed: {message}");
                }
                LlmEvent::Completed | LlmEvent::Cancelled => {}
            }
        }

        if generation_terminal {
            println!();
            while pending_mouth_utterances > 0 {
                drain_mouth_events_without_llm(&mouth_rx, &mut pending_mouth_utterances)?;
                std::thread::sleep(Duration::from_millis(5));
            }
            break;
        }
    }

    Ok(())
}

#[cfg(not(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
)))]
pub(crate) fn run_continue(_command: ContinueCommand) -> Result<()> {
    anyhow::bail!(
        "listenbury dev continue requires the `audio-cpal`, `asr-whisper`, `llm-llama-cpp`, and `tts-piper` features"
    )
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn wrap_live_input(text: &str) -> String {
    format!(
        "\n\n--- LIVE EVENT: user ---\n{}\n--- END LIVE EVENT ---\n\n",
        text.trim()
    )
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn wrap_time_event(message: &str) -> String {
    format!("\n\n--- LIVE EVENT: clock ---\n{message}\n--- END LIVE EVENT ---\n\n")
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn wrap_ear_event(message: &str) -> String {
    format!("\n\n--- LIVE EVENT: ear ---\n{message}\n--- END LIVE EVENT ---\n\n")
}

#[cfg(test)]
fn wrap_mouth_event(message: &str) -> String {
    format!("\n\n--- LIVE EVENT: mouth ---\n{message}\n--- END LIVE EVENT ---\n\n")
}

#[cfg(test)]
fn wrap_runtime_event(message: &str) -> String {
    format!("\n\n--- LIVE EVENT: runtime ---\n{message}\n--- END LIVE EVENT ---\n\n")
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn wrap_source_event(message: &str) -> String {
    format!("\n\n--- LIVE EVENT: source ---\n{message}\n--- END LIVE EVENT ---\n\n")
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn current_time_message() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::ZERO);
    format!(
        "The current Unix time is {}.{:03} seconds.",
        now.as_secs(),
        now.subsec_millis()
    )
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
#[derive(Debug)]
struct ContinueLlmSession {
    llm: LlamaCppEngine,
    id: GenerationId,
    mode: crate::cli::PromptMode,
    max_tokens: Option<usize>,
    rolling: RollingContextManager,
    paused: bool,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
impl ContinueLlmSession {
    fn start(
        mut llm: LlamaCppEngine,
        mode: crate::cli::PromptMode,
        system_prompt: String,
        max_tokens: Option<usize>,
        context_size: u32,
        verbatim_turns: usize,
    ) -> Result<Self> {
        let rolling =
            RollingContextManager::new(system_prompt, context_size, max_tokens, verbatim_turns);
        let (prompt, stop) = build_prompt(mode, &rolling.prompt_body());
        let id = llm.start(GenerationRequest {
            prompt: prompt.clone(),
            max_tokens,
            stop,
        })?;
        let mut session = Self {
            llm,
            id,
            mode,
            max_tokens,
            rolling,
            paused: false,
        };
        session.rolling.note_context_loaded(&prompt);
        Ok(session)
    }

    fn poll(&mut self) -> Result<Vec<LlmEvent>> {
        self.llm.poll(self.id)
    }

    fn cancel(&mut self) -> Result<()> {
        self.llm.cancel(self.id)
    }

    fn set_paused(&mut self, paused: bool) -> Result<()> {
        self.llm.set_paused(self.id, paused)?;
        self.paused = paused;
        Ok(())
    }

    fn record_generated_text(&mut self, text: &str) {
        self.rolling.record_generated_text(text);
    }

    fn remember_spoken(&mut self, text: &str) {
        self.rolling.remember_spoken(text);
    }

    fn append_prompt_packet(&mut self, packet: PromptPacket) -> Result<()> {
        let append_text = self.rolling.record_prompt_packet(packet);
        if self.rolling.should_restart_before_append(&append_text) {
            self.restart_with_compact_prompt()
        } else {
            self.rolling.note_appended_text(&append_text);
            self.llm.append_prompt(self.id, append_text)
        }
    }

    fn restart_with_compact_prompt(&mut self) -> Result<()> {
        self.cancel_current_generation()?;
        let (prompt, stop) = build_prompt(self.mode, &self.rolling.prompt_body());
        self.id = self.llm.start(GenerationRequest {
            prompt: prompt.clone(),
            max_tokens: self.max_tokens,
            stop,
        })?;
        self.rolling.note_context_loaded(&prompt);
        if self.paused {
            self.llm.set_paused(self.id, true)?;
        }
        Ok(())
    }

    fn cancel_current_generation(&mut self) -> Result<()> {
        self.llm.cancel(self.id)?;
        loop {
            let events = self.llm.poll(self.id)?;
            if events.iter().any(|event| {
                matches!(
                    event,
                    LlmEvent::Completed | LlmEvent::Cancelled | LlmEvent::Error { .. }
                )
            }) {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
#[derive(Debug, Clone)]
struct PromptPacket {
    text: String,
    memory: PromptMemory,
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
impl PromptPacket {
    fn listened(text: String) -> Self {
        let trimmed = text.trim().to_string();
        Self {
            text: wrap_live_input(&trimmed),
            memory: PromptMemory::Listened(trimmed),
        }
    }

    fn heard(text: String) -> Self {
        let trimmed = text.trim().to_string();
        Self {
            text: wrap_ear_event(&format!("Heard: {trimmed}")),
            memory: PromptMemory::Listened(trimmed),
        }
    }

    fn spoken(text: String) -> Self {
        Self {
            text: String::new(),
            memory: PromptMemory::Spoken(text.trim().to_string()),
        }
    }

    fn clock(message: String) -> Self {
        Self {
            text: wrap_time_event(&message),
            memory: PromptMemory::Clock(message),
        }
    }

    fn source(message: String) -> Self {
        Self {
            text: wrap_source_event(&message),
            memory: PromptMemory::Source(message),
        }
    }
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
#[derive(Debug, Clone)]
enum PromptMemory {
    Listened(String),
    Spoken(String),
    Clock(String),
    Source(String),
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct CognitivePage {
    kind: PageKind,
    summary: Option<String>,
    events: Vec<String>,
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PageKind {
    Persona,
    Conversation,
    AuditoryScene,
    Intention,
    Scratch,
    Memory,
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConversationTurnKind {
    Listened,
    Spoken,
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
impl ConversationTurnKind {
    fn label(self) -> &'static str {
        match self {
            Self::Listened => "Listened",
            Self::Spoken => "Spoken",
        }
    }
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct ConversationTurn {
    kind: ConversationTurnKind,
    text: String,
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
#[derive(Debug)]
struct RollingContextManager {
    persona: CognitivePage,
    memory: CognitivePage,
    auditory_scene: CognitivePage,
    intention: CognitivePage,
    scratch: CognitivePage,
    recent_turns: std::collections::VecDeque<ConversationTurn>,
    verbatim_turns: usize,
    token_budget: usize,
    active_estimated_tokens: usize,
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
impl RollingContextManager {
    fn new(
        system_prompt: String,
        context_size: u32,
        max_tokens: Option<usize>,
        verbatim_turns: usize,
    ) -> Self {
        let token_budget = rolling_prompt_token_budget(context_size, max_tokens);
        Self {
            persona: CognitivePage {
                kind: PageKind::Persona,
                summary: Some(system_prompt),
                events: Vec::new(),
            },
            memory: CognitivePage {
                kind: PageKind::Memory,
                summary: None,
                events: Vec::new(),
            },
            auditory_scene: CognitivePage {
                kind: PageKind::AuditoryScene,
                summary: None,
                events: Vec::new(),
            },
            intention: CognitivePage {
                kind: PageKind::Intention,
                summary: Some(
                    "Continue naturally. Do not mention internal page management unless relevant."
                        .to_string(),
                ),
                events: Vec::new(),
            },
            scratch: CognitivePage {
                kind: PageKind::Scratch,
                summary: None,
                events: Vec::new(),
            },
            recent_turns: std::collections::VecDeque::new(),
            verbatim_turns,
            token_budget,
            active_estimated_tokens: 0,
        }
    }

    fn prompt_body(&self) -> String {
        let persona = self.persona.summary.as_deref().unwrap_or_default();
        let working_memory = self
            .memory
            .summary
            .as_deref()
            .unwrap_or("No older conversation has been summarized yet.");
        let auditory_scene = if self.auditory_scene.events.is_empty() {
            "No current prompt-worthy auditory scene events.".to_string()
        } else {
            self.auditory_scene.events.join("\n")
        };
        let scratch = if self.scratch.events.is_empty() {
            "No source or scratch observations are loaded.".to_string()
        } else {
            self.scratch.events.join("\n")
        };
        let recent_verbatim = if self.recent_turns.is_empty() {
            "No listened/spoken turns yet.".to_string()
        } else {
            self.recent_turns
                .iter()
                .map(|turn| {
                    format!(
                        "{}: {}",
                        turn.kind.label(),
                        compact_prompt_line(&turn.text, MAX_VERBATIM_TURN_CHARS)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        let next_action = self.intention.summary.as_deref().unwrap_or_default();

        format!(
            "{persona}\n\n\
             <working_memory>\n{working_memory}\n</working_memory>\n\n\
             <auditory_scene>\n{auditory_scene}\n</auditory_scene>\n\n\
             <recent_verbatim>\n{recent_verbatim}\n</recent_verbatim>\n\n\
             <scratch>\n{scratch}\n</scratch>\n\n\
             <next_action>\n{next_action}\n</next_action>\n"
        )
    }

    fn record_prompt_packet(&mut self, packet: PromptPacket) -> String {
        let PromptPacket { text, memory } = packet;
        match memory {
            PromptMemory::Listened(text) => self.push_turn(ConversationTurnKind::Listened, text),
            PromptMemory::Spoken(text) => self.push_turn(ConversationTurnKind::Spoken, text),
            PromptMemory::Clock(message) => self.set_auditory_scene(message),
            PromptMemory::Source(message) => self.push_scratch(message),
        }
        self.compact_until_within_budget();
        text
    }

    fn remember_spoken(&mut self, text: &str) {
        let packet = PromptPacket::spoken(text.to_string());
        let _ = self.record_prompt_packet(packet);
    }

    fn record_generated_text(&mut self, text: &str) {
        self.active_estimated_tokens = self
            .active_estimated_tokens
            .saturating_add(estimate_prompt_tokens(text));
    }

    fn note_context_loaded(&mut self, prompt: &str) {
        self.active_estimated_tokens = estimate_prompt_tokens(prompt);
    }

    fn note_appended_text(&mut self, text: &str) {
        self.active_estimated_tokens = self
            .active_estimated_tokens
            .saturating_add(estimate_prompt_tokens(text));
    }

    fn should_restart_before_append(&mut self, text: &str) -> bool {
        self.compact_until_within_budget();
        self.active_estimated_tokens
            .saturating_add(estimate_prompt_tokens(text))
            > self.token_budget
    }

    fn push_turn(&mut self, kind: ConversationTurnKind, text: String) {
        let text = compact_prompt_line(&text, MAX_VERBATIM_TURN_CHARS);
        if text.is_empty() {
            return;
        }
        self.recent_turns.push_back(ConversationTurn { kind, text });
        while self.recent_turns.len() > self.verbatim_turns {
            self.retire_oldest_turn();
        }
    }

    fn set_auditory_scene(&mut self, message: String) {
        self.auditory_scene.events.clear();
        self.auditory_scene.events.push(format!(
            "Clock: {}",
            compact_prompt_line(&message, MAX_SCRATCH_EVENT_CHARS)
        ));
    }

    fn push_scratch(&mut self, message: String) {
        let line = compact_prompt_line(&message, MAX_SCRATCH_EVENT_CHARS);
        if line.is_empty() {
            return;
        }
        self.scratch.events.push(format!("Source: {line}"));
        while self.scratch.events.len() > MAX_SCRATCH_EVENTS {
            self.scratch.events.remove(0);
        }
    }

    fn compact_until_within_budget(&mut self) {
        while estimate_prompt_tokens(&self.prompt_body()) > self.token_budget {
            if !self.recent_turns.is_empty() {
                self.retire_oldest_turn();
                continue;
            }
            if !self.scratch.events.is_empty() {
                self.scratch.events.remove(0);
                continue;
            }
            self.truncate_memory_summary();
            break;
        }
    }

    fn retire_oldest_turn(&mut self) {
        let Some(turn) = self.recent_turns.pop_front() else {
            return;
        };
        let line = format!(
            "- {}: {}",
            turn.kind.label(),
            compact_prompt_line(&turn.text, MAX_SUMMARY_TURN_CHARS)
        );
        let summary = self.memory.summary.get_or_insert_with(String::new);
        if !summary.is_empty() {
            summary.push('\n');
        }
        summary.push_str(&line);
        if summary.len() > MAX_WORKING_MEMORY_CHARS {
            let keep_from = summary.len() - MAX_WORKING_MEMORY_CHARS;
            let keep_from = next_char_boundary(summary, keep_from);
            *summary = format!("[older memory compressed]\n{}", &summary[keep_from..]);
        }
    }

    fn truncate_memory_summary(&mut self) {
        let Some(summary) = self.memory.summary.as_mut() else {
            return;
        };
        if summary.len() <= MIN_WORKING_MEMORY_CHARS {
            return;
        }
        let keep_from = summary.len() - MIN_WORKING_MEMORY_CHARS;
        let keep_from = next_char_boundary(summary, keep_from);
        *summary = format!("[older memory compressed]\n{}", &summary[keep_from..]);
    }
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const MAX_VERBATIM_TURN_CHARS: usize = 1_200;
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const MAX_SUMMARY_TURN_CHARS: usize = 220;
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const MAX_WORKING_MEMORY_CHARS: usize = 2_400;
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const MIN_WORKING_MEMORY_CHARS: usize = 1_200;
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const MAX_SCRATCH_EVENTS: usize = 3;
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const MAX_SCRATCH_EVENT_CHARS: usize = 1_000;

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn rolling_prompt_token_budget(context_size: u32, max_tokens: Option<usize>) -> usize {
    let context_size = usize::try_from(context_size).unwrap_or(usize::MAX);
    let reserved_generation = max_tokens.unwrap_or(512).max(256);
    context_size
        .saturating_sub(reserved_generation)
        .saturating_mul(3)
        / 4
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn estimate_prompt_tokens(text: &str) -> usize {
    text.chars().count().saturating_add(3) / 4
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn compact_prompt_line(text: &str, max_chars: usize) -> String {
    let mut line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if line.chars().count() <= max_chars {
        return line;
    }
    line = line.chars().take(max_chars.saturating_sub(3)).collect();
    line.push_str("...");
    line
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn next_char_boundary(text: &str, mut index: usize) -> usize {
    while index < text.len() && !text.is_char_boundary(index) {
        index += 1;
    }
    index
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn execute_source_command(command: &SourceCommand) -> String {
    match command {
        SourceCommand::ListFiles => execute_list_source_files(),
        SourceCommand::ViewFile { path, page } => execute_view_source_file(path, *page),
    }
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn execute_list_source_files() -> String {
    let mut files: Vec<_> = source_bundle().keys().cloned().collect();
    files.sort();
    let mut response = String::from("Available source files:\n");
    for file in files {
        response.push_str(&file);
        response.push('\n');
    }
    response
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn execute_view_source_file(path: &str, page: usize) -> String {
    let normalized = path.trim().trim_start_matches("./");
    let page = page.max(1);
    let Some(content) = source_bundle().get(normalized) else {
        return format!("File not found: {normalized}");
    };
    let lines: Vec<_> = content.lines().collect();
    let start = (page - 1) * SOURCE_PAGE_LINES;
    if start >= lines.len() {
        return format!(
            "File {normalized} has only {} lines (page {page} is past EOF).",
            lines.len()
        );
    }
    let end = (start + SOURCE_PAGE_LINES).min(lines.len());
    format!(
        "--- {normalized} (lines {} to {} of {}) ---\n{}\n---",
        start + 1,
        end,
        lines.len(),
        lines[start..end].join("\n")
    )
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn source_bundle() -> &'static std::collections::HashMap<String, String> {
    static BUNDLE: OnceLock<std::collections::HashMap<String, String>> = OnceLock::new();
    BUNDLE.get_or_init(|| {
        let bundle = include_str!(concat!(env!("OUT_DIR"), "/listenbury_source.txt"));
        let mut map = std::collections::HashMap::new();
        let mut current_file = String::new();
        let mut current_content = String::new();

        for line in bundle.lines() {
            if let Some(path) = line.strip_prefix("@@@FILE: ") {
                if !current_file.is_empty() {
                    map.insert(current_file.clone(), current_content.clone());
                    current_content.clear();
                }
                current_file = path.to_string();
            } else {
                current_content.push_str(line);
                current_content.push('\n');
            }
        }
        if !current_file.is_empty() {
            map.insert(current_file, current_content);
        }
        map
    })
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn append_pending_live_events(
    llm_session: &mut ContinueLlmSession,
    stdin_rx: &crossbeam_channel::Receiver<std::result::Result<String, String>>,
    ear_rx: &crossbeam_channel::Receiver<ContinueEarEvent>,
    mouth_rx: &crossbeam_channel::Receiver<ContinueMouthEvent>,
    pending_mouth_utterances: &mut usize,
    mouth_playback_paused: &mut bool,
    next_time_event_at: &mut Instant,
    defer_live_events: bool,
    deferred_live_events: &mut VecDeque<PromptPacket>,
) -> Result<()> {
    if !defer_live_events {
        flush_deferred_live_events(llm_session, deferred_live_events)?;
    }

    let now = Instant::now();
    if now >= *next_time_event_at {
        append_or_defer_live_event(
            llm_session,
            PromptPacket::clock(current_time_message()),
            defer_live_events,
            deferred_live_events,
            "failed to append time event to live generation",
        )?;
        *next_time_event_at = now + TIME_EVENT_INTERVAL;
    }

    for stdin_event in stdin_rx.try_iter() {
        match stdin_event {
            Ok(text) => append_or_defer_live_event(
                llm_session,
                PromptPacket::listened(text),
                defer_live_events,
                deferred_live_events,
                "failed to append stdin text to live generation",
            )?,
            Err(message) => anyhow::bail!("failed to read stdin: {message}"),
        }
    }

    for ear_event in ear_rx.try_iter() {
        match ear_event {
            ContinueEarEvent::Transcript { ref text } => eprintln!("[dev continue] heard: {text}"),
            ContinueEarEvent::ListeningStarted { .. }
            | ContinueEarEvent::SpeechStarted
            | ContinueEarEvent::SpeechStopped
            | ContinueEarEvent::Error { .. } => {}
        }
        if let Some(packet) = ear_event.prompt_packet() {
            append_or_defer_live_event(
                llm_session,
                packet,
                defer_live_events,
                deferred_live_events,
                "failed to append ear event to live generation",
            )?;
        }
        if let ContinueEarEvent::Error { message } = ear_event {
            anyhow::bail!("dev continue ear failed: {message}");
        }
    }

    drain_mouth_events_into_llm(
        llm_session,
        mouth_rx,
        pending_mouth_utterances,
        mouth_playback_paused,
        defer_live_events,
        deferred_live_events,
    )?;

    Ok(())
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn append_or_defer_live_event(
    llm_session: &mut ContinueLlmSession,
    packet: PromptPacket,
    defer_live_events: bool,
    deferred_live_events: &mut VecDeque<PromptPacket>,
    context: &'static str,
) -> Result<()> {
    if defer_live_events {
        deferred_live_events.push_back(packet);
        return Ok(());
    }

    flush_deferred_live_events(llm_session, deferred_live_events)?;
    llm_session.append_prompt_packet(packet).context(context)
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn flush_deferred_live_events(
    llm_session: &mut ContinueLlmSession,
    deferred_live_events: &mut VecDeque<PromptPacket>,
) -> Result<()> {
    while let Some(packet) = deferred_live_events.pop_front() {
        llm_session
            .append_prompt_packet(packet)
            .context("failed to append deferred live event to live generation")?;
    }
    Ok(())
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn drain_mouth_events_into_llm(
    _llm_session: &mut ContinueLlmSession,
    mouth_rx: &crossbeam_channel::Receiver<ContinueMouthEvent>,
    pending_mouth_utterances: &mut usize,
    mouth_playback_paused: &mut bool,
    _defer_live_events: bool,
    _deferred_live_events: &mut VecDeque<PromptPacket>,
) -> Result<()> {
    loop {
        match mouth_rx.try_recv() {
            Ok(mouth_event) => {
                *pending_mouth_utterances = pending_mouth_utterances
                    .saturating_sub(mouth_event.completed_pending_speech_count());
                mouth_event.apply_playback_state(mouth_playback_paused);
                match mouth_event {
                    ContinueMouthEvent::WorkerStarted => {}
                    ContinueMouthEvent::SpeechPlaybackStarted { text, .. } => {
                        eprintln!("[dev continue] speaking: {text}");
                    }
                    ContinueMouthEvent::SpeechError { message, .. } => {
                        anyhow::bail!("dev continue mouth failed: {message}");
                    }
                    ContinueMouthEvent::SpeechQueued { .. }
                    | ContinueMouthEvent::SpeechSynthesisStarted { .. }
                    | ContinueMouthEvent::SpeechPlaybackCompleted { .. }
                    | ContinueMouthEvent::SpeechInterrupted { .. }
                    | ContinueMouthEvent::SpeechQueueCleared { .. }
                    | ContinueMouthEvent::SpeechPaused
                    | ContinueMouthEvent::SpeechResumed => {}
                }
            }
            Err(crossbeam_channel::TryRecvError::Empty) => return Ok(()),
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                if *pending_mouth_utterances > 0 {
                    anyhow::bail!("dev continue mouth worker disconnected with pending speech");
                }
                return Ok(());
            }
        }
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn drain_mouth_events_without_llm(
    mouth_rx: &crossbeam_channel::Receiver<ContinueMouthEvent>,
    pending_mouth_utterances: &mut usize,
) -> Result<()> {
    loop {
        match mouth_rx.try_recv() {
            Ok(mouth_event) => {
                *pending_mouth_utterances = pending_mouth_utterances
                    .saturating_sub(mouth_event.completed_pending_speech_count());
                match mouth_event {
                    ContinueMouthEvent::WorkerStarted => {}
                    ContinueMouthEvent::SpeechPlaybackStarted { text, .. } => {
                        eprintln!("[dev continue] speaking: {text}");
                    }
                    ContinueMouthEvent::SpeechError { message, .. } => {
                        anyhow::bail!("dev continue mouth failed: {message}");
                    }
                    ContinueMouthEvent::SpeechQueued { .. }
                    | ContinueMouthEvent::SpeechSynthesisStarted { .. }
                    | ContinueMouthEvent::SpeechPlaybackCompleted { .. }
                    | ContinueMouthEvent::SpeechInterrupted { .. }
                    | ContinueMouthEvent::SpeechQueueCleared { .. }
                    | ContinueMouthEvent::SpeechPaused
                    | ContinueMouthEvent::SpeechResumed => {}
                }
            }
            Err(crossbeam_channel::TryRecvError::Empty) => return Ok(()),
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                if *pending_mouth_utterances > 0 {
                    anyhow::bail!("dev continue mouth worker disconnected with pending speech");
                }
                return Ok(());
            }
        }
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
struct ContinueEarConfig {
    whisper_model: PathBuf,
    vad_backend: VadBackendKind,
    capture_enabled: Arc<AtomicBool>,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
#[allow(dead_code)]
enum ContinueEarEvent {
    ListeningStarted {
        device: String,
        sample_rate_hz: u32,
        channels: u16,
        vad: VadBackendKind,
    },
    SpeechStarted,
    SpeechStopped,
    Transcript {
        text: String,
    },
    Error {
        message: String,
    },
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
impl ContinueEarEvent {
    #[allow(dead_code)]
    fn to_message(&self) -> String {
        match self {
            Self::ListeningStarted {
                device,
                sample_rate_hz,
                channels,
                vad,
            } => format!(
                "listening_started: device={device:?} sample_rate_hz={sample_rate_hz} channels={channels} vad={}",
                vad.as_str()
            ),
            Self::SpeechStarted => "speech_started".to_string(),
            Self::SpeechStopped => "speech_stopped".to_string(),
            Self::Transcript { text } => format!("Heard: {}", text.trim()),
            Self::Error { message } => format!("error: {message}"),
        }
    }

    fn prompt_packet(&self) -> Option<PromptPacket> {
        match self {
            Self::Transcript { text } => Some(PromptPacket::heard(text.clone())),
            Self::ListeningStarted { .. }
            | Self::SpeechStarted
            | Self::SpeechStopped
            | Self::Error { .. } => None,
        }
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
struct ContinueEar {
    stop: Arc<AtomicBool>,
    _stream: cpal::Stream,
    processor: Option<JoinHandle<()>>,
    asr: Option<JoinHandle<()>>,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
impl ContinueEar {
    fn start(
        config: ContinueEarConfig,
    ) -> Result<(Self, crossbeam_channel::Receiver<ContinueEarEvent>)> {
        let mut recognizer =
            WhisperSpeechRecognizer::new(&config.whisper_model).with_context(|| {
                format!(
                    "failed to load Whisper model at {}",
                    config.whisper_model.display()
                )
            })?;
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow::anyhow!("no default input device available"))?;
        let device_name = device
            .name()
            .unwrap_or_else(|_| "<unknown input device>".to_string());
        let supported_config = device
            .default_input_config()
            .with_context(|| format!("failed to read default input config for {device_name}"))?;
        let stream_config = supported_config.config();
        let input_sample_rate_hz = stream_config.sample_rate.0;
        let input_channels = stream_config.channels;
        anyhow::ensure!(
            input_channels > 0,
            "default input device reported zero channels"
        );

        let stop = Arc::new(AtomicBool::new(false));
        let (sample_tx, sample_rx) = crossbeam_channel::bounded::<f32>(CALLBACK_SAMPLE_CAPACITY);
        let (asr_tx, asr_rx) = crossbeam_channel::bounded::<Vec<AudioFrame>>(8);
        let (event_tx, event_rx) = crossbeam_channel::unbounded::<ContinueEarEvent>();
        let dropped_in_callback = Arc::new(AtomicUsize::new(0));
        let err_fn = |err| eprintln!("input stream error: {err}");
        let stream = match supported_config.sample_format() {
            cpal::SampleFormat::F32 => build_continue_input_stream::<f32>(
                &device,
                &stream_config,
                sample_tx.clone(),
                Arc::clone(&dropped_in_callback),
                Arc::clone(&config.capture_enabled),
                err_fn,
            )?,
            cpal::SampleFormat::F64 => build_continue_input_stream::<f64>(
                &device,
                &stream_config,
                sample_tx.clone(),
                Arc::clone(&dropped_in_callback),
                Arc::clone(&config.capture_enabled),
                err_fn,
            )?,
            cpal::SampleFormat::I8 => build_continue_input_stream::<i8>(
                &device,
                &stream_config,
                sample_tx.clone(),
                Arc::clone(&dropped_in_callback),
                Arc::clone(&config.capture_enabled),
                err_fn,
            )?,
            cpal::SampleFormat::I16 => build_continue_input_stream::<i16>(
                &device,
                &stream_config,
                sample_tx.clone(),
                Arc::clone(&dropped_in_callback),
                Arc::clone(&config.capture_enabled),
                err_fn,
            )?,
            cpal::SampleFormat::I32 => build_continue_input_stream::<i32>(
                &device,
                &stream_config,
                sample_tx.clone(),
                Arc::clone(&dropped_in_callback),
                Arc::clone(&config.capture_enabled),
                err_fn,
            )?,
            cpal::SampleFormat::I64 => build_continue_input_stream::<i64>(
                &device,
                &stream_config,
                sample_tx.clone(),
                Arc::clone(&dropped_in_callback),
                Arc::clone(&config.capture_enabled),
                err_fn,
            )?,
            cpal::SampleFormat::U8 => build_continue_input_stream::<u8>(
                &device,
                &stream_config,
                sample_tx.clone(),
                Arc::clone(&dropped_in_callback),
                Arc::clone(&config.capture_enabled),
                err_fn,
            )?,
            cpal::SampleFormat::U16 => build_continue_input_stream::<u16>(
                &device,
                &stream_config,
                sample_tx.clone(),
                Arc::clone(&dropped_in_callback),
                Arc::clone(&config.capture_enabled),
                err_fn,
            )?,
            cpal::SampleFormat::U32 => build_continue_input_stream::<u32>(
                &device,
                &stream_config,
                sample_tx.clone(),
                Arc::clone(&dropped_in_callback),
                Arc::clone(&config.capture_enabled),
                err_fn,
            )?,
            cpal::SampleFormat::U64 => build_continue_input_stream::<u64>(
                &device,
                &stream_config,
                sample_tx,
                Arc::clone(&dropped_in_callback),
                Arc::clone(&config.capture_enabled),
                err_fn,
            )?,
            sample_format => anyhow::bail!("unsupported input sample format: {sample_format:?}"),
        };
        stream
            .play()
            .with_context(|| format!("failed to start capture from {device_name}"))?;

        eprintln!(
            "dev continue ear listening on {device_name}: {} Hz, {} channel(s), vad={}.",
            input_sample_rate_hz,
            input_channels,
            config.vad_backend.as_str()
        );
        let _ = event_tx.send(ContinueEarEvent::ListeningStarted {
            device: device_name,
            sample_rate_hz: input_sample_rate_hz,
            channels: input_channels,
            vad: config.vad_backend,
        });

        let stop_for_asr = Arc::clone(&stop);
        let event_tx_for_asr = event_tx.clone();
        let asr = std::thread::Builder::new()
            .name("listenbury-dev-continue-asr".to_string())
            .spawn(move || {
                while !stop_for_asr.load(Ordering::Relaxed) {
                    match asr_rx.recv_timeout(Duration::from_millis(20)) {
                        Ok(frames) => match transcribe_group(&frames, &mut recognizer) {
                            Ok(text) if !text.is_empty() => {
                                if event_tx_for_asr
                                    .send(ContinueEarEvent::Transcript { text })
                                    .is_err()
                                {
                                    return;
                                }
                            }
                            Ok(_) => {}
                            Err(error) => {
                                let _ = event_tx_for_asr.send(ContinueEarEvent::Error {
                                    message: error.to_string(),
                                });
                            }
                        },
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return,
                    }
                }
            })
            .context("failed to spawn dev continue ASR worker")?;

        let stop_for_processor = Arc::clone(&stop);
        let processor = std::thread::Builder::new()
            .name("listenbury-dev-continue-ear".to_string())
            .spawn(move || {
                if let Err(error) = run_continue_ear_processor(
                    sample_rx,
                    asr_tx,
                    event_tx.clone(),
                    stop_for_processor,
                    config.vad_backend,
                    input_sample_rate_hz,
                    input_channels,
                ) {
                    let _ = event_tx.send(ContinueEarEvent::Error {
                        message: error.to_string(),
                    });
                }
            })
            .context("failed to spawn dev continue ear worker")?;

        Ok((
            Self {
                stop,
                _stream: stream,
                processor: Some(processor),
                asr: Some(asr),
            },
            event_rx,
        ))
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
impl Drop for ContinueEar {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.processor.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.asr.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
struct ContinueEarState {
    vad: Box<dyn VoiceActivityDetector>,
    segmenter: BreathGroupSegmenter,
    active_groups: HashMap<BreathGroupId, Vec<AudioFrame>>,
    frame_time_ms: u64,
    last_vad_state: Option<bool>,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn run_continue_ear_processor(
    sample_rx: crossbeam_channel::Receiver<f32>,
    asr_tx: crossbeam_channel::Sender<Vec<AudioFrame>>,
    event_tx: crossbeam_channel::Sender<ContinueEarEvent>,
    stop: Arc<AtomicBool>,
    vad_backend: VadBackendKind,
    input_sample_rate_hz: u32,
    input_channels: u16,
) -> Result<()> {
    let input_frame_samples =
        frame_samples_per_callback_frame(input_sample_rate_hz, input_channels);
    let (frame_sample_rate_hz, frame_channels) =
        vad_frame_format(vad_backend, input_sample_rate_hz, input_channels);
    let mut pending = VecDeque::<f32>::new();
    let mut state = ContinueEarState {
        vad: create_vad_backend(vad_backend)?,
        segmenter: BreathGroupSegmenter::default(),
        active_groups: HashMap::new(),
        frame_time_ms: 0,
        last_vad_state: None,
    };

    while !stop.load(Ordering::Relaxed) {
        match sample_rx.recv_timeout(Duration::from_millis(20)) {
            Ok(sample) => pending.push_back(sample),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
        while let Ok(sample) = sample_rx.try_recv() {
            pending.push_back(sample);
        }
        drain_pending_continue_ear_frames(
            &mut pending,
            input_frame_samples,
            input_sample_rate_hz,
            input_channels,
            frame_sample_rate_hz,
            frame_channels,
            &mut state,
            &asr_tx,
            &event_tx,
        )?;
    }

    while let Ok(sample) = sample_rx.try_recv() {
        pending.push_back(sample);
    }
    drain_pending_continue_ear_frames(
        &mut pending,
        input_frame_samples,
        input_sample_rate_hz,
        input_channels,
        frame_sample_rate_hz,
        frame_channels,
        &mut state,
        &asr_tx,
        &event_tx,
    )?;
    for (_, frames) in state.active_groups.drain() {
        if !frames.is_empty() && asr_tx.send(frames).is_err() {
            break;
        }
    }

    Ok(())
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn drain_pending_continue_ear_frames(
    pending: &mut VecDeque<f32>,
    input_frame_samples: usize,
    input_sample_rate_hz: u32,
    input_channels: u16,
    frame_sample_rate_hz: u32,
    frame_channels: u16,
    state: &mut ContinueEarState,
    asr_tx: &crossbeam_channel::Sender<Vec<AudioFrame>>,
    event_tx: &crossbeam_channel::Sender<ContinueEarEvent>,
) -> Result<()> {
    while pending.len() >= input_frame_samples {
        let mut samples = Vec::with_capacity(input_frame_samples);
        for _ in 0..input_frame_samples {
            if let Some(sample) = pending.pop_front() {
                samples.push(sample);
            }
        }
        if samples.len() < input_frame_samples {
            break;
        }
        let samples = convert_frame_samples(
            &samples,
            input_sample_rate_hz,
            input_channels,
            frame_sample_rate_hz,
            frame_channels,
        );
        let frame = AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: frame_sample_rate_hz,
            channels: frame_channels,
            samples,
        };
        process_continue_ear_frame(frame, state, asr_tx, event_tx)?;
    }
    Ok(())
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn process_continue_ear_frame(
    frame: AudioFrame,
    state: &mut ContinueEarState,
    asr_tx: &crossbeam_channel::Sender<Vec<AudioFrame>>,
    event_tx: &crossbeam_channel::Sender<ContinueEarEvent>,
) -> Result<()> {
    let frame_duration_ms = frame_duration_ms(&frame);
    let vad_result = state.vad.process_frame(&frame)?;
    if listenbury::developer_diagnostics_enabled()
        && state.last_vad_state != Some(vad_result.is_speech)
    {
        eprintln!(
            "[dev continue ear] vad t_ms={} speech={} prob={:.3}",
            state.frame_time_ms, vad_result.is_speech, vad_result.speech_prob
        );
        state.last_vad_state = Some(vad_result.is_speech);
    }

    let events = state.segmenter.process(vad_result);
    for event in &events {
        match event {
            HearingEvent::SpeechStarted => {
                let _ = event_tx.send(ContinueEarEvent::SpeechStarted);
            }
            HearingEvent::BreathGroupOpened { id } => {
                state.active_groups.entry(*id).or_default();
            }
            HearingEvent::BreathGroupClosed { .. } => {
                let _ = event_tx.send(ContinueEarEvent::SpeechStopped);
            }
            HearingEvent::SpeechContinued { .. } | HearingEvent::PauseStarted => {}
        }
    }
    for group in state.active_groups.values_mut() {
        group.push(frame.clone());
    }
    for event in events {
        if let HearingEvent::BreathGroupClosed { id, .. } = event {
            if let Some(group_frames) = state.active_groups.remove(&id) {
                if !group_frames.is_empty() && asr_tx.send(group_frames).is_err() {
                    return Ok(());
                }
            }
        }
    }
    state.frame_time_ms = state.frame_time_ms.saturating_add(frame_duration_ms);
    Ok(())
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn build_continue_input_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_tx: crossbeam_channel::Sender<f32>,
    dropped_in_callback: Arc<AtomicUsize>,
    capture_enabled: Arc<AtomicBool>,
    err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream>
where
    T: Sample + SizedSample,
    f32: FromSample<T>,
{
    device
        .build_input_stream(
            config,
            move |data: &[T], _| {
                if !capture_enabled.load(Ordering::Relaxed) {
                    return;
                }
                for sample in data {
                    if sample_tx.try_send(sample.to_sample::<f32>()).is_err() {
                        dropped_in_callback.fetch_add(1, Ordering::Relaxed);
                    }
                }
            },
            err_fn,
            None,
        )
        .context("failed to build input stream")
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn vad_frame_format(
    vad_backend: VadBackendKind,
    input_sample_rate_hz: u32,
    input_channels: u16,
) -> (u32, u16) {
    match vad_backend {
        VadBackendKind::WebRtc => (WEBRTC_VAD_SAMPLE_RATE_HZ, MONO_CHANNELS),
        VadBackendKind::Energy | VadBackendKind::Silero => (input_sample_rate_hz, input_channels),
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn convert_frame_samples(
    samples: &[f32],
    input_sample_rate_hz: u32,
    input_channels: u16,
    frame_sample_rate_hz: u32,
    frame_channels: u16,
) -> Vec<f32> {
    if input_sample_rate_hz == frame_sample_rate_hz && input_channels == frame_channels {
        return samples.to_vec();
    }

    let mut converted = if input_channels != frame_channels && frame_channels == MONO_CHANNELS {
        mix_to_mono(samples, input_channels)
    } else {
        samples.to_vec()
    };

    if input_sample_rate_hz != frame_sample_rate_hz {
        converted = resample_linear(&converted, input_sample_rate_hz, frame_sample_rate_hz);
    }

    converted
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn frame_samples_per_callback_frame(sample_rate_hz: u32, channels: u16) -> usize {
    let samples_per_channel = usize::try_from(sample_rate_hz / 100).unwrap_or(1).max(1);
    samples_per_channel.saturating_mul(usize::from(channels).max(1))
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn frame_duration_ms(frame: &AudioFrame) -> u64 {
    if frame.sample_rate_hz == 0 || frame.channels == 0 {
        return 0;
    }
    let samples_per_channel = frame.samples.len() as f64 / f64::from(frame.channels);
    ((samples_per_channel / f64::from(frame.sample_rate_hz)) * 1000.0).round() as u64
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn mix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    let channel_count = usize::from(channels).max(1);
    if channel_count == 1 {
        return samples.to_vec();
    }
    samples
        .chunks_exact(channel_count)
        .map(|frame| frame.iter().sum::<f32>() / f32::from(channels))
        .collect()
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn resample_linear(samples: &[f32], source_rate_hz: u32, target_rate_hz: u32) -> Vec<f32> {
    if samples.is_empty() || source_rate_hz == target_rate_hz {
        return samples.to_vec();
    }

    let output_len = ((samples.len() as f64 * f64::from(target_rate_hz))
        / f64::from(source_rate_hz))
    .round() as usize;
    let mut output = Vec::with_capacity(output_len);
    let source_step = f64::from(source_rate_hz) / f64::from(target_rate_hz);

    for output_idx in 0..output_len {
        let source_pos = output_idx as f64 * source_step;
        let left_idx = source_pos.floor() as usize;
        let right_idx = (left_idx + 1).min(samples.len() - 1);
        let fraction = (source_pos - left_idx as f64) as f32;
        let left = samples[left_idx];
        let right = samples[right_idx];
        output.push(left + (right - left) * fraction);
    }

    output
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
struct ContinueMouth {
    tx: crossbeam_channel::Sender<ContinueMouthCommand>,
    worker: Option<JoinHandle<()>>,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
impl ContinueMouth {
    fn start(
        tts: PiperTextToSpeech,
        capture_enabled: Arc<AtomicBool>,
    ) -> Result<(Self, crossbeam_channel::Receiver<ContinueMouthEvent>)> {
        let (tx, rx) = crossbeam_channel::unbounded();
        let (event_tx, event_rx) = crossbeam_channel::unbounded();
        let worker = std::thread::Builder::new()
            .name("listenbury-dev-continue-mouth".to_string())
            .spawn(move || run_continue_mouth_worker(tts, rx, event_tx, capture_enabled))
            .context("failed to spawn dev continue mouth worker")?;
        Ok((
            Self {
                tx,
                worker: Some(worker),
            },
            event_rx,
        ))
    }

    fn enqueue_runtime_event(&mut self, event: &ContinueRuntimeEvent) -> Result<bool> {
        match event {
            ContinueRuntimeEvent::UtteranceCompleted { id, content } => {
                if strip_emoji(content).trim().is_empty() {
                    return Ok(false);
                }

                self.tx
                    .send(ContinueMouthCommand::Speak {
                        id: *id,
                        text: content.to_string(),
                    })
                    .context("failed to send speech to dev continue mouth worker")?;
                Ok(true)
            }
            ContinueRuntimeEvent::SpeechControl { command } => {
                let command = match command {
                    SpeechControlCommand::Shutup => ContinueMouthCommand::Shutup,
                    SpeechControlCommand::Pause => ContinueMouthCommand::Pause,
                    SpeechControlCommand::Resume => ContinueMouthCommand::Resume,
                };
                self.tx
                    .send(command)
                    .context("failed to send speech control to dev continue mouth worker")?;
                Ok(false)
            }
            ContinueRuntimeEvent::UtteranceStarted { .. } => Ok(false),
            ContinueRuntimeEvent::SourceCommand { .. } => Ok(false),
        }
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
impl Drop for ContinueMouth {
    fn drop(&mut self) {
        let _ = self.tx.send(ContinueMouthCommand::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
enum ContinueMouthCommand {
    Speak { id: u64, text: String },
    Shutup,
    Pause,
    Resume,
    Shutdown,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
#[allow(dead_code)]
enum ContinueMouthEvent {
    WorkerStarted,
    SpeechQueued { id: u64, text: String },
    SpeechSynthesisStarted { id: u64, text: String },
    SpeechPlaybackStarted { id: u64, text: String },
    SpeechPlaybackCompleted { id: u64, text: String },
    SpeechInterrupted { id: u64, text: String },
    SpeechQueueCleared { count: usize },
    SpeechPaused,
    SpeechResumed,
    SpeechError { id: u64, message: String },
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
impl ContinueMouthEvent {
    #[allow(dead_code)]
    fn to_message(&self) -> String {
        match self {
            Self::WorkerStarted => "worker_started".to_string(),
            Self::SpeechQueued { id, text } => {
                format!(
                    "speech_queued: id={id}\ncontent:\n{}",
                    sanitize_runtime_event_content(text)
                )
            }
            Self::SpeechSynthesisStarted { id, text } => {
                format!(
                    "speech_synthesis_started: id={id}\ncontent:\n{}",
                    sanitize_runtime_event_content(text)
                )
            }
            Self::SpeechPlaybackStarted { id, text } => {
                format!(
                    "speech_playback_started: id={id}\ncontent:\n{}",
                    sanitize_runtime_event_content(text)
                )
            }
            Self::SpeechPlaybackCompleted { id, text } => {
                format!(
                    "speech_playback_completed: id={id}\ncontent:\n{}",
                    sanitize_runtime_event_content(text)
                )
            }
            Self::SpeechInterrupted { id, text } => {
                format!(
                    "speech_interrupted: id={id}\ncontent:\n{}",
                    sanitize_runtime_event_content(text)
                )
            }
            Self::SpeechQueueCleared { count } => {
                format!("speech_queue_cleared: count={count}")
            }
            Self::SpeechPaused => "speech_paused".to_string(),
            Self::SpeechResumed => "speech_resumed".to_string(),
            Self::SpeechError { id, message } => {
                format!(
                    "speech_error: id={id}\nmessage:\n{}",
                    sanitize_runtime_event_content(message)
                )
            }
        }
    }

    fn completed_pending_speech_count(&self) -> usize {
        match self {
            Self::SpeechPlaybackCompleted { .. }
            | Self::SpeechInterrupted { .. }
            | Self::SpeechError { .. } => 1,
            Self::SpeechQueueCleared { count } => *count,
            Self::WorkerStarted
            | Self::SpeechQueued { .. }
            | Self::SpeechSynthesisStarted { .. }
            | Self::SpeechPlaybackStarted { .. }
            | Self::SpeechPaused
            | Self::SpeechResumed => 0,
        }
    }

    fn apply_playback_state(&self, paused: &mut bool) {
        match self {
            Self::SpeechPaused => *paused = true,
            Self::SpeechResumed
            | Self::SpeechPlaybackCompleted { .. }
            | Self::SpeechInterrupted { .. }
            | Self::SpeechError { .. } => *paused = false,
            Self::WorkerStarted
            | Self::SpeechQueued { .. }
            | Self::SpeechSynthesisStarted { .. }
            | Self::SpeechPlaybackStarted { .. }
            | Self::SpeechQueueCleared { .. } => {}
        }
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn run_continue_mouth_worker(
    mut tts: PiperTextToSpeech,
    rx: crossbeam_channel::Receiver<ContinueMouthCommand>,
    event_tx: crossbeam_channel::Sender<ContinueMouthEvent>,
    capture_enabled: Arc<AtomicBool>,
) {
    let _ = event_tx.send(ContinueMouthEvent::WorkerStarted);
    let mut pending = VecDeque::<(u64, String)>::new();
    let mut paused = false;
    loop {
        let command = if let Some((id, text)) = pending.pop_front() {
            ContinueMouthCommand::Speak { id, text }
        } else {
            match rx.recv() {
                Ok(command) => command,
                Err(_) => return,
            }
        };
        match command {
            ContinueMouthCommand::Speak { id, text } => {
                match run_continue_mouth_speech(
                    id,
                    text,
                    &mut tts,
                    &rx,
                    &mut pending,
                    &event_tx,
                    &capture_enabled,
                    &mut paused,
                ) {
                    Ok(MouthWorkerFlow::Continue) | Err(_) => {}
                    Ok(MouthWorkerFlow::Shutdown) => return,
                }
            }
            ContinueMouthCommand::Shutup => {
                let _ = tts.stop();
                if send_cleared_mouth_queue_event(&rx, &mut pending, &event_tx) {
                    return;
                }
            }
            ContinueMouthCommand::Pause => pause_mouth_playback(&event_tx, &mut paused),
            ContinueMouthCommand::Resume => resume_mouth_playback(&event_tx, &mut paused),
            ContinueMouthCommand::Shutdown => return,
        }
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MouthWorkerFlow {
    Continue,
    Shutdown,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MouthControlFlow {
    Continue,
    StopCurrent,
    Shutdown,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
#[derive(Debug)]
enum MouthAudioOutcome {
    Frames(Vec<AudioFrame>),
    Interrupted,
    Shutdown,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MouthPlaybackOutcome {
    Completed,
    Interrupted,
    Shutdown,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn pause_mouth_playback(
    event_tx: &crossbeam_channel::Sender<ContinueMouthEvent>,
    paused: &mut bool,
) {
    if !*paused {
        *paused = true;
        let _ = event_tx.send(ContinueMouthEvent::SpeechPaused);
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn resume_mouth_playback(
    event_tx: &crossbeam_channel::Sender<ContinueMouthEvent>,
    paused: &mut bool,
) {
    if *paused {
        *paused = false;
        let _ = event_tx.send(ContinueMouthEvent::SpeechResumed);
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn drain_mouth_control_commands(
    rx: &crossbeam_channel::Receiver<ContinueMouthCommand>,
    pending: &mut VecDeque<(u64, String)>,
    event_tx: &crossbeam_channel::Sender<ContinueMouthEvent>,
    tts: &mut PiperTextToSpeech,
    paused: &mut bool,
) -> MouthControlFlow {
    loop {
        match rx.try_recv() {
            Ok(ContinueMouthCommand::Speak { id, text }) => pending.push_back((id, text)),
            Ok(ContinueMouthCommand::Pause) => pause_mouth_playback(event_tx, paused),
            Ok(ContinueMouthCommand::Resume) => resume_mouth_playback(event_tx, paused),
            Ok(ContinueMouthCommand::Shutup) => {
                let _ = tts.stop();
                if send_cleared_mouth_queue_event(rx, pending, event_tx) {
                    return MouthControlFlow::Shutdown;
                }
                return MouthControlFlow::StopCurrent;
            }
            Ok(ContinueMouthCommand::Shutdown) => {
                let _ = tts.stop();
                send_cleared_mouth_queue_event(rx, pending, event_tx);
                return MouthControlFlow::Shutdown;
            }
            Err(crossbeam_channel::TryRecvError::Empty) => return MouthControlFlow::Continue,
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                return MouthControlFlow::Shutdown;
            }
        }
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn send_cleared_mouth_queue_event(
    rx: &crossbeam_channel::Receiver<ContinueMouthCommand>,
    pending: &mut VecDeque<(u64, String)>,
    event_tx: &crossbeam_channel::Sender<ContinueMouthEvent>,
) -> bool {
    let mut cleared = pending.len();
    let mut shutdown = false;
    pending.clear();
    loop {
        match rx.try_recv() {
            Ok(ContinueMouthCommand::Speak { .. }) => cleared += 1,
            Ok(ContinueMouthCommand::Shutdown) => shutdown = true,
            Ok(ContinueMouthCommand::Shutup)
            | Ok(ContinueMouthCommand::Pause)
            | Ok(ContinueMouthCommand::Resume) => {}
            Err(crossbeam_channel::TryRecvError::Empty)
            | Err(crossbeam_channel::TryRecvError::Disconnected) => break,
        }
    }
    let _ = event_tx.send(ContinueMouthEvent::SpeechQueueCleared { count: cleared });
    shutdown
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn run_continue_mouth_speech(
    id: u64,
    text: String,
    tts: &mut PiperTextToSpeech,
    rx: &crossbeam_channel::Receiver<ContinueMouthCommand>,
    pending: &mut VecDeque<(u64, String)>,
    event_tx: &crossbeam_channel::Sender<ContinueMouthEvent>,
    capture_enabled: &AtomicBool,
    paused: &mut bool,
) -> Result<MouthWorkerFlow> {
    event_tx
        .send(ContinueMouthEvent::SpeechQueued {
            id,
            text: text.clone(),
        })
        .ok();
    event_tx
        .send(ContinueMouthEvent::SpeechSynthesisStarted {
            id,
            text: text.clone(),
        })
        .ok();

    if let Err(error) = tts.enqueue(SpeechPlan::from(SpeechUnit::CompleteSentence(text.clone()))) {
        let _ = event_tx.send(ContinueMouthEvent::SpeechError {
            id,
            message: error.to_string(),
        });
        return Err(error);
    }

    let frames = match collect_continue_mouth_audio(
        tts,
        Duration::from_secs(30),
        rx,
        pending,
        event_tx,
        paused,
    ) {
        Ok(MouthAudioOutcome::Frames(frames)) => frames,
        Ok(MouthAudioOutcome::Interrupted) => {
            let _ = event_tx.send(ContinueMouthEvent::SpeechInterrupted { id, text });
            return Ok(MouthWorkerFlow::Continue);
        }
        Ok(MouthAudioOutcome::Shutdown) => return Ok(MouthWorkerFlow::Shutdown),
        Err(error) => {
            let _ = event_tx.send(ContinueMouthEvent::SpeechError {
                id,
                message: error.to_string(),
            });
            return Err(error);
        }
    };

    event_tx
        .send(ContinueMouthEvent::SpeechPlaybackStarted {
            id,
            text: text.clone(),
        })
        .ok();
    capture_enabled.store(false, Ordering::Relaxed);
    let playback = play_continue_audio_frames_interruptible(
        &frames,
        "listenbury dev continue speech",
        rx,
        pending,
        event_tx,
        tts,
        paused,
    );
    capture_enabled.store(true, Ordering::Relaxed);
    match playback {
        Ok(MouthPlaybackOutcome::Completed) => {}
        Ok(MouthPlaybackOutcome::Interrupted) => {
            let _ = event_tx.send(ContinueMouthEvent::SpeechInterrupted { id, text });
            return Ok(MouthWorkerFlow::Continue);
        }
        Ok(MouthPlaybackOutcome::Shutdown) => return Ok(MouthWorkerFlow::Shutdown),
        Err(error) => {
            let _ = event_tx.send(ContinueMouthEvent::SpeechError {
                id,
                message: error.to_string(),
            });
            return Err(error);
        }
    }
    event_tx
        .send(ContinueMouthEvent::SpeechPlaybackCompleted { id, text })
        .ok();
    Ok(MouthWorkerFlow::Continue)
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn collect_continue_mouth_audio(
    tts: &mut PiperTextToSpeech,
    timeout: Duration,
    rx: &crossbeam_channel::Receiver<ContinueMouthCommand>,
    pending: &mut VecDeque<(u64, String)>,
    event_tx: &crossbeam_channel::Sender<ContinueMouthEvent>,
    paused: &mut bool,
) -> Result<MouthAudioOutcome> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        match drain_mouth_control_commands(rx, pending, event_tx, tts, paused) {
            MouthControlFlow::Continue => {}
            MouthControlFlow::StopCurrent => return Ok(MouthAudioOutcome::Interrupted),
            MouthControlFlow::Shutdown => return Ok(MouthAudioOutcome::Shutdown),
        }
        let frames = tts.poll_audio()?;
        if !frames.is_empty() {
            return Ok(MouthAudioOutcome::Frames(frames));
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    anyhow::bail!("Piper produced no audio frames before timeout")
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn play_continue_audio_frames_interruptible(
    frames: &[AudioFrame],
    source: &str,
    rx: &crossbeam_channel::Receiver<ContinueMouthCommand>,
    pending: &mut VecDeque<(u64, String)>,
    event_tx: &crossbeam_channel::Sender<ContinueMouthEvent>,
    tts: &mut PiperTextToSpeech,
    paused: &mut bool,
) -> Result<MouthPlaybackOutcome> {
    let Some(first_frame) = frames.first() else {
        anyhow::bail!("no audio frames available for playback from {source}");
    };
    let sample_rate = first_frame.sample_rate_hz;
    let channels = first_frame.channels;
    anyhow::ensure!(
        sample_rate > 0,
        "audio from {source} has invalid sample rate"
    );
    anyhow::ensure!(
        channels > 0,
        "audio from {source} has invalid channel count"
    );

    let total_samples: usize = frames.iter().map(|frame| frame.samples.len()).sum();
    let mut audio_samples = Vec::with_capacity(total_samples);
    for frame in frames {
        anyhow::ensure!(
            frame.sample_rate_hz == sample_rate,
            "audio from {source} changed sample rate mid-stream ({} -> {})",
            sample_rate,
            frame.sample_rate_hz
        );
        anyhow::ensure!(
            frame.channels == channels,
            "audio from {source} changed channel count mid-stream ({} -> {})",
            channels,
            frame.channels
        );
        audio_samples.extend_from_slice(&frame.samples);
    }

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow::anyhow!("no default output device available"))?;
    let device_name = device
        .name()
        .unwrap_or_else(|_| "<unknown output device>".to_string());
    let supported = select_continue_output_config(&device, sample_rate, channels)?;
    let stream_config = supported
        .with_sample_rate(cpal::SampleRate(sample_rate))
        .config();

    let playback_cursor = Arc::new(AtomicUsize::new(0));
    let playback_paused = Arc::new(AtomicBool::new(*paused));
    let samples = Arc::new(audio_samples);
    let done_threshold = samples.len();
    let err_fn = |err| eprintln!("output stream error: {err}");
    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => build_continue_output_stream::<f32>(
            &device,
            &stream_config,
            Arc::clone(&samples),
            Arc::clone(&playback_cursor),
            Arc::clone(&playback_paused),
            err_fn,
        )?,
        cpal::SampleFormat::F64 => build_continue_output_stream::<f64>(
            &device,
            &stream_config,
            Arc::clone(&samples),
            Arc::clone(&playback_cursor),
            Arc::clone(&playback_paused),
            err_fn,
        )?,
        cpal::SampleFormat::I8 => build_continue_output_stream::<i8>(
            &device,
            &stream_config,
            Arc::clone(&samples),
            Arc::clone(&playback_cursor),
            Arc::clone(&playback_paused),
            err_fn,
        )?,
        cpal::SampleFormat::I16 => build_continue_output_stream::<i16>(
            &device,
            &stream_config,
            Arc::clone(&samples),
            Arc::clone(&playback_cursor),
            Arc::clone(&playback_paused),
            err_fn,
        )?,
        cpal::SampleFormat::I32 => build_continue_output_stream::<i32>(
            &device,
            &stream_config,
            Arc::clone(&samples),
            Arc::clone(&playback_cursor),
            Arc::clone(&playback_paused),
            err_fn,
        )?,
        cpal::SampleFormat::I64 => build_continue_output_stream::<i64>(
            &device,
            &stream_config,
            Arc::clone(&samples),
            Arc::clone(&playback_cursor),
            Arc::clone(&playback_paused),
            err_fn,
        )?,
        cpal::SampleFormat::U8 => build_continue_output_stream::<u8>(
            &device,
            &stream_config,
            Arc::clone(&samples),
            Arc::clone(&playback_cursor),
            Arc::clone(&playback_paused),
            err_fn,
        )?,
        cpal::SampleFormat::U16 => build_continue_output_stream::<u16>(
            &device,
            &stream_config,
            Arc::clone(&samples),
            Arc::clone(&playback_cursor),
            Arc::clone(&playback_paused),
            err_fn,
        )?,
        cpal::SampleFormat::U32 => build_continue_output_stream::<u32>(
            &device,
            &stream_config,
            Arc::clone(&samples),
            Arc::clone(&playback_cursor),
            Arc::clone(&playback_paused),
            err_fn,
        )?,
        cpal::SampleFormat::U64 => build_continue_output_stream::<u64>(
            &device,
            &stream_config,
            Arc::clone(&samples),
            Arc::clone(&playback_cursor),
            Arc::clone(&playback_paused),
            err_fn,
        )?,
        sample_format => anyhow::bail!("unsupported output sample format: {sample_format:?}"),
    };
    stream
        .play()
        .with_context(|| format!("failed to start playback on {device_name}"))?;

    while playback_cursor.load(Ordering::Relaxed) < done_threshold {
        match drain_mouth_control_commands(rx, pending, event_tx, tts, paused) {
            MouthControlFlow::Continue => {
                playback_paused.store(*paused, Ordering::Relaxed);
            }
            MouthControlFlow::StopCurrent => {
                drop(stream);
                return Ok(MouthPlaybackOutcome::Interrupted);
            }
            MouthControlFlow::Shutdown => {
                drop(stream);
                return Ok(MouthPlaybackOutcome::Shutdown);
            }
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    std::thread::sleep(Duration::from_millis(20));
    drop(stream);

    let audio_duration = continue_playback_duration(total_samples, sample_rate, channels);
    println!(
        "Played with {device_name}: {} Hz, {channels} channel(s), {:.2}s from {source}",
        sample_rate,
        audio_duration.as_secs_f64(),
    );

    Ok(MouthPlaybackOutcome::Completed)
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn build_continue_output_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    samples: Arc<Vec<f32>>,
    playback_cursor: Arc<AtomicUsize>,
    playback_paused: Arc<AtomicBool>,
    err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream>
where
    T: Sample + SizedSample + FromSample<f32>,
{
    device
        .build_output_stream(
            config,
            move |output: &mut [T], _| {
                for out in output.iter_mut() {
                    if playback_paused.load(Ordering::Relaxed) {
                        *out = T::from_sample(0.0);
                        continue;
                    }
                    let idx = playback_cursor.fetch_add(1, Ordering::Relaxed);
                    let sample = samples.get(idx).copied().unwrap_or(0.0);
                    *out = T::from_sample(sample);
                }
            },
            err_fn,
            None,
        )
        .context("failed to build output stream")
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn select_continue_output_config(
    device: &cpal::Device,
    sample_rate: u32,
    channels: u16,
) -> Result<SupportedStreamConfigRange> {
    let mut candidates = device
        .supported_output_configs()
        .context("failed to list output stream configs")?;
    candidates
        .find(|config| {
            config.channels() == channels
                && config.min_sample_rate().0 <= sample_rate
                && config.max_sample_rate().0 >= sample_rate
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no output stream supports {} Hz, {} channel(s)",
                sample_rate,
                channels
            )
        })
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn continue_playback_duration(total_samples: usize, sample_rate: u32, channels: u16) -> Duration {
    if sample_rate == 0 || channels == 0 {
        return Duration::ZERO;
    }
    let frames = total_samples as f64 / f64::from(channels);
    Duration::from_secs_f64(frames / f64::from(sample_rate))
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn build_initial_prompt(prompt_words: &[String]) -> String {
    let seed = if prompt_words.is_empty() {
        DEFAULT_CONTINUE_PROMPT.to_string()
    } else {
        prompt_words.join(" ")
    };
    format!("{seed}\n\n{LIVE_EVENT_INSTRUCTIONS}\n\n")
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
#[derive(Debug, Default)]
struct SpeechEventDetector {
    pending: String,
    in_speech: bool,
    next_utterance_id: u64,
    current_utterance_id: Option<u64>,
    current_utterance_content: String,
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
#[derive(Debug, Clone, PartialEq, Eq)]
enum ContinueRuntimeEvent {
    UtteranceStarted { id: u64 },
    UtteranceCompleted { id: u64, content: String },
    SpeechControl { command: SpeechControlCommand },
    SourceCommand { command: SourceCommand },
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpeechControlCommand {
    Shutup,
    Pause,
    Resume,
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
#[derive(Debug, Clone, PartialEq, Eq)]
enum SourceCommand {
    ListFiles,
    ViewFile { path: String, page: usize },
}

#[cfg(test)]
impl ContinueRuntimeEvent {
    fn to_message(&self) -> String {
        match self {
            Self::UtteranceStarted { id } => {
                format!("utterance_started: id={id}")
            }
            Self::UtteranceCompleted { id, content } => {
                format!(
                    "utterance_completed: id={id}\ncontent:\n{}",
                    sanitize_runtime_event_content(content)
                )
            }
            Self::SpeechControl { command } => format!("speech_control: {}", command.as_str()),
            Self::SourceCommand { command } => match command {
                SourceCommand::ListFiles => "source_command: list_files".to_string(),
                SourceCommand::ViewFile { path, page } => {
                    format!(
                        "source_command: view_file\npath: {}\npage: {page}",
                        sanitize_runtime_event_content(path)
                    )
                }
            },
        }
    }
}

#[cfg(test)]
impl SpeechControlCommand {
    fn as_str(self) -> &'static str {
        match self {
            Self::Shutup => "shutup",
            Self::Pause => "pause",
            Self::Resume => "resume",
        }
    }
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
impl SpeechEventDetector {
    fn defers_live_events(&self) -> bool {
        self.in_speech
            || longest_marker_prefix_suffix_len(&self.pending) > 0
            || incomplete_source_tag_start(&self.pending).is_some()
    }

    fn ingest(&mut self, text: &str) -> Vec<ContinueRuntimeEvent> {
        self.pending.push_str(text);
        let mut events = Vec::new();

        loop {
            if self.in_speech {
                let Some(next_marker) = next_any_speech_marker(&self.pending) else {
                    self.commit_pending_speech_text_before_marker_prefix(&mut events);
                    return events;
                };
                let speech_text = self.pending[..next_marker.index].to_string();
                let marker_end = next_marker.index + next_marker.len;
                self.pending.drain(..marker_end);

                self.append_speech_text(&speech_text, &mut events);
                match next_marker.kind {
                    SpeechMarkerKind::Start => {
                        if let Some(event) = self.flush_current_utterance() {
                            events.push(event);
                        }
                        events.push(self.start_utterance());
                    }
                    SpeechMarkerKind::End => {
                        if let Some(event) = self.flush_current_utterance() {
                            events.push(event);
                        }
                    }
                    SpeechMarkerKind::Control(command) => {
                        events.push(ContinueRuntimeEvent::SpeechControl { command });
                    }
                    SpeechMarkerKind::Source(command) => {
                        events.push(ContinueRuntimeEvent::SourceCommand { command });
                    }
                }
            } else {
                let Some(next_marker) = next_any_open_speech_marker(&self.pending) else {
                    self.trim_pending_to_marker_prefix_or_source_tag();
                    return events;
                };
                let marker_end = next_marker.index + next_marker.len;
                self.pending.drain(..marker_end);
                match next_marker.kind {
                    SpeechMarkerKind::Start => events.push(self.start_utterance()),
                    SpeechMarkerKind::Control(command) => {
                        events.push(ContinueRuntimeEvent::SpeechControl { command });
                    }
                    SpeechMarkerKind::Source(command) => {
                        events.push(ContinueRuntimeEvent::SourceCommand { command });
                    }
                    SpeechMarkerKind::End => {}
                }
            }
        }
    }

    fn start_utterance(&mut self) -> ContinueRuntimeEvent {
        self.in_speech = true;
        self.current_utterance_content.clear();
        self.open_utterance()
    }

    fn open_utterance(&mut self) -> ContinueRuntimeEvent {
        let id = self.next_utterance_id;
        self.next_utterance_id += 1;
        self.current_utterance_id = Some(id);
        ContinueRuntimeEvent::UtteranceStarted { id }
    }

    fn append_speech_text(&mut self, text: &str, events: &mut Vec<ContinueRuntimeEvent>) {
        if text.is_empty() {
            return;
        }
        self.ensure_utterance_started_if_needed(text, events);
        self.current_utterance_content.push_str(text);
        self.emit_completed_sentences(events);
    }

    fn ensure_utterance_started_if_needed(
        &mut self,
        text: &str,
        events: &mut Vec<ContinueRuntimeEvent>,
    ) {
        if self.current_utterance_id.is_none() && !text.trim().is_empty() {
            events.push(self.start_utterance());
        }
    }

    fn emit_completed_sentences(&mut self, events: &mut Vec<ContinueRuntimeEvent>) {
        while let Some(end) = seams_sentence_end(&self.current_utterance_content) {
            let sentence = self.current_utterance_content[..end].trim().to_string();
            self.current_utterance_content.drain(..end);

            if let Some(id) = self.current_utterance_id.take() {
                events.push(ContinueRuntimeEvent::UtteranceCompleted {
                    id,
                    content: sentence,
                });
            }

            let leading_whitespace = self.current_utterance_content.len()
                - self.current_utterance_content.trim_start().len();
            if leading_whitespace > 0 {
                self.current_utterance_content.drain(..leading_whitespace);
            }
            if !self.current_utterance_content.trim().is_empty() {
                events.push(self.open_utterance());
            }
        }
    }

    fn flush_current_utterance(&mut self) -> Option<ContinueRuntimeEvent> {
        self.in_speech = false;
        let id = self.current_utterance_id.take()?;
        let content = self.current_utterance_content.trim().to_string();
        self.current_utterance_content.clear();
        if content.is_empty() {
            return None;
        }
        Some(ContinueRuntimeEvent::UtteranceCompleted { id, content })
    }

    fn commit_pending_speech_text_before_marker_prefix(
        &mut self,
        events: &mut Vec<ContinueRuntimeEvent>,
    ) {
        let keep = longest_marker_prefix_suffix_len(&self.pending);
        let emit_len = self.pending.len() - keep;
        let speech_text = self.pending[..emit_len].to_string();
        self.pending = self.pending[emit_len..].to_string();
        self.append_speech_text(&speech_text, events);
    }

    fn trim_pending_to_marker_prefix_or_source_tag(&mut self) {
        if let Some(start) = incomplete_source_tag_start(&self.pending) {
            self.pending = self.pending[start..].to_string();
            return;
        }
        let keep = longest_marker_prefix_suffix_len(&self.pending);
        if keep == 0 {
            self.pending.clear();
        } else {
            let start = self.pending.len() - keep;
            self.pending = self.pending[start..].to_string();
        }
    }
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
#[derive(Debug, Clone, PartialEq, Eq)]
enum SpeechMarkerKind {
    Start,
    End,
    Control(SpeechControlCommand),
    Source(SourceCommand),
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct SpeechMarker {
    kind: SpeechMarkerKind,
    index: usize,
    len: usize,
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn next_any_speech_marker(text: &str) -> Option<SpeechMarker> {
    [
        next_speech_marker(text, SpeechMarkerKind::Start),
        next_speech_marker(text, SpeechMarkerKind::End),
        next_speech_marker(
            text,
            SpeechMarkerKind::Control(SpeechControlCommand::Shutup),
        ),
        next_speech_marker(text, SpeechMarkerKind::Control(SpeechControlCommand::Pause)),
        next_speech_marker(
            text,
            SpeechMarkerKind::Control(SpeechControlCommand::Resume),
        ),
        next_source_marker(text),
    ]
    .into_iter()
    .flatten()
    .min_by_key(|marker| marker.index)
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn next_any_open_speech_marker(text: &str) -> Option<SpeechMarker> {
    [
        next_speech_marker(text, SpeechMarkerKind::Start),
        next_speech_marker(
            text,
            SpeechMarkerKind::Control(SpeechControlCommand::Shutup),
        ),
        next_speech_marker(text, SpeechMarkerKind::Control(SpeechControlCommand::Pause)),
        next_speech_marker(
            text,
            SpeechMarkerKind::Control(SpeechControlCommand::Resume),
        ),
        next_source_marker(text),
    ]
    .into_iter()
    .flatten()
    .min_by_key(|marker| marker.index)
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn next_source_marker(text: &str) -> Option<SpeechMarker> {
    let list_marker = text
        .find(SOURCE_LIST_FILES_MARKER)
        .map(|index| SpeechMarker {
            kind: SpeechMarkerKind::Source(SourceCommand::ListFiles),
            index,
            len: SOURCE_LIST_FILES_MARKER.len(),
        });

    [
        list_marker,
        next_view_source_marker(text, SOURCE_VIEW_FILE_START),
        next_view_source_marker(text, SOURCE_READ_FILE_START),
        next_view_source_marker(text, SOURCE_READ_SOURCE_FILE_START),
    ]
    .into_iter()
    .flatten()
    .min_by_key(|marker| marker.index)
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn next_view_source_marker(text: &str, start_marker: &'static str) -> Option<SpeechMarker> {
    let start = text.find(start_marker)?;
    let tag = &text[start..];
    let end_rel = tag.find("/>")?;
    let raw_tag = &tag[..end_rel + 2];
    let attrs = &raw_tag[start_marker.len()..raw_tag.len() - 2];
    let path = tag_attribute(attrs, "path").or_else(|| tag_attribute(attrs, "file"))?;
    let page = tag_attribute(attrs, "page")
        .and_then(|page| page.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    Some(SpeechMarker {
        kind: SpeechMarkerKind::Source(SourceCommand::ViewFile { path, page }),
        index: start,
        len: end_rel + 2,
    })
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn tag_attribute(attrs: &str, key: &str) -> Option<String> {
    let needle = format!("{key}=");
    let start = attrs.find(&needle)? + needle.len();
    let rest = attrs[start..].trim_start();
    let quote = rest.chars().next()?;
    if quote == '"' || quote == '\'' {
        let value = &rest[quote.len_utf8()..];
        let end = value.find(quote)?;
        return Some(value[..end].to_string());
    }
    let end = rest
        .find(|ch: char| ch.is_whitespace() || ch == '/')
        .unwrap_or(rest.len());
    Some(rest[..end].to_string())
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn incomplete_source_tag_start(text: &str) -> Option<usize> {
    [
        SOURCE_VIEW_FILE_START,
        SOURCE_READ_FILE_START,
        SOURCE_READ_SOURCE_FILE_START,
    ]
    .into_iter()
    .filter_map(|start_marker| {
        let start = text.rfind(start_marker)?;
        let rest = &text[start..];
        if rest.contains("/>") {
            None
        } else {
            Some(start)
        }
    })
    .min()
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn next_speech_marker(text: &str, kind: SpeechMarkerKind) -> Option<SpeechMarker> {
    speech_markers(kind.clone())
        .into_iter()
        .filter_map(|marker| {
            text.find(marker).map(|index| SpeechMarker {
                kind: kind.clone(),
                index,
                len: marker.len(),
            })
        })
        .min_by_key(|marker| marker.index)
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn speech_markers(kind: SpeechMarkerKind) -> [&'static str; 1] {
    match kind {
        SpeechMarkerKind::Start => [INLINE_SPEECH_START_MARKER],
        SpeechMarkerKind::End => [INLINE_SPEECH_END_MARKER],
        SpeechMarkerKind::Control(SpeechControlCommand::Shutup) => [SPEECH_SHUTUP_MARKER],
        SpeechMarkerKind::Control(SpeechControlCommand::Pause) => [SPEECH_PAUSE_MARKER],
        SpeechMarkerKind::Control(SpeechControlCommand::Resume) => [SPEECH_RESUME_MARKER],
        SpeechMarkerKind::Source(_) => unreachable!("source markers are variable-length"),
    }
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn all_speech_markers() -> [&'static str; 9] {
    [
        INLINE_SPEECH_START_MARKER,
        INLINE_SPEECH_END_MARKER,
        SPEECH_SHUTUP_MARKER,
        SPEECH_PAUSE_MARKER,
        SPEECH_RESUME_MARKER,
        SOURCE_LIST_FILES_MARKER,
        SOURCE_VIEW_FILE_START,
        SOURCE_READ_FILE_START,
        SOURCE_READ_SOURCE_FILE_START,
    ]
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn longest_marker_prefix_suffix_len(text: &str) -> usize {
    all_speech_markers()
        .into_iter()
        .flat_map(|marker| {
            marker
                .char_indices()
                .skip(1)
                .map(|(index, _)| index)
                .chain(std::iter::once(marker.len()))
                .filter(|&len| len <= text.len())
                .filter(|&len| text.ends_with(&marker[..len]))
        })
        .max()
        .unwrap_or(0)
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn sentence_detector() -> &'static SentenceDetectorDialog {
    static DETECTOR: OnceLock<SentenceDetectorDialog> = OnceLock::new();
    DETECTOR.get_or_init(|| {
        SentenceDetectorDialog::new().expect("failed to initialize seams sentence detector")
    })
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn seams_sentence_end(text: &str) -> Option<usize> {
    let detector = sentence_detector();
    let sentences = detector
        .detect_sentences_borrowed(text)
        .expect("failed to split speech with seams");
    let mut search_from = 0;
    for sentence in sentences {
        if let Some(rel) = text[search_from..].find(sentence.raw_content) {
            let start = search_from + rel;
            let end = start + sentence.raw_content.len();
            search_from = end;
            if sentence.raw_content.trim().ends_with(['.', '?', '!']) {
                return Some(end);
            }
        }
    }
    None
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
#[allow(dead_code)]
fn sanitize_runtime_event_content(content: &str) -> String {
    content
        .replace("--- END LIVE EVENT ---", "[end live event]")
        .replace("--- LIVE EVENT:", "[live event]")
}

#[cfg(test)]
mod tests {
    use super::{
        ContinueRuntimeEvent, PromptPacket, RollingContextManager, SourceCommand,
        SpeechControlCommand, SpeechEventDetector, build_initial_prompt, execute_list_source_files,
        execute_view_source_file, wrap_ear_event, wrap_live_input, wrap_mouth_event,
        wrap_runtime_event, wrap_source_event, wrap_time_event,
    };

    #[test]
    fn stdin_append_is_wrapped_as_live_input() {
        assert_eq!(
            wrap_live_input("turn toward the window\n"),
            "\n\n--- LIVE EVENT: user ---\nturn toward the window\n--- END LIVE EVENT ---\n\n"
        );
    }

    #[test]
    fn time_append_is_wrapped_as_live_input() {
        assert_eq!(
            wrap_time_event("The current Unix time is 42.000 seconds."),
            "\n\n--- LIVE EVENT: clock ---\nThe current Unix time is 42.000 seconds.\n--- END LIVE EVENT ---\n\n"
        );
    }

    #[test]
    fn heard_speech_is_wrapped_as_live_input() {
        assert_eq!(
            wrap_ear_event("Heard: hello from the room"),
            "\n\n--- LIVE EVENT: ear ---\nHeard: hello from the room\n--- END LIVE EVENT ---\n\n"
        );
    }

    #[test]
    fn mouth_event_is_wrapped_as_live_input() {
        assert_eq!(
            wrap_mouth_event("speech_playback_completed: id=3"),
            "\n\n--- LIVE EVENT: mouth ---\nspeech_playback_completed: id=3\n--- END LIVE EVENT ---\n\n"
        );
    }

    #[test]
    fn runtime_event_is_wrapped_as_live_input() {
        assert_eq!(
            wrap_runtime_event("utterance_started: id=0"),
            "\n\n--- LIVE EVENT: runtime ---\nutterance_started: id=0\n--- END LIVE EVENT ---\n\n"
        );
        assert_eq!(
            ContinueRuntimeEvent::UtteranceStarted { id: 7 }.to_message(),
            "utterance_started: id=7"
        );
    }

    #[test]
    fn source_event_is_wrapped_as_live_input() {
        assert_eq!(
            wrap_source_event("Available source files:\nsrc/main.rs\n"),
            "\n\n--- LIVE EVENT: source ---\nAvailable source files:\nsrc/main.rs\n\n--- END LIVE EVENT ---\n\n"
        );
    }

    #[test]
    fn initial_prompt_includes_live_event_hygiene() {
        let prompt = build_initial_prompt(&["Think continuously.".to_string()]);

        assert!(prompt.starts_with("Think continuously.\n\n"));
        assert!(prompt.contains("Live events may appear in the transcript"));
        assert!(prompt.contains("Do not copy live event delimiters or runtime event text."));
        assert!(prompt.contains("Do not write system, assistant, analysis, channel"));
        assert!(prompt.contains("<sp>words to say aloud :)</sp>"));
        assert!(prompt.contains("event text is never inserted inside speech"));
        assert!(prompt.contains("<shutup/> immediately halts current speech"));
        assert!(prompt.contains("<pause/> pauses speech playback"));
        assert!(prompt.contains("<resume/> resumes paused speech"));
        assert!(prompt.contains("<list_files/> lists bundled source files"));
        assert!(prompt.contains("<view_file path=\"src/main.rs\" page=\"1\"/>"));
        assert!(!prompt.contains("--- SPEECH ---"));
        assert!(prompt.contains("Emoji inside speech tags are instructions to your countenance"));
    }

    #[test]
    fn rolling_context_keeps_persona_top_and_recent_turns_verbatim() {
        let mut context =
            RollingContextManager::new("Identity and rules.".to_string(), 4096, None, 2);

        context.record_prompt_packet(PromptPacket::listened("first heard turn".to_string()));
        context.record_prompt_packet(PromptPacket::spoken("first spoken turn".to_string()));
        context.record_prompt_packet(PromptPacket::listened("latest heard turn".to_string()));

        let prompt = context.prompt_body();
        assert!(prompt.starts_with("Identity and rules."));
        assert!(prompt.contains("<working_memory>"));
        assert!(prompt.contains("first heard turn"));
        assert!(prompt.contains("<recent_verbatim>"));
        let recent = prompt
            .split("<recent_verbatim>")
            .nth(1)
            .and_then(|text| text.split("</recent_verbatim>").next())
            .expect("recent verbatim page should be present");
        assert!(!recent.contains("Listened: first heard turn"));
        assert!(prompt.contains("Spoken: first spoken turn"));
        assert!(prompt.contains("Listened: latest heard turn"));
    }

    #[test]
    fn rolling_context_tracks_clock_as_scene_not_verbatim_turn() {
        let mut context = RollingContextManager::new("Identity.".to_string(), 4096, None, 4);

        context.record_prompt_packet(PromptPacket::clock(
            "The current Unix time is 42.000 seconds.".to_string(),
        ));

        let prompt = context.prompt_body();
        assert!(prompt.contains("<auditory_scene>"));
        assert!(prompt.contains("Clock: The current Unix time is 42.000 seconds."));
        assert!(prompt.contains("No listened/spoken turns yet."));
    }

    #[test]
    fn speech_detector_parses_inline_speech() {
        let mut detector = SpeechEventDetector::default();

        assert_eq!(
            detector
                .ingest("<sp>:) This is how I speak. Parse here. And here. And here...live</sp>"),
            vec![
                ContinueRuntimeEvent::UtteranceStarted { id: 0 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 0,
                    content: ":) This is how I speak.".to_string()
                },
                ContinueRuntimeEvent::UtteranceStarted { id: 1 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 1,
                    content: "Parse here.".to_string()
                },
                ContinueRuntimeEvent::UtteranceStarted { id: 2 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 2,
                    content: "And here.".to_string()
                },
                ContinueRuntimeEvent::UtteranceStarted { id: 3 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 3,
                    content: "And here...live".to_string()
                }
            ]
        );
    }

    #[test]
    fn speech_detector_handles_split_inline_marker() {
        let mut detector = SpeechEventDetector::default();

        assert!(detector.ingest("<s").is_empty());
        assert_eq!(
            detector.ingest("p>Hello</sp>"),
            vec![
                ContinueRuntimeEvent::UtteranceStarted { id: 0 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 0,
                    content: "Hello".to_string()
                }
            ]
        );
    }

    #[test]
    fn speech_detector_emits_utterance_started_on_marker() {
        let mut detector = SpeechEventDetector::default();

        assert_eq!(
            detector.ingest("thinking <sp>Hello"),
            vec![ContinueRuntimeEvent::UtteranceStarted { id: 0 }]
        );
    }

    #[test]
    fn speech_detector_handles_split_marker() {
        let mut detector = SpeechEventDetector::default();

        assert!(detector.ingest("thinking <s").is_empty());
        assert_eq!(
            detector.ingest("p>Hello"),
            vec![ContinueRuntimeEvent::UtteranceStarted { id: 0 }]
        );
    }

    #[test]
    fn speech_detector_rearms_after_end_marker() {
        let mut detector = SpeechEventDetector::default();

        assert_eq!(
            detector.ingest("<sp>First</sp>Later <sp>Second"),
            vec![
                ContinueRuntimeEvent::UtteranceStarted { id: 0 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 0,
                    content: "First".to_string()
                },
                ContinueRuntimeEvent::UtteranceStarted { id: 1 }
            ]
        );
    }

    #[test]
    fn speech_detector_emits_utterance_completed_on_end_marker() {
        let mut detector = SpeechEventDetector::default();

        assert_eq!(
            detector.ingest("<sp>Hello there.</sp>"),
            vec![
                ContinueRuntimeEvent::UtteranceStarted { id: 0 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 0,
                    content: "Hello there.".to_string()
                }
            ]
        );
    }

    #[test]
    fn speech_detector_treats_nested_start_as_recovery_boundary() {
        let mut detector = SpeechEventDetector::default();

        assert_eq!(
            detector.ingest("<sp>Hello<sp>what happens up here?"),
            vec![
                ContinueRuntimeEvent::UtteranceStarted { id: 0 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 0,
                    content: "Hello".to_string()
                },
                ContinueRuntimeEvent::UtteranceStarted { id: 1 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 1,
                    content: "what happens up here?".to_string()
                }
            ]
        );
    }

    #[test]
    fn speech_detector_emits_complete_sentences_from_head_before_end_marker() {
        let mut detector = SpeechEventDetector::default();

        assert_eq!(
            detector.ingest("<sp>First sentence. Second"),
            vec![
                ContinueRuntimeEvent::UtteranceStarted { id: 0 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 0,
                    content: "First sentence.".to_string()
                },
                ContinueRuntimeEvent::UtteranceStarted { id: 1 }
            ]
        );
        assert!(detector.ingest(" sentence").is_empty());
        assert_eq!(
            detector.ingest(".</sp>"),
            vec![ContinueRuntimeEvent::UtteranceCompleted {
                id: 1,
                content: "Second sentence.".to_string()
            }]
        );
    }

    #[test]
    fn speech_detector_captures_content_across_chunks() {
        let mut detector = SpeechEventDetector::default();

        assert_eq!(
            detector.ingest("<sp>Hello "),
            vec![ContinueRuntimeEvent::UtteranceStarted { id: 0 }]
        );
        assert_eq!(
            detector.ingest("there.</sp>"),
            vec![ContinueRuntimeEvent::UtteranceCompleted {
                id: 0,
                content: "Hello there.".to_string()
            }]
        );
    }

    #[test]
    fn speech_detector_ignores_legacy_block_delimiters() {
        let mut detector = SpeechEventDetector::default();

        assert!(
            detector
                .ingest("--- SPEECH ---no fallback--- END SPEECH ---")
                .is_empty()
        );
    }

    #[test]
    fn speech_detector_defers_live_events_inside_speech() {
        let mut detector = SpeechEventDetector::default();

        assert!(!detector.defers_live_events());
        assert_eq!(
            detector.ingest("<sp>Hello"),
            vec![ContinueRuntimeEvent::UtteranceStarted { id: 0 }]
        );
        assert!(detector.defers_live_events());
        assert_eq!(
            detector.ingest("</sp>"),
            vec![ContinueRuntimeEvent::UtteranceCompleted {
                id: 0,
                content: "Hello".to_string()
            }]
        );
        assert!(!detector.defers_live_events());
    }

    #[test]
    fn speech_detector_defers_live_events_during_partial_markers() {
        let mut detector = SpeechEventDetector::default();

        assert!(detector.ingest("<pau").is_empty());
        assert!(detector.defers_live_events());
        assert_eq!(
            detector.ingest("se/>"),
            vec![ContinueRuntimeEvent::SpeechControl {
                command: SpeechControlCommand::Pause
            }]
        );
        assert!(!detector.defers_live_events());
    }

    #[test]
    fn speech_detector_parses_source_file_tags() {
        let mut detector = SpeechEventDetector::default();

        assert_eq!(
            detector.ingest(
                "inspect <list_files/> then <view_file path=\"src/cli/commands/continue_generation.rs\" page=\"2\"/>"
            ),
            vec![
                ContinueRuntimeEvent::SourceCommand {
                    command: SourceCommand::ListFiles
                },
                ContinueRuntimeEvent::SourceCommand {
                    command: SourceCommand::ViewFile {
                        path: "src/cli/commands/continue_generation.rs".to_string(),
                        page: 2
                    }
                }
            ]
        );
    }

    #[test]
    fn speech_detector_defers_live_events_during_partial_source_tag() {
        let mut detector = SpeechEventDetector::default();

        assert!(
            detector
                .ingest("<view_file path=\"src/main.rs\"")
                .is_empty()
        );
        assert!(detector.defers_live_events());
        assert_eq!(
            detector.ingest(" page=\"1\"/>"),
            vec![ContinueRuntimeEvent::SourceCommand {
                command: SourceCommand::ViewFile {
                    path: "src/main.rs".to_string(),
                    page: 1
                }
            }]
        );
        assert!(!detector.defers_live_events());
    }

    #[test]
    fn source_bundle_lists_and_views_files() {
        let files = execute_list_source_files();
        assert!(files.contains("src/cli/commands/continue_generation.rs"));

        let page = execute_view_source_file("src/cli/commands/continue_generation.rs", 1);
        assert!(page.contains("--- src/cli/commands/continue_generation.rs"));
        assert!(page.contains("use crate::cli::ContinueCommand;"));
    }

    #[test]
    fn speech_detector_parses_self_closing_speech_controls_outside_speech() {
        let mut detector = SpeechEventDetector::default();

        assert_eq!(
            detector.ingest("thinking <pause/> then <resume/> and <shutup/>"),
            vec![
                ContinueRuntimeEvent::SpeechControl {
                    command: SpeechControlCommand::Pause
                },
                ContinueRuntimeEvent::SpeechControl {
                    command: SpeechControlCommand::Resume
                },
                ContinueRuntimeEvent::SpeechControl {
                    command: SpeechControlCommand::Shutup
                }
            ]
        );
    }

    #[test]
    fn speech_detector_parses_self_closing_speech_controls_inside_speech() {
        let mut detector = SpeechEventDetector::default();

        assert_eq!(
            detector.ingest("<sp>Hello. <pause/>Wait here. <resume/>Go now.</sp>"),
            vec![
                ContinueRuntimeEvent::UtteranceStarted { id: 0 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 0,
                    content: "Hello.".to_string()
                },
                ContinueRuntimeEvent::SpeechControl {
                    command: SpeechControlCommand::Pause
                },
                ContinueRuntimeEvent::UtteranceStarted { id: 1 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 1,
                    content: "Wait here.".to_string()
                },
                ContinueRuntimeEvent::SpeechControl {
                    command: SpeechControlCommand::Resume
                },
                ContinueRuntimeEvent::UtteranceStarted { id: 2 },
                ContinueRuntimeEvent::UtteranceCompleted {
                    id: 2,
                    content: "Go now.".to_string()
                }
            ]
        );
    }
}
