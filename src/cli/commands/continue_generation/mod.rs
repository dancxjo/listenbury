use crate::cli::ContinueCommand;
use anyhow::Result;

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use crate::cli::commands::llama::build_prompt;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use crate::cli::commands::mic_transcribe::transcribe_group_with_finality;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use crate::cli::commands::prepare_audio_playback;
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
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use chrono::{Local, SecondsFormat};
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
use cpal::{FromSample, Sample, SizedSample};
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use listenbury::ExactTimestamp;
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use listenbury::VadBackendKind;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::WhisperSpeechRecognizer;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::audio::capture::{
    boost_current_thread_for_capture, callback_sample_queue_capacity,
};
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
use listenbury::hearing::environment::EnvironmentalSoundClip;
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use listenbury::hearing::environment::{EnvironmentalSound, EnvironmentalSoundObserver};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::hearing::vad::{VadResult, VoiceActivityDetector, create_vad_backend};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::hearing::{
    AuditoryFrameAnalysis, AuditoryRouting, AuditorySceneAnalyzer, SpeakerReferenceMask,
};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::live_trace::{
    DiskTraceWriter, JsonlTraceWriter, LiveTraceRecorder, SessionId, SseBroadcaster, TeeSink,
    TraceRuntimeMetadata, TraceSessionMetadata,
};
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use listenbury::mind::llm::LlmEvent;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::mind::llm::{GenerationId, GenerationRequest, LlmEngine};
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use listenbury::mouth::planner::strip_emoji;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::mouth::planner::{SpeechPlan, SpeechUnit};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::mouth::tts::TextToSpeech;
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use listenbury::word::{TimedWordStream, WordStreamId, WordStreamSource};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::word::{
    TranscriptWord, transcript_to_energy_snapped_word_stream, transcript_to_word_stream,
};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::{AudioFrame, LlamaCppConfig, LlamaCppEngine, PiperTextToSpeech};
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use serde::{Deserialize, Serialize};
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use serde_json::{Value, json};
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
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use std::sync::{Arc, Mutex};
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use std::thread::JoinHandle;
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use std::time::{Duration, Instant};
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use tsrun::{
    Guarded, InternalModule, Interpreter, InterpreterConfig, JsError, JsValue, StepResult, api,
    js_value_to_json,
};

mod ear;
mod mouth;
mod prompt;
mod source;
mod trace;

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use ear::TranscriptStabilityState;
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use ear::{ContinueEarEvent, TranscriptSpeculativePlanner};
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use mouth::{ContinueMouthCommand, mouth_command_for_runtime_event};
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use prompt::ContinuePromptGate;
#[cfg(test)]
use prompt::ContinuePromptGateConfig;
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use source::SourceCommand;
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use source::{
    SourceCommandExecution, execute_grep_source, execute_list_source_files, execute_search_source,
    execute_source_command, execute_view_source_file,
};
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use trace::{
    current_time_message, next_time_event_interval, wrap_ear_event, wrap_live_input,
    wrap_source_event, wrap_time_event,
};
#[cfg(test)]
use trace::{wrap_mouth_event, wrap_runtime_event};

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
const LIVE_EVENT_INSTRUCTIONS: &str = "Live events may appear in the transcript while you are generating.\nTreat them as observations from outside.\nDo not assume a user is currently present; there may be nobody in the room or nobody addressing you.\nClock events arrive frequently, about once per second but at slightly irregular intervals, with local ISO-8601 time and timezone offset so you can track timing, pauses, and elapsed time.\nDo not copy live event delimiters or runtime event text.\nDo not write system, assistant, analysis, channel, message, thoughts, or template tokens.\nContinue naturally as Pete.\nPlain generated text is Pete's internal thought only. It is not spoken aloud. It does not happen in the real world. It is private internal monologue inside the system.\nThe only way to affect the real world is to run small TypeScript modules with <ts>code</ts>.\nTypeScript runs through tsrun with only the internal module \"pete:will\" available; it cannot use arbitrary imports, filesystem, network, or processes. Import the builders you need from \"pete:will\", for example: import { say, listFiles } from \"pete:will\";. Make the final expression a command object or array from these builders: say(text, options?), shutup(), pause(), resume(), listFiles(), readSourceFile(path, page?), readFile(path, page?), searchSource(query, limit?), grepSource(pattern, limit?).\nUse say(text) for words the user should hear. If speech should intentionally talk over active user speech, use say(text, { interrupt: true }); otherwise TTS waits for VAD to clear before starting. Speak sparingly: after you say something, leave room for the interlocutor to answer instead of immediately saying more. Do not narrate every clock tick, quiet moment, or idle thought aloud. If nobody is present or addressing you, prefer internal thought and do not speak just to fill silence.\nIf you are bored, alone, or waiting for something to happen, you may explore Pete's own source code with listFiles(), readSourceFile(path, page?), searchSource(query, limit?), or grepSource(pattern, limit?) instead of speaking into silence.\nUse shutup() to halt current speech and clear queued speech, pause() to pause playback, and resume() to resume paused playback.\nTypeScript source and command results are reported back as live source events. Use TypeScript tags outside speech.";
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const SOURCE_TYPESCRIPT_START: &str = "<ts>";
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const SOURCE_TYPESCRIPT_END: &str = "</ts>";
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
const WEBRTC_VAD_SAMPLE_RATE_HZ: u32 = 16_000;
#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
const MONO_CHANNELS: u16 = 1;
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const TIME_EVENT_INTERVAL_BASE_MS: u64 = 1_000;
#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const TIME_EVENT_INTERVAL_JITTER_MS: u64 = 300;
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
type ContinueLiveTrace =
    LiveTraceRecorder<TeeSink<Option<DiskTraceWriter>, Option<SseBroadcaster>>>;

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn continue_trace_session_metadata(
    session_id: SessionId,
    trace_started_at: ExactTimestamp,
    command: &crate::cli::ContinueCommand,
) -> TraceSessionMetadata {
    let mut runtime = TraceRuntimeMetadata::new("listenbury dev continue");
    runtime.mode = Some(if command.duplex_trace_scenario.is_some() {
        "duplex_trace_scenario".to_string()
    } else {
        "live_duplex".to_string()
    });
    runtime.configuration = serde_json::from_value(json!({
        "web": command.web,
        "web_host": command.web_host,
        "web_port": command.web_port,
        "vad": format!("{:?}", command.vad),
        "mode": format!("{:?}", command.mode),
        "context_size": command.context_size,
        "verbatim_turns": command.verbatim_turns,
        "max_tokens": command.max_tokens,
        "tts_vad_pause_ms": command.tts_vad_pause_ms,
        "tts_vad_listen_ms": command.tts_vad_listen_ms,
        "duplex_trace_scenario": command.duplex_trace_scenario.map(|value| format!("{value:?}")),
        "llm_model": command.llm_model.as_ref().map(|path| path.display().to_string()),
        "whisper_model": command.whisper_model.as_ref().map(|path| path.display().to_string()),
        "piper_bin": command.piper_bin.as_ref().map(|path| path.display().to_string()),
        "piper_voice": command.piper_voice.as_ref().map(|path| path.display().to_string()),
    }))
    .expect("continue trace runtime configuration should serialize to an object");
    TraceSessionMetadata::new(session_id, trace_started_at, runtime)
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn browser_host_for_bind_host(bind_host: &str) -> String {
    match bind_host {
        "0.0.0.0" => "127.0.0.1".to_string(),
        "::" => "[::1]".to_string(),
        _ => {
            let looks_like_ipv6 =
                bind_host.contains(':') && !bind_host.starts_with('[') && !bind_host.ends_with(']');
            if looks_like_ipv6 {
                format!("[{bind_host}]")
            } else {
                bind_host.to_string()
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
fn live_asr_text_to_word_stream(stream_id: WordStreamId, transcript: &str) -> TimedWordStream {
    // `transcribe_group` currently returns finalized text chunks only, without
    // per-word timing/confidence from the ASR backend. We still emit a
    // first-class TimedWordStream artifact and preserve metadata fields as
    // `None` when unavailable.
    let words = transcript
        .split_whitespace()
        .map(|word| TranscriptWord {
            text: word.to_string(),
            start_ms: None,
            end_ms: None,
            confidence: None,
        })
        .collect::<Vec<_>>();
    let mut stream = transcript_to_word_stream(stream_id, &words);
    stream.source = WordStreamSource::LiveAsr;
    stream
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn emit_live_asr_trace_events(
    trace: &mut ContinueLiveTrace,
    turn: u64,
    text: &str,
    stream: &TimedWordStream,
    occurred_at: ExactTimestamp,
) -> Result<()> {
    let mut transcript_event = trace.event(turn, "transcript", occurred_at);
    transcript_event.text = Some(text.to_string());
    trace.emit(transcript_event)?;

    let mut stream_event = trace.event(turn, "asr_timed_word_stream", occurred_at);
    stream_event.artifact =
        Some(serde_json::to_value(stream).context("serialize ASR TimedWordStream artifact")?);
    trace.emit(stream_event)
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn emit_live_asr_candidate_trace_event(
    trace: &mut ContinueLiveTrace,
    turn: u64,
    event: &listenbury::speech::transcript::TranscriptCandidateEvent,
    stability: Option<&TranscriptStabilityState>,
    occurred_at: ExactTimestamp,
) -> Result<()> {
    let mut candidate_event = trace.event(turn, "transcript_candidate", occurred_at);
    candidate_event.text = Some(match event {
        listenbury::speech::transcript::TranscriptCandidateEvent::CandidateStarted { id } => {
            format!("candidate_started id={}", id.0)
        }
        listenbury::speech::transcript::TranscriptCandidateEvent::CandidateUpdated {
            id, ..
        } => {
            format!("candidate_updated id={}", id.0)
        }
        listenbury::speech::transcript::TranscriptCandidateEvent::CandidateReplaced {
            old,
            new,
            reason,
        } => format!(
            "candidate_replaced old={} new={} reason={reason:?}",
            old.0, new.0
        ),
        listenbury::speech::transcript::TranscriptCandidateEvent::CandidateFinalized {
            id, ..
        } => {
            format!("candidate_finalized id={}", id.0)
        }
        listenbury::speech::transcript::TranscriptCandidateEvent::CandidateCancelled { id } => {
            format!("candidate_cancelled id={}", id.0)
        }
    });
    if let Some(stability) = stability {
        candidate_event.artifact = Some(json!({
            "candidate_id": stability.candidate_id.0,
            "stable_text": stability.stable_text,
            "unstable_text": stability.unstable_text,
            "confidence": stability.confidence,
        }));
    }
    trace.emit(candidate_event)
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn emit_live_ear_trace_event(
    trace: &mut ContinueLiveTrace,
    turn: u64,
    event: &ContinueEarEvent,
) -> Result<()> {
    let mut trace_event = match event {
        ContinueEarEvent::ListeningStarted { .. } => {
            trace.event(0, "listening_started", ExactTimestamp::now())
        }
        ContinueEarEvent::SpeechStarted => {
            trace.event(turn, "speech_started", ExactTimestamp::now())
        }
        ContinueEarEvent::SpeechStopped => {
            trace.event(turn, "speech_stopped", ExactTimestamp::now())
        }
        ContinueEarEvent::AuditoryObservation { text } => {
            let mut trace_event = trace.event(turn, "auditory_observation", ExactTimestamp::now());
            trace_event.text = Some(text.clone());
            trace_event
        }
        ContinueEarEvent::EnvironmentalSound { sound } => {
            let mut trace_event = trace.event(turn, "environmental_sound", ExactTimestamp::now());
            trace_event.text = Some(sound.description.clone());
            trace_event
        }
        ContinueEarEvent::SelfVoiceHeard { .. } => {
            let mut trace_event = trace.event(turn, "self_voice_heard", ExactTimestamp::now());
            trace_event.text = Some(event.to_message());
            trace_event
        }
        ContinueEarEvent::OverlapDetected { .. } => {
            let mut trace_event = trace.event(turn, "overlap_detected", ExactTimestamp::now());
            trace_event.text = Some(event.to_message());
            trace_event
        }
        ContinueEarEvent::Error { message } => {
            let mut trace_event = trace.event(turn, "ear_error", ExactTimestamp::now());
            trace_event.text = Some(message.clone());
            trace_event
        }
        ContinueEarEvent::Transcript { .. } | ContinueEarEvent::TranscriptCandidate { .. } => {
            return Ok(());
        }
    };

    if trace_event.text.is_none() {
        trace_event.text = Some(event.to_message());
    }
    trace.emit(trace_event)
}

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
    if let Some(scenario) = command.duplex_trace_scenario {
        let jsonl_path = command
            .jsonl
            .as_deref()
            .context("--jsonl is required when --duplex-trace-scenario is enabled")?;
        let events = build_duplex_trace_scenario_events(
            scenario,
            Duration::from_millis(command.tts_vad_pause_ms),
            Duration::from_millis(command.tts_vad_listen_ms),
        );
        write_duplex_trace_scenario_jsonl(jsonl_path, &events)?;
        eprintln!(
            "dev continue duplex trace scenario {:?} wrote {} events to {}",
            scenario,
            events.len(),
            jsonl_path.display()
        );
        return Ok(());
    }

    let model_path = resolve_llm_model(command.llm_model.clone())?;
    let llm_placement = llm_runtime_placement(
        &model_path,
        command.llm_gpu_layers,
        DEFAULT_CONTINUE_LLAMA_GPU_LAYERS,
    )?;
    let prompt_format = continue_prompt_format_for_model(&model_path, command.mode);
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
        prompt_format,
        system_prompt,
        max_tokens,
        command.context_size,
        command.verbatim_turns,
    )
    .context("failed to start continued llama.cpp generation")?;
    let piper_bin = resolve_piper_bin(command.piper_bin.clone())?;
    let piper_voice = resolve_piper_voice(command.piper_voice.clone())?;
    let whisper_model = resolve_whisper_model(command.whisper_model.clone())?;
    let vad_backend = command.vad.as_backend_kind();
    let capture_enabled = Arc::new(AtomicBool::new(true));
    let speaker_reference = Arc::new(Mutex::new(SpeakerReferenceMask::default()));
    let trace_started_at = ExactTimestamp::now();
    let trace_session_id = SessionId::new();
    let trace_writer = command
        .jsonl
        .as_deref()
        .map(|path| {
            DiskTraceWriter::create(
                path,
                continue_trace_session_metadata(trace_session_id, trace_started_at, &command),
            )
        })
        .transpose()
        .context("failed to create dev continue live trace writer")?;
    let live_audio = command
        .web
        .then(listenbury::web::LiveSessionAudioStore::new);
    let (browser_audio_tx, browser_audio_rx) = if command.web {
        let (tx, rx) = crossbeam_channel::bounded::<AudioFrame>(128);
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };
    let broadcaster = if command.web {
        let bc = SseBroadcaster::new();
        let server_bc = bc.clone();
        let bind_host = command.web_host.clone();
        let server = listenbury::web::bind(listenbury::web::ServeConfig {
            host: bind_host.clone(),
            port: command.web_port,
            payload: None,
            trace: None,
            broadcaster: Some(server_bc),
            live_audio: live_audio.clone(),
            input_control: listenbury::web::WebInputControl::new(
                Some(Arc::clone(&capture_enabled)),
                browser_audio_tx,
            ),
        })
        .context("failed to start embedded web viewer")?;
        let web_port = server.local_addr().port();
        let browser_host = browser_host_for_bind_host(&bind_host);
        let url = format!("http://{}:{}/", browser_host, web_port);
        std::thread::spawn(move || {
            if let Err(e) = server.serve() {
                eprintln!("embedded web server error: {e:#}");
            }
        });
        println!("Listenbury web viewer available at {url}");
        Some(bc)
    } else {
        None
    };
    let mut live_trace = LiveTraceRecorder::with_session_id(
        trace_session_id,
        trace_started_at,
        TeeSink(trace_writer, broadcaster),
    );
    let mut live_trace_turn = 0u64;
    live_trace.emit_now(0, "capture_started", ExactTimestamp::now())?;
    let (mut mouth, mouth_rx) = ContinueMouth::start(
        PiperTextToSpeech::new(piper_config_for_voice(piper_bin, piper_voice)?),
        Arc::clone(&capture_enabled),
        Arc::clone(&speaker_reference),
    )?;
    let (_ear, ear_rx) = ContinueEar::start(ContinueEarConfig {
        whisper_model,
        vad_backend,
        capture_enabled: Arc::clone(&capture_enabled),
        speaker_reference,
        live_audio,
        browser_audio_rx,
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
        "listenbury dev continue: streaming one generation; stdin lines, mic transcripts, and jittered ~1s time events append to the live context. Ctrl-C cancels."
    );

    let mut cancelled = false;
    let mut time_event_jitter_state = initial_time_event_jitter_state();
    let mut next_time_event_at =
        Instant::now() + next_time_event_interval(&mut time_event_jitter_state);
    let mut speech_events = SpeechEventDetector::default();
    let mut harmony_filter = llm_session.uses_harmony().then(HarmonyFinalFilter::default);
    let mut pending_mouth_utterances = 0usize;
    let mut llm_paused_for_mouth = false;
    let mut mouth_playback_paused = false;
    let mut deferred_live_events = VecDeque::<PromptPacket>::new();
    let mut prompt_gate = ContinuePromptGate::default();
    let mut tts_vad = DuplexTurnController::new(DuplexTurnControllerConfig {
        pause_after: Duration::from_millis(command.tts_vad_pause_ms),
        listen_for: Duration::from_millis(command.tts_vad_listen_ms),
    });
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
            &mut time_event_jitter_state,
            speech_events.defers_live_events(),
            &mut deferred_live_events,
            &mut mouth,
            &mut tts_vad,
            &mut prompt_gate,
            &mut live_trace,
            &mut live_trace_turn,
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
            if let LlmEvent::Token { text } = event {
                llm_session.record_generated_text(text);
            }
        }
        let visible_events = if let Some(filter) = &mut harmony_filter {
            filter.filter_events(&events)
        } else {
            events.clone()
        };

        for event in &visible_events {
            match event {
                LlmEvent::Token { text } => {
                    if !llm_session.uses_harmony() {
                        print!("{text}");
                        std::io::stdout().flush()?;
                    }
                    for speech_event in speech_events.ingest(text) {
                        if let ContinueRuntimeEvent::UtteranceCompleted { content, .. } =
                            &speech_event
                        {
                            if clean_spoken_content(content).is_some() {
                                llm_session.remember_spoken(content);
                            }
                        }
                        if let ContinueRuntimeEvent::SourceCommand { command } = &speech_event {
                            let source_result = execute_source_command(command);
                            eprintln!("[dev continue] source result:\n{}", source_result.message);
                            if !generation_terminal {
                                append_or_defer_live_event(
                                    &mut llm_session,
                                    PromptPacket::source(source_result.message.clone()),
                                    speech_events.defers_live_events(),
                                    &mut deferred_live_events,
                                    "failed to append source event to live generation",
                                )?;
                            }
                            for runtime_event in source_result.runtime_events {
                                if let ContinueRuntimeEvent::UtteranceCompleted {
                                    content, ..
                                } = &runtime_event
                                {
                                    if clean_spoken_content(content).is_some() {
                                        llm_session.remember_spoken(content);
                                    }
                                }
                                prepare_tts_runtime_event(
                                    &runtime_event,
                                    &mut mouth,
                                    &mut tts_vad,
                                    &mut llm_session,
                                    speech_events.defers_live_events(),
                                    &mut deferred_live_events,
                                )?;
                                if mouth.enqueue_runtime_event(&runtime_event)? {
                                    pending_mouth_utterances += 1;
                                }
                            }
                        }
                        prepare_tts_runtime_event(
                            &speech_event,
                            &mut mouth,
                            &mut tts_vad,
                            &mut llm_session,
                            speech_events.defers_live_events(),
                            &mut deferred_live_events,
                        )?;
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
            if !cancelled {
                llm_session
                    .start_with_compact_prompt()
                    .context("failed to restart continued llama.cpp generation")?;
                harmony_filter = llm_session.uses_harmony().then(HarmonyFinalFilter::default);
                continue;
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

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn initial_time_event_jitter_state() -> u64 {
    let now = Local::now();
    let nanos = now
        .timestamp_nanos_opt()
        .unwrap_or_else(|| now.timestamp_millis().saturating_mul(1_000_000));
    let seed = nanos as u64 ^ u64::from(std::process::id());
    if seed == 0 {
        0x9e37_79b9_7f4a_7c15
    } else {
        seed
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContinuePromptFormat {
    Legacy(crate::cli::PromptMode),
    GptOssHarmony,
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
fn continue_prompt_format_for_model(
    model_path: &std::path::Path,
    legacy_mode: crate::cli::PromptMode,
) -> ContinuePromptFormat {
    let filename = model_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if filename.contains("gpt-oss") {
        ContinuePromptFormat::GptOssHarmony
    } else {
        ContinuePromptFormat::Legacy(legacy_mode)
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
fn build_continue_prompt(format: ContinuePromptFormat, prompt_body: &str) -> (String, Vec<String>) {
    match format {
        ContinuePromptFormat::Legacy(mode) => build_prompt(mode, prompt_body),
        ContinuePromptFormat::GptOssHarmony => (
            format!(
                "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\n\nReasoning: low\n\n# Valid channels: analysis, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions\n\nYou are Pete Listenbury. Use the analysis channel for private internal monologue. Use the final channel only to emit a real-world action. Final channel content must be exactly one or more <ts>...</ts> TypeScript blocks, or empty. Never write plain conversational text in final. Never put Harmony template tokens in final channel content.\n\nTo speak to the user, write final content like <ts>say(\"Hello, I can hear you.\")</ts>. If speech should intentionally talk over active user speech, use <ts>say(\"Excuse me.\", {{ interrupt: true }})</ts>; otherwise TTS waits for VAD to clear before starting. Speak sparingly: after one say command, leave room for the interlocutor to answer before saying more. Do not use say for clock ticks, quiet moments, or idle narration. To inspect code, write final content like <ts>listFiles()</ts>. The TypeScript builders say, shutup, pause, resume, listFiles, readSourceFile, readFile, searchSource, and grepSource are already available in scope; imports from \"pete:will\" are also allowed.<|end|><|start|>user<|message|>{prompt_body}<|end|><|start|>assistant"
            ),
            harmony_continue_stops(),
        ),
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
fn harmony_continue_stops() -> Vec<String> {
    vec![
        "<|return|>".to_string(),
        "<|start|>user".to_string(),
        "<|start|>system".to_string(),
        "<|start|>developer".to_string(),
    ]
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
    prompt_format: ContinuePromptFormat,
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
        prompt_format: ContinuePromptFormat,
        system_prompt: String,
        max_tokens: Option<usize>,
        context_size: u32,
        verbatim_turns: usize,
    ) -> Result<Self> {
        let rolling =
            RollingContextManager::new(system_prompt, context_size, max_tokens, verbatim_turns);
        let (prompt, stop) = build_continue_prompt(prompt_format, &rolling.prompt_body());
        let id = llm.start(GenerationRequest {
            prompt: prompt.clone(),
            max_tokens,
            stop,
        })?;
        let mut session = Self {
            llm,
            id,
            prompt_format,
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
        let formatted = self.format_live_append(&append_text);
        if self.rolling.should_restart_before_append(&formatted) {
            self.restart_with_compact_prompt()
        } else {
            self.rolling.note_appended_text(&formatted);
            self.llm.append_prompt(self.id, formatted)
        }
    }

    fn restart_with_compact_prompt(&mut self) -> Result<()> {
        self.cancel_current_generation()?;
        self.start_with_compact_prompt()
    }

    fn start_with_compact_prompt(&mut self) -> Result<()> {
        let (prompt, stop) = build_continue_prompt(self.prompt_format, &self.rolling.prompt_body());
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

    fn uses_harmony(&self) -> bool {
        matches!(self.prompt_format, ContinuePromptFormat::GptOssHarmony)
    }

    fn format_live_append(&self, text: &str) -> String {
        match self.prompt_format {
            ContinuePromptFormat::GptOssHarmony => {
                format!("<|end|><|start|>user<|message|>{text}<|end|><|start|>assistant")
            }
            ContinuePromptFormat::Legacy(_) => text.to_string(),
        }
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

    fn ear_observation(text: String) -> Self {
        let trimmed = text.trim().to_string();
        Self {
            text: wrap_ear_event(&trimmed),
            memory: PromptMemory::AuditoryObservation(trimmed),
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
    AuditoryObservation(String),
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
            PromptMemory::AuditoryObservation(message) => self.set_ear_scene(message),
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

    fn set_ear_scene(&mut self, message: String) {
        self.auditory_scene.events.clear();
        self.auditory_scene.events.push(format!(
            "Ear: {}",
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
#[derive(Debug, Clone, PartialEq, Eq)]
enum TypeScriptCommand {
    Say { text: String, interrupt: bool },
    Shutup,
    Pause,
    Resume,
    ListFiles,
    ReadSourceFile { file: String, page: usize },
    SearchSource { query: String, limit: usize },
    GrepSource { pattern: String, limit: usize },
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
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum TypeScriptCommandPayload {
    Say {
        text: String,
        #[serde(default)]
        interrupt: bool,
    },
    Shutup,
    Pause,
    Resume,
    ListFiles,
    ReadSourceFile {
        file: String,
        page: Option<usize>,
    },
    SearchSource {
        query: String,
        limit: Option<usize>,
    },
    GrepSource {
        pattern: String,
        limit: Option<usize>,
    },
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
fn execute_typescript_source(script: &str) -> SourceCommandExecution {
    match execute_typescript_commands(script) {
        Ok(commands) => execute_typescript_command_results(script, &commands),
        Err(error) => SourceCommandExecution {
            message: format!("TypeScript failed:\n{error}"),
            runtime_events: Vec::new(),
        },
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
fn execute_typescript_command_results(
    script: &str,
    commands: &[TypeScriptCommand],
) -> SourceCommandExecution {
    let mut response = String::from("TypeScript executed.\nSource:\n");
    response.push_str(script.trim());
    if commands.is_empty() {
        response.push_str("\n\nNo commands returned.");
        return SourceCommandExecution {
            message: response,
            runtime_events: Vec::new(),
        };
    }

    response.push_str("\n\nResults:");
    let mut runtime_events = Vec::new();
    for command in commands {
        let (name, output) = match command {
            TypeScriptCommand::Say { text, interrupt } => {
                runtime_events.push(ContinueRuntimeEvent::UtteranceCompleted {
                    id: next_typescript_utterance_id(),
                    content: text.trim().to_string(),
                    interrupt: *interrupt,
                });
                (
                    "say",
                    format!(
                        "Speech queued{}: {}",
                        if *interrupt { " (interrupt)" } else { "" },
                        text.trim()
                    ),
                )
            }
            TypeScriptCommand::Shutup => {
                runtime_events.push(ContinueRuntimeEvent::SpeechControl {
                    command: SpeechControlCommand::Shutup,
                });
                (
                    "shutup",
                    "Speech playback stopped and queue cleared.".to_string(),
                )
            }
            TypeScriptCommand::Pause => {
                runtime_events.push(ContinueRuntimeEvent::SpeechControl {
                    command: SpeechControlCommand::Pause,
                });
                ("pause", "Speech playback paused.".to_string())
            }
            TypeScriptCommand::Resume => {
                runtime_events.push(ContinueRuntimeEvent::SpeechControl {
                    command: SpeechControlCommand::Resume,
                });
                ("resume", "Speech playback resumed.".to_string())
            }
            TypeScriptCommand::ListFiles => ("list_files", execute_list_source_files()),
            TypeScriptCommand::ReadSourceFile { file, page } => {
                ("read_source_file", execute_view_source_file(file, *page))
            }
            TypeScriptCommand::SearchSource { query, limit } => {
                ("search_source", execute_search_source(query, *limit))
            }
            TypeScriptCommand::GrepSource { pattern, limit } => {
                ("grep_source", execute_grep_source(pattern, *limit))
            }
        };
        response.push_str("\n\n[");
        response.push_str(name);
        response.push_str("]\n");
        response.push_str(&output);
    }
    SourceCommandExecution {
        message: response,
        runtime_events,
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
fn next_typescript_utterance_id() -> u64 {
    static NEXT_ID: AtomicU64 = AtomicU64::new(10_000);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
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
fn execute_typescript_commands(script: &str) -> Result<Vec<TypeScriptCommand>> {
    if script.trim().is_empty() {
        return Ok(Vec::new());
    }
    let script = typescript_source_with_default_will_imports(script);

    let config = InterpreterConfig {
        internal_modules: vec![will_typescript_module()],
        ..Default::default()
    };
    let mut interp = Interpreter::with_config(config);
    interp
        .prepare(&script, Some(tsrun::ModulePath::new("/listenbury-will.ts")))
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
    let payloads = parse_typescript_command_payloads(command_value)?;
    Ok(payloads
        .into_iter()
        .filter_map(|payload| match payload {
            TypeScriptCommandPayload::Say { text, interrupt } => {
                non_empty_text(&text).map(|text| TypeScriptCommand::Say {
                    text: text.to_string(),
                    interrupt,
                })
            }
            TypeScriptCommandPayload::Shutup => Some(TypeScriptCommand::Shutup),
            TypeScriptCommandPayload::Pause => Some(TypeScriptCommand::Pause),
            TypeScriptCommandPayload::Resume => Some(TypeScriptCommand::Resume),
            TypeScriptCommandPayload::ListFiles => Some(TypeScriptCommand::ListFiles),
            TypeScriptCommandPayload::ReadSourceFile { file, page } => {
                let file = file.trim();
                (!file.is_empty()).then(|| TypeScriptCommand::ReadSourceFile {
                    file: file.to_string(),
                    page: page.unwrap_or(1).max(1),
                })
            }
            TypeScriptCommandPayload::SearchSource { query, limit } => {
                non_empty_text(&query).map(|query| TypeScriptCommand::SearchSource {
                    query: query.to_string(),
                    limit: limit.unwrap_or(12).max(1),
                })
            }
            TypeScriptCommandPayload::GrepSource { pattern, limit } => non_empty_text(&pattern)
                .map(|pattern| TypeScriptCommand::GrepSource {
                    pattern: pattern.to_string(),
                    limit: limit.unwrap_or(12).max(1),
                }),
        })
        .collect())
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
fn typescript_source_with_default_will_imports(script: &str) -> String {
    if script.contains("\"pete:will\"") || script.contains("'pete:will'") {
        return script.to_string();
    }

    format!(
        "import {{ say, shutup, pause, resume, listFiles, readSourceFile, readFile, searchSource, grepSource }} from \"pete:will\";\n{script}"
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
fn non_empty_text(text: &str) -> Option<&str> {
    let trimmed = text.trim();
    (!trimmed.is_empty()).then_some(trimmed)
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
fn tsrun_error(err: JsError) -> anyhow::Error {
    anyhow::anyhow!("TypeScript execution failed: {err}")
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
fn parse_typescript_command_payloads(value: Value) -> Result<Vec<TypeScriptCommandPayload>> {
    match value {
        Value::Null => Ok(Vec::new()),
        Value::Array(items) => items
            .into_iter()
            .filter(|item| !item.is_null())
            .map(serde_json::from_value)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into),
        Value::Object(_) => Ok(vec![serde_json::from_value(value)?]),
        other => {
            anyhow::bail!("TypeScript must return a command object or command array, got {other}")
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
fn will_typescript_module() -> InternalModule {
    InternalModule::native("pete:will")
        .with_function("say", ts_say, 2)
        .with_function("shutup", ts_shutup, 0)
        .with_function("pause", ts_pause, 0)
        .with_function("resume", ts_resume, 0)
        .with_function("listFiles", ts_list_files, 0)
        .with_function("readSourceFile", ts_read_source_file, 2)
        .with_function("readFile", ts_read_source_file, 2)
        .with_function("searchSource", ts_search_source, 2)
        .with_function("grepSource", ts_grep_source, 2)
        .build()
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
fn command_value(interp: &mut Interpreter, value: Value) -> std::result::Result<Guarded, JsError> {
    let guard = api::create_guard(interp);
    let value = api::create_from_json(interp, &guard, &value)?;
    Ok(Guarded::with_guard(value, guard))
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
fn string_arg(args: &[JsValue], index: usize) -> String {
    args.get(index)
        .and_then(JsValue::as_str)
        .unwrap_or_default()
        .to_string()
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
fn optional_positive_integer_arg(args: &[JsValue], index: usize) -> Option<usize> {
    args.get(index)
        .and_then(JsValue::as_number)
        .filter(|number| number.is_finite() && *number > 0.0)
        .map(|number| number.floor() as usize)
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

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
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

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn ts_shutup(
    interp: &mut Interpreter,
    _this: JsValue,
    _args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(interp, json!({ "kind": "shutup" }))
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
fn ts_pause(
    interp: &mut Interpreter,
    _this: JsValue,
    _args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(interp, json!({ "kind": "pause" }))
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
fn ts_resume(
    interp: &mut Interpreter,
    _this: JsValue,
    _args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(interp, json!({ "kind": "resume" }))
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
fn ts_list_files(
    interp: &mut Interpreter,
    _this: JsValue,
    _args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    command_value(interp, json!({ "kind": "list_files" }))
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
fn ts_read_source_file(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    let mut value = json!({ "kind": "read_source_file", "file": string_arg(args, 0) });
    if let Some(page) = optional_positive_integer_arg(args, 1) {
        value["page"] = json!(page);
    }
    command_value(interp, value)
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
fn ts_search_source(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    let mut value = json!({ "kind": "search_source", "query": string_arg(args, 0) });
    if let Some(limit) = optional_positive_integer_arg(args, 1) {
        value["limit"] = json!(limit);
    }
    command_value(interp, value)
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
fn ts_grep_source(
    interp: &mut Interpreter,
    _this: JsValue,
    args: &[JsValue],
) -> std::result::Result<Guarded, JsError> {
    let mut value = json!({ "kind": "grep_source", "pattern": string_arg(args, 0) });
    if let Some(limit) = optional_positive_integer_arg(args, 1) {
        value["limit"] = json!(limit);
    }
    command_value(interp, value)
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
    time_event_jitter_state: &mut u64,
    defer_live_events: bool,
    deferred_live_events: &mut VecDeque<PromptPacket>,
    mouth: &mut ContinueMouth,
    tts_vad: &mut DuplexTurnController,
    prompt_gate: &mut ContinuePromptGate,
    live_trace: &mut ContinueLiveTrace,
    live_trace_turn: &mut u64,
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
        *next_time_event_at = now + next_time_event_interval(time_event_jitter_state);
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

    drain_mouth_events_into_llm(
        llm_session,
        mouth_rx,
        pending_mouth_utterances,
        mouth_playback_paused,
        defer_live_events,
        deferred_live_events,
    )?;

    for ear_event in ear_rx.try_iter() {
        match &ear_event {
            ContinueEarEvent::Transcript {
                text,
                timed_word_stream,
                occurred_at,
            } => {
                eprintln!("[dev continue] heard: {text}");
                *live_trace_turn = live_trace_turn.saturating_add(1);
                emit_live_asr_trace_events(
                    live_trace,
                    *live_trace_turn,
                    text,
                    timed_word_stream,
                    *occurred_at,
                )?;
            }
            ContinueEarEvent::TranscriptCandidate {
                event,
                stability,
                occurred_at,
            } => {
                eprintln!("[ear] {}", ear_event.to_message());
                emit_live_asr_candidate_trace_event(
                    live_trace,
                    live_trace_turn.saturating_add(1),
                    event,
                    stability.as_ref(),
                    *occurred_at,
                )?;
            }
            ContinueEarEvent::ListeningStarted { .. }
            | ContinueEarEvent::SpeechStarted
            | ContinueEarEvent::SpeechStopped
            | ContinueEarEvent::AuditoryObservation { .. }
            | ContinueEarEvent::EnvironmentalSound { .. }
            | ContinueEarEvent::SelfVoiceHeard { .. }
            | ContinueEarEvent::OverlapDetected { .. }
            | ContinueEarEvent::Error { .. } => {
                emit_live_ear_trace_event(
                    live_trace,
                    live_trace_turn.saturating_add(1),
                    &ear_event,
                )?;
            }
        }
        for packet in prompt_gate.consider_ear_event(&ear_event, Instant::now()) {
            append_or_defer_live_event(
                llm_session,
                packet,
                defer_live_events,
                deferred_live_events,
                "failed to append ear event to live generation",
            )?;
        }
        if let ContinueEarEvent::Error { message } = &ear_event {
            anyhow::bail!("dev continue ear failed: {message}");
        }
        tts_vad.handle_ear_event(
            &ear_event,
            mouth,
            llm_session,
            defer_live_events,
            deferred_live_events,
            *pending_mouth_utterances,
        )?;
    }

    tts_vad.poll(
        mouth,
        llm_session,
        defer_live_events,
        deferred_live_events,
        *pending_mouth_utterances,
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
fn write_duplex_trace_scenario_jsonl(path: &std::path::Path, events: &[Value]) -> Result<()> {
    let mut writer = if listenbury::live_trace::trace_path_looks_like_jsonl(path) {
        EitherTraceScenarioWriter::Jsonl(JsonlTraceWriter::create(path)?)
    } else {
        EitherTraceScenarioWriter::Session(listenbury::live_trace::TraceSessionWriter::create(
            path,
            TraceSessionMetadata::new(SessionId::new(), ExactTimestamp::now(), {
                let mut runtime = TraceRuntimeMetadata::new("listenbury dev continue");
                runtime.mode = Some("duplex_trace_scenario".to_string());
                runtime
            }),
        )?)
    };
    for event in events {
        writer.write(event)?;
    }
    Ok(())
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
enum EitherTraceScenarioWriter {
    Jsonl(JsonlTraceWriter),
    Session(listenbury::live_trace::TraceSessionWriter),
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
impl EitherTraceScenarioWriter {
    fn write(&mut self, value: &Value) -> Result<()> {
        match self {
            Self::Jsonl(writer) => writer.write(value),
            Self::Session(writer) => writer.write(value),
        }
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn build_duplex_trace_scenario_events(
    scenario: crate::cli::DuplexTraceScenarioOption,
    pause_after: Duration,
    listen_for: Duration,
) -> Vec<Value> {
    match scenario {
        crate::cli::DuplexTraceScenarioOption::OverlapYield => {
            build_duplex_overlap_yield_trace_events(pause_after, listen_for)
        }
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn build_duplex_overlap_yield_trace_events(
    pause_after: Duration,
    listen_for: Duration,
) -> Vec<Value> {
    let started_at = Instant::now();
    let mut controller = DuplexTurnController::new(DuplexTurnControllerConfig {
        pause_after,
        listen_for,
    });
    let mut prompt_gate = ContinuePromptGate::default();
    let mut events = Vec::new();
    let pending_mouth_utterances = 2usize;

    let mut push = |kind: &'static str, t_ms: u64, details: Value| {
        events.push(json!({
            "scenario": "overlap-yield",
            "kind": kind,
            "t_ms": t_ms,
            "details": details
        }));
    };
    push(
        "playback_started",
        0,
        json!({
            "speaker": "pete",
            "utterance": "Synthetic overlap/yield diagnostic"
        }),
    );

    push(
        "routing_decision",
        120,
        json!({
            "case": "short_overlap_blip",
            "routing": "MixedSelfAndExternal",
            "self_correlation": 0.62,
            "residual_ratio": 0.41,
            "estimated_delay_ms": 47,
            "decision": "route_residual_to_vad_asr",
        }),
    );
    let short_overlap = ContinueEarEvent::OverlapDetected {
        self_confidence: 0.66,
        external_confidence: 0.82,
        duration_ms: 45,
    };
    let short_overlap_at = started_at + Duration::from_millis(120);
    controller.observe_ear_event(&short_overlap, short_overlap_at);
    for packet in prompt_gate.consider_ear_event(&short_overlap, short_overlap_at) {
        push(
            "prompt_gate_output",
            120,
            json!({
                "source": "overlap_detected",
                "summary": packet.text,
            }),
        );
    }
    let short_action = controller.next_action(
        started_at
            + Duration::from_millis(120)
            + pause_after.saturating_sub(Duration::from_millis(1)),
        pending_mouth_utterances,
    );
    push(
        "controller_decision",
        120 + pause_after
            .saturating_sub(Duration::from_millis(1))
            .as_millis() as u64,
        json!({
            "case": "short_overlap_blip",
            "decision": "continue",
            "raw_action": short_action.map(|action| format!("{action:?}")),
        }),
    );

    push(
        "routing_decision",
        1_000,
        json!({
            "case": "sustained_overlap",
            "routing": "ExternalSpeechCandidate",
            "self_correlation": 0.22,
            "residual_ratio": 0.91,
            "estimated_delay_ms": 0,
            "decision": "route_external_to_vad_asr",
        }),
    );
    let sustained_overlap_a = ContinueEarEvent::OverlapDetected {
        self_confidence: 0.35,
        external_confidence: 0.93,
        duration_ms: 400,
    };
    let sustained_overlap_a_at = started_at + Duration::from_millis(1_000);
    controller.observe_ear_event(&sustained_overlap_a, sustained_overlap_a_at);
    for packet in prompt_gate.consider_ear_event(&sustained_overlap_a, sustained_overlap_a_at) {
        push(
            "prompt_gate_output",
            1_000,
            json!({
                "source": "overlap_detected",
                "summary": packet.text,
            }),
        );
    }
    let sustained_overlap_b = ContinueEarEvent::OverlapDetected {
        self_confidence: 0.31,
        external_confidence: 0.90,
        duration_ms: 380,
    };
    let sustained_overlap_b_at = started_at + Duration::from_millis(1_080);
    controller.observe_ear_event(&sustained_overlap_b, sustained_overlap_b_at);
    for packet in prompt_gate.consider_ear_event(&sustained_overlap_b, sustained_overlap_b_at) {
        push(
            "prompt_gate_output",
            1_080,
            json!({
                "source": "overlap_detected",
                "summary": packet.text,
            }),
        );
    }
    let pause_at = 1_000 + pause_after.as_millis() as u64 + 1;
    let pause_action = controller.next_action(
        started_at + Duration::from_millis(pause_at),
        pending_mouth_utterances,
    );
    push(
        "controller_decision",
        pause_at,
        json!({
            "case": "sustained_overlap",
            "decision": "yield_pause",
            "raw_action": pause_action.map(|action| format!("{action:?}")),
            "mouth_command": "pause",
        }),
    );
    let clear_at = pause_at + listen_for.as_millis() as u64 + 1;
    let clear_action = controller.next_action(
        started_at + Duration::from_millis(clear_at),
        pending_mouth_utterances,
    );
    push(
        "controller_decision",
        clear_at,
        json!({
            "case": "sustained_overlap",
            "decision": "yield_clear_queue",
            "raw_action": clear_action.map(|action| format!("{action:?}")),
            "mouth_command": "shutup",
            "generation_restart": true,
        }),
    );

    push(
        "routing_decision",
        2_100,
        json!({
            "case": "mixed_echo_and_user_speech",
            "routing": "MixedSelfAndExternal",
            "self_correlation": 0.71,
            "residual_ratio": 0.53,
            "estimated_delay_ms": 55,
            "decision": "route_residual_to_vad_asr",
        }),
    );
    push(
        "vad_decision",
        2_130,
        json!({
            "input": "residual_external",
            "speech": true,
            "confidence": 0.87,
        }),
    );
    push(
        "asr_result",
        2_260,
        json!({
            "source": "residual_external",
            "transcript": "sorry Pete, one sec",
        }),
    );

    push(
        "routing_decision",
        3_000,
        json!({
            "case": "echo_only_return",
            "routing": "EchoOnly",
            "self_correlation": 0.96,
            "residual_ratio": 0.03,
            "estimated_delay_ms": 48,
            "decision": "suppress_asr",
        }),
    );
    push(
        "vad_decision",
        3_010,
        json!({
            "input": "speaker_echo",
            "speech": false,
            "suppressed": true,
            "false_interruption": false,
        }),
    );
    push(
        "playback_finished",
        3_400,
        json!({
            "speaker": "pete",
            "status": "diagnostic_complete"
        }),
    );

    events
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
const DUPLEX_TURN_MIN_OVERLAP_EXTERNAL_CONFIDENCE: f32 = 0.45;

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
#[derive(Debug, Clone, Copy)]
struct DuplexTurnControllerConfig {
    pause_after: Duration,
    listen_for: Duration,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
#[derive(Debug)]
struct DuplexTurnController {
    config: DuplexTurnControllerConfig,
    external_speech_active: bool,
    external_speech_started_at: Option<Instant>,
    paused_for_external_speech: bool,
    listen_deadline: Option<Instant>,
    turn_gap_deadline: Option<Instant>,
    holding_turn_gap_speech: bool,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DuplexTurnAction {
    Pause,
    Resume,
    Clear,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
impl DuplexTurnController {
    fn new(config: DuplexTurnControllerConfig) -> Self {
        Self {
            config,
            external_speech_active: false,
            external_speech_started_at: None,
            paused_for_external_speech: false,
            listen_deadline: None,
            turn_gap_deadline: None,
            holding_turn_gap_speech: false,
        }
    }

    fn observe_ear_event(&mut self, event: &ContinueEarEvent, now: Instant) {
        match event {
            ContinueEarEvent::SpeechStarted => {
                self.mark_external_speech_started(now, 1.0);
            }
            ContinueEarEvent::OverlapDetected {
                external_confidence,
                ..
            } if *external_confidence >= DUPLEX_TURN_MIN_OVERLAP_EXTERNAL_CONFIDENCE => {
                self.mark_external_speech_started(now, *external_confidence);
            }
            ContinueEarEvent::SpeechStopped => {
                self.mark_external_speech_stopped(now);
            }
            ContinueEarEvent::Transcript { .. } => {
                if !self.external_speech_active && self.turn_gap_deadline.is_none() {
                    self.turn_gap_deadline = Some(now + self.config.listen_for);
                }
            }
            ContinueEarEvent::TranscriptCandidate { .. } => {}
            ContinueEarEvent::ListeningStarted { .. }
            | ContinueEarEvent::AuditoryObservation { .. }
            | ContinueEarEvent::EnvironmentalSound { .. }
            | ContinueEarEvent::SelfVoiceHeard { .. }
            | ContinueEarEvent::OverlapDetected { .. }
            | ContinueEarEvent::Error { .. } => {}
        }
    }

    fn handle_ear_event(
        &mut self,
        event: &ContinueEarEvent,
        mouth: &mut ContinueMouth,
        llm_session: &mut ContinueLlmSession,
        defer_live_events: bool,
        deferred_live_events: &mut VecDeque<PromptPacket>,
        pending_mouth_utterances: usize,
    ) -> Result<()> {
        let now = Instant::now();
        self.observe_ear_event(event, now);
        self.poll(
            mouth,
            llm_session,
            defer_live_events,
            deferred_live_events,
            pending_mouth_utterances,
        )
    }

    fn prepare_runtime_event(
        &mut self,
        event: &ContinueRuntimeEvent,
        mouth: &mut ContinueMouth,
        llm_session: &mut ContinueLlmSession,
        defer_live_events: bool,
        deferred_live_events: &mut VecDeque<PromptPacket>,
    ) -> Result<()> {
        let Some(action) = self.prepare_runtime_action(event) else {
            return Ok(());
        };
        match action {
            DuplexTurnAction::Pause => {
                let message = if self.holding_turn_gap_speech {
                    "TTS prepared during the quiet turn gap; playback is held until the gap stays quiet."
                } else {
                    "TTS start deferred because external speech is currently detected."
                };
                self.pause_for_external_speech(
                    mouth,
                    llm_session,
                    defer_live_events,
                    deferred_live_events,
                    message,
                )?
            }
            DuplexTurnAction::Resume => self.resume_after_external_speech(
                mouth,
                llm_session,
                defer_live_events,
                deferred_live_events,
                "TTS resumed because this say command was marked interrupt=true.",
            )?,
            DuplexTurnAction::Clear => {}
        }
        Ok(())
    }

    fn prepare_runtime_action(&mut self, event: &ContinueRuntimeEvent) -> Option<DuplexTurnAction> {
        match event {
            ContinueRuntimeEvent::UtteranceCompleted {
                interrupt: true, ..
            } if self.paused_for_external_speech => Some(DuplexTurnAction::Resume),
            ContinueRuntimeEvent::UtteranceCompleted {
                interrupt: false, ..
            } if self.external_speech_active && !self.paused_for_external_speech => {
                self.listen_deadline = None;
                Some(DuplexTurnAction::Pause)
            }
            ContinueRuntimeEvent::UtteranceCompleted {
                interrupt: false, ..
            } if self.should_hold_for_turn_gap() => {
                self.holding_turn_gap_speech = true;
                Some(DuplexTurnAction::Pause)
            }
            ContinueRuntimeEvent::UtteranceCompleted { .. }
            | ContinueRuntimeEvent::SpeechControl { .. }
            | ContinueRuntimeEvent::SourceCommand { .. } => None,
        }
    }

    fn poll(
        &mut self,
        mouth: &mut ContinueMouth,
        llm_session: &mut ContinueLlmSession,
        defer_live_events: bool,
        deferred_live_events: &mut VecDeque<PromptPacket>,
        pending_mouth_utterances: usize,
    ) -> Result<()> {
        let Some(action) = self.next_action(Instant::now(), pending_mouth_utterances) else {
            return Ok(());
        };
        match action {
            DuplexTurnAction::Pause => self.pause_for_external_speech(
                mouth,
                llm_session,
                defer_live_events,
                deferred_live_events,
                "TTS auto-paused because external speech was detected while Pete was speaking.",
            )?,
            DuplexTurnAction::Resume => self.resume_after_external_speech(
                mouth,
                llm_session,
                defer_live_events,
                deferred_live_events,
                "TTS resumed after the interruption listen window stayed quiet.",
            )?,
            DuplexTurnAction::Clear => self.clear_after_external_speech(
                mouth,
                llm_session,
                defer_live_events,
                deferred_live_events,
                "TTS queue cleared because external speech continued during the interruption listen window.",
            )?,
        }
        Ok(())
    }

    fn mark_external_speech_started(&mut self, now: Instant, _confidence: f32) {
        if !self.external_speech_active {
            self.external_speech_started_at = Some(now);
        }
        self.external_speech_active = true;
    }

    fn mark_external_speech_stopped(&mut self, now: Instant) {
        self.external_speech_active = false;
        self.external_speech_started_at = None;
        if self.paused_for_external_speech {
            self.listen_deadline = Some(now + self.config.listen_for);
        } else {
            self.turn_gap_deadline = Some(now + self.config.listen_for);
        }
    }

    fn should_hold_for_turn_gap(&self) -> bool {
        self.turn_gap_deadline
            .is_some_and(|deadline| Instant::now() < deadline)
            && !self.paused_for_external_speech
    }

    fn next_action(
        &mut self,
        now: Instant,
        pending_mouth_utterances: usize,
    ) -> Option<DuplexTurnAction> {
        if pending_mouth_utterances == 0 {
            self.paused_for_external_speech = false;
            self.listen_deadline = None;
            self.holding_turn_gap_speech = false;
            return None;
        }

        if self.holding_turn_gap_speech {
            if self.external_speech_active {
                self.holding_turn_gap_speech = false;
                self.turn_gap_deadline = None;
                self.paused_for_external_speech = false;
                self.listen_deadline = None;
                return Some(DuplexTurnAction::Clear);
            }
            if self
                .turn_gap_deadline
                .is_some_and(|deadline| now >= deadline)
            {
                self.holding_turn_gap_speech = false;
                self.turn_gap_deadline = None;
                self.paused_for_external_speech = false;
                self.listen_deadline = None;
                return Some(DuplexTurnAction::Resume);
            }
        }

        if self.external_speech_active && !self.paused_for_external_speech {
            if let Some(started_at) = self.external_speech_started_at {
                let elapsed = now.checked_duration_since(started_at).unwrap_or_default();
                if elapsed >= self.config.pause_after {
                    self.paused_for_external_speech = true;
                    self.listen_deadline = Some(now + self.config.listen_for);
                    return Some(DuplexTurnAction::Pause);
                }
            }
        }

        if self.paused_for_external_speech
            && self.listen_deadline.is_some_and(|deadline| now >= deadline)
        {
            self.paused_for_external_speech = false;
            self.listen_deadline = None;
            return if self.external_speech_active {
                Some(DuplexTurnAction::Clear)
            } else {
                Some(DuplexTurnAction::Resume)
            };
        }
        None
    }

    fn pause_for_external_speech(
        &mut self,
        mouth: &mut ContinueMouth,
        llm_session: &mut ContinueLlmSession,
        defer_live_events: bool,
        deferred_live_events: &mut VecDeque<PromptPacket>,
        message: &'static str,
    ) -> Result<()> {
        self.paused_for_external_speech = true;
        send_duplex_turn_control(
            mouth,
            llm_session,
            defer_live_events,
            deferred_live_events,
            SpeechControlCommand::Pause,
            message,
        )
    }

    fn resume_after_external_speech(
        &mut self,
        mouth: &mut ContinueMouth,
        llm_session: &mut ContinueLlmSession,
        defer_live_events: bool,
        deferred_live_events: &mut VecDeque<PromptPacket>,
        message: &'static str,
    ) -> Result<()> {
        self.paused_for_external_speech = false;
        self.listen_deadline = None;
        self.turn_gap_deadline = None;
        self.holding_turn_gap_speech = false;
        send_duplex_turn_control(
            mouth,
            llm_session,
            defer_live_events,
            deferred_live_events,
            SpeechControlCommand::Resume,
            message,
        )
    }

    fn clear_after_external_speech(
        &mut self,
        mouth: &mut ContinueMouth,
        llm_session: &mut ContinueLlmSession,
        defer_live_events: bool,
        deferred_live_events: &mut VecDeque<PromptPacket>,
        message: &'static str,
    ) -> Result<()> {
        self.paused_for_external_speech = false;
        self.listen_deadline = None;
        self.turn_gap_deadline = None;
        self.holding_turn_gap_speech = false;
        send_duplex_turn_control(
            mouth,
            llm_session,
            defer_live_events,
            deferred_live_events,
            SpeechControlCommand::Shutup,
            message,
        )?;
        append_or_defer_live_event(
            llm_session,
            PromptPacket::source(
                "Live generation was restarted because sustained overlap forced a yield."
                    .to_string(),
            ),
            defer_live_events,
            deferred_live_events,
            "failed to append duplex generation restart event to live generation",
        )?;
        llm_session
            .restart_with_compact_prompt()
            .context("failed to restart continued llama.cpp generation after duplex queue clear")
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn prepare_tts_runtime_event(
    event: &ContinueRuntimeEvent,
    mouth: &mut ContinueMouth,
    tts_vad: &mut DuplexTurnController,
    llm_session: &mut ContinueLlmSession,
    defer_live_events: bool,
    deferred_live_events: &mut VecDeque<PromptPacket>,
) -> Result<()> {
    tts_vad.prepare_runtime_event(
        event,
        mouth,
        llm_session,
        defer_live_events,
        deferred_live_events,
    )
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn send_duplex_turn_control(
    mouth: &mut ContinueMouth,
    llm_session: &mut ContinueLlmSession,
    defer_live_events: bool,
    deferred_live_events: &mut VecDeque<PromptPacket>,
    command: SpeechControlCommand,
    message: &'static str,
) -> Result<()> {
    mouth.enqueue_runtime_event(&ContinueRuntimeEvent::SpeechControl { command })?;
    append_or_defer_live_event(
        llm_session,
        PromptPacket::source(message.to_string()),
        defer_live_events,
        deferred_live_events,
        "failed to append duplex turn control event to live generation",
    )
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn append_mouth_runtime_trace(
    llm_session: &mut ContinueLlmSession,
    defer_live_events: bool,
    deferred_live_events: &mut VecDeque<PromptPacket>,
    message: String,
) -> Result<()> {
    append_or_defer_live_event(
        llm_session,
        PromptPacket::source(message),
        defer_live_events,
        deferred_live_events,
        "failed to append mouth runtime event to live generation",
    )
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn drain_mouth_events_into_llm(
    llm_session: &mut ContinueLlmSession,
    mouth_rx: &crossbeam_channel::Receiver<ContinueMouthEvent>,
    pending_mouth_utterances: &mut usize,
    mouth_playback_paused: &mut bool,
    defer_live_events: bool,
    deferred_live_events: &mut VecDeque<PromptPacket>,
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
                    ContinueMouthEvent::SpeechInterrupted { id, text } => {
                        append_mouth_runtime_trace(
                            llm_session,
                            defer_live_events,
                            deferred_live_events,
                            format!(
                                "Duplex runtime: speech interrupted while speaking (id={id}, text={:?}).",
                                compact_prompt_line(&text, MAX_VERBATIM_TURN_CHARS)
                            ),
                        )?;
                    }
                    ContinueMouthEvent::SpeechQueueCleared { count } => {
                        append_mouth_runtime_trace(
                            llm_session,
                            defer_live_events,
                            deferred_live_events,
                            format!(
                                "Duplex runtime: cleared {count} queued speech unit(s) after sustained overlap."
                            ),
                        )?;
                    }
                    ContinueMouthEvent::SpeechPaused => {
                        append_mouth_runtime_trace(
                            llm_session,
                            defer_live_events,
                            deferred_live_events,
                            "Duplex runtime: speech playback paused to yield the floor."
                                .to_string(),
                        )?;
                    }
                    ContinueMouthEvent::SpeechResumed => {
                        append_mouth_runtime_trace(
                            llm_session,
                            defer_live_events,
                            deferred_live_events,
                            "Duplex runtime: speech playback resumed.".to_string(),
                        )?;
                    }
                    ContinueMouthEvent::SpeechError { message, .. } => {
                        anyhow::bail!("dev continue mouth failed: {message}");
                    }
                    ContinueMouthEvent::SpeechQueued { .. }
                    | ContinueMouthEvent::SpeechSynthesisStarted { .. }
                    | ContinueMouthEvent::SpeechPlaybackCompleted { .. } => {}
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
    speaker_reference: Arc<Mutex<SpeakerReferenceMask>>,
    live_audio: Option<listenbury::web::LiveSessionAudioStore>,
    browser_audio_rx: Option<crossbeam_channel::Receiver<AudioFrame>>,
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
        let mut recognizer = WhisperSpeechRecognizer::new_quiet(&config.whisper_model)
            .with_context(|| {
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
        let sample_capacity = callback_sample_queue_capacity(input_sample_rate_hz, input_channels);
        let (sample_tx, sample_rx) = crossbeam_channel::bounded::<f32>(sample_capacity);
        let (asr_tx, asr_rx) = crossbeam_channel::bounded::<ContinueAsrWorkItem>(8);
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
            "dev continue ear listening on {device_name}: {} Hz, {} channel(s), vad={}, sample_queue={}.",
            input_sample_rate_hz,
            input_channels,
            config.vad_backend.as_str(),
            sample_capacity
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
                let mut next_stream_id = 1u64;
                let mut speculative_planner = TranscriptSpeculativePlanner::default();
                while !stop_for_asr.load(Ordering::Relaxed) {
                    match asr_rx.recv_timeout(Duration::from_millis(20)) {
                        Ok(work) => match transcribe_group_with_finality(
                            &work.frames,
                            &mut recognizer,
                            work.is_final,
                        ) {
                            Ok(output) => {
                                let observed_at = ExactTimestamp::now();
                                for event in output.candidate_events {
                                    let stability = speculative_planner.observe(&event);
                                    if event_tx_for_asr
                                        .send(ContinueEarEvent::TranscriptCandidate {
                                            event,
                                            stability,
                                            occurred_at: observed_at,
                                        })
                                        .is_err()
                                    {
                                        return;
                                    }
                                }

                                if !output.text.is_empty() {
                                    let timed_word_stream = if output.words.is_empty() {
                                        live_asr_text_to_word_stream(
                                            WordStreamId(next_stream_id),
                                            &output.text,
                                        )
                                    } else {
                                        let mut stream = transcript_to_energy_snapped_word_stream(
                                            WordStreamId(next_stream_id),
                                            &output.words,
                                            &work.frames,
                                        );
                                        stream.source = WordStreamSource::LiveAsr;
                                        stream
                                    };
                                    next_stream_id = next_stream_id.saturating_add(1);
                                    if event_tx_for_asr
                                        .send(ContinueEarEvent::Transcript {
                                            text: output.text,
                                            timed_word_stream,
                                            occurred_at: observed_at,
                                        })
                                        .is_err()
                                    {
                                        return;
                                    }
                                }
                            }
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
                    Arc::clone(&config.speaker_reference),
                    config.live_audio.clone(),
                    config.browser_audio_rx,
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
    active_groups: HashMap<BreathGroupId, ActiveAsrGroup>,
    environment: EnvironmentalSoundObserver,
    auditory_scene: AuditorySceneAnalyzer,
    live_audio: Option<listenbury::web::LiveSessionAudioStore>,
    frame_time_ms: u64,
    vad_observation: VadObservationState,
    last_self_hearing_observation_ms: Option<u64>,
    last_overlap_observation_ms: Option<u64>,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
#[derive(Debug, Clone)]
struct ActiveAsrGroup {
    frames: Vec<AudioFrame>,
    next_prospective_at_ms: u64,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
impl ActiveAsrGroup {
    fn new(opened_at_ms: u64) -> Self {
        Self {
            frames: Vec::new(),
            next_prospective_at_ms: opened_at_ms.saturating_add(PROSPECTIVE_ASR_INITIAL_MS),
        }
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
#[derive(Debug, Clone)]
struct ContinueAsrWorkItem {
    frames: Vec<AudioFrame>,
    is_final: bool,
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
enum VadObservationKind {
    Silence,
    Voice,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
#[derive(Debug, Clone, Copy)]
struct VadObservationState {
    kind: VadObservationKind,
    started_at_ms: u64,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
const SELF_HEARING_OBSERVATION_INTERVAL_MS: u64 = 2_000;

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
const OVERLAP_OBSERVATION_INTERVAL_MS: u64 = 500;

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const ENVIRONMENTAL_ASR_SILENCE_PADDING_MS: u64 = 250;

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
const PROSPECTIVE_ASR_INITIAL_MS: u64 = 300;

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
const PROSPECTIVE_ASR_INTERVAL_MS: u64 = 250;

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn run_continue_ear_processor(
    sample_rx: crossbeam_channel::Receiver<f32>,
    asr_tx: crossbeam_channel::Sender<ContinueAsrWorkItem>,
    event_tx: crossbeam_channel::Sender<ContinueEarEvent>,
    stop: Arc<AtomicBool>,
    vad_backend: VadBackendKind,
    input_sample_rate_hz: u32,
    input_channels: u16,
    speaker_reference: Arc<Mutex<SpeakerReferenceMask>>,
    live_audio: Option<listenbury::web::LiveSessionAudioStore>,
    browser_audio_rx: Option<crossbeam_channel::Receiver<AudioFrame>>,
) -> Result<()> {
    boost_current_thread_for_capture("listenbury-dev-continue-ear");

    let input_frame_samples =
        frame_samples_per_callback_frame(input_sample_rate_hz, input_channels);
    let (frame_sample_rate_hz, frame_channels) =
        vad_frame_format(vad_backend, input_sample_rate_hz, input_channels);
    let mut pending = VecDeque::<f32>::new();
    let mut state = ContinueEarState {
        vad: create_vad_backend(vad_backend)?,
        segmenter: BreathGroupSegmenter::default(),
        active_groups: HashMap::new(),
        environment: EnvironmentalSoundObserver::default(),
        auditory_scene: AuditorySceneAnalyzer::new(speaker_reference),
        live_audio,
        frame_time_ms: 0,
        vad_observation: VadObservationState {
            kind: VadObservationKind::Silence,
            started_at_ms: 0,
        },
        last_self_hearing_observation_ms: None,
        last_overlap_observation_ms: None,
    };

    while !stop.load(Ordering::Relaxed) {
        drain_browser_audio_frames(&browser_audio_rx, &mut state, &asr_tx, &event_tx)?;
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

    drain_browser_audio_frames(&browser_audio_rx, &mut state, &asr_tx, &event_tx)?;
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
    for (_, group) in state.active_groups.drain() {
        if !queue_final_asr_work(&asr_tx, group.frames) {
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
fn drain_browser_audio_frames(
    browser_audio_rx: &Option<crossbeam_channel::Receiver<AudioFrame>>,
    state: &mut ContinueEarState,
    asr_tx: &crossbeam_channel::Sender<ContinueAsrWorkItem>,
    event_tx: &crossbeam_channel::Sender<ContinueEarEvent>,
) -> Result<()> {
    let Some(browser_audio_rx) = browser_audio_rx else {
        return Ok(());
    };
    while let Ok(frame) = browser_audio_rx.try_recv() {
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
fn drain_pending_continue_ear_frames(
    pending: &mut VecDeque<f32>,
    input_frame_samples: usize,
    input_sample_rate_hz: u32,
    input_channels: u16,
    frame_sample_rate_hz: u32,
    frame_channels: u16,
    state: &mut ContinueEarState,
    asr_tx: &crossbeam_channel::Sender<ContinueAsrWorkItem>,
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
            voice_signatures: Vec::new(),
        };
        process_continue_ear_frame(frame, state, asr_tx, event_tx)?;
    }
    Ok(())
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
fn vad_observation_message(kind: VadObservationKind, duration_ms: u64) -> String {
    match kind {
        VadObservationKind::Silence => format!("I heard silence for {duration_ms} ms."),
        VadObservationKind::Voice => {
            format!(
                "I heard what sounded like a voice for {}.",
                format_seconds_duration(duration_ms)
            )
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
fn format_seconds_duration(duration_ms: u64) -> String {
    if duration_ms < 1_000 {
        format!("{:.2} s", duration_ms as f64 / 1_000.0)
    } else if duration_ms < 10_000 {
        format!("{:.1} s", duration_ms as f64 / 1_000.0)
    } else {
        format!("{} s", duration_ms / 1_000)
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn send_vad_observation_transition(
    state: &mut ContinueEarState,
    next_kind: VadObservationKind,
    event_tx: &crossbeam_channel::Sender<ContinueEarEvent>,
) {
    if state.vad_observation.kind == next_kind {
        return;
    }

    let duration_ms = state
        .frame_time_ms
        .saturating_sub(state.vad_observation.started_at_ms);
    if duration_ms > 0 {
        let _ = event_tx.send(ContinueEarEvent::AuditoryObservation {
            text: vad_observation_message(state.vad_observation.kind, duration_ms),
        });
    }
    state.vad_observation = VadObservationState {
        kind: next_kind,
        started_at_ms: state.frame_time_ms,
    };
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
    asr_tx: &crossbeam_channel::Sender<ContinueAsrWorkItem>,
    event_tx: &crossbeam_channel::Sender<ContinueEarEvent>,
) -> Result<()> {
    if let Some(live_audio) = &state.live_audio {
        live_audio.push_frame(frame.clone());
    }
    let frame_duration_ms = frame_duration_ms(&frame);
    let analysis = state.auditory_scene.analyze(frame)?;
    log_auditory_frame_if_enabled(&analysis);
    match analysis.routing {
        AuditoryRouting::EchoOnly => {
            send_self_hearing_event_if_due(state, event_tx, &analysis);
            state.frame_time_ms = state.frame_time_ms.saturating_add(frame_duration_ms);
            return Ok(());
        }
        AuditoryRouting::MixedSelfAndExternal => {
            send_overlap_event_if_due(state, event_tx, &analysis, frame_duration_ms);
            if let Some(residual) = analysis.external_residual_frame().cloned() {
                process_continue_vad_and_asr_frame(residual, state, asr_tx, event_tx)?;
            }
        }
        AuditoryRouting::ExternalSpeechCandidate => {
            process_continue_vad_and_asr_frame(analysis.frame, state, asr_tx, event_tx)?;
        }
        AuditoryRouting::EnvironmentalSoundCandidate => {
            if let Some(clip) = state
                .environment
                .observe_frame_with_audio(&analysis.frame, false)
            {
                send_environmental_sound_clip(clip, asr_tx, event_tx);
            }
            process_continue_segmenter_silence_frame(
                analysis.frame.clone(),
                state,
                asr_tx,
                event_tx,
            )?;
            send_vad_observation_transition(state, VadObservationKind::Silence, event_tx);
        }
        AuditoryRouting::SilenceOrNoise => {
            if let Some(clip) = state
                .environment
                .observe_frame_with_audio(&analysis.frame, false)
            {
                if clip.sound.label.as_deref() != Some("silence") {
                    send_environmental_sound_clip(clip, asr_tx, event_tx);
                }
            }
            process_continue_segmenter_silence_frame(
                analysis.frame.clone(),
                state,
                asr_tx,
                event_tx,
            )?;
            send_vad_observation_transition(state, VadObservationKind::Silence, event_tx);
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
fn send_environmental_sound_clip(
    clip: EnvironmentalSoundClip,
    asr_tx: &crossbeam_channel::Sender<ContinueAsrWorkItem>,
    event_tx: &crossbeam_channel::Sender<ContinueEarEvent>,
) {
    let EnvironmentalSoundClip { sound, frames } = clip;
    let should_transcribe = sound.label.as_deref() != Some("silence") && !frames.is_empty();
    let _ = event_tx.send(ContinueEarEvent::EnvironmentalSound { sound });

    if should_transcribe {
        let frames = padded_environmental_asr_frames(&frames, ENVIRONMENTAL_ASR_SILENCE_PADDING_MS);
        if !frames.is_empty() {
            let _ = asr_tx.send(ContinueAsrWorkItem {
                frames,
                is_final: true,
            });
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
fn padded_environmental_asr_frames(frames: &[AudioFrame], padding_ms: u64) -> Vec<AudioFrame> {
    let Some(first) = frames.first() else {
        return Vec::new();
    };
    let frame_duration_ms = frame_duration_ms(first);
    if frame_duration_ms == 0 || padding_ms == 0 {
        return frames.to_vec();
    }

    let padding_frames = padding_ms.div_ceil(frame_duration_ms) as usize;
    let silence_frame = || AudioFrame {
        captured_at: ExactTimestamp::now(),
        sample_rate_hz: first.sample_rate_hz,
        channels: first.channels,
        samples: vec![0.0; first.samples.len()],
        voice_signatures: Vec::new(),
    };

    let mut padded = Vec::with_capacity(frames.len().saturating_add(padding_frames * 2));
    padded.extend((0..padding_frames).map(|_| silence_frame()));
    padded.extend(frames.iter().cloned());
    padded.extend((0..padding_frames).map(|_| silence_frame()));
    padded
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn log_auditory_frame_if_enabled(analysis: &AuditoryFrameAnalysis) {
    if !listenbury::developer_diagnostics_enabled() {
        return;
    }
    if analysis.routing == AuditoryRouting::SilenceOrNoise {
        return;
    }
    eprintln!(
        "[ear] routing={:?} rms={:.4} zcr={:.3} brightness={:.3} vad_speech={} voice_score={:.3} env_score={:.3} noise_floor={:.4} corr={:.3} residual={:.3} delay_ms={} env_hysteresis={} reason={}",
        analysis.routing,
        analysis.diagnostics.rms,
        analysis.diagnostics.zero_crossing_rate,
        analysis.diagnostics.brightness,
        analysis.external.vad_candidate,
        analysis.diagnostics.voice_score,
        analysis.diagnostics.environment_score,
        analysis.diagnostics.noise_floor_rms,
        analysis.self_voice.correlation,
        analysis.self_voice.residual_ratio,
        analysis.self_voice.delay_ms,
        analysis.diagnostics.environmental_hysteresis_frames,
        analysis.diagnostics.routing_reason
    );
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn process_continue_vad_and_asr_frame(
    frame: AudioFrame,
    state: &mut ContinueEarState,
    asr_tx: &crossbeam_channel::Sender<ContinueAsrWorkItem>,
    event_tx: &crossbeam_channel::Sender<ContinueEarEvent>,
) -> Result<()> {
    let vad_result = state.vad.process_frame(&frame)?;
    if let Some(sound) = state
        .environment
        .observe_frame(&frame, vad_result.is_speech)
    {
        let _ = event_tx.send(ContinueEarEvent::EnvironmentalSound { sound });
    }
    let events = state.segmenter.process(vad_result);
    process_continue_segmenter_events(frame, state, asr_tx, event_tx, events)
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn process_continue_segmenter_silence_frame(
    frame: AudioFrame,
    state: &mut ContinueEarState,
    asr_tx: &crossbeam_channel::Sender<ContinueAsrWorkItem>,
    event_tx: &crossbeam_channel::Sender<ContinueEarEvent>,
) -> Result<()> {
    let events = state.segmenter.process(VadResult {
        speech_prob: 0.0,
        is_speech: false,
    });
    process_continue_segmenter_events(frame, state, asr_tx, event_tx, events)
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn process_continue_segmenter_events(
    frame: AudioFrame,
    state: &mut ContinueEarState,
    asr_tx: &crossbeam_channel::Sender<ContinueAsrWorkItem>,
    event_tx: &crossbeam_channel::Sender<ContinueEarEvent>,
    events: Vec<HearingEvent>,
) -> Result<()> {
    for event in &events {
        match event {
            HearingEvent::SpeechStarted => {
                send_vad_observation_transition(state, VadObservationKind::Voice, event_tx);
                let _ = event_tx.send(ContinueEarEvent::SpeechStarted);
            }
            HearingEvent::BreathGroupOpened { id } => {
                state
                    .active_groups
                    .entry(*id)
                    .or_insert_with(|| ActiveAsrGroup::new(state.frame_time_ms));
            }
            HearingEvent::BreathGroupClosed { .. } => {
                let _ = event_tx.send(ContinueEarEvent::SpeechStopped);
                send_vad_observation_transition(state, VadObservationKind::Silence, event_tx);
            }
            HearingEvent::SpeechContinued { .. } | HearingEvent::PauseStarted => {}
        }
    }
    for group in state.active_groups.values_mut() {
        group.frames.push(frame.clone());
    }
    for event in events {
        if let HearingEvent::BreathGroupClosed { id, .. } = event {
            if let Some(group) = state.active_groups.remove(&id) {
                if !queue_final_asr_work(asr_tx, group.frames) {
                    return Ok(());
                }
            }
        }
    }
    let frame_end_ms = state
        .frame_time_ms
        .saturating_add(frame_duration_ms(&frame));
    for group in state.active_groups.values_mut() {
        if group.frames.is_empty() || frame_end_ms < group.next_prospective_at_ms {
            continue;
        }
        queue_prospective_asr_snapshot(asr_tx, group.frames.clone());
        group.next_prospective_at_ms = frame_end_ms.saturating_add(PROSPECTIVE_ASR_INTERVAL_MS);
    }
    Ok(())
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn queue_final_asr_work(
    asr_tx: &crossbeam_channel::Sender<ContinueAsrWorkItem>,
    frames: Vec<AudioFrame>,
) -> bool {
    if frames.is_empty() {
        return true;
    }
    asr_tx
        .send(ContinueAsrWorkItem {
            frames,
            is_final: true,
        })
        .is_ok()
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn queue_prospective_asr_snapshot(
    asr_tx: &crossbeam_channel::Sender<ContinueAsrWorkItem>,
    frames: Vec<AudioFrame>,
) {
    if frames.is_empty() {
        return;
    }
    let _ = asr_tx.try_send(ContinueAsrWorkItem {
        frames,
        is_final: false,
    });
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn send_self_hearing_event_if_due(
    state: &mut ContinueEarState,
    event_tx: &crossbeam_channel::Sender<ContinueEarEvent>,
    analysis: &AuditoryFrameAnalysis,
) {
    if !rate_limit_elapsed(
        state.last_self_hearing_observation_ms,
        state.frame_time_ms,
        SELF_HEARING_OBSERVATION_INTERVAL_MS,
    ) {
        return;
    }
    state.last_self_hearing_observation_ms = Some(state.frame_time_ms);
    let _ = event_tx.send(ContinueEarEvent::SelfVoiceHeard {
        delay_ms: analysis.self_voice.delay_ms,
        gain: analysis.self_voice.gain,
        confidence: analysis.self_voice.confidence,
    });
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn send_overlap_event_if_due(
    state: &mut ContinueEarState,
    event_tx: &crossbeam_channel::Sender<ContinueEarEvent>,
    analysis: &AuditoryFrameAnalysis,
    duration_ms: u64,
) {
    if !rate_limit_elapsed(
        state.last_overlap_observation_ms,
        state.frame_time_ms,
        OVERLAP_OBSERVATION_INTERVAL_MS,
    ) {
        return;
    }
    state.last_overlap_observation_ms = Some(state.frame_time_ms);
    let _ = event_tx.send(ContinueEarEvent::OverlapDetected {
        self_confidence: analysis.self_voice.confidence,
        external_confidence: analysis.external.confidence,
        duration_ms,
    });
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn rate_limit_elapsed(last_at_ms: Option<u64>, now_ms: u64, interval_ms: u64) -> bool {
    last_at_ms
        .map(|last| now_ms.saturating_sub(last) >= interval_ms)
        .unwrap_or(true)
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

    let mut current_channels = input_channels;
    let mut converted = if input_channels != frame_channels && frame_channels == MONO_CHANNELS {
        current_channels = MONO_CHANNELS;
        mix_to_mono(samples, input_channels)
    } else {
        samples.to_vec()
    };

    if input_sample_rate_hz != frame_sample_rate_hz {
        converted = resample_interleaved(
            &converted,
            input_sample_rate_hz,
            frame_sample_rate_hz,
            current_channels,
        );
    }

    if current_channels != frame_channels {
        converted = convert_channels(&converted, current_channels, frame_channels);
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

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
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
fn convert_channels(samples: &[f32], source_channels: u16, target_channels: u16) -> Vec<f32> {
    if source_channels == target_channels {
        return samples.to_vec();
    }

    if target_channels == MONO_CHANNELS {
        return mix_to_mono(samples, source_channels);
    }

    let source_channel_count = usize::from(source_channels).max(1);
    let target_channel_count = usize::from(target_channels).max(1);
    if source_channel_count == 1 {
        let mut converted = Vec::with_capacity(samples.len().saturating_mul(target_channel_count));
        for sample in samples {
            converted.extend(std::iter::repeat_n(*sample, target_channel_count));
        }
        return converted;
    }

    let mut converted = Vec::with_capacity(
        samples
            .len()
            .saturating_div(source_channel_count)
            .saturating_mul(target_channel_count),
    );
    for frame in samples.chunks_exact(source_channel_count) {
        for channel_idx in 0..target_channel_count {
            converted.push(frame[channel_idx.min(source_channel_count - 1)]);
        }
    }
    converted
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
fn resample_interleaved(
    samples: &[f32],
    source_rate_hz: u32,
    target_rate_hz: u32,
    channels: u16,
) -> Vec<f32> {
    let channel_count = usize::from(channels).max(1);
    if channel_count == 1 {
        return resample_linear(samples, source_rate_hz, target_rate_hz);
    }

    let frame_count = samples.len() / channel_count;
    if frame_count == 0 || source_rate_hz == target_rate_hz {
        return samples.to_vec();
    }

    let output_frame_count = ((frame_count as f64 * f64::from(target_rate_hz))
        / f64::from(source_rate_hz))
    .round() as usize;
    let mut output = Vec::with_capacity(output_frame_count.saturating_mul(channel_count));
    let source_step = f64::from(source_rate_hz) / f64::from(target_rate_hz);

    for output_frame_idx in 0..output_frame_count {
        let source_pos = output_frame_idx as f64 * source_step;
        let left_frame_idx = source_pos.floor() as usize;
        let right_frame_idx = (left_frame_idx + 1).min(frame_count - 1);
        let fraction = (source_pos - left_frame_idx as f64) as f32;
        for channel_idx in 0..channel_count {
            let left = samples[left_frame_idx * channel_count + channel_idx];
            let right = samples[right_frame_idx * channel_count + channel_idx];
            output.push(left + (right - left) * fraction);
        }
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
        speaker_reference: Arc<Mutex<SpeakerReferenceMask>>,
    ) -> Result<(Self, crossbeam_channel::Receiver<ContinueMouthEvent>)> {
        let (tx, rx) = crossbeam_channel::unbounded();
        let (event_tx, event_rx) = crossbeam_channel::unbounded();
        let worker = std::thread::Builder::new()
            .name("listenbury-dev-continue-mouth".to_string())
            .spawn(move || {
                run_continue_mouth_worker(tts, rx, event_tx, capture_enabled, speaker_reference)
            })
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
        let Some((command, pending_speech)) = mouth_command_for_runtime_event(event) else {
            return Ok(false);
        };
        self.tx
            .send(command)
            .context("failed to send runtime event to dev continue mouth worker")?;
        Ok(pending_speech)
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
            | Self::SpeechQueueCleared { .. }
            | Self::SpeechError { .. } => *paused = false,
            Self::WorkerStarted
            | Self::SpeechQueued { .. }
            | Self::SpeechSynthesisStarted { .. }
            | Self::SpeechPlaybackStarted { .. } => {}
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
    speaker_reference: Arc<Mutex<SpeakerReferenceMask>>,
) {
    let _ = event_tx.send(ContinueMouthEvent::WorkerStarted);
    let mut pending = VecDeque::<PendingMouthSpeech>::new();
    let mut paused = false;
    loop {
        let command = if let Some(speech) = pending.pop_front() {
            ContinueMouthCommand::Speak {
                id: speech.id,
                text: speech.text,
                interrupt: speech.interrupt,
            }
        } else {
            match rx.recv() {
                Ok(command) => command,
                Err(_) => return,
            }
        };
        match command {
            ContinueMouthCommand::Speak {
                id,
                text,
                interrupt,
            } => {
                match run_continue_mouth_speech(
                    id,
                    text,
                    interrupt,
                    &mut tts,
                    &rx,
                    &mut pending,
                    &event_tx,
                    &capture_enabled,
                    &speaker_reference,
                    &mut paused,
                ) {
                    Ok(MouthWorkerFlow::Continue) | Err(_) => {}
                    Ok(MouthWorkerFlow::Shutdown) => return,
                }
            }
            ContinueMouthCommand::Shutup => {
                let _ = tts.stop();
                resume_mouth_playback(&event_tx, &mut paused);
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
#[derive(Debug, Clone, PartialEq, Eq)]
struct PendingMouthSpeech {
    id: u64,
    text: String,
    interrupt: bool,
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
    pending: &mut VecDeque<PendingMouthSpeech>,
    event_tx: &crossbeam_channel::Sender<ContinueMouthEvent>,
    tts: &mut PiperTextToSpeech,
    paused: &mut bool,
) -> MouthControlFlow {
    loop {
        match rx.try_recv() {
            Ok(ContinueMouthCommand::Speak {
                id,
                text,
                interrupt,
            }) => pending.push_back(PendingMouthSpeech {
                id,
                text,
                interrupt,
            }),
            Ok(ContinueMouthCommand::Pause) => pause_mouth_playback(event_tx, paused),
            Ok(ContinueMouthCommand::Resume) => resume_mouth_playback(event_tx, paused),
            Ok(ContinueMouthCommand::Shutup) => {
                let _ = tts.stop();
                resume_mouth_playback(event_tx, paused);
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
    pending: &mut VecDeque<PendingMouthSpeech>,
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
    _interrupt: bool,
    tts: &mut PiperTextToSpeech,
    rx: &crossbeam_channel::Receiver<ContinueMouthCommand>,
    pending: &mut VecDeque<PendingMouthSpeech>,
    event_tx: &crossbeam_channel::Sender<ContinueMouthEvent>,
    _capture_enabled: &AtomicBool,
    speaker_reference: &Arc<Mutex<SpeakerReferenceMask>>,
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
    let playback = play_continue_audio_frames_interruptible(
        &frames,
        "listenbury dev continue speech",
        rx,
        pending,
        event_tx,
        tts,
        speaker_reference,
        paused,
    );
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
    pending: &mut VecDeque<PendingMouthSpeech>,
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
    pending: &mut VecDeque<PendingMouthSpeech>,
    event_tx: &crossbeam_channel::Sender<ContinueMouthEvent>,
    tts: &mut PiperTextToSpeech,
    speaker_reference: &Arc<Mutex<SpeakerReferenceMask>>,
    paused: &mut bool,
) -> Result<MouthPlaybackOutcome> {
    let playback = prepare_audio_playback(frames, source)?;
    let playback_cursor = Arc::new(AtomicUsize::new(0));
    let playback_paused = Arc::new(AtomicBool::new(*paused));
    let done_threshold = playback.sample_count();
    let stream =
        playback.build_stream(Arc::clone(&playback_cursor), Arc::clone(&playback_paused))?;
    stream
        .play()
        .with_context(|| format!("failed to start playback on {}", playback.device_name))?;
    {
        let started_at = ExactTimestamp::now();
        let reference_frame = playback.as_audio_frame(started_at);
        let mut speaker_reference = speaker_reference
            .lock()
            .map_err(|_| anyhow::anyhow!("speaker reference mask lock poisoned"))?;
        speaker_reference.mark_output_started(&[reference_frame], started_at);
    }

    while playback_cursor.load(Ordering::Relaxed) < done_threshold {
        match drain_mouth_control_commands(rx, pending, event_tx, tts, paused) {
            MouthControlFlow::Continue => {
                playback_paused.store(*paused, Ordering::Relaxed);
            }
            MouthControlFlow::StopCurrent => {
                if let Ok(mut speaker_reference) = speaker_reference.lock() {
                    speaker_reference.mark_output_finished();
                }
                drop(stream);
                return Ok(MouthPlaybackOutcome::Interrupted);
            }
            MouthControlFlow::Shutdown => {
                if let Ok(mut speaker_reference) = speaker_reference.lock() {
                    speaker_reference.mark_output_finished();
                }
                drop(stream);
                return Ok(MouthPlaybackOutcome::Shutdown);
            }
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    std::thread::sleep(Duration::from_millis(20));
    drop(stream);
    if let Ok(mut speaker_reference) = speaker_reference.lock() {
        speaker_reference.mark_output_finished();
    }

    let audio_duration = playback.duration();
    println!(
        "Played with {}: {} Hz, {} channel(s), {:.2}s from {source}",
        playback.device_name,
        playback.sample_rate_hz,
        playback.channels,
        audio_duration.as_secs_f64(),
    );

    Ok(MouthPlaybackOutcome::Completed)
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
    UtteranceCompleted {
        id: u64,
        content: String,
        interrupt: bool,
    },
    SpeechControl {
        command: SpeechControlCommand,
    },
    SourceCommand {
        command: SourceCommand,
    },
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

#[cfg(test)]
impl ContinueRuntimeEvent {
    fn to_message(&self) -> String {
        match self {
            Self::UtteranceCompleted {
                id,
                content,
                interrupt,
            } => {
                format!(
                    "utterance_completed: id={id} interrupt={interrupt}\ncontent:\n{}",
                    sanitize_runtime_event_content(content)
                )
            }
            Self::SpeechControl { command } => format!("speech_control: {}", command.as_str()),
            Self::SourceCommand { command } => match command {
                SourceCommand::RunTypeScript { source } => {
                    format!(
                        "source_command: typescript\nsource:\n{}",
                        sanitize_runtime_event_content(source)
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
#[derive(Debug, Default)]
struct HarmonyFinalFilter {
    pending: String,
    in_final: bool,
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
impl HarmonyFinalFilter {
    fn filter_events(&mut self, events: &[LlmEvent]) -> Vec<LlmEvent> {
        let mut filtered = Vec::new();
        for event in events {
            match event {
                LlmEvent::Token { text } => {
                    let text = self.push(text);
                    if !text.is_empty() {
                        filtered.push(LlmEvent::Token { text });
                    }
                }
                LlmEvent::Completed | LlmEvent::Cancelled | LlmEvent::Error { .. } => {
                    let text = self.finish();
                    if !text.is_empty() {
                        filtered.push(LlmEvent::Token { text });
                    }
                    filtered.push(event.clone());
                }
            }
        }
        filtered
    }

    fn push(&mut self, text: &str) -> String {
        self.pending.push_str(text);
        self.drain(false)
    }

    fn finish(&mut self) -> String {
        self.drain(true)
    }

    fn drain(&mut self, completed: bool) -> String {
        let mut visible = String::new();
        loop {
            if self.in_final {
                if let Some((start, marker)) = first_marker(&self.pending, HARMONY_FINAL_ENDS) {
                    visible.push_str(&self.pending[..start]);
                    self.pending.drain(..start + marker.len());
                    self.in_final = false;
                    continue;
                }
                let keep_from = if completed {
                    self.pending.len()
                } else {
                    possible_marker_prefix_start(&self.pending, HARMONY_FINAL_ENDS)
                };
                visible.push_str(&self.pending[..keep_from]);
                self.pending.drain(..keep_from);
                break;
            }

            if let Some((start, marker)) = first_marker(&self.pending, HARMONY_FINAL_STARTS) {
                self.pending.drain(..start + marker.len());
                self.in_final = true;
                continue;
            }
            if completed {
                self.pending.clear();
            } else {
                keep_possible_marker_prefix(&mut self.pending, HARMONY_FINAL_STARTS);
            }
            break;
        }
        visible
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
const HARMONY_FINAL_STARTS: &[&str] = &[
    "<|channel|>final<|message|>",
    "<|start|>assistant<|channel|>final<|message|>",
];

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
const HARMONY_FINAL_ENDS: &[&str] = &["<|end|>", "<|return|>", "<|start|>"];

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
fn first_marker<'a>(text: &str, markers: &'a [&str]) -> Option<(usize, &'a str)> {
    markers
        .iter()
        .filter_map(|marker| text.find(marker).map(|index| (index, *marker)))
        .min_by_key(|(index, _)| *index)
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
fn keep_possible_marker_prefix(text: &mut String, markers: &[&str]) {
    let keep_from = possible_marker_prefix_start(text, markers);
    text.drain(..keep_from);
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
fn possible_marker_prefix_start(text: &str, markers: &[&str]) -> usize {
    (0..text.len())
        .find(|&index| {
            text.is_char_boundary(index)
                && markers.iter().any(|marker| {
                    let suffix = &text[index..];
                    !suffix.is_empty() && suffix.len() < marker.len() && marker.starts_with(suffix)
                })
        })
        .unwrap_or(text.len())
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
        longest_marker_prefix_suffix_len(&self.pending) > 0
            || incomplete_source_tag_start(&self.pending).is_some()
    }

    fn ingest(&mut self, text: &str) -> Vec<ContinueRuntimeEvent> {
        self.pending.push_str(text);
        let mut events = Vec::new();

        loop {
            let Some(marker) = next_typescript_marker(&self.pending) else {
                self.trim_pending_to_marker_prefix_or_source_tag();
                return events;
            };
            let marker_end = marker.index + marker.len;
            self.pending.drain(..marker_end);
            events.push(ContinueRuntimeEvent::SourceCommand {
                command: marker.command,
            });
        }
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
struct TypeScriptMarker {
    command: SourceCommand,
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
fn next_typescript_marker(text: &str) -> Option<TypeScriptMarker> {
    let start = text.find(SOURCE_TYPESCRIPT_START)?;
    let content_start = start + SOURCE_TYPESCRIPT_START.len();
    let rest = &text[content_start..];
    let end_rel = rest.find(SOURCE_TYPESCRIPT_END)?;
    let source = rest[..end_rel].trim().to_string();
    Some(TypeScriptMarker {
        command: SourceCommand::RunTypeScript { source },
        index: start,
        len: SOURCE_TYPESCRIPT_START.len() + end_rel + SOURCE_TYPESCRIPT_END.len(),
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
fn incomplete_source_tag_start(text: &str) -> Option<usize> {
    text.rfind(SOURCE_TYPESCRIPT_START).and_then(|start| {
        let rest = &text[start..];
        (!rest.contains(SOURCE_TYPESCRIPT_END)).then_some(start)
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
fn all_speech_markers() -> [&'static str; 2] {
    [SOURCE_TYPESCRIPT_START, SOURCE_TYPESCRIPT_END]
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
#[allow(dead_code)]
fn sanitize_runtime_event_content(content: &str) -> String {
    content
        .replace("--- END LIVE EVENT ---", "[end live event]")
        .replace("--- LIVE EVENT:", "[live event]")
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
fn clean_spoken_content(content: &str) -> Option<String> {
    if contains_template_token(content) {
        return None;
    }

    let content = strip_emoji(content).trim().to_string();
    if content.is_empty() || contains_template_token(&content) {
        None
    } else {
        Some(content)
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
fn contains_template_token(content: &str) -> bool {
    let lower = content.to_ascii_lowercase();
    [
        "<|",
        "|>",
        "<|start|>",
        "<|end|>",
        "<|message|>",
        "<|channel|>",
        "assistant<|",
        "system<|",
        "analysis<|",
        "thoughts<|",
        "assistant<|channel|>",
        "assistant<|message|>",
    ]
    .into_iter()
    .any(|marker| lower.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::VadObservationKind;
    use super::{
        ContinueEarEvent, ContinueMouthCommand, ContinuePromptFormat, ContinuePromptGate,
        ContinuePromptGateConfig, ContinueRuntimeEvent, HarmonyFinalFilter, PromptPacket,
        RollingContextManager, SourceCommand, SpeechControlCommand, SpeechEventDetector,
        TIME_EVENT_INTERVAL_BASE_MS, TIME_EVENT_INTERVAL_JITTER_MS, TranscriptSpeculativePlanner,
        TypeScriptCommand, build_continue_prompt, build_initial_prompt, clean_spoken_content,
        continue_prompt_format_for_model, current_time_message, execute_list_source_files,
        execute_typescript_commands, execute_typescript_source, execute_view_source_file,
        mouth_command_for_runtime_event, next_time_event_interval, padded_environmental_asr_frames,
        vad_observation_message, wrap_ear_event, wrap_live_input, wrap_mouth_event,
        wrap_runtime_event, wrap_source_event, wrap_time_event,
    };
    use listenbury::AudioFrame;
    use listenbury::ExactTimestamp;
    use listenbury::mind::llm::LlmEvent;
    use listenbury::speech::transcript::{
        TranscriptCandidateEvent, TranscriptCandidateId, TranscriptReplacementReason,
    };
    use listenbury::word::{TimedWordStream, WordStreamId, WordStreamSource};

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
    fn vad_observations_are_short_ear_events() {
        assert_eq!(
            vad_observation_message(VadObservationKind::Silence, 160),
            "I heard silence for 160 ms."
        );
        assert_eq!(
            vad_observation_message(VadObservationKind::Voice, 2_410),
            "I heard what sounded like a voice for 2.4 s."
        );

        let packet = PromptPacket::ear_observation("I heard silence for 160 ms.".to_string());
        assert_eq!(
            packet.text,
            "\n\n--- LIVE EVENT: ear ---\nI heard silence for 160 ms.\n--- END LIVE EVENT ---\n\n"
        );
    }

    #[test]
    fn prompt_gate_deduplicates_repeated_observations() {
        let mut gate = ContinuePromptGate::new(ContinuePromptGateConfig {
            duplicate_suppression_window: std::time::Duration::from_millis(500),
            auditory_min_interval: std::time::Duration::from_millis(0),
            overlap_summary_threshold: 3,
        });
        let now = std::time::Instant::now();
        let event = ContinueEarEvent::AuditoryObservation {
            text: "I heard silence for 160 ms.".to_string(),
        };

        assert_eq!(gate.consider_ear_event(&event, now).len(), 1);
        assert!(
            gate.consider_ear_event(&event, now + std::time::Duration::from_millis(100))
                .is_empty()
        );
        assert_eq!(
            gate.consider_ear_event(&event, now + std::time::Duration::from_millis(800))
                .len(),
            1
        );
    }

    #[test]
    fn prompt_gate_suppresses_auditory_bursts() {
        let mut gate = ContinuePromptGate::new(ContinuePromptGateConfig {
            duplicate_suppression_window: std::time::Duration::from_millis(0),
            auditory_min_interval: std::time::Duration::from_millis(1_000),
            overlap_summary_threshold: 3,
        });
        let now = std::time::Instant::now();

        let first = gate.consider_ear_event(
            &ContinueEarEvent::AuditoryObservation {
                text: "first".to_string(),
            },
            now,
        );
        let second = gate.consider_ear_event(
            &ContinueEarEvent::AuditoryObservation {
                text: "second".to_string(),
            },
            now + std::time::Duration::from_millis(50),
        );

        assert_eq!(first.len(), 1);
        assert!(second.is_empty());
    }

    #[test]
    fn prompt_gate_coalesces_overlap_into_summary() {
        let mut gate = ContinuePromptGate::new(ContinuePromptGateConfig {
            duplicate_suppression_window: std::time::Duration::from_millis(0),
            auditory_min_interval: std::time::Duration::from_millis(0),
            overlap_summary_threshold: 2,
        });
        let now = std::time::Instant::now();

        assert!(
            gate.consider_ear_event(
                &ContinueEarEvent::OverlapDetected {
                    self_confidence: 0.8,
                    external_confidence: 0.7,
                    duration_ms: 90
                },
                now
            )
            .is_empty()
        );

        let packets = gate.consider_ear_event(
            &ContinueEarEvent::OverlapDetected {
                self_confidence: 0.82,
                external_confidence: 0.72,
                duration_ms: 92,
            },
            now + std::time::Duration::from_millis(10),
        );

        assert_eq!(packets.len(), 1);
        assert!(
            packets[0]
                .text
                .contains("Pete heard overlapping speech while speaking.")
        );
        assert!(!packets[0].text.contains("self_confidence"));
    }

    #[test]
    fn prompt_gate_allows_transcript_passthrough() {
        let mut gate = ContinuePromptGate::new(ContinuePromptGateConfig {
            duplicate_suppression_window: std::time::Duration::from_secs(30),
            auditory_min_interval: std::time::Duration::from_secs(30),
            overlap_summary_threshold: 3,
        });
        let now = std::time::Instant::now();
        let event = ContinueEarEvent::Transcript {
            text: "hello".to_string(),
            timed_word_stream: TimedWordStream::new(WordStreamId(1), WordStreamSource::LiveAsr),
            occurred_at: ExactTimestamp { unix_nanos: 0 },
        };

        let first = gate.consider_ear_event(&event, now);
        let second = gate.consider_ear_event(&event, now + std::time::Duration::from_millis(1));

        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), 1);
        assert!(first[0].text.contains("Heard: hello"));
        assert!(second[0].text.contains("Heard: hello"));
    }

    #[test]
    fn prompt_gate_treats_speech_stopped_as_response_prep_signal() {
        let mut gate = ContinuePromptGate::new(ContinuePromptGateConfig {
            duplicate_suppression_window: std::time::Duration::from_secs(30),
            auditory_min_interval: std::time::Duration::from_millis(0),
            overlap_summary_threshold: 3,
        });

        let packets =
            gate.consider_ear_event(&ContinueEarEvent::SpeechStopped, std::time::Instant::now());

        assert_eq!(packets.len(), 1);
        assert!(packets[0].text.contains("External speech stopped"));
        assert!(packets[0].text.contains("prepare a response"));
    }

    #[test]
    fn environmental_asr_padding_adds_silence_before_and_after_clip() {
        let clip = vec![AudioFrame {
            captured_at: ExactTimestamp { unix_nanos: 0 },
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![0.5; 160],
            voice_signatures: Vec::new(),
        }];

        let padded = padded_environmental_asr_frames(&clip, 20);

        assert_eq!(padded.len(), 5);
        assert!(padded[0].samples.iter().all(|sample| *sample == 0.0));
        assert!(padded[1].samples.iter().all(|sample| *sample == 0.0));
        assert_eq!(padded[2].samples, vec![0.5; 160]);
        assert!(padded[3].samples.iter().all(|sample| *sample == 0.0));
        assert!(padded[4].samples.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn transcript_speculative_planner_tracks_stable_and_unstable_segments() {
        let mut planner = TranscriptSpeculativePlanner::default();
        assert!(
            planner
                .observe(&TranscriptCandidateEvent::CandidateStarted {
                    id: TranscriptCandidateId(1),
                })
                .is_none()
        );

        let first = planner
            .observe(&TranscriptCandidateEvent::CandidateUpdated {
                id: TranscriptCandidateId(1),
                text: "can you".to_string(),
                stable_prefix_len: "can ".len(),
                confidence: Some(0.7),
            })
            .expect("stability update");
        assert_eq!(first.stable_text, "can ");
        assert_eq!(first.unstable_text, "you");
        assert_eq!(first.confidence, Some(0.7));

        let second = planner
            .observe(&TranscriptCandidateEvent::CandidateUpdated {
                id: TranscriptCandidateId(1),
                text: "can you tell".to_string(),
                stable_prefix_len: "can you".len(),
                confidence: Some(0.8),
            })
            .expect("stability update");
        assert_eq!(second.stable_text, "can you");
        assert_eq!(second.unstable_text, " tell");
        assert_eq!(second.confidence, Some(0.8));
    }

    #[test]
    fn transcript_candidate_head_replacement_is_exposed_beyond_recognizer() {
        let event = ContinueEarEvent::TranscriptCandidate {
            event: TranscriptCandidateEvent::CandidateReplaced {
                old: TranscriptCandidateId(3),
                new: TranscriptCandidateId(4),
                reason: TranscriptReplacementReason::HeadChanged {
                    stable_prefix_len: 0,
                },
            },
            stability: None,
            occurred_at: ExactTimestamp { unix_nanos: 42 },
        };

        assert!(event.to_message().contains(
            "transcript_candidate_replaced: old=3 new=4 reason=head_changed stable_prefix_len=0"
        ));
        assert!(event.direct_prompt_packet().is_none());
    }

    #[test]
    fn transcript_speculative_planner_snaps_to_character_boundary() {
        let mut planner = TranscriptSpeculativePlanner::default();
        let state = planner
            .observe(&TranscriptCandidateEvent::CandidateUpdated {
                id: TranscriptCandidateId(1),
                text: "héllo".to_string(),
                stable_prefix_len: 2,
                confidence: None,
            })
            .expect("stability update");
        assert_eq!(state.stable_text, "hé");
        assert_eq!(state.unstable_text, "llo");
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
            ContinueRuntimeEvent::SpeechControl {
                command: SpeechControlCommand::Pause
            }
            .to_message(),
            "speech_control: pause"
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
        assert!(prompt.contains("Do not assume a user is currently present"));
        assert!(prompt.contains("there may be nobody in the room"));
        assert!(prompt.contains("Clock events arrive frequently"));
        assert!(prompt.contains("about once per second but at slightly irregular intervals"));
        assert!(prompt.contains("local ISO-8601 time and timezone offset"));
        assert!(prompt.contains("track timing, pauses, and elapsed time"));
        assert!(prompt.contains("Do not copy live event delimiters or runtime event text."));
        assert!(prompt.contains("Do not write system, assistant, analysis, channel, message"));
        assert!(prompt.contains("Plain generated text is Pete's internal thought only"));
        assert!(prompt.contains("It is not spoken aloud"));
        assert!(prompt.contains("does not happen in the real world"));
        assert!(prompt.contains("The only way to affect the real world"));
        assert!(prompt.contains("Speak sparingly"));
        assert!(prompt.contains("Do not narrate every clock tick"));
        assert!(prompt.contains("do not speak just to fill silence"));
        assert!(prompt.contains("If you are bored, alone, or waiting"));
        assert!(prompt.contains("explore Pete's own source code"));
        assert!(prompt.contains("say(text, options?), shutup(), pause(), resume()"));
        assert!(prompt.contains("say(text, { interrupt: true })"));
        assert!(prompt.contains("run small TypeScript modules with <ts>code</ts>"));
        assert!(prompt.contains("internal module \"pete:will\""));
        assert!(prompt.contains("Import the builders you need from \"pete:will\""));
        assert!(prompt.contains("import { say, listFiles } from \"pete:will\""));
        assert!(prompt.contains("listFiles(), readSourceFile(path, page?)"));
        assert!(prompt.contains("Use shutup() to halt current speech"));
        assert!(!prompt.contains("<sp>words to say aloud :)</sp>"));
        assert!(!prompt.contains("<shutup/>"));
        assert!(!prompt.contains("<list_files/>"));
        assert!(!prompt.contains("--- SPEECH ---"));
    }

    #[test]
    fn gpt_oss_continue_uses_harmony_prompt_format() {
        assert_eq!(
            continue_prompt_format_for_model(
                std::path::Path::new("models/llama/gpt-oss-20b-mxfp4.gguf"),
                crate::cli::PromptMode::Raw,
            ),
            ContinuePromptFormat::GptOssHarmony
        );

        let (prompt, stops) =
            build_continue_prompt(ContinuePromptFormat::GptOssHarmony, "Pete context.");
        assert!(prompt.starts_with("<|start|>system<|message|>"));
        assert!(prompt.contains("# Valid channels: analysis, final."));
        assert!(prompt.contains("<|start|>developer<|message|># Instructions"));
        assert!(prompt.contains("Final channel content must be exactly one or more <ts>"));
        assert!(prompt.contains("<ts>say(\"Hello, I can hear you.\")</ts>"));
        assert!(prompt.contains("already available in scope"));
        assert!(prompt.contains("leave room for the interlocutor"));
        assert!(prompt.contains("Do not use say for clock ticks"));
        assert!(prompt.contains("interrupt: true"));
        assert!(prompt.contains("<|start|>user<|message|>Pete context.<|end|>"));
        assert!(prompt.ends_with("<|start|>assistant"));
        assert!(stops.iter().any(|stop| stop == "<|return|>"));
        assert!(!stops.iter().any(|stop| stop == "<|end|>"));
    }

    #[test]
    fn harmony_filter_only_emits_final_channel() {
        let mut filter = HarmonyFinalFilter::default();
        let events = filter.filter_events(&[
            LlmEvent::Token {
                text: "<|channel|>analysis<|message|>Think privately.".to_string(),
            },
            LlmEvent::Token {
                text: "<|end|><|start|>assistant<|channel|>final<|message|><ts>import { say } from \"pete:will\"; say(\"Hi\")</ts>".to_string(),
            },
            LlmEvent::Completed,
        ]);

        assert!(matches!(
            events.as_slice(),
            [LlmEvent::Token { text }, LlmEvent::Completed]
                if text == "<ts>import { say } from \"pete:will\"; say(\"Hi\")</ts>"
        ));
    }

    #[test]
    fn spoken_content_rejects_chat_template_tokens() {
        assert_eq!(
            clean_spoken_content("<|end|><|start|>assistant<|channel|>sp<|message|>Hey!"),
            None
        );
        assert_eq!(
            clean_spoken_content("assistant<|channel|>analysis<|message|>We respond."),
            None
        );
        assert_eq!(
            clean_spoken_content("I can hear you."),
            Some("I can hear you.".to_string())
        );
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
    fn current_time_message_includes_local_iso_time_with_offset() {
        let message = current_time_message();

        assert!(message.starts_with("The current local time is "));
        assert!(message.contains(". Unix time is "));
        let local_time = message
            .strip_prefix("The current local time is ")
            .and_then(|text| text.split(". Unix time is ").next())
            .expect("clock message should contain local time");
        assert!(local_time.contains('T'));
        assert!(
            local_time.len() >= 6,
            "local time should include a timezone offset"
        );
        let offset = &local_time[local_time.len() - 6..];
        assert!(
            matches!(offset.as_bytes()[0], b'+' | b'-')
                && offset.as_bytes()[3] == b':'
                && offset[1..3].chars().all(|ch| ch.is_ascii_digit())
                && offset[4..6].chars().all(|ch| ch.is_ascii_digit()),
            "local time should end with an ISO timezone offset, got {local_time}"
        );
    }

    #[test]
    fn time_event_interval_is_jittered_around_one_second() {
        let min_ms = TIME_EVENT_INTERVAL_BASE_MS - TIME_EVENT_INTERVAL_JITTER_MS;
        let max_ms = TIME_EVENT_INTERVAL_BASE_MS + TIME_EVENT_INTERVAL_JITTER_MS;
        let mut state = 0x1234_5678_9abc_def0;
        let intervals = (0..20)
            .map(|_| next_time_event_interval(&mut state).as_millis() as u64)
            .collect::<Vec<_>>();

        assert!(
            intervals
                .iter()
                .all(|millis| (min_ms..=max_ms).contains(millis)),
            "intervals should stay near one second: {intervals:?}"
        );
        assert!(
            intervals.windows(2).any(|pair| pair[0] != pair[1]),
            "intervals should vary: {intervals:?}"
        );
    }

    #[test]
    fn speech_detector_parses_typescript_tag() {
        let mut detector = SpeechEventDetector::default();

        assert_eq!(
            detector.ingest(
                "think <ts>import { listFiles } from \"pete:will\";\nlistFiles()</ts> then continue"
            ),
            vec![ContinueRuntimeEvent::SourceCommand {
                command: SourceCommand::RunTypeScript {
                    source: "import { listFiles } from \"pete:will\";\nlistFiles()".to_string()
                }
            }]
        );
    }

    #[test]
    fn speech_detector_defers_live_events_during_partial_typescript_tag() {
        let mut detector = SpeechEventDetector::default();

        assert!(
            detector
                .ingest("<ts>import { listFiles } from \"pete:will\";\nlist")
                .is_empty()
        );
        assert!(detector.defers_live_events());
        assert_eq!(
            detector.ingest("Files()</ts>"),
            vec![ContinueRuntimeEvent::SourceCommand {
                command: SourceCommand::RunTypeScript {
                    source: "import { listFiles } from \"pete:will\";\nlistFiles()".to_string()
                }
            }]
        );
        assert!(!detector.defers_live_events());
    }

    #[test]
    fn typescript_executes_pete_will_commands() {
        let commands = execute_typescript_commands(
            r#"import { listFiles, readFile, grepSource, say, pause, resume, shutup } from "pete:will";
[listFiles(), readFile("src/main.rs", 1 + 1), grepSource("ContinueCommand", 2), say("check complete"), pause(), resume(), shutup()]"#,
        )
        .expect("typescript should execute");

        assert_eq!(
            commands,
            vec![
                TypeScriptCommand::ListFiles,
                TypeScriptCommand::ReadSourceFile {
                    file: "src/main.rs".to_string(),
                    page: 2
                },
                TypeScriptCommand::GrepSource {
                    pattern: "ContinueCommand".to_string(),
                    limit: 2
                },
                TypeScriptCommand::Say {
                    text: "check complete".to_string(),
                    interrupt: false
                },
                TypeScriptCommand::Pause,
                TypeScriptCommand::Resume,
                TypeScriptCommand::Shutup
            ]
        );
    }

    #[test]
    fn typescript_builders_are_available_without_explicit_import() {
        let commands = execute_typescript_commands(
            r#"[listFiles(), readSourceFile("src/main.rs"), say("I can hear you.")]"#,
        )
        .expect("default pete:will imports should be injected");

        assert_eq!(
            commands,
            vec![
                TypeScriptCommand::ListFiles,
                TypeScriptCommand::ReadSourceFile {
                    file: "src/main.rs".to_string(),
                    page: 1
                },
                TypeScriptCommand::Say {
                    text: "I can hear you.".to_string(),
                    interrupt: false
                }
            ]
        );
    }

    #[test]
    fn typescript_source_reports_command_results() {
        let output = execute_typescript_source(
            r#"import { grepSource } from "pete:will";
grepSource("build_initial_prompt", 1)"#,
        );

        assert!(output.message.contains("TypeScript executed."));
        assert!(output.message.contains("[grep_source]"));
        assert!(output.message.contains("build_initial_prompt"));
        assert!(output.runtime_events.is_empty());
    }

    #[test]
    fn typescript_say_and_controls_emit_runtime_events() {
        let output = execute_typescript_source(
            r#"import { say, pause, resume, shutup } from "pete:will";
[say("I can hear you."), pause(), resume(), shutup()]"#,
        );

        assert!(output.message.contains("[say]"));
        assert!(output.message.contains("[pause]"));
        assert_eq!(output.runtime_events.len(), 4);
        assert!(matches!(
            &output.runtime_events[0],
            ContinueRuntimeEvent::UtteranceCompleted { content, interrupt, .. } if content == "I can hear you." && !interrupt
        ));
        assert_eq!(
            output.runtime_events[1],
            ContinueRuntimeEvent::SpeechControl {
                command: SpeechControlCommand::Pause
            }
        );
        assert_eq!(
            output.runtime_events[2],
            ContinueRuntimeEvent::SpeechControl {
                command: SpeechControlCommand::Resume
            }
        );
        assert_eq!(
            output.runtime_events[3],
            ContinueRuntimeEvent::SpeechControl {
                command: SpeechControlCommand::Shutup
            }
        );
    }

    #[test]
    fn typescript_say_accepts_interrupt_option() {
        let commands = execute_typescript_commands(r#"say("Hold on.", { interrupt: true })"#)
            .expect("say interrupt option should execute");

        assert_eq!(
            commands,
            vec![TypeScriptCommand::Say {
                text: "Hold on.".to_string(),
                interrupt: true
            }]
        );

        let output = execute_typescript_source(r#"say("Hold on.", true)"#);
        assert!(matches!(
            &output.runtime_events[0],
            ContinueRuntimeEvent::UtteranceCompleted { content, interrupt, .. }
                if content == "Hold on." && *interrupt
        ));
    }

    #[test]
    fn say_runtime_event_maps_to_mouth_speak_command() {
        assert_eq!(
            mouth_command_for_runtime_event(&ContinueRuntimeEvent::UtteranceCompleted {
                id: 42,
                content: "I can hear you.".to_string(),
                interrupt: false
            }),
            Some((
                ContinueMouthCommand::Speak {
                    id: 42,
                    text: "I can hear you.".to_string(),
                    interrupt: false
                },
                true
            ))
        );

        assert_eq!(
            mouth_command_for_runtime_event(&ContinueRuntimeEvent::SpeechControl {
                command: SpeechControlCommand::Pause
            }),
            Some((ContinueMouthCommand::Pause, false))
        );
    }

    #[test]
    fn source_bundle_lists_and_views_files() {
        let files = execute_list_source_files();
        assert!(files.contains("src/cli/commands/continue_generation/mod.rs"));

        let page = execute_view_source_file("src/cli/commands/continue_generation/mod.rs", 1);
        assert!(page.contains("--- src/cli/commands/continue_generation/mod.rs"));
        assert!(page.contains("use crate::cli::ContinueCommand;"));
    }

    #[cfg(all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    ))]
    #[test]
    fn duplex_controller_ignores_low_confidence_overlap() {
        let mut controller = super::DuplexTurnController::new(super::DuplexTurnControllerConfig {
            pause_after: std::time::Duration::from_millis(150),
            listen_for: std::time::Duration::from_millis(300),
        });
        let started_at = std::time::Instant::now();

        controller.observe_ear_event(
            &super::ContinueEarEvent::OverlapDetected {
                self_confidence: 0.95,
                external_confidence: 0.2,
                duration_ms: 20,
            },
            started_at,
        );
        assert_eq!(
            controller.next_action(started_at + std::time::Duration::from_millis(500), 1),
            None
        );
    }

    #[cfg(all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    ))]
    #[test]
    fn duplex_controller_ignores_brief_high_confidence_overlap() {
        let mut controller = super::DuplexTurnController::new(super::DuplexTurnControllerConfig {
            pause_after: std::time::Duration::from_millis(150),
            listen_for: std::time::Duration::from_millis(300),
        });
        let started_at = std::time::Instant::now();

        controller.observe_ear_event(
            &super::ContinueEarEvent::OverlapDetected {
                self_confidence: 0.3,
                external_confidence: 0.9,
                duration_ms: 30,
            },
            started_at,
        );
        assert_eq!(
            controller.next_action(started_at + std::time::Duration::from_millis(149), 1),
            None
        );
    }

    #[cfg(all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    ))]
    #[test]
    fn duplex_controller_pauses_for_sustained_overlap() {
        let mut controller = super::DuplexTurnController::new(super::DuplexTurnControllerConfig {
            pause_after: std::time::Duration::from_millis(150),
            listen_for: std::time::Duration::from_millis(300),
        });
        let started_at = std::time::Instant::now();

        controller.observe_ear_event(
            &super::ContinueEarEvent::OverlapDetected {
                self_confidence: 0.4,
                external_confidence: 0.8,
                duration_ms: 20,
            },
            started_at,
        );
        assert_eq!(controller.next_action(started_at, 1), None);
        assert_eq!(
            controller.next_action(started_at + std::time::Duration::from_millis(151), 1),
            Some(super::DuplexTurnAction::Pause)
        );
    }

    #[cfg(all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    ))]
    #[test]
    fn duplex_controller_clears_queue_after_sustained_overlap() {
        let mut controller = super::DuplexTurnController::new(super::DuplexTurnControllerConfig {
            pause_after: std::time::Duration::from_millis(150),
            listen_for: std::time::Duration::from_millis(300),
        });
        let started_at = std::time::Instant::now();

        controller.observe_ear_event(
            &super::ContinueEarEvent::OverlapDetected {
                self_confidence: 0.4,
                external_confidence: 0.8,
                duration_ms: 20,
            },
            started_at,
        );
        assert_eq!(
            controller.next_action(started_at + std::time::Duration::from_millis(151), 2),
            Some(super::DuplexTurnAction::Pause)
        );
        assert_eq!(
            controller.next_action(started_at + std::time::Duration::from_millis(452), 2),
            Some(super::DuplexTurnAction::Clear)
        );
    }

    #[cfg(all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    ))]
    #[test]
    fn duplex_controller_ignores_silence_and_noise() {
        let mut controller = super::DuplexTurnController::new(super::DuplexTurnControllerConfig {
            pause_after: std::time::Duration::from_millis(150),
            listen_for: std::time::Duration::from_millis(300),
        });
        let now = std::time::Instant::now();

        assert_eq!(
            controller.next_action(now + std::time::Duration::from_secs(10), 1),
            None
        );
    }

    #[cfg(all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    ))]
    #[test]
    fn duplex_controller_resumes_after_silence() {
        let mut controller = super::DuplexTurnController::new(super::DuplexTurnControllerConfig {
            pause_after: std::time::Duration::from_millis(150),
            listen_for: std::time::Duration::from_millis(300),
        });
        let started_at = std::time::Instant::now();

        controller.observe_ear_event(
            &super::ContinueEarEvent::OverlapDetected {
                self_confidence: 0.4,
                external_confidence: 0.8,
                duration_ms: 20,
            },
            started_at,
        );
        assert_eq!(
            controller.next_action(started_at + std::time::Duration::from_millis(151), 1),
            Some(super::DuplexTurnAction::Pause)
        );
        controller.observe_ear_event(
            &super::ContinueEarEvent::SpeechStopped,
            started_at + std::time::Duration::from_millis(200),
        );
        assert_eq!(
            controller.next_action(started_at + std::time::Duration::from_millis(499), 1),
            None
        );
        assert_eq!(
            controller.next_action(started_at + std::time::Duration::from_millis(500), 1),
            Some(super::DuplexTurnAction::Resume)
        );
    }

    #[cfg(all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    ))]
    #[test]
    fn duplex_controller_holds_prepared_tts_during_turn_gap() {
        let mut controller = super::DuplexTurnController::new(super::DuplexTurnControllerConfig {
            pause_after: std::time::Duration::from_millis(150),
            listen_for: std::time::Duration::from_millis(300),
        });
        let stopped_at = std::time::Instant::now();

        controller.observe_ear_event(&super::ContinueEarEvent::SpeechStopped, stopped_at);
        assert_eq!(
            controller.prepare_runtime_action(&super::ContinueRuntimeEvent::UtteranceCompleted {
                id: 7,
                content: "Okay.".to_string(),
                interrupt: false,
            }),
            Some(super::DuplexTurnAction::Pause)
        );
        controller.paused_for_external_speech = true;

        assert_eq!(
            controller.next_action(stopped_at + std::time::Duration::from_millis(299), 1),
            None
        );
        assert_eq!(
            controller.next_action(stopped_at + std::time::Duration::from_millis(300), 1),
            Some(super::DuplexTurnAction::Resume)
        );
    }

    #[cfg(all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    ))]
    #[test]
    fn duplex_controller_cancels_prepared_tts_when_speech_resumes_during_turn_gap() {
        let mut controller = super::DuplexTurnController::new(super::DuplexTurnControllerConfig {
            pause_after: std::time::Duration::from_millis(150),
            listen_for: std::time::Duration::from_millis(300),
        });
        let stopped_at = std::time::Instant::now();

        controller.observe_ear_event(&super::ContinueEarEvent::SpeechStopped, stopped_at);
        assert_eq!(
            controller.prepare_runtime_action(&super::ContinueRuntimeEvent::UtteranceCompleted {
                id: 7,
                content: "Okay.".to_string(),
                interrupt: false,
            }),
            Some(super::DuplexTurnAction::Pause)
        );
        controller.paused_for_external_speech = true;
        controller.observe_ear_event(
            &super::ContinueEarEvent::SpeechStarted,
            stopped_at + std::time::Duration::from_millis(120),
        );

        assert_eq!(
            controller.next_action(stopped_at + std::time::Duration::from_millis(121), 1),
            Some(super::DuplexTurnAction::Clear)
        );
    }

    #[cfg(all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    ))]
    #[test]
    fn duplex_controller_allows_interrupting_utterances_to_resume() {
        let mut controller = super::DuplexTurnController::new(super::DuplexTurnControllerConfig {
            pause_after: std::time::Duration::from_millis(150),
            listen_for: std::time::Duration::from_millis(300),
        });
        let started_at = std::time::Instant::now();

        controller.observe_ear_event(
            &super::ContinueEarEvent::Transcript {
                text: "wait".to_string(),
                timed_word_stream: TimedWordStream::new(WordStreamId(1), WordStreamSource::LiveAsr),
                occurred_at: ExactTimestamp { unix_nanos: 0 },
            },
            started_at,
        );
        assert_eq!(
            controller.prepare_runtime_action(&super::ContinueRuntimeEvent::UtteranceCompleted {
                id: 7,
                content: "Hold on.".to_string(),
                interrupt: false,
            }),
            Some(super::DuplexTurnAction::Pause)
        );
        controller.paused_for_external_speech = true;
        assert_eq!(
            controller.prepare_runtime_action(&super::ContinueRuntimeEvent::UtteranceCompleted {
                id: 8,
                content: "Excuse me.".to_string(),
                interrupt: true,
            }),
            Some(super::DuplexTurnAction::Resume)
        );
    }

    #[cfg(all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    ))]
    #[test]
    fn duplex_trace_scenario_records_short_overlap_without_yielding() {
        let events = super::build_duplex_trace_scenario_events(
            crate::cli::DuplexTraceScenarioOption::OverlapYield,
            std::time::Duration::from_millis(150),
            std::time::Duration::from_millis(300),
        );

        let short = events.iter().find(|event| {
            event["kind"] == "controller_decision"
                && event["details"]["case"] == "short_overlap_blip"
        });
        assert!(short.is_some());
        let short = short.unwrap();
        assert_eq!(short["details"]["decision"], "continue");
        assert!(short["details"]["raw_action"].is_null());
    }

    #[cfg(all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    ))]
    #[test]
    fn duplex_trace_scenario_records_sustained_overlap_as_yield_and_clear() {
        let events = super::build_duplex_trace_scenario_events(
            crate::cli::DuplexTraceScenarioOption::OverlapYield,
            std::time::Duration::from_millis(150),
            std::time::Duration::from_millis(300),
        );

        let sustained_pause = events.iter().find(|event| {
            event["kind"] == "controller_decision"
                && event["details"]["case"] == "sustained_overlap"
                && event["details"]["decision"] == "yield_pause"
        });
        assert!(sustained_pause.is_some());
        assert_eq!(
            sustained_pause.unwrap()["details"]["raw_action"],
            serde_json::Value::String("Pause".to_string())
        );

        let sustained_clear = events.iter().find(|event| {
            event["kind"] == "controller_decision"
                && event["details"]["case"] == "sustained_overlap"
                && event["details"]["decision"] == "yield_clear_queue"
        });
        assert!(sustained_clear.is_some());
        assert_eq!(
            sustained_clear.unwrap()["details"]["raw_action"],
            serde_json::Value::String("Clear".to_string())
        );
    }
}
