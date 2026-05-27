use super::{
    ContinueCommand, DuplexTraceScenarioOption, LiveHalfDuplexCommand, PromptMode,
    VadBackendOption, commands,
};
use anyhow::{Context, Result};
use std::path::PathBuf;

const DEFAULT_CONTEXT_SIZE: u32 = 8192;
const DEFAULT_VERBATIM_TURNS: usize = 8;
const DEFAULT_TTS_VAD_PAUSE_MS: u64 = 250;
const DEFAULT_TTS_VAD_LISTEN_MS: u64 = 700;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LiveSessionMode {
    HalfDuplex,
    Duplex,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct LiveSessionConfig {
    pub(crate) mode: LiveSessionMode,
    pub(crate) input: LiveInputConfig,
    pub(crate) hearing: HearingConfig,
    pub(crate) asr: AsrConfig,
    pub(crate) llm: LlmConfig,
    pub(crate) mouth_playback: MouthPlaybackConfig,
    pub(crate) tracing: TraceConfig,
    pub(crate) memory: MemoryConfig,
    pub(crate) web_bridge: WebBridgeConfig,
    runtime: Option<LiveRuntimeCompatibilityShim>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LiveInputConfig {
    pub(crate) seconds: Option<u64>,
    pub(crate) vad: VadBackendOption,
    pub(crate) vad_profile: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HearingConfig {
    pub(crate) vad: VadBackendOption,
    pub(crate) vad_profile: Option<PathBuf>,
    pub(crate) tts_vad_pause_ms: u64,
    pub(crate) tts_vad_listen_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AsrConfig {
    pub(crate) whisper_model: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LlmConfig {
    pub(crate) model: Option<PathBuf>,
    pub(crate) gpu_layers: Option<u32>,
    pub(crate) mode: PromptMode,
    pub(crate) max_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MouthPlaybackConfig {
    pub(crate) piper_bin: Option<PathBuf>,
    pub(crate) piper_voice: Option<PathBuf>,
    pub(crate) hifigan: bool,
    pub(crate) hifigan_model: Option<PathBuf>,
    pub(crate) skip_gan: bool,
    pub(crate) no_backchannels: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TraceConfig {
    pub(crate) jsonl: Option<PathBuf>,
    pub(crate) duplex_trace_scenario: Option<DuplexTraceScenarioOption>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MemoryConfig {
    pub(crate) context_size: u32,
    pub(crate) verbatim_turns: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WebBridgeConfig {
    pub(crate) enabled: bool,
    pub(crate) host: String,
    pub(crate) port: u16,
}

#[derive(Debug)]
enum LiveRuntimeCompatibilityShim {
    HalfDuplex(LiveHalfDuplexCommand),
    Duplex(ContinueCommand),
}

impl LiveSessionConfig {
    pub(crate) fn from_listen_command(command: LiveHalfDuplexCommand) -> Self {
        if command.duplex {
            return Self::from_continue_command(continue_command_from_listen_command(command));
        }

        Self {
            mode: LiveSessionMode::HalfDuplex,
            input: LiveInputConfig {
                seconds: command.seconds,
                vad: command.vad,
                vad_profile: command.vad_profile.clone(),
            },
            hearing: HearingConfig {
                vad: command.vad,
                vad_profile: command.vad_profile.clone(),
                tts_vad_pause_ms: DEFAULT_TTS_VAD_PAUSE_MS,
                tts_vad_listen_ms: DEFAULT_TTS_VAD_LISTEN_MS,
            },
            asr: AsrConfig {
                whisper_model: command.whisper_model.clone(),
            },
            llm: LlmConfig {
                model: command.llm_model.clone(),
                gpu_layers: command.llm_gpu_layers,
                mode: PromptMode::Spoken,
                max_tokens: None,
            },
            mouth_playback: MouthPlaybackConfig {
                piper_bin: command.piper_bin.clone(),
                piper_voice: command.piper_voice.clone(),
                hifigan: command.hifigan,
                hifigan_model: command.hifigan_model.clone(),
                skip_gan: command.skip_gan,
                no_backchannels: command.no_backchannels,
            },
            tracing: TraceConfig {
                jsonl: command.jsonl.clone(),
                duplex_trace_scenario: None,
            },
            memory: MemoryConfig {
                context_size: DEFAULT_CONTEXT_SIZE,
                verbatim_turns: DEFAULT_VERBATIM_TURNS,
            },
            web_bridge: WebBridgeConfig {
                enabled: command.web,
                host: command.web_host.clone(),
                port: command.web_port,
            },
            runtime: Some(LiveRuntimeCompatibilityShim::HalfDuplex(command)),
        }
    }

    pub(crate) fn from_continue_command(command: ContinueCommand) -> Self {
        Self {
            mode: LiveSessionMode::Duplex,
            input: LiveInputConfig {
                seconds: None,
                vad: command.vad,
                vad_profile: command.vad_profile.clone(),
            },
            hearing: HearingConfig {
                vad: command.vad,
                vad_profile: command.vad_profile.clone(),
                tts_vad_pause_ms: command.tts_vad_pause_ms,
                tts_vad_listen_ms: command.tts_vad_listen_ms,
            },
            asr: AsrConfig {
                whisper_model: command.whisper_model.clone(),
            },
            llm: LlmConfig {
                model: command.llm_model.clone(),
                gpu_layers: command.llm_gpu_layers,
                mode: command.mode,
                max_tokens: command.max_tokens,
            },
            mouth_playback: MouthPlaybackConfig {
                piper_bin: command.piper_bin.clone(),
                piper_voice: command.piper_voice.clone(),
                hifigan: command.hifigan,
                hifigan_model: command.hifigan_model.clone(),
                skip_gan: command.skip_gan,
                no_backchannels: false,
            },
            tracing: TraceConfig {
                jsonl: command.jsonl.clone(),
                duplex_trace_scenario: command.duplex_trace_scenario,
            },
            memory: MemoryConfig {
                context_size: command.context_size,
                verbatim_turns: command.verbatim_turns,
            },
            web_bridge: WebBridgeConfig {
                enabled: command.web,
                host: command.web_host.clone(),
                port: command.web_port,
            },
            runtime: Some(LiveRuntimeCompatibilityShim::Duplex(command)),
        }
    }
}

/// First-class live runtime session entrypoint.
///
/// This currently executes through command-level runtime functions as a temporary compatibility
/// shim while orchestration continues moving out of CLI command modules.
pub(crate) struct LiveSession {
    runtime: Option<LiveRuntimeCompatibilityShim>,
}

impl LiveSession {
    pub(crate) fn new(config: LiveSessionConfig) -> Result<Self> {
        Ok(Self {
            runtime: config.runtime,
        })
    }

    pub(crate) fn run(&mut self) -> Result<()> {
        match self
            .runtime
            .take()
            .context("live session has already run")?
        {
            LiveRuntimeCompatibilityShim::HalfDuplex(command) => {
                commands::run_live_half_duplex(command)
            }
            LiveRuntimeCompatibilityShim::Duplex(command) => commands::run_continue(command),
        }
    }

    pub(crate) fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

pub(crate) fn continue_command_from_listen_command(
    command: LiveHalfDuplexCommand,
) -> ContinueCommand {
    ContinueCommand {
        llm_model: command.llm_model,
        llm_gpu_layers: command.llm_gpu_layers,
        piper_bin: command.piper_bin,
        piper_voice: command.piper_voice,
        hifigan: command.hifigan,
        hifigan_model: command.hifigan_model,
        skip_gan: command.skip_gan,
        whisper_model: command.whisper_model,
        vad: command.vad,
        vad_profile: command.vad_profile,
        mode: PromptMode::Raw,
        max_tokens: None,
        context_size: command.context_size,
        verbatim_turns: DEFAULT_VERBATIM_TURNS,
        tts_vad_pause_ms: DEFAULT_TTS_VAD_PAUSE_MS,
        tts_vad_listen_ms: DEFAULT_TTS_VAD_LISTEN_MS,
        web: command.web,
        web_host: command.web_host,
        web_port: command.web_port,
        duplex_trace_scenario: None,
        jsonl: command.jsonl,
        prompt: Vec::new(),
    }
}
