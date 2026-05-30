use anyhow::Result;
use clap::{Args, Subcommand, ValueEnum};
use listenbury::{VadBackendKind, VadProfile};
use std::path::{Path, PathBuf};

#[derive(Debug, Args)]
pub(crate) struct TextCommand {
    #[arg(required = true, num_args = 1.., trailing_var_arg = true)]
    pub(crate) text: Vec<String>,
}

#[derive(Debug, Args)]
pub(crate) struct RecordWavCommand {
    pub(crate) output_wav: PathBuf,
    #[arg(long, default_value_t = 5)]
    pub(crate) seconds: u64,
}

#[derive(Debug, Args)]
pub(crate) struct PlayWavCommand {
    pub(crate) input_wav: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct LoopTraceCommand {
    /// Pipeline profile to trace.
    #[arg(long, value_enum, default_value_t = LoopTraceProfile::Mock)]
    pub(crate) profile: LoopTraceProfile,
    /// Synthetic capture window represented in the trace.
    #[arg(long, default_value_t = 20)]
    pub(crate) duration: u64,
    /// Use the real microphone path. Implies --profile ear.
    #[arg(long)]
    pub(crate) real_mic: bool,
    /// Use the real ASR path. Implies --profile ear.
    #[arg(long)]
    pub(crate) real_asr: bool,
    /// Explicitly request the synthetic microphone path.
    #[arg(long)]
    pub(crate) mock_mic: bool,
    /// Explicitly request the synthetic LLM path.
    #[arg(long)]
    pub(crate) mock_llm: bool,
    /// Explicitly request the synthetic mouth/playback path.
    #[arg(long)]
    pub(crate) mock_mouth: bool,
    /// Print the latency summary as JSON.
    #[arg(long, conflicts_with = "pretty")]
    pub(crate) json: bool,
    /// Print the human-readable latency summary.
    #[arg(long)]
    pub(crate) pretty: bool,
    /// Disable synthetic self-hearing capture/transcription events.
    #[arg(long)]
    pub(crate) no_self_hearing: bool,
    /// Destination JSONL trace file.
    #[arg(long, default_value = "out/loop-trace.jsonl")]
    pub(crate) write: PathBuf,
    #[arg(long, alias = "model-path")]
    pub(crate) whisper_model: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = VadBackendOption::WebRtc)]
    pub(crate) vad: VadBackendOption,
    /// TOML profile emitted by `listenbury vad calibrate-room --toml`.
    #[arg(long)]
    pub(crate) vad_profile: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct EarScopeCommand {
    /// Live microphone capture duration.
    #[arg(long, default_value_t = 10)]
    pub(crate) duration: u64,
    /// Destination WAV file for the normalized 16 kHz mono capture.
    #[arg(long, default_value = "out/ear-scope.wav")]
    pub(crate) output_wav: PathBuf,
    /// Destination PNG timeline diagnostic.
    #[arg(long, default_value = "out/ear-scope.png")]
    pub(crate) output_png: PathBuf,
    /// Destination per-frame JSONL diagnostics.
    #[arg(long, default_value = "out/ear-scope.jsonl")]
    pub(crate) output_jsonl: PathBuf,
    #[arg(long, value_enum, default_value_t = VadBackendOption::WebRtc)]
    pub(crate) vad: VadBackendOption,
    /// TOML profile emitted by `listenbury vad calibrate-room --toml`.
    #[arg(long)]
    pub(crate) vad_profile: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct ThinkSayCommand {
    /// Use a deterministic local token stream instead of the configured LLM backend.
    #[arg(long)]
    pub(crate) mock_llm: bool,
    /// Record mouth queueing without synthesizing or playing audio.
    #[arg(long)]
    pub(crate) mock_mouth: bool,
    /// Mouth backend to use when --mock-mouth is not set.
    #[arg(long, value_enum, default_value_t = ThinkSayMouthOption::Current)]
    pub(crate) mouth: ThinkSayMouthOption,
    #[arg(long, alias = "model-path")]
    pub(crate) llm_model: Option<PathBuf>,
    /// Number of llama.cpp layers to offload to the GPU. Use 0 for CPU-only LLM inference.
    #[arg(long)]
    pub(crate) llm_gpu_layers: Option<u32>,
    /// Maximum tokens the LLM may generate.
    #[arg(long, default_value_t = 80)]
    pub(crate) max_tokens: u32,
    /// Flush only sentence/newline boundaries unless the LLM completes.
    #[arg(long, conflicts_with = "breath_group")]
    pub(crate) sentence_by_sentence: bool,
    /// Allow breath-group-sized fallback flushes when punctuation does not arrive.
    #[arg(long, conflicts_with = "sentence_by_sentence")]
    pub(crate) breath_group: bool,
    /// Optional JSONL trace destination.
    #[arg(long)]
    pub(crate) dump: Option<PathBuf>,
    #[arg(required = true, num_args = 1.., trailing_var_arg = true)]
    pub(crate) prompt: Vec<String>,
}

#[derive(Debug, Args)]
pub(crate) struct GoCommand {
    #[arg(long, alias = "model-path")]
    pub(crate) llm_model: Option<PathBuf>,
    /// Number of llama.cpp layers to offload to the GPU. Use 0 for CPU-only LLM inference.
    #[arg(long)]
    pub(crate) llm_gpu_layers: Option<u32>,
    #[arg(long)]
    pub(crate) piper_bin: Option<PathBuf>,
    #[arg(long)]
    pub(crate) piper_voice: Option<PathBuf>,
    /// Use the source-filter acoustic model plus HiFi-GAN vocoder for Pete's voice.
    #[arg(long)]
    pub(crate) hifigan: bool,
    #[arg(long = "hifigan-model", requires = "hifigan")]
    pub(crate) hifigan_model: Option<PathBuf>,
    /// Debug the mel path with the non-neural mel debug renderer instead of HiFi-GAN.
    #[arg(long, alias = "hifigan-fallback", requires = "hifigan")]
    pub(crate) skip_gan: bool,
    /// Optional generated-token cap per draft generation. By default, each generation continues until Ctrl-C or context fills.
    #[arg(long)]
    pub(crate) max_tokens: Option<u32>,
    /// llama.cpp context size for the live stream.
    #[arg(long, default_value_t = 8192)]
    pub(crate) context_size: u32,
    /// Tokens reserved when rebuilding the compact stream prompt.
    #[arg(long, default_value_t = 512)]
    pub(crate) reserved_generation_tokens: u32,
    /// Number of recent observations retained verbatim after compaction.
    #[arg(long, default_value_t = 16)]
    pub(crate) memory_events: usize,
    /// Maximum generated token fragments to let Pete get ahead of mouth/ear pacing.
    #[arg(long, default_value_t = 8)]
    pub(crate) lookahead_tokens: usize,
    /// Flush a partial speakable unit after this many characters even without punctuation.
    #[arg(long, default_value_t = 80)]
    pub(crate) lookahead_chars: usize,
    /// Require the mouth playback completion signal before allowing more visible generation.
    #[arg(long, default_value_t = true)]
    pub(crate) require_self_hearing: bool,
    /// Record the stream without synthesizing or playing audio.
    #[arg(long)]
    pub(crate) mock_mouth: bool,
    #[arg(long)]
    pub(crate) whisper_model: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = VadBackendOption::WebRtc)]
    pub(crate) vad: VadBackendOption,
    /// TOML profile emitted by `listenbury vad calibrate-room --toml`.
    #[arg(long)]
    pub(crate) vad_profile: Option<PathBuf>,
    /// Initial stream seed. If omitted, Pete wakes into an open live session.
    #[arg(num_args = 0.., trailing_var_arg = true)]
    pub(crate) prompt: Vec<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum ThinkSayMouthOption {
    #[default]
    Current,
    Piper,
    Klatt,
    Diphone,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum LoopTraceProfile {
    #[default]
    Mock,
    Ear,
}

impl LoopTraceCommand {
    pub(crate) fn effective_profile(&self) -> LoopTraceProfile {
        if self.real_mic || self.real_asr {
            LoopTraceProfile::Ear
        } else {
            self.profile
        }
    }
}

#[derive(Debug, Args)]
pub(crate) struct VadTraceCommand {
    pub(crate) input_wav: PathBuf,
    #[arg(long)]
    pub(crate) jsonl: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = VadBackendOption::WebRtc)]
    pub(crate) vad: VadBackendOption,
    /// TOML profile emitted by `listenbury vad calibrate-room --toml`.
    #[arg(long)]
    pub(crate) vad_profile: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct BreathTranscribeCommand {
    pub(crate) input_wav: PathBuf,
    #[arg(long)]
    pub(crate) whisper_model: Option<PathBuf>,
    #[arg(long, default_value_t = 100)]
    pub(crate) pre_roll_ms: u64,
    #[arg(long, default_value_t = 100)]
    pub(crate) trailing_pad_ms: u64,
    #[arg(long, default_value_t = 150)]
    pub(crate) min_group_ms: u64,
    #[arg(long, default_value_t = 0)]
    pub(crate) max_group_ms: u64,
}

#[derive(Debug, Args)]
pub(crate) struct MicTranscribeCommand {
    #[arg(long, default_value_t = 30)]
    pub(crate) seconds: u64,
    #[arg(long)]
    pub(crate) until_ctrl_c: bool,
    #[arg(long, alias = "model-path")]
    pub(crate) whisper_model: Option<PathBuf>,
    /// Slower Whisper model used for rolling refinement in --web mode.
    #[arg(long)]
    pub(crate) refine_whisper_model: Option<PathBuf>,
    /// Rolling audio window refined by the slower model in --web mode.
    #[arg(long, default_value_t = 90)]
    pub(crate) refine_window_seconds: u64,
    /// Minimum delay between queued rolling refinement passes in --web mode.
    #[arg(long, default_value_t = 1_500)]
    pub(crate) refine_interval_ms: u64,
    #[arg(long, value_enum, default_value_t = VadBackendOption::WebRtc)]
    pub(crate) vad: VadBackendOption,
    /// TOML profile emitted by `listenbury vad calibrate-room --toml`.
    #[arg(long)]
    pub(crate) vad_profile: Option<PathBuf>,
    /// Start the screenplay and WaveDeck browser viewers; microphone capture runs until Ctrl-C.
    #[arg(long)]
    pub(crate) web: bool,
    /// Host for the embedded web viewer (requires --web).
    #[arg(long, default_value = "127.0.0.1")]
    pub(crate) web_host: String,
    /// Port for the embedded web viewer (requires --web).
    #[arg(long, default_value_t = 8787)]
    pub(crate) web_port: u16,
}

#[derive(Debug, Args)]
pub(crate) struct LlamaTurnCommand {
    #[arg(long, alias = "model-path")]
    pub(crate) llm_model: Option<PathBuf>,
    /// Number of llama.cpp layers to offload to the GPU. Use 0 for CPU-only LLM inference.
    #[arg(long)]
    pub(crate) llm_gpu_layers: Option<u32>,
    /// Prompt framing to apply before generation.
    #[arg(long, value_enum, default_value_t = PromptMode::Spoken)]
    pub(crate) mode: PromptMode,
    /// Maximum tokens the LLM may generate.
    #[arg(long, default_value_t = 48)]
    pub(crate) max_tokens: u32,
    #[arg(required = true, num_args = 1.., trailing_var_arg = true)]
    pub(crate) prompt: Vec<String>,
}

#[derive(Debug, Args)]
pub(crate) struct DraftPeteLineCommand {
    #[arg(long, alias = "model-path")]
    pub(crate) llm_model: Option<PathBuf>,
    /// Number of llama.cpp layers to offload to the GPU. Use 0 for CPU-only LLM inference.
    #[arg(long)]
    pub(crate) llm_gpu_layers: Option<u32>,
    #[arg(long)]
    pub(crate) piper_bin: Option<PathBuf>,
    #[arg(long)]
    pub(crate) piper_voice: Option<PathBuf>,
    #[arg(long)]
    pub(crate) whisper_model: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = VadBackendOption::WebRtc)]
    pub(crate) vad: VadBackendOption,
    /// TOML profile emitted by `listenbury vad calibrate-room --toml`.
    #[arg(long)]
    pub(crate) vad_profile: Option<PathBuf>,
    /// Optional generated-token cap. By default, continue until Ctrl-C or context fills.
    #[arg(long)]
    pub(crate) max_tokens: Option<u32>,
    /// llama.cpp context size for this raw completion.
    #[arg(long, default_value_t = 8192)]
    pub(crate) context_size: u32,
    /// Record open-mouth chunks without synthesizing or playing audio.
    #[arg(long)]
    pub(crate) mock_mouth: bool,
}

#[derive(Debug, Args)]
pub(crate) struct ContinueCommand {
    #[arg(long, alias = "model-path")]
    pub(crate) llm_model: Option<PathBuf>,
    /// Number of llama.cpp layers to offload to the GPU. Use 0 for CPU-only LLM inference.
    #[arg(long)]
    pub(crate) llm_gpu_layers: Option<u32>,
    #[arg(long)]
    pub(crate) piper_bin: Option<PathBuf>,
    #[arg(long)]
    pub(crate) piper_voice: Option<PathBuf>,
    /// Use the source-filter acoustic model plus HiFi-GAN vocoder for Pete's voice.
    #[arg(long)]
    pub(crate) hifigan: bool,
    #[arg(long = "hifigan-model", requires = "hifigan")]
    pub(crate) hifigan_model: Option<PathBuf>,
    /// Debug the mel path with the non-neural mel debug renderer instead of HiFi-GAN.
    #[arg(long, alias = "hifigan-fallback", requires = "hifigan")]
    pub(crate) skip_gan: bool,
    #[arg(long)]
    pub(crate) whisper_model: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = VadBackendOption::WebRtc)]
    pub(crate) vad: VadBackendOption,
    /// TOML profile emitted by `listenbury vad calibrate-room --toml`.
    #[arg(long)]
    pub(crate) vad_profile: Option<PathBuf>,
    /// Prompt framing to apply to the initial prompt only. Stdin appends are inserted raw.
    #[arg(long, value_enum, default_value_t = PromptMode::Raw)]
    pub(crate) mode: PromptMode,
    /// Optional maximum generated-token cap. By default, continue until Ctrl-C or context fills.
    #[arg(long)]
    pub(crate) max_tokens: Option<u32>,
    /// llama.cpp context size for the live session.
    #[arg(long, default_value_t = 8192)]
    pub(crate) context_size: u32,
    /// Tokens reserved for generation when budgeting live prompt assembly.
    #[arg(long, default_value_t = 512)]
    pub(crate) reserved_generation_tokens: u32,
    /// Number of recent listened/spoken turns to keep verbatim before summarizing older turns.
    #[arg(long, default_value_t = 8)]
    pub(crate) verbatim_turns: usize,
    #[arg(long, value_enum, default_value_t = ModelProfile::Tiny)]
    pub(crate) model_profile: ModelProfile,
    #[arg(long)]
    pub(crate) no_backchannels: bool,
    /// Continuous VAD speech duration before TTS auto-pauses while Pete is speaking.
    #[arg(long, default_value_t = 250)]
    pub(crate) tts_vad_pause_ms: u64,
    /// Time to listen after an auto-pause before clearing or resuming TTS.
    #[arg(long, default_value_t = 700)]
    pub(crate) tts_vad_listen_ms: u64,
    /// Start the WaveDeck browser viewer alongside the continuous duplex loop (live events streamed via SSE).
    #[arg(long)]
    pub(crate) web: bool,
    /// Host for the embedded web viewer (requires --web).
    #[arg(long, default_value = "127.0.0.1")]
    pub(crate) web_host: String,
    /// Port for the embedded web viewer (requires --web).
    #[arg(long, default_value_t = 8787)]
    pub(crate) web_port: u16,
    /// Run a synthetic duplex diagnostic scenario instead of the live mic/speaker loop.
    #[arg(long, value_enum)]
    pub(crate) duplex_trace_scenario: Option<DuplexTraceScenarioOption>,
    /// Write either a JSONL trace file or a structured trace-session directory (required when --duplex-trace-scenario is set).
    #[arg(long)]
    pub(crate) jsonl: Option<PathBuf>,
    /// Capture native Linux camera frames from a V4L2 device and vectorize them.
    #[arg(long)]
    pub(crate) native_video: bool,
    /// V4L2 camera device for --native-video.
    #[arg(long, default_value = "/dev/video0")]
    pub(crate) video_device: PathBuf,
    /// Capture width for --native-video.
    #[arg(long, default_value_t = 320)]
    pub(crate) video_width: u32,
    /// Capture height for --native-video.
    #[arg(long, default_value_t = 240)]
    pub(crate) video_height: u32,
    /// Frames per second for --native-video.
    #[arg(long, default_value_t = 2)]
    pub(crate) video_fps: u32,
    /// Mark native video artifacts as retained. The default is vector-only.
    #[arg(long)]
    pub(crate) retain_video_images: bool,
    /// Initial prompt text. If omitted, generation starts from Pete's continuous-awareness seed.
    #[arg(num_args = 0.., trailing_var_arg = true)]
    pub(crate) prompt: Vec<String>,
}

#[derive(Debug, Args)]
pub(crate) struct TraceViewerExportCommand {
    /// Runtime trace JSONL file or trace-session directory generated by `listenbury listen --jsonl ...` or `listenbury dev continue --jsonl ...`.
    pub(crate) input_jsonl: PathBuf,
    /// Destination browser viewer payload JSON.
    pub(crate) output_json: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct ProsodyPlanCommand {
    /// Forced-alignment JSON, usually adapted from WhisperX or MFA output.
    pub(crate) alignment_json: PathBuf,
    /// Praat nuclei/silence JSON.
    pub(crate) praat_json: PathBuf,
    /// Destination normalized prosody timing plan JSON.
    pub(crate) output_json: PathBuf,
    /// Optional SSML file with mark and break tags derived from the plan.
    #[arg(long)]
    pub(crate) ssml_output: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct MelRoundtripCommand {
    /// Reference WAV to extract SpeechT5 log-mel features from.
    pub(crate) input_wav: PathBuf,
    /// HiFi-GAN ONNX model compatible with SpeechT5 log-mel spectrograms.
    #[arg(long = "hifigan-model")]
    pub(crate) hifigan_model: PathBuf,
    /// Destination reconstructed WAV path.
    #[arg(long, default_value = "out/listenbury-mel-roundtrip.wav")]
    pub(crate) output_wav: PathBuf,
    /// Optional text dump of the extracted mel matrix.
    #[arg(long)]
    pub(crate) mel_dump: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct SingDemoCommand {
    #[arg(
        long,
        value_enum,
        conflicts_with_all = ["piper", "klatt", "diphone", "hifigan", "speecht5"]
    )]
    pub(crate) backend: Option<SingDemoBackendOption>,
    #[arg(long, conflicts_with_all = ["backend", "klatt", "diphone", "hifigan", "speecht5"])]
    pub(crate) piper: bool,
    #[arg(long, conflicts_with_all = ["backend", "piper", "diphone", "hifigan", "speecht5"])]
    pub(crate) klatt: bool,
    #[arg(long, conflicts_with_all = ["backend", "piper", "klatt", "hifigan", "speecht5"])]
    pub(crate) diphone: bool,
    #[arg(long, conflicts_with_all = ["backend", "piper", "klatt", "diphone", "speecht5"])]
    pub(crate) hifigan: bool,
    #[arg(long, conflicts_with_all = ["backend", "piper", "klatt", "diphone", "hifigan"])]
    pub(crate) speecht5: bool,
    #[arg(long = "diphone-voice", requires = "diphone")]
    pub(crate) mbrola_voice: Option<PathBuf>,
    #[arg(long)]
    pub(crate) output_wav: Option<PathBuf>,
    #[arg(long)]
    pub(crate) piper_bin: Option<PathBuf>,
    #[arg(long, alias = "model-path")]
    pub(crate) piper_voice: Option<PathBuf>,
    #[arg(long = "hifigan-model")]
    pub(crate) hifigan_model: Option<PathBuf>,
    /// Debug the mel path with the non-neural mel debug renderer instead of HiFi-GAN.
    #[arg(long, alias = "hifigan-fallback")]
    pub(crate) skip_gan: bool,
}

impl SingDemoCommand {
    pub(crate) fn selected_backend(&self) -> SingDemoBackendOption {
        if self.piper {
            SingDemoBackendOption::Piper
        } else if self.klatt {
            SingDemoBackendOption::Klatt
        } else if self.diphone {
            SingDemoBackendOption::Mbrola
        } else if self.hifigan || self.speecht5 {
            SingDemoBackendOption::Speecht5
        } else {
            self.backend.unwrap_or(SingDemoBackendOption::Riper)
        }
    }
}

#[derive(Debug, Args)]
pub(crate) struct WebCommand {
    #[arg(long, default_value = "127.0.0.1")]
    pub(crate) host: String,
    #[arg(long, default_value_t = 8787)]
    pub(crate) port: u16,
    #[arg(long)]
    pub(crate) payload: Option<PathBuf>,
    #[arg(long)]
    pub(crate) trace: Option<PathBuf>,
    #[arg(long)]
    pub(crate) open: bool,
}

#[derive(Debug, Args)]
pub(crate) struct TranscribeCommand {
    pub(crate) input_wav: Option<PathBuf>,
    #[arg(long, alias = "model-path")]
    pub(crate) whisper_model: Option<PathBuf>,
    /// Slower Whisper model used for rolling refinement in --web mode.
    #[arg(long)]
    pub(crate) refine_whisper_model: Option<PathBuf>,
    /// Rolling audio window refined by the slower model in --web mode.
    #[arg(long, default_value_t = 90)]
    pub(crate) refine_window_seconds: u64,
    /// Minimum delay between queued rolling refinement passes in --web mode.
    #[arg(long, default_value_t = 1_500)]
    pub(crate) refine_interval_ms: u64,
    #[arg(long, default_value_t = 30)]
    pub(crate) seconds: u64,
    #[arg(long)]
    pub(crate) until_ctrl_c: bool,
    #[arg(long, value_enum, default_value_t = VadBackendOption::WebRtc)]
    pub(crate) vad: VadBackendOption,
    /// TOML profile emitted by `listenbury vad calibrate-room --toml`.
    #[arg(long)]
    pub(crate) vad_profile: Option<PathBuf>,
    /// Start the screenplay and WaveDeck browser viewers; microphone capture runs until Ctrl-C.
    #[arg(long)]
    pub(crate) web: bool,
    /// Host for the embedded web viewer (requires --web).
    #[arg(long, default_value = "127.0.0.1")]
    pub(crate) web_host: String,
    /// Port for the embedded web viewer (requires --web).
    #[arg(long, default_value_t = 8787)]
    pub(crate) web_port: u16,
}

#[derive(Debug, Args)]
pub(crate) struct SayCommand {
    #[arg(long)]
    pub(crate) piper: bool,
    /// Select the internal Riper route. This is the default, but the flag is accepted for explicit inspection commands.
    #[arg(long, conflicts_with_all = ["piper", "klatt", "hifigan", "speecht5", "diphone", "rp"])]
    pub(crate) riper: bool,
    #[arg(long)]
    pub(crate) piper_bin: Option<PathBuf>,
    #[arg(long, alias = "model-path")]
    pub(crate) piper_voice: Option<PathBuf>,
    #[arg(long)]
    pub(crate) output_wav: Option<PathBuf>,
    /// Print the selected speech pipeline before synthesis.
    #[arg(long, alias = "trace-speech-pipeline")]
    pub(crate) dump_pipeline: bool,
    /// Print phoneme and phone translations before synthesis.
    #[arg(long)]
    pub(crate) dump_phonemes: bool,
    /// Print the shared JSON phone plan and exit before synthesis.
    #[arg(long)]
    pub(crate) dump_phone_plan: bool,
    /// Print the Piper-compatible ONNX tensor contract before synthesis.
    #[arg(long)]
    pub(crate) dump_piper_tensors: bool,
    #[arg(long, conflicts_with_all = ["piper", "hifigan", "speecht5", "diphone"])]
    pub(crate) klatt: bool,
    #[arg(long, conflicts_with_all = ["piper", "klatt", "speecht5", "diphone", "rp"])]
    /// Native SpeechT5 acoustic route with SpeechT5 HiFi-GAN vocoding.
    pub(crate) hifigan: bool,
    /// Native SpeechT5 acoustic route: text/tokenizer -> SpeechT5 encoder/decoder mel -> SpeechT5 HiFi-GAN.
    #[arg(long, conflicts_with_all = ["piper", "klatt", "hifigan", "diphone", "rp"])]
    pub(crate) speecht5: bool,
    #[arg(long = "hifigan-model")]
    pub(crate) hifigan_model: Option<PathBuf>,
    /// Debug the mel path with the non-neural mel debug renderer instead of HiFi-GAN.
    #[arg(long, alias = "hifigan-fallback", requires = "hifigan")]
    pub(crate) skip_gan: bool,
    #[arg(long, conflicts_with_all = ["piper", "klatt", "hifigan", "speecht5", "mbrola_voice"])]
    pub(crate) rp: bool,
    #[arg(long, conflicts_with_all = ["piper", "klatt", "hifigan", "speecht5"])]
    pub(crate) diphone: bool,
    #[arg(long = "diphone-voice", requires = "diphone")]
    pub(crate) mbrola_voice: Option<PathBuf>,
    #[arg(num_args = 0.., trailing_var_arg = true)]
    pub(crate) words: Vec<String>,
}

#[derive(Debug, Args)]
pub(crate) struct MbrolaRenderCommand {
    #[arg(long)]
    pub(crate) voice: PathBuf,
    #[arg(long)]
    pub(crate) phones: PathBuf,
    #[arg(long)]
    pub(crate) out: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct MbrolaInventoryCommand {
    /// Path to the MBROLA voice database file.
    #[arg(long)]
    pub(crate) voice: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct MbrolaAuditCommand {
    /// Path to the MBROLA voice database file.
    #[arg(long)]
    pub(crate) voice: PathBuf,
    /// Path to the MBROLA `.pho` plan to audit against.
    #[arg(long)]
    pub(crate) plan: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct RiperCompareCommand {
    #[arg(long)]
    pub(crate) piper_bin: Option<PathBuf>,
    #[arg(long, alias = "model-path")]
    pub(crate) piper_voice: Option<PathBuf>,
    #[arg(long)]
    pub(crate) riper_voice: Option<PathBuf>,
    #[arg(long)]
    pub(crate) riper_config: Option<PathBuf>,
    #[arg(long)]
    pub(crate) process_output_wav: Option<PathBuf>,
    #[arg(long)]
    pub(crate) riper_output_wav: Option<PathBuf>,
    #[arg(long)]
    pub(crate) phonemes: Option<String>,
    /// Text to compare. Defaults to a phonetic pangram-style utterance.
    #[arg(num_args = 0.., trailing_var_arg = true)]
    pub(crate) words: Vec<String>,
}

#[derive(Debug, Args)]
pub(crate) struct EchoCommand {
    pub(crate) input_wav: PathBuf,
    #[arg(long)]
    pub(crate) whisper_model: Option<PathBuf>,
    #[arg(long, alias = "model-path")]
    pub(crate) piper_voice: Option<PathBuf>,
    #[arg(long)]
    pub(crate) riper_config: Option<PathBuf>,
    #[arg(long)]
    pub(crate) output_wav: Option<PathBuf>,
    #[arg(long)]
    pub(crate) comparison_json: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct RoundTripWavCommand {
    pub(crate) input_wav: PathBuf,
    #[arg(long)]
    pub(crate) whisper_model: Option<PathBuf>,
    #[arg(long)]
    pub(crate) llm_model: Option<PathBuf>,
    /// Number of llama.cpp layers to offload to the GPU. Use 0 for CPU-only LLM inference.
    #[arg(long)]
    pub(crate) llm_gpu_layers: Option<u32>,
    #[arg(long)]
    pub(crate) piper_bin: Option<PathBuf>,
    #[arg(long)]
    pub(crate) piper_voice: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct DogfoodTwoCommand {
    /// Initial utterance that instance A speaks on turn one.
    #[arg(long, default_value = "Hello.")]
    pub(crate) seed: String,
    /// Total number of TTS→ASR exchanges to run (hard-capped at 32).
    #[arg(long, default_value_t = 6)]
    pub(crate) turns: usize,
    /// Maximum tokens the LLM may generate per turn.
    #[arg(long, default_value_t = 128)]
    pub(crate) max_tokens: u32,
    #[arg(long, alias = "model-path")]
    pub(crate) whisper_model: Option<PathBuf>,
    #[arg(long)]
    pub(crate) llm_model: Option<PathBuf>,
    /// Number of llama.cpp layers to offload to the GPU. Defaults lower for CUDA live mode so ASR and LLM fit together.
    #[arg(long)]
    pub(crate) llm_gpu_layers: Option<u32>,
    #[arg(long)]
    pub(crate) piper_bin: Option<PathBuf>,
    /// Piper voice used by instance A.
    #[arg(long)]
    pub(crate) piper_voice_a: Option<PathBuf>,
    /// Piper voice used by instance B (defaults to the same voice as A).
    #[arg(long)]
    pub(crate) piper_voice_b: Option<PathBuf>,
    /// Write a JSONL trace file to this path.
    #[arg(long)]
    pub(crate) jsonl: Option<PathBuf>,
    /// Save per-turn WAV files to this directory.
    #[arg(long)]
    pub(crate) save_audio_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum PromptMode {
    Raw,
    #[default]
    Spoken,
    Chat,
    Inner,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum ModelProfile {
    #[default]
    Tiny,
}

#[derive(Debug, Args)]
pub(crate) struct LiveHalfDuplexCommand {
    /// Stop listening after this many seconds. By default, listen until Ctrl-C.
    #[arg(long)]
    pub(crate) seconds: Option<u64>,
    /// Write either a JSONL trace file or a structured trace-session directory.
    #[arg(long)]
    pub(crate) jsonl: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = ModelProfile::Tiny)]
    pub(crate) model_profile: ModelProfile,
    #[arg(long)]
    pub(crate) no_backchannels: bool,
    #[arg(long)]
    pub(crate) whisper_model: Option<PathBuf>,
    #[arg(long)]
    pub(crate) llm_model: Option<PathBuf>,
    /// Number of llama.cpp layers to offload to the GPU. Defaults lower for CUDA live mode so ASR and LLM fit together.
    #[arg(long)]
    pub(crate) llm_gpu_layers: Option<u32>,
    /// llama.cpp context window size for live prompt plus generation.
    #[arg(long, default_value_t = 8192)]
    pub(crate) context_size: u32,
    /// Tokens reserved for generation when budgeting live prompt assembly.
    #[arg(long, default_value_t = 512)]
    pub(crate) reserved_generation_tokens: u32,
    #[arg(long)]
    pub(crate) piper_bin: Option<PathBuf>,
    #[arg(long)]
    pub(crate) piper_voice: Option<PathBuf>,
    /// Use the source-filter acoustic model plus HiFi-GAN vocoder for Pete's voice.
    #[arg(long)]
    pub(crate) hifigan: bool,
    #[arg(long = "hifigan-model", requires = "hifigan")]
    pub(crate) hifigan_model: Option<PathBuf>,
    /// Debug the mel path with the non-neural mel debug renderer instead of HiFi-GAN.
    #[arg(long, alias = "hifigan-fallback", requires = "hifigan")]
    pub(crate) skip_gan: bool,
    #[arg(long, value_enum, default_value_t = VadBackendOption::WebRtc)]
    pub(crate) vad: VadBackendOption,
    /// TOML profile emitted by `listenbury vad calibrate-room --toml`.
    #[arg(long)]
    pub(crate) vad_profile: Option<PathBuf>,
    /// Start the WaveDeck browser viewer alongside the listen loop (live events streamed via SSE).
    #[arg(long)]
    pub(crate) web: bool,
    /// Capture native Linux camera frames from a V4L2 device and vectorize them.
    #[arg(long)]
    pub(crate) native_video: bool,
    /// V4L2 camera device for --native-video.
    #[arg(long, default_value = "/dev/video0")]
    pub(crate) video_device: PathBuf,
    /// Capture width for --native-video.
    #[arg(long, default_value_t = 320)]
    pub(crate) video_width: u32,
    /// Capture height for --native-video.
    #[arg(long, default_value_t = 240)]
    pub(crate) video_height: u32,
    /// Frames per second for --native-video.
    #[arg(long, default_value_t = 2)]
    pub(crate) video_fps: u32,
    /// Mark native video artifacts as retained. The default is vector-only.
    #[arg(long)]
    pub(crate) retain_video_images: bool,
    /// Run the continuous duplex development pipeline instead of the half-duplex loop.
    #[arg(long)]
    pub(crate) duplex: bool,
    /// Host for the embedded web viewer (requires --web).
    #[arg(long, default_value = "127.0.0.1")]
    pub(crate) web_host: String,
    /// Port for the embedded web viewer (requires --web).
    #[arg(long, default_value_t = 8787)]
    pub(crate) web_port: u16,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum VadBackendOption {
    Energy,
    #[default]
    #[value(name = "webrtc")]
    WebRtc,
    Silero,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub(crate) enum DuplexTraceScenarioOption {
    #[value(name = "overlap-yield")]
    OverlapYield,
}

impl VadBackendOption {
    pub(crate) fn as_backend_kind(self) -> VadBackendKind {
        match self {
            Self::Energy => VadBackendKind::Energy,
            Self::WebRtc => VadBackendKind::WebRtc,
            Self::Silero => VadBackendKind::Silero,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ResolvedVadConfig {
    pub(crate) backend: VadBackendKind,
    pub(crate) profile: Option<VadProfile>,
}

pub(crate) fn resolve_vad_config(
    selected: VadBackendOption,
    profile_path: Option<&Path>,
) -> Result<ResolvedVadConfig> {
    let profile = profile_path.map(VadProfile::read_toml).transpose()?;
    Ok(ResolvedVadConfig {
        backend: profile
            .as_ref()
            .map(|profile| profile.backend)
            .unwrap_or_else(|| selected.as_backend_kind()),
        profile,
    })
}

#[derive(Debug, Subcommand)]
pub(crate) enum ModelsCommand {
    Menu,
    Fetch(ModelsFetchCommand),
    List,
    Use(ModelsUseCommand),
    Status(ModelsStatusCommand),
    Verify(ModelsVerifyCommand),
    Repair(ModelsFetchCommand),
    Path,
}

#[derive(Debug, Args)]
pub(crate) struct ModelsFetchCommand {
    /// Fetch one model bundle by name instead of the currently selected models.
    pub(crate) model: Option<String>,
    /// Fetch every registered asset.
    #[arg(long)]
    pub(crate) all: bool,
    /// Verify existing files against manifest checksums/sizes and repair invalid assets.
    #[arg(long)]
    pub(crate) verify: bool,
    /// Maximum number of model assets to download at once.
    #[arg(long, default_value_t = 2)]
    pub(crate) jobs: usize,
}

#[derive(Debug, Args, Default)]
pub(crate) struct ModelsStatusCommand {
    /// Verify existing model files against manifest checksums/sizes.
    #[arg(long)]
    pub(crate) verify: bool,
}

/// Arguments for `listenbury models verify`.
#[derive(Debug, Args, Default)]
pub(crate) struct ModelsVerifyCommand {
    /// Verify one model bundle by name instead of all registered assets.
    pub(crate) model: Option<String>,
}

#[derive(Debug, Args)]
pub(crate) struct ModelsUseCommand {
    #[arg(value_enum)]
    pub(crate) kind: ModelsUseKind,
    pub(crate) model: String,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub(crate) enum ModelsUseKind {
    Llm,
    #[value(alias = "embedding", alias = "embed", alias = "text")]
    TextEmbedding,
    Voice,
    Acoustic,
    Vocoder,
    Whisper,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum SingDemoBackendOption {
    #[default]
    Klatt,
    Riper,
    Mbrola,
    Piper,
    Hifigan,
    Speecht5,
}

#[derive(Debug, Subcommand)]
pub(crate) enum SyntheticCacheCommand {
    Prewarm(SyntheticCachePrewarmCommand),
}

#[derive(Debug, Args)]
pub(crate) struct SyntheticCachePrewarmCommand {
    #[arg(long)]
    pub(crate) piper_bin: Option<PathBuf>,
    #[arg(long)]
    pub(crate) piper_voice: Option<PathBuf>,
    #[arg(long)]
    pub(crate) listenbury_home: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct SoundscapeDebugCommand {
    /// Print a built-in sample debug view demonstrating the output format.
    #[arg(long)]
    pub(crate) sample: bool,
    /// Path to a JSON file containing a soundscape debug input
    /// (`frame`, `voice_count`, `hypotheses`, `transcripts`).
    #[arg(long)]
    pub(crate) input: Option<PathBuf>,
    /// Pretty-print the JSON output (default: compact).
    #[arg(long)]
    pub(crate) pretty: bool,
}

#[derive(Debug, Subcommand)]
pub(crate) enum DiphoneCacheCommand {
    /// Forge a single diphone and store it in the cache.
    Forge(DiphoneCacheForgeCommand),
    /// Build a full diphone inventory cache from a phone list file.
    Build(DiphoneCacheBuildCommand),
    /// List cached diphone entries and metadata.
    List(DiphoneCacheListCommand),
    /// Audit a .pho/PhoneTimedPlan against MBROLA, cache, and neural forge availability.
    AuditPlan(DiphoneAuditPlanCommand),
}

#[derive(Debug, Subcommand)]
pub(crate) enum DiphoneCommand {
    /// Forge a single diphone and store it in the cache.
    Forge(DiphoneCacheForgeCommand),
    /// Create a cache-backed diphone voice from a Piper/Riper ONNX model.
    Wizard(DiphoneWizardCommand),
    /// Create a MBROLA voice database from a diphone PCM cache folder.
    #[command(alias = "mbrola")]
    MbrolaDatabase(DiphoneMbrolaDatabaseCommand),
    /// Build a full diphone inventory cache.
    #[command(alias = "build")]
    CacheBuild(DiphoneCacheBuildCommand),
    /// List cached diphone entries and metadata.
    CacheList(DiphoneCacheListCommand),
    /// Audit a .pho/PhoneTimedPlan against MBROLA, cache, and neural forge availability.
    AuditPlan(DiphoneAuditPlanCommand),
}

#[derive(Debug, Subcommand)]
pub(crate) enum VadCommand {
    /// Capture/read room tone and emit baseline stats + an initial VAD profile.
    CalibrateRoom(VadCalibrateRoomCommand),
    /// Calibrate one backend against optional labeled intervals.
    Calibrate(VadCalibrateCommand),
    /// Compare multiple backends against optional labeled intervals.
    Compare(VadCompareCommand),
}

#[derive(Debug, Args)]
pub(crate) struct VadCalibrateRoomCommand {
    /// Optional WAV path. If omitted, capture from the default input device.
    #[arg(long)]
    pub(crate) audio: Option<PathBuf>,
    /// Capture duration used when --audio is omitted.
    #[arg(long, default_value_t = 20)]
    pub(crate) seconds: u64,
    /// Optional JSON report output path.
    #[arg(long)]
    pub(crate) json: Option<PathBuf>,
    /// Optional TOML profile output path.
    #[arg(long)]
    pub(crate) toml: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = VadBackendOption::Energy)]
    pub(crate) vad: VadBackendOption,
}

#[derive(Debug, Args)]
pub(crate) struct VadCalibrateCommand {
    /// Fixture WAV used for calibration.
    #[arg(long)]
    pub(crate) audio: PathBuf,
    /// Optional labeled intervals JSON.
    #[arg(long)]
    pub(crate) labels: Option<PathBuf>,
    /// Optional JSON report output path.
    #[arg(long)]
    pub(crate) json: Option<PathBuf>,
    /// Optional TOML profile output path.
    #[arg(long)]
    pub(crate) toml: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = VadBackendOption::Energy)]
    pub(crate) vad: VadBackendOption,
}

#[derive(Debug, Args)]
pub(crate) struct VadCompareCommand {
    /// Fixture WAV used for backend comparison.
    #[arg(long)]
    pub(crate) audio: PathBuf,
    /// Optional labeled intervals JSON.
    #[arg(long)]
    pub(crate) labels: Option<PathBuf>,
    /// Optional JSON report output path.
    #[arg(long)]
    pub(crate) json: Option<PathBuf>,
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_values_t = [
            VadBackendOption::Energy,
            VadBackendOption::WebRtc,
            VadBackendOption::Silero
        ]
    )]
    pub(crate) backends: Vec<VadBackendOption>,
}

#[derive(Debug, Args)]
pub(crate) struct DiphoneCacheForgeCommand {
    /// Path to the Piper ONNX model file.
    #[arg(long)]
    pub(crate) model: PathBuf,
    /// Path to the Piper voice config JSON.
    #[arg(long)]
    pub(crate) config: PathBuf,
    /// Left phone symbol of the diphone to forge (e.g. `h`).
    #[arg(long)]
    pub(crate) left: String,
    /// Right phone symbol of the diphone to forge (e.g. `@`).
    #[arg(long)]
    pub(crate) right: String,
    /// Directory for the diphone cache (defaults to `./diphone-cache`).
    #[arg(long, default_value = "diphone-cache")]
    pub(crate) cache_dir: PathBuf,
    /// Directory for forge debug outputs (carrier/extracted/normalized WAV + JSON).
    #[arg(long)]
    pub(crate) debug_dir: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct DiphoneCacheBuildCommand {
    /// Path to the Piper ONNX model file.
    #[arg(long)]
    pub(crate) model: PathBuf,
    /// Path to the Piper voice config JSON.
    #[arg(long)]
    pub(crate) config: PathBuf,
    /// Inventory name (e.g. `en-us-basic`) or a text file with one phone symbol per line.
    #[arg(long)]
    pub(crate) inventory: String,
    /// Re-forge and overwrite units even when the cache entry already exists.
    #[arg(long)]
    pub(crate) force: bool,
    /// Directory for the diphone cache (defaults to `./diphone-cache`).
    #[arg(long, default_value = "diphone-cache")]
    pub(crate) cache_dir: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct DiphoneMbrolaDatabaseCommand {
    /// Directory containing diphone cache `.json`/`.pcm` pairs.
    #[arg(long, alias = "pcm-folder")]
    pub(crate) pcm_dir: PathBuf,
    /// Output MBROLA database path.
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Output database sample rate. Defaults to the cache metadata sample rate.
    #[arg(long)]
    pub(crate) sample_rate: Option<u32>,
    /// MBROLA frame period in samples.
    #[arg(long, default_value_t = 80)]
    pub(crate) mbr_period: usize,
}

#[derive(Debug, Args)]
pub(crate) struct DiphoneWizardCommand {
    /// Path to the Piper ONNX model file.
    pub(crate) model: PathBuf,
    /// Local voice directory/name to create (e.g. `ryan`).
    pub(crate) voice: PathBuf,
    /// Path to the Piper voice config JSON. Defaults to `<model>.json`, then `<stem>.json`.
    #[arg(long)]
    pub(crate) config: Option<PathBuf>,
    /// Inventory name or phone-list file to prewarm.
    #[arg(long, default_value = "en-us-basic")]
    pub(crate) inventory: String,
    /// Re-forge and overwrite units even when the cache entry already exists.
    #[arg(long)]
    pub(crate) force: bool,
}

#[derive(Debug, Args)]
pub(crate) struct DiphoneCacheListCommand {
    /// Path to the Piper ONNX model file.
    #[arg(long)]
    pub(crate) model: PathBuf,
    /// Path to the Piper voice config JSON.
    #[arg(long)]
    pub(crate) config: PathBuf,
    /// Directory for the diphone cache (defaults to `./diphone-cache`).
    #[arg(long, default_value = "diphone-cache")]
    pub(crate) cache_dir: PathBuf,
    /// Print full per-entry metadata.
    #[arg(long)]
    pub(crate) verbose: bool,
}

#[derive(Debug, Args)]
pub(crate) struct DiphoneAuditPlanCommand {
    /// Path to the Piper ONNX model file.
    #[arg(long)]
    pub(crate) model: PathBuf,
    /// Path to the Piper voice config JSON.
    #[arg(long)]
    pub(crate) config: PathBuf,
    /// Path to a MBROLA voice database file for exact-coverage checks.
    #[arg(long)]
    pub(crate) mbrola_voice: Option<PathBuf>,
    /// Path to the .pho or PhoneTimedPlan JSON to audit.
    #[arg(long)]
    pub(crate) plan: PathBuf,
    /// Directory for the diphone cache (defaults to `./diphone-cache`).
    #[arg(long, default_value = "diphone-cache")]
    pub(crate) cache_dir: PathBuf,
}
