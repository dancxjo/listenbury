pub mod audio;
pub mod config;
pub mod event;
pub mod hearing;
pub mod live_trace;
pub mod memory;
pub mod mind;
#[cfg(feature = "model-download")]
pub mod models;
pub mod mouth;
pub mod runtime;
pub mod speculative;
pub mod speech;
pub mod text_stability;
pub mod time;
pub mod vision;
pub mod word;

pub use audio::frame::AudioFrame;
pub use audio::{AudioInput, AudioOutput};
pub use event::{
    AudioEvent, HearingEvent, MindEvent, MouthEvent, PeteEvent, TranscriptEvent, UtteranceId,
    VisionEvent,
};
pub use hearing::{BreathGroupSegmenter, VadBackendKind, create_vad_backend};
pub use mind::controller::{
    BackchannelId, ConversationController, ConversationMessage, ConversationRole,
    DEFAULT_FILLER_ACTIVATION_DELAY_MS, DEFAULT_FILLER_REPEAT_COOLDOWN_MS, FillerContext,
    FillerDecision, FillerPlanner, FillerPlannerConfig, RuntimePacket,
};
#[cfg(feature = "llm-llama-cpp")]
pub use mind::llama_cpp::{LlamaCppConfig, LlamaCppEngine};
pub use mind::llm::{GenerationId, GenerationRequest, LlmEngine, LlmEvent, MockLlmEngine};
pub use mind::turn::{TurnState, TurnTracker};
#[cfg(feature = "tts-piper")]
pub use mouth::piper::{PiperConfig, PiperTextToSpeech};
pub use mouth::planner::{
    ExpressiveUnit, FaceCommand, MouthCommand, SpeechPlan, SpeechPlanner, SpeechPlannerConfig,
    SpeechUnit, strip_emoji,
};
pub use mouth::player::{PlaybackEvent, PlaybackUnitId, Player, SequentialPlayer};
pub use runtime::{developer_diagnostics_enabled, set_developer_diagnostics_enabled};
pub use speech::breath_asr::{BreathAsrConfig, BreathAudioSegment, collect_breath_segments};
#[cfg(feature = "asr-whisper")]
pub use speech::whisper::WhisperSpeechRecognizer;
pub use text_stability::{shared_prefix_len, stable_prefix_len};
pub use time::{ExactTimestamp, Timed};
