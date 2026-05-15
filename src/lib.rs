pub mod audio;
pub mod config;
pub mod event;
pub mod hearing;
pub mod mind;
#[cfg(feature = "model-download")]
pub mod models;
pub mod mouth;
pub mod runtime;
pub mod speech;
pub mod time;
pub mod vision;

pub use audio::frame::AudioFrame;
pub use audio::{AudioInput, AudioOutput};
pub use event::{
    AudioEvent, HearingEvent, MindEvent, MouthEvent, PeteEvent, TranscriptEvent, UtteranceId,
    VisionEvent,
};
pub use hearing::breath::{
    BreathGroupConfig, BreathGroupEndReason, BreathGroupId, BreathGroupSegmenter,
};
pub use hearing::suppression::{SelfHearingState, SuppressionDecision, SUPPRESSION_TAIL_MS};
pub use hearing::vad::{
    create_vad_backend, EnergyVad, VadBackendKind, VadResult, VoiceActivityDetector,
};
pub use mind::controller::{
    BackchannelId, ConversationController, FillerContext, FillerDecision, FillerPlanner,
    FillerPlannerConfig, RuntimePacket, DEFAULT_FILLER_REPEAT_COOLDOWN_MS,
};
#[cfg(feature = "llm-llama-cpp")]
pub use mind::llama_cpp::{LlamaCppConfig, LlamaCppEngine};
pub use mind::llm::{GenerationId, GenerationRequest, LlmEngine, LlmEvent, MockLlmEngine};
pub use mind::turn::{TurnState, TurnTracker};
#[cfg(feature = "tts-piper")]
pub use mouth::piper::{PiperConfig, PiperTextToSpeech};
pub use mouth::planner::{
    strip_emoji, ExpressiveUnit, FaceCommand, MouthCommand, SpeechPlan, SpeechPlanner,
    SpeechPlannerConfig, SpeechUnit,
};
pub use mouth::tts::TextToSpeech;
pub use runtime::{developer_diagnostics_enabled, set_developer_diagnostics_enabled};
pub use speech::breath_asr::{collect_breath_segments, BreathAsrConfig, BreathAudioSegment};
#[cfg(feature = "asr-whisper")]
pub use speech::whisper::WhisperSpeechRecognizer;
pub use time::{ExactTimestamp, Timed};
