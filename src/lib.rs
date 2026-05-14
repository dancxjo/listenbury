pub mod audio;
pub mod config;
pub mod event;
pub mod hearing;
pub mod mind;
#[cfg(feature = "model-download")]
pub mod models;
pub mod mouth;
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
pub use hearing::vad::{EnergyVad, VadResult, VoiceActivityDetector};
#[cfg(feature = "llm-llama-cpp")]
pub use mind::llama_cpp::{LlamaCppConfig, LlamaCppEngine};
pub use mind::llm::{GenerationId, GenerationRequest, LlmEngine, LlmEvent, MockLlmEngine};
pub use mind::controller::{
    BackchannelId, ConversationController, FillerContext, FillerDecision, FillerPlanner,
    FillerPlannerConfig, RuntimePacket, DEFAULT_FILLER_REPEAT_COOLDOWN_MS,
};
pub use mind::turn::{TurnState, TurnTracker};
#[cfg(feature = "tts-piper")]
pub use mouth::piper::{PiperConfig, PiperTextToSpeech};
pub use mouth::planner::{MouthCommand, SpeechPlan, SpeechPlanner, SpeechUnit};
pub use mouth::tts::TextToSpeech;
#[cfg(feature = "asr-whisper")]
pub use speech::whisper::WhisperSpeechRecognizer;
pub use time::{ExactTimestamp, Timed};
