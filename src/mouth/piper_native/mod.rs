#[cfg(feature = "tts-piper-native")]
pub mod backend;
pub mod config;
pub mod phoneme;

#[cfg(feature = "tts-piper-native")]
pub use backend::{NativePiperBackend, NativePiperPcm, PiperModelContract};
pub use config::{PiperVoiceConfig, PiperVoiceConfigError};
pub use phoneme::{
    PiperIdSequence, PiperPhoneme, PiperPhonemeIdConversionError, PiperPhonemeSequence,
};
