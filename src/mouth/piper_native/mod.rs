#[cfg(feature = "tts-piper-native")]
pub mod backend;
pub mod config;
pub mod g2p;
pub mod phoneme;
pub mod text;

#[cfg(feature = "tts-piper-native")]
pub use backend::{NativePiperBackend, NativePiperPcm, PiperModelContract};
pub use config::{PiperVoiceConfig, PiperVoiceConfigError};
pub use g2p::{
    G2pError, GraphemeToPhoneme, PhoneLengthClass, PhoneLengthHint, PhonemizedUnit,
    SimpleEnglishG2p,
};
pub use phoneme::{
    PiperIdSequence, PiperPhoneme, PiperPhonemeIdConversionError, PiperPhonemeSequence,
};
pub use text::{
    NormalizedText, NormalizedToken, ProsodyBoundaryHint, ProsodyCommitment,
    TextNormalizationError, TextNormalizer,
};
