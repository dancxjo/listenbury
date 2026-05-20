#[cfg(feature = "tts-riper")]
pub mod backend;
pub mod config;
pub mod encoder;
pub mod g2p;
pub mod phoneme;
pub mod prosody_controls;
pub mod prosody_planner;
pub mod text;

#[cfg(feature = "tts-riper")]
pub use backend::{PiperModelContract, RiperBackend, RiperPcm};
pub use config::{PiperVoiceConfig, PiperVoiceConfigError};
pub use encoder::PiperEncoder;
pub use g2p::{
    G2pError, GraphemeToPhoneme, LexicalStressLevel, LexicalStressSource, LexicalStressTarget,
    PhoneLengthClass, PhoneLengthHint, PhoneTimingHint, PhonemeProsodyCandidate,
    PhonemeProsodyCandidateEvent, PhonemeProsodyCandidateTracker, PhonemeProsodyPhonemizer,
    PhonemizedUnit, SimpleEnglishG2p, SpeechCandidateId, TimingHintSource, WordProsodyTarget,
    WordTimingHint,
};
pub use phoneme::{
    PiperIdSequence, PiperPhoneme, PiperPhonemeIdConversionError, PiperPhonemeSequence,
};
pub use prosody_controls::{
    ControlStatusEntry, PiperBoundaryOverride, PiperPauseOverride, PiperPhonemeDurationOverride,
    PiperProsodyControls, PiperSynthesisDiagnostics, ProsodyControlStatus,
};
pub use prosody_planner::{
    BoundaryState, BreathGroupCandidate, BreathGroupId, BreathGroupProsodyPlanner, PauseOp,
    PauseStrengthClass, ProsodyAccentKind, ProsodyBoundaryHintOp, ProsodyContour, ProsodyEnergy,
    ProsodyEnergyClass, ProsodyList, ProsodyOp, ProsodyOperation, ProsodyOverlay,
    ProsodyOverlaySource, ProsodyPitchShape, ProsodyRateClass, ProsodyTarget,
    RiperProsodyRealization,
};
pub use text::{
    NormalizedText, NormalizedToken, ProsodyBoundaryHint, ProsodyCommitment,
    TextNormalizationError, TextNormalizer,
};
