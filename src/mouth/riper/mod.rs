#[cfg(feature = "tts-riper")]
pub mod backend;
pub mod config;
pub mod encoder;
pub mod espeak_ng_rules;
pub mod evidence;
pub mod g2p;
pub mod morphophonology;
pub mod phoneme;
pub mod prosody_audit;
pub mod prosody_controls;
pub mod prosody_planner;
pub mod sentence_analysis;
pub mod text;

#[cfg(feature = "tts-riper")]
pub use backend::{PiperModelContract, RiperBackend, RiperPcm};
pub use config::{PiperVoiceConfig, PiperVoiceConfigError};
pub use encoder::PiperEncoder;
pub use espeak_ng_rules::{
    EspeakNgSeedRuleTable, LinguisticVarieties, LinguisticVarietyRuleTable, PhonemeMappingRule,
    PronunciationOverrideRule, PunctuationProsodyRule, RuleContextConstraint, RuleProvenance,
    StressRule, ToRuleDescriptor, VoiceVariantRule, WeakFormRule, export_rule_table_to_json,
    import_rule_table_from_str, load_seed_rule_table,
};
pub use evidence::{
    AnalysisClaim, AnalysisSourceKind, AnalysisTarget, ClaimId, ClaimKind, ClaimValue,
    ConflictEntry, ResolvedAnalysis, SpanState, claim_from_environment_match, next_claim_id,
    resolve_claims, source_default_priority,
};
pub use g2p::{
    G2pError, GraphemeToPhoneme, LexicalStressLevel, LexicalStressSource, LexicalStressTarget,
    PhoneLengthClass, PhoneLengthHint, PhoneTimingHint, PhonemeProsodyCandidate,
    PhonemeProsodyCandidateEvent, PhonemeProsodyCandidateTracker, PhonemeProsodyPhonemizer,
    PhonemizedUnit, SimpleEnglishG2p, SpeechCandidateId, TimingHintSource, WordProsodyTarget,
    WordTimingHint,
};
pub use morphophonology::{
    AnalysisSource, DisplayNotation, MorphemeAnalysis, MorphemeBoundary, MorphemeFeatures,
    MorphemeKind, MorphologicalAnalysis, MorphophonologyResult, PhonologicalForm,
    PhonologicalStress, RealizedPhoneSequence, StressPattern, UnderlyingPhonologicalForm,
    WordPronunciation, analyze_word,
};
pub use phoneme::{
    PiperIdSequence, PiperPhoneme, PiperPhonemeIdConversionError, PiperPhonemeSequence,
};
pub use prosody_audit::{
    PauseReason, PhoLikeDiagnosticEntry, PhoLikeDiagnostics, PhraseBoundaryKind, ProminenceClass,
    ProsodyRealizationStatus, RiperStyleProfile, SpeechToken, Stress, WordProsodyInfo,
};
pub use prosody_controls::{
    ControlStatusEntry, PiperBoundaryOverride, PiperPauseOverride, PiperPhonemeDurationOverride,
    PiperProsodyControls, PiperSynthesisDiagnostics, ProsodyControlStatus,
};
pub use prosody_planner::{
    BoundaryState, BreathGroupCandidate, BreathGroupId, BreathGroupProsodyPlanner,
    FocusAccentDiagnostic, FocusAccentReason, FocusAccentStatus, PauseOp, PauseStrengthClass,
    ProsodyAccentKind, ProsodyBoundaryHintOp, ProsodyContour, ProsodyEnergy, ProsodyEnergyClass,
    ProsodyList, ProsodyOp, ProsodyOperation, ProsodyOverlay, ProsodyOverlaySource,
    ProsodyPitchShape, ProsodyRateClass, ProsodyTarget, RiperProsodyRealization,
};
pub use sentence_analysis::{
    ContextPredicate, EnvironmentPattern, HeuristicSentenceAnalyzer, PartOfSpeech, ProsodicRole,
    ReductionClass, ReductionDiagnostic, ReductionStatus, SentenceAnalysis, SentenceAnalyzer,
    SyntacticLink, SyntacticLinkKind, SyntacticLinkParse, SyntacticLinkSource, SyntacticRole,
    TokenAnalysis, WordIndex,
};
pub use text::{
    NormalizedText, NormalizedToken, ProsodyBoundaryHint, ProsodyCommitment,
    TextNormalizationError, TextNormalizer,
};
