#[cfg(feature = "tts-riper")]
pub mod backend;
pub mod config;
pub mod echo;
pub mod encoder;
pub mod evidence;
pub mod g2p;
pub mod morphophonology;
pub mod phoneme;
pub mod prosody_audit;
pub mod prosody_controls;
pub mod prosody_planner;
pub mod sentence_analysis;
pub mod text;

pub use crate::linguistic::language_pack_rules::{
    BoundaryProsodyRuleSeed, LanguagePackEnvironmentRule, LexicalProsodyFlag,
    LexicalProsodyFlagFact, LinguisticVarieties, LinguisticVarietyRuleTable, MatchedWordSpan,
    MorphophonologyOutput, MorphophonologyRule, MultiWordPronunciationRule, MultiWordRuleMatch,
    MultiWordRuleOutput, PhonemeMappingRule, PhraseRuleEntry, PronunciationOverrideRule,
    PronunciationRuleCatalog, RuleContextConstraint, RuleOutput, SourceProvenance,
    SpellingRepairHint, StemRetranslationPolicy, StressRule, ToRuleDescriptor, VoiceVariantRule,
    WeakFormRule, convert_multi_word_rule, convert_punctuation_prosody_rule,
    convert_weak_form_rule, english_lexical_flag_facts_for_rule,
    english_native_morphophonology_rules, english_pack_phrase_rules,
    english_pack_punctuation_rules, english_pack_weak_form_rules, export_rule_catalog_to_json,
    import_rule_catalog_from_str, load_pronunciation_rule_catalog, match_multi_word_rule,
    rule_matches_context,
};
#[cfg(feature = "tts-riper")]
pub use backend::{PiperModelContract, RiperBackend, RiperPcm};
pub use config::{PiperVoiceConfig, PiperVoiceConfigError};
pub use echo::{
    EchoComparisonRecord, EchoMatchedAccent, EchoMatchedPause, EchoProsodyObservation,
    EchoProsodyPlan, EchoWordProsodyObservation,
};
pub use encoder::PiperEncoder;
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
    ProsodyPitchShape, ProsodyRateClass, ProsodyTarget, RepairCue, RepairPlan, RepairStrategy,
    RestartScope, RiperProsodyRealization, SpeechCommitState, SpeechCursor,
};
pub use sentence_analysis::{
    ContextPredicate, EnvironmentPattern, HeuristicSentenceAnalyzer, OrthographicEmphasisKind,
    PartOfSpeech, ProsodicRole, ProsodyEnvironmentFacts, ReductionClass, ReductionDiagnostic,
    ReductionStatus, SentenceAnalysis, SentenceAnalyzer, SyntacticLink, SyntacticLinkKind,
    SyntacticLinkParse, SyntacticLinkSource, SyntacticRole, TokenAnalysis, WordIndex,
    WordSyntaxFacts,
};
pub use text::{
    NormalizedText, NormalizedToken, ProsodyBoundaryHint, ProsodyCommitment,
    TextNormalizationError, TextNormalizer,
};
