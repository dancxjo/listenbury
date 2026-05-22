pub mod audio;
pub mod config;
pub mod diagnostics;
pub mod event;
pub mod hearing;
pub mod linguistic;
pub mod live_trace;
pub mod memory;
pub mod mind;
#[cfg(feature = "model-download")]
pub mod models;
pub mod mouth;
pub mod playback_check;
pub mod prosody;
pub mod runtime;
pub mod runtime_event;
pub mod segmentation;
pub mod soundscape;
pub mod span;
pub mod speculative;
pub mod speech;
pub mod speech_timeline;
pub mod text_stability;
pub mod time;
pub mod trace;
pub mod vision;
pub mod voice;
pub mod web;
pub mod word;

pub use audio::frame::AudioFrame;
pub use audio::{
    AcousticAnalysis, AudioInput, AudioOutput, SpeechLikelihood, SpeechLikelihoodConfig,
    VoiceSignature, VoiceSignatureId, VoiceSignatureLabel, VoiceSignatureSource,
    analyze_audio_frames, analyze_mono_samples, build_speech_likelihood_stream,
};
pub use diagnostics::{developer_diagnostics_enabled, set_developer_diagnostics_enabled};
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
pub use mouth::read_aloud::{
    ReadAloudAudioPreparer, ReadAloudCandidate, ReadAloudCandidateEvent, ReadAloudCandidateTracker,
    SpeechCandidateCommitment, SpeechCandidateId,
};
pub use runtime_event::{EventSource, RuntimeEvent, RuntimeEventKind};
pub use segmentation::{
    BoundaryEvidence, BoundaryHypothesis, BoundaryHypothesisConfig, BoundaryKind,
    NucleusDetectionConfig, NucleusEvidence, SyllableExpansionConfig, SyllableIsland,
    VowelNucleusCandidate, WordRegionConfig, detect_nuclei, emit_ranked_boundary_hypotheses,
    extract_syllable_islands, generate_landmark_hypotheses, rank_word_region_hypotheses,
};
pub use soundscape::{
    AcousticContribution, AcousticMixture, AcousticMixtureId, AttributionEvidence, AudioSpan,
    ClusterId, DebugHypothesis, DebugOverlapMixture, DebugSource, DebugTranscriptEvent, EventId,
    IsolationEvaluation, IsolationPolicy, MixtureComponent, MixtureId, NoopSourceSeparator,
    OverlapMixture, PlaybackCancellationSeparator, SeparationMethod, SeparationRequest,
    SeparationResult, SoundEvent, SoundEventKind, SoundSource, Soundscape, SoundscapeContext,
    SoundscapeDebugView, SoundscapeFrame, SoundscapeId, SoundscapePipelineAdapter,
    SourceAttributedTranscript, SourceAttributor, SourceCriterion, SourceHypothesis, SourceId,
    SourceKind, SourceLabel, SourceOperation, SourceSeparator, SuppressionTarget, TimePoint,
    TimeRange, TrackingTarget, Voice, VoiceAttribution, VoiceId, VoiceKind, VoiceLabel,
    VoiceRoleInSpan, apply_separation_requests, detect_overlaps, evaluate_policies,
    self_hearing_suppression_policy,
};
pub use span::{
    Alignment, AlignmentGraph, AlignmentKind, AlignmentOffset, Cursor, Modality, Span, SpanId,
    SpanRevision, SpanState, Text, TextId,
};
pub use speech::breath_asr::{BreathAsrConfig, BreathAudioSegment, collect_breath_segments};
pub use speech::canonical_plan::{
    CanonicalArticulationHints, CanonicalEnergyPlan, CanonicalPhoneProvenance,
    CanonicalPhoneTiming, CanonicalPitchPlan, CanonicalPitchTarget, CanonicalSpeechBreak,
    CanonicalSpeechPhone, CanonicalSpeechPlan, CanonicalSpeechPlanMetadata,
    CanonicalSpeechPlanSource, CanonicalSpeechSegment, CanonicalStress, CanonicalSyllableRole,
    canonical_speech_plan_from_prosody_timing, canonical_speech_plan_to_phone_timed_plan,
    canonical_speech_plan_to_piper_phoneme_sequence, canonical_speech_plan_to_piper_timing,
};
pub use speech::prosody_timing::{
    AlignedPhone, AlignedWord, BreakReason, BreathGroup, ExternalAlignmentCommand, ForcedAlignment,
    PiperTimingBreak, PiperTimingPhone, PiperTimingPlan, PraatCommandConfig, PraatNucleus,
    PraatProsodyAnalysis, PraatSilence, ProsodyPhone, ProsodySegment, ProsodyTimingConfig,
    ProsodyTimingPlan, forced_alignment_from_json, plan_prosody_timing, praat_analysis_from_json,
    prosody_plan_to_piper_timing, prosody_plan_to_ssml, run_external_alignment, run_praat_analysis,
};
#[cfg(feature = "asr-whisper")]
pub use speech::whisper::WhisperSpeechRecognizer;
pub use speech_timeline::{AudioClipId, SessionId, SpeechUnitId, TranscriptRevisionId, TurnId};
pub use text_stability::{shared_prefix_len, stable_prefix_len};
pub use time::{
    Clock, ExactTimestamp, FakeClock, NormalizedTimestamp, SessionClock, SystemClock, Timed,
};
pub use vision::{
    AvSyncConfig, EvidenceScore, PhonemeClass, VisualEvidenceStatus, VisualProvenance,
    VisualSpeechClaim, VisualSpeechClaimKind, VisualSpeechFrame, VisualSpeechTrace, VowelShape,
};
pub use voice::mbrola::{
    FallbackReason, FallbackResult, JoinPoint, ManifestError, MbrolaDatabase, MbrolaPhone,
    MbrolaPitchTarget, MbrolaRenderer, MbrolaRendererConfig, MbrolaSymbolMap, MbrolaVoice,
    PhoneTimedPlan, PhoneTimedRenderer, RenderReport, UnitAssemblyReport, VoiceManifest,
    assemble_unit, fallback_warning, left_half_samples, phone_timed_plan_to_pho,
    prosody_timing_plan_to_phone_timed_plan, read_pho_file, resolve_left_half, resolve_right_half,
    right_half_samples, write_pho_file,
};
pub use voice::tract::{
    FormantEstimation, GlottalSourceEstimate, GlottalSourceTarget, KlattRenderConfig,
    NoiseEstimate, PhoneAcousticTarget, PhoneRenderTarget, SourceFilterFrame, SourceFilterTrack,
    VocalTractFilterEstimate, VocalTractFilterTarget, VoicingEstimate,
    default_english_phone_targets, estimate_f0_autocorrelation, phone_render_targets_from_string,
    render_phone, render_phone_string, render_targets_from_sung_syllable,
    render_targets_from_syllable, source_filter_track_from_acoustic,
    source_filter_track_from_acoustic_full,
};
pub use voice::vocal_plausibility::{
    VocalPlausibility, VocalPlausibilityConfig, assess_vocal_plausibility,
};
