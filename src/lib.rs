pub mod acoustic;
pub mod audio;
pub mod config;
pub mod diagnostics;
pub mod event;
pub mod hearing;
pub mod linguistic;
pub mod live_trace;
pub mod loop_trace;
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
pub mod vocoder;
pub mod voice;
pub mod web;
pub mod word;

pub use acoustic::{
    AcousticFrameTrack, AcousticInput, AcousticModelBackend, MelFrame,
    MelTemporalDiscontinuityStats, NeuralAcousticModel, NeuralAcousticModelKind,
    NeuralAcousticOnnxConfig, NeuralAcousticTensorNames, NeuralAcousticTrackContract,
    NeuralMelOutputLayout, NeuralPhoneIdMap, SingingPlan, SourceFilterAcousticModel,
    SpeechT5OnnxAcousticGenerator, SpeechT5OnnxPaths, acoustic_model_by_id, list_acoustic_models,
    mel_frame_delta_energy, summarize_mel_temporal_discontinuity, temporal_smooth_mel_frames,
};
pub use audio::frame::AudioFrame;
pub use audio::{
    AcousticAnalysis, AudioInput, AudioOutput, SpeechLikelihood, SpeechLikelihoodConfig,
    VoiceSignature, VoiceSignatureId, VoiceSignatureLabel, VoiceSignatureSource,
    VoiceVectorObservation, analyze_audio_frames, analyze_mono_samples,
    build_speech_likelihood_stream, voice_vector_from_audio_frames,
};
pub use config::{ListenburyConfig, VadProfile};
pub use diagnostics::{developer_diagnostics_enabled, set_developer_diagnostics_enabled};
pub use event::{
    AudioEvent, HearingEvent, MindEvent, MouthEvent, PeteEvent, TranscriptEvent, UtteranceId,
    VisionEvent,
};
pub use hearing::{
    BreathGroupSegmenter, UtteranceEndReason, UtteranceSmoother, UtteranceSmootherConfig,
    UtteranceSmootherEvent, UtteranceSmootherState, VadBackendKind, create_vad_backend,
};
pub use loop_trace::{
    LatencyBucket, LatencySummary, MockLoopTraceConfig, TraceEvent, append_mock_downstream_trace,
    mock_interaction_trace, mock_payload, real_payload, summarize_latency, write_trace_jsonl,
};
pub use memory::{
    DEFAULT_KNOWN_VOICE_REGISTRY_PATH, DeterministicKnownVoiceEmbeddingProvider,
    KNOWN_VOICE_EMBEDDING_BACKEND, KNOWN_VOICE_LOCALITY, KNOWN_VOICE_QDRANT_COLLECTION,
    KnownVoiceEmbeddingProvider, KnownVoiceMemoryStore, QdrantKnownVoiceMatcher,
};
pub use mind::context::{
    ContextBudget, ContextGraph, ContextNode, ContextNodeRole, ContextProvider,
    ConversationContext, ConversationTurn, DEFAULT_CONTEXT_MAX_CHARS,
    DEFAULT_GRAPH_SUMMARY_CHARS_PER_TOKEN, DEFAULT_GRAPH_SUMMARY_MAX_CHARS, DEFAULT_SELF_NODE_ID,
    DEFAULT_SELF_NODE_LABEL, EmbeddingRecall, EmbeddingRecallProvider, ExpandedContextGraph,
    ExpandedEdge, ExpandedNode, GraphExpansionRequest, GraphNeighborhoodSummary,
    GraphNeighborhoodSummaryConfig, GraphNeighborhoodSummaryStats, GraphNodeId, GraphNodeRef,
    GraphTraversalEdge, PinScope, PinnedContextNode, QdrantEmbeddingRecall, RecallHit, RecallQuery,
    RecallSource, StubContextProvider, TraversalDirection, TraversalPathEdge, TraversalProvenance,
    build_conversation_context, expand_context_graph,
};
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
    ExpressiveUnit, FaceCommand, MouthCommand, MouthSyntheticPlan, SyntheticPlanner,
    SyntheticPlannerConfig, SyntheticUnit, strip_emoji,
};
pub use mouth::player::{PlaybackEvent, PlaybackUnitId, Player, SequentialPlayer};
pub use mouth::read_aloud::{
    ReadAloudAudioPreparer, ReadAloudCandidate, ReadAloudCandidateEvent, ReadAloudCandidateTracker,
    SyntheticCandidateCommitment, SyntheticCandidateId,
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
    ClusterId, DebugHypothesis, DebugOverlapMixture, DebugSource, DebugTranscriptEvent,
    EmbeddingRef, EnrollmentQuality, EnrollmentSource, EventId, IsolationEvaluation,
    IsolationPolicy, KnownVoice, KnownVoiceRegistry, MixtureComponent, MixtureId, MockVoiceMatcher,
    NoopSourceSeparator, OverlapMixture, PlaybackCancellationSeparator, SeparationMethod,
    SeparationRequest, SeparationResult, SoundEvent, SoundEventKind, SoundSource, Soundscape,
    SoundscapeContext, SoundscapeDebugView, SoundscapeFrame, SoundscapeId,
    SoundscapePipelineAdapter, SourceAttributedTranscript, SourceAttributor, SourceCriterion,
    SourceHypothesis, SourceId, SourceKind, SourceLabel, SourceOperation, SourceSeparator,
    SuppressionTarget, TimePoint, TimeRange, TrackingTarget, Voice, VoiceAttribution,
    VoiceAttributionAlternative, VoiceAttributionSource, VoiceEnrollmentSample,
    VoiceEnrollmentSampleId, VoiceId, VoiceKind, VoiceLabel, VoiceMatcher, VoiceRoleInSpan,
    apply_separation_requests, detect_overlaps, evaluate_policies, self_hearing_suppression_policy,
};
pub use span::{
    Alignment, AlignmentGraph, AlignmentKind, AlignmentOffset, Cursor, Modality, Span, SpanId,
    SpanRevision, SpanState, Text, TextId,
};
pub use speech::breath_asr::{BreathAsrConfig, BreathAudioSegment, collect_breath_segments};
pub use speech::phone_plan::{LexicalStatus, PhonePlan, PhoneSpan, WordPlan};
pub use speech::pipeline::{
    AcousticPlan, AcousticPlanner, AudioRender, LinguisticAnalyzer, LinguisticPlan, MouthSink,
    ProsodyPlanner, SpeechPipeline, SpeechStageDescriptor, SpeechStageKind, VocoderRenderer,
};
pub use speech::prosody_timing::{
    AlignedPhone, AlignedWord, BreakReason, BreathGroup, ExternalAlignmentCommand, ForcedAlignment,
    PiperTimingBreak, PiperTimingPhone, PiperTimingPlan, PraatCommandConfig, PraatNucleus,
    PraatProsodyAnalysis, PraatSilence, ProsodyPhone, ProsodySegment, ProsodyTimingConfig,
    ProsodyTimingPlan, forced_alignment_from_json, plan_prosody_timing, praat_analysis_from_json,
    prosody_plan_to_piper_timing, prosody_plan_to_ssml, run_external_alignment, run_praat_analysis,
};
pub use speech::recognizer::{
    SpeechRecognizer, StreamingPartialKind, StreamingRecognition, StreamingRecognizerBackend,
    StreamingSpeechRecognizer,
};
pub use speech::synthetic_plan::{
    ArticulationHints, EnergyPlan, PhoneProvenance, PhoneTiming, PitchPlan, PitchTarget, Stress,
    SyllableRole, SyntheticBreak, SyntheticPhone, SyntheticPlan, SyntheticPlanMetadata,
    SyntheticPlanSource, SyntheticSegment, synthetic_plan_from_prosody_timing,
    synthetic_plan_to_phone_timed_plan, synthetic_plan_to_piper_phoneme_sequence,
    synthetic_plan_to_piper_timing,
};
#[cfg(feature = "asr-whisper")]
pub use speech::whisper::WhisperSpeechRecognizer;
pub use speech::work::{
    AcousticChunk, AcousticPhoneTiming, AcousticStage, ArticulatoryChunk, AudioTime,
    BlockingVocoderRenderer, Boundary, BoundaryHint, BreathPlan, BufferWatermarks,
    CANONICAL_SYNTHETIC_WORK_FLOW, Cadence, ChunkId, CoarseTextChunk, CommitHorizons, Commitment,
    Curve, CurvePoint, LingChunk, LingPhonePlan, LingStage, LingWordPlan, LpcNetChunk, MelChunk,
    MelF0Chunk, PartialProsodyChunk, PhraseShape, PipelineTime, PlaybackStage,
    RealtimeVocoderRenderer, RenderStage, RenderStatus, Renderer, RepairPlan, RepairStrategy,
    RepresentationKind, RepresentationStage, RepresentationStream, SpeechWorkGraph, StageReadiness,
    StageStatus, StreamChunk, StreamStage, SyntheticClock, SyntheticClockKind, SyntheticEvent,
    SyntheticPipelineWatermarks, SyntheticRepresentation, SyntheticStageRuntimePolicy,
    SyntheticWorkGraph, SyntheticWorkStageKind, TextChunk, TextSource, TextStage, TickStage,
    TimedItem, VoiceProfile, WaveChunk, WavePassthroughRenderer, WorkBudget, WorldChunk,
    render_plan_to_representation,
};
pub use speech_timeline::{AudioClipId, SessionId, SyntheticUnitId, TranscriptRevisionId, TurnId};
pub use text_stability::{shared_prefix_len, stable_prefix_len};
pub use time::{
    Clock, ExactTimestamp, FakeClock, NormalizedTimestamp, SessionClock, SystemClock, Timed,
};
pub use vision::{
    AvSyncConfig, EvidenceScore, PhonemeClass, VisualEvidenceStatus, VisualProvenance,
    VisualSpeechClaim, VisualSpeechClaimKind, VisualSpeechFrame, VisualSpeechTrace, VowelShape,
};
#[cfg(target_os = "linux")]
pub use vision::{
    LinuxVideoCaptureConfig, NativeVideoCaptureHandle, ffmpeg_linux_video_args,
    spawn_linux_video_vector_capture,
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
