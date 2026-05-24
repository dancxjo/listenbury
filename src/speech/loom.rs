use anyhow::Result;

/// Shared canonical speech fabric for both heard and generated speech material.
pub const CANONICAL_SPEECH_LOOM_ID: &str = "canonical-speech-loom";

/// Canonical speech artifacts that workers can observe, revise, or emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpeechArtifactKind {
    Text,
    NormalizedText,
    InputAudio,
    AudioTile,
    VadSegment,
    AsrHypothesis,
    TokenHypothesis,
    WordHypothesis,
    PhoneHypothesis,
    Alignment,
    SourceAttribution,
    ObservedProsody,
    LexicalPlan,
    PhoneSequence,
    SyllableStressPlan,
    ProsodyPlan,
    PhoneTimedPlan,
    PhoneIds,
    DiphonePlan,
    AcousticTrack,
    AcousticFeatureTile,
    TemporalSmoothedFeatureTile,
    MelSpectrogramTile,
    F0Track,
    /// Temporary compatibility artifact for legacy combined mel/F0 vocoder inputs.
    MelF0Track,
    WaveformTile,
    AudioFrames,
}

/// Coverage over the canonical speech fabric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpeechSpan {
    Unknown,
    Samples { start: u64, end: u64 },
    Time { start_ms: u64, end_ms: u64 },
    Tokens { start: usize, end: usize },
    Phones { start: usize, end: usize },
    Syllables { start: usize, end: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpeechSourceOrigin {
    SelfVoice,
    OtherVoice,
    Synthesized,
    Echo,
    Recording,
    Overlap,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechProvenance {
    pub origin: SpeechSourceOrigin,
    pub speaker_id: Option<&'static str>,
    pub source_id: Option<&'static str>,
}

impl SpeechProvenance {
    pub fn new(origin: SpeechSourceOrigin) -> Self {
        Self {
            origin,
            speaker_id: None,
            source_id: None,
        }
    }

    pub fn with_speaker(mut self, speaker_id: &'static str) -> Self {
        self.speaker_id = Some(speaker_id);
        self
    }

    pub fn with_source(mut self, source_id: &'static str) -> Self {
        self.source_id = Some(source_id);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeechArtifactContent {
    Pending,
    Opaque(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpeechDependencyKind {
    Input,
    Evidence,
    TimingAnchor,
    DerivedFrom,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechArtifactDependency {
    pub artifact_id: &'static str,
    pub relation: SpeechDependencyKind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeechArtifact {
    pub id: &'static str,
    pub kind: SpeechArtifactKind,
    pub span: SpeechSpan,
    pub content: SpeechArtifactContent,
    pub provenance: SpeechProvenance,
    pub confidence: f32,
    pub revision: u32,
    pub dependencies: Vec<SpeechArtifactDependency>,
}

impl SpeechArtifact {
    pub fn placeholder(
        id: &'static str,
        kind: SpeechArtifactKind,
        span: SpeechSpan,
        provenance: SpeechProvenance,
        content: SpeechArtifactContent,
    ) -> Self {
        Self {
            id,
            kind,
            span,
            content,
            provenance,
            confidence: 1.0,
            revision: 0,
            dependencies: Vec::new(),
        }
    }
}

/// Small reusable worker contract for future composable speech work.
pub trait SpeechWorker<I, O> {
    fn id(&self) -> &'static str;
    fn run(&mut self, input: I) -> Result<O>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpeechWorkKind {
    FusedBackend,
    ExternalProcess,
    Recognizer,
    Planner,
    DiphoneSelector,
    AcousticModel,
    FeatureTransform,
    CompatibilityBridge,
    Vocoder,
    Renderer,
    Attribution,
    Fusion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpeechSpanBehavior {
    Global,
    StreamingTimeAligned,
    TokenAligned,
    PhoneAligned,
    SyllableAligned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpeechOrdering {
    InOrder,
    OutOfOrder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpeechDependencyPolicy {
    ExplicitArtifacts,
    ImplicitContext,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpeechConfidencePolicy {
    EmitsConfidence,
    RequiresConfidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpeechRevisionPolicy {
    Immutable,
    Revisable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommitState {
    Draft,
    Planned,
    AcousticReady,
    Buffered,
    Playing,
    Played,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpeechDeadlineBehavior {
    BestEffort,
    CommitFrontier(CommitState),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechWorkerDescriptor {
    pub id: &'static str,
    pub consumes: Vec<SpeechArtifactKind>,
    pub produces: Vec<SpeechArtifactKind>,
    pub work_kind: SpeechWorkKind,
    pub span_behavior: SpeechSpanBehavior,
    pub ordering: SpeechOrdering,
    pub dependency_policy: SpeechDependencyPolicy,
    pub confidence_policy: SpeechConfidencePolicy,
    pub revision_policy: SpeechRevisionPolicy,
    pub deadline_behavior: SpeechDeadlineBehavior,
    pub commit_state: CommitState,
}

impl SpeechWorkerDescriptor {
    pub fn new(
        id: &'static str,
        consumes: Vec<SpeechArtifactKind>,
        produces: Vec<SpeechArtifactKind>,
        work_kind: SpeechWorkKind,
    ) -> Self {
        Self {
            id,
            consumes,
            produces,
            work_kind,
            span_behavior: SpeechSpanBehavior::StreamingTimeAligned,
            ordering: SpeechOrdering::OutOfOrder,
            dependency_policy: SpeechDependencyPolicy::ExplicitArtifacts,
            confidence_policy: SpeechConfidencePolicy::EmitsConfidence,
            revision_policy: SpeechRevisionPolicy::Revisable,
            deadline_behavior: SpeechDeadlineBehavior::BestEffort,
            commit_state: CommitState::Draft,
        }
    }

    pub fn with_span_behavior(mut self, span_behavior: SpeechSpanBehavior) -> Self {
        self.span_behavior = span_behavior;
        self
    }

    pub fn with_ordering(mut self, ordering: SpeechOrdering) -> Self {
        self.ordering = ordering;
        self
    }

    pub fn with_dependency_policy(mut self, dependency_policy: SpeechDependencyPolicy) -> Self {
        self.dependency_policy = dependency_policy;
        self
    }

    pub fn with_confidence_policy(mut self, confidence_policy: SpeechConfidencePolicy) -> Self {
        self.confidence_policy = confidence_policy;
        self
    }

    pub fn with_revision_policy(mut self, revision_policy: SpeechRevisionPolicy) -> Self {
        self.revision_policy = revision_policy;
        self
    }

    pub fn with_deadline_behavior(mut self, deadline_behavior: SpeechDeadlineBehavior) -> Self {
        self.deadline_behavior = deadline_behavior;
        self
    }

    pub fn with_commit_state(mut self, commit_state: CommitState) -> Self {
        self.commit_state = commit_state;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechLoom {
    pub id: &'static str,
    pub projection: &'static str,
    pub workers: Vec<SpeechWorkerDescriptor>,
}

impl SpeechLoom {
    pub fn new(projection: &'static str, workers: Vec<SpeechWorkerDescriptor>) -> Self {
        Self {
            id: CANONICAL_SPEECH_LOOM_ID,
            projection,
            workers,
        }
    }
}

/// Compatibility view over the current `say` backend graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentBackendGraphView {
    pub id: &'static str,
    pub workers: Vec<SpeechWorkerDescriptor>,
    pub fused: bool,
}

impl CurrentBackendGraphView {
    pub fn new(id: &'static str, workers: Vec<SpeechWorkerDescriptor>, fused: bool) -> Self {
        Self { id, workers, fused }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CurrentSayBackendKind {
    PiperCompat,
    PiperProcess,
    Klatt,
    MbrolaDiphone,
    SourceFilterHifigan,
}

impl CurrentSayBackendKind {
    pub fn loom(self) -> SpeechLoom {
        SpeechLoom::new(self.projection(), self.workers())
    }

    pub fn current_backend_graph(self) -> CurrentBackendGraphView {
        CurrentBackendGraphView::new(self.route_id(), self.workers(), self.fused())
    }

    fn projection(self) -> &'static str {
        match self {
            Self::PiperCompat => "current-backend/piper-compat",
            Self::PiperProcess => "current-backend/piper-process",
            Self::Klatt => "current-backend/klatt",
            Self::MbrolaDiphone => "current-backend/mbrola-diphone",
            Self::SourceFilterHifigan => "current-backend/source-filter-hifigan",
        }
    }

    fn route_id(self) -> &'static str {
        match self {
            Self::PiperCompat => "piper-compat",
            Self::PiperProcess => "piper-process",
            Self::Klatt => "klatt",
            Self::MbrolaDiphone => "mbrola-diphone",
            Self::SourceFilterHifigan => "source-filter-hifigan",
        }
    }

    fn fused(self) -> bool {
        matches!(self, Self::PiperCompat | Self::PiperProcess)
    }

    fn workers(self) -> Vec<SpeechWorkerDescriptor> {
        match self {
            Self::PiperCompat => vec![
                SpeechWorkerDescriptor::new(
                    "piper-compatible-onnx",
                    vec![SpeechArtifactKind::Text, SpeechArtifactKind::PhoneIds],
                    vec![SpeechArtifactKind::AudioFrames],
                    SpeechWorkKind::FusedBackend,
                )
                .with_span_behavior(SpeechSpanBehavior::Global)
                .with_ordering(SpeechOrdering::InOrder)
                .with_revision_policy(SpeechRevisionPolicy::Immutable)
                .with_deadline_behavior(SpeechDeadlineBehavior::CommitFrontier(
                    CommitState::Buffered,
                ))
                .with_commit_state(CommitState::Buffered),
            ],
            Self::PiperProcess => vec![
                SpeechWorkerDescriptor::new(
                    "piper-process-backend",
                    vec![SpeechArtifactKind::Text],
                    vec![SpeechArtifactKind::AudioFrames],
                    SpeechWorkKind::ExternalProcess,
                )
                .with_span_behavior(SpeechSpanBehavior::Global)
                .with_ordering(SpeechOrdering::InOrder)
                .with_dependency_policy(SpeechDependencyPolicy::ImplicitContext)
                .with_revision_policy(SpeechRevisionPolicy::Immutable)
                .with_deadline_behavior(SpeechDeadlineBehavior::CommitFrontier(
                    CommitState::Buffered,
                ))
                .with_commit_state(CommitState::Buffered),
            ],
            Self::Klatt => vec![
                SpeechWorkerDescriptor::new(
                    "klatt-formant-renderer",
                    vec![SpeechArtifactKind::PhoneTimedPlan],
                    vec![SpeechArtifactKind::AudioFrames],
                    SpeechWorkKind::Renderer,
                )
                .with_span_behavior(SpeechSpanBehavior::PhoneAligned)
                .with_ordering(SpeechOrdering::InOrder)
                .with_deadline_behavior(SpeechDeadlineBehavior::CommitFrontier(
                    CommitState::Buffered,
                ))
                .with_commit_state(CommitState::Buffered),
            ],
            Self::MbrolaDiphone => vec![
                SpeechWorkerDescriptor::new(
                    "mbrola-diphone-selection",
                    vec![SpeechArtifactKind::PhoneTimedPlan],
                    vec![SpeechArtifactKind::DiphonePlan],
                    SpeechWorkKind::DiphoneSelector,
                )
                .with_span_behavior(SpeechSpanBehavior::PhoneAligned)
                .with_deadline_behavior(SpeechDeadlineBehavior::CommitFrontier(
                    CommitState::Planned,
                ))
                .with_commit_state(CommitState::Planned),
                SpeechWorkerDescriptor::new(
                    "mbrola-diphone-renderer",
                    vec![SpeechArtifactKind::DiphonePlan],
                    vec![SpeechArtifactKind::AudioFrames],
                    SpeechWorkKind::Renderer,
                )
                .with_span_behavior(SpeechSpanBehavior::PhoneAligned)
                .with_ordering(SpeechOrdering::InOrder)
                .with_deadline_behavior(SpeechDeadlineBehavior::CommitFrontier(
                    CommitState::Buffered,
                ))
                .with_commit_state(CommitState::Buffered),
            ],
            Self::SourceFilterHifigan => vec![
                SpeechWorkerDescriptor::new(
                    "source-filter-acoustic-generator",
                    vec![SpeechArtifactKind::PhoneTimedPlan],
                    vec![
                        SpeechArtifactKind::AcousticFeatureTile,
                        SpeechArtifactKind::F0Track,
                    ],
                    SpeechWorkKind::AcousticModel,
                )
                .with_span_behavior(SpeechSpanBehavior::PhoneAligned)
                .with_deadline_behavior(SpeechDeadlineBehavior::CommitFrontier(
                    CommitState::AcousticReady,
                ))
                .with_commit_state(CommitState::AcousticReady),
                SpeechWorkerDescriptor::new(
                    "source-filter-temporal-smoother",
                    vec![SpeechArtifactKind::AcousticFeatureTile],
                    vec![SpeechArtifactKind::TemporalSmoothedFeatureTile],
                    SpeechWorkKind::FeatureTransform,
                )
                .with_span_behavior(SpeechSpanBehavior::PhoneAligned)
                .with_commit_state(CommitState::AcousticReady),
                SpeechWorkerDescriptor::new(
                    "source-filter-mel-compat-bridge",
                    vec![
                        SpeechArtifactKind::TemporalSmoothedFeatureTile,
                        SpeechArtifactKind::F0Track,
                    ],
                    vec![
                        SpeechArtifactKind::MelSpectrogramTile,
                        SpeechArtifactKind::MelF0Track,
                    ],
                    SpeechWorkKind::CompatibilityBridge,
                )
                .with_span_behavior(SpeechSpanBehavior::StreamingTimeAligned)
                .with_commit_state(CommitState::AcousticReady),
                SpeechWorkerDescriptor::new(
                    "hifigan-vocoder",
                    vec![
                        SpeechArtifactKind::MelSpectrogramTile,
                        SpeechArtifactKind::F0Track,
                        SpeechArtifactKind::MelF0Track,
                    ],
                    vec![
                        SpeechArtifactKind::WaveformTile,
                        SpeechArtifactKind::AudioFrames,
                    ],
                    SpeechWorkKind::Vocoder,
                )
                .with_span_behavior(SpeechSpanBehavior::StreamingTimeAligned)
                .with_ordering(SpeechOrdering::InOrder)
                .with_deadline_behavior(SpeechDeadlineBehavior::CommitFrontier(
                    CommitState::Buffered,
                ))
                .with_commit_state(CommitState::Buffered),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CANONICAL_SPEECH_LOOM_ID, CommitState, CurrentSayBackendKind, SpeechArtifact,
        SpeechArtifactContent, SpeechArtifactKind, SpeechProvenance, SpeechSourceOrigin,
        SpeechSpan, SpeechWorkKind,
    };

    #[test]
    fn phone_hypotheses_keep_attribution_in_metadata() {
        let heard = SpeechArtifact::placeholder(
            "heard-phone",
            SpeechArtifactKind::PhoneHypothesis,
            SpeechSpan::Phones { start: 0, end: 2 },
            SpeechProvenance::new(SpeechSourceOrigin::OtherVoice).with_speaker("guest-1"),
            SpeechArtifactContent::Opaque("heard"),
        );
        let generated = SpeechArtifact::placeholder(
            "generated-phone",
            SpeechArtifactKind::PhoneHypothesis,
            SpeechSpan::Phones { start: 0, end: 2 },
            SpeechProvenance::new(SpeechSourceOrigin::Synthesized).with_source("tts-buffer"),
            SpeechArtifactContent::Opaque("generated"),
        );

        assert_eq!(heard.kind, generated.kind);
        assert_ne!(heard.provenance.origin, generated.provenance.origin);
    }

    #[test]
    fn piper_compat_graph_is_a_view_over_the_canonical_loom() {
        let loom = CurrentSayBackendKind::PiperCompat.loom();
        let graph = CurrentSayBackendKind::PiperCompat.current_backend_graph();

        assert_eq!(loom.id, CANONICAL_SPEECH_LOOM_ID);
        assert_eq!(loom.projection, "current-backend/piper-compat");
        assert_eq!(graph.id, "piper-compat");
        assert!(graph.fused);
        assert_eq!(graph.workers.len(), 1);
        let worker = &graph.workers[0];
        assert_eq!(worker.id, "piper-compatible-onnx");
        assert_eq!(
            worker.consumes,
            vec![SpeechArtifactKind::Text, SpeechArtifactKind::PhoneIds]
        );
        assert_eq!(worker.produces, vec![SpeechArtifactKind::AudioFrames]);
        assert_eq!(worker.work_kind, SpeechWorkKind::FusedBackend);
        assert_eq!(worker.commit_state, CommitState::Buffered);
    }

    #[test]
    fn klatt_graph_reports_phone_timed_renderer() {
        let loom = CurrentSayBackendKind::Klatt.loom();
        let graph = CurrentSayBackendKind::Klatt.current_backend_graph();

        assert_eq!(loom.id, CANONICAL_SPEECH_LOOM_ID);
        assert_eq!(loom.projection, "current-backend/klatt");
        assert_eq!(graph.id, "klatt");
        assert!(!graph.fused);
        assert_eq!(graph.workers.len(), 1);
        let worker = &graph.workers[0];
        assert_eq!(worker.id, "klatt-formant-renderer");
        assert_eq!(worker.consumes, vec![SpeechArtifactKind::PhoneTimedPlan]);
        assert_eq!(worker.produces, vec![SpeechArtifactKind::AudioFrames]);
        assert_eq!(worker.work_kind, SpeechWorkKind::Renderer);
    }

    #[test]
    fn mbrola_graph_reports_selection_then_render() {
        let loom = CurrentSayBackendKind::MbrolaDiphone.loom();
        let graph = CurrentSayBackendKind::MbrolaDiphone.current_backend_graph();

        assert_eq!(loom.id, CANONICAL_SPEECH_LOOM_ID);
        assert_eq!(loom.projection, "current-backend/mbrola-diphone");
        assert_eq!(graph.id, "mbrola-diphone");
        assert!(!graph.fused);
        assert_eq!(graph.workers.len(), 2);
        assert_eq!(graph.workers[0].id, "mbrola-diphone-selection");
        assert_eq!(
            graph.workers[0].consumes,
            vec![SpeechArtifactKind::PhoneTimedPlan]
        );
        assert_eq!(
            graph.workers[0].produces,
            vec![SpeechArtifactKind::DiphonePlan]
        );
        assert_eq!(graph.workers[0].work_kind, SpeechWorkKind::DiphoneSelector);
        assert_eq!(graph.workers[0].commit_state, CommitState::Planned);
        assert_eq!(graph.workers[1].id, "mbrola-diphone-renderer");
        assert_eq!(
            graph.workers[1].consumes,
            vec![SpeechArtifactKind::DiphonePlan]
        );
        assert_eq!(
            graph.workers[1].produces,
            vec![SpeechArtifactKind::AudioFrames]
        );
        assert_eq!(graph.workers[1].work_kind, SpeechWorkKind::Renderer);
        assert_eq!(graph.workers[1].commit_state, CommitState::Buffered);
    }

    #[test]
    fn source_filter_hifigan_graph_reports_acoustic_then_vocoder() {
        let loom = CurrentSayBackendKind::SourceFilterHifigan.loom();
        let graph = CurrentSayBackendKind::SourceFilterHifigan.current_backend_graph();

        assert_eq!(loom.id, CANONICAL_SPEECH_LOOM_ID);
        assert_eq!(loom.projection, "current-backend/source-filter-hifigan");
        assert_eq!(graph.id, "source-filter-hifigan");
        assert!(!graph.fused);
        assert_eq!(graph.workers.len(), 4);
        assert_eq!(graph.workers[0].id, "source-filter-acoustic-generator");
        assert_eq!(
            graph.workers[0].consumes,
            vec![SpeechArtifactKind::PhoneTimedPlan]
        );
        assert_eq!(
            graph.workers[0].produces,
            vec![
                SpeechArtifactKind::AcousticFeatureTile,
                SpeechArtifactKind::F0Track
            ]
        );
        assert_eq!(graph.workers[0].work_kind, SpeechWorkKind::AcousticModel);
        assert_eq!(graph.workers[0].commit_state, CommitState::AcousticReady);
        assert_eq!(graph.workers[1].id, "source-filter-temporal-smoother");
        assert_eq!(
            graph.workers[1].consumes,
            vec![SpeechArtifactKind::AcousticFeatureTile]
        );
        assert_eq!(
            graph.workers[1].produces,
            vec![SpeechArtifactKind::TemporalSmoothedFeatureTile]
        );
        assert_eq!(graph.workers[1].work_kind, SpeechWorkKind::FeatureTransform);
        assert_eq!(graph.workers[1].commit_state, CommitState::AcousticReady);
        assert_eq!(graph.workers[2].id, "source-filter-mel-compat-bridge");
        assert_eq!(
            graph.workers[2].consumes,
            vec![
                SpeechArtifactKind::TemporalSmoothedFeatureTile,
                SpeechArtifactKind::F0Track
            ]
        );
        assert_eq!(
            graph.workers[2].produces,
            vec![
                SpeechArtifactKind::MelSpectrogramTile,
                SpeechArtifactKind::MelF0Track
            ]
        );
        assert_eq!(
            graph.workers[2].work_kind,
            SpeechWorkKind::CompatibilityBridge
        );
        assert_eq!(graph.workers[2].commit_state, CommitState::AcousticReady);
        assert_eq!(graph.workers[3].id, "hifigan-vocoder");
        assert_eq!(
            graph.workers[3].consumes,
            vec![
                SpeechArtifactKind::MelSpectrogramTile,
                SpeechArtifactKind::F0Track,
                SpeechArtifactKind::MelF0Track
            ]
        );
        assert_eq!(
            graph.workers[3].produces,
            vec![
                SpeechArtifactKind::WaveformTile,
                SpeechArtifactKind::AudioFrames
            ]
        );
        assert_eq!(graph.workers[3].work_kind, SpeechWorkKind::Vocoder);
        assert_eq!(graph.workers[3].commit_state, CommitState::Buffered);
    }
}
