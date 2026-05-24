use anyhow::Result;
use std::collections::HashMap;

use super::breath_asr::BreathAudioSegment;
use super::synthetic_plan::SyntheticPlan;
use crate::word::TranscriptWord;

/// Shared canonical speech fabric for both heard and generated synthesis material.
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
    Phones,
    Syllables,
    Stress,
    Prominence,
    PhraseProsody,
    SyllableProsody,
    PhoneTiming,
    PhoneFeatures,
    MelTile,
    FormantTile,
    DiphoneTile,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimeSpan {
    pub start_ms: u64,
    pub end_ms: u64,
}

impl TimeSpan {
    pub fn new(start_ms: u64, end_ms: u64) -> Option<Self> {
        (end_ms >= start_ms).then_some(Self { start_ms, end_ms })
    }
}

fn seconds_to_ms(seconds: f64) -> u64 {
    (seconds.max(0.0) * 1000.0).round() as u64
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SampleSpan {
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenSpan {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WordSpan {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhoneSpan {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SyllableSpan {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BreathGroupSpan {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProsodicPhraseSpan {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AcousticFrameSpan {
    pub start: usize,
    pub end: usize,
}

/// Coverage over the canonical speech fabric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpeechSpan {
    Unknown,
    Samples(SampleSpan),
    Time(TimeSpan),
    Tokens(TokenSpan),
    Words(WordSpan),
    Phones(PhoneSpan),
    Syllables(SyllableSpan),
    BreathGroups(BreathGroupSpan),
    ProsodicPhrases(ProsodicPhraseSpan),
    AcousticFrames(AcousticFrameSpan),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttributionSourceKind {
    LiveVoice,
    SynthesizedWaveform,
    Echo,
    Recording,
    Overlap,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfidenceBool {
    pub value: Option<bool>,
    pub confidence: f32,
}

impl ConfidenceBool {
    pub fn certain(value: bool, confidence: f32) -> Self {
        Self {
            value: Some(value),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    pub fn uncertain(confidence: f32) -> Self {
        Self {
            value: None,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerHypothesis {
    pub speaker_id: String,
    pub confidence: f32,
}

impl SpeakerHypothesis {
    pub fn new(speaker_id: impl Into<String>, confidence: f32) -> Self {
        Self {
            speaker_id: speaker_id.into(),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AudioChannel(pub String);

#[derive(Debug, Clone, PartialEq)]
pub struct SpeechAttribution {
    pub speaker: Option<SpeakerHypothesis>,
    pub is_self: ConfidenceBool,
    pub channel: Option<AudioChannel>,
    pub source: AttributionSourceKind,
    pub evidence: Vec<String>,
}

impl SpeechAttribution {
    pub fn new(source: AttributionSourceKind) -> Self {
        Self {
            speaker: None,
            is_self: ConfidenceBool::uncertain(0.0),
            channel: None,
            source,
            evidence: Vec::new(),
        }
    }

    pub fn with_speaker(mut self, speaker_id: impl Into<String>, confidence: f32) -> Self {
        self.speaker = Some(SpeakerHypothesis::new(speaker_id, confidence));
        self
    }

    pub fn with_is_self(mut self, value: Option<bool>, confidence: f32) -> Self {
        self.is_self = match value {
            Some(value) => ConfidenceBool::certain(value, confidence),
            None => ConfidenceBool::uncertain(confidence),
        };
        self
    }

    pub fn with_channel(mut self, channel: impl Into<String>) -> Self {
        self.channel = Some(AudioChannel(channel.into()));
        self
    }

    pub fn with_evidence(mut self, artifact_id: impl Into<String>) -> Self {
        self.evidence.push(artifact_id.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechProvenance {
    pub origin: SpeechSourceOrigin,
    pub speaker_id: Option<String>,
    pub source_id: Option<String>,
}

impl SpeechProvenance {
    pub fn new(origin: SpeechSourceOrigin) -> Self {
        Self {
            origin,
            speaker_id: None,
            source_id: None,
        }
    }

    pub fn with_speaker(mut self, speaker_id: impl Into<String>) -> Self {
        self.speaker_id = Some(speaker_id.into());
        self
    }

    pub fn with_source(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = Some(source_id.into());
        self
    }

    pub fn canonical_attribution(&self) -> SpeechAttribution {
        let (source, is_self) = match self.origin {
            SpeechSourceOrigin::SelfVoice => (
                AttributionSourceKind::LiveVoice,
                ConfidenceBool::certain(true, 1.0),
            ),
            SpeechSourceOrigin::OtherVoice => (
                AttributionSourceKind::LiveVoice,
                ConfidenceBool::certain(false, 1.0),
            ),
            SpeechSourceOrigin::Synthesized => (
                AttributionSourceKind::SynthesizedWaveform,
                ConfidenceBool::uncertain(0.0),
            ),
            SpeechSourceOrigin::Echo => (
                AttributionSourceKind::Echo,
                ConfidenceBool::certain(true, 0.7),
            ),
            SpeechSourceOrigin::Recording => (
                AttributionSourceKind::Recording,
                ConfidenceBool::uncertain(0.5),
            ),
            SpeechSourceOrigin::Overlap => (
                AttributionSourceKind::Overlap,
                ConfidenceBool::uncertain(0.5),
            ),
            SpeechSourceOrigin::Unknown => (
                AttributionSourceKind::Unknown,
                ConfidenceBool::uncertain(0.0),
            ),
        };
        let mut attribution = SpeechAttribution::new(source);
        attribution.is_self = is_self;
        if let Some(speaker_id) = &self.speaker_id {
            attribution.speaker = Some(SpeakerHypothesis::new(speaker_id.clone(), 1.0));
        }
        if let Some(source_id) = &self.source_id {
            attribution.evidence.push(source_id.clone());
        }
        attribution
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SpeechArtifactContent {
    Pending,
    Opaque(&'static str),
    Text(String),
    PhoneSymbol(String),
    Scalar(f32),
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
    pub artifact_id: String,
    pub relation: SpeechDependencyKind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeechArtifact {
    pub id: String,
    pub kind: SpeechArtifactKind,
    pub span: SpeechSpan,
    pub content: SpeechArtifactContent,
    pub provenance: SpeechProvenance,
    pub attributions: Vec<SpeechAttribution>,
    pub confidence: f32,
    pub revision: u32,
    pub dependencies: Vec<SpeechArtifactDependency>,
}

impl SpeechArtifact {
    pub fn placeholder(
        id: impl Into<String>,
        kind: SpeechArtifactKind,
        span: SpeechSpan,
        provenance: SpeechProvenance,
        content: SpeechArtifactContent,
    ) -> Self {
        let attribution = provenance.canonical_attribution();
        Self {
            id: id.into(),
            kind,
            span,
            content,
            provenance,
            attributions: vec![attribution],
            confidence: 1.0,
            revision: 0,
            dependencies: Vec::new(),
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn with_revision(mut self, revision: u32) -> Self {
        self.revision = revision;
        self
    }

    pub fn with_dependency(
        mut self,
        artifact_id: impl Into<String>,
        relation: SpeechDependencyKind,
    ) -> Self {
        self.dependencies.push(SpeechArtifactDependency {
            artifact_id: artifact_id.into(),
            relation,
        });
        self
    }

    pub fn with_attribution(mut self, attribution: SpeechAttribution) -> Self {
        self.attributions.push(attribution);
        self
    }

    pub fn with_attributions(mut self, attributions: Vec<SpeechAttribution>) -> Self {
        self.attributions = attributions;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeechDocument {
    pub id: String,
    pub artifacts: HashMap<String, SpeechArtifact>,
    pub commit_frontier_ms: Option<u64>,
}

impl SpeechDocument {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            artifacts: HashMap::new(),
            commit_frontier_ms: None,
        }
    }

    pub fn upsert(&mut self, artifact: SpeechArtifact) {
        self.artifacts.insert(artifact.id.clone(), artifact);
    }

    pub fn mark_commit_frontier_ms(&mut self, frontier_ms: u64) {
        self.commit_frontier_ms = Some(frontier_ms);
    }

    pub fn replace_uncommitted(
        &mut self,
        old_artifact_id: &str,
        replacement: SpeechArtifact,
    ) -> Result<()> {
        let frontier_ms = self.commit_frontier_ms.unwrap_or_default();
        let old = self
            .artifacts
            .get(old_artifact_id)
            .ok_or_else(|| anyhow::anyhow!("artifact `{old_artifact_id}` does not exist"))?;
        if let SpeechSpan::Time(span) = old.span {
            anyhow::ensure!(
                span.start_ms >= frontier_ms,
                "artifact `{old_artifact_id}` is already committed at frontier {frontier_ms}ms"
            );
        }
        self.artifacts.remove(old_artifact_id);
        self.upsert(replacement);
        Ok(())
    }
}

pub fn asr_word_hypothesis_artifacts(
    words: &[TranscriptWord],
    source_id: &str,
) -> Vec<SpeechArtifact> {
    words
        .iter()
        .enumerate()
        .map(|(word_index, word)| {
            let span = match (word.start_ms, word.end_ms) {
                (Some(start_ms), Some(end_ms)) if end_ms >= start_ms => {
                    SpeechSpan::Time(TimeSpan { start_ms, end_ms })
                }
                _ => SpeechSpan::Words(WordSpan {
                    start: word_index,
                    end: word_index + 1,
                }),
            };
            SpeechArtifact::placeholder(
                format!("asr-word-{word_index}"),
                SpeechArtifactKind::WordHypothesis,
                span,
                SpeechProvenance::new(SpeechSourceOrigin::OtherVoice).with_source(source_id),
                SpeechArtifactContent::Text(word.text.clone()),
            )
            .with_confidence(word.confidence.unwrap_or(0.0))
        })
        .collect()
}

pub fn asr_vad_segment_artifacts(
    segments: &[BreathAudioSegment],
    source_id: &str,
) -> Vec<SpeechArtifact> {
    segments
        .iter()
        .enumerate()
        .map(|(segment_index, segment)| {
            SpeechArtifact::placeholder(
                format!("asr-vad-{segment_index}"),
                SpeechArtifactKind::VadSegment,
                SpeechSpan::Time(TimeSpan {
                    start_ms: segment.start_ms,
                    end_ms: segment.end_ms,
                }),
                SpeechProvenance::new(SpeechSourceOrigin::Recording).with_source(source_id),
                SpeechArtifactContent::Pending,
            )
        })
        .collect()
}

pub fn tts_intended_phone_artifacts(plan: &SyntheticPlan, source_id: &str) -> Vec<SpeechArtifact> {
    let mut phone_index = 0usize;
    let mut artifacts = Vec::new();
    for (word_index, segment) in plan.segments.iter().enumerate() {
        for phone in &segment.phones {
            let timing = TimeSpan {
                start_ms: seconds_to_ms(phone.timing.t0),
                end_ms: seconds_to_ms(phone.timing.t1.max(phone.timing.t0)),
            };
            let id = format!("tts-phone-{phone_index}");
            artifacts.push(
                SpeechArtifact::placeholder(
                    id.clone(),
                    SpeechArtifactKind::PhoneHypothesis,
                    SpeechSpan::Phones(PhoneSpan {
                        start: phone_index,
                        end: phone_index + 1,
                    }),
                    SpeechProvenance::new(SpeechSourceOrigin::Synthesized).with_source(source_id),
                    SpeechArtifactContent::PhoneSymbol(phone.symbol.clone()),
                )
                .with_dependency(
                    format!("tts-word-{word_index}"),
                    SpeechDependencyKind::DerivedFrom,
                ),
            );
            artifacts.push(
                SpeechArtifact::placeholder(
                    format!("{id}-timing"),
                    SpeechArtifactKind::PhoneTiming,
                    SpeechSpan::Time(timing),
                    SpeechProvenance::new(SpeechSourceOrigin::Synthesized).with_source(source_id),
                    SpeechArtifactContent::PhoneSymbol(phone.symbol.clone()),
                )
                .with_dependency(id, SpeechDependencyKind::TimingAnchor),
            );
            phone_index += 1;
        }
    }
    artifacts
}

pub fn tts_alignment_from_asr_word_hypotheses(
    asr_words: &[SpeechArtifact],
    source_id: &str,
) -> Option<SpeechArtifact> {
    let mut range: Option<TimeSpan> = None;
    let mut dependencies = Vec::new();
    for word in asr_words {
        if let SpeechSpan::Time(span) = word.span {
            range = Some(match range {
                Some(current) => TimeSpan {
                    start_ms: current.start_ms.min(span.start_ms),
                    end_ms: current.end_ms.max(span.end_ms),
                },
                None => span,
            });
            dependencies.push(word.id.clone());
        }
    }
    range.map(|covered| {
        let mut artifact = SpeechArtifact::placeholder(
            "tts-reused-asr-alignment",
            SpeechArtifactKind::Alignment,
            SpeechSpan::Time(covered),
            SpeechProvenance::new(SpeechSourceOrigin::Synthesized).with_source(source_id),
            SpeechArtifactContent::Opaque("reused-asr-word-alignment"),
        );
        artifact.dependencies = dependencies
            .into_iter()
            .map(|artifact_id| SpeechArtifactDependency {
                artifact_id,
                relation: SpeechDependencyKind::Evidence,
            })
            .collect();
        artifact
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhoneComparison {
    pub phone_index: usize,
    pub intended: String,
    pub heard: String,
}

pub fn compare_intended_vs_heard_phones(document: &SpeechDocument) -> Vec<PhoneComparison> {
    let mut intended = HashMap::<usize, String>::new();
    let mut heard = HashMap::<usize, String>::new();
    for artifact in document.artifacts.values() {
        if artifact.kind != SpeechArtifactKind::PhoneHypothesis {
            continue;
        }
        let SpeechSpan::Phones(span) = artifact.span else {
            continue;
        };
        let SpeechArtifactContent::PhoneSymbol(symbol) = &artifact.content else {
            continue;
        };
        let classify_from_provenance = || match artifact.provenance.origin {
            SpeechSourceOrigin::Synthesized => Some("intended"),
            SpeechSourceOrigin::SelfVoice
            | SpeechSourceOrigin::OtherVoice
            | SpeechSourceOrigin::Echo
            | SpeechSourceOrigin::Recording
            | SpeechSourceOrigin::Overlap => Some("heard"),
            SpeechSourceOrigin::Unknown => None,
        };

        let classify_from_attribution =
            artifact
                .attributions
                .iter()
                .find_map(|attribution| match attribution.source {
                    AttributionSourceKind::SynthesizedWaveform => Some("intended"),
                    AttributionSourceKind::LiveVoice
                    | AttributionSourceKind::Echo
                    | AttributionSourceKind::Recording
                    | AttributionSourceKind::Overlap => Some("heard"),
                    AttributionSourceKind::Unknown => None,
                });

        match classify_from_attribution.or_else(classify_from_provenance) {
            Some("intended") => {
                intended.insert(span.start, symbol.clone());
            }
            Some("heard") => {
                heard.insert(span.start, symbol.clone());
            }
            _ => {}
        }
    }

    let mut mismatches = Vec::new();
    for (phone_index, intended_phone) in intended {
        if let Some(heard_phone) = heard.get(&phone_index) {
            if heard_phone != &intended_phone {
                mismatches.push(PhoneComparison {
                    phone_index,
                    intended: intended_phone,
                    heard: heard_phone.clone(),
                });
            }
        }
    }
    mismatches.sort_by_key(|m| m.phone_index);
    mismatches
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
        AttributionSourceKind, CANONICAL_SPEECH_LOOM_ID, CommitState, CurrentSayBackendKind,
        PhoneComparison, PhoneSpan, SpeechArtifact, SpeechArtifactContent, SpeechArtifactKind,
        SpeechAttribution, SpeechDocument, SpeechProvenance, SpeechSourceOrigin, SpeechSpan,
        SpeechWorkKind, SyllableSpan, TimeSpan, asr_word_hypothesis_artifacts,
        compare_intended_vs_heard_phones, tts_alignment_from_asr_word_hypotheses,
    };
    use crate::word::TranscriptWord;

    #[test]
    fn phone_hypotheses_keep_attribution_in_metadata() {
        let heard = SpeechArtifact::placeholder(
            "heard-phone",
            SpeechArtifactKind::PhoneHypothesis,
            SpeechSpan::Phones(PhoneSpan { start: 0, end: 2 }),
            SpeechProvenance::new(SpeechSourceOrigin::OtherVoice).with_speaker("guest-1"),
            SpeechArtifactContent::Opaque("heard"),
        );
        let generated = SpeechArtifact::placeholder(
            "generated-phone",
            SpeechArtifactKind::PhoneHypothesis,
            SpeechSpan::Phones(PhoneSpan { start: 0, end: 2 }),
            SpeechProvenance::new(SpeechSourceOrigin::Synthesized).with_source("tts-buffer"),
            SpeechArtifactContent::Opaque("generated"),
        );

        assert_eq!(heard.kind, generated.kind);
        assert_ne!(heard.provenance.origin, generated.provenance.origin);
        assert_eq!(
            heard.attributions[0].source,
            AttributionSourceKind::LiveVoice
        );
        assert_eq!(
            generated.attributions[0].source,
            AttributionSourceKind::SynthesizedWaveform
        );
    }

    #[test]
    fn canonical_artifact_supports_competing_attributions_for_same_span() {
        let canonical = SpeechArtifact::placeholder(
            "phone-span-0",
            SpeechArtifactKind::PhoneHypothesis,
            SpeechSpan::Phones(PhoneSpan { start: 0, end: 1 }),
            SpeechProvenance::new(SpeechSourceOrigin::Unknown),
            SpeechArtifactContent::PhoneSymbol("AH".to_string()),
        )
        .with_attributions(vec![
            SpeechAttribution::new(AttributionSourceKind::LiveVoice)
                .with_speaker("speaker-a", 0.71)
                .with_is_self(None, 0.45)
                .with_evidence("asr-hyp-a"),
            SpeechAttribution::new(AttributionSourceKind::LiveVoice)
                .with_speaker("speaker-b", 0.63)
                .with_is_self(Some(false), 0.67)
                .with_evidence("asr-hyp-b"),
        ]);

        assert_eq!(canonical.kind, SpeechArtifactKind::PhoneHypothesis);
        assert_eq!(canonical.attributions.len(), 2);
        assert_eq!(
            canonical.attributions[0]
                .speaker
                .as_ref()
                .map(|speaker| speaker.speaker_id.as_str()),
            Some("speaker-a")
        );
        assert_eq!(
            canonical.attributions[0]
                .speaker
                .as_ref()
                .map(|speaker| speaker.confidence),
            Some(0.71)
        );
        assert_eq!(canonical.attributions[0].is_self.value, None);
        assert_eq!(canonical.attributions[0].is_self.confidence, 0.45);
        assert_eq!(
            canonical.attributions[1]
                .speaker
                .as_ref()
                .map(|speaker| speaker.speaker_id.as_str()),
            Some("speaker-b")
        );
        assert_eq!(
            canonical.attributions[1]
                .speaker
                .as_ref()
                .map(|speaker| speaker.confidence),
            Some(0.63)
        );
        assert_eq!(canonical.attributions[1].is_self.value, Some(false));
    }

    #[test]
    fn self_and_other_heard_audio_share_canonical_artifact_kinds() {
        let self_artifacts = vec![
            SpeechArtifact::placeholder(
                "self-phone",
                SpeechArtifactKind::PhoneHypothesis,
                SpeechSpan::Phones(PhoneSpan { start: 0, end: 1 }),
                SpeechProvenance::new(SpeechSourceOrigin::SelfVoice),
                SpeechArtifactContent::PhoneSymbol("HH".to_string()),
            ),
            SpeechArtifact::placeholder(
                "self-syllable",
                SpeechArtifactKind::Syllables,
                SpeechSpan::Syllables(SyllableSpan { start: 0, end: 1 }),
                SpeechProvenance::new(SpeechSourceOrigin::SelfVoice),
                SpeechArtifactContent::Opaque("self-syllable"),
            ),
            SpeechArtifact::placeholder(
                "self-prosody",
                SpeechArtifactKind::SyllableProsody,
                SpeechSpan::Syllables(SyllableSpan { start: 0, end: 1 }),
                SpeechProvenance::new(SpeechSourceOrigin::SelfVoice),
                SpeechArtifactContent::Opaque("self-prosody"),
            ),
        ];
        let other_artifacts = vec![
            SpeechArtifact::placeholder(
                "other-phone",
                SpeechArtifactKind::PhoneHypothesis,
                SpeechSpan::Phones(PhoneSpan { start: 0, end: 1 }),
                SpeechProvenance::new(SpeechSourceOrigin::OtherVoice),
                SpeechArtifactContent::PhoneSymbol("HH".to_string()),
            ),
            SpeechArtifact::placeholder(
                "other-syllable",
                SpeechArtifactKind::Syllables,
                SpeechSpan::Syllables(SyllableSpan { start: 0, end: 1 }),
                SpeechProvenance::new(SpeechSourceOrigin::OtherVoice),
                SpeechArtifactContent::Opaque("other-syllable"),
            ),
            SpeechArtifact::placeholder(
                "other-prosody",
                SpeechArtifactKind::SyllableProsody,
                SpeechSpan::Syllables(SyllableSpan { start: 0, end: 1 }),
                SpeechProvenance::new(SpeechSourceOrigin::OtherVoice),
                SpeechArtifactContent::Opaque("other-prosody"),
            ),
        ];

        for (self_artifact, other_artifact) in self_artifacts.iter().zip(other_artifacts.iter()) {
            assert_eq!(self_artifact.kind, other_artifact.kind);
            assert_eq!(self_artifact.span, other_artifact.span);
            assert_eq!(
                self_artifact.attributions[0].source,
                other_artifact.attributions[0].source
            );
            assert_ne!(
                self_artifact.attributions[0].is_self.value,
                other_artifact.attributions[0].is_self.value
            );
        }
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

    #[test]
    fn asr_timing_artifacts_are_reused_for_tts_alignment() {
        let asr_words = vec![
            TranscriptWord {
                text: "hello".to_string(),
                start_ms: Some(100),
                end_ms: Some(300),
                confidence: Some(0.92),
            },
            TranscriptWord {
                text: "there".to_string(),
                start_ms: Some(320),
                end_ms: Some(500),
                confidence: Some(0.88),
            },
        ];
        let asr_artifacts = asr_word_hypothesis_artifacts(&asr_words, "whisper");
        let alignment = tts_alignment_from_asr_word_hypotheses(&asr_artifacts, "tts-planner")
            .expect("tts alignment should be created from ASR timings");
        assert_eq!(alignment.kind, SpeechArtifactKind::Alignment);
        assert_eq!(
            alignment.span,
            SpeechSpan::Time(TimeSpan {
                start_ms: 100,
                end_ms: 500
            })
        );
        assert_eq!(alignment.dependencies.len(), 2);
        assert!(
            alignment
                .dependencies
                .iter()
                .all(|dep| dep.artifact_id.starts_with("asr-word-"))
        );
    }

    #[test]
    fn compares_tts_intended_phones_against_self_heard_asr() {
        let mut doc = SpeechDocument::new("self-monitor");
        doc.upsert(
            SpeechArtifact::placeholder(
                "tts-intended-0",
                SpeechArtifactKind::PhoneHypothesis,
                SpeechSpan::Phones(PhoneSpan { start: 0, end: 1 }),
                SpeechProvenance::new(SpeechSourceOrigin::Unknown).with_source("tts"),
                SpeechArtifactContent::PhoneSymbol("HH".to_string()),
            )
            .with_attribution(
                SpeechAttribution::new(AttributionSourceKind::SynthesizedWaveform)
                    .with_is_self(Some(true), 0.9),
            ),
        );
        doc.upsert(
            SpeechArtifact::placeholder(
                "tts-intended-1",
                SpeechArtifactKind::PhoneHypothesis,
                SpeechSpan::Phones(PhoneSpan { start: 1, end: 2 }),
                SpeechProvenance::new(SpeechSourceOrigin::Unknown).with_source("tts"),
                SpeechArtifactContent::PhoneSymbol("EH".to_string()),
            )
            .with_attribution(
                SpeechAttribution::new(AttributionSourceKind::SynthesizedWaveform)
                    .with_is_self(Some(true), 0.9),
            ),
        );
        doc.upsert(
            SpeechArtifact::placeholder(
                "asr-heard-0",
                SpeechArtifactKind::PhoneHypothesis,
                SpeechSpan::Phones(PhoneSpan { start: 0, end: 1 }),
                SpeechProvenance::new(SpeechSourceOrigin::Unknown).with_source("asr"),
                SpeechArtifactContent::PhoneSymbol("HH".to_string()),
            )
            .with_attribution(
                SpeechAttribution::new(AttributionSourceKind::LiveVoice).with_is_self(None, 0.55),
            ),
        );
        doc.upsert(
            SpeechArtifact::placeholder(
                "asr-heard-1",
                SpeechArtifactKind::PhoneHypothesis,
                SpeechSpan::Phones(PhoneSpan { start: 1, end: 2 }),
                SpeechProvenance::new(SpeechSourceOrigin::Unknown).with_source("asr"),
                SpeechArtifactContent::PhoneSymbol("IH".to_string()),
            )
            .with_attribution(
                SpeechAttribution::new(AttributionSourceKind::LiveVoice).with_is_self(None, 0.51),
            ),
        );

        assert_eq!(
            compare_intended_vs_heard_phones(&doc),
            vec![PhoneComparison {
                phone_index: 1,
                intended: "EH".to_string(),
                heard: "IH".to_string(),
            }]
        );
        let heard = doc
            .artifacts
            .get("asr-heard-0")
            .expect("asr-heard-0 should be in doc");
        assert_eq!(heard.attributions[1].is_self.value, None);
        assert_eq!(heard.attributions[1].is_self.confidence, 0.55);
    }
}
