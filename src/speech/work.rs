use std::collections::VecDeque;
use std::time::Duration;

use anyhow::Result;

use crate::acoustic::MelFrame;
use crate::audio::frame::AudioFrame;
use crate::time::ExactTimestamp;
use crate::vocoder::{BackendFamily, VocoderBackend, VocoderInput};
use crate::voice::articulator::{
    PartialProsodyPhone, PhoneTimedRenderTarget, PitchHint, RenderPlan,
};
use crate::voice::tract::SourceFilterTrack;

pub type ChunkId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SyntheticWorkStageKind {
    TextStream,
    LinguisticPlanStream,
    AcousticPlanStream,
    SpectralFrameStream,
    RenderFrameStream,
    WaveformStream,
    AudioSink,
}

pub const CANONICAL_SYNTHETIC_WORK_FLOW: &[SyntheticWorkStageKind] = &[
    SyntheticWorkStageKind::TextStream,
    SyntheticWorkStageKind::LinguisticPlanStream,
    SyntheticWorkStageKind::AcousticPlanStream,
    SyntheticWorkStageKind::SpectralFrameStream,
    SyntheticWorkStageKind::RenderFrameStream,
    SyntheticWorkStageKind::WaveformStream,
    SyntheticWorkStageKind::AudioSink,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AudioTime {
    pub sample_rate_hz: u32,
    pub sample_index: u64,
}

impl AudioTime {
    pub const fn zero(sample_rate_hz: u32) -> Self {
        Self {
            sample_rate_hz,
            sample_index: 0,
        }
    }

    pub fn advance_samples(self, samples: usize) -> Self {
        Self {
            sample_rate_hz: self.sample_rate_hz,
            sample_index: self.sample_index.saturating_add(samples as u64),
        }
    }

    pub fn as_duration(self) -> Duration {
        if self.sample_rate_hz == 0 {
            return Duration::ZERO;
        }
        Duration::from_secs_f64(self.sample_index as f64 / self.sample_rate_hz as f64)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineTime {
    pub audio: AudioTime,
    pub wall: Option<ExactTimestamp>,
}

impl PipelineTime {
    pub const fn from_audio(audio: AudioTime) -> Self {
        Self { audio, wall: None }
    }

    pub const fn with_wall_time(mut self, wall: ExactTimestamp) -> Self {
        self.wall = Some(wall);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkBudget {
    pub max_items: usize,
    pub max_work: Duration,
}

impl WorkBudget {
    pub const fn new(max_items: usize, max_work: Duration) -> Self {
        Self {
            max_items,
            max_work,
        }
    }

    pub const fn single_item() -> Self {
        Self {
            max_items: 1,
            max_work: Duration::ZERO,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SyntheticClockKind {
    Audio,
    Frame,
    Linguistic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SyntheticClock {
    Audio {
        sample_rate_hz: u32,
    },
    Frame {
        sample_rate_hz: u32,
        hop_samples: usize,
    },
    Linguistic,
}

impl SyntheticClock {
    pub const fn kind(self) -> SyntheticClockKind {
        match self {
            Self::Audio { .. } => SyntheticClockKind::Audio,
            Self::Frame { .. } => SyntheticClockKind::Frame,
            Self::Linguistic => SyntheticClockKind::Linguistic,
        }
    }

    pub fn nominal_period(self) -> Option<Duration> {
        match self {
            Self::Audio { sample_rate_hz } if sample_rate_hz > 0 => {
                Some(Duration::from_secs_f64(1.0 / f64::from(sample_rate_hz)))
            }
            Self::Frame {
                sample_rate_hz,
                hop_samples,
            } if sample_rate_hz > 0 => Some(Duration::from_secs_f64(
                hop_samples as f64 / f64::from(sample_rate_hz),
            )),
            Self::Audio { .. } | Self::Frame { .. } | Self::Linguistic => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SyntheticStageRuntimePolicy {
    pub clock: SyntheticClock,
    pub lookahead_target: Option<Duration>,
    pub minimum_commit: Boundary,
    pub maximum_latency: Duration,
}

impl SyntheticStageRuntimePolicy {
    pub const fn new(
        clock: SyntheticClock,
        lookahead_target: Option<Duration>,
        minimum_commit: Boundary,
        maximum_latency: Duration,
    ) -> Self {
        Self {
            clock,
            lookahead_target,
            minimum_commit,
            maximum_latency,
        }
    }

    pub fn audio_sink(sample_rate_hz: u32, watermarks: &SyntheticPipelineWatermarks) -> Self {
        Self::new(
            SyntheticClock::Audio { sample_rate_hz },
            Some(watermarks.audio_sink.target),
            Boundary::None,
            watermarks.audio_sink.high,
        )
    }

    pub fn representation_frames(
        sample_rate_hz: u32,
        hop_samples: usize,
        watermarks: &SyntheticPipelineWatermarks,
    ) -> Self {
        Self::new(
            SyntheticClock::Frame {
                sample_rate_hz,
                hop_samples,
            },
            Some(watermarks.representation.target),
            Boundary::Phone,
            watermarks.representation.high,
        )
    }

    pub fn linguistic_phrase(watermarks: &SyntheticPipelineWatermarks) -> Self {
        Self::new(
            SyntheticClock::Linguistic,
            None,
            Boundary::Phrase,
            watermarks.acoustic.high,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoundaryHint {
    Token,
    Word,
    ClauseMaybe,
    Sentence,
    ForcedBreak,
    Correction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextSource {
    User,
    Llm,
    Script,
    Repair,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextChunk {
    pub text: String,
    pub boundary: BoundaryHint,
    pub source: TextSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Commitment {
    Draft,
    Planned,
    Committed,
    Spoken,
    Cancelled,
}

impl Commitment {
    pub const fn can_revise(self) -> bool {
        matches!(self, Self::Draft | Self::Planned)
    }

    pub const fn can_mutate(self) -> bool {
        matches!(self, Self::Draft)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Boundary {
    None,
    Phone,
    Syllable,
    Word,
    Phrase,
    BreathGroup,
    Sentence,
    Turn,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TimedItem<T> {
    pub item: T,
    pub start: Option<AudioTime>,
    pub end: Option<AudioTime>,
    pub commitment: Commitment,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreamChunk<T> {
    pub id: ChunkId,
    pub parent: Option<ChunkId>,
    pub items: Vec<TimedItem<T>>,
    pub boundary: Boundary,
    pub revision: u64,
}

pub type RepresentationStream<T> = StreamChunk<T>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordPlan {
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhonePlan {
    pub symbol: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyllablePlan {
    pub text: String,
    pub phone_span: std::ops::Range<usize>,
    pub stressed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhraseShape {
    pub boundary: Boundary,
    pub accent_targets: Vec<usize>,
    pub final_cadence: Option<Cadence>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cadence {
    Falling,
    Rising,
    Suspensive,
    Flat,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LingChunk {
    pub id: ChunkId,
    pub words: Vec<WordPlan>,
    pub phones: Vec<PhonePlan>,
    pub syllables: Vec<SyllablePlan>,
    pub phrase: PhraseShape,
    pub commitment: Commitment,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AcousticPhoneTiming {
    pub phone: String,
    pub start: AudioTime,
    pub end: AudioTime,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Curve<T> {
    pub points: Vec<CurvePoint<T>>,
}

impl<T> Curve<T> {
    pub fn empty() -> Self {
        Self { points: Vec::new() }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CurvePoint<T> {
    pub at: AudioTime,
    pub value: T,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BreathPlan {
    pub inhale_before: bool,
    pub phrase_break_after: Option<Duration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceProfile {
    pub id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AcousticChunk {
    pub id: ChunkId,
    pub phone_timing: Vec<AcousticPhoneTiming>,
    pub f0: Curve<f32>,
    pub energy: Curve<f32>,
    pub voicing: Curve<f32>,
    pub breath: BreathPlan,
    pub articulatory: Option<ArticulatoryChunk>,
    pub voice: VoiceProfile,
    pub commitment: Commitment,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MelChunk {
    pub config: String,
    pub frames: Vec<MelFrame>,
    pub frame_hop_samples: usize,
    pub sample_rate_hz: u32,
    pub time_start: AudioTime,
    pub commitment: Commitment,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MelF0Chunk {
    pub mel: Vec<MelFrame>,
    pub f0_hz: Vec<f32>,
    pub voiced: Vec<bool>,
    pub frame_hop_samples: usize,
    pub sample_rate_hz: u32,
    pub time_start: AudioTime,
    pub commitment: Commitment,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WorldChunk {
    pub f0_hz: Vec<f32>,
    pub spectral_envelope: Vec<f32>,
    pub aperiodicity: Vec<f32>,
    pub sample_rate_hz: u32,
    pub time_start: AudioTime,
    pub commitment: Commitment,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LpcNetChunk {
    pub features: Vec<f32>,
    pub frame_count: usize,
    pub sample_rate_hz: u32,
    pub time_start: AudioTime,
    pub commitment: Commitment,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArticulatoryChunk {
    pub targets: Vec<PhoneTimedRenderTarget>,
    pub time_start: AudioTime,
    pub commitment: Commitment,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PartialProsodyChunk {
    pub text: String,
    pub phones: Vec<PartialProsodyPhone>,
    pub pitch_hints: Vec<PitchHint>,
    pub time_start: AudioTime,
    pub commitment: Commitment,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoarseTextChunk {
    pub text: String,
    pub ssml_hint: Option<String>,
    pub commitment: Commitment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RepresentationKind {
    Mel,
    MelF0,
    World,
    LpcNet,
    Articulatory,
    PhoneTimed,
    PartialProsody,
    CoarseText,
    SourceFilterTrack,
    Wave,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SyntheticRepresentation {
    Mel(MelChunk),
    MelF0(MelF0Chunk),
    World(WorldChunk),
    LpcNet(LpcNetChunk),
    Articulatory(ArticulatoryChunk),
    PhoneTimed(Vec<PhoneTimedRenderTarget>),
    PartialProsody(PartialProsodyChunk),
    CoarseText(CoarseTextChunk),
    SourceFilterTrack(SourceFilterTrack),
    Wave(WaveChunk),
}

impl SyntheticRepresentation {
    pub fn kind(&self) -> RepresentationKind {
        match self {
            Self::Mel(_) => RepresentationKind::Mel,
            Self::MelF0(_) => RepresentationKind::MelF0,
            Self::World(_) => RepresentationKind::World,
            Self::LpcNet(_) => RepresentationKind::LpcNet,
            Self::Articulatory(_) => RepresentationKind::Articulatory,
            Self::PhoneTimed(_) => RepresentationKind::PhoneTimed,
            Self::PartialProsody(_) => RepresentationKind::PartialProsody,
            Self::CoarseText(_) => RepresentationKind::CoarseText,
            Self::SourceFilterTrack(_) => RepresentationKind::SourceFilterTrack,
            Self::Wave(_) => RepresentationKind::Wave,
        }
    }

    pub fn commitment(&self) -> Commitment {
        match self {
            Self::Mel(chunk) => chunk.commitment,
            Self::MelF0(chunk) => chunk.commitment,
            Self::World(chunk) => chunk.commitment,
            Self::LpcNet(chunk) => chunk.commitment,
            Self::Articulatory(chunk) => chunk.commitment,
            Self::PhoneTimed(_) => Commitment::Planned,
            Self::PartialProsody(chunk) => chunk.commitment,
            Self::CoarseText(chunk) => chunk.commitment,
            Self::SourceFilterTrack(_) => Commitment::Planned,
            Self::Wave(chunk) => chunk.commitment,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WaveChunk {
    pub samples: Vec<f32>,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub start_time: AudioTime,
    pub end_time: AudioTime,
    pub commitment: Commitment,
}

impl WaveChunk {
    pub fn new(
        samples: Vec<f32>,
        sample_rate_hz: u32,
        channels: u16,
        start_time: AudioTime,
        commitment: Commitment,
    ) -> Self {
        let channel_count = usize::from(channels.max(1));
        let frame_count = samples.len() / channel_count;
        let end_time = start_time.advance_samples(frame_count);
        Self {
            samples,
            sample_rate_hz,
            channels,
            start_time,
            end_time,
            commitment,
        }
    }

    pub fn to_audio_frame(&self, captured_at: ExactTimestamp) -> AudioFrame {
        AudioFrame {
            captured_at,
            sample_rate_hz: self.sample_rate_hz,
            channels: self.channels,
            samples: self.samples.clone(),
            voice_signatures: Vec::new(),
        }
    }
}

impl From<AudioFrame> for WaveChunk {
    fn from(frame: AudioFrame) -> Self {
        let start_time = AudioTime::zero(frame.sample_rate_hz);
        Self::new(
            frame.samples,
            frame.sample_rate_hz,
            frame.channels,
            start_time,
            Commitment::Committed,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SyntheticEvent {
    Say(WaveChunk),
    Pause(Duration),
    FadeOut(Duration),
    Repair(RepairPlan),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairPlan {
    pub strategy: RepairStrategy,
    pub replacement_text: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RepairStrategy {
    ContinueAsIfCorrect,
    MicroPauseReplacement,
    IMeanResume,
    FullRestatement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CommitHorizons {
    pub draft_until: AudioTime,
    pub planned_until: AudioTime,
    pub audio_until: AudioTime,
    pub spoken_until: AudioTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferWatermarks {
    pub low: Duration,
    pub target: Duration,
    pub high: Duration,
}

impl BufferWatermarks {
    pub const fn new(low: Duration, target: Duration, high: Duration) -> Self {
        Self { low, target, high }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SyntheticPipelineWatermarks {
    pub text_lookahead_words: std::ops::RangeInclusive<usize>,
    pub acoustic: BufferWatermarks,
    pub representation: BufferWatermarks,
    pub wave: BufferWatermarks,
    pub audio_sink: BufferWatermarks,
}

impl Default for SyntheticPipelineWatermarks {
    fn default() -> Self {
        Self {
            text_lookahead_words: 3..=12,
            acoustic: BufferWatermarks::new(
                Duration::from_millis(300),
                Duration::from_millis(500),
                Duration::from_millis(800),
            ),
            representation: BufferWatermarks::new(
                Duration::from_millis(200),
                Duration::from_millis(350),
                Duration::from_millis(500),
            ),
            wave: BufferWatermarks::new(
                Duration::from_millis(100),
                Duration::from_millis(200),
                Duration::from_millis(300),
            ),
            audio_sink: BufferWatermarks::new(
                Duration::from_millis(40),
                Duration::from_millis(80),
                Duration::from_millis(120),
            ),
        }
    }
}

impl SyntheticPipelineWatermarks {
    pub fn low_latency() -> Self {
        Self {
            text_lookahead_words: 1..=6,
            acoustic: BufferWatermarks::new(
                Duration::from_millis(200),
                Duration::from_millis(300),
                Duration::from_millis(400),
            ),
            representation: BufferWatermarks::new(
                Duration::from_millis(120),
                Duration::from_millis(180),
                Duration::from_millis(250),
            ),
            wave: BufferWatermarks::new(
                Duration::from_millis(80),
                Duration::from_millis(120),
                Duration::from_millis(150),
            ),
            audio_sink: BufferWatermarks::new(
                Duration::from_millis(30),
                Duration::from_millis(45),
                Duration::from_millis(60),
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StageReadiness {
    NeedsInput,
    Ready,
    WaitingForLookahead,
    Late,
    Blocked,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageStatus {
    pub id: &'static str,
    pub readiness: StageReadiness,
    pub input_len: usize,
    pub output_len: usize,
}

impl StageStatus {
    pub const fn new(
        id: &'static str,
        readiness: StageReadiness,
        input_len: usize,
        output_len: usize,
    ) -> Self {
        Self {
            id,
            readiness,
            input_len,
            output_len,
        }
    }
}

pub trait TickStage {
    fn id(&self) -> &'static str;
    fn tick(&mut self, now: PipelineTime, budget: WorkBudget) -> StageStatus;
    fn status(&self) -> StageStatus;
}

pub trait StreamStage: TickStage {
    type Input;
    type Output;

    fn accept(&mut self, input: Self::Input);
    fn drain(&mut self) -> Vec<Self::Output>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderStatus {
    Accepted {
        queued: usize,
    },
    Unsupported {
        renderer: &'static str,
        kind: RepresentationKind,
    },
}

pub trait Renderer: TickStage {
    fn accepts(&self, kind: RepresentationKind) -> bool;
    fn push(&mut self, input: SyntheticRepresentation) -> RenderStatus;
    fn drain(&mut self) -> Vec<WaveChunk>;
}

pub struct BlockingVocoderRenderer<B> {
    id: &'static str,
    backend: B,
    pending: VecDeque<SyntheticRepresentation>,
    rendered: VecDeque<WaveChunk>,
    last_status: StageStatus,
}

impl<B> BlockingVocoderRenderer<B>
where
    B: VocoderBackend,
{
    pub fn new(id: &'static str, backend: B) -> Self {
        Self {
            id,
            backend,
            pending: VecDeque::new(),
            rendered: VecDeque::new(),
            last_status: StageStatus::new(id, StageReadiness::NeedsInput, 0, 0),
        }
    }

    fn render_representation(
        &mut self,
        input: &SyntheticRepresentation,
    ) -> Result<Vec<AudioFrame>> {
        match input {
            SyntheticRepresentation::Mel(chunk) => {
                self.backend.render(VocoderInput::Mel(&chunk.frames))
            }
            SyntheticRepresentation::MelF0(chunk) => self.backend.render(VocoderInput::MelF0 {
                mel: &chunk.mel,
                f0_hz: &chunk.f0_hz,
                voiced: &chunk.voiced,
            }),
            SyntheticRepresentation::Articulatory(chunk) => self
                .backend
                .render(VocoderInput::PhoneTimed(&chunk.targets)),
            SyntheticRepresentation::PhoneTimed(targets) => {
                self.backend.render(VocoderInput::PhoneTimed(targets))
            }
            SyntheticRepresentation::PartialProsody(chunk) => {
                self.backend.render(VocoderInput::PartialProsody {
                    text: &chunk.text,
                    phones: &chunk.phones,
                    pitch_hints: &chunk.pitch_hints,
                })
            }
            SyntheticRepresentation::CoarseText(chunk) => {
                self.backend.render(VocoderInput::CoarseText {
                    text: &chunk.text,
                    ssml_hint: chunk.ssml_hint.as_deref(),
                })
            }
            SyntheticRepresentation::SourceFilterTrack(track) => {
                self.backend.render(VocoderInput::SourceFilterTrack(track))
            }
            SyntheticRepresentation::World(_)
            | SyntheticRepresentation::LpcNet(_)
            | SyntheticRepresentation::Wave(_) => Ok(Vec::new()),
        }
    }
}

impl<B> TickStage for BlockingVocoderRenderer<B>
where
    B: VocoderBackend,
{
    fn id(&self) -> &'static str {
        self.id
    }

    fn tick(&mut self, now: PipelineTime, budget: WorkBudget) -> StageStatus {
        let mut processed = 0usize;
        while processed < budget.max_items {
            let Some(input) = self.pending.pop_front() else {
                break;
            };
            if let SyntheticRepresentation::Wave(chunk) = input {
                self.rendered.push_back(chunk);
                processed += 1;
                continue;
            }
            match self.render_representation(&input) {
                Ok(frames) => {
                    let mut cursor = now.audio;
                    for frame in frames {
                        let mut chunk = WaveChunk::from(frame);
                        if cursor.sample_rate_hz != chunk.sample_rate_hz {
                            cursor = AudioTime::zero(chunk.sample_rate_hz);
                        }
                        let channel_count = usize::from(chunk.channels.max(1));
                        let frame_count = chunk.samples.len() / channel_count;
                        chunk.start_time = cursor;
                        chunk.end_time = cursor.advance_samples(frame_count);
                        cursor = chunk.end_time;
                        self.rendered.push_back(chunk);
                    }
                }
                Err(_) => {
                    self.last_status = StageStatus::new(
                        self.id,
                        StageReadiness::Blocked,
                        self.pending.len(),
                        self.rendered.len(),
                    );
                    return self.last_status.clone();
                }
            }
            processed += 1;
        }

        let readiness = if self.pending.is_empty() {
            StageReadiness::NeedsInput
        } else {
            StageReadiness::Ready
        };
        self.last_status =
            StageStatus::new(self.id, readiness, self.pending.len(), self.rendered.len());
        self.last_status.clone()
    }

    fn status(&self) -> StageStatus {
        self.last_status.clone()
    }
}

impl<B> Renderer for BlockingVocoderRenderer<B>
where
    B: VocoderBackend,
{
    fn accepts(&self, kind: RepresentationKind) -> bool {
        let descriptor = self.backend.descriptor();
        let capabilities = descriptor.capabilities;
        match kind {
            RepresentationKind::Mel => capabilities.accepts_mel,
            RepresentationKind::MelF0 => capabilities.accepts_mel_f0,
            RepresentationKind::Articulatory | RepresentationKind::PhoneTimed => {
                capabilities.accepts_phone_timed
            }
            RepresentationKind::PartialProsody => capabilities.accepts_partial_prosody,
            RepresentationKind::CoarseText => capabilities.accepts_coarse_text,
            RepresentationKind::SourceFilterTrack => matches!(
                descriptor.family,
                BackendFamily::FormantSourceFilter | BackendFamily::NeuralSourceFilter
            ),
            RepresentationKind::Wave => true,
            RepresentationKind::World | RepresentationKind::LpcNet => false,
        }
    }

    fn push(&mut self, input: SyntheticRepresentation) -> RenderStatus {
        let kind = input.kind();
        if !self.accepts(kind) {
            return RenderStatus::Unsupported {
                renderer: self.id,
                kind,
            };
        }
        self.pending.push_back(input);
        RenderStatus::Accepted {
            queued: self.pending.len(),
        }
    }

    fn drain(&mut self) -> Vec<WaveChunk> {
        self.rendered.drain(..).collect()
    }
}

pub struct WavePassthroughRenderer {
    pending: VecDeque<WaveChunk>,
    rendered: VecDeque<WaveChunk>,
}

impl WavePassthroughRenderer {
    pub fn new() -> Self {
        Self {
            pending: VecDeque::new(),
            rendered: VecDeque::new(),
        }
    }
}

impl Default for WavePassthroughRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl TickStage for WavePassthroughRenderer {
    fn id(&self) -> &'static str {
        "wave-passthrough-renderer"
    }

    fn tick(&mut self, _now: PipelineTime, budget: WorkBudget) -> StageStatus {
        let mut processed = 0usize;
        while processed < budget.max_items {
            let Some(chunk) = self.pending.pop_front() else {
                break;
            };
            self.rendered.push_back(chunk);
            processed += 1;
        }
        self.status()
    }

    fn status(&self) -> StageStatus {
        let readiness = if self.pending.is_empty() {
            StageReadiness::NeedsInput
        } else {
            StageReadiness::Ready
        };
        StageStatus::new(
            self.id(),
            readiness,
            self.pending.len(),
            self.rendered.len(),
        )
    }
}

impl Renderer for WavePassthroughRenderer {
    fn accepts(&self, kind: RepresentationKind) -> bool {
        kind == RepresentationKind::Wave
    }

    fn push(&mut self, input: SyntheticRepresentation) -> RenderStatus {
        match input {
            SyntheticRepresentation::Wave(chunk) => {
                self.pending.push_back(chunk);
                RenderStatus::Accepted {
                    queued: self.pending.len(),
                }
            }
            other => RenderStatus::Unsupported {
                renderer: self.id(),
                kind: other.kind(),
            },
        }
    }

    fn drain(&mut self) -> Vec<WaveChunk> {
        self.rendered.drain(..).collect()
    }
}

pub struct SyntheticWorkGraph {
    stages: Vec<Box<dyn TickStage>>,
    watermarks: SyntheticPipelineWatermarks,
}

impl SyntheticWorkGraph {
    pub fn new(watermarks: SyntheticPipelineWatermarks) -> Self {
        Self {
            stages: Vec::new(),
            watermarks,
        }
    }

    pub fn with_default_watermarks() -> Self {
        Self::new(SyntheticPipelineWatermarks::default())
    }

    pub fn add_stage(&mut self, stage: Box<dyn TickStage>) {
        self.stages.push(stage);
    }

    pub fn tick(&mut self, now: PipelineTime, budget: WorkBudget) -> Vec<StageStatus> {
        self.stages
            .iter_mut()
            .map(|stage| stage.tick(now, budget))
            .collect()
    }

    pub fn statuses(&self) -> Vec<StageStatus> {
        self.stages.iter().map(|stage| stage.status()).collect()
    }

    pub fn watermarks(&self) -> SyntheticPipelineWatermarks {
        self.watermarks.clone()
    }
}

pub fn render_plan_to_representation(
    plan: RenderPlan,
    time_start: AudioTime,
    commitment: Commitment,
) -> SyntheticRepresentation {
    match plan {
        RenderPlan::PhoneTimed(targets) => {
            SyntheticRepresentation::Articulatory(ArticulatoryChunk {
                targets,
                time_start,
                commitment,
            })
        }
        RenderPlan::PartialProsody {
            text,
            phones,
            pitch_hints,
        } => SyntheticRepresentation::PartialProsody(PartialProsodyChunk {
            text,
            phones,
            pitch_hints,
            time_start,
            commitment,
        }),
        RenderPlan::CoarseText { text, ssml_hint } => {
            SyntheticRepresentation::CoarseText(CoarseTextChunk {
                text,
                ssml_hint,
                commitment,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocoder::{BackendCapabilities, BackendFamily, VocoderDescriptor};

    struct SilentMelBackend;

    impl VocoderBackend for SilentMelBackend {
        fn id(&self) -> &'static str {
            "silent-mel"
        }

        fn descriptor(&self) -> VocoderDescriptor {
            let mut capabilities = BackendCapabilities::unsupported();
            capabilities.accepts_mel = true;
            VocoderDescriptor {
                id: "silent-mel",
                family: BackendFamily::NeuralVocoder,
                capabilities,
                sample_rate_hz: 24_000,
                backend_kind: None,
                detail: None,
                notes: &["test backend"],
            }
        }

        fn render(&mut self, input: VocoderInput<'_>) -> Result<Vec<AudioFrame>> {
            match input {
                VocoderInput::Mel(frames) => Ok(vec![AudioFrame {
                    captured_at: ExactTimestamp::from_unix_nanos(1),
                    sample_rate_hz: 24_000,
                    channels: 1,
                    samples: vec![0.0; frames.len() * 240],
                    voice_signatures: Vec::new(),
                }]),
                _ => Ok(Vec::new()),
            }
        }
    }

    #[test]
    fn mel_is_one_representation_not_the_graph_contract() {
        let mel = SyntheticRepresentation::Mel(MelChunk {
            config: "debug-mel".to_string(),
            frames: vec![MelFrame {
                bins: vec![0.0, 1.0],
            }],
            frame_hop_samples: 256,
            sample_rate_hz: 24_000,
            time_start: AudioTime::zero(24_000),
            commitment: Commitment::Planned,
        });
        let wave = SyntheticRepresentation::Wave(WaveChunk::new(
            vec![0.0; 240],
            24_000,
            1,
            AudioTime::zero(24_000),
            Commitment::Committed,
        ));

        assert_eq!(mel.kind(), RepresentationKind::Mel);
        assert_eq!(wave.kind(), RepresentationKind::Wave);
        assert_eq!(mel.commitment(), Commitment::Planned);
        assert_eq!(wave.commitment(), Commitment::Committed);
    }

    #[test]
    fn canonical_flow_names_the_stream_boundaries() {
        assert_eq!(
            CANONICAL_SYNTHETIC_WORK_FLOW,
            &[
                SyntheticWorkStageKind::TextStream,
                SyntheticWorkStageKind::LinguisticPlanStream,
                SyntheticWorkStageKind::AcousticPlanStream,
                SyntheticWorkStageKind::SpectralFrameStream,
                SyntheticWorkStageKind::RenderFrameStream,
                SyntheticWorkStageKind::WaveformStream,
                SyntheticWorkStageKind::AudioSink,
            ]
        );
    }

    #[test]
    fn stage_clocks_make_audio_frame_and_linguistic_ticks_explicit() {
        let audio = SyntheticClock::Audio {
            sample_rate_hz: 48_000,
        };
        let frame = SyntheticClock::Frame {
            sample_rate_hz: 24_000,
            hop_samples: 256,
        };
        let linguistic = SyntheticClock::Linguistic;

        assert_eq!(audio.kind(), SyntheticClockKind::Audio);
        assert_eq!(frame.kind(), SyntheticClockKind::Frame);
        assert_eq!(linguistic.kind(), SyntheticClockKind::Linguistic);
        assert_eq!(audio.nominal_period().unwrap().as_nanos(), 20_833);
        assert_eq!(frame.nominal_period().unwrap().as_micros(), 10_666);
        assert_eq!(linguistic.nominal_period(), None);
    }

    #[test]
    fn stage_runtime_policies_bind_clocks_to_watermarks() {
        let watermarks = SyntheticPipelineWatermarks::low_latency();
        let sink = SyntheticStageRuntimePolicy::audio_sink(48_000, &watermarks);
        let representation =
            SyntheticStageRuntimePolicy::representation_frames(24_000, 256, &watermarks);
        let ling = SyntheticStageRuntimePolicy::linguistic_phrase(&watermarks);

        assert_eq!(sink.clock.kind(), SyntheticClockKind::Audio);
        assert_eq!(sink.lookahead_target, Some(Duration::from_millis(45)));
        assert_eq!(sink.maximum_latency, Duration::from_millis(60));
        assert_eq!(representation.clock.kind(), SyntheticClockKind::Frame);
        assert_eq!(representation.minimum_commit, Boundary::Phone);
        assert_eq!(
            representation.lookahead_target,
            Some(Duration::from_millis(180))
        );
        assert_eq!(ling.clock.kind(), SyntheticClockKind::Linguistic);
        assert_eq!(ling.minimum_commit, Boundary::Phrase);
    }

    #[test]
    fn wave_passthrough_renderer_obeys_tick_budget() {
        let mut renderer = WavePassthroughRenderer::new();
        for _ in 0..2 {
            let status = renderer.push(SyntheticRepresentation::Wave(WaveChunk::new(
                vec![0.0; 10],
                10,
                1,
                AudioTime::zero(10),
                Commitment::Committed,
            )));
            assert!(matches!(status, RenderStatus::Accepted { .. }));
        }

        let status = renderer.tick(
            PipelineTime::from_audio(AudioTime::zero(10)),
            WorkBudget::single_item(),
        );

        assert_eq!(status.input_len, 1);
        assert_eq!(status.output_len, 1);
        assert_eq!(renderer.drain().len(), 1);
        assert_eq!(renderer.status().input_len, 1);
    }

    #[test]
    fn blocking_vocoder_renderer_adapts_mel_representation() {
        let mut renderer = BlockingVocoderRenderer::new("test-hifigan-slot", SilentMelBackend);
        let status = renderer.push(SyntheticRepresentation::Mel(MelChunk {
            config: "test".to_string(),
            frames: vec![MelFrame { bins: vec![0.0] }, MelFrame { bins: vec![1.0] }],
            frame_hop_samples: 240,
            sample_rate_hz: 24_000,
            time_start: AudioTime::zero(24_000),
            commitment: Commitment::Planned,
        }));
        assert!(matches!(status, RenderStatus::Accepted { queued: 1 }));

        let status = renderer.tick(
            PipelineTime::from_audio(AudioTime::zero(24_000)),
            WorkBudget::single_item(),
        );
        let rendered = renderer.drain();

        assert_eq!(status.input_len, 0);
        assert_eq!(status.output_len, 1);
        assert_eq!(rendered.len(), 1);
        assert_eq!(rendered[0].samples.len(), 480);
        assert_eq!(rendered[0].end_time.sample_index, 480);
    }

    #[test]
    fn graph_ticks_registered_stages() {
        let mut graph = SyntheticWorkGraph::with_default_watermarks();
        graph.add_stage(Box::new(WavePassthroughRenderer::new()));

        let statuses = graph.tick(
            PipelineTime::from_audio(AudioTime::zero(48_000)),
            WorkBudget::single_item(),
        );

        assert_eq!(statuses.len(), 1);
        assert_eq!(statuses[0].id, "wave-passthrough-renderer");
        assert_eq!(
            graph.watermarks().audio_sink.target,
            Duration::from_millis(80)
        );
    }
}
