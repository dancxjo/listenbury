use anyhow::Result;

/// Reusable artifact kinds that speech synthesis stages can consume or produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpeechArtifactKind {
    Text,
    NormalizedText,
    LexicalPlan,
    PhoneSequence,
    SyllableStressPlan,
    ProsodyPlan,
    PhoneTimedPlan,
    PhoneIds,
    DiphonePlan,
    AcousticTrack,
    MelF0Track,
    AudioFrames,
}

/// Small reusable stage contract for future composable speech pipelines.
pub trait SpeechStage<I, O> {
    fn id(&self) -> &'static str;
    fn run(&mut self, input: I) -> Result<O>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StageImplementationKind {
    FusedBackend,
    ExternalProcess,
    Planner,
    DiphoneSelector,
    AcousticModel,
    Vocoder,
    Renderer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageDescriptor {
    pub id: &'static str,
    pub consumes: Vec<SpeechArtifactKind>,
    pub produces: SpeechArtifactKind,
    pub implementation: StageImplementationKind,
}

impl StageDescriptor {
    pub fn new(
        id: &'static str,
        consumes: Vec<SpeechArtifactKind>,
        produces: SpeechArtifactKind,
        implementation: StageImplementationKind,
    ) -> Self {
        Self {
            id,
            consumes,
            produces,
            implementation,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineDescriptor {
    pub id: &'static str,
    pub stages: Vec<StageDescriptor>,
    pub fused: bool,
}

impl PipelineDescriptor {
    pub fn new(id: &'static str, stages: Vec<StageDescriptor>, fused: bool) -> Self {
        Self { id, stages, fused }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpeechPipelineKind {
    PiperCompat,
    PiperProcess,
    Klatt,
    MbrolaDiphone,
    SourceFilterHifigan,
}

impl SpeechPipelineKind {
    pub fn descriptor(self) -> PipelineDescriptor {
        match self {
            Self::PiperCompat => PipelineDescriptor::new(
                "piper-compat",
                vec![StageDescriptor::new(
                    "piper-compatible-onnx",
                    vec![SpeechArtifactKind::Text, SpeechArtifactKind::PhoneIds],
                    SpeechArtifactKind::AudioFrames,
                    StageImplementationKind::FusedBackend,
                )],
                true,
            ),
            Self::PiperProcess => PipelineDescriptor::new(
                "piper-process",
                vec![StageDescriptor::new(
                    "piper-process-backend",
                    vec![SpeechArtifactKind::Text],
                    SpeechArtifactKind::AudioFrames,
                    StageImplementationKind::ExternalProcess,
                )],
                true,
            ),
            Self::Klatt => PipelineDescriptor::new(
                "klatt",
                vec![StageDescriptor::new(
                    "klatt-formant-renderer",
                    vec![SpeechArtifactKind::PhoneTimedPlan],
                    SpeechArtifactKind::AudioFrames,
                    StageImplementationKind::Renderer,
                )],
                false,
            ),
            Self::MbrolaDiphone => PipelineDescriptor::new(
                "mbrola-diphone",
                vec![
                    StageDescriptor::new(
                        "mbrola-diphone-selection",
                        vec![SpeechArtifactKind::PhoneTimedPlan],
                        SpeechArtifactKind::DiphonePlan,
                        StageImplementationKind::DiphoneSelector,
                    ),
                    StageDescriptor::new(
                        "mbrola-diphone-renderer",
                        vec![SpeechArtifactKind::DiphonePlan],
                        SpeechArtifactKind::AudioFrames,
                        StageImplementationKind::Renderer,
                    ),
                ],
                false,
            ),
            Self::SourceFilterHifigan => PipelineDescriptor::new(
                "source-filter-hifigan",
                vec![
                    StageDescriptor::new(
                        "source-filter-acoustic-generator",
                        vec![SpeechArtifactKind::PhoneTimedPlan],
                        SpeechArtifactKind::MelF0Track,
                        StageImplementationKind::AcousticModel,
                    ),
                    StageDescriptor::new(
                        "hifigan-vocoder",
                        vec![SpeechArtifactKind::MelF0Track],
                        SpeechArtifactKind::AudioFrames,
                        StageImplementationKind::Vocoder,
                    ),
                ],
                false,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SpeechArtifactKind, SpeechPipelineKind, StageImplementationKind};

    #[test]
    fn piper_compat_descriptor_reports_fused_onnx_stage() {
        let descriptor = SpeechPipelineKind::PiperCompat.descriptor();
        assert_eq!(descriptor.id, "piper-compat");
        assert!(descriptor.fused);
        assert_eq!(descriptor.stages.len(), 1);
        let stage = &descriptor.stages[0];
        assert_eq!(stage.id, "piper-compatible-onnx");
        assert_eq!(
            stage.consumes,
            vec![SpeechArtifactKind::Text, SpeechArtifactKind::PhoneIds]
        );
        assert_eq!(stage.produces, SpeechArtifactKind::AudioFrames);
        assert_eq!(stage.implementation, StageImplementationKind::FusedBackend);
    }

    #[test]
    fn klatt_descriptor_reports_phone_timed_renderer() {
        let descriptor = SpeechPipelineKind::Klatt.descriptor();
        assert_eq!(descriptor.id, "klatt");
        assert!(!descriptor.fused);
        assert_eq!(descriptor.stages.len(), 1);
        let stage = &descriptor.stages[0];
        assert_eq!(stage.id, "klatt-formant-renderer");
        assert_eq!(stage.consumes, vec![SpeechArtifactKind::PhoneTimedPlan]);
        assert_eq!(stage.produces, SpeechArtifactKind::AudioFrames);
        assert_eq!(stage.implementation, StageImplementationKind::Renderer);
    }

    #[test]
    fn mbrola_descriptor_reports_selection_then_render() {
        let descriptor = SpeechPipelineKind::MbrolaDiphone.descriptor();
        assert_eq!(descriptor.id, "mbrola-diphone");
        assert!(!descriptor.fused);
        assert_eq!(descriptor.stages.len(), 2);
        assert_eq!(descriptor.stages[0].id, "mbrola-diphone-selection");
        assert_eq!(
            descriptor.stages[0].consumes,
            vec![SpeechArtifactKind::PhoneTimedPlan]
        );
        assert_eq!(
            descriptor.stages[0].produces,
            SpeechArtifactKind::DiphonePlan
        );
        assert_eq!(
            descriptor.stages[0].implementation,
            StageImplementationKind::DiphoneSelector
        );
        assert_eq!(descriptor.stages[1].id, "mbrola-diphone-renderer");
        assert_eq!(
            descriptor.stages[1].consumes,
            vec![SpeechArtifactKind::DiphonePlan]
        );
        assert_eq!(
            descriptor.stages[1].produces,
            SpeechArtifactKind::AudioFrames
        );
        assert_eq!(
            descriptor.stages[1].implementation,
            StageImplementationKind::Renderer
        );
    }

    #[test]
    fn source_filter_hifigan_descriptor_reports_acoustic_then_vocoder() {
        let descriptor = SpeechPipelineKind::SourceFilterHifigan.descriptor();
        assert_eq!(descriptor.id, "source-filter-hifigan");
        assert!(!descriptor.fused);
        assert_eq!(descriptor.stages.len(), 2);
        assert_eq!(descriptor.stages[0].id, "source-filter-acoustic-generator");
        assert_eq!(
            descriptor.stages[0].consumes,
            vec![SpeechArtifactKind::PhoneTimedPlan]
        );
        assert_eq!(
            descriptor.stages[0].produces,
            SpeechArtifactKind::MelF0Track
        );
        assert_eq!(
            descriptor.stages[0].implementation,
            StageImplementationKind::AcousticModel
        );
        assert_eq!(descriptor.stages[1].id, "hifigan-vocoder");
        assert_eq!(
            descriptor.stages[1].consumes,
            vec![SpeechArtifactKind::MelF0Track]
        );
        assert_eq!(
            descriptor.stages[1].produces,
            SpeechArtifactKind::AudioFrames
        );
        assert_eq!(
            descriptor.stages[1].implementation,
            StageImplementationKind::Vocoder
        );
    }
}
