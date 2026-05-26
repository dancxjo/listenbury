use anyhow::Result;

use crate::audio::frame::AudioFrame;
use crate::speech::phone_plan::PhonePlan;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechStageKind {
    Analyzer,
    ProsodyPlanner,
    AcousticModel,
    Vocoder,
    MouthSink,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechStageDescriptor {
    pub kind: SpeechStageKind,
    pub id: String,
    pub detail: String,
}

impl SpeechStageDescriptor {
    pub fn new(kind: SpeechStageKind, id: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            kind,
            id: id.into(),
            detail: detail.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinguisticPlan {
    pub text: String,
    pub phone_plan: PhonePlan,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AcousticPlan {
    pub route_id: String,
    pub text: String,
    pub phone_plan: PhonePlan,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioRender {
    pub frames: Vec<AudioFrame>,
    pub source_label: String,
}

pub trait LinguisticAnalyzer: Send {
    fn describe(&self) -> SpeechStageDescriptor;
    fn analyze(&mut self, text: &str) -> Result<LinguisticPlan>;
}

pub trait ProsodyPlanner: Send {
    fn describe(&self) -> SpeechStageDescriptor;
    fn plan(&mut self, linguistic: LinguisticPlan) -> Result<PhonePlan>;
}

pub trait AcousticPlanner: Send {
    fn describe(&self) -> SpeechStageDescriptor;
    fn plan(&mut self, phone_plan: PhonePlan, text: &str) -> Result<AcousticPlan>;
}

pub trait VocoderRenderer: Send {
    fn describe(&self) -> SpeechStageDescriptor;
    fn render(&mut self, acoustic: AcousticPlan) -> Result<AudioRender>;
}

pub trait MouthSink: Send {
    fn describe(&self) -> SpeechStageDescriptor;
    fn accept(&mut self, audio: AudioRender) -> Result<()>;
}

pub struct SpeechPipeline {
    pub analyzer: Box<dyn LinguisticAnalyzer>,
    pub planner: Box<dyn ProsodyPlanner>,
    pub acoustic: Box<dyn AcousticPlanner>,
    pub renderer: Box<dyn VocoderRenderer>,
    pub mouth: Box<dyn MouthSink>,
}

impl SpeechPipeline {
    pub fn new(
        analyzer: Box<dyn LinguisticAnalyzer>,
        planner: Box<dyn ProsodyPlanner>,
        acoustic: Box<dyn AcousticPlanner>,
        renderer: Box<dyn VocoderRenderer>,
        mouth: Box<dyn MouthSink>,
    ) -> Self {
        Self {
            analyzer,
            planner,
            acoustic,
            renderer,
            mouth,
        }
    }

    pub fn describe(&self) -> Vec<SpeechStageDescriptor> {
        vec![
            self.analyzer.describe(),
            self.planner.describe(),
            self.acoustic.describe(),
            self.renderer.describe(),
            self.mouth.describe(),
        ]
    }

    pub fn run(&mut self, text: &str) -> Result<()> {
        let linguistic = self.analyzer.analyze(text)?;
        let phone_plan = self.planner.plan(linguistic)?;
        let acoustic = self.acoustic.plan(phone_plan, text)?;
        let audio = self.renderer.render(acoustic)?;
        self.mouth.accept(audio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speech::phone_plan::{LexicalStatus, PhoneSpan, WordPlan};

    #[derive(Default)]
    struct RecordingStage {
        calls: Vec<&'static str>,
    }

    fn phone_plan() -> PhonePlan {
        PhonePlan {
            source_text: "hello".to_string(),
            words: vec![WordPlan {
                text: "hello".to_string(),
                start_phone: 0,
                end_phone: 1,
                phones: vec!["hh".to_string()],
                stress: None,
                lexical_status: LexicalStatus::Resolved,
            }],
            phones: vec![PhoneSpan {
                phone: "hh".to_string(),
                start_ms: 0.0,
                duration_ms: 80.0,
                pitch: None,
                energy: None,
                syllable_index: None,
                word_index: Some(0),
            }],
        }
    }

    impl LinguisticAnalyzer for RecordingStage {
        fn describe(&self) -> SpeechStageDescriptor {
            SpeechStageDescriptor::new(SpeechStageKind::Analyzer, "test-analyzer", "test")
        }

        fn analyze(&mut self, text: &str) -> Result<LinguisticPlan> {
            self.calls.push("analyze");
            Ok(LinguisticPlan {
                text: text.to_string(),
                phone_plan: phone_plan(),
            })
        }
    }

    impl ProsodyPlanner for RecordingStage {
        fn describe(&self) -> SpeechStageDescriptor {
            SpeechStageDescriptor::new(SpeechStageKind::ProsodyPlanner, "test-planner", "test")
        }

        fn plan(&mut self, linguistic: LinguisticPlan) -> Result<PhonePlan> {
            self.calls.push("plan");
            Ok(linguistic.phone_plan)
        }
    }

    impl AcousticPlanner for RecordingStage {
        fn describe(&self) -> SpeechStageDescriptor {
            SpeechStageDescriptor::new(SpeechStageKind::AcousticModel, "test-acoustic", "test")
        }

        fn plan(&mut self, phone_plan: PhonePlan, text: &str) -> Result<AcousticPlan> {
            self.calls.push("acoustic");
            Ok(AcousticPlan {
                route_id: "test".to_string(),
                text: text.to_string(),
                phone_plan,
                detail: "test acoustic".to_string(),
            })
        }
    }

    impl VocoderRenderer for RecordingStage {
        fn describe(&self) -> SpeechStageDescriptor {
            SpeechStageDescriptor::new(SpeechStageKind::Vocoder, "test-renderer", "test")
        }

        fn render(&mut self, acoustic: AcousticPlan) -> Result<AudioRender> {
            self.calls.push("render");
            assert_eq!(acoustic.text, "hello");
            Ok(AudioRender {
                frames: Vec::new(),
                source_label: "test".to_string(),
            })
        }
    }

    impl MouthSink for RecordingStage {
        fn describe(&self) -> SpeechStageDescriptor {
            SpeechStageDescriptor::new(SpeechStageKind::MouthSink, "test-mouth", "test")
        }

        fn accept(&mut self, _audio: AudioRender) -> Result<()> {
            self.calls.push("mouth");
            Ok(())
        }
    }

    #[test]
    fn speech_pipeline_exposes_ordered_stage_descriptions() {
        let pipeline = SpeechPipeline::new(
            Box::<RecordingStage>::default(),
            Box::<RecordingStage>::default(),
            Box::<RecordingStage>::default(),
            Box::<RecordingStage>::default(),
            Box::<RecordingStage>::default(),
        );

        let kinds = pipeline
            .describe()
            .into_iter()
            .map(|stage| stage.kind)
            .collect::<Vec<_>>();

        assert_eq!(
            kinds,
            vec![
                SpeechStageKind::Analyzer,
                SpeechStageKind::ProsodyPlanner,
                SpeechStageKind::AcousticModel,
                SpeechStageKind::Vocoder,
                SpeechStageKind::MouthSink,
            ]
        );
    }

    #[test]
    fn speech_pipeline_runs_text_to_mouth() {
        let mut pipeline = SpeechPipeline::new(
            Box::<RecordingStage>::default(),
            Box::<RecordingStage>::default(),
            Box::<RecordingStage>::default(),
            Box::<RecordingStage>::default(),
            Box::<RecordingStage>::default(),
        );

        pipeline.run("hello").expect("pipeline run");
    }
}
