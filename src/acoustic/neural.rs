use anyhow::{Result, bail};

use crate::acoustic::{
    AcousticFrameTrack, AcousticInput, AcousticModelBackend, registry::AcousticModelDescriptor,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuralAcousticModelKind {
    FastSpeech2,
    Matcha,
    VitsPiper,
    SpeechT5,
}

impl NeuralAcousticModelKind {
    pub const fn id(self) -> &'static str {
        match self {
            Self::FastSpeech2 => "fastspeech2",
            Self::Matcha => "matcha",
            Self::VitsPiper => "vits-piper",
            Self::SpeechT5 => "speecht5",
        }
    }

    pub const fn display_name(self) -> &'static str {
        match self {
            Self::FastSpeech2 => "FastSpeech2",
            Self::Matcha => "Matcha",
            Self::VitsPiper => "VITS/Piper",
            Self::SpeechT5 => "SpeechT5",
        }
    }
}

pub struct NeuralAcousticModel {
    kind: NeuralAcousticModelKind,
}

pub type FastSpeech2AcousticModel = NeuralAcousticModel;
pub type MatchaAcousticModel = NeuralAcousticModel;
pub type VitsPiperAcousticModel = NeuralAcousticModel;
pub type SpeechT5AcousticModel = NeuralAcousticModel;

impl NeuralAcousticModel {
    pub const fn new(kind: NeuralAcousticModelKind) -> Self {
        Self { kind }
    }

    pub fn descriptor_for(kind: NeuralAcousticModelKind) -> AcousticModelDescriptor {
        AcousticModelDescriptor {
            id: kind.id(),
            notes: &[
                "Neural acoustic backend slot that produces AcousticFrameTrack mel/F0 output.",
                "Model loading and frontend-specific token adapters are intentionally explicit; source-filter remains the deterministic proxy backend.",
            ],
        }
    }
}

impl AcousticModelBackend for NeuralAcousticModel {
    fn id(&self) -> &'static str {
        self.kind.id()
    }

    fn generate(&mut self, _input: AcousticInput<'_>) -> Result<AcousticFrameTrack> {
        bail!(
            "{} acoustic backend is registered but requires model-specific tokenization and ONNX/session wiring before it can generate AcousticFrameTrack",
            self.kind.display_name()
        )
    }
}
