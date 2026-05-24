use anyhow::{Result, bail};

use crate::acoustic::{
    AcousticModelBackend, NeuralAcousticModel, NeuralAcousticModelKind, SourceFilterAcousticModel,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcousticModelDescriptor {
    pub id: &'static str,
    pub notes: &'static [&'static str],
}

pub fn list_acoustic_models() -> Vec<AcousticModelDescriptor> {
    vec![
        SourceFilterAcousticModel::descriptor(),
        NeuralAcousticModel::descriptor_for(NeuralAcousticModelKind::FastSpeech2),
        NeuralAcousticModel::descriptor_for(NeuralAcousticModelKind::Matcha),
        NeuralAcousticModel::descriptor_for(NeuralAcousticModelKind::VitsPiper),
        NeuralAcousticModel::descriptor_for(NeuralAcousticModelKind::SpeechT5),
    ]
}

pub fn acoustic_model_by_id(id: &str) -> Result<Box<dyn AcousticModelBackend>> {
    match id {
        "source-filter" => Ok(Box::new(SourceFilterAcousticModel)),
        "fastspeech2" => Ok(Box::new(NeuralAcousticModel::new(
            NeuralAcousticModelKind::FastSpeech2,
        ))),
        "matcha" => Ok(Box::new(NeuralAcousticModel::new(
            NeuralAcousticModelKind::Matcha,
        ))),
        "vits-piper" => Ok(Box::new(NeuralAcousticModel::new(
            NeuralAcousticModelKind::VitsPiper,
        ))),
        "speecht5" => Ok(Box::new(NeuralAcousticModel::new(
            NeuralAcousticModelKind::SpeechT5,
        ))),
        _ => bail!("unknown acoustic model id `{id}`"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_names_source_filter_and_neural_acoustic_slots() {
        let ids = list_acoustic_models()
            .into_iter()
            .map(|descriptor| descriptor.id)
            .collect::<Vec<_>>();

        assert_eq!(
            ids,
            vec![
                "source-filter",
                "fastspeech2",
                "matcha",
                "vits-piper",
                "speecht5"
            ]
        );
    }

    #[test]
    fn neural_backend_slots_are_not_source_filter() {
        let backend = acoustic_model_by_id("speecht5").expect("registered backend");

        assert_eq!(backend.id(), "speecht5");
    }
}
