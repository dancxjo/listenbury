use anyhow::{Result, bail};

use crate::acoustic::{AcousticModelBackend, SourceFilterAcousticModel};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcousticModelDescriptor {
    pub id: &'static str,
    pub notes: &'static [&'static str],
}

pub fn list_acoustic_models() -> Vec<AcousticModelDescriptor> {
    vec![SourceFilterAcousticModel::descriptor()]
}

pub fn acoustic_model_by_id(id: &str) -> Result<Box<dyn AcousticModelBackend>> {
    match id {
        "source-filter" => Ok(Box::new(SourceFilterAcousticModel)),
        _ => bail!("unknown acoustic model id `{id}`"),
    }
}
