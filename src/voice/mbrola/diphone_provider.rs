use anyhow::Result;
use std::path::PathBuf;

use super::database::{MbrolaDatabase, MbrolaDatabaseError};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DiphoneKey {
    pub left: String,
    pub right: String,
}

impl DiphoneKey {
    pub fn new(left: impl Into<String>, right: impl Into<String>) -> Self {
        Self {
            left: left.into(),
            right: right.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DiphoneUnitSource {
    MbrolaExact,
    MbrolaBoundaryFallback,
    CacheHit,
    NeuralGenerated,
    Substitute,
    SyntheticSilence,
    SyntheticNoise,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DiphoneUnitMetadata {
    pub requested_key: Option<DiphoneKey>,
    pub warning: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiphoneUnit {
    pub key: DiphoneKey,
    pub samples: Vec<f32>,
    pub sample_rate_hz: u32,
    pub halfseg_samples: usize,
    pub source: DiphoneUnitSource,
    pub metadata: DiphoneUnitMetadata,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiphoneLookup {
    pub unit: DiphoneUnit,
}

pub trait DiphoneProvider {
    fn get_diphone(&mut self, left: &str, right: &str) -> Result<DiphoneLookup>;
}

pub struct MbrolaDiphoneProvider<'db> {
    database: &'db MbrolaDatabase,
}

impl<'db> MbrolaDiphoneProvider<'db> {
    pub fn new(database: &'db MbrolaDatabase) -> Self {
        Self { database }
    }
}

impl DiphoneProvider for MbrolaDiphoneProvider<'_> {
    fn get_diphone(&mut self, left: &str, right: &str) -> Result<DiphoneLookup> {
        let diphone = self.database.diphone(left, right).ok_or_else(|| {
            MbrolaDatabaseError::MissingDiphone {
                left: left.to_string(),
                right: right.to_string(),
            }
        })?;
        let samples = self.database.samples_for_diphone(diphone)?;
        Ok(DiphoneLookup {
            unit: DiphoneUnit {
                key: DiphoneKey::new(diphone.left.as_str(), diphone.right.as_str()),
                samples,
                sample_rate_hz: self.database.sample_rate_hz,
                halfseg_samples: diphone.halfseg_samples,
                source: DiphoneUnitSource::MbrolaExact,
                metadata: DiphoneUnitMetadata::default(),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mbrola_provider_returns_exact_us3_lookup_when_available() {
        let path = PathBuf::from("data/mbrola/us3/us3");
        if !path.is_file() {
            eprintln!("skipping us3 provider lookup test; run `just fetch`");
            return;
        }

        let database = MbrolaDatabase::load(&path).expect("load us3 database");
        let mut provider = MbrolaDiphoneProvider::new(&database);
        let lookup = provider
            .get_diphone("h", "@")
            .expect("lookup exact h-@ diphone");

        assert_eq!(lookup.unit.key, DiphoneKey::new("h", "@"));
        assert_eq!(lookup.unit.source, DiphoneUnitSource::MbrolaExact);
        assert!(!lookup.unit.samples.is_empty());
    }
}
