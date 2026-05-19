#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VarietyTag(pub String);

impl VarietyTag {
    pub fn new(tag: impl Into<String>) -> Self {
        Self(tag.into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phonology {
    pub name: String,
}

impl Phonology {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lexicon {
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinguisticRuntimeProfile {
    Realtime,
    Batch,
}

impl Default for LinguisticRuntimeProfile {
    fn default() -> Self {
        Self::Realtime
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinguisticVariety {
    pub tag: Option<VarietyTag>,
    pub name: String,
    pub phonology: Phonology,
    pub lexicon: Option<Lexicon>,
    pub runtime: LinguisticRuntimeProfile,
}

impl LinguisticVariety {
    pub fn tagged(tag: VarietyTag, name: impl Into<String>, phonology: Phonology) -> Self {
        Self {
            tag: Some(tag),
            name: name.into(),
            phonology,
            lexicon: None,
            runtime: LinguisticRuntimeProfile::default(),
        }
    }

    pub fn untagged(name: impl Into<String>, phonology: Phonology) -> Self {
        Self {
            tag: None,
            name: name.into(),
            phonology,
            lexicon: None,
            runtime: LinguisticRuntimeProfile::default(),
        }
    }
}
