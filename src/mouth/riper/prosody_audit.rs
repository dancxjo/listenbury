use serde::{Deserialize, Serialize};

use crate::mouth::riper::sentence_analysis::{OrthographicEmphasisKind, ReductionDiagnostic};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stress {
    None,
    Primary,
    Secondary,
    Reduced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProminenceClass {
    Weak,
    Content,
    Focused,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WordProsodyInfo {
    pub word_index: usize,
    pub text_range: std::ops::Range<usize>,
    pub phoneme_range: std::ops::Range<usize>,
    pub lexical_stress: Vec<Stress>,
    pub orthographic_emphasis: OrthographicEmphasisKind,
    pub prominence_class: ProminenceClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhraseBoundaryKind {
    None,
    Word,
    MinorPhrase,
    MajorPhrase,
    PossibleFinal,
    FinalFalling,
    FinalRising,
    Exclamation,
    Parenthetical,
    Vocative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PauseReason {
    WordBoundary,
    PhraseBoundary,
    SentenceBoundary,
    ParagraphBoundary,
    ExplicitBreak,
    Breath,
    Repair,
    DirectAddressBoundary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpeechToken {
    TextWord(String),
    Boundary(PhraseBoundaryKind),
    PhoneticOverride {
        display: String,
        phones: Vec<String>,
        stress: Vec<Stress>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RiperStyleProfile {
    pub base_rate: f32,
    pub pitch_range: f32,
    pub energy: f32,
    pub pause_scale: f32,
    pub focus_strength: f32,
    pub finality_strength: f32,
}

impl RiperStyleProfile {
    pub fn preset(name: &str) -> Option<Self> {
        match name {
            "neutral" => Some(Self {
                base_rate: 1.0,
                pitch_range: 1.0,
                energy: 1.0,
                pause_scale: 1.0,
                focus_strength: 1.0,
                finality_strength: 1.0,
            }),
            "warm" => Some(Self {
                base_rate: 0.97,
                pitch_range: 0.95,
                energy: 1.04,
                pause_scale: 1.08,
                focus_strength: 1.05,
                finality_strength: 1.0,
            }),
            "dry" => Some(Self {
                base_rate: 1.03,
                pitch_range: 0.82,
                energy: 0.92,
                pause_scale: 0.94,
                focus_strength: 0.9,
                finality_strength: 1.03,
            }),
            "excited" => Some(Self {
                base_rate: 1.12,
                pitch_range: 1.2,
                energy: 1.18,
                pause_scale: 0.86,
                focus_strength: 1.24,
                finality_strength: 1.08,
            }),
            "solemn" => Some(Self {
                base_rate: 0.88,
                pitch_range: 0.78,
                energy: 0.9,
                pause_scale: 1.24,
                focus_strength: 0.94,
                finality_strength: 1.14,
            }),
            "storytelling" => Some(Self {
                base_rate: 0.95,
                pitch_range: 1.1,
                energy: 1.06,
                pause_scale: 1.16,
                focus_strength: 1.14,
                finality_strength: 1.04,
            }),
            "fast_backchannel" | "fastbackchannel" => Some(Self {
                base_rate: 1.25,
                pitch_range: 0.92,
                energy: 1.02,
                pause_scale: 0.7,
                focus_strength: 0.88,
                finality_strength: 0.84,
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProsodyRealizationStatus {
    Requested,
    Realized,
    Approximated,
    Advisory,
    Ignored,
    Deferred,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhoLikeDiagnosticEntry {
    pub word: String,
    pub span: Option<String>,
    pub phoneme: String,
    pub duration_hint: Option<u64>,
    pub stress: Vec<Stress>,
    pub accent: Option<String>,
    pub boundary: Option<PhraseBoundaryKind>,
    pub pause: Option<u64>,
    pub classification: Option<String>,
    pub pause_behavior: Option<String>,
    pub pitch_hint: Option<String>,
    pub reduction: Option<ReductionDiagnostic>,
    pub capitalization_effect: Option<String>,
    pub realization_status: ProsodyRealizationStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhoLikeDiagnostics {
    pub candidate_id: u64,
    pub entries: Vec<PhoLikeDiagnosticEntry>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn style_profile_presets_cover_audit_names() {
        for name in [
            "neutral",
            "warm",
            "dry",
            "excited",
            "solemn",
            "storytelling",
            "fast_backchannel",
        ] {
            assert!(
                RiperStyleProfile::preset(name).is_some(),
                "missing style profile preset `{name}`"
            );
        }
    }
}
