use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::mouth::riper::{
    LexicalStressLevel, PhonemizedUnit, SimpleEnglishG2p,
    morphophonology::{PhonologicalStress, StressPattern},
};
use crate::speech::synthetic_plan::PitchTarget;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PhonePlan {
    pub source_text: String,
    pub words: Vec<WordPlan>,
    pub phones: Vec<PhoneSpan>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct WordPlan {
    pub text: String,
    pub start_phone: usize,
    pub end_phone: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub phones: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stress: Option<StressPattern>,
    pub lexical_status: LexicalStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PhoneSpan {
    pub phone: String,
    pub start_ms: f32,
    pub duration_ms: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pitch: Option<PitchTarget>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub energy: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub syllable_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub word_index: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalStatus {
    Resolved,
    Ambiguous,
    NeedsRightContext,
}

impl PhonePlan {
    pub fn from_text_with_riper_g2p(text: &str) -> anyhow::Result<Self> {
        let unit = SimpleEnglishG2p::default()
            .phonemize_unit(text)
            .with_context(|| format!("failed to build phone plan for `{text}`"))?;
        Ok(Self::from_riper_phonemized_unit(text, &unit))
    }

    pub fn from_riper_phonemized_unit(source_text: &str, unit: &PhonemizedUnit) -> Self {
        let mut source_to_plan_index = vec![None; unit.phonemes.phonemes.len()];
        let mut phones = Vec::new();
        let mut cursor_ms = 0.0_f32;

        for (source_index, phoneme) in unit.phonemes.phonemes.iter().enumerate() {
            if !is_speakable_plan_symbol(&phoneme.0) {
                continue;
            }

            let duration_ms = unit
                .length_hints
                .get(source_index)
                .map(|hint| duration_ms_for_length_class(hint.class))
                .unwrap_or(100.0);
            let plan_index = phones.len();
            source_to_plan_index[source_index] = Some(plan_index);
            phones.push(PhoneSpan {
                phone: normalize_phone_symbol(&phoneme.0),
                start_ms: cursor_ms,
                duration_ms,
                pitch: None,
                energy: None,
                syllable_index: None,
                word_index: unit.phoneme_to_word.get(source_index).copied().flatten(),
            });
            cursor_ms += duration_ms;
        }

        let words = unit
            .word_targets
            .iter()
            .map(|target| {
                let phone_indices = target
                    .phoneme_range
                    .clone()
                    .filter_map(|source_index| {
                        source_to_plan_index.get(source_index).copied().flatten()
                    })
                    .collect::<Vec<_>>();
                let start_phone = phone_indices.first().copied().unwrap_or(phones.len());
                let end_phone = phone_indices
                    .last()
                    .map(|index| index.saturating_add(1))
                    .unwrap_or(start_phone);
                let word_phones = phone_indices
                    .iter()
                    .filter_map(|index| phones.get(*index))
                    .map(|phone| phone.phone.clone())
                    .collect::<Vec<_>>();

                WordPlan {
                    text: target.normalized_text.clone(),
                    start_phone,
                    end_phone,
                    phones: word_phones,
                    stress: stress_pattern_for_word(unit, target.phoneme_range.clone()),
                    lexical_status: LexicalStatus::Resolved,
                }
            })
            .collect();

        Self {
            source_text: source_text.to_string(),
            words,
            phones,
        }
    }
}

fn is_speakable_plan_symbol(symbol: &str) -> bool {
    !matches!(symbol, " " | "|")
}

fn normalize_phone_symbol(symbol: &str) -> String {
    let base = symbol
        .strip_suffix(['0', '1', '2'])
        .filter(|base| is_stress_marked_vowel(base))
        .unwrap_or(symbol);
    base.to_ascii_lowercase()
}

fn is_stress_marked_vowel(symbol: &str) -> bool {
    matches!(
        symbol,
        "AA" | "AE"
            | "AH"
            | "AO"
            | "AW"
            | "AY"
            | "EH"
            | "ER"
            | "EY"
            | "IH"
            | "IY"
            | "OW"
            | "OY"
            | "UH"
            | "UW"
    )
}

fn duration_ms_for_length_class(class: crate::mouth::riper::PhoneLengthClass) -> f32 {
    match class {
        crate::mouth::riper::PhoneLengthClass::Short => 70.0,
        crate::mouth::riper::PhoneLengthClass::Medium => 120.0,
        crate::mouth::riper::PhoneLengthClass::Long => 145.0,
    }
}

fn stress_pattern_for_word(
    unit: &PhonemizedUnit,
    range: std::ops::Range<usize>,
) -> Option<StressPattern> {
    let mut levels_by_phone = vec![None; range.len()];
    for stress in &unit.lexical_stress {
        if range.contains(&stress.phoneme_index) {
            levels_by_phone[stress.phoneme_index - range.start] = Some(match stress.stress {
                LexicalStressLevel::Primary => PhonologicalStress::Primary,
                LexicalStressLevel::Secondary => PhonologicalStress::Secondary,
                LexicalStressLevel::Unstressed => PhonologicalStress::Unstressed,
            });
        }
    }

    levels_by_phone
        .iter()
        .any(Option::is_some)
        .then_some(StressPattern { levels_by_phone })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn riper_g2p_phone_plan_keeps_word_phone_spans() {
        let plan = PhonePlan::from_text_with_riper_g2p("Don't be jealous of my boogy.")
            .expect("fallback G2P should produce an inspectable phone plan");

        assert_eq!(plan.source_text, "Don't be jealous of my boogy.");
        assert_eq!(plan.words[0].text, "don't");
        assert_eq!(plan.words[0].start_phone, 0);
        assert_eq!(plan.words[0].phones, ["d", "ow", "n", "t"]);
        assert!(plan.phones.iter().all(|phone| phone.duration_ms > 0.0));
        assert_eq!(plan.phones[0].phone, "d");
        assert_eq!(plan.phones[0].word_index, Some(0));
    }
}
