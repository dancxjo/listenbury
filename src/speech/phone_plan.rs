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
        let lexical_stress = lexical_stress_by_source_phone(unit);
        let word_final_phones = word_final_source_phones(unit);
        let mut phones = Vec::new();
        let mut cursor_ms = 0.0_f32;

        for (source_index, phoneme) in unit.phonemes.phonemes.iter().enumerate() {
            if !is_speakable_plan_symbol(&phoneme.0) {
                continue;
            }

            let duration_ms =
                shaped_phone_duration_ms(unit, &lexical_stress, &word_final_phones, source_index);
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

fn lexical_stress_by_source_phone(unit: &PhonemizedUnit) -> Vec<Option<LexicalStressLevel>> {
    let mut stress = vec![None; unit.phonemes.phonemes.len()];
    for target in &unit.lexical_stress {
        if let Some(slot) = stress.get_mut(target.phoneme_index) {
            *slot = Some(target.stress);
        }
    }
    stress
}

fn word_final_source_phones(unit: &PhonemizedUnit) -> Vec<bool> {
    let mut word_final = vec![false; unit.phonemes.phonemes.len()];
    for target in &unit.word_targets {
        let source_index = target.phoneme_range.clone().rev().find(|index| {
            unit.phonemes
                .phonemes
                .get(*index)
                .is_some_and(|phoneme| is_speakable_plan_symbol(&phoneme.0))
        });
        if let Some(source_index) = source_index {
            if let Some(slot) = word_final.get_mut(source_index) {
                *slot = true;
            }
        }
    }
    word_final
}

fn shaped_phone_duration_ms(
    unit: &PhonemizedUnit,
    lexical_stress: &[Option<LexicalStressLevel>],
    word_final_phones: &[bool],
    source_index: usize,
) -> f32 {
    let symbol = unit.phonemes.phonemes[source_index].0.as_str();
    let class = PhoneDurationClass::for_symbol(symbol);
    let mut duration = class.base_duration_ms();

    match lexical_stress.get(source_index).copied().flatten() {
        Some(LexicalStressLevel::Primary) => duration *= 1.25,
        Some(LexicalStressLevel::Secondary) => duration *= 1.10,
        Some(LexicalStressLevel::Unstressed) if class.is_vocalic() => duration *= 0.75,
        _ => {}
    }

    if word_final_phones
        .get(source_index)
        .copied()
        .unwrap_or(false)
        && class.is_sonorant()
    {
        duration *= 1.15;
    }

    if is_phrase_final_phone(unit, source_index) {
        duration *= 1.30;
    }

    if class == PhoneDurationClass::Stop && next_speakable_phone_is_stop(unit, source_index) {
        duration *= 0.75;
    }

    duration.round().clamp(35.0, 240.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhoneDurationClass {
    Vowel,
    Diphthong,
    Approximant,
    Nasal,
    Fricative,
    Stop,
    Affricate,
    H,
    Unknown,
}

impl PhoneDurationClass {
    fn for_symbol(symbol: &str) -> Self {
        match strip_arpabet_stress(symbol).as_str() {
            "AA" | "AE" | "AH" | "AO" | "EH" | "ER" | "IH" | "IY" | "UH" | "UW" => Self::Vowel,
            "AW" | "AY" | "EY" | "OW" | "OY" => Self::Diphthong,
            "L" | "R" | "W" | "Y" => Self::Approximant,
            "M" | "N" | "NG" => Self::Nasal,
            "F" | "V" | "TH" | "DH" | "S" | "Z" | "SH" | "ZH" => Self::Fricative,
            "P" | "B" | "T" | "D" | "K" | "G" => Self::Stop,
            "CH" | "JH" => Self::Affricate,
            "HH" => Self::H,
            _ => Self::Unknown,
        }
    }

    fn base_duration_ms(self) -> f32 {
        match self {
            Self::Vowel => 120.0,
            Self::Diphthong => 145.0,
            Self::Approximant => 90.0,
            Self::Nasal => 85.0,
            Self::Fricative => 75.0,
            Self::Stop => 55.0,
            Self::Affricate => 80.0,
            Self::H => 55.0,
            Self::Unknown => 90.0,
        }
    }

    fn is_vocalic(self) -> bool {
        matches!(self, Self::Vowel | Self::Diphthong)
    }

    fn is_sonorant(self) -> bool {
        matches!(
            self,
            Self::Vowel | Self::Diphthong | Self::Approximant | Self::Nasal
        )
    }
}

fn strip_arpabet_stress(symbol: &str) -> String {
    symbol
        .strip_suffix(['0', '1', '2'])
        .filter(|base| is_stress_marked_vowel(base))
        .unwrap_or(symbol)
        .to_ascii_uppercase()
}

fn is_phrase_final_phone(unit: &PhonemizedUnit, source_index: usize) -> bool {
    for phoneme in unit.phonemes.phonemes.iter().skip(source_index + 1) {
        match phoneme.0.as_str() {
            "|" => return true,
            symbol if is_speakable_plan_symbol(symbol) => return false,
            _ => {}
        }
    }
    true
}

fn next_speakable_phone_is_stop(unit: &PhonemizedUnit, source_index: usize) -> bool {
    unit.phonemes
        .phonemes
        .iter()
        .skip(source_index + 1)
        .find(|phoneme| is_speakable_plan_symbol(&phoneme.0))
        .is_some_and(|phoneme| {
            PhoneDurationClass::for_symbol(&phoneme.0) == PhoneDurationClass::Stop
        })
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

    #[test]
    fn phone_plan_shapes_duration_by_stress_class_and_position() {
        let plan = PhonePlan::from_text_with_riper_g2p("Hello, my ragtime gal.")
            .expect("acceptance phrase should produce a phone plan");

        let hello = &plan.words[0];
        assert_eq!(hello.text, "hello");
        let hello_hh = &plan.phones[hello.start_phone];
        let hello_ah = &plan.phones[hello.start_phone + 1];
        let hello_l = &plan.phones[hello.start_phone + 2];
        let hello_ow = &plan.phones[hello.start_phone + 3];

        assert_eq!(hello_hh.phone, "hh");
        assert_eq!(hello_hh.duration_ms, 55.0);
        assert_eq!(hello_ah.phone, "ah");
        assert_eq!(hello_ah.duration_ms, 90.0);
        assert_eq!(hello_l.phone, "l");
        assert_eq!(hello_l.duration_ms, 90.0);
        assert_eq!(hello_ow.phone, "ow");
        assert!(
            hello_ow.duration_ms > hello_ah.duration_ms,
            "stressed word-final diphthong should be longer than unstressed vowel: {:?}",
            plan.phones
        );

        let ragtime = plan
            .words
            .iter()
            .find(|word| word.text == "ragtime")
            .expect("ragtime word plan");
        let ragtime_ae = plan.phones[ragtime.start_phone..ragtime.end_phone]
            .iter()
            .find(|phone| phone.phone == "ae")
            .expect("ragtime primary vowel");
        let ragtime_t = plan.phones[ragtime.start_phone..ragtime.end_phone]
            .iter()
            .find(|phone| phone.phone == "t")
            .expect("ragtime stop");
        assert_eq!(ragtime_ae.duration_ms, 150.0);
        assert_eq!(ragtime_t.duration_ms, 55.0);

        let gal = plan
            .words
            .iter()
            .find(|word| word.text == "gal")
            .expect("gal word plan");
        let gal_ae = plan.phones[gal.start_phone..gal.end_phone]
            .iter()
            .find(|phone| phone.phone == "ae")
            .expect("gal primary vowel");
        let final_l = &plan.phones[gal.end_phone - 1];
        assert_eq!(gal_ae.duration_ms, 150.0);
        assert_eq!(final_l.phone, "l");
        assert!(
            final_l.duration_ms > hello_l.duration_ms,
            "phrase-final word-final sonorant should lengthen: {:?}",
            plan.phones
        );

        let distinct_durations = plan
            .phones
            .iter()
            .map(|phone| phone.duration_ms as u32)
            .collect::<std::collections::BTreeSet<_>>();
        assert!(
            distinct_durations.len() >= 5,
            "acceptance phrase should no longer be near-uniform: {:?}",
            plan.phones
        );
    }
}
