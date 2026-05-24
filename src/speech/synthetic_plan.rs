use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::linguistic::inventory::FeatureBundle;
use crate::mouth::riper::{PiperPhoneme, PiperPhonemeSequence};
use crate::prosody::note_target::NoteTarget;
use crate::speech::prosody_timing::{
    BreakReason, BreathGroup, PiperTimingBreak, PiperTimingPhone, PiperTimingPlan,
    ProsodyTimingPlan,
};
use crate::voice::mbrola::{MbrolaPhone, MbrolaPitchTarget, MbrolaSymbolMap, PhoneTimedPlan};

const DEFAULT_NUCLEUS_PITCH_TARGETS: &[(u8, f32)] = &[(0, 125.0), (60, 135.0), (100, 128.0)];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SyntheticPlan {
    pub utterance_id: String,
    pub segments: Vec<SyntheticSegment>,
    pub breath_groups: Vec<BreathGroup>,
    pub metadata: SyntheticPlanMetadata,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SyntheticSegment {
    pub word: String,
    pub t0: f64,
    pub t1: f64,
    pub phones: Vec<SyntheticPhone>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub break_after: Option<SyntheticBreak>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SyntheticPhone {
    pub symbol: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<FeatureBundle>,
    pub timing: PhoneTiming,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub syllable_role: Option<SyllableRole>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stress: Option<Stress>,
    #[serde(default)]
    pub nucleus: bool,
    #[serde(default)]
    pub pitch: PitchPlan,
    #[serde(default)]
    pub energy: EnergyPlan,
    #[serde(default)]
    pub articulation: ArticulationHints,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<NoteTarget>,
    pub provenance: PhoneProvenance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SyntheticBreak {
    pub millis: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<BreakReason>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct PhoneTiming {
    pub t0: f64,
    pub t1: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_duration_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct PitchPlan {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub targets: Vec<PitchTarget>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PitchTarget {
    pub percent: u8,
    pub hz: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct EnergyPlan {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ArticulationHints {
    #[serde(default)]
    pub legato: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyllableRole {
    Onset,
    Nucleus,
    Coda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stress {
    Primary,
    Secondary,
    Unstressed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhoneProvenance {
    ForcedAlignment,
    G2p,
    EspeakPiperMapping,
    ManualSingingNote,
    RewriteRepairLayer,
    CachedDiphone,
    GeneratedDiphone,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SyntheticPlanMetadata {
    pub source: SyntheticPlanSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticPlanSource {
    ProsodyTimingPlan,
    Other,
}

pub fn synthetic_plan_from_prosody_timing(plan: &ProsodyTimingPlan) -> SyntheticPlan {
    let segments = plan
        .segments
        .iter()
        .map(|segment| SyntheticSegment {
            word: segment.word.clone(),
            t0: segment.t0,
            t1: segment.t1,
            phones: segment
                .phones
                .iter()
                .map(|phone| SyntheticPhone {
                    symbol: phone.p.clone(),
                    features: None,
                    timing: PhoneTiming {
                        t0: phone.t0,
                        t1: phone.t1,
                        target_duration_ms: phone.pace_target_ms,
                    },
                    syllable_role: if phone.nucleus {
                        Some(SyllableRole::Nucleus)
                    } else {
                        None
                    },
                    stress: None,
                    nucleus: phone.nucleus,
                    pitch: PitchPlan::default(),
                    energy: EnergyPlan::default(),
                    articulation: ArticulationHints::default(),
                    note: None,
                    provenance: PhoneProvenance::ForcedAlignment,
                })
                .collect(),
            break_after: segment.break_hint_ms.map(|millis| SyntheticBreak {
                millis,
                reason: segment.break_reason,
            }),
        })
        .collect();

    SyntheticPlan {
        utterance_id: plan.utterance_id.clone(),
        segments,
        breath_groups: plan.breath_groups.clone(),
        metadata: SyntheticPlanMetadata {
            source: SyntheticPlanSource::ProsodyTimingPlan,
        },
    }
}

pub fn synthetic_plan_to_piper_timing(plan: &SyntheticPlan) -> PiperTimingPlan {
    let mut phonemes = Vec::new();
    let mut breaks = Vec::new();

    for (word_index, segment) in plan.segments.iter().enumerate() {
        for phone in &segment.phones {
            phonemes.push(PiperTimingPhone {
                p: phone.symbol.clone(),
                source_word_index: word_index,
                target_duration_ms: phone
                    .timing
                    .target_duration_ms
                    .unwrap_or_else(|| seconds_to_ms((phone.timing.t1 - phone.timing.t0).max(0.0))),
                nucleus: phone.nucleus,
            });
        }

        if let Some(break_after) = &segment.break_after {
            if let Some(reason) = break_after.reason {
                breaks.push(PiperTimingBreak {
                    after_word_index: word_index,
                    millis: break_after.millis,
                    reason,
                });
            }
        }
    }

    PiperTimingPlan { phonemes, breaks }
}

pub fn synthetic_plan_to_phone_timed_plan(
    plan: &SyntheticPlan,
    symbol_map: &MbrolaSymbolMap,
) -> Result<PhoneTimedPlan> {
    let mut phones = Vec::new();

    for segment in &plan.segments {
        for phone in &segment.phones {
            let duration_ms = phone
                .timing
                .target_duration_ms
                .unwrap_or_else(|| seconds_to_ms((phone.timing.t1 - phone.timing.t0).max(0.0)))
                .clamp(1, u64::from(u32::MAX)) as u32;
            let mut mbrola_phone =
                MbrolaPhone::new(symbol_map.map_phone(&phone.symbol)?, duration_ms);
            if phone.pitch.targets.is_empty() {
                if phone.nucleus {
                    mbrola_phone.pitch_targets = DEFAULT_NUCLEUS_PITCH_TARGETS
                        .iter()
                        .map(|(percent, hz)| MbrolaPitchTarget {
                            percent: *percent,
                            hz: *hz,
                        })
                        .collect();
                }
            } else {
                mbrola_phone.pitch_targets = phone
                    .pitch
                    .targets
                    .iter()
                    .map(|target| MbrolaPitchTarget {
                        percent: target.percent,
                        hz: target.hz,
                    })
                    .collect();
            }
            phones.push(mbrola_phone);
        }

        if let Some(break_after) = &segment.break_after {
            phones.push(MbrolaPhone::new(
                "_",
                break_after.millis.clamp(1, u64::from(u32::MAX)) as u32,
            ));
        }
    }

    Ok(PhoneTimedPlan::new(phones))
}

pub fn synthetic_plan_to_piper_phoneme_sequence(plan: &SyntheticPlan) -> PiperPhonemeSequence {
    let mut phonemes = Vec::new();
    for segment in &plan.segments {
        for phone in &segment.phones {
            phonemes.push(PiperPhoneme(phone.symbol.clone()));
        }
    }
    PiperPhonemeSequence { phonemes }
}

fn seconds_to_ms(seconds: f64) -> u64 {
    if !seconds.is_finite() || seconds <= 0.0 {
        return 0;
    }
    (seconds * 1000.0).round() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prosody::note_target::{
        CentsOffset, MidiNote, NoteArticulation, NoteDuration, PitchTarget as NotePitchTarget,
        TimePoint, Velocity,
    };
    use crate::speech::prosody_timing::{BreakReason, BreathGroup, ProsodyPhone, ProsodySegment};
    use crate::voice::mbrola::MbrolaSymbolMap;

    #[test]
    fn synthetic_plan_from_prosody_preserves_timing_nucleus_breaks_and_breath_groups() {
        let plan = ProsodyTimingPlan {
            utterance_id: "utt".to_string(),
            segments: vec![ProsodySegment {
                word: "hello".to_string(),
                t0: 0.0,
                t1: 0.4,
                phones: vec![ProsodyPhone {
                    p: "EH1".to_string(),
                    t0: 0.1,
                    t1: 0.25,
                    nucleus: true,
                    pace_target_ms: Some(162),
                }],
                break_hint_ms: Some(160),
                break_reason: Some(BreakReason::Punctuation),
            }],
            breath_groups: vec![BreathGroup { t0: 0.0, t1: 0.4 }],
        };

        let synthetic_plan = synthetic_plan_from_prosody_timing(&plan);

        assert_eq!(synthetic_plan.utterance_id, "utt");
        assert_eq!(synthetic_plan.segments[0].word, "hello");
        assert_eq!(synthetic_plan.segments[0].phones[0].symbol, "EH1");
        assert_eq!(synthetic_plan.segments[0].phones[0].timing.t0, 0.1);
        assert_eq!(synthetic_plan.segments[0].phones[0].timing.t1, 0.25);
        assert_eq!(
            synthetic_plan.segments[0].phones[0]
                .timing
                .target_duration_ms,
            Some(162)
        );
        assert!(synthetic_plan.segments[0].phones[0].nucleus);
        assert_eq!(
            synthetic_plan.segments[0].break_after,
            Some(SyntheticBreak {
                millis: 160,
                reason: Some(BreakReason::Punctuation)
            })
        );
        assert_eq!(
            synthetic_plan.breath_groups,
            vec![BreathGroup { t0: 0.0, t1: 0.4 }]
        );
    }

    #[test]
    fn synthetic_plan_lowers_to_mbrola_and_piper_without_reordering() {
        let mut synthetic_plan = SyntheticPlan {
            utterance_id: "utt".to_string(),
            segments: vec![SyntheticSegment {
                word: "hello".to_string(),
                t0: 0.0,
                t1: 0.4,
                phones: vec![SyntheticPhone {
                    symbol: "EH1".to_string(),
                    features: Some(FeatureBundle::unknown_phone()),
                    timing: PhoneTiming {
                        t0: 0.1,
                        t1: 0.25,
                        target_duration_ms: Some(162),
                    },
                    syllable_role: Some(SyllableRole::Nucleus),
                    stress: Some(Stress::Primary),
                    nucleus: true,
                    pitch: PitchPlan::default(),
                    energy: EnergyPlan::default(),
                    articulation: ArticulationHints::default(),
                    note: Some(NoteTarget {
                        pitch: NotePitchTarget::with_tuning(
                            MidiNote::new(64).unwrap(),
                            CentsOffset::default(),
                        ),
                        onset: TimePoint::from_millis(0),
                        duration: NoteDuration::from_millis(200),
                        velocity: Velocity::new(96).unwrap(),
                        articulation: NoteArticulation::Legato,
                    }),
                    provenance: PhoneProvenance::ManualSingingNote,
                }],
                break_after: Some(SyntheticBreak {
                    millis: 160,
                    reason: Some(BreakReason::Punctuation),
                }),
            }],
            breath_groups: vec![BreathGroup { t0: 0.0, t1: 0.4 }],
            metadata: SyntheticPlanMetadata {
                source: SyntheticPlanSource::Other,
            },
        };

        let mbrola =
            synthetic_plan_to_phone_timed_plan(&synthetic_plan, &MbrolaSymbolMap::us1_starter())
                .unwrap();
        assert_eq!(mbrola.phones[0].duration_ms, 162);
        assert_eq!(mbrola.phones[0].symbol, "E");
        assert_eq!(
            mbrola.phones[0].pitch_targets,
            vec![
                MbrolaPitchTarget {
                    percent: 0,
                    hz: 125.0
                },
                MbrolaPitchTarget {
                    percent: 60,
                    hz: 135.0
                },
                MbrolaPitchTarget {
                    percent: 100,
                    hz: 128.0
                },
            ]
        );
        assert_eq!(mbrola.phones[1], MbrolaPhone::new("_", 160));

        let piper = synthetic_plan_to_piper_phoneme_sequence(&synthetic_plan);
        assert_eq!(piper.phonemes, vec![PiperPhoneme("EH1".to_string())]);

        synthetic_plan.segments[0].phones[0]
            .pitch
            .targets
            .push(PitchTarget {
                percent: 50,
                hz: 200.0,
            });
        let with_pitch =
            synthetic_plan_to_phone_timed_plan(&synthetic_plan, &MbrolaSymbolMap::identity())
                .unwrap();
        assert_eq!(with_pitch.phones[0].pitch_targets.len(), 1);
        assert_eq!(with_pitch.phones[0].pitch_targets[0].hz, 200.0);
        assert!(synthetic_plan.segments[0].phones[0].features.is_some());
        assert!(synthetic_plan.segments[0].phones[0].note.is_some());
    }
}
