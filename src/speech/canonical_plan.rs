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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalSpeechPlan {
    pub utterance_id: String,
    pub segments: Vec<CanonicalSpeechSegment>,
    pub breath_groups: Vec<BreathGroup>,
    pub metadata: CanonicalSpeechPlanMetadata,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalSpeechSegment {
    pub word: String,
    pub t0: f64,
    pub t1: f64,
    pub phones: Vec<CanonicalSpeechPhone>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub break_after: Option<CanonicalSpeechBreak>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalSpeechPhone {
    pub symbol: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<FeatureBundle>,
    pub timing: CanonicalPhoneTiming,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub syllable_role: Option<CanonicalSyllableRole>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stress: Option<CanonicalStress>,
    #[serde(default)]
    pub nucleus: bool,
    #[serde(default)]
    pub pitch: CanonicalPitchPlan,
    #[serde(default)]
    pub energy: CanonicalEnergyPlan,
    #[serde(default)]
    pub articulation: CanonicalArticulationHints,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<NoteTarget>,
    pub provenance: CanonicalPhoneProvenance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalSpeechBreak {
    pub millis: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<BreakReason>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalPhoneTiming {
    pub t0: f64,
    pub t1: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_duration_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalPitchPlan {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub targets: Vec<CanonicalPitchTarget>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalPitchTarget {
    pub percent: u8,
    pub hz: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalEnergyPlan {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalArticulationHints {
    #[serde(default)]
    pub legato: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalSyllableRole {
    Onset,
    Nucleus,
    Coda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalStress {
    Primary,
    Secondary,
    Unstressed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalPhoneProvenance {
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
pub struct CanonicalSpeechPlanMetadata {
    pub source: CanonicalSpeechPlanSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalSpeechPlanSource {
    ProsodyTimingPlan,
    Other,
}

pub fn canonical_speech_plan_from_prosody_timing(plan: &ProsodyTimingPlan) -> CanonicalSpeechPlan {
    let segments = plan
        .segments
        .iter()
        .map(|segment| CanonicalSpeechSegment {
            word: segment.word.clone(),
            t0: segment.t0,
            t1: segment.t1,
            phones: segment
                .phones
                .iter()
                .map(|phone| CanonicalSpeechPhone {
                    symbol: phone.p.clone(),
                    features: None,
                    timing: CanonicalPhoneTiming {
                        t0: phone.t0,
                        t1: phone.t1,
                        target_duration_ms: phone.pace_target_ms,
                    },
                    syllable_role: if phone.nucleus {
                        Some(CanonicalSyllableRole::Nucleus)
                    } else {
                        None
                    },
                    stress: None,
                    nucleus: phone.nucleus,
                    pitch: CanonicalPitchPlan::default(),
                    energy: CanonicalEnergyPlan::default(),
                    articulation: CanonicalArticulationHints::default(),
                    note: None,
                    provenance: CanonicalPhoneProvenance::ForcedAlignment,
                })
                .collect(),
            break_after: segment.break_hint_ms.map(|millis| CanonicalSpeechBreak {
                millis,
                reason: segment.break_reason,
            }),
        })
        .collect();

    CanonicalSpeechPlan {
        utterance_id: plan.utterance_id.clone(),
        segments,
        breath_groups: plan.breath_groups.clone(),
        metadata: CanonicalSpeechPlanMetadata {
            source: CanonicalSpeechPlanSource::ProsodyTimingPlan,
        },
    }
}

pub fn canonical_speech_plan_to_piper_timing(plan: &CanonicalSpeechPlan) -> PiperTimingPlan {
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

pub fn canonical_speech_plan_to_phone_timed_plan(
    plan: &CanonicalSpeechPlan,
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
                    mbrola_phone.pitch_targets = vec![
                        MbrolaPitchTarget {
                            percent: 0,
                            hz: 125.0,
                        },
                        MbrolaPitchTarget {
                            percent: 60,
                            hz: 135.0,
                        },
                        MbrolaPitchTarget {
                            percent: 100,
                            hz: 128.0,
                        },
                    ];
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

pub fn canonical_speech_plan_to_piper_phoneme_sequence(
    plan: &CanonicalSpeechPlan,
) -> PiperPhonemeSequence {
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
        CentsOffset, MidiNote, NoteArticulation, NoteDuration, PitchTarget, TimePoint, Velocity,
    };
    use crate::speech::prosody_timing::{BreakReason, BreathGroup, ProsodyPhone, ProsodySegment};
    use crate::voice::mbrola::MbrolaSymbolMap;

    #[test]
    fn canonical_from_prosody_preserves_timing_nucleus_breaks_and_breath_groups() {
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

        let canonical = canonical_speech_plan_from_prosody_timing(&plan);

        assert_eq!(canonical.utterance_id, "utt");
        assert_eq!(canonical.segments[0].word, "hello");
        assert_eq!(canonical.segments[0].phones[0].symbol, "EH1");
        assert_eq!(canonical.segments[0].phones[0].timing.t0, 0.1);
        assert_eq!(canonical.segments[0].phones[0].timing.t1, 0.25);
        assert_eq!(
            canonical.segments[0].phones[0].timing.target_duration_ms,
            Some(162)
        );
        assert!(canonical.segments[0].phones[0].nucleus);
        assert_eq!(
            canonical.segments[0].break_after,
            Some(CanonicalSpeechBreak {
                millis: 160,
                reason: Some(BreakReason::Punctuation)
            })
        );
        assert_eq!(
            canonical.breath_groups,
            vec![BreathGroup { t0: 0.0, t1: 0.4 }]
        );
    }

    #[test]
    fn canonical_lowers_to_mbrola_and_piper_without_reordering() {
        let mut canonical = CanonicalSpeechPlan {
            utterance_id: "utt".to_string(),
            segments: vec![CanonicalSpeechSegment {
                word: "hello".to_string(),
                t0: 0.0,
                t1: 0.4,
                phones: vec![CanonicalSpeechPhone {
                    symbol: "EH1".to_string(),
                    features: Some(FeatureBundle::unknown_phone()),
                    timing: CanonicalPhoneTiming {
                        t0: 0.1,
                        t1: 0.25,
                        target_duration_ms: Some(162),
                    },
                    syllable_role: Some(CanonicalSyllableRole::Nucleus),
                    stress: Some(CanonicalStress::Primary),
                    nucleus: true,
                    pitch: CanonicalPitchPlan::default(),
                    energy: CanonicalEnergyPlan::default(),
                    articulation: CanonicalArticulationHints::default(),
                    note: Some(NoteTarget {
                        pitch: PitchTarget::with_tuning(
                            MidiNote::new(64).unwrap(),
                            CentsOffset::default(),
                        ),
                        onset: TimePoint::from_millis(0),
                        duration: NoteDuration::from_millis(200),
                        velocity: Velocity::new(96).unwrap(),
                        articulation: NoteArticulation::Legato,
                    }),
                    provenance: CanonicalPhoneProvenance::ManualSingingNote,
                }],
                break_after: Some(CanonicalSpeechBreak {
                    millis: 160,
                    reason: Some(BreakReason::Punctuation),
                }),
            }],
            breath_groups: vec![BreathGroup { t0: 0.0, t1: 0.4 }],
            metadata: CanonicalSpeechPlanMetadata {
                source: CanonicalSpeechPlanSource::Other,
            },
        };

        let mbrola =
            canonical_speech_plan_to_phone_timed_plan(&canonical, &MbrolaSymbolMap::us1_starter())
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

        let piper = canonical_speech_plan_to_piper_phoneme_sequence(&canonical);
        assert_eq!(piper.phonemes, vec![PiperPhoneme("EH1".to_string())]);

        canonical.segments[0].phones[0]
            .pitch
            .targets
            .push(CanonicalPitchTarget {
                percent: 50,
                hz: 200.0,
            });
        let with_pitch =
            canonical_speech_plan_to_phone_timed_plan(&canonical, &MbrolaSymbolMap::identity())
                .unwrap();
        assert_eq!(with_pitch.phones[0].pitch_targets.len(), 1);
        assert_eq!(with_pitch.phones[0].pitch_targets[0].hz, 200.0);
        assert!(canonical.segments[0].phones[0].features.is_some());
        assert!(canonical.segments[0].phones[0].note.is_some());
    }
}
