use serde::{Deserialize, Serialize};

use crate::audio::streaming_prosody::{
    ProsodyAccentCandidate, ProsodyPauseBoundary, ProsodyPhraseCandidate, ProsodyProvenance,
    StreamingProsodyFrame, StreamingProsodyUpdate,
};
use crate::mouth::riper::g2p::{LexicalStressLevel, PhonemizedUnit};
use crate::mouth::riper::prosody_audit::PauseReason;
use crate::mouth::riper::prosody_controls::{
    PiperBoundaryOverride, PiperPauseOverride, PiperPhonemeDurationOverride, PiperProsodyControls,
    PiperSynthesisDiagnostics, ProsodyControlStatus,
};
use crate::mouth::riper::prosody_planner::{
    PauseOp, PauseStrengthClass, ProsodyAccentKind, ProsodyBoundaryHintOp, ProsodyEnergyClass,
    ProsodyOp, ProsodyPitchShape, ProsodyRateClass, ProsodyTarget,
};
use crate::mouth::riper::text::ProsodyCommitment;
use crate::mouth::riper::{
    AnalysisClaim, AnalysisSourceKind, AnalysisTarget, ClaimKind, ClaimValue,
};
use crate::word::TranscriptWord;

/// Allow small ASR/prosody timing disagreement when matching a pause back to
/// the preceding word boundary in offline echo mode.
const PAUSE_MATCH_TOLERANCE_MS: u64 = 80;
/// Treat quarter-second pauses as strong enough to resemble a sentence-level
/// break instead of an intra-phrase hesitation.
const STRONG_PAUSE_MS: u64 = 250;
/// Stretch accented phonemes toward a clearly noticeable but still speech-like
/// advisory duration hint when exporting to current Riper controls.
const ACCENT_DURATION_HINT_MS: u64 = 120;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EchoWordProsodyObservation {
    pub word_index: usize,
    pub text: String,
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
    pub confidence: Option<f32>,
    pub mean_loudness_dbfs: Option<f32>,
    pub mean_pitch_hz: Option<f32>,
    pub speech_rate_proxy: Option<f32>,
    pub pause_after_ms: Option<u64>,
    pub phrase_boundary_after: bool,
    pub accent_peak_ms: Option<u64>,
    pub phone_timing_hints_ms: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EchoProsodyObservation {
    pub source_text: String,
    pub source_duration_ms: u64,
    pub words: Vec<EchoWordProsodyObservation>,
    pub loudness_contour_dbfs: Vec<f32>,
    pub pitch_contour_hz: Vec<Option<f32>>,
    pub pauses: Vec<ProsodyPauseBoundary>,
    pub phrase_boundaries: Vec<ProsodyPhraseCandidate>,
    pub accent_peaks: Vec<ProsodyAccentCandidate>,
    pub speech_rate: f32,
    pub confidence: f32,
    pub provenance: ProsodyProvenance,
    pub claims: Vec<AnalysisClaim>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EchoProsodyPlan {
    pub claims: Vec<AnalysisClaim>,
    pub prosody_ops: Vec<ProsodyOp>,
    pub controls: PiperProsodyControls,
    pub pause_after_word_indices: Vec<usize>,
    pub accent_word_indices: Vec<usize>,
    pub advisory_controls: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EchoMatchedPause {
    pub after_word_index: usize,
    pub source_pause_ms: u64,
    pub control_status: ProsodyControlStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EchoMatchedAccent {
    pub word_index: usize,
    pub source_peak_ms: u64,
    pub control_status: ProsodyControlStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EchoComparisonRecord {
    pub source_text: String,
    pub source_duration_ms: u64,
    pub riper_duration_ms: u64,
    pub matched_pauses: Vec<EchoMatchedPause>,
    pub matched_accents: Vec<EchoMatchedAccent>,
    pub unrealized_controls: Vec<String>,
}

impl EchoProsodyObservation {
    pub fn from_streaming_updates(
        source_text: impl Into<String>,
        transcript_words: &[TranscriptWord],
        updates: &[StreamingProsodyUpdate],
    ) -> Self {
        let source_text = source_text.into();
        let frames: Vec<&StreamingProsodyFrame> =
            updates.iter().map(|update| &update.frame).collect();
        let pauses: Vec<_> = updates
            .iter()
            .filter_map(|update| update.pause.clone())
            .collect();
        let phrase_boundaries: Vec<_> = updates
            .iter()
            .filter_map(|update| update.phrase_candidate.clone())
            .collect();
        let accent_peaks: Vec<_> = updates
            .iter()
            .filter_map(|update| update.accent_candidate.clone())
            .collect();
        let loudness_contour_dbfs = frames.iter().map(|frame| frame.loudness_dbfs).collect();
        let pitch_contour_hz = frames.iter().map(|frame| frame.pitch_hz).collect();
        let speech_rate =
            mean_f32(frames.iter().map(|frame| frame.speech_rate_proxy)).unwrap_or(0.0);
        let confidence = mean_f32(frames.iter().map(|frame| frame.confidence)).unwrap_or(0.0);
        let provenance = updates
            .last()
            .map(|update| update.model.provenance)
            .unwrap_or(ProsodyProvenance::Provisional);
        let source_duration_ms = transcript_words
            .iter()
            .filter_map(|word| word.end_ms)
            .chain(frames.iter().map(|frame| frame.frame_end_ms))
            .chain(pauses.iter().map(|pause| pause.end_ms))
            .max()
            .unwrap_or_default();

        let words = transcript_words
            .iter()
            .enumerate()
            .map(|(word_index, word)| {
                let overlapping_frames = word_window_frames(word, &frames);
                let mean_loudness_dbfs =
                    mean_f32(overlapping_frames.iter().map(|frame| frame.loudness_dbfs));
                let mean_pitch_hz =
                    mean_f32(overlapping_frames.iter().filter_map(|frame| frame.pitch_hz));
                let speech_rate_proxy = mean_f32(
                    overlapping_frames
                        .iter()
                        .map(|frame| frame.speech_rate_proxy),
                );
                let accent_peak_ms =
                    word.start_ms
                        .zip(word.end_ms)
                        .and_then(|(start_ms, end_ms)| {
                            accent_peaks
                                .iter()
                                .find(|accent| accent.at_ms >= start_ms && accent.at_ms <= end_ms)
                                .map(|accent| accent.at_ms)
                        });
                let pause_after_ms = word.end_ms.and_then(|end_ms| {
                    pause_after_word(word_index, end_ms, transcript_words, &pauses)
                });
                let phrase_boundary_after = word.end_ms.is_some_and(|end_ms| {
                    phrase_boundaries.iter().any(|boundary| {
                        boundary.end_ms.abs_diff(end_ms) <= PAUSE_MATCH_TOLERANCE_MS
                    })
                });

                EchoWordProsodyObservation {
                    word_index,
                    text: word.text.clone(),
                    start_ms: word.start_ms,
                    end_ms: word.end_ms,
                    confidence: word.confidence,
                    mean_loudness_dbfs,
                    mean_pitch_hz,
                    speech_rate_proxy,
                    pause_after_ms,
                    phrase_boundary_after,
                    accent_peak_ms,
                    phone_timing_hints_ms: Vec::new(),
                }
            })
            .collect::<Vec<_>>();

        let claims = claims_from_observation(&words, speech_rate);

        Self {
            source_text,
            source_duration_ms,
            words,
            loudness_contour_dbfs,
            pitch_contour_hz,
            pauses,
            phrase_boundaries,
            accent_peaks,
            speech_rate,
            confidence,
            provenance,
            claims,
        }
    }
}

impl EchoProsodyPlan {
    pub fn from_observation(
        observation: &EchoProsodyObservation,
        phonemized: Option<&PhonemizedUnit>,
    ) -> Self {
        let mut prosody_ops = Vec::new();
        let mut controls = PiperProsodyControls::default();
        let mut pause_after_word_indices = Vec::new();
        let mut accent_word_indices = Vec::new();
        let mut advisory_controls = Vec::new();

        if let Some(rate) = classify_rate(observation.speech_rate) {
            prosody_ops.push(ProsodyOp::AdjustRate {
                target: ProsodyTarget::WholeCandidate,
                rate,
            });
            controls.length_scale = Some(match rate {
                ProsodyRateClass::Slower => 1.15,
                ProsodyRateClass::Neutral => 1.0,
                ProsodyRateClass::Faster => 0.9,
            });
        }

        for word in &observation.words {
            if let Some(pause_ms) = word.pause_after_ms {
                pause_after_word_indices.push(word.word_index);
                prosody_ops.push(ProsodyOp::InsertPause(PauseOp {
                    after: ProsodyTarget::WordIndex {
                        index: word.word_index,
                    },
                    millis: pause_ms,
                    strength: if pause_ms >= STRONG_PAUSE_MS {
                        PauseStrengthClass::Strong
                    } else {
                        PauseStrengthClass::Medium
                    },
                    reason: if word.word_index + 1 == observation.words.len() {
                        PauseReason::SentenceBoundary
                    } else if word.phrase_boundary_after {
                        PauseReason::PhraseBoundary
                    } else {
                        PauseReason::WordBoundary
                    },
                    commitment: ProsodyCommitment::Committed,
                }));
                controls.boundary_overrides.push(PiperBoundaryOverride {
                    after_index: word.word_index,
                    strong: pause_ms >= STRONG_PAUSE_MS,
                });
                if word.word_index + 1 == observation.words.len() {
                    controls.pause_overrides.push(PiperPauseOverride {
                        millis: pause_ms,
                        label: "trailing_observed_pause".to_string(),
                    });
                }
            }

            if let Some(accent_peak_ms) = word.accent_peak_ms {
                accent_word_indices.push(word.word_index);
                prosody_ops.push(ProsodyOp::SetAccent {
                    target: ProsodyTarget::WordIndex {
                        index: word.word_index,
                    },
                    kind: ProsodyAccentKind::Focus,
                    strength: 72,
                });
                if let Some(phoneme_index) = accented_phoneme_index(word.word_index, phonemized) {
                    controls
                        .phoneme_duration_overrides
                        .push(PiperPhonemeDurationOverride {
                            phoneme_index,
                            millis: ACCENT_DURATION_HINT_MS,
                        });
                }
                advisory_controls.push(format!(
                    "accent peak near word {} at {} ms remains advisory in the current Riper runtime",
                    word.word_index, accent_peak_ms
                ));
            }
        }

        if let Some(shape) = infer_pitch_shape(&observation.pitch_contour_hz) {
            prosody_ops.push(ProsodyOp::SetPitchShape {
                target: ProsodyTarget::WholeCandidate,
                shape,
                strength: 64,
            });
            advisory_controls.push(format!(
                "pitch contour {:?} is exported as an advisory hint until direct contour control exists",
                shape
            ));
        }

        if let Some(loudness_threshold) = mean_f32(
            observation
                .words
                .iter()
                .filter_map(|word| word.mean_loudness_dbfs),
        ) {
            for word in &observation.words {
                if word
                    .mean_loudness_dbfs
                    .is_some_and(|dbfs| dbfs >= loudness_threshold + 3.0)
                {
                    prosody_ops.push(ProsodyOp::AdjustEnergy {
                        target: ProsodyTarget::WordIndex {
                            index: word.word_index,
                        },
                        energy: ProsodyEnergyClass::Higher,
                    });
                    advisory_controls.push(format!(
                        "energy peak on word {} is advisory only in the current Riper runtime",
                        word.word_index
                    ));
                }
            }
        }

        for word in &observation.words {
            if word.phrase_boundary_after {
                prosody_ops.push(ProsodyOp::SetBoundary {
                    target: ProsodyTarget::WordIndex {
                        index: word.word_index,
                    },
                    boundary: if word.pause_after_ms.unwrap_or_default() >= STRONG_PAUSE_MS {
                        ProsodyBoundaryHintOp::FinalClosure
                    } else {
                        ProsodyBoundaryHintOp::Continuing
                    },
                });
            }
        }

        Self {
            claims: observation.claims.clone(),
            prosody_ops,
            controls,
            pause_after_word_indices,
            accent_word_indices,
            advisory_controls,
        }
    }
}

impl EchoComparisonRecord {
    pub fn from_plan(
        observation: &EchoProsodyObservation,
        plan: &EchoProsodyPlan,
        diagnostics: &PiperSynthesisDiagnostics,
    ) -> Self {
        let last_word_index = observation.words.len().saturating_sub(1);
        let matched_pauses = observation
            .words
            .iter()
            .filter_map(|word| {
                word.pause_after_ms.map(|pause_ms| EchoMatchedPause {
                    after_word_index: word.word_index,
                    source_pause_ms: pause_ms,
                    control_status: if word.word_index == last_word_index
                        && !plan.controls.pause_overrides.is_empty()
                    {
                        ProsodyControlStatus::Approximated
                    } else {
                        ProsodyControlStatus::AdvisoryOnly
                    },
                })
            })
            .collect();
        let matched_accents = observation
            .words
            .iter()
            .filter_map(|word| {
                word.accent_peak_ms.map(|source_peak_ms| EchoMatchedAccent {
                    word_index: word.word_index,
                    source_peak_ms,
                    control_status: if plan.accent_word_indices.contains(&word.word_index) {
                        ProsodyControlStatus::AdvisoryOnly
                    } else {
                        ProsodyControlStatus::Deferred
                    },
                })
            })
            .collect();
        let unrealized_controls = diagnostics
            .control_statuses
            .iter()
            .filter(|status| {
                !matches!(
                    status.status,
                    ProsodyControlStatus::Realized | ProsodyControlStatus::Approximated
                )
            })
            .map(|status| format!("{}: {}", status.name, status.detail))
            .chain(plan.advisory_controls.iter().cloned())
            .collect();

        Self {
            source_text: observation.source_text.clone(),
            source_duration_ms: observation.source_duration_ms,
            riper_duration_ms: diagnostics.pcm_duration_ms,
            matched_pauses,
            matched_accents,
            unrealized_controls,
        }
    }
}

fn claims_from_observation(
    words: &[EchoWordProsodyObservation],
    speech_rate: f32,
) -> Vec<AnalysisClaim> {
    let mut claims = Vec::new();

    for word in words {
        if word.accent_peak_ms.is_some() {
            let mut claim = AnalysisClaim::new(
                AnalysisTarget::WordIndex(word.word_index),
                ClaimKind::ProsodicRole,
                ClaimValue::ProsodicRole("observed_accent_peak".to_string()),
                AnalysisSourceKind::AcousticEvidence,
                word.confidence.unwrap_or(0.8),
                format!(
                    "offline echo observed an accent-like energy peak for `{}`",
                    word.text
                ),
            );
            claim.commit();
            claims.push(claim);
        }

        if let Some(pause_ms) = word.pause_after_ms {
            let mut claim = AnalysisClaim::new(
                AnalysisTarget::Boundary {
                    left_word: Some(word.word_index),
                    right_word: Some(word.word_index + 1),
                },
                ClaimKind::BoundaryKind,
                ClaimValue::BoundaryKind(if pause_ms >= STRONG_PAUSE_MS {
                    "observed_strong_pause".to_string()
                } else {
                    "observed_minor_pause".to_string()
                }),
                AnalysisSourceKind::AcousticEvidence,
                word.confidence.unwrap_or(0.75),
                format!(
                    "offline echo measured a {} ms pause after `{}`",
                    pause_ms, word.text
                ),
            );
            claim.commit();
            claims.push(claim);
        }
    }

    if !words.is_empty() {
        let indices = words.iter().map(|word| word.word_index).collect::<Vec<_>>();
        let mut claim = AnalysisClaim::new(
            AnalysisTarget::WordRange(indices),
            ClaimKind::ProsodicRole,
            ClaimValue::ProsodicRole(
                match classify_rate(speech_rate) {
                    Some(ProsodyRateClass::Slower) => "observed_rate_slower",
                    Some(ProsodyRateClass::Faster) => "observed_rate_faster",
                    _ => "observed_rate_neutral",
                }
                .to_string(),
            ),
            AnalysisSourceKind::AcousticEvidence,
            0.8,
            format!(
                "offline echo averaged speech-rate proxy {:.3} across the utterance",
                speech_rate
            ),
        );
        claim.commit();
        claims.push(claim);
    }

    claims
}

fn word_window_frames<'a>(
    word: &TranscriptWord,
    frames: &'a [&StreamingProsodyFrame],
) -> Vec<&'a StreamingProsodyFrame> {
    match (word.start_ms, word.end_ms) {
        (Some(start_ms), Some(end_ms)) => frames
            .iter()
            .copied()
            .filter(|frame| frame.frame_start_ms < end_ms && frame.frame_end_ms > start_ms)
            .collect(),
        _ => Vec::new(),
    }
}

fn pause_after_word(
    word_index: usize,
    word_end_ms: u64,
    transcript_words: &[TranscriptWord],
    pauses: &[ProsodyPauseBoundary],
) -> Option<u64> {
    let next_word_start_ms = transcript_words
        .get(word_index + 1)
        .and_then(|word| word.start_ms);
    pauses.iter().find_map(|pause| {
        let near_word_end = pause.start_ms >= word_end_ms.saturating_sub(PAUSE_MATCH_TOLERANCE_MS);
        let before_next_word = next_word_start_ms
            .map(|next_start| pause.end_ms <= next_start.saturating_add(PAUSE_MATCH_TOLERANCE_MS))
            .unwrap_or(true);
        (near_word_end && before_next_word).then_some(pause.end_ms.saturating_sub(pause.start_ms))
    })
}

fn mean_f32(values: impl Iterator<Item = f32>) -> Option<f32> {
    let (sum, count) = values.fold((0.0_f32, 0usize), |(sum, count), value| {
        (sum + value, count + 1)
    });
    (count > 0).then_some(sum / count as f32)
}

fn classify_rate(speech_rate: f32) -> Option<ProsodyRateClass> {
    if speech_rate >= 0.72 {
        Some(ProsodyRateClass::Faster)
    } else if speech_rate <= 0.42 {
        Some(ProsodyRateClass::Slower)
    } else if speech_rate > 0.0 {
        Some(ProsodyRateClass::Neutral)
    } else {
        None
    }
}

fn infer_pitch_shape(pitch_contour_hz: &[Option<f32>]) -> Option<ProsodyPitchShape> {
    let voiced = pitch_contour_hz
        .iter()
        .copied()
        .flatten()
        .collect::<Vec<_>>();
    let first = *voiced.first()?;
    let last = *voiced.last()?;
    let delta = last - first;
    if delta >= 12.0 {
        Some(ProsodyPitchShape::Rise)
    } else if delta <= -12.0 {
        Some(ProsodyPitchShape::Fall)
    } else {
        None
    }
}

fn accented_phoneme_index(word_index: usize, phonemized: Option<&PhonemizedUnit>) -> Option<usize> {
    let phonemized = phonemized?;
    let word_target = phonemized
        .word_targets
        .iter()
        .find(|target| target.word_index == word_index)?;
    phonemized
        .lexical_stress
        .iter()
        .find(|stress| {
            stress.word_index == word_index
                && matches!(
                    stress.stress,
                    LexicalStressLevel::Primary | LexicalStressLevel::Secondary
                )
        })
        .map(|stress| stress.phoneme_index)
        .or_else(|| Some(word_target.phoneme_range.start))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::streaming_prosody::{
        ProsodyAccentCandidate, ProsodyPauseBoundary, ProsodyPhraseCandidate, ProsodyProvenance,
        RollingProsodyModel, StreamingProsodyFrame, StreamingProsodyUpdate,
    };
    use crate::mouth::riper::SimpleEnglishG2p;

    fn update(
        frame_start_ms: u64,
        frame_end_ms: u64,
        loudness_dbfs: f32,
        pitch_hz: Option<f32>,
        speech_rate_proxy: f32,
        pause: Option<ProsodyPauseBoundary>,
        phrase_candidate: Option<ProsodyPhraseCandidate>,
        accent_candidate: Option<ProsodyAccentCandidate>,
    ) -> StreamingProsodyUpdate {
        StreamingProsodyUpdate {
            frame: StreamingProsodyFrame {
                frame_start_ms,
                frame_end_ms,
                rms_loudness: 0.2,
                loudness_dbfs,
                pitch_hz,
                voicing_confidence: 0.8,
                energy_contour: 0.1,
                pause_marker: pause.is_some(),
                speech_rate_proxy,
                spectral_tilt: None,
                confidence: 0.85,
                provenance: ProsodyProvenance::Confirmed,
                revision: 1,
            },
            model: RollingProsodyModel {
                current_pitch_range_hz: Some((110.0, 180.0)),
                recent_loudness_contour: vec![loudness_dbfs],
                likely_stress_peaks_ms: accent_candidate
                    .iter()
                    .map(|accent| accent.at_ms)
                    .collect(),
                recent_pause_boundaries: pause.iter().cloned().collect(),
                provisional_phrase_boundaries_ms: phrase_candidate
                    .iter()
                    .map(|candidate| candidate.end_ms)
                    .collect(),
                confidence: 0.85,
                provenance: ProsodyProvenance::Confirmed,
                revision: 1,
            },
            contour: Some(0.1),
            pause,
            phrase_candidate,
            accent_candidate,
            observed_feature_latency_ms: 10,
            latency_target_ms: 50,
            captured_at_unix_ns: 0,
            processed_at_unix_ns: 0,
        }
    }

    #[test]
    fn observation_captures_pause_accent_rate_and_claims() {
        let words = vec![
            TranscriptWord {
                text: "hello".to_string(),
                start_ms: Some(0),
                end_ms: Some(180),
                confidence: Some(0.9),
            },
            TranscriptWord {
                text: "world".to_string(),
                start_ms: Some(260),
                end_ms: Some(460),
                confidence: Some(0.88),
            },
        ];
        let updates = vec![
            update(
                0,
                120,
                -18.0,
                Some(140.0),
                0.76,
                None,
                None,
                Some(ProsodyAccentCandidate {
                    at_ms: 90,
                    confidence: 0.9,
                    provenance: ProsodyProvenance::Confirmed,
                    revision: 1,
                }),
            ),
            update(
                180,
                260,
                -52.0,
                None,
                0.1,
                Some(ProsodyPauseBoundary {
                    start_ms: 180,
                    end_ms: 260,
                }),
                Some(ProsodyPhraseCandidate {
                    start_ms: 180,
                    end_ms: 260,
                    confidence: 0.8,
                    provenance: ProsodyProvenance::Confirmed,
                    revision: 1,
                }),
                None,
            ),
            update(260, 460, -16.0, Some(124.0), 0.74, None, None, None),
        ];

        let observation =
            EchoProsodyObservation::from_streaming_updates("hello world", &words, &updates);

        assert_eq!(observation.words.len(), 2);
        assert_eq!(observation.words[0].accent_peak_ms, Some(90));
        assert_eq!(observation.words[0].pause_after_ms, Some(80));
        assert!(observation.words[0].phrase_boundary_after);
        assert!(observation.speech_rate > 0.4);
        assert!(observation
            .claims
            .iter()
            .any(|claim| matches!(claim.value, ClaimValue::ProsodicRole(ref value) if value == "observed_accent_peak")));
        assert!(observation
            .claims
            .iter()
            .any(|claim| matches!(claim.value, ClaimValue::BoundaryKind(ref value) if value == "observed_minor_pause")));
    }

    #[test]
    fn plan_maps_pause_rate_and_accent_into_controls() {
        let observation = EchoProsodyObservation {
            source_text: "hello world".to_string(),
            source_duration_ms: 460,
            words: vec![
                EchoWordProsodyObservation {
                    word_index: 0,
                    text: "hello".to_string(),
                    start_ms: Some(0),
                    end_ms: Some(180),
                    confidence: Some(0.9),
                    mean_loudness_dbfs: Some(-14.0),
                    mean_pitch_hz: Some(150.0),
                    speech_rate_proxy: Some(0.78),
                    pause_after_ms: Some(90),
                    phrase_boundary_after: true,
                    accent_peak_ms: Some(90),
                    phone_timing_hints_ms: Vec::new(),
                },
                EchoWordProsodyObservation {
                    word_index: 1,
                    text: "world".to_string(),
                    start_ms: Some(260),
                    end_ms: Some(460),
                    confidence: Some(0.88),
                    mean_loudness_dbfs: Some(-21.0),
                    mean_pitch_hz: Some(128.0),
                    speech_rate_proxy: Some(0.76),
                    pause_after_ms: Some(280),
                    phrase_boundary_after: true,
                    accent_peak_ms: None,
                    phone_timing_hints_ms: Vec::new(),
                },
            ],
            loudness_contour_dbfs: vec![-14.0, -21.0],
            pitch_contour_hz: vec![Some(150.0), Some(128.0)],
            pauses: vec![ProsodyPauseBoundary {
                start_ms: 180,
                end_ms: 270,
            }],
            phrase_boundaries: vec![ProsodyPhraseCandidate {
                start_ms: 180,
                end_ms: 270,
                confidence: 0.8,
                provenance: ProsodyProvenance::Confirmed,
                revision: 1,
            }],
            accent_peaks: vec![ProsodyAccentCandidate {
                at_ms: 90,
                confidence: 0.9,
                provenance: ProsodyProvenance::Confirmed,
                revision: 1,
            }],
            speech_rate: 0.78,
            confidence: 0.85,
            provenance: ProsodyProvenance::Confirmed,
            claims: Vec::new(),
        };
        let phonemized = SimpleEnglishG2p::default()
            .phonemize_unit("hello world")
            .expect("known words should phonemize");

        let plan = EchoProsodyPlan::from_observation(&observation, Some(&phonemized));

        assert_eq!(plan.controls.length_scale, Some(0.9));
        assert_eq!(plan.controls.boundary_overrides.len(), 2);
        assert_eq!(plan.controls.pause_overrides.len(), 1);
        assert!(!plan.controls.phoneme_duration_overrides.is_empty());
        assert!(plan
            .prosody_ops
            .iter()
            .any(|op| matches!(op, ProsodyOp::SetAccent { .. })));
        assert!(plan
            .prosody_ops
            .iter()
            .any(|op| matches!(op, ProsodyOp::SetPitchShape { .. })));
        assert!(!plan.advisory_controls.is_empty());
    }

    #[test]
    fn comparison_reports_unrealized_controls() {
        let observation = EchoProsodyObservation {
            source_text: "hello".to_string(),
            source_duration_ms: 200,
            words: vec![EchoWordProsodyObservation {
                word_index: 0,
                text: "hello".to_string(),
                start_ms: Some(0),
                end_ms: Some(200),
                confidence: Some(0.9),
                mean_loudness_dbfs: Some(-15.0),
                mean_pitch_hz: Some(160.0),
                speech_rate_proxy: Some(0.8),
                pause_after_ms: Some(150),
                phrase_boundary_after: true,
                accent_peak_ms: Some(90),
                phone_timing_hints_ms: Vec::new(),
            }],
            loudness_contour_dbfs: vec![-15.0],
            pitch_contour_hz: vec![Some(160.0), Some(140.0)],
            pauses: vec![ProsodyPauseBoundary {
                start_ms: 200,
                end_ms: 350,
            }],
            phrase_boundaries: Vec::new(),
            accent_peaks: vec![ProsodyAccentCandidate {
                at_ms: 90,
                confidence: 0.9,
                provenance: ProsodyProvenance::Confirmed,
                revision: 1,
            }],
            speech_rate: 0.8,
            confidence: 0.85,
            provenance: ProsodyProvenance::Confirmed,
            claims: Vec::new(),
        };
        let plan = EchoProsodyPlan {
            claims: Vec::new(),
            prosody_ops: Vec::new(),
            controls: PiperProsodyControls {
                length_scale: Some(0.9),
                pause_overrides: vec![PiperPauseOverride {
                    millis: 150,
                    label: "trailing_observed_pause".to_string(),
                }],
                phoneme_duration_overrides: vec![PiperPhonemeDurationOverride {
                    phoneme_index: 0,
                    millis: 120,
                }],
                boundary_overrides: vec![PiperBoundaryOverride {
                    after_index: 0,
                    strong: false,
                }],
                ..Default::default()
            },
            pause_after_word_indices: vec![0],
            accent_word_indices: vec![0],
            advisory_controls: vec!["pitch contour Fall is advisory".to_string()],
        };
        let diagnostics = PiperSynthesisDiagnostics {
            input_phoneme_ids: vec![1, 2, 3],
            applied_length_scale: 0.9,
            applied_noise_scale: 1.0,
            applied_noise_w: 1.0,
            inserted_pause_ms: 150,
            pcm_duration_ms: 420,
            control_statuses: vec![
                crate::mouth::riper::ControlStatusEntry {
                    name: "length_scale".to_string(),
                    status: ProsodyControlStatus::Realized,
                    detail: "length_scale overridden".to_string(),
                },
                crate::mouth::riper::ControlStatusEntry {
                    name: "phoneme_duration_override[0]".to_string(),
                    status: ProsodyControlStatus::AdvisoryOnly,
                    detail: "advisory only".to_string(),
                },
            ],
        };

        let comparison = EchoComparisonRecord::from_plan(&observation, &plan, &diagnostics);

        assert_eq!(comparison.source_duration_ms, 200);
        assert_eq!(comparison.riper_duration_ms, 420);
        assert_eq!(
            comparison.matched_pauses[0].control_status,
            ProsodyControlStatus::Approximated
        );
        assert_eq!(
            comparison.matched_accents[0].control_status,
            ProsodyControlStatus::AdvisoryOnly
        );
        assert!(comparison
            .unrealized_controls
            .iter()
            .any(|entry| entry.contains("phoneme_duration_override[0]")));
    }
}
