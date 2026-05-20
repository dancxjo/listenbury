use serde::{Deserialize, Serialize};

use crate::mouth::riper::g2p::{LexicalStressLevel, PhonemeProsodyCandidate, SpeechCandidateId};
use crate::mouth::riper::prosody_audit::{
    PauseReason, PhoLikeDiagnosticEntry, PhoLikeDiagnostics, PhraseBoundaryKind,
    ProsodyRealizationStatus, Stress,
};
use crate::mouth::riper::text::{ProsodyBoundaryHint, ProsodyCommitment, detect_vocative_spans};

const PAUSE_MS_DEFAULT: u64 = 140;
const PAUSE_MS_FINAL_CLOSURE: u64 = 260;
const PAUSE_MS_BREATH: u64 = 180;
const PAUSE_MS_VOCATIVE_REDUCTION: u64 = 60;
const BREATH_PAUSE_WORD_INTERVAL: usize = 9;
const BREATH_PAUSE_MIN_WORDS_AFTER: usize = 4;
const CONTOUR_CONTINUING: (f32, f32, f32) = (0.82_f32, 0.10_f32, 1.0_f32);
const CONTOUR_PHRASE_BREAK: (f32, f32, f32) = (0.74_f32, 0.58_f32, 0.95_f32);
const CONTOUR_POSSIBLE_CLOSURE: (f32, f32, f32) = (0.34_f32, 0.76_f32, 0.90_f32);
const CONTOUR_FINAL_CLOSURE: (f32, f32, f32) = (0.08_f32, 0.92_f32, 0.86_f32);
const VOCATIVE_CONTINUATION_BIAS_FLOOR: f32 = 0.85;
const VOCATIVE_PAUSE_LIKELIHOOD_CEILING: f32 = 0.4;
const VOCATIVE_RATE_HINT_FLOOR: f32 = 1.04;
const FUNCTION_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "to", "of", "in", "on", "at",
    "for", "from", "with", "by", "as", "is", "are", "was", "were", "be", "been", "am", "it",
    "this", "that", "these", "those", "he", "she", "they", "we", "you", "i", "me", "my", "your",
    "our", "their", "because",
];
const FOCUS_INTENSIFIERS: &[&str] = &["so", "very", "really", "especially", "extremely"];
const FOCUS_PRECISION_ADVERBS: &[&str] = &["precisely", "exactly", "specifically", "particularly"];
const FOCUS_CONTRAST_MARKERS: &[&str] = &["but", "not", "instead", "rather"];
const FOCUS_CORRECTIVE_PARTICLES: &[&str] = &["actually", "even", "only", "just"];
const FUNCTION_WORD_GIVEN_STRENGTH: u8 = 24;
const FUNCTION_WORD_DEEMPHASIS_STRENGTH: u8 = 42;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BreathGroupId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryState {
    Continuing,
    PossibleClosure,
    FinalClosure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProsodyEnergy {
    Low,
    Neutral,
    Elevated,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProsodyContour {
    pub energy: ProsodyEnergy,
    pub continuation_bias: f32,
    pub pause_likelihood: f32,
    pub speaking_rate_hint: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BreathGroupCandidate {
    pub id: BreathGroupId,
    pub source_candidate_id: SpeechCandidateId,
    pub text: String,
    pub contour: ProsodyContour,
    pub boundary_state: BoundaryState,
    pub commitment: ProsodyCommitment,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProsodyOverlay {
    pub target: ProsodyTarget,
    pub operation: ProsodyOperation,
    pub strength: u8,
    pub source: ProsodyOverlaySource,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProsodyTarget {
    WholeCandidate,
    WordIndex { index: usize },
    WordRange { start: usize, end: usize },
    PhonemeRange { start: usize, end: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProsodyOperation {
    Emphasize,
    Deemphasize,
    Sarcasm,
    Skepticism,
    Anger,
    Warmth,
    Continuation,
    Finality,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProsodyOverlaySource {
    Emoji(String),
    PromptTag(String),
    RuntimeAffect,
    Inference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProsodyPitchShape {
    Level,
    Rise,
    Fall,
    RiseFall,
    FallRise,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProsodyAccentKind {
    LexicalStress,
    Contrastive,
    Focus,
    GivenInformation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProsodyRateClass {
    Slower,
    Neutral,
    Faster,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProsodyEnergyClass {
    Lower,
    Neutral,
    Higher,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PauseStrengthClass {
    Light,
    Medium,
    Strong,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PauseOp {
    pub after: ProsodyTarget,
    pub millis: u64,
    pub strength: PauseStrengthClass,
    pub reason: PauseReason,
    pub commitment: ProsodyCommitment,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProsodyBoundaryHintOp {
    Continuing,
    PossibleClosure,
    FinalClosure,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProsodyOp {
    SetBaseContour(ProsodyContour),
    SetAccent {
        target: ProsodyTarget,
        kind: ProsodyAccentKind,
        strength: u8,
    },
    PreserveLexicalStress {
        target: ProsodyTarget,
        stress: LexicalStressLevel,
    },
    SetPitchShape {
        target: ProsodyTarget,
        shape: ProsodyPitchShape,
        strength: u8,
    },
    AdjustRate {
        target: ProsodyTarget,
        rate: ProsodyRateClass,
    },
    AdjustEnergy {
        target: ProsodyTarget,
        energy: ProsodyEnergyClass,
    },
    ApplyRhetoric {
        target: ProsodyTarget,
        op: ProsodyOperation,
        strength: u8,
    },
    InsertPause(PauseOp),
    SetBoundary {
        target: ProsodyTarget,
        boundary: ProsodyBoundaryHintOp,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProsodyList {
    pub base: BreathGroupCandidate,
    pub ops: Vec<ProsodyOp>,
    pub focus_diagnostics: Vec<FocusAccentDiagnostic>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RiperProsodyRealization {
    pub candidate_id: SpeechCandidateId,
    pub phone_duration_overrides_ms: Vec<Option<u64>>,
    pub word_duration_overrides_ms: Vec<Option<u64>>,
    pub pauses: Vec<PauseOp>,
    pub realized_ops: Vec<ProsodyOp>,
    pub advisory_ops: Vec<ProsodyOp>,
    pub diagnostics: Vec<String>,
    pub focus_diagnostics: Vec<FocusAccentDiagnostic>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FocusAccentReason {
    Intensifier,
    PrecisionAdverb,
    ContrastMarker,
    CorrectiveParticle,
    QuotedWord,
    FinalContentWord,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FocusAccentStatus {
    Provisional,
    Prepared,
    Playable,
    Committed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FocusAccentDiagnostic {
    pub word: String,
    pub word_index: usize,
    pub reason: FocusAccentReason,
    pub strength: u8,
    pub candidate_id: SpeechCandidateId,
    pub status: FocusAccentStatus,
}

impl ProsodyList {
    pub fn apply_overlay(&mut self, overlay: ProsodyOverlay) {
        self.ops.push(ProsodyOp::ApplyRhetoric {
            target: overlay.target,
            op: overlay.operation,
            strength: overlay.strength,
        });
    }

    pub fn realize_for_riper(
        &self,
        candidate: &PhonemeProsodyCandidate,
    ) -> RiperProsodyRealization {
        let mut phone_duration_overrides_ms = vec![None; candidate.phonemes.phonemes.len()];
        let mut word_duration_overrides_ms = vec![None; candidate.word_targets.len()];
        let mut pauses = Vec::new();
        let mut realized_ops = Vec::new();
        let mut advisory_ops = Vec::new();
        let mut diagnostics = Vec::new();

        for op in &self.ops {
            match op {
                ProsodyOp::AdjustRate { target, rate } => {
                    let factor = match rate {
                        ProsodyRateClass::Slower => 1.15_f32,
                        ProsodyRateClass::Neutral => 1.0_f32,
                        ProsodyRateClass::Faster => 0.9_f32,
                    };
                    if apply_word_duration_factor(
                        candidate,
                        target,
                        factor,
                        &mut word_duration_overrides_ms,
                    ) {
                        realized_ops.push(op.clone());
                    } else {
                        advisory_ops.push(op.clone());
                    }
                }
                ProsodyOp::PreserveLexicalStress { target, stress } => {
                    let factor = match stress {
                        LexicalStressLevel::Primary => 1.2_f32,
                        LexicalStressLevel::Secondary => 1.1_f32,
                        LexicalStressLevel::Unstressed => 0.95_f32,
                    };
                    if apply_phone_duration_factor(
                        candidate,
                        target,
                        factor,
                        &mut phone_duration_overrides_ms,
                    ) {
                        realized_ops.push(op.clone());
                    } else {
                        advisory_ops.push(op.clone());
                    }
                }
                ProsodyOp::InsertPause(pause) => {
                    pauses.push(pause.clone());
                    realized_ops.push(op.clone());
                }
                ProsodyOp::SetBoundary { .. }
                | ProsodyOp::SetBaseContour(_)
                | ProsodyOp::SetAccent { .. } => {
                    advisory_ops.push(op.clone());
                }
                ProsodyOp::SetPitchShape { .. }
                | ProsodyOp::AdjustEnergy { .. }
                | ProsodyOp::ApplyRhetoric { .. } => {
                    advisory_ops.push(op.clone());
                }
            }
        }

        if !advisory_ops.is_empty() {
            diagnostics.push(
                "Riper currently applies duration and pause hints; pitch/energy/accent controls remain advisory"
                    .to_string(),
            );
        }
        diagnostics.push(format!(
            "realized_ops={}, advisory_ops={}",
            realized_ops.len(),
            advisory_ops.len()
        ));

        RiperProsodyRealization {
            candidate_id: candidate.id,
            phone_duration_overrides_ms,
            word_duration_overrides_ms,
            pauses,
            realized_ops,
            advisory_ops,
            diagnostics,
            focus_diagnostics: self.focus_diagnostics.clone(),
        }
    }

    pub fn pho_like_diagnostics(&self, candidate: &PhonemeProsodyCandidate) -> PhoLikeDiagnostics {
        let pauses = self
            .ops
            .iter()
            .filter_map(|op| match op {
                ProsodyOp::InsertPause(pause) => Some(pause.millis),
                _ => None,
            })
            .collect::<Vec<_>>();
        let pause_hint = pauses.last().copied();
        let direct_address_pause_op = self.ops.iter().find_map(|op| match op {
            ProsodyOp::InsertPause(pause)
                if matches!(pause.reason, PauseReason::DirectAddressBoundary) =>
            {
                Some(pause)
            }
            _ => None,
        });
        let vocative_spans = detect_vocative_spans(&candidate.text);

        let entries = candidate
            .word_targets
            .iter()
            .map(|word_target| {
                let phoneme = candidate.phonemes.phonemes[word_target.phoneme_range.clone()]
                    .iter()
                    .map(|symbol| symbol.0.clone())
                    .collect::<Vec<_>>()
                    .join(" ");
                let duration_hint = candidate
                    .word_hints
                    .iter()
                    .find(|hint| hint.word_index == word_target.word_index)
                    .and_then(|hint| hint.approximate_duration_ms);
                let stress = candidate
                    .lexical_stress
                    .iter()
                    .filter(|stress| {
                        stress.phoneme_index >= word_target.phoneme_range.start
                            && stress.phoneme_index < word_target.phoneme_range.end
                    })
                    .map(|stress| match stress.stress {
                        LexicalStressLevel::Primary => Stress::Primary,
                        LexicalStressLevel::Secondary => Stress::Secondary,
                        LexicalStressLevel::Unstressed => Stress::Reduced,
                    })
                    .collect::<Vec<_>>();
                let accent = self.ops.iter().find_map(|op| match op {
                    ProsodyOp::SetAccent {
                        target: ProsodyTarget::WordIndex { index },
                        kind,
                        ..
                    } if *index == word_target.word_index => Some(format!("{kind:?}")),
                    _ => None,
                });
                let pitch_hint = self.ops.iter().find_map(|op| match op {
                    ProsodyOp::SetPitchShape {
                        target: ProsodyTarget::WordIndex { index },
                        shape,
                        ..
                    } if *index == word_target.word_index => Some(format!("{shape:?}")),
                    ProsodyOp::SetPitchShape {
                        target: ProsodyTarget::WholeCandidate,
                        shape,
                        ..
                    } => Some(format!("{shape:?}")),
                    _ => None,
                });
                let is_vocative_span = vocative_spans.iter().any(|span| {
                    span.start < word_target.text_range.end
                        && span.end > word_target.text_range.start
                });
                PhoLikeDiagnosticEntry {
                    word: word_target.normalized_text.clone(),
                    span: if is_vocative_span {
                        Some(
                            candidate.text[word_target.text_range.clone()]
                                .trim_matches(|ch: char| !ch.is_ascii_alphabetic())
                                .to_string(),
                        )
                    } else {
                        None
                    },
                    phoneme,
                    duration_hint,
                    stress,
                    accent,
                    boundary: if word_target.word_index + 1 == candidate.word_targets.len() {
                        Some(candidate.boundary_kind)
                    } else {
                        None
                    },
                    pause: if word_target.word_index + 1 == candidate.word_targets.len() {
                        pause_hint
                    } else {
                        None
                    },
                    classification: if is_vocative_span {
                        Some("vocative".to_string())
                    } else {
                        None
                    },
                    pause_behavior: if is_vocative_span {
                        Some(if direct_address_pause_op.is_some() {
                            "reduced".to_string()
                        } else {
                            "suppressed".to_string()
                        })
                    } else {
                        None
                    },
                    pitch_hint,
                    realization_status: ProsodyRealizationStatus::Requested,
                }
            })
            .collect();

        PhoLikeDiagnostics {
            candidate_id: candidate.id.0,
            entries,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct BreathGroupProsodyPlanner {
    active: Option<ProsodyList>,
}

impl BreathGroupProsodyPlanner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn active(&self) -> Option<&ProsodyList> {
        self.active.as_ref()
    }

    pub fn cancel_candidate(
        &mut self,
        source_candidate_id: SpeechCandidateId,
    ) -> Option<ProsodyList> {
        let mut active = self.active.take()?;
        if active.base.source_candidate_id == source_candidate_id {
            active.base.commitment = ProsodyCommitment::Cancelled;
            return Some(active);
        }

        self.active = Some(active);
        None
    }

    pub fn plan_candidate(&mut self, candidate: &PhonemeProsodyCandidate) -> ProsodyList {
        let previous = self
            .active
            .as_ref()
            .filter(|active| active.base.source_candidate_id == candidate.id)
            .map(|active| &active.base.contour);

        let contour = build_contour(candidate, previous);
        let boundary_state = boundary_state(candidate);
        let base = BreathGroupCandidate {
            id: BreathGroupId(candidate.id.0),
            source_candidate_id: candidate.id,
            text: candidate.text.clone(),
            contour: contour.clone(),
            boundary_state,
            commitment: candidate.commitment,
        };

        let mut ops = vec![ProsodyOp::SetBaseContour(contour)];
        ops.push(ProsodyOp::SetBoundary {
            target: ProsodyTarget::WholeCandidate,
            boundary: match boundary_state {
                BoundaryState::Continuing => ProsodyBoundaryHintOp::Continuing,
                BoundaryState::PossibleClosure => ProsodyBoundaryHintOp::PossibleClosure,
                BoundaryState::FinalClosure => ProsodyBoundaryHintOp::FinalClosure,
            },
        });
        ops.push(ProsodyOp::SetPitchShape {
            target: ProsodyTarget::WholeCandidate,
            shape: match boundary_state {
                BoundaryState::Continuing => ProsodyPitchShape::Rise,
                BoundaryState::PossibleClosure => ProsodyPitchShape::FallRise,
                BoundaryState::FinalClosure => ProsodyPitchShape::Fall,
            },
            strength: if matches!(boundary_state, BoundaryState::FinalClosure) {
                96
            } else {
                64
            },
        });
        ops.push(ProsodyOp::AdjustEnergy {
            target: ProsodyTarget::WholeCandidate,
            energy: match base.contour.energy {
                ProsodyEnergy::Low => ProsodyEnergyClass::Lower,
                ProsodyEnergy::Neutral => ProsodyEnergyClass::Neutral,
                ProsodyEnergy::Elevated => ProsodyEnergyClass::Higher,
            },
        });

        if base.contour.pause_likelihood >= 0.5 {
            ops.push(ProsodyOp::InsertPause(PauseOp {
                after: ProsodyTarget::WholeCandidate,
                millis: if matches!(base.boundary_state, BoundaryState::FinalClosure) {
                    PAUSE_MS_FINAL_CLOSURE
                } else if matches!(candidate.boundary_kind, PhraseBoundaryKind::Vocative) {
                    PAUSE_MS_DEFAULT.saturating_sub(PAUSE_MS_VOCATIVE_REDUCTION)
                } else {
                    PAUSE_MS_DEFAULT
                },
                strength: if matches!(base.boundary_state, BoundaryState::FinalClosure) {
                    PauseStrengthClass::Strong
                } else if matches!(candidate.boundary_kind, PhraseBoundaryKind::Vocative) {
                    PauseStrengthClass::Light
                } else {
                    PauseStrengthClass::Medium
                },
                reason: if matches!(candidate.boundary_kind, PhraseBoundaryKind::Vocative)
                    && !matches!(base.boundary_state, BoundaryState::FinalClosure)
                {
                    PauseReason::DirectAddressBoundary
                } else if matches!(base.boundary_state, BoundaryState::FinalClosure) {
                    PauseReason::SentenceBoundary
                } else {
                    PauseReason::PhraseBoundary
                },
                commitment: base.commitment,
            }));
        }
        ops.extend(default_breath_pause_ops(candidate, base.commitment));
        let focus_plan = default_emphasis_ops(candidate, boundary_state);
        ops.extend(focus_plan.ops);

        let planned = ProsodyList {
            base,
            ops,
            focus_diagnostics: focus_plan.diagnostics,
        };
        self.active = Some(planned.clone());
        planned
    }
}

fn boundary_state(candidate: &PhonemeProsodyCandidate) -> BoundaryState {
    if matches!(candidate.commitment, ProsodyCommitment::Committed)
        || matches!(
            candidate.boundary_hint,
            ProsodyBoundaryHint::FinalSentenceEnd
        )
    {
        return BoundaryState::FinalClosure;
    }

    match candidate.boundary_hint {
        ProsodyBoundaryHint::PossibleSentenceEnd => BoundaryState::PossibleClosure,
        ProsodyBoundaryHint::None | ProsodyBoundaryHint::PhraseBreak => BoundaryState::Continuing,
        ProsodyBoundaryHint::FinalSentenceEnd => unreachable!("final sentence end handled above"),
    }
}

fn build_contour(
    candidate: &PhonemeProsodyCandidate,
    previous: Option<&ProsodyContour>,
) -> ProsodyContour {
    let (mut base_continuation, mut pause_likelihood, mut speaking_rate_hint) =
        match candidate.boundary_hint {
            ProsodyBoundaryHint::None => CONTOUR_CONTINUING,
            ProsodyBoundaryHint::PhraseBreak => CONTOUR_PHRASE_BREAK,
            ProsodyBoundaryHint::PossibleSentenceEnd => CONTOUR_POSSIBLE_CLOSURE,
            ProsodyBoundaryHint::FinalSentenceEnd => CONTOUR_FINAL_CLOSURE,
        };
    if matches!(candidate.boundary_kind, PhraseBoundaryKind::Vocative)
        && !matches!(
            candidate.boundary_hint,
            ProsodyBoundaryHint::FinalSentenceEnd
        )
    {
        base_continuation = base_continuation.max(VOCATIVE_CONTINUATION_BIAS_FLOOR);
        pause_likelihood = pause_likelihood.min(VOCATIVE_PAUSE_LIKELIHOOD_CEILING);
        speaking_rate_hint = speaking_rate_hint.max(VOCATIVE_RATE_HINT_FLOOR);
    }

    let continuation_bias = if candidate.stable_prefix_len > 0 {
        if let Some(previous) = previous {
            base_continuation.max(previous.continuation_bias * 0.5)
        } else {
            base_continuation
        }
    } else {
        base_continuation
    };

    let energy = match candidate.commitment {
        ProsodyCommitment::Cancelled => ProsodyEnergy::Low,
        ProsodyCommitment::Prepared | ProsodyCommitment::Playable => ProsodyEnergy::Elevated,
        ProsodyCommitment::Provisional | ProsodyCommitment::Committed => ProsodyEnergy::Neutral,
    };

    ProsodyContour {
        energy,
        continuation_bias,
        pause_likelihood,
        speaking_rate_hint,
    }
}

fn default_emphasis_ops(
    candidate: &PhonemeProsodyCandidate,
    boundary_state: BoundaryState,
) -> FocusAccentPlan {
    let mut ops = Vec::new();
    let mut diagnostics = Vec::new();
    let mut focus_by_word = std::collections::HashMap::<usize, FocusSelection>::new();
    let word_count = candidate.word_targets.len();
    if word_count == 0 {
        return FocusAccentPlan { ops, diagnostics };
    }

    for target in &candidate.word_targets {
        let word = target.normalized_text.as_str();
        if has_quote_emphasis(candidate, target.text_range.start, target.text_range.end) {
            promote_focus(
                &mut focus_by_word,
                target.word_index,
                FocusAccentReason::QuotedWord,
                82,
            );
        }
        if FOCUS_INTENSIFIERS.contains(&word) {
            promote_focus(
                &mut focus_by_word,
                target.word_index,
                FocusAccentReason::Intensifier,
                80,
            );
        }
        if FOCUS_PRECISION_ADVERBS.contains(&word) {
            promote_focus(
                &mut focus_by_word,
                target.word_index,
                FocusAccentReason::PrecisionAdverb,
                78,
            );
        }
        if FOCUS_CONTRAST_MARKERS.contains(&word) {
            promote_focus(
                &mut focus_by_word,
                target.word_index,
                FocusAccentReason::ContrastMarker,
                76,
            );
        }
        if FOCUS_CORRECTIVE_PARTICLES.contains(&word) {
            promote_focus(
                &mut focus_by_word,
                target.word_index,
                FocusAccentReason::CorrectiveParticle,
                74,
            );
        }
    }

    if !matches!(boundary_state, BoundaryState::Continuing)
        && let Some(final_content) = candidate
            .word_targets
            .iter()
            .rev()
            .find(|target| is_content_word(&target.normalized_text))
    {
        promote_focus(
            &mut focus_by_word,
            final_content.word_index,
            FocusAccentReason::FinalContentWord,
            60,
        );
    }

    for target in &candidate.word_targets {
        let focus = focus_by_word.get(&target.word_index).copied();
        if let Some(focus) = focus {
            ops.push(ProsodyOp::SetAccent {
                target: ProsodyTarget::WordIndex {
                    index: target.word_index,
                },
                kind: focus_reason_accent_kind(focus.reason),
                strength: focus.strength,
            });
            ops.push(ProsodyOp::ApplyRhetoric {
                target: ProsodyTarget::WordIndex {
                    index: target.word_index,
                },
                op: ProsodyOperation::Emphasize,
                strength: focus.strength,
            });
            diagnostics.push(FocusAccentDiagnostic {
                word: target.normalized_text.clone(),
                word_index: target.word_index,
                reason: focus.reason,
                strength: focus.strength,
                candidate_id: candidate.id,
                status: focus_status_from_commitment(candidate.commitment),
            });
        } else if !is_content_word(&target.normalized_text) {
            ops.push(ProsodyOp::SetAccent {
                target: ProsodyTarget::WordIndex {
                    index: target.word_index,
                },
                kind: ProsodyAccentKind::GivenInformation,
                strength: FUNCTION_WORD_GIVEN_STRENGTH,
            });
            ops.push(ProsodyOp::AdjustEnergy {
                target: ProsodyTarget::WordIndex {
                    index: target.word_index,
                },
                energy: ProsodyEnergyClass::Lower,
            });
            ops.push(ProsodyOp::ApplyRhetoric {
                target: ProsodyTarget::WordIndex {
                    index: target.word_index,
                },
                op: ProsodyOperation::Deemphasize,
                strength: FUNCTION_WORD_DEEMPHASIS_STRENGTH,
            });
        }

        let should_accelerate_function_word_rate = !focus_by_word.contains_key(&target.word_index)
            && !is_content_word(&target.normalized_text);
        ops.push(ProsodyOp::AdjustRate {
            target: ProsodyTarget::WordIndex {
                index: target.word_index,
            },
            rate: if should_accelerate_function_word_rate
                || is_parenthetical(candidate, target.text_range.start, target.text_range.end)
            {
                ProsodyRateClass::Faster
            } else {
                ProsodyRateClass::Neutral
            },
        });
    }

    for lexical in &candidate.lexical_stress {
        ops.push(ProsodyOp::PreserveLexicalStress {
            target: ProsodyTarget::PhonemeRange {
                start: lexical.phoneme_index,
                end: lexical.phoneme_index + 1,
            },
            stress: lexical.stress,
        });
    }

    if word_count <= 5
        && let Some(final_content) = candidate
            .word_targets
            .iter()
            .rev()
            .find(|target| is_content_word(&target.normalized_text))
    {
        ops.push(ProsodyOp::SetPitchShape {
            target: ProsodyTarget::WordIndex {
                index: final_content.word_index,
            },
            shape: match boundary_state {
                BoundaryState::FinalClosure => ProsodyPitchShape::Fall,
                BoundaryState::Continuing | BoundaryState::PossibleClosure => {
                    ProsodyPitchShape::RiseFall
                }
            },
            strength: 84,
        });
    }

    FocusAccentPlan { ops, diagnostics }
}

fn default_breath_pause_ops(
    candidate: &PhonemeProsodyCandidate,
    commitment: ProsodyCommitment,
) -> Vec<ProsodyOp> {
    candidate
        .word_targets
        .iter()
        .filter(|target| {
            let words_after = candidate
                .word_targets
                .len()
                .saturating_sub(target.word_index + 1);
            target.word_index + 1 >= BREATH_PAUSE_WORD_INTERVAL
                && (target.word_index + 1) % BREATH_PAUSE_WORD_INTERVAL == 0
                && words_after >= BREATH_PAUSE_MIN_WORDS_AFTER
        })
        .map(|target| {
            ProsodyOp::InsertPause(PauseOp {
                after: ProsodyTarget::WordIndex {
                    index: target.word_index,
                },
                millis: PAUSE_MS_BREATH,
                strength: PauseStrengthClass::Light,
                reason: PauseReason::Breath,
                commitment,
            })
        })
        .collect()
}

fn is_content_word(word: &str) -> bool {
    !FUNCTION_WORDS.contains(&word)
}

#[derive(Debug, Clone, PartialEq)]
struct FocusAccentPlan {
    ops: Vec<ProsodyOp>,
    diagnostics: Vec<FocusAccentDiagnostic>,
}

#[derive(Clone, Copy)]
struct FocusSelection {
    reason: FocusAccentReason,
    strength: u8,
}

fn promote_focus(
    focus_by_word: &mut std::collections::HashMap<usize, FocusSelection>,
    word_index: usize,
    reason: FocusAccentReason,
    strength: u8,
) {
    let entry = focus_by_word
        .entry(word_index)
        .or_insert(FocusSelection { reason, strength });
    if strength > entry.strength {
        *entry = FocusSelection { reason, strength };
    }
}

fn focus_status_from_commitment(commitment: ProsodyCommitment) -> FocusAccentStatus {
    match commitment {
        ProsodyCommitment::Provisional => FocusAccentStatus::Provisional,
        ProsodyCommitment::Prepared => FocusAccentStatus::Prepared,
        ProsodyCommitment::Playable => FocusAccentStatus::Playable,
        ProsodyCommitment::Committed => FocusAccentStatus::Committed,
        ProsodyCommitment::Cancelled => FocusAccentStatus::Cancelled,
    }
}

fn focus_reason_accent_kind(reason: FocusAccentReason) -> ProsodyAccentKind {
    if matches!(reason, FocusAccentReason::ContrastMarker) {
        ProsodyAccentKind::Contrastive
    } else {
        ProsodyAccentKind::Focus
    }
}

fn has_quote_emphasis(candidate: &PhonemeProsodyCandidate, start: usize, end: usize) -> bool {
    let before = candidate.text[..start].chars().next_back();
    let after = candidate.text[end..].chars().next();
    before.is_some_and(is_quote_mark) || after.is_some_and(is_quote_mark)
}

fn is_parenthetical(candidate: &PhonemeProsodyCandidate, start: usize, end: usize) -> bool {
    let before = candidate.text[..start].chars().next_back();
    let after = candidate.text[end..].chars().next();
    before == Some('(') || after == Some(')')
}

fn is_quote_mark(ch: char) -> bool {
    matches!(ch, '"' | '\'' | '“' | '”' | '‘' | '’')
}

fn apply_phone_duration_factor(
    candidate: &PhonemeProsodyCandidate,
    target: &ProsodyTarget,
    factor: f32,
    out: &mut [Option<u64>],
) -> bool {
    let mut any = false;
    for idx in target_phoneme_indexes(candidate, target) {
        if let Some(current) = candidate
            .phone_hints
            .iter()
            .find(|hint| hint.phoneme_index == idx)
            .and_then(|hint| hint.approximate_duration_ms)
        {
            out[idx] = Some((current as f32 * factor).round().max(1.0) as u64);
            any = true;
        }
    }
    any
}

fn apply_word_duration_factor(
    candidate: &PhonemeProsodyCandidate,
    target: &ProsodyTarget,
    factor: f32,
    out: &mut [Option<u64>],
) -> bool {
    let mut any = false;
    for idx in target_word_indexes(candidate, target) {
        if let Some(current) = candidate
            .word_hints
            .iter()
            .find(|hint| hint.word_index == idx)
            .and_then(|hint| hint.approximate_duration_ms)
        {
            out[idx] = Some((current as f32 * factor).round().max(1.0) as u64);
            any = true;
        }
    }
    any
}

fn target_word_indexes(candidate: &PhonemeProsodyCandidate, target: &ProsodyTarget) -> Vec<usize> {
    let word_len = candidate.word_targets.len();
    match target {
        ProsodyTarget::WholeCandidate => candidate
            .word_targets
            .iter()
            .map(|w| w.word_index)
            .collect(),
        ProsodyTarget::WordIndex { index } => {
            if *index < word_len {
                vec![*index]
            } else {
                Vec::new()
            }
        }
        ProsodyTarget::WordRange { start, end } => (*start..(*end).min(word_len)).collect(),
        ProsodyTarget::PhonemeRange { start, end } => (*start..*end)
            .filter_map(|idx| candidate.phoneme_to_word.get(idx).and_then(|word| *word))
            .collect(),
    }
}

fn target_phoneme_indexes(
    candidate: &PhonemeProsodyCandidate,
    target: &ProsodyTarget,
) -> Vec<usize> {
    match target {
        ProsodyTarget::WholeCandidate => (0..candidate.phonemes.phonemes.len()).collect(),
        ProsodyTarget::WordIndex { index } => candidate
            .word_targets
            .iter()
            .find(|w| w.word_index == *index)
            .map(|w| (w.phoneme_range.start..w.phoneme_range.end).collect())
            .unwrap_or_default(),
        ProsodyTarget::WordRange { start, end } => candidate
            .word_targets
            .iter()
            .filter(|w| w.word_index >= *start && w.word_index < *end)
            .flat_map(|w| w.phoneme_range.start..w.phoneme_range.end)
            .collect(),
        ProsodyTarget::PhonemeRange { start, end } => (*start..*end).collect(),
    }
}

#[cfg(test)]
mod tests {
    use crate::mouth::riper::g2p::{
        PhonemeProsodyCandidateEvent, PhonemeProsodyCandidateTracker, SimpleEnglishG2p,
    };

    use super::*;

    fn latest_candidate(events: &[PhonemeProsodyCandidateEvent]) -> &PhonemeProsodyCandidate {
        events
            .iter()
            .rev()
            .find_map(|event| match event {
                PhonemeProsodyCandidateEvent::CandidateUpdated { candidate } => Some(candidate),
                _ => None,
            })
            .expect("updated candidate event")
    }

    fn focus_diag<'a>(
        diagnostics: &'a [FocusAccentDiagnostic],
        word: &str,
    ) -> Option<&'a FocusAccentDiagnostic> {
        diagnostics.iter().find(|diag| diag.word == word)
    }

    fn first_pause(planned: &ProsodyList) -> Option<&PauseOp> {
        planned.ops.iter().find_map(|op| match op {
            ProsodyOp::InsertPause(pause) => Some(pause),
            _ => None,
        })
    }

    #[test]
    fn stable_extension_without_cadence_reset() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let mut planner = BreathGroupProsodyPlanner::new();

        let first = tracker.ingest_text("I see").expect("candidate");
        let first_plan = planner.plan_candidate(latest_candidate(&first));

        let second = tracker.ingest_text("I see okay").expect("candidate");
        let second_candidate = latest_candidate(&second);
        let second_plan = planner.plan_candidate(second_candidate);

        assert_eq!(
            second_plan.base.source_candidate_id,
            first_plan.base.source_candidate_id
        );
        assert_eq!(second_candidate.stable_prefix_len, "I see".len());
        assert!(
            second_plan.base.contour.continuation_bias >= first_plan.base.contour.continuation_bias
        );
        assert_eq!(second_plan.base.boundary_state, BoundaryState::Continuing);
    }

    #[test]
    fn delayed_sentence_closure() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let mut planner = BreathGroupProsodyPlanner::new();

        let first = tracker.ingest_text("I see").expect("candidate");
        let first_plan = planner.plan_candidate(latest_candidate(&first));
        assert_eq!(first_plan.base.boundary_state, BoundaryState::Continuing);

        let second = tracker.ingest_text("I see.").expect("candidate");
        let second_plan = planner.plan_candidate(latest_candidate(&second));
        assert_eq!(
            second_plan.base.boundary_state,
            BoundaryState::PossibleClosure
        );
        assert!(second_plan.base.contour.continuation_bias > 0.35);
    }

    #[test]
    fn abbreviation_initial_continuation() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let mut planner = BreathGroupProsodyPlanner::new();

        let first = tracker.ingest_text("F.").expect("candidate");
        let first_plan = planner.plan_candidate(latest_candidate(&first));
        assert_eq!(first_plan.base.boundary_state, BoundaryState::Continuing);
        assert!(first_plan.base.contour.continuation_bias > 0.7);
    }

    #[test]
    fn interruption_cancellation() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let mut planner = BreathGroupProsodyPlanner::new();

        let first = tracker.ingest_text("Okay.").expect("candidate");
        let first_candidate = latest_candidate(&first).clone();
        let first_plan = planner.plan_candidate(&first_candidate);

        let second = tracker.ingest_text("I see.").expect("candidate");
        let cancelled = second
            .iter()
            .find_map(|event| match event {
                PhonemeProsodyCandidateEvent::CandidateCancelled { id } => Some(*id),
                _ => None,
            })
            .expect("cancel event");

        let cancelled_plan = planner
            .cancel_candidate(cancelled)
            .expect("cancelled plan available");
        assert_eq!(
            cancelled_plan.base.source_candidate_id,
            first_plan.base.source_candidate_id
        );
        assert_eq!(cancelled_plan.base.commitment, ProsodyCommitment::Cancelled);

        let second_plan = planner.plan_candidate(latest_candidate(&second));
        assert_ne!(
            second_plan.base.source_candidate_id,
            first_plan.base.source_candidate_id
        );
    }

    #[test]
    fn phrase_continuation_after_comma() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let mut planner = BreathGroupProsodyPlanner::new();

        let candidate = tracker.ingest_text("I see, okay").expect("candidate");
        let planned = planner.plan_candidate(latest_candidate(&candidate));
        assert_eq!(planned.base.boundary_state, BoundaryState::Continuing);
        assert!(planned.base.contour.pause_likelihood >= 0.5);
        assert!(planned.base.contour.continuation_bias > 0.7);
    }

    #[test]
    fn vocative_fixtures_suppress_hard_comma_pause() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let mut planner = BreathGroupProsodyPlanner::new();

        for fixture in [
            "Thank you, Dave.",
            "Listen, professor, this matters.",
            "You see, interlocutor, the system has revealed itself.",
        ] {
            let candidate = tracker.ingest_text(fixture).expect("candidate");
            let latest = latest_candidate(&candidate);
            assert_eq!(latest.boundary_kind, PhraseBoundaryKind::Vocative);
            let planned = planner.plan_candidate(latest);
            assert!(
                first_pause(&planned).is_none(),
                "vocative fixture should suppress hard comma pauses: {fixture}"
            );
        }
    }

    #[test]
    fn parenthetical_and_apposition_keep_phrase_separation() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let mut planner = BreathGroupProsodyPlanner::new();

        for fixture in [
            "The machine, unfortunately, exploded.",
            "My brother, who lives in Tacoma, arrived.",
        ] {
            let candidate = tracker.ingest_text(fixture).expect("candidate");
            let latest = latest_candidate(&candidate);
            assert_ne!(latest.boundary_kind, PhraseBoundaryKind::Vocative);
            let planned = planner.plan_candidate(latest);
            assert!(
                first_pause(&planned).is_some(),
                "contrast fixture should preserve phrase separation: {fixture}"
            );
        }
    }

    #[test]
    fn default_emphasis_planner_marks_content_vs_function_words() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let mut planner = BreathGroupProsodyPlanner::new();

        let candidate = tracker.ingest_text("the warm light").expect("candidate");
        let planned = planner.plan_candidate(latest_candidate(&candidate));

        assert!(planned.ops.iter().any(|op| matches!(
            op,
            ProsodyOp::SetAccent {
                target: ProsodyTarget::WordIndex { index: 0 },
                kind: ProsodyAccentKind::GivenInformation,
                ..
            }
        )));
        assert!(planned.ops.iter().any(|op| matches!(
            op,
            ProsodyOp::AdjustRate {
                target: ProsodyTarget::WordIndex { index: 1 },
                rate: ProsodyRateClass::Neutral,
            }
        )));
        assert!(planned.ops.iter().any(|op| matches!(
            op,
            ProsodyOp::AdjustRate {
                target: ProsodyTarget::WordIndex { index: 0 },
                rate: ProsodyRateClass::Faster,
            }
        )));
        assert!(
            planned
                .ops
                .iter()
                .any(|op| matches!(op, ProsodyOp::PreserveLexicalStress { .. }))
        );
    }

    #[test]
    fn focus_fixture_accents_precisely_so_and_final_small() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let mut planner = BreathGroupProsodyPlanner::new();

        let candidate = tracker
            .ingest_text(
                "University politics are vicious precisely because the stakes are so small.",
            )
            .expect("candidate");
        let planned = planner.plan_candidate(latest_candidate(&candidate));

        let precisely =
            focus_diag(&planned.focus_diagnostics, "precisely").expect("precisely focus");
        assert_eq!(precisely.reason, FocusAccentReason::PrecisionAdverb);
        let so = focus_diag(&planned.focus_diagnostics, "so").expect("so focus");
        assert_eq!(so.reason, FocusAccentReason::Intensifier);
        let small = focus_diag(&planned.focus_diagnostics, "small").expect("small focus");
        assert_eq!(small.reason, FocusAccentReason::FinalContentWord);
        assert!(small.strength < so.strength);
        assert_eq!(precisely.status, FocusAccentStatus::Provisional);
    }

    #[test]
    fn focus_fixture_precision_and_corrective_words() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let mut planner = BreathGroupProsodyPlanner::new();

        let exactly = tracker
            .ingest_text("That is exactly what I meant.")
            .expect("candidate");
        let exactly_plan = planner.plan_candidate(latest_candidate(&exactly));
        assert_eq!(
            focus_diag(&exactly_plan.focus_diagnostics, "exactly")
                .expect("exactly focus")
                .reason,
            FocusAccentReason::PrecisionAdverb
        );

        let only = tracker
            .ingest_text("I only said the first one.")
            .expect("candidate");
        let only_plan = planner.plan_candidate(latest_candidate(&only));
        assert_eq!(
            focus_diag(&only_plan.focus_diagnostics, "only")
                .expect("only focus")
                .reason,
            FocusAccentReason::CorrectiveParticle
        );
        assert!(
            focus_diag(&only_plan.focus_diagnostics, "the").is_none(),
            "function words stay de-emphasized unless contrastive"
        );
    }

    #[test]
    fn focus_fixture_contrast_markers_and_intensifiers() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let mut planner = BreathGroupProsodyPlanner::new();

        let contrast = tracker
            .ingest_text("It is not broken, but delayed.")
            .expect("candidate");
        let contrast_plan = planner.plan_candidate(latest_candidate(&contrast));
        assert_eq!(
            focus_diag(&contrast_plan.focus_diagnostics, "not")
                .expect("not focus")
                .reason,
            FocusAccentReason::ContrastMarker
        );
        assert_eq!(
            focus_diag(&contrast_plan.focus_diagnostics, "but")
                .expect("but focus")
                .reason,
            FocusAccentReason::ContrastMarker
        );

        let intensifier = tracker
            .ingest_text("This is really very good.")
            .expect("candidate");
        let intensifier_plan = planner.plan_candidate(latest_candidate(&intensifier));
        assert_eq!(
            focus_diag(&intensifier_plan.focus_diagnostics, "really")
                .expect("really focus")
                .reason,
            FocusAccentReason::Intensifier
        );
        assert_eq!(
            focus_diag(&intensifier_plan.focus_diagnostics, "very")
                .expect("very focus")
                .reason,
            FocusAccentReason::Intensifier
        );
    }

    #[test]
    fn focus_fixture_not_blue_marks_not_as_contrastive() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let mut planner = BreathGroupProsodyPlanner::new();

        let candidate = tracker
            .ingest_text("I said red, not blue.")
            .expect("candidate");
        let planned = planner.plan_candidate(latest_candidate(&candidate));
        assert_eq!(
            focus_diag(&planned.focus_diagnostics, "not")
                .expect("not focus")
                .reason,
            FocusAccentReason::ContrastMarker
        );
    }

    #[test]
    fn focus_planning_is_revision_safe_across_incremental_updates() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let mut planner = BreathGroupProsodyPlanner::new();

        let first = tracker
            .ingest_text("University politics are vicious precisely...")
            .expect("candidate");
        let first_plan = planner.plan_candidate(latest_candidate(&first));
        assert_eq!(
            focus_diag(&first_plan.focus_diagnostics, "precisely")
                .expect("precisely focus")
                .reason,
            FocusAccentReason::PrecisionAdverb
        );
        assert!(focus_diag(&first_plan.focus_diagnostics, "so").is_none());

        let second = tracker
            .ingest_text("University politics are vicious precisely because...")
            .expect("candidate");
        let second_plan = planner.plan_candidate(latest_candidate(&second));
        assert_eq!(
            focus_diag(&second_plan.focus_diagnostics, "precisely")
                .expect("precisely focus")
                .reason,
            FocusAccentReason::PrecisionAdverb
        );
        assert!(focus_diag(&second_plan.focus_diagnostics, "so").is_none());

        let third = tracker
            .ingest_text(
                "University politics are vicious precisely because the stakes are so small.",
            )
            .expect("candidate");
        let third_plan = planner.plan_candidate(latest_candidate(&third));
        assert_eq!(
            focus_diag(&third_plan.focus_diagnostics, "precisely")
                .expect("precisely focus")
                .reason,
            FocusAccentReason::PrecisionAdverb
        );
        assert_eq!(
            focus_diag(&third_plan.focus_diagnostics, "so")
                .expect("so focus")
                .reason,
            FocusAccentReason::Intensifier
        );
    }

    #[test]
    fn provisional_cadence_uses_non_final_pitch_shape_until_commitment() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let mut planner = BreathGroupProsodyPlanner::new();

        let first = tracker
            .ingest_text("I think this works.")
            .expect("candidate");
        let provisional = planner.plan_candidate(latest_candidate(&first));
        assert!(provisional.ops.iter().any(|op| matches!(
            op,
            ProsodyOp::SetPitchShape {
                target: ProsodyTarget::WholeCandidate,
                shape: ProsodyPitchShape::FallRise,
                ..
            }
        )));

        let mut committed_candidate = latest_candidate(&first).clone();
        committed_candidate.mark_committed();
        let final_plan = planner.plan_candidate(&committed_candidate);
        assert!(final_plan.ops.iter().any(|op| matches!(
            op,
            ProsodyOp::SetPitchShape {
                target: ProsodyTarget::WholeCandidate,
                shape: ProsodyPitchShape::Fall,
                ..
            }
        )));
    }

    #[test]
    fn pause_planning_assigns_explicit_pause_reason() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let mut planner = BreathGroupProsodyPlanner::new();

        let first = tracker.ingest_text("I see.").expect("candidate");
        let provisional = planner.plan_candidate(latest_candidate(&first));
        let pause = provisional.ops.iter().find_map(|op| match op {
            ProsodyOp::InsertPause(pause) => Some(pause),
            _ => None,
        });
        assert_eq!(
            pause.map(|p| p.reason),
            Some(PauseReason::PhraseBoundary),
            "provisional sentence endings stay phrase-like until committed"
        );

        let mut committed_candidate = latest_candidate(&first).clone();
        committed_candidate.mark_committed();
        let committed = planner.plan_candidate(&committed_candidate);
        let committed_pause = committed.ops.iter().find_map(|op| match op {
            ProsodyOp::InsertPause(pause) => Some(pause),
            _ => None,
        });
        assert_eq!(
            committed_pause.map(|p| p.reason),
            Some(PauseReason::SentenceBoundary)
        );
    }

    #[test]
    fn long_runs_plan_light_breath_pauses() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let mut planner = BreathGroupProsodyPlanner::new();

        let candidate = tracker
            .ingest_text(
                "We represent the lollipop guild because the machine needs another minute before returning today.",
            )
            .expect("candidate");
        let planned = planner.plan_candidate(latest_candidate(&candidate));
        let breath_pause = planned.ops.iter().find_map(|op| match op {
            ProsodyOp::InsertPause(pause) if matches!(pause.reason, PauseReason::Breath) => {
                Some(pause)
            }
            _ => None,
        });

        assert_eq!(
            breath_pause.map(|pause| pause.millis),
            Some(PAUSE_MS_BREATH)
        );
        assert_eq!(
            breath_pause.map(|pause| &pause.after),
            Some(&ProsodyTarget::WordIndex { index: 8 }),
            "breath should fall after the ninth word"
        );
    }

    #[test]
    fn riper_realization_reports_realized_vs_advisory_hints() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let mut planner = BreathGroupProsodyPlanner::new();

        let candidate_events = tracker.ingest_text("I see, okay.").expect("candidate");
        let candidate = latest_candidate(&candidate_events);
        let planned = planner.plan_candidate(candidate);
        let realized = planned.realize_for_riper(candidate);

        assert!(!realized.realized_ops.is_empty());
        assert!(!realized.advisory_ops.is_empty());
        assert!(!realized.diagnostics.is_empty());
        assert_eq!(
            realized.focus_diagnostics, planned.focus_diagnostics,
            "realization should preserve planner focus diagnostics"
        );
        assert!(
            realized
                .diagnostics
                .iter()
                .any(|line| line.contains("advisory"))
        );
    }

    #[test]
    fn emits_pho_like_diagnostics_for_seed_sentence() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let mut planner = BreathGroupProsodyPlanner::new();
        let candidate = tracker
            .ingest_text(
                "University politics are vicious precisely because the stakes are so small.",
            )
            .expect("candidate");
        let latest = latest_candidate(&candidate);
        let planned = planner.plan_candidate(latest);
        let diagnostics = planned.pho_like_diagnostics(latest);

        assert_eq!(diagnostics.candidate_id, latest.id.0);
        assert!(!diagnostics.entries.is_empty());
        let precisely = diagnostics
            .entries
            .iter()
            .find(|entry| entry.word == "precisely")
            .expect("precisely diagnostic");
        assert!(
            precisely.accent.is_some(),
            "focus diagnostics should emit accent hints for precision adverbs"
        );
        let final_entry = diagnostics.entries.last().expect("final entry");
        assert!(final_entry.boundary.is_some());
        assert_eq!(
            final_entry.realization_status,
            ProsodyRealizationStatus::Requested
        );
    }

    #[test]
    fn diagnostics_expose_vocative_classification_and_pause_behavior() {
        let mut tracker = PhonemeProsodyCandidateTracker::new(SimpleEnglishG2p::default());
        let mut planner = BreathGroupProsodyPlanner::new();
        let candidate = tracker.ingest_text("Thank you, Dave.").expect("candidate");
        let latest = latest_candidate(&candidate);
        let planned = planner.plan_candidate(latest);
        let diagnostics = planned.pho_like_diagnostics(latest);
        let dave = diagnostics
            .entries
            .iter()
            .find(|entry| entry.word == "dave")
            .expect("dave diagnostic");
        assert_eq!(dave.span.as_deref(), Some("Dave"));
        assert_eq!(dave.classification.as_deref(), Some("vocative"));
        assert_eq!(dave.pause_behavior.as_deref(), Some("suppressed"));
    }
}
