use crate::mouth::piper_native::g2p::{PhonemeProsodyCandidate, SpeechCandidateId};
use crate::mouth::piper_native::text::{ProsodyBoundaryHint, ProsodyCommitment};

const PAUSE_MS_DEFAULT: u64 = 140;
const PAUSE_MS_FINAL_CLOSURE: u64 = 260;
const CONTOUR_CONTINUING: (f32, f32, f32) = (0.82_f32, 0.10_f32, 1.0_f32);
const CONTOUR_PHRASE_BREAK: (f32, f32, f32) = (0.74_f32, 0.58_f32, 0.95_f32);
const CONTOUR_POSSIBLE_CLOSURE: (f32, f32, f32) = (0.34_f32, 0.76_f32, 0.90_f32);
const CONTOUR_FINAL_CLOSURE: (f32, f32, f32) = (0.08_f32, 0.92_f32, 0.86_f32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BreathGroupId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryState {
    Continuing,
    PossibleClosure,
    FinalClosure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProsodyEnergy {
    Low,
    Neutral,
    Elevated,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProsodyContour {
    pub energy: ProsodyEnergy,
    pub continuation_bias: f32,
    pub pause_likelihood: f32,
    pub speaking_rate_hint: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BreathGroupCandidate {
    pub id: BreathGroupId,
    pub source_candidate_id: SpeechCandidateId,
    pub text: String,
    pub contour: ProsodyContour,
    pub boundary_state: BoundaryState,
    pub commitment: ProsodyCommitment,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProsodyOverlay {
    pub target: ProsodyTarget,
    pub operation: ProsodyOperation,
    pub strength: u8,
    pub source: ProsodyOverlaySource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProsodyTarget {
    WholeCandidate,
    WordIndex { index: usize },
    WordRange { start: usize, end: usize },
    PhonemeRange { start: usize, end: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProsodyOverlaySource {
    Emoji(String),
    PromptTag(String),
    RuntimeAffect,
    Inference,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PauseOp {
    pub after: ProsodyTarget,
    pub millis: u64,
    pub commitment: ProsodyCommitment,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProsodyBoundaryHintOp {
    Continuing,
    PossibleClosure,
    FinalClosure,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProsodyOp {
    SetBaseContour(ProsodyContour),
    Stress {
        target: ProsodyTarget,
        strength: u8,
    },
    ApplyRhetoric {
        target: ProsodyTarget,
        op: ProsodyOperation,
        strength: u8,
    },
    Stretch {
        target: ProsodyTarget,
        factor: f32,
    },
    Compress {
        target: ProsodyTarget,
        factor: f32,
    },
    InsertPause(PauseOp),
    SetBoundary {
        target: ProsodyTarget,
        boundary: ProsodyBoundaryHintOp,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProsodyList {
    pub base: BreathGroupCandidate,
    pub ops: Vec<ProsodyOp>,
}

impl ProsodyList {
    pub fn apply_overlay(&mut self, overlay: ProsodyOverlay) {
        self.ops.push(ProsodyOp::ApplyRhetoric {
            target: overlay.target,
            op: overlay.operation,
            strength: overlay.strength,
        });
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

        if base.contour.pause_likelihood >= 0.5 {
            ops.push(ProsodyOp::InsertPause(PauseOp {
                after: ProsodyTarget::WholeCandidate,
                millis: if matches!(base.boundary_state, BoundaryState::FinalClosure) {
                    PAUSE_MS_FINAL_CLOSURE
                } else {
                    PAUSE_MS_DEFAULT
                },
                commitment: base.commitment,
            }));
        }

        let planned = ProsodyList { base, ops };
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
    let (base_continuation, pause_likelihood, speaking_rate_hint) = match candidate.boundary_hint {
        ProsodyBoundaryHint::None => CONTOUR_CONTINUING,
        ProsodyBoundaryHint::PhraseBreak => CONTOUR_PHRASE_BREAK,
        ProsodyBoundaryHint::PossibleSentenceEnd => CONTOUR_POSSIBLE_CLOSURE,
        ProsodyBoundaryHint::FinalSentenceEnd => CONTOUR_FINAL_CLOSURE,
    };

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

#[cfg(test)]
mod tests {
    use crate::mouth::piper_native::g2p::{
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
}
