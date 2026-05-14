use crate::mind::llm::LlmEvent;
use crate::mind::turn::{TurnState, TurnTracker};
use crate::mouth::planner::{ExpressiveUnit, MouthCommand, SpeechPlan, SpeechPlanner, SpeechUnit};

pub const DEFAULT_FILLER_REPEAT_COOLDOWN_MS: u64 = 60_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackchannelId {
    Okay,
    Right,
    ISee,
    MmHm,
    LetMeThink,
    OneThingJumpsOut,
    ThatMakesSense,
}

impl BackchannelId {
    pub fn text(self) -> &'static str {
        match self {
            BackchannelId::Okay => "Okay.",
            BackchannelId::Right => "Right.",
            BackchannelId::ISee => "I see.",
            BackchannelId::MmHm => "Mm-hm.",
            BackchannelId::LetMeThink => "Let me think.",
            BackchannelId::OneThingJumpsOut => "One thing jumps out.",
            BackchannelId::ThatMakesSense => "That makes sense.",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FillerDecision {
    Silence,
    PlayCachedBackchannel { id: BackchannelId },
    SynthesizeBackchannel { text: String },
}

#[derive(Debug, Clone)]
pub struct FillerContext {
    pub turn_state: TurnState,
    pub transcript_so_far: Option<String>,
    pub vad_confidence: f32,
    pub silence_duration_ms: u64,
    pub main_llm_started_at_ms: Option<u64>,
    pub main_llm_has_emitted_token: bool,
    pub main_llm_has_safe_speech_unit: bool,
    pub user_interrupted_recently: bool,
    pub now_ms: u64,
    pub user_turn_id: Option<u64>,
}

impl Default for FillerContext {
    fn default() -> Self {
        Self {
            turn_state: TurnState::Idle,
            transcript_so_far: None,
            vad_confidence: 0.0,
            silence_duration_ms: 0,
            main_llm_started_at_ms: None,
            main_llm_has_emitted_token: false,
            main_llm_has_safe_speech_unit: false,
            user_interrupted_recently: false,
            now_ms: 0,
            user_turn_id: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FillerPlannerConfig {
    pub min_silence_for_filler_ms: u64,
    pub repeat_cooldown_ms: u64,
    pub allow_multiple_fillers_per_turn: bool,
}

impl Default for FillerPlannerConfig {
    fn default() -> Self {
        Self {
            min_silence_for_filler_ms: 800,
            repeat_cooldown_ms: DEFAULT_FILLER_REPEAT_COOLDOWN_MS,
            allow_multiple_fillers_per_turn: false,
        }
    }
}

#[derive(Debug, Default)]
pub struct FillerPlanner {
    config: FillerPlannerConfig,
    last_filler: Option<(BackchannelId, u64)>,
    last_filler_turn_id: Option<u64>,
}

impl FillerPlanner {
    pub fn new(config: FillerPlannerConfig) -> Self {
        Self {
            config,
            ..Self::default()
        }
    }

    pub fn decide(&mut self, ctx: &FillerContext) -> FillerDecision {
        if ctx.turn_state != TurnState::PeteThinking
            || ctx.user_interrupted_recently
            || ctx.main_llm_has_safe_speech_unit
            || ctx.main_llm_has_emitted_token
            || ctx.silence_duration_ms < self.config.min_silence_for_filler_ms
            || ctx.vad_confidence >= 0.5
            || ctx.main_llm_started_at_ms.is_none()
        {
            return FillerDecision::Silence;
        }

        if !self.config.allow_multiple_fillers_per_turn
            && ctx.user_turn_id.is_some()
            && ctx.user_turn_id == self.last_filler_turn_id
        {
            return FillerDecision::Silence;
        }

        let selected = select_backchannel(ctx.transcript_so_far.as_deref());
        if let Some((last_id, last_used_at_ms)) = self.last_filler {
            let still_in_cooldown = ctx.now_ms.saturating_sub(last_used_at_ms) < self.config.repeat_cooldown_ms;
            if still_in_cooldown && last_id == selected {
                return FillerDecision::Silence;
            }
        }

        self.last_filler = Some((selected, ctx.now_ms));
        self.last_filler_turn_id = ctx.user_turn_id;
        FillerDecision::PlayCachedBackchannel { id: selected }
    }
}

fn select_backchannel(transcript_so_far: Option<&str>) -> BackchannelId {
    let Some(transcript) = transcript_so_far.map(str::trim) else {
        return BackchannelId::Okay;
    };
    if transcript.is_empty() {
        return BackchannelId::Okay;
    }

    if transcript.ends_with('?') {
        BackchannelId::LetMeThink
    } else if transcript.len() > 80 {
        BackchannelId::OneThingJumpsOut
    } else {
        BackchannelId::Okay
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimePacket {
    UserStartedSpeaking,
    UserStoppedSpeaking,
    TranscriptUpdated { text: String, confidence: f32 },
    BackchannelPlayed { id: BackchannelId },
    SpeechUnitCommitted { text: String },
    TtsQueueChanged { queued_ms: u64 },
    FaceChanged { emoji: String },
    InterruptionDetected,
}

#[derive(Debug, Default)]
pub struct ConversationController {
    pub turn_tracker: TurnTracker,
    pub filler_planner: FillerPlanner,
    pub speech_planner: SpeechPlanner,
    pending_runtime_packets: Vec<RuntimePacket>,
    runtime_context: Vec<RuntimePacket>,
}

impl ConversationController {
    pub fn record_runtime_packet(&mut self, packet: RuntimePacket) {
        self.pending_runtime_packets.push(packet);
    }

    pub fn apply_safe_boundary_updates(&mut self) {
        self.runtime_context
            .append(&mut self.pending_runtime_packets);
    }

    pub fn runtime_context(&self) -> &[RuntimePacket] {
        &self.runtime_context
    }

    pub fn decide_filler_command(&mut self, ctx: &FillerContext) -> Option<MouthCommand> {
        match self.filler_planner.decide(ctx) {
            FillerDecision::Silence => None,
            FillerDecision::PlayCachedBackchannel { id } => {
                self.record_runtime_packet(RuntimePacket::BackchannelPlayed { id });
                Some(MouthCommand::Speak(SpeechPlan::from(SpeechUnit::Backchannel(
                    id.text().to_string(),
                ))))
            }
            FillerDecision::SynthesizeBackchannel { text } => {
                Some(MouthCommand::Speak(SpeechPlan::from(SpeechUnit::Backchannel(
                    text,
                ))))
            }
        }
    }

    pub fn ingest_llm_events(&mut self, events: &[LlmEvent]) -> Vec<ExpressiveUnit> {
        self.speech_planner.ingest(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thinking_context(now_ms: u64, turn_id: u64) -> FillerContext {
        FillerContext {
            turn_state: TurnState::PeteThinking,
            transcript_so_far: Some("Can you explain this?".to_string()),
            vad_confidence: 0.1,
            silence_duration_ms: 1_200,
            main_llm_started_at_ms: Some(now_ms.saturating_sub(300)),
            main_llm_has_emitted_token: false,
            main_llm_has_safe_speech_unit: false,
            user_interrupted_recently: false,
            now_ms,
            user_turn_id: Some(turn_id),
        }
    }

    #[test]
    fn planner_prefers_silence_when_llm_already_has_tokens() {
        let mut planner = FillerPlanner::default();
        let mut ctx = thinking_context(10_000, 1);
        ctx.main_llm_has_emitted_token = true;
        assert_eq!(planner.decide(&ctx), FillerDecision::Silence);
    }

    #[test]
    fn planner_chooses_cached_backchannel_while_waiting() {
        let mut planner = FillerPlanner::default();
        let ctx = thinking_context(10_000, 1);
        assert_eq!(
            planner.decide(&ctx),
            FillerDecision::PlayCachedBackchannel {
                id: BackchannelId::LetMeThink
            }
        );
    }

    #[test]
    fn planner_avoids_repeating_same_filler_within_cooldown() {
        let mut planner = FillerPlanner::default();
        let first = thinking_context(10_000, 1);
        let second = thinking_context(10_500, 2);
        assert!(matches!(
            planner.decide(&first),
            FillerDecision::PlayCachedBackchannel { .. }
        ));
        assert_eq!(planner.decide(&second), FillerDecision::Silence);
    }

    #[test]
    fn planner_emits_only_one_filler_per_turn_by_default() {
        let mut planner = FillerPlanner::default();
        let first = thinking_context(10_000, 9);
        let second = thinking_context(80_000, 9);
        assert!(matches!(
            planner.decide(&first),
            FillerDecision::PlayCachedBackchannel { .. }
        ));
        assert_eq!(planner.decide(&second), FillerDecision::Silence);
    }

    #[test]
    fn controller_paths_cached_backchannel_to_speech_unit() {
        let mut controller = ConversationController::default();
        let ctx = thinking_context(10_000, 1);
        let command = controller.decide_filler_command(&ctx);
        assert!(matches!(
            command,
            Some(MouthCommand::Speak(plan))
                if plan.unit() == &SpeechUnit::Backchannel("Let me think.".to_string())
        ));
    }

    #[test]
    fn controller_appends_runtime_packets_at_safe_boundaries() {
        let mut controller = ConversationController::default();
        controller.record_runtime_packet(RuntimePacket::UserStartedSpeaking);
        controller.record_runtime_packet(RuntimePacket::InterruptionDetected);
        assert!(controller.runtime_context().is_empty());

        controller.apply_safe_boundary_updates();
        assert_eq!(
            controller.runtime_context(),
            &[
                RuntimePacket::UserStartedSpeaking,
                RuntimePacket::InterruptionDetected
            ]
        );
    }
}
