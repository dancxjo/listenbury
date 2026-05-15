use crate::event::HearingEvent;
use crate::mind::llm::LlmEvent;
use crate::mind::turn::{TurnState, TurnTracker};
use crate::mouth::planner::{ExpressiveUnit, MouthCommand, SpeechPlan, SpeechPlanner, SpeechUnit};

pub const DEFAULT_FILLER_REPEAT_COOLDOWN_MS: u64 = 60_000;
pub const DEFAULT_INTERRUPT_BLIP_MS: u64 = 80;
pub const DEFAULT_INTERRUPT_FADE_THRESHOLD_MS: u64 = 160;
pub const DEFAULT_INTERRUPT_STOP_THRESHOLD_MS: u64 = 450;
pub const DEFAULT_INTERRUPT_FADEOUT_MS: u64 = 180;

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
            BackchannelId::OneThingJumpsOut => "Well, boy howdy! I declare.",
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
            let still_in_cooldown =
                ctx.now_ms.saturating_sub(last_used_at_ms) < self.config.repeat_cooldown_ms;
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

#[derive(Debug, Clone, Copy)]
pub struct InterruptionPolicy {
    pub ignore_blip_ms: u64,
    pub fade_threshold_ms: u64,
    pub stop_threshold_ms: u64,
    pub fade_out_ms: u64,
}

impl Default for InterruptionPolicy {
    fn default() -> Self {
        Self {
            ignore_blip_ms: DEFAULT_INTERRUPT_BLIP_MS,
            fade_threshold_ms: DEFAULT_INTERRUPT_FADE_THRESHOLD_MS,
            stop_threshold_ms: DEFAULT_INTERRUPT_STOP_THRESHOLD_MS,
            fade_out_ms: DEFAULT_INTERRUPT_FADEOUT_MS,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InterruptionDecision {
    pub mouth_command: Option<MouthCommand>,
    pub cancel_generation: bool,
    pub clear_tts_queue: bool,
}

impl Default for InterruptionDecision {
    fn default() -> Self {
        Self {
            mouth_command: None,
            cancel_generation: false,
            clear_tts_queue: false,
        }
    }
}

#[derive(Debug)]
pub struct ConversationController {
    pub turn_tracker: TurnTracker,
    pub filler_planner: FillerPlanner,
    pub speech_planner: SpeechPlanner,
    pub interruption_policy: InterruptionPolicy,
    pending_runtime_packets: Vec<RuntimePacket>,
    runtime_context: Vec<RuntimePacket>,
    interruption_started_at_ms: Option<u64>,
    interruption_faded: bool,
    interruption_recorded: bool,
}

impl Default for ConversationController {
    fn default() -> Self {
        Self {
            turn_tracker: TurnTracker::default(),
            filler_planner: FillerPlanner::default(),
            speech_planner: SpeechPlanner::default(),
            interruption_policy: InterruptionPolicy::default(),
            pending_runtime_packets: Vec::new(),
            runtime_context: Vec::new(),
            interruption_started_at_ms: None,
            interruption_faded: false,
            interruption_recorded: false,
        }
    }
}

impl ConversationController {
    pub fn on_pete_speech_started(&mut self) {
        self.turn_tracker.on_pete_speech_started();
        self.reset_interruption_state();
        self.interruption_recorded = false;
    }

    pub fn on_hearing_event(&mut self, event: &HearingEvent, now_ms: u64) -> InterruptionDecision {
        let was_pete_outputting = matches!(
            self.turn_tracker.state(),
            TurnState::PeteSpeaking | TurnState::PeteInterrupted
        );
        self.turn_tracker.on_hearing_event(event);

        match event {
            HearingEvent::SpeechStarted => {
                if was_pete_outputting {
                    self.interruption_started_at_ms.get_or_insert(now_ms);
                }
                InterruptionDecision::default()
            }
            HearingEvent::SpeechContinued { .. } => self.interruption_decision(now_ms),
            HearingEvent::PauseStarted | HearingEvent::BreathGroupClosed { .. } => {
                self.reset_interruption_state();
                InterruptionDecision::default()
            }
            HearingEvent::BreathGroupOpened { .. } => InterruptionDecision::default(),
        }
    }

    fn interruption_decision(&mut self, now_ms: u64) -> InterruptionDecision {
        let Some(started_at_ms) = self.interruption_started_at_ms else {
            return InterruptionDecision::default();
        };

        let elapsed_ms = now_ms.saturating_sub(started_at_ms);
        if elapsed_ms <= self.interruption_policy.ignore_blip_ms {
            return InterruptionDecision::default();
        }

        if elapsed_ms >= self.interruption_policy.stop_threshold_ms {
            self.turn_tracker.on_pete_interrupted();
            self.record_interruption_packet_once();
            self.reset_interruption_state();
            return InterruptionDecision {
                mouth_command: Some(MouthCommand::StopNow),
                cancel_generation: true,
                clear_tts_queue: true,
            };
        }

        if elapsed_ms >= self.interruption_policy.fade_threshold_ms && !self.interruption_faded {
            self.turn_tracker.on_pete_interrupted();
            self.record_interruption_packet_once();
            self.interruption_faded = true;
            return InterruptionDecision {
                mouth_command: Some(MouthCommand::FadeOut {
                    millis: self.interruption_policy.fade_out_ms,
                }),
                cancel_generation: true,
                clear_tts_queue: false,
            };
        }

        InterruptionDecision::default()
    }

    fn reset_interruption_state(&mut self) {
        self.interruption_started_at_ms = None;
        self.interruption_faded = false;
    }

    fn record_interruption_packet_once(&mut self) {
        if self.interruption_recorded {
            return;
        }
        self.record_runtime_packet(RuntimePacket::InterruptionDetected);
        self.interruption_recorded = true;
    }

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
                Some(MouthCommand::Speak(SpeechPlan::from(
                    SpeechUnit::Backchannel(id.text().to_string()),
                )))
            }
            FillerDecision::SynthesizeBackchannel { text } => Some(MouthCommand::Speak(
                SpeechPlan::from(SpeechUnit::Backchannel(text)),
            )),
        }
    }

    pub fn ingest_llm_events(&mut self, events: &[LlmEvent]) -> Vec<ExpressiveUnit> {
        self.speech_planner.ingest(events)
    }
}

#[cfg(test)]
mod tests {
    use crate::event::HearingEvent;

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

    #[test]
    fn interruption_policy_ignores_brief_user_blips() {
        let mut controller = ConversationController::default();
        controller.on_pete_speech_started();
        let started = controller.on_hearing_event(&HearingEvent::SpeechStarted, 1_000);
        let continued =
            controller.on_hearing_event(&HearingEvent::SpeechContinued { speech_prob: 0.9 }, 1_060);

        assert!(started.mouth_command.is_none());
        assert!(continued.mouth_command.is_none());
        assert_eq!(controller.turn_tracker.state(), TurnState::PeteSpeaking);
    }

    #[test]
    fn interruption_policy_fades_then_stops_for_sustained_speech() {
        let mut controller = ConversationController::default();
        controller.on_pete_speech_started();
        controller.on_hearing_event(&HearingEvent::SpeechStarted, 5_000);

        let fade =
            controller.on_hearing_event(&HearingEvent::SpeechContinued { speech_prob: 0.9 }, 5_180);
        assert!(matches!(
            fade.mouth_command,
            Some(MouthCommand::FadeOut { millis: 180 })
        ));
        assert!(fade.cancel_generation);
        assert!(!fade.clear_tts_queue);

        let stop =
            controller.on_hearing_event(&HearingEvent::SpeechContinued { speech_prob: 0.9 }, 5_500);
        assert!(matches!(stop.mouth_command, Some(MouthCommand::StopNow)));
        assert!(stop.cancel_generation);
        assert!(stop.clear_tts_queue);
        assert_eq!(controller.turn_tracker.state(), TurnState::PeteInterrupted);

        controller.apply_safe_boundary_updates();
        assert_eq!(
            controller
                .runtime_context()
                .iter()
                .filter(|packet| matches!(packet, RuntimePacket::InterruptionDetected))
                .count(),
            1
        );
    }
}
