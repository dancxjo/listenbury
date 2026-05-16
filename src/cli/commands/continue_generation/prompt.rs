#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use super::*;

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
#[derive(Debug, Clone, Copy)]
pub(super) struct ContinuePromptGateConfig {
    pub(super) duplicate_suppression_window: Duration,
    pub(super) auditory_min_interval: Duration,
    pub(super) overlap_summary_threshold: usize,
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
impl Default for ContinuePromptGateConfig {
    fn default() -> Self {
        Self {
            duplicate_suppression_window: Duration::from_millis(1_500),
            auditory_min_interval: Duration::from_millis(800),
            overlap_summary_threshold: 2,
        }
    }
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
trait PromptGate {
    fn consider_ear_event(&mut self, event: &ContinueEarEvent, now: Instant) -> Vec<PromptPacket>;
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
#[derive(Debug)]
pub(super) struct ContinuePromptGate {
    config: ContinuePromptGateConfig,
    last_emitted_packet: Option<String>,
    last_emitted_at: Option<Instant>,
    last_auditory_at: Option<Instant>,
    pending_overlap_count: usize,
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
impl Default for ContinuePromptGate {
    fn default() -> Self {
        Self::new(ContinuePromptGateConfig::default())
    }
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
impl ContinuePromptGate {
    pub(super) fn new(config: ContinuePromptGateConfig) -> Self {
        Self {
            config,
            last_emitted_packet: None,
            last_emitted_at: None,
            last_auditory_at: None,
            pending_overlap_count: 0,
        }
    }

    pub(super) fn consider_ear_event(
        &mut self,
        event: &ContinueEarEvent,
        now: Instant,
    ) -> Vec<PromptPacket> {
        <Self as PromptGate>::consider_ear_event(self, event, now)
    }

    fn flush_overlap_summary(&mut self, now: Instant) -> Option<PromptPacket> {
        if self.pending_overlap_count == 0 {
            return None;
        }
        self.pending_overlap_count = 0;
        Some(PromptPacket::ear_observation(
            "Pete heard overlapping speech while speaking.".to_string(),
        ))
        .and_then(|packet| self.filter_packet(packet, now))
    }

    fn filter_packet(&mut self, packet: PromptPacket, now: Instant) -> Option<PromptPacket> {
        let is_important = matches!(packet.memory, PromptMemory::Listened(_));
        if !is_important
            && matches!(packet.memory, PromptMemory::AuditoryObservation(_))
            && self.last_auditory_at.is_some_and(|last| {
                now.saturating_duration_since(last) < self.config.auditory_min_interval
            })
        {
            return None;
        }

        let signature = packet.text.clone();
        if !is_important
            && self.last_emitted_packet.as_deref() == Some(signature.as_str())
            && self.last_emitted_at.is_some_and(|last| {
                now.saturating_duration_since(last) < self.config.duplicate_suppression_window
            })
        {
            return None;
        }

        if matches!(packet.memory, PromptMemory::AuditoryObservation(_)) {
            self.last_auditory_at = Some(now);
        }
        self.last_emitted_packet = Some(signature);
        self.last_emitted_at = Some(now);
        Some(packet)
    }
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
impl PromptGate for ContinuePromptGate {
    fn consider_ear_event(&mut self, event: &ContinueEarEvent, now: Instant) -> Vec<PromptPacket> {
        let mut packets = Vec::new();
        if !matches!(
            event,
            ContinueEarEvent::OverlapDetected { .. } | ContinueEarEvent::SelfVoiceHeard { .. }
        ) {
            if let Some(packet) = self.flush_overlap_summary(now) {
                packets.push(packet);
            }
        }

        match event {
            ContinueEarEvent::OverlapDetected { .. } => {
                self.pending_overlap_count += 1;
                if self.pending_overlap_count >= self.config.overlap_summary_threshold {
                    if let Some(packet) = self.flush_overlap_summary(now) {
                        packets.push(packet);
                    }
                }
            }
            ContinueEarEvent::SelfVoiceHeard { .. } => {
                // Ignore raw self-hearing telemetry here; overlap summaries carry the salient signal.
            }
            _ => {
                if let Some(packet) = event
                    .direct_prompt_packet()
                    .and_then(|packet| self.filter_packet(packet, now))
                {
                    packets.push(packet);
                }
            }
        }

        packets
    }
}
