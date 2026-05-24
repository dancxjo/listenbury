use crate::audio::AudioFrame;
use crate::hearing::VadResult;
use crate::mouth::player::PlaybackEvent;
use crate::soundscape::{
    AcousticContribution, AcousticMixture, AttributionEvidence, EventId, ExpectedSound, MixtureId,
    ObservedSound, SoundEvent, SoundEventKind, SoundSource, SoundscapeFrame,
    SourceAttributedTranscript, SourceHypothesis, SourceId, SourceKind, SourceLabel, TimePoint,
    TimeRange, TranscriptHypothesis,
};
use crate::speech::transcript::TranscriptChunk;

const DEFAULT_PLAYBACK_LABEL: &str = "Playback";

/// Thin adapter between current pipeline events and soundscape inputs/outputs.
///
/// Current event seams wired here:
/// - audio input: [`AudioFrame`]
/// - playback output: [`PlaybackEvent::SyntheticStarted`]
/// - VAD: [`VadResult`]
/// - ASR: [`TranscriptChunk`]
#[derive(Debug, Clone)]
pub struct SoundscapePipelineAdapter {
    playback_source_id: SourceId,
    microphone_source_id: SourceId,
    playback_label: String,
    unknown_voice_ordinal: u32,
}

impl Default for SoundscapePipelineAdapter {
    fn default() -> Self {
        Self {
            playback_source_id: SourceId::new(),
            microphone_source_id: SourceId::new(),
            playback_label: DEFAULT_PLAYBACK_LABEL.to_string(),
            unknown_voice_ordinal: 1,
        }
    }
}

impl SoundscapePipelineAdapter {
    pub fn with_playback_label(playback_label: impl Into<String>) -> Self {
        Self {
            playback_label: playback_label.into(),
            ..Self::default()
        }
    }

    pub fn observed_from_audio_vad_asr(
        &self,
        frame: &AudioFrame,
        vad: Option<VadResult>,
        asr: Option<&TranscriptChunk>,
    ) -> ObservedSound {
        let mut observed = ObservedSound::from_audio_frame(frame);
        if let Some(chunk) = asr {
            let confidence = vad
                .map(|result| result.speech_prob.clamp(0.0, 1.0))
                .unwrap_or(if chunk.is_final { 0.9 } else { 0.6 });
            observed.transcript_hypotheses.push(TranscriptHypothesis {
                text: chunk.text.clone(),
                confidence,
            });
        }
        observed
    }

    pub fn expected_from_playback_event(
        &self,
        event: &PlaybackEvent,
        rendered_frame: Option<&AudioFrame>,
    ) -> Option<ExpectedSound> {
        let (text, at) = match event {
            PlaybackEvent::SyntheticStarted { text, at, .. } => (text, *at),
            _ => return None,
        };
        let start_ms = nanos_to_millis(at.unix_nanos);
        let duration_ms = rendered_frame
            .map(audio_frame_duration_millis)
            .unwrap_or_default();
        let range = TimeRange::new(
            TimePoint::from_millis(start_ms),
            TimePoint::from_millis(start_ms.saturating_add(duration_ms)),
        );
        let expected_samples = rendered_frame
            .map(|frame| frame.samples.clone())
            .unwrap_or_default();
        Some(ExpectedSound {
            source_id: self.playback_source_id,
            expected_range: range,
            expected_text: Some(text.clone()),
            expected_samples,
            confidence: 0.9,
        })
    }

    pub fn emit_frame(
        &self,
        observed: &ObservedSound,
        expected: Option<&ExpectedSound>,
        vad: Option<VadResult>,
    ) -> SoundscapeFrame {
        let mut sources = Vec::new();
        let mut events = Vec::new();
        let mut contributions = Vec::new();

        if expected.is_some() {
            sources.push(SoundSource {
                id: self.playback_source_id,
                kind: SourceKind::Playback,
                label: SourceLabel::Playback(self.playback_label.clone()),
                confidence: 0.9,
            });
        }

        let voice_confidence = vad
            .map(|result| result.speech_prob.clamp(0.0, 1.0))
            .unwrap_or(0.5);
        let is_voice_active = vad.map(|result| result.is_speech).unwrap_or(true);
        if is_voice_active || !observed.transcript_hypotheses.is_empty() {
            sources.push(SoundSource {
                id: self.microphone_source_id,
                kind: SourceKind::Voice,
                label: SourceLabel::UnknownVoice {
                    ordinal: self.unknown_voice_ordinal,
                },
                confidence: voice_confidence,
            });
            events.push(SoundEvent {
                id: EventId::new(),
                source_id: self.microphone_source_id,
                kind: SoundEventKind::VoiceActivity,
                range: observed.range,
                confidence: voice_confidence,
            });
            contributions.push(AcousticContribution {
                source_id: self.microphone_source_id,
                gain: voice_confidence.max(0.01),
            });
        }

        if let Some(expected) = expected {
            events.push(SoundEvent {
                id: EventId::new(),
                source_id: self.playback_source_id,
                kind: SoundEventKind::PlaybackActivity,
                range: expected.expected_range,
                confidence: expected.confidence.clamp(0.0, 1.0),
            });
            contributions.push(AcousticContribution {
                source_id: self.playback_source_id,
                gain: expected.confidence.clamp(0.0, 1.0).max(0.01),
            });
        }

        let range = expected
            .map(|candidate| span_union(observed.range, candidate.expected_range))
            .unwrap_or(observed.range);
        let mixtures = if contributions.len() > 1 {
            vec![AcousticMixture {
                id: MixtureId::new(),
                event_ids: events.iter().map(|event| event.id).collect(),
                contributions,
            }]
        } else {
            Vec::new()
        };

        SoundscapeFrame {
            range,
            sources,
            events,
            mixtures,
        }
    }

    pub fn source_attributed_transcript(
        &self,
        observed: &ObservedSound,
        asr: &TranscriptChunk,
        vad: Option<VadResult>,
    ) -> SourceAttributedTranscript {
        let attribution_confidence = vad
            .map(|result| result.speech_prob.clamp(0.0, 1.0))
            .unwrap_or(0.5);
        SourceAttributedTranscript {
            range: observed.range,
            source_hypothesis: SourceHypothesis {
                source_id: Some(self.microphone_source_id),
                kind: SourceKind::Voice,
                range: observed.range,
                confidence: attribution_confidence,
                evidence: vec![AttributionEvidence::EnergyChange],
            },
            source_label: SourceLabel::UnknownVoice {
                ordinal: self.unknown_voice_ordinal,
            },
            text: asr.text.clone(),
            transcript_confidence: if asr.is_final { 0.9 } else { 0.6 },
            attribution_confidence,
            overlap: None,
        }
    }
}

fn nanos_to_millis(unix_nanos: u128) -> u64 {
    (unix_nanos / 1_000_000).min(u128::from(u64::MAX)) as u64
}

fn audio_frame_duration_millis(frame: &AudioFrame) -> u64 {
    let channel_count = usize::from(frame.channels.max(1));
    let per_channel_samples = frame.samples.len().div_ceil(channel_count);
    if frame.sample_rate_hz == 0 {
        0
    } else {
        ((per_channel_samples as u128).saturating_mul(1_000) / u128::from(frame.sample_rate_hz))
            .min(u128::from(u64::MAX)) as u64
    }
}

fn span_union(a: TimeRange, b: TimeRange) -> TimeRange {
    TimeRange::new(
        TimePoint::from_millis(a.start.millis.min(b.start.millis)),
        TimePoint::from_millis(a.end.millis.max(b.end.millis)),
    )
}

#[cfg(test)]
mod tests {
    use crate::mouth::player::PlaybackUnitId;
    use crate::time::ExactTimestamp;

    use super::*;

    #[test]
    fn adapters_cover_audio_playback_vad_and_asr() {
        let adapter = SoundscapePipelineAdapter::default();
        let mic = AudioFrame {
            captured_at: ExactTimestamp::from_unix_nanos(1_000_000_000),
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![0.2; 160],
            voice_signatures: Vec::new(),
        };
        let vad = VadResult {
            speech_prob: 0.88,
            is_speech: true,
        };
        let asr = TranscriptChunk {
            text: "hello from microphone".to_string(),
            is_final: true,
        };
        let playback = PlaybackEvent::SyntheticStarted {
            id: PlaybackUnitId(7),
            text: "hello from speaker".to_string(),
            at: ExactTimestamp::from_unix_nanos(1_005_000_000),
        };
        let render = AudioFrame {
            captured_at: ExactTimestamp::from_unix_nanos(1_005_000_000),
            sample_rate_hz: 22_050,
            channels: 1,
            samples: vec![0.1; 220],
            voice_signatures: Vec::new(),
        };

        let observed = adapter.observed_from_audio_vad_asr(&mic, Some(vad), Some(&asr));
        let expected = adapter
            .expected_from_playback_event(&playback, Some(&render))
            .expect("playback expected sound");
        let frame = adapter.emit_frame(&observed, Some(&expected), Some(vad));
        let transcript = adapter.source_attributed_transcript(&observed, &asr, Some(vad));

        assert_eq!(observed.transcript_hypotheses.len(), 1);
        assert_eq!(
            expected.expected_text.as_deref(),
            Some("hello from speaker"),
            "playback event text should map into expected sound"
        );
        assert!(
            frame
                .events
                .iter()
                .any(|event| event.kind == SoundEventKind::VoiceActivity)
        );
        assert!(
            frame
                .events
                .iter()
                .any(|event| event.kind == SoundEventKind::PlaybackActivity)
        );
        assert_eq!(transcript.text, "hello from microphone");
        assert_eq!(
            transcript.source_label,
            SourceLabel::UnknownVoice { ordinal: 1 }
        );
    }
}
