use serde::{Deserialize, Serialize};

use crate::soundscape::{AcousticMixture, SoundEvent, SoundSource, TimeRange};

/// Time-bounded view of the active sources, events, and mixtures.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoundscapeFrame {
    pub range: TimeRange,
    pub sources: Vec<SoundSource>,
    pub events: Vec<SoundEvent>,
    pub mixtures: Vec<AcousticMixture>,
}

#[cfg(test)]
mod tests {
    use crate::soundscape::{
        AcousticContribution, AcousticMixture, EventId, MixtureId, SoundEvent, SoundEventKind,
        SoundSource, SoundscapeFrame, SourceId, SourceKind, SourceLabel, TimePoint, TimeRange,
    };

    #[test]
    fn constructs_frame_with_pete_playback_unknown_voice_and_room_noise() {
        let range = TimeRange::new(TimePoint::from_millis(1_000), TimePoint::from_millis(1_350));

        let pete_playback_source_id = SourceId::new();
        let unknown_voice_source_id = SourceId::new();
        let room_noise_source_id = SourceId::new();

        let sources = vec![
            SoundSource {
                id: pete_playback_source_id,
                kind: SourceKind::Playback,
                label: SourceLabel::Playback("Pete".into()),
                confidence: 0.96,
            },
            SoundSource {
                id: unknown_voice_source_id,
                kind: SourceKind::Voice,
                label: SourceLabel::UnknownVoice { ordinal: 1 },
                confidence: 0.82,
            },
            SoundSource {
                id: room_noise_source_id,
                kind: SourceKind::EnvironmentalNoise,
                label: SourceLabel::RoomNoise,
                confidence: 0.91,
            },
        ];

        let playback_event_id = EventId::new();
        let voice_event_id = EventId::new();
        let noise_event_id = EventId::new();
        let events = vec![
            SoundEvent {
                id: playback_event_id,
                source_id: pete_playback_source_id,
                kind: SoundEventKind::PlaybackActivity,
                range,
                confidence: 0.94,
            },
            SoundEvent {
                id: voice_event_id,
                source_id: unknown_voice_source_id,
                kind: SoundEventKind::VoiceActivity,
                range,
                confidence: 0.8,
            },
            SoundEvent {
                id: noise_event_id,
                source_id: room_noise_source_id,
                kind: SoundEventKind::EnvironmentalNoise,
                range,
                confidence: 0.87,
            },
        ];

        let mixtures = vec![AcousticMixture {
            id: MixtureId::new(),
            event_ids: vec![playback_event_id, voice_event_id, noise_event_id],
            contributions: vec![
                AcousticContribution {
                    source_id: pete_playback_source_id,
                    gain: 0.63,
                },
                AcousticContribution {
                    source_id: unknown_voice_source_id,
                    gain: 0.22,
                },
                AcousticContribution {
                    source_id: room_noise_source_id,
                    gain: 0.15,
                },
            ],
        }];

        let frame = SoundscapeFrame {
            range,
            sources,
            events,
            mixtures,
        };

        assert_eq!(frame.sources.len(), 3);
        assert_eq!(frame.events.len(), 3);
        assert_eq!(frame.mixtures.len(), 1);
        assert_eq!(frame.sources[0].label.display_label(), "_PETE PLAYBACK_");
        assert_eq!(frame.sources[1].label.display_label(), "_UNKNOWN VOICE #1_");
        assert_eq!(frame.sources[2].label.display_label(), "_ROOM NOISE_");
    }
}
