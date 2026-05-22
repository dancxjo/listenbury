use serde::{Deserialize, Serialize};

use crate::soundscape::{
    IsolationPolicy, SoundscapeFrame, SourceCriterion, SourceId, SourceKind, SourceLabel,
    SourceOperation,
};

/// A source selected for destructive filtering.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SuppressionTarget {
    pub source_id: SourceId,
    pub strength: f32,
}

/// A source selected for observational tracking.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrackingTarget {
    pub source_id: SourceId,
    pub strength: f32,
}

/// Output of policy evaluation for one frame.
///
/// Suppression targets (destructive filtering) and tracking targets
/// (observational) are intentionally separated.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct IsolationEvaluation {
    pub suppressions: Vec<SuppressionTarget>,
    pub tracking: Vec<TrackingTarget>,
}

/// Transitional policy shim for self-hearing suppression.
pub fn self_hearing_suppression_policy(pete_voice_source_id: SourceId) -> IsolationPolicy {
    IsolationPolicy::suppress(SourceCriterion::KnownSource(pete_voice_source_id), 1.0)
}

/// Evaluate source-isolation policies against one soundscape frame.
///
/// First pass support includes `Suppress` and `Track`.
pub fn evaluate_policies(
    frame: &SoundscapeFrame,
    policies: &[IsolationPolicy],
) -> IsolationEvaluation {
    let mut result = IsolationEvaluation::default();
    for source in &frame.sources {
        for policy in policies {
            if !matches_criterion(source, policy.criterion) {
                continue;
            }
            let strength = policy.strength.clamp(0.0, 1.0);
            match policy.operation {
                SourceOperation::Suppress => {
                    result.suppressions.push(SuppressionTarget {
                        source_id: source.id,
                        strength,
                    });
                }
                SourceOperation::Track => {
                    result.tracking.push(TrackingTarget {
                        source_id: source.id,
                        strength,
                    });
                }
                SourceOperation::Enhance | SourceOperation::Extract => {}
            }
        }
    }
    result
}

fn matches_criterion(source: &crate::soundscape::SoundSource, criterion: SourceCriterion) -> bool {
    match criterion {
        SourceCriterion::KnownSource(source_id) => source.id == source_id,
        SourceCriterion::NotKnownSource(source_id) => source.id != source_id,
        SourceCriterion::MatchesSignature(_signature_id) => false,
        SourceCriterion::HumanSpeech => matches!(source.kind, SourceKind::Voice),
        SourceCriterion::SyntheticSpeech => {
            matches!(
                source.kind,
                SourceKind::SyntheticVoice | SourceKind::KnownSelfVoice
            )
        }
        SourceCriterion::Playback => matches!(source.kind, SourceKind::Playback),
        SourceCriterion::Foreground => !matches!(
            source.label,
            SourceLabel::BackgroundVoice { .. } | SourceLabel::RoomNoise
        ),
        SourceCriterion::Background => {
            matches!(
                source.label,
                SourceLabel::BackgroundVoice { .. } | SourceLabel::RoomNoise
            )
        }
        SourceCriterion::UnknownVoice => matches!(source.label, SourceLabel::UnknownVoice { .. }),
        SourceCriterion::CurrentlyAddressingPete => false,
    }
}

#[cfg(test)]
mod tests {
    use crate::soundscape::{
        IsolationPolicy, MixtureId, SoundEvent, SoundEventKind, SoundSource, SoundscapeFrame,
        SourceCriterion, SourceId, SourceKind, SourceLabel, TimePoint, TimeRange,
        evaluate_policies, self_hearing_suppression_policy,
    };

    #[test]
    fn pete_playback_suppression_and_unknown_voice_tracking_share_policy_machinery() {
        let range = TimeRange::new(TimePoint::from_millis(1_000), TimePoint::from_millis(1_100));
        let pete_playback_source = SourceId::new();
        let unknown_voice_source = SourceId::new();
        let frame = SoundscapeFrame {
            range,
            sources: vec![
                SoundSource {
                    id: pete_playback_source,
                    kind: SourceKind::Playback,
                    label: SourceLabel::Playback("Pete".into()),
                    confidence: 0.98,
                },
                SoundSource {
                    id: unknown_voice_source,
                    kind: SourceKind::Voice,
                    label: SourceLabel::UnknownVoice { ordinal: 1 },
                    confidence: 0.83,
                },
            ],
            events: vec![
                SoundEvent {
                    id: Default::default(),
                    source_id: pete_playback_source,
                    kind: SoundEventKind::PlaybackActivity,
                    range,
                    confidence: 0.9,
                },
                SoundEvent {
                    id: Default::default(),
                    source_id: unknown_voice_source,
                    kind: SoundEventKind::VoiceActivity,
                    range,
                    confidence: 0.8,
                },
            ],
            mixtures: vec![crate::soundscape::AcousticMixture {
                id: MixtureId::new(),
                event_ids: vec![],
                contributions: vec![],
            }],
        };

        let policies = vec![
            self_hearing_suppression_policy(pete_playback_source),
            IsolationPolicy::track(SourceCriterion::UnknownVoice, 0.7),
        ];
        let evaluation = evaluate_policies(&frame, &policies);

        assert_eq!(
            evaluation.suppressions,
            vec![crate::soundscape::SuppressionTarget {
                source_id: pete_playback_source,
                strength: 1.0
            }]
        );
        assert_eq!(
            evaluation.tracking,
            vec![crate::soundscape::TrackingTarget {
                source_id: unknown_voice_source,
                strength: 0.7
            }]
        );
    }
}
