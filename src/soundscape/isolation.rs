use serde::{Deserialize, Serialize};

use crate::audio::frame::AudioFrame;
use crate::hearing::suppression::SpeakerReferenceMask;
use crate::soundscape::{
    IsolationPolicy, SoundSource, SoundscapeContext, SoundscapeFrame, SourceCriterion, SourceId,
    SourceKind, SourceLabel, SourceOperation,
};
use crate::time::ExactTimestamp;

/// Audio chunk passed into source-separation backends.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AudioSpan {
    pub frames: Vec<AudioFrame>,
}

/// Backend-agnostic source-separation adapter.
pub trait SourceSeparator {
    fn separate(
        &mut self,
        input: &AudioSpan,
        target: &SourceCriterion,
        context: &SoundscapeContext,
    ) -> SeparationResult;
}

#[derive(Debug, Clone, PartialEq)]
pub struct SeparationResult {
    pub selected: Option<AudioSpan>,
    pub residual: Option<AudioSpan>,
    pub confidence: f32,
    pub method: SeparationMethod,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SeparationMethod {
    None,
    HeuristicMask,
    PlaybackCancellation,
    EmbeddingGuided,
    ExternalModel(String),
}

/// Separation/effect request emitted by policy evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SeparationRequest {
    pub source_id: SourceId,
    pub operation: SourceOperation,
    pub criterion: SourceCriterion,
    pub strength: f32,
}

/// Trivial separator implementation for tests and baseline wiring.
#[derive(Debug, Default)]
pub struct NoopSourceSeparator;

impl SourceSeparator for NoopSourceSeparator {
    fn separate(
        &mut self,
        input: &AudioSpan,
        _target: &SourceCriterion,
        _context: &SoundscapeContext,
    ) -> SeparationResult {
        SeparationResult {
            selected: None,
            residual: Some(input.clone()),
            confidence: 1.0,
            method: SeparationMethod::None,
        }
    }
}

/// Adapter that uses speaker-reference cancellation for playback extraction.
#[derive(Debug, Clone)]
pub struct PlaybackCancellationSeparator {
    mask: SpeakerReferenceMask,
}

impl PlaybackCancellationSeparator {
    pub fn new() -> Self {
        Self {
            mask: SpeakerReferenceMask::new(),
        }
    }

    pub fn mark_playback_reference(&mut self, frames: &[AudioFrame], started_at: ExactTimestamp) {
        self.mask.mark_output_started(frames, started_at);
    }
}

impl Default for PlaybackCancellationSeparator {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceSeparator for PlaybackCancellationSeparator {
    fn separate(
        &mut self,
        input: &AudioSpan,
        target: &SourceCriterion,
        context: &SoundscapeContext,
    ) -> SeparationResult {
        if !matches!(target, SourceCriterion::Playback)
            || context.expected_playback_source.is_none()
        {
            return NoopSourceSeparator.separate(input, target, context);
        }

        let mut selected_frames = Vec::with_capacity(input.frames.len());
        let mut residual_frames = Vec::with_capacity(input.frames.len());
        let mut confidence_sum = 0.0;

        for frame in &input.frames {
            let decision = self.mask.analyze_frame(frame);
            confidence_sum += decision.correlation.abs();
            selected_frames.push(decision.self_frame);
            residual_frames.push(decision.residual_frame);
        }

        let confidence = if input.frames.is_empty() {
            0.0
        } else {
            (confidence_sum / input.frames.len() as f32).clamp(0.0, 1.0)
        };

        SeparationResult {
            selected: Some(AudioSpan {
                frames: selected_frames,
            }),
            residual: Some(AudioSpan {
                frames: residual_frames,
            }),
            confidence,
            method: SeparationMethod::PlaybackCancellation,
        }
    }
}

/// Apply separation requests with any pluggable backend.
pub fn apply_separation_requests(
    separator: &mut dyn SourceSeparator,
    input: &AudioSpan,
    context: &SoundscapeContext,
    requests: &[SeparationRequest],
) -> Vec<SeparationResult> {
    requests
        .iter()
        .map(|request| separator.separate(input, &request.criterion, context))
        .collect()
}

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
    pub separation_requests: Vec<SeparationRequest>,
}

/// Transitional policy shim for self-hearing suppression.
pub fn self_hearing_suppression_policy(source_id: SourceId) -> IsolationPolicy {
    IsolationPolicy::suppress(SourceCriterion::KnownSource(source_id), 1.0)
}

/// Evaluate source-isolation policies against one soundscape frame.
pub fn evaluate_policies(
    frame: &SoundscapeFrame,
    policies: &[IsolationPolicy],
) -> IsolationEvaluation {
    let mut result = IsolationEvaluation::default();
    for policy in policies {
        let strength = policy.strength.clamp(0.0, 1.0);
        match policy.operation {
            SourceOperation::Suppress => {
                for source in &frame.sources {
                    if !matches_criterion(source, policy.criterion) {
                        continue;
                    }
                    result.suppressions.push(SuppressionTarget {
                        source_id: source.id,
                        strength,
                    });
                }
            }
            SourceOperation::Track => {
                for source in &frame.sources {
                    if !matches_criterion(source, policy.criterion) {
                        continue;
                    }
                    result.tracking.push(TrackingTarget {
                        source_id: source.id,
                        strength,
                    });
                }
            }
            SourceOperation::Enhance | SourceOperation::Extract => {
                for source in &frame.sources {
                    if !matches_criterion(source, policy.criterion) {
                        continue;
                    }
                    result.separation_requests.push(SeparationRequest {
                        source_id: source.id,
                        operation: policy.operation,
                        criterion: policy.criterion,
                        strength,
                    });
                }
            }
        }
    }
    result
}

fn matches_criterion(source: &SoundSource, criterion: SourceCriterion) -> bool {
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
    use crate::audio::frame::AudioFrame;
    use crate::soundscape::{
        AudioSpan, IsolationPolicy, MixtureId, NoopSourceSeparator, PlaybackCancellationSeparator,
        SeparationMethod, SeparationRequest, SoundEvent, SoundEventKind, SoundSource,
        SoundscapeContext, SoundscapeFrame, SourceCriterion, SourceId, SourceKind, SourceLabel,
        SourceOperation, SourceSeparator, TimePoint, TimeRange, apply_separation_requests,
        evaluate_policies, self_hearing_suppression_policy,
    };
    use crate::time::ExactTimestamp;

    #[test]
    fn suppresses_playback_and_tracks_unknown_voice() {
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
        assert!(evaluation.separation_requests.is_empty());
    }

    #[test]
    fn extraction_and_enhancement_policies_emit_separation_requests() {
        let range = TimeRange::new(TimePoint::from_millis(1_000), TimePoint::from_millis(1_100));
        let playback_source = SourceId::new();
        let unknown_voice_source_1 = SourceId::new();
        let unknown_voice_source_2 = SourceId::new();
        let frame = SoundscapeFrame {
            range,
            sources: vec![
                SoundSource {
                    id: playback_source,
                    kind: SourceKind::Playback,
                    label: SourceLabel::Playback("Pete".into()),
                    confidence: 0.95,
                },
                SoundSource {
                    id: unknown_voice_source_1,
                    kind: SourceKind::Voice,
                    label: SourceLabel::UnknownVoice { ordinal: 1 },
                    confidence: 0.8,
                },
                SoundSource {
                    id: unknown_voice_source_2,
                    kind: SourceKind::Voice,
                    label: SourceLabel::UnknownVoice { ordinal: 2 },
                    confidence: 0.78,
                },
            ],
            events: vec![],
            mixtures: vec![],
        };
        let policies = vec![
            IsolationPolicy {
                operation: SourceOperation::Extract,
                criterion: SourceCriterion::UnknownVoice,
                strength: 0.8,
            },
            IsolationPolicy {
                operation: SourceOperation::Enhance,
                criterion: SourceCriterion::Playback,
                strength: 0.6,
            },
        ];

        let evaluation = evaluate_policies(&frame, &policies);

        assert_eq!(
            evaluation.separation_requests,
            vec![
                SeparationRequest {
                    source_id: unknown_voice_source_1,
                    operation: SourceOperation::Extract,
                    criterion: SourceCriterion::UnknownVoice,
                    strength: 0.8,
                },
                SeparationRequest {
                    source_id: unknown_voice_source_2,
                    operation: SourceOperation::Extract,
                    criterion: SourceCriterion::UnknownVoice,
                    strength: 0.8,
                },
                SeparationRequest {
                    source_id: playback_source,
                    operation: SourceOperation::Enhance,
                    criterion: SourceCriterion::Playback,
                    strength: 0.6,
                },
            ]
        );
    }

    #[test]
    fn noop_separator_handles_policy_requests_without_backend_specifics() {
        let input = AudioSpan {
            frames: vec![AudioFrame {
                captured_at: ExactTimestamp::now(),
                sample_rate_hz: 16_000,
                channels: 1,
                samples: vec![0.1, -0.2, 0.3],
                voice_signatures: Vec::new(),
            }],
        };
        let requests = vec![SeparationRequest {
            source_id: SourceId::new(),
            operation: SourceOperation::Extract,
            criterion: SourceCriterion::UnknownVoice,
            strength: 0.9,
        }];
        let mut separator = NoopSourceSeparator;

        let results = apply_separation_requests(
            &mut separator,
            &input,
            &SoundscapeContext::default(),
            &requests,
        );

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].method, SeparationMethod::None);
        assert_eq!(results[0].selected, None);
        assert_eq!(results[0].residual, Some(input));
    }

    #[test]
    fn playback_cancellation_separator_extracts_playback_and_residual_tracks() {
        let playback_source = SourceId::new();
        let frame = AudioFrame {
            captured_at: ExactTimestamp {
                unix_nanos: 1_000_000_000,
            },
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![0.2; 160],
            voice_signatures: Vec::new(),
        };
        let input = AudioSpan {
            frames: vec![frame.clone()],
        };
        let mut separator = PlaybackCancellationSeparator::new();
        separator.mark_playback_reference(
            &[frame],
            ExactTimestamp {
                unix_nanos: 1_000_000_000,
            },
        );

        let result = separator.separate(
            &input,
            &SourceCriterion::Playback,
            &SoundscapeContext {
                expected_playback_source: Some(playback_source),
                ..SoundscapeContext::default()
            },
        );

        assert_eq!(result.method, SeparationMethod::PlaybackCancellation);
        assert!(result.selected.is_some());
        assert!(result.residual.is_some());
    }
}
