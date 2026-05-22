//! Developer-facing debug surface for soundscape inspection.
//!
//! This module provides [`SoundscapeDebugView`] — a JSON-serialisable snapshot
//! that assembles sources, attribution hypotheses, overlap mixtures,
//! transcript events, and voice-count estimates into a single structure that
//! is easy to print, diff, and reason about.
//!
//! # Example
//!
//! ```
//! use listenbury::soundscape::{
//!     SoundscapeDebugView, SoundscapeFrame, VoiceCount,
//!     SoundSource, SourceId, SourceKind, SourceLabel,
//!     TimePoint, TimeRange,
//! };
//!
//! let range = TimeRange::new(TimePoint::from_millis(12_000), TimePoint::from_millis(15_000));
//! let frame = SoundscapeFrame { range, sources: vec![], events: vec![], mixtures: vec![] };
//! let voice_count = VoiceCount {
//!     active_now: 0, recently_heard: 0, known: 0, unknown: 0, confidence: 0.0,
//! };
//! let view = SoundscapeDebugView::from_components(&frame, voice_count, &[], &[], &[]);
//! assert_eq!(view.range, "12.000..15.000");
//! ```

use serde::{Deserialize, Serialize};

use crate::soundscape::{
    AttributionEvidence, OverlapMixture, SoundscapeFrame, SourceAttributedTranscript,
    SourceHypothesis, SourceKind, TimeRange, VoiceCount,
};

/// Developer-facing debug snapshot of the soundscape state.
///
/// Serialises to a JSON object that is easy to read, diff, and include in
/// regression fixtures.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoundscapeDebugView {
    /// Time range this snapshot covers, as `"start_seconds..end_seconds"`.
    pub range: String,
    /// Voice-count estimates for the frame.
    pub voice_count: VoiceCount,
    /// Sources active in the frame.
    pub sources: Vec<DebugSource>,
    /// Overlap mixtures detected in the frame (empty when no overlap).
    pub overlaps: Vec<DebugOverlapMixture>,
    /// Raw attribution hypotheses for the frame.
    pub hypotheses: Vec<DebugHypothesis>,
    /// Source-attributed transcript events.
    pub events: Vec<DebugTranscriptEvent>,
}

/// Debug entry for one [`SoundSource`](crate::soundscape::SoundSource).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DebugSource {
    /// Script-friendly label, e.g. `"_PETE VOICE_"`.
    pub label: String,
    /// Source kind.
    pub kind: SourceKind,
    /// Attribution confidence in `[0.0, 1.0]`.
    pub confidence: f32,
}

/// Debug entry for one [`OverlapMixture`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DebugOverlapMixture {
    /// Time range of the mixture, as `"start_seconds..end_seconds"`.
    pub range: String,
    /// Number of concurrent voice-like sources.
    pub voice_count: usize,
    /// `true` when two or more sources overlap.
    pub is_overlapping: bool,
    /// Detection confidence.
    pub confidence: f32,
    /// Individual participant hypotheses.
    pub components: Vec<DebugHypothesis>,
}

/// Debug entry for one [`SourceHypothesis`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DebugHypothesis {
    /// Source kind.
    pub kind: SourceKind,
    /// Time range covered by the hypothesis.
    pub range: String,
    /// Attribution confidence in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Human-readable list of evidence labels, e.g. `"MatchesPlaybackBuffer(0.88)"`.
    pub evidence: Vec<String>,
}

/// Debug entry for one [`SourceAttributedTranscript`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DebugTranscriptEvent {
    /// Script-friendly source label.
    pub label: String,
    /// Recognised text.
    pub text: String,
    /// ASR confidence in the words.
    pub transcript_confidence: f32,
    /// Source-attribution confidence.
    pub attribution_confidence: f32,
    /// `true` when this event is part of an overlapping acoustic mixture.
    pub overlapped: bool,
}

impl SoundscapeDebugView {
    /// Assemble a debug snapshot from soundscape components.
    ///
    /// # Parameters
    ///
    /// - `frame` — the current soundscape frame (sources, events, mixtures).
    /// - `voice_count` — voice-count estimates for the frame window.
    /// - `hypotheses` — raw attribution hypotheses for the frame.
    /// - `overlaps` — overlap mixtures detected for this frame.
    /// - `transcripts` — source-attributed transcript events.
    pub fn from_components(
        frame: &SoundscapeFrame,
        voice_count: VoiceCount,
        hypotheses: &[SourceHypothesis],
        overlaps: &[OverlapMixture],
        transcripts: &[SourceAttributedTranscript],
    ) -> Self {
        Self {
            range: format_range(frame.range),
            voice_count,
            sources: frame
                .sources
                .iter()
                .map(|s| DebugSource {
                    label: s.label.display_label(),
                    kind: s.kind,
                    confidence: s.confidence,
                })
                .collect(),
            overlaps: overlaps
                .iter()
                .map(|m| DebugOverlapMixture {
                    range: format_range(m.range),
                    voice_count: m.voice_count(),
                    is_overlapping: m.is_overlapping(),
                    confidence: m.confidence,
                    components: m
                        .components
                        .iter()
                        .map(|c| hypothesis_to_debug(&c.source_hypothesis))
                        .collect(),
                })
                .collect(),
            hypotheses: hypotheses.iter().map(hypothesis_to_debug).collect(),
            events: transcripts
                .iter()
                .map(|t| DebugTranscriptEvent {
                    label: t.display_label(),
                    text: t.text.clone(),
                    transcript_confidence: t.transcript_confidence,
                    attribution_confidence: t.attribution_confidence,
                    overlapped: t.is_overlapped(),
                })
                .collect(),
        }
    }
}

fn format_range(range: TimeRange) -> String {
    let start = range.start.millis as f64 / 1_000.0;
    let end = range.end.millis as f64 / 1_000.0;
    format!("{start:.3}..{end:.3}")
}

fn hypothesis_to_debug(h: &SourceHypothesis) -> DebugHypothesis {
    DebugHypothesis {
        kind: h.kind,
        range: format_range(h.range),
        confidence: h.confidence,
        evidence: h.evidence.iter().map(evidence_label).collect(),
    }
}

fn evidence_label(ev: &AttributionEvidence) -> String {
    match ev {
        AttributionEvidence::MatchesExpectedPlayback { confidence, .. } => {
            format!("MatchesExpectedPlayback({confidence:.2})")
        }
        AttributionEvidence::MatchesExpectedSelfOutput { confidence, .. } => {
            format!("MatchesExpectedSelfOutput({confidence:.2})")
        }
        AttributionEvidence::MatchesPlaybackBuffer { confidence } => {
            format!("MatchesPlaybackBuffer({confidence:.2})")
        }
        AttributionEvidence::MatchesVoiceSignature { confidence, .. } => {
            format!("MatchesVoiceSignature({confidence:.2})")
        }
        AttributionEvidence::SpeakerEmbeddingCluster { confidence, .. } => {
            format!("SpeakerEmbeddingCluster({confidence:.2})")
        }
        AttributionEvidence::PitchContinuity { confidence } => {
            format!("PitchContinuity({confidence:.2})")
        }
        AttributionEvidence::SpectralContinuity { confidence } => {
            format!("SpectralContinuity({confidence:.2})")
        }
        AttributionEvidence::LexicalContinuity { confidence } => {
            format!("LexicalContinuity({confidence:.2})")
        }
        AttributionEvidence::EnergyChange => "EnergyChange".to_string(),
        AttributionEvidence::OverlapDetected => "OverlapDetected".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soundscape::{
        MixtureComponent, SoundSource, SoundscapeFrame, SourceAttributedTranscript,
        SourceId, SourceLabel, TimePoint, TimeRange,
    };

    fn range(start_ms: u64, end_ms: u64) -> TimeRange {
        TimeRange::new(
            TimePoint::from_millis(start_ms),
            TimePoint::from_millis(end_ms),
        )
    }

    fn empty_frame(r: TimeRange) -> SoundscapeFrame {
        SoundscapeFrame {
            range: r,
            sources: vec![],
            events: vec![],
            mixtures: vec![],
        }
    }

    fn zero_voice_count() -> VoiceCount {
        VoiceCount {
            active_now: 0,
            recently_heard: 0,
            known: 0,
            unknown: 0,
            confidence: 0.0,
        }
    }

    #[test]
    fn range_formatted_as_seconds_with_three_decimal_places() {
        let view = SoundscapeDebugView::from_components(
            &empty_frame(range(12_000, 15_000)),
            zero_voice_count(),
            &[],
            &[],
            &[],
        );
        assert_eq!(view.range, "12.000..15.000");
    }

    #[test]
    fn sources_include_label_kind_and_confidence() {
        let pete_id = SourceId::new();
        let unknown_id = SourceId::new();
        let frame = SoundscapeFrame {
            range: range(0, 3_000),
            sources: vec![
                SoundSource {
                    id: pete_id,
                    kind: SourceKind::KnownSelfVoice,
                    label: SourceLabel::NamedVoice("Pete".into()),
                    confidence: 0.96,
                },
                SoundSource {
                    id: unknown_id,
                    kind: SourceKind::Voice,
                    label: SourceLabel::UnknownVoice { ordinal: 1 },
                    confidence: 0.68,
                },
            ],
            events: vec![],
            mixtures: vec![],
        };
        let voice_count = VoiceCount {
            active_now: 2,
            recently_heard: 2,
            known: 1,
            unknown: 1,
            confidence: 0.74,
        };
        let view =
            SoundscapeDebugView::from_components(&frame, voice_count, &[], &[], &[]);
        assert_eq!(view.sources.len(), 2);
        assert_eq!(view.sources[0].label, "_PETE VOICE_");
        assert_eq!(view.sources[0].kind, SourceKind::KnownSelfVoice);
        assert!((view.sources[0].confidence - 0.96).abs() < f32::EPSILON);
        assert_eq!(view.sources[1].label, "_UNKNOWN VOICE #1_");
        assert_eq!(view.sources[1].kind, SourceKind::Voice);
    }

    #[test]
    fn hypotheses_include_evidence_labels() {
        let hypotheses = vec![SourceHypothesis {
            source_id: None,
            kind: SourceKind::Playback,
            range: range(0, 500),
            confidence: 0.85,
            evidence: vec![
                AttributionEvidence::MatchesPlaybackBuffer { confidence: 0.88 },
                AttributionEvidence::EnergyChange,
            ],
        }];
        let view = SoundscapeDebugView::from_components(
            &empty_frame(range(0, 1_000)),
            zero_voice_count(),
            &hypotheses,
            &[],
            &[],
        );
        assert_eq!(view.hypotheses.len(), 1);
        assert_eq!(
            view.hypotheses[0].evidence,
            vec!["MatchesPlaybackBuffer(0.88)", "EnergyChange"]
        );
    }

    #[test]
    fn transcript_events_preserve_confidence_and_overlap_flag() {
        let r = range(0, 500);
        let transcripts = vec![SourceAttributedTranscript {
            range: r,
            source_hypothesis: SourceHypothesis {
                source_id: None,
                kind: SourceKind::Voice,
                range: r,
                confidence: 0.62,
                evidence: vec![],
            },
            source_label: SourceLabel::UnknownVoice { ordinal: 1 },
            text: "wait, what?".to_string(),
            transcript_confidence: 0.71,
            attribution_confidence: 0.62,
            overlap: None,
        }];
        let view = SoundscapeDebugView::from_components(
            &empty_frame(r),
            VoiceCount {
                active_now: 1,
                recently_heard: 1,
                known: 0,
                unknown: 1,
                confidence: 0.62,
            },
            &[],
            &[],
            &transcripts,
        );
        assert_eq!(view.events.len(), 1);
        assert_eq!(view.events[0].label, "_UNKNOWN VOICE #1_");
        assert_eq!(view.events[0].text, "wait, what?");
        assert!((view.events[0].transcript_confidence - 0.71).abs() < f32::EPSILON);
        assert!((view.events[0].attribution_confidence - 0.62).abs() < f32::EPSILON);
        assert!(!view.events[0].overlapped);
    }

    #[test]
    fn overlap_mixtures_include_component_hypotheses() {
        let h1 = SourceHypothesis {
            source_id: None,
            kind: SourceKind::Voice,
            range: range(0, 600),
            confidence: 0.8,
            evidence: vec![AttributionEvidence::OverlapDetected],
        };
        let h2 = SourceHypothesis {
            source_id: None,
            kind: SourceKind::Voice,
            range: range(200, 800),
            confidence: 0.7,
            evidence: vec![AttributionEvidence::OverlapDetected],
        };
        let overlaps = vec![OverlapMixture {
            range: range(0, 800),
            confidence: 0.85,
            components: vec![
                MixtureComponent {
                    source_hypothesis: h1.clone(),
                    relative_energy: Some(0.6),
                },
                MixtureComponent {
                    source_hypothesis: h2.clone(),
                    relative_energy: Some(0.4),
                },
            ],
        }];
        let view = SoundscapeDebugView::from_components(
            &empty_frame(range(0, 1_000)),
            VoiceCount {
                active_now: 2,
                recently_heard: 2,
                known: 0,
                unknown: 2,
                confidence: 0.75,
            },
            &[],
            &overlaps,
            &[],
        );
        assert_eq!(view.overlaps.len(), 1);
        assert!(view.overlaps[0].is_overlapping);
        assert_eq!(view.overlaps[0].voice_count, 2);
        assert_eq!(view.overlaps[0].components.len(), 2);
        assert!(view.overlaps[0].components[0]
            .evidence
            .contains(&"OverlapDetected".to_string()));
    }

    #[test]
    fn serialises_to_json_without_error() {
        let frame = SoundscapeFrame {
            range: range(12_000, 15_000),
            sources: vec![SoundSource {
                id: SourceId::new(),
                kind: SourceKind::KnownSelfVoice,
                label: SourceLabel::NamedVoice("Pete".into()),
                confidence: 0.96,
            }],
            events: vec![],
            mixtures: vec![],
        };
        let view = SoundscapeDebugView::from_components(
            &frame,
            VoiceCount {
                active_now: 1,
                recently_heard: 1,
                known: 1,
                unknown: 0,
                confidence: 0.96,
            },
            &[],
            &[],
            &[],
        );
        let json = serde_json::to_string_pretty(&view).expect("serialise");
        assert!(json.contains("12.000..15.000"));
        assert!(json.contains("_PETE VOICE_"));
    }

    /// Validates that a hand-crafted fixture round-trips through JSON without
    /// loss.  This serves as the golden/snapshot fixture required by the issue.
    #[test]
    fn golden_fixture_round_trips() {
        let fixture = include_str!("../../fixtures/soundscape/sample_debug_view.json");
        let view: SoundscapeDebugView =
            serde_json::from_str(fixture).expect("fixture should deserialise");

        assert_eq!(view.range, "12.000..15.000");
        assert_eq!(view.voice_count.active_now, 2);
        assert_eq!(view.voice_count.known, 1);
        assert_eq!(view.voice_count.unknown, 2);
        assert_eq!(view.sources.len(), 2);
        assert_eq!(view.sources[0].label, "_PETE VOICE_");
        assert_eq!(view.sources[1].label, "_UNKNOWN VOICE #1_");
        assert_eq!(view.hypotheses.len(), 2);
        assert_eq!(view.events.len(), 1);
        assert_eq!(view.events[0].text, "wait, what?");
    }
}
