//! Source-attributed transcript events for timeline and screenplay views.
//!
//! Transcript text is coupled with a [`SourceHypothesis`] so the timeline can
//! show what was actually heard and who (or what) was speaking.  Attribution
//! uncertainty is preserved rather than flattened away, and transcript
//! confidence is kept separate from attribution confidence.

use serde::{Deserialize, Serialize};

use crate::soundscape::{MixtureId, SourceHypothesis, SourceLabel, TimeRange};

/// A stable identifier for an acoustic mixture of overlapping sources.
///
/// When two or more voice-like sources overlap in time the [`OverlapMixture`]
/// is assigned a [`AcousticMixtureId`] so that individual
/// [`SourceAttributedTranscript`] entries can reference their parent mixture.
///
/// [`OverlapMixture`]: crate::soundscape::OverlapMixture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AcousticMixtureId(pub MixtureId);

impl AcousticMixtureId {
    pub fn new() -> Self {
        Self(MixtureId::new())
    }
}

impl Default for AcousticMixtureId {
    fn default() -> Self {
        Self::new()
    }
}

/// A transcript hypothesis tied to a single soundscape source.
///
/// This type carries *both* the transcript text confidence (how sure the ASR
/// engine is about the words) and the attribution confidence (how sure the
/// source-attribution layer is about who spoke them).  Both dimensions of
/// uncertainty are preserved so the timeline and screenplay layers can render
/// them with appropriate visual weight.
///
/// # Display labels
///
/// Use [`source_label`](SourceAttributedTranscript::source_label) to obtain a
/// script-friendly cue such as `_PETE VOICE_` or `_UNKNOWN VOICE #1_`.
///
/// # Example
///
/// ```
/// use listenbury::soundscape::{
///     AcousticMixtureId, SourceAttributedTranscript, SourceHypothesis,
///     SourceId, SourceKind, SourceLabel, TimePoint, TimeRange,
/// };
///
/// let hypothesis = SourceHypothesis {
///     source_id: None,
///     kind: SourceKind::Voice,
///     range: TimeRange::new(TimePoint::from_millis(0), TimePoint::from_millis(500)),
///     confidence: 0.72,
///     evidence: vec![],
/// };
///
/// let event = SourceAttributedTranscript {
///     range: TimeRange::new(TimePoint::from_millis(0), TimePoint::from_millis(500)),
///     source_hypothesis: hypothesis,
///     source_label: SourceLabel::UnknownVoice { ordinal: 1 },
///     text: "wait, what?".to_string(),
///     transcript_confidence: 0.85,
///     attribution_confidence: 0.72,
///     overlap: None,
/// };
///
/// assert_eq!(event.display_label(), "_UNKNOWN VOICE #1_");
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceAttributedTranscript {
    /// The acoustic time window covered by this transcript segment.
    pub range: TimeRange,
    /// Attribution hypothesis identifying the source that produced the audio.
    pub source_hypothesis: SourceHypothesis,
    /// Screenplay-friendly source label for rendering.
    pub source_label: SourceLabel,
    /// The recognised transcript text.
    pub text: String,
    /// ASR engine confidence in the recognised words (0.0–1.0).
    pub transcript_confidence: f32,
    /// Source-attribution confidence — how certain we are about the speaker
    /// identity (0.0–1.0).  Distinct from [`transcript_confidence`](Self::transcript_confidence).
    pub attribution_confidence: f32,
    /// When this segment is part of an overlapping acoustic mixture the
    /// mixture is identified here.  `None` when no overlap was detected.
    pub overlap: Option<AcousticMixtureId>,
}

impl SourceAttributedTranscript {
    /// Returns the script-friendly display label for this transcript segment,
    /// e.g. `_PETE VOICE_`, `_UNKNOWN VOICE #1_`, `_BACKGROUND VOICE #2_`.
    pub fn display_label(&self) -> String {
        self.source_label.display_label()
    }

    /// Returns `true` when this segment was produced during an acoustic
    /// mixture (overlapping speakers or indistinct sources).
    pub fn is_overlapped(&self) -> bool {
        self.overlap.is_some()
    }

    /// Returns `true` when the transcript text is likely indistinct.
    ///
    /// Indistinct segments have low transcript confidence (< 0.4) or are part
    /// of an overlapping mixture where multiple sources contribute energy.
    pub fn is_indistinct(&self) -> bool {
        self.transcript_confidence < 0.4 || self.is_overlapped()
    }

    /// Format the segment as a screenplay line: `LABEL: text`.
    ///
    /// When the segment is indistinct the text is rendered as `[indistinct]`.
    pub fn screenplay_line(&self) -> String {
        let label = self.display_label();
        let body = if self.is_indistinct() && self.text.is_empty() {
            "[indistinct]".to_string()
        } else {
            self.text.clone()
        };
        format!("{}: {}", label, body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soundscape::{SourceId, SourceKind, TimePoint};

    fn range(start_ms: u64, end_ms: u64) -> TimeRange {
        TimeRange::new(
            TimePoint::from_millis(start_ms),
            TimePoint::from_millis(end_ms),
        )
    }

    fn make_transcript(
        label: SourceLabel,
        text: &str,
        transcript_confidence: f32,
        attribution_confidence: f32,
        overlap: Option<AcousticMixtureId>,
    ) -> SourceAttributedTranscript {
        let r = range(0, 500);
        SourceAttributedTranscript {
            range: r,
            source_hypothesis: SourceHypothesis {
                source_id: None,
                kind: SourceKind::Voice,
                range: r,
                confidence: attribution_confidence,
                evidence: vec![],
            },
            source_label: label,
            text: text.to_string(),
            transcript_confidence,
            attribution_confidence,
            overlap,
        }
    }

    #[test]
    fn known_pete_voice_display_label() {
        let event = make_transcript(
            SourceLabel::NamedVoice("Pete".to_string()),
            "I'm going to make the timing model...",
            0.92,
            0.95,
            None,
        );
        assert_eq!(event.display_label(), "_PETE VOICE_");
        assert_eq!(
            event.screenplay_line(),
            "_PETE VOICE_: I'm going to make the timing model..."
        );
        assert!(!event.is_overlapped());
        assert!(!event.is_indistinct());
    }

    #[test]
    fn unknown_voice_display_label() {
        let event = make_transcript(
            SourceLabel::UnknownVoice { ordinal: 1 },
            "wait, what?",
            0.85,
            0.72,
            None,
        );
        assert_eq!(event.display_label(), "_UNKNOWN VOICE #1_");
        assert_eq!(event.screenplay_line(), "_UNKNOWN VOICE #1_: wait, what?");
        assert!(!event.is_overlapped());
    }

    #[test]
    fn background_voice_display_label() {
        let event = make_transcript(
            SourceLabel::BackgroundVoice { ordinal: 2 },
            "[indistinct]",
            0.35,
            0.60,
            None,
        );
        assert_eq!(event.display_label(), "_BACKGROUND VOICE #2_");
        assert!(event.is_indistinct(), "low transcript confidence should mark as indistinct");
    }

    #[test]
    fn overlapped_segment_is_indistinct() {
        let mixture_id = AcousticMixtureId::new();
        let event = make_transcript(
            SourceLabel::UnknownVoice { ordinal: 1 },
            "something",
            0.75,
            0.55,
            Some(mixture_id),
        );
        assert!(event.is_overlapped());
        assert!(event.is_indistinct(), "overlapped segments are always treated as indistinct");
    }

    #[test]
    fn indistinct_empty_text_renders_placeholder() {
        let mixture_id = AcousticMixtureId::new();
        let event = make_transcript(
            SourceLabel::BackgroundVoice { ordinal: 1 },
            "",
            0.20,
            0.50,
            Some(mixture_id),
        );
        assert_eq!(event.screenplay_line(), "_BACKGROUND VOICE #1_: [indistinct]");
    }

    #[test]
    fn transcript_and_attribution_confidence_are_independent() {
        let event = make_transcript(
            SourceLabel::UnknownVoice { ordinal: 3 },
            "hello there",
            0.90,
            0.30,
            None,
        );
        // High transcript confidence but low attribution confidence.
        assert!(!event.is_indistinct(), "high transcript confidence should not be indistinct even with low attribution confidence");
        assert_eq!(event.transcript_confidence, 0.90);
        assert_eq!(event.attribution_confidence, 0.30);
    }

    #[test]
    fn known_source_id_is_preserved() {
        let source_id = SourceId::new();
        let r = range(1_000, 1_500);
        let event = SourceAttributedTranscript {
            range: r,
            source_hypothesis: SourceHypothesis {
                source_id: Some(source_id),
                kind: SourceKind::KnownSelfVoice,
                range: r,
                confidence: 0.98,
                evidence: vec![],
            },
            source_label: SourceLabel::NamedVoice("Pete".to_string()),
            text: "Testing, testing.".to_string(),
            transcript_confidence: 0.97,
            attribution_confidence: 0.98,
            overlap: None,
        };
        assert_eq!(event.source_hypothesis.source_id, Some(source_id));
        assert_eq!(event.display_label(), "_PETE VOICE_");
    }

    #[test]
    fn serialization_round_trip() {
        let event = make_transcript(
            SourceLabel::UnknownVoice { ordinal: 2 },
            "Some text here.",
            0.88,
            0.76,
            None,
        );
        let json = serde_json::to_string(&event).expect("serialize");
        let restored: SourceAttributedTranscript =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(event, restored);
    }
}
