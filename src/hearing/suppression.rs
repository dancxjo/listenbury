use std::time::Duration;

use crate::time::ExactTimestamp;

/// Additional silence to suppress after Pete's TTS output ends, to absorb room echo.
pub const SUPPRESSION_TAIL_MS: u64 = 300;

/// Tracks when Pete (the TTS assistant) is emitting audio output, and provides
/// a policy for suppressing incoming microphone frames during that window.
///
/// The suppression window covers the period from when Pete starts speaking
/// through his full audio duration plus a configurable tail buffer
/// ([`SUPPRESSION_TAIL_MS`]).  This prevents Whisper / VAD from treating
/// Pete's own voice as user input.
#[derive(Debug, Clone)]
pub struct SelfHearingState {
    /// Whether Pete is actively emitting TTS audio right now.
    pub pete_speaking: bool,
    /// When the current (or most recent) TTS output began.
    pub output_started_at: Option<ExactTimestamp>,
    /// The end of the suppression window: output duration + tail buffer.
    pub output_expected_until: Option<ExactTimestamp>,
    /// The text of the utterance Pete is (or was last) speaking.
    pub current_utterance_text: Option<String>,
}

/// Decision returned by the suppression policy for each incoming mic frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuppressionDecision {
    /// Allow the frame to be processed normally.
    Allow,
    /// Drop the frame; Pete is speaking or the echo tail is still active.
    Suppress,
    /// Process the frame with reduced weight (hook for future full-duplex AEC).
    Attenuate,
}

impl SelfHearingState {
    /// Create a new [`SelfHearingState`] with no active suppression.
    pub fn new() -> Self {
        Self {
            pete_speaking: false,
            output_started_at: None,
            output_expected_until: None,
            current_utterance_text: None,
        }
    }

    /// Record the start of Pete's TTS output.
    ///
    /// `expected_duration` should be the estimated play-time of the audio. A
    /// fixed tail buffer ([`SUPPRESSION_TAIL_MS`]) is added so that residual
    /// room echo is also suppressed after playback ends.
    pub fn mark_output_started(
        &mut self,
        utterance_text: impl Into<String>,
        expected_duration: Duration,
    ) {
        let now = ExactTimestamp::now();
        let tail_nanos = u128::from(SUPPRESSION_TAIL_MS) * 1_000_000;
        let window_nanos = expected_duration.as_nanos().saturating_add(tail_nanos);
        self.pete_speaking = true;
        self.output_started_at = Some(now);
        self.output_expected_until = Some(ExactTimestamp {
            unix_nanos: now.unix_nanos.saturating_add(window_nanos),
        });
        self.current_utterance_text = Some(utterance_text.into());
    }

    /// Record the end of Pete's TTS playback.
    ///
    /// Clears [`pete_speaking`](SelfHearingState::pete_speaking), but the tail
    /// window ([`output_expected_until`](SelfHearingState::output_expected_until))
    /// remains active so that post-output room echo is still suppressed.
    pub fn mark_output_finished(&mut self) {
        self.pete_speaking = false;
    }

    /// Decide how a microphone frame arriving *now* should be treated.
    ///
    /// Returns [`SuppressionDecision::Suppress`] while Pete is speaking or
    /// within the tail window after he stops.  Returns
    /// [`SuppressionDecision::Allow`] once the window has fully elapsed and no
    /// output is active.
    pub fn suppression_decision(&self) -> SuppressionDecision {
        self.suppression_decision_at(ExactTimestamp::now())
    }

    /// Decide how a microphone frame captured at `timestamp` should be treated.
    ///
    /// This is preferred for queued audio frames: a frame captured during Pete's
    /// suppression window should still be dropped even if it is processed after
    /// the wall-clock window has elapsed.
    pub fn suppression_decision_at(&self, timestamp: ExactTimestamp) -> SuppressionDecision {
        if self.pete_speaking {
            return SuppressionDecision::Suppress;
        }
        if let Some(until) = self.output_expected_until {
            if timestamp.unix_nanos <= until.unix_nanos {
                return SuppressionDecision::Suppress;
            }
        }
        SuppressionDecision::Allow
    }
}

impl Default for SelfHearingState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::time::ExactTimestamp;

    use super::{SUPPRESSION_TAIL_MS, SelfHearingState, SuppressionDecision};

    #[test]
    fn initial_state_allows_all_frames() {
        let state = SelfHearingState::new();
        assert_eq!(state.suppression_decision(), SuppressionDecision::Allow);
    }

    #[test]
    fn suppresses_frames_while_pete_is_speaking() {
        let mut state = SelfHearingState::new();
        state.mark_output_started("Hello there.", Duration::from_secs(2));
        assert_eq!(state.suppression_decision(), SuppressionDecision::Suppress);
    }

    #[test]
    fn still_suppresses_during_tail_window_after_output_finished() {
        let mut state = SelfHearingState::new();
        // Zero-length audio: the suppression window is the tail only, which
        // extends SUPPRESSION_TAIL_MS into the future, so we are still inside it.
        state.mark_output_started("Hi.", Duration::ZERO);
        state.mark_output_finished();
        assert!(!state.pete_speaking);
        assert_eq!(state.suppression_decision(), SuppressionDecision::Suppress);
    }

    #[test]
    fn allows_frames_after_window_expires() {
        let mut state = SelfHearingState::new();
        // Manually set an already-expired window (unix_nanos=2 is far in the past).
        state.pete_speaking = false;
        state.output_started_at = Some(ExactTimestamp { unix_nanos: 1 });
        state.output_expected_until = Some(ExactTimestamp { unix_nanos: 2 });
        assert_eq!(state.suppression_decision(), SuppressionDecision::Allow);
    }

    #[test]
    fn suppresses_queued_frames_captured_during_tail_window() {
        let mut state = SelfHearingState::new();
        state.pete_speaking = false;
        state.output_started_at = Some(ExactTimestamp { unix_nanos: 1_000 });
        state.output_expected_until = Some(ExactTimestamp { unix_nanos: 2_000 });

        assert_eq!(
            state.suppression_decision_at(ExactTimestamp { unix_nanos: 1_500 }),
            SuppressionDecision::Suppress
        );
        assert_eq!(
            state.suppression_decision_at(ExactTimestamp { unix_nanos: 2_001 }),
            SuppressionDecision::Allow
        );
    }

    #[test]
    fn records_utterance_text_on_output_started() {
        let mut state = SelfHearingState::new();
        state.mark_output_started("Test sentence.", Duration::from_millis(500));
        assert_eq!(
            state.current_utterance_text.as_deref(),
            Some("Test sentence.")
        );
    }

    #[test]
    fn output_expected_until_includes_tail_buffer() {
        let before = ExactTimestamp::now();
        let mut state = SelfHearingState::new();
        let audio_duration = Duration::from_millis(500);
        state.mark_output_started("text", audio_duration);
        let after = ExactTimestamp::now();

        let until = state.output_expected_until.unwrap();
        let tail_nanos = u128::from(SUPPRESSION_TAIL_MS) * 1_000_000;
        let min_expected = before
            .unix_nanos
            .saturating_add(audio_duration.as_nanos())
            .saturating_add(tail_nanos);
        let max_expected = after
            .unix_nanos
            .saturating_add(audio_duration.as_nanos())
            .saturating_add(tail_nanos);

        assert!(
            until.unix_nanos >= min_expected && until.unix_nanos <= max_expected,
            "until={} not in [{min_expected}, {max_expected}]",
            until.unix_nanos,
        );
    }

    #[test]
    fn mark_output_finished_clears_pete_speaking_flag() {
        let mut state = SelfHearingState::new();
        state.mark_output_started("Something.", Duration::from_secs(1));
        assert!(state.pete_speaking);
        state.mark_output_finished();
        assert!(!state.pete_speaking);
    }
}
