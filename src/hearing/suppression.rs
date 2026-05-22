use std::time::Duration;

use crate::audio::frame::AudioFrame;
use crate::soundscape::{IsolationPolicy, SourceId, self_hearing_suppression_policy};
use crate::time::ExactTimestamp;

/// Additional silence to suppress after Pete's TTS output ends, to absorb room echo.
pub const SUPPRESSION_TAIL_MS: u64 = 300;
/// Suppress a small window before speaker intent/playback, covering callback
/// and clock jitter between queued mic frames and output scheduling.
pub const SUPPRESSION_PRE_ROLL_MS: u64 = 100;
/// Additional acoustic reference lifetime after playback ends.
pub const SPEAKER_REFERENCE_TAIL_MS: u64 = 500;
const SPEAKER_REFERENCE_MAX_DELAY_MS: i64 = 450;
const SPEAKER_REFERENCE_DELAY_STEP_MS: i64 = 15;
const SPEAKER_REFERENCE_MIN_CORRELATION: f32 = 0.72;
const SPEAKER_REFERENCE_MAX_RESIDUAL_RATIO: f32 = 0.35;

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
        self.mark_output_started_at(utterance_text, expected_duration, now);
    }

    /// Record that Pete intends to speak, before audio is necessarily ready.
    ///
    /// In half-duplex mode the microphone gate should close before output can
    /// become audible.  The final window end is filled in once the audio
    /// duration is known via [`mark_output_started`](Self::mark_output_started).
    pub fn mark_output_intent(&mut self, utterance_text: impl Into<String>) {
        self.mark_output_intent_at(utterance_text, ExactTimestamp::now());
    }

    /// Timestamp-injectable variant of
    /// [`mark_output_intent`](Self::mark_output_intent), primarily for tests
    /// and callers that already captured the scheduling time.
    pub fn mark_output_intent_at(
        &mut self,
        utterance_text: impl Into<String>,
        intended_at: ExactTimestamp,
    ) {
        self.pete_speaking = true;
        let intended_start = suppression_start_with_pre_roll(intended_at);
        self.output_started_at = Some(
            self.output_started_at
                .map_or(intended_start, |current| current.min(intended_start)),
        );
        self.current_utterance_text = Some(utterance_text.into());
    }

    /// Timestamp-injectable variant of
    /// [`mark_output_started`](Self::mark_output_started).
    pub fn mark_output_started_at(
        &mut self,
        utterance_text: impl Into<String>,
        expected_duration: Duration,
        started_at: ExactTimestamp,
    ) {
        let tail_nanos = u128::from(SUPPRESSION_TAIL_MS) * 1_000_000;
        let window_nanos = expected_duration.as_nanos().saturating_add(tail_nanos);
        self.pete_speaking = true;
        self.output_started_at = Some(
            self.output_started_at
                .unwrap_or_else(|| suppression_start_with_pre_roll(started_at)),
        );
        self.output_expected_until = Some(ExactTimestamp {
            unix_nanos: started_at.unix_nanos.saturating_add(window_nanos),
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

    /// Transitional adapter exposing self-hearing suppression as an isolation
    /// policy keyed by Pete's source id.
    pub fn isolation_policy_for_source(pete_voice_source_id: SourceId) -> IsolationPolicy {
        self_hearing_suppression_policy(pete_voice_source_id)
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
        if self.pete_speaking
            && self
                .output_started_at
                .is_none_or(|started_at| timestamp.unix_nanos >= started_at.unix_nanos)
        {
            return SuppressionDecision::Suppress;
        }
        if let Some(until) = self.output_expected_until
            && timestamp.unix_nanos <= until.unix_nanos
        {
            return SuppressionDecision::Suppress;
        }
        SuppressionDecision::Allow
    }
}

fn suppression_start_with_pre_roll(at: ExactTimestamp) -> ExactTimestamp {
    ExactTimestamp {
        unix_nanos: at
            .unix_nanos
            .saturating_sub(u128::from(SUPPRESSION_PRE_ROLL_MS) * 1_000_000),
    }
}

impl Default for SelfHearingState {
    fn default() -> Self {
        Self::new()
    }
}

/// Keeps a short reference copy of audio being sent to the speaker and decides
/// whether incoming microphone frames are just delayed speaker bleed.
#[derive(Debug, Clone)]
pub struct SpeakerReferenceMask {
    reference: Option<SpeakerReference>,
    tail: Duration,
}

#[derive(Debug, Clone)]
struct SpeakerReference {
    started_at: ExactTimestamp,
    sample_rate_hz: u32,
    samples: Vec<f32>,
    expected_until: ExactTimestamp,
}

/// Diagnostic result from comparing one mic frame with the speaker reference.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerReferenceDecision {
    pub decision: SuppressionDecision,
    pub correlation: f32,
    pub residual_ratio: f32,
    pub delay_ms: i64,
    pub gain: f32,
    pub residual_frame: AudioFrame,
    pub self_frame: AudioFrame,
}

impl SpeakerReferenceMask {
    pub fn new() -> Self {
        Self {
            reference: None,
            tail: Duration::from_millis(SPEAKER_REFERENCE_TAIL_MS),
        }
    }

    /// Record the audio that is about to be played through Pete's speaker.
    pub fn mark_output_started(&mut self, frames: &[AudioFrame], started_at: ExactTimestamp) {
        let Some(first_frame) = frames.first() else {
            self.reference = None;
            return;
        };
        if first_frame.sample_rate_hz == 0 || first_frame.channels == 0 {
            self.reference = None;
            return;
        }

        let sample_rate_hz = first_frame.sample_rate_hz;
        let channels = first_frame.channels;
        let mut samples = Vec::new();
        for frame in frames {
            if frame.sample_rate_hz != sample_rate_hz || frame.channels != channels {
                self.reference = None;
                return;
            }
            samples.extend(downmix_mono(&frame.samples, channels));
        }
        if samples.is_empty() {
            self.reference = None;
            return;
        }

        let duration_nanos =
            (samples.len() as u128).saturating_mul(1_000_000_000) / u128::from(sample_rate_hz);
        let expected_until = ExactTimestamp {
            unix_nanos: started_at
                .unix_nanos
                .saturating_add(duration_nanos)
                .saturating_add(self.tail.as_nanos()),
        };
        self.reference = Some(SpeakerReference {
            started_at,
            sample_rate_hz,
            samples,
            expected_until,
        });
    }

    pub fn mark_output_finished(&mut self) {
        if let Some(reference) = &mut self.reference {
            reference.expected_until = ExactTimestamp {
                unix_nanos: ExactTimestamp::now()
                    .unix_nanos
                    .saturating_add(self.tail.as_nanos()),
            };
        }
    }

    pub fn suppression_decision_for_frame(&mut self, frame: &AudioFrame) -> SuppressionDecision {
        self.analyze_frame(frame).decision
    }

    pub fn analyze_frame(&mut self, frame: &AudioFrame) -> SpeakerReferenceDecision {
        self.expire(frame.captured_at);
        let Some(reference) = &self.reference else {
            return empty_speaker_reference_decision(frame, SuppressionDecision::Allow);
        };
        if frame.sample_rate_hz == 0 || frame.channels == 0 || frame.samples.is_empty() {
            return empty_speaker_reference_decision(frame, SuppressionDecision::Allow);
        }

        let mic = downmix_mono(&frame.samples, frame.channels);
        if mic.is_empty() || rms_energy(&mic) <= f32::EPSILON {
            return empty_speaker_reference_decision(frame, SuppressionDecision::Allow);
        }

        let mut best = empty_speaker_reference_decision(frame, SuppressionDecision::Allow);
        for delay_ms in
            (0..=SPEAKER_REFERENCE_MAX_DELAY_MS).step_by(SPEAKER_REFERENCE_DELAY_STEP_MS as usize)
        {
            if let Some(candidate) = compare_with_reference(reference, frame, &mic, delay_ms)
                && candidate.correlation > best.correlation
            {
                best = candidate;
            }
        }

        if best.correlation >= SPEAKER_REFERENCE_MIN_CORRELATION
            && best.residual_ratio <= SPEAKER_REFERENCE_MAX_RESIDUAL_RATIO
        {
            best.decision = SuppressionDecision::Suppress;
        }
        best
    }

    fn expire(&mut self, now: ExactTimestamp) {
        if self
            .reference
            .as_ref()
            .is_some_and(|reference| now.unix_nanos > reference.expected_until.unix_nanos)
        {
            self.reference = None;
        }
    }
}

impl Default for SpeakerReferenceMask {
    fn default() -> Self {
        Self::new()
    }
}

fn compare_with_reference(
    reference: &SpeakerReference,
    frame: &AudioFrame,
    mic: &[f32],
    delay_ms: i64,
) -> Option<SpeakerReferenceDecision> {
    let elapsed_nanos = frame
        .captured_at
        .unix_nanos
        .checked_sub(reference.started_at.unix_nanos)?;
    let delayed_nanos = elapsed_nanos.checked_sub((delay_ms as u128).saturating_mul(1_000_000))?;
    let ref_start =
        delayed_nanos.saturating_mul(u128::from(reference.sample_rate_hz)) / 1_000_000_000;
    let ref_start = usize::try_from(ref_start).ok()?;
    let ref_step = reference.sample_rate_hz as f64 / frame.sample_rate_hz as f64;
    let reference_window =
        sample_reference_window(&reference.samples, ref_start, ref_step, mic.len())?;
    let (correlation, residual_ratio, gain) = correlation_and_residual_gain(mic, &reference_window);
    let (self_frame, residual_frame) = decompose_frame(frame, &reference_window, gain);
    Some(speaker_reference_decision(
        SuppressionDecision::Allow,
        correlation,
        residual_ratio,
        delay_ms,
        gain,
        residual_frame,
        self_frame,
    ))
}

fn sample_reference_window(
    reference: &[f32],
    ref_start: usize,
    ref_step: f64,
    len: usize,
) -> Option<Vec<f32>> {
    let last = ref_start as f64 + ref_step * len.saturating_sub(1) as f64;
    if last as usize >= reference.len() {
        return None;
    }
    let mut output = Vec::with_capacity(len);
    for i in 0..len {
        let pos = ref_start as f64 + ref_step * i as f64;
        let idx = pos.floor() as usize;
        let frac = (pos - idx as f64) as f32;
        let a = reference.get(idx).copied().unwrap_or(0.0);
        let b = reference.get(idx + 1).copied().unwrap_or(a);
        output.push(a + (b - a) * frac);
    }
    Some(output)
}

fn correlation_and_residual_gain(mic: &[f32], reference: &[f32]) -> (f32, f32, f32) {
    let mut dot = 0.0f32;
    let mut mic_energy = 0.0f32;
    let mut ref_energy = 0.0f32;
    for (&mic_sample, &ref_sample) in mic.iter().zip(reference) {
        dot += mic_sample * ref_sample;
        mic_energy += mic_sample * mic_sample;
        ref_energy += ref_sample * ref_sample;
    }
    if mic_energy <= f32::EPSILON || ref_energy <= f32::EPSILON {
        return (0.0, 1.0, 0.0);
    }
    let gain = dot / ref_energy;
    let mut residual_energy = 0.0f32;
    for (&mic_sample, &ref_sample) in mic.iter().zip(reference) {
        let residual = mic_sample - gain * ref_sample;
        residual_energy += residual * residual;
    }
    let correlation = (dot.abs() / (mic_energy.sqrt() * ref_energy.sqrt())).clamp(0.0, 1.0);
    let residual_ratio = (residual_energy / mic_energy).clamp(0.0, 1.0);
    (correlation, residual_ratio, gain)
}

fn decompose_frame(
    frame: &AudioFrame,
    reference_window: &[f32],
    gain: f32,
) -> (AudioFrame, AudioFrame) {
    let channel_count = usize::from(frame.channels).max(1);
    let mut self_samples = Vec::with_capacity(frame.samples.len());
    let mut residual_samples = Vec::with_capacity(frame.samples.len());
    for (chunk, &reference_sample) in frame.samples.chunks(channel_count).zip(reference_window) {
        let self_sample = gain * reference_sample;
        for &sample in chunk {
            self_samples.push(self_sample);
            residual_samples.push(sample - self_sample);
        }
    }
    (
        AudioFrame {
            captured_at: frame.captured_at,
            sample_rate_hz: frame.sample_rate_hz,
            channels: frame.channels,
            samples: self_samples,
            voice_signatures: Vec::new(),
        },
        AudioFrame {
            captured_at: frame.captured_at,
            sample_rate_hz: frame.sample_rate_hz,
            channels: frame.channels,
            samples: residual_samples,
            voice_signatures: Vec::new(),
        },
    )
}

fn speaker_reference_decision(
    decision: SuppressionDecision,
    correlation: f32,
    residual_ratio: f32,
    delay_ms: i64,
    gain: f32,
    residual_frame: AudioFrame,
    self_frame: AudioFrame,
) -> SpeakerReferenceDecision {
    SpeakerReferenceDecision {
        decision,
        correlation,
        residual_ratio,
        delay_ms,
        gain,
        residual_frame,
        self_frame,
    }
}

fn empty_speaker_reference_decision(
    frame: &AudioFrame,
    decision: SuppressionDecision,
) -> SpeakerReferenceDecision {
    let self_frame = AudioFrame {
        captured_at: frame.captured_at,
        sample_rate_hz: frame.sample_rate_hz,
        channels: frame.channels,
        samples: vec![0.0; frame.samples.len()],
        voice_signatures: Vec::new(),
    };
    speaker_reference_decision(decision, 0.0, 1.0, 0, 0.0, frame.clone(), self_frame)
}

fn downmix_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    let channels = usize::from(channels.max(1));
    samples
        .chunks(channels)
        .map(|chunk| chunk.iter().copied().sum::<f32>() / chunk.len() as f32)
        .collect()
}

fn rms_energy(samples: &[f32]) -> f32 {
    samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::audio::frame::AudioFrame;
    use crate::soundscape::{SourceCriterion, SourceId, SourceOperation};
    use crate::time::ExactTimestamp;

    use super::{
        SUPPRESSION_PRE_ROLL_MS, SUPPRESSION_TAIL_MS, SelfHearingState, SpeakerReferenceMask,
        SuppressionDecision,
    };

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
    fn speaker_intent_suppresses_before_audio_duration_is_known() {
        let mut state = SelfHearingState::new();
        let intent_at = ExactTimestamp {
            unix_nanos: 10_000_000_000,
        };
        state.mark_output_intent_at("One moment.", intent_at);

        assert_eq!(
            state.suppression_decision_at(intent_at),
            SuppressionDecision::Suppress
        );
        assert_eq!(
            state.suppression_decision_at(ExactTimestamp {
                unix_nanos: intent_at
                    .unix_nanos
                    .saturating_sub(u128::from(SUPPRESSION_PRE_ROLL_MS) * 1_000_000)
            }),
            SuppressionDecision::Suppress
        );
    }

    #[test]
    fn active_speaking_does_not_suppress_frames_before_pre_roll() {
        let mut state = SelfHearingState::new();
        let intent_at = ExactTimestamp {
            unix_nanos: 10_000_000_000,
        };
        state.mark_output_intent_at("One moment.", intent_at);

        assert_eq!(
            state.suppression_decision_at(ExactTimestamp {
                unix_nanos: intent_at
                    .unix_nanos
                    .saturating_sub(u128::from(SUPPRESSION_PRE_ROLL_MS + 1) * 1_000_000)
            }),
            SuppressionDecision::Allow
        );
    }

    #[test]
    fn audio_start_keeps_original_intent_as_window_start() {
        let mut state = SelfHearingState::new();
        let intent_at = ExactTimestamp {
            unix_nanos: 10_000_000_000,
        };
        let playback_at = ExactTimestamp {
            unix_nanos: 12_000_000_000,
        };
        state.mark_output_intent_at("One moment.", intent_at);
        state.mark_output_started_at("One moment.", Duration::from_millis(500), playback_at);

        assert_eq!(
            state.output_started_at,
            Some(ExactTimestamp {
                unix_nanos: intent_at
                    .unix_nanos
                    .saturating_sub(u128::from(SUPPRESSION_PRE_ROLL_MS) * 1_000_000)
            })
        );
        assert_eq!(
            state.output_expected_until,
            Some(ExactTimestamp {
                unix_nanos: playback_at
                    .unix_nanos
                    .saturating_add(u128::from(500 + SUPPRESSION_TAIL_MS) * 1_000_000)
            })
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

    #[test]
    fn self_hearing_can_be_expressed_as_known_source_suppression_policy() {
        let pete_source_id = SourceId::new();
        let policy = SelfHearingState::isolation_policy_for_source(pete_source_id);
        assert_eq!(policy.operation, SourceOperation::Suppress);
        assert_eq!(
            policy.criterion,
            SourceCriterion::KnownSource(pete_source_id)
        );
        assert_eq!(policy.strength, 1.0);
    }

    #[test]
    fn speaker_reference_mask_suppresses_delayed_echo() {
        let sample_rate_hz = 16_000;
        let started_at = ExactTimestamp {
            unix_nanos: 1_000_000_000,
        };
        let reference = test_noise(3_200);
        let frames = vec![AudioFrame {
            captured_at: started_at,
            sample_rate_hz,
            channels: 1,
            samples: reference.clone(),
            voice_signatures: Vec::new(),
        }];
        let mut mask = SpeakerReferenceMask::new();
        mask.mark_output_started(&frames, started_at);

        let mic_start = 800;
        let mic = reference[mic_start..mic_start + 160]
            .iter()
            .map(|sample| sample * 0.4)
            .collect::<Vec<_>>();
        let decision = mask.analyze_frame(&AudioFrame {
            captured_at: ExactTimestamp {
                unix_nanos: started_at.unix_nanos + 50_000_000,
            },
            sample_rate_hz,
            channels: 1,
            samples: mic,
            voice_signatures: Vec::new(),
        });

        assert_eq!(decision.decision, SuppressionDecision::Suppress);
        assert!(decision.correlation > 0.99);
        assert!(decision.residual_ratio < 0.01);
    }

    #[test]
    fn speaker_reference_mask_allows_speech_mixed_with_echo() {
        let sample_rate_hz = 16_000;
        let started_at = ExactTimestamp {
            unix_nanos: 1_000_000_000,
        };
        let reference = test_noise(3_200);
        let frames = vec![AudioFrame {
            captured_at: started_at,
            sample_rate_hz,
            channels: 1,
            samples: reference.clone(),
            voice_signatures: Vec::new(),
        }];
        let mut mask = SpeakerReferenceMask::new();
        mask.mark_output_started(&frames, started_at);

        let mic_start = 800;
        let user = test_noise(160).into_iter().rev().collect::<Vec<_>>();
        let mic = reference[mic_start..mic_start + 160]
            .iter()
            .zip(user)
            .map(|(echo, user)| echo * 0.25 + user * 0.8)
            .collect::<Vec<_>>();
        let decision = mask.analyze_frame(&AudioFrame {
            captured_at: ExactTimestamp {
                unix_nanos: started_at.unix_nanos + 50_000_000,
            },
            sample_rate_hz,
            channels: 1,
            samples: mic,
            voice_signatures: Vec::new(),
        });

        assert_eq!(decision.decision, SuppressionDecision::Allow);
    }

    fn test_noise(len: usize) -> Vec<f32> {
        let mut state = 0x1234_5678u32;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let value = ((state >> 8) as f32 / 16_777_215.0) * 2.0 - 1.0;
                value * 0.5
            })
            .collect()
    }
}
