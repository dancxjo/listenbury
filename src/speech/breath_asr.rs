use crate::audio::frame::AudioFrame;
use crate::event::HearingEvent;
use crate::hearing::breath::{BreathGroupId, BreathGroupSegmenter};
use crate::hearing::vad::{EnergyVad, VoiceActivityDetector};
use crate::time::ExactTimestamp;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub struct BreathAsrConfig {
    pub pre_roll_ms: u64,
    pub trailing_pad_ms: u64,
    pub min_group_ms: u64,
    pub max_group_ms: u64,
}

impl Default for BreathAsrConfig {
    fn default() -> Self {
        Self {
            pre_roll_ms: 100,
            trailing_pad_ms: 100,
            min_group_ms: 150,
            max_group_ms: 15_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BreathAudioSegment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub start_captured_at: ExactTimestamp,
    pub end_captured_at: ExactTimestamp,
    pub frames: Vec<AudioFrame>,
}

impl BreathAudioSegment {
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

pub fn collect_breath_segments(
    frames: &[AudioFrame],
    config: BreathAsrConfig,
) -> anyhow::Result<Vec<BreathAudioSegment>> {
    let mut vad = EnergyVad::default();
    let mut segmenter = BreathGroupSegmenter::default();
    let mut group_start_ms = HashMap::<BreathGroupId, u64>::new();
    let mut closed_groups = Vec::<(u64, u64)>::new();

    let mut frame_starts = Vec::with_capacity(frames.len());
    let mut frame_ends = Vec::with_capacity(frames.len());
    let mut t_ms = 0_u64;

    for frame in frames {
        let start_ms = t_ms;
        let end_ms = start_ms.saturating_add(frame_duration_ms(frame));
        frame_starts.push(start_ms);
        frame_ends.push(end_ms);

        let vad_result = vad.process_frame(frame)?;
        for hearing_event in segmenter.process(vad_result) {
            match hearing_event {
                HearingEvent::BreathGroupOpened { id } => {
                    group_start_ms.insert(id, start_ms);
                }
                HearingEvent::BreathGroupClosed { id, .. } => {
                    if let Some(opened_ms) = group_start_ms.remove(&id) {
                        closed_groups.push((opened_ms, end_ms));
                    }
                }
                HearingEvent::SpeechStarted
                | HearingEvent::SpeechContinued { .. }
                | HearingEvent::PauseStarted => {}
            }
        }

        t_ms = end_ms;
    }

    let mut segments = Vec::new();
    for (opened_ms, closed_ms) in closed_groups {
        let padded_start_ms = opened_ms.saturating_sub(config.pre_roll_ms);
        let padded_end_ms = closed_ms.saturating_add(config.trailing_pad_ms).min(t_ms);

        for (start_ms, end_ms) in split_by_max_duration(padded_start_ms, padded_end_ms, config) {
            if let Some(segment) =
                build_segment(frames, &frame_starts, &frame_ends, start_ms, end_ms)
            {
                segments.push(segment);
            }
        }
    }

    Ok(segments)
}

fn split_by_max_duration(start_ms: u64, end_ms: u64, config: BreathAsrConfig) -> Vec<(u64, u64)> {
    if end_ms <= start_ms {
        return Vec::new();
    }
    if config.max_group_ms == 0 {
        return if end_ms.saturating_sub(start_ms) >= config.min_group_ms {
            vec![(start_ms, end_ms)]
        } else {
            Vec::new()
        };
    }

    let mut ranges = Vec::new();
    let mut cursor = start_ms;
    while cursor < end_ms {
        let next = cursor.saturating_add(config.max_group_ms).min(end_ms);
        if next.saturating_sub(cursor) >= config.min_group_ms {
            ranges.push((cursor, next));
        }
        cursor = next;
    }
    ranges
}

fn build_segment(
    frames: &[AudioFrame],
    frame_starts: &[u64],
    frame_ends: &[u64],
    start_ms: u64,
    end_ms: u64,
) -> Option<BreathAudioSegment> {
    let start_idx = frame_ends
        .iter()
        .position(|frame_end| *frame_end > start_ms)?;
    let end_idx = frame_starts
        .iter()
        .position(|frame_start| *frame_start >= end_ms)
        .unwrap_or(frames.len());

    if end_idx <= start_idx {
        return None;
    }

    let segment_frames = frames[start_idx..end_idx].to_vec();
    let first = segment_frames.first()?;
    let last = segment_frames.last()?;

    Some(BreathAudioSegment {
        start_ms: frame_starts[start_idx],
        end_ms: frame_ends[end_idx - 1],
        start_captured_at: first.captured_at,
        end_captured_at: last.captured_at,
        frames: segment_frames,
    })
}

fn frame_duration_ms(frame: &AudioFrame) -> u64 {
    if frame.sample_rate_hz == 0 || frame.channels == 0 {
        return 0;
    }
    let samples_per_channel = frame.samples.len() as f64 / f64::from(frame.channels);
    ((samples_per_channel / f64::from(frame.sample_rate_hz)) * 1000.0).round() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(amplitude: f32) -> AudioFrame {
        AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![amplitude; 160],
        }
    }

    #[test]
    fn silence_yields_no_segments() {
        let frames = vec![make_frame(0.0); 60];
        let segments = collect_breath_segments(&frames, BreathAsrConfig::default()).unwrap();
        assert!(segments.is_empty());
    }

    #[test]
    fn collects_multiple_groups_and_resets_between_them() {
        let mut frames = Vec::new();
        frames.extend(std::iter::repeat_with(|| make_frame(0.0)).take(5));
        frames.extend(std::iter::repeat_with(|| make_frame(0.3)).take(6));
        frames.extend(std::iter::repeat_with(|| make_frame(0.0)).take(12));
        frames.extend(std::iter::repeat_with(|| make_frame(0.3)).take(6));
        frames.extend(std::iter::repeat_with(|| make_frame(0.0)).take(12));

        let segments = collect_breath_segments(
            &frames,
            BreathAsrConfig {
                pre_roll_ms: 20,
                trailing_pad_ms: 20,
                min_group_ms: 40,
                max_group_ms: 2_000,
            },
        )
        .unwrap();

        assert_eq!(segments.len(), 2);
        assert!(segments[0].duration_ms() >= 40);
        assert!(segments[1].start_ms >= segments[0].end_ms);
    }

    #[test]
    fn pre_roll_and_trailing_pad_expand_segment_window() {
        let mut frames = Vec::new();
        frames.extend(std::iter::repeat_with(|| make_frame(0.0)).take(4));
        frames.extend(std::iter::repeat_with(|| make_frame(0.3)).take(6));
        frames.extend(std::iter::repeat_with(|| make_frame(0.0)).take(12));

        let segments = collect_breath_segments(
            &frames,
            BreathAsrConfig {
                pre_roll_ms: 100,
                trailing_pad_ms: 100,
                min_group_ms: 40,
                max_group_ms: 2_000,
            },
        )
        .unwrap();

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start_ms, 0);
        assert!(segments[0].end_ms >= 200);
    }

    #[test]
    fn splits_long_groups_by_max_duration() {
        let mut frames = Vec::new();
        frames.extend(std::iter::repeat_with(|| make_frame(0.3)).take(70));
        frames.extend(std::iter::repeat_with(|| make_frame(0.0)).take(12));

        let segments = collect_breath_segments(
            &frames,
            BreathAsrConfig {
                pre_roll_ms: 0,
                trailing_pad_ms: 0,
                min_group_ms: 50,
                max_group_ms: 200,
            },
        )
        .unwrap();

        assert!(!segments.is_empty());
        assert!(segments.iter().all(|segment| segment.duration_ms() <= 200));
    }
}
