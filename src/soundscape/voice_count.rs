use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::soundscape::{AttributionEvidence, SourceHypothesis, SourceId, SourceKind, TimeRange};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceCount {
    pub active_now: usize,
    pub recently_heard: usize,
    pub known: usize,
    pub unknown: usize,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceActivityFrame {
    pub range: TimeRange,
    pub estimated_voice_count: usize,
    pub source_hypotheses: Vec<SourceHypothesis>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoiceCountConfig {
    pub recent_window_millis: u64,
    pub continuity_gap_millis: u64,
}

impl Default for VoiceCountConfig {
    fn default() -> Self {
        Self {
            recent_window_millis: 1_500,
            continuity_gap_millis: 250,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RollingVoiceCountEstimator {
    config: VoiceCountConfig,
    tracks: Vec<TrackedVoice>,
    next_unknown_ordinal: u64,
}

impl RollingVoiceCountEstimator {
    pub fn new(config: VoiceCountConfig) -> Self {
        Self {
            config,
            ..Self::default()
        }
    }

    pub fn update(
        &mut self,
        range: TimeRange,
        source_hypotheses: Vec<SourceHypothesis>,
    ) -> (VoiceActivityFrame, VoiceCount) {
        self.evict_stale(range);

        let mut hypotheses = source_hypotheses;
        hypotheses.retain(is_voice_like);

        let mut updated_this_frame: HashMap<usize, TimeRange> = HashMap::new();
        let mut hypothesis_track_keys = Vec::with_capacity(hypotheses.len());

        for hypothesis in &hypotheses {
            let track_index = self.resolve_track_for_hypothesis(hypothesis, &updated_this_frame);
            self.update_track(track_index, hypothesis);
            updated_this_frame.insert(track_index, hypothesis.range);
            hypothesis_track_keys.push(self.tracks[track_index].key);
        }

        let estimated_voice_count =
            max_overlapping_voices(&hypotheses, &hypothesis_track_keys, range);

        let recent_cutoff = range
            .end
            .millis
            .saturating_sub(self.config.recent_window_millis);
        let recently_heard_keys: HashSet<ResolvedSourceKey> = self
            .tracks
            .iter()
            .filter(|track| track.last_range.end.millis >= recent_cutoff)
            .map(|track| track.key)
            .collect();

        let known = recently_heard_keys
            .iter()
            .filter(|key| matches!(key, ResolvedSourceKey::Known(_)))
            .count();
        let unknown = recently_heard_keys.len().saturating_sub(known);

        let active_confidences: Vec<f32> = hypotheses
            .iter()
            .map(|hypothesis| hypothesis.confidence.clamp(0.0, 1.0))
            .collect();

        let confidence = if active_confidences.is_empty() {
            0.0
        } else {
            active_confidences.iter().sum::<f32>() / active_confidences.len() as f32
        };

        (
            VoiceActivityFrame {
                range,
                estimated_voice_count,
                source_hypotheses: hypotheses,
            },
            VoiceCount {
                active_now: estimated_voice_count,
                recently_heard: recently_heard_keys.len(),
                known,
                unknown,
                confidence,
            },
        )
    }

    fn evict_stale(&mut self, frame_range: TimeRange) {
        let recent_cutoff = frame_range
            .end
            .millis
            .saturating_sub(self.config.recent_window_millis);
        self.tracks
            .retain(|track| track.last_range.end.millis >= recent_cutoff);
    }

    fn resolve_track_for_hypothesis(
        &mut self,
        hypothesis: &SourceHypothesis,
        updated_this_frame: &HashMap<usize, TimeRange>,
    ) -> usize {
        if let Some(source_id) = hypothesis.source_id {
            if let Some(index) = self
                .tracks
                .iter()
                .position(|track| track.key == ResolvedSourceKey::Known(source_id))
            {
                return index;
            }

            let new_index = self.tracks.len();
            self.tracks.push(TrackedVoice::new(
                ResolvedSourceKey::Known(source_id),
                hypothesis.range,
                hypothesis.confidence,
            ));
            return new_index;
        }

        if let Some(index) = self.find_continuing_unknown_track(hypothesis, updated_this_frame) {
            return index;
        }

        let new_index = self.tracks.len();
        self.tracks.push(TrackedVoice::new(
            ResolvedSourceKey::Unknown {
                ordinal: self.next_unknown_ordinal,
            },
            hypothesis.range,
            hypothesis.confidence,
        ));
        self.next_unknown_ordinal = self.next_unknown_ordinal.saturating_add(1);
        new_index
    }

    fn find_continuing_unknown_track(
        &self,
        hypothesis: &SourceHypothesis,
        updated_this_frame: &HashMap<usize, TimeRange>,
    ) -> Option<usize> {
        let continuity_hint = hypothesis_continuity_hint(hypothesis);

        self.tracks
            .iter()
            .enumerate()
            .filter(|(_, track)| matches!(track.key, ResolvedSourceKey::Unknown { .. }))
            .filter(|(_, track)| {
                let gap = hypothesis
                    .range
                    .start
                    .millis
                    .saturating_sub(track.last_range.end.millis);
                gap <= self.config.continuity_gap_millis
            })
            .filter(|(index, track)| {
                if let Some(updated_range) = updated_this_frame.get(index) {
                    !ranges_overlap(*updated_range, hypothesis.range)
                } else {
                    !ranges_overlap(track.last_range, hypothesis.range)
                }
            })
            .max_by_key(|(_, track)| {
                let gap = hypothesis
                    .range
                    .start
                    .millis
                    .saturating_sub(track.last_range.end.millis);
                let continuity_bonus = if continuity_hint { 1_000_u64 } else { 0_u64 };
                continuity_bonus
                    .saturating_add(self.config.continuity_gap_millis.saturating_sub(gap))
            })
            .map(|(index, _)| index)
    }

    fn update_track(&mut self, track_index: usize, hypothesis: &SourceHypothesis) {
        if let Some(track) = self.tracks.get_mut(track_index) {
            track.last_range = hypothesis.range;
            track.last_confidence = hypothesis.confidence.clamp(0.0, 1.0);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum ResolvedSourceKey {
    Known(SourceId),
    Unknown { ordinal: u64 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct TrackedVoice {
    key: ResolvedSourceKey,
    last_range: TimeRange,
    last_confidence: f32,
}

impl TrackedVoice {
    fn new(key: ResolvedSourceKey, last_range: TimeRange, last_confidence: f32) -> Self {
        Self {
            key,
            last_range,
            last_confidence: last_confidence.clamp(0.0, 1.0),
        }
    }
}

fn is_voice_like(hypothesis: &SourceHypothesis) -> bool {
    matches!(
        hypothesis.kind,
        SourceKind::Voice
            | SourceKind::SyntheticVoice
            | SourceKind::KnownSelfVoice
            | SourceKind::Unknown
    )
}

fn ranges_overlap(left: TimeRange, right: TimeRange) -> bool {
    left.start.millis <= right.end.millis && right.start.millis <= left.end.millis
}

fn max_overlapping_voices(
    hypotheses: &[SourceHypothesis],
    keys: &[ResolvedSourceKey],
    range: TimeRange,
) -> usize {
    let mut boundaries = Vec::with_capacity(hypotheses.len() * 2);
    for (hypothesis, key) in hypotheses.iter().zip(keys.iter()) {
        let clipped_start = hypothesis.range.start.millis.max(range.start.millis);
        let clipped_end = hypothesis.range.end.millis.min(range.end.millis);
        if clipped_start > clipped_end {
            continue;
        }
        boundaries.push((clipped_start, true, *key));
        boundaries.push((clipped_end, false, *key));
    }

    boundaries.sort_by_key(|(millis, is_start, _)| (*millis, !*is_start));

    let mut active = HashSet::new();
    let mut max_active = 0usize;

    for (_, is_start, key) in boundaries {
        if is_start {
            active.insert(key);
            max_active = max_active.max(active.len());
        } else {
            active.remove(&key);
        }
    }

    max_active
}

fn hypothesis_continuity_hint(hypothesis: &SourceHypothesis) -> bool {
    hypothesis.evidence.iter().any(|evidence| {
        matches!(
            evidence,
            AttributionEvidence::PitchContinuity { .. }
                | AttributionEvidence::SpectralContinuity { .. }
                | AttributionEvidence::LexicalContinuity { .. }
        )
    })
}

#[cfg(test)]
mod tests {
    use crate::soundscape::{
        AttributionEvidence, RollingVoiceCountEstimator, SourceHypothesis, SourceId, SourceKind,
        TimePoint, TimeRange,
    };

    fn range(start: u64, end: u64) -> TimeRange {
        TimeRange::new(TimePoint::from_millis(start), TimePoint::from_millis(end))
    }

    fn unknown_voice(start: u64, end: u64, confidence: f32) -> SourceHypothesis {
        SourceHypothesis {
            source_id: None,
            kind: SourceKind::Voice,
            range: range(start, end),
            confidence,
            evidence: vec![],
        }
    }

    #[test]
    fn one_voice_continues_across_adjacent_short_spans() {
        let mut estimator = RollingVoiceCountEstimator::default();

        let (_, first_count) = estimator.update(range(0, 300), vec![unknown_voice(0, 300, 0.92)]);
        assert_eq!(first_count.active_now, 1);
        assert_eq!(first_count.recently_heard, 1);
        assert_eq!(first_count.known, 0);
        assert_eq!(first_count.unknown, 1);

        let second = SourceHypothesis {
            evidence: vec![AttributionEvidence::PitchContinuity { confidence: 0.8 }],
            ..unknown_voice(320, 620, 0.9)
        };
        let (frame, second_count) = estimator.update(range(320, 620), vec![second]);

        assert_eq!(frame.estimated_voice_count, 1);
        assert_eq!(second_count.active_now, 1);
        assert_eq!(second_count.recently_heard, 1);
        assert_eq!(second_count.known, 0);
        assert_eq!(second_count.unknown, 1);
        assert!(second_count.confidence > 0.0);
    }

    #[test]
    fn two_alternating_known_voices_are_tracked_recently() {
        let mut estimator = RollingVoiceCountEstimator::default();
        let speaker_a = SourceId::new();
        let speaker_b = SourceId::new();

        let first_hypothesis = SourceHypothesis {
            source_id: Some(speaker_a),
            kind: SourceKind::Voice,
            range: range(0, 400),
            confidence: 0.94,
            evidence: vec![],
        };
        let (_, first_count) = estimator.update(range(0, 400), vec![first_hypothesis]);
        assert_eq!(first_count.active_now, 1);
        assert_eq!(first_count.recently_heard, 1);

        let second_hypothesis = SourceHypothesis {
            source_id: Some(speaker_b),
            kind: SourceKind::Voice,
            range: range(500, 900),
            confidence: 0.9,
            evidence: vec![],
        };
        let (frame, second_count) = estimator.update(range(500, 900), vec![second_hypothesis]);

        assert_eq!(frame.estimated_voice_count, 1);
        assert_eq!(second_count.active_now, 1);
        assert_eq!(second_count.recently_heard, 2);
        assert_eq!(second_count.known, 2);
        assert_eq!(second_count.unknown, 0);
    }

    #[test]
    fn two_overlapping_unknown_voices_are_counted_explicitly() {
        let mut estimator = RollingVoiceCountEstimator::default();

        let overlap_a = unknown_voice(0, 700, 0.88);
        let overlap_b = SourceHypothesis {
            evidence: vec![AttributionEvidence::OverlapDetected],
            ..unknown_voice(200, 900, 0.83)
        };

        let (frame, count) = estimator.update(range(0, 900), vec![overlap_a, overlap_b]);

        assert_eq!(frame.estimated_voice_count, 2);
        assert_eq!(count.active_now, 2);
        assert_eq!(count.recently_heard, 2);
        assert_eq!(count.known, 0);
        assert_eq!(count.unknown, 2);
    }
}
