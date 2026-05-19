use serde::{Deserialize, Serialize};

use crate::audio::frame::AudioFrame;
use crate::time::ExactTimestamp;

pub const PROSODY_FEATURE_LATENCY_TARGET_MS: u64 = 50;
pub const ECHO_PLANNING_LATENCY_TARGET_MS: u64 = 150;

const PAUSE_THRESHOLD_DBFS: f32 = -44.0;
const PAUSE_MIN_MS: u64 = 120;
const ACCENT_DELTA_THRESHOLD: f32 = 0.04;
const MODEL_HISTORY_LIMIT: usize = 12;
const NANOS_PER_MS: u128 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProsodyProvenance {
    Provisional,
    Revised,
    Confirmed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamingProsodyFrame {
    pub frame_start_ms: u64,
    pub frame_end_ms: u64,
    pub rms_loudness: f32,
    pub loudness_dbfs: f32,
    pub pitch_hz: Option<f32>,
    pub voicing_confidence: f32,
    pub energy_contour: f32,
    pub pause_marker: bool,
    pub speech_rate_proxy: f32,
    pub spectral_tilt: Option<f32>,
    pub confidence: f32,
    pub provenance: ProsodyProvenance,
    pub revision: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProsodyPauseBoundary {
    pub start_ms: u64,
    pub end_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProsodyPhraseCandidate {
    pub start_ms: u64,
    pub end_ms: u64,
    pub confidence: f32,
    pub provenance: ProsodyProvenance,
    pub revision: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProsodyAccentCandidate {
    pub at_ms: u64,
    pub confidence: f32,
    pub provenance: ProsodyProvenance,
    pub revision: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RollingProsodyModel {
    pub current_pitch_range_hz: Option<(f32, f32)>,
    pub recent_loudness_contour: Vec<f32>,
    pub likely_stress_peaks_ms: Vec<u64>,
    pub recent_pause_boundaries: Vec<ProsodyPauseBoundary>,
    pub provisional_phrase_boundaries_ms: Vec<u64>,
    pub confidence: f32,
    pub provenance: ProsodyProvenance,
    pub revision: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamingProsodyUpdate {
    pub frame: StreamingProsodyFrame,
    pub model: RollingProsodyModel,
    pub contour: Option<f32>,
    pub pause: Option<ProsodyPauseBoundary>,
    pub phrase_candidate: Option<ProsodyPhraseCandidate>,
    pub accent_candidate: Option<ProsodyAccentCandidate>,
    pub observed_feature_latency_ms: u64,
    pub latency_target_ms: u64,
    pub captured_at_unix_ns: u128,
    pub processed_at_unix_ns: u128,
}

#[derive(Debug, Clone)]
pub struct StreamingProsodyAnalyzer {
    model: RollingProsodyModel,
    last_rms: Option<f32>,
    pause_start_ms: Option<u64>,
    last_pause_end_ms: Option<u64>,
    frame_counter: u64,
    last_evidence_at: Option<ExactTimestamp>,
}

impl Default for StreamingProsodyAnalyzer {
    fn default() -> Self {
        Self {
            model: RollingProsodyModel {
                current_pitch_range_hz: None,
                recent_loudness_contour: Vec::new(),
                likely_stress_peaks_ms: Vec::new(),
                recent_pause_boundaries: Vec::new(),
                provisional_phrase_boundaries_ms: Vec::new(),
                confidence: 0.0,
                provenance: ProsodyProvenance::Provisional,
                revision: 0,
            },
            last_rms: None,
            pause_start_ms: None,
            last_pause_end_ms: None,
            frame_counter: 0,
            last_evidence_at: None,
        }
    }
}

impl StreamingProsodyAnalyzer {
    pub fn ingest_frame(
        &mut self,
        frame: &AudioFrame,
        frame_start_ms: u64,
    ) -> Option<StreamingProsodyUpdate> {
        if frame.samples.is_empty() || frame.channels == 0 || frame.sample_rate_hz == 0 {
            return None;
        }
        let frame_duration_ms = frame_duration_ms(frame);
        let frame_end_ms = frame_start_ms.saturating_add(frame_duration_ms);
        let mono = mono_samples(frame);
        if mono.is_empty() {
            return None;
        }
        self.frame_counter = self.frame_counter.saturating_add(1);
        self.model.revision = self.model.revision.saturating_add(1);

        let rms = rms(&mono);
        let dbfs = dbfs_from_rms(rms);
        let pitch = estimate_pitch_hz(&mono, frame.sample_rate_hz);
        let voicing_confidence = pitch
            .map(|_| estimate_voicing_confidence(&mono, dbfs))
            .unwrap_or(0.0);
        let energy_contour = self.last_rms.map_or(0.0, |prev| rms - prev);
        self.last_rms = Some(rms);
        let speech_rate_proxy = (voicing_confidence * 1.4).clamp(0.0, 1.0);
        let spectral_tilt = estimate_spectral_tilt(&mono);
        let pause_marker = dbfs <= PAUSE_THRESHOLD_DBFS;

        if let Some(pitch_hz) = pitch {
            self.model.current_pitch_range_hz = Some(match self.model.current_pitch_range_hz {
                Some((min_hz, max_hz)) => (min_hz.min(pitch_hz), max_hz.max(pitch_hz)),
                None => (pitch_hz, pitch_hz),
            });
        }
        push_with_limit(
            &mut self.model.recent_loudness_contour,
            dbfs,
            MODEL_HISTORY_LIMIT,
        );

        let accent_candidate = if energy_contour >= ACCENT_DELTA_THRESHOLD && !pause_marker {
            let at_ms = frame_start_ms.saturating_add(frame_duration_ms / 2);
            push_with_limit(
                &mut self.model.likely_stress_peaks_ms,
                at_ms,
                MODEL_HISTORY_LIMIT,
            );
            Some(ProsodyAccentCandidate {
                at_ms,
                confidence: (0.4 + voicing_confidence * 0.6).clamp(0.0, 1.0),
                provenance: ProsodyProvenance::Provisional,
                revision: self.model.revision,
            })
        } else {
            None
        };

        let pause = self.update_pause_model(frame_start_ms, frame_end_ms, pause_marker);
        let phrase_candidate = pause.as_ref().map(|pause_boundary| {
            push_with_limit(
                &mut self.model.provisional_phrase_boundaries_ms,
                pause_boundary.end_ms,
                MODEL_HISTORY_LIMIT,
            );
            ProsodyPhraseCandidate {
                start_ms: pause_boundary.start_ms,
                end_ms: pause_boundary.end_ms,
                confidence: 0.7,
                provenance: ProsodyProvenance::Revised,
                revision: self.model.revision,
            }
        });

        self.model.confidence = ((voicing_confidence
            + (1.0 - (dbfs.abs() / 80.0).clamp(0.0, 1.0))
            + if pause_marker { 0.65 } else { 0.35 })
            / 3.0)
            .clamp(0.0, 1.0);
        self.model.provenance = if self.frame_counter < 3 {
            ProsodyProvenance::Provisional
        } else if pause.is_some() {
            ProsodyProvenance::Confirmed
        } else {
            ProsodyProvenance::Revised
        };

        let captured_at = frame.captured_at;
        let processed_at = ExactTimestamp::now();
        self.last_evidence_at = Some(captured_at);
        let observed_feature_latency_ms = saturating_elapsed_ms(captured_at, processed_at);

        let update = StreamingProsodyUpdate {
            frame: StreamingProsodyFrame {
                frame_start_ms,
                frame_end_ms,
                rms_loudness: rms,
                loudness_dbfs: dbfs,
                pitch_hz: pitch,
                voicing_confidence,
                energy_contour,
                pause_marker,
                speech_rate_proxy,
                spectral_tilt,
                confidence: self.model.confidence,
                provenance: self.model.provenance,
                revision: self.model.revision,
            },
            model: self.model.clone(),
            contour: Some(energy_contour),
            pause,
            phrase_candidate,
            accent_candidate,
            observed_feature_latency_ms,
            latency_target_ms: PROSODY_FEATURE_LATENCY_TARGET_MS,
            captured_at_unix_ns: captured_at.unix_nanos,
            processed_at_unix_ns: processed_at.unix_nanos,
        };
        Some(update)
    }

    pub fn latest_model(&self) -> &RollingProsodyModel {
        &self.model
    }

    pub fn last_evidence_at(&self) -> Option<ExactTimestamp> {
        self.last_evidence_at
    }

    fn update_pause_model(
        &mut self,
        frame_start_ms: u64,
        frame_end_ms: u64,
        pause_marker: bool,
    ) -> Option<ProsodyPauseBoundary> {
        if pause_marker {
            let start = self.pause_start_ms.get_or_insert(frame_start_ms);
            let duration = frame_end_ms.saturating_sub(*start);
            if duration >= PAUSE_MIN_MS {
                let pause = ProsodyPauseBoundary {
                    start_ms: *start,
                    end_ms: frame_end_ms,
                };
                let should_record = self.last_pause_end_ms.is_none_or(|end| end != frame_end_ms);
                if should_record {
                    push_with_limit(
                        &mut self.model.recent_pause_boundaries,
                        pause.clone(),
                        MODEL_HISTORY_LIMIT,
                    );
                    self.last_pause_end_ms = Some(frame_end_ms);
                }
                return Some(pause);
            }
            return None;
        }
        self.pause_start_ms = None;
        None
    }
}

fn frame_duration_ms(frame: &AudioFrame) -> u64 {
    if frame.sample_rate_hz == 0 || frame.channels == 0 {
        return 0;
    }
    let samples_per_channel = frame.samples.len() as f64 / f64::from(frame.channels);
    ((samples_per_channel / f64::from(frame.sample_rate_hz)) * 1000.0).round() as u64
}

fn mono_samples(frame: &AudioFrame) -> Vec<f32> {
    let channels = usize::from(frame.channels).max(1);
    if channels == 1 {
        return frame.samples.clone();
    }
    frame
        .samples
        .chunks_exact(channels)
        .map(|chunk| chunk.iter().copied().sum::<f32>() / channels as f32)
        .collect()
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum = samples.iter().map(|sample| sample * sample).sum::<f32>();
    (sum / samples.len() as f32).sqrt()
}

fn dbfs_from_rms(rms: f32) -> f32 {
    let eps = 1e-6f32;
    (20.0 * (rms.max(eps)).log10()).clamp(-120.0, 0.0)
}

fn estimate_voicing_confidence(samples: &[f32], dbfs: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let voiced_ratio = samples
        .iter()
        .filter(|sample| sample.abs() >= 0.015)
        .count() as f32
        / samples.len() as f32;
    let loudness_weight = ((dbfs + 60.0) / 60.0).clamp(0.0, 1.0);
    (voiced_ratio * 0.7 + loudness_weight * 0.3).clamp(0.0, 1.0)
}

fn estimate_pitch_hz(samples: &[f32], sample_rate_hz: u32) -> Option<f32> {
    if samples.len() < 8 || sample_rate_hz == 0 {
        return None;
    }
    let mut crossings = 0usize;
    for window in samples.windows(2) {
        let a = window[0];
        let b = window[1];
        if (a <= 0.0 && b > 0.0) || (a >= 0.0 && b < 0.0) {
            crossings = crossings.saturating_add(1);
        }
    }
    if crossings < 2 {
        return None;
    }
    let seconds = samples.len() as f32 / sample_rate_hz as f32;
    if seconds <= 0.0 {
        return None;
    }
    let estimate = (crossings as f32 / 2.0) / seconds;
    if (70.0..=420.0).contains(&estimate) {
        Some(estimate)
    } else {
        None
    }
}

fn estimate_spectral_tilt(samples: &[f32]) -> Option<f32> {
    if samples.len() < 4 {
        return None;
    }
    let half = samples.len() / 2;
    if half == 0 {
        return None;
    }
    let low = samples[..half]
        .iter()
        .map(|sample| sample.abs())
        .sum::<f32>()
        / half as f32;
    let high = samples[half..]
        .iter()
        .map(|sample| sample.abs())
        .sum::<f32>()
        / (samples.len() - half) as f32;
    Some((low - high).clamp(-1.0, 1.0))
}

fn push_with_limit<T>(values: &mut Vec<T>, value: T, limit: usize) {
    values.push(value);
    if values.len() > limit {
        let drain_count = values.len().saturating_sub(limit);
        values.drain(0..drain_count);
    }
}

pub fn saturating_elapsed_ms(start: ExactTimestamp, end: ExactTimestamp) -> u64 {
    let nanos = end.unix_nanos.saturating_sub(start.unix_nanos);
    u64::try_from(nanos / NANOS_PER_MS).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame_with_level(
        dbfs: f32,
        sample_rate_hz: u32,
        duration_ms: u64,
        captured_ms: u128,
    ) -> AudioFrame {
        let channels = 1u16;
        let sample_count = usize::try_from((u64::from(sample_rate_hz) * duration_ms) / 1_000)
            .unwrap_or(1)
            .max(1);
        let linear = 10f32.powf(dbfs / 20.0);
        AudioFrame {
            captured_at: ExactTimestamp {
                unix_nanos: captured_ms.saturating_mul(NANOS_PER_MS),
            },
            sample_rate_hz,
            channels,
            samples: vec![linear; sample_count],
            voice_signatures: Vec::new(),
        }
    }

    fn sine_frame(
        freq_hz: f32,
        sample_rate_hz: u32,
        duration_ms: u64,
        amplitude: f32,
        captured_ms: u128,
    ) -> AudioFrame {
        let sample_count = usize::try_from((u64::from(sample_rate_hz) * duration_ms) / 1_000)
            .unwrap_or(1)
            .max(1);
        let mut samples = Vec::with_capacity(sample_count);
        for i in 0..sample_count {
            let t = i as f32 / sample_rate_hz as f32;
            samples.push((2.0 * std::f32::consts::PI * freq_hz * t).sin() * amplitude);
        }
        AudioFrame {
            captured_at: ExactTimestamp {
                unix_nanos: captured_ms.saturating_mul(NANOS_PER_MS),
            },
            sample_rate_hz,
            channels: 1,
            samples,
            voice_signatures: Vec::new(),
        }
    }

    #[test]
    fn rolling_model_tracks_pitch_and_revisions() {
        let mut analyzer = StreamingProsodyAnalyzer::default();
        let first = sine_frame(180.0, 16_000, 20, 0.5, 1);
        let second = sine_frame(220.0, 16_000, 20, 0.5, 2);
        let first_update = analyzer.ingest_frame(&first, 0).expect("first update");
        let second_update = analyzer.ingest_frame(&second, 20).expect("second update");
        let (min_hz, max_hz) = second_update
            .model
            .current_pitch_range_hz
            .expect("pitch range");
        assert!(max_hz > min_hz);
        assert!((max_hz - min_hz) >= 5.0);
        assert!(second_update.model.revision > first_update.model.revision);
    }

    #[test]
    fn pause_detection_and_phrase_candidate_emit_before_sentence_finalization() {
        let mut analyzer = StreamingProsodyAnalyzer::default();
        let mut pause_detected = None;
        for idx in 0..8 {
            let frame = frame_with_level(-70.0, 16_000, 20, idx + 1);
            let update = analyzer
                .ingest_frame(&frame, (idx.saturating_mul(20)) as u64)
                .expect("prosody update");
            if update.pause.is_some() {
                pause_detected = Some(update);
                break;
            }
        }
        let pause_update = pause_detected.expect("pause should be detected after rolling silence");
        assert!(pause_update.pause.is_some());
        assert!(pause_update.phrase_candidate.is_some());
    }

    #[test]
    fn accent_candidates_follow_energy_peaks() {
        let mut analyzer = StreamingProsodyAnalyzer::default();
        let low = frame_with_level(-28.0, 16_000, 20, 1);
        let high = frame_with_level(-6.0, 16_000, 20, 2);
        analyzer.ingest_frame(&low, 0).expect("low frame");
        let update = analyzer.ingest_frame(&high, 20).expect("high frame");
        assert!(update.accent_candidate.is_some());
        assert!(update.frame.energy_contour > 0.0);
    }

    #[test]
    fn provenance_moves_from_provisional_to_revised_and_confirmed() {
        let mut analyzer = StreamingProsodyAnalyzer::default();
        let speech_a = sine_frame(160.0, 16_000, 20, 0.5, 1);
        let speech_b = sine_frame(170.0, 16_000, 20, 0.5, 2);
        let speech_c = sine_frame(180.0, 16_000, 20, 0.5, 3);
        let first = analyzer.ingest_frame(&speech_a, 0).expect("first");
        let second = analyzer.ingest_frame(&speech_b, 20).expect("second");
        let third = analyzer.ingest_frame(&speech_c, 40).expect("third");
        assert_eq!(first.model.provenance, ProsodyProvenance::Provisional);
        assert_eq!(second.model.provenance, ProsodyProvenance::Provisional);
        assert_eq!(third.model.provenance, ProsodyProvenance::Revised);

        let mut confirmed = None;
        for idx in 0..8 {
            let silence = frame_with_level(-72.0, 16_000, 20, 10 + idx);
            let update = analyzer
                .ingest_frame(&silence, (60 + idx.saturating_mul(20)) as u64)
                .expect("silence update");
            if update.model.provenance == ProsodyProvenance::Confirmed {
                confirmed = Some(update);
                break;
            }
        }
        assert!(confirmed.is_some());
    }

    #[test]
    fn latency_measurement_uses_capture_timestamp() {
        let mut analyzer = StreamingProsodyAnalyzer::default();
        let frame = sine_frame(190.0, 16_000, 20, 0.5, 0);
        let update = analyzer.ingest_frame(&frame, 0).expect("update");
        assert_eq!(update.latency_target_ms, PROSODY_FEATURE_LATENCY_TARGET_MS);
        assert!(update.observed_feature_latency_ms > 0);
    }
}
