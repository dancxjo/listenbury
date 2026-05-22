use serde::{Deserialize, Serialize};

use crate::voice::tract::{SourceFilterFrame, VocalTractFilterEstimate};

const DEFAULT_MIN_F1_HZ: f32 = 200.0;
const DEFAULT_MAX_F1_HZ: f32 = 1_100.0;
const DEFAULT_MIN_F2_HZ: f32 = 600.0;
const DEFAULT_MAX_F2_HZ: f32 = 3_200.0;
const DEFAULT_MIN_F3_HZ: f32 = 1_400.0;
const DEFAULT_MAX_F3_HZ: f32 = 4_200.0;
const DEFAULT_MIN_F1_F2_SEPARATION_HZ: f32 = 250.0;
const DEFAULT_MIN_F2_F3_SEPARATION_HZ: f32 = 350.0;
const DEFAULT_MAX_FORMANT_DELTA_HZ: f32 = 380.0;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VocalPlausibilityConfig {
    pub min_f1_hz: f32,
    pub max_f1_hz: f32,
    pub min_f2_hz: f32,
    pub max_f2_hz: f32,
    pub min_f3_hz: f32,
    pub max_f3_hz: f32,
    pub min_f1_f2_separation_hz: f32,
    pub min_f2_f3_separation_hz: f32,
    pub max_formant_delta_hz: f32,
}

impl Default for VocalPlausibilityConfig {
    fn default() -> Self {
        Self {
            min_f1_hz: DEFAULT_MIN_F1_HZ,
            max_f1_hz: DEFAULT_MAX_F1_HZ,
            min_f2_hz: DEFAULT_MIN_F2_HZ,
            max_f2_hz: DEFAULT_MAX_F2_HZ,
            min_f3_hz: DEFAULT_MIN_F3_HZ,
            max_f3_hz: DEFAULT_MAX_F3_HZ,
            min_f1_f2_separation_hz: DEFAULT_MIN_F1_F2_SEPARATION_HZ,
            min_f2_f3_separation_hz: DEFAULT_MIN_F2_F3_SEPARATION_HZ,
            max_formant_delta_hz: DEFAULT_MAX_FORMANT_DELTA_HZ,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VocalPlausibility {
    pub formant_confidence: f32,
    pub spacing_confidence: f32,
    pub trajectory_confidence: f32,
    pub plausibility: f32,
}

pub fn assess_vocal_plausibility(
    frame: &SourceFilterFrame,
    prev_frame: Option<&SourceFilterFrame>,
    next_frame: Option<&SourceFilterFrame>,
    config: &VocalPlausibilityConfig,
) -> VocalPlausibility {
    let formant_confidence = mean_formant_confidence(&frame.filter);
    let spacing_confidence = formant_spacing_confidence(&frame.filter, config);
    let trajectory_confidence =
        formant_trajectory_confidence(frame, prev_frame, next_frame, config.max_formant_delta_hz);
    let plausibility = (formant_confidence * 0.35
        + spacing_confidence * 0.45
        + trajectory_confidence * 0.20)
        .clamp(0.0, 1.0);
    VocalPlausibility {
        formant_confidence,
        spacing_confidence,
        trajectory_confidence,
        plausibility,
    }
}

fn mean_formant_confidence(filter: &VocalTractFilterEstimate) -> f32 {
    let confidences = [filter.f1.as_ref(), filter.f2.as_ref(), filter.f3.as_ref()]
        .into_iter()
        .flatten()
        .map(|formant| formant.confidence)
        .collect::<Vec<_>>();
    if confidences.is_empty() {
        return 0.0;
    }
    (confidences.iter().sum::<f32>() / confidences.len() as f32).clamp(0.0, 1.0)
}

fn formant_spacing_confidence(filter: &VocalTractFilterEstimate, config: &VocalPlausibilityConfig) -> f32 {
    let (f1, f2, f3) = match (&filter.f1, &filter.f2, &filter.f3) {
        (Some(f1), Some(f2), Some(f3)) => (f1.frequency_hz, f2.frequency_hz, f3.frequency_hz),
        _ => return 0.0,
    };
    if !(config.min_f1_hz..=config.max_f1_hz).contains(&f1)
        || !(config.min_f2_hz..=config.max_f2_hz).contains(&f2)
        || !(config.min_f3_hz..=config.max_f3_hz).contains(&f3)
    {
        return 0.0;
    }

    let f1_f2 = f2 - f1;
    let f2_f3 = f3 - f2;
    if f1_f2 <= 0.0 || f2_f3 <= 0.0 {
        return 0.0;
    }

    let s1 = (f1_f2 / config.min_f1_f2_separation_hz).clamp(0.0, 1.0);
    let s2 = (f2_f3 / config.min_f2_f3_separation_hz).clamp(0.0, 1.0);
    (s1 * 0.5 + s2 * 0.5).clamp(0.0, 1.0)
}

fn formant_trajectory_confidence(
    frame: &SourceFilterFrame,
    prev_frame: Option<&SourceFilterFrame>,
    next_frame: Option<&SourceFilterFrame>,
    max_delta_hz: f32,
) -> f32 {
    let max_delta_hz = max_delta_hz.max(1.0);
    let mut smoothness = Vec::new();
    for (get_formant, present) in [
        (
            |f: &SourceFilterFrame| f.filter.f1.as_ref().map(|x| x.frequency_hz),
            frame.filter.f1.is_some(),
        ),
        (
            |f: &SourceFilterFrame| f.filter.f2.as_ref().map(|x| x.frequency_hz),
            frame.filter.f2.is_some(),
        ),
        (
            |f: &SourceFilterFrame| f.filter.f3.as_ref().map(|x| x.frequency_hz),
            frame.filter.f3.is_some(),
        ),
    ] {
        if !present {
            continue;
        }
        if let (Some(cur), Some(prev)) = (get_formant(frame), prev_frame.and_then(get_formant)) {
            smoothness.push(1.0 - ((cur - prev).abs() / max_delta_hz).clamp(0.0, 1.0));
        }
        if let (Some(cur), Some(next)) = (get_formant(frame), next_frame.and_then(get_formant)) {
            smoothness.push(1.0 - ((cur - next).abs() / max_delta_hz).clamp(0.0, 1.0));
        }
    }
    if smoothness.is_empty() {
        0.5
    } else {
        (smoothness.iter().sum::<f32>() / smoothness.len() as f32).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice::tract::{FormantEstimation, GlottalSourceEstimate, NoiseEstimate, VoicingEstimate};

    fn frame(f1: f32, f2: f32, f3: f32, confidence: f32) -> SourceFilterFrame {
        SourceFilterFrame {
            frame_start_ms: 0,
            frame_end_ms: 10,
            voicing: VoicingEstimate::default(),
            source: GlottalSourceEstimate::default(),
            filter: VocalTractFilterEstimate {
                f1: Some(FormantEstimation {
                    frequency_hz: f1,
                    bandwidth_hz: Some(90.0),
                    amplitude_db: -8.0,
                    confidence,
                }),
                f2: Some(FormantEstimation {
                    frequency_hz: f2,
                    bandwidth_hz: Some(120.0),
                    amplitude_db: -10.0,
                    confidence,
                }),
                f3: Some(FormantEstimation {
                    frequency_hz: f3,
                    bandwidth_hz: Some(140.0),
                    amplitude_db: -14.0,
                    confidence,
                }),
                f4: None,
                nasality: None,
            },
            noise: NoiseEstimate::default(),
            confidence,
        }
    }

    #[test]
    fn plausible_spacing_scores_higher_than_implausible_spacing() {
        let cfg = VocalPlausibilityConfig::default();
        let plausible = frame(520.0, 1_500.0, 2_450.0, 0.9);
        let implausible = frame(500.0, 640.0, 760.0, 0.9);

        let plausible_score = assess_vocal_plausibility(&plausible, None, None, &cfg).plausibility;
        let implausible_score = assess_vocal_plausibility(&implausible, None, None, &cfg).plausibility;

        assert!(
            plausible_score > implausible_score,
            "expected plausible spacing to score higher: plausible={plausible_score} implausible={implausible_score}"
        );
    }

    #[test]
    fn smooth_trajectory_scores_higher_than_jumpy_trajectory() {
        let cfg = VocalPlausibilityConfig::default();
        let prev = frame(510.0, 1_470.0, 2_410.0, 0.9);
        let smooth = frame(530.0, 1_490.0, 2_440.0, 0.9);
        let jumpy = frame(940.0, 2_520.0, 3_920.0, 0.9);

        let smooth_score = assess_vocal_plausibility(&smooth, Some(&prev), None, &cfg).trajectory_confidence;
        let jumpy_score = assess_vocal_plausibility(&jumpy, Some(&prev), None, &cfg).trajectory_confidence;

        assert!(
            smooth_score > jumpy_score,
            "expected smooth trajectory to score higher: smooth={smooth_score} jumpy={jumpy_score}"
        );
    }
}
