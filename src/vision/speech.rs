//! Backend visual-speech feature extraction and evidence mapping.
//!
//! This module is intentionally identity-free. Camera frames are treated as a
//! local, consenting diagnostic input and converted immediately into low-rate
//! mouth-motion features. The ordinary artifact is the derived feature trace,
//! not raw video.

use std::ops::Range;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::audio::hypothesis::{
    HypothesisSource, SpanHypothesis, SpanHypothesisId, SpanHypothesisKind,
};
use crate::audio::lattice::FusionInput;
use crate::vision::VisionFrame;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceScore {
    pub value: f32,
    pub confidence: f32,
}

impl EvidenceScore {
    pub fn new(value: f32, confidence: f32) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    pub fn weighted(self) -> f32 {
        self.value * self.confidence
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VisualSpeechFrame {
    #[serde(with = "range_ms_array")]
    pub time_range_ms: Range<u64>,
    pub mouth_open: EvidenceScore,
    pub lip_closure: EvidenceScore,
    pub lip_rounding: EvidenceScore,
    pub jaw_opening: EvidenceScore,
    pub visibility: EvidenceScore,
    pub provenance: VisualProvenance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VisualProvenance {
    pub source: String,
    pub frame_index: Option<u64>,
    pub capture_time_ms: Option<u64>,
    pub audio_offset_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhonemeClass {
    BilabialStop,
    BilabialNasal,
    Labiodental,
    RoundedVowelOrGlide,
    OpenVowel,
    AlveolarStop,
    VelarStop,
    Unknown,
}

impl PhonemeClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BilabialStop => "bilabial_stop",
            Self::BilabialNasal => "bilabial_nasal",
            Self::Labiodental => "labiodental",
            Self::RoundedVowelOrGlide => "rounded_vowel_or_glide",
            Self::OpenVowel => "open_vowel",
            Self::AlveolarStop => "alveolar_stop",
            Self::VelarStop => "velar_stop",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VowelShape {
    Rounded,
    Open,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisualSpeechClaimKind {
    SupportsPhonemeClass(PhonemeClass),
    ConflictsWithPhonemeClass(PhonemeClass),
    SupportsBoundary,
    ConflictsWithBoundary,
    SupportsVowelShape(VowelShape),
    VisibilityInsufficient,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisualEvidenceStatus {
    Strong,
    Weak,
    AdvisoryOnly,
    Unusable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VisualSpeechClaim {
    pub target_time_ms: u64,
    pub kind: VisualSpeechClaimKind,
    pub source: String,
    pub supports: Vec<String>,
    pub conflicts: Vec<String>,
    pub confidence: f32,
    pub status: VisualEvidenceStatus,
    pub rationale: String,
    pub provenance: VisualProvenance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VisualSpeechTrace {
    pub frames: Vec<VisualSpeechFrame>,
    pub sync: AvSyncConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AvSyncConfig {
    pub manual_video_offset_ms: i64,
    pub max_usable_desync_ms: u64,
    pub confidence_decay_per_ms: f32,
}

impl Default for AvSyncConfig {
    fn default() -> Self {
        Self {
            manual_video_offset_ms: 0,
            max_usable_desync_ms: 80,
            confidence_decay_per_ms: 0.012,
        }
    }
}

/// Extract backend mouth-motion features from a transient RGBA frame.
///
/// The first-pass extractor deliberately uses a conservative lower-face ROI
/// heuristic rather than storing frames or performing identity recognition. A
/// future local landmark backend can replace this implementation while keeping
/// [`VisualSpeechFrame`] stable.
pub fn extract_visual_speech_frame_from_rgba(
    frame: &VisionFrame,
    time_range_ms: Range<u64>,
    mut provenance: VisualProvenance,
) -> Option<VisualSpeechFrame> {
    let width = usize::try_from(frame.width).ok()?;
    let height = usize::try_from(frame.height).ok()?;
    if width == 0
        || height == 0
        || frame.bytes.len() < width.saturating_mul(height).saturating_mul(4)
    {
        return None;
    }

    let x0 = width * 34 / 100;
    let x1 = width * 66 / 100;
    let y0 = height * 56 / 100;
    let y1 = height * 82 / 100;
    if x1 <= x0 || y1 <= y0 {
        return None;
    }

    let mut lumas = Vec::with_capacity((x1 - x0) * (y1 - y0));
    let mut min_luma = 255.0_f32;
    let mut max_luma = 0.0_f32;
    let mut sum = 0.0_f32;
    for y in y0..y1 {
        for x in x0..x1 {
            let i = (y * width + x) * 4;
            let r = frame.bytes[i] as f32;
            let g = frame.bytes[i + 1] as f32;
            let b = frame.bytes[i + 2] as f32;
            let luma = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;
            min_luma = min_luma.min(luma);
            max_luma = max_luma.max(luma);
            sum += luma;
            lumas.push(luma);
        }
    }

    let count = lumas.len() as f32;
    let mean = sum / count.max(1.0);
    let variance = lumas
        .iter()
        .map(|luma| {
            let delta = *luma - mean;
            delta * delta
        })
        .sum::<f32>()
        / count.max(1.0);
    let contrast = (max_luma - min_luma).clamp(0.0, 1.0);
    let threshold = (mean - variance.sqrt() * 0.35).clamp(0.05, 0.65);
    let dark_count = lumas.iter().filter(|luma| **luma <= threshold).count() as f32;
    let dark_ratio = (dark_count / count.max(1.0)).clamp(0.0, 1.0);

    let mid_x0 = width * 43 / 100;
    let mid_x1 = width * 57 / 100;
    let mid_y0 = height * 60 / 100;
    let mid_y1 = height * 78 / 100;
    let central_dark_ratio = dark_ratio_in_region(
        &frame.bytes,
        width,
        mid_x0,
        mid_x1,
        mid_y0,
        mid_y1,
        threshold,
    )?;

    let visibility_value = if contrast < 0.04 {
        0.25
    } else if !(0.08..=0.92).contains(&mean) {
        0.35
    } else {
        (0.55 + contrast * 1.4).clamp(0.0, 1.0)
    };
    let visibility_confidence = (0.45 + contrast).clamp(0.0, 1.0);
    let feature_confidence = (visibility_value * visibility_confidence).clamp(0.0, 1.0);

    let mouth_open = dark_ratio;
    let lip_closure = (1.0 - dark_ratio * 1.35).clamp(0.0, 1.0);
    let lip_rounding =
        (central_dark_ratio * (1.0 - (dark_ratio - central_dark_ratio).abs())).clamp(0.0, 1.0);

    provenance.notes.push(
        "backend lower-face ROI heuristic; raw frame was transient and is not persisted"
            .to_string(),
    );

    Some(VisualSpeechFrame {
        time_range_ms,
        mouth_open: EvidenceScore::new(mouth_open, feature_confidence),
        lip_closure: EvidenceScore::new(lip_closure, feature_confidence),
        lip_rounding: EvidenceScore::new(lip_rounding, feature_confidence * 0.75),
        jaw_opening: EvidenceScore::new(mouth_open, feature_confidence * 0.85),
        visibility: EvidenceScore::new(visibility_value, visibility_confidence),
        provenance,
    })
}

pub fn visual_speech_claims_for_phone_hypotheses(
    frames: &[VisualSpeechFrame],
    phone_hypotheses: &[SpanHypothesis],
    sync: AvSyncConfig,
) -> Vec<VisualSpeechClaim> {
    phone_hypotheses
        .iter()
        .filter_map(|hypothesis| {
            let class = phoneme_class_for_symbol(&hypothesis.label);
            visual_claim_for_candidate(frames, hypothesis, class, sync)
        })
        .collect()
}

pub fn visual_fusion_inputs_for_phone_hypotheses(
    frames: &[VisualSpeechFrame],
    phone_hypotheses: &[SpanHypothesis],
    sync: AvSyncConfig,
) -> Vec<(SpanHypothesisId, FusionInput)> {
    phone_hypotheses
        .iter()
        .filter_map(|hypothesis| {
            let class = phoneme_class_for_symbol(&hypothesis.label);
            let claim = visual_claim_for_candidate(frames, hypothesis, class, sync)?;
            let score = if claim
                .conflicts
                .iter()
                .any(|label| label == &hypothesis.label)
            {
                1.0 - claim.confidence
            } else if claim
                .supports
                .iter()
                .any(|label| label == &hypothesis.label)
            {
                0.5 + claim.confidence * 0.5
            } else {
                0.5
            };
            Some((
                hypothesis.id.clone(),
                FusionInput {
                    visual_speech_evidence: Some(score.clamp(0.0, 1.0)),
                    ..FusionInput::default()
                },
            ))
        })
        .collect()
}

pub fn visual_claim_hypotheses(claims: &[VisualSpeechClaim]) -> Vec<SpanHypothesis> {
    claims
        .iter()
        .map(|claim| {
            let label = match claim.kind {
                VisualSpeechClaimKind::SupportsPhonemeClass(class) => {
                    format!("visual_supports_{}", class.as_str())
                }
                VisualSpeechClaimKind::ConflictsWithPhonemeClass(class) => {
                    format!("visual_conflicts_{}", class.as_str())
                }
                VisualSpeechClaimKind::SupportsBoundary => "visual_supports_boundary".to_string(),
                VisualSpeechClaimKind::ConflictsWithBoundary => {
                    "visual_conflicts_boundary".to_string()
                }
                VisualSpeechClaimKind::SupportsVowelShape(shape) => {
                    format!("visual_supports_{shape:?}").to_ascii_lowercase()
                }
                VisualSpeechClaimKind::VisibilityInsufficient => {
                    "visual_visibility_insufficient".to_string()
                }
            };
            SpanHypothesis::new(
                SpanHypothesisKind::PhoneClassCandidate,
                label,
                claim.target_time_ms,
                claim.target_time_ms.saturating_add(1),
                claim.confidence,
                claim.confidence,
                HypothesisSource::VisualSpeech,
                vec![claim.source.clone()],
                json!(claim),
            )
        })
        .collect()
}

fn visual_claim_for_candidate(
    frames: &[VisualSpeechFrame],
    hypothesis: &SpanHypothesis,
    class: PhonemeClass,
    sync: AvSyncConfig,
) -> Option<VisualSpeechClaim> {
    let window_start = hypothesis.start_ms.saturating_sub(70);
    let window_end = hypothesis.end_ms.saturating_add(70);
    let nearby: Vec<&VisualSpeechFrame> = frames
        .iter()
        .filter(|frame| {
            frame.time_range_ms.start < window_end && frame.time_range_ms.end > window_start
        })
        .collect();
    if nearby.is_empty() {
        return None;
    }

    let best_visibility = nearby
        .iter()
        .map(|frame| frame.visibility.weighted())
        .fold(0.0_f32, f32::max);
    let status = evidence_status(best_visibility, sync);
    let target_time_ms =
        hypothesis.start_ms + hypothesis.end_ms.saturating_sub(hypothesis.start_ms) / 2;
    let provenance = nearby
        .iter()
        .max_by(|a, b| {
            a.visibility
                .weighted()
                .partial_cmp(&b.visibility.weighted())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?
        .provenance
        .clone();

    if matches!(status, VisualEvidenceStatus::Unusable) {
        return Some(VisualSpeechClaim {
            target_time_ms,
            kind: VisualSpeechClaimKind::VisibilityInsufficient,
            source: "visual_speech.visibility".to_string(),
            supports: Vec::new(),
            conflicts: vec![hypothesis.label.clone()],
            confidence: 0.0,
            status,
            rationale: "Mouth visibility or audiovisual timing was insufficient.".to_string(),
            provenance,
        });
    }

    let max_closure = nearby
        .iter()
        .map(|frame| frame.lip_closure.weighted())
        .fold(0.0_f32, f32::max);
    let max_rounding = nearby
        .iter()
        .map(|frame| frame.lip_rounding.weighted())
        .fold(0.0_f32, f32::max);
    let max_open = nearby
        .iter()
        .map(|frame| {
            frame
                .mouth_open
                .weighted()
                .max(frame.jaw_opening.weighted())
        })
        .fold(0.0_f32, f32::max);
    let timing_weight = timing_confidence_weight(sync, provenance.audio_offset_ms);
    let status_weight = match status {
        VisualEvidenceStatus::Strong => 1.0,
        VisualEvidenceStatus::Weak => 0.7,
        VisualEvidenceStatus::AdvisoryOnly => 0.35,
        VisualEvidenceStatus::Unusable => 0.0,
    };

    match class {
        PhonemeClass::BilabialStop | PhonemeClass::BilabialNasal => {
            let confidence =
                ((1.0 - max_closure) * best_visibility * timing_weight * status_weight)
                    .clamp(0.0, 0.85);
            if max_closure < 0.35 {
                Some(VisualSpeechClaim {
                    target_time_ms,
                    kind: VisualSpeechClaimKind::ConflictsWithPhonemeClass(class),
                    source: "visual_speech.lip_closure".to_string(),
                    supports: Vec::new(),
                    conflicts: vec![hypothesis.label.clone()],
                    confidence,
                    status,
                    rationale:
                        "No visible lip closure near expected bilabial closure/release window."
                            .to_string(),
                    provenance,
                })
            } else {
                Some(VisualSpeechClaim {
                    target_time_ms,
                    kind: VisualSpeechClaimKind::SupportsPhonemeClass(class),
                    source: "visual_speech.lip_closure".to_string(),
                    supports: vec![hypothesis.label.clone()],
                    conflicts: Vec::new(),
                    confidence: (max_closure * timing_weight * status_weight).clamp(0.0, 0.85),
                    status,
                    rationale: "Visible lip closure near expected bilabial consonant window."
                        .to_string(),
                    provenance,
                })
            }
        }
        PhonemeClass::RoundedVowelOrGlide if max_rounding > 0.35 => Some(VisualSpeechClaim {
            target_time_ms,
            kind: VisualSpeechClaimKind::SupportsPhonemeClass(class),
            source: "visual_speech.lip_rounding".to_string(),
            supports: vec![hypothesis.label.clone()],
            conflicts: Vec::new(),
            confidence: (max_rounding * timing_weight * status_weight).clamp(0.0, 0.8),
            status,
            rationale:
                "Lip rounding/protrusion evidence overlaps the rounded vowel/glide candidate."
                    .to_string(),
            provenance,
        }),
        PhonemeClass::OpenVowel if max_open > 0.45 => Some(VisualSpeechClaim {
            target_time_ms,
            kind: VisualSpeechClaimKind::SupportsVowelShape(VowelShape::Open),
            source: "visual_speech.mouth_open".to_string(),
            supports: vec![hypothesis.label.clone()],
            conflicts: Vec::new(),
            confidence: (max_open * timing_weight * status_weight).clamp(0.0, 0.8),
            status,
            rationale: "Wide mouth/jaw opening overlaps the open vowel candidate.".to_string(),
            provenance,
        }),
        PhonemeClass::AlveolarStop if max_closure < 0.35 => Some(VisualSpeechClaim {
            target_time_ms,
            kind: VisualSpeechClaimKind::SupportsPhonemeClass(class),
            source: "visual_speech.no_lip_closure".to_string(),
            supports: vec![hypothesis.label.clone()],
            conflicts: Vec::new(),
            confidence: ((1.0 - max_closure) * best_visibility * timing_weight * status_weight)
                .clamp(0.0, 0.65),
            status,
            rationale: "Absence of lip closure is compatible with a non-bilabial stop candidate."
                .to_string(),
            provenance,
        }),
        _ => None,
    }
}

fn evidence_status(visibility: f32, sync: AvSyncConfig) -> VisualEvidenceStatus {
    let desync = sync.manual_video_offset_ms.unsigned_abs();
    if visibility < 0.15 || desync > sync.max_usable_desync_ms.saturating_mul(2) {
        VisualEvidenceStatus::Unusable
    } else if visibility < 0.35 || desync > sync.max_usable_desync_ms {
        VisualEvidenceStatus::AdvisoryOnly
    } else if visibility < 0.6 {
        VisualEvidenceStatus::Weak
    } else {
        VisualEvidenceStatus::Strong
    }
}

fn timing_confidence_weight(sync: AvSyncConfig, frame_audio_offset_ms: Option<i64>) -> f32 {
    let offset = frame_audio_offset_ms.unwrap_or(sync.manual_video_offset_ms);
    let desync = offset.unsigned_abs();
    if desync > sync.max_usable_desync_ms.saturating_mul(2) {
        return 0.0;
    }
    (1.0 - desync as f32 * sync.confidence_decay_per_ms).clamp(0.0, 1.0)
}

fn phoneme_class_for_symbol(symbol: &str) -> PhonemeClass {
    let upper = symbol
        .trim_matches(|c: char| c == '/' || c.is_ascii_digit())
        .to_ascii_uppercase();
    match upper.as_str() {
        "P" | "B" => PhonemeClass::BilabialStop,
        "M" => PhonemeClass::BilabialNasal,
        "F" | "V" => PhonemeClass::Labiodental,
        "UW" | "UW0" | "UW1" | "UW2" | "UH" | "OW" | "OW0" | "OW1" | "OW2" | "W" => {
            PhonemeClass::RoundedVowelOrGlide
        }
        "AA" | "AA0" | "AA1" | "AA2" | "AE" | "AE0" | "AE1" | "AE2" => PhonemeClass::OpenVowel,
        "T" | "D" => PhonemeClass::AlveolarStop,
        "K" | "G" => PhonemeClass::VelarStop,
        _ => PhonemeClass::Unknown,
    }
}

fn dark_ratio_in_region(
    bytes: &[u8],
    width: usize,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    threshold: f32,
) -> Option<f32> {
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let mut total = 0usize;
    let mut dark = 0usize;
    for y in y0..y1 {
        for x in x0..x1 {
            let i = (y * width + x) * 4;
            if i + 2 >= bytes.len() {
                return None;
            }
            let r = bytes[i] as f32;
            let g = bytes[i + 1] as f32;
            let b = bytes[i + 2] as f32;
            let luma = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;
            total += 1;
            if luma <= threshold {
                dark += 1;
            }
        }
    }
    Some(dark as f32 / total.max(1) as f32)
}

mod range_ms_array {
    use std::ops::Range;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(range: &Range<u64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        [range.start, range.end].serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Range<u64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let [start, end] = <[u64; 2]>::deserialize(deserializer)?;
        Ok(start..end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::hypothesis::{HypothesisSource, SpanHypothesisKind};
    use crate::time::ExactTimestamp;

    fn provenance() -> VisualProvenance {
        VisualProvenance {
            source: "test".to_string(),
            frame_index: Some(1),
            capture_time_ms: Some(123),
            audio_offset_ms: Some(0),
            notes: Vec::new(),
        }
    }

    fn visual_frame(start: u64, closure: f32, visibility: f32) -> VisualSpeechFrame {
        VisualSpeechFrame {
            time_range_ms: start..start + 33,
            mouth_open: EvidenceScore::new(1.0 - closure, 0.8),
            lip_closure: EvidenceScore::new(closure, 0.9),
            lip_rounding: EvidenceScore::new(0.1, 0.5),
            jaw_opening: EvidenceScore::new(1.0 - closure, 0.8),
            visibility: EvidenceScore::new(visibility, 0.95),
            provenance: provenance(),
        }
    }

    fn phone(label: &str, start: u64, end: u64) -> SpanHypothesis {
        SpanHypothesis::new(
            SpanHypothesisKind::PronunciationAlignment,
            label,
            start,
            end,
            0.5,
            0.5,
            HypothesisSource::ViterbiAlignment,
            vec![],
            json!(null),
        )
    }

    #[test]
    fn serializes_time_range_as_pair_for_json_traces() {
        let json = serde_json::to_string(&visual_frame(1200, 0.91, 0.95)).unwrap();
        assert!(json.contains("\"timeRangeMs\":[1200,1233]"));
        assert!(!json.contains("bytes"));
    }

    #[test]
    fn backend_rgba_extractor_returns_derived_features_without_raw_video() {
        let width = 40;
        let height = 40;
        let mut bytes = vec![180_u8; width * height * 4];
        for px in bytes.chunks_exact_mut(4) {
            px[3] = 255;
        }
        for y in 24..30 {
            for x in 16..24 {
                let i = (y * width + x) * 4;
                bytes[i] = 10;
                bytes[i + 1] = 10;
                bytes[i + 2] = 10;
            }
        }
        let frame = VisionFrame {
            captured_at: ExactTimestamp::now(),
            width: width as u32,
            height: height as u32,
            bytes,
        };
        let features =
            extract_visual_speech_frame_from_rgba(&frame, 0..33, provenance()).expect("features");
        assert!(features.mouth_open.value > 0.0);
        let json = serde_json::to_value(&features).unwrap();
        assert!(json.get("bytes").is_none());
    }

    #[test]
    fn did_vs_deep_without_final_lip_closure_weakens_p_and_supports_d() {
        let frames = vec![visual_frame(800, 0.05, 0.95)];
        let deep_p = phone("P", 820, 860);
        let did_d = phone("D", 820, 860);
        let claims = visual_speech_claims_for_phone_hypotheses(
            &frames,
            &[deep_p.clone(), did_d.clone()],
            AvSyncConfig::default(),
        );

        assert!(claims.iter().any(|claim| {
            matches!(
                claim.kind,
                VisualSpeechClaimKind::ConflictsWithPhonemeClass(PhonemeClass::BilabialStop)
            ) && claim.conflicts == vec!["P"]
        }));
        assert!(claims.iter().any(|claim| {
            matches!(
                claim.kind,
                VisualSpeechClaimKind::SupportsPhonemeClass(PhonemeClass::AlveolarStop)
            ) && claim.supports == vec!["D"]
        }));

        let visual_inputs = visual_fusion_inputs_for_phone_hypotheses(
            &frames,
            &[deep_p, did_d],
            AvSyncConfig::default(),
        );
        assert_eq!(visual_inputs.len(), 2);
        assert!(visual_inputs[0].1.visual_speech_evidence.unwrap() < 0.5);
        assert!(visual_inputs[1].1.visual_speech_evidence.unwrap() > 0.5);
    }

    #[test]
    fn desync_downweights_visual_evidence_to_unusable() {
        let sync = AvSyncConfig {
            manual_video_offset_ms: 250,
            ..Default::default()
        };
        let claims = visual_speech_claims_for_phone_hypotheses(
            &[visual_frame(800, 0.05, 0.95)],
            &[phone("P", 820, 860)],
            sync,
        );
        assert_eq!(claims[0].status, VisualEvidenceStatus::Unusable);
        assert_eq!(claims[0].confidence, 0.0);
    }

    #[test]
    fn did_deep_fixture_contains_derived_visual_trace_only() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("examples/browser-transcript-player/fixtures/visual-speech-did-deep.json");
        let fixture: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).expect("fixture")).expect("json");
        assert_eq!(fixture["rawVideoPersistedByDefault"], false);
        assert!(fixture["visualSpeechFrames"].as_array().unwrap().len() >= 2);
        assert!(fixture.to_string().contains("lipClosure"));
        assert!(!fixture.to_string().contains("rawVideoBytes"));
    }
}
