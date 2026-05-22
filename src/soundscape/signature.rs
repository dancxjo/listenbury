use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::soundscape::SourceId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VoiceSignatureId(pub Uuid);

impl VoiceSignatureId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for VoiceSignatureId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PitchProfile {
    pub median_hz: f32,
    pub range_hz: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FormantProfile {
    pub median_f1_hz: Option<f32>,
    pub median_f2_hz: Option<f32>,
    pub vowel_coloration: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimbreProfile {
    pub spectral_centroid_hz: Option<f32>,
    pub brightness: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProsodyProfile {
    pub pitch_variation: Option<f32>,
    pub pause_ratio: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RateProfile {
    pub syllables_per_second: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceSignatureObservation {
    pub source_id: Option<SourceId>,
    pub embedding: Option<Vec<f32>>,
    pub pitch_profile: Option<PitchProfile>,
    pub formant_profile: Option<FormantProfile>,
    pub timbre_profile: Option<TimbreProfile>,
    pub prosody_profile: Option<ProsodyProfile>,
    pub speaking_rate: Option<RateProfile>,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceSignature {
    pub id: VoiceSignatureId,
    pub source_id: Option<SourceId>,
    pub embedding: Option<Vec<f32>>,
    pub pitch_profile: Option<PitchProfile>,
    pub formant_profile: Option<FormantProfile>,
    pub timbre_profile: Option<TimbreProfile>,
    pub prosody_profile: Option<ProsodyProfile>,
    pub speaking_rate: Option<RateProfile>,
    pub sample_count: usize,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceSignatureMatch {
    pub score: f32,
    pub embedding_similarity: Option<f32>,
    pub pitch_similarity: Option<f32>,
    pub speaking_rate_similarity: Option<f32>,
}

impl VoiceSignature {
    pub fn new(source_id: Option<SourceId>) -> Self {
        Self {
            id: VoiceSignatureId::new(),
            source_id,
            embedding: None,
            pitch_profile: None,
            formant_profile: None,
            timbre_profile: None,
            prosody_profile: None,
            speaking_rate: None,
            sample_count: 0,
            confidence: 0.0,
        }
    }

    pub fn update_with_observation(&mut self, observation: VoiceSignatureObservation) {
        if observation.source_id.is_some() {
            self.source_id = observation.source_id;
        }
        if observation.embedding.is_some() {
            self.embedding = observation.embedding;
        }
        if observation.pitch_profile.is_some() {
            self.pitch_profile = observation.pitch_profile;
        }
        if observation.formant_profile.is_some() {
            self.formant_profile = observation.formant_profile;
        }
        if observation.timbre_profile.is_some() {
            self.timbre_profile = observation.timbre_profile;
        }
        if observation.prosody_profile.is_some() {
            self.prosody_profile = observation.prosody_profile;
        }
        if observation.speaking_rate.is_some() {
            self.speaking_rate = observation.speaking_rate;
        }

        let bounded = observation.confidence.clamp(0.0, 1.0);
        self.sample_count += 1;
        self.confidence = ((self.confidence * (self.sample_count as f32 - 1.0)) + bounded)
            / self.sample_count as f32;
    }

    pub fn compare(&self, other: &Self) -> VoiceSignatureMatch {
        let embedding_similarity = match (&self.embedding, &other.embedding) {
            (Some(left), Some(right)) if left.len() == right.len() && !left.is_empty() => {
                Some(cosine_similarity(left, right))
            }
            _ => None,
        };
        let pitch_similarity = match (&self.pitch_profile, &other.pitch_profile) {
            (Some(left), Some(right)) => Some(similarity_from_distance(
                (left.median_hz - right.median_hz).abs(),
                120.0,
            )),
            _ => None,
        };
        let speaking_rate_similarity = match (&self.speaking_rate, &other.speaking_rate) {
            (Some(left), Some(right)) => Some(similarity_from_distance(
                (left.syllables_per_second - right.syllables_per_second).abs(),
                3.0,
            )),
            _ => None,
        };

        let mut weighted = 0.0;
        let mut total_weight = 0.0;
        if let Some(value) = embedding_similarity {
            weighted += value * 0.45;
            total_weight += 0.45;
        }
        if let Some(value) = pitch_similarity {
            weighted += value * 0.30;
            total_weight += 0.30;
        }
        if let Some(value) = speaking_rate_similarity {
            weighted += value * 0.25;
            total_weight += 0.25;
        }

        let score = if total_weight > 0.0 {
            (weighted / total_weight).clamp(0.0, 1.0)
        } else {
            0.0
        };

        VoiceSignatureMatch {
            score,
            embedding_similarity,
            pitch_similarity,
            speaking_rate_similarity,
        }
    }
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (a, b) in left.iter().zip(right.iter()) {
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }
    ((dot / (left_norm.sqrt() * right_norm.sqrt())) + 1.0) / 2.0
}

fn similarity_from_distance(distance: f32, scale: f32) -> f32 {
    (1.0 - (distance / scale)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_increments_sample_count_and_averages_confidence() {
        let mut signature = VoiceSignature::new(None);

        signature.update_with_observation(VoiceSignatureObservation {
            source_id: None,
            embedding: Some(vec![0.2, 0.3, 0.4]),
            pitch_profile: Some(PitchProfile {
                median_hz: 175.0,
                range_hz: 45.0,
            }),
            formant_profile: None,
            timbre_profile: None,
            prosody_profile: None,
            speaking_rate: Some(RateProfile {
                syllables_per_second: 4.2,
            }),
            confidence: 0.8,
        });

        signature.update_with_observation(VoiceSignatureObservation {
            source_id: None,
            embedding: None,
            pitch_profile: None,
            formant_profile: None,
            timbre_profile: None,
            prosody_profile: None,
            speaking_rate: None,
            confidence: 0.6,
        });

        assert_eq!(signature.sample_count, 2);
        assert!((signature.confidence - 0.7).abs() < 1e-6);
    }

    #[test]
    fn compare_scores_simple_similar_signatures_highly() {
        let mut left = VoiceSignature::new(None);
        left.update_with_observation(VoiceSignatureObservation {
            source_id: None,
            embedding: Some(vec![0.6, 0.4, 0.2]),
            pitch_profile: Some(PitchProfile {
                median_hz: 180.0,
                range_hz: 60.0,
            }),
            formant_profile: None,
            timbre_profile: None,
            prosody_profile: None,
            speaking_rate: Some(RateProfile {
                syllables_per_second: 4.4,
            }),
            confidence: 0.9,
        });

        let mut right = VoiceSignature::new(None);
        right.update_with_observation(VoiceSignatureObservation {
            source_id: None,
            embedding: Some(vec![0.58, 0.42, 0.2]),
            pitch_profile: Some(PitchProfile {
                median_hz: 183.0,
                range_hz: 58.0,
            }),
            formant_profile: None,
            timbre_profile: None,
            prosody_profile: None,
            speaking_rate: Some(RateProfile {
                syllables_per_second: 4.5,
            }),
            confidence: 0.88,
        });

        let result = left.compare(&right);
        assert!(result.score > 0.85);
        assert!(result.embedding_similarity.unwrap_or_default() > 0.95);
    }
}
