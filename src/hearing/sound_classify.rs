#[derive(Debug, Clone, Copy)]
pub struct SoundFeatures {
    pub duration_ms: u64,
    pub centroid_hz: f32,
    pub repetition_hz: f32,
    pub salience: f32,
    pub temporal_density: f32,
}

#[derive(Debug, Clone)]
pub struct SoundClassification {
    pub label: Option<String>,
    pub description: String,
    pub confidence: f32,
}

pub fn describe_sound(features: SoundFeatures) -> SoundClassification {
    let salience = features.salience.clamp(0.0, 1.0);
    if features.duration_ms >= 600
        && features.centroid_hz <= 320.0
        && features.temporal_density < 0.12
    {
        return SoundClassification {
            label: Some("sustained_hum".to_string()),
            description: "I heard sustained fan-like noise.".to_string(),
            confidence: (0.55 + salience * 0.4).clamp(0.0, 1.0),
        };
    }

    if features.repetition_hz >= 3.0
        && features.repetition_hz <= 18.0
        && features.duration_ms <= 220
        && features.temporal_density >= 0.10
    {
        return SoundClassification {
            label: Some("repetitive_ticks".to_string()),
            description: "I heard repetitive keyboard-like ticks.".to_string(),
            confidence: (0.50 + salience * 0.45).clamp(0.0, 1.0),
        };
    }

    if features.duration_ms <= 160 && features.centroid_hz >= 1_800.0 {
        return SoundClassification {
            label: Some("chirp".to_string()),
            description: "I heard a short high-pitched chirp.".to_string(),
            confidence: (0.45 + salience * 0.5).clamp(0.0, 1.0),
        };
    }

    if features.duration_ms <= 180 {
        return SoundClassification {
            label: Some("impact".to_string()),
            description: "I heard a sharp impact sound.".to_string(),
            confidence: (0.40 + salience * 0.5).clamp(0.0, 1.0),
        };
    }

    SoundClassification {
        label: None,
        description: "I heard non-speech environmental noise.".to_string(),
        confidence: (0.25 + salience * 0.4).clamp(0.0, 1.0),
    }
}
