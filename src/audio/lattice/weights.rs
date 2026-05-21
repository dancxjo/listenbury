use serde::{Deserialize, Serialize};

/// Per-signal weights used by [`FusionInput::weighted_confidence_with`] and
/// the blending factor used by [`fuse_hypotheses`].
///
/// Each field controls how strongly the corresponding evidence signal
/// contributes relative to the others when computing a weighted average.
/// A higher weight means that signal dominates the final score.
///
/// # Design rationale
///
/// | Field                        | Default | Why                                                                  |
/// |------------------------------|---------|----------------------------------------------------------------------|
/// | `asr_confidence`             | 3.0     | Whisper ASR is the most direct and reliable recogniser signal.       |
/// | `energy_alignment_quality`   | 1.5     | Energy-landmark snapping is fast and largely signal-free.            |
/// | `timing_coherence`           | 1.25    | Temporal ordering is a strong prior; impossible orderings are penalised. |
/// | `phone_segmentation_agreement` | 1.0   | Phone-level agreement improves precision but needs a pronunciation.  |
/// | `pronunciation_fit`          | 1.0     | Complements segmentation; same ballpark reliability.                 |
/// | `mechanical_recognizer_score` | 1.0    | DTW/Viterbi scores are useful but noisier than Whisper.              |
/// | `visual_speech_evidence`     | 0.9     | Lip-reading adds signal but is optional/not always present.          |
/// | `spectral_evidence`          | 0.75    | Spectral shape is auxiliary; useful but not decisive on its own.     |
/// | `prosody_consistency`        | 0.5     | Prosodic cues are soft—agree-if-present, but rarely flip a winner.   |
/// | `external_evidence_blend`    | 3.0     | How aggressively fused external evidence overrides base confidence.  |
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusionWeights {
    /// Weight for ASR (e.g. Whisper) confidence signal.
    /// Highest default weight: this is the most direct recogniser signal.
    pub asr_confidence: f32,
    /// Weight for energy-landmark snapping quality.
    /// Rewards hypotheses whose boundaries align with acoustic energy peaks.
    pub energy_alignment_quality: f32,
    /// Weight for agreement between phone segmentation and expected pronunciation.
    pub phone_segmentation_agreement: f32,
    /// Weight for pronunciation fit score.
    pub pronunciation_fit: f32,
    /// Weight for spectral evidence (e.g. formant shape matching).
    pub spectral_evidence: f32,
    /// Weight for prosody consistency (pitch contour, duration).
    /// Low default: prosodic cues rarely flip a winner on their own.
    pub prosody_consistency: f32,
    /// Weight for timing coherence (penalises temporally impossible orderings).
    pub timing_coherence: f32,
    /// Weight for mechanical recogniser aggregate score (DTW/Viterbi).
    pub mechanical_recognizer_score: f32,
    /// Weight for time-synchronised visual speech evidence (lip-reading backend).
    pub visual_speech_evidence: f32,
    /// Blending factor for external fused evidence vs. a hypothesis's own confidence.
    ///
    /// Used in `fuse_hypotheses`: `(own_confidence + external * blend) / (1 + blend)`.
    /// A higher value causes the fused external evidence to dominate the base score.
    pub external_evidence_blend: f32,
}

impl Default for FusionWeights {
    /// Returns the balanced default weights that preserve the original heuristic behaviour.
    fn default() -> Self {
        FusionProfile::Default.into()
    }
}

/// Named weighting profiles for the speech hypothesis fusion pipeline.
///
/// Each variant maps to a [`FusionWeights`] configuration tuned for a
/// particular operating scenario. Profiles are converted to [`FusionWeights`]
/// via `FusionWeights::from(profile)` and have no runtime overhead beyond a
/// single struct copy at construction time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionProfile {
    /// Balanced default: exactly matches the original heuristic numeric weights.
    ///
    /// Suitable as a general-purpose baseline when no specific operating mode
    /// is required.
    Default,

    /// Conservative: strongly amplifies the highest-reliability (ASR) signal
    /// and suppresses speculative cues such as prosody and spectral shape.
    ///
    /// Use when false positives are more costly than false negatives—for example
    /// in transcription-quality pipelines where an incorrect word is worse than
    /// a missed one.
    Conservative,

    /// Realtime: de-emphasises latency-heavy ASR and instead privileges fast
    /// local signals (energy alignment, timing coherence, spectral shape).
    ///
    /// Use when end-to-end latency matters more than peak accuracy, such as in
    /// live captions or voice-command gating.
    Realtime,

    /// AcousticHeavy: boosts spectral, energy, and mechanical-recogniser evidence
    /// relative to ASR. Useful when ASR confidence scores are unreliable or
    /// unavailable (e.g. a degraded-channel or far-field scenario).
    AcousticHeavy,

    /// VisualAssist: amplifies visual speech evidence for multi-modal scenarios
    /// where lip-reading supplements audio (e.g. noisy environments or
    /// accessibility-focused deployments).
    VisualAssist,
}

impl From<FusionProfile> for FusionWeights {
    fn from(profile: FusionProfile) -> Self {
        match profile {
            FusionProfile::Default => Self {
                asr_confidence: 3.0,
                energy_alignment_quality: 1.5,
                phone_segmentation_agreement: 1.0,
                pronunciation_fit: 1.0,
                spectral_evidence: 0.75,
                prosody_consistency: 0.5,
                timing_coherence: 1.25,
                mechanical_recognizer_score: 1.0,
                visual_speech_evidence: 0.9,
                external_evidence_blend: 3.0,
            },
            FusionProfile::Conservative => Self {
                // Strongly favour ASR; suppress speculative acoustic cues.
                asr_confidence: 5.0,
                energy_alignment_quality: 1.0,
                phone_segmentation_agreement: 0.75,
                pronunciation_fit: 0.75,
                spectral_evidence: 0.5,
                prosody_consistency: 0.25,
                timing_coherence: 1.0,
                mechanical_recognizer_score: 0.75,
                visual_speech_evidence: 0.5,
                external_evidence_blend: 4.0,
            },
            FusionProfile::Realtime => Self {
                // De-emphasise latency-heavy ASR; prefer fast local signals.
                asr_confidence: 1.5,
                energy_alignment_quality: 2.5,
                phone_segmentation_agreement: 0.75,
                pronunciation_fit: 0.75,
                spectral_evidence: 1.5,
                prosody_consistency: 0.5,
                timing_coherence: 2.0,
                mechanical_recognizer_score: 1.5,
                visual_speech_evidence: 0.5,
                external_evidence_blend: 2.0,
            },
            FusionProfile::AcousticHeavy => Self {
                // Boost acoustic signals; use ASR only as a soft prior.
                asr_confidence: 1.0,
                energy_alignment_quality: 3.0,
                phone_segmentation_agreement: 2.0,
                pronunciation_fit: 1.5,
                spectral_evidence: 3.0,
                prosody_consistency: 1.0,
                timing_coherence: 1.5,
                mechanical_recognizer_score: 3.0,
                visual_speech_evidence: 0.5,
                external_evidence_blend: 2.5,
            },
            FusionProfile::VisualAssist => Self {
                // Amplify lip-reading evidence for multi-modal deployments.
                asr_confidence: 2.0,
                energy_alignment_quality: 1.0,
                phone_segmentation_agreement: 0.75,
                pronunciation_fit: 1.0,
                spectral_evidence: 0.5,
                prosody_consistency: 0.5,
                timing_coherence: 1.0,
                mechanical_recognizer_score: 0.75,
                visual_speech_evidence: 4.0,
                external_evidence_blend: 3.0,
            },
        }
    }
}
