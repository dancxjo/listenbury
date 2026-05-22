use serde::{Deserialize, Serialize};

use crate::audio::acoustic::EnergyLandmarks;
use crate::audio::features::{AcousticFeatureFrame, AcousticFeatureStream};
use crate::audio::speech_likelihood::SpeechLikelihood;
use crate::segmentation::nuclei::{NucleusEvidence, VowelNucleusCandidate};
use crate::segmentation::syllable_regions::SyllableIsland;
use crate::segmentation::word_regions::{rank_word_region_hypotheses, WordRegionConfig};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryKind {
    SpeechRegion,
    SyllableIsland,
    PossibleWordRegion,
    NoiseEvent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryEvidence {
    EnergyRise,
    EnergyFall,
    VoicingOnset,
    VoicingOffset,
    FormantOnset,
    FormantOffset,
    VowelNucleus,
    SpectralFluxPeak,
    SilenceGap,
    NoiseRejected,
    MatchesExpectedPhoneShape,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoundaryHypothesis {
    pub start_time: f32,
    pub end_time: f32,
    pub kind: BoundaryKind,
    pub confidence: f32,
    pub evidence: Vec<BoundaryEvidence>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoundaryHypothesisConfig {
    pub min_speech_confidence: f32,
    pub max_noise_confidence: f32,
    pub min_voiced_confidence: f32,
    pub min_formant_confidence: f32,
    pub min_flux_for_peak: f32,
    pub word_regions: WordRegionConfig,
}

impl Default for BoundaryHypothesisConfig {
    fn default() -> Self {
        Self {
            min_speech_confidence: 0.28,
            max_noise_confidence: 0.72,
            min_voiced_confidence: 0.30,
            min_formant_confidence: 0.25,
            min_flux_for_peak: 0.20,
            word_regions: WordRegionConfig::default(),
        }
    }
}

pub fn emit_ranked_boundary_hypotheses(
    landmarks: &EnergyLandmarks,
    likelihoods: &[SpeechLikelihood],
    features: &AcousticFeatureStream,
    nuclei: &[VowelNucleusCandidate],
    islands: &[SyllableIsland],
    config: &BoundaryHypothesisConfig,
) -> Vec<BoundaryHypothesis> {
    let mut hypotheses = generate_landmark_hypotheses(landmarks, Some(features));

    hypotheses.extend(speech_regions_from_likelihoods(
        likelihoods,
        features,
        config,
    ));
    hypotheses.extend(
        islands
            .iter()
            .map(|island| BoundaryHypothesis {
                start_time: island.start_time,
                end_time: island.end_time,
                kind: BoundaryKind::SyllableIsland,
                confidence: island.confidence.clamp(0.0, 1.0),
                evidence: island_evidence(island, config),
            })
            .collect::<Vec<_>>(),
    );

    hypotheses.extend(rank_word_region_hypotheses(
        islands,
        likelihoods,
        features,
        &config.word_regions,
    ));

    if nuclei.is_empty() {
        hypotheses.push(BoundaryHypothesis {
            start_time: 0.0,
            end_time: 0.0,
            kind: BoundaryKind::NoiseEvent,
            confidence: 0.35,
            evidence: vec![BoundaryEvidence::NoiseRejected],
        });
    }

    rank_hypotheses(hypotheses)
}

pub fn generate_landmark_hypotheses(
    landmarks: &EnergyLandmarks,
    features: Option<&AcousticFeatureStream>,
) -> Vec<BoundaryHypothesis> {
    let mut hypotheses = Vec::new();

    for &onset in &landmarks.onsets {
        let end = landmarks
            .offsets
            .iter()
            .copied()
            .find(|offset| *offset >= onset)
            .unwrap_or(onset);
        hypotheses.push(BoundaryHypothesis {
            start_time: onset as f32,
            end_time: end as f32,
            kind: BoundaryKind::SpeechRegion,
            confidence: span_energy_confidence(features, onset, end),
            evidence: vec![BoundaryEvidence::EnergyRise, BoundaryEvidence::EnergyFall],
        });
    }

    for &offset in &landmarks.offsets {
        hypotheses.push(BoundaryHypothesis {
            start_time: offset as f32,
            end_time: offset as f32,
            kind: BoundaryKind::SpeechRegion,
            confidence: span_energy_confidence(features, offset, offset),
            evidence: vec![BoundaryEvidence::EnergyFall],
        });
    }

    for silence in &landmarks.silences {
        let duration_ms = silence.end_ms.saturating_sub(silence.start_ms);
        hypotheses.push(BoundaryHypothesis {
            start_time: silence.start_ms as f32,
            end_time: silence.end_ms as f32,
            kind: BoundaryKind::NoiseEvent,
            confidence: silence_confidence(duration_ms),
            evidence: vec![
                BoundaryEvidence::SilenceGap,
                BoundaryEvidence::NoiseRejected,
            ],
        });
    }

    for &valley in &landmarks.valleys {
        hypotheses.push(BoundaryHypothesis {
            start_time: valley as f32,
            end_time: valley as f32,
            kind: BoundaryKind::NoiseEvent,
            confidence: 0.30,
            evidence: vec![
                BoundaryEvidence::EnergyFall,
                BoundaryEvidence::EnergyRise,
                BoundaryEvidence::NoiseRejected,
            ],
        });
    }

    rank_hypotheses(hypotheses)
}

fn speech_regions_from_likelihoods(
    likelihoods: &[SpeechLikelihood],
    features: &AcousticFeatureStream,
    config: &BoundaryHypothesisConfig,
) -> Vec<BoundaryHypothesis> {
    let frame_count = likelihoods.len().min(features.frames.len());
    if frame_count == 0 {
        return Vec::new();
    }

    let mut regions = Vec::new();
    let mut region_start: Option<usize> = None;

    for idx in 0..=frame_count {
        let is_speech = if idx < frame_count {
            let l = &likelihoods[idx];
            l.speech_confidence >= config.min_speech_confidence
                && l.noise_confidence <= config.max_noise_confidence
        } else {
            false
        };

        match (region_start, is_speech) {
            (None, true) => region_start = Some(idx),
            (Some(start), false) => {
                let end = idx.saturating_sub(1);
                let l_slice = &likelihoods[start..=end];
                let f_slice = &features.frames[start..=end];
                regions.push(build_speech_region(l_slice, f_slice, config));
                region_start = None;
            }
            _ => {}
        }
    }

    regions
}

fn build_speech_region(
    likelihoods: &[SpeechLikelihood],
    frames: &[AcousticFeatureFrame],
    config: &BoundaryHypothesisConfig,
) -> BoundaryHypothesis {
    let start_time = frames
        .first()
        .map(|f| f.frame_start_ms as f32)
        .unwrap_or(0.0);
    let end_time = frames
        .last()
        .map(|f| f.frame_end_ms as f32)
        .unwrap_or(start_time);

    let mut evidence = vec![BoundaryEvidence::EnergyRise, BoundaryEvidence::EnergyFall];
    if likelihoods
        .iter()
        .any(|l| l.voiced_confidence >= config.min_voiced_confidence)
    {
        evidence.push(BoundaryEvidence::VoicingOnset);
        evidence.push(BoundaryEvidence::VoicingOffset);
    }
    if likelihoods
        .iter()
        .any(|l| l.formant_confidence >= config.min_formant_confidence)
    {
        evidence.push(BoundaryEvidence::FormantOnset);
        evidence.push(BoundaryEvidence::FormantOffset);
    }
    if frames
        .iter()
        .any(|frame| frame.spectral_flux >= config.min_flux_for_peak)
    {
        evidence.push(BoundaryEvidence::SpectralFluxPeak);
    }

    let confidence = likelihoods
        .iter()
        .map(|l| l.speech_confidence)
        .fold(0.0_f32, f32::max)
        .clamp(0.0, 1.0);

    BoundaryHypothesis {
        start_time,
        end_time,
        kind: BoundaryKind::SpeechRegion,
        confidence,
        evidence,
    }
}

fn island_evidence(
    island: &SyllableIsland,
    config: &BoundaryHypothesisConfig,
) -> Vec<BoundaryEvidence> {
    let mut evidence = vec![BoundaryEvidence::VowelNucleus];

    if island
        .nucleus
        .evidence
        .iter()
        .any(|item| matches!(item, NucleusEvidence::Voiced))
        || island.confidence >= config.min_voiced_confidence
    {
        evidence.push(BoundaryEvidence::VoicingOnset);
        evidence.push(BoundaryEvidence::VoicingOffset);
    }

    if island
        .nucleus
        .evidence
        .iter()
        .any(|item| matches!(item, NucleusEvidence::Formant))
    {
        evidence.push(BoundaryEvidence::FormantOnset);
        evidence.push(BoundaryEvidence::FormantOffset);
    }

    evidence.push(BoundaryEvidence::MatchesExpectedPhoneShape);
    evidence
}

fn rank_hypotheses(mut hypotheses: Vec<BoundaryHypothesis>) -> Vec<BoundaryHypothesis> {
    hypotheses.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
    hypotheses
}

fn span_energy_confidence(
    features: Option<&AcousticFeatureStream>,
    start_ms: u64,
    end_ms: u64,
) -> f32 {
    let Some(features) = features else {
        return 0.50;
    };

    let mut count = 0usize;
    let mut total = 0.0_f32;
    for frame in &features.frames {
        if frame.frame_end_ms < start_ms || frame.frame_start_ms > end_ms {
            continue;
        }
        total += (frame.rms_energy * 8.0).clamp(0.0, 1.0);
        count += 1;
    }

    if count == 0 {
        0.50
    } else {
        (total / count as f32).clamp(0.0, 1.0)
    }
}

fn silence_confidence(duration_ms: u64) -> f32 {
    if duration_ms >= 300 {
        0.85
    } else if duration_ms >= 120 {
        0.68
    } else {
        0.45
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::acoustic::{analyze_mono_samples, EnergySilence};
    use crate::audio::features::build_feature_stream;
    use crate::audio::speech_likelihood::{build_speech_likelihood_stream, SpeechLikelihoodConfig};
    use crate::segmentation::nuclei::{detect_nuclei, NucleusDetectionConfig};
    use crate::segmentation::syllable_regions::{
        extract_syllable_islands, SyllableExpansionConfig,
    };
    use crate::voice::tract::source_filter_track_from_acoustic_full;

    fn empty_landmarks() -> EnergyLandmarks {
        EnergyLandmarks {
            onsets: Vec::new(),
            offsets: Vec::new(),
            valleys: Vec::new(),
            silences: Vec::new(),
            peaks: Vec::new(),
        }
    }

    fn vowelish_harmonic(sample_rate: u32, seconds: f32) -> Vec<f32> {
        let count = (sample_rate as f32 * seconds).round() as usize;
        (0..count)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                let f0 = 145.0_f32;
                (2.0 * std::f32::consts::PI * f0 * t).sin() * 0.22
                    + (2.0 * std::f32::consts::PI * f0 * 2.0 * t).sin() * 0.11
                    + (2.0 * std::f32::consts::PI * f0 * 3.0 * t).sin() * 0.06
            })
            .collect()
    }

    fn click_burst(sample_rate: u32, seconds: f32) -> Vec<f32> {
        let count = (sample_rate as f32 * seconds).round() as usize;
        let mut samples = vec![0.0_f32; count];
        let step = (sample_rate / 30) as usize;
        for idx in (0..count).step_by(step.max(1)) {
            samples[idx] = 0.85;
        }
        samples
    }

    fn pipeline(
        samples: &[f32],
        sample_rate: u32,
    ) -> (
        EnergyLandmarks,
        AcousticFeatureStream,
        Vec<SpeechLikelihood>,
        Vec<VowelNucleusCandidate>,
        Vec<SyllableIsland>,
    ) {
        let analysis = analyze_mono_samples(samples, sample_rate);
        let features = build_feature_stream(
            samples,
            sample_rate,
            &analysis.energy_envelope,
            analysis.spectrogram.levels.first(),
        );
        let source_filter = source_filter_track_from_acoustic_full(&analysis, samples);
        let likelihoods = build_speech_likelihood_stream(
            &features,
            Some(&source_filter),
            &SpeechLikelihoodConfig::default(),
        );
        let nuclei = detect_nuclei(&likelihoods, &features, &NucleusDetectionConfig::default());
        let islands = extract_syllable_islands(
            &nuclei,
            &likelihoods,
            &features,
            &SyllableExpansionConfig::default(),
        );

        (
            analysis.energy_landmarks,
            features,
            likelihoods,
            nuclei,
            islands,
        )
    }

    #[test]
    fn landmark_noise_regions_include_noise_event() {
        let mut landmarks = empty_landmarks();
        landmarks.silences.push(EnergySilence {
            start_ms: 120,
            end_ms: 360,
        });

        let hypotheses = generate_landmark_hypotheses(&landmarks, None);
        assert!(hypotheses.iter().any(|candidate| {
            candidate.kind == BoundaryKind::NoiseEvent
                && candidate
                    .evidence
                    .contains(&BoundaryEvidence::NoiseRejected)
        }));
    }

    #[test]
    fn speechlike_signal_emits_ranked_speech_and_word_candidates() {
        let sample_rate = 16_000_u32;
        let samples = vowelish_harmonic(sample_rate, 0.40);
        let (landmarks, features, likelihoods, nuclei, islands) = pipeline(&samples, sample_rate);

        let hypotheses = emit_ranked_boundary_hypotheses(
            &landmarks,
            &likelihoods,
            &features,
            &nuclei,
            &islands,
            &BoundaryHypothesisConfig::default(),
        );

        assert!(!hypotheses.is_empty(), "expected ranked hypotheses");
        assert!(hypotheses
            .iter()
            .any(|candidate| candidate.kind == BoundaryKind::SpeechRegion));
        assert!(hypotheses
            .iter()
            .any(|candidate| candidate.kind == BoundaryKind::SyllableIsland));
        assert!(hypotheses
            .iter()
            .any(|candidate| candidate.kind == BoundaryKind::PossibleWordRegion));

        for pair in hypotheses.windows(2) {
            assert!(
                pair[0].confidence >= pair[1].confidence,
                "hypotheses should be sorted by descending confidence"
            );
        }
    }

    #[test]
    fn noisy_signal_emits_noise_event_without_high_confidence_word_region() {
        let sample_rate = 16_000_u32;
        let samples = click_burst(sample_rate, 0.45);
        let (landmarks, features, likelihoods, nuclei, islands) = pipeline(&samples, sample_rate);

        let hypotheses = emit_ranked_boundary_hypotheses(
            &landmarks,
            &likelihoods,
            &features,
            &nuclei,
            &islands,
            &BoundaryHypothesisConfig::default(),
        );

        let has_noise_event = hypotheses.iter().any(|candidate| {
            candidate.kind == BoundaryKind::NoiseEvent
                && candidate
                    .evidence
                    .contains(&BoundaryEvidence::NoiseRejected)
        });
        assert!(has_noise_event, "expected explicit NoiseEvent hypothesis");

        let best_word_confidence = hypotheses
            .iter()
            .filter(|candidate| candidate.kind == BoundaryKind::PossibleWordRegion)
            .map(|candidate| candidate.confidence)
            .fold(0.0_f32, f32::max);
        assert!(
            best_word_confidence <= 0.55,
            "noise should not become high-confidence PossibleWordRegion"
        );
    }
}
