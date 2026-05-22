use serde::{Deserialize, Serialize};

use crate::audio::features::AcousticFeatureStream;
use crate::audio::speech_likelihood::SpeechLikelihood;
use crate::segmentation::boundary_hypotheses::{
    BoundaryEvidence, BoundaryHypothesis, BoundaryKind,
};
use crate::segmentation::nuclei::NucleusEvidence;
use crate::segmentation::syllable_regions::SyllableIsland;

// Confidence blend for speech-like word regions:
// speech + voicing + formants + nucleus/island support, minus noise.
const WORD_SPEECH_WEIGHT: f32 = 0.35;
const WORD_VOICED_WEIGHT: f32 = 0.30;
const WORD_FORMANT_WEIGHT: f32 = 0.25;
const WORD_ISLAND_WEIGHT: f32 = 0.10;
const WORD_NOISE_PENALTY: f32 = 0.35;
const NOISE_DEMOTION_FACTOR: f32 = 0.45;
const NOISE_EVENT_MAX_CONFIDENCE: f32 = 0.45;
const SILENCE_SPEECH_THRESHOLD: f32 = 0.12;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WordRegionConfig {
    pub min_voiced_confidence: f32,
    pub min_formant_confidence: f32,
    pub min_speech_confidence: f32,
    pub max_noise_confidence: f32,
    pub min_flux_for_peak: f32,
}

impl Default for WordRegionConfig {
    fn default() -> Self {
        Self {
            // Require clear periodicity and resonance before elevating a region
            // to a possible word.
            min_voiced_confidence: 0.30,
            min_formant_confidence: 0.25,
            min_speech_confidence: 0.30,
            max_noise_confidence: 0.65,
            min_flux_for_peak: 0.20,
        }
    }
}

pub fn rank_word_region_hypotheses(
    islands: &[SyllableIsland],
    likelihoods: &[SpeechLikelihood],
    features: &AcousticFeatureStream,
    config: &WordRegionConfig,
) -> Vec<BoundaryHypothesis> {
    let frame_count = likelihoods.len().min(features.frames.len());
    if frame_count == 0 || islands.is_empty() {
        return Vec::new();
    }

    let mut hypotheses = islands
        .iter()
        .filter_map(|island| {
            let indices = overlapping_frame_indices(
                island.start_time,
                island.end_time,
                features,
                frame_count,
            );
            if indices.is_empty() {
                return None;
            }

            let mut speech_peak = 0.0_f32;
            let mut voiced_peak = 0.0_f32;
            let mut formant_peak = 0.0_f32;
            let mut noise_total = 0.0_f32;
            let mut has_flux_peak = false;

            for &idx in &indices {
                let likelihood = &likelihoods[idx];
                let frame = &features.frames[idx];
                speech_peak = speech_peak.max(likelihood.speech_confidence);
                voiced_peak = voiced_peak.max(likelihood.voiced_confidence);
                formant_peak = formant_peak.max(likelihood.formant_confidence);
                noise_total += likelihood.noise_confidence;
                has_flux_peak |= frame.spectral_flux >= config.min_flux_for_peak;
            }

            let noise_mean = noise_total / indices.len() as f32;
            let mut evidence = vec![BoundaryEvidence::VowelNucleus];

            if voiced_peak >= config.min_voiced_confidence {
                evidence.push(BoundaryEvidence::VoicingOnset);
                evidence.push(BoundaryEvidence::VoicingOffset);
            }
            if formant_peak >= config.min_formant_confidence {
                evidence.push(BoundaryEvidence::FormantOnset);
                evidence.push(BoundaryEvidence::FormantOffset);
            }
            if has_flux_peak {
                evidence.push(BoundaryEvidence::SpectralFluxPeak);
            }
            if has_silence_gap_around(island, likelihoods, features) {
                evidence.push(BoundaryEvidence::SilenceGap);
            }

            let has_nucleus_voicing = island
                .nucleus
                .evidence
                .iter()
                .any(|item| matches!(item, NucleusEvidence::Voiced));
            let has_nucleus_formant = island
                .nucleus
                .evidence
                .iter()
                .any(|item| matches!(item, NucleusEvidence::Formant));
            let has_nucleus_vowel = island
                .nucleus
                .evidence
                .iter()
                .any(|item| matches!(item, NucleusEvidence::VowelLike));

            let raw_confidence = ((speech_peak * WORD_SPEECH_WEIGHT)
                + (voiced_peak * WORD_VOICED_WEIGHT)
                + (formant_peak * WORD_FORMANT_WEIGHT)
                + (island.confidence * WORD_ISLAND_WEIGHT)
                - (noise_mean * WORD_NOISE_PENALTY))
                .clamp(0.0, 1.0);

            let speechlike = speech_peak >= config.min_speech_confidence
                && noise_mean <= config.max_noise_confidence
                && has_nucleus_vowel
                && has_nucleus_voicing
                && has_nucleus_formant;

            let (kind, confidence) = if speechlike {
                evidence.push(BoundaryEvidence::MatchesExpectedPhoneShape);
                (BoundaryKind::PossibleWordRegion, raw_confidence)
            } else {
                evidence.push(BoundaryEvidence::NoiseRejected);
                (
                    BoundaryKind::NoiseEvent,
                    (raw_confidence * NOISE_DEMOTION_FACTOR)
                        .clamp(0.0, NOISE_EVENT_MAX_CONFIDENCE),
                )
            };

            Some(BoundaryHypothesis {
                start_time: island.start_time,
                end_time: island.end_time,
                kind,
                confidence,
                evidence,
            })
        })
        .collect::<Vec<_>>();

    hypotheses.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
    hypotheses
}

fn overlapping_frame_indices(
    start_ms: f32,
    end_ms: f32,
    features: &AcousticFeatureStream,
    frame_count: usize,
) -> Vec<usize> {
    let mut indices = Vec::new();
    for idx in 0..frame_count {
        let frame = &features.frames[idx];
        if frame.frame_end_ms as f32 >= start_ms && frame.frame_start_ms as f32 <= end_ms {
            indices.push(idx);
        }
    }
    indices
}

fn has_silence_gap_around(
    island: &SyllableIsland,
    likelihoods: &[SpeechLikelihood],
    features: &AcousticFeatureStream,
) -> bool {
    let before = features
        .frames
        .iter()
        .zip(likelihoods.iter())
        .rev()
        .find(|(frame, _)| frame.frame_end_ms as f32 <= island.start_time)
        .map(|(_, l)| l.speech_confidence)
        .unwrap_or(0.0);

    let after = features
        .frames
        .iter()
        .zip(likelihoods.iter())
        .find(|(frame, _)| frame.frame_start_ms as f32 >= island.end_time)
        .map(|(_, l)| l.speech_confidence)
        .unwrap_or(0.0);

    before < SILENCE_SPEECH_THRESHOLD || after < SILENCE_SPEECH_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::acoustic::analyze_mono_samples;
    use crate::audio::features::{build_feature_stream, AcousticFeatureFrame};
    use crate::audio::speech_likelihood::{build_speech_likelihood_stream, SpeechLikelihoodConfig};
    use crate::segmentation::nuclei::{detect_nuclei, NucleusDetectionConfig};
    use crate::segmentation::syllable_regions::{
        extract_syllable_islands, SyllableExpansionConfig,
    };
    use crate::voice::tract::source_filter_track_from_acoustic_full;

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
        Vec<SpeechLikelihood>,
        AcousticFeatureStream,
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
        (likelihoods, features, islands)
    }

    #[test]
    fn noisy_clicks_demote_word_region_to_noise_event() {
        let sample_rate = 16_000_u32;
        let samples = click_burst(sample_rate, 0.40);
        let (_likelihoods, _features, islands) = pipeline(&samples, sample_rate);

        assert!(
            islands.is_empty(),
            "click burst fixture should not produce speech islands"
        );
    }

    #[test]
    fn energy_only_candidate_without_voicing_or_formants_is_demoted_to_noise() {
        let islands = vec![SyllableIsland {
            start_time: 0.0,
            nucleus: crate::segmentation::nuclei::VowelNucleusCandidate {
                start_time: 0.0,
                peak_time: 10.0,
                end_time: 20.0,
                confidence: 0.95,
                evidence: Vec::new(),
            },
            end_time: 20.0,
            confidence: 0.95,
        }];
        let likelihoods = vec![SpeechLikelihood {
            speech_confidence: 0.88,
            voiced_confidence: 0.08,
            vowel_like_confidence: 0.12,
            consonant_like_confidence: 0.65,
            noise_confidence: 0.25,
            formant_confidence: 0.09,
        }];
        let features = AcousticFeatureStream {
            hop_ms: 10.0,
            frames: vec![AcousticFeatureFrame {
                frame_start_ms: 0,
                frame_end_ms: 20,
                spectral_flux: 0.22,
                ..AcousticFeatureFrame::default()
            }],
        };

        let hypotheses = rank_word_region_hypotheses(
            &islands,
            &likelihoods,
            &features,
            &WordRegionConfig::default(),
        );
        assert_eq!(hypotheses.len(), 1);
        let candidate = &hypotheses[0];
        assert_eq!(candidate.kind, BoundaryKind::NoiseEvent);
        assert!(
            candidate
                .evidence
                .contains(&BoundaryEvidence::NoiseRejected),
            "demoted candidate should preserve rejection evidence"
        );
        assert!(
            candidate.confidence <= 0.45,
            "energy-only candidate should be demoted below high-confidence word range"
        );
    }

    #[test]
    fn speechlike_island_produces_possible_word_region_with_evidence() {
        let sample_rate = 16_000_u32;
        let samples = vowelish_harmonic(sample_rate, 0.32);
        let (likelihoods, features, islands) = pipeline(&samples, sample_rate);
        assert!(
            !islands.is_empty(),
            "expected syllable island from speechlike signal"
        );

        let hypotheses = rank_word_region_hypotheses(
            &islands,
            &likelihoods,
            &features,
            &WordRegionConfig::default(),
        );

        let best = hypotheses
            .iter()
            .find(|candidate| candidate.kind == BoundaryKind::PossibleWordRegion)
            .expect("expected possible word region hypothesis");

        assert!(
            best.evidence.contains(&BoundaryEvidence::VowelNucleus),
            "expected vowel nucleus evidence"
        );
        assert!(
            best.confidence > 0.35,
            "speechlike possible word region should have meaningful confidence"
        );
    }

    #[test]
    fn noise_event_confidence_stays_below_word_confidence_threshold_without_voicing_formants() {
        let sample_rate = 16_000_u32;
        let mut samples = click_burst(sample_rate, 0.30);
        samples.extend(vec![0.0_f32; (sample_rate as f32 * 0.12).round() as usize]);
        samples.extend(click_burst(sample_rate, 0.30));

        let (likelihoods, features, islands) = pipeline(&samples, sample_rate);
        let hypotheses = rank_word_region_hypotheses(
            &islands,
            &likelihoods,
            &features,
            &WordRegionConfig::default(),
        );

        let max_word_confidence = hypotheses
            .iter()
            .filter(|candidate| candidate.kind == BoundaryKind::PossibleWordRegion)
            .map(|candidate| candidate.confidence)
            .fold(0.0_f32, f32::max);

        assert!(
            max_word_confidence <= 0.55,
            "energy-only regions should not become high-confidence word candidates"
        );
    }
}
