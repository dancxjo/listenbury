use serde::{Deserialize, Serialize};

use crate::audio::features::{AcousticFeatureFrame, AcousticFeatureStream};
use crate::voice::tract::{SourceFilterFrame, SourceFilterTrack};
use crate::voice::vocal_plausibility::{VocalPlausibilityConfig, assess_vocal_plausibility};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechLikelihoodConfig {
    pub min_snr_db: f32,
    pub max_snr_db: f32,
    pub min_energy_over_noise: f32,
    pub max_energy_over_noise: f32,
    pub vowel_nucleus_threshold: f32,
    pub consonant_support_radius_frames: usize,
    pub isolated_consonant_penalty: f32,
    pub vocal_plausibility: VocalPlausibilityConfig,
}

impl Default for SpeechLikelihoodConfig {
    fn default() -> Self {
        Self {
            min_snr_db: 1.5,
            max_snr_db: 18.0,
            min_energy_over_noise: 0.35,
            max_energy_over_noise: 4.0,
            vowel_nucleus_threshold: 0.35,
            consonant_support_radius_frames: 3,
            isolated_consonant_penalty: 0.25,
            vocal_plausibility: VocalPlausibilityConfig::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechLikelihood {
    pub speech_confidence: f32,
    pub voiced_confidence: f32,
    pub vowel_like_confidence: f32,
    pub consonant_like_confidence: f32,
    pub noise_confidence: f32,
    pub formant_confidence: f32,
}

pub fn build_speech_likelihood_stream(
    features: &AcousticFeatureStream,
    source_filter_track: Option<&SourceFilterTrack>,
    config: &SpeechLikelihoodConfig,
) -> Vec<SpeechLikelihood> {
    let frame_count = features.frames.len();
    let mut provisional = Vec::with_capacity(frame_count);
    let mut nuclei = vec![false; frame_count];

    for idx in 0..frame_count {
        let frame = &features.frames[idx];
        let sf = source_filter_track.and_then(|track| track.frames.get(idx));
        let voiced = voiced_confidence(sf);
        let formant = formant_confidence_at(source_filter_track, idx, config);
        let energy = energy_confidence(frame, config);
        let vowel =
            ((voiced * 0.45) + (formant * 0.40) + low_band_dominance_confidence(frame) * 0.15)
                .clamp(0.0, 1.0);
        nuclei[idx] = vowel >= config.vowel_nucleus_threshold;
        provisional.push((energy, voiced, formant, vowel));
    }

    (0..frame_count)
        .map(|idx| {
            let frame = &features.frames[idx];
            let sf = source_filter_track.and_then(|track| track.frames.get(idx));
            let (energy, voiced, formant, vowel) = provisional[idx];
            let has_nearby_nucleus =
                nearby_nucleus(&nuclei, idx, config.consonant_support_radius_frames);
            let mut consonant = consonant_like_confidence(frame, sf);
            if !has_nearby_nucleus {
                consonant *= config.isolated_consonant_penalty.clamp(0.0, 1.0);
            }
            let noise = noise_confidence(frame, sf, has_nearby_nucleus, voiced, formant);
            let speech = ((energy * 0.35)
                + (vowel * 0.30)
                + (consonant * 0.20)
                + (voiced * 0.10)
                + (formant * 0.05)
                - (noise * 0.42))
                .clamp(0.0, 1.0);
            SpeechLikelihood {
                speech_confidence: speech,
                voiced_confidence: voiced,
                vowel_like_confidence: vowel,
                consonant_like_confidence: consonant,
                noise_confidence: noise,
                formant_confidence: formant,
            }
        })
        .collect()
}

fn energy_confidence(frame: &AcousticFeatureFrame, config: &SpeechLikelihoodConfig) -> f32 {
    let snr = normalize(frame.snr_db, config.min_snr_db, config.max_snr_db);
    let over_noise = normalize(
        frame.energy_over_noise,
        config.min_energy_over_noise,
        config.max_energy_over_noise,
    );
    (snr * 0.55 + over_noise * 0.45).clamp(0.0, 1.0)
}

fn voiced_confidence(source_filter: Option<&SourceFilterFrame>) -> f32 {
    let Some(source_filter) = source_filter else {
        return 0.0;
    };
    let v = &source_filter.voicing;
    ((v.voicing_probability * 0.65) + (v.f0_confidence * 0.35)).clamp(0.0, 1.0)
}

fn formant_confidence_at(
    source_filter_track: Option<&SourceFilterTrack>,
    idx: usize,
    config: &SpeechLikelihoodConfig,
) -> f32 {
    let Some(track) = source_filter_track else {
        return 0.0;
    };
    let Some(frame) = track.frames.get(idx) else {
        return 0.0;
    };
    let prev = idx.checked_sub(1).and_then(|i| track.frames.get(i));
    let next = track.frames.get(idx + 1);
    assess_vocal_plausibility(frame, prev, next, &config.vocal_plausibility).plausibility
}

fn low_band_dominance_confidence(frame: &AcousticFeatureFrame) -> f32 {
    ((frame.low_band_energy_db - frame.high_band_energy_db + 5.0) / 20.0).clamp(0.0, 1.0)
}

fn consonant_like_confidence(
    frame: &AcousticFeatureFrame,
    source_filter: Option<&SourceFilterFrame>,
) -> f32 {
    let zcr = normalize(frame.zero_crossing_rate, 0.08, 0.30);
    let flux = normalize(frame.spectral_flux, 0.03, 0.30);
    let energetic = normalize(frame.energy_over_noise, 0.2, 3.0);
    let frication = source_filter
        .map(|sf| sf.noise.frication_energy)
        .unwrap_or(frame.broadband_noise_likeness);
    ((zcr * 0.30) + (flux * 0.30) + (frication * 0.25) + (energetic * 0.15)).clamp(0.0, 1.0)
}

fn noise_confidence(
    frame: &AcousticFeatureFrame,
    source_filter: Option<&SourceFilterFrame>,
    has_nearby_nucleus: bool,
    voiced_confidence: f32,
    formant_confidence: f32,
) -> f32 {
    let mut broadband = frame.broadband_noise_likeness;
    if let Some(sf) = source_filter {
        broadband = (broadband * 0.7 + sf.noise.noise_ratio * 0.3).clamp(0.0, 1.0);
    }
    let evidence_absent = (1.0 - voiced_confidence).max(1.0 - formant_confidence);
    let unsupported = if has_nearby_nucleus { 0.0 } else { 1.0 };
    let clickiness = normalize(frame.spectral_flux, 0.16, 0.45)
        * normalize(frame.zero_crossing_rate, 0.08, 0.26);
    let base = ((broadband * 0.50)
        + (evidence_absent * 0.30)
        + (unsupported * 0.15)
        + (clickiness * 0.05))
        .clamp(0.0, 1.0);
    if has_nearby_nucleus {
        (base * 0.55).clamp(0.0, 1.0)
    } else {
        base
    }
}

fn nearby_nucleus(nuclei: &[bool], idx: usize, radius: usize) -> bool {
    if nuclei.is_empty() {
        return false;
    }
    let start = idx.saturating_sub(radius);
    let end = (idx + radius + 1).min(nuclei.len());
    nuclei[start..end].iter().any(|is_nucleus| *is_nucleus)
}

fn normalize(value: f32, min: f32, max: f32) -> f32 {
    if max <= min {
        return 0.0;
    }
    ((value - min) / (max - min)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::acoustic::analyze_mono_samples;
    use crate::audio::features::build_feature_stream;
    use crate::voice::tract::source_filter_track_from_acoustic_full;

    fn deterministic_noise(len: usize) -> Vec<f32> {
        const LCG_A: u32 = 1_664_525;
        const LCG_C: u32 = 1_013_904_223;
        const LCG_SEED: u32 = 0xA5A5_17F3;
        const PRECISION_SHIFT: u32 = 8;
        let mut state: u32 = LCG_SEED;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
                let unit = ((state >> PRECISION_SHIFT) as f32)
                    / ((u32::MAX >> PRECISION_SHIFT) as f32);
                (unit * 2.0 - 1.0) * 0.20
            })
            .collect()
    }

    fn vowelish_harmonic(sample_rate: u32, seconds: f32) -> Vec<f32> {
        let count = (sample_rate as f32 * seconds).round() as usize;
        (0..count)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                let f0 = 145.0;
                (2.0 * std::f32::consts::PI * f0 * t).sin() * 0.22
                    + (2.0 * std::f32::consts::PI * f0 * 2.0 * t).sin() * 0.11
                    + (2.0 * std::f32::consts::PI * f0 * 3.0 * t).sin() * 0.06
            })
            .collect()
    }

    fn click_burst(sample_rate: u32, seconds: f32) -> Vec<f32> {
        let count = (sample_rate as f32 * seconds).round() as usize;
        let mut samples = vec![0.0; count];
        let step = (sample_rate / 30) as usize;
        for idx in (0..count).step_by(step.max(1)) {
            samples[idx] = 0.85;
        }
        samples
    }

    fn fricative_like_noise(sample_rate: u32, seconds: f32) -> Vec<f32> {
        let mut samples = deterministic_noise((sample_rate as f32 * seconds).round() as usize);
        for (idx, sample) in samples.iter_mut().enumerate() {
            let t = idx as f32 / sample_rate as f32;
            *sample = (*sample * 0.7) + (2.0 * std::f32::consts::PI * 3_000.0 * t).sin() * 0.03;
        }
        samples
    }

    fn mean_speech_confidence(samples: &[f32], sample_rate: u32) -> f32 {
        let analysis = analyze_mono_samples(samples, sample_rate);
        let feature_stream = build_feature_stream(
            samples,
            sample_rate,
            &analysis.energy_envelope,
            analysis.spectrogram.levels.first(),
        );
        let source_filter = source_filter_track_from_acoustic_full(&analysis, samples);
        let likelihoods = build_speech_likelihood_stream(
            &feature_stream,
            Some(&source_filter),
            &SpeechLikelihoodConfig::default(),
        );
        if likelihoods.is_empty() {
            return 0.0;
        }
        likelihoods
            .iter()
            .map(|frame| frame.speech_confidence)
            .sum::<f32>()
            / likelihoods.len() as f32
    }

    #[test]
    fn vowel_like_material_scores_higher_than_broadband_noise_at_similar_level() {
        let sample_rate = 16_000;
        let vowel = vowelish_harmonic(sample_rate, 0.6);
        let noise = deterministic_noise((sample_rate as f32 * 0.6).round() as usize);

        let vowel_speech = mean_speech_confidence(&vowel, sample_rate);
        let noise_speech = mean_speech_confidence(&noise, sample_rate);

        assert!(
            vowel_speech > noise_speech,
            "expected vowel-like speech confidence > broadband noise: vowel={vowel_speech} noise={noise_speech}"
        );
    }

    #[test]
    fn isolated_clicks_stay_low_confidence_without_neighboring_nuclei() {
        let sample_rate = 16_000;
        let clicks = click_burst(sample_rate, 0.5);
        let click_speech = mean_speech_confidence(&clicks, sample_rate);
        assert!(
            click_speech < 0.35,
            "expected isolated clicks to remain low-confidence speech, got {click_speech}"
        );
    }

    #[test]
    fn fricative_like_noise_is_retained_near_vowel_nuclei() {
        let sample_rate = 16_000;
        let mut samples = Vec::new();
        samples.extend(vowelish_harmonic(sample_rate, 0.22));
        samples.extend(fricative_like_noise(sample_rate, 0.06));
        samples.extend(vowelish_harmonic(sample_rate, 0.22));

        let analysis = analyze_mono_samples(&samples, sample_rate);
        let feature_stream = build_feature_stream(
            &samples,
            sample_rate,
            &analysis.energy_envelope,
            analysis.spectrogram.levels.first(),
        );
        let source_filter = source_filter_track_from_acoustic_full(&analysis, &samples);
        let likelihoods = build_speech_likelihood_stream(
            &feature_stream,
            Some(&source_filter),
            &SpeechLikelihoodConfig::default(),
        );
        let mid = likelihoods.len() / 2;
        let span = &likelihoods[mid.saturating_sub(2)..(mid + 3).min(likelihoods.len())];
        let consonant_support = span
            .iter()
            .map(|frame| frame.consonant_like_confidence)
            .sum::<f32>()
            / span.len() as f32;
        let speech_support = span
            .iter()
            .map(|frame| frame.speech_confidence)
            .sum::<f32>()
            / span.len() as f32;

        assert!(
            consonant_support > 0.15,
            "expected consonant-like evidence near nuclei, got {consonant_support}"
        );
        assert!(
            speech_support > 0.25,
            "expected fricative-like bridge near vowels to be retained, got {speech_support}"
        );
    }
}
