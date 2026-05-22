//! Vowel nucleus detection from speech-likelihood frames.
//!
//! Detects probable vowel centres from per-frame [`SpeechLikelihood`] evidence
//! and groups contiguous high-confidence frames into [`VowelNucleusCandidate`]
//! regions.  Each candidate records the peak evidence frame and the acoustic
//! evidence types that contributed, so downstream stages can reason about
//! confidence rather than re-analysing raw audio.

use serde::{Deserialize, Serialize};

use crate::audio::features::{AcousticFeatureFrame, AcousticFeatureStream};
use crate::audio::speech_likelihood::SpeechLikelihood;

// ---------------------------------------------------------------------------
// Evidence
// ---------------------------------------------------------------------------

/// Acoustic evidence type that contributed to a vowel nucleus candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NucleusEvidence {
    /// Vowel-like harmonic structure (voiced + low-band dominance evidence).
    VowelLike,
    /// Periodic voicing evidence (F0 present, voicing probability high).
    Voiced,
    /// Formant pattern consistent with a vowel resonance.
    Formant,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Thresholds and weights for vowel nucleus detection.
///
/// All thresholds are in the 0.0–1.0 confidence range unless noted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NucleusDetectionConfig {
    /// Minimum combined nucleus score to include a frame in a nucleus region.
    pub nucleus_score_threshold: f32,
    /// Weight of `vowel_like_confidence` in the nucleus score.
    pub vowel_weight: f32,
    /// Weight of `voiced_confidence` in the nucleus score.
    pub voiced_weight: f32,
    /// Weight of `formant_confidence` in the nucleus score.
    pub formant_weight: f32,
    /// Frames with `noise_confidence` above this value are excluded from
    /// nucleus regions, even if other evidence is strong.
    pub max_noise_for_nucleus: f32,
    /// Minimum vowel-like confidence for a frame to contribute
    /// `NucleusEvidence::VowelLike` to the candidate's evidence list.
    pub vowel_evidence_min: f32,
    /// Minimum voiced confidence for a frame to contribute
    /// `NucleusEvidence::Voiced` to the evidence list.
    pub voiced_evidence_min: f32,
    /// Minimum formant confidence for a frame to contribute
    /// `NucleusEvidence::Formant` to the evidence list.
    pub formant_evidence_min: f32,
}

impl Default for NucleusDetectionConfig {
    fn default() -> Self {
        Self {
            nucleus_score_threshold: 0.40,
            vowel_weight: 0.50,
            voiced_weight: 0.30,
            formant_weight: 0.20,
            max_noise_for_nucleus: 0.60,
            vowel_evidence_min: 0.30,
            voiced_evidence_min: 0.30,
            formant_evidence_min: 0.25,
        }
    }
}

// ---------------------------------------------------------------------------
// Output type
// ---------------------------------------------------------------------------

/// A candidate vowel nucleus region detected from acoustic evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VowelNucleusCandidate {
    /// Start time of the nucleus region, in milliseconds.
    pub start_time: f32,
    /// Time of the peak-evidence frame (centre of mass of the region), in ms.
    pub peak_time: f32,
    /// End time of the nucleus region, in milliseconds.
    pub end_time: f32,
    /// Overall confidence for this nucleus candidate (0.0–1.0).
    pub confidence: f32,
    /// Acoustic evidence types that contributed to this candidate.
    pub evidence: Vec<NucleusEvidence>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Detect vowel nucleus candidates from aligned speech-likelihood and feature
/// frames.
///
/// `likelihoods` and `features` must be frame-aligned: `likelihoods[i]` must
/// have been derived from `features.frames[i]`.  If the slices differ in
/// length, only the shorter prefix is examined.
///
/// Each contiguous run of frames whose nucleus score meets
/// `config.nucleus_score_threshold` becomes one [`VowelNucleusCandidate`].
/// Multiple candidates are returned without merging so that competing
/// evidence can be passed upward intact.
pub fn detect_nuclei(
    likelihoods: &[SpeechLikelihood],
    features: &AcousticFeatureStream,
    config: &NucleusDetectionConfig,
) -> Vec<VowelNucleusCandidate> {
    if likelihoods.is_empty() || features.frames.is_empty() {
        return Vec::new();
    }

    let frame_count = likelihoods.len().min(features.frames.len());

    // Compute a nucleus score for every frame.
    let scores: Vec<f32> = (0..frame_count)
        .map(|i| nucleus_score(&likelihoods[i], config))
        .collect();

    // Group contiguous above-threshold frames into candidate regions.
    let mut candidates = Vec::new();
    let mut region_start: Option<usize> = None;

    for i in 0..=frame_count {
        let above = i < frame_count && scores[i] >= config.nucleus_score_threshold;
        match (region_start, above) {
            (None, true) => region_start = Some(i),
            (Some(start), false) => {
                candidates.push(build_candidate(
                    start,
                    i,
                    &scores[start..i],
                    &likelihoods[start..i],
                    &features.frames[start..i],
                    config,
                ));
                region_start = None;
            }
            _ => {}
        }
    }

    candidates
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Compute the nucleus score for a single frame.
fn nucleus_score(l: &SpeechLikelihood, config: &NucleusDetectionConfig) -> f32 {
    if l.noise_confidence > config.max_noise_for_nucleus {
        return 0.0;
    }
    (l.vowel_like_confidence * config.vowel_weight
        + l.voiced_confidence * config.voiced_weight
        + l.formant_confidence * config.formant_weight)
        .clamp(0.0, 1.0)
}

/// Build a [`VowelNucleusCandidate`] from a contiguous region of frames.
fn build_candidate(
    _region_offset: usize,
    _region_end: usize,
    scores: &[f32],
    likelihoods: &[SpeechLikelihood],
    frames: &[AcousticFeatureFrame],
    config: &NucleusDetectionConfig,
) -> VowelNucleusCandidate {
    debug_assert_eq!(scores.len(), frames.len());
    debug_assert_eq!(likelihoods.len(), frames.len());

    let peak_local = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let start_time = frames.first().map_or(0.0, |f| f.frame_start_ms as f32);
    let end_time = frames.last().map_or(0.0, |f| f.frame_end_ms as f32);
    let peak_time = frames
        .get(peak_local)
        .map_or(start_time, |f| midpoint_ms(f));

    let confidence = scores
        .iter()
        .cloned()
        .fold(0.0_f32, f32::max);

    let evidence = collect_evidence(likelihoods, config);

    VowelNucleusCandidate {
        start_time,
        peak_time,
        end_time,
        confidence,
        evidence,
    }
}

/// Return which evidence types are present across the nucleus region.
fn collect_evidence(
    likelihoods: &[SpeechLikelihood],
    config: &NucleusDetectionConfig,
) -> Vec<NucleusEvidence> {
    let mut ev = Vec::new();
    if likelihoods
        .iter()
        .any(|l| l.vowel_like_confidence >= config.vowel_evidence_min)
    {
        ev.push(NucleusEvidence::VowelLike);
    }
    if likelihoods
        .iter()
        .any(|l| l.voiced_confidence >= config.voiced_evidence_min)
    {
        ev.push(NucleusEvidence::Voiced);
    }
    if likelihoods
        .iter()
        .any(|l| l.formant_confidence >= config.formant_evidence_min)
    {
        ev.push(NucleusEvidence::Formant);
    }
    ev
}

fn midpoint_ms(f: &AcousticFeatureFrame) -> f32 {
    (f.frame_start_ms as f32 + f.frame_end_ms as f32) / 2.0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::acoustic::analyze_mono_samples;
    use crate::audio::features::build_feature_stream;
    use crate::audio::speech_likelihood::{SpeechLikelihoodConfig, build_speech_likelihood_stream};
    use crate::voice::tract::source_filter_track_from_acoustic_full;

    // Shared synthetic signal helpers (same approach as speech_likelihood tests).

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

    fn silence(sample_rate: u32, seconds: f32) -> Vec<f32> {
        vec![0.0_f32; (sample_rate as f32 * seconds).round() as usize]
    }

    fn pipeline(samples: &[f32], sample_rate: u32) -> (AcousticFeatureStream, Vec<SpeechLikelihood>) {
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
        (features, likelihoods)
    }

    /// A synthetic vowel-ish region should yield at least one nucleus candidate
    /// and its peak should fall inside the voiced region.
    #[test]
    fn vowel_region_yields_nucleus_candidate() {
        let sample_rate = 16_000_u32;
        let vowel_duration_s = 0.30_f32;
        let samples = vowelish_harmonic(sample_rate, vowel_duration_s);

        let (features, likelihoods) = pipeline(&samples, sample_rate);
        let candidates =
            detect_nuclei(&likelihoods, &features, &NucleusDetectionConfig::default());

        assert!(
            !candidates.is_empty(),
            "expected at least one nucleus candidate for vowel-ish signal, got none"
        );

        let vowel_end_ms = vowel_duration_s * 1000.0;
        let peak = candidates
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .unwrap();
        assert!(
            peak.peak_time >= 0.0 && peak.peak_time <= vowel_end_ms,
            "peak_time {:.1} ms should be inside the vowel region (0–{:.0} ms)",
            peak.peak_time,
            vowel_end_ms
        );
    }

    /// Isolated click bursts (no nearby vowel evidence) must not yield any
    /// nucleus candidates.
    #[test]
    fn click_burst_yields_no_nucleus() {
        let sample_rate = 16_000_u32;
        let samples = click_burst(sample_rate, 0.40);

        let (features, likelihoods) = pipeline(&samples, sample_rate);
        let candidates =
            detect_nuclei(&likelihoods, &features, &NucleusDetectionConfig::default());

        assert!(
            candidates.is_empty(),
            "expected no nucleus candidates for isolated click burst, got {}",
            candidates.len()
        );
    }

    /// Two vowel regions separated by silence should produce two distinct
    /// nucleus candidates (not merged into one).
    #[test]
    fn two_separated_vowels_yield_two_nuclei() {
        let sample_rate = 16_000_u32;
        let mut samples = Vec::new();
        samples.extend(vowelish_harmonic(sample_rate, 0.20));
        samples.extend(silence(sample_rate, 0.12));
        samples.extend(vowelish_harmonic(sample_rate, 0.20));

        let (features, likelihoods) = pipeline(&samples, sample_rate);
        let candidates =
            detect_nuclei(&likelihoods, &features, &NucleusDetectionConfig::default());

        assert!(
            candidates.len() >= 2,
            "expected at least 2 nucleus candidates for two separated vowels, got {}",
            candidates.len()
        );

        // The two best candidates should not substantially overlap.
        if candidates.len() >= 2 {
            let a = &candidates[0];
            let b = &candidates[1];
            assert!(
                a.end_time <= b.start_time || b.end_time <= a.start_time,
                "nucleus candidates should not overlap: [{:.0}, {:.0}] vs [{:.0}, {:.0}]",
                a.start_time,
                a.end_time,
                b.start_time,
                b.end_time
            );
        }
    }

    /// Nucleus candidates should include `VowelLike` evidence for a genuine
    /// vowel signal.
    #[test]
    fn vowel_nucleus_carries_vowel_like_evidence() {
        let sample_rate = 16_000_u32;
        let samples = vowelish_harmonic(sample_rate, 0.25);

        let (features, likelihoods) = pipeline(&samples, sample_rate);
        let candidates =
            detect_nuclei(&likelihoods, &features, &NucleusDetectionConfig::default());

        let has_vowel_evidence = candidates
            .iter()
            .any(|c| c.evidence.contains(&NucleusEvidence::VowelLike));
        assert!(
            has_vowel_evidence,
            "expected at least one nucleus with VowelLike evidence"
        );
    }
}
