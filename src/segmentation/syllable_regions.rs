//! Expansion of vowel nuclei into syllable islands.
//!
//! Given a list of [`VowelNucleusCandidate`]s, this module walks outward from
//! each nucleus, absorbing adjacent frames that look like onset or coda
//! consonant material.  The result is a [`SyllableIsland`]: a region that
//! spans the nucleus plus any plausible flanking consonantal material.
//!
//! Room-noise bursts are excluded because the expansion stops as soon as
//! frame-level speech confidence drops below threshold or noise confidence
//! rises above threshold, with no "look-ahead" that could accidentally
//! incorporate isolated energy spikes.

use serde::{Deserialize, Serialize};

use crate::audio::features::{AcousticFeatureFrame, AcousticFeatureStream};
use crate::audio::speech_likelihood::SpeechLikelihood;
use crate::segmentation::nuclei::VowelNucleusCandidate;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Thresholds for expanding a nucleus into a syllable island.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SyllableExpansionConfig {
    /// Minimum `speech_confidence` required for a frame to be absorbed as
    /// onset or coda material.
    pub min_speech_confidence: f32,
    /// Maximum `noise_confidence` tolerated in onset or coda frames.
    /// Frames above this threshold stop expansion immediately.
    pub max_noise_confidence: f32,
    /// Maximum expansion on each side of the nucleus, in milliseconds.
    /// Expansion halts at this distance even if speech evidence continues.
    pub max_expansion_ms: f32,
}

impl Default for SyllableExpansionConfig {
    fn default() -> Self {
        Self {
            min_speech_confidence: 0.20,
            max_noise_confidence: 0.65,
            max_expansion_ms: 120.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Output type
// ---------------------------------------------------------------------------

/// A syllable island: a vowel nucleus together with adjacent onset / coda
/// consonant material absorbed from the surrounding frames.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SyllableIsland {
    /// Start time of the island (may precede the nucleus onset), in ms.
    pub start_time: f32,
    /// The vowel nucleus at the core of this island.
    pub nucleus: VowelNucleusCandidate,
    /// End time of the island (may follow the nucleus offset), in ms.
    pub end_time: f32,
    /// Confidence for this island, inherited from the nucleus.
    pub confidence: f32,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Expand each nucleus into a syllable island by absorbing adjacent onset and
/// coda frames.
///
/// `likelihoods` and `features` must be frame-aligned with each other and
/// must cover at least the time range of all `nuclei`.  If they differ in
/// length, only the shorter prefix is considered.
///
/// Each nucleus produces exactly one [`SyllableIsland`].  Islands may overlap
/// when nuclei are very close together; callers are responsible for any
/// downstream merging policy.
pub fn extract_syllable_islands(
    nuclei: &[VowelNucleusCandidate],
    likelihoods: &[SpeechLikelihood],
    features: &AcousticFeatureStream,
    config: &SyllableExpansionConfig,
) -> Vec<SyllableIsland> {
    if nuclei.is_empty() {
        return Vec::new();
    }

    let frame_count = likelihoods.len().min(features.frames.len());
    if frame_count == 0 {
        return Vec::new();
    }

    nuclei
        .iter()
        .map(|nucleus| {
            let start_time =
                expand_left(nucleus.start_time, likelihoods, &features.frames, config, frame_count);
            let end_time =
                expand_right(nucleus.end_time, likelihoods, &features.frames, config, frame_count);
            SyllableIsland {
                start_time,
                nucleus: nucleus.clone(),
                end_time,
                confidence: nucleus.confidence,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Walk left from `nucleus_start_ms`, absorbing frames that meet the speech
/// threshold.  Returns the earliest frame start time reached, or
/// `nucleus_start_ms` if no eligible frames are found.
fn expand_left(
    nucleus_start_ms: f32,
    likelihoods: &[SpeechLikelihood],
    frames: &[AcousticFeatureFrame],
    config: &SyllableExpansionConfig,
    frame_count: usize,
) -> f32 {
    // Find the first frame that starts at or after the nucleus.
    let nucleus_frame = first_frame_at_or_after(frames, nucleus_start_ms, frame_count);
    let min_time_ms = nucleus_start_ms - config.max_expansion_ms;
    let mut boundary_ms = nucleus_start_ms;

    for i in (0..nucleus_frame).rev() {
        let frame = &frames[i];
        if (frame.frame_start_ms as f32) < min_time_ms {
            break;
        }
        let l = &likelihoods[i];
        if l.speech_confidence < config.min_speech_confidence
            || l.noise_confidence > config.max_noise_confidence
        {
            break;
        }
        boundary_ms = frame.frame_start_ms as f32;
    }

    boundary_ms
}

/// Walk right from `nucleus_end_ms`, absorbing frames that meet the speech
/// threshold.  Returns the latest frame end time reached, or `nucleus_end_ms`
/// if no eligible frames are found.
fn expand_right(
    nucleus_end_ms: f32,
    likelihoods: &[SpeechLikelihood],
    frames: &[AcousticFeatureFrame],
    config: &SyllableExpansionConfig,
    frame_count: usize,
) -> f32 {
    // First frame that begins after the nucleus ends.
    let after_nucleus = first_frame_after(frames, nucleus_end_ms, frame_count);
    let max_time_ms = nucleus_end_ms + config.max_expansion_ms;
    let mut boundary_ms = nucleus_end_ms;

    for i in after_nucleus..frame_count {
        let frame = &frames[i];
        if (frame.frame_end_ms as f32) > max_time_ms {
            break;
        }
        let l = &likelihoods[i];
        if l.speech_confidence < config.min_speech_confidence
            || l.noise_confidence > config.max_noise_confidence
        {
            break;
        }
        boundary_ms = frame.frame_end_ms as f32;
    }

    boundary_ms
}

/// Index of the first frame whose `frame_start_ms >= time_ms`.
fn first_frame_at_or_after(
    frames: &[AcousticFeatureFrame],
    time_ms: f32,
    frame_count: usize,
) -> usize {
    frames[..frame_count]
        .iter()
        .position(|f| f.frame_start_ms as f32 >= time_ms)
        .unwrap_or(frame_count)
}

/// Index of the first frame whose `frame_start_ms > time_ms`.
fn first_frame_after(
    frames: &[AcousticFeatureFrame],
    time_ms: f32,
    frame_count: usize,
) -> usize {
    frames[..frame_count]
        .iter()
        .position(|f| f.frame_start_ms as f32 > time_ms)
        .unwrap_or(frame_count)
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
    use crate::segmentation::nuclei::{NucleusDetectionConfig, detect_nuclei};
    use crate::voice::tract::source_filter_track_from_acoustic_full;

    // Shared synthetic signal helpers.

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

    fn fricative_like_noise(sample_rate: u32, seconds: f32) -> Vec<f32> {
        const LCG_A: u32 = 1_664_525;
        const LCG_C: u32 = 1_013_904_223;
        const LCG_SEED: u32 = 0xA5A5_17F3;
        let mut state: u32 = LCG_SEED;
        let count = (sample_rate as f32 * seconds).round() as usize;
        let mut samples: Vec<f32> = (0..count)
            .map(|_| {
                state = state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
                let unit = (state >> 8) as f32 / (u32::MAX >> 8) as f32;
                (unit * 2.0 - 1.0) * 0.20
            })
            .collect();
        for (idx, sample) in samples.iter_mut().enumerate() {
            let t = idx as f32 / sample_rate as f32;
            *sample = (*sample * 0.7) + (2.0 * std::f32::consts::PI * 3_000.0 * t).sin() * 0.03;
        }
        samples
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
    ) -> (AcousticFeatureStream, Vec<SpeechLikelihood>) {
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

    /// A click burst should not produce any syllable islands.
    #[test]
    fn click_burst_yields_no_syllable_island() {
        let sample_rate = 16_000_u32;
        let samples = click_burst(sample_rate, 0.40);

        let (features, likelihoods) = pipeline(&samples, sample_rate);
        let nuclei = detect_nuclei(&likelihoods, &features, &NucleusDetectionConfig::default());
        let islands = extract_syllable_islands(
            &nuclei,
            &likelihoods,
            &features,
            &SyllableExpansionConfig::default(),
        );

        assert!(
            islands.is_empty(),
            "expected no syllable islands for click burst, got {}",
            islands.len()
        );
    }

    /// A consonant-vowel-consonant pattern should yield one syllable island
    /// that spans beyond the nucleus alone.
    #[test]
    fn cvc_pattern_yields_island_wider_than_nucleus() {
        let sample_rate = 16_000_u32;
        let onset_s = 0.06_f32;
        let vowel_s = 0.18_f32;
        let coda_s = 0.06_f32;

        let mut samples = Vec::new();
        samples.extend(fricative_like_noise(sample_rate, onset_s));
        samples.extend(vowelish_harmonic(sample_rate, vowel_s));
        samples.extend(fricative_like_noise(sample_rate, coda_s));

        let (features, likelihoods) = pipeline(&samples, sample_rate);
        let nuclei = detect_nuclei(&likelihoods, &features, &NucleusDetectionConfig::default());

        assert!(
            !nuclei.is_empty(),
            "expected at least one nucleus in CVC pattern"
        );

        let islands = extract_syllable_islands(
            &nuclei,
            &likelihoods,
            &features,
            &SyllableExpansionConfig::default(),
        );

        assert!(
            !islands.is_empty(),
            "expected at least one syllable island in CVC pattern"
        );

        let island = islands
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .unwrap();

        let island_span = island.end_time - island.start_time;
        let nucleus_span = island.nucleus.end_time - island.nucleus.start_time;

        assert!(
            island_span > nucleus_span,
            "island span ({:.1} ms) should exceed nucleus span ({:.1} ms) in a CVC pattern",
            island_span,
            nucleus_span
        );
    }

    /// Islands from two separate nuclei should remain distinct (not collapsed
    /// into a single island even when they share the same feature stream).
    #[test]
    fn two_nuclei_produce_two_islands() {
        let sample_rate = 16_000_u32;
        let mut samples = Vec::new();
        samples.extend(vowelish_harmonic(sample_rate, 0.18));
        // A short silence gap separates the two vowels.
        samples.extend(vec![0.0_f32; (sample_rate as f32 * 0.12).round() as usize]);
        samples.extend(vowelish_harmonic(sample_rate, 0.18));

        let (features, likelihoods) = pipeline(&samples, sample_rate);
        let nuclei = detect_nuclei(&likelihoods, &features, &NucleusDetectionConfig::default());

        assert!(
            nuclei.len() >= 2,
            "expected two nuclei for two separated vowels, got {}",
            nuclei.len()
        );

        let islands = extract_syllable_islands(
            &nuclei,
            &likelihoods,
            &features,
            &SyllableExpansionConfig::default(),
        );

        assert!(
            islands.len() >= 2,
            "expected two islands for two separated nuclei, got {}",
            islands.len()
        );
    }
}
