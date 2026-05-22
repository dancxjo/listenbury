//! Acoustic segmentation: vowel nuclei and syllable islands.
//!
//! This module implements a middle segmentation layer that sits between raw
//! per-frame acoustic evidence and higher-level linguistic structures such as
//! word-region hypotheses.
//!
//! ## Pipeline
//!
//! 1. Feed [`SpeechLikelihood`] frames (from [`crate::audio::speech_likelihood`])
//!    into [`detect_nuclei`] to find probable vowel centres.
//! 2. Pass the resulting [`VowelNucleusCandidate`] list and the same feature
//!    frames into [`extract_syllable_islands`] to absorb adjacent onset and
//!    coda consonant material.
//! 3. Forward [`SyllableIsland`]s upstream to phoneme alignment, ASR, and
//!    mimicry code.
//!
//! Energy-based word cuts are **not** the right anchor for word boundaries;
//! vowel nuclei provide a more stable acoustic anchor because they carry
//! formant and voicing evidence at their core.
//!
//! [`SpeechLikelihood`]: crate::audio::speech_likelihood::SpeechLikelihood

pub mod boundary_hypotheses;
pub mod nuclei;
pub mod syllable_regions;
pub mod word_regions;

pub use boundary_hypotheses::{
    BoundaryEvidence, BoundaryHypothesis, BoundaryHypothesisConfig, BoundaryKind,
    emit_ranked_boundary_hypotheses, generate_landmark_hypotheses,
};
pub use nuclei::{NucleusDetectionConfig, NucleusEvidence, VowelNucleusCandidate, detect_nuclei};
pub use syllable_regions::{SyllableExpansionConfig, SyllableIsland, extract_syllable_islands};
pub use word_regions::{WordRegionConfig, rank_word_region_hypotheses};
