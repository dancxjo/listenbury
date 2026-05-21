use crate::linguistic::phonology::Phone;
use crate::prosody::syllable::{DiagnosticKind, SyllableDiagnostic};

/// Verdict returned when the profile evaluates an onset cluster.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnsetVerdict {
    /// The phones that were evaluated (cloned from the input references).
    pub cluster: Vec<Phone>,
    /// Whether this cluster is a legal onset under the active profile.
    pub is_legal: bool,
    /// Human-readable explanation of the decision, using IPA.
    pub reason: String,
}

impl OnsetVerdict {
    /// Convert this verdict into a [`SyllableDiagnostic`].
    pub fn as_diagnostic(&self) -> SyllableDiagnostic {
        let kind = if self.is_legal {
            DiagnosticKind::LegalOnset
        } else {
            DiagnosticKind::RejectedOnset
        };
        SyllableDiagnostic::new(kind, self.reason.clone())
    }

    /// Concatenate the IPA strings of all phones in this verdict's cluster.
    pub fn cluster_ipa(&self) -> String {
        self.cluster.iter().map(|p| p.ipa.as_str()).collect()
    }
}

/// A phonotactic profile that the syllabifier consults to determine the
/// legality of onset and coda clusters.
///
/// Methods take slices of [`Phone`] references (`&[&Phone]`) so that the
/// syllabifier can build candidate clusters without heap-allocating clones.
///
/// Implement this trait to add new variety profiles.
pub trait PhonotacticProfile {
    /// Display name of this variety/profile, e.g. `"General American English"`.
    fn variety_name(&self) -> &str;

    /// Returns `true` if `phone` is a nucleus (vowel or syllabic consonant)
    /// in this variety.
    fn is_nucleus(&self, phone: &Phone) -> bool;

    /// Returns a detailed verdict for whether `cluster` is a legal onset.
    ///
    /// An empty cluster is always legal (null onset).
    fn onset_verdict(&self, cluster: &[&Phone]) -> OnsetVerdict;

    /// Returns `true` if `cluster` is a legal onset.
    ///
    /// Convenience wrapper around [`onset_verdict`][Self::onset_verdict].
    fn is_legal_onset(&self, cluster: &[&Phone]) -> bool {
        cluster.is_empty() || self.onset_verdict(cluster).is_legal
    }

    /// Returns `true` if `cluster` is a legal coda.
    ///
    /// An empty cluster is always legal.
    fn is_legal_coda(&self, cluster: &[&Phone]) -> bool;
}
