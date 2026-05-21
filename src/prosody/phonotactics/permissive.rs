use crate::linguistic::phonology::Phone;
use crate::prosody::phonotactics::profile::{OnsetVerdict, PhonotacticProfile};

/// A maximally permissive phonotactic profile that accepts every non-empty
/// onset and coda cluster.
///
/// Useful in unit tests to exercise the syllabifier algorithm independently
/// of phonotactic constraints.
pub struct PermissiveProfile;

impl PhonotacticProfile for PermissiveProfile {
    fn variety_name(&self) -> &str {
        "Permissive (test)"
    }

    fn is_nucleus(&self, phone: &Phone) -> bool {
        matches!(
            phone.ipa.as_str(),
            "ɑ" | "æ"
                | "ʌ"
                | "ɔ"
                | "aʊ"
                | "aɪ"
                | "ɛ"
                | "ɝ"
                | "eɪ"
                | "ɪ"
                | "iː"
                | "oʊ"
                | "ɔɪ"
                | "ʊ"
                | "uː"
        )
    }

    fn onset_verdict(&self, cluster: &[&Phone]) -> OnsetVerdict {
        OnsetVerdict {
            cluster: cluster.iter().map(|p| (*p).clone()).collect(),
            is_legal: true,
            reason: "permissive profile accepts all onsets".into(),
        }
    }

    fn is_legal_coda(&self, _cluster: &[&Phone]) -> bool {
        true
    }
}
