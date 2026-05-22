//! Phonotactic profiles for IPA-based syllabification.
//!
//! A [`PhonotacticProfile`] encapsulates the rules that determine which
//! [`Phone`] sequences are legal onsets or codas in a given linguistic variety.
//! The syllabifier queries a profile rather than embedding hard-coded rules,
//! so phonotactic assumptions are separable from the algorithm and can be
//! swapped for a different English variety or a completely different language.
//!
//! # Input phone layer
//!
//! Phonotactic profiles operate exclusively over **broad phonemic phones**
//! ([`PhoneDecompositionPolicy::KeepPhonemic`]).  Affricates such as `/tʃ/`
//! (ARPABET `CH`) and `/dʒ/` (ARPABET `JH`) are single phonotactic units;
//! decomposing them into `[t, ʃ]` or `[d, ʒ]` before consulting the tables
//! would make multi-phone coda entries like `[n, tʃ]` and `[n, dʒ]`
//! invisible.  Diphthong nuclei (`/oʊ/`, `/aɪ/`, …) are likewise kept whole.
//!
//! Decomposition for singing or acoustic rendering is applied **after**
//! syllabification, via
//! [`crate::prosody::syllable::SungSyllable::with_decomposition_policy`].
//!
//! All profile methods receive **[`Phone`] references** from the phonology
//! layer; the `phone.ipa` field carries the IPA surface form, and phone
//! comparisons respect the variety's [`PhoneEqualityOptions`] so that
//! aspiration and other diacritic detail can be treated as non-contrastive.
//!
//! The primary English implementation is [`EnglishPhonotactics`], constructed
//! with [`EnglishPhonotactics::for_variety`].  A maximally permissive
//! [`PermissiveProfile`] is also provided for testing.
//!
//! # Example
//!
//! ```
//! use listenbury::prosody::phonotactics::{EnglishPhonotactics, EnglishVariety, PhonotacticProfile};
//! use listenbury::linguistic::phonology::Phone;
//!
//! let profile = EnglishPhonotactics::for_variety(EnglishVariety::GeneralAmerican);
//!
//! // /stɹ/ is a legal English onset (e.g. "strong", "stream")
//! let stɹ = [Phone::mapped("s"), Phone::mapped("t"), Phone::mapped("ɹ")];
//! assert!(profile.is_legal_onset(&stɹ.iter().collect::<Vec<_>>()));
//!
//! // /tl/ is not a legal onset in General American English
//! let tl = [Phone::mapped("t"), Phone::mapped("l")];
//! assert!(!profile.is_legal_onset(&tl.iter().collect::<Vec<_>>()));
//!
//! // Vowels are nuclei
//! assert!(profile.is_nucleus(&Phone::mapped("ɛ")));
//! assert!(!profile.is_nucleus(&Phone::mapped("t")));
//! ```

pub use crate::linguistic::inventory::general_american_english;
pub use crate::linguistic::variety::EnglishVariety;

mod english;
mod permissive;
mod profile;
pub(crate) mod tables;

pub use english::EnglishPhonotactics;
pub use permissive::PermissiveProfile;
pub use profile::{OnsetVerdict, PhonotacticProfile};

#[cfg(test)]
mod tests;
