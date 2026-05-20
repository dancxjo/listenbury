use crate::linguistic::{
    phoneme::PhonemeText,
    pronounce::{OrthographyToPhonemes, PhonologyError},
    variety::LinguisticVariety,
};

/// A pronunciation service that bridges free-form text to a neutral phoneme
/// representation.
///
/// This wraps an [`OrthographyToPhonemes`] realizer together with a
/// [`LinguisticVariety`] context, providing a simple text-in / [`PhonemeText`]-out
/// interface for use by synthesis backends and other consumers of the shared
/// phonological substrate.
///
/// # Construction
///
/// Use [`PronunciationService::new`] to supply a custom realizer and variety.
/// For the native Piper path, [`SimpleEnglishG2p`] with a General American
/// English variety is the recommended default.
///
/// [`SimpleEnglishG2p`]: crate::mouth::piper_native::SimpleEnglishG2p
pub struct PronunciationService {
    pronouncer: Box<dyn OrthographyToPhonemes + Send + Sync>,
    variety: LinguisticVariety,
}

impl PronunciationService {
    /// Create a new `PronunciationService` backed by the given pronouncer and
    /// linguistic variety context.
    pub fn new(
        pronouncer: impl OrthographyToPhonemes + Send + Sync + 'static,
        variety: LinguisticVariety,
    ) -> Self {
        Self {
            pronouncer: Box::new(pronouncer),
            variety,
        }
    }

    /// Realize `text` into a neutral [`PhonemeText`] representation using the
    /// configured pronouncer and variety.
    pub fn realize_text(&self, text: &str) -> Result<PhonemeText, PhonologyError> {
        self.pronouncer.realize_text(&self.variety, text)
    }

    /// Return the [`LinguisticVariety`] used by this service.
    pub fn variety(&self) -> &LinguisticVariety {
        &self.variety
    }
}

impl std::fmt::Debug for PronunciationService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PronunciationService")
            .field("variety", &self.variety)
            .finish_non_exhaustive()
    }
}

/// Build the default English (US) variety used by [`PronunciationService`] when
/// backed by the native Piper path.
#[cfg(feature = "tts-piper-native")]
pub fn default_english_variety() -> LinguisticVariety {
    use crate::linguistic::variety::{Phonology, VarietyTag};
    LinguisticVariety::tagged(
        VarietyTag::new("en_US"),
        "English (US)",
        Phonology::new("General American"),
    )
}
