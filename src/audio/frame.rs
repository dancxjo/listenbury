use crate::audio::voice_signature::VoiceSignature;
use crate::time::ExactTimestamp;

/// A chunk of PCM audio together with its capture metadata.
///
/// The `voice_signatures` field carries **zero or more** speaker hypotheses for
/// this frame.  Frames that have not been analysed for speaker identity carry
/// an empty list, which is the default.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioFrame {
    pub captured_at: ExactTimestamp,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
    /// Zero or more voice-signature annotations for this frame.
    ///
    /// * 0 entries – silence, or the frame has not been processed for speaker
    ///   identity.
    /// * 1 entry – a single speaker hypothesis.
    /// * N entries – overlapping speakers or competing hypotheses.
    pub voice_signatures: Vec<VoiceSignature>,
}

impl Default for AudioFrame {
    fn default() -> Self {
        Self {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: 0,
            channels: 0,
            samples: Vec::new(),
            voice_signatures: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::voice_signature::{VoiceSignatureLabel, VoiceSignatureSource};

    #[test]
    fn default_frame_has_no_voice_signatures() {
        let frame = AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![0.0],
            voice_signatures: Vec::new(),
        };
        assert!(frame.voice_signatures.is_empty());
    }

    #[test]
    fn frame_can_carry_one_voice_signature() {
        let sig = VoiceSignature::new(VoiceSignatureLabel::User, 0.9, VoiceSignatureSource::Manual);
        let frame = AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![0.0],
            voice_signatures: vec![sig],
        };
        assert_eq!(frame.voice_signatures.len(), 1);
        assert_eq!(frame.voice_signatures[0].label, VoiceSignatureLabel::User);
    }

    #[test]
    fn frame_can_carry_multiple_voice_signatures() {
        let sig_user = VoiceSignature::new(
            VoiceSignatureLabel::User,
            0.8,
            VoiceSignatureSource::EmbeddingModel,
        );
        let sig_pete = VoiceSignature::new(
            VoiceSignatureLabel::PeteSelfVoice,
            0.6,
            VoiceSignatureSource::SelfVoiceSuppression,
        );
        let sig_unknown = VoiceSignature::new(
            VoiceSignatureLabel::Unknown,
            0.3,
            VoiceSignatureSource::Heuristic,
        );
        let frame = AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![0.0],
            voice_signatures: vec![sig_user, sig_pete, sig_unknown],
        };
        assert_eq!(frame.voice_signatures.len(), 3);
    }
}
