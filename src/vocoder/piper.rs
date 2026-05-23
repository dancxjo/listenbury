use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Result, bail};

use crate::audio::frame::AudioFrame;
use crate::vocoder::{
    BackendCapabilities, BackendFamily, VocoderBackend, VocoderDescriptor, VocoderInput,
};
use crate::voice::articulator::{RenderPlan, SungBackendDetail, SungBackendKind};

#[cfg(feature = "tts-piper")]
use crate::{PiperConfig, PiperTextToSpeech};
#[cfg(feature = "tts-piper")]
use crate::{
    mouth::planner::{SpeechPlan, SpeechUnit},
    mouth::tts::TextToSpeech,
};
#[cfg(feature = "tts-piper")]
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct PiperBackendConfig {
    pub piper_bin: PathBuf,
    pub piper_voice: PathBuf,
    pub timeout: Duration,
}

pub struct PiperBackend {
    config: Option<PiperBackendConfig>,
}

impl PiperBackend {
    pub fn new(config: Option<PiperBackendConfig>) -> Self {
        Self { config }
    }

    pub fn descriptor() -> VocoderDescriptor {
        VocoderDescriptor {
            id: "piper",
            family: BackendFamily::TextTtsProcess,
            capabilities: BackendCapabilities {
                accepts_phone_timed: false,
                accepts_partial_prosody: false,
                accepts_coarse_text: true,
                accepts_mel: false,
                accepts_mel_f0: false,
                honors_explicit_duration: false,
                honors_explicit_f0: false,
                honors_vibrato: false,
                streaming_safe: true,
            },
            sample_rate_hz: 22_050,
            backend_kind: SungBackendKind::Piper,
            detail: SungBackendDetail::CoarseHintsOnly,
            notes: &[
                "Piper currently consumes only coarse shared-plan text hints.",
                "Piper currently ignores shared phones, note timing detail, and vibrato.",
            ],
        }
    }

    fn text_from_input(input: VocoderInput<'_>) -> Result<String> {
        match input {
            VocoderInput::RenderPlan(RenderPlan::CoarseText { text, .. }) => Ok(text.clone()),
            VocoderInput::RenderPlan(RenderPlan::PhoneTimed(_)) | VocoderInput::PhoneTimed(_) => {
                bail!(
                    "piper backend accepts only degraded coarse text input; phone-timed input is unsupported"
                )
            }
            VocoderInput::CoarseText { text, .. } => Ok(text.to_string()),
            _ => bail!("piper backend accepts only coarse text input"),
        }
    }

    #[cfg(feature = "tts-piper")]
    fn collect_tts_audio(
        tts: &mut impl TextToSpeech,
        timeout: Duration,
    ) -> Result<Vec<AudioFrame>> {
        let deadline = Instant::now() + timeout;
        let quiet_after_audio = Duration::from_millis(100);
        let mut frames = Vec::new();
        let mut last_audio_at = None;

        while Instant::now() < deadline {
            let new_frames = tts.poll_audio()?;
            if new_frames.is_empty() {
                if let Some(last_audio_at) = last_audio_at
                    && Instant::now().duration_since(last_audio_at) >= quiet_after_audio
                {
                    break;
                }
            } else {
                frames.extend(new_frames);
                last_audio_at = Some(Instant::now());
            }

            std::thread::sleep(Duration::from_millis(10));
        }

        if frames.is_empty() {
            bail!("Piper produced no audio frames before timeout");
        }

        Ok(frames)
    }
}

impl VocoderBackend for PiperBackend {
    fn id(&self) -> &'static str {
        Self::descriptor().id
    }

    fn descriptor(&self) -> VocoderDescriptor {
        Self::descriptor()
    }

    fn render(&mut self, input: VocoderInput<'_>) -> Result<Vec<AudioFrame>> {
        let text = Self::text_from_input(input)?;

        #[cfg(not(feature = "tts-piper"))]
        {
            let _ = text;
            bail!("vocoder `piper` is registered but unavailable: build with feature `tts-piper`")
        }

        #[cfg(feature = "tts-piper")]
        {
            let config = self.config.as_ref().ok_or_else(|| {
                anyhow::anyhow!("piper backend requires explicit piper_bin and piper_voice config")
            })?;
            let mut piper_config =
                PiperConfig::new(config.piper_bin.clone(), config.piper_voice.clone());
            let inferred_config_path = config.piper_voice.with_extension("onnx.json");
            if inferred_config_path.exists() {
                piper_config.config_path = Some(inferred_config_path);
            }
            let mut tts = PiperTextToSpeech::new(piper_config);
            tts.enqueue(SpeechPlan::from(SpeechUnit::FullTurn(text)))?;
            Self::collect_tts_audio(&mut tts, config.timeout)
        }
    }
}
