use anyhow::{Result, bail, ensure};

use crate::audio::frame::AudioFrame;
use crate::time::ExactTimestamp;
use crate::vocoder::{
    BackendCapabilities, BackendFamily, VocoderBackend, VocoderDescriptor, VocoderInput,
};
use crate::voice::articulator::{
    RenderPlan, SungBackendDetail, SungBackendKind, klatt_render_targets_from_phone_timed,
};
use crate::voice::tract::klatt::{KlattRenderConfig, render_phone_string};
use crate::voice::tract::targets::default_english_phone_targets;

pub struct KlattBackend;

impl KlattBackend {
    pub fn descriptor() -> VocoderDescriptor {
        VocoderDescriptor {
            id: "klatt",
            family: BackendFamily::FormantSourceFilter,
            capabilities: BackendCapabilities {
                accepts_phone_timed: true,
                accepts_partial_prosody: false,
                accepts_coarse_text: false,
                accepts_mel: false,
                accepts_mel_f0: false,
                honors_explicit_duration: true,
                honors_explicit_f0: true,
                honors_vibrato: true,
                streaming_safe: false,
            },
            sample_rate_hz: KlattRenderConfig::default().sample_rate,
            backend_kind: Some(SungBackendKind::Klatt),
            detail: Some(SungBackendDetail::PhoneTimed),
            notes: &[
                "Klatt consumes the shared phone-timed plan and nucleus-driven pitch sampling.",
                "Per-syllable vibrato now modulates sustained nucleus F0 in the trajectory layer.",
            ],
        }
    }

    fn render_phone_timed(
        neutral_targets: &[crate::voice::articulator::PhoneTimedRenderTarget],
    ) -> Result<Vec<AudioFrame>> {
        let config = KlattRenderConfig::default();
        let target_table = default_english_phone_targets();
        ensure!(
            !neutral_targets.is_empty(),
            "klatt backend received an empty phone-timed render plan"
        );
        let missing_phones: Vec<String> = neutral_targets
            .iter()
            .map(|target| target.phone.ipa.as_str())
            .filter(|ipa| !target_table.contains_key(*ipa))
            .map(str::to_string)
            .collect();
        ensure!(
            missing_phones.is_empty(),
            "klatt backend cannot render phone(s): {}",
            missing_phones.join(", ")
        );

        let klatt_targets = klatt_render_targets_from_phone_timed(neutral_targets, &target_table);
        let pcm = render_phone_string(&klatt_targets, &config);
        ensure!(!pcm.is_empty(), "klatt backend produced no audio");

        Ok(vec![AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: config.sample_rate,
            channels: 1,
            samples: pcm,
            voice_signatures: Vec::new(),
        }])
    }
}

impl VocoderBackend for KlattBackend {
    fn id(&self) -> &'static str {
        Self::descriptor().id
    }

    fn descriptor(&self) -> VocoderDescriptor {
        Self::descriptor()
    }

    fn render(&mut self, input: VocoderInput<'_>) -> Result<Vec<AudioFrame>> {
        match input {
            VocoderInput::RenderPlan(RenderPlan::PhoneTimed(targets)) => {
                Self::render_phone_timed(targets)
            }
            VocoderInput::PhoneTimed(targets) => Self::render_phone_timed(targets),
            _ => bail!("klatt backend requires phone-timed input"),
        }
    }
}
