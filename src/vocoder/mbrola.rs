use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use crate::audio::frame::AudioFrame;
use crate::vocoder::{
    BackendCapabilities, BackendFamily, VocoderBackend, VocoderDescriptor, VocoderInput,
};
use crate::voice::articulator::{RenderPlan, SungBackendDetail, SungBackendKind};
use crate::{MbrolaPhone, MbrolaPitchTarget, MbrolaRenderer, PhoneTimedPlan};

const PITCH_TARGET_MID_PERCENT: u8 = 50;
const PITCH_TARGET_END_PERCENT: u8 = 100;
const PITCH_TARGET_MID_RATIO: f32 = 1.02;
const PITCH_TARGET_END_RATIO: f32 = 0.99;

pub struct MbrolaBackend {
    voice_path: PathBuf,
}

impl MbrolaBackend {
    pub fn new(voice_path: PathBuf) -> Self {
        Self { voice_path }
    }

    pub fn descriptor() -> VocoderDescriptor {
        VocoderDescriptor {
            id: "mbrola",
            family: BackendFamily::DiphoneTdPsola,
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
            sample_rate_hz: 16_000,
            backend_kind: Some(SungBackendKind::Mbrola),
            detail: Some(SungBackendDetail::PhoneTimed),
            notes: &[
                "MBROLA loads a real voice database and validates the shared phone-timed plan against its symbol map.",
                "Native MBROLA TD-PSOLA now matches shared phone durations and pitch targets while stitching real database waveforms without calling Klatt or the mbrola binary.",
            ],
        }
    }

    fn render_phone_timed(
        &self,
        targets: &[crate::voice::articulator::PhoneTimedRenderTarget],
    ) -> Result<Vec<AudioFrame>> {
        let renderer =
            MbrolaRenderer::from_voice_path(None, &self.voice_path).with_context(|| {
                format!("failed to load MBROLA voice {}", self.voice_path.display())
            })?;
        let mut phones = Vec::with_capacity(targets.len());
        for target in targets {
            let symbol = renderer
                .voice()
                .symbol_map
                .map_phone(&target.phone.ipa)
                .with_context(|| {
                    format!(
                        "failed to map sung phone `{}` to MBROLA voice `{}`",
                        target.phone.ipa,
                        renderer.voice().name
                    )
                })?;
            let duration_ms = target.duration_ms.clamp(1, u64::from(u32::MAX)) as u32;
            let pitch_targets = target
                .f0_hz
                .map(|hz| {
                    vec![
                        MbrolaPitchTarget { percent: 0, hz },
                        MbrolaPitchTarget {
                            percent: PITCH_TARGET_MID_PERCENT,
                            hz: hz * PITCH_TARGET_MID_RATIO,
                        },
                        MbrolaPitchTarget {
                            percent: PITCH_TARGET_END_PERCENT,
                            hz: hz * PITCH_TARGET_END_RATIO,
                        },
                    ]
                })
                .unwrap_or_default();
            phones.push(MbrolaPhone {
                symbol,
                duration_ms,
                pitch_targets,
            });
        }
        let phone_plan = PhoneTimedPlan::new(phones);
        renderer.render_phone_plan_to_frames(&phone_plan).context(
            "native MBROLA diphone renderer failed while using the shared Riper phone-timed plan",
        )
    }
}

impl VocoderBackend for MbrolaBackend {
    fn id(&self) -> &'static str {
        Self::descriptor().id
    }

    fn descriptor(&self) -> VocoderDescriptor {
        Self::descriptor()
    }

    fn render(&mut self, input: VocoderInput<'_>) -> Result<Vec<AudioFrame>> {
        match input {
            VocoderInput::RenderPlan(RenderPlan::PhoneTimed(targets)) => {
                self.render_phone_timed(targets)
            }
            VocoderInput::PhoneTimed(targets) => self.render_phone_timed(targets),
            _ => bail!("mbrola backend requires phone-timed input"),
        }
    }
}
