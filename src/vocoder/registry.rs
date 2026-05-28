use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Result, bail};

use crate::vocoder::bigvgan::BigVganBackend;
use crate::vocoder::diffwave::DiffwaveBackend;
use crate::vocoder::hifigan::HifiganBackend;
use crate::vocoder::klatt::KlattBackend;
use crate::vocoder::mbrola::MbrolaBackend;
use crate::vocoder::mel_debug::MelDebugRendererBackend;
use crate::vocoder::neural_onnx::RiperOnnxDirectBackend;
use crate::vocoder::piper::{PiperBackend, PiperBackendConfig};
use crate::vocoder::riper::RiperKlattFallbackBackend;
use crate::vocoder::source_filter::NeuralSourceFilterBackend;
use crate::vocoder::{SpeechSynthesizer, VocoderDescriptor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SingDemoBackendSelector {
    Klatt,
    Riper,
    Mbrola,
    Piper,
    Hifigan,
}

#[derive(Debug, Clone, Default)]
pub struct VocoderConfig {
    pub mbrola_voice: Option<PathBuf>,
    pub piper_bin: Option<PathBuf>,
    pub piper_voice: Option<PathBuf>,
    pub piper_timeout: Option<Duration>,
    pub hifigan_model: Option<PathBuf>,
    pub skip_gan: bool,
}

pub fn backend_for_option(
    option: SingDemoBackendSelector,
    config: VocoderConfig,
) -> Result<Box<dyn SpeechSynthesizer>> {
    match option {
        SingDemoBackendSelector::Klatt => Ok(Box::new(KlattBackend)),
        SingDemoBackendSelector::Riper => Ok(Box::new(RiperKlattFallbackBackend::new())),
        SingDemoBackendSelector::Mbrola => {
            let voice = config.mbrola_voice.ok_or_else(|| {
                anyhow::anyhow!(
                    "MBROLA backend requires a voice path to be specified in the configuration"
                )
            })?;
            Ok(Box::new(MbrolaBackend::new(voice)))
        }
        SingDemoBackendSelector::Piper => {
            let piper = {
                let piper_bin = config
                    .piper_bin
                    .ok_or_else(|| anyhow::anyhow!("piper backend requires a piper binary path"))?;
                let piper_voice = config
                    .piper_voice
                    .ok_or_else(|| anyhow::anyhow!("piper backend requires a piper voice path"))?;
                Some(PiperBackendConfig {
                    piper_bin,
                    piper_voice,
                    timeout: config.piper_timeout.unwrap_or(Duration::from_secs(30)),
                })
            };
            Ok(Box::new(PiperBackend::new(piper)))
        }
        SingDemoBackendSelector::Hifigan => {
            if config.skip_gan {
                return Ok(Box::new(MelDebugRendererBackend::new()));
            }

            #[cfg(feature = "piper-compat")]
            {
                let model_path = config.hifigan_model.ok_or_else(|| {
                    anyhow::anyhow!(
                        "HiFi-GAN model is unavailable: specify --hifigan-model or install the default hifigan-speecht5 model; use --skip-gan only for the mel debug renderer"
                    )
                })?;
                Ok(Box::new(HifiganBackend::load(model_path)?))
            }

            #[cfg(not(feature = "piper-compat"))]
            {
                bail!(
                    "hifigan backend requires the `piper-compat` feature unless --skip-gan is used"
                )
            }
        }
    }
}

pub fn list_backends() -> Vec<VocoderDescriptor> {
    vec![
        KlattBackend::descriptor(),
        RiperKlattFallbackBackend::descriptor(),
        MbrolaBackend::descriptor(),
        PiperBackend::descriptor(),
        RiperOnnxDirectBackend::descriptor(),
        HifiganBackend::descriptor(),
        MelDebugRendererBackend::descriptor(),
        BigVganBackend::descriptor(),
        DiffwaveBackend::descriptor(),
        NeuralSourceFilterBackend::descriptor(),
    ]
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn backend_by_id(id: &str) -> Result<Box<dyn SpeechSynthesizer>> {
    match id {
        "riper-onnx-direct" => Ok(Box::new(RiperOnnxDirectBackend)),
        "hifigan" => bail!(
            "HiFi-GAN model is unavailable: construct the hifigan backend with a model path; use `mel-debug-renderer` only when that debug renderer is explicitly selected"
        ),
        "mel-debug-renderer" => Ok(Box::new(MelDebugRendererBackend::new())),
        "bigvgan" => Ok(Box::new(BigVganBackend)),
        "diffwave" => Ok(Box::new(DiffwaveBackend)),
        "source-filter-neural" => Ok(Box::new(NeuralSourceFilterBackend)),
        _ => bail!("unknown vocoder backend id `{id}`"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::frame::AudioFrame;
    use crate::prosody::singing::SungPhrase;
    use crate::prosody::syllable::{PhoneSpan, SungSyllable, TimedPhoneRef};
    use crate::prosody::{
        note_target::{
            MidiNote, NoteArticulation, NoteDuration, NoteTarget, PitchTarget, TimePoint, Velocity,
        },
        vibrato::Vibrato,
    };
    use crate::vocoder::VocoderInput;
    use crate::voice::articulator::{
        SungBackendKind, articulate, backend_detail_expectation, render_plan_for_backend,
    };
    use crate::voice::tract::targets::default_english_phone_targets;
    use std::time::Duration;

    fn ragtime_phrase() -> SungPhrase {
        fn timed_phone(ipa: &str, start_ms: u64, end_ms: u64) -> TimedPhoneRef {
            TimedPhoneRef::new(
                crate::linguistic::phonology::Phone::new_ipa(ipa),
                TimePoint::from_millis(start_ms),
                TimePoint::from_millis(end_ms),
            )
            .expect("timed phone")
        }

        fn note_target(midi: u8, onset_ms: u64, duration_ms: u64) -> NoteTarget {
            NoteTarget {
                pitch: PitchTarget::new(MidiNote::new(midi).expect("midi")),
                onset: TimePoint::from_millis(onset_ms),
                duration: NoteDuration::from_millis(duration_ms),
                velocity: Velocity::mezzo_forte(),
                articulation: NoteArticulation::Neutral,
            }
        }

        fn push(
            phrase: &mut SungPhrase,
            start: u64,
            text: &str,
            phones: &[(&str, u64)],
            midi: u8,
            vibrato: Option<Vibrato>,
        ) -> u64 {
            let mut t = start;
            let refs = phones
                .iter()
                .map(|(ipa, dur)| {
                    let end = t + dur;
                    let p = timed_phone(ipa, t, end);
                    t = end;
                    p
                })
                .collect::<Vec<_>>();
            let mut syllable = SungSyllable::new(
                text,
                refs,
                PhoneSpan::new(0, 1).unwrap(),
                PhoneSpan::new(1, 2).unwrap(),
                PhoneSpan::new(2, 2).unwrap_or_else(|_| PhoneSpan::new(1, 2).unwrap()),
                None,
                Some(note_target(midi, start, t - start)),
            )
            .expect("syllable");
            if let Some(vibrato) = vibrato {
                syllable = syllable.with_vibrato(Some(vibrato));
            }
            phrase.push(syllable).expect("push syllable");
            t
        }

        let mut phrase = SungPhrase::new();
        let mut t = 0;
        t = push(&mut phrase, t, "hel", &[("h", 40), ("ɛ", 120)], 60, None);
        t = push(&mut phrase, t, "lo", &[("l", 40), ("oʊ", 220)], 64, None);
        let _ = push(
            &mut phrase,
            t,
            "gaaaaaal",
            &[("ɡ", 60), ("æ", 960)],
            62,
            Some(Vibrato::new(
                5.0,
                25.0,
                Duration::from_millis(300),
                Duration::from_millis(180),
                0.0,
            )),
        );
        phrase
    }

    fn frame_samples(frames: &[AudioFrame]) -> usize {
        frames.iter().map(|frame| frame.samples.len()).sum()
    }

    fn synthetic_mel_frames() -> Vec<crate::vocoder::MelFrame> {
        (0..6)
            .map(|frame_index| crate::vocoder::MelFrame {
                bins: (0..80)
                    .map(|bin_index| {
                        let envelope = 1.0 - (bin_index as f32 / 80.0);
                        ((0.12 + frame_index as f32 * 0.01) * envelope.max(0.05)).ln()
                    })
                    .collect(),
            })
            .collect()
    }

    #[test]
    fn klatt_backend_renders_non_empty_audio() {
        let phrase = ragtime_phrase();
        let plan = articulate(&phrase);
        let target_table = default_english_phone_targets();
        let render_plan =
            render_plan_for_backend(SungBackendKind::Klatt, &plan, 0.7, &target_table);

        let mut backend =
            backend_for_option(SingDemoBackendSelector::Klatt, VocoderConfig::default())
                .expect("klatt backend");
        let frames = backend
            .render(VocoderInput::RenderPlan(&render_plan))
            .expect("klatt render");
        assert!(!frames.is_empty());
        assert!(frame_samples(&frames) > 0);
    }

    #[test]
    fn riper_fallback_backend_renders_and_identifies_itself() {
        let phrase = ragtime_phrase();
        let plan = articulate(&phrase);
        let target_table = default_english_phone_targets();
        let render_plan = render_plan_for_backend(
            SungBackendKind::RiperKlattFallback,
            &plan,
            0.7,
            &target_table,
        );

        let mut backend =
            backend_for_option(SingDemoBackendSelector::Riper, VocoderConfig::default())
                .expect("riper fallback backend");
        assert_eq!(backend.id(), "riper-klatt-fallback");
        let frames = backend
            .render(VocoderInput::RenderPlan(&render_plan))
            .expect("riper fallback render");
        assert!(frame_samples(&frames) > 0);
    }

    #[test]
    fn mbrola_backend_descriptor_is_distinct_from_klatt() {
        let backends = list_backends();
        let mbrola = backends
            .iter()
            .find(|d| d.id == "mbrola")
            .expect("mbrola descriptor");
        let klatt = backends
            .iter()
            .find(|d| d.id == "klatt")
            .expect("klatt descriptor");
        assert_ne!(mbrola.backend_kind, klatt.backend_kind);
        assert_eq!(mbrola.backend_kind, Some(SungBackendKind::Mbrola));
        assert_eq!(
            Some(backend_detail_expectation(
                mbrola
                    .backend_kind
                    .expect("mbrola should have a render contract"),
            )),
            mbrola.detail
        );
    }

    #[test]
    fn piper_backend_rejects_phone_timed_input() {
        let mut backend = PiperBackend::new(None);
        let err = backend
            .render(VocoderInput::PhoneTimed(&[]))
            .expect_err("piper should reject phone-timed input");
        assert!(
            err.to_string()
                .contains("accepts only degraded coarse text input")
        );
    }

    #[test]
    fn mel_debug_renderer_backend_renders_mel_f0_audio_when_selected() {
        let mel = synthetic_mel_frames();
        let f0_hz = vec![220.0, 225.0, 230.0, 235.0, 240.0, 245.0];
        let voiced = vec![true, true, true, true, true, true];
        let mut backend =
            backend_by_id("mel-debug-renderer").expect("mel debug backend construction");

        let frames = backend
            .render(VocoderInput::MelF0 {
                mel: &mel,
                f0_hz: &f0_hz,
                voiced: &voiced,
            })
            .expect("mel debug render");

        assert_eq!(backend.id(), "mel-debug-renderer");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].sample_rate_hz, 16_000);
        assert_eq!(frames[0].channels, 1);
        assert_eq!(frame_samples(&frames), mel.len() * 256);
        assert!(frames[0].samples.iter().any(|sample| sample.abs() > 0.0));
    }

    #[test]
    fn hifigan_backend_by_id_reports_missing_model() {
        let err = match backend_by_id("hifigan") {
            Ok(_) => panic!("hifigan requires a model path"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("HiFi-GAN model is unavailable"));
    }

    #[test]
    fn mel_debug_renderer_backend_rejects_mismatched_f0_tracks() {
        let mel = synthetic_mel_frames();
        let f0_hz = vec![220.0];
        let voiced = vec![true; mel.len()];
        let mut backend =
            backend_by_id("mel-debug-renderer").expect("mel debug backend construction");

        let err = backend
            .render(VocoderInput::MelF0 {
                mel: &mel,
                f0_hz: &f0_hz,
                voiced: &voiced,
            })
            .expect_err("mel debug renderer should validate F0 length");

        assert!(err.to_string().contains("F0 values"));
    }

    #[test]
    fn remaining_neural_stubs_are_registered_and_return_clear_errors() {
        for id in [
            "bigvgan",
            "diffwave",
            "source-filter-neural",
            "riper-onnx-direct",
        ] {
            let descriptor = list_backends()
                .into_iter()
                .find(|d| d.id == id)
                .expect("stub descriptor should be registered");
            assert!(
                descriptor.capabilities.accepts_mel
                    || descriptor.capabilities.accepts_mel_f0
                    || descriptor.capabilities.accepts_partial_prosody
            );
            let mut backend = backend_by_id(id).expect("stub backend construction");
            let err = backend
                .render(VocoderInput::CoarseText {
                    text: "hello",
                    ssml_hint: None,
                })
                .expect_err("stub should return unimplemented error");
            assert!(
                err.to_string()
                    .contains("registered but not implemented yet")
            );
        }
    }
}
