use std::path::PathBuf;
use std::time::Duration;

use anyhow::{bail, Result};

use crate::vocoder::bigvgan::BigVganBackend;
use crate::vocoder::diffwave::DiffwaveBackend;
use crate::vocoder::hifigan::HifiganBackend;
use crate::vocoder::klatt::KlattBackend;
use crate::vocoder::mbrola::MbrolaBackend;
use crate::vocoder::neural_onnx::RiperOnnxDirectBackend;
use crate::vocoder::piper::PiperBackend;
#[cfg(feature = "tts-piper")]
use crate::vocoder::piper::PiperBackendConfig;
use crate::vocoder::riper::RiperKlattFallbackBackend;
use crate::vocoder::source_filter::NeuralSourceFilterBackend;
use crate::vocoder::{VocoderBackend, VocoderDescriptor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SingDemoBackendSelector {
    Klatt,
    Riper,
    Mbrola,
    Piper,
}

#[derive(Debug, Clone, Default)]
pub struct VocoderConfig {
    pub mbrola_voice: Option<PathBuf>,
    pub piper_bin: Option<PathBuf>,
    pub piper_voice: Option<PathBuf>,
    pub piper_timeout: Option<Duration>,
}

pub fn backend_for_option(
    option: SingDemoBackendSelector,
    config: VocoderConfig,
) -> Result<Box<dyn VocoderBackend>> {
    match option {
        SingDemoBackendSelector::Klatt => Ok(Box::new(KlattBackend)),
        SingDemoBackendSelector::Riper => Ok(Box::new(RiperKlattFallbackBackend::new())),
        SingDemoBackendSelector::Mbrola => {
            let voice = config.mbrola_voice.ok_or_else(|| {
                anyhow::anyhow!("mbrola backend requires a resolved voice path in VocoderConfig")
            })?;
            Ok(Box::new(MbrolaBackend::new(voice)))
        }
        SingDemoBackendSelector::Piper => {
            #[cfg(feature = "tts-piper")]
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
            #[cfg(not(feature = "tts-piper"))]
            let piper = None;
            Ok(Box::new(PiperBackend::new(piper)))
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
        BigVganBackend::descriptor(),
        DiffwaveBackend::descriptor(),
        NeuralSourceFilterBackend::descriptor(),
    ]
}

pub fn backend_by_id(id: &str) -> Result<Box<dyn VocoderBackend>> {
    match id {
        "riper-onnx-direct" => Ok(Box::new(RiperOnnxDirectBackend)),
        "hifigan" => Ok(Box::new(HifiganBackend)),
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
        articulate, backend_detail_expectation, render_plan_for_backend, SungBackendKind,
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
        assert_eq!(mbrola.backend_kind, SungBackendKind::Mbrola);
        assert_eq!(
            backend_detail_expectation(mbrola.backend_kind),
            mbrola.detail
        );
    }

    #[test]
    fn piper_backend_rejects_phone_timed_input() {
        let mut backend =
            backend_for_option(SingDemoBackendSelector::Piper, VocoderConfig::default())
                .expect("piper backend");
        let err = backend
            .render(VocoderInput::PhoneTimed(&[]))
            .expect_err("piper should reject phone-timed input");
        assert!(err
            .to_string()
            .contains("accepts only degraded coarse text input"));
    }

    #[test]
    fn neural_stubs_are_registered_and_return_clear_errors() {
        for id in [
            "hifigan",
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
            assert!(err
                .to_string()
                .contains("registered but not implemented yet"));
        }
    }
}
