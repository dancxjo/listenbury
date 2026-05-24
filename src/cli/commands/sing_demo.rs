use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};

use crate::cli::{SingDemoBackendOption, SingDemoCommand};

#[cfg(feature = "tts-riper")]
use crate::cli::model_paths::resolve_hifigan_model;
#[cfg(feature = "tts-piper")]
use crate::cli::model_paths::resolve_piper_voice;
#[cfg(feature = "tts-piper")]
use crate::cli::piper::resolve_piper_bin;
use listenbury::acoustic::{AcousticInput, AcousticModelBackend, SourceFilterAcousticModel};
use listenbury::audio::{frame::AudioFrame, write_wav};
use listenbury::linguistic::phonology::Phone;
use listenbury::prosody::note_target::{
    MidiNote, NoteArticulation, NoteDuration, NoteTarget, PitchTarget, TimePoint, Velocity,
};
use listenbury::prosody::singing::SungPhrase;
use listenbury::prosody::syllable::{PhoneSpan, SungSyllable, TimedPhoneRef};
use listenbury::prosody::vibrato::Vibrato;
use listenbury::vocoder::{
    SingDemoBackendSelector, VocoderConfig, VocoderInput, backend_for_option,
};
use listenbury::voice::articulator::{
    articulate, backend_detail_expectation, render_plan_for_backend,
};
use listenbury::voice::tract::targets::default_english_phone_targets;

pub(crate) fn run_sing_demo(command: SingDemoCommand) -> Result<()> {
    let phrase = build_ragtime_phrase()?;
    let plan = articulate(&phrase);
    let backend = command.selected_backend();
    let selector = selector_for_backend(backend);
    let config = vocoder_config_for_command(backend, &command)?;
    let mut renderer = backend_for_option(selector, config)?;
    let descriptor = renderer.descriptor();
    if backend == SingDemoBackendOption::Hifigan {
        println!(
            "sing-demo backend: {} (MelF0 via source-filter acoustic model)",
            descriptor.id
        );
        for note in descriptor.notes {
            println!("sing-demo note: {note}");
        }
        println!(
            "sing-demo hifigan mode: {}",
            if command.skip_gan {
                "skip-gan deterministic mel debug"
            } else {
                "ONNX HiFi-GAN"
            }
        );
        let mut acoustic = SourceFilterAcousticModel;
        let acoustic_track = acoustic
            .generate(AcousticInput::Singing(&phrase))
            .context("failed to generate source-filter mel/F0 track for sing-demo --hifigan")?;
        let frames = renderer
            .render(VocoderInput::MelF0 {
                mel: &acoustic_track.mel,
                f0_hz: &acoustic_track.f0_hz,
                voiced: &acoustic_track.voiced,
            })
            .context("failed to render sing-demo mel/F0 track with HiFi-GAN")?;
        let output_path = command
            .output_wav
            .unwrap_or_else(|| default_output_wav_path(backend));
        write_demo_wav(&output_path, &frames)?;
        return Ok(());
    }

    let backend_kind = descriptor.backend_kind.ok_or_else(|| {
        anyhow::anyhow!(
            "backend `{}` has no sing-demo render contract",
            descriptor.id
        )
    })?;
    let detail = descriptor
        .detail
        .unwrap_or_else(|| backend_detail_expectation(backend_kind));
    let target_table = default_english_phone_targets();
    let render_plan = render_plan_for_backend(backend_kind, &plan, 0.7, &target_table);
    println!("sing-demo backend: {} ({detail:?})", descriptor.id);

    for note in descriptor.notes {
        println!("sing-demo note: {note}");
    }

    let frames = renderer.render(VocoderInput::RenderPlan(&render_plan))?;

    let output_path = command
        .output_wav
        .unwrap_or_else(|| default_output_wav_path(backend));
    write_demo_wav(&output_path, &frames)?;
    Ok(())
}

fn default_output_wav_path(backend: SingDemoBackendOption) -> PathBuf {
    PathBuf::from(format!("out/hello-ragtime-{}.wav", backend.as_str()))
}

fn selector_for_backend(backend: SingDemoBackendOption) -> SingDemoBackendSelector {
    match backend {
        SingDemoBackendOption::Klatt => SingDemoBackendSelector::Klatt,
        SingDemoBackendOption::Riper => SingDemoBackendSelector::Riper,
        SingDemoBackendOption::Mbrola => SingDemoBackendSelector::Mbrola,
        SingDemoBackendOption::Piper => SingDemoBackendSelector::Piper,
        SingDemoBackendOption::Hifigan => SingDemoBackendSelector::Hifigan,
    }
}

fn vocoder_config_for_command(
    backend: SingDemoBackendOption,
    command: &SingDemoCommand,
) -> Result<VocoderConfig> {
    let mut config = VocoderConfig::default();
    anyhow::ensure!(
        backend == SingDemoBackendOption::Hifigan
            || (!command.skip_gan && command.hifigan_model.is_none()),
        "listenbury sing: --skip-gan and --hifigan-model only apply when the HiFi-GAN backend is selected"
    );
    if backend == SingDemoBackendOption::Mbrola {
        config.mbrola_voice = Some(resolve_sing_mbrola_voice(command.mbrola_voice.clone())?);
    }

    if backend == SingDemoBackendOption::Hifigan {
        config.skip_gan = command.skip_gan;
        #[cfg(feature = "tts-riper")]
        if !command.skip_gan {
            config.hifigan_model = Some(resolve_hifigan_model(command.hifigan_model.clone())?);
        }
        #[cfg(not(feature = "tts-riper"))]
        anyhow::ensure!(
            command.skip_gan,
            "listenbury sing --hifigan requires the `tts-riper` feature unless --skip-gan is used"
        );
    }

    if backend == SingDemoBackendOption::Piper {
        #[cfg(feature = "tts-piper")]
        {
            config.piper_voice = Some(resolve_piper_voice(command.piper_voice.clone())?);
            config.piper_bin = Some(resolve_piper_bin(command.piper_bin.clone())?);
            config.piper_timeout = Some(Duration::from_secs(30));
        }
    }

    Ok(config)
}

fn write_demo_wav(output_path: &Path, frames: &[AudioFrame]) -> Result<()> {
    if let Some(parent) = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output directory {}", parent.display()))?;
    }

    write_wav(output_path, frames)?;
    let sample_count: usize = frames.iter().map(|frame| frame.samples.len()).sum();
    println!(
        "Wrote ragtime sing-demo WAV: {} frames / {} samples -> {}",
        frames.len(),
        sample_count,
        output_path.display()
    );
    Ok(())
}

fn timed_phone(ipa: &str, start_ms: u64, end_ms: u64) -> Result<TimedPhoneRef> {
    TimedPhoneRef::new(
        Phone::new_ipa(ipa),
        TimePoint::from_millis(start_ms),
        TimePoint::from_millis(end_ms),
    )
    .context("build timed ragtime phone")
}

fn note_target(midi: u8, onset_ms: u64, duration_ms: u64) -> Result<NoteTarget> {
    let midi = MidiNote::new(midi).context("invalid ragtime demo midi note")?;
    Ok(NoteTarget {
        pitch: PitchTarget::new(midi),
        onset: TimePoint::from_millis(onset_ms),
        duration: NoteDuration::from_millis(duration_ms),
        velocity: Velocity::mezzo_forte(),
        articulation: NoteArticulation::Neutral,
    })
}

/// `(IPA phone, duration_ms)` for one timed phone segment inside a syllable.
type PhoneSegment<'a> = (&'a str, u64);

fn push_syllable(
    phrase: &mut SungPhrase,
    start_ms: u64,
    text: &str,
    phone_segments: &[PhoneSegment<'_>],
    constituent_ends: (usize, usize),
    midi: u8,
    vibrato: Option<Vibrato>,
) -> Result<u64> {
    let (onset_end, nucleus_end) = constituent_ends;
    let mut cursor = start_ms;
    let mut phones = Vec::with_capacity(phone_segments.len());
    for (ipa, duration_ms) in phone_segments {
        let end = cursor.saturating_add(*duration_ms);
        phones.push(timed_phone(ipa, cursor, end)?);
        cursor = end;
    }
    anyhow::ensure!(
        !phones.is_empty(),
        "ragtime demo syllable `{text}` had no phone segments"
    );

    let len = phones.len();
    let onset = PhoneSpan::new(0, onset_end).map_err(|error| {
        anyhow::anyhow!("invalid onset span for ragtime syllable `{text}`: {error:?}")
    })?;
    let nucleus = PhoneSpan::new(onset_end, nucleus_end).map_err(|error| {
        anyhow::anyhow!("invalid nucleus span for ragtime syllable `{text}`: {error:?}")
    })?;
    let coda = PhoneSpan::new(nucleus_end, len).map_err(|error| {
        anyhow::anyhow!("invalid coda span for ragtime syllable `{text}`: {error:?}")
    })?;
    let mut syllable = SungSyllable::new(
        text,
        phones,
        onset,
        nucleus,
        coda,
        None,
        Some(note_target(
            midi,
            start_ms,
            cursor.saturating_sub(start_ms),
        )?),
    )
    .map_err(|error| anyhow::anyhow!("build ragtime syllable `{text}` failed: {error:?}"))?;
    if let Some(vibrato) = vibrato {
        syllable = syllable.with_vibrato(Some(vibrato));
    }
    phrase
        .push(syllable)
        .map_err(|error| anyhow::anyhow!("append ragtime syllable `{text}` failed: {error:?}"))?;
    Ok(cursor)
}

fn build_ragtime_phrase() -> Result<SungPhrase> {
    let mut phrase = SungPhrase::new();
    let mut t = 0_u64;

    t = push_syllable(
        &mut phrase,
        t,
        "hel",
        &[("h", 40), ("ɛ", 120), ("l", 60)],
        (1, 2),
        60,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "lo",
        &[("l", 40), ("oʊ", 220)],
        (1, 2),
        64,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "my",
        &[("m", 60), ("ɑɪ", 180)],
        (1, 2),
        67,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "ba",
        &[("b", 40), ("eɪ", 160)],
        (1, 2),
        69,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "by",
        &[("b", 40), ("i", 220)],
        (1, 2),
        67,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "hel",
        &[("h", 40), ("ɛ", 120), ("l", 60)],
        (1, 2),
        60,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "lo",
        &[("l", 40), ("oʊ", 220)],
        (1, 2),
        64,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "my",
        &[("m", 60), ("ɑɪ", 180)],
        (1, 2),
        67,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "dar",
        &[("d", 40), ("ɑ", 120), ("ɹ", 60)],
        (1, 2),
        69,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "ling",
        &[("l", 40), ("ɪ", 100), ("ŋ", 80)],
        (1, 2),
        67,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "hel",
        &[("h", 40), ("ɛ", 120), ("l", 60)],
        (1, 2),
        60,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "lo",
        &[("l", 40), ("oʊ", 220)],
        (1, 2),
        64,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "my",
        &[("m", 60), ("ɑɪ", 180)],
        (1, 2),
        67,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "rag",
        &[("ɹ", 40), ("æ", 120), ("ɡ", 80)],
        (1, 2),
        65,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "time",
        &[("t", 50), ("ɑɪ", 170), ("m", 80)],
        (1, 2),
        64,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "gaaaaaal",
        &[("ɡ", 60), ("æ", 960), ("l", 100)],
        (1, 2),
        62,
        Some(Vibrato::new(
            5.0,                        // rate_hz
            25.0,                       // depth_cents
            Duration::from_millis(300), // onset
            Duration::from_millis(180), // ramp
            0.0,                        // phase
        )),
    )?;
    let _ = t;

    Ok(phrase)
}

impl SingDemoBackendOption {
    fn as_str(self) -> &'static str {
        match self {
            Self::Klatt => "klatt",
            Self::Riper => "riper",
            Self::Mbrola => "mbrola",
            Self::Piper => "piper",
            Self::Hifigan => "hifigan",
        }
    }
}

fn resolve_sing_mbrola_voice(explicit: Option<PathBuf>) -> Result<PathBuf> {
    explicit
        .or_else(|| std::env::var_os("LISTENBURY_MBROLA_VOICE").map(PathBuf::from))
        .or_else(|| std::env::var_os("MBROLA_VOICE").map(PathBuf::from))
        .or_else(|| {
            let fetched = PathBuf::from("data/mbrola/us3/us3");
            fetched.is_file().then_some(fetched)
        })
        .with_context(|| {
            "failed to find diphone voice; run `just fetch` or set LISTENBURY_MBROLA_VOICE / MBROLA_VOICE / --diphone-voice"
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use listenbury::vocoder::list_backends;
    use listenbury::voice::articulator::SungBackendKind;

    #[test]
    fn ragtime_phrase_keeps_final_gal_nucleus_long() {
        let phrase = build_ragtime_phrase().expect("ragtime phrase should build");
        let gal = phrase
            .syllables
            .last()
            .expect("ragtime phrase should include final gal");
        assert_eq!(gal.text, "gaaaaaal");
        assert!(
            gal.vibrato.is_some(),
            "final gal should carry a light vibrato"
        );

        let onset = &gal.phones[gal.onset.start..gal.onset.end];
        let nucleus = &gal.phones[gal.nucleus.start..gal.nucleus.end];
        let coda = &gal.phones[gal.coda.start..gal.coda.end];
        let onset_ms: u64 = onset
            .iter()
            .map(|phone| phone.end.millis.saturating_sub(phone.start.millis))
            .sum();
        let nucleus_ms: u64 = nucleus
            .iter()
            .map(|phone| phone.end.millis.saturating_sub(phone.start.millis))
            .sum();
        let coda_ms: u64 = coda
            .iter()
            .map(|phone| phone.end.millis.saturating_sub(phone.start.millis))
            .sum();
        assert!(
            nucleus_ms >= (onset_ms + coda_ms) * 4,
            "expected sustained final nucleus, got onset={onset_ms}ms nucleus={nucleus_ms}ms coda={coda_ms}ms"
        );
    }

    #[test]
    fn ragtime_shared_plan_preserves_sustained_final_nucleus() {
        let phrase = build_ragtime_phrase().expect("ragtime phrase should build");
        let plan = articulate(&phrase);
        let final_span = plan
            .syllables
            .last()
            .expect("shared plan should include final span");
        assert_eq!(final_span.text, "gaaaaaal");

        let onset_ms: u64 = plan.gestures.gestures[final_span.onset.start..final_span.onset.end]
            .iter()
            .map(|gesture| gesture.duration_ms)
            .sum();
        let nucleus_ms: u64 = plan.gestures.gestures
            [final_span.nucleus.start..final_span.nucleus.end]
            .iter()
            .map(|gesture| gesture.duration_ms)
            .sum();
        let coda_ms: u64 = plan.gestures.gestures[final_span.coda.start..final_span.coda.end]
            .iter()
            .map(|gesture| gesture.duration_ms)
            .sum();
        assert!(
            nucleus_ms >= (onset_ms + coda_ms) * 4,
            "shared plan should preserve long final nucleus, got onset={onset_ms} nucleus={nucleus_ms} coda={coda_ms}"
        );
    }

    #[test]
    fn backend_degradation_notes_are_explicit() {
        let backends = list_backends();
        let klatt = backends
            .iter()
            .find(|descriptor| descriptor.id == "klatt")
            .expect("klatt descriptor")
            .notes
            .join(" ");
        assert!(
            klatt.contains("vibrato now modulates sustained nucleus F0"),
            "klatt note should call out trajectory vibrato support"
        );

        let riper = backends
            .iter()
            .find(|descriptor| descriptor.id == "riper-klatt-fallback")
            .expect("riper fallback descriptor")
            .notes
            .join(" ");
        assert!(
            riper.contains("RiperKlattFallback"),
            "riper should advertise that it currently routes through the explicit fallback path"
        );
        assert!(
            riper.contains("Klatt source/filter"),
            "riper should advertise that its sung path is currently Klatt-backed"
        );

        let piper = backends
            .iter()
            .find(|descriptor| descriptor.id == "piper")
            .expect("piper descriptor")
            .notes
            .join(" ");
        assert!(
            piper.contains("ignores shared phones"),
            "piper degradation should be explicit"
        );
    }

    #[test]
    fn klatt_demo_plan_renders_non_empty_audio() {
        let phrase = build_ragtime_phrase().expect("ragtime phrase should build");
        let plan = articulate(&phrase);
        let target_table = default_english_phone_targets();
        let render_plan =
            render_plan_for_backend(SungBackendKind::Klatt, &plan, 0.7, &target_table);
        let mut backend =
            backend_for_option(SingDemoBackendSelector::Klatt, VocoderConfig::default())
                .expect("klatt backend");
        let frames = backend
            .render(VocoderInput::RenderPlan(&render_plan))
            .expect("klatt sing-demo should synthesize");
        let sample_count: usize = frames.iter().map(|frame| frame.samples.len()).sum();
        assert!(!frames.is_empty(), "klatt sing-demo should emit frames");
        assert!(
            sample_count > 0,
            "klatt sing-demo should emit non-empty audio"
        );
    }

    #[test]
    fn riper_demo_plan_renders_from_shared_phone_timing() {
        let phrase = build_ragtime_phrase().expect("ragtime phrase should build");
        let plan = articulate(&phrase);
        let target_table = default_english_phone_targets();
        assert_eq!(
            selector_for_backend(SingDemoBackendOption::Riper),
            SingDemoBackendSelector::Riper
        );
        assert_eq!(
            backend_detail_expectation(SungBackendKind::RiperKlattFallback),
            listenbury::voice::articulator::SungBackendDetail::PhoneTimedViaKlattFallback
        );
        let render_plan = render_plan_for_backend(
            SungBackendKind::RiperKlattFallback,
            &plan,
            0.7,
            &target_table,
        );
        let mut backend =
            backend_for_option(SingDemoBackendSelector::Riper, VocoderConfig::default())
                .expect("riper backend");
        let frames = backend
            .render(VocoderInput::RenderPlan(&render_plan))
            .expect(
                "riper sing-demo should synthesize through the Klatt-backed phone-timed fallback",
            );
        let sample_count: usize = frames.iter().map(|frame| frame.samples.len()).sum();
        assert!(!frames.is_empty(), "riper sing-demo should emit frames");
        assert!(
            sample_count > 0,
            "riper sing-demo should emit non-empty audio"
        );
    }

    #[test]
    fn hifigan_demo_renders_from_source_filter_mel_f0() {
        let phrase = build_ragtime_phrase().expect("ragtime phrase should build");
        assert_eq!(
            selector_for_backend(SingDemoBackendOption::Hifigan),
            SingDemoBackendSelector::Hifigan
        );
        let mut acoustic = SourceFilterAcousticModel;
        let acoustic_track = acoustic
            .generate(AcousticInput::Singing(&phrase))
            .expect("hifigan sing-demo should generate mel/F0");
        let mut backend = backend_for_option(
            SingDemoBackendSelector::Hifigan,
            VocoderConfig {
                skip_gan: true,
                ..VocoderConfig::default()
            },
        )
        .expect("hifigan backend");
        let frames = backend
            .render(VocoderInput::MelF0 {
                mel: &acoustic_track.mel,
                f0_hz: &acoustic_track.f0_hz,
                voiced: &acoustic_track.voiced,
            })
            .expect("hifigan sing-demo should synthesize from mel/F0");
        let sample_count: usize = frames.iter().map(|frame| frame.samples.len()).sum();
        assert!(!frames.is_empty(), "hifigan sing-demo should emit frames");
        assert!(
            sample_count > 0,
            "hifigan sing-demo should emit non-empty audio"
        );
    }

    #[test]
    fn mbrola_backend_kind_is_distinct_and_phone_timed() {
        let backends = list_backends();
        let mbrola = backends
            .iter()
            .find(|descriptor| descriptor.id == "mbrola")
            .expect("mbrola descriptor");
        let klatt = backends
            .iter()
            .find(|descriptor| descriptor.id == "klatt")
            .expect("klatt descriptor");
        assert_eq!(mbrola.backend_kind, Some(SungBackendKind::Mbrola));
        assert_ne!(
            mbrola.backend_kind, klatt.backend_kind,
            "MBROLA should no longer be identified as Klatt"
        );
    }

    #[test]
    fn sing_demo_contains_no_direct_synth_backend_impls() {
        let source = include_str!("sing_demo.rs");
        assert!(
            !source.contains("fn synthesize_klatt_from_plan"),
            "sing_demo should orchestrate through the vocoder layer"
        );
        assert!(
            !source.contains("fn synthesize_mbrola_from_plan"),
            "sing_demo should orchestrate through the vocoder layer"
        );
    }
}
