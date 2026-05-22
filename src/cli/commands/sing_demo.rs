use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};

use crate::cli::{SingDemoBackendOption, SingDemoCommand};

#[cfg(feature = "tts-piper")]
use crate::cli::model_paths::resolve_piper_voice;
#[cfg(feature = "tts-piper")]
use crate::cli::piper::{collect_tts_audio, piper_config_for_voice, resolve_piper_bin};

#[cfg(feature = "tts-piper")]
use listenbury::PiperTextToSpeech;
use listenbury::audio::{frame::AudioFrame, write_wav};
use listenbury::linguistic::phonology::Phone;
use listenbury::mouth::planner::{SpeechPlan, SpeechUnit};
use listenbury::mouth::tts::TextToSpeech;
use listenbury::prosody::note_target::{
    MidiNote, NoteArticulation, NoteDuration, NoteTarget, PitchTarget, TimePoint, Velocity,
};
use listenbury::prosody::singing::SungPhrase;
use listenbury::prosody::syllable::{PhoneSpan, SungSyllable, TimedPhoneRef};
use listenbury::prosody::vibrato::Vibrato;
use listenbury::time::ExactTimestamp;
use listenbury::voice::articulator::{
    RenderPlan, SungBackendKind, articulate, backend_detail_expectation, render_plan_for_backend,
};
use listenbury::voice::tract::klatt::{KlattRenderConfig, render_phone_string};
use listenbury::voice::tract::targets::default_english_phone_targets;

pub(crate) fn run_sing_demo(command: SingDemoCommand) -> Result<()> {
    let phrase = build_ragtime_phrase()?;
    let plan = articulate(&phrase);
    let backend = command.selected_backend();
    let backend_kind = render_kind_for_backend(backend);
    let detail = backend_detail_expectation(backend_kind);
    let target_table = default_english_phone_targets();
    let render_plan = render_plan_for_backend(backend_kind, &plan, 0.7, &target_table);
    println!("sing-demo backend: {backend:?} ({detail:?})");

    for note in backend_degradation_notes(backend) {
        println!("sing-demo note: {note}");
    }

    let frames = match backend {
        SingDemoBackendOption::Klatt => synthesize_klatt_from_plan(render_plan)?,
        SingDemoBackendOption::Riper => synthesize_riper_from_plan(render_plan, &command)?,
        SingDemoBackendOption::Piper => synthesize_piper_from_plan(render_plan, &command)?,
    };

    let output_path = command
        .output_wav
        .unwrap_or_else(|| default_output_wav_path(backend));
    write_demo_wav(&output_path, &frames)?;
    Ok(())
}

fn default_output_wav_path(backend: SingDemoBackendOption) -> PathBuf {
    PathBuf::from(format!("out/hello-ragtime-{}.wav", backend.as_str()))
}

fn synthesize_klatt_from_plan(plan: RenderPlan) -> Result<Vec<AudioFrame>> {
    let config = KlattRenderConfig::default();
    let target_table = default_english_phone_targets();
    let RenderPlan::PhoneTimed(targets) = plan else {
        anyhow::bail!("Klatt backend requires a phone-timed render plan");
    };
    anyhow::ensure!(
        !targets.is_empty(),
        "listenbury dev sing-demo --backend klatt produced an empty phone target plan"
    );
    let missing_phones: Vec<String> = targets
        .iter()
        .map(|target| target.phone.ipa.as_str())
        .filter(|ipa| !target_table.contains_key(*ipa))
        .map(str::to_string)
        .collect();
    anyhow::ensure!(
        missing_phones.is_empty(),
        "listenbury dev sing-demo --backend klatt cannot render phone(s): {}",
        missing_phones.join(", ")
    );

    let pcm = render_phone_string(&targets, &config);
    anyhow::ensure!(
        !pcm.is_empty(),
        "listenbury dev sing-demo --backend klatt produced no audio"
    );

    Ok(vec![AudioFrame {
        captured_at: ExactTimestamp::now(),
        sample_rate_hz: config.sample_rate,
        channels: 1,
        samples: pcm,
        voice_signatures: Vec::new(),
    }])
}

fn synthesize_riper_from_plan(
    plan: RenderPlan,
    _command: &SingDemoCommand,
) -> Result<Vec<AudioFrame>> {
    synthesize_klatt_from_plan(plan)
        .context("Riper sing-demo Klatt vocoder failed to render the shared phone-timed plan")
}

fn synthesize_piper_from_plan(
    plan: RenderPlan,
    command: &SingDemoCommand,
) -> Result<Vec<AudioFrame>> {
    #[cfg(not(feature = "tts-piper"))]
    {
        let _ = (plan, command);
        anyhow::bail!("listenbury dev sing-demo --backend piper requires the `tts-piper` feature");
    }

    #[cfg(feature = "tts-piper")]
    {
        #[cfg(feature = "tts-riper")]
        {
            synthesize_piper_text_from_plan(plan, command)
        }
        #[cfg(not(feature = "tts-riper"))]
        {
            synthesize_piper_text_from_plan(plan, command)
        }
    }
}

#[cfg(all(feature = "tts-piper", feature = "tts-riper"))]
fn synthesize_piper_text_from_plan(
    plan: RenderPlan,
    command: &SingDemoCommand,
) -> Result<Vec<AudioFrame>> {
    let text = text_from_text_render_plan(plan)?;
    let piper_voice = resolve_piper_voice(command.piper_voice.clone())?;
    let piper_bin = resolve_piper_bin(command.piper_bin.clone())?;
    let piper_config = piper_config_for_voice(piper_bin, piper_voice)?;

    let mut tts = PiperTextToSpeech::new(piper_config);
    tts.enqueue(SpeechPlan::from(SpeechUnit::FullTurn(text)))?;
    collect_tts_audio(&mut tts, Duration::from_secs(30))
}

#[cfg(all(feature = "tts-piper", not(feature = "tts-riper")))]
fn synthesize_piper_text_from_plan(
    plan: RenderPlan,
    command: &SingDemoCommand,
) -> Result<Vec<AudioFrame>> {
    let text = text_from_text_render_plan(plan)?;
    let piper_voice = resolve_piper_voice(command.piper_voice.clone())?;
    let piper_bin = resolve_piper_bin(command.piper_bin.clone())?;
    let piper_config = piper_config_for_voice(piper_bin, piper_voice)?;
    let mut tts = PiperTextToSpeech::new(piper_config);
    tts.enqueue(SpeechPlan::from(SpeechUnit::FullTurn(text)))?;
    collect_tts_audio(&mut tts, Duration::from_secs(30))
}

fn text_from_text_render_plan(plan: RenderPlan) -> Result<String> {
    match plan {
        RenderPlan::PartialProsody { text, .. } | RenderPlan::CoarseText { text, .. } => Ok(text),
        RenderPlan::PhoneTimed(_) => {
            anyhow::bail!("text backend requires a degraded text render plan")
        }
    }
}

fn backend_degradation_notes(backend: SingDemoBackendOption) -> &'static [&'static str] {
    match backend {
        SingDemoBackendOption::Klatt => &[
            "Klatt consumes the shared phone-timed plan and nucleus-driven pitch sampling.",
            "TODO: thread per-syllable vibrato from the shared plan into backend F0 modulation.",
        ],
        SingDemoBackendOption::Riper => &[
            "Riper sing-demo consumes the shared phone-timed plan before vocoder rendering.",
            "Riper's current sung vocoder path is Klatt source/filter until the ONNX path grows direct F0 and duration controls.",
        ],
        SingDemoBackendOption::Piper => &[
            "Piper currently consumes only coarse shared-plan text hints.",
            "Piper currently ignores shared phones, note timing detail, and vibrato.",
        ],
    }
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
    onset_end: usize,
    nucleus_end: usize,
    midi: u8,
    vibrato: Option<Vibrato>,
) -> Result<u64> {
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
        1,
        2,
        60,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "lo",
        &[("l", 40), ("oʊ", 220)],
        1,
        2,
        64,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "my",
        &[("m", 60), ("ɑɪ", 180)],
        1,
        2,
        67,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "ba",
        &[("b", 40), ("eɪ", 160)],
        1,
        2,
        69,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "by",
        &[("b", 40), ("i", 220)],
        1,
        2,
        67,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "hel",
        &[("h", 40), ("ɛ", 120), ("l", 60)],
        1,
        2,
        60,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "lo",
        &[("l", 40), ("oʊ", 220)],
        1,
        2,
        64,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "my",
        &[("m", 60), ("ɑɪ", 180)],
        1,
        2,
        67,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "dar",
        &[("d", 40), ("ɑ", 120), ("ɹ", 60)],
        1,
        2,
        69,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "ling",
        &[("l", 40), ("ɪ", 100), ("ŋ", 80)],
        1,
        2,
        67,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "hel",
        &[("h", 40), ("ɛ", 120), ("l", 60)],
        1,
        2,
        60,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "lo",
        &[("l", 40), ("oʊ", 220)],
        1,
        2,
        64,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "my",
        &[("m", 60), ("ɑɪ", 180)],
        1,
        2,
        67,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "rag",
        &[("ɹ", 40), ("æ", 120), ("ɡ", 80)],
        1,
        2,
        65,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "time",
        &[("t", 50), ("ɑɪ", 170), ("m", 80)],
        1,
        2,
        64,
        None,
    )?;
    t = push_syllable(
        &mut phrase,
        t,
        "gaaaaaal",
        &[("ɡ", 60), ("æ", 960), ("l", 100)],
        1,
        2,
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

fn render_kind_for_backend(backend: SingDemoBackendOption) -> SungBackendKind {
    match backend {
        SingDemoBackendOption::Klatt | SingDemoBackendOption::Riper => SungBackendKind::Klatt,
        SingDemoBackendOption::Piper => SungBackendKind::Piper,
    }
}

impl SingDemoBackendOption {
    fn as_str(self) -> &'static str {
        match self {
            Self::Klatt => "klatt",
            Self::Riper => "riper",
            Self::Piper => "piper",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let klatt = backend_degradation_notes(SingDemoBackendOption::Klatt).join(" ");
        assert!(
            klatt.contains("TODO: thread per-syllable vibrato"),
            "klatt degradation should call out current vibrato limitation"
        );

        let riper = backend_degradation_notes(SingDemoBackendOption::Riper).join(" ");
        assert!(
            riper.contains("shared phone-timed plan"),
            "riper should advertise that it uses the shared sung plan"
        );

        let piper = backend_degradation_notes(SingDemoBackendOption::Piper).join(" ");
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
        let frames =
            synthesize_klatt_from_plan(render_plan).expect("klatt sing-demo should synthesize");
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
        let render_plan = render_plan_for_backend(
            render_kind_for_backend(SingDemoBackendOption::Riper),
            &plan,
            0.7,
            &target_table,
        );
        let frames = synthesize_riper_from_plan(
            render_plan,
            &SingDemoCommand {
                backend: None,
                riper: true,
                klatt: false,
                output_wav: None,
                piper_bin: None,
                piper_voice: None,
            },
        )
        .expect("riper sing-demo should synthesize through the shared phone-timed plan");
        let sample_count: usize = frames.iter().map(|frame| frame.samples.len()).sum();
        assert!(!frames.is_empty(), "riper sing-demo should emit frames");
        assert!(
            sample_count > 0,
            "riper sing-demo should emit non-empty audio"
        );
    }
}
