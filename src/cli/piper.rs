#[cfg(feature = "audio-cpal")]
use crate::cli::commands::{play_audio_frame_stream, play_audio_frames};
use crate::cli::model_paths::resolve_piper_voice;
#[cfg(all(feature = "asr-whisper", feature = "piper-compat"))]
use crate::cli::model_paths::resolve_whisper_model;
#[cfg(feature = "piper-compat")]
use crate::cli::model_paths::{resolve_hifigan_model, resolve_speecht5_acoustic_dir};
use crate::cli::{EchoCommand, RiperCompareCommand, SayCommand};
use anyhow::{Context, Result};
#[cfg(all(feature = "asr-whisper", feature = "piper-compat"))]
use listenbury::WhisperSpeechRecognizer;
use listenbury::audio::frame::AudioFrame;
#[cfg(all(feature = "asr-whisper", feature = "piper-compat"))]
use listenbury::audio::read_wav_as_whisper_frames;
#[cfg(all(feature = "asr-whisper", feature = "piper-compat"))]
use listenbury::audio::streaming_prosody::StreamingProsodyAnalyzer;
use listenbury::audio::write_wav;
#[cfg(test)]
use listenbury::audio::write_wav_bytes;
use listenbury::linguistic::phonology::{Phone, PhoneString};
#[cfg(feature = "piper-compat")]
use listenbury::mouth::backend::TtsBackend;
#[cfg(feature = "piper-compat")]
use listenbury::mouth::piper::{PiperBackendPreference, ProcessPiperBackend};
use listenbury::mouth::planner::{MouthSyntheticPlan, SyntheticUnit};
#[cfg(feature = "piper-compat")]
use listenbury::mouth::riper::phoneme::espeak_compatible_sequence;
#[cfg(feature = "piper-compat")]
use listenbury::mouth::riper::{
    BreathGroupProsodyPlanner, PhonemeProsodyCandidate, PhonemeProsodyCandidateEvent,
    PiperIdSequence, PiperPhoneme, PiperPhonemeSequence, PiperTextIdTrace, PiperVoiceConfig,
    ProsodyEnergy, ProsodyList, ProsodyOp, ProsodyPitchShape, ProsodyTarget, RiperBackend,
    RiperProsodyRealization, SentenceAnalysis, SimpleEnglishG2p, SyntacticLinkKind,
    SyntacticLinkParse,
};
#[cfg(all(feature = "asr-whisper", feature = "piper-compat"))]
use listenbury::mouth::riper::{EchoComparisonRecord, EchoProsodyObservation, EchoProsodyPlan};
use listenbury::mouth::tts::TextToSpeech;
use listenbury::speech::loom::{CurrentBackendGraphView, CurrentSayBackendKind, SpeechLoom};
use listenbury::speech::phone_plan::PhonePlan;
#[cfg(all(feature = "asr-whisper", feature = "piper-compat"))]
use listenbury::speech::recognizer::SpeechRecognizer;
use listenbury::time::ExactTimestamp;
#[cfg(feature = "piper-compat")]
use listenbury::vocoder::{
    HifiganBackend, MelDebugRendererBackend, SpeechSynthesizer, VocoderInput,
};
#[cfg(feature = "piper-compat")]
use listenbury::voice::articulator::PhoneTimedRenderTarget;
#[cfg(feature = "piper-compat")]
use listenbury::voice::diphone::{DiphoneCache, DiphoneVoiceManifest, NeuralDiphoneProvider};
#[cfg(feature = "piper-compat")]
use listenbury::voice::mbrola::render_phone_plan_with_diphone_provider_to_frames;
#[cfg(feature = "piper-compat")]
use listenbury::voice::mbrola::{MbrolaPhone, MbrolaPitchTarget, MbrolaRenderer, PhoneTimedPlan};
use listenbury::voice::tract::klatt::{KlattRenderConfig, render_phone_string};
use listenbury::voice::tract::targets::{
    default_english_phone_targets, phone_render_targets_from_string,
};
#[cfg(feature = "piper-compat")]
use listenbury::{
    AcousticFrameTrack, AcousticInput, AcousticModelBackend, MelFrame,
    MelTemporalDiscontinuityStats, SourceFilterAcousticModel, summarize_mel_temporal_discontinuity,
    temporal_smooth_mel_frames,
};
use listenbury::{PiperConfig, PiperTextToSpeech};
#[cfg(feature = "piper-compat")]
use listenbury::{SpeechT5OnnxAcousticGenerator, SpeechT5OnnxPaths};
#[cfg(feature = "audio-cpal")]
use std::io::BufRead;
#[cfg(feature = "piper-compat")]
use std::io::Write;
use std::path::{Path, PathBuf};
#[cfg(feature = "piper-compat")]
use std::process::{Command, Stdio};
#[cfg(feature = "audio-cpal")]
use std::thread;
use std::time::{Duration, Instant};

#[cfg(not(feature = "piper-compat"))]
const KLATT_SUPPORTED_WORDS: [&str; 6] = ["baby", "darling", "gal", "hello", "my", "ragtime"];
#[cfg(feature = "piper-compat")]
const HIFIGAN_TEMPORAL_BANDING_MEAN_DELTA_THRESHOLD: f32 = 0.20;
#[cfg(feature = "piper-compat")]
const HIFIGAN_TEMPORAL_BANDING_P95_DELTA_THRESHOLD: f32 = 0.30;
#[cfg(feature = "piper-compat")]
const HIFIGAN_SMOOTHING_EFFECT_RATIO: f32 = 0.85;

pub(crate) fn run_say(command: SayCommand) -> Result<()> {
    let piper_args = SayArgs::from_command(command)?;
    let loom = say_speech_loom(&piper_args);
    let backend_graph = say_backend_graph(&piper_args);
    tracing::debug!(
        speech_loom = loom.id,
        loom_projection = loom.projection,
        backend_graph = backend_graph.id,
        fused = backend_graph.fused,
        "listenbury say selected current backend graph over speech loom"
    );
    if piper_args.dump_pipeline {
        print_say_pipeline(&piper_args);
    }
    if piper_args.dump_phonemes && !piper_args.stdin_stream {
        print_say_phonemes(&piper_args)?;
    }
    if piper_args.dump_piper_tensors && !piper_args.stdin_stream {
        print_say_piper_tensors(&piper_args)?;
    }
    if piper_args.dump_phone_plan && !piper_args.stdin_stream {
        print_say_phone_plan(&piper_args)?;
        return Ok(());
    }
    if piper_args.stdin_stream {
        return run_say_stdin_stream(piper_args);
    }

    if should_use_klatt_backend(&piper_args) {
        let frames = synthesize_klatt_for_say(&piper_args.text)?;
        if let Some(output_path) = piper_args.output_wav {
            write_say_wav(&output_path, &frames)?;
        } else {
            play_say_audio_with_source(&frames, "Klatt")?;
        }
        return Ok(());
    }

    if should_use_speecht5_backend(&piper_args) {
        let frames = synthesize_speecht5_for_say(&piper_args)?;
        if let Some(output_path) = piper_args.output_wav {
            write_say_wav(&output_path, &frames)?;
        } else {
            play_say_audio_with_source(&frames, "SpeechT5")?;
        }
        return Ok(());
    }

    if should_use_source_filter_hifigan_backend(&piper_args) {
        let frames = synthesize_hifigan_for_say(&piper_args)?;
        if let Some(output_path) = piper_args.output_wav {
            write_say_wav(&output_path, &frames)?;
        } else {
            play_say_audio_with_source(&frames, "HiFi-GAN")?;
        }
        return Ok(());
    }

    if should_use_mbrola_backend(&piper_args) {
        let mut tts = say_mbrola_tts_for_args(&piper_args)?;
        tts.enqueue(MouthSyntheticPlan::from(SyntheticUnit::FullTurn(
            piper_args.text,
        )))?;
        let frames = collect_tts_audio(&mut tts, Duration::from_secs(30))?;

        if let Some(output_path) = piper_args.output_wav {
            write_say_wav(&output_path, &frames)?;
        } else {
            play_say_audio_with_source(&frames, "MBROLA")?;
        }
        return Ok(());
    }

    let piper_voice = resolve_piper_voice(piper_args.piper_voice.clone())?;
    let mut tts = say_tts_for_args(&piper_args, piper_voice)?;
    tts.enqueue(MouthSyntheticPlan::from(SyntheticUnit::FullTurn(
        piper_args.text,
    )))?;
    let frames = collect_tts_audio(&mut tts, Duration::from_secs(30))?;

    if let Some(output_path) = piper_args.output_wav {
        write_say_wav(&output_path, &frames)?;
    } else {
        play_say_audio(&frames)?;
    }

    Ok(())
}

fn run_say_stdin_stream(piper_args: SayArgs) -> Result<()> {
    anyhow::ensure!(
        piper_args.output_wav.is_none(),
        "listenbury say - streams to speaker playback; omit --output-wav"
    );

    #[cfg(not(feature = "audio-cpal"))]
    {
        let _ = piper_args;
        anyhow::bail!("listenbury say - needs the `audio-cpal` feature for speaker playback");
    }

    #[cfg(feature = "audio-cpal")]
    {
        let (frame_tx, frame_rx) = crossbeam_channel::bounded::<Vec<AudioFrame>>(8);
        let playback = thread::spawn(move || play_audio_frame_stream(frame_rx, "Piper stdin TTS"));

        let synthesis_result = if should_use_klatt_backend(&piper_args) {
            stream_klatt_stdin_to_frames(piper_args, frame_tx)
        } else if should_use_source_filter_hifigan_backend(&piper_args) {
            stream_hifigan_stdin_to_frames(piper_args, frame_tx)
        } else if should_use_speecht5_backend(&piper_args) {
            stream_speecht5_stdin_to_frames(piper_args, frame_tx)
        } else if should_use_mbrola_backend(&piper_args) {
            stream_mbrola_stdin_to_frames(piper_args, frame_tx)
        } else {
            stream_piper_stdin_to_frames(piper_args, frame_tx)
        };

        let playback_result = playback
            .join()
            .map_err(|_| anyhow::anyhow!("Piper stdin playback thread panicked"))?;
        if synthesis_result.is_err() {
            synthesis_result
        } else {
            playback_result?;
            Ok(())
        }
    }
}

#[cfg(feature = "audio-cpal")]
fn stream_klatt_stdin_to_frames(
    mut piper_args: SayArgs,
    frame_tx: crossbeam_channel::Sender<Vec<AudioFrame>>,
) -> Result<()> {
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let text = line.context("failed to read stdin for listenbury say -")?;
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        piper_args.text = text.to_string();
        if piper_args.dump_phonemes {
            print_say_phonemes(&piper_args)?;
        }
        if piper_args.dump_piper_tensors {
            print_say_piper_tensors(&piper_args)?;
        }
        if piper_args.dump_phone_plan {
            print_say_phone_plan(&piper_args)?;
            continue;
        }
        let frames = synthesize_klatt_for_say(text)?;
        frame_tx
            .send(frames)
            .context("failed to send Klatt stdin audio to playback")?;
    }
    Ok(())
}

#[cfg(feature = "audio-cpal")]
fn stream_hifigan_stdin_to_frames(
    mut piper_args: SayArgs,
    frame_tx: crossbeam_channel::Sender<Vec<AudioFrame>>,
) -> Result<()> {
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let text = line.context("failed to read stdin for listenbury say --hifigan -")?;
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        piper_args.text = text.to_string();
        if piper_args.dump_phonemes {
            print_say_phonemes(&piper_args)?;
        }
        if piper_args.dump_piper_tensors {
            print_say_piper_tensors(&piper_args)?;
        }
        if piper_args.dump_phone_plan {
            print_say_phone_plan(&piper_args)?;
            continue;
        }
        let frames = synthesize_hifigan_for_say(&piper_args)?;
        frame_tx
            .send(frames)
            .context("failed to send HiFi-GAN stdin audio to playback")?;
    }
    Ok(())
}

#[cfg(feature = "audio-cpal")]
fn stream_speecht5_stdin_to_frames(
    mut piper_args: SayArgs,
    frame_tx: crossbeam_channel::Sender<Vec<AudioFrame>>,
) -> Result<()> {
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let text = line.context("failed to read stdin for listenbury say --speecht5 -")?;
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        piper_args.text = text.to_string();
        if piper_args.dump_phonemes {
            print_say_phonemes(&piper_args)?;
        }
        if piper_args.dump_piper_tensors {
            print_say_piper_tensors(&piper_args)?;
        }
        if piper_args.dump_phone_plan {
            print_say_phone_plan(&piper_args)?;
            continue;
        }
        let frames = synthesize_speecht5_for_say(&piper_args)?;
        frame_tx
            .send(frames)
            .context("failed to send SpeechT5 stdin audio to playback")?;
    }
    Ok(())
}

#[cfg(feature = "audio-cpal")]
fn stream_piper_stdin_to_frames(
    piper_args: SayArgs,
    frame_tx: crossbeam_channel::Sender<Vec<AudioFrame>>,
) -> Result<()> {
    let piper_voice = resolve_piper_voice(piper_args.piper_voice.clone())?;
    let mut tts = say_tts_for_args(&piper_args, piper_voice)?;
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let text = line.context("failed to read stdin for listenbury say -")?;
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        if piper_args.dump_phonemes {
            let line_args = piper_args.clone_for_text(text);
            print_say_phonemes(&line_args)?;
        }
        if piper_args.dump_piper_tensors {
            let line_args = piper_args.clone_for_text(text);
            print_say_piper_tensors(&line_args)?;
        }
        if piper_args.dump_phone_plan {
            let line_args = piper_args.clone_for_text(text);
            print_say_phone_plan(&line_args)?;
            continue;
        }
        tts.enqueue(MouthSyntheticPlan::from(SyntheticUnit::FullTurn(
            text.to_string(),
        )))?;
        let frames = collect_tts_audio(&mut tts, Duration::from_secs(30))?;
        frame_tx
            .send(frames)
            .context("failed to send Piper stdin audio to playback")?;
    }
    Ok(())
}

#[cfg(feature = "audio-cpal")]
fn stream_mbrola_stdin_to_frames(
    piper_args: SayArgs,
    frame_tx: crossbeam_channel::Sender<Vec<AudioFrame>>,
) -> Result<()> {
    let mut tts = say_mbrola_tts_for_args(&piper_args)?;
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let text = line.context("failed to read stdin for listenbury say --diphone -")?;
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        if piper_args.dump_phonemes {
            let line_args = piper_args.clone_for_text(text);
            print_say_phonemes(&line_args)?;
        }
        if piper_args.dump_piper_tensors {
            let line_args = piper_args.clone_for_text(text);
            print_say_piper_tensors(&line_args)?;
        }
        if piper_args.dump_phone_plan {
            let line_args = piper_args.clone_for_text(text);
            print_say_phone_plan(&line_args)?;
            continue;
        }
        tts.enqueue(MouthSyntheticPlan::from(SyntheticUnit::FullTurn(
            text.to_string(),
        )))?;
        let frames = collect_tts_audio(&mut tts, Duration::from_secs(30))?;
        frame_tx
            .send(frames)
            .context("failed to send MBROLA stdin audio to playback")?;
    }
    Ok(())
}

fn say_tts_for_args(args: &SayArgs, piper_voice: PathBuf) -> Result<PiperTextToSpeech> {
    if args.piper {
        let piper_bin = resolve_piper_bin(args.piper_bin.clone())?;
        return Ok(PiperTextToSpeech::new(piper_config_for_voice(
            piper_bin,
            piper_voice,
        )?));
    }

    say_riper_tts_for_voice(piper_voice)
}

#[cfg(feature = "piper-compat")]
fn say_mbrola_tts_for_args(args: &SayArgs) -> Result<PiperTextToSpeech> {
    let voice = resolve_mbrola_voice(args.mbrola_voice.clone())?;
    if voice == received_pronunciation_mbrola_voice() && !voice.is_file() {
        anyhow::bail!(
            "failed to find RP MBROLA voice {}; run `just fetch` first",
            voice.display()
        );
    }
    Ok(PiperTextToSpeech::with_backend(MbrolaTextBackend::load(
        voice,
    )?))
}

#[cfg(not(feature = "piper-compat"))]
fn say_mbrola_tts_for_args(_args: &SayArgs) -> Result<PiperTextToSpeech> {
    anyhow::bail!("listenbury say --diphone requires the `piper-compat` feature")
}

#[cfg(feature = "piper-compat")]
fn say_riper_tts_for_voice(piper_voice: PathBuf) -> Result<PiperTextToSpeech> {
    Ok(PiperTextToSpeech::new_with_backend_preference(
        piper_config_for_riper_voice(piper_voice)?,
        PiperBackendPreference::RiperWithProcessFallback,
    ))
}

#[cfg(not(feature = "piper-compat"))]
fn say_riper_tts_for_voice(_piper_voice: PathBuf) -> Result<PiperTextToSpeech> {
    anyhow::bail!(
        "listenbury say requires the `piper-compat` feature (or pass --piper for the external Piper binary)"
    )
}

#[cfg(feature = "piper-compat")]
enum MbrolaTextBackend {
    Native {
        renderer: MbrolaRenderer,
        phonemizer: SimpleEnglishG2p,
    },
    DiphoneCache {
        provider: NeuralDiphoneProvider,
        phonemizer: SimpleEnglishG2p,
        voice_name: String,
        sample_rate_hz: u32,
        source_period_samples: usize,
    },
}

#[cfg(feature = "piper-compat")]
impl MbrolaTextBackend {
    fn load(voice_path: PathBuf) -> Result<Self> {
        if let Some(manifest) = DiphoneVoiceManifest::load_if_present(&voice_path)? {
            return Self::load_diphone_cache(manifest);
        }
        Ok(Self::Native {
            renderer: MbrolaRenderer::from_voice_path(None, voice_path)?,
            phonemizer: SimpleEnglishG2p::default(),
        })
    }

    fn load_diphone_cache(manifest: DiphoneVoiceManifest) -> Result<Self> {
        let config_json = std::fs::read_to_string(&manifest.config).with_context(|| {
            format!(
                "failed to read Piper voice config {}",
                manifest.config.display()
            )
        })?;
        let config = PiperVoiceConfig::from_json_str(&config_json).with_context(|| {
            format!(
                "failed to parse Piper voice config {}",
                manifest.config.display()
            )
        })?;
        let sample_rate_hz = config.sample_rate_hz;
        let backend = RiperBackend::load(&manifest.model, config).with_context(|| {
            format!(
                "failed to load cache-backed diphone voice model {}",
                manifest.model.display()
            )
        })?;
        let cache = DiphoneCache::open(&manifest.cache_dir).with_context(|| {
            format!(
                "failed to open diphone cache {}",
                manifest.cache_dir.display()
            )
        })?;
        Ok(Self::DiphoneCache {
            provider: NeuralDiphoneProvider::new(backend, cache),
            phonemizer: SimpleEnglishG2p::default(),
            voice_name: manifest.name,
            sample_rate_hz,
            source_period_samples: neutral_source_period_samples(sample_rate_hz),
        })
    }
}

#[cfg(feature = "piper-compat")]
impl TtsBackend for MbrolaTextBackend {
    fn synthesize(&mut self, text: &str) -> Result<Vec<AudioFrame>> {
        match self {
            Self::Native {
                renderer,
                phonemizer,
            } => {
                let voice = renderer.voice();
                let plan = planned_phone_timed_plan_for_text(
                    *phonemizer,
                    text,
                    |symbol| Ok(voice.symbol_map.map_phone(symbol)?),
                    &voice.name,
                )?;
                let frames = renderer
                    .render_phone_plan_to_frames(&plan)
                    .with_context(|| format!("native MBROLA diphone render failed for `{text}`"))?;
                anyhow::ensure!(
                    !frames.is_empty(),
                    "MBROLA produced no audio frames for `{text}`"
                );
                Ok(frames)
            }
            Self::DiphoneCache {
                provider,
                phonemizer,
                voice_name,
                sample_rate_hz,
                source_period_samples,
            } => {
                let plan = planned_phone_timed_plan_for_text(
                    *phonemizer,
                    text,
                    |symbol| Ok(symbol.to_string()),
                    voice_name,
                )?;
                let frames = render_phone_plan_with_diphone_provider_to_frames(
                    &plan,
                    provider,
                    *sample_rate_hz,
                    *source_period_samples,
                )
                .with_context(|| {
                    format!("cache-backed MBROLA-compatible render failed for `{text}`")
                })?;
                anyhow::ensure!(
                    !frames.is_empty(),
                    "cache-backed diphone voice produced no audio frames for `{text}`"
                );
                Ok(frames)
            }
        }
    }
}

#[cfg(feature = "piper-compat")]
fn planned_phone_timed_plan_for_text(
    phonemizer: SimpleEnglishG2p,
    text: &str,
    map_phone: impl Fn(&str) -> Result<String>,
    voice_name: &str,
) -> Result<PhoneTimedPlan> {
    let mut tracker = listenbury::mouth::riper::PhonemeProsodyCandidateTracker::new(phonemizer);
    let mut candidate = tracker
        .ingest_text(text)
        .with_context(|| format!("failed to realize Riper phonemes for `{text}`"))?
        .into_iter()
        .find_map(|event| match event {
            PhonemeProsodyCandidateEvent::CandidateUpdated { candidate } => Some(candidate),
            _ => None,
        })
        .with_context(|| format!("Riper produced no candidate for `{text}`"))?;
    candidate.mark_committed();

    let mut planner = BreathGroupProsodyPlanner::new();
    let planned = planner.plan_candidate(&candidate);
    phone_timed_plan_from_planned_prosody(&candidate, &planned, map_phone, text, voice_name)
}

#[cfg(feature = "piper-compat")]
fn phone_timed_plan_from_planned_prosody(
    candidate: &PhonemeProsodyCandidate,
    planned: &ProsodyList,
    map_phone: impl Fn(&str) -> Result<String>,
    text: &str,
    voice_name: &str,
) -> Result<PhoneTimedPlan> {
    let realization = planned.realize_for_riper(candidate);
    let pauses_after_word = pauses_after_words(candidate, &realization);
    let utterance_pauses = utterance_pauses(&realization);
    let mut phones = Vec::new();

    for (index, phoneme) in candidate.phonemes.phonemes.iter().enumerate() {
        let symbol = phoneme.0.as_str();
        let word_index = candidate.phoneme_to_word.get(index).and_then(|word| *word);
        if word_index.is_none() || !is_renderable_riper_symbol(symbol) {
            continue;
        }

        let mapped = map_phone(symbol).with_context(|| {
            format!(
                "failed to map Riper phone `{symbol}` to diphone voice `{voice_name}` while rendering `{text}`"
            )
        })?;
        let duration_ms = planned_phone_duration_ms(candidate, &realization, index);
        let pitch_targets = if mbrola_symbol_is_pitch_bearing(symbol) {
            planned_pitch_targets(candidate, planned, index)
        } else {
            Vec::new()
        };
        phones.push(MbrolaPhone {
            symbol: mapped,
            duration_ms,
            pitch_targets,
        });

        if let Some(word_index) = word_index
            && is_last_phone_in_word(candidate, index, word_index)
        {
            if let Some(pauses) = pauses_after_word.get(word_index) {
                phones.extend(pauses.iter().map(|millis| MbrolaPhone::new("_", *millis)));
            }
        }
    }

    phones.extend(
        utterance_pauses
            .into_iter()
            .map(|millis| MbrolaPhone::new("_", millis)),
    );
    if phones.last().is_none_or(|phone| phone.symbol != "_") {
        phones.push(MbrolaPhone::new("_", 80));
    }
    anyhow::ensure!(
        phones.iter().any(|phone| phone.symbol != "_"),
        "Riper produced no cache-renderable phones for `{text}` with diphone voice `{voice_name}`"
    );
    Ok(PhoneTimedPlan::new(phones))
}

#[cfg(feature = "piper-compat")]
fn planned_phone_duration_ms(
    candidate: &PhonemeProsodyCandidate,
    realization: &RiperProsodyRealization,
    phoneme_index: usize,
) -> u32 {
    let base = candidate
        .phone_hints
        .iter()
        .find(|hint| hint.phoneme_index == phoneme_index)
        .and_then(|hint| hint.approximate_duration_ms)
        .unwrap_or(90);
    let mut duration = base as f32;

    if let Some(word_index) = candidate
        .phoneme_to_word
        .get(phoneme_index)
        .and_then(|word| *word)
        && let Some(Some(word_duration)) = realization.word_duration_overrides_ms.get(word_index)
        && let Some(word_base) = candidate
            .word_hints
            .iter()
            .find(|hint| hint.word_index == word_index)
            .and_then(|hint| hint.approximate_duration_ms)
        && word_base > 0
    {
        duration *= *word_duration as f32 / word_base as f32;
    }

    if let Some(Some(phone_duration)) = realization.phone_duration_overrides_ms.get(phoneme_index)
        && base > 0
    {
        duration *= *phone_duration as f32 / base as f32;
    }

    duration.round().clamp(1.0, u32::MAX as f32) as u32
}

#[cfg(feature = "piper-compat")]
fn planned_pitch_targets(
    candidate: &PhonemeProsodyCandidate,
    planned: &ProsodyList,
    phoneme_index: usize,
) -> Vec<MbrolaPitchTarget> {
    let Some((shape, strength)) = planned
        .ops
        .iter()
        .filter_map(|op| match op {
            ProsodyOp::SetPitchShape {
                target,
                shape,
                strength,
            } if prosody_target_contains_phoneme(candidate, target, phoneme_index) => {
                Some((*shape, *strength))
            }
            _ => None,
        })
        .last()
    else {
        return Vec::new();
    };

    pitch_targets_for_shape(shape, strength, planned.base.contour.energy)
}

#[cfg(feature = "piper-compat")]
fn pitch_targets_for_shape(
    shape: ProsodyPitchShape,
    strength: u8,
    energy: ProsodyEnergy,
) -> Vec<MbrolaPitchTarget> {
    let base_hz = match energy {
        ProsodyEnergy::Low => 118.0,
        ProsodyEnergy::Neutral => 125.0,
        ProsodyEnergy::Elevated => 132.0,
    };
    let range_hz = 8.0 + 18.0 * (f32::from(strength.min(100)) / 100.0);
    let targets = match shape {
        ProsodyPitchShape::Level => vec![(0, base_hz), (100, base_hz)],
        ProsodyPitchShape::Rise => vec![
            (0, base_hz - range_hz * 0.35),
            (65, base_hz + range_hz * 0.40),
            (100, base_hz + range_hz * 0.60),
        ],
        ProsodyPitchShape::Fall => vec![
            (0, base_hz + range_hz * 0.55),
            (55, base_hz),
            (100, base_hz - range_hz * 0.55),
        ],
        ProsodyPitchShape::RiseFall => vec![
            (0, base_hz - range_hz * 0.30),
            (50, base_hz + range_hz * 0.65),
            (100, base_hz - range_hz * 0.20),
        ],
        ProsodyPitchShape::FallRise => vec![
            (0, base_hz + range_hz * 0.35),
            (50, base_hz - range_hz * 0.45),
            (100, base_hz + range_hz * 0.25),
        ],
    };

    targets
        .into_iter()
        .map(|(percent, hz)| MbrolaPitchTarget {
            percent,
            hz: hz.max(60.0),
        })
        .collect()
}

#[cfg(feature = "piper-compat")]
fn pauses_after_words(
    candidate: &PhonemeProsodyCandidate,
    realization: &RiperProsodyRealization,
) -> Vec<Vec<u32>> {
    let mut pauses = vec![Vec::new(); candidate.word_targets.len()];
    for pause in &realization.pauses {
        match &pause.after {
            ProsodyTarget::WordIndex { index } => {
                if let Some(slot) = pauses.get_mut(*index) {
                    slot.push(clamp_duration_u64_to_u32(pause.millis));
                }
            }
            ProsodyTarget::WordRange { end, .. } => {
                if let Some(index) = end.checked_sub(1)
                    && let Some(slot) = pauses.get_mut(index)
                {
                    slot.push(clamp_duration_u64_to_u32(pause.millis));
                }
            }
            ProsodyTarget::PhonemeRange { end, .. } => {
                if let Some(index) = end
                    .checked_sub(1)
                    .and_then(|idx| candidate.phoneme_to_word.get(idx))
                    .and_then(|word| *word)
                    && let Some(slot) = pauses.get_mut(index)
                {
                    slot.push(clamp_duration_u64_to_u32(pause.millis));
                }
            }
            ProsodyTarget::WholeCandidate => {}
        }
    }
    pauses
}

#[cfg(feature = "piper-compat")]
fn utterance_pauses(realization: &RiperProsodyRealization) -> Vec<u32> {
    realization
        .pauses
        .iter()
        .filter_map(|pause| {
            matches!(pause.after, ProsodyTarget::WholeCandidate)
                .then(|| clamp_duration_u64_to_u32(pause.millis))
        })
        .collect()
}

#[cfg(feature = "piper-compat")]
fn prosody_target_contains_phoneme(
    candidate: &PhonemeProsodyCandidate,
    target: &ProsodyTarget,
    phoneme_index: usize,
) -> bool {
    match target {
        ProsodyTarget::WholeCandidate => true,
        ProsodyTarget::WordIndex { index } => {
            candidate
                .phoneme_to_word
                .get(phoneme_index)
                .and_then(|word| *word)
                == Some(*index)
        }
        ProsodyTarget::WordRange { start, end } => candidate
            .phoneme_to_word
            .get(phoneme_index)
            .and_then(|word| *word)
            .is_some_and(|word| word >= *start && word < *end),
        ProsodyTarget::PhonemeRange { start, end } => {
            phoneme_index >= *start && phoneme_index < *end
        }
    }
}

#[cfg(feature = "piper-compat")]
fn is_last_phone_in_word(
    candidate: &PhonemeProsodyCandidate,
    phoneme_index: usize,
    word_index: usize,
) -> bool {
    candidate
        .phoneme_to_word
        .iter()
        .enumerate()
        .skip(phoneme_index + 1)
        .find_map(|(_, word)| word.map(|word| word != word_index))
        .unwrap_or(true)
}

#[cfg(feature = "piper-compat")]
fn is_renderable_riper_symbol(symbol: &str) -> bool {
    !symbol.trim().is_empty()
        && !matches!(symbol, "_" | "^" | "$" | "|" | "‖" | "." | "," | "!" | "?")
}

#[cfg(feature = "piper-compat")]
fn clamp_duration_u64_to_u32(millis: u64) -> u32 {
    millis.clamp(1, u64::from(u32::MAX)) as u32
}

#[cfg(feature = "piper-compat")]
fn mbrola_symbol_is_pitch_bearing(symbol: &str) -> bool {
    let base = symbol.trim_end_matches(|ch: char| ch.is_ascii_digit());
    matches!(
        base,
        "AA" | "AE"
            | "AH"
            | "AO"
            | "AW"
            | "AY"
            | "EH"
            | "ER"
            | "EY"
            | "IH"
            | "IY"
            | "OW"
            | "OY"
            | "UH"
            | "UW"
            | "i"
            | "ɪ"
            | "e"
            | "ɛ"
            | "æ"
            | "ə"
            | "ʌ"
            | "ɑ"
            | "ɔ"
            | "o"
            | "ʊ"
            | "u"
    )
}

#[cfg(feature = "piper-compat")]
fn neutral_source_period_samples(sample_rate_hz: u32) -> usize {
    (sample_rate_hz / 125).max(1) as usize
}

#[cfg(feature = "piper-compat")]
fn resolve_mbrola_voice(explicit: Option<PathBuf>) -> Result<PathBuf> {
    explicit
        .or_else(|| std::env::var_os("LISTENBURY_MBROLA_VOICE").map(PathBuf::from))
        .or_else(|| std::env::var_os("MBROLA_VOICE").map(PathBuf::from))
        .or_else(|| {
            let fetched = PathBuf::from("data/mbrola/us3/us3");
            fetched.is_file().then_some(fetched)
        })
        .or_else(|| {
            let fetched = PathBuf::from("data/mbrola/us1/us1");
            fetched.is_file().then_some(fetched)
        })
        .with_context(|| {
            "failed to find diphone voice; run `just fetch` or set LISTENBURY_MBROLA_VOICE / MBROLA_VOICE / --diphone-voice"
        })
}

pub(crate) fn run_riper_compare(command: RiperCompareCommand) -> Result<()> {
    #[cfg(not(feature = "piper-compat"))]
    {
        let _ = command;
        anyhow::bail!(
            "listenbury riper-compare requires the `piper-compat` feature to compare Riper synthesis"
        );
    }

    #[cfg(feature = "piper-compat")]
    {
        run_riper_compare_impl(command)
    }
}

pub(crate) fn run_echo(command: EchoCommand) -> Result<()> {
    #[cfg(not(all(feature = "asr-whisper", feature = "piper-compat")))]
    {
        let _ = command;
        anyhow::bail!(
            "listenbury echo requires both the `asr-whisper` and `piper-compat` features"
        );
    }

    #[cfg(all(feature = "asr-whisper", feature = "piper-compat"))]
    {
        run_echo_impl(command)
    }
}

#[cfg(all(feature = "asr-whisper", feature = "piper-compat"))]
fn run_echo_impl(command: EchoCommand) -> Result<()> {
    let whisper_model = resolve_whisper_model(command.whisper_model)?;
    let riper_model_path = resolve_piper_voice(command.piper_voice)?;
    let riper_config_path = command
        .riper_config
        .unwrap_or_else(|| riper_model_path.with_extension("onnx.json"));
    let output_wav = command
        .output_wav
        .unwrap_or_else(|| PathBuf::from("out/riper-echo.wav"));
    let comparison_json = command.comparison_json;

    let frames = read_wav_as_whisper_frames(&command.input_wav, 1_600).with_context(|| {
        format!(
            "failed to read echo input WAV at {}",
            command.input_wav.display()
        )
    })?;
    anyhow::ensure!(
        !frames.is_empty(),
        "echo input WAV produced no audio frames: {}",
        command.input_wav.display()
    );

    let mut recognizer = WhisperSpeechRecognizer::new_quiet(&whisper_model).with_context(|| {
        format!(
            "failed to initialize Whisper model at {}",
            whisper_model.display()
        )
    })?;
    let mut analyzer = StreamingProsodyAnalyzer::default();
    let mut updates = Vec::new();
    let mut frame_start_ms = 0u64;
    for frame in &frames {
        recognizer.push_frame(frame)?;
        if let Some(update) = analyzer.ingest_frame(frame, frame_start_ms) {
            updates.push(update);
        }
        frame_start_ms = frame_start_ms.saturating_add(frame_duration_ms(frame));
    }

    let transcript = recognizer.poll_timed_transcript_with_finality(true)?;
    if transcript.text.trim().is_empty() {
        anyhow::bail!(
            "Whisper produced no transcript for echo input {}",
            command.input_wav.display()
        );
    }

    let observation = EchoProsodyObservation::from_streaming_updates(
        transcript.text.clone(),
        &transcript.words,
        &updates,
    );
    let phonemized = SimpleEnglishG2p::default()
        .phonemize_unit(&transcript.text)
        .with_context(|| {
            format!(
                "failed to phonemize echoed transcript `{}`",
                transcript.text
            )
        })?;
    let plan = EchoProsodyPlan::from_observation(&observation, Some(&phonemized));
    let voice_config = read_riper_voice_config(&riper_config_path)?;
    let ids = phonemized
        .phonemes
        .to_piper_text_ids_compatible(&voice_config)
        .with_context(|| {
            format!(
                "Riper voice config at {} cannot map one or more phonemes for `{}`",
                riper_config_path.display(),
                transcript.text
            )
        })?;

    let mut backend = RiperBackend::load(&riper_model_path, voice_config).with_context(|| {
        format!(
            "failed to initialize Riper backend from model {}",
            riper_model_path.display()
        )
    })?;
    let (echo_frames, diagnostics) = backend
        .synthesize_id_frames_with_controls(&ids, Some(&plan.controls))
        .with_context(|| {
            format!(
                "failed to synthesize echoed transcript `{}`",
                transcript.text
            )
        })?;
    let comparison = EchoComparisonRecord::from_plan(&observation, &plan, &diagnostics);

    write_say_wav(&output_wav, &echo_frames)?;
    println!("Heard: {}", transcript.text);
    println!("{}", serde_json::to_string_pretty(&comparison)?);

    if let Some(path) = comparison_json {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create comparison output directory {}",
                    parent.display()
                )
            })?;
        }
        std::fs::write(&path, serde_json::to_vec_pretty(&comparison)?).with_context(|| {
            format!("failed to write echo comparison JSON to {}", path.display())
        })?;
        println!("Wrote {}", path.display());
    }

    Ok(())
}

#[cfg(feature = "piper-compat")]
fn run_riper_compare_impl(command: RiperCompareCommand) -> Result<()> {
    let args = RiperCompareArgs::from_command(command)?;

    let piper_bin = resolve_piper_bin(args.piper_bin.clone())?;
    let process_voice = resolve_piper_voice(args.piper_voice.clone())?;
    let process_config = piper_config_for_voice(piper_bin.clone(), process_voice)?;
    let process_stats = synthesize_process_for_compare(&process_config, &args.text)?;
    let process_phonemes =
        process_native_phonemes_for_compare(process_config.config_path.as_deref(), &args.text);

    let riper_model_path = args
        .riper_voice
        .clone()
        .unwrap_or_else(|| process_config.model_path.clone());
    let riper_config_path = args
        .riper_config
        .clone()
        .or_else(|| process_config.config_path.clone())
        .unwrap_or_else(|| riper_model_path.with_extension("onnx.json"));
    let riper_voice_config = read_riper_voice_config(&riper_config_path)?;
    let riper_phonemes = resolve_riper_phoneme_report(&args, &riper_voice_config)?;
    let riper_contract = RiperBackend::load(&riper_model_path, riper_voice_config.clone())
        .and_then(|backend| backend.validate_model_contract())
        .with_context(|| {
            format!(
                "failed to inspect Riper ONNX contract for {}",
                riper_model_path.display()
            )
        })?;
    let riper_stats = synthesize_riper_for_compare(
        &riper_model_path,
        &riper_voice_config,
        &riper_phonemes.ids,
        &args.text,
    )?;

    report_compare_phonemes(&process_phonemes, &riper_phonemes);
    print!(
        "{}",
        format_piper_tensor_dump(
            "riper-compare",
            &args.text,
            &riper_model_path,
            &riper_config_path,
            false,
            &riper_phonemes.trace,
            &riper_voice_config,
            &riper_contract.input_names,
        )
    );
    report_compare_stats(&process_stats, &riper_stats);

    if let Some(output) = args.process_output_wav {
        write_say_wav(&output, &process_stats.frames)?;
    }
    if let Some(output) = args.riper_output_wav {
        write_say_wav(&output, &riper_stats.frames)?;
    }

    Ok(())
}

#[cfg(feature = "piper-compat")]
#[derive(Debug)]
struct RiperCompareArgs {
    piper_bin: Option<PathBuf>,
    piper_voice: Option<PathBuf>,
    riper_voice: Option<PathBuf>,
    riper_config: Option<PathBuf>,
    process_output_wav: Option<PathBuf>,
    riper_output_wav: Option<PathBuf>,
    phonemes: Option<String>,
    text: String,
}

#[cfg(feature = "piper-compat")]
impl RiperCompareArgs {
    fn from_command(command: RiperCompareCommand) -> Result<Self> {
        anyhow::ensure!(
            !command.words.is_empty(),
            "missing text to compare; try `listenbury riper-compare \"I see.\"`"
        );
        let text = command.words.join(" ");
        anyhow::ensure!(
            !text.trim().is_empty(),
            "missing text to compare; try `listenbury riper-compare \"I see.\"`"
        );

        Ok(Self {
            piper_bin: command.piper_bin,
            piper_voice: command.piper_voice,
            riper_voice: command.riper_voice,
            riper_config: command.riper_config,
            process_output_wav: command.process_output_wav,
            riper_output_wav: command.riper_output_wav,
            phonemes: command.phonemes,
            text,
        })
    }
}

#[cfg(feature = "piper-compat")]
#[derive(Debug, Clone)]
struct SynthesisStats {
    frames: Vec<AudioFrame>,
    runtime: Duration,
    audio: AudioStats,
}

#[cfg(feature = "piper-compat")]
#[derive(Debug, Clone, PartialEq)]
struct AudioStats {
    sample_rate_hz: u32,
    channels: u16,
    sample_count: usize,
    duration_ms: f64,
    rms: f32,
    peak_abs: f32,
}

#[cfg(feature = "piper-compat")]
#[derive(Debug, Clone, PartialEq, Eq)]
struct ProcessNativePhonemes {
    voice: String,
    mnemonic: std::result::Result<String, String>,
    ipa: std::result::Result<String, String>,
}

#[cfg(feature = "piper-compat")]
#[derive(Debug, Clone, PartialEq)]
struct RiperPhonemeReport {
    source: &'static str,
    phonemes: PiperPhonemeSequence,
    compatible_phonemes: Option<PiperPhonemeSequence>,
    ids: PiperIdSequence,
    trace: PiperTextIdTrace,
    sentence_analysis: Option<SentenceAnalysis>,
}

#[cfg(feature = "piper-compat")]
impl AudioStats {
    fn from_frames(frames: &[AudioFrame], label: &str) -> Result<Self> {
        let Some(first) = frames.first() else {
            anyhow::bail!("{label} synthesis produced no frames");
        };
        anyhow::ensure!(
            first.sample_rate_hz > 0,
            "{label} synthesis produced an invalid sample rate of 0 Hz"
        );
        anyhow::ensure!(
            first.channels > 0,
            "{label} synthesis produced an invalid channel count of 0"
        );

        let mut samples = Vec::new();
        for frame in frames {
            anyhow::ensure!(
                frame.sample_rate_hz == first.sample_rate_hz,
                "{label} synthesis changed sample rate mid-stream ({} -> {})",
                first.sample_rate_hz,
                frame.sample_rate_hz
            );
            anyhow::ensure!(
                frame.channels == first.channels,
                "{label} synthesis changed channel count mid-stream ({} -> {})",
                first.channels,
                frame.channels
            );
            samples.extend_from_slice(&frame.samples);
        }

        let sample_count = samples.len();
        let (rms, peak_abs) = if sample_count == 0 {
            (0.0, 0.0)
        } else {
            let square_sum = samples.iter().map(|sample| sample * sample).sum::<f32>();
            let rms = (square_sum / sample_count as f32).sqrt();
            let peak_abs = samples
                .iter()
                .map(|sample| sample.abs())
                .fold(0.0_f32, f32::max);
            (rms, peak_abs)
        };

        let duration_ms = (sample_count as f64 / f64::from(first.sample_rate_hz)) * 1000.0;

        Ok(Self {
            sample_rate_hz: first.sample_rate_hz,
            channels: first.channels,
            sample_count,
            duration_ms,
            rms,
            peak_abs,
        })
    }
}

#[cfg(feature = "piper-compat")]
fn synthesize_process_for_compare(config: &PiperConfig, text: &str) -> Result<SynthesisStats> {
    let mut backend = ProcessPiperBackend::new(config.clone());
    let t0 = Instant::now();
    let frames = backend.synthesize(text)?;
    let runtime = t0.elapsed();
    let audio = AudioStats::from_frames(&frames, "process")?;
    Ok(SynthesisStats {
        frames,
        runtime,
        audio,
    })
}

#[cfg(feature = "piper-compat")]
fn synthesize_riper_for_compare(
    model_path: &Path,
    config: &PiperVoiceConfig,
    ids: &PiperIdSequence,
    text: &str,
) -> Result<SynthesisStats> {
    let mut backend = RiperBackend::load(model_path, config.clone()).with_context(|| {
        format!(
            "failed to initialize Riper backend from model {}",
            model_path.display()
        )
    })?;
    let t0 = Instant::now();
    let frames = backend.synthesize_id_frames(ids).with_context(|| {
        format!(
            "Riper synthesis failed for model {} and text `{}`",
            model_path.display(),
            text
        )
    })?;
    let runtime = t0.elapsed();
    let audio = AudioStats::from_frames(&frames, "riper")?;
    Ok(SynthesisStats {
        frames,
        runtime,
        audio,
    })
}

#[cfg(feature = "piper-compat")]
fn resolve_riper_phoneme_report(
    args: &RiperCompareArgs,
    config: &PiperVoiceConfig,
) -> Result<RiperPhonemeReport> {
    let (source, phoneme_sequence, sentence_analysis) = if let Some(raw) = args.phonemes.as_ref() {
        let symbols: Vec<_> = raw.split_whitespace().collect();
        anyhow::ensure!(
            !symbols.is_empty(),
            "Riper phoneme override was empty; pass symbols like --phonemes \"OW K EY |\""
        );
        let sentence_analysis = SimpleEnglishG2p::default()
            .phonemize_unit(&args.text)
            .ok()
            .map(|unit| unit.sentence_analysis);
        (
            "override",
            PiperPhonemeSequence {
                phonemes: symbols
                    .into_iter()
                    .map(|symbol| PiperPhoneme(symbol.to_string()))
                    .collect(),
            },
            sentence_analysis,
        )
    } else {
        let unit = SimpleEnglishG2p::default()
            .phonemize_unit(&args.text)
            .with_context(|| {
                format!("failed to realize Riper phonemes for text `{}`", args.text)
            })?;
        ("riper-g2p", unit.phonemes, Some(unit.sentence_analysis))
    };

    let ids = phoneme_sequence
        .to_piper_text_ids_compatible(config)
        .with_context(|| {
            format!(
                "Riper voice config cannot map one or more phonemes for `{}`; pass --phonemes to override",
                args.text
            )
        })?;
    let trace = phoneme_sequence
        .to_piper_text_id_trace(config)
        .with_context(|| format!("failed to build Piper ID trace for `{}`", args.text))?;
    let compatible_phonemes = espeak_compatible_sequence(&phoneme_sequence, config).ok();

    Ok(RiperPhonemeReport {
        source,
        phonemes: phoneme_sequence,
        compatible_phonemes,
        ids,
        trace,
        sentence_analysis,
    })
}

#[cfg(feature = "piper-compat")]
fn read_riper_voice_config(path: &Path) -> Result<PiperVoiceConfig> {
    let json = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read Riper config at {}", path.display()))?;
    PiperVoiceConfig::from_json_str(&json)
        .with_context(|| format!("failed to parse Riper config JSON at {}", path.display()))
}

#[cfg(feature = "piper-compat")]
fn process_native_phonemes_for_compare(
    config_path: Option<&Path>,
    text: &str,
) -> ProcessNativePhonemes {
    let voice = config_path
        .and_then(espeak_voice_from_config)
        .unwrap_or_else(|| "en-us".to_string());

    ProcessNativePhonemes {
        voice: voice.clone(),
        mnemonic: run_espeak_ng_phonemes(&voice, text, EspeakPhonemeNotation::Mnemonic),
        ipa: run_espeak_ng_phonemes(&voice, text, EspeakPhonemeNotation::Ipa),
    }
}

#[cfg(feature = "piper-compat")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EspeakPhonemeNotation {
    Mnemonic,
    Ipa,
}

#[cfg(feature = "piper-compat")]
fn espeak_voice_from_config(path: &Path) -> Option<String> {
    let contents = std::fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&contents).ok()?;
    value
        .get("espeak")
        .and_then(|espeak| espeak.get("voice"))
        .and_then(|voice| voice.as_str())
        .filter(|voice| !voice.trim().is_empty())
        .map(str::to_string)
}

#[cfg(feature = "piper-compat")]
fn run_espeak_ng_phonemes(
    voice: &str,
    text: &str,
    notation: EspeakPhonemeNotation,
) -> std::result::Result<String, String> {
    let mut command = Command::new("espeak-ng");
    command.arg("-q").arg("--sep= ").arg("-v").arg(voice);
    match notation {
        EspeakPhonemeNotation::Mnemonic => {
            command.arg("-x");
        }
        EspeakPhonemeNotation::Ipa => {
            command.arg("--ipa");
        }
    }
    command
        .arg("--stdin")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|error| format!("failed to spawn espeak-ng: {error}"))?;
    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| "failed to open espeak-ng stdin".to_string())?;
        stdin
            .write_all(text.as_bytes())
            .map_err(|error| format!("failed to write text to espeak-ng stdin: {error}"))?;
        stdin
            .write_all(b"\n")
            .map_err(|error| format!("failed to finish espeak-ng stdin: {error}"))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|error| format!("failed to read espeak-ng output: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "espeak-ng exited with {}: {}",
            output.status,
            stderr.trim()
        ));
    }

    Ok(String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" "))
}

#[cfg(feature = "piper-compat")]
fn report_compare_phonemes(process: &ProcessNativePhonemes, riper: &RiperPhonemeReport) {
    println!("process native phonemes (eSpeak {}, -x):", process.voice);
    println!("  {}", render_phoneme_result(&process.mnemonic));
    println!("process native IPA (eSpeak {}):", process.voice);
    println!("  {}", render_phoneme_result(&process.ipa));
    println!("Riper phonemes ({}):", riper.source);
    println!("  {}", format_phoneme_sequence(&riper.phonemes));
    if let Some(compatible) = &riper.compatible_phonemes {
        println!("Riper Piper-compatible phonemes:");
        println!("  {}", format_phoneme_sequence(compatible));
    }
    println!("Riper phoneme ids:");
    println!("  {:?}", riper.ids.ids);
    report_link_grammar(&riper.sentence_analysis);
}

#[cfg(feature = "piper-compat")]
fn format_piper_tensor_dump(
    backend_id: &str,
    text: &str,
    model_path: &Path,
    config_path: &Path,
    external_process: bool,
    trace: &PiperTextIdTrace,
    voice_config: &PiperVoiceConfig,
    input_names: &[String],
) -> String {
    let input_len = trace.ids_after_framing.len();
    let scales = piper_inference_scales(voice_config);
    let sid_input = input_names
        .iter()
        .find(|name| name.as_str() == "sid" || name.as_str() == "speaker_id");

    let mut output = String::new();
    output.push_str(&format!("piper tensor dump: {backend_id}\n"));
    output.push_str(&format!("input text: {text}\n"));
    output.push_str(&format!("model: {}\n", model_path.display()));
    output.push_str(&format!("config: {}\n", config_path.display()));
    if external_process {
        output.push_str("external Piper process tensors:\n");
        output.push_str(
            "  unavailable: the process API returns audio only; it does not expose final ONNX input IDs\n",
        );
        output.push_str("Listenbury internal-compatible tensor candidate:\n");
    } else {
        output.push_str("Listenbury internal Piper-compatible tensors:\n");
    }
    output.push_str(&format!("  source symbols: {:?}\n", trace.source_symbols));
    output.push_str(&format!(
        "  text symbols after termination: {:?}\n",
        trace.text_symbols
    ));
    output.push_str(&format!(
        "  symbols before id mapping: {:?}\n",
        trace.symbols_before_id_mapping
    ));
    output.push_str(&format!(
        "  ids before BOS/PAD/EOS framing: {:?}\n",
        trace.ids_before_framing
    ));
    output.push_str(&format!(
        "  ids after BOS/PAD/EOS framing: {:?}\n",
        trace.ids_after_framing
    ));
    output.push_str("  input tensor:\n");
    output.push_str(&format!("    shape=[1, {input_len}]\n"));
    output.push_str(&format!("    values={:?}\n", trace.ids_after_framing));
    output.push_str("  input_lengths:\n");
    output.push_str("    shape=[1]\n");
    output.push_str(&format!("    values=[{input_len}]\n"));
    output.push_str("  scales:\n");
    output.push_str("    shape=[3]\n");
    output.push_str(&format!(
        "    values=[{:.6}, {:.6}, {:.6}]\n",
        scales[0], scales[1], scales[2]
    ));
    output.push_str("  sid:\n");
    if let Some(name) = sid_input {
        output.push_str(&format!("    input={name}\n"));
        output.push_str("    shape=[1]\n");
        output.push_str("    values=[0]\n");
    } else {
        output.push_str("    absent\n");
    }
    output
}

#[cfg(feature = "piper-compat")]
fn piper_inference_scales(config: &PiperVoiceConfig) -> [f32; 3] {
    [
        config.noise_scale.unwrap_or(0.667),
        config.length_scale.unwrap_or(1.0),
        config.noise_w.unwrap_or(0.8),
    ]
}

#[cfg(feature = "piper-compat")]
fn report_link_grammar(sentence_analysis: &Option<SentenceAnalysis>) {
    let Some(analysis) = sentence_analysis else {
        println!("Riper link grammar:");
        println!("  (unavailable for this phoneme override)");
        return;
    };

    println!("Riper link grammar:");
    println!("  tokens: {}", format_sentence_tokens(analysis));
    let noun_phrases = format_noun_compounds(analysis);
    if !noun_phrases.is_empty() {
        println!("  noun phrases: {}", noun_phrases.join("; "));
    }
    for (index, parse) in analysis.link_parses.iter().enumerate() {
        println!(
            "  parse #{} rank {:.2}: {}",
            index + 1,
            parse.rank,
            format_syntactic_links(analysis, parse)
        );
    }
}

#[cfg(feature = "piper-compat")]
fn format_sentence_tokens(analysis: &SentenceAnalysis) -> String {
    analysis
        .tokens
        .iter()
        .filter_map(|token| {
            token
                .word_index
                .map(|word_index| format!("{word_index}:{}:{:?}", token.text, token.pos))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(feature = "piper-compat")]
fn format_noun_compounds(analysis: &SentenceAnalysis) -> Vec<String> {
    let words = word_texts(analysis);
    analysis
        .link_parses
        .first()
        .map(|parse| {
            parse
                .links
                .iter()
                .filter(|link| link.kind == SyntacticLinkKind::NounCompound)
                .filter_map(|link| {
                    let left = words.get(link.left)?.as_ref()?;
                    let right = words.get(link.right)?.as_ref()?;
                    Some(format!("{left} {right}"))
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(feature = "piper-compat")]
fn format_syntactic_links(analysis: &SentenceAnalysis, parse: &SyntacticLinkParse) -> String {
    if parse.links.is_empty() {
        return "(none)".to_string();
    }

    let words = word_texts(analysis);
    parse
        .links
        .iter()
        .filter_map(|link| {
            let left = words.get(link.left)?.as_ref()?;
            let right = words.get(link.right)?.as_ref()?;
            Some(format!(
                "{left} -{:?}/{:.2}-> {right}",
                link.kind, link.confidence
            ))
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(feature = "piper-compat")]
fn word_texts(analysis: &SentenceAnalysis) -> Vec<Option<String>> {
    let mut words = Vec::new();
    for token in &analysis.tokens {
        let Some(word_index) = token.word_index else {
            continue;
        };
        if words.len() <= word_index {
            words.resize(word_index + 1, None);
        }
        words[word_index] = Some(token.text.clone());
    }
    words
}

#[cfg(feature = "piper-compat")]
fn render_phoneme_result(result: &std::result::Result<String, String>) -> String {
    match result {
        Ok(value) if value.is_empty() => "(empty)".to_string(),
        Ok(value) => value.clone(),
        Err(error) => format!("(unavailable: {error})"),
    }
}

#[cfg(feature = "piper-compat")]
fn format_phoneme_sequence(sequence: &PiperPhonemeSequence) -> String {
    sequence
        .phonemes
        .iter()
        .map(|phoneme| phoneme.0.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_phone_string(phone_string: &PhoneString) -> String {
    phone_string.ipa_segments().join(" ")
}

#[cfg(feature = "piper-compat")]
fn klatt_phone_string_from_riper_symbols(sequence: &PiperPhonemeSequence) -> Result<PhoneString> {
    let mut phones = Vec::new();
    let mut unsupported_symbols = Vec::new();

    for phoneme in &sequence.phonemes {
        match klatt_ipa_segments_for_riper_symbol(&phoneme.0) {
            Some(segments) => phones.extend(segments.iter().copied().map(Phone::new_ipa)),
            None => unsupported_symbols.push(phoneme.0.clone()),
        }
    }

    if !unsupported_symbols.is_empty() {
        unsupported_symbols.sort_unstable();
        unsupported_symbols.dedup();
        anyhow::bail!(
            "cannot convert Riper phoneme(s) for acoustic phones: {}",
            unsupported_symbols.join(", ")
        );
    }

    Ok(PhoneString { phones })
}

#[cfg(feature = "piper-compat")]
fn phone_segments_for_riper_symbols(symbols: &[&str]) -> std::result::Result<String, String> {
    let mut phones = Vec::new();
    let mut unsupported_symbols = Vec::new();
    for symbol in symbols {
        match klatt_ipa_segments_for_riper_symbol(symbol) {
            Some(segments) => phones.extend(segments.iter().copied()),
            None => unsupported_symbols.push((*symbol).to_string()),
        }
    }
    if unsupported_symbols.is_empty() {
        Ok(phones.join(" "))
    } else {
        unsupported_symbols.sort_unstable();
        unsupported_symbols.dedup();
        Err(unsupported_symbols.join(", "))
    }
}

#[cfg(feature = "piper-compat")]
fn backend_phone_translation_note(args: &SayArgs) -> String {
    if should_use_klatt_backend(args) {
        "acoustic IPA targets -> Klatt formant renderer".to_string()
    } else if should_use_source_filter_hifigan_backend(args) {
        "acoustic IPA targets -> source-filter mel/F0 -> mel debug renderer".to_string()
    } else if should_use_speecht5_backend(args) {
        "Riper phonemes -> SpeechT5 tokenizer/model tokens -> SpeechT5 mel -> HiFi-GAN".to_string()
    } else if should_use_mbrola_backend(args) {
        "Riper surface phones -> MBROLA voice symbol map -> timed .pho plan".to_string()
    } else if args.piper {
        "Riper phonemes -> Piper/eSpeak-compatible symbols -> external Piper IDs".to_string()
    } else {
        "Riper phonemes -> Piper/eSpeak-compatible symbols -> internal Piper IDs".to_string()
    }
}

#[cfg(feature = "piper-compat")]
fn report_compare_stats(process: &SynthesisStats, riper: &SynthesisStats) {
    println!("process runtime: {}ms", process.runtime.as_millis());
    println!("Riper inference: {}ms", riper.runtime.as_millis());
    println!(
        "process sample rate: {} Hz | Riper sample rate: {} Hz",
        process.audio.sample_rate_hz, riper.audio.sample_rate_hz
    );
    println!(
        "process duration: {:.2}ms | Riper duration: {:.2}ms",
        process.audio.duration_ms, riper.audio.duration_ms
    );
    println!(
        "process samples: {} | Riper samples: {}",
        process.audio.sample_count, riper.audio.sample_count
    );
    println!(
        "process rms/peak: {:.5}/{:.5} | Riper rms/peak: {:.5}/{:.5}",
        process.audio.rms, process.audio.peak_abs, riper.audio.rms, riper.audio.peak_abs
    );
}

#[derive(Debug, Clone)]
struct SayArgs {
    piper: bool,
    piper_bin: Option<PathBuf>,
    piper_voice: Option<PathBuf>,
    #[cfg(feature = "piper-compat")]
    hifigan_model: Option<PathBuf>,
    mbrola: bool,
    mbrola_voice: Option<PathBuf>,
    output_wav: Option<PathBuf>,
    dump_pipeline: bool,
    dump_phonemes: bool,
    dump_phone_plan: bool,
    dump_piper_tensors: bool,
    klatt: bool,
    hifigan: bool,
    speecht5: bool,
    skip_gan: bool,
    stdin_stream: bool,
    text: String,
}

impl SayArgs {
    fn clone_for_text(&self, text: &str) -> Self {
        let mut args = self.clone();
        args.text = text.to_string();
        args
    }

    fn from_command(command: SayCommand) -> Result<Self> {
        let mut piper = command.piper;
        let mut klatt = command.klatt;
        let mut hifigan = command.hifigan;
        let mut speecht5 = command.speecht5;
        let mut skip_gan = command.skip_gan;
        let mut dump_pipeline = command.dump_pipeline;
        let mut dump_phonemes = command.dump_phonemes;
        let mut dump_phone_plan = command.dump_phone_plan;
        let mut dump_piper_tensors = command.dump_piper_tensors;
        let mut rp = command.rp;
        let mut diphone = command.diphone;
        let mut words = command
            .words
            .into_iter()
            .filter_map(|word| {
                if word == "--piper" {
                    piper = true;
                    None
                } else if word == "--klatt" {
                    klatt = true;
                    None
                } else if word == "--hifigan" {
                    hifigan = true;
                    None
                } else if word == "--speecht5" {
                    speecht5 = true;
                    None
                } else if word == "--skip-gan" || word == "--hifigan-fallback" {
                    skip_gan = true;
                    None
                } else if word == "--dump-pipeline" || word == "--trace-speech-pipeline" {
                    dump_pipeline = true;
                    None
                } else if word == "--dump-phonemes" {
                    dump_phonemes = true;
                    None
                } else if word == "--dump-phone-plan" {
                    dump_phone_plan = true;
                    None
                } else if word == "--dump-piper-tensors" {
                    dump_piper_tensors = true;
                    None
                } else if word == "--riper" {
                    None
                } else if word == "--diphone" {
                    diphone = true;
                    None
                } else if word == "--rp" {
                    rp = true;
                    None
                } else {
                    Some(word)
                }
            })
            .collect::<Vec<_>>();
        let explicit_piper_bin = command.piper_bin.is_some();
        let explicit_hifigan_model = command.hifigan_model.is_some();
        let mut piper_bin = command.piper_bin;
        let mut piper_voice = command.piper_voice;

        if piper_bin.is_none() && words.first().is_some_and(|word| looks_like_piper_bin(word)) {
            piper_bin = Some(PathBuf::from(words.remove(0)));
        }

        if piper_voice.is_none() && words.first().is_some_and(|word| word.ends_with(".onnx")) {
            piper_voice = Some(PathBuf::from(words.remove(0)));
        }

        let mut mbrola_voice = command.mbrola_voice;
        if rp && mbrola_voice.is_none() {
            mbrola_voice = Some(received_pronunciation_mbrola_voice());
        }

        let mbrola = diphone || mbrola_voice.is_some() || rp;
        if words.is_empty() && mbrola {
            words.push("Hello, my baby.".to_string());
        }
        anyhow::ensure!(!words.is_empty(), "missing text to speak; try `say hello`");
        anyhow::ensure!(
            !(piper && (klatt || hifigan || speecht5 || mbrola)),
            "listenbury say: --piper cannot be combined with --klatt, --hifigan, --speecht5, or --diphone"
        );
        anyhow::ensure!(
            hifigan || !skip_gan,
            "listenbury say: --skip-gan only applies when --hifigan is selected"
        );
        anyhow::ensure!(
            piper || !explicit_piper_bin,
            "listenbury say: --piper-bin only applies to the external Piper binary; pass --piper"
        );
        anyhow::ensure!(
            hifigan || speecht5 || !explicit_hifigan_model,
            "listenbury say: --hifigan-model only applies when --hifigan or --speecht5 is selected"
        );
        anyhow::ensure!(
            [klatt, hifigan, speecht5, mbrola]
                .into_iter()
                .filter(|set| *set)
                .count()
                <= 1,
            "listenbury say: choose only one of --klatt, --hifigan, --speecht5, or the MBROLA/RP voice path"
        );
        let stdin_stream = words.len() == 1 && words[0] == "-";

        Ok(Self {
            piper_bin,
            piper_voice,
            #[cfg(feature = "piper-compat")]
            hifigan_model: command.hifigan_model,
            mbrola,
            mbrola_voice,
            output_wav: command.output_wav,
            dump_pipeline,
            dump_phonemes,
            dump_phone_plan,
            dump_piper_tensors,
            piper,
            klatt,
            hifigan,
            speecht5,
            skip_gan,
            stdin_stream,
            text: if stdin_stream {
                String::new()
            } else {
                words.join(" ")
            },
        })
    }
}

fn received_pronunciation_mbrola_voice() -> PathBuf {
    PathBuf::from("data/mbrola/en1/en1")
}

fn should_use_klatt_backend(args: &SayArgs) -> bool {
    args.klatt
}

fn should_use_source_filter_hifigan_backend(args: &SayArgs) -> bool {
    args.hifigan && args.skip_gan
}

fn should_use_speecht5_backend(args: &SayArgs) -> bool {
    args.speecht5 || (args.hifigan && !args.skip_gan)
}

fn should_use_mbrola_backend(args: &SayArgs) -> bool {
    args.mbrola
}

fn say_backend_kind(args: &SayArgs) -> CurrentSayBackendKind {
    if should_use_klatt_backend(args) {
        CurrentSayBackendKind::Klatt
    } else if should_use_source_filter_hifigan_backend(args) {
        CurrentSayBackendKind::SourceFilterHifigan
    } else if should_use_speecht5_backend(args) {
        CurrentSayBackendKind::SpeechT5Hifigan
    } else if should_use_mbrola_backend(args) {
        CurrentSayBackendKind::MbrolaDiphone
    } else if args.piper {
        CurrentSayBackendKind::PiperProcess
    } else {
        CurrentSayBackendKind::PiperCompat
    }
}

fn say_speech_loom(args: &SayArgs) -> SpeechLoom {
    say_backend_kind(args).loom()
}

fn say_backend_graph(args: &SayArgs) -> CurrentBackendGraphView {
    say_backend_kind(args).current_backend_graph()
}

fn print_say_pipeline(args: &SayArgs) {
    print!("{}", format_say_pipeline(args));
}

fn print_say_phonemes(args: &SayArgs) -> Result<()> {
    print!("{}", format_say_phonemes(args)?);
    Ok(())
}

fn print_say_phone_plan(args: &SayArgs) -> Result<()> {
    let plan = PhonePlan::from_text_with_riper_g2p(&args.text)?;
    println!("{}", serde_json::to_string_pretty(&plan)?);
    Ok(())
}

fn print_say_piper_tensors(args: &SayArgs) -> Result<()> {
    print!("{}", format_say_piper_tensors(args)?);
    Ok(())
}

fn format_say_pipeline(args: &SayArgs) -> String {
    let backend_graph = say_backend_graph(args);
    let mut output = String::new();
    output.push_str(&format!("speech pipeline: {}\n", backend_graph.id));
    output.push_str("input text\n");
    for stage in say_pipeline_stages(args) {
        output.push_str(&format!("  -> {stage}\n"));
    }
    output.push_str("workers:\n");
    for worker in backend_graph.workers {
        output.push_str(&format!("  - {}\n", worker.id));
    }
    output
}

#[cfg(feature = "piper-compat")]
fn format_say_phonemes(args: &SayArgs) -> Result<String> {
    let unit = SimpleEnglishG2p::default()
        .phonemize_unit(&args.text)
        .with_context(|| format!("failed to phonemize `{}`", args.text))?;
    let riper_symbols = format_phoneme_sequence(&unit.phonemes);
    let acoustic_phones = klatt_phone_string_from_riper_symbols(&unit.phonemes)?;

    let mut output = String::new();
    output.push_str(&format!("phoneme dump: {}\n", say_backend_graph(args).id));
    output.push_str(&format!("input text: {}\n", args.text));
    output.push_str(&format!("riper phonemes: {riper_symbols}\n"));
    output.push_str(&format!(
        "acoustic phones: {}\n",
        format_phone_string(&acoustic_phones)
    ));
    output.push_str("word phones:\n");
    for target in &unit.word_targets {
        let symbols = unit.phonemes.phonemes[target.phoneme_range.clone()]
            .iter()
            .map(|phoneme| phoneme.0.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let phones = phone_segments_for_riper_symbols(
            &unit.phonemes.phonemes[target.phoneme_range.clone()]
                .iter()
                .map(|phoneme| phoneme.0.as_str())
                .collect::<Vec<_>>(),
        )
        .unwrap_or_else(|error| format!("(unavailable: {error})"));
        output.push_str(&format!(
            "  - {}: {} -> {}\n",
            target.normalized_text, symbols, phones
        ));
    }
    output.push_str(&format!(
        "backend phones: {}\n",
        backend_phone_translation_note(args)
    ));
    Ok(output)
}

#[cfg(feature = "piper-compat")]
fn format_say_piper_tensors(args: &SayArgs) -> Result<String> {
    anyhow::ensure!(
        !should_use_klatt_backend(args)
            && !should_use_source_filter_hifigan_backend(args)
            && !should_use_speecht5_backend(args)
            && !should_use_mbrola_backend(args),
        "listenbury say --dump-piper-tensors only applies to --piper or the default piper-compat route"
    );

    let piper_voice = resolve_piper_voice(args.piper_voice.clone())?;
    let piper_config = if args.piper {
        let piper_bin = resolve_piper_bin(args.piper_bin.clone())?;
        piper_config_for_voice(piper_bin, piper_voice)?
    } else {
        piper_config_for_riper_voice(piper_voice)?
    };
    let config_path = piper_config
        .config_path
        .clone()
        .unwrap_or_else(|| piper_config.model_path.with_extension("onnx.json"));
    let voice_config = read_riper_voice_config(&config_path)?;
    let unit = SimpleEnglishG2p::default()
        .phonemize_unit(&args.text)
        .with_context(|| format!("failed to phonemize `{}`", args.text))?;
    let trace = unit
        .phonemes
        .to_piper_text_id_trace(&voice_config)
        .with_context(|| {
            format!(
                "failed to build Piper ID trace for model config {}",
                config_path.display()
            )
        })?;
    let contract = RiperBackend::load(&piper_config.model_path, voice_config.clone())
        .and_then(|backend| backend.validate_model_contract())
        .with_context(|| {
            format!(
                "failed to inspect Piper ONNX contract for {}",
                piper_config.model_path.display()
            )
        })?;
    Ok(format_piper_tensor_dump(
        say_backend_graph(args).id,
        &args.text,
        &piper_config.model_path,
        &config_path,
        args.piper,
        &trace,
        &voice_config,
        &contract.input_names,
    ))
}

#[cfg(not(feature = "piper-compat"))]
fn format_say_piper_tensors(_args: &SayArgs) -> Result<String> {
    anyhow::bail!("listenbury say --dump-piper-tensors requires the `piper-compat` feature")
}

#[cfg(not(feature = "piper-compat"))]
fn format_say_phonemes(args: &SayArgs) -> Result<String> {
    let acoustic_phones = klatt_phone_string_for_text(&args.text)?;
    let mut output = String::new();
    output.push_str(&format!("phoneme dump: {}\n", say_backend_graph(args).id));
    output.push_str(&format!("input text: {}\n", args.text));
    output.push_str(&format!(
        "acoustic phones: {}\n",
        format_phone_string(&acoustic_phones)
    ));
    output.push_str("backend phones: klatt/demo lexicon -> acoustic IPA targets\n");
    Ok(output)
}

fn say_pipeline_stages(args: &SayArgs) -> Vec<String> {
    let mut stages = vec![
        "text normalizer: Riper/SimpleEnglishG2p".to_string(),
        format!("language variety: {}", say_language_variety(args)),
        "tokenizer: Riper sentence analysis".to_string(),
        "pronunciation/rules: language-pack English G2P".to_string(),
        "phones".to_string(),
        "syllables".to_string(),
        "timing plan".to_string(),
    ];

    if should_use_klatt_backend(args) {
        stages.extend([
            "acoustic generator: klatt".to_string(),
            "mel/features: disabled".to_string(),
            "vocoder: disabled".to_string(),
        ]);
    } else if should_use_source_filter_hifigan_backend(args) {
        stages.extend([
            "acoustic generator: source-filter".to_string(),
            hifigan_feature_stage(),
            hifigan_vocoder_stage(args),
        ]);
    } else if should_use_speecht5_backend(args) {
        stages.extend([
            "tokenizer: SpeechT5 tokenizer".to_string(),
            "acoustic generator: SpeechT5 encoder/decoder ONNX".to_string(),
            "mel/features: SpeechT5 mel spectrogram".to_string(),
            speecht5_vocoder_stage(args),
        ]);
    } else if should_use_mbrola_backend(args) {
        stages.extend([
            "acoustic generator: MBROLA-compatible diphone renderer".to_string(),
            mbrola_voice_stage(args),
            "mel/features: disabled".to_string(),
            "vocoder: disabled".to_string(),
        ]);
    } else if args.piper {
        stages.extend([
            "external process: piper".to_string(),
            "acoustic generator: piper process fused".to_string(),
            "mel/features: piper process internal".to_string(),
            "vocoder: piper process internal".to_string(),
        ]);
    } else {
        stages.extend([
            "acoustic generator: piper-compatible ONNX/Riper".to_string(),
            "mel/features: piper-compatible internal".to_string(),
            "vocoder: piper-compatible internal".to_string(),
        ]);
    }

    stages.push(output_stage(args));
    stages
}

fn say_language_variety(args: &SayArgs) -> &'static str {
    if args.mbrola_voice.as_deref() == Some(received_pronunciation_mbrola_voice().as_path()) {
        "en-GB-RP"
    } else {
        "en-US"
    }
}

fn hifigan_feature_stage() -> String {
    let smoothing = std::env::var_os("LISTENBURY_HIFIGAN_TEMPORAL_SMOOTHING")
        .map(|value| value.to_string_lossy().trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "0.0".to_string());
    format!(
        "mel/features: source-filter spectral proxy for HiFi-GAN, temporal smoothing {smoothing}"
    )
}

fn hifigan_vocoder_stage(args: &SayArgs) -> String {
    if args.skip_gan {
        "vocoder: deterministic-fallback mel debug renderer (explicit --skip-gan/--hifigan-fallback)".to_string()
    } else {
        #[cfg(feature = "piper-compat")]
        if let Some(model) = &args.hifigan_model {
            return format!("vocoder: hifigan ONNX ({})", model.display());
        }
        "vocoder: hifigan ONNX (model resolved at runtime)".to_string()
    }
}

fn speecht5_vocoder_stage(args: &SayArgs) -> String {
    if args.skip_gan {
        return "vocoder: deterministic-fallback mel debug renderer (explicit --skip-gan/--hifigan-fallback)".to_string();
    }
    #[cfg(feature = "piper-compat")]
    if let Some(model) = &args.hifigan_model {
        return format!("vocoder: SpeechT5 HiFi-GAN ONNX ({})", model.display());
    }
    "vocoder: SpeechT5 HiFi-GAN ONNX (model resolved at runtime)".to_string()
}

fn mbrola_voice_stage(args: &SayArgs) -> String {
    match &args.mbrola_voice {
        Some(path) => format!("diphone voice: {}", path.display()),
        None => "diphone voice: default MBROLA-compatible voice".to_string(),
    }
}

fn output_stage(args: &SayArgs) -> String {
    match &args.output_wav {
        Some(path) => format!("wav writer: {}", path.display()),
        None => "output: speaker playback".to_string(),
    }
}

#[cfg(feature = "piper-compat")]
fn synthesize_hifigan_for_say(args: &SayArgs) -> Result<Vec<AudioFrame>> {
    synthesize_hifigan_text(
        &args.text,
        args.hifigan_model.clone(),
        args.skip_gan,
        "listenbury say --hifigan",
    )
}

#[cfg(feature = "piper-compat")]
fn synthesize_hifigan_text(
    text: &str,
    hifigan_model: Option<PathBuf>,
    skip_gan: bool,
    command_label: &str,
) -> Result<Vec<AudioFrame>> {
    let phone_string = klatt_phone_string_for_text(text)?;
    let target_table = default_english_phone_targets();
    let missing_phones: Vec<String> = phone_string
        .phones
        .iter()
        .map(|phone| phone.ipa.as_str())
        .filter(|ipa| !target_table.contains_key(*ipa))
        .map(str::to_string)
        .collect();
    anyhow::ensure!(
        missing_phones.is_empty(),
        "{command_label} cannot render phone(s): {}",
        missing_phones.join(", ")
    );
    let phone_targets =
        phone_render_targets_from_string(&phone_string, Some(150.0), 0.7, &target_table)
            .into_iter()
            .map(|target| PhoneTimedRenderTarget {
                phone: target.phone,
                duration_ms: target.duration_ms,
                f0_hz: target.f0_hz,
                amplitude: target.amplitude,
                vibrato: target.vibrato,
            })
            .collect::<Vec<_>>();
    let mut acoustic = SourceFilterAcousticModel;
    let acoustic_track = acoustic
        .generate(AcousticInput::PhoneTimed(&phone_targets))
        .with_context(|| {
            format!("{command_label} failed to generate acoustic frames for `{text}`")
        })?;
    HifiganBackend::validate_acoustic_contract(
        acoustic_track.sample_rate_hz,
        acoustic_track.hop_samples,
    )?;
    let temporal_smoothing = hifigan_temporal_smoothing_amount()?;
    let raw_mel = &acoustic_track.mel;
    let raw_discontinuity = summarize_mel_temporal_discontinuity(raw_mel);
    let smoothed_mel = if temporal_smoothing > 0.0 {
        Some(temporal_smooth_mel_frames(raw_mel, temporal_smoothing))
    } else {
        None
    };
    let hifigan_input_mel = smoothed_mel.as_deref().unwrap_or(raw_mel);
    let hifigan_input_discontinuity = summarize_mel_temporal_discontinuity(hifigan_input_mel);
    let source_filter_frames = maybe_render_source_filter_reference(
        hifigan_input_mel,
        &acoustic_track.f0_hz,
        &acoustic_track.voiced,
    )?;
    let mut backend = if skip_gan {
        eprintln!(
            "WARNING: {command_label} using deterministic-fallback mel debug renderer, not ONNX HiFi-GAN."
        );
        tracing::warn!(
            command = command_label,
            mode = "deterministic-fallback",
            "HiFi-GAN route is using the explicit mel debug renderer"
        );
        Box::new(MelDebugRendererBackend::new()) as Box<dyn SpeechSynthesizer>
    } else {
        let model_path = resolve_hifigan_model(hifigan_model)?;
        eprintln!(
            "{command_label} using ONNX HiFi-GAN model: {}",
            model_path.display()
        );
        tracing::info!(
            command = command_label,
            mode = "onnx",
            model = %model_path.display(),
            "HiFi-GAN route is using ONNX vocoding"
        );
        Box::new(HifiganBackend::load(model_path)?) as Box<dyn SpeechSynthesizer>
    };
    let frames = backend
        .render(VocoderInput::MelF0 {
            mel: hifigan_input_mel,
            f0_hz: &acoustic_track.f0_hz,
            voiced: &acoustic_track.voiced,
        })
        .with_context(|| format!("{command_label} failed to render `{text}`"))?;
    anyhow::ensure!(
        !frames.is_empty(),
        "{command_label} produced no audio for `{text}`"
    );
    maybe_write_hifigan_debug_artifacts(
        text,
        &acoustic_track,
        raw_mel,
        hifigan_input_mel,
        temporal_smoothing,
        raw_discontinuity,
        hifigan_input_discontinuity,
        &source_filter_frames,
        &frames,
    )?;
    Ok(frames)
}

#[cfg(not(feature = "piper-compat"))]
fn synthesize_hifigan_for_say(_args: &SayArgs) -> Result<Vec<AudioFrame>> {
    anyhow::bail!("listenbury say --hifigan requires the `piper-compat` feature")
}

#[cfg(feature = "piper-compat")]
fn synthesize_speecht5_for_say(args: &SayArgs) -> Result<Vec<AudioFrame>> {
    let acoustic_dir = resolve_speecht5_acoustic_dir()?;
    let mut acoustic =
        SpeechT5OnnxAcousticGenerator::load(SpeechT5OnnxPaths::from_dir(&acoustic_dir))
            .with_context(|| {
                format!(
                    "failed to load SpeechT5 acoustic model from {}",
                    acoustic_dir.display()
                )
            })?;
    let acoustic_track = acoustic
        .generate_text(&args.text)
        .with_context(|| format!("SpeechT5 failed to generate mel frames for `{}`", args.text))?;
    HifiganBackend::validate_acoustic_contract(
        acoustic_track.sample_rate_hz,
        acoustic_track.hop_samples,
    )?;

    let mut backend = if args.skip_gan {
        eprintln!(
            "WARNING: listenbury say --hifigan using deterministic-fallback mel debug renderer, not ONNX HiFi-GAN."
        );
        Box::new(MelDebugRendererBackend::new()) as Box<dyn SpeechSynthesizer>
    } else {
        let model_path = resolve_hifigan_model(args.hifigan_model.clone())?;
        eprintln!(
            "listenbury say using SpeechT5 acoustic model: {}",
            acoustic_dir.display()
        );
        eprintln!(
            "listenbury say using SpeechT5 HiFi-GAN model: {}",
            model_path.display()
        );
        Box::new(HifiganBackend::load(model_path)?) as Box<dyn SpeechSynthesizer>
    };
    let frames = backend
        .render(VocoderInput::MelF0 {
            mel: &acoustic_track.mel,
            f0_hz: &acoustic_track.f0_hz,
            voiced: &acoustic_track.voiced,
        })
        .with_context(|| format!("SpeechT5 HiFi-GAN failed to render `{}`", args.text))?;
    anyhow::ensure!(
        !frames.is_empty(),
        "SpeechT5 HiFi-GAN produced no audio for `{}`",
        args.text
    );
    Ok(frames)
}

#[cfg(not(feature = "piper-compat"))]
fn synthesize_speecht5_for_say(_args: &SayArgs) -> Result<Vec<AudioFrame>> {
    anyhow::bail!("listenbury say --speecht5 requires the `piper-compat` feature")
}

#[cfg(feature = "piper-compat")]
fn maybe_render_source_filter_reference(
    mel: &[MelFrame],
    f0_hz: &[f32],
    voiced: &[bool],
) -> Result<Vec<AudioFrame>> {
    if std::env::var_os("LISTENBURY_HIFIGAN_DEBUG_DIR").is_none() {
        return Ok(Vec::new());
    }
    let mut debug_renderer = MelDebugRendererBackend::new();
    debug_renderer
        .render(VocoderInput::MelF0 { mel, f0_hz, voiced })
        .context("failed to render mel debug source-filter A/B reference")
}

#[cfg(feature = "piper-compat")]
fn maybe_write_hifigan_debug_artifacts(
    text: &str,
    acoustic_track: &AcousticFrameTrack,
    raw_mel: &[MelFrame],
    hifigan_input_mel: &[MelFrame],
    smoothing_amount: f32,
    raw_discontinuity: MelTemporalDiscontinuityStats,
    hifigan_input_discontinuity: MelTemporalDiscontinuityStats,
    source_filter_frames: &[AudioFrame],
    hifigan_frames: &[AudioFrame],
) -> Result<()> {
    let Some(debug_dir) = std::env::var_os("LISTENBURY_HIFIGAN_DEBUG_DIR").map(PathBuf::from)
    else {
        return Ok(());
    };
    std::fs::create_dir_all(&debug_dir).with_context(|| {
        format!(
            "failed to create HiFi-GAN debug directory {}",
            debug_dir.display()
        )
    })?;
    let stem = format!(
        "utterance-{}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        std::process::id()
    );

    let raw_mel_path = debug_dir.join(format!("{stem}-source-filter-mel-raw.txt"));
    write_hifigan_mel_dump(&raw_mel_path, text, acoustic_track, raw_mel).with_context(|| {
        format!(
            "failed to write HiFi-GAN raw source-filter mel debug dump {}",
            raw_mel_path.display()
        )
    })?;

    let mel_path = debug_dir.join(format!("{stem}-pre-vocoder-mel.txt"));
    write_hifigan_mel_dump(&mel_path, text, acoustic_track, hifigan_input_mel).with_context(
        || {
            format!(
                "failed to write HiFi-GAN vocoder-input mel debug dump {}",
                mel_path.display()
            )
        },
    )?;

    let diagnostics_path = debug_dir.join(format!("{stem}-temporal-diagnostics.txt"));
    let mut diagnostics = String::new();
    diagnostics.push_str(&format!("text={text}\n"));
    diagnostics.push_str("mel_contract=source-filter-spectral-proxy (shape/timing validated)\n");
    diagnostics.push_str(&format!(
        "temporal_smoothing_amount={smoothing_amount:.3}\n"
    ));
    diagnostics.push_str(&format!(
        "raw_frame_pairs={} raw_mean_abs_delta={:.6} raw_p95_abs_delta={:.6} raw_max_abs_delta={:.6}\n",
        raw_discontinuity.frame_pairs,
        raw_discontinuity.mean_abs_delta,
        raw_discontinuity.p95_abs_delta,
        raw_discontinuity.max_abs_delta
    ));
    diagnostics.push_str(&format!(
        "input_frame_pairs={} input_mean_abs_delta={:.6} input_p95_abs_delta={:.6} input_max_abs_delta={:.6}\n",
        hifigan_input_discontinuity.frame_pairs,
        hifigan_input_discontinuity.mean_abs_delta,
        hifigan_input_discontinuity.p95_abs_delta,
        hifigan_input_discontinuity.max_abs_delta
    ));
    diagnostics.push_str(&format!(
        "artifact_attribution={}\n",
        hifigan_artifact_attribution(
            raw_discontinuity,
            hifigan_input_discontinuity,
            smoothing_amount,
        )
    ));
    std::fs::write(&diagnostics_path, diagnostics).with_context(|| {
        format!(
            "failed to write HiFi-GAN temporal diagnostics {}",
            diagnostics_path.display()
        )
    })?;

    tracing::debug!(
        raw_mean_abs_delta = raw_discontinuity.mean_abs_delta,
        input_mean_abs_delta = hifigan_input_discontinuity.mean_abs_delta,
        smoothing_amount,
        attribution = %hifigan_artifact_attribution(
            raw_discontinuity,
            hifigan_input_discontinuity,
            smoothing_amount,
        ),
        "hifigan temporal modulation diagnostics"
    );

    if !source_filter_frames.is_empty() {
        let source_filter_path = debug_dir.join(format!("{stem}-source-filter-reference.wav"));
        write_wav(&source_filter_path, source_filter_frames).with_context(|| {
            format!(
                "failed to write HiFi-GAN source-filter reference {}",
                source_filter_path.display()
            )
        })?;
    }
    let hifigan_path = debug_dir.join(format!("{stem}-hifigan-output.wav"));
    write_wav(&hifigan_path, hifigan_frames).with_context(|| {
        format!(
            "failed to write HiFi-GAN debug wav {}",
            hifigan_path.display()
        )
    })?;

    tracing::debug!(
        debug_dir = %debug_dir.display(),
        raw_mel_dump = %raw_mel_path.display(),
        mel_dump = %mel_path.display(),
        diagnostics = %diagnostics_path.display(),
        source_filter_frames = source_filter_frames.len(),
        hifigan_frames = hifigan_frames.len(),
        "wrote HiFi-GAN debug artifacts"
    );
    Ok(())
}

#[cfg(feature = "piper-compat")]
fn write_hifigan_mel_dump(
    path: &Path,
    text: &str,
    acoustic_track: &AcousticFrameTrack,
    mel: &[MelFrame],
) -> Result<()> {
    let mut mel_dump = String::new();
    mel_dump.push_str(&format!("text={text}\n"));
    mel_dump.push_str(&format!(
        "sample_rate_hz={} hop_samples={} frame_count={} mel_bins={}\n",
        acoustic_track.sample_rate_hz,
        acoustic_track.hop_samples,
        mel.len(),
        mel.first().map(|frame| frame.bins.len()).unwrap_or(0)
    ));
    for frame in mel {
        let row = frame
            .bins
            .iter()
            .map(|value| format!("{value:.6}"))
            .collect::<Vec<_>>()
            .join(",");
        mel_dump.push_str(&row);
        mel_dump.push('\n');
    }
    std::fs::write(path, mel_dump)?;
    Ok(())
}

#[cfg(feature = "piper-compat")]
fn hifigan_temporal_smoothing_amount() -> Result<f32> {
    let Some(value) = std::env::var_os("LISTENBURY_HIFIGAN_TEMPORAL_SMOOTHING")
        .map(|value| value.to_string_lossy().trim().to_string())
    else {
        return Ok(0.0);
    };
    if value.is_empty() {
        return Ok(0.0);
    }
    let amount = value.parse::<f32>().with_context(|| {
        format!("invalid LISTENBURY_HIFIGAN_TEMPORAL_SMOOTHING value `{value}`")
    })?;
    anyhow::ensure!(
        (0.0..=1.0).contains(&amount),
        "LISTENBURY_HIFIGAN_TEMPORAL_SMOOTHING must be in [0.0, 1.0], got {amount}"
    );
    Ok(amount)
}

#[cfg(feature = "piper-compat")]
fn hifigan_artifact_attribution(
    raw: MelTemporalDiscontinuityStats,
    input: MelTemporalDiscontinuityStats,
    smoothing_amount: f32,
) -> &'static str {
    let temporal_banding_detected = raw.mean_abs_delta
        >= HIFIGAN_TEMPORAL_BANDING_MEAN_DELTA_THRESHOLD
        || raw.p95_abs_delta >= HIFIGAN_TEMPORAL_BANDING_P95_DELTA_THRESHOLD;
    let smoothing_reduced_modulation = smoothing_amount > 0.0
        && input.mean_abs_delta <= raw.mean_abs_delta * HIFIGAN_SMOOTHING_EFFECT_RATIO;
    if temporal_banding_detected && smoothing_reduced_modulation {
        "temporal_banding_primary"
    } else if temporal_banding_detected {
        "temporal_banding_present_contract_or_model_mismatch_also_possible"
    } else {
        "contract_or_representation_mismatch_more_likely_than_temporal_banding"
    }
}

fn synthesize_klatt_for_say(text: &str) -> Result<Vec<AudioFrame>> {
    let phone_string = klatt_phone_string_for_text(text)?;
    let config = KlattRenderConfig::default();
    let target_table = default_english_phone_targets();
    let targets = phone_render_targets_from_string(&phone_string, Some(150.0), 0.7, &target_table);
    let missing_phones: Vec<String> = phone_string
        .phones
        .iter()
        .map(|phone| phone.ipa.as_str())
        .filter(|ipa| !target_table.contains_key(*ipa))
        .map(str::to_string)
        .collect();
    anyhow::ensure!(
        missing_phones.is_empty(),
        "listenbury say --klatt cannot render phone(s): {}",
        missing_phones.join(", ")
    );

    let pcm = render_phone_string(&targets, &config);
    anyhow::ensure!(
        !pcm.is_empty(),
        "listenbury say --klatt produced no audio for `{text}`"
    );
    Ok(vec![AudioFrame {
        captured_at: ExactTimestamp::now(),
        sample_rate_hz: config.sample_rate,
        channels: 1,
        samples: pcm,
        voice_signatures: Vec::new(),
    }])
}

fn klatt_phone_string_for_text(text: &str) -> Result<PhoneString> {
    #[cfg(feature = "piper-compat")]
    {
        klatt_phone_string_from_riper(text)
    }

    #[cfg(not(feature = "piper-compat"))]
    {
        klatt_phone_string_from_demo_lexicon(text)
    }
}

#[cfg(feature = "piper-compat")]
fn klatt_phone_string_from_riper(text: &str) -> Result<PhoneString> {
    let unit = SimpleEnglishG2p::default()
        .phonemize_unit(text)
        .with_context(|| format!("listenbury say --klatt could not phonemize `{text}`"))?;
    let mut phones = Vec::new();
    let mut unsupported_symbols = Vec::new();

    for phoneme in &unit.phonemes.phonemes {
        match klatt_ipa_segments_for_riper_symbol(&phoneme.0) {
            Some(segments) => phones.extend(segments.iter().copied().map(Phone::new_ipa)),
            None => unsupported_symbols.push(phoneme.0.clone()),
        }
    }

    anyhow::ensure!(
        !phones.is_empty(),
        "listenbury say --klatt could not find any speakable phones in `{text}`"
    );
    if !unsupported_symbols.is_empty() {
        unsupported_symbols.sort_unstable();
        unsupported_symbols.dedup();
        anyhow::bail!(
            "listenbury say --klatt cannot convert Riper phoneme(s) for Klatt: {}",
            unsupported_symbols.join(", ")
        );
    }

    Ok(PhoneString { phones })
}

#[cfg(feature = "piper-compat")]
fn klatt_ipa_segments_for_riper_symbol(symbol: &str) -> Option<&'static [&'static str]> {
    let stress = symbol.chars().next_back();
    let base = symbol
        .strip_suffix(['0', '1', '2'])
        .filter(|base| is_riper_vowel_symbol(base))
        .unwrap_or(symbol);

    Some(match (symbol, base) {
        (" " | "|", _) => &[],
        ("AH0", _) => &["ə"],
        ("AH1" | "AH2", _) => &["ʌ"],
        (_, "AA") => &["ɑ"],
        (_, "AE") => &["æ"],
        (_, "AH") => {
            if matches!(stress, Some('0')) {
                &["ə"]
            } else {
                &["ʌ"]
            }
        }
        (_, "AO") => &["ɔ"],
        (_, "AW") => &["ɑ", "ʊ"],
        (_, "AY") => &["ɑ", "ɪ"],
        (_, "B") => &["b"],
        (_, "CH") => &["t", "ʃ"],
        (_, "D") => &["d"],
        (_, "DH") => &["ð"],
        (_, "DX") => &["d"],
        (_, "EH") => &["ɛ"],
        (_, "ER") => &["ə", "ɹ"],
        (_, "EY") => &["e", "ɪ"],
        (_, "F") => &["f"],
        (_, "G") => &["ɡ"],
        (_, "HH") => &["h"],
        (_, "IH") => &["ɪ"],
        (_, "IY") => &["i"],
        (_, "JH") => &["d", "ʒ"],
        (_, "K") => &["k"],
        (_, "L") => &["l"],
        (_, "M") => &["m"],
        (_, "N") => &["n"],
        (_, "NG") => &["ŋ"],
        (_, "OW") => &["o", "ʊ"],
        (_, "OY") => &["ɔ", "ɪ"],
        (_, "P") => &["p"],
        (_, "R") => &["ɹ"],
        (_, "S") => &["s"],
        (_, "SH") => &["ʃ"],
        (_, "T") => &["t"],
        (_, "TH") => &["θ"],
        (_, "TS") => &["t", "s"],
        (_, "UH") => &["ʊ"],
        (_, "UW") => &["u"],
        (_, "V") => &["v"],
        (_, "W") => &["w"],
        (_, "Y") => &["j"],
        (_, "Z") => &["z"],
        (_, "ZH") => &["ʒ"],
        (_, "i") => &["i"],
        (_, "ɪ") => &["ɪ"],
        (_, "e") => &["e"],
        (_, "ɛ") => &["ɛ"],
        (_, "æ") => &["æ"],
        (_, "ə") => &["ə"],
        (_, "ʌ") => &["ʌ"],
        (_, "ɑ") => &["ɑ"],
        (_, "ɔ") => &["ɔ"],
        (_, "o") => &["o"],
        (_, "ʊ") => &["ʊ"],
        (_, "u") => &["u"],
        (_, "m") => &["m"],
        (_, "n") => &["n"],
        (_, "ŋ") => &["ŋ"],
        (_, "l") => &["l"],
        (_, "ɹ") => &["ɹ"],
        (_, "j") => &["j"],
        (_, "w") => &["w"],
        (_, "s") => &["s"],
        (_, "z") => &["z"],
        (_, "ʃ") => &["ʃ"],
        (_, "ʒ") => &["ʒ"],
        (_, "f") => &["f"],
        (_, "v") => &["v"],
        (_, "θ") => &["θ"],
        (_, "ð") => &["ð"],
        (_, "h") => &["h"],
        (_, "p") => &["p"],
        (_, "b") => &["b"],
        (_, "t") => &["t"],
        (_, "d") => &["d"],
        (_, "k") => &["k"],
        (_, "ɡ") => &["ɡ"],
        (_, "ɾ") => &["d"],
        _ => return None,
    })
}

#[cfg(feature = "piper-compat")]
fn is_riper_vowel_symbol(symbol: &str) -> bool {
    matches!(
        symbol,
        "AA" | "AE"
            | "AH"
            | "AO"
            | "AW"
            | "AY"
            | "EH"
            | "ER"
            | "EY"
            | "IH"
            | "IY"
            | "OW"
            | "OY"
            | "UH"
            | "UW"
    )
}

#[cfg(not(feature = "piper-compat"))]
fn klatt_phone_string_from_demo_lexicon(text: &str) -> Result<PhoneString> {
    let mut phones = Vec::new();
    let mut unknown_words = Vec::new();
    for token in text.split_whitespace() {
        let word = token
            .trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '\'')
            .to_ascii_lowercase();
        if word.is_empty() {
            continue;
        }

        let Some(word_phones) = klatt_word_phones(word.as_str()) else {
            unknown_words.push(word);
            continue;
        };
        phones.extend(word_phones.iter().copied().map(Phone::new_ipa));
    }

    anyhow::ensure!(
        !phones.is_empty(),
        "listenbury say --klatt could not find any speakable words in `{text}`"
    );
    if !unknown_words.is_empty() {
        unknown_words.sort_unstable();
        unknown_words.dedup();
        anyhow::bail!(
            "listenbury say --klatt does not yet know word(s): {}. Supported words: {}",
            unknown_words.join(", "),
            KLATT_SUPPORTED_WORDS.join(", ")
        );
    }

    Ok(PhoneString { phones })
}

#[cfg(not(feature = "piper-compat"))]
fn klatt_word_phones(word: &str) -> Option<&'static [&'static str]> {
    const HELLO: [&str; 5] = ["h", "ɛ", "l", "o", "ʊ"];
    const MY: [&str; 3] = ["m", "ɑ", "ɪ"];
    const BABY: [&str; 5] = ["b", "e", "ɪ", "b", "i"];
    const DARLING: [&str; 6] = ["d", "ɑ", "ɹ", "l", "ɪ", "ŋ"];
    const RAGTIME: [&str; 7] = ["ɹ", "æ", "ɡ", "t", "ɑ", "ɪ", "m"];
    const GAL: [&str; 3] = ["ɡ", "æ", "l"];
    match word {
        "hello" => Some(&HELLO),
        "my" => Some(&MY),
        "baby" => Some(&BABY),
        "darling" => Some(&DARLING),
        "ragtime" => Some(&RAGTIME),
        "gal" => Some(&GAL),
        _ => None,
    }
}

fn write_say_wav(output_path: &Path, frames: &[AudioFrame]) -> Result<()> {
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
        "Wrote {} frames / {} samples to {}",
        frames.len(),
        sample_count,
        output_path.display()
    );

    Ok(())
}

#[cfg(feature = "audio-cpal")]
fn play_say_audio(frames: &[AudioFrame]) -> Result<()> {
    play_say_audio_with_source(frames, "Piper TTS")
}

#[cfg(feature = "audio-cpal")]
fn play_say_audio_with_source(frames: &[AudioFrame], source: &str) -> Result<()> {
    play_audio_frames(frames, source)
}

#[cfg(not(feature = "audio-cpal"))]
fn play_say_audio(_frames: &[AudioFrame]) -> Result<()> {
    anyhow::bail!(
        "listenbury say needs the `audio-cpal` feature for speaker playback; pass --output-wav <path> to write a WAV instead"
    )
}

#[cfg(not(feature = "audio-cpal"))]
fn play_say_audio_with_source(frames: &[AudioFrame], _source: &str) -> Result<()> {
    play_say_audio(frames)
}

fn looks_like_piper_bin(word: &str) -> bool {
    let path = Path::new(word);
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.contains("piper"))
}

pub(crate) fn resolve_piper_bin(explicit: Option<PathBuf>) -> Result<PathBuf> {
    explicit
        .or_else(|| std::env::var_os("LISTENBURY_PIPER_BIN").map(PathBuf::from))
        .or_else(|| find_piper_executable("piper"))
        .or_else(|| find_piper_executable("piper-tts.piper-cli"))
        .with_context(|| {
            "failed to find Piper executable; install `piper` or set LISTENBURY_PIPER_BIN / --piper-bin"
        })
}

fn find_piper_executable(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path)
            .map(|dir| dir.join(name))
            .find(|candidate| candidate.is_file())
    })
}

#[cfg(feature = "piper-compat")]
fn frame_duration_ms(frame: &AudioFrame) -> u64 {
    if frame.sample_rate_hz == 0 || frame.channels == 0 {
        return 0;
    }
    let channel_count = u64::from(frame.channels);
    let sample_count = frame.samples.len() as u64;
    // (samples / channels / sample_rate) * 1000, reordered to preserve integer precision.
    sample_count.saturating_mul(1_000)
        / channel_count.saturating_mul(u64::from(frame.sample_rate_hz))
}

pub(crate) fn piper_config_for_voice(
    piper_bin: impl Into<PathBuf>,
    model_path: impl Into<PathBuf>,
) -> Result<PiperConfig> {
    let piper_bin = piper_bin.into();
    let model_path = prepare_piper_model_path(&piper_bin, model_path.into())?;
    piper_config_for_model_path(piper_bin, model_path)
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
pub(crate) fn hifigan_text_to_speech(
    hifigan_model: Option<PathBuf>,
    skip_gan: bool,
) -> Result<PiperTextToSpeech> {
    #[cfg(feature = "piper-compat")]
    {
        Ok(PiperTextToSpeech::with_backend(HifiganTextBackend {
            hifigan_model,
            skip_gan,
        }))
    }

    #[cfg(not(feature = "piper-compat"))]
    {
        let _ = (hifigan_model, skip_gan);
        anyhow::bail!("listenbury live --hifigan requires the `piper-compat` feature")
    }
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper",
    feature = "piper-compat"
))]
struct HifiganTextBackend {
    hifigan_model: Option<PathBuf>,
    skip_gan: bool,
}

#[cfg(all(
    feature = "audio-cpal",
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper",
    feature = "piper-compat"
))]
impl TtsBackend for HifiganTextBackend {
    fn synthesize(&mut self, text: &str) -> Result<Vec<AudioFrame>> {
        synthesize_hifigan_text(
            text,
            self.hifigan_model.clone(),
            self.skip_gan,
            "listenbury live --hifigan",
        )
    }
}

#[cfg(feature = "piper-compat")]
fn piper_config_for_riper_voice(model_path: impl Into<PathBuf>) -> Result<PiperConfig> {
    piper_config_for_model_path("piper", model_path.into())
}

fn piper_config_for_model_path(
    piper_bin: impl Into<PathBuf>,
    model_path: impl Into<PathBuf>,
) -> Result<PiperConfig> {
    let piper_bin = piper_bin.into();
    let model_path = model_path.into();
    let inferred_config_path = model_path.with_extension("onnx.json");
    let mut config = PiperConfig::new(piper_bin, model_path);
    if inferred_config_path.exists() {
        if let Some(sample_rate_hz) = read_piper_sample_rate_hz(&inferred_config_path)? {
            config.sample_rate_hz = sample_rate_hz;
        }
        config.config_path = Some(inferred_config_path);
    }
    Ok(config)
}

pub(crate) fn collect_tts_audio(
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
        anyhow::bail!("Piper produced no audio frames before timeout");
    }

    Ok(frames)
}

fn read_piper_sample_rate_hz(path: &std::path::Path) -> Result<Option<u32>> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read Piper config at {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse Piper config at {}", path.display()))?;

    Ok(value
        .get("audio")
        .and_then(|audio| audio.get("sample_rate"))
        .and_then(|sample_rate| sample_rate.as_u64())
        .and_then(|sample_rate| u32::try_from(sample_rate).ok()))
}

fn prepare_piper_model_path(piper_bin: &Path, model_path: PathBuf) -> Result<PathBuf> {
    if !piper_model_needs_snap_copy(piper_bin, &model_path) {
        return Ok(model_path);
    }

    let destination_dir = Path::new("out/piper-models");
    std::fs::create_dir_all(destination_dir)
        .context("failed to create Snap-readable Piper model directory")?;

    let model_filename = model_path
        .file_name()
        .context("Piper model path has no filename")?;
    let copied_model_path = destination_dir.join(model_filename);
    copy_if_needed(&model_path, &copied_model_path)?;

    let config_path = model_path.with_extension("onnx.json");
    if config_path.exists() {
        let config_filename = config_path
            .file_name()
            .context("Piper config path has no filename")?;
        copy_if_needed(&config_path, &destination_dir.join(config_filename))?;
    }

    Ok(copied_model_path)
}

fn piper_model_needs_snap_copy(piper_bin: &Path, model_path: &Path) -> bool {
    if !uses_snap_piper(piper_bin) {
        return false;
    }

    has_hidden_component(model_path)
        || model_path
            .canonicalize()
            .is_ok_and(|path| has_hidden_component(&path))
}

fn uses_snap_piper(piper_bin: &Path) -> bool {
    piper_bin
        .to_str()
        .is_some_and(|path| path.starts_with("/snap/bin/") || path.contains("piper-tts.piper-cli"))
}

fn has_hidden_component(path: &Path) -> bool {
    path.components().any(|component| {
        component
            .as_os_str()
            .to_str()
            .is_some_and(|part| part.starts_with('.') && part != "." && part != "..")
    })
}

fn copy_if_needed(source: &Path, destination: &Path) -> Result<()> {
    let should_copy = match (source.metadata(), destination.metadata()) {
        (Ok(source_meta), Ok(destination_meta)) => source_meta.len() != destination_meta.len(),
        (Ok(_), Err(_)) => true,
        (Err(error), _) => {
            return Err(error).with_context(|| format!("failed to inspect {}", source.display()));
        }
    };

    if should_copy {
        std::fs::copy(source, destination).with_context(|| {
            format!(
                "failed to copy Piper asset from {} to {}",
                source.display(),
                destination.display()
            )
        })?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use listenbury::time::ExactTimestamp;

    #[test]
    fn say_args_treats_single_word_as_text() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec!["hello".to_string()],
        })
        .expect("single word should be text");

        assert!(args.piper_bin.is_none());
        assert!(args.piper_voice.is_none());
        assert_eq!(args.text, "hello");
    }

    #[test]
    fn say_args_accepts_dump_pipeline_flag() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: true,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec!["hello".to_string()],
        })
        .expect("--dump-pipeline should parse without changing the default route");

        assert!(args.dump_pipeline);
        let dump = format_say_pipeline(&args);
        assert!(dump.contains("speech pipeline: piper-compat"));
        assert!(dump.contains("-> acoustic generator: piper-compatible ONNX/Riper"));
        assert!(dump.contains("-> vocoder: piper-compatible internal"));
    }

    #[test]
    fn say_args_accepts_dump_phonemes_flag() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: true,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec!["hello".to_string()],
        })
        .expect("--dump-phonemes should parse without changing the default route");

        assert!(args.dump_phonemes);
        let dump = format_say_phonemes(&args).expect("phoneme dump should format");
        assert!(dump.contains("phoneme dump: piper-compat"));
        assert!(dump.contains("riper phonemes:"));
        assert!(dump.contains("acoustic phones:"));
        assert!(dump.contains("word phones:"));
    }

    #[test]
    fn say_args_accepts_dump_phone_plan_flag() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: true,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: true,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec!["Don't be jealous.".to_string()],
        })
        .expect("--dump-phone-plan should parse without synthesis setup");

        assert!(args.dump_phone_plan);
        let plan = PhonePlan::from_text_with_riper_g2p(&args.text).expect("plan should build");
        assert_eq!(plan.words[0].phones, ["d", "ow", "n", "t"]);
    }

    #[test]
    fn say_args_accepts_trailing_dump_phonemes_flag() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: true,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec!["hello".to_string(), "--dump-phonemes".to_string()],
        })
        .expect("trailing phoneme dump flag should be accepted");

        assert!(args.dump_phonemes);
        assert_eq!(args.text, "hello");
        let dump = format_say_phonemes(&args).expect("phoneme dump should format");
        assert!(dump.contains("phoneme dump: speecht5-hifigan"));
        assert!(dump.contains("SpeechT5"));
    }

    #[test]
    fn say_args_accepts_trailing_trace_speech_pipeline_flag() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: true,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec!["hello".to_string(), "--trace-speech-pipeline".to_string()],
        })
        .expect("trailing pipeline trace flag should be accepted");

        assert!(args.dump_pipeline);
        assert_eq!(args.text, "hello");
        let dump = format_say_pipeline(&args);
        assert!(dump.contains("speech pipeline: klatt"));
        assert!(dump.contains("-> acoustic generator: klatt"));
        assert!(dump.contains("-> mel/features: disabled"));
        assert!(dump.contains("-> vocoder: disabled"));
    }

    #[test]
    fn say_args_accepts_legacy_piper_bin_position() {
        let args = SayArgs::from_command(SayCommand {
            piper: true,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec![
                "/snap/bin/piper-tts.piper-cli".to_string(),
                "hello".to_string(),
            ],
        })
        .expect("legacy Piper executable should be accepted when --piper is selected");

        assert_eq!(
            args.piper_bin,
            Some(PathBuf::from("/snap/bin/piper-tts.piper-cli"))
        );
        assert!(args.piper_voice.is_none());
        assert_eq!(args.text, "hello");
    }

    #[test]
    fn say_args_accepts_legacy_voice_position() {
        let args = SayArgs::from_command(SayCommand {
            piper: true,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec![
                "/snap/bin/piper-tts.piper-cli".to_string(),
                "voice.onnx".to_string(),
                "hello".to_string(),
            ],
        })
        .expect("legacy Piper executable and voice should be accepted");

        assert_eq!(
            args.piper_bin,
            Some(PathBuf::from("/snap/bin/piper-tts.piper-cli"))
        );
        assert_eq!(args.piper_voice, Some(PathBuf::from("voice.onnx")));
        assert_eq!(args.text, "hello");
    }

    #[test]
    fn say_args_accepts_trailing_riper_flag() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec![
                "hello".to_string(),
                "there".to_string(),
                "--riper".to_string(),
            ],
        })
        .expect("--riper should be accepted as an explicit default route");
        assert_eq!(args.text, "hello there");
        assert!(!args.piper);
    }

    #[test]
    fn say_args_accepts_trailing_klatt_flag() {
        let error = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec!["hello".to_string(), "my".to_string(), "--klatt".to_string()],
        })
        .expect("Klatt is a default Riper-path backend");
        assert!(error.klatt);
        assert_eq!(error.text, "hello my");
    }

    #[test]
    fn say_args_accepts_klatt() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: true,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec!["hello".to_string()],
        })
        .expect("klatt should parse as a Riper backend alternative");
        assert!(args.klatt);
        assert!(should_use_klatt_backend(&args));
        assert_eq!(say_backend_graph(&args).id, "klatt");
        assert_eq!(say_speech_loom(&args).projection, "current-backend/klatt");
    }

    #[test]
    fn say_args_accepts_hifigan() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: true,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec!["hello".to_string()],
        })
        .expect("hifigan should parse, selecting the SpeechT5 acoustic route");
        assert!(!args.klatt);
        assert!(args.hifigan);
        assert!(should_use_speecht5_backend(&args));
        assert!(!should_use_source_filter_hifigan_backend(&args));
        assert_eq!(say_backend_graph(&args).id, "speecht5-hifigan");
        assert_eq!(
            say_speech_loom(&args).projection,
            "current-backend/speecht5-hifigan"
        );
        let dump = format_say_pipeline(&args);
        assert!(dump.contains("speech pipeline: speecht5-hifigan"));
        assert!(dump.contains("-> tokenizer: SpeechT5 tokenizer"));
        assert!(dump.contains("-> acoustic generator: SpeechT5 encoder/decoder ONNX"));
        assert!(dump.contains("-> mel/features: SpeechT5 mel spectrogram"));
        assert!(dump.contains("-> vocoder: SpeechT5 HiFi-GAN ONNX"));
        assert!(!dump.contains("source-filter spectral proxy"));
    }

    #[test]
    fn say_args_accepts_speecht5_as_native_acoustic_route() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: true,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec!["hello".to_string()],
        })
        .expect("speecht5 should parse as a native acoustic route");
        assert!(!args.hifigan);
        assert!(args.speecht5);
        assert!(should_use_speecht5_backend(&args));
        assert_eq!(say_backend_graph(&args).id, "speecht5-hifigan");
        assert_eq!(
            say_speech_loom(&args).projection,
            "current-backend/speecht5-hifigan"
        );
        let dump = format_say_pipeline(&args);
        assert!(dump.contains("speech pipeline: speecht5-hifigan"));
        assert!(dump.contains("-> tokenizer: SpeechT5 tokenizer"));
        assert!(dump.contains("-> acoustic generator: SpeechT5 encoder/decoder ONNX"));
        assert!(dump.contains("-> mel/features: SpeechT5 mel spectrogram"));
        assert!(dump.contains("-> vocoder: SpeechT5 HiFi-GAN ONNX"));
        assert!(!dump.contains("source-filter spectral proxy"));
    }

    #[test]
    fn say_args_accepts_trailing_speecht5_flag() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec!["hello".to_string(), "--speecht5".to_string()],
        })
        .expect("trailing SpeechT5 flag should be accepted");
        assert!(args.speecht5);
        assert_eq!(args.text, "hello");
    }

    #[test]
    fn say_args_rejects_hifigan_model_without_hifigan_or_speecht5() {
        let error = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: false,
            hifigan_model: Some(PathBuf::from("speecht5_hifigan.onnx")),
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec!["hello".to_string()],
        })
        .expect_err("--hifigan-model should require a HiFi-GAN route");
        assert!(
            error
                .to_string()
                .contains("--hifigan-model only applies when --hifigan or --speecht5"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn say_args_accepts_trailing_hifigan_flag() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec!["hello".to_string(), "--hifigan".to_string()],
        })
        .expect("trailing HiFi-GAN flag should be accepted");
        assert!(args.hifigan);
        assert_eq!(args.text, "hello");
    }

    #[test]
    fn say_args_accepts_skip_gan_as_hifigan_modifier() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: true,
            speecht5: false,
            hifigan_model: None,
            skip_gan: true,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec!["hello".to_string()],
        })
        .expect("--hifigan --skip-gan should select the source-filter mel debug route");
        assert!(args.hifigan);
        assert!(args.skip_gan);
        assert!(should_use_source_filter_hifigan_backend(&args));
        assert!(!should_use_speecht5_backend(&args));
        assert_eq!(say_backend_graph(&args).id, "source-filter-hifigan");
    }

    #[test]
    fn say_args_accepts_trailing_hifigan_fallback_alias() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec![
                "hello".to_string(),
                "--hifigan".to_string(),
                "--hifigan-fallback".to_string(),
            ],
        })
        .expect("trailing --hifigan-fallback should select the mel debug route");
        assert!(args.hifigan);
        assert!(args.skip_gan);
        assert_eq!(args.text, "hello");
    }

    #[test]
    fn say_args_rejects_skip_gan_without_hifigan() {
        let error = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: true,
            mbrola_voice: None,
            words: vec!["And sudd....".to_string(), "--skip-gan".to_string()],
        })
        .expect_err("--skip-gan should not select the mel debug route by itself");
        assert!(error.to_string().contains("--skip-gan only applies"));
    }

    #[test]
    fn say_args_accepts_diphone_voice() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: true,
            mbrola_voice: Some(PathBuf::from("voices/us1")),
            words: vec!["hello".to_string()],
        })
        .expect("diphone should select the diphone voice backend");
        assert!(!args.klatt);
        assert!(should_use_mbrola_backend(&args));
        assert_eq!(args.mbrola_voice, Some(PathBuf::from("voices/us1")));
        assert_eq!(say_backend_graph(&args).id, "mbrola-diphone");
    }

    #[test]
    fn say_args_accepts_diphone() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: true,
            mbrola_voice: None,
            words: vec!["hello".to_string()],
        })
        .expect("diphone should select the diphone voice backend");
        assert!(args.mbrola);
        assert_eq!(say_backend_graph(&args).id, "mbrola-diphone");
    }

    #[test]
    fn say_backend_graph_defaults_to_piper_compat() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec!["hello".to_string()],
        })
        .expect("default say route should parse");
        let backend_graph = say_backend_graph(&args);
        let loom = say_speech_loom(&args);
        assert_eq!(backend_graph.id, "piper-compat");
        assert!(backend_graph.fused);
        assert_eq!(backend_graph.workers.len(), 1);
        assert_eq!(backend_graph.workers[0].id, "piper-compatible-onnx");
        assert_eq!(loom.projection, "current-backend/piper-compat");
    }

    #[test]
    fn say_backend_graph_reports_external_piper_process() {
        let args = SayArgs::from_command(SayCommand {
            piper: true,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec!["hello".to_string()],
        })
        .expect("external piper route should parse");
        let backend_graph = say_backend_graph(&args);
        assert_eq!(backend_graph.id, "piper-process");
        assert!(backend_graph.fused);
        assert_eq!(backend_graph.workers.len(), 1);
        assert_eq!(backend_graph.workers[0].id, "piper-process-backend");
    }

    #[test]
    fn say_backend_graph_reports_klatt_worker_contract() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: true,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec!["hello".to_string()],
        })
        .expect("klatt route should parse");
        let backend_graph = say_backend_graph(&args);
        assert_eq!(backend_graph.id, "klatt");
        assert!(!backend_graph.fused);
        assert_eq!(backend_graph.workers.len(), 1);
        assert_eq!(backend_graph.workers[0].id, "klatt-formant-renderer");
    }

    #[test]
    fn say_backend_graph_reports_mbrola_internal_workers() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: true,
            mbrola_voice: None,
            words: vec!["hello".to_string()],
        })
        .expect("diphone route should parse");
        let backend_graph = say_backend_graph(&args);
        assert_eq!(backend_graph.id, "mbrola-diphone");
        assert!(!backend_graph.fused);
        assert_eq!(backend_graph.workers.len(), 2);
        assert_eq!(backend_graph.workers[0].id, "mbrola-diphone-selection");
        assert_eq!(backend_graph.workers[1].id, "mbrola-diphone-renderer");
        let dump = format_say_pipeline(&args);
        assert!(dump.contains("-> acoustic generator: MBROLA-compatible diphone renderer"));
        assert!(dump.contains("-> diphone voice: default MBROLA-compatible voice"));
    }

    #[test]
    fn say_backend_graph_reports_hifigan_speecht5_workers() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: true,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec!["hello".to_string()],
        })
        .expect("hifigan route should parse");
        let backend_graph = say_backend_graph(&args);
        let loom = say_speech_loom(&args);
        assert_eq!(backend_graph.id, "speecht5-hifigan");
        assert!(!backend_graph.fused);
        assert_eq!(backend_graph.workers.len(), 3);
        assert_eq!(backend_graph.workers[0].id, "speecht5-tokenizer");
        assert_eq!(
            backend_graph.workers[1].id,
            "speecht5-encoder-decoder-acoustic-generator"
        );
        assert_eq!(backend_graph.workers[2].id, "speecht5-hifigan-vocoder");
        assert_eq!(loom.projection, "current-backend/speecht5-hifigan");
    }

    #[test]
    fn say_backend_graph_reports_hifigan_fallback_feature_bridge_workers() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: true,
            speecht5: false,
            hifigan_model: None,
            skip_gan: true,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec!["hello".to_string()],
        })
        .expect("hifigan fallback route should parse");
        let backend_graph = say_backend_graph(&args);
        let loom = say_speech_loom(&args);
        assert_eq!(backend_graph.id, "source-filter-hifigan");
        assert!(!backend_graph.fused);
        assert_eq!(backend_graph.workers.len(), 4);
        assert_eq!(
            backend_graph.workers[0].id,
            "source-filter-acoustic-generator"
        );
        assert_eq!(
            backend_graph.workers[1].id,
            "source-filter-temporal-smoother"
        );
        assert_eq!(
            backend_graph.workers[2].id,
            "source-filter-mel-compat-bridge"
        );
        assert_eq!(backend_graph.workers[3].id, "hifigan-vocoder");
        assert_eq!(loom.projection, "current-backend/source-filter-hifigan");
    }

    #[test]
    fn say_args_rp_selects_en1_mbrola_voice() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: true,
            diphone: false,
            mbrola_voice: None,
            words: vec!["hello".to_string()],
        })
        .expect("RP shorthand should select the en1 MBROLA voice");
        assert!(args.mbrola);
        assert_eq!(
            args.mbrola_voice,
            Some(received_pronunciation_mbrola_voice())
        );
    }

    #[test]
    fn say_args_accepts_trailing_rp_flag() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec!["hello".to_string(), "--rp".to_string()],
        })
        .expect("trailing RP shorthand should be accepted");
        assert!(args.mbrola);
        assert_eq!(args.text, "hello");
        assert_eq!(
            args.mbrola_voice,
            Some(received_pronunciation_mbrola_voice())
        );
    }

    #[test]
    fn say_args_rejects_rp_with_klatt() {
        let error = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: true,
            diphone: false,
            mbrola_voice: None,
            words: vec!["hello".to_string(), "--klatt".to_string()],
        })
        .expect_err("RP shorthand should conflict with Klatt");
        assert!(
            error.to_string().contains("MBROLA/RP voice path"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn say_args_uses_default_diphone_demo_text() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: false,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: true,
            mbrola_voice: None,
            words: Vec::new(),
        })
        .expect("diphone should have a default smoke utterance");
        assert_eq!(args.text, "Hello, my baby.");
    }

    #[test]
    fn say_args_treats_dash_as_stdin_stream() {
        let args = SayArgs::from_command(SayCommand {
            piper: false,
            riper: false,
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            dump_pipeline: false,
            dump_phonemes: false,
            dump_phone_plan: false,
            dump_piper_tensors: false,
            klatt: true,
            hifigan: false,
            speecht5: false,
            hifigan_model: None,
            skip_gan: false,
            rp: false,
            diphone: false,
            mbrola_voice: None,
            words: vec!["-".to_string()],
        })
        .expect("dash should select stdin streaming");

        assert!(args.stdin_stream);
        assert!(args.klatt);
        assert!(args.text.is_empty());
    }

    #[test]
    fn klatt_phrase_renders_non_empty_audio_and_wav_bytes() {
        let frames =
            synthesize_klatt_for_say("Hello, my baby. Hello, my darling. Hello, my ragtime gal.")
                .expect("klatt phrase should synthesize");
        assert_eq!(frames.len(), 1);
        assert!(!frames[0].samples.is_empty());
        let wav = write_wav_bytes(&frames).expect("frames should serialize as WAV");
        assert!(wav.len() > 44, "WAV payload should include audio data");
    }

    #[test]
    fn klatt_phrase_unknown_word_reports_clear_error() {
        let error = synthesize_klatt_for_say("Hello 💥")
            .expect_err("unsupported text should produce a clear error");
        assert!(error.to_string().contains("could not phonemize"));
    }

    #[test]
    #[cfg(feature = "piper-compat")]
    fn klatt_uses_riper_pronunciation_for_mixed_prose() {
        let frames = synthesize_klatt_for_say(
            "MBROLA was created by Thierry Dutoit. It's a speech synthesizer based on the concatenation of diphones.",
        )
        .expect("Klatt should synthesize prose via Riper pronunciation machinery");
        assert_eq!(frames.len(), 1);
        assert!(!frames[0].samples.is_empty());
    }

    #[test]
    #[cfg(feature = "piper-compat")]
    fn klatt_riper_phone_bridge_splits_diphthongs_and_affricates() {
        let phone_string = klatt_phone_string_for_text("Okay, Charlie.")
            .expect("Riper phones should convert to Klatt render phones");
        let ipas = phone_string.ipa_segments();
        assert!(ipas.windows(2).any(|phones| phones == ["o", "ʊ"]));
        assert!(ipas.windows(2).any(|phones| phones == ["t", "ʃ"]));
    }

    #[test]
    #[cfg(feature = "piper-compat")]
    fn diphone_plan_uses_planned_durations_pitches_and_pause() {
        let plan = planned_phone_timed_plan_for_text(
            SimpleEnglishG2p::default(),
            "The red machine.",
            |symbol| Ok(symbol.to_string()),
            "test",
        )
        .expect("planned diphone plan");

        assert!(
            plan.phones.iter().any(|phone| phone.symbol != "_"
                && phone.duration_ms != 75
                && phone.duration_ms != 145),
            "planned durations should replace the old canned consonant/vowel defaults: {:?}",
            plan.phones
        );
        assert!(
            plan.phones.iter().any(|phone| phone
                .pitch_targets
                .iter()
                .any(|target| (target.hz - 135.0).abs() > 0.01)),
            "planned pitch shapes should replace the old canned vowel pitch triplet: {:?}",
            plan.phones
        );
        assert_eq!(
            plan.phones.last(),
            Some(&MbrolaPhone::new("_", 260)),
            "committed full-turn say should use the planner final pause"
        );
    }

    #[test]
    #[cfg(feature = "piper-compat")]
    fn frame_duration_ms_handles_zero_values() {
        let frame = AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: 0,
            channels: 1,
            samples: vec![0.0; 1600],
            voice_signatures: Vec::new(),
        };
        assert_eq!(frame_duration_ms(&frame), 0);

        let frame = AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: 16_000,
            channels: 0,
            samples: vec![0.0; 1600],
            voice_signatures: Vec::new(),
        };
        assert_eq!(frame_duration_ms(&frame), 0);
    }

    #[test]
    #[cfg(feature = "piper-compat")]
    fn frame_duration_ms_preserves_fractional_millisecond_precision() {
        let frame = AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: 16_000,
            channels: 2,
            samples: vec![0.0; 3_200],
            voice_signatures: Vec::new(),
        };

        assert_eq!(frame_duration_ms(&frame), 100);
    }

    #[test]
    #[cfg(feature = "piper-compat")]
    fn riper_compare_args_joins_words_into_text() {
        let args = RiperCompareArgs::from_command(RiperCompareCommand {
            piper_bin: None,
            piper_voice: None,
            riper_voice: None,
            riper_config: None,
            process_output_wav: None,
            riper_output_wav: None,
            phonemes: None,
            words: vec!["Okay.".to_string(), "Again.".to_string()],
        })
        .expect("words should parse");

        assert_eq!(args.text, "Okay. Again.");
    }

    #[test]
    #[cfg(unix)]
    fn snap_piper_copy_check_follows_symlink_to_hidden_directory() {
        let root = std::env::temp_dir().join(format!(
            "listenbury-piper-symlink-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let hidden_dir = root.join(".models");
        let visible_dir = root.join("voices");
        std::fs::create_dir_all(&hidden_dir).expect("create hidden model directory");
        std::fs::create_dir_all(&visible_dir).expect("create visible voice directory");

        let hidden_model = hidden_dir.join("ryan.onnx");
        std::fs::write(&hidden_model, b"model").expect("write hidden model");
        let visible_model = visible_dir.join("ryan.onnx");
        std::os::unix::fs::symlink(&hidden_model, &visible_model).expect("create model symlink");

        assert!(piper_model_needs_snap_copy(
            Path::new("/snap/bin/piper-tts.piper-cli"),
            &visible_model,
        ));
        assert!(!piper_model_needs_snap_copy(
            Path::new("/usr/bin/piper"),
            &visible_model,
        ));

        std::fs::remove_dir_all(root).expect("remove test directory");
    }

    #[test]
    #[cfg(feature = "piper-compat")]
    fn espeak_compatible_ids_match_piper_debug_shape_for_okay() {
        let config = PiperVoiceConfig::from_json_str(
            r#"
            {
              "audio": { "sample_rate": 22050 },
              "phoneme_id_map": {
                "_": [0],
                "^": [1],
                "$": [2],
                ".": [10],
                "e": [18],
                "k": [23],
                "o": [27],
                "ɪ": [74],
                "ʊ": [100],
                "ˈ": [120]
              }
            }
            "#,
        )
        .expect("voice config should parse");
        let sequence = PiperPhonemeSequence {
            phonemes: ["OW", "K", "EY", "|"]
                .into_iter()
                .map(|symbol| PiperPhoneme(symbol.to_string()))
                .collect(),
        };

        let ids = sequence
            .to_piper_ids_compatible(&config)
            .expect("ARPAbet symbols should map to eSpeak Piper IDs");

        assert_eq!(
            ids,
            PiperIdSequence {
                ids: vec![1, 0, 27, 0, 100, 0, 23, 0, 18, 0, 74, 0, 10, 0, 2]
            }
        );
    }

    #[test]
    #[cfg(feature = "piper-compat")]
    fn espeak_compatible_ids_support_lollipop_guild_sentence_symbols() {
        let config = PiperVoiceConfig::from_json_str(
            r#"
            {
              "audio": { "sample_rate": 22050 },
              "phoneme_id_map": {
                "_": [0],
                "^": [1],
                "$": [2],
                " ": [3],
                ".": [10],
                "a": [11],
                "d": [12],
                "i": [13],
                "l": [14],
                "n": [15],
                "p": [16],
                "t": [17],
                "w": [18],
                "z": [19],
                "ð": [20],
                "ɡ": [21],
                "ɪ": [22],
                "ɛ": [23],
                "ɑ": [24],
                "ə": [25],
                "ɹ": [26]
              }
            }
            "#,
        )
        .expect("voice config should parse");
        let sequence = PiperPhonemeSequence {
            phonemes: [
                "W", "IY", " ", "R", "EH", "P", "R", "IH", "Z", "EH", "N", "T", " ", "DH", "AH0",
                " ", "L", "AA", "L", "IY", "P", "AA", "P", " ", "G", "IH", "L", "D", "|",
            ]
            .into_iter()
            .map(|symbol| PiperPhoneme(symbol.to_string()))
            .collect(),
        };

        sequence
            .to_piper_ids_compatible(&config)
            .expect("sentence ARPAbet symbols should map to eSpeak Piper IDs");
    }

    #[test]
    #[cfg(feature = "piper-compat")]
    fn espeak_compatible_ids_map_arpabet_flap_symbol() {
        let config = PiperVoiceConfig::from_json_str(
            r#"
            {
              "audio": { "sample_rate": 22050 },
              "phoneme_id_map": {
                "_": [0],
                "^": [1],
                "$": [2],
                "b": [10],
                "l": [11],
                "ɑ": [12],
                "ə": [13],
                "ɾ": [14]
              }
            }
            "#,
        )
        .expect("voice config should parse");
        let sequence = PiperPhonemeSequence {
            phonemes: ["B", "AA", "DX", "AH0", "L"]
                .into_iter()
                .map(|symbol| PiperPhoneme(symbol.to_string()))
                .collect(),
        };

        let ids = sequence
            .to_piper_ids_compatible(&config)
            .expect("flapped Riper sequence should map to eSpeak Piper IDs");

        assert_eq!(
            ids,
            PiperIdSequence {
                ids: vec![1, 0, 10, 0, 12, 0, 14, 0, 13, 0, 11, 0, 2]
            }
        );
    }

    #[test]
    #[cfg(feature = "piper-compat")]
    fn audio_stats_computes_duration_rms_and_peak() {
        let stats = AudioStats::from_frames(
            &[AudioFrame {
                captured_at: ExactTimestamp::now(),
                sample_rate_hz: 20,
                channels: 1,
                samples: vec![0.0, 0.5, -1.0, 0.5],
                voice_signatures: Vec::new(),
            }],
            "test",
        )
        .expect("stats should compute");

        assert_eq!(stats.sample_rate_hz, 20);
        assert_eq!(stats.channels, 1);
        assert_eq!(stats.sample_count, 4);
        assert!((stats.duration_ms - 200.0).abs() < 0.0001);
        assert!((stats.rms - 0.6123724).abs() < 0.0001);
        assert!((stats.peak_abs - 1.0).abs() < 0.0001);
    }
}
