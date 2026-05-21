#[cfg(feature = "audio-cpal")]
use crate::cli::commands::play_audio_frames;
use crate::cli::model_paths::resolve_piper_voice;
use crate::cli::{RiperCompareCommand, SayCommand};
use anyhow::{Context, Result};
use listenbury::audio::frame::AudioFrame;
use listenbury::audio::write_wav;
#[cfg(feature = "tts-riper")]
use listenbury::mouth::backend::TtsBackend;
#[cfg(feature = "tts-riper")]
use listenbury::mouth::piper::{PiperBackendPreference, ProcessPiperBackend};
use listenbury::mouth::planner::{SpeechPlan, SpeechUnit};
#[cfg(feature = "tts-riper")]
use listenbury::mouth::riper::phoneme::espeak_compatible_sequence;
#[cfg(feature = "tts-riper")]
use listenbury::mouth::riper::{
    PiperIdSequence, PiperPhoneme, PiperPhonemeSequence, PiperVoiceConfig, RiperBackend,
    SentenceAnalysis, SimpleEnglishG2p, SyntacticLinkKind, SyntacticLinkParse,
};
use listenbury::mouth::tts::TextToSpeech;
use listenbury::{PiperConfig, PiperTextToSpeech};
#[cfg(feature = "tts-riper")]
use std::io::Write;
use std::path::{Path, PathBuf};
#[cfg(feature = "tts-riper")]
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub(crate) fn run_say(command: SayCommand) -> Result<()> {
    let piper_args = SayArgs::from_command(command)?;
    #[cfg(not(feature = "tts-riper"))]
    if piper_args.riper {
        anyhow::bail!("listenbury say --riper requires the `tts-riper` feature");
    }

    let piper_voice = resolve_piper_voice(piper_args.piper_voice.clone())?;
    let mut tts = say_tts_for_args(&piper_args, piper_voice)?;
    tts.enqueue(SpeechPlan::from(SpeechUnit::FullTurn(piper_args.text)))?;
    let frames = collect_tts_audio(&mut tts, Duration::from_secs(30))?;

    if let Some(output_path) = piper_args.output_wav {
        write_say_wav(&output_path, &frames)?;
    } else {
        play_say_audio(&frames)?;
    }

    Ok(())
}

fn say_tts_for_args(args: &SayArgs, piper_voice: PathBuf) -> Result<PiperTextToSpeech> {
    if args.riper {
        return say_riper_tts_for_voice(piper_voice);
    }

    let piper_bin = resolve_piper_bin(args.piper_bin.clone())?;
    Ok(PiperTextToSpeech::new(piper_config_for_voice(
        piper_bin,
        piper_voice,
    )?))
}

#[cfg(feature = "tts-riper")]
fn say_riper_tts_for_voice(piper_voice: PathBuf) -> Result<PiperTextToSpeech> {
    Ok(PiperTextToSpeech::new_with_backend_preference(
        piper_config_for_riper_voice(piper_voice)?,
        PiperBackendPreference::Riper,
    ))
}

#[cfg(not(feature = "tts-riper"))]
fn say_riper_tts_for_voice(_piper_voice: PathBuf) -> Result<PiperTextToSpeech> {
    anyhow::bail!("listenbury say --riper requires the `tts-riper` feature")
}

pub(crate) fn run_riper_compare(command: RiperCompareCommand) -> Result<()> {
    #[cfg(not(feature = "tts-riper"))]
    {
        let _ = command;
        anyhow::bail!(
            "listenbury riper-compare requires the `tts-riper` feature to compare Riper synthesis"
        );
    }

    #[cfg(feature = "tts-riper")]
    {
        run_riper_compare_impl(command)
    }
}

#[cfg(feature = "tts-riper")]
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
    let riper_stats = synthesize_riper_for_compare(
        &riper_model_path,
        &riper_voice_config,
        &riper_phonemes.ids,
        &args.text,
    )?;

    report_compare_phonemes(&process_phonemes, &riper_phonemes);
    report_compare_stats(&process_stats, &riper_stats);

    if let Some(output) = args.process_output_wav {
        write_say_wav(&output, &process_stats.frames)?;
    }
    if let Some(output) = args.riper_output_wav {
        write_say_wav(&output, &riper_stats.frames)?;
    }

    Ok(())
}

#[cfg(feature = "tts-riper")]
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

#[cfg(feature = "tts-riper")]
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

#[cfg(feature = "tts-riper")]
#[derive(Debug, Clone)]
struct SynthesisStats {
    frames: Vec<AudioFrame>,
    runtime: Duration,
    audio: AudioStats,
}

#[cfg(feature = "tts-riper")]
#[derive(Debug, Clone, PartialEq)]
struct AudioStats {
    sample_rate_hz: u32,
    channels: u16,
    sample_count: usize,
    duration_ms: f64,
    rms: f32,
    peak_abs: f32,
}

#[cfg(feature = "tts-riper")]
#[derive(Debug, Clone, PartialEq, Eq)]
struct ProcessNativePhonemes {
    voice: String,
    mnemonic: std::result::Result<String, String>,
    ipa: std::result::Result<String, String>,
}

#[cfg(feature = "tts-riper")]
#[derive(Debug, Clone, PartialEq)]
struct RiperPhonemeReport {
    source: &'static str,
    phonemes: PiperPhonemeSequence,
    compatible_phonemes: Option<PiperPhonemeSequence>,
    ids: PiperIdSequence,
    sentence_analysis: Option<SentenceAnalysis>,
}

#[cfg(feature = "tts-riper")]
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

#[cfg(feature = "tts-riper")]
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

#[cfg(feature = "tts-riper")]
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

#[cfg(feature = "tts-riper")]
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
        .to_piper_ids_compatible(config)
        .with_context(|| {
            format!(
                "Riper voice config cannot map one or more phonemes for `{}`; pass --phonemes to override",
                args.text
            )
        })?;
    let compatible_phonemes = espeak_compatible_sequence(&phoneme_sequence, config).ok();

    Ok(RiperPhonemeReport {
        source,
        phonemes: phoneme_sequence,
        compatible_phonemes,
        ids,
        sentence_analysis,
    })
}

#[cfg(feature = "tts-riper")]
fn read_riper_voice_config(path: &Path) -> Result<PiperVoiceConfig> {
    let json = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read Riper config at {}", path.display()))?;
    PiperVoiceConfig::from_json_str(&json)
        .with_context(|| format!("failed to parse Riper config JSON at {}", path.display()))
}

#[cfg(feature = "tts-riper")]
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

#[cfg(feature = "tts-riper")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EspeakPhonemeNotation {
    Mnemonic,
    Ipa,
}

#[cfg(feature = "tts-riper")]
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

#[cfg(feature = "tts-riper")]
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

#[cfg(feature = "tts-riper")]
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

#[cfg(feature = "tts-riper")]
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

#[cfg(feature = "tts-riper")]
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

#[cfg(feature = "tts-riper")]
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

#[cfg(feature = "tts-riper")]
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

#[cfg(feature = "tts-riper")]
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

#[cfg(feature = "tts-riper")]
fn render_phoneme_result(result: &std::result::Result<String, String>) -> String {
    match result {
        Ok(value) if value.is_empty() => "(empty)".to_string(),
        Ok(value) => value.clone(),
        Err(error) => format!("(unavailable: {error})"),
    }
}

#[cfg(feature = "tts-riper")]
fn format_phoneme_sequence(sequence: &PiperPhonemeSequence) -> String {
    sequence
        .phonemes
        .iter()
        .map(|phoneme| phoneme.0.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(feature = "tts-riper")]
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

#[derive(Debug)]
struct SayArgs {
    piper_bin: Option<PathBuf>,
    piper_voice: Option<PathBuf>,
    output_wav: Option<PathBuf>,
    riper: bool,
    text: String,
}

impl SayArgs {
    fn from_command(command: SayCommand) -> Result<Self> {
        let mut riper = command.riper;
        let mut words = command
            .words
            .into_iter()
            .filter_map(|word| {
                if word == "--riper" {
                    riper = true;
                    None
                } else {
                    Some(word)
                }
            })
            .collect::<Vec<_>>();
        let mut piper_bin = command.piper_bin;
        let mut piper_voice = command.piper_voice;

        if piper_bin.is_none() && words.first().is_some_and(|word| looks_like_piper_bin(word)) {
            piper_bin = Some(PathBuf::from(words.remove(0)));
        }

        if piper_voice.is_none() && words.first().is_some_and(|word| word.ends_with(".onnx")) {
            piper_voice = Some(PathBuf::from(words.remove(0)));
        }

        anyhow::ensure!(!words.is_empty(), "missing text to speak; try `say hello`");

        Ok(Self {
            piper_bin,
            piper_voice,
            output_wav: command.output_wav,
            riper,
            text: words.join(" "),
        })
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
    play_audio_frames(frames, "Piper TTS")
}

#[cfg(not(feature = "audio-cpal"))]
fn play_say_audio(_frames: &[AudioFrame]) -> Result<()> {
    anyhow::bail!(
        "listenbury say needs the `audio-cpal` feature for speaker playback; pass --output-wav <path> to write a WAV instead"
    )
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

pub(crate) fn piper_config_for_voice(
    piper_bin: impl Into<PathBuf>,
    model_path: impl Into<PathBuf>,
) -> Result<PiperConfig> {
    let piper_bin = piper_bin.into();
    let model_path = prepare_piper_model_path(&piper_bin, model_path.into())?;
    piper_config_for_model_path(piper_bin, model_path)
}

#[cfg(feature = "tts-riper")]
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
            if let Some(last_audio_at) = last_audio_at {
                if Instant::now().duration_since(last_audio_at) >= quiet_after_audio {
                    break;
                }
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
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            riper: false,
            words: vec!["hello".to_string()],
        })
        .expect("single word should be text");

        assert!(args.piper_bin.is_none());
        assert!(args.piper_voice.is_none());
        assert_eq!(args.text, "hello");
    }

    #[test]
    fn say_args_accepts_legacy_piper_bin_position() {
        let args = SayArgs::from_command(SayCommand {
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            riper: false,
            words: vec![
                "/snap/bin/piper-tts.piper-cli".to_string(),
                "hello".to_string(),
            ],
        })
        .expect("legacy Piper executable should be accepted");

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
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            riper: false,
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
            piper_bin: None,
            piper_voice: None,
            output_wav: None,
            riper: false,
            words: vec![
                "hello".to_string(),
                "there".to_string(),
                "--riper".to_string(),
            ],
        })
        .expect("trailing Riper flag should be accepted");

        assert!(args.riper);
        assert_eq!(args.text, "hello there");
    }

    #[test]
    #[cfg(feature = "tts-riper")]
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
    #[cfg(feature = "tts-riper")]
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
                ids: vec![1, 0, 27, 0, 100, 0, 23, 0, 120, 0, 18, 0, 74, 0, 10, 0, 2]
            }
        );
    }

    #[test]
    #[cfg(feature = "tts-riper")]
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
    #[cfg(feature = "tts-riper")]
    fn espeak_compatible_ids_preserve_riper_flap_symbol() {
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
            phonemes: ["B", "AA", "ɾ", "AH0", "L"]
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
    #[cfg(feature = "tts-riper")]
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
