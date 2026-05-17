#[cfg(feature = "audio-cpal")]
use crate::cli::commands::play_audio_frames;
use crate::cli::model_paths::resolve_piper_voice;
use crate::cli::{PiperCompareCommand, SayCommand};
use anyhow::{Context, Result};
use listenbury::audio::frame::AudioFrame;
use listenbury::audio::write_wav;
#[cfg(feature = "tts-piper-native")]
use listenbury::mouth::backend::TtsBackend;
#[cfg(feature = "tts-piper-native")]
use listenbury::mouth::piper::ProcessPiperBackend;
#[cfg(feature = "tts-piper-native")]
use listenbury::mouth::piper_native::{
    GraphemeToPhoneme, NativePiperBackend, PiperIdSequence, PiperPhoneme, PiperPhonemeSequence,
    PiperVoiceConfig, SimpleEnglishG2p,
};
use listenbury::mouth::planner::{SpeechPlan, SpeechUnit};
use listenbury::mouth::tts::TextToSpeech;
use listenbury::{PiperConfig, PiperTextToSpeech};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub(crate) fn run_say(command: SayCommand) -> Result<()> {
    let piper_args = SayArgs::from_command(command)?;
    let piper_bin = resolve_piper_bin(piper_args.piper_bin)?;
    let piper_voice = resolve_piper_voice(piper_args.piper_voice)?;
    let mut tts = PiperTextToSpeech::new(piper_config_for_voice(piper_bin, piper_voice)?);
    tts.enqueue(SpeechPlan::from(SpeechUnit::FullTurn(piper_args.text)))?;
    let frames = collect_tts_audio(&mut tts, Duration::from_secs(30))?;

    if let Some(output_path) = piper_args.output_wav {
        write_say_wav(&output_path, &frames)?;
    } else {
        play_say_audio(&frames)?;
    }

    Ok(())
}

pub(crate) fn run_piper_compare(command: PiperCompareCommand) -> Result<()> {
    #[cfg(not(feature = "tts-piper-native"))]
    {
        let _ = command;
        anyhow::bail!(
            "listenbury piper-compare requires the `tts-piper-native` feature to compare native synthesis"
        );
    }

    #[cfg(feature = "tts-piper-native")]
    {
        run_piper_compare_impl(command)
    }
}

#[cfg(feature = "tts-piper-native")]
fn run_piper_compare_impl(command: PiperCompareCommand) -> Result<()> {
    let args = PiperCompareArgs::from_command(command)?;
    let piper_bin = resolve_piper_bin(args.piper_bin.clone())?;
    let process_voice = resolve_piper_voice(args.piper_voice.clone())?;
    let process_config = piper_config_for_voice(piper_bin.clone(), process_voice)?;
    let process_stats = synthesize_process_for_compare(&process_config, &args.text)?;

    let native_model_path = args
        .native_voice
        .clone()
        .unwrap_or_else(|| process_config.model_path.clone());
    let native_config_path = args
        .native_config
        .clone()
        .or_else(|| process_config.config_path.clone())
        .unwrap_or_else(|| native_model_path.with_extension("onnx.json"));
    let native_voice_config = read_native_voice_config(&native_config_path)?;
    let native_ids = resolve_native_ids(&args, &native_voice_config)?;
    let native_stats = synthesize_native_for_compare(
        &native_model_path,
        &native_voice_config,
        &native_ids,
        &args.text,
    )?;

    report_compare_stats(&process_stats, &native_stats);

    if let Some(output) = args.process_output_wav {
        write_say_wav(&output, &process_stats.frames)?;
    }
    if let Some(output) = args.native_output_wav {
        write_say_wav(&output, &native_stats.frames)?;
    }

    Ok(())
}

#[cfg(feature = "tts-piper-native")]
#[derive(Debug)]
struct PiperCompareArgs {
    piper_bin: Option<PathBuf>,
    piper_voice: Option<PathBuf>,
    native_voice: Option<PathBuf>,
    native_config: Option<PathBuf>,
    process_output_wav: Option<PathBuf>,
    native_output_wav: Option<PathBuf>,
    phonemes: Option<String>,
    text: String,
}

#[cfg(feature = "tts-piper-native")]
impl PiperCompareArgs {
    fn from_command(command: PiperCompareCommand) -> Result<Self> {
        anyhow::ensure!(
            !command.words.is_empty(),
            "missing text to compare; try `listenbury piper-compare \"I see.\"`"
        );
        let text = command.words.join(" ");
        anyhow::ensure!(
            !text.trim().is_empty(),
            "missing text to compare; try `listenbury piper-compare \"I see.\"`"
        );

        Ok(Self {
            piper_bin: command.piper_bin,
            piper_voice: command.piper_voice,
            native_voice: command.native_voice,
            native_config: command.native_config,
            process_output_wav: command.process_output_wav,
            native_output_wav: command.native_output_wav,
            phonemes: command.phonemes,
            text,
        })
    }
}

#[cfg(feature = "tts-piper-native")]
#[derive(Debug, Clone)]
struct SynthesisStats {
    frames: Vec<AudioFrame>,
    runtime: Duration,
    audio: AudioStats,
}

#[cfg(feature = "tts-piper-native")]
#[derive(Debug, Clone, PartialEq)]
struct AudioStats {
    sample_rate_hz: u32,
    channels: u16,
    sample_count: usize,
    duration_ms: f64,
    rms: f32,
    peak_abs: f32,
}

#[cfg(feature = "tts-piper-native")]
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

#[cfg(feature = "tts-piper-native")]
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

#[cfg(feature = "tts-piper-native")]
fn synthesize_native_for_compare(
    model_path: &Path,
    config: &PiperVoiceConfig,
    ids: &PiperIdSequence,
    text: &str,
) -> Result<SynthesisStats> {
    let mut backend = NativePiperBackend::load(model_path, config.clone()).with_context(|| {
        format!(
            "failed to initialize native Piper backend from model {}",
            model_path.display()
        )
    })?;
    let t0 = Instant::now();
    let frames = backend.synthesize_id_frames(ids).with_context(|| {
        format!(
            "native Piper synthesis failed for model {} and text `{}`",
            model_path.display(),
            text
        )
    })?;
    let runtime = t0.elapsed();
    let audio = AudioStats::from_frames(&frames, "native")?;
    Ok(SynthesisStats {
        frames,
        runtime,
        audio,
    })
}

#[cfg(feature = "tts-piper-native")]
fn resolve_native_ids(
    args: &PiperCompareArgs,
    config: &PiperVoiceConfig,
) -> Result<PiperIdSequence> {
    let phoneme_sequence = if let Some(raw) = args.phonemes.as_ref() {
        let symbols: Vec<_> = raw.split_whitespace().collect();
        anyhow::ensure!(
            !symbols.is_empty(),
            "native phoneme override was empty; pass symbols like --phonemes \"OW K EY |\""
        );
        PiperPhonemeSequence {
            phonemes: symbols
                .into_iter()
                .map(|symbol| PiperPhoneme(symbol.to_string()))
                .collect(),
        }
    } else {
        let g2p = SimpleEnglishG2p::default();
        g2p.phonemize(&args.text)
            .with_context(|| format!("failed to phonemize text `{}` for native Piper", args.text))?
    };

    phoneme_sequence
        .to_piper_ids(config)
        .or_else(|_| espeak_compatible_ids(&phoneme_sequence, config))
        .with_context(|| {
            format!(
                "native voice config cannot map one or more phonemes for `{}`; pass --phonemes to override",
                args.text
            )
        })
}

#[cfg(feature = "tts-piper-native")]
fn espeak_compatible_ids(
    phoneme_sequence: &PiperPhonemeSequence,
    config: &PiperVoiceConfig,
) -> std::result::Result<
    PiperIdSequence,
    listenbury::mouth::piper_native::PiperPhonemeIdConversionError,
> {
    let sequence = espeak_compatible_sequence(phoneme_sequence, config)?;
    sequence.to_piper_ids(config)
}

#[cfg(feature = "tts-piper-native")]
fn espeak_compatible_sequence(
    phoneme_sequence: &PiperPhonemeSequence,
    config: &PiperVoiceConfig,
) -> std::result::Result<
    PiperPhonemeSequence,
    listenbury::mouth::piper_native::PiperPhonemeIdConversionError,
> {
    let mut symbols = vec![PiperPhoneme("^".to_string())];
    for phoneme in &phoneme_sequence.phonemes {
        let expanded = expand_espeak_phoneme(&phoneme.0, config).ok_or_else(|| {
            listenbury::mouth::piper_native::PiperPhonemeIdConversionError::UnknownPhoneme {
                symbol: phoneme.0.clone(),
            }
        })?;
        symbols.extend(expanded.into_iter().map(PiperPhoneme));
    }
    symbols.push(PiperPhoneme("$".to_string()));

    let mut interspersed = Vec::with_capacity(symbols.len().saturating_mul(2).saturating_sub(1));
    for (index, symbol) in symbols.into_iter().enumerate() {
        if index > 0 {
            interspersed.push(PiperPhoneme("_".to_string()));
        }
        interspersed.push(symbol);
    }

    Ok(PiperPhonemeSequence {
        phonemes: interspersed,
    })
}

#[cfg(feature = "tts-piper-native")]
fn expand_espeak_phoneme(symbol: &str, config: &PiperVoiceConfig) -> Option<Vec<String>> {
    let expanded = match symbol {
        "AA" => &["ɑ"][..],
        "AH" => &["ə"],
        "AY" => &["a", "ɪ"],
        "D" => &["d"],
        "EH" => &["ɛ"],
        "ER" => &["ɚ"],
        "EY" => &["ˈ", "e", "ɪ"],
        "F" => &["f"],
        "IH" => &["ɪ"],
        "IY" => &["i"],
        "JH" => &["d", "ʒ"],
        "K" => &["k"],
        "L" => &["l"],
        "NG" => &["ŋ"],
        "OW" => &["o", "ʊ"],
        "R" => &["ɹ"],
        "S" => &["s"],
        "T" => &["t"],
        "TS" => &["t", "s"],
        "|" => &["."],
        _ if config.phoneme_id_map.contains_key(symbol) => return Some(vec![symbol.to_string()]),
        _ => return None,
    };

    expanded
        .iter()
        .all(|symbol| config.phoneme_id_map.contains_key(*symbol))
        .then(|| {
            expanded
                .iter()
                .map(|symbol| (*symbol).to_string())
                .collect()
        })
}

#[cfg(feature = "tts-piper-native")]
fn read_native_voice_config(path: &Path) -> Result<PiperVoiceConfig> {
    let json = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read native Piper config at {}", path.display()))?;
    PiperVoiceConfig::from_json_str(&json).with_context(|| {
        format!(
            "failed to parse native Piper config JSON at {}",
            path.display()
        )
    })
}

#[cfg(feature = "tts-piper-native")]
fn report_compare_stats(process: &SynthesisStats, native: &SynthesisStats) {
    println!("process runtime: {}ms", process.runtime.as_millis());
    println!("native inference: {}ms", native.runtime.as_millis());
    println!(
        "process sample rate: {} Hz | native sample rate: {} Hz",
        process.audio.sample_rate_hz, native.audio.sample_rate_hz
    );
    println!(
        "process duration: {:.2}ms | native duration: {:.2}ms",
        process.audio.duration_ms, native.audio.duration_ms
    );
    println!(
        "process samples: {} | native samples: {}",
        process.audio.sample_count, native.audio.sample_count
    );
    println!(
        "process rms/peak: {:.5}/{:.5} | native rms/peak: {:.5}/{:.5}",
        process.audio.rms, process.audio.peak_abs, native.audio.rms, native.audio.peak_abs
    );
}

#[derive(Debug)]
struct SayArgs {
    piper_bin: Option<PathBuf>,
    piper_voice: Option<PathBuf>,
    output_wav: Option<PathBuf>,
    text: String,
}

impl SayArgs {
    fn from_command(command: SayCommand) -> Result<Self> {
        let mut words = command.words;
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
    #[cfg(feature = "tts-piper-native")]
    fn piper_compare_args_joins_words_into_text() {
        let args = PiperCompareArgs::from_command(PiperCompareCommand {
            piper_bin: None,
            piper_voice: None,
            native_voice: None,
            native_config: None,
            process_output_wav: None,
            native_output_wav: None,
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
    #[cfg(feature = "tts-piper-native")]
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

        let ids = espeak_compatible_ids(&sequence, &config)
            .expect("ARPAbet symbols should map to eSpeak Piper IDs");

        assert_eq!(
            ids,
            PiperIdSequence {
                ids: vec![1, 0, 27, 0, 100, 0, 23, 0, 120, 0, 18, 0, 74, 0, 10, 0, 2]
            }
        );
    }

    #[test]
    #[cfg(feature = "tts-piper-native")]
    fn audio_stats_computes_duration_rms_and_peak() {
        let stats = AudioStats::from_frames(
            &[AudioFrame {
                captured_at: ExactTimestamp::now(),
                sample_rate_hz: 20,
                channels: 1,
                samples: vec![0.0, 0.5, -1.0, 0.5],
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
