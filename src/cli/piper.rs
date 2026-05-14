use crate::cli::PiperSayCommand;
use crate::cli::model_paths::resolve_piper_voice;
use anyhow::{Context, Result};
use listenbury::audio::frame::AudioFrame;
use listenbury::audio::write_wav;
use listenbury::mouth::planner::{SpeechPlan, SpeechUnit};
use listenbury::mouth::tts::TextToSpeech;
use listenbury::{PiperConfig, PiperTextToSpeech};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub(crate) fn run_piper_say(command: PiperSayCommand) -> Result<()> {
    let piper_args = PiperSayArgs::from_command(command)?;
    let piper_bin = resolve_piper_bin(piper_args.piper_bin);
    let piper_voice = resolve_piper_voice(piper_args.piper_voice)?;
    let mut tts = PiperTextToSpeech::new(piper_config_for_voice(piper_bin, piper_voice)?);
    tts.enqueue(SpeechPlan::from(SpeechUnit::FullTurn(piper_args.text)))?;
    let frames = collect_tts_audio(&mut tts, Duration::from_secs(30))?;

    std::fs::create_dir_all("out").context("failed to create out directory")?;
    let output_path = std::path::Path::new("out/listenbury-piper-test.wav");
    write_wav(output_path, &frames)?;

    let sample_count: usize = frames.iter().map(|frame| frame.samples.len()).sum();
    println!(
        "Wrote {} frames / {} samples to {}",
        frames.len(),
        sample_count,
        output_path.display()
    );

    Ok(())
}

#[derive(Debug)]
struct PiperSayArgs {
    piper_bin: Option<PathBuf>,
    piper_voice: Option<PathBuf>,
    text: String,
}

impl PiperSayArgs {
    fn from_command(command: PiperSayCommand) -> Result<Self> {
        let mut words = command.words;
        let mut piper_bin = command.piper_bin;
        let mut piper_voice = command.piper_voice;

        if piper_bin.is_none() && words.first().is_some_and(|word| looks_like_piper_bin(word)) {
            piper_bin = Some(PathBuf::from(words.remove(0)));
        }

        if piper_voice.is_none() && words.first().is_some_and(|word| word.ends_with(".onnx")) {
            piper_voice = Some(PathBuf::from(words.remove(0)));
        }

        anyhow::ensure!(
            !words.is_empty(),
            "missing text to speak; try `piper-say hello`"
        );

        Ok(Self {
            piper_bin,
            piper_voice,
            text: words.join(" "),
        })
    }
}

fn looks_like_piper_bin(word: &str) -> bool {
    let path = Path::new(word);
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.contains("piper"))
}

pub(crate) fn resolve_piper_bin(explicit: Option<PathBuf>) -> PathBuf {
    explicit
        .or_else(|| std::env::var_os("LISTENBURY_PIPER_BIN").map(PathBuf::from))
        .or_else(|| find_piper_executable("piper"))
        .or_else(|| find_piper_executable("piper-tts.piper-cli"))
        .unwrap_or_else(|| PathBuf::from("piper"))
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
    if !uses_snap_piper(piper_bin) || !has_hidden_component(&model_path) {
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

    #[test]
    fn piper_say_args_treats_single_word_as_text() {
        let args = PiperSayArgs::from_command(PiperSayCommand {
            piper_bin: None,
            piper_voice: None,
            words: vec!["hello".to_string()],
        })
        .expect("single word should be text");

        assert!(args.piper_bin.is_none());
        assert!(args.piper_voice.is_none());
        assert_eq!(args.text, "hello");
    }

    #[test]
    fn piper_say_args_accepts_legacy_piper_bin_position() {
        let args = PiperSayArgs::from_command(PiperSayCommand {
            piper_bin: None,
            piper_voice: None,
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
    fn piper_say_args_accepts_legacy_voice_position() {
        let args = PiperSayArgs::from_command(PiperSayCommand {
            piper_bin: None,
            piper_voice: None,
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
}
