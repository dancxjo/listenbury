use anyhow::{Context, Result};
use listenbury::audio::frame::AudioFrame;
use listenbury::hearing::breath::BreathGroupSegmenter;
use listenbury::hearing::vad::{EnergyVad, VoiceActivityDetector};
use listenbury::mind::llm::{GenerationRequest, LlmEngine, LlmEvent, MockLlmEngine};
#[cfg(feature = "model-download")]
use listenbury::models::{
    FetchOutcome, default_asset_paths, default_assets_status, fetch_default_assets,
    paths::resolve_listenbury_home,
};
#[cfg(feature = "tts-piper")]
use listenbury::mouth::planner::SpeechPlan;
use listenbury::mouth::planner::SpeechPlanner;
#[cfg(feature = "tts-piper")]
use listenbury::mouth::tts::TextToSpeech;
#[cfg(feature = "asr-whisper")]
use listenbury::speech::recognizer::SpeechRecognizer;
use listenbury::time::ExactTimestamp;
#[cfg(feature = "llm-llama-cpp")]
use listenbury::{LlamaCppConfig, LlamaCppEngine};
#[cfg(feature = "tts-piper")]
use listenbury::{PiperConfig, PiperTextToSpeech};

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let mut args = std::env::args().skip(1);
    let Some(command) = args.next() else {
        print_usage();
        return Ok(());
    };

    match command.as_str() {
        "fake-turn" => {
            let user_text = args.collect::<Vec<_>>().join(" ");
            if user_text.is_empty() {
                anyhow::bail!("usage: listenbury fake-turn \"hello there\"");
            }
            run_fake_turn(user_text)
        }
        "demo-vad" => run_demo_vad(),
        "llama-turn" => {
            let Some(model_path) = args.next() else {
                anyhow::bail!("usage: listenbury llama-turn <model.gguf> \"prompt\"");
            };
            let prompt = args.collect::<Vec<_>>().join(" ");
            if prompt.is_empty() {
                anyhow::bail!("usage: listenbury llama-turn <model.gguf> \"prompt\"");
            }
            run_llama_turn(model_path, prompt)
        }
        "transcribe-synthetic" => {
            let Some(model_path) = args.next() else {
                anyhow::bail!("usage: listenbury transcribe-synthetic <model.bin>");
            };
            run_transcribe_synthetic(model_path)
        }
        "piper-say" => {
            let Some(piper_bin) = args.next() else {
                anyhow::bail!("usage: listenbury piper-say <piper-bin> <voice.onnx> \"text\"");
            };
            let Some(model_path) = args.next() else {
                anyhow::bail!("usage: listenbury piper-say <piper-bin> <voice.onnx> \"text\"");
            };
            let text = args.collect::<Vec<_>>().join(" ");
            if text.is_empty() {
                anyhow::bail!("usage: listenbury piper-say <piper-bin> <voice.onnx> \"text\"");
            }
            run_piper_say(piper_bin, model_path, text)
        }
        "models" => run_models(args),
        _ => {
            print_usage();
            Ok(())
        }
    }
}

fn print_usage() {
    println!("Usage:");
    println!("  listenbury fake-turn \"hello there\"");
    println!("  listenbury demo-vad");
    println!("  listenbury llama-turn <model.gguf> \"prompt\"");
    println!("  listenbury transcribe-synthetic <model.bin>");
    println!("  listenbury piper-say <piper-bin> <voice.onnx> \"text\"");
    println!("  listenbury models <fetch|status|path>");
}

#[cfg(feature = "model-download")]
fn run_models(mut args: impl Iterator<Item = String>) -> Result<()> {
    let Some(subcommand) = args.next() else {
        anyhow::bail!("usage: listenbury models <fetch|status|path>");
    };

    match subcommand.as_str() {
        "path" => {
            let home = resolve_listenbury_home()?;
            println!("listenbury_home={}", home.display());
            println!("models_dir={}", home.join("models").display());
            println!("bin_dir={}", home.join("bin").display());
            for (asset, path) in default_asset_paths()? {
                println!("{}={}", asset.id, path.display());
            }
            Ok(())
        }
        "status" => {
            for status in default_assets_status()? {
                let state = if status.present { "present" } else { "missing" };
                println!("{} {} {}", status.asset_id, state, status.path.display());
            }
            Ok(())
        }
        "fetch" => {
            let mut had_failure = false;
            for result in fetch_default_assets()? {
                match result.outcome {
                    FetchOutcome::SkippedExisting => {
                        println!("{} skipped {}", result.asset_id, result.path.display());
                    }
                    FetchOutcome::Downloaded => {
                        println!("{} downloaded {}", result.asset_id, result.path.display());
                    }
                    FetchOutcome::Failed => {
                        had_failure = true;
                        println!(
                            "{} failed {} ({})",
                            result.asset_id,
                            result.path.display(),
                            result.error.as_deref().unwrap_or("unknown error")
                        );
                    }
                }
            }
            if had_failure {
                anyhow::bail!("one or more model assets failed to fetch");
            }
            Ok(())
        }
        _ => anyhow::bail!("usage: listenbury models <fetch|status|path>"),
    }
}

#[cfg(not(feature = "model-download"))]
fn run_models(_args: impl Iterator<Item = String>) -> Result<()> {
    anyhow::bail!("listenbury was built without the `model-download` feature")
}

fn run_fake_turn(user_text: String) -> Result<()> {
    let mut llm = MockLlmEngine::with_response(vec!["I ".into(), "heard ".into(), "you.".into()]);
    let request = GenerationRequest {
        prompt: format!("User said: {user_text}"),
        max_tokens: None,
    };

    let id = llm.start(request).context("failed to start generation")?;
    let mut planner = SpeechPlanner::default();

    loop {
        let events = llm.poll(id)?;
        if events.is_empty() {
            continue;
        }

        for event in &events {
            if let LlmEvent::Token { text } = event {
                print!("{text}");
            }
        }

        if let Some(plan) = planner.ingest(&events) {
            println!();
            println!("SpeechPlan: {plan:?}");
        }

        if events.iter().any(|event| {
            matches!(
                event,
                LlmEvent::Completed | LlmEvent::Cancelled | LlmEvent::Error { .. }
            )
        }) {
            break;
        }
    }

    Ok(())
}

#[cfg(feature = "llm-llama-cpp")]
fn run_llama_turn(model_path: String, prompt: String) -> Result<()> {
    let config = LlamaCppConfig {
        model_path: model_path.into(),
        ..Default::default()
    };
    let mut llm = LlamaCppEngine::new(config).context("failed to initialize llama.cpp engine")?;
    let id = llm
        .start(GenerationRequest {
            prompt,
            max_tokens: None,
        })
        .context("failed to start llama.cpp generation")?;

    loop {
        let events = llm.poll(id)?;
        if events.is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(5));
            continue;
        }

        for event in &events {
            match event {
                LlmEvent::Token { text } => {
                    print!("{text}");
                    use std::io::Write;
                    std::io::stdout().flush()?;
                }
                LlmEvent::Error { message } => {
                    anyhow::bail!("llama.cpp generation failed: {message}");
                }
                LlmEvent::Completed | LlmEvent::Cancelled => {}
            }
        }

        if events.iter().any(|event| {
            matches!(
                event,
                LlmEvent::Completed | LlmEvent::Cancelled | LlmEvent::Error { .. }
            )
        }) {
            println!();
            break;
        }
    }

    Ok(())
}

#[cfg(not(feature = "llm-llama-cpp"))]
fn run_llama_turn(_model_path: String, _prompt: String) -> Result<()> {
    anyhow::bail!("listenbury was built without the `llm-llama-cpp` feature")
}

fn run_demo_vad() -> Result<()> {
    let mut vad = EnergyVad::new(0.02);
    let mut segmenter = BreathGroupSegmenter::default();

    let amplitudes = [
        0.0_f32, 0.0, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    for amp in amplitudes {
        let frame = AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![amp; 160],
        };
        let vad_result = vad.process_frame(&frame)?;
        for event in segmenter.process(vad_result) {
            println!("{event:?}");
        }
    }

    Ok(())
}

#[cfg(feature = "asr-whisper")]
fn run_transcribe_synthetic(model_path: String) -> Result<()> {
    let mut recognizer = listenbury::WhisperSpeechRecognizer::new(&model_path)
        .with_context(|| format!("failed to load Whisper model at {model_path}"))?;

    recognizer.push_frame(&AudioFrame {
        captured_at: ExactTimestamp::now(),
        sample_rate_hz: 16_000,
        channels: 1,
        samples: vec![0.0; 16_000],
    })?;

    let chunks = recognizer.poll_chunks()?;
    if chunks.is_empty() {
        println!("No transcript chunks produced.");
        return Ok(());
    }

    for chunk in chunks {
        println!("{chunk:?}");
    }

    Ok(())
}

#[cfg(not(feature = "asr-whisper"))]
fn run_transcribe_synthetic(_model_path: String) -> Result<()> {
    anyhow::bail!("listenbury was built without the `asr-whisper` feature")
}

#[cfg(feature = "tts-piper")]
fn run_piper_say(piper_bin: String, model_path: String, text: String) -> Result<()> {
    let model_path = std::path::PathBuf::from(model_path);
    let inferred_config_path = model_path.with_extension("onnx.json");
    let mut config = PiperConfig::new(piper_bin, model_path);
    if inferred_config_path.exists() {
        if let Some(sample_rate_hz) = read_piper_sample_rate_hz(&inferred_config_path)? {
            config.sample_rate_hz = sample_rate_hz;
        }
        config.config_path = Some(inferred_config_path);
    }

    let mut tts = PiperTextToSpeech::new(config);
    tts.enqueue(SpeechPlan::FullTurn(text))?;

    let mut frames = Vec::new();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let quiet_after_audio = std::time::Duration::from_millis(100);
    let mut last_audio_at = None;
    while std::time::Instant::now() < deadline {
        let new_frames = tts.poll_audio()?;
        if new_frames.is_empty() {
            if let Some(last_audio_at) = last_audio_at {
                if std::time::Instant::now().duration_since(last_audio_at) >= quiet_after_audio {
                    break;
                }
            }
        } else {
            frames.extend(new_frames);
            last_audio_at = Some(std::time::Instant::now());
        }

        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    if frames.is_empty() {
        anyhow::bail!("Piper produced no audio frames before timeout");
    }

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

#[cfg(not(feature = "tts-piper"))]
fn run_piper_say(_piper_bin: String, _model_path: String, _text: String) -> Result<()> {
    anyhow::bail!("listenbury was built without the `tts-piper` feature")
}

#[cfg(feature = "tts-piper")]
fn write_wav(path: &std::path::Path, frames: &[AudioFrame]) -> Result<()> {
    let Some(first_frame) = frames.first() else {
        anyhow::bail!("cannot write WAV without audio frames");
    };

    let spec = hound::WavSpec {
        channels: first_frame.channels,
        sample_rate: first_frame.sample_rate_hz,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("failed to create WAV at {}", path.display()))?;

    for frame in frames {
        anyhow::ensure!(
            frame.channels == first_frame.channels,
            "Piper frame channel count changed from {} to {}",
            first_frame.channels,
            frame.channels
        );
        anyhow::ensure!(
            frame.sample_rate_hz == first_frame.sample_rate_hz,
            "Piper frame sample rate changed from {} to {}",
            first_frame.sample_rate_hz,
            frame.sample_rate_hz
        );

        for sample in &frame.samples {
            writer.write_sample(f32_to_i16(*sample))?;
        }
    }

    writer.finalize()?;
    Ok(())
}

#[cfg(feature = "tts-piper")]
fn f32_to_i16(sample: f32) -> i16 {
    let sample = sample.clamp(-1.0, 1.0);
    (sample * i16::MAX as f32) as i16
}

#[cfg(feature = "tts-piper")]
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
