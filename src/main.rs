use anyhow::{Context, Result};
use listenbury::audio::frame::AudioFrame;
use listenbury::hearing::breath::BreathGroupSegmenter;
use listenbury::hearing::vad::{EnergyVad, VoiceActivityDetector};
use listenbury::mind::llm::{GenerationRequest, LlmEngine, LlmEvent, MockLlmEngine};
use listenbury::mouth::planner::SpeechPlanner;
#[cfg(feature = "asr-whisper")]
use listenbury::speech::recognizer::SpeechRecognizer;
use listenbury::time::ExactTimestamp;
#[cfg(feature = "llm-llama-cpp")]
use listenbury::{LlamaCppConfig, LlamaCppEngine};

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
