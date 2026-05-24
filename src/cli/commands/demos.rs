#[cfg(feature = "tts-piper")]
use anyhow::Context;
use anyhow::Result;
use listenbury::audio::frame::AudioFrame;
use listenbury::hearing::breath::BreathGroupSegmenter;
use listenbury::hearing::vad::{EnergyVad, VoiceActivityDetector};
#[cfg(feature = "tts-piper")]
use listenbury::mind::llm::{GenerationRequest, LlmEngine, LlmEvent, MockLlmEngine};
#[cfg(feature = "tts-piper")]
use listenbury::mouth::planner::SyntheticPlanner;
use listenbury::time::ExactTimestamp;

#[cfg(feature = "tts-piper")]
pub(crate) fn run_fake_turn(user_text: String) -> Result<()> {
    let mut llm = MockLlmEngine::with_response(vec!["I ".into(), "heard ".into(), "you.".into()]);
    let request = GenerationRequest {
        prompt: format!("User said: {user_text}"),
        max_tokens: None,
        stop: Vec::new(),
    };

    let id = llm.start(request).context("failed to start generation")?;
    let mut planner = SyntheticPlanner::default();

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

        for unit in planner.ingest(&events) {
            println!();
            println!("ExpressiveUnit: {unit:?}");
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

#[cfg(not(feature = "tts-piper"))]
pub(crate) fn run_fake_turn(_user_text: String) -> Result<()> {
    anyhow::bail!("listenbury was built without the `tts-piper` feature")
}

pub(crate) fn run_demo_vad() -> Result<()> {
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
            voice_signatures: Vec::new(),
        };
        let vad_result = vad.process_frame(&frame)?;
        for event in segmenter.process(vad_result) {
            println!("{event:?}");
        }
    }

    Ok(())
}
