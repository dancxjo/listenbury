use crate::cli::RoundTripWavCommand;
use anyhow::Result;

#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use crate::cli::model_paths::{
    llm_runtime_placement, resolve_llm_model, resolve_piper_voice, resolve_whisper_model,
};
#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use crate::cli::piper::{collect_tts_audio, piper_config_for_voice, resolve_piper_bin};
#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use anyhow::Context;
#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::audio::{read_wav_as_whisper_frames, write_wav};
#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::mind::llm::{GenerationRequest, LlmEngine, LlmEvent};
#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::mouth::planner::{ExpressiveUnit, SpeechPlan, SpeechPlanner, SpeechUnit};
#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::mouth::tts::TextToSpeech;
#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::speech::recognizer::SpeechRecognizer;
#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::{BreathAsrConfig, collect_breath_segments};
#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use listenbury::{LlamaCppConfig, LlamaCppEngine, PiperTextToSpeech};
#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use std::io::Write;
#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use std::path::{Path, PathBuf};
#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use std::time::Duration;

#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
pub(crate) fn run_round_trip_wav(command: RoundTripWavCommand) -> Result<()> {
    let paths = RoundTripModelPaths::discover(command)?;
    let frames = read_wav_as_whisper_frames(&paths.input_wav, 1_600)?;
    let transcript = transcribe_frames(&paths, &frames)?;
    println!("Heard: {transcript}");

    let plan = generate_speech_plan(&paths, &transcript)?;
    let mut tts =
        PiperTextToSpeech::new(piper_config_for_voice(paths.piper_bin, paths.piper_voice)?);
    tts.enqueue(plan)?;
    let audio = collect_tts_audio(&mut tts, Duration::from_secs(30))?;

    std::fs::create_dir_all("out").context("failed to create out directory")?;
    let output_path = Path::new("out/listenbury-round-trip.wav");
    write_wav(output_path, &audio)?;
    println!("Wrote {}", output_path.display());

    Ok(())
}

#[cfg(not(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
)))]
pub(crate) fn run_round_trip_wav(_command: RoundTripWavCommand) -> Result<()> {
    anyhow::bail!(
        "listenbury was built without the `asr-whisper`, `llm-llama-cpp`, and `tts-piper` features"
    )
}

#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
#[derive(Debug, Clone)]
struct RoundTripModelPaths {
    input_wav: PathBuf,
    whisper_model: PathBuf,
    llm_model: PathBuf,
    llm_gpu_layers: Option<u32>,
    piper_bin: PathBuf,
    piper_voice: PathBuf,
}

#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
impl RoundTripModelPaths {
    fn discover(command: RoundTripWavCommand) -> Result<Self> {
        let llm_model = resolve_llm_model(command.llm_model)?;
        let llm_placement = llm_runtime_placement(&llm_model, command.llm_gpu_layers, None)?;
        Ok(Self {
            input_wav: command.input_wav,
            whisper_model: resolve_whisper_model(command.whisper_model)?,
            llm_model,
            llm_gpu_layers: llm_placement.gpu_layers,
            piper_bin: resolve_piper_bin(command.piper_bin)?,
            piper_voice: resolve_piper_voice(command.piper_voice)?,
        })
    }
}

#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn transcribe_frames(
    paths: &RoundTripModelPaths,
    frames: &[listenbury::AudioFrame],
) -> Result<String> {
    let segments = collect_breath_segments(frames, BreathAsrConfig::default())?;
    let mut transcripts = Vec::new();
    for segment in segments {
        let mut recognizer = listenbury::WhisperSpeechRecognizer::new(&paths.whisper_model)
            .with_context(|| {
                format!(
                    "failed to load Whisper model at {}",
                    paths.whisper_model.display()
                )
            })?;
        for frame in &segment.frames {
            recognizer.push_frame(frame)?;
        }
        transcripts.extend(
            recognizer
                .poll_chunks()?
                .into_iter()
                .map(|chunk| chunk.text),
        );
    }

    Ok(transcripts.join(" "))
}

#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn generate_speech_plan(paths: &RoundTripModelPaths, transcript: &str) -> Result<SpeechPlan> {
    let mut llm = LlamaCppEngine::new(LlamaCppConfig {
        model_path: paths.llm_model.clone(),
        gpu_layers: paths.llm_gpu_layers,
        cpu_only: paths.llm_gpu_layers == Some(0),
        ..Default::default()
    })
    .with_context(|| {
        format!(
            "failed to initialize llama.cpp with {}",
            paths.llm_model.display()
        )
    })?;

    let generation_id = llm
        .start(GenerationRequest {
            prompt: build_round_trip_prompt(transcript),
            max_tokens: Some(96),
            stop: Vec::new(),
        })
        .context("failed to start llama.cpp generation")?;

    let mut planner = SpeechPlanner::default();
    let mut spoken_parts = Vec::new();
    loop {
        let events = llm.poll(generation_id)?;
        if events.is_empty() {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        }

        print_llm_events(&events)?;

        for unit in planner.ingest(&events) {
            let ExpressiveUnit::Speech(plan) = unit else {
                continue;
            };
            spoken_parts.push(plan.text().to_string());
        }

        if events.iter().any(is_terminal_llm_event) {
            println!();
            break;
        }
    }

    let text = spoken_parts.join(" ");
    let text = text.trim();
    let response = if text.is_empty() {
        "I heard you, but I lost my words.".to_string()
    } else {
        text.to_string()
    };
    Ok(SpeechPlan::from(SpeechUnit::FullTurn(response)))
}

#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn print_llm_events(events: &[LlmEvent]) -> Result<()> {
    for event in events {
        match event {
            LlmEvent::Token { text } => {
                print!("{text}");
                std::io::stdout().flush()?;
            }
            LlmEvent::Error { message } => {
                anyhow::bail!("llama.cpp generation failed: {message}");
            }
            LlmEvent::Completed | LlmEvent::Cancelled => {}
        }
    }
    Ok(())
}

#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn is_terminal_llm_event(event: &LlmEvent) -> bool {
    matches!(
        event,
        LlmEvent::Completed | LlmEvent::Cancelled | LlmEvent::Error { .. }
    )
}

#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn build_round_trip_prompt(transcript: &str) -> String {
    format!(
        "<|system|>\nYou are Pete, speaking aloud through a TTS system.\nWrite in short, complete spoken sentences.\nDo not rely on long subordinate clauses.\nPrefer natural sentence boundaries.\nEach sentence should be speakable on its own.</s>\n<|user|>\n{transcript}</s>\n<|assistant|>\n"
    )
}
