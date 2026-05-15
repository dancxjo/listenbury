//! Two-instance mouth-to-ear dogfood diagnostic.
//!
//! Runs two named Listenbury instances (A and B) that exchange turns through
//! the full TTS → audio-bridge → ASR pipeline, proving the loop:
//!
//! ```text
//! A mouth → TTS audio → B ear → ASR → B mind → B mouth → TTS audio → A ear → …
//! ```

use crate::cli::DogfoodTwoCommand;
use anyhow::Result;

// ── Safety guard helpers ─────────────────────────────────────────────────────
//
// These are plain functions with no feature requirements so they can be
// exercised by unit tests even when the full asr+llm+tts feature set is absent.

/// Returns `true` when `transcript` is blank or contains only whitespace.
#[cfg_attr(
    not(all(
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )),
    allow(dead_code)
)]
pub(crate) fn is_empty_transcript(transcript: &str) -> bool {
    transcript.trim().is_empty()
}

/// Returns `true` when `new_entry` (after trimming) equals the most recent
/// entry in `history`, indicating the pipeline is stuck in a repetition loop.
#[cfg_attr(
    not(all(
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )),
    allow(dead_code)
)]
pub(crate) fn is_repeated_transcript(history: &[String], new_entry: &str) -> bool {
    history
        .last()
        .is_some_and(|last| last.trim() == new_entry.trim())
}

// ── Full implementation (requires all three feature flags) ───────────────────

#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use crate::cli::model_paths::{resolve_llm_model, resolve_piper_voice, resolve_whisper_model};
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
use listenbury::mouth::planner::{SpeechPlan, SpeechUnit};
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
use listenbury::{
    BreathAsrConfig, LlamaCppConfig, LlamaCppEngine, PiperTextToSpeech, WhisperSpeechRecognizer,
    collect_breath_segments,
};
#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use serde::Serialize;
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
use std::path::Path;
#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
use std::time::{Duration, Instant};

/// Hard ceiling on turns regardless of the `--turns` flag.
#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
const MAX_HARD_TURNS: usize = 32;

/// Maximum wall-clock seconds to wait for a single TTS synthesis to complete.
#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
const MAX_TTS_TIMEOUT_SECS: u64 = 30;

/// Maximum synthesised audio duration (ms) per turn; longer audio triggers a
/// hard stop to prevent runaway generation.
#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
const MAX_AUDIO_DURATION_MS: u64 = 60_000;

// ── JSONL trace record ────────────────────────────────────────────────────────

#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
#[derive(Serialize)]
struct TurnRecord {
    turn: usize,
    speaker: String,
    listener: String,
    input_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    generated_text: Option<String>,
    speech_text: String,
    audio_duration_ms: u64,
    asr_transcript: String,
    timings: TurnTimings,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_reason: Option<String>,
}

#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
#[derive(Serialize)]
struct TurnTimings {
    #[serde(skip_serializing_if = "Option::is_none")]
    llm_ms: Option<u64>,
    tts_ms: u64,
    asr_ms: u64,
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
pub(crate) fn run_dogfood_two(command: DogfoodTwoCommand) -> Result<()> {
    let turns = command.turns.min(MAX_HARD_TURNS);

    // ── Resolve model paths ──────────────────────────────────────────────────
    let whisper_model = resolve_whisper_model(command.whisper_model)?;
    let llm_model = resolve_llm_model(command.llm_model)?;
    let piper_bin = resolve_piper_bin(command.piper_bin)?;
    let piper_voice_a = resolve_piper_voice(command.piper_voice_a)?;
    // Instance B defaults to the same voice as A when --piper-voice-b is absent.
    let piper_voice_b = resolve_piper_voice(command.piper_voice_b)?;

    // Pre-build Piper configs (cloned each turn to create a fresh TTS worker).
    let piper_config_a = piper_config_for_voice(piper_bin.clone(), piper_voice_a)?;
    let piper_config_b = piper_config_for_voice(piper_bin, piper_voice_b)?;

    // ── LLM engine ───────────────────────────────────────────────────────────
    let mut llm = LlamaCppEngine::new(LlamaCppConfig {
        model_path: llm_model,
        ..Default::default()
    })
    .context("failed to initialise LLM engine")?;

    // ── Output directories / files ───────────────────────────────────────────
    if let Some(dir) = &command.save_audio_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("failed to create audio dir {}", dir.display()))?;
    }

    let mut jsonl_writer: Option<std::io::BufWriter<std::fs::File>> =
        if let Some(path) = &command.jsonl {
            let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
            if let Some(parent) = parent {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!("failed to create JSONL directory {}", parent.display())
                })?;
            }
            let file = std::fs::File::create(path)
                .with_context(|| format!("failed to create JSONL file at {}", path.display()))?;
            Some(std::io::BufWriter::new(file))
        } else {
            None
        };

    // ── Conversation state ───────────────────────────────────────────────────
    let mut current_input = command.seed.clone();
    let mut speaker_is_a = true;
    let mut history_a: Vec<String> = Vec::new();
    let mut history_b: Vec<String> = Vec::new();

    // ── Main loop ────────────────────────────────────────────────────────────
    for turn_num in 1..=turns {
        let speaker = if speaker_is_a { "A" } else { "B" };
        let listener = if speaker_is_a { "B" } else { "A" };

        println!("\n=== Turn {turn_num} / {turns}  {speaker} → {listener} ===");

        // Step 1 – Determine speech text
        //   Turn 1: A speaks the seed directly (no LLM generation).
        //   All other turns: current speaker generates a response via LLM.
        let (speech_text, generated_text, llm_ms) = if turn_num == 1 {
            println!("[A] Seed: {}", command.seed);
            (command.seed.clone(), None, None)
        } else {
            let t = Instant::now();
            let text = generate_response(&mut llm, speaker, &current_input, command.max_tokens)
                .with_context(|| format!("LLM generation failed on turn {turn_num}"))?;
            let elapsed = t.elapsed().as_millis() as u64;
            println!("[{speaker}] Generated: {text}");
            (text.clone(), Some(text), Some(elapsed))
        };

        // Step 2 – TTS synthesis
        let piper_config = if speaker_is_a {
            piper_config_a.clone()
        } else {
            piper_config_b.clone()
        };
        let mut tts = PiperTextToSpeech::new(piper_config);
        tts.enqueue(SpeechPlan::from(SpeechUnit::FullTurn(speech_text.clone())))?;

        let tts_t = Instant::now();
        let audio = collect_tts_audio(&mut tts, Duration::from_secs(MAX_TTS_TIMEOUT_SECS))
            .with_context(|| format!("TTS synthesis failed on turn {turn_num}"))?;
        let tts_ms = tts_t.elapsed().as_millis() as u64;

        // Compute synthesised audio duration
        let total_samples: usize = audio.iter().map(|f| f.samples.len()).sum();
        let sample_rate = audio.first().map(|f| f.sample_rate_hz).unwrap_or(22_050);
        let audio_duration_ms = (total_samples as u64 * 1_000) / u64::from(sample_rate);

        // Guard: synthesised audio too long
        if audio_duration_ms > MAX_AUDIO_DURATION_MS {
            let record = TurnRecord {
                turn: turn_num,
                speaker: speaker.to_string(),
                listener: listener.to_string(),
                input_text: current_input.clone(),
                generated_text,
                speech_text,
                audio_duration_ms,
                asr_transcript: String::new(),
                timings: TurnTimings {
                    llm_ms,
                    tts_ms,
                    asr_ms: 0,
                },
                stop_reason: Some("audio_too_long".to_string()),
            };
            emit_record(&mut jsonl_writer, &record)?;
            println!("Stopping: audio too long ({audio_duration_ms} ms)");
            break;
        }

        // Step 3 – Optionally save per-turn WAV
        if let Some(dir) = &command.save_audio_dir {
            let name = format!("turn-{turn_num:03}-{speaker}-to-{listener}.wav");
            write_wav(&dir.join(&name), &audio)
                .with_context(|| format!("failed to save audio to {name}"))?;
        }

        // Step 4 – ASR: bridge audio through a temporary WAV file
        let asr_t = Instant::now();
        let asr_result = transcribe_audio_bridge(&audio, &whisper_model, turn_num);
        let asr_ms = asr_t.elapsed().as_millis() as u64;

        let (asr_transcript, asr_stop) = match asr_result {
            Ok(t) => (t, None),
            Err(e) => (String::new(), Some(format!("asr_failure: {e}"))),
        };

        println!("[{listener}] Heard: {asr_transcript}");

        // Step 5 – Safety guards
        let listener_history = if speaker_is_a { &history_b } else { &history_a };

        let stop_reason = asr_stop.or_else(|| {
            if is_empty_transcript(&asr_transcript) {
                Some("empty_transcript".to_string())
            } else if is_repeated_transcript(listener_history, &asr_transcript) {
                Some("repeated_transcript".to_string())
            } else {
                None
            }
        });

        // Step 6 – Emit structured JSONL trace
        let record = TurnRecord {
            turn: turn_num,
            speaker: speaker.to_string(),
            listener: listener.to_string(),
            input_text: current_input.clone(),
            generated_text,
            speech_text,
            audio_duration_ms,
            asr_transcript: asr_transcript.clone(),
            timings: TurnTimings {
                llm_ms,
                tts_ms,
                asr_ms,
            },
            stop_reason: stop_reason.clone(),
        };
        emit_record(&mut jsonl_writer, &record)?;

        if let Some(reason) = stop_reason {
            println!("Stopping: {reason}");
            break;
        }

        // Advance state for the next turn
        if speaker_is_a {
            history_b.push(asr_transcript.clone());
        } else {
            history_a.push(asr_transcript.clone());
        }
        current_input = asr_transcript;
        speaker_is_a = !speaker_is_a;
    }

    println!("\nDogfood run complete.");
    Ok(())
}

/// Stub for builds that lack the full feature set.
#[cfg(not(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
)))]
pub(crate) fn run_dogfood_two(_command: DogfoodTwoCommand) -> Result<()> {
    anyhow::bail!(
        "listenbury was built without the `asr-whisper`, `llm-llama-cpp`, and `tts-piper` features"
    )
}

// ── Internal helpers (feature-gated) ─────────────────────────────────────────

/// Generate a spoken response for `instance_id` given `input` text from the
/// other instance.  Returns the raw token stream trimmed of surrounding
/// whitespace.
#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn generate_response(
    llm: &mut LlamaCppEngine,
    instance_id: &str,
    input: &str,
    max_tokens: u32,
) -> Result<String> {
    let max_tokens = usize::try_from(max_tokens).context("max_tokens does not fit in usize")?;
    let generation_id = llm
        .start(GenerationRequest {
            prompt: build_prompt(instance_id, input),
            max_tokens: Some(max_tokens),
        })
        .context("failed to start LLM generation")?;

    let mut tokens = String::new();
    loop {
        let events = llm.poll(generation_id)?;
        if events.is_empty() {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        }

        for event in &events {
            match event {
                LlmEvent::Token { text } => {
                    tokens.push_str(text);
                    print!("{text}");
                    std::io::stdout().flush()?;
                }
                LlmEvent::Error { message } => {
                    anyhow::bail!("LLM error during generation: {message}");
                }
                LlmEvent::Completed | LlmEvent::Cancelled => {}
            }
        }

        if events.iter().any(|e| {
            matches!(
                e,
                LlmEvent::Completed | LlmEvent::Cancelled | LlmEvent::Error { .. }
            )
        }) {
            println!();
            break;
        }
    }

    Ok(tokens.trim().to_string())
}

/// Build the prompt for a dogfood turn.
#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn build_prompt(instance_id: &str, input: &str) -> String {
    format!(
        "<|system|>\n\
         You are instance {instance_id} of a conversational AI, speaking aloud through a TTS system.\n\
         Keep your response to one or two short, complete spoken sentences.\n\
         Do not use bullet points, lists, or special characters.\n\
         Each sentence should be speakable on its own.</s>\n\
         <|user|>\n{input}</s>\n\
         <|assistant|>\n"
    )
}

/// Convert TTS audio frames into a Whisper transcript by bouncing through a
/// temporary WAV file (which handles sample-rate conversion and channel mixing)
/// before running the Whisper recogniser.
#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn transcribe_audio_bridge(
    audio: &[listenbury::AudioFrame],
    whisper_model: &Path,
    turn_num: usize,
) -> Result<String> {
    // Write to a per-turn temp file to avoid collision between parallel runs.
    let tmp_path = std::env::temp_dir().join(format!(
        "listenbury-dogfood-bridge-{}-{turn_num}.wav",
        std::process::id()
    ));

    write_wav(&tmp_path, audio)
        .with_context(|| format!("failed to write bridge WAV at {}", tmp_path.display()))?;

    let frames = read_wav_as_whisper_frames(&tmp_path, 1_600)
        .with_context(|| format!("failed to read bridge WAV at {}", tmp_path.display()));

    // Best-effort cleanup – ignore errors so a read failure still surfaces.
    let _ = std::fs::remove_file(&tmp_path);

    let frames = frames?;

    let segments = collect_breath_segments(&frames, BreathAsrConfig::default())?;

    let mut transcripts: Vec<String> = Vec::new();

    if segments.is_empty() {
        // Fallback: feed the whole buffer to a single recogniser instance.
        let mut recognizer = WhisperSpeechRecognizer::new(whisper_model).with_context(|| {
            format!(
                "failed to load Whisper model at {}",
                whisper_model.display()
            )
        })?;
        for frame in &frames {
            recognizer.push_frame(frame)?;
        }
        transcripts.extend(recognizer.poll_chunks()?.into_iter().map(|c| c.text));
    } else {
        for segment in &segments {
            let mut recognizer =
                WhisperSpeechRecognizer::new(whisper_model).with_context(|| {
                    format!(
                        "failed to load Whisper model at {}",
                        whisper_model.display()
                    )
                })?;
            for frame in &segment.frames {
                recognizer.push_frame(frame)?;
            }
            transcripts.extend(recognizer.poll_chunks()?.into_iter().map(|c| c.text));
        }
    }

    Ok(transcripts.join(" ").trim().to_string())
}

/// Append one JSONL record to the optional writer.
#[cfg(all(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn emit_record(
    writer: &mut Option<std::io::BufWriter<std::fs::File>>,
    record: &TurnRecord,
) -> Result<()> {
    if let Some(w) = writer {
        serde_json::to_writer(&mut *w, record)?;
        w.write_all(b"\n")?;
        w.flush()?;
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── is_empty_transcript ──────────────────────────────────────────────────

    #[test]
    fn empty_string_is_empty_transcript() {
        assert!(is_empty_transcript(""));
    }

    #[test]
    fn whitespace_only_is_empty_transcript() {
        assert!(is_empty_transcript("   "));
        assert!(is_empty_transcript("\t\n"));
    }

    #[test]
    fn non_empty_transcript_is_not_empty() {
        assert!(!is_empty_transcript("Hello."));
        assert!(!is_empty_transcript("  word  "));
    }

    // ── is_repeated_transcript ───────────────────────────────────────────────

    #[test]
    fn empty_history_never_repeats() {
        assert!(!is_repeated_transcript(&[], "Hello."));
    }

    #[test]
    fn single_entry_history_repeats_on_exact_match() {
        let history = vec!["Hello.".to_string()];
        assert!(is_repeated_transcript(&history, "Hello."));
    }

    #[test]
    fn single_entry_history_does_not_repeat_on_different_entry() {
        let history = vec!["Hello.".to_string()];
        assert!(!is_repeated_transcript(&history, "Goodbye."));
    }

    #[test]
    fn repeat_check_trims_whitespace() {
        let history = vec!["Hello.".to_string()];
        assert!(is_repeated_transcript(&history, "  Hello.  "));
        assert!(is_repeated_transcript(&history, "Hello.\n"));
    }

    #[test]
    fn only_the_last_history_entry_is_compared() {
        let history = vec!["Hello.".to_string(), "Goodbye.".to_string()];
        // "Hello." was the second-to-last, so it does NOT trigger the guard.
        assert!(!is_repeated_transcript(&history, "Hello."));
        // "Goodbye." is the last entry, so it does trigger the guard.
        assert!(is_repeated_transcript(&history, "Goodbye."));
    }

    #[test]
    fn alternating_transcripts_do_not_trigger_repetition_guard() {
        let history = vec!["Hello.".to_string(), "Goodbye.".to_string()];
        // The new entry matches an earlier entry but not the last → no repeat.
        assert!(!is_repeated_transcript(&history, "Hello."));
    }
}
