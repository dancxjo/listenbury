use crate::cli::LlamaTurnCommand;
#[cfg(feature = "llm-llama-cpp")]
use crate::cli::PromptMode;
#[cfg(feature = "llm-llama-cpp")]
use crate::cli::model_paths::{llm_runtime_placement, resolve_llm_model};
#[cfg(feature = "llm-llama-cpp")]
use anyhow::Context;
use anyhow::Result;
#[cfg(feature = "llm-llama-cpp")]
use listenbury::mind::llm::{GenerationRequest, LlmEngine, LlmEvent};
#[cfg(feature = "llm-llama-cpp")]
use listenbury::{LlamaCppConfig, LlamaCppEngine};
#[cfg(feature = "llm-llama-cpp")]
use std::io::Write;
#[cfg(feature = "llm-llama-cpp")]
use std::path::PathBuf;

#[cfg(feature = "llm-llama-cpp")]
pub(crate) fn run_llama_turn(command: LlamaTurnCommand) -> Result<()> {
    let args = LlamaTurnArgs::from_command(command)?;
    let model_path = resolve_llm_model(args.llm_model)?;
    let llm_placement = llm_runtime_placement(&model_path, args.llm_gpu_layers, None)?;
    let config = LlamaCppConfig {
        model_path,
        gpu_layers: llm_placement.gpu_layers,
        cpu_only: llm_placement.cpu_only,
        ..Default::default()
    };
    let mut llm = LlamaCppEngine::new(config).context("failed to initialize llama.cpp engine")?;
    let id = llm
        .start(GenerationRequest {
            prompt: args.prompt,
            max_tokens: Some(args.max_tokens),
            stop: args.stop,
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
pub(crate) fn run_llama_turn(_command: LlamaTurnCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `llm-llama-cpp` feature")
}

#[cfg(feature = "llm-llama-cpp")]
#[derive(Debug)]
struct LlamaTurnArgs {
    llm_model: Option<PathBuf>,
    llm_gpu_layers: Option<u32>,
    prompt: String,
    max_tokens: usize,
    stop: Vec<String>,
}

#[cfg(feature = "llm-llama-cpp")]
impl LlamaTurnArgs {
    fn from_command(command: LlamaTurnCommand) -> Result<Self> {
        let mut prompt = command.prompt;
        let mut llm_model = command.llm_model;
        let max_tokens =
            usize::try_from(command.max_tokens).context("max_tokens does not fit in usize")?;
        anyhow::ensure!(max_tokens > 0, "max_tokens must be greater than zero");

        if llm_model.is_none() && prompt.first().is_some_and(|word| word.ends_with(".gguf")) {
            llm_model = Some(PathBuf::from(prompt.remove(0)));
        }

        anyhow::ensure!(
            !prompt.is_empty(),
            "missing prompt; try `llama-turn \"hello\"`"
        );

        let user_prompt = prompt.join(" ");
        let (prompt, stop) = build_prompt(command.mode, &user_prompt);

        Ok(Self {
            llm_model,
            llm_gpu_layers: command.llm_gpu_layers,
            prompt,
            max_tokens,
            stop,
        })
    }
}

#[cfg(feature = "llm-llama-cpp")]
pub(crate) fn build_prompt(mode: PromptMode, user_prompt: &str) -> (String, Vec<String>) {
    match mode {
        PromptMode::Raw => (user_prompt.to_string(), Vec::new()),
        PromptMode::Spoken => (
            format!(
                "<|system|>\n\
                 You are Pete Listenbury, a low-latency spoken agent.\n\
                 You answer in brief, natural speech.\n\
                 Never write songs, poems, Markdown, labels, scripts, or multiple paragraphs.\n\
                 Give one conversational reply only, usually 3 to 15 words.</s>\n\
                 <|user|>\n\
                 {user_prompt}</s>\n\
                 <|assistant|>\n"
            ),
            spoken_stops(),
        ),
        PromptMode::Chat => (
            format!(
                "<|system|>\n\
                 You are Pete Listenbury, a brief conversational agent.\n\
                 Reply naturally and directly. One assistant turn only.</s>\n\
                 <|user|>\n\
                 {user_prompt}</s>\n\
                 <|assistant|>\n"
            ),
            role_stops(),
        ),
        PromptMode::Inner => (
            format!(
                "<|system|>\n\
                 You are Pete Listenbury thinking privately before speech.\n\
                 Keep the thought short, practical, and unformatted.</s>\n\
                 <|user|>\n\
                 {user_prompt}</s>\n\
                 <|assistant|>\n"
            ),
            role_stops(),
        ),
    }
}

#[cfg(feature = "llm-llama-cpp")]
fn role_stops() -> Vec<String> {
    vec![
        "\nUser:".into(),
        "\nPete:".into(),
        "\nAssistant:".into(),
        "\n<|user|>".into(),
        "</s>".into(),
    ]
}

#[cfg(feature = "llm-llama-cpp")]
fn spoken_stops() -> Vec<String> {
    let mut stops = role_stops();
    stops.extend([
        "\n(Verse".into(),
        "\n(Chorus".into(),
        "\nVerse".into(),
        "\nChorus".into(),
        "\n[".into(),
    ]);
    stops
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "llm-llama-cpp")]
    use super::*;

    #[cfg(feature = "llm-llama-cpp")]
    #[test]
    fn llama_turn_args_treats_single_argument_as_prompt() {
        let args = LlamaTurnArgs::from_command(LlamaTurnCommand {
            llm_model: None,
            llm_gpu_layers: None,
            mode: PromptMode::Raw,
            max_tokens: 48,
            prompt: vec!["hello".to_string()],
        })
        .expect("single argument should be prompt");

        assert!(args.llm_model.is_none());
        assert_eq!(args.prompt, "hello");
    }

    #[cfg(feature = "llm-llama-cpp")]
    #[test]
    fn llama_turn_args_accepts_legacy_model_position() {
        let args = LlamaTurnArgs::from_command(LlamaTurnCommand {
            llm_model: None,
            llm_gpu_layers: None,
            mode: PromptMode::Raw,
            max_tokens: 48,
            prompt: vec![
                "llama-3.2-3b-instruct.gguf".to_string(),
                "hello".to_string(),
            ],
        })
        .expect("legacy model path should be accepted");

        assert_eq!(
            args.llm_model,
            Some(PathBuf::from("llama-3.2-3b-instruct.gguf"))
        );
        assert_eq!(args.prompt, "hello");
    }

    #[cfg(feature = "llm-llama-cpp")]
    #[test]
    fn llama_turn_args_wraps_spoken_prompt_by_default() {
        let args = LlamaTurnArgs::from_command(LlamaTurnCommand {
            llm_model: None,
            llm_gpu_layers: None,
            mode: PromptMode::Spoken,
            max_tokens: 32,
            prompt: vec!["Can you hear me?".to_string()],
        })
        .expect("spoken prompt should parse");

        assert_eq!(args.max_tokens, 32);
        assert!(args.prompt.contains("You are Pete Listenbury"));
        assert!(args.prompt.contains("<|user|>\nCan you hear me?</s>"));
        assert!(args.prompt.ends_with("<|assistant|>\n"));
        assert!(args.stop.contains(&"\n(Verse".to_string()));
        assert!(args.stop.contains(&"\nUser:".to_string()));
    }
}
