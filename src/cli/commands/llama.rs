use crate::cli::LlamaTurnCommand;
#[cfg(feature = "llm-llama-cpp")]
use crate::cli::model_paths::resolve_llm_model;
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
    let config = LlamaCppConfig {
        model_path,
        ..Default::default()
    };
    let mut llm = LlamaCppEngine::new(config).context("failed to initialize llama.cpp engine")?;
    let id = llm
        .start(GenerationRequest {
            prompt: args.prompt,
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
    prompt: String,
}

#[cfg(feature = "llm-llama-cpp")]
impl LlamaTurnArgs {
    fn from_command(command: LlamaTurnCommand) -> Result<Self> {
        let mut prompt = command.prompt;
        let mut llm_model = command.llm_model;

        if llm_model.is_none() && prompt.first().is_some_and(|word| word.ends_with(".gguf")) {
            llm_model = Some(PathBuf::from(prompt.remove(0)));
        }

        anyhow::ensure!(
            !prompt.is_empty(),
            "missing prompt; try `llama-turn \"hello\"`"
        );

        Ok(Self {
            llm_model,
            prompt: prompt.join(" "),
        })
    }
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
            prompt: vec!["tinyllama.gguf".to_string(), "hello".to_string()],
        })
        .expect("legacy model path should be accepted");

        assert_eq!(args.llm_model, Some(PathBuf::from("tinyllama.gguf")));
        assert_eq!(args.prompt, "hello");
    }
}
