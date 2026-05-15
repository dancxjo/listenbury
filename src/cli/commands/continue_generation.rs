use crate::cli::ContinueCommand;
use anyhow::Result;

#[cfg(feature = "llm-llama-cpp")]
use crate::cli::commands::llama::build_prompt;
#[cfg(feature = "llm-llama-cpp")]
use crate::cli::model_paths::{llm_runtime_placement, resolve_llm_model};
#[cfg(feature = "llm-llama-cpp")]
use anyhow::Context;
#[cfg(feature = "llm-llama-cpp")]
use listenbury::mind::llm::{GenerationRequest, LlmEngine, LlmEvent};
#[cfg(feature = "llm-llama-cpp")]
use listenbury::{LlamaCppConfig, LlamaCppEngine};
#[cfg(feature = "llm-llama-cpp")]
use std::io::{BufRead, Write};
#[cfg(feature = "llm-llama-cpp")]
use std::sync::Arc;
#[cfg(feature = "llm-llama-cpp")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "llm-llama-cpp")]
use std::time::Duration;

#[cfg(feature = "llm-llama-cpp")]
pub(crate) fn run_continue(command: ContinueCommand) -> Result<()> {
    let max_tokens =
        usize::try_from(command.max_tokens).context("max_tokens does not fit in usize")?;
    anyhow::ensure!(max_tokens > 0, "max_tokens must be greater than zero");
    anyhow::ensure!(
        command.context_size > 0,
        "context_size must be greater than zero"
    );

    let model_path = resolve_llm_model(command.llm_model)?;
    let llm_placement = llm_runtime_placement(&model_path, command.llm_gpu_layers, None)?;
    let config = LlamaCppConfig {
        model_path,
        gpu_layers: llm_placement.gpu_layers,
        cpu_only: llm_placement.cpu_only,
        context_size: command.context_size,
        ..Default::default()
    };

    let initial_prompt = if command.prompt.is_empty() {
        "Continue generating while new context is appended.\n\n".to_string()
    } else {
        command.prompt.join(" ")
    };
    let (prompt, stop) = build_prompt(command.mode, &initial_prompt);
    let mut llm = LlamaCppEngine::new(config).context("failed to initialize llama.cpp engine")?;
    let id = llm
        .start(GenerationRequest {
            prompt,
            max_tokens: Some(max_tokens),
            stop,
        })
        .context("failed to start continued llama.cpp generation")?;

    let interrupted = Arc::new(AtomicBool::new(false));
    ctrlc::set_handler({
        let interrupted = Arc::clone(&interrupted);
        move || {
            interrupted.store(true, Ordering::Relaxed);
        }
    })
    .context("failed to install Ctrl-C handler")?;

    let (stdin_tx, stdin_rx) =
        crossbeam_channel::unbounded::<std::result::Result<String, String>>();
    std::thread::Builder::new()
        .name("listenbury-dev-continue-stdin".to_string())
        .spawn(move || {
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if stdin_tx.send(Ok(line)).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = stdin_tx.send(Err(error.to_string()));
                        break;
                    }
                }
            }
        })
        .context("failed to spawn stdin reader")?;

    eprintln!(
        "listenbury dev continue: streaming one generation; stdin lines append to the live context. Ctrl-C cancels."
    );

    let mut cancelled = false;
    loop {
        if interrupted.load(Ordering::Relaxed) && !cancelled {
            llm.cancel(id)?;
            cancelled = true;
        }

        for stdin_event in stdin_rx.try_iter() {
            match stdin_event {
                Ok(text) => llm
                    .append_prompt(id, wrap_live_input(&text))
                    .context("failed to append stdin text to live generation")?,
                Err(message) => anyhow::bail!("failed to read stdin: {message}"),
            }
        }

        let events = llm.poll(id)?;
        if events.is_empty() {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        }

        for event in &events {
            match event {
                LlmEvent::Token { text } => {
                    print!("{text}");
                    std::io::stdout().flush()?;
                }
                LlmEvent::Error { message } => {
                    anyhow::bail!("continued llama.cpp generation failed: {message}");
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
pub(crate) fn run_continue(_command: ContinueCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `llm-llama-cpp` feature")
}

#[cfg(any(feature = "llm-llama-cpp", test))]
fn wrap_live_input(text: &str) -> String {
    format!(
        "\n<live_input>\nUSER: {}\n</live_input>\n<assistant_continues>\n",
        text.trim()
    )
}

#[cfg(test)]
mod tests {
    use super::wrap_live_input;

    #[test]
    fn stdin_append_is_wrapped_as_live_input() {
        assert_eq!(
            wrap_live_input("turn toward the window\n"),
            "\n<live_input>\nUSER: turn toward the window\n</live_input>\n<assistant_continues>\n"
        );
    }
}
