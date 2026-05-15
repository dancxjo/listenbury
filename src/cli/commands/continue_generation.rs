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
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(feature = "llm-llama-cpp")]
const DEFAULT_CONTINUE_PROMPT: &str = "You are Pete Listenbury, an experiment in artificial awareness. Please continuously generate thoughts as new input arrives from the outside world. Try to understand what's going on around you and make new friends.\n\n";
#[cfg(feature = "llm-llama-cpp")]
const TIME_EVENT_INTERVAL: Duration = Duration::from_secs(10);

#[cfg(feature = "llm-llama-cpp")]
pub(crate) fn run_continue(command: ContinueCommand) -> Result<()> {
    let max_tokens = command
        .max_tokens
        .map(|max_tokens| usize::try_from(max_tokens).context("max_tokens does not fit in usize"))
        .transpose()?;
    if let Some(max_tokens) = max_tokens {
        anyhow::ensure!(max_tokens > 0, "max_tokens must be greater than zero");
    }
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
        DEFAULT_CONTINUE_PROMPT.to_string()
    } else {
        command.prompt.join(" ")
    };
    let (prompt, stop) = build_prompt(command.mode, &initial_prompt);
    let mut llm = LlamaCppEngine::new(config).context("failed to initialize llama.cpp engine")?;
    let id = llm
        .start(GenerationRequest {
            prompt,
            max_tokens,
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
        "listenbury dev continue: streaming one generation; stdin lines and 10s time events append to the live context. Ctrl-C cancels."
    );

    let mut cancelled = false;
    let mut next_time_event_at = Instant::now() + TIME_EVENT_INTERVAL;
    loop {
        if interrupted.load(Ordering::Relaxed) && !cancelled {
            llm.cancel(id)?;
            cancelled = true;
        }

        let now = Instant::now();
        if now >= next_time_event_at {
            llm.append_prompt(id, wrap_time_event(&current_time_message()))
                .context("failed to append time event to live generation")?;
            next_time_event_at = now + TIME_EVENT_INTERVAL;
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

#[cfg(any(feature = "llm-llama-cpp", test))]
fn wrap_time_event(message: &str) -> String {
    format!("\n<live_input>\nTIME: {message}\n</live_input>\n<assistant_continues>\n")
}

#[cfg(feature = "llm-llama-cpp")]
fn current_time_message() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::ZERO);
    format!(
        "The current Unix time is {}.{:03} seconds.",
        now.as_secs(),
        now.subsec_millis()
    )
}

#[cfg(test)]
mod tests {
    use super::{wrap_live_input, wrap_time_event};

    #[test]
    fn stdin_append_is_wrapped_as_live_input() {
        assert_eq!(
            wrap_live_input("turn toward the window\n"),
            "\n<live_input>\nUSER: turn toward the window\n</live_input>\n<assistant_continues>\n"
        );
    }

    #[test]
    fn time_append_is_wrapped_as_live_input() {
        assert_eq!(
            wrap_time_event("The current Unix time is 42.000 seconds."),
            "\n<live_input>\nTIME: The current Unix time is 42.000 seconds.\n</live_input>\n<assistant_continues>\n"
        );
    }
}
