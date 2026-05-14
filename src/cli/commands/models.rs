use crate::cli::ModelsCommand;
use anyhow::Result;

#[cfg(feature = "model-download")]
use anyhow::Context;
#[cfg(feature = "model-download")]
use indicatif::{ProgressBar, ProgressStyle};
#[cfg(feature = "model-download")]
use listenbury::models::{
    FetchOutcome, default_asset_paths, default_assets_status, fetch_default_assets,
    paths::resolve_listenbury_home,
};
#[cfg(feature = "model-download")]
use owo_colors::OwoColorize;

#[cfg(feature = "model-download")]
pub(crate) fn run_models(command: ModelsCommand) -> Result<()> {
    match command {
        ModelsCommand::Path => {
            let home = resolve_listenbury_home()?;
            println!("{}={}", "listenbury_home".cyan(), home.display());
            println!("{}={}", "models_dir".cyan(), home.join("models").display());
            println!("{}={}", "bin_dir".cyan(), home.join("bin").display());
            for (asset, path) in default_asset_paths()? {
                println!("{}={}", asset.id.cyan(), path.display());
            }
            Ok(())
        }
        ModelsCommand::Status => {
            for status in default_assets_status()? {
                let state = if status.present {
                    "present".green().to_string()
                } else {
                    "missing".red().to_string()
                };
                println!(
                    "{} {} {}",
                    status.asset_id.bold(),
                    state,
                    status.path.display()
                );
            }
            Ok(())
        }
        ModelsCommand::Fetch => {
            let spinner = ProgressBar::new_spinner();
            let style = ProgressStyle::with_template("{spinner:.cyan} {msg}")
                .context("failed to create spinner style")?;
            spinner.set_style(style);
            spinner.enable_steady_tick(std::time::Duration::from_millis(100));
            spinner.set_message("Fetching default model assets...");

            let results = fetch_default_assets()?;
            spinner.finish_and_clear();
            let mut had_failure = false;
            for result in results {
                match result.outcome {
                    FetchOutcome::SkippedExisting => {
                        println!(
                            "{} {} {}",
                            result.asset_id.bold(),
                            "skipped".yellow(),
                            result.path.display()
                        );
                    }
                    FetchOutcome::Downloaded => {
                        println!(
                            "{} {} {}",
                            result.asset_id.bold(),
                            "downloaded".green(),
                            result.path.display()
                        );
                    }
                    FetchOutcome::Failed => {
                        had_failure = true;
                        println!(
                            "{} {} {} ({})",
                            result.asset_id.bold(),
                            "failed".red(),
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
    }
}

#[cfg(not(feature = "model-download"))]
pub(crate) fn run_models(_command: ModelsCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `model-download` feature")
}
