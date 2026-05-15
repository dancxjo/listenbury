use crate::cli::ModelsCommand;
use anyhow::Result;

#[cfg(feature = "model-download")]
use anyhow::Context;
#[cfg(feature = "model-download")]
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
#[cfg(feature = "model-download")]
use listenbury::models::{
    default_asset_paths, default_assets_status, fetch_default_assets_with_progress,
    paths::resolve_listenbury_home, FetchOutcome,
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
            let bars = MultiProgress::new();
            let overall = bars.add(ProgressBar::new(default_asset_paths()?.len() as u64));
            let overall_style = ProgressStyle::with_template(
                "{spinner:.cyan} {msg} [{wide_bar:.cyan/blue}] {pos}/{len} ETA {eta_precise}",
            )
            .context("failed to create overall progress style")?
            .progress_chars("=>-");
            overall.set_style(overall_style);
            overall.enable_steady_tick(std::time::Duration::from_millis(100));
            overall.set_message("Fetching default model assets...");

            let progress = bars.add(ProgressBar::new(0));
            let download_style = ProgressStyle::with_template(
                "{spinner:.cyan} {msg} [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} {bytes_per_sec} ETA {eta_precise}",
            )
            .context("failed to create download progress style")?
            .progress_chars("=>-");
            progress.set_style(download_style);
            progress.enable_steady_tick(std::time::Duration::from_millis(100));
            progress.set_message("Waiting for first download...");

            let results = fetch_default_assets_with_progress(|asset_progress| {
                overall.set_length(asset_progress.asset_count as u64);
                overall.set_position(asset_progress.asset_index as u64);
                overall.set_message(format!("Fetching {}...", asset_progress.asset_id));

                match asset_progress.total_bytes {
                    Some(total_bytes) => progress.set_length(total_bytes),
                    None => progress.unset_length(),
                }
                progress.set_position(asset_progress.downloaded_bytes);
                progress.set_message(format!(
                    "{} -> {}",
                    asset_progress.asset_id,
                    asset_progress.path.display()
                ));
            })?;
            overall.set_position(overall.length().unwrap_or(0));
            overall.finish_and_clear();
            progress.finish_and_clear();
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
