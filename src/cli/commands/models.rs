use crate::cli::{ModelsCommand, ModelsFetchCommand, ModelsUseCommand, ModelsUseKind};
use anyhow::Result;

#[cfg(feature = "model-download")]
use anyhow::Context;
#[cfg(feature = "model-download")]
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
#[cfg(feature = "model-download")]
use listenbury::models::{
    FetchOutcome, bundle_present, default_asset_paths, default_assets_status,
    fetch_all_assets_with_progress, fetch_bundle_with_progress,
    fetch_selected_assets_with_progress, find_bundle,
    manifest::{MODEL_BUNDLES, ModelBundle, ModelKind},
    paths::resolve_listenbury_home,
    read_model_selection, selected_bundle, write_model_selection,
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
            println!(
                "{}={}",
                "selection".cyan(),
                listenbury::models::model_selection_path()?.display()
            );
            for (asset, path) in default_asset_paths()? {
                println!("{}={}", asset.id.cyan(), path.display());
            }
            Ok(())
        }
        ModelsCommand::List => {
            let llm = selected_bundle(ModelKind::Llm)?.id;
            let voice = selected_bundle(ModelKind::Voice)?.id;
            for kind in [ModelKind::Llm, ModelKind::Voice, ModelKind::Whisper] {
                println!("{}", listenbury::models::model_kind_label(kind).bold());
                for bundle in MODEL_BUNDLES.iter().filter(|bundle| bundle.kind == kind) {
                    let marker = if (kind == ModelKind::Llm && bundle.id == llm)
                        || (kind == ModelKind::Voice && bundle.id == voice)
                    {
                        "*"
                    } else {
                        " "
                    };
                    let state = if bundle_present(bundle)? {
                        "present".green().to_string()
                    } else {
                        "missing".red().to_string()
                    };
                    println!(
                        "{} {} {:<28} {}",
                        marker,
                        bundle.id.bold(),
                        bundle.display_name,
                        state
                    );
                }
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
        ModelsCommand::Use(command) => use_model(command),
        ModelsCommand::Fetch(command) => fetch_models(command),
    }
}

#[cfg(feature = "model-download")]
fn use_model(command: ModelsUseCommand) -> Result<()> {
    let kind = match command.kind {
        ModelsUseKind::Llm => ModelKind::Llm,
        ModelsUseKind::Voice => ModelKind::Voice,
    };
    let bundle = find_bundle(kind, &command.model).with_context(|| {
        format!(
            "unknown {} model `{}`; run `listenbury models list`",
            listenbury::models::model_kind_label(kind),
            command.model
        )
    })?;
    let mut selection = read_model_selection()?;
    match command.kind {
        ModelsUseKind::Llm => selection.llm = Some(bundle.id.to_string()),
        ModelsUseKind::Voice => selection.voice = Some(bundle.id.to_string()),
    }
    write_model_selection(&selection)?;
    println!(
        "{} {} {}",
        "selected".green(),
        listenbury::models::model_kind_label(kind),
        bundle.display_name.bold()
    );
    Ok(())
}

#[cfg(feature = "model-download")]
fn fetch_models(command: ModelsFetchCommand) -> Result<()> {
    let result = if command.all {
        progress_fetch("Fetching every registered model asset...", None)
    } else if let Some(model) = command.model {
        let bundle = find_bundle(ModelKind::Llm, &model)
            .or_else(|| find_bundle(ModelKind::Voice, &model))
            .or_else(|| find_bundle(ModelKind::Whisper, &model))
            .with_context(|| format!("unknown model `{model}`; run `listenbury models list`"))?;
        progress_fetch(
            &format!("Fetching {}...", bundle.display_name),
            Some(bundle),
        )
    } else {
        progress_fetch("Fetching selected model assets...", None)
    }?;
    print_fetch_results(result)
}

#[cfg(feature = "model-download")]
fn progress_fetch(
    message: &str,
    bundle: Option<&ModelBundle>,
) -> Result<Vec<listenbury::models::FetchResult>> {
    let bars = MultiProgress::new();
    let overall = bars.add(ProgressBar::new(0));
    let overall_style = ProgressStyle::with_template(
        "{spinner:.cyan} {msg} [{wide_bar:.cyan/blue}] {pos}/{len} ETA {eta_precise}",
    )
    .context("failed to create overall progress style")?
    .progress_chars("=>-");
    overall.set_style(overall_style);
    overall.enable_steady_tick(std::time::Duration::from_millis(100));
    overall.set_message(message.to_string());

    let progress = bars.add(ProgressBar::new(0));
    let download_style = ProgressStyle::with_template(
                "{spinner:.cyan} {msg} [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} {bytes_per_sec} ETA {eta_precise}",
            )
            .context("failed to create download progress style")?
            .progress_chars("=>-");
    progress.set_style(download_style);
    progress.enable_steady_tick(std::time::Duration::from_millis(100));
    progress.set_message("Waiting for first download...");

    let results = match (message.contains("every registered"), bundle) {
        (true, _) => fetch_all_assets_with_progress(|asset_progress| {
            update_progress(&overall, &progress, asset_progress);
        })?,
        (_, Some(bundle)) => fetch_bundle_with_progress(bundle, |asset_progress| {
            update_progress(&overall, &progress, asset_progress);
        })?,
        _ => fetch_selected_assets_with_progress(|asset_progress| {
            update_progress(&overall, &progress, asset_progress);
        })?,
    };
    overall.set_position(overall.length().unwrap_or(0));
    overall.finish_and_clear();
    progress.finish_and_clear();
    Ok(results)
}

#[cfg(feature = "model-download")]
fn update_progress(
    overall: &ProgressBar,
    progress: &ProgressBar,
    asset_progress: listenbury::models::FetchProgress,
) {
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
}

#[cfg(feature = "model-download")]
fn print_fetch_results(results: Vec<listenbury::models::FetchResult>) -> Result<()> {
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

#[cfg(not(feature = "model-download"))]
pub(crate) fn run_models(_command: ModelsCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `model-download` feature")
}
