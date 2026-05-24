use crate::cli::ModelsCommand;
#[cfg(feature = "model-download")]
use crate::cli::download_progress::DownloadProgress;
#[cfg(feature = "model-download")]
use crate::cli::{
    ModelsFetchCommand, ModelsStatusCommand, ModelsUseCommand, ModelsUseKind, ModelsVerifyCommand,
};
use anyhow::Result;

#[cfg(feature = "model-download")]
use anyhow::Context;
#[cfg(feature = "model-download")]
use inquire::Select;
#[cfg(feature = "model-download")]
use listenbury::models::{
    FetchOutcome, bundle_assets, bundle_present, default_asset_paths,
    default_assets_status_with_verification,
    download::{AssetIntegrityState, verify_existing_asset},
    fetch_all_assets_with_progress_and_jobs_and_verify,
    fetch_bundle_with_progress_and_jobs_and_verify,
    fetch_selected_assets_with_progress_and_jobs_and_verify, find_bundle,
    manifest::{DEFAULT_MODELS, MODEL_BUNDLES, ModelBundle, ModelKind},
    paths::{asset_path, resolve_listenbury_home},
    read_model_selection, selected_bundle, write_model_selection,
};
#[cfg(feature = "model-download")]
use owo_colors::OwoColorize;
#[cfg(feature = "model-download")]
use std::fmt;

#[cfg(feature = "model-download")]
pub(crate) fn run_models(command: Option<ModelsCommand>) -> Result<()> {
    match command {
        None | Some(ModelsCommand::Menu) => model_menu(),
        Some(ModelsCommand::Path) => {
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
        Some(ModelsCommand::List) => print_models_list(),
        Some(ModelsCommand::Status(command)) => print_status(command),
        Some(ModelsCommand::Verify(command)) => verify_models(command),
        Some(ModelsCommand::Repair(command)) => repair_models(command),
        Some(ModelsCommand::Use(command)) => use_model(command),
        Some(ModelsCommand::Fetch(command)) => fetch_models(command),
    }
}

#[cfg(feature = "model-download")]
fn model_menu() -> Result<()> {
    let category = Select::new(
        "Model category",
        vec![
            CategoryChoice::new(ModelKind::Llm)?,
            CategoryChoice::new(ModelKind::Voice)?,
            CategoryChoice::new(ModelKind::Whisper)?,
        ],
    )
    .prompt()
    .context("model menu was cancelled")?;

    let bundles = MODEL_BUNDLES
        .iter()
        .filter(|bundle| bundle.kind == category.kind)
        .map(BundleChoice::new)
        .collect::<Result<Vec<_>>>()?;
    let current = selected_bundle(category.kind)?.id;
    let starting_cursor = bundles
        .iter()
        .position(|choice| choice.bundle.id == current)
        .unwrap_or(0);
    let selected = Select::new(&format!("{} model", category.name), bundles)
        .with_starting_cursor(starting_cursor)
        .prompt()
        .context("model selection was cancelled")?;

    select_bundle(category.kind, selected.bundle)
}

#[cfg(feature = "model-download")]
fn print_models_list() -> Result<()> {
    let llm = selected_bundle(ModelKind::Llm)?.id;
    let voice = selected_bundle(ModelKind::Voice)?.id;
    let whisper = selected_bundle(ModelKind::Whisper)?.id;
    for kind in [ModelKind::Llm, ModelKind::Voice, ModelKind::Whisper] {
        println!("{}", listenbury::models::model_kind_label(kind).bold());
        for bundle in MODEL_BUNDLES.iter().filter(|bundle| bundle.kind == kind) {
            let marker = if (kind == ModelKind::Llm && bundle.id == llm)
                || (kind == ModelKind::Voice && bundle.id == voice)
                || (kind == ModelKind::Whisper && bundle.id == whisper)
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

#[cfg(feature = "model-download")]
fn use_model(command: ModelsUseCommand) -> Result<()> {
    let kind = match command.kind {
        ModelsUseKind::Llm => ModelKind::Llm,
        ModelsUseKind::Voice => ModelKind::Voice,
        ModelsUseKind::Whisper => ModelKind::Whisper,
    };
    let bundle = find_bundle(kind, &command.model).with_context(|| {
        format!(
            "unknown {} model `{}`; run `listenbury models list`",
            listenbury::models::model_kind_label(kind),
            command.model
        )
    })?;
    select_bundle(kind, bundle)
}

#[cfg(feature = "model-download")]
fn select_bundle(kind: ModelKind, bundle: &ModelBundle) -> Result<()> {
    let mut selection = read_model_selection()?;
    match kind {
        ModelKind::Llm => selection.llm = Some(bundle.id.to_string()),
        ModelKind::Voice => selection.voice = Some(bundle.id.to_string()),
        ModelKind::Whisper => selection.whisper = Some(bundle.id.to_string()),
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
#[derive(Clone)]
struct CategoryChoice {
    kind: ModelKind,
    name: &'static str,
    label: String,
}

#[cfg(feature = "model-download")]
impl CategoryChoice {
    fn new(kind: ModelKind) -> Result<Self> {
        let name = match kind {
            ModelKind::Llm => "LLM",
            ModelKind::Voice => "Voice",
            ModelKind::Whisper => "Whisper",
        };
        let selected = selected_bundle(kind)?;
        let label = format!(
            "{name:<7} {}",
            format!("current: {}", selected.display_name).dimmed()
        );
        Ok(Self { kind, name, label })
    }
}

#[cfg(feature = "model-download")]
impl fmt::Display for CategoryChoice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.label)
    }
}

#[cfg(feature = "model-download")]
#[derive(Clone)]
struct BundleChoice {
    bundle: &'static ModelBundle,
    label: String,
}

#[cfg(feature = "model-download")]
impl BundleChoice {
    fn new(bundle: &'static ModelBundle) -> Result<Self> {
        let state = if bundle_present(bundle)? {
            "present".green().to_string()
        } else {
            "missing".red().to_string()
        };
        Ok(Self {
            bundle,
            label: format!("{:<36} {}", bundle.display_name, state),
        })
    }
}

#[cfg(feature = "model-download")]
impl fmt::Display for BundleChoice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.label)
    }
}

#[cfg(feature = "model-download")]
fn fetch_models(command: ModelsFetchCommand) -> Result<()> {
    let jobs = command.jobs.max(1);
    let result = if command.all {
        progress_fetch(
            "Fetching every registered model asset...",
            None,
            jobs,
            command.verify,
        )
    } else if let Some(model) = command.model {
        let bundle = find_bundle(ModelKind::Llm, &model)
            .or_else(|| find_bundle(ModelKind::Voice, &model))
            .or_else(|| find_bundle(ModelKind::Whisper, &model))
            .with_context(|| format!("unknown model `{model}`; run `listenbury models list`"))?;
        progress_fetch(
            &format!("Fetching {}...", bundle.display_name),
            Some(bundle),
            jobs,
            command.verify,
        )
    } else {
        progress_fetch(
            "Fetching selected model assets...",
            None,
            jobs,
            command.verify,
        )
    }?;
    print_fetch_results(result)
}

#[cfg(feature = "model-download")]
fn progress_fetch(
    message: &str,
    bundle: Option<&ModelBundle>,
    jobs: usize,
    verify_existing: bool,
) -> Result<Vec<listenbury::models::FetchResult>> {
    let mut progress = DownloadProgress::new(message)?;

    let results = match (message.contains("every registered"), bundle) {
        (true, _) => fetch_all_assets_with_progress_and_jobs_and_verify(
            jobs,
            verify_existing,
            |asset_progress| {
                progress.update(asset_progress);
            },
        )?,
        (_, Some(bundle)) => fetch_bundle_with_progress_and_jobs_and_verify(
            bundle,
            jobs,
            verify_existing,
            |asset_progress| {
                progress.update(asset_progress);
            },
        )?,
        _ => fetch_selected_assets_with_progress_and_jobs_and_verify(
            jobs,
            verify_existing,
            |asset_progress| {
                progress.update(asset_progress);
            },
        )?,
    };
    progress.finish_and_clear();
    Ok(results)
}

#[cfg(feature = "model-download")]
fn print_status(command: ModelsStatusCommand) -> Result<()> {
    for status in default_assets_status_with_verification(command.verify)? {
        let state = integrity_label(status.integrity);
        println!(
            "{} {} {} source={} license={} url={}",
            status.asset_id.bold(),
            state,
            status.path.display(),
            status.source.unwrap_or("unknown"),
            status.license.unwrap_or("unknown"),
            status.url
        );
    }
    Ok(())
}

#[cfg(feature = "model-download")]
fn repair_models(command: ModelsFetchCommand) -> Result<()> {
    fetch_models(ModelsFetchCommand {
        verify: true,
        ..command
    })
}

#[cfg(feature = "model-download")]
fn verify_models(command: ModelsVerifyCommand) -> Result<()> {
    let home = resolve_listenbury_home()?;
    let assets: Vec<_> = if let Some(model) = &command.model {
        let bundle = find_bundle(ModelKind::Llm, model)
            .or_else(|| find_bundle(ModelKind::Voice, model))
            .or_else(|| find_bundle(ModelKind::Whisper, model))
            .with_context(|| format!("unknown model `{model}`; run `listenbury models list`"))?;
        bundle_assets(bundle)?
    } else {
        DEFAULT_MODELS.iter().collect()
    };

    let mut any_invalid = false;
    for asset in assets {
        let path = asset_path(&home, asset);
        let integrity = verify_existing_asset(&path, asset, true)?;
        let state = integrity_label(integrity);
        let checksum_note = match asset.sha256 {
            Some(_) => String::new(),
            None => " (no checksum in manifest)".dimmed().to_string(),
        };
        println!("{} {}{}", asset.id.bold(), state, checksum_note);
        match integrity {
            AssetIntegrityState::PresentInvalidSize
            | AssetIntegrityState::PresentInvalidChecksum
            | AssetIntegrityState::Missing => {
                any_invalid = true;
            }
            _ => {}
        }
    }
    if any_invalid {
        anyhow::bail!("one or more model assets failed verification");
    }
    Ok(())
}

#[cfg(feature = "model-download")]
fn integrity_label(state: AssetIntegrityState) -> String {
    match state {
        AssetIntegrityState::Missing => "missing".red().to_string(),
        AssetIntegrityState::PresentUnverified => "present-unverified".yellow().to_string(),
        AssetIntegrityState::PresentValid => "present-valid".green().to_string(),
        AssetIntegrityState::PresentInvalidSize => "present-invalid-size".red().to_string(),
        AssetIntegrityState::PresentInvalidChecksum => "present-invalid-checksum".red().to_string(),
        AssetIntegrityState::UnknownChecksum => "unknown-checksum".yellow().to_string(),
    }
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
pub(crate) fn run_models(_command: Option<ModelsCommand>) -> Result<()> {
    anyhow::bail!("listenbury was built without the `model-download` feature")
}
