pub mod download;
pub mod manifest;
pub mod paths;

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use anyhow::{Context, Result};

use crate::models::download::{
    AssetIntegrityState, fetch_asset_with_progress_and_verify, verify_existing_asset,
};
use crate::models::manifest::{DEFAULT_MODELS, MODEL_BUNDLES, ModelAsset, ModelBundle, ModelKind};
use crate::models::paths::{asset_path, resolve_listenbury_home};

#[derive(Debug, Clone)]
pub struct ModelStatus {
    pub asset_id: &'static str,
    pub path: PathBuf,
    pub integrity: AssetIntegrityState,
    pub url: &'static str,
    pub expected_size_bytes: Option<u64>,
    pub sha256: Option<&'static str>,
    pub license: Option<&'static str>,
    pub source: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchOutcome {
    SkippedExisting,
    Downloaded,
    Failed,
}

#[derive(Debug, Clone)]
pub struct FetchResult {
    pub asset_id: &'static str,
    pub path: PathBuf,
    pub outcome: FetchOutcome,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FetchProgress {
    pub asset_id: &'static str,
    pub asset_index: usize,
    pub asset_count: usize,
    pub path: PathBuf,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
}

#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
pub struct ModelSelection {
    pub llm: Option<String>,
    pub voice: Option<String>,
    pub vocoder: Option<String>,
    pub whisper: Option<String>,
}

pub fn model_selection_path() -> Result<PathBuf> {
    Ok(resolve_listenbury_home()?
        .join("models")
        .join("selection.json"))
}

pub fn read_model_selection() -> Result<ModelSelection> {
    let path = model_selection_path()?;
    match std::fs::read_to_string(&path) {
        Ok(contents) => serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse model selection {}", path.display())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(ModelSelection::default()),
        Err(error) => {
            Err(error).with_context(|| format!("failed to read model selection {}", path.display()))
        }
    }
}

pub fn write_model_selection(selection: &ModelSelection) -> Result<()> {
    let path = model_selection_path()?;
    let parent = path
        .parent()
        .context("model selection path had no parent directory")?;
    std::fs::create_dir_all(parent).with_context(|| {
        format!(
            "failed to create model selection directory {}",
            parent.display()
        )
    })?;
    let contents =
        serde_json::to_string_pretty(selection).context("failed to encode model selection")?;
    std::fs::write(&path, format!("{contents}\n"))
        .with_context(|| format!("failed to write model selection {}", path.display()))
}

pub fn default_bundle_id(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::Llm => "llama-3-2-3b-instruct-q4-k-m",
        ModelKind::Voice => "ryan",
        ModelKind::Vocoder => "speecht5-hifigan",
        ModelKind::Whisper => "whisper-large-v3-turbo",
    }
}

pub fn find_asset(asset_id: &str) -> Option<&'static ModelAsset> {
    DEFAULT_MODELS.iter().find(|asset| asset.id == asset_id)
}

pub fn find_bundle(kind: ModelKind, name: &str) -> Option<&'static ModelBundle> {
    let normalized = normalize_model_name(name);
    MODEL_BUNDLES.iter().find(|bundle| {
        bundle.kind == kind
            && (normalize_model_name(bundle.id) == normalized
                || normalize_model_name(bundle.display_name) == normalized
                || bundle
                    .aliases
                    .iter()
                    .any(|alias| normalize_model_name(alias) == normalized)
                || bundle
                    .asset_ids
                    .iter()
                    .any(|asset_id| normalize_model_name(asset_id) == normalized))
    })
}

pub fn selected_bundle(kind: ModelKind) -> Result<&'static ModelBundle> {
    let env_value = match kind {
        ModelKind::Llm => std::env::var("PETE_LLM")
            .ok()
            .or_else(|| std::env::var("LISTENBURY_LLM").ok()),
        ModelKind::Voice => std::env::var("PETE_VOICE")
            .ok()
            .or_else(|| std::env::var("LISTENBURY_VOICE").ok()),
        ModelKind::Vocoder => std::env::var("PETE_VOCODER")
            .ok()
            .or_else(|| std::env::var("LISTENBURY_VOCODER").ok()),
        ModelKind::Whisper => std::env::var("PETE_WHISPER")
            .ok()
            .or_else(|| std::env::var("LISTENBURY_WHISPER").ok()),
    };
    let selection = if env_value.is_none() {
        Some(read_model_selection()?)
    } else {
        None
    };
    let configured = env_value
        .or_else(|| {
            selection.and_then(|selection| match kind {
                ModelKind::Llm => selection.llm,
                ModelKind::Voice => selection.voice,
                ModelKind::Vocoder => selection.vocoder,
                ModelKind::Whisper => selection.whisper,
            })
        })
        .unwrap_or_else(|| default_bundle_id(kind).to_string());

    find_bundle(kind, &configured).with_context(|| {
        format!(
            "unknown {} model `{configured}`; run `listenbury models list`",
            model_kind_label(kind)
        )
    })
}

pub fn model_kind_label(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::Llm => "llm",
        ModelKind::Voice => "voice",
        ModelKind::Vocoder => "vocoder",
        ModelKind::Whisper => "whisper",
    }
}

pub fn bundle_primary_path(bundle: &ModelBundle) -> Result<PathBuf> {
    let asset = find_asset(bundle.primary_asset_id).with_context(|| {
        format!(
            "model asset `{}` is not registered",
            bundle.primary_asset_id
        )
    })?;
    let home = resolve_listenbury_home()?;
    Ok(asset_path(&home, asset))
}

pub fn bundle_assets(bundle: &ModelBundle) -> Result<Vec<&'static ModelAsset>> {
    bundle
        .asset_ids
        .iter()
        .map(|asset_id| {
            find_asset(asset_id)
                .with_context(|| format!("model asset `{asset_id}` is not registered"))
        })
        .collect()
}

pub fn bundle_present(bundle: &ModelBundle) -> Result<bool> {
    let home = resolve_listenbury_home()?;
    for asset in bundle_assets(bundle)? {
        if !asset_path(&home, asset)
            .metadata()
            .map(|meta| meta.len() > 0)
            .unwrap_or(false)
        {
            return Ok(false);
        }
    }
    Ok(true)
}

pub fn fetch_bundle_with_progress(
    bundle: &ModelBundle,
    progress: impl FnMut(FetchProgress),
) -> Result<Vec<FetchResult>> {
    fetch_bundle_with_progress_and_jobs(bundle, 1, progress)
}

pub fn fetch_bundle_with_progress_and_jobs(
    bundle: &ModelBundle,
    jobs: usize,
    mut progress: impl FnMut(FetchProgress),
) -> Result<Vec<FetchResult>> {
    fetch_bundle_with_progress_and_jobs_and_verify(bundle, jobs, false, &mut progress)
}

pub fn fetch_bundle_with_progress_and_jobs_and_verify(
    bundle: &ModelBundle,
    jobs: usize,
    verify_existing: bool,
    mut progress: impl FnMut(FetchProgress),
) -> Result<Vec<FetchResult>> {
    let home = resolve_listenbury_home()?;
    let assets = bundle_assets(bundle)?;
    fetch_asset_refs_at_home(&home, &assets, jobs, verify_existing, &mut progress)
}

pub fn fetch_selected_assets_with_progress(
    progress: impl FnMut(FetchProgress),
) -> Result<Vec<FetchResult>> {
    fetch_selected_assets_with_progress_and_jobs(1, progress)
}

pub fn fetch_selected_assets_with_progress_and_jobs(
    jobs: usize,
    mut progress: impl FnMut(FetchProgress),
) -> Result<Vec<FetchResult>> {
    fetch_selected_assets_with_progress_and_jobs_and_verify(jobs, false, &mut progress)
}

pub fn fetch_selected_assets_with_progress_and_jobs_and_verify(
    jobs: usize,
    verify_existing: bool,
    mut progress: impl FnMut(FetchProgress),
) -> Result<Vec<FetchResult>> {
    let bundles = [
        selected_bundle(ModelKind::Whisper)?,
        selected_bundle(ModelKind::Llm)?,
        selected_bundle(ModelKind::Voice)?,
        selected_bundle(ModelKind::Vocoder)?,
    ];
    fetch_bundles_with_progress(&bundles, jobs, verify_existing, &mut progress)
}

pub fn fetch_bundles_with_progress(
    bundles: &[&ModelBundle],
    jobs: usize,
    verify_existing: bool,
    progress: &mut impl FnMut(FetchProgress),
) -> Result<Vec<FetchResult>> {
    let home = resolve_listenbury_home()?;
    let assets = assets_for_bundles(bundles)?;
    fetch_asset_refs_at_home(&home, &assets, jobs, verify_existing, progress)
}

pub fn fetch_all_assets_with_progress(
    progress: impl FnMut(FetchProgress),
) -> Result<Vec<FetchResult>> {
    fetch_all_assets_with_progress_and_jobs(1, progress)
}

pub fn fetch_all_assets_with_progress_and_jobs(
    jobs: usize,
    mut progress: impl FnMut(FetchProgress),
) -> Result<Vec<FetchResult>> {
    fetch_all_assets_with_progress_and_jobs_and_verify(jobs, false, &mut progress)
}

pub fn fetch_all_assets_with_progress_and_jobs_and_verify(
    jobs: usize,
    verify_existing: bool,
    mut progress: impl FnMut(FetchProgress),
) -> Result<Vec<FetchResult>> {
    let home = resolve_listenbury_home()?;
    let assets = DEFAULT_MODELS.iter().collect::<Vec<_>>();
    fetch_asset_refs_at_home(&home, &assets, jobs, verify_existing, &mut progress)
}

fn assets_for_bundles(bundles: &[&ModelBundle]) -> Result<Vec<&'static ModelAsset>> {
    let mut assets = Vec::new();
    for bundle in bundles {
        for asset in bundle_assets(bundle)? {
            if !assets
                .iter()
                .any(|existing: &&ModelAsset| existing.id == asset.id)
            {
                assets.push(asset);
            }
        }
    }
    Ok(assets)
}

fn normalize_model_name(name: &str) -> String {
    name.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

pub fn default_asset_paths() -> Result<Vec<(&'static ModelAsset, PathBuf)>> {
    let home = resolve_listenbury_home()?;
    Ok(DEFAULT_MODELS
        .iter()
        .map(|asset| (asset, asset_path(&home, asset)))
        .collect())
}

pub fn default_assets_status() -> Result<Vec<ModelStatus>> {
    default_assets_status_with_verification(false)
}

pub fn default_assets_status_with_verification(verify: bool) -> Result<Vec<ModelStatus>> {
    let home = resolve_listenbury_home()?;
    Ok(DEFAULT_MODELS
        .iter()
        .map(|asset| {
            let path = asset_path(&home, asset);
            let integrity = verify_existing_asset(&path, asset, verify)?;
            Ok(ModelStatus {
                asset_id: asset.id,
                path,
                integrity,
                url: asset.url,
                expected_size_bytes: asset.expected_size_bytes,
                sha256: asset.sha256,
                license: asset.license,
                source: asset.source,
            })
        })
        .collect::<Result<Vec<_>>>()?)
}

pub fn fetch_default_assets() -> Result<Vec<FetchResult>> {
    fetch_default_assets_with_progress(|_| {})
}

pub fn fetch_default_assets_with_progress(
    mut progress: impl FnMut(FetchProgress),
) -> Result<Vec<FetchResult>> {
    fetch_selected_assets_with_progress(&mut progress)
}

#[cfg(test)]
fn fetch_assets_at_home(
    home: &Path,
    assets: &[ModelAsset],
    progress: &mut impl FnMut(FetchProgress),
) -> Result<Vec<FetchResult>> {
    let assets = assets.iter().collect::<Vec<_>>();
    fetch_asset_refs_at_home(home, &assets, 1, false, progress)
}

fn fetch_asset_refs_at_home(
    home: &Path,
    assets: &[&ModelAsset],
    jobs: usize,
    verify_existing: bool,
    progress: &mut impl FnMut(FetchProgress),
) -> Result<Vec<FetchResult>> {
    let jobs = jobs.max(1);
    let mut results = Vec::with_capacity(assets.len());
    for chunk_start in (0..assets.len()).step_by(jobs) {
        let chunk_end = (chunk_start + jobs).min(assets.len());
        let chunk = &assets[chunk_start..chunk_end];
        let (sender, receiver) = mpsc::channel();
        std::thread::scope(|scope| {
            for (offset, asset) in chunk.iter().enumerate() {
                let asset = *asset;
                let sender = sender.clone();
                let asset_index = chunk_start + offset;
                let path = asset_path(home, asset);
                scope.spawn(move || {
                    let result = fetch_asset_with_progress_and_verify(
                        home,
                        asset,
                        verify_existing,
                        |downloaded_bytes, total_bytes| {
                            let _ = sender.send(FetchEvent::Progress(FetchProgress {
                                asset_id: asset.id,
                                asset_index,
                                asset_count: assets.len(),
                                path: path.clone(),
                                downloaded_bytes,
                                total_bytes,
                            }));
                        },
                    );
                    let outcome = match result {
                        Ok(downloaded) => FetchResult {
                            asset_id: asset.id,
                            path,
                            outcome: if downloaded {
                                FetchOutcome::Downloaded
                            } else {
                                FetchOutcome::SkippedExisting
                            },
                            error: None,
                        },
                        Err(error) => FetchResult {
                            asset_id: asset.id,
                            path,
                            outcome: FetchOutcome::Failed,
                            error: Some(error.to_string()),
                        },
                    };
                    let _ = sender.send(FetchEvent::Result(asset_index, outcome));
                });
            }
            drop(sender);
            let mut chunk_results = Vec::with_capacity(chunk.len());
            for event in receiver {
                match event {
                    FetchEvent::Progress(asset_progress) => progress(asset_progress),
                    FetchEvent::Result(asset_index, result) => {
                        chunk_results.push((asset_index, result));
                    }
                }
            }
            chunk_results.sort_by_key(|(asset_index, _)| *asset_index);
            results.extend(chunk_results.into_iter().map(|(_, result)| result));
        });
    }
    Ok(results)
}

enum FetchEvent {
    Progress(FetchProgress),
    Result(usize, FetchResult),
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{FetchOutcome, default_bundle_id, fetch_assets_at_home, find_bundle};
    use crate::models::manifest::{ModelAsset, ModelKind};

    fn temp_dir(label: &str) -> PathBuf {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("listenbury-{label}-{}-{ts}", std::process::id()))
    }

    #[test]
    fn fetch_skips_existing_non_empty_files() {
        let home = temp_dir("models-skip-existing");
        let asset_path = home.join("models/test/small.bin");
        fs::create_dir_all(asset_path.parent().expect("parent")).expect("mkdir");
        fs::write(&asset_path, b"already-here").expect("write seed model");

        let assets = [ModelAsset {
            id: "test-asset",
            filename: "small.bin",
            relative_path: "models/test/small.bin",
            url: "http://127.0.0.1:9/unreachable",
            expected_size_bytes: None,
            sha256: None,
            license: None,
            source: None,
        }];

        let results = fetch_assets_at_home(&home, &assets, &mut |_| {})
            .expect("fetch should skip existing file");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].outcome, FetchOutcome::SkippedExisting);
        assert_eq!(results[0].path, asset_path);
        assert!(results[0].error.is_none());
    }

    #[test]
    fn gemma_aliases_resolve_to_llm_bundles() {
        let gemma3 = find_bundle(ModelKind::Llm, "gemma3").expect("gemma3 bundle");
        assert_eq!(gemma3.id, "gemma-3-4b-it-q4-k-m");

        let legacy_gemma3 =
            find_bundle(ModelKind::Llm, "gemma-3-4b-it-q4-0").expect("legacy gemma3 bundle");
        assert_eq!(legacy_gemma3.id, "gemma-3-4b-it-q4-k-m");

        let gemma4 = find_bundle(ModelKind::Llm, "gemma4").expect("gemma4 bundle");
        assert_eq!(gemma4.id, "gemma-4-e4b-it-q4-k-m");
    }

    #[test]
    fn whisper_aliases_resolve_to_whisper_bundles() {
        let base = find_bundle(ModelKind::Whisper, "base").expect("base bundle");
        assert_eq!(base.id, "whisper-base");

        let base_en = find_bundle(ModelKind::Whisper, "base.en").expect("base.en bundle");
        assert_eq!(base_en.id, "whisper-base-en");

        let turbo = find_bundle(ModelKind::Whisper, "turbo").expect("turbo bundle");
        assert_eq!(turbo.id, "whisper-large-v3-turbo");
    }

    #[test]
    fn whisper_defaults_to_multilingual_v3_turbo() {
        assert_eq!(
            default_bundle_id(ModelKind::Whisper),
            "whisper-large-v3-turbo"
        );
    }
}
