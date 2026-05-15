pub mod download;
pub mod manifest;
pub mod paths;

use std::path::PathBuf;

use anyhow::Result;

use crate::models::download::fetch_asset_with_progress;
use crate::models::manifest::{DEFAULT_MODELS, ModelAsset};
use crate::models::paths::{asset_path, resolve_listenbury_home};

#[derive(Debug, Clone)]
pub struct ModelStatus {
    pub asset_id: &'static str,
    pub path: PathBuf,
    pub present: bool,
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

pub fn default_asset_paths() -> Result<Vec<(&'static ModelAsset, PathBuf)>> {
    let home = resolve_listenbury_home()?;
    Ok(DEFAULT_MODELS
        .iter()
        .map(|asset| (asset, asset_path(&home, asset)))
        .collect())
}

pub fn default_assets_status() -> Result<Vec<ModelStatus>> {
    let home = resolve_listenbury_home()?;
    Ok(DEFAULT_MODELS
        .iter()
        .map(|asset| {
            let path = asset_path(&home, asset);
            let present = path.metadata().map(|meta| meta.len() > 0).unwrap_or(false);
            ModelStatus {
                asset_id: asset.id,
                path,
                present,
            }
        })
        .collect())
}

pub fn fetch_default_assets() -> Result<Vec<FetchResult>> {
    fetch_default_assets_with_progress(|_| {})
}

pub fn fetch_default_assets_with_progress(
    mut progress: impl FnMut(FetchProgress),
) -> Result<Vec<FetchResult>> {
    let home = resolve_listenbury_home()?;
    fetch_assets_at_home(&home, DEFAULT_MODELS, &mut progress)
}

fn fetch_assets_at_home(
    home: &std::path::Path,
    assets: &[ModelAsset],
    progress: &mut impl FnMut(FetchProgress),
) -> Result<Vec<FetchResult>> {
    let mut results = Vec::with_capacity(assets.len());
    for (asset_index, asset) in assets.iter().enumerate() {
        let path = asset_path(home, asset);
        match fetch_asset_with_progress(home, asset, |downloaded_bytes, total_bytes| {
            progress(FetchProgress {
                asset_id: asset.id,
                asset_index,
                asset_count: assets.len(),
                path: path.clone(),
                downloaded_bytes,
                total_bytes,
            });
        }) {
            Ok(downloaded) => results.push(FetchResult {
                asset_id: asset.id,
                path,
                outcome: if downloaded {
                    FetchOutcome::Downloaded
                } else {
                    FetchOutcome::SkippedExisting
                },
                error: None,
            }),
            Err(error) => results.push(FetchResult {
                asset_id: asset.id,
                path,
                outcome: FetchOutcome::Failed,
                error: Some(error.to_string()),
            }),
        }
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{FetchOutcome, fetch_assets_at_home};
    use crate::models::manifest::ModelAsset;

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
            expected_size_hint: None,
        }];

        let results = fetch_assets_at_home(&home, &assets, &mut |_| {})
            .expect("fetch should skip existing file");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].outcome, FetchOutcome::SkippedExisting);
        assert_eq!(results[0].path, asset_path);
        assert!(results[0].error.is_none());
    }
}
