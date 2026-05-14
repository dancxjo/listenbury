use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

use crate::models::manifest::ModelAsset;
use crate::models::paths::asset_path;

pub fn fetch_asset(home: &Path, asset: &ModelAsset) -> Result<bool> {
    let target_path = asset_path(home, asset);
    if is_non_empty_file(&target_path)? {
        return Ok(false);
    }

    let parent = target_path
        .parent()
        .context("asset path had no parent directory")?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create model directory {}", parent.display()))?;

    let timestamp_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before UNIX_EPOCH")?
        .as_nanos();
    let temp_path = parent.join(format!(
        ".{}.{}.{}.part",
        asset.filename,
        std::process::id(),
        timestamp_nanos
    ));

    let response = ureq::get(asset.url)
        .call()
        .with_context(|| format!("failed to download {}", asset.url))?;

    let mut body = response.into_body();
    let mut reader = body.as_reader();
    let mut file = BufWriter::new(
        File::create(&temp_path)
            .with_context(|| format!("failed to create temp file {}", temp_path.display()))?,
    );
    std::io::copy(&mut reader, &mut file).with_context(|| {
        format!(
            "failed to write download for {} to {}",
            asset.id,
            temp_path.display()
        )
    })?;
    file.flush()
        .with_context(|| format!("failed to flush {}", temp_path.display()))?;
    drop(file);

    fs::rename(&temp_path, &target_path).with_context(|| {
        format!(
            "failed to move completed download {} to {}",
            temp_path.display(),
            target_path.display()
        )
    })?;
    Ok(true)
}

fn is_non_empty_file(path: &Path) -> Result<bool> {
    if !path.is_file() {
        return Ok(false);
    }
    let metadata = fs::metadata(path)
        .with_context(|| format!("failed to read file metadata {}", path.display()))?;
    Ok(metadata.len() > 0)
}
