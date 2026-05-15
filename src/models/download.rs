use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

use crate::models::manifest::ModelAsset;
use crate::models::paths::asset_path;

pub fn fetch_asset(home: &Path, asset: &ModelAsset) -> Result<bool> {
    fetch_asset_with_progress(home, asset, |_, _| {})
}

pub fn fetch_asset_with_progress(
    home: &Path,
    asset: &ModelAsset,
    mut progress: impl FnMut(u64, Option<u64>),
) -> Result<bool> {
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
    let total_bytes = response.body().content_length().or(asset.expected_size_hint);

    let mut body = response.into_body();
    let mut reader = body.as_reader();
    let mut file = BufWriter::new(
        File::create(&temp_path)
            .with_context(|| format!("failed to create temp file {}", temp_path.display()))?,
    );
    copy_with_progress(&mut reader, &mut file, total_bytes, &mut progress).with_context(|| {
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

fn copy_with_progress(
    reader: &mut impl Read,
    writer: &mut impl Write,
    total_bytes: Option<u64>,
    progress: &mut impl FnMut(u64, Option<u64>),
) -> Result<u64> {
    let mut downloaded = 0;
    let mut buffer = [0; 64 * 1024];
    progress(downloaded, total_bytes);
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        writer.write_all(&buffer[..read])?;
        downloaded += read as u64;
        progress(downloaded, total_bytes);
    }
    Ok(downloaded)
}

fn is_non_empty_file(path: &Path) -> Result<bool> {
    match fs::metadata(path) {
        Ok(metadata) => Ok(metadata.is_file() && metadata.len() > 0),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => {
            Err(error).with_context(|| format!("failed to read file metadata {}", path.display()))
        }
    }
}
