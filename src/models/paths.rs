use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::models::manifest::ModelAsset;

pub fn resolve_listenbury_home() -> Result<PathBuf> {
    if let Some(home) = std::env::var_os("LISTENBURY_HOME") {
        let home = PathBuf::from(home);
        if home.as_os_str().is_empty() {
            anyhow::bail!("LISTENBURY_HOME is set but empty");
        }
        return Ok(home);
    }

    let base = dirs::data_local_dir().context("failed to resolve local data directory")?;
    Ok(base.join("listenbury"))
}

pub fn asset_path(home: &std::path::Path, asset: &ModelAsset) -> PathBuf {
    home.join(asset.relative_path)
}
