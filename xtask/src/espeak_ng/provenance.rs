use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

pub const UPSTREAM_URL: &str = "https://github.com/espeak-ng/espeak-ng.git";
pub const CACHE_DIR: &str = ".target/espeak-ng-src";
pub const META_FILE: &str = ".listenbury-espeak-meta.json";
pub const CONVERTER_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpstreamMetadata {
    pub upstream_url: String,
    pub revision: String,
    pub fetched_unix_secs: u64,
    pub source_license: String,
}

pub fn repo_root() -> Result<PathBuf> {
    std::env::current_dir().context("failed to determine current directory")
}

pub fn cache_path() -> Result<PathBuf> {
    Ok(repo_root()?.join(CACHE_DIR))
}

pub fn metadata_path() -> Result<PathBuf> {
    Ok(cache_path()?.join(META_FILE))
}

pub fn ensure_cache_exists() -> Result<PathBuf> {
    let cache = cache_path()?;
    if !cache.exists() {
        bail!(
            "missing eSpeak-ng cache at {} (run `cargo xtask espeak-ng fetch` first)",
            cache.display()
        );
    }
    Ok(cache)
}

pub fn run_git(cache: &Path, args: &[&str]) -> Result<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(cache)
        .args(args)
        .output()
        .with_context(|| format!("failed to run git {:?}", args))?;
    if !out.status.success() {
        bail!(
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

pub fn current_revision(cache: &Path) -> Result<String> {
    run_git(cache, &["rev-parse", "HEAD"])
}

pub fn read_source_license(cache: &Path) -> String {
    for candidate in ["COPYING", "LICENSE", "README.md"] {
        let path = cache.join(candidate);
        if let Ok(content) = fs::read_to_string(path)
            && let Some(first) = content.lines().find(|line| !line.trim().is_empty())
        {
            return first.trim().to_string();
        }
    }
    "GPL-3.0-or-later".to_string()
}

pub fn write_metadata(cache: &Path, revision: String) -> Result<UpstreamMetadata> {
    let fetched_unix_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock before UNIX_EPOCH")?
        .as_secs();
    let metadata = UpstreamMetadata {
        upstream_url: UPSTREAM_URL.to_string(),
        revision,
        fetched_unix_secs,
        source_license: read_source_license(cache),
    };
    let path = metadata_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(
        &path,
        serde_json::to_string_pretty(&metadata).context("failed to serialize metadata")?,
    )
    .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(metadata)
}

pub fn load_metadata() -> Result<Option<UpstreamMetadata>> {
    let path = metadata_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let raw =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let metadata = serde_json::from_str(&raw).context("failed to parse metadata json")?;
    Ok(Some(metadata))
}
