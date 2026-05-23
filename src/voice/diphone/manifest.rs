//! Manifest for a local cache-backed diphone voice.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub const DIPHONE_VOICE_MANIFEST_FILE: &str = "listenbury-diphone-voice.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiphoneVoiceManifest {
    pub format_version: u32,
    pub name: String,
    pub model: PathBuf,
    pub config: PathBuf,
    pub cache_dir: PathBuf,
    pub inventory: String,
    pub sample_rate_hz: u32,
}

impl DiphoneVoiceManifest {
    pub fn new(
        name: impl Into<String>,
        model: impl Into<PathBuf>,
        config: impl Into<PathBuf>,
        cache_dir: impl Into<PathBuf>,
        inventory: impl Into<String>,
        sample_rate_hz: u32,
    ) -> Self {
        Self {
            format_version: 1,
            name: name.into(),
            model: model.into(),
            config: config.into(),
            cache_dir: cache_dir.into(),
            inventory: inventory.into(),
            sample_rate_hz,
        }
    }

    pub fn manifest_path(path: impl AsRef<Path>) -> PathBuf {
        let path = path.as_ref();
        if path.is_dir() {
            path.join(DIPHONE_VOICE_MANIFEST_FILE)
        } else {
            path.to_path_buf()
        }
    }

    pub fn load_if_present(path: impl AsRef<Path>) -> Result<Option<Self>> {
        let manifest_path = Self::manifest_path(path);
        if !manifest_path.is_file() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&manifest_path).with_context(|| {
            format!(
                "failed to read diphone voice manifest {}",
                manifest_path.display()
            )
        })?;
        let manifest = serde_json::from_str(&text).with_context(|| {
            format!(
                "failed to parse diphone voice manifest {}",
                manifest_path.display()
            )
        })?;
        Ok(Some(manifest))
    }

    pub fn write_pretty(&self, path: impl AsRef<Path>) -> Result<()> {
        let manifest_path = Self::manifest_path(path);
        if let Some(parent) = manifest_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create diphone voice directory {}",
                    parent.display()
                )
            })?;
        }
        let json = serde_json::to_string_pretty(self)
            .context("failed to serialize diphone voice manifest")?;
        std::fs::write(&manifest_path, json.as_bytes()).with_context(|| {
            format!(
                "failed to write diphone voice manifest {}",
                manifest_path.display()
            )
        })
    }
}
