use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use listenbury::models::manifest::ModelKind;
#[cfg(feature = "model-download")]
use listenbury::models::{
    bundle_assets, bundle_primary_path, find_asset,
    manifest::ModelBundle,
    paths::{asset_path, resolve_listenbury_home},
    selected_bundle,
};

#[cfg(feature = "llm-llama-cpp")]
pub(crate) fn resolve_llm_model(explicit: Option<PathBuf>) -> Result<PathBuf> {
    resolve_model_path(
        explicit,
        "LISTENBURY_LLM_MODEL",
        "llama.cpp model",
        "--llm-model",
        Some("llama-3-2-3b-instruct-q4-k-m"),
        Some(ModelKind::Llm),
        |path| path.extension().is_some_and(|ext| ext == "gguf"),
    )
}

#[cfg(feature = "asr-whisper")]
pub(crate) fn resolve_whisper_model(explicit: Option<PathBuf>) -> Result<PathBuf> {
    resolve_model_path(
        explicit,
        "LISTENBURY_WHISPER_MODEL",
        "Whisper model",
        "--whisper-model",
        Some("whisper-tiny-en"),
        Some(ModelKind::Whisper),
        |path| {
            path.extension().is_some_and(|ext| ext == "bin")
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.contains("ggml"))
        },
    )
}

#[cfg(feature = "tts-piper")]
pub(crate) fn resolve_piper_voice(explicit: Option<PathBuf>) -> Result<PathBuf> {
    resolve_model_path(
        explicit,
        "LISTENBURY_PIPER_VOICE",
        "Piper voice",
        "--piper-voice",
        Some("piper-ryan-medium"),
        Some(ModelKind::Voice),
        |path| path.extension().is_some_and(|ext| ext == "onnx"),
    )
}

#[cfg(any(
    feature = "asr-whisper",
    feature = "llm-llama-cpp",
    feature = "tts-piper"
))]
fn resolve_model_path(
    explicit: Option<PathBuf>,
    env_var: &str,
    label: &str,
    flag: &str,
    default_asset_id: Option<&str>,
    selected_kind: Option<ModelKind>,
    matches: impl Fn(&Path) -> bool,
) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path);
    }

    if let Some(path) = std::env::var_os(env_var) {
        return Ok(PathBuf::from(path));
    }

    #[cfg(feature = "model-download")]
    if let Some(kind) = selected_kind {
        let bundle = selected_bundle(kind)?;
        ensure_bundle_available(bundle)?;
        return bundle_primary_path(bundle);
    }

    #[cfg(feature = "model-download")]
    if let Some(asset_id) = default_asset_id {
        let path = default_asset_path(asset_id)?;
        if is_non_empty_file(&path) {
            return Ok(path);
        }
    }

    if let Some(path) = discover_model_file(&matches)? {
        return Ok(path);
    }

    let fetch_hint = if default_asset_id.is_some() && cfg!(feature = "model-download") {
        ", or run `cargo run -- models fetch`"
    } else {
        ""
    };
    anyhow::bail!("could not discover {label}; set {env_var}, pass {flag}{fetch_hint}")
}

#[cfg(feature = "model-download")]
fn default_asset_path(asset_id: &str) -> Result<PathBuf> {
    let Some(asset) = find_asset(asset_id) else {
        anyhow::bail!("default model asset `{asset_id}` is not registered");
    };
    let home = resolve_listenbury_home()?;
    Ok(asset_path(&home, asset))
}

#[cfg(feature = "model-download")]
fn ensure_bundle_available(bundle: &ModelBundle) -> Result<()> {
    let home = resolve_listenbury_home()?;
    let assets = bundle_assets(bundle)?;
    let missing: Vec<_> = assets
        .iter()
        .filter(|asset| !is_non_empty_file(&asset_path(&home, asset)))
        .copied()
        .collect();
    if missing.is_empty() {
        return Ok(());
    }

    eprintln!(
        "{} model `{}` is missing locally; downloading {} asset(s). This can take a while...",
        listenbury::models::model_kind_label(bundle.kind),
        bundle.display_name,
        missing.len()
    );
    for asset in missing {
        eprintln!(
            "downloading {} -> {}",
            asset.id,
            asset_path(&home, asset).display()
        );
        listenbury::models::download::fetch_asset(&home, asset)?;
    }
    Ok(())
}

#[cfg(feature = "model-download")]
fn is_non_empty_file(path: &Path) -> bool {
    path.metadata().map(|meta| meta.len() > 0).unwrap_or(false)
}

fn discover_model_file(matches: &impl Fn(&Path) -> bool) -> Result<Option<PathBuf>> {
    let models_dir = Path::new("models");
    if !models_dir.exists() {
        return Ok(None);
    }

    let mut stack = vec![models_dir.to_path_buf()];
    let mut found = Vec::new();

    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)
            .with_context(|| format!("failed to read model directory {}", dir.display()))?
        {
            let entry = entry
                .with_context(|| format!("failed to inspect model directory {}", dir.display()))?;
            let path = entry.path();
            let file_type = entry
                .file_type()
                .with_context(|| format!("failed to inspect {}", path.display()))?;
            if file_type.is_dir() {
                stack.push(path);
            } else if file_type.is_file() && matches(&path) {
                found.push(path);
            }
        }
    }

    found.sort();
    Ok(found.into_iter().next())
}
