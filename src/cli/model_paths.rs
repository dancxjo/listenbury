use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

#[cfg(feature = "model-download")]
use crate::cli::download_progress::DownloadProgress;
#[cfg(feature = "model-download")]
use listenbury::models::{
    FetchProgress, bundle_assets, bundle_primary_path, find_asset,
    manifest::{ModelBundle, ModelKind},
    paths::{asset_path, resolve_listenbury_home},
    selected_bundle,
};

#[cfg(not(feature = "model-download"))]
#[derive(Debug, Clone, Copy)]
enum ModelKind {
    #[cfg(feature = "llm-llama-cpp")]
    Llm,
    #[cfg(feature = "tts-piper")]
    Voice,
    #[cfg(feature = "asr-whisper")]
    Whisper,
}

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

#[cfg(feature = "llm-llama-cpp")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LlmRuntimePlacement {
    pub(crate) gpu_layers: Option<u32>,
    pub(crate) cpu_only: bool,
}

#[cfg(feature = "llm-llama-cpp")]
pub(crate) fn llm_runtime_placement(
    model_path: &Path,
    explicit_gpu_layers: Option<u32>,
    default_gpu_layers: Option<u32>,
) -> Result<LlmRuntimePlacement> {
    if let Some(gpu_layers) = explicit_gpu_layers {
        return Ok(LlmRuntimePlacement {
            gpu_layers: Some(gpu_layers),
            cpu_only: gpu_layers == 0,
        });
    }

    if default_gpu_layers.is_none() && llm_model_needs_cpu_runtime(model_path) {
        return Ok(LlmRuntimePlacement {
            gpu_layers: Some(0),
            cpu_only: true,
        });
    }

    Ok(LlmRuntimePlacement {
        gpu_layers: default_gpu_layers,
        cpu_only: default_gpu_layers == Some(0),
    })
}

#[cfg(feature = "llm-llama-cpp")]
fn llm_model_needs_cpu_runtime(model_path: &Path) -> bool {
    llm_model_filename(model_path).contains("gpt-oss")
}

#[cfg(feature = "llm-llama-cpp")]
fn llm_model_filename(model_path: &Path) -> String {
    let filename = model_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    filename
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "llm-llama-cpp")]
    #[test]
    fn gpt_oss_defaults_to_cpu_without_gpu_default() {
        let placement =
            llm_runtime_placement(Path::new("gpt-oss-20b-mxfp4.gguf"), None, None).unwrap();

        assert_eq!(placement.gpu_layers, Some(0));
        assert!(placement.cpu_only);
    }

    #[cfg(feature = "llm-llama-cpp")]
    #[test]
    fn gpt_oss_uses_cuda_default_when_provided() {
        let placement =
            llm_runtime_placement(Path::new("gpt-oss-20b-mxfp4.gguf"), None, Some(999)).unwrap();

        assert_eq!(placement.gpu_layers, Some(999));
        assert!(!placement.cpu_only);
    }
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

    #[cfg(not(feature = "model-download"))]
    let _ = selected_kind;

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
    let mut progress = DownloadProgress::new(format!(
        "Downloading {} model `{}`...",
        listenbury::models::model_kind_label(bundle.kind),
        bundle.display_name
    ))?;
    let asset_count = missing.len();
    for (asset_index, asset) in missing.into_iter().enumerate() {
        let path = asset_path(&home, asset);
        listenbury::models::download::fetch_asset_with_progress(
            &home,
            asset,
            |downloaded_bytes, total_bytes| {
                progress.update(FetchProgress {
                    asset_id: asset.id,
                    asset_index,
                    asset_count,
                    path: path.clone(),
                    downloaded_bytes,
                    total_bytes,
                });
            },
        )?;
    }
    progress.finish_and_clear();
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
