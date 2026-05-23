use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tempfile::tempdir;

use super::{
    RulesMode,
    dictionary::convert_list,
    profile::convert_profiles,
    provenance::{
        CONVERTER_VERSION, current_revision, ensure_cache_exists, load_metadata, repo_root,
    },
    rules::convert_rules,
};

const DEFAULT_REGEN_BASE: &str = "data/language-varieties";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeneratedManifest {
    lang: String,
    source: String,
    source_revision: String,
    source_license: String,
    converter_version: String,
    outputs: Vec<String>,
}

fn hash_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn collect_files(root: &Path, dir: &Path, acc: &mut BTreeMap<String, String>) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(root, &path, acc)?;
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();
        acc.insert(rel, hash_file(&path)?);
    }
    Ok(())
}

pub fn convert_all(lang: &str, out: &Path) -> Result<()> {
    let cache = ensure_cache_exists()?;
    fs::create_dir_all(out).with_context(|| format!("failed to create {}", out.display()))?;

    let profiles_dir = out.join("profiles");
    let dictionary_dir = out.join("dictionary");
    let rules_dir = out.join("rules-inventory");

    convert_profiles(lang, &profiles_dir)?;
    convert_list(lang, &dictionary_dir)?;
    convert_rules(lang, &rules_dir, RulesMode::Inventory)?;
    convert_rules(lang, &rules_dir, RulesMode::NativeSubset)?;

    let source_revision = current_revision(&cache)?;
    let source_license = load_metadata()?
        .map(|metadata| metadata.source_license)
        .unwrap_or_else(|| "GPL-3.0-or-later".to_string());

    let mut outputs = Vec::new();
    for entry in fs::read_dir(out).with_context(|| format!("failed to read {}", out.display()))? {
        let entry = entry?;
        outputs.push(entry.file_name().to_string_lossy().to_string());
    }
    outputs.sort();

    let manifest = GeneratedManifest {
        lang: lang.to_string(),
        source: "espeak-ng".to_string(),
        source_revision,
        source_license,
        converter_version: CONVERTER_VERSION.to_string(),
        outputs,
    };

    let manifest_path = out.join("manifest.toml");
    fs::write(
        &manifest_path,
        toml::to_string_pretty(&manifest).context("failed to encode manifest TOML")?,
    )
    .with_context(|| format!("failed to write {}", manifest_path.display()))?;

    println!("Wrote eSpeak-ng generated outputs to {}", out.display());
    Ok(())
}

fn default_regen_out(lang: &str) -> Result<PathBuf> {
    Ok(repo_root()?
        .join(DEFAULT_REGEN_BASE)
        .join(lang)
        .join("generated")
        .join("espeak-ng"))
}

pub fn regen(lang: &str) -> Result<()> {
    let out = default_regen_out(lang)?;
    convert_all(lang, &out)
}

pub fn diff(lang: &str, out: Option<&Path>) -> Result<()> {
    let target_out = if let Some(out) = out {
        out.to_path_buf()
    } else {
        default_regen_out(lang)?
    };

    let tmp = tempdir().context("failed to create temporary directory")?;
    convert_all(lang, tmp.path())?;

    if !target_out.exists() {
        bail!(
            "target generated directory {} does not exist (run `cargo xtask espeak-ng regen --lang {lang}`)",
            target_out.display()
        );
    }

    let mut current = BTreeMap::new();
    let mut expected = BTreeMap::new();
    collect_files(&target_out, &target_out, &mut current)?;
    collect_files(tmp.path(), tmp.path(), &mut expected)?;

    let mut drift = Vec::new();
    for (path, hash) in &expected {
        match current.get(path) {
            Some(existing) if existing == hash => {}
            Some(_) => drift.push(format!("modified: {path}")),
            None => drift.push(format!("missing: {path}")),
        }
    }
    for path in current.keys() {
        if !expected.contains_key(path) {
            drift.push(format!("extra: {path}"));
        }
    }

    if drift.is_empty() {
        println!("No drift detected for {}", target_out.display());
        return Ok(());
    }

    println!("Drift detected for {}", target_out.display());
    for item in &drift {
        println!("  - {item}");
    }
    bail!("generated output differs from regenerated eSpeak-ng conversion")
}

#[cfg(test)]
mod tests {
    use super::collect_files;
    use anyhow::Result;
    use std::{collections::BTreeMap, fs};
    use tempfile::tempdir;

    #[test]
    fn collect_files_keeps_paths_relative_to_root() -> Result<()> {
        let tmp = tempdir()?;
        let root = tmp.path();
        fs::create_dir_all(root.join("profiles"))?;
        fs::create_dir_all(root.join("dictionary"))?;
        fs::write(root.join("profiles/en-US.toml"), "profile")?;
        fs::write(root.join("dictionary/dictionary.toml"), "dictionary")?;

        let mut files = BTreeMap::new();
        collect_files(root, root, &mut files)?;

        assert!(files.contains_key("profiles/en-US.toml"));
        assert!(files.contains_key("dictionary/dictionary.toml"));
        Ok(())
    }
}
