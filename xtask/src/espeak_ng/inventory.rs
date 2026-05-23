use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    RulesMode,
    dictionary::convert_list,
    profile::convert_profiles,
    provenance::{current_revision, ensure_cache_exists, load_metadata},
    rules::{convert_rules, parse_rules_content},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InventoryFile {
    path: String,
    sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InventoryReport {
    lang: String,
    upstream_revision: String,
    source_license: String,
    files: Vec<InventoryFile>,
    relevant_sections: Vec<String>,
    supported_categories: Vec<String>,
    unsupported_categories: Vec<String>,
}

fn checksum(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let hash = Sha256::digest(&bytes);
    Ok(format!("{hash:x}"))
}

fn push_file(cache: &Path, path: PathBuf, files: &mut Vec<InventoryFile>) -> Result<()> {
    if !path.exists() || !path.is_file() {
        return Ok(());
    }
    let rel = path
        .strip_prefix(cache)
        .unwrap_or(&path)
        .to_string_lossy()
        .to_string();
    files.push(InventoryFile {
        path: rel,
        sha256: checksum(&path)?,
    });
    Ok(())
}

fn list_prefixed_files(cache: &Path, base: &str, prefix: &str) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let dir = cache.join(base);
    if !dir.exists() {
        return Ok(files);
    }
    for entry in fs::read_dir(&dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with(prefix))
        {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

pub fn inventory(lang: &str, json_out: Option<&Path>) -> Result<()> {
    let cache = ensure_cache_exists()?;
    let revision = current_revision(&cache)?;
    let source_license = load_metadata()?
        .map(|metadata| metadata.source_license)
        .unwrap_or_else(|| "GPL-3.0-or-later".to_string());

    let mut files = Vec::new();
    for direct in [
        format!("dictsource/{lang}_rules"),
        format!("dictsource/{lang}_list"),
        format!("dictsource/{lang}_extra"),
        "phsource/phonemes".to_string(),
    ] {
        push_file(&cache, cache.join(direct), &mut files)?;
    }

    for profile in list_prefixed_files(&cache, "espeak-ng-data/lang/gmw", lang)? {
        push_file(&cache, profile, &mut files)?;
    }
    for voice in list_prefixed_files(&cache, "espeak-ng-data/voices/mb", "mb-en")? {
        push_file(&cache, voice, &mut files)?;
    }
    for voice in list_prefixed_files(&cache, "espeak-ng-data/voices/mb", "mb-us")? {
        push_file(&cache, voice, &mut files)?;
    }

    files.sort_by(|a, b| a.path.cmp(&b.path));

    let mut relevant_sections = BTreeSet::new();
    for file in &files {
        if file.path.contains("_rules") {
            relevant_sections.insert("rules".to_string());
        }
        if file.path.contains("_list") || file.path.contains("_extra") {
            relevant_sections.insert("dictionary".to_string());
        }
        if file.path.contains("lang/gmw") {
            relevant_sections.insert("profiles".to_string());
        }
        if file.path.contains("phsource/phonemes") {
            relevant_sections.insert("phoneme-table".to_string());
        }
        if file.path.contains("voices/mb") {
            relevant_sections.insert("mbrola-voices".to_string());
        }
    }

    let mut supported_categories = BTreeSet::new();
    supported_categories
        .insert("profile-fields:name/language/phonemes/dictrules/replace/stress".to_string());
    supported_categories
        .insert("dictionary:explicit/stress/multiword/symbol/number/alt".to_string());
    supported_categories
        .insert("rules-inventory:.Lnn/.replace/.group/conditions/operators".to_string());

    let mut unsupported_categories = BTreeSet::new();
    let rules_path = cache.join(format!("dictsource/{lang}_rules"));
    if rules_path.exists() {
        let rules_content = fs::read_to_string(&rules_path)
            .with_context(|| format!("failed to read {}", rules_path.display()))?;
        let rules = parse_rules_content(
            lang,
            &format!("dictsource/{lang}_rules"),
            &rules_content,
            &revision,
            &source_license,
            RulesMode::Inventory,
        );
        if !rules.unsupported.is_empty() {
            unsupported_categories.insert(format!("rules-constructs:{}", rules.unsupported.len()));
        }
    }

    let report = InventoryReport {
        lang: lang.to_string(),
        upstream_revision: revision,
        source_license,
        files,
        relevant_sections: relevant_sections.into_iter().collect(),
        supported_categories: supported_categories.into_iter().collect(),
        unsupported_categories: unsupported_categories.into_iter().collect(),
    };

    println!("eSpeak-ng inventory for lang={lang}");
    println!("  revision: {}", report.upstream_revision);
    println!("  files: {}", report.files.len());
    for file in &report.files {
        println!("  - {} ({})", file.path, file.sha256);
    }

    if let Some(path) = json_out {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        fs::write(
            path,
            serde_json::to_string_pretty(&report).context("failed to encode inventory json")?,
        )
        .with_context(|| format!("failed to write {}", path.display()))?;
        println!("Wrote inventory JSON to {}", path.display());
    }

    Ok(())
}

#[allow(dead_code)]
fn _compile_use_for_converter_signatures(lang: &str, path: &Path) {
    let _ = convert_profiles(lang, path);
    let _ = convert_list(lang, path);
    let _ = convert_rules(lang, path, RulesMode::Inventory);
}
