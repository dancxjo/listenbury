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
    discovery::{discover_dictionary_files, discover_profile_files, discover_voice_files},
    provenance::{current_revision, ensure_cache_exists, load_metadata},
    rules::parse_rules_content,
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

pub fn inventory(lang: &str, json_out: Option<&Path>) -> Result<()> {
    let cache = ensure_cache_exists()?;
    let revision = current_revision(&cache)?;
    let source_license = load_metadata()?
        .map(|metadata| metadata.source_license)
        .unwrap_or_else(|| "GPL-3.0-or-later".to_string());

    let mut files = Vec::new();
    for dictionary_file in discover_dictionary_files(&cache, lang) {
        push_file(&cache, dictionary_file, &mut files)?;
    }
    push_file(&cache, cache.join("phsource/phonemes"), &mut files)?;

    for profile in discover_profile_files(&cache, lang)? {
        push_file(&cache, profile, &mut files)?;
    }
    for voice in discover_voice_files(&cache, lang)? {
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
        if file.path.contains("espeak-ng-data/lang/") {
            relevant_sections.insert("profiles".to_string());
        }
        if file.path.contains("phsource/phonemes") {
            relevant_sections.insert("phoneme-table".to_string());
        }
        if file.path.contains("voices/") {
            relevant_sections.insert("voices".to_string());
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
