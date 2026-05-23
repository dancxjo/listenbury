use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageSource {
    pub lang: String,
    pub profile_files: Vec<String>,
    pub dictionary_files: Vec<String>,
    pub voice_files: Vec<String>,
}

fn relative_path(cache: &Path, path: &Path) -> String {
    path.strip_prefix(cache)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
}

fn visit_files(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            visit_files(&path, files)?;
        } else if path.is_file() {
            files.push(path);
        }
    }
    Ok(())
}

pub fn file_matches_lang(file_name: &str, lang: &str) -> bool {
    let file_name = file_name.to_lowercase();
    let lang = lang.to_lowercase();
    file_name == lang || file_name.starts_with(&format!("{lang}-"))
}

pub fn discover_profile_files(cache: &Path, lang: &str) -> Result<Vec<PathBuf>> {
    let mut all = Vec::new();
    visit_files(&cache.join("espeak-ng-data/lang"), &mut all)?;
    let mut files: Vec<PathBuf> = all
        .into_iter()
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| file_matches_lang(name, lang))
        })
        .collect();
    files.sort();
    Ok(files)
}

pub fn discover_dictionary_files(cache: &Path, lang: &str) -> Vec<PathBuf> {
    ["rules", "list", "extra", "listx", "emoji"]
        .into_iter()
        .map(|suffix| cache.join(format!("dictsource/{lang}_{suffix}")))
        .filter(|path| path.exists() && path.is_file())
        .collect()
}

pub fn language_tags_in_profile(content: &str) -> Vec<String> {
    let mut tags = Vec::new();
    for raw in content.lines() {
        let line = raw.split_once("//").map_or(raw, |(head, _)| head).trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        if parts.next() == Some("language") {
            if let Some(tag) = parts.next() {
                tags.push(tag.to_lowercase());
            }
        }
    }
    tags
}

pub fn discover_voice_files(cache: &Path, lang: &str) -> Result<Vec<PathBuf>> {
    let mut all = Vec::new();
    visit_files(&cache.join("espeak-ng-data/voices"), &mut all)?;
    let mut files = Vec::new();
    for path in all {
        let content = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let tags = language_tags_in_profile(&content);
        if tags.iter().any(|tag| file_matches_lang(tag, lang)) {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

pub fn discover_languages(cache: &Path) -> Result<Vec<LanguageSource>> {
    let mut by_lang: BTreeMap<String, LanguageSource> = BTreeMap::new();

    let mut profile_files = Vec::new();
    visit_files(&cache.join("espeak-ng-data/lang"), &mut profile_files)?;
    for path in profile_files {
        let Some(id) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let entry = by_lang.entry(id.to_string()).or_insert(LanguageSource {
            lang: id.to_string(),
            profile_files: Vec::new(),
            dictionary_files: Vec::new(),
            voice_files: Vec::new(),
        });
        entry.profile_files.push(relative_path(cache, &path));
    }

    let dict_dir = cache.join("dictsource");
    if dict_dir.exists() {
        for entry in fs::read_dir(&dict_dir)
            .with_context(|| format!("failed to read {}", dict_dir.display()))?
        {
            let path = entry?.path();
            if !path.is_file() {
                continue;
            }
            let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let Some((lang, suffix)) = file_name.rsplit_once('_') else {
                continue;
            };
            if !matches!(suffix, "rules" | "list" | "extra" | "listx" | "emoji") {
                continue;
            }
            let entry = by_lang.entry(lang.to_string()).or_insert(LanguageSource {
                lang: lang.to_string(),
                profile_files: Vec::new(),
                dictionary_files: Vec::new(),
                voice_files: Vec::new(),
            });
            entry.dictionary_files.push(relative_path(cache, &path));
        }
    }

    let langs: BTreeSet<String> = by_lang.keys().cloned().collect();
    let mut voice_files = Vec::new();
    visit_files(&cache.join("espeak-ng-data/voices"), &mut voice_files)?;
    for path in voice_files {
        let content = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let rel = relative_path(cache, &path);
        for tag in language_tags_in_profile(&content) {
            for lang in &langs {
                if file_matches_lang(&tag, lang) {
                    if let Some(entry) = by_lang.get_mut(lang) {
                        entry.voice_files.push(rel.clone());
                    }
                }
            }
        }
    }

    let mut languages: Vec<LanguageSource> = by_lang.into_values().collect();
    for language in &mut languages {
        language.profile_files.sort();
        language.profile_files.dedup();
        language.dictionary_files.sort();
        language.dictionary_files.dedup();
        language.voice_files.sort();
        language.voice_files.dedup();
    }
    languages.sort_by(|a, b| a.lang.cmp(&b.lang));
    Ok(languages)
}
