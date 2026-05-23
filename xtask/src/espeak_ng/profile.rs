use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::provenance::{CONVERTER_VERSION, current_revision, ensure_cache_exists, load_metadata};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageTag {
    pub tag: String,
    pub priority: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Replacement {
    pub flags: i32,
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsupportedRecord {
    pub line: usize,
    pub directive: String,
    pub content: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StressProfile {
    pub length: Vec<i32>,
    pub amplitude: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvertedProfile {
    pub id: String,
    pub source: String,
    pub source_file: String,
    pub source_revision: String,
    pub source_license: String,
    pub converter_version: String,
    pub name: Option<String>,
    pub language_tags: Vec<LanguageTag>,
    pub phonemes: Option<String>,
    pub dictrules: Vec<i32>,
    pub stress: StressProfile,
    pub replacements: Vec<Replacement>,
    pub unsupported: Vec<UnsupportedRecord>,
}

fn strip_comment(line: &str) -> &str {
    line.split_once("//").map_or(line, |(head, _)| head)
}

fn parse_i32_values(values: &[&str]) -> Vec<i32> {
    values
        .iter()
        .filter_map(|value| value.parse::<i32>().ok())
        .collect()
}

pub fn parse_profile_content(
    id: &str,
    source_file: &str,
    content: &str,
    source_revision: &str,
    source_license: &str,
) -> ConvertedProfile {
    let mut name = None;
    let mut language_tags = Vec::new();
    let mut phonemes = None;
    let mut dictrules = Vec::new();
    let mut stress = StressProfile::default();
    let mut replacements = Vec::new();
    let mut unsupported = Vec::new();

    for (index, raw) in content.lines().enumerate() {
        let line_no = index + 1;
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(directive) = parts.next() else {
            continue;
        };
        let rest: Vec<&str> = parts.collect();
        match directive {
            "name" => {
                if !rest.is_empty() {
                    name = Some(rest.join(" "));
                }
            }
            "language" => {
                if let Some(tag) = rest.first() {
                    let priority = rest.get(1).and_then(|value| value.parse::<i32>().ok());
                    language_tags.push(LanguageTag {
                        tag: tag.to_lowercase(),
                        priority,
                    });
                }
            }
            "phonemes" => {
                if let Some(value) = rest.first() {
                    phonemes = Some((*value).to_string());
                }
            }
            "dictrules" => dictrules.extend(parse_i32_values(&rest)),
            "stressLength" | "stresslength" => stress.length = parse_i32_values(&rest),
            "stressAmp" | "stressamp" => stress.amplitude = parse_i32_values(&rest),
            "replace" => {
                if rest.len() >= 3 {
                    if let Ok(flags) = rest[0].parse::<i32>() {
                        replacements.push(Replacement {
                            flags,
                            from: rest[1].to_string(),
                            to: rest[2].to_string(),
                        });
                    }
                } else {
                    unsupported.push(UnsupportedRecord {
                        line: line_no,
                        directive: directive.to_string(),
                        content: line.to_string(),
                    });
                }
            }
            _ => unsupported.push(UnsupportedRecord {
                line: line_no,
                directive: directive.to_string(),
                content: line.to_string(),
            }),
        }
    }

    ConvertedProfile {
        id: id.to_string(),
        source: "espeak-ng".to_string(),
        source_file: source_file.to_string(),
        source_revision: source_revision.to_string(),
        source_license: source_license.to_string(),
        converter_version: CONVERTER_VERSION.to_string(),
        name,
        language_tags,
        phonemes,
        dictrules,
        stress,
        replacements,
        unsupported,
    }
}

fn discover_profile_files(cache: &Path, lang: &str) -> Result<Vec<PathBuf>> {
    let base = cache.join("espeak-ng-data/lang/gmw");
    let mut files = Vec::new();
    if !base.exists() {
        return Ok(files);
    }

    for entry in
        fs::read_dir(&base).with_context(|| format!("failed to read {}", base.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name == lang || file_name.starts_with(&format!("{lang}-")) {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

pub fn convert_profiles(lang: &str, out: &Path) -> Result<()> {
    let cache = ensure_cache_exists()?;
    let source_revision = current_revision(&cache)?;
    let source_license = load_metadata()?
        .map(|metadata| metadata.source_license)
        .unwrap_or_else(|| "GPL-3.0-or-later".to_string());

    fs::create_dir_all(out).with_context(|| format!("failed to create {}", out.display()))?;
    let mut seen_ids = BTreeSet::new();
    for profile_path in discover_profile_files(&cache, lang)? {
        let id = profile_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(lang)
            .to_string();
        if !seen_ids.insert(id.clone()) {
            continue;
        }
        let source_file = profile_path
            .strip_prefix(&cache)
            .unwrap_or(&profile_path)
            .to_string_lossy()
            .to_string();
        let content = fs::read_to_string(&profile_path)
            .with_context(|| format!("failed to read {}", profile_path.display()))?;
        let converted = parse_profile_content(
            &id,
            &source_file,
            &content,
            &source_revision,
            &source_license,
        );
        let target = out.join(format!("{id}.toml"));
        fs::write(
            &target,
            toml::to_string_pretty(&converted).context("failed to encode profile TOML")?,
        )
        .with_context(|| format!("failed to write {}", target.display()))?;
    }

    println!("Wrote profile conversions to {}", out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_profile_content;

    #[test]
    fn parses_supported_profile_fields() {
        let content = r#"
name English (America)
language en-us 5
language en 1
phonemes en-us
dictrules 3 6
stressLength 140 120 190 170 0 0 255 300
stressAmp 17 16 19 19 19 19 21 19
replace 3 I i
formant 120
"#;
        let profile = parse_profile_content(
            "en-US",
            "espeak-ng-data/lang/gmw/en-US",
            content,
            "abc",
            "GPL",
        );

        assert_eq!(profile.name.as_deref(), Some("English (America)"));
        assert_eq!(profile.language_tags.len(), 2);
        assert_eq!(profile.phonemes.as_deref(), Some("en-us"));
        assert_eq!(profile.dictrules, vec![3, 6]);
        assert_eq!(profile.stress.length[0], 140);
        assert_eq!(profile.stress.amplitude[0], 17);
        assert_eq!(profile.replacements.len(), 1);
        assert_eq!(profile.unsupported.len(), 1);
    }
}
