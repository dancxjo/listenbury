use std::{fs, path::Path};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::provenance::{CONVERTER_VERSION, current_revision, ensure_cache_exists, load_metadata};

const SUPPORTED_DICTIONARY_FLAGS: &[char] = &['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictionaryEntry {
    pub token: String,
    pub pronunciation: String,
    pub kind: String,
    pub flags: Vec<String>,
    pub source_file: String,
    pub source_line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AltDefinition {
    pub name: String,
    pub value: String,
    pub source_file: String,
    pub source_line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsupportedFlag {
    pub token: String,
    pub flag: String,
    pub source_file: String,
    pub source_line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvertedDictionary {
    pub lang: String,
    pub source: String,
    pub source_revision: String,
    pub source_license: String,
    pub converter_version: String,
    pub entries: Vec<DictionaryEntry>,
    pub alt_definitions: Vec<AltDefinition>,
    pub unsupported_flags: Vec<UnsupportedFlag>,
}

fn strip_comment(line: &str) -> &str {
    line.split_once("//").map_or(line, |(head, _)| head)
}

fn classify_entry(token: &str, pronunciation: &str) -> String {
    if token.contains('_') {
        return "multiword".to_string();
    }
    if token
        .chars()
        .all(|ch| ch.is_ascii_digit() || matches!(ch, ',' | '.' | '-' | '/'))
    {
        return "number_fragment".to_string();
    }
    if token.chars().count() == 1
        && token
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_alphabetic())
    {
        return "letter_name".to_string();
    }
    if token.starts_with('&') || token.starts_with('\\') || token.starts_with('@') {
        return "symbol_name".to_string();
    }
    if pronunciation
        .chars()
        .all(|ch| ch.is_ascii_whitespace() || matches!(ch, '\'' | ',' | ';' | ':' | '`'))
    {
        return "stress_only".to_string();
    }
    "explicit_pronunciation".to_string()
}

fn parse_token_and_flags(raw_token: &str) -> (String, Vec<String>) {
    if let Some((token, flags)) = raw_token.split_once('/') {
        let parsed_flags = flags.chars().map(|ch| ch.to_string()).collect();
        (token.to_string(), parsed_flags)
    } else {
        (raw_token.to_string(), Vec::new())
    }
}

fn parse_dictionary_file(
    source_file: &str,
    content: &str,
    entries: &mut Vec<DictionaryEntry>,
    alt_definitions: &mut Vec<AltDefinition>,
    unsupported_flags: &mut Vec<UnsupportedFlag>,
) {
    for (index, raw_line) in content.lines().enumerate() {
        let line_no = index + 1;
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }

        if let Some(rest) = line.strip_prefix("$alt") {
            let mut parts = rest.trim().split_whitespace();
            let name = parts.next().unwrap_or_default().to_string();
            let value = parts.collect::<Vec<_>>().join(" ");
            alt_definitions.push(AltDefinition {
                name,
                value,
                source_file: source_file.to_string(),
                source_line: line_no,
            });
            continue;
        }

        let mut parts = line.split_whitespace();
        let Some(raw_token) = parts.next() else {
            continue;
        };
        let pronunciation = parts.collect::<Vec<_>>().join(" ");
        if pronunciation.is_empty() {
            continue;
        }

        let (token, flags) = parse_token_and_flags(raw_token);
        let kind = classify_entry(&token, &pronunciation);

        for flag in &flags {
            let is_supported = flag
                .chars()
                .next()
                .is_some_and(|value| SUPPORTED_DICTIONARY_FLAGS.contains(&value));
            if !is_supported {
                unsupported_flags.push(UnsupportedFlag {
                    token: token.clone(),
                    flag: flag.clone(),
                    source_file: source_file.to_string(),
                    source_line: line_no,
                });
            }
        }

        entries.push(DictionaryEntry {
            token,
            pronunciation,
            kind,
            flags,
            source_file: source_file.to_string(),
            source_line: line_no,
        });
    }
}

pub fn convert_list(lang: &str, out: &Path) -> Result<()> {
    let cache = ensure_cache_exists()?;
    let source_revision = current_revision(&cache)?;
    let source_license = load_metadata()?
        .map(|metadata| metadata.source_license)
        .unwrap_or_else(|| "GPL-3.0-or-later".to_string());

    fs::create_dir_all(out).with_context(|| format!("failed to create {}", out.display()))?;

    let mut entries = Vec::new();
    let mut alt_definitions = Vec::new();
    let mut unsupported_flags = Vec::new();

    for source_file in [
        format!("dictsource/{lang}_list"),
        format!("dictsource/{lang}_extra"),
    ] {
        let path = cache.join(&source_file);
        if !path.exists() {
            continue;
        }
        let content = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        parse_dictionary_file(
            &source_file,
            &content,
            &mut entries,
            &mut alt_definitions,
            &mut unsupported_flags,
        );
    }

    entries.sort_by(|a, b| (&a.token, a.source_line).cmp(&(&b.token, b.source_line)));
    alt_definitions.sort_by(|a, b| (&a.name, a.source_line).cmp(&(&b.name, b.source_line)));
    unsupported_flags.sort_by(|a, b| {
        (&a.token, &a.flag, a.source_line).cmp(&(&b.token, &b.flag, b.source_line))
    });

    let converted = ConvertedDictionary {
        lang: lang.to_string(),
        source: "espeak-ng".to_string(),
        source_revision,
        source_license,
        converter_version: CONVERTER_VERSION.to_string(),
        entries,
        alt_definitions,
        unsupported_flags,
    };

    let target = out.join("dictionary.toml");
    fs::write(
        &target,
        toml::to_string_pretty(&converted).context("failed to encode dictionary TOML")?,
    )
    .with_context(|| format!("failed to write {}", target.display()))?;

    println!("Wrote dictionary conversion to {}", target.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_dictionary_file;

    #[test]
    fn inventories_dictionary_categories_and_unsupported_flags() {
        let sample = r#"
$alt1 weak_form
A EY
foo/bar F UW
new_york N UW Y AO R K
123 W AH N T UW TH R IY
"#;
        let mut entries = Vec::new();
        let mut alt = Vec::new();
        let mut unsupported = Vec::new();
        parse_dictionary_file(
            "dictsource/en_list",
            sample,
            &mut entries,
            &mut alt,
            &mut unsupported,
        );

        assert_eq!(alt.len(), 1);
        assert!(entries.iter().any(|entry| entry.kind == "letter_name"));
        assert!(entries.iter().any(|entry| entry.kind == "multiword"));
        assert!(entries.iter().any(|entry| entry.kind == "number_fragment"));
        assert!(!unsupported.is_empty());
    }
}
