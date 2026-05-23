use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::{
    RulesMode,
    provenance::{CONVERTER_VERSION, current_revision, ensure_cache_exists, load_metadata},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LetterGroup {
    pub id: String,
    pub value: String,
    pub source_line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaceRule {
    pub from: String,
    pub to: String,
    pub source_line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupSummary {
    pub name: String,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsupportedConstruct {
    pub directive: String,
    pub source_line: usize,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvertedRulesInventory {
    pub lang: String,
    pub mode: String,
    pub source: String,
    pub source_file: String,
    pub source_revision: String,
    pub source_license: String,
    pub converter_version: String,
    pub letter_groups: Vec<LetterGroup>,
    pub replace_rules: Vec<ReplaceRule>,
    pub groups: Vec<GroupSummary>,
    pub condition_flags_seen: Vec<String>,
    pub suffix_prefix_operators_seen: Vec<String>,
    pub special_symbols_seen: Vec<String>,
    pub unsupported: Vec<UnsupportedConstruct>,
}

fn strip_comment(line: &str) -> &str {
    line.split_once("//").map_or(line, |(head, _)| head)
}

fn extract_condition_flags(line: &str, seen: &mut BTreeSet<String>) {
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '?' {
            let mut flag = String::from("?");
            i += 1;
            if i < chars.len() && chars[i] == '!' {
                flag.push('!');
                i += 1;
            }
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                flag.push(chars[i]);
                i += 1;
            }
            if i > start {
                seen.insert(flag);
                continue;
            }
        }
        i += 1;
    }
}

pub fn parse_rules_content(
    lang: &str,
    source_file: &str,
    content: &str,
    source_revision: &str,
    source_license: &str,
    mode: RulesMode,
) -> ConvertedRulesInventory {
    let mut letter_groups = Vec::new();
    let mut replace_rules = Vec::new();
    let mut group_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut condition_flags_seen = BTreeSet::new();
    let mut operators_seen = BTreeSet::new();
    let mut symbols_seen = BTreeSet::new();
    let mut unsupported = Vec::new();

    for (index, raw_line) in content.lines().enumerate() {
        let line_no = index + 1;
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }

        extract_condition_flags(line, &mut condition_flags_seen);

        for ch in ['+', '-', '<', '>', '^', '$'] {
            if line.contains(ch) {
                operators_seen.insert(ch.to_string());
            }
        }
        for ch in ['_', '#', '%', '&', '@', ':', '(', ')'] {
            if line.contains(ch) {
                symbols_seen.insert(ch.to_string());
            }
        }

        if line.starts_with(".L") {
            let mut parts = line.split_whitespace();
            let id = parts
                .next()
                .unwrap_or_default()
                .trim_start_matches('.')
                .to_string();
            let value = parts.collect::<Vec<_>>().join(" ");
            letter_groups.push(LetterGroup {
                id,
                value,
                source_line: line_no,
            });
            continue;
        }

        if line.starts_with(".replace") {
            let mut parts = line.split_whitespace();
            let _ = parts.next();
            let from = parts.next().unwrap_or_default().to_string();
            let to = parts.next().unwrap_or_default().to_string();
            if !from.is_empty() && !to.is_empty() {
                replace_rules.push(ReplaceRule {
                    from,
                    to,
                    source_line: line_no,
                });
            } else {
                unsupported.push(UnsupportedConstruct {
                    directive: ".replace".to_string(),
                    source_line: line_no,
                    content: line.to_string(),
                });
            }
            continue;
        }

        if line.starts_with(".group") {
            let name = line
                .split_whitespace()
                .nth(1)
                .unwrap_or("unnamed")
                .to_string();
            *group_counts.entry(name).or_insert(0) += 1;
            continue;
        }

        if line.starts_with('.') {
            unsupported.push(UnsupportedConstruct {
                directive: line.split_whitespace().next().unwrap_or(".").to_string(),
                source_line: line_no,
                content: line.to_string(),
            });
        }
    }

    let groups = group_counts
        .into_iter()
        .map(|(name, count)| GroupSummary { name, count })
        .collect();

    let replace_rules = if matches!(mode, RulesMode::NativeSubset) {
        let mut filtered = Vec::new();
        for rule in replace_rules {
            if rule.from.chars().all(|ch| ch.is_ascii_alphabetic()) {
                filtered.push(rule);
            } else {
                unsupported.push(UnsupportedConstruct {
                    directive: ".replace".to_string(),
                    source_line: rule.source_line,
                    content: format!("filtered-out-native-subset: {} -> {}", rule.from, rule.to),
                });
            }
        }
        filtered
    } else {
        replace_rules
    };

    ConvertedRulesInventory {
        lang: lang.to_string(),
        mode: match mode {
            RulesMode::Inventory => "inventory",
            RulesMode::NativeSubset => "native-subset",
        }
        .to_string(),
        source: "espeak-ng".to_string(),
        source_file: source_file.to_string(),
        source_revision: source_revision.to_string(),
        source_license: source_license.to_string(),
        converter_version: CONVERTER_VERSION.to_string(),
        letter_groups,
        replace_rules,
        groups,
        condition_flags_seen: condition_flags_seen.into_iter().collect(),
        suffix_prefix_operators_seen: operators_seen.into_iter().collect(),
        special_symbols_seen: symbols_seen.into_iter().collect(),
        unsupported,
    }
}

pub fn convert_rules(lang: &str, out: &Path, mode: RulesMode) -> Result<()> {
    let cache = ensure_cache_exists()?;
    let source_file = format!("dictsource/{lang}_rules");
    let path = cache.join(&source_file);
    if !path.exists() {
        bail!("missing rules file {}", path.display());
    }

    let source_revision = current_revision(&cache)?;
    let source_license = load_metadata()?
        .map(|metadata| metadata.source_license)
        .unwrap_or_else(|| "GPL-3.0-or-later".to_string());

    let content =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let converted = parse_rules_content(
        lang,
        &source_file,
        &content,
        &source_revision,
        &source_license,
        mode,
    );

    fs::create_dir_all(out).with_context(|| format!("failed to create {}", out.display()))?;
    let suffix = if matches!(mode, RulesMode::NativeSubset) {
        "rules-native-subset.toml"
    } else {
        "rules-inventory.toml"
    };
    let target = out.join(suffix);
    fs::write(
        &target,
        toml::to_string_pretty(&converted).context("failed to encode rules TOML")?,
    )
    .with_context(|| format!("failed to write {}", target.display()))?;

    println!("Wrote rules conversion to {}", target.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_rules_content;
    use crate::espeak_ng::RulesMode;

    #[test]
    fn inventories_groups_replacements_and_unsupported() {
        let sample = r#"
.L01 aeiou
.group vowels
.replace aa ah
?3 a) A:
.unknown x y z
"#;
        let inventory = parse_rules_content(
            "en",
            "dictsource/en_rules",
            sample,
            "sha",
            "GPL",
            RulesMode::Inventory,
        );
        assert_eq!(inventory.letter_groups.len(), 1);
        assert_eq!(inventory.groups.len(), 1);
        assert_eq!(inventory.replace_rules.len(), 1);
        assert!(
            inventory
                .condition_flags_seen
                .iter()
                .any(|flag| flag == "?3")
        );
        assert!(
            inventory
                .unsupported
                .iter()
                .any(|item| item.directive == ".unknown")
        );
    }
}
