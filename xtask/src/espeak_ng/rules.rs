use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use anyhow::{Context, Result};
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
pub struct MorphAction {
    pub kind: String,
    pub count: usize,
    pub flags: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleRecord {
    pub group: Option<String>,
    pub condition: Option<String>,
    pub pre: Option<String>,
    pub match_text: String,
    pub post: Option<String>,
    pub phonemes: String,
    pub score_delta: i32,
    pub morph_actions: Vec<MorphAction>,
    pub language_switch: Option<String>,
    pub raw: String,
    pub source_line: usize,
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
    pub rules: Vec<RuleRecord>,
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

fn split_condition(line: &str) -> (Option<String>, &str) {
    let Some(split_at) = line.find(char::is_whitespace) else {
        return (None, line);
    };
    let (first, rest) = line.split_at(split_at);
    if first.starts_with('?')
        && first[1..]
            .trim_start_matches('!')
            .chars()
            .all(|ch| ch.is_ascii_digit())
    {
        return (Some(first.to_string()), rest.trim_start());
    }
    (None, line)
}

fn parse_morph_actions(post: Option<&str>) -> Vec<MorphAction> {
    let mut actions = Vec::new();
    let Some(post) = post else {
        return actions;
    };
    let chars: Vec<char> = post.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let kind = match chars[i] {
            'S' => "suffix",
            'P' => "prefix",
            _ => {
                i += 1;
                continue;
            }
        };
        i += 1;
        let start = i;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        if i == start {
            continue;
        }
        let count = chars[start..i]
            .iter()
            .collect::<String>()
            .parse::<usize>()
            .unwrap_or_default();
        let flags_start = i;
        while i < chars.len() && chars[i].is_ascii_alphabetic() {
            i += 1;
        }
        actions.push(MorphAction {
            kind: kind.to_string(),
            count,
            flags: chars[flags_start..i].iter().collect(),
        });
    }
    actions
}

fn parse_rule_record(
    group: Option<&str>,
    condition: Option<String>,
    line: &str,
    source_line: usize,
) -> Option<RuleRecord> {
    let mut parts = line.split_whitespace();
    let first = parts.next()?;
    let mut pre = None;
    let match_text;
    if let Some(prefix) = first.strip_suffix(')') {
        pre = Some(prefix.to_string());
        match_text = parts.next()?.to_string();
    } else {
        match_text = first.to_string();
    }

    let mut post = None;
    let mut phoneme_parts = Vec::new();
    for part in parts {
        if post.is_none() && part.starts_with('(') {
            post = Some(part.trim_start_matches('(').to_string());
        } else {
            phoneme_parts.push(part.to_string());
        }
    }
    let phonemes = phoneme_parts.join(" ");
    if match_text.is_empty() || phonemes.is_empty() {
        return None;
    }
    let score_delta = post.as_deref().map_or(0, |post| {
        let increases = post.chars().filter(|ch| *ch == '+').count() as i32;
        let decreases = post.chars().filter(|ch| *ch == '<').count() as i32;
        (increases - decreases) * 20
    });
    let language_switch = phoneme_parts
        .iter()
        .find_map(|part| part.strip_prefix("_^_").map(|lang| lang.to_string()));
    let morph_actions = parse_morph_actions(post.as_deref());

    Some(RuleRecord {
        group: group.map(|value| value.to_string()),
        condition,
        pre,
        match_text,
        post,
        phonemes,
        score_delta,
        morph_actions,
        language_switch,
        raw: line.to_string(),
        source_line,
    })
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
    let mut rules = Vec::new();
    let mut group_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut condition_flags_seen = BTreeSet::new();
    let mut operators_seen = BTreeSet::new();
    let mut symbols_seen = BTreeSet::new();
    let mut unsupported = Vec::new();
    let mut current_group: Option<String> = None;
    let mut in_replace = false;

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

        let (condition, line_without_condition) = split_condition(line);
        let line = line_without_condition;

        if line.starts_with(".L") {
            in_replace = false;
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
            in_replace = from.is_empty() && to.is_empty();
            if !from.is_empty() && !to.is_empty() {
                replace_rules.push(ReplaceRule {
                    from,
                    to,
                    source_line: line_no,
                });
            } else if !from.is_empty() || !to.is_empty() {
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
            *group_counts.entry(name.clone()).or_insert(0) += 1;
            current_group = Some(name);
            in_replace = false;
            continue;
        }

        if line.starts_with('.') {
            in_replace = false;
            unsupported.push(UnsupportedConstruct {
                directive: line.split_whitespace().next().unwrap_or(".").to_string(),
                source_line: line_no,
                content: line.to_string(),
            });
            continue;
        }

        if in_replace {
            let mut parts = line.split_whitespace();
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

        if let Some(rule) =
            parse_rule_record(current_group.as_deref(), condition.clone(), line, line_no)
        {
            rules.push(rule);
        } else {
            unsupported.push(UnsupportedConstruct {
                directive: "rule".to_string(),
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
        rules,
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

    let source_revision = current_revision(&cache)?;
    let source_license = load_metadata()?
        .map(|metadata| metadata.source_license)
        .unwrap_or_else(|| "GPL-3.0-or-later".to_string());

    let content = if path.exists() {
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?
    } else {
        String::new()
    };
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
