#[cfg(any(test, feature = "asr-whisper"))]
use std::sync::OnceLock;

#[cfg(any(test, feature = "asr-whisper"))]
const SOURCE_PAGE_LINES: usize = 50;

#[cfg(any(test, feature = "asr-whisper"))]
pub(in crate::cli::commands) fn execute_list_source_files() -> String {
    let mut files: Vec<_> = source_bundle().keys().cloned().collect();
    files.sort();
    let mut response = String::from("Available source files:\n");
    for file in files {
        response.push_str(&file);
        response.push('\n');
    }
    response
}

#[cfg(any(test, feature = "asr-whisper"))]
pub(in crate::cli::commands) fn execute_view_source_file(path: &str, page: usize) -> String {
    let normalized = path.trim().trim_start_matches("./");
    let page = page.max(1);
    let Some(content) = source_bundle().get(normalized) else {
        return format!("File not found: {normalized}");
    };
    let lines: Vec<_> = content.lines().collect();
    let start = (page - 1) * SOURCE_PAGE_LINES;
    if start >= lines.len() {
        return format!(
            "File {normalized} has only {} lines (page {page} is past EOF).",
            lines.len()
        );
    }
    let end = (start + SOURCE_PAGE_LINES).min(lines.len());
    format!(
        "--- {normalized} (lines {} to {} of {}) ---\n{}\n---",
        start + 1,
        end,
        lines.len(),
        lines[start..end].join("\n")
    )
}

#[cfg(any(test, feature = "asr-whisper"))]
pub(in crate::cli::commands) fn execute_search_source(query: &str, limit: usize) -> String {
    search_source_lines(query, limit, false)
}

#[cfg(any(test, feature = "asr-whisper"))]
pub(in crate::cli::commands) fn execute_grep_source(pattern: &str, limit: usize) -> String {
    search_source_lines(pattern, limit, true)
}

#[cfg(any(test, feature = "asr-whisper"))]
fn search_source_lines(needle: &str, limit: usize, literal: bool) -> String {
    let needle = needle.trim();
    if needle.is_empty() {
        return "Search query was empty.".to_string();
    }

    let max_results = limit.clamp(1, 30);
    let folded_needle = needle.to_lowercase();
    let mut files: Vec<_> = source_bundle().iter().collect();
    files.sort_by_key(|(file, _)| *file);

    let mut results = Vec::new();
    for (file, content) in files {
        for (index, line) in content.lines().enumerate() {
            if line.to_lowercase().contains(&folded_needle) {
                results.push(format!(
                    "{}:{}: {}",
                    file,
                    index + 1,
                    compact_source_line(line.trim(), 220)
                ));
                if results.len() >= max_results {
                    break;
                }
            }
        }
        if results.len() >= max_results {
            break;
        }
    }

    if results.is_empty() {
        format!(
            "No source matches for {}: {}",
            if literal { "pattern" } else { "query" },
            needle
        )
    } else {
        format!(
            "Source matches for {} \"{}\":\n{}",
            if literal { "pattern" } else { "query" },
            needle,
            results.join("\n")
        )
    }
}

#[cfg(any(test, feature = "asr-whisper"))]
fn compact_source_line(text: &str, max_chars: usize) -> String {
    let mut line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if line.chars().count() <= max_chars {
        return line;
    }
    line = line.chars().take(max_chars.saturating_sub(3)).collect();
    line.push_str("...");
    line
}

#[cfg(any(test, feature = "asr-whisper"))]
fn source_bundle() -> &'static std::collections::HashMap<String, String> {
    static BUNDLE: OnceLock<std::collections::HashMap<String, String>> = OnceLock::new();
    BUNDLE.get_or_init(|| {
        let bundle = include_str!(concat!(env!("OUT_DIR"), "/listenbury_source.txt"));
        let mut map = std::collections::HashMap::new();
        let mut current_file = String::new();
        let mut current_content = String::new();

        for line in bundle.lines() {
            if let Some(path) = line.strip_prefix("@@@FILE: ") {
                if !current_file.is_empty() {
                    map.insert(current_file.clone(), current_content.clone());
                    current_content.clear();
                }
                current_file = path.to_string();
            } else {
                current_content.push_str(line);
                current_content.push('\n');
            }
        }
        if !current_file.is_empty() {
            map.insert(current_file, current_content);
        }
        map
    })
}
