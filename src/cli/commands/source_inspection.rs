#[cfg(any(test, feature = "asr-whisper"))]
use std::sync::OnceLock;

#[cfg(any(test, feature = "asr-whisper"))]
const SOURCE_PAGE_LINES: usize = 120;
#[cfg(any(test, feature = "asr-whisper"))]
const SOURCE_LIST_PAGE_SIZE: usize = 80;
#[cfg(any(test, feature = "asr-whisper"))]
const MIN_SOURCE_LIST_PAGE_SIZE: usize = 20;
#[cfg(any(test, feature = "asr-whisper"))]
const MAX_SOURCE_LIST_PAGE_SIZE: usize = 200;

#[cfg(any(test, feature = "asr-whisper"))]
pub(in crate::cli::commands) fn execute_list_source_files_page(
    page: usize,
    page_size: Option<usize>,
) -> String {
    let mut files: Vec<_> = source_bundle().keys().cloned().collect();
    files.sort();
    let page = page.max(1);
    let page_size = page_size
        .unwrap_or(SOURCE_LIST_PAGE_SIZE)
        .clamp(MIN_SOURCE_LIST_PAGE_SIZE, MAX_SOURCE_LIST_PAGE_SIZE);
    let total = files.len();
    let page_count = total.max(1).div_ceil(page_size);
    let start = (page - 1) * page_size;
    if start >= total {
        return format!(
            "Available source files page {page} is past the end ({page_count} page(s), {total} file(s), {page_size} files/page). Use listFiles({page_count}) to see the last page."
        );
    }
    let end = (start + page_size).min(total);
    let mut response = format!(
        "Available source files page {page}/{page_count} (files {} to {} of {total}, {page_size} files/page):\n",
        start + 1,
        end,
    );
    if page < page_count {
        response.push_str(&format!(
            "Use listFiles({}) to continue after this page.\n",
            page + 1
        ));
    }
    for file in &files[start..end] {
        response.push_str(&file);
        response.push('\n');
    }
    if page < page_count {
        response.push_str(&format!(
            "More source files are available. Use listFiles({}) to continue.\n",
            page + 1
        ));
    } else {
        response.push_str(
            "End of source file list. Use readSourceFile(path, page?) to inspect a file.\n",
        );
    }
    response
}

#[cfg(any(test, feature = "asr-whisper"))]
pub(in crate::cli::commands) fn execute_view_source_file(path: &str, page: usize) -> String {
    execute_view_source_file_page(path, page, SOURCE_PAGE_LINES)
}

#[cfg(any(test, feature = "asr-whisper"))]
pub(in crate::cli::commands) fn execute_view_source_file_page(
    path: &str,
    page: usize,
    page_lines: usize,
) -> String {
    let normalized = path.trim().trim_start_matches("./");
    let page = page.max(1);
    let page_lines = page_lines.clamp(20, 240);
    let Some(content) = source_bundle().get(normalized) else {
        return format!("File not found: {normalized}");
    };
    let lines: Vec<_> = content.lines().collect();
    let page_count = lines.len().max(1).div_ceil(page_lines);
    let start = (page - 1) * page_lines;
    if start >= lines.len() {
        return format!(
            "File {normalized} has only {} lines ({page_count} page(s) at {page_lines} lines/page; page {page} is past EOF).",
            lines.len(),
        );
    }
    let end = (start + page_lines).min(lines.len());
    format!(
        "--- {normalized} page {page}/{page_count} (lines {} to {} of {}, {page_lines} lines/page) ---\n{}\n---",
        start + 1,
        end,
        lines.len(),
        lines[start..end].join("\n")
    )
}

#[cfg(any(test, feature = "asr-whisper"))]
pub(in crate::cli::commands) fn execute_view_source_file_line(
    path: &str,
    line: usize,
    page_lines: usize,
) -> String {
    let line = line.max(1);
    let page_lines = page_lines.clamp(20, 240);
    let page = (line - 1) / page_lines + 1;
    execute_view_source_file_page(path, page, page_lines)
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
