use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let whisper_enabled = std::env::var_os("CARGO_FEATURE_ASR_WHISPER").is_some();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").ok();

    if whisper_enabled && target_os.as_deref() == Some("linux") {
        println!("cargo:rustc-link-arg=-Wl,--allow-multiple-definition");
        println!("cargo:rustc-link-lib=gomp");
    }

    write_source_bundle();
}

fn write_source_bundle() {
    let Some(out_dir) = env::var_os("OUT_DIR") else {
        return;
    };
    let dest_path = Path::new(&out_dir).join("listenbury_source.txt");
    let root = env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=README.md");
    println!("cargo:rerun-if-changed=justfile");
    println!("cargo:rerun-if-changed=src");

    let mut paths = Vec::new();
    for root_file in ["Cargo.toml", "build.rs", "justfile", "README.md"] {
        let path = root.join(root_file);
        if path.exists() {
            paths.push(path);
        }
    }

    collect_source_paths(&root.join("src"), &mut paths);
    paths.sort();

    let mut out = String::new();
    for path in paths {
        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };
        let rel_path = path.strip_prefix(&root).unwrap_or(&path);
        out.push_str(&format!("@@@FILE: {}\n", rel_path.display()));
        out.push_str(&content);
        out.push('\n');
    }

    if let Err(error) = fs::write(&dest_path, out) {
        eprintln!(
            "failed to write source bundle {}: {error}",
            dest_path.display()
        );
    }
}

fn collect_source_paths(dir: &Path, paths: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("");
            if matches!(name, "target" | ".git" | "node_modules") {
                continue;
            }
            collect_source_paths(&path, paths);
            continue;
        }

        let ext = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or_default();
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if matches!(ext, "rs" | "toml" | "md") || name == "Dockerfile" {
            paths.push(path);
        }
    }
}
