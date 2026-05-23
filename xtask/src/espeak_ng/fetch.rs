use std::{fs, path::Path, process::Command};

use anyhow::{Context, Result, bail};

use super::provenance::{
    UPSTREAM_URL, cache_path, current_revision, ensure_cache_exists, load_metadata,
    read_source_license, run_git, write_metadata,
};

fn clone_repo(cache: &Path) -> Result<()> {
    let parent = cache.parent().context("cache path has no parent")?;
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;

    let out = Command::new("git")
        .arg("clone")
        .arg(UPSTREAM_URL)
        .arg(cache)
        .output()
        .context("failed to run git clone")?;
    if !out.status.success() {
        bail!(
            "git clone failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(())
}

fn default_remote_ref(cache: &Path) -> Result<String> {
    let remote_head = run_git(cache, &["symbolic-ref", "refs/remotes/origin/HEAD"])?;
    Ok(remote_head.trim().to_string())
}

pub fn fetch(rev: Option<&str>) -> Result<()> {
    let cache = cache_path()?;
    if !cache.join(".git").exists() {
        clone_repo(&cache)?;
    }

    run_git(&cache, &["fetch", "--tags", "--prune", "origin"])?;

    if let Some(rev) = rev {
        run_git(&cache, &["checkout", "--detach", rev])?;
    } else {
        let remote_ref = default_remote_ref(&cache).unwrap_or_else(|_| "origin/master".to_string());
        run_git(&cache, &["checkout", "--detach", &remote_ref])?;
    }

    let revision = current_revision(&cache)?;
    let metadata = write_metadata(&cache, revision.clone())?;

    println!(
        "Fetched eSpeak-ng: {} @ {} (license: {})",
        metadata.upstream_url, metadata.revision, metadata.source_license
    );
    Ok(())
}

pub fn status() -> Result<()> {
    let cache = ensure_cache_exists()?;
    let revision = current_revision(&cache)?;
    let metadata = load_metadata()?;

    println!("cache: {}", cache.display());
    println!("revision: {revision}");
    println!("license: {}", read_source_license(&cache));
    if let Some(metadata) = metadata {
        println!("metadata revision: {}", metadata.revision);
        println!("metadata fetched_unix_secs: {}", metadata.fetched_unix_secs);
    } else {
        println!(
            "metadata: missing ({})",
            cache.join(super::provenance::META_FILE).display()
        );
    }
    Ok(())
}
