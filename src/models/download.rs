use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

use crate::models::manifest::ModelAsset;
use crate::models::paths::asset_path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetIntegrityState {
    Missing,
    PresentUnverified,
    PresentValid,
    PresentInvalidSize,
    PresentInvalidChecksum,
    UnknownChecksum,
}

pub fn fetch_asset(home: &Path, asset: &ModelAsset) -> Result<bool> {
    fetch_asset_with_progress(home, asset, |_, _| {})
}

pub fn fetch_asset_with_progress(
    home: &Path,
    asset: &ModelAsset,
    mut progress: impl FnMut(u64, Option<u64>),
) -> Result<bool> {
    fetch_asset_with_progress_and_verify(home, asset, false, &mut progress)
}

pub fn fetch_asset_with_progress_and_verify(
    home: &Path,
    asset: &ModelAsset,
    verify_existing: bool,
    mut progress: impl FnMut(u64, Option<u64>),
) -> Result<bool> {
    let target_path = asset_path(home, asset);
    let parent = target_path
        .parent()
        .context("asset path had no parent directory")?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create model directory {}", parent.display()))?;

    let existing_bytes = file_len(&target_path)?;
    let remote_bytes = remote_content_length(asset).or(asset.expected_size_bytes);
    if let Some(existing_bytes) = existing_bytes
        && existing_bytes > 0
    {
        if verify_existing {
            match verify_existing_asset(&target_path, asset, true)? {
                AssetIntegrityState::PresentValid | AssetIntegrityState::UnknownChecksum => {
                    cleanup_partial_files(parent, asset, None)?;
                    return Ok(false);
                }
                AssetIntegrityState::PresentInvalidSize
                | AssetIntegrityState::PresentInvalidChecksum => {
                    remove_file_if_exists(&target_path)?;
                    cleanup_partial_files(parent, asset, None)?;
                    return download_fresh(asset, &target_path, remote_bytes, &mut progress);
                }
                AssetIntegrityState::Missing | AssetIntegrityState::PresentUnverified => {}
            }
        }
        match remote_bytes {
            Some(total_bytes) if existing_bytes == total_bytes => {
                cleanup_partial_files(parent, asset, None)?;
                return Ok(false);
            }
            Some(total_bytes) if existing_bytes < total_bytes => {
                return resume_download(
                    asset,
                    &target_path,
                    &target_path,
                    existing_bytes,
                    Some(total_bytes),
                    &mut progress,
                );
            }
            Some(_) => {}
            None => return Ok(false),
        }
    }

    if let Some(partial_path) = reusable_partial_path(parent, asset, remote_bytes)? {
        let existing_bytes = file_len(&partial_path)?.unwrap_or(0);
        return resume_download(
            asset,
            &partial_path,
            &target_path,
            existing_bytes,
            remote_bytes,
            &mut progress,
        );
    }

    download_fresh(asset, &target_path, remote_bytes, &mut progress)
}

pub fn verify_existing_asset(
    path: &Path,
    asset: &ModelAsset,
    verify: bool,
) -> Result<AssetIntegrityState> {
    let Some(actual_bytes) = file_len(path)? else {
        return Ok(AssetIntegrityState::Missing);
    };
    if actual_bytes == 0 {
        return Ok(AssetIntegrityState::Missing);
    }
    if !verify {
        return Ok(AssetIntegrityState::PresentUnverified);
    }
    if let Some(expected_size_bytes) = asset.expected_size_bytes
        && expected_size_bytes != actual_bytes
    {
        return Ok(AssetIntegrityState::PresentInvalidSize);
    }
    let Some(expected_sha256) = asset.sha256 else {
        return Ok(AssetIntegrityState::UnknownChecksum);
    };
    let actual_sha256 = file_sha256_hex(path)?;
    if actual_sha256.eq_ignore_ascii_case(expected_sha256) {
        Ok(AssetIntegrityState::PresentValid)
    } else {
        Ok(AssetIntegrityState::PresentInvalidChecksum)
    }
}

fn download_fresh(
    asset: &ModelAsset,
    target_path: &Path,
    expected_total_bytes: Option<u64>,
    progress: &mut impl FnMut(u64, Option<u64>),
) -> Result<bool> {
    let parent = target_path
        .parent()
        .context("asset path had no parent directory")?;
    let temp_path = fresh_partial_path(parent, asset)?;

    let response = ureq::get(asset.url)
        .call()
        .with_context(|| format!("failed to download {}", asset.url))?;
    let total_bytes = response.body().content_length().or(expected_total_bytes);

    let mut body = response.into_body();
    let mut reader = body.as_reader();
    let mut file = BufWriter::new(
        File::create(&temp_path)
            .with_context(|| format!("failed to create temp file {}", temp_path.display()))?,
    );
    copy_with_progress(&mut reader, &mut file, total_bytes, progress).with_context(|| {
        format!(
            "failed to write download for {} to {}",
            asset.id,
            temp_path.display()
        )
    })?;
    file.flush()
        .with_context(|| format!("failed to flush {}", temp_path.display()))?;
    drop(file);

    finalize_download(asset, &temp_path, target_path, total_bytes)?;
    Ok(true)
}

fn resume_download(
    asset: &ModelAsset,
    download_path: &Path,
    target_path: &Path,
    existing_bytes: u64,
    expected_total_bytes: Option<u64>,
    progress: &mut impl FnMut(u64, Option<u64>),
) -> Result<bool> {
    let response = ureq::get(asset.url)
        .header("Range", format!("bytes={existing_bytes}-"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .with_context(|| format!("failed to resume download {}", asset.url))?;

    match response.status().as_u16() {
        206 => {
            let total_bytes = content_range_total(response.headers())
                .or_else(|| {
                    response
                        .body()
                        .content_length()
                        .map(|remaining| existing_bytes + remaining)
                })
                .or(expected_total_bytes);
            let mut body = response.into_body();
            let mut reader = body.as_reader();
            let mut file = BufWriter::new(
                OpenOptions::new()
                    .append(true)
                    .open(download_path)
                    .with_context(|| {
                        format!("failed to open partial file {}", download_path.display())
                    })?,
            );
            copy_with_progress_from(
                &mut reader,
                &mut file,
                existing_bytes,
                total_bytes,
                progress,
            )
            .with_context(|| {
                format!(
                    "failed to append resumed download for {} to {}",
                    asset.id,
                    download_path.display()
                )
            })?;
            file.flush()
                .with_context(|| format!("failed to flush {}", download_path.display()))?;
            drop(file);
            finalize_download(asset, download_path, target_path, total_bytes)?;
            Ok(true)
        }
        200 => download_fresh(asset, target_path, expected_total_bytes, progress),
        416 => {
            let total_bytes = content_range_total(response.headers()).or(expected_total_bytes);
            if total_bytes.is_some_and(|total_bytes| existing_bytes >= total_bytes) {
                finalize_download(asset, download_path, target_path, total_bytes)?;
                Ok(false)
            } else {
                download_fresh(asset, target_path, expected_total_bytes, progress)
            }
        }
        status => bail!(
            "server returned HTTP {status} while resuming {} from byte {existing_bytes}",
            asset.url
        ),
    }
}

fn reusable_partial_path(
    parent: &Path,
    asset: &ModelAsset,
    expected_total_bytes: Option<u64>,
) -> Result<Option<PathBuf>> {
    let prefix = partial_filename_prefix(asset);
    let mut best: Option<(PathBuf, u64)> = None;
    for entry in fs::read_dir(parent)
        .with_context(|| format!("failed to read model directory {}", parent.display()))?
    {
        let entry = entry
            .with_context(|| format!("failed to inspect model directory {}", parent.display()))?;
        let path = entry.path();
        let Some(filename) = path.file_name().and_then(|filename| filename.to_str()) else {
            continue;
        };
        if !filename.starts_with(&prefix) || !filename.ends_with(".part") {
            continue;
        }
        let Some(len) = file_len(&path)? else {
            continue;
        };
        if len == 0 || expected_total_bytes.is_some_and(|total_bytes| len > total_bytes) {
            continue;
        }
        if best.as_ref().is_none_or(|(_, best_len)| len > *best_len) {
            best = Some((path, len));
        }
    }
    Ok(best.map(|(path, _)| path))
}

fn fresh_partial_path(parent: &Path, asset: &ModelAsset) -> Result<PathBuf> {
    let timestamp_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before UNIX_EPOCH")?
        .as_nanos();
    Ok(parent.join(format!(
        "{}{}.{}.part",
        partial_filename_prefix(asset),
        std::process::id(),
        timestamp_nanos
    )))
}

fn partial_filename_prefix(asset: &ModelAsset) -> String {
    format!(".{}.", asset.filename)
}

fn remote_content_length(asset: &ModelAsset) -> Option<u64> {
    let response = ureq::head(asset.url)
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .ok()?;
    if !response.status().is_success() {
        return None;
    }
    content_length_header(response.headers()).or_else(|| response.body().content_length())
}

fn copy_with_progress(
    reader: &mut impl Read,
    writer: &mut impl Write,
    total_bytes: Option<u64>,
    progress: &mut impl FnMut(u64, Option<u64>),
) -> Result<u64> {
    copy_with_progress_from(reader, writer, 0, total_bytes, progress)
}

fn copy_with_progress_from(
    reader: &mut impl Read,
    writer: &mut impl Write,
    initial_downloaded: u64,
    total_bytes: Option<u64>,
    progress: &mut impl FnMut(u64, Option<u64>),
) -> Result<u64> {
    let mut downloaded = initial_downloaded;
    let mut buffer = [0; 64 * 1024];
    progress(downloaded, total_bytes);
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        writer.write_all(&buffer[..read])?;
        downloaded += read as u64;
        progress(downloaded, total_bytes);
    }
    Ok(downloaded)
}

fn file_len(path: &Path) -> Result<Option<u64>> {
    match fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => Ok(Some(metadata.len())),
        Ok(_) => Ok(None),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => {
            Err(error).with_context(|| format!("failed to read file metadata {}", path.display()))
        }
    }
}

fn file_sha256_hex(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("failed to open file {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn remove_file_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("failed to remove {}", path.display())),
    }
}

fn finalize_download(
    asset: &ModelAsset,
    download_path: &Path,
    target_path: &Path,
    total_bytes: Option<u64>,
) -> Result<()> {
    let verify_result = verify_downloaded_asset(asset, download_path, total_bytes);
    if let Err(error) = verify_result {
        remove_file_if_exists(download_path)?;
        return Err(error);
    }
    if download_path != target_path {
        fs::rename(download_path, target_path).with_context(|| {
            format!(
                "failed to move completed download {} to {}",
                download_path.display(),
                target_path.display()
            )
        })?;
    }
    if let Some(parent) = target_path.parent() {
        cleanup_partial_files(parent, asset, Some(target_path))?;
    }
    Ok(())
}

fn verify_downloaded_asset(
    asset: &ModelAsset,
    path: &Path,
    total_bytes: Option<u64>,
) -> Result<()> {
    let actual_bytes = file_len(path)?.unwrap_or(0);
    if let Some(total_bytes) = total_bytes
        && actual_bytes != total_bytes
    {
        bail!(
            "downloaded file {} is {actual_bytes} bytes, expected {total_bytes} bytes",
            path.display()
        );
    }
    if let Some(expected_size_bytes) = asset.expected_size_bytes
        && actual_bytes != expected_size_bytes
    {
        bail!(
            "downloaded file {} is {actual_bytes} bytes, expected manifest size {expected_size_bytes} bytes",
            path.display()
        );
    }
    if let Some(expected_sha256) = asset.sha256 {
        let actual_sha256 = file_sha256_hex(path)?;
        if !actual_sha256.eq_ignore_ascii_case(expected_sha256) {
            bail!(
                "downloaded file {} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}",
                path.display()
            );
        }
    }
    Ok(())
}

fn cleanup_partial_files(parent: &Path, asset: &ModelAsset, keep: Option<&Path>) -> Result<()> {
    let prefix = partial_filename_prefix(asset);
    for entry in fs::read_dir(parent)
        .with_context(|| format!("failed to read model directory {}", parent.display()))?
    {
        let entry = entry
            .with_context(|| format!("failed to inspect model directory {}", parent.display()))?;
        let path = entry.path();
        if keep.is_some_and(|keep| path == keep) {
            continue;
        }
        let Some(filename) = path.file_name().and_then(|filename| filename.to_str()) else {
            continue;
        };
        if filename.starts_with(&prefix) && filename.ends_with(".part") {
            fs::remove_file(&path)
                .with_context(|| format!("failed to remove stale partial {}", path.display()))?;
        }
    }
    Ok(())
}

fn content_range_total(headers: &ureq::http::HeaderMap) -> Option<u64> {
    let value = headers.get("Content-Range")?.to_str().ok()?;
    value.rsplit('/').next()?.parse().ok()
}

fn content_length_header(headers: &ureq::http::HeaderMap) -> Option<u64> {
    headers.get("Content-Length")?.to_str().ok()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::path::PathBuf;
    use std::sync::mpsc;
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{AssetIntegrityState, fetch_asset_with_progress, verify_existing_asset};
    use crate::models::manifest::ModelAsset;

    fn temp_dir(label: &str) -> PathBuf {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("listenbury-{label}-{}-{ts}", std::process::id()))
    }

    #[test]
    fn fetch_resumes_partial_file_with_range_request() {
        let body = b"already-finished-model".to_vec();
        let server_body = body.clone();
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let url = format!("http://{}/model.bin", listener.local_addr().expect("addr"));
        let (range_tx, range_rx) = mpsc::channel();

        let server = thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().expect("accept request");
                let request = read_request(&mut stream);
                if request.starts_with("HEAD ") {
                    write_response(
                        &mut stream,
                        "HTTP/1.1 200 OK",
                        &[("Content-Length", &server_body.len().to_string())],
                        &[],
                    );
                    continue;
                }

                let range = request
                    .lines()
                    .filter_map(|line| line.split_once(':'))
                    .find_map(|(name, value)| {
                        name.eq_ignore_ascii_case("range")
                            .then(|| value.trim().strip_prefix("bytes="))
                            .flatten()
                    })
                    .expect("range header")
                    .trim()
                    .trim_end_matches('-')
                    .parse::<usize>()
                    .expect("range start");
                range_tx.send(range).expect("send range");
                let content_range = format!(
                    "bytes {range}-{}/{}",
                    server_body.len() - 1,
                    server_body.len()
                );
                write_response(
                    &mut stream,
                    "HTTP/1.1 206 Partial Content",
                    &[
                        ("Content-Length", &(server_body.len() - range).to_string()),
                        ("Content-Range", &content_range),
                    ],
                    &server_body[range..],
                );
            }
        });

        let home = temp_dir("models-resume-partial");
        let asset_path = home.join("models/test/model.bin");
        fs::create_dir_all(asset_path.parent().expect("parent")).expect("mkdir");
        fs::write(&asset_path, b"already-").expect("write partial model");
        let asset = ModelAsset {
            id: "test-asset",
            filename: "model.bin",
            relative_path: "models/test/model.bin",
            url: Box::leak(url.into_boxed_str()),
            expected_size_bytes: None,
            sha256: None,
            license: None,
            source: None,
        };

        let mut progress_updates = Vec::new();
        let downloaded = fetch_asset_with_progress(&home, &asset, |downloaded, total| {
            progress_updates.push((downloaded, total));
        })
        .expect("resume download");

        assert!(downloaded);
        assert_eq!(range_rx.recv().expect("range"), b"already-".len());
        assert_eq!(fs::read(&asset_path).expect("read model"), body);
        assert!(progress_updates.contains(&(b"already-".len() as u64, Some(body.len() as u64))));
        assert_eq!(
            progress_updates.last(),
            Some(&(body.len() as u64, Some(body.len() as u64)))
        );
        server.join().expect("server");
    }

    #[test]
    fn fetch_resumes_largest_temp_partial_file() {
        let body = b"already-finished-model".to_vec();
        let server_body = body.clone();
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let url = format!("http://{}/model.bin", listener.local_addr().expect("addr"));
        let (range_tx, range_rx) = mpsc::channel();

        let server = thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().expect("accept request");
                let request = read_request(&mut stream);
                if request.starts_with("HEAD ") {
                    write_response(
                        &mut stream,
                        "HTTP/1.1 200 OK",
                        &[("Content-Length", &server_body.len().to_string())],
                        &[],
                    );
                    continue;
                }

                let range = request_range_start(&request);
                range_tx.send(range).expect("send range");
                let content_range = format!(
                    "bytes {range}-{}/{}",
                    server_body.len() - 1,
                    server_body.len()
                );
                write_response(
                    &mut stream,
                    "HTTP/1.1 206 Partial Content",
                    &[
                        ("Content-Length", &(server_body.len() - range).to_string()),
                        ("Content-Range", &content_range),
                    ],
                    &server_body[range..],
                );
            }
        });

        let home = temp_dir("models-resume-temp-partial");
        let model_dir = home.join("models/test");
        let asset_path = model_dir.join("model.bin");
        let smaller_partial = model_dir.join(".model.bin.1.1.part");
        let larger_partial = model_dir.join(".model.bin.2.2.part");
        fs::create_dir_all(&model_dir).expect("mkdir");
        fs::write(&smaller_partial, b"already").expect("write smaller partial");
        fs::write(&larger_partial, b"already-").expect("write larger partial");
        let asset = ModelAsset {
            id: "test-asset",
            filename: "model.bin",
            relative_path: "models/test/model.bin",
            url: Box::leak(url.into_boxed_str()),
            expected_size_bytes: None,
            sha256: None,
            license: None,
            source: None,
        };

        let mut progress_updates = Vec::new();
        let downloaded = fetch_asset_with_progress(&home, &asset, |downloaded, total| {
            progress_updates.push((downloaded, total));
        })
        .expect("resume temp download");

        assert!(downloaded);
        assert_eq!(range_rx.recv().expect("range"), b"already-".len());
        assert_eq!(fs::read(&asset_path).expect("read model"), body);
        assert!(!larger_partial.exists());
        assert!(!smaller_partial.exists());
        assert!(progress_updates.contains(&(b"already-".len() as u64, Some(body.len() as u64))));
        server.join().expect("server");
    }

    #[test]
    fn verify_reports_valid_checksum() {
        let home = temp_dir("models-verify-valid");
        let asset_path = home.join("models/test/model.bin");
        let body = b"tiny-model";
        fs::create_dir_all(asset_path.parent().expect("parent")).expect("mkdir");
        fs::write(&asset_path, body).expect("write model");

        let asset = test_asset(
            "model.bin",
            "models/test/model.bin",
            body.len() as u64,
            Some("ccdbfb9993be88c536b0b7cd2abe60eda83c7ce1ad530c6a2ada81510ff1548c"),
        );
        let integrity = verify_existing_asset(&asset_path, &asset, true).expect("verify existing");
        assert_eq!(integrity, AssetIntegrityState::PresentValid);
    }

    #[test]
    fn verify_reports_invalid_checksum() {
        let home = temp_dir("models-verify-bad-checksum");
        let asset_path = home.join("models/test/model.bin");
        fs::create_dir_all(asset_path.parent().expect("parent")).expect("mkdir");
        fs::write(&asset_path, b"tiny-model").expect("write model");

        let asset = test_asset(
            "model.bin",
            "models/test/model.bin",
            b"tiny-model".len() as u64,
            Some("0000000000000000000000000000000000000000000000000000000000000000"),
        );
        let integrity = verify_existing_asset(&asset_path, &asset, true).expect("verify existing");
        assert_eq!(integrity, AssetIntegrityState::PresentInvalidChecksum);
    }

    #[test]
    fn verify_reports_invalid_size() {
        let home = temp_dir("models-verify-bad-size");
        let asset_path = home.join("models/test/model.bin");
        let body = b"tiny-model";
        fs::create_dir_all(asset_path.parent().expect("parent")).expect("mkdir");
        fs::write(&asset_path, body).expect("write model");

        let asset = test_asset(
            "model.bin",
            "models/test/model.bin",
            body.len() as u64 + 1,
            None,
        );
        let integrity = verify_existing_asset(&asset_path, &asset, true).expect("verify existing");
        assert_eq!(integrity, AssetIntegrityState::PresentInvalidSize);
    }

    #[test]
    fn verify_reports_unknown_checksum_when_missing() {
        let home = temp_dir("models-verify-unknown-checksum");
        let asset_path = home.join("models/test/model.bin");
        let body = b"tiny-model";
        fs::create_dir_all(asset_path.parent().expect("parent")).expect("mkdir");
        fs::write(&asset_path, body).expect("write model");

        let asset = test_asset(
            "model.bin",
            "models/test/model.bin",
            body.len() as u64,
            None,
        );
        let integrity = verify_existing_asset(&asset_path, &asset, true).expect("verify existing");
        assert_eq!(integrity, AssetIntegrityState::UnknownChecksum);
    }

    #[test]
    fn fetch_rejects_invalid_checksum_without_promoting_temp_file() {
        let body = b"tiny-model".to_vec();
        let server_body = body.clone();
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let url = format!("http://{}/model.bin", listener.local_addr().expect("addr"));
        let server = thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().expect("accept request");
                let request = read_request(&mut stream);
                if request.starts_with("HEAD ") {
                    write_response(
                        &mut stream,
                        "HTTP/1.1 200 OK",
                        &[("Content-Length", &server_body.len().to_string())],
                        &[],
                    );
                    continue;
                }
                assert!(request.starts_with("GET "), "unexpected request: {request}");
                write_response(
                    &mut stream,
                    "HTTP/1.1 200 OK",
                    &[("Content-Length", &server_body.len().to_string())],
                    &server_body,
                );
            }
        });

        let home = temp_dir("models-reject-invalid-checksum");
        let asset_path = home.join("models/test/model.bin");
        fs::create_dir_all(asset_path.parent().expect("parent")).expect("mkdir");
        let asset = ModelAsset {
            id: "test-asset",
            filename: "model.bin",
            relative_path: "models/test/model.bin",
            url: Box::leak(url.into_boxed_str()),
            expected_size_bytes: Some(body.len() as u64),
            sha256: Some("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"),
            license: Some("unknown"),
            source: Some("fixture-server"),
        };

        let result = fetch_asset_with_progress(&home, &asset, |_, _| {});
        assert!(result.is_err(), "checksum mismatch should fail");
        assert!(
            !asset_path.exists(),
            "invalid download should not become final"
        );

        let model_dir = asset_path.parent().expect("parent");
        let has_part = fs::read_dir(model_dir)
            .expect("read model dir")
            .filter_map(Result::ok)
            .any(|entry| entry.file_name().to_string_lossy().ends_with(".part"));
        assert!(!has_part, "checksum mismatch should clean partial files");
        server.join().expect("server");
    }

    #[test]
    fn verify_reports_missing_when_file_absent() {
        let home = temp_dir("models-verify-missing");
        let asset_path = home.join("models/test/model.bin");
        // Deliberately do NOT create the file.

        let asset = test_asset(
            "model.bin",
            "models/test/model.bin",
            10,
            Some("ccdbfb9993be88c536b0b7cd2abe60eda83c7ce1ad530c6a2ada81510ff1548c"),
        );
        let integrity = verify_existing_asset(&asset_path, &asset, true).expect("verify existing");
        assert_eq!(integrity, AssetIntegrityState::Missing);
    }

    #[test]
    fn verify_reports_present_unverified_without_verify_flag() {
        let home = temp_dir("models-verify-unverified");
        let asset_path = home.join("models/test/model.bin");
        let body = b"tiny-model";
        fs::create_dir_all(asset_path.parent().expect("parent")).expect("mkdir");
        fs::write(&asset_path, body).expect("write model");

        let asset = test_asset(
            "model.bin",
            "models/test/model.bin",
            body.len() as u64,
            Some("ccdbfb9993be88c536b0b7cd2abe60eda83c7ce1ad530c6a2ada81510ff1548c"),
        );
        // verify=false → should report PresentUnverified even if checksum would match.
        let integrity = verify_existing_asset(&asset_path, &asset, false).expect("verify existing");
        assert_eq!(integrity, AssetIntegrityState::PresentUnverified);
    }

    fn test_asset(
        filename: &'static str,
        relative_path: &'static str,
        expected_size_bytes: u64,
        sha256: Option<&'static str>,
    ) -> ModelAsset {
        ModelAsset {
            id: "test-asset",
            filename,
            relative_path,
            url: "http://127.0.0.1:9/unreachable",
            expected_size_bytes: Some(expected_size_bytes),
            sha256,
            license: Some("test"),
            source: Some("tests"),
        }
    }

    fn read_request(stream: &mut TcpStream) -> String {
        let mut request = Vec::new();
        let mut buffer = [0; 1024];
        loop {
            let read = stream.read(&mut buffer).expect("read request");
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..read]);
            if request.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
        String::from_utf8(request).expect("utf8 request")
    }

    fn request_range_start(request: &str) -> usize {
        request
            .lines()
            .filter_map(|line| line.split_once(':'))
            .find_map(|(name, value)| {
                name.eq_ignore_ascii_case("range")
                    .then(|| value.trim().strip_prefix("bytes="))
                    .flatten()
            })
            .expect("range header")
            .trim()
            .trim_end_matches('-')
            .parse::<usize>()
            .expect("range start")
    }

    fn write_response(stream: &mut TcpStream, status: &str, headers: &[(&str, &str)], body: &[u8]) {
        write!(stream, "{status}\r\n").expect("write status");
        for (name, value) in headers {
            write!(stream, "{name}: {value}\r\n").expect("write header");
        }
        write!(stream, "Connection: close\r\n\r\n").expect("write terminator");
        stream.write_all(body).expect("write body");
    }
}
