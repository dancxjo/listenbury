use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};

use crate::models::manifest::ModelAsset;
use crate::models::paths::asset_path;

pub fn fetch_asset(home: &Path, asset: &ModelAsset) -> Result<bool> {
    fetch_asset_with_progress(home, asset, |_, _| {})
}

pub fn fetch_asset_with_progress(
    home: &Path,
    asset: &ModelAsset,
    mut progress: impl FnMut(u64, Option<u64>),
) -> Result<bool> {
    let target_path = asset_path(home, asset);
    let parent = target_path
        .parent()
        .context("asset path had no parent directory")?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create model directory {}", parent.display()))?;

    let existing_bytes = file_len(&target_path)?;
    let remote_bytes = remote_content_length(asset).or(asset.expected_size_hint);
    if let Some(existing_bytes) = existing_bytes {
        if existing_bytes > 0 {
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

    fs::rename(&temp_path, &target_path).with_context(|| {
        format!(
            "failed to move completed download {} to {}",
            temp_path.display(),
            target_path.display()
        )
    })?;
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

fn finalize_download(
    asset: &ModelAsset,
    download_path: &Path,
    target_path: &Path,
    total_bytes: Option<u64>,
) -> Result<()> {
    let Some(total_bytes) = total_bytes else {
        return Ok(());
    };
    let actual_bytes = file_len(download_path)?.unwrap_or(0);
    if actual_bytes != total_bytes {
        bail!(
            "downloaded file {} is {actual_bytes} bytes, expected {total_bytes} bytes",
            download_path.display()
        );
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

    use super::fetch_asset_with_progress;
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
            expected_size_hint: None,
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
            expected_size_hint: None,
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
