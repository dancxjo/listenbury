use anyhow::Result;
use std::process::{Command, Stdio};

use crate::cli::WebCommand;

pub(crate) fn run_web(command: WebCommand) -> Result<()> {
    let url = format!("http://{}:{}/", command.host, command.port);
    if command.open {
        let open_url = url.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(350));
            open_browser(&open_url);
        });
    }

    listenbury::web::serve(listenbury::web::ServeConfig {
        host: command.host,
        port: command.port,
        payload: command.payload,
        trace: command.trace,
        broadcaster: None,
        live_audio: None,
    })
}

fn open_browser(url: &str) {
    #[cfg(target_os = "macos")]
    let mut command = {
        let mut command = Command::new("open");
        command.arg(url);
        command
    };
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut command = Command::new("cmd");
        command.arg("/C").arg("start").arg("").arg(url);
        command
    };
    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    let mut command = {
        let mut command = Command::new("xdg-open");
        command.arg(url);
        command
    };

    match command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
    {
        Ok(output) if output.status.success() => {}
        Ok(output) => {
            let detail = String::from_utf8_lossy(&output.stderr);
            if let Some(detail) = concise_opener_error(&detail) {
                eprintln!(
                    "Unable to open browser automatically; open {url} manually. Opener reported: {detail}"
                );
            } else {
                eprintln!("Unable to open browser automatically; open {url} manually.");
            }
        }
        Err(error) => {
            eprintln!("Unable to open browser automatically; open {url} manually: {error}");
        }
    }
}

fn concise_opener_error(stderr: &str) -> Option<String> {
    let line = stderr
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty())?;
    Some(line.chars().take(240).collect())
}

#[cfg(test)]
mod tests {
    use super::concise_opener_error;

    #[test]
    fn opener_error_uses_last_non_empty_line() {
        let stderr = "Unable to connect to VS Code server\n\nxdg-open: no method available\n";

        assert_eq!(
            concise_opener_error(stderr).as_deref(),
            Some("xdg-open: no method available")
        );
    }
}
