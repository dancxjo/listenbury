use anyhow::Result;

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
    })
}

fn open_browser(url: &str) {
    #[cfg(target_os = "macos")]
    let mut command = std::process::Command::new("open");
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut command = std::process::Command::new("cmd");
        command.arg("/C").arg("start");
        command
    };
    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    let mut command = std::process::Command::new("xdg-open");

    match command.arg(url).spawn() {
        Ok(_) => {}
        Err(error) => eprintln!("failed to open browser for {url}: {error}"),
    }
}
