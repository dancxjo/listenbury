use std::sync::Arc;

use anyhow::Result;
use listenbury::SystemClock;
use listenbury::playback_check::run_playback_check;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let events = run_playback_check(Arc::new(SystemClock), |_| {})?;
    for event in events {
        println!(
            "{:?}\t{}\t{:?}",
            event.kind, event.at.unix_nanos, event.text
        );
    }

    Ok(())
}
