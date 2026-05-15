use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use listenbury::models::FetchProgress;

pub(crate) struct DownloadProgress {
    bars: MultiProgress,
    overall: ProgressBar,
    download_style: ProgressStyle,
    asset_bars: HashMap<&'static str, ProgressBar>,
    finished: bool,
}

impl DownloadProgress {
    pub(crate) fn new(message: impl Into<String>) -> Result<Self> {
        let bars = MultiProgress::new();
        let overall = bars.add(ProgressBar::new(0));
        let overall_style = ProgressStyle::with_template(
            "{spinner:.cyan} {msg} [{wide_bar:.cyan/blue}] {pos}/{len} ETA {eta_precise}",
        )
        .context("failed to create overall progress style")?
        .progress_chars("=>-");
        overall.set_style(overall_style);
        overall.enable_steady_tick(Duration::from_millis(100));
        overall.set_message(message.into());

        let download_style = ProgressStyle::with_template(
            "{spinner:.cyan} {msg} [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} {bytes_per_sec} ETA {eta_precise}",
        )
        .context("failed to create download progress style")?
        .progress_chars("=>-");

        Ok(Self {
            bars,
            overall,
            download_style,
            asset_bars: HashMap::new(),
            finished: false,
        })
    }

    pub(crate) fn update(&mut self, asset_progress: FetchProgress) {
        self.overall.set_length(asset_progress.asset_count as u64);
        self.overall.set_position(
            self.overall
                .position()
                .max(asset_progress.asset_index as u64),
        );
        self.overall
            .set_message(format!("Fetching {}...", asset_progress.asset_id));

        let progress = self
            .asset_bars
            .entry(asset_progress.asset_id)
            .or_insert_with(|| {
                let progress = self.bars.add(ProgressBar::new(0));
                progress.set_style(self.download_style.clone());
                progress.enable_steady_tick(Duration::from_millis(100));
                progress
            });
        match asset_progress.total_bytes {
            Some(total_bytes) => progress.set_length(total_bytes),
            None => progress.unset_length(),
        }
        progress.set_position(asset_progress.downloaded_bytes);
        progress.set_message(format!(
            "{} -> {}",
            asset_progress.asset_id,
            asset_progress.path.display()
        ));
    }

    pub(crate) fn finish_and_clear(&mut self) {
        self.overall
            .set_position(self.overall.length().unwrap_or(0));
        self.overall.finish_and_clear();
        for progress in self.asset_bars.values() {
            progress.finish_and_clear();
        }
        self.finished = true;
    }
}

impl Drop for DownloadProgress {
    fn drop(&mut self) {
        if !self.finished {
            self.finish_and_clear();
        }
    }
}
