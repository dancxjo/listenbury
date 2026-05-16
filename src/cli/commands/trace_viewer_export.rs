use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

use anyhow::{Context, Result};

use crate::cli::TraceViewerExportCommand;

pub(crate) fn run_trace_viewer_export(command: TraceViewerExportCommand) -> Result<()> {
    let input = File::open(&command.input_jsonl)
        .with_context(|| format!("open live trace JSONL at {}", command.input_jsonl.display()))?;
    let reader = BufReader::new(input);
    let payload =
        listenbury::trace::viewer_payload::live_trace_jsonl_reader_to_viewer_payload(reader)
            .with_context(|| {
                format!(
                    "convert live trace JSONL {} into viewer payload",
                    command.input_jsonl.display()
                )
            })?;

    if let Some(parent) = command
        .output_json
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create output directory {}", parent.display()))?;
    }

    let output = File::create(&command.output_json).with_context(|| {
        format!(
            "create viewer payload output file {}",
            command.output_json.display()
        )
    })?;
    let mut writer = BufWriter::new(output);
    serde_json::to_writer_pretty(&mut writer, &payload).with_context(|| {
        format!(
            "serialize viewer payload to {}",
            command.output_json.display()
        )
    })?;
    writer
        .write_all(b"\n")
        .with_context(|| format!("finalize {}", command.output_json.display()))?;
    writer
        .flush()
        .with_context(|| format!("flush {}", command.output_json.display()))?;

    println!(
        "wrote viewer payload with {} stream lanes, {} events, {} markers to {}",
        payload.streams.len(),
        payload.events.len(),
        payload.markers.len(),
        command.output_json.display()
    );
    Ok(())
}
