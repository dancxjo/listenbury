use anyhow::{Context, Result};

use crate::cli::MbrolaRenderCommand;

pub(crate) fn run_mbrola_render(command: MbrolaRenderCommand) -> Result<()> {
    let report = listenbury::voice::mbrola::render::render_raw_pho(
        None,
        command.voice,
        &command.phones,
        &command.out,
    )
    .with_context(|| {
        format!(
            "failed to render MBROLA .pho {} to {}",
            command.phones.display(),
            command.out.display()
        )
    })?;

    println!(
        "Rendered {} phones / {} ms with {} voice {} to {}",
        report.phone_count,
        report.duration_ms,
        report.backend,
        report.voice_name,
        report.out_wav.display()
    );
    Ok(())
}
