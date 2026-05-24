use std::fs;

use anyhow::{Context, Result};

use crate::cli::ProsodyPlanCommand;

pub(crate) fn run_prosody_plan(command: ProsodyPlanCommand) -> Result<()> {
    let alignment_json = fs::read_to_string(&command.alignment_json).with_context(|| {
        format!(
            "read forced-alignment JSON {}",
            command.alignment_json.display()
        )
    })?;
    let praat_json = fs::read_to_string(&command.praat_json)
        .with_context(|| format!("read Praat prosody JSON {}", command.praat_json.display()))?;

    let alignment = listenbury::forced_alignment_from_json(&alignment_json)?;
    let praat = listenbury::praat_analysis_from_json(&praat_json)?;
    let plan = listenbury::plan_prosody_timing(
        alignment,
        praat,
        &listenbury::ProsodyTimingConfig::default(),
    );
    let synthetic_plan = listenbury::synthetic_plan_from_prosody_timing(&plan);

    let plan_json = serde_json::to_string_pretty(&plan).context("serialize prosody timing plan")?;
    fs::write(&command.output_json, plan_json).with_context(|| {
        format!(
            "write prosody timing plan {}",
            command.output_json.display()
        )
    })?;

    if let Some(ssml_output) = command.ssml_output {
        fs::write(&ssml_output, listenbury::prosody_plan_to_ssml(&plan))
            .with_context(|| format!("write SSML {}", ssml_output.display()))?;
    }

    println!(
        "wrote prosody plan with {} words, {} breath groups to {}",
        synthetic_plan.segments.len(),
        synthetic_plan.breath_groups.len(),
        command.output_json.display()
    );
    Ok(())
}
