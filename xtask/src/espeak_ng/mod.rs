mod convert;
mod dictionary;
mod fetch;
mod inventory;
mod profile;
mod provenance;
mod rules;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Subcommand, ValueEnum};

#[derive(Debug, Args)]
pub struct EspeakNgCommand {
    #[command(subcommand)]
    command: EspeakNgSubcommand,
}

#[derive(Debug, Subcommand)]
enum EspeakNgSubcommand {
    /// Clone or update the local eSpeak-ng source cache.
    Fetch {
        #[arg(long)]
        rev: Option<String>,
    },
    /// Show local eSpeak-ng source cache status.
    Status,
    /// Inventory eSpeak-ng source files for a language.
    Inventory {
        #[arg(long, default_value = "en")]
        lang: String,
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Convert language/voice profile files.
    ConvertProfiles {
        #[arg(long, default_value = "en")]
        lang: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Convert dictionary/list metadata.
    ConvertList {
        #[arg(long, default_value = "en")]
        lang: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Convert rules into inventory or native subset.
    ConvertRules {
        #[arg(long, default_value = "en")]
        lang: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_enum, default_value_t = RulesMode::Inventory)]
        mode: RulesMode,
    },
    /// Run all eSpeak-ng converters.
    Convert {
        #[arg(long, default_value = "en")]
        lang: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Regenerate deterministic eSpeak-derived output in default location.
    Regen {
        #[arg(long, default_value = "en")]
        lang: String,
    },
    /// Diff regenerated output against existing generated files.
    Diff {
        #[arg(long, default_value = "en")]
        lang: String,
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum RulesMode {
    Inventory,
    NativeSubset,
}

pub fn run(cmd: EspeakNgCommand) -> Result<()> {
    match cmd.command {
        EspeakNgSubcommand::Fetch { rev } => fetch::fetch(rev.as_deref()),
        EspeakNgSubcommand::Status => fetch::status(),
        EspeakNgSubcommand::Inventory { lang, json } => {
            inventory::inventory(&lang, json.as_deref())
        }
        EspeakNgSubcommand::ConvertProfiles { lang, out } => profile::convert_profiles(&lang, &out),
        EspeakNgSubcommand::ConvertList { lang, out } => dictionary::convert_list(&lang, &out),
        EspeakNgSubcommand::ConvertRules { lang, out, mode } => {
            rules::convert_rules(&lang, &out, mode)
        }
        EspeakNgSubcommand::Convert { lang, out } => convert::convert_all(&lang, &out),
        EspeakNgSubcommand::Regen { lang } => convert::regen(&lang),
        EspeakNgSubcommand::Diff { lang, out } => convert::diff(&lang, out.as_deref()),
    }
}
