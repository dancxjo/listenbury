mod espeak_ng;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(author, version, about = "Development tasks for Listenbury")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    EspeakNg(espeak_ng::EspeakNgCommand),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::EspeakNg(cmd) => espeak_ng::run(cmd),
    }
}
