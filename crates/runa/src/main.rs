//! `runa` command-line binary: clap CLI, figment config, output, server.
//! Skeleton for P0.1: `--version` plus a `doctor` stub.

use clap::Command;

fn cli() -> Command {
    Command::new("runa")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Run AI models locally (GGUF) or via OpenAI/Anthropic APIs")
        .arg_required_else_help(true)
        .subcommand(
            Command::new("doctor")
                .about("Probe hardware and compiled backends (stub in P0.1)"),
        )
}

fn main() {
    let matches = cli().get_matches();
    match matches.subcommand_name() {
        Some("doctor") => {
            println!("runa doctor: stub (P0.1) — hardware probe lands in P1.6");
        }
        _ => unreachable!("clap handles help/version/errors"),
    }
}
