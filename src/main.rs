use anyhow::Result;
use clap::Parser;
use silero::Silero;
use std::path::PathBuf;

#[derive(Parser)]
struct Opts {
    #[clap(short, long)]
    input: Vec<PathBuf>,
    #[clap(short, long)]
    output_dir: Option<PathBuf>,
}

fn main() -> Result<()> {
    env_logger::init();
    let opts = Opts::parse();
    let silero = Silero::default()?;
    let output_dir = opts.output_dir.unwrap_or_default();
    silero.stt(&opts.input, &output_dir)?;
    Ok(())
}
