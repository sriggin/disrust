#[cfg(feature = "cuda")]
use clap::{Parser, Subcommand};

#[cfg(feature = "cuda")]
#[derive(Parser)]
#[command(about = "High-performance ONNX/CUDA inference tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[cfg(feature = "cuda")]
#[derive(Subcommand)]
enum Command {
    Serve(disrust::server::ServeArgs),
    Verify(disrust::verify::VerifyArgs),
}

#[cfg(feature = "cuda")]
fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => disrust::server::run(args),
        Command::Verify(args) => disrust::verify::run(args),
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("disrust requires the `cuda` feature");
    std::process::exit(1);
}
