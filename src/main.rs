use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(about = "High-performance ONNX inference tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Serve(disrust::server::ServeArgs),
    Verify(disrust::verify::VerifyArgs),
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => disrust::server::run(args),
        Command::Verify(args) => disrust::verify::run(args),
    }
}
