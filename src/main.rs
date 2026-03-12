#[cfg(feature = "cuda")]
fn main() {
    disrust::server::run();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("disrust requires the `cuda` feature");
    std::process::exit(1);
}
