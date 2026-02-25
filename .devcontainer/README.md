# Dev Container for disrust

## What’s included

- **Image:** `rust:bookworm` with `rustfmt`, `clippy`, `gdb`, and a non-root `vscode` user.
- **Run args:** `seccomp=unconfined` so io_uring works when you run the server inside the container.
- **VS Code:** Rust Analyzer extension is recommended via `devcontainer.json`.

## What you need to do

1. **Open in Dev Container (Cursor / VS Code)**  
   - Install the “Dev Containers” extension if you don’t have it.  
   - Open this repo, then run **“Dev Containers: Reopen in Container”** from the command palette (or use the prompt when opening the folder).  
   - The first open will build the image; later opens reuse it.

2. **Use the container for development**  
   - Terminal, build, test, and run commands all execute inside the container (Linux).  
   - `cargo build`, `cargo test`, `cargo run`, `cargo run --bin client`, etc. work as in CLAUDE.md.

3. **Running the server inside the container**  
   - The dev container is started with `--security-opt seccomp=unconfined` so io_uring is allowed.  
   - You can run `./target/release/disrust` or `cargo run` and use the server on the configured port (e.g. 9900).  
   - Forward the port in Cursor/VS Code if you want to access it from the host.

## Optional: pin Rust version

To require Rust 1.85+ (for edition 2024) when using rustup locally or in CI, add a `rust-toolchain.toml` in the repo root:

```toml
[toolchain]
channel = "1.85"
```

The dev container image uses the Rust version provided by `rust:bookworm`; the Dockerfile can be changed to a specific version tag (e.g. `rust:1.85-bookworm`) if needed.
