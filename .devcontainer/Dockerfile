# Rust on Debian bookworm (use recent tag for edition 2024, e.g. 1.85+)
FROM rust:bookworm

# Install dev tools and ensure non-root user for dev container
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdb \
    && rm -rf /var/lib/apt/lists/*

# Add vscode user (standard for dev containers)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Install Rust components (clippy, rustfmt, rust-analyzer optional)
RUN rustup component add rustfmt clippy

USER $USERNAME
WORKDIR /workspaces/disrust
