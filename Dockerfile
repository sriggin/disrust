FROM rust:slim AS builder
WORKDIR /app
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release && \
    cp target/release/disrust /disrust && \
    cp target/release/client /client

FROM debian:bookworm-slim
COPY --from=builder /disrust /usr/local/bin/disrust
COPY --from=builder /client /usr/local/bin/client
EXPOSE 9900
CMD ["disrust"]
