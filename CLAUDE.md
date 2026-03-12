# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`disrust` is a high-performance TCP inference server built with Rust, using io_uring for asynchronous I/O and the LMAX disruptor pattern for lock-free inter-thread communication. The server processes fixed-size feature vectors (16 f32 values per vector; see `constants::FEATURE_DIM`) and runs ONNX inference through ONNX Runtime with the CUDA execution provider.

## Dev Environment

io_uring and SO_REUSEPORT require Linux. On Linux, run `cargo` commands directly. On macOS, prefix every `cargo` command with `docker exec <container>` where `<container>` is the dev container name (randomly assigned by Docker — find it with `docker ps`). The container is started via `docker-compose.yml`.

## Run Commands

```bash
# Start server (default port 9900)
./target/release/disrust
./target/release/disrust --port 9900

# Run with metrics (deltas every 10s to stdout)
cargo build --release --features metrics && ./target/release/disrust --model model.onnx

# Client smoke / pipeline / benchmark
cargo run --bin client
cargo run --bin client -- 9900 pipeline
cargo run --bin client -- 9900 bench <num_connections> <requests_per_conn>

# Benchmarks
cargo bench --bench buffer_pool_bench -- --alloc-sizes
cargo bench --bench buffer_pool_bench -- --pool-sizes
```

## Architecture

### Threading Model

Currently **1 ingress IO thread + 2 ONNX consumer threads**. The disruptor is SPSC (`build_single_producer`). Expanding to N IO threads requires switching to `build_multi_producer` and revisiting the single-producer assumptions in the ingress and GPU consumer pipeline.

- **Ingress IO Thread**: io_uring event loop for accept/read only. Parses protocol and publishes `InferenceEvent`s.
- **Submission Consumer**: batches `InferenceEvent`s and submits ONNX Runtime runs.
- **Completion Consumer**: waits for completion, serializes responses, and writes directly to client sockets via its own io_uring ring.

### Communication Flow

1. **Request path** (ingress IO thread → submission consumer): `request_flow::process_requests_from_buffer()` parses bytes, allocates pool space, and publishes `InferenceEvent`s to the SPSC disruptor.
2. **Completion path** (completion consumer → socket): the completion consumer reads finished outputs and submits direct socket writes from its own io_uring ring.

### io_uring Wrapper

`src/bin/disrust_gpu/io_thread_gpu.rs` defines a local `IoUring` struct (shadows the crate type) that wraps `io_uring::IoUring`. The rest of the file interacts only with the local wrapper:
- `push(&mut self, sqe: &Entry)` — submits an SQE, flushing the SQ to the kernel if full
- `wait(&mut self, n: usize)` — `submit_and_wait`
- `drain_cqes(&mut self) -> Vec<(u64, i32)>` — collects eagerly to release the CQ borrow before any SQE submissions in the same iteration
- The ingress thread encodes `OP_ACCEPT` and `OP_READ` in `user_data`; the completion consumer owns the write side separately.

### Connection State

Each `Connection` owns:
- `read_buf` / `read_len` — accumulates partial reads; compacted after each parse pass
- `read_inflight` — prevents duplicate SQE submissions
- `next_request_seq` — per-connection sequence counter

`Connection` has a `Drop` impl that closes the fd. Remove from `Slab` is sufficient to close — no manual `libc::close` needed at call sites.

### Buffer Pool

`BufferPool` is a ring-based arena over pinned host memory. The ingress IO thread holds a `&'static BufferPool` (via `from_raw_ptr`) and the ONNX path reads directly from those allocations.

- `alloc(len) -> Result<PoolSliceMut, AllocError>` — advances write cursor; wraps to 0 if allocation would straddle the end
- `PoolSliceMut::freeze() -> PoolSlice` — makes immutable; stored in `InferenceEvent`
- `PoolSlice::Drop` — advances read cursor, reclaiming space

Pool size critically affects performance due to cache locality (see PERFORMANCE.md).

### lib/binary Split

`lib.rs` exports shared request-path and GPU runtime code. The binary-specific io_uring ingress thread lives under `src/bin/disrust_gpu`.

### Protocol

Wire format is little-endian binary:

- **Request**: `[u32 num_vectors][f32 × num_vectors × FEATURE_DIM]`
- **Response**: `[u8 num_vectors][f32 × num_vectors]`

`protocol::try_parse_request()` returns `Complete { num_vectors, bytes_consumed }`, `Incomplete`, or `Error`. Multiple requests may be pipelined; the parse loop in `request_flow` consumes all complete requests per read.

### Critical Constants

- `config.rs`: `GPU_DISRUPTOR_SIZE`, `GPU_BUFFER_POOL_CAPACITY`, `SESSION_POOL_SIZE`, `MAX_SESSION_BATCH_SIZE`, `MAX_IO_THREADS`, `READ_BUF_SIZE`, `SLAB_CAPACITY`
- `constants.rs`: `FEATURE_DIM = 16`, `MAX_VECTORS_PER_REQUEST = 64`

## Metrics (`--features metrics`)

Background thread prints deltas every 10s:
- **Throughput**: `published` (requests published to the disruptor)
- **Stalls**: `req_ring_full`, `pool_exh`, `pool_too_large`
- **Gauges**: `req_occ`, `req_max`, `pool_max_in_use`

## Notes

- Edition 2024 Rust required
- Release profile: LTO + single codegen unit
- io_uring requires Linux 5.1+ (5.6+ recommended)
