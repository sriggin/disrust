# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`disrust` is a high-performance TCP inference server built with Rust, using io_uring for asynchronous I/O and the LMAX disruptor pattern for lock-free inter-thread communication. The server processes fixed-size feature vectors (16 f32 values per vector; see `constants::FEATURE_DIM`) and returns inference results with minimal latency.

## Dev Environment

io_uring and SO_REUSEPORT require Linux. On Linux, run `cargo` commands directly. On macOS, prefix every `cargo` command with `docker exec <container>` where `<container>` is the dev container name (randomly assigned by Docker — find it with `docker ps`). The container is started via `docker-compose.yml`.

## Run Commands

```bash
# Start server (default port 9900)
./target/release/disrust
./target/release/disrust --port 9900

# Run with metrics (deltas every 10s to stdout)
cargo build --release --features metrics && ./target/release/disrust

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

Currently **1 IO thread + 1 batch processor thread**. The disruptor is SPSC (`build_single_producer`). Expanding to N IO threads requires switching to `build_multi_producer`, making `IoThread::producer` a `MultiProducer` (Clone), and `BatchProcessor::poller` to `EventPoller<InferenceEvent, MultiProducerBarrier>`. Each IO thread already gets a dedicated response channel and `BufferPool`.

- **IO Thread**: io_uring event loop — accept, read, write, eventfd. Parses protocol, publishes to request disruptor, writes responses.
- **Batch Processor**: Busy-spins on request disruptor, runs inference (currently: sum of feature vector — replace in `batch_processor.rs`), publishes responses to per-thread response queues.

### Communication Flow

1. **Request path** (IO thread → batch processor): IO thread calls `request_flow::process_requests_from_buffer()`, which parses bytes, allocates pool space, and publishes `InferenceEvent` to the SPSC disruptor.
2. **Response path** (batch processor → IO thread): Batch processor publishes `InferenceResponse` to a per-thread SPSC queue, then calls `signal()` which writes to an `eventfd`. This triggers an `OP_EVENTFD` completion in the IO thread's ring, which drains the response queue and submits writev.

### io_uring Wrapper

`io_thread.rs` defines a local `IoUring` struct (shadows the crate type) that wraps `io_uring::IoUring`. The rest of the file interacts only with the local wrapper:
- `push(&mut self, sqe: &Entry)` — submits an SQE, flushing the SQ to the kernel if full
- `wait(&mut self, n: usize)` — `submit_and_wait`
- `drain_cqes(&mut self) -> Vec<(u64, i32)>` — collects eagerly to release the CQ borrow before any SQE submissions in the same iteration
- `fd(&self) -> RawFd` — for future `MSG_RING` cross-thread posting

Four operations are encoded into the 64-bit `user_data` field (`OP_ACCEPT`, `OP_READ`, `OP_WRITE`, `OP_EVENTFD`) with the connection key in the low 16 bits.

### Connection State

Each `Connection` owns:
- `read_buf` / `read_len` — accumulates partial reads; compacted after each parse pass
- `write_headers` / `write_payloads` / `write_segments` — filled per-response before building iovecs
- `pending_iovecs` — scatter-gather list built by `build_iovecs()` for `Writev`
- `read_inflight` / `write_inflight` — prevent duplicate SQE submissions
- `next_request_seq` — per-connection sequence counter

`Connection` has a `Drop` impl that closes the fd. Remove from `Slab` is sufficient to close — no manual `libc::close` needed at call sites.

### Buffer Pool

`BufferPool` is a ring-based arena. Each IO thread holds a `&'static BufferPool` (via `leak_new`). The batch processor holds a separate result pool for large response payloads.

- `alloc(len) -> Result<PoolSliceMut, AllocError>` — advances write cursor; wraps to 0 if allocation would straddle the end
- `PoolSliceMut::freeze() -> PoolSlice` — makes immutable; stored in `InferenceEvent`
- `PoolSlice::Drop` — advances read cursor, reclaiming space

Pool size critically affects performance due to cache locality (see PERFORMANCE.md).

### lib/binary Split

`lib.rs` exports everything **except** `io_thread` and `config`. This keeps the library testable without io_uring. Integration tests and benchmarks drive `request_flow` and `response_flow` directly. `io_thread` is compiled only by the binary.

### Protocol

Wire format is little-endian binary:

- **Request**: `[u32 num_vectors][f32 × num_vectors × FEATURE_DIM]`
- **Response**: `[u8 num_vectors][f32 × num_vectors]`

`protocol::try_parse_request()` returns `Complete { num_vectors, bytes_consumed }`, `Incomplete`, or `Error`. Multiple requests may be pipelined; the parse loop in `request_flow` consumes all complete requests per read.

### Critical Constants

- `config.rs`: `DISRUPTOR_SIZE`, `RESPONSE_QUEUE_SIZE`, `BUFFER_POOL_CAPACITY`, `RESULT_POOL_CAPACITY`, `MAX_IO_THREADS`, `READ_BUF_SIZE`, `SLAB_CAPACITY`
- `constants.rs`: `FEATURE_DIM = 16`, `MAX_VECTORS_PER_REQUEST = 64`

## Metrics (`--features metrics`)

Background thread prints deltas every 10s:
- **Throughput**: `published` (to disruptor), `sent` (responses written)
- **Stalls**: `req_ring_full`, `resp_ring_full`, `pool_exh`, `pool_too_large`
- **Batch processor**: `poll_events` vs `poll_no_events`, `stall_pct`
- **Gauges**: `req_occ`, `resp_occ`, `req_max`, `resp_max`, `pool_max_in_use`

## Notes

- Edition 2024 Rust required
- Release profile: LTO + single codegen unit
- io_uring requires Linux 5.1+ (5.6+ recommended)
