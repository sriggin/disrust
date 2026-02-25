# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`disrust` is a high-performance TCP inference server built with Rust, using io_uring for asynchronous I/O and the LMAX disruptor pattern for lock-free inter-thread communication. The server processes fixed-size feature vectors (128 f32 values) and returns inference results with minimal latency.

## Build and Run Commands

```bash
# Build release binary
cargo build --release

# Run server (auto-detects CPU count for IO threads)
./target/release/disrust

# Run server with specific thread count and port
./target/release/disrust <num_io_threads> <port>
# Example: ./target/release/disrust 4 9900

# Build and run debug build
cargo run

# Run client smoke test
cargo run --bin client

# Run client pipeline test
cargo run --bin client -- 9900 pipeline

# Run client benchmark
cargo run --bin client -- 9900 bench <num_connections> <requests_per_conn>
# Example: cargo run --bin client -- 9900 bench 8 100000

# Run tests (buffer_pool.rs contains tests)
cargo test

# Run buffer pool benchmarks
cargo bench --bench buffer_pool_bench -- --alloc-sizes
cargo bench --bench buffer_pool_bench -- --pool-sizes

# Profile buffer pool with perf (Linux only, runs 100M iterations)
cargo bench --no-run
perf record -g target/release/deps/profile_buffer_pool-*
perf report

# Run with metrics (deltas every 10s to stdout; low-overhead atomics)
cargo build --release --features metrics
./target/release/disrust --port 9900
```

## Metrics (optional, `--features metrics`)

When built with `--features metrics`, a background thread prints **deltas every 10 seconds to stdout** with minimal overhead (atomic counters only, no locks or allocations in the hot path). Output includes:

- **Throughput:** `published` (requests enqueued to disruptor), `sent` (responses written to clients)
- **Stalls:** `req_ring_full`, `resp_ring_full`, `pool_exh`, `pool_too_large` (deltas = occurrences in the interval)
- **Batch processor utilization:** `poll_events` vs `poll_no_events` and `stall_pct` (fraction of poll cycles with no events; high = processor often spinning idle)
- **Gauges:** current and max in-flight (`req_occ`, `resp_occ`, `req_max`, `resp_max`), `pool_max_in_use`

Use this to spot backpressure, underutilization, or pool exhaustion without paying for a full metrics stack.

## Architecture Overview

### Threading Model

The server uses a multi-threaded architecture with clear separation of concerns:

- **IO Threads** (N threads, one per CPU core - 1): Handle network I/O using io_uring, parse protocol, publish requests to disruptor, write responses
- **Batch Processor** (1 thread): Consumes requests from disruptor, runs inference (currently POC: sum of feature vectors), publishes responses back to IO threads

### Communication Flow

1. **Request path** (MPSC): IO threads → Batch processor
   - Uses multi-producer disruptor (`build_multi_producer`)
   - IO threads publish `InferenceEvent` to shared ring buffer
   - Batch processor polls events with busy-spin wait strategy

2. **Response path** (SPSC per IO thread): Batch processor → IO thread
   - Each IO thread has dedicated single-producer response channel
   - Batch processor writes to per-thread response queue
   - Uses `eventfd` to wake io_uring when responses are ready

### Key Design Patterns

- **io_uring**: All I/O operations (accept, read, write, eventfd) use io_uring submission/completion queues for zero-copy async I/O
- **SO_REUSEPORT**: Multiple IO threads bind to the same port, kernel load-balances incoming connections
- **Disruptor Pattern**: Lock-free ring buffers with busy-spin for ultra-low latency
- **Slab Allocator**: Connection state stored in `Slab<Connection>` for efficient allocation/deallocation
- **Buffer Pool**: Per-thread ring-based buffer pools allocate variable-length feature data. Each IO thread has a dedicated `BufferPool` (currently 2GB worst-case sizing, but can be right-sized to 8-32 MB for typical workloads). Features are allocated as `PoolSlice` which automatically returns memory when dropped. Pool size critically affects performance due to cache locality - see PERFORMANCE.md

### Protocol

Wire format is little-endian binary:

**Request**: `[u32 num_vectors][f32 * num_vectors * FEATURE_DIM]`
- `num_vectors`: 1-64 vectors per request
- Each vector: 128 f32 values (FEATURE_DIM = 128)

**Response**: `[u32 num_vectors][f32 * num_vectors]`
- One f32 result per input vector

### Critical Constants

- `DISRUPTOR_SIZE: 65536` - Request ring buffer size
- `RESPONSE_QUEUE_SIZE: 8192` - Per-thread response queue size
- `FEATURE_DIM: 128` - Fixed feature vector dimension
- `MAX_VECTORS_PER_REQUEST: 64` - Protocol limit
- `READ_BUF_SIZE: 65536` - Per-connection read buffer
- `BUFFER_POOL_CAPACITY: DISRUPTOR_SIZE * MAX_VECTORS_PER_REQUEST * FEATURE_DIM` - Per-thread buffer pool size (~2GB). **Note:** This is worst-case sizing assuming all in-flight requests are at max size. Real workloads are typically 1-8 vectors, so this could be reduced to 8-32 MB for significant performance gains (2.5x speedup due to better cache locality). Minimum: `DISRUPTOR_SIZE * FEATURE_DIM` (~32MB)

## File Organization

- `main.rs`: Entry point, spawns IO threads and batch processor, creates sockets with SO_REUSEPORT
- `io_thread.rs`: IoThread implementation, io_uring event loop, connection management
- `batch_processor.rs`: BatchProcessor implementation, consumes requests and produces responses
- `protocol.rs`: Wire protocol parsing and serialization (try_parse_request, copy_features, write_response)
- `ring_types.rs`: Disruptor event types (InferenceEvent, InferenceResponse) and constants
- `response_queue.rs`: SPSC response channel builder and eventfd signaling
- `buffer_pool.rs`: Ring-based buffer pool with automatic memory reclamation via `PoolSlice`/`PoolSliceMut` RAII types. Each IO thread has a dedicated pool instance
- `bin/client.rs`: Test client with smoke test, pipeline test, and benchmark modes

## Important Implementation Details

### io_uring Operations

Four operation types encoded in user_data field:
- `OP_ACCEPT`: Accept new connections on listen socket
- `OP_READ`: Read from client connection
- `OP_WRITE`: Write response to client
- `OP_EVENTFD`: Read from eventfd (signals responses available)

Use `push_sqe()` helper which handles submission queue full by flushing to kernel.

### Connection State Machine

Each connection tracks:
- `read_inflight/write_inflight`: Prevent duplicate submissions
- `read_buf`: Accumulates partial protocol messages
- `write_buf`: Queues responses for transmission
- `next_request_seq`: Per-connection sequence number for ordering

### Request Parsing

`protocol::try_parse_request()` returns:
- `Complete`: Full request parsed, returns num_vectors and bytes_consumed
- `Incomplete`: Need more data, returns minimum bytes needed
- `Error`: Protocol violation (num_vectors == 0 or > MAX)

Multiple requests can be pipelined in a single read buffer; parse loop consumes all complete requests, then compacts remaining data.

### Response Signaling

After publishing responses, batch processor calls `response_producers[thread_id].signal()` which writes to eventfd. This wakes io_uring on the IO thread via OP_EVENTFD completion, which then drains the response queue.

### Buffer Pool Memory Management

Each IO thread has a dedicated `Arc<BufferPool>` with ring-based allocation:
- **Allocation**: IO thread calls `pool.alloc(len)` returning `PoolSliceMut` for writing feature data
- **Freezing**: `PoolSliceMut::freeze()` converts to immutable `PoolSlice` which is stored in `InferenceEvent`
- **Automatic Reclamation**: When `PoolSlice` is dropped (disruptor slot overwritten), `Drop` impl advances read cursor to free space
- **Single-threaded**: Despite using `Arc`, all pool operations happen on one IO thread. Uses `Relaxed` atomic ordering
- **Cache Locality**: Pool size critically affects performance. Small pools (~8-32 MB) are 2.5x faster than current 2GB sizing due to cache behavior

See PERFORMANCE.md for detailed benchmarks and right-sizing opportunities.

## Development Notes

- Edition 2024 Rust required
- Release profile uses LTO and single codegen unit for maximum performance
- The current "inference" is a POC placeholder (sum of feature vector) - replace `batch_processor.rs` logic for real ML models
- SO_REUSEPORT requires Linux kernel support
- io_uring requires Linux 5.1+ (use 5.6+ for best stability)
