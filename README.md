# disrust

`disrust` is a high-throughput ONNX inference server built around:

- `io_uring` for network I/O
- a disruptor-style request path for inference admission
- a shared buffer pool for request features
- batched ONNX Runtime execution
- a decoupled response writer

The repo also includes a purpose-built load generator in [src/bin/client.rs](/home/sriggin/dev/sean/disrust/src/bin/client.rs) for smoke tests, pipelined runs, sustained load, and profiling.

## What This App Is

This is not a general web server or RPC framework.

It is a specialized TCP inference service optimized for:

- small fixed-shape requests
- high request rates
- aggressive batching into ONNX Runtime
- explicit performance measurement and bottleneck analysis

The core design goal is simple:

- keep the inference pipeline predictable
- keep hot-path allocations and synchronization under control
- make performance bottlenecks inspectable with built-in metrics and `perf`

## How It Works

At a high level, the server pipeline is:

1. One or more ingress `io-*` threads accept TCP connections and read request bytes with `io_uring`.
2. Ingress parses framed requests and copies features into the shared buffer pool.
3. Parsed requests are published as `InferenceEvent`s into a multi-producer disruptor ring.
4. A merged `inference` thread drains events, builds batches, submits ONNX Runtime work, retires completed batches, and encodes responses.
5. The writer thread flushes queued responses back to client sockets.

Important architectural points:

- ingress is sharded with `SO_REUSEPORT`
- the request path is multi-producer into a shared ring
- submission and completion are merged into one inference lane
- connection identity is logical and shard-aware, not based on raw file descriptors
- the buffer pool is global because batching and CUDA-backed execution want a single memory region
- response writing is decoupled from completion so writes can be coalesced per connection

## Current Structure

Main server/runtime areas:

- entrypoints:
  - [src/main.rs](/home/sriggin/dev/sean/disrust/src/main.rs)
  - [src/lib.rs](/home/sriggin/dev/sean/disrust/src/lib.rs)
- ingress/server wiring:
  - [src/server/mod.rs](/home/sriggin/dev/sean/disrust/src/server/mod.rs)
  - [src/server/ingress.rs](/home/sriggin/dev/sean/disrust/src/server/ingress.rs)
- inference pipeline:
  - [src/pipeline/inference.rs](/home/sriggin/dev/sean/disrust/src/pipeline/inference.rs)
  - [src/pipeline/writer.rs](/home/sriggin/dev/sean/disrust/src/pipeline/writer.rs)
  - [src/pipeline/connection_registry.rs](/home/sriggin/dev/sean/disrust/src/pipeline/connection_registry.rs)
- shared core pieces:
  - [src/buffer_pool.rs](/home/sriggin/dev/sean/disrust/src/buffer_pool.rs)
  - [src/request_flow.rs](/home/sriggin/dev/sean/disrust/src/request_flow.rs)
  - [src/ring_types.rs](/home/sriggin/dev/sean/disrust/src/ring_types.rs)
  - [src/connection_id.rs](/home/sriggin/dev/sean/disrust/src/connection_id.rs)
  - [src/metrics.rs](/home/sriggin/dev/sean/disrust/src/metrics.rs)
- client/load generator:
  - [src/bin/client.rs](/home/sriggin/dev/sean/disrust/src/bin/client.rs)

Supporting docs:

- metrics definitions: [METRICS.md](/home/sriggin/dev/sean/disrust/METRICS.md)
- client behavior and caveats: [CLIENT.md](/home/sriggin/dev/sean/disrust/CLIENT.md)
- merged inference lane notes: [MERGE_INFERENCE.md](/home/sriggin/dev/sean/disrust/MERGE_INFERENCE.md)
- multithreaded ingress design: [docs/MULTITHREAD_IO.md](/home/sriggin/dev/sean/disrust/docs/MULTITHREAD_IO.md)
- buf-ring exploration: [docs/BUF_RING.md](/home/sriggin/dev/sean/disrust/docs/BUF_RING.md)
- accumulated takeaways: [docs/LEARNINGS.md](/home/sriggin/dev/sean/disrust/docs/LEARNINGS.md)

## What To Expect

The current system is in a healthy state, but a few expectations matter:

- the main productive server hotspot is usually ingress/read-side TCP work
- batching is generally not the bottleneck once the server is fed properly
- the merged inference lane is correctness-stable, but not yet clearly a universal performance win over the old split shape
- the writer/inference response path is no longer the dominant active bottleneck after write decoupling
- end-to-end client latency is often larger than the measured middle of the server
- the built-in client can itself become the limiter if it is under-provisioned

That means:

- high QPS with stable median latency is normal
- `perf` may show large sample shares on threads that are mostly yielding rather than doing useful work
- client-reported latency must be interpreted alongside server metrics, not in isolation

## Running

Server:

```bash
cargo run --release --bin disrust -- serve --model tests/models/ort_verify_model.onnx
```

Client smoke test:

```bash
cargo run --release --bin client -- --port 9900 smoke
```

Sustain run:

```bash
cargo run --release --bin client -- --port 9900 sustain --threads 1 --connections 4 --window 64 --vectors 1 --warmup 3 --duration 10
```

Notes:

- `client --threads N` means `N` independent client workers, each running the full configured workload shape
- only client worker threads are pinned; the reporting path is not
- `disrust serve --io-threads N` enables `SO_REUSEPORT` ingress sharding

## Profiling And Repeatable Runs

The main harness for sustained runs and profiling is:

- [scripts/run_sustain_capture.sh](/home/sriggin/dev/sean/disrust/scripts/run_sustain_capture.sh)

It can:

- build release binaries
- start the server
- wait for readiness
- run the client sustain workload
- optionally collect `perf stat` and `perf record`
- emit a self-contained run bundle under `artifacts/sustain_capture/...`

Typical examples:

```bash
scripts/run_sustain_capture.sh --no-cuda
```

```bash
CONNECTIONS=16 CLIENT_THREADS=2 scripts/run_sustain_capture.sh --no-cuda --perf-all --server-io-threads 4
```

## Canonical Test Model

The repo's single canonical trivial ONNX fixture is:

- [tests/models/ort_verify_model.onnx](/home/sriggin/dev/sean/disrust/tests/models/ort_verify_model.onnx)

Its structure is intentionally simple:

- input: `float32[batch, 16]`
- output: `float32[batch]`
- computation: weighted dot product against weights `[1, 2, ..., 16]`

For a row with values `x[0..15]`, the expected output is:

```text
sum(x[i] * (i + 1)) for i in 0..15
```

You can verify exact outputs through the built-in verification command:

```bash
cargo run --release --bin disrust -- verify --model tests/models/ort_verify_model.onnx --validate-generated-model --cpu
```

That command feeds deterministic generated inputs and checks the exact weighted-dot result for each output row. Use `--cuda` instead of `--cpu` when validating the CUDA execution path.

## Validation

Useful local checks:

```bash
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --lib
cargo test --test request_flow_integration
```

The real TCP ingress integration test is:

- [tests/ingress_tcp_integration.rs](/home/sriggin/dev/sean/disrust/tests/ingress_tcp_integration.rs)

It requires real `AF_INET` sockets and may not run inside restrictive sandboxes.

## Caveats

- The request protocol and transport path are intentionally specialized.
- Some latency attribution gaps remain at the receive and client-observation edges.
- The current global buffer pool requires a serialized allocation+publish gate under multithreaded ingress to preserve correctness.
- Wide/shallow and narrow/deep workloads stress different parts of the system and should not be interpreted as equivalent.
- The merged inference lane should not use blocking helpers that assume a separate completion thread exists.

## Why This README Exists

This repo accumulated a lot of performance work quickly. The point of this README is to make the
current structure and expectations obvious to a fresh reader so they can answer:

- what this server is
- why it is structured this way
- how to run it
- where to look next when investigating behavior
