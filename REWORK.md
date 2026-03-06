# GPU Inference Pipeline

This document describes the GPU inference pipeline, which runs as a **separate binary** (`disrust-gpu`) alongside the existing CPU baseline (`disrust`). The existing binary and all its supporting code are unchanged. See `GPU_INFERENCE.md` for research background and decision rationale.

---

## Structure

```
disrust        (existing, unchanged)   CPU pipeline, batch_processor POC
disrust-gpu    (new, additive)         GPU pipeline, ORT/CUDA, this document
```

Shared types receive additive-only changes: a new field on `InferenceEvent`, new constructors on `BufferPool`/`PoolSlice`, and a new parameter on `request_flow`. No existing behaviour is altered.

---

## Pipeline Topology

```
IO Thread(s)
    │ publishes InferenceEvent (includes fd)
    ▼
[request ring]
    │
    ▼
SubmissionConsumer   (accumulates batch, submits to ORT/GPU)
    │ pushes (end_sequence, session_idx, OrtRunHandle) to batch queue
    ▼
[batch queue SPSC]
    │
    ▼
CompletionConsumer   (gated on Submission sequence)
    │ awaits GPU handle, reads output tensor
    │ serializes wire bytes, submits OP_WRITE directly
    ▼
[Completion Consumer's own io_uring ring]
    │ write errors ignored; IO thread discovers dead conns via OP_READ
    ▼
client fds
```

The Completion Consumer's sequence barrier is gated on the Submission Consumer's published sequence — it cannot read slot N until Submission has advanced past it.

---

## Settled Decisions

### Session pool

Design around a pool of N ORT sessions (N configurable, default 1). With N=1 the pipeline degenerates to single-session, one-batch-in-flight behavior — no behavioral difference from a simpler design, so there is no cost to building for N. With N>1, the Submission Consumer round-robins across sessions; the batch queue carries which session index is associated with each batch entry so the Completion Consumer knows which future to await.

Multiple sessions are viable here because:
- The model is a Hummingbird-converted GBT → NN, producing GEMM-based ONNX ops
- SM occupancy per session is expected to be low at these model sizes (1–10MB source, even at 100× expansion after conversion)
- Weight duplication per session is affordable at these sizes
- N=1 always works as a safe default pending profiling

### Input path: buffer pool over pinned memory, zero-copy H2D

The IO buffer pool's backing memory is allocated with `cudaMallocHost` externally before the pool is constructed. The device-mapped base pointer is obtained once via `cuMemHostGetDevicePointer` and stored on the `SubmissionConsumer`. The pool has no knowledge of CUDA.

At batch submission time, the Submission Consumer computes the device pointer for the batch's input span as:
```
device_ptr = device_base + (batch_start_ptr - host_base)
```

Allocations within a batch are contiguous in the pool ring by construction — the IO thread allocates sequentially, one slot at a time. The Submission Consumer passes a single device pointer covering the entire batch to ORT via `TensorRefMut::from_raw` with `AllocationDevice::CUDA_PINNED` — no H2D copy.

**Wrap-around:** when the pool ring wraps mid-batch, the wrapping slot's data starts at a lower address than the previous slot's end. The Submission Consumer detects this via `PoolSlice::is_contiguous` on consecutive slots. A non-contiguous slot is a forced batch boundary: submit the pre-wrap batch to GPU, push `BatchEntry`, advance the disruptor sequence, start a new batch from that slot. No staging copy — the common path is always a direct device pointer, and the wrap case just produces two smaller batches instead of one.

The `PoolSlice` for each slot is released immediately after the batch is submitted to ORT. The Completion Consumer does not touch input data.

### Output path: pre-allocated per-session pinned output tensor

Each ORT session owns a fixed output tensor allocated with `cudaMallocHost`:
- Shape: `[MAX_BATCH_VECTORS]` f32, where `MAX_BATCH_VECTORS = max_session_batch_size × MAX_VECTORS_PER_REQUEST`
- Maximum size: a few KB — negligible overhead
- Bound once via `IoBinding::bind_output` at session construction
- ORT overwrites it in place each batch run

After the Completion Consumer awaits GPU completion, it walks ring slots in order, tracking a running `offset` starting at 0. For each slot it reads `output_buf[offset..offset+num_vectors]` and advances `offset` by `num_vectors`. No allocation, no PoolSlice, no D2H copy (host-accessible pinned memory).

**Verification required:** ORT's behaviour when a variable-size input batch (fewer than `MAX_BATCH_VECTORS` vectors) runs against a fixed-size pre-bound output tensor must be confirmed. The expected behaviour is that ORT writes exactly `actual_batch_vectors` results starting at index 0 of the bound buffer; results beyond that index are not written and not read. This must be validated against the `ort` 2.x CUDA EP before committing to this output binding strategy.

### Completion Consumer loop

1. Drains write CQEs from the previous batch's OP_WRITEs (non-blocking; by the time the next GPU handle is awaited these are complete — this is the point at which per-slot write buffers are confirmed free for reuse)
2. Pops `(end_sequence, session_idx, OrtRunHandle)` from the batch queue
3. Awaits the handle (blocks until GPU batch completes)
4. Walks ring slots from last processed sequence to `end_sequence`, tracking a running `output_offset`
5. For each slot: reads `fd`, `conn_id`, `num_vectors`; reads `output_buf[output_offset..output_offset+num_vectors]`; serializes wire bytes into a pre-allocated per-slot buffer; submits OP_WRITE to `fd` via the Completion Consumer's own io_uring ring; advances `output_offset` by `num_vectors`
6. Flushes the ring (submit pending SQEs to kernel)
7. Advances its disruptor sequence

### Batch queue

SPSC channel between Submission and Completion consumers. Each entry:
```rust
struct BatchEntry {
    end_sequence: u64,  // last ring slot (inclusive) in this batch
    session_idx: usize, // which session in the pool was used
    handle: OrtRunHandle,
}
```

Fixed-capacity hand-rolled SPSC ring sized to `SESSION_POOL_SIZE` (at most one in-flight entry per session). `OrtRunHandle` is a non-trivial type and cannot be treated as raw bytes, so `spsc-bip-buffer` is not appropriate here.

### Response write path: two rings, no connection state coordination

The Completion Consumer writes directly to client fds via its own io_uring ring — no response queue, no eventfd, no IO thread involvement in writes.

Write CQE errors (EPIPE, EBADF, ECONNRESET) are ignored. The IO thread's next OP_READ on the same fd will return ≤0, triggering the existing connection cleanup path.

There is a narrow race where a closed fd is reused by a new accept before a stale OP_WRITE completes. In practice this window is extremely tight. Accepted as a known limitation; a generation counter per conn_id slot can close it if it proves to be a real operational problem.

---

## What Exists (Done)

Additive changes to shared types:

| Item | Location | Notes |
|---|---|---|
| `fd: i32` in `InferenceEvent` | `ring_types.rs` | Fits in existing 4-byte padding; 64-byte assert preserved |
| `BufferPool::from_raw_ptr` | `buffer_pool.rs` | Unsafe constructor; `_backing: None` for external memory |
| `PoolSlice::is_contiguous` | `buffer_pool.rs` | Pointer comparison; detects pool ring wrap-around |
| `fd` parameter in `request_flow` | `request_flow.rs` | Propagates fd into every published slot |

New files:

| Item | Location | Notes |
|---|---|---|
| `IoThreadGpu` | `src/bin/io_thread_gpu.rs` | OP_ACCEPT + OP_READ only; no write path |
| GPU binary entry point | `src/bin/disrust_gpu.rs` | Stub; consumers not yet wired |

---

## Remaining Work

1. **Config** — add `SESSION_POOL_SIZE`, `MAX_SESSION_BATCH_SIZE` to `config.rs`
2. **ORT session wrapper** — session construction with CUDA EP, IoBinding setup, pre-allocated pinned output tensor per session; verify variable-batch behaviour against fixed output tensor
3. **Batch queue** — `BatchEntry` type, fixed-capacity SPSC ring sized to `SESSION_POOL_SIZE`
4. **`SubmissionConsumer`** — disruptor polling, batch accumulation with `is_contiguous` wrap detection, ORT submission via `TensorRefMut::from_raw`, batch queue push, PoolSlice release
5. **`CompletionConsumer`** — batch queue pop, GPU await, output tensor read, serialize into per-slot buffers, OP_WRITE via own ring, deferred CQE drain, disruptor sequence advance
6. **Wire `disrust_gpu.rs`** — `cudaMallocHost` pool allocation, `cuMemHostGetDevicePointer` for device base, session pool construction, consumer thread spawning

---

## Open Questions

- Batching policy: max batch size, deadline for partial batches, busy-spin vs timed wait in Submission Consumer
- Whether `disable_device_sync` + manual stream sync is worth pursuing once basic pipeline is working
- Whether CUDA Graphs are viable given fixed-batch-size requirement and open deferred-sync bug (#20392)
- GPU error handling mid-batch: define failure response vs connection close
- `request_seq` ordering: strict per-connection or best-effort (currently best-effort, no change planned)
