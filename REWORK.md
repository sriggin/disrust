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
    │ fatal write errors swallowed; IO thread discovers dead conns via OP_READ
    ▼
client fds
```

The Completion Consumer's sequence barrier is gated on the Submission Consumer's published sequence — it cannot read slot N until Submission has advanced past it.

---

## ORT Upfront Verification

The following must be confirmed against the `ort` 2.x crate with CUDA EP **before** building `SubmissionConsumer` or `CompletionConsumer`. Each is a go/no-go for a specific design decision.

### 1. Variable-batch run against a fixed pre-bound output tensor

Run a batch of K vectors (K < `MAX_BATCH_VECTORS`) against a session with a pre-bound output tensor allocated for `MAX_BATCH_VECTORS`. Verify ORT writes exactly K results at indices [0, K) and leaves the rest unmodified. If ORT rejects the shape mismatch or writes unpredictably, the pre-allocated output tensor strategy must change — options are per-batch dynamic output binding or a dynamically-shaped output tensor.

### 2. True zero-copy with `AllocationDevice::CUDA_PINNED` and `TensorRefMut::from_raw`

Confirm that passing a `cuMemHostGetDevicePointer` result as input with `AllocationDevice::CUDA_PINNED` causes ORT's CUDA EP to issue a kernel that reads directly from the pinned host mapping without an internal D2D staging copy. If ORT stages a copy, zero-copy is not achieved (correctness is unaffected, but the zero-copy design rationale changes and pool sizing can be relaxed).

### 3. `OrtRunHandle` / async execution semantics

Confirm that the async run API returns a handle immediately (before GPU completion) and that awaiting the handle resolves after the GPU kernel finishes and the pinned output tensor is fully written. If the CUDA EP's async path is synchronous (blocks on return), the two-consumer pipeline is still correct but degenerates to serial GPU execution — the performance model changes but the design holds.

### 4. Dynamic input shape via `IoBinding`

Confirm that `IoBinding::bind_input` with a CUDA_PINNED pointer and a per-batch shape (varying total `num_vectors` per call) is accepted without requiring a fixed input shape registered at session construction time.

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

**Wrap-around:** when the pool ring wraps mid-batch, the wrapping slot's data starts at a lower address than the previous slot's end. The Submission Consumer detects this via `PoolSlice::is_contiguous` on consecutive slots. A non-contiguous slot is a forced batch boundary: submit the pre-wrap batch to GPU, push `BatchEntry`, then submit the post-wrap events as a second batch immediately and push a second `BatchEntry` — all within the same guard drop. The post-wrap batch cannot be deferred to a later poll cycle: the alignment guarantee requires all `BatchEntry`s for a guard's events to be pushed before the guard is dropped. Deferral would mean dropping the guard without a `BatchEntry` for the post-wrap slots, which the CompletionConsumer would see as unaccounted events. With `SESSION_POOL_SIZE = 1` this produces two sequential submissions to the same session; ORT serializes concurrent `Run()` calls on a single session internally, so this is safe.

**PoolSlice lifetime:** because the GPU reads directly from pinned host memory throughout kernel execution (zero-copy), `PoolSlice`s cannot be released at submission time. Releasing early would allow the IO thread to reallocate the same physical pool region while the GPU kernel is still issuing DMA reads. However, no explicit transfer is needed: `PoolSlice`s stay in `InferenceEvent.features` in the ring slots. The Completion Consumer's disruptor barrier prevents the IO thread from overwriting any slot in the batch until the Completion Consumer advances its sequence — which only happens after GPU completion and response writes. When the IO thread publishes to a recycled slot, the old `PoolSlice` is dropped by assignment, advancing the read cursor at the correct time.

### Output path: pre-allocated per-session pinned output tensor

Each ORT session owns a fixed output tensor allocated with `cudaMallocHost`:
- Shape: `[MAX_BATCH_VECTORS]` f32, where `MAX_BATCH_VECTORS = MAX_SESSION_BATCH_SIZE × MAX_VECTORS_PER_REQUEST`
- Maximum size: a few KB — negligible overhead
- Bound once via `IoBinding::bind_output` at session construction
- ORT overwrites it in place each batch run

After the Completion Consumer awaits GPU completion, it walks ring slots in order, tracking a running `offset` starting at 0. For each slot it reads `output_buf[offset..offset+num_vectors]` and advances `offset` by `num_vectors`. No allocation, no PoolSlice, no D2H copy (host-accessible pinned memory).

**Verification required:** see ORT Upfront Verification §1 above. This strategy is conditional on ORT writing exactly `actual_batch_vectors` results at index 0 of the pre-bound buffer.

### Completion Consumer loop

The Completion Consumer's sequence cursor (not an active guard) is what gates the producer from reusing ring slots. The cursor only advances when a poll guard is dropped. Therefore: as long as the guard is held, those slots cannot be recycled regardless of GPU execution state — but the guard must also be held until all GPU handles in its range complete, because dropping the guard advances the cursor past all covered slots simultaneously. If the guard covered two batches and was dropped after only the first GPU await, the cursor advances past the second batch's slots, eventually allowing the producer to overwrite them via ring wraparound before GPU 2 completes.

Core loop structure (common to both write-path options):

1. `poll()` on the EventPoller → blocks until the SubmissionConsumer has published at least one event; returns a guard over all events in range `[cursor+1, submission_published]`
2. For each `BatchEntry` within this guard's range (there may be more than one if the SubmissionConsumer ran multiple cycles while this consumer was awaiting GPU):
   - Pop `(end_sequence, session_idx, OrtRunHandle)` from the batch queue
   - Await the handle (guard is held throughout — cursor has not advanced)
   - Walk `entry.end_sequence - cursor` events from the guard, tracking a running `output_offset`; for each event: read `fd`, `conn_id`, `num_vectors`; read `output_buf[output_offset..output_offset+num_vectors]`; dispatch response (see write-path option); advance `output_offset` by `num_vectors`; advance local cursor to `entry.end_sequence`
3. Drop guard → Completion Consumer sequence advances to `submission_published`

**Option A additions:** before step 2, drain write CQEs from the previous guard's OP_WRITEs (confirms per-slot write buffers are free). After step 2's inner walk, submit OP_WRITEs via the Completion Consumer's own io_uring ring and flush. Requires double-buffered per-slot write buffers (`2 × MAX_SESSION_BATCH_SIZE` slots) due to deferred CQE drain.

**Option B additions:** after step 2's inner walk, push serialized bytes into the per-IO-thread response queue and signal via eventfd. No write ring, no write buffers, no CQE drain step.

**io_uring CQ sizing (Option A only):** the kernel silently drops CQEs when the Completion Queue ring is full (sets `IORING_SQ_CQ_OVERFLOW`; there is no error returned to the application — entries are simply lost). The Completion Consumer's io_uring instance must be sized so the CQ can never overflow between CQE drains. The maximum pending CQEs between drains equals the maximum OP_WRITEs submitted in one guard's range: up to `SESSION_POOL_SIZE × MAX_SESSION_BATCH_SIZE` entries. Set the SQ to that value and the CQ is automatically 2× — sufficient.

### Batch queue

SPSC channel between Submission and Completion consumers. Each entry:
```rust
struct BatchEntry {
    end_sequence: u64,  // last ring slot (inclusive) in this batch
    session_idx: usize, // which session in the pool was used
    handle: OrtRunHandle,
}
```

`BatchEntry` carries only batch-level metadata. Per-slot metadata (`fd`, `conn_id`, `num_vectors`) is read directly from the disruptor ring slots by the CompletionConsumer — those slots remain live because the CompletionConsumer's own barrier prevents the producer from overwriting them until the CompletionConsumer advances its sequence.

`PoolSlice`s likewise stay in `InferenceEvent.features` in the ring slots; the same barrier holds them alive. When the IO thread eventually publishes to a recycled slot, assigning the new `PoolSlice` to `slot.features` drops the old one (advancing the read cursor). GPU has already completed before the CompletionConsumer advances its sequence, so the timing is always: GPU done → CompletionConsumer advances → IO thread overwrites slot → `PoolSlice::drop` → read cursor advances. No explicit transfer into `BatchEntry` is needed.

`OrtRunHandle` is non-trivial and cannot sit in a pre-allocated ring slot, which is why `BatchEntry` exists at all. `spsc-bip-buffer` remains unsuitable; a hand-rolled SPSC ring of `BatchEntry` is required.

**Alignment guarantee:** when CompletionConsumer's `poll()` returns a guard, all `BatchEntry`s covering those events are already in the queue. This holds because the SubmissionConsumer drops its cursor (making those events visible to CompletionConsumer) only after pushing all `BatchEntry`s for its entire guard. The CompletionConsumer iterates the guard's events, consuming one `BatchEntry` per batch (advancing to the next when `local_cursor == entry.end_sequence`); the iterator exhausts at the same point as the last `BatchEntry`'s range.

**Batch queue capacity:** must be `SESSION_POOL_SIZE + 1`, not `SESSION_POOL_SIZE`. A pool ring wrap mid-cycle forces two batches from one SubmissionConsumer guard. With capacity = `SESSION_POOL_SIZE`, the SubmissionConsumer blocks trying to push the second `BatchEntry` while its guard is still held. The CompletionConsumer, gated on SubmissionConsumer's cursor (which hasn't advanced because the guard is held), sees no events and spins. Deadlock. The `+1` absorbs exactly one wrap-induced extra batch per cycle, which is the maximum possible (pool ring can wrap at most once per SubmissionConsumer poll cycle given the pool and ring sizing constraints).

### Response write path: **UNRESOLVED — two architectural options**

This section describes an open design decision. The original plan had the Completion Consumer writing directly to client fds via its own io_uring ring. That design has a correctness gap and the tradeoff must be decided before implementation.

**The problem:** EAGAIN is not limited to dead connections. A slow but live client — one undergoing a GC pause, slow postprocessing, or any IO wait — can cause the TCP receive window to shrink to zero and EAGAIN to surface. Swallowing EAGAIN on a live connection silently drops a response, breaking the protocol's 1:1 ordering invariant: the client later receives a response for the wrong request with no indication anything went wrong. Partial writes (some bytes delivered, rest dropped) are a harder version of the same problem: closing after a partial write leaves a corrupt partial frame in the client's receive buffer.

**Option A — Completion Consumer writes directly, terminate on error:**
The Completion Consumer writes via its own io_uring ring. Fatal connection errors (`EPIPE`, `EBADF`, `ECONNRESET`) are swallowed and counted via metrics — the IO thread discovers the dead connection on the next OP_READ. EAGAIN and partial writes trigger `shutdown(fd, SHUT_WR)` (not `close`) — the IO thread's next OP_READ returns 0 and triggers cleanup; the client receives TCP EOF rather than silent corruption. Partial writes with some bytes already delivered leave a corrupt partial frame on the client side regardless.
*Pros:* low latency on the common path, no cross-thread coordination.
*Cons:* requires Completion Consumer to call `shutdown()` (limited fd management); partial writes cannot be cleanly recovered; per-slot write buffers and double-buffering add complexity; CQE drain machinery required.

**Option B — Route responses through the IO thread:**
The Completion Consumer serializes responses and pushes them into a per-IO-thread response queue, then signals via eventfd, identical to the CPU pipeline. The IO thread writes via its existing bip-buffer and writev machinery, handles all backpressure and partial writes naturally, and owns all fd lifecycle. The Completion Consumer has no io_uring ring and no write buffers.
*Pros:* structurally correct for all write scenarios including partial writes and slow clients; simplifies the Completion Consumer significantly (eliminates per-slot write buffers, double-buffering, CQE drain, and the write ring entirely); reuses proven CPU pipeline machinery.
*Cons:* one cross-thread hop plus eventfd syscall per response batch; partially surrenders the latency advantage of the two-ring design.

**Decision required:** whether the latency cost of Option B is acceptable given GPU execution time dominates the pipeline. If GPU latency is 1–5ms, the cross-thread eventfd overhead (~1–5µs) is negligible and Option B is strictly preferable. If sub-microsecond tail latency on the write path matters, Option A with partial-write acceptance may be justified.

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
| GPU binary entry point | `src/bin/disrust_gpu.rs` | Stub; consumers and ring not yet wired |

---

## Remaining Work

1. **Config** — add GPU-path constants to `config.rs` (separate from CPU constants):
   - `SESSION_POOL_SIZE` (default 1)
   - `MAX_SESSION_BATCH_SIZE` (max disruptor slots per GPU batch)
   - `SUBMISSION_TIMEOUT_NS` — maximum nanoseconds the SubmissionConsumer busy-spins on an empty ring before yielding; prevents a dedicated core running at 100% CPU with no GPU work pending. Default: ~100_000 ns. Not a "hold partial batch" timer — with the disruptor poll-guard model, every poll cycle flushes whatever is available; this timeout governs the idle spin between cycles.
   - `GPU_DISRUPTOR_SIZE` — do not reuse `DISRUPTOR_SIZE` (65536 slots × max request size = 268MB pinned memory; far too large). Size to `SESSION_POOL_SIZE × MAX_SESSION_BATCH_SIZE × 4` as headroom.
   - `GPU_BUFFER_POOL_CAPACITY` = `GPU_DISRUPTOR_SIZE × MAX_VECTORS_PER_REQUEST × FEATURE_DIM`. With `GPU_DISRUPTOR_SIZE` in the low hundreds, this is low tens of MB of pinned memory — affordable.

   **Ring/pool sizing constraint:** `GPU_DISRUPTOR_SIZE` must be ≥ `SESSION_POOL_SIZE × MAX_SESSION_BATCH_SIZE`. The IO thread (producer) is gated on the CompletionConsumer's sequence. If the ring is smaller than the maximum number of in-flight slots across all in-flight GPU batches, the IO thread stalls even when the GPU is still executing. With `SESSION_POOL_SIZE = 1`, the ring needs only to be ≥ `MAX_SESSION_BATCH_SIZE`; the 4× headroom above gives the IO thread room to run ahead while the GPU executes.

2. **ORT session wrapper** — session construction with CUDA EP, IoBinding setup, pre-allocated pinned output tensor per session; **run ORT verification steps (§ ORT Upfront Verification) before proceeding**

3. **Batch queue** — `BatchEntry` type (`end_sequence`, `session_idx`, `handle`), fixed-capacity SPSC ring with capacity `SESSION_POOL_SIZE + 1` (the `+1` prevents deadlock when a pool ring wrap forces two batches from one SubmissionConsumer guard cycle)

4. **`SubmissionConsumer`** — disruptor polling, batch accumulation with `is_contiguous` wrap detection, ORT submission via `TensorRefMut::from_raw`, batch queue push, idle timeout via `SUBMISSION_TIMEOUT_NS`.

   The disruptor poll guard covers all events delivered in one `poll()` call. The guard must be held across all ORT submissions from that cycle (there may be multiple due to wrap-around or max-size splits); the guard drop advances the SubmissionConsumer's sequence. If a batch queue push blocks (queue full — all sessions in flight), this also stalls the guard drop, which stalls the IO thread's ring slot reuse. This is correct backpressure. The SubmissionConsumer does not accumulate events across poll cycles; batching occurs naturally from events that arrive between consecutive polls.

   **Wrap handling within a cycle:** a wrap-induced boundary submits the pre-wrap batch immediately and begins accumulating the post-wrap batch. However, the post-wrap batch is **not submitted** within that same guard — it carries forward into the next poll cycle. Since a guard cannot span multiple poll cycles, the SubmissionConsumer must defer the post-wrap partial batch state as a field (e.g., `pending_batch_start: Option<SequenceIdx>`) that the next `poll()` call picks up. If the next cycle returns no events (idle), `SUBMISSION_TIMEOUT_NS` governs when the pending partial batch is flushed — same as any other idle partial batch. This ensures a single submission path with no special-case timing for wrap.

5. **`CompletionConsumer`** — batch queue pop, GPU await, output tensor read, disruptor sequence advance. Write dispatch depends on the response write path decision: Option A adds per-slot write buffers, OP_WRITE via own ring, write CQE drain, and error handling; Option B adds response queue push and eventfd signal per batch.

6. **Wire `disrust_gpu.rs`** — `cudaMallocHost` pool allocation, `cuMemHostGetDevicePointer` for device base, session pool construction, consumer thread spawning. The correct two-consumer builder chain is:
   ```rust
   let builder = build_single_producer(GPU_DISRUPTOR_SIZE, InferenceEvent::factory, BusySpin);
   let (submission_poller, builder) = builder.event_poller();
   let (completion_poller, builder) = builder.and_then().event_poller();
   let producer = builder.build();
   ```
   `event_poller()` alone is non-gating; `.and_then()` installs the dependency barrier. `completion_poller` has type `EventPoller<InferenceEvent, SingleConsumerBarrier>` and is gated on `submission_poller`'s cursor. The producer is gated on `completion_poller`'s cursor (the slowest consumer). **Note:** the CompletionConsumer must hold its poll guard across all GPU awaits for all `BatchEntry`s within that guard's range — guard drop advances the cursor and frees slots for the producer. Multiple `BatchEntry`s may fall within one guard (if the SubmissionConsumer ran multiple cycles); the number of slots per entry is `entry.end_sequence - cursor`.

---

## Open Questions

- Whether `disable_device_sync` + manual stream sync is worth pursuing once basic pipeline is working
- Whether CUDA Graphs are viable given fixed-batch-size requirement and open deferred-sync bug (#20392)
- `request_seq` ordering: strict per-connection or best-effort (currently best-effort, no change planned)
- **GPU error handling mid-batch:** if `OrtRunHandle` resolves with an error, all connections in that batch receive no response, breaking the 1:1 invariant. Options: close all connections in the batch, send a synthetic error response (requires protocol change), or treat GPU failure as fatal and restart the process. Must be decided before implementing `CompletionConsumer`.
