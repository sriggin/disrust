# GPU Inference Pipeline

This document describes the ONNX/CUDA inference pipeline used by `disrust`. See `GPU_INFERENCE.md` for research background and decision rationale.

---

## Structure

```
disrust        ONNX/CUDA pipeline, this document
```

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

**PoolSlice lifetime:** because the GPU reads directly from pinned host memory throughout kernel execution (zero-copy), `PoolSlice`s cannot be released at submission time. Releasing early would allow the IO thread to reallocate the same physical pool region while the GPU kernel is still issuing DMA reads.

`PoolSlice` ownership stays on the ring slot (`InferenceEvent.features`) throughout submission and GPU execution; no transfer into `BatchEntry` is required. The Completion Consumer holds the poll guard while awaiting GPU handles for the covered ranges, so those slots cannot be recycled by the producer during that window.

After a batch handle resolves, while iterating that batch's slot range, CompletionConsumer submits the write and then calls `slot.features.release()` eagerly for each slot. This is the intended reclamation point (to avoid pool saturation from dead data). Later producer overwrite still drops the old `PoolSlice`, but `release()` is idempotent, so that drop is a no-op for already-released slices.

### Output path: pre-allocated per-session pinned output tensor

Each ORT session owns a fixed output tensor allocated with `cudaMallocHost`:
- Shape: `[MAX_BATCH_VECTORS]` f32, where `MAX_BATCH_VECTORS = MAX_SESSION_BATCH_SIZE × MAX_VECTORS_PER_REQUEST`
- Maximum size: a few KB — negligible overhead
- Bound once via `IoBinding::bind_output` at session construction
- ORT overwrites it in place each batch run

After the Completion Consumer awaits GPU completion, it walks ring slots in order, tracking a running `offset` starting at 0. For each slot it reads `output_buf[offset..offset+num_vectors]` and advances `offset` by `num_vectors`. No allocation, no PoolSlice, no D2H copy (host-accessible pinned memory).

**Verification required:** see ORT Upfront Verification §1 above. This strategy is conditional on ORT writing exactly `actual_batch_vectors` results at index 0 of the pre-bound buffer.

### Completion Consumer loop

Core invariants:
- **Invariant 1 (PoolSlice reclaim):** `PoolSlice` is released eagerly by CompletionConsumer per slot, after that slot's batch GPU completion, not at submission time.
- **Invariant 2 (Batch boundary):** GPU batch boundaries are defined by SubmissionConsumer via ordered `BatchEntry` boundaries (`end_sequence`), not by poll-guard boundaries.

The Completion Consumer's sequence cursor (not an active guard) is what gates the producer from reusing ring slots. The cursor only advances when a poll guard is dropped. Therefore: as long as the guard is held, those slots cannot be recycled regardless of GPU execution state — but the guard must also be held until all GPU handles in its range complete, because dropping the guard advances the cursor past all covered slots simultaneously. If the guard covered two batches and was dropped after only the first GPU await, the cursor advances past the second batch's slots, eventually allowing the producer to overwrite them via ring wraparound before GPU 2 completes.

Core loop structure:

1. Drain write CQEs and retire outstanding writes from the previous buffer set.
2. `poll()` on the EventPoller → blocks until the SubmissionConsumer has published at least one event; returns a guard over all events in range `[cursor+1, submission_published]`
3. For each `BatchEntry` within this guard's range:
   - Pop `(end_sequence, session_idx, OrtRunHandle)` from the batch queue
   - Assert `end_sequence > local_cursor` and `end_sequence <= submission_published`
   - Await the handle (guard is held throughout — cursor has not advanced)
   - Reset `output_offset = 0`
   - Walk `end_sequence - local_cursor` slots from the guard; for each slot: read `fd`, `num_vectors`; encode wire format from `output_buf[output_offset..output_offset+num_vectors]` into the slot's write buffer; submit OP_WRITE; eagerly call `slot.features.release()`; advance `output_offset` by `num_vectors`
   - Set `local_cursor = end_sequence`
4. Flush the io_uring SQ
5. Drop guard → Completion Consumer sequence advances to `submission_published`

**Write completion accounting (required):**
- CQEs can arrive out of order across connections and SQEs; FIFO CQE arrival must not be assumed.
- The consumer must track an exact outstanding write count per buffer set (A/B), incrementing on each submitted OP_WRITE and decrementing on each corresponding write CQE.
- A buffer set is reusable only when its outstanding count reaches zero.
- "Drain CQEs once" is not sufficient proof of safety; reuse is gated by the counter, not by a single drain call.

This gives the intended phase ordering per cycle:
- **Phase A:** retire CQEs until the previous set's outstanding count is zero (safe to reuse).
- **Phase B:** process one poll guard end-to-end (await GPU handles, encode full guard range, submit all writes for the current set), then swap sets.

**Guard vs GPU batch (`BatchEntry`) delineation:**
- A poll guard is a retention/liveness boundary; it is not the semantic GPU batch boundary.
- Ordered `BatchEntry` boundaries define true batch boundaries.
- One guard may contain multiple `BatchEntry` ranges.
- Within one guard, process `BatchEntry`s in order:
  1. await that batch's `OrtRunHandle`
  2. encode and submit OP_WRITEs for that batch's covered slots
  3. release those slots' `PoolSlice`s eagerly
- Do **not** require "await all write CQEs for batch N before starting batch N+1" as the default flow.
  Write completions are retired by the per-buffer-set outstanding counter, and reuse is gated by counter==0.
  This preserves overlap and avoids unnecessary serialization.

**Performance assumption:**
- Do not assume socket write completions are always faster than GPU inference.
- On healthy clients they are often fast, but slow receivers/backpressure can make write completion the long pole.
- Correctness must depend on explicit completion accounting, not timing assumptions.

**io_uring CQ sizing:** the kernel silently drops CQEs when the Completion Queue ring is full (sets `IORING_SQ_CQ_OVERFLOW`; entries are lost with no error returned). The CQ must never overflow between the CQE drain at step 1 and the next cycle's drain. The maximum pending CQEs equals the maximum OP_WRITEs from one guard: up to `SESSION_POOL_SIZE × MAX_SESSION_BATCH_SIZE` entries. Set the SQ to that value; the CQ is automatically 2× — sufficient.

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

`PoolSlice`s stay in `InferenceEvent.features` in the ring slots; CompletionConsumer explicitly releases them while processing each batch range after GPU completion. No explicit ownership transfer into `BatchEntry` is needed.

`OrtRunHandle` is non-trivial and cannot sit in a pre-allocated ring slot, which is why `BatchEntry` exists at all. `spsc-bip-buffer` remains unsuitable; a hand-rolled SPSC ring of `BatchEntry` is required.

**Submission visibility invariant:** the SubmissionConsumer must not publish any slot sequence to CompletionConsumer unless the corresponding `BatchEntry` boundary has already been enqueued in the batch queue. Given current poll-guard semantics (cursor advances on guard drop), this means all `BatchEntry`s covering that guard's published slots are enqueued before dropping the guard.

**Completion boundary invariant:** within a guard, popped `BatchEntry.end_sequence` values must be strictly increasing and the final one must equal `submission_published`.

Debug assertions should enforce this in development builds.

**Batch queue capacity:** must be `SESSION_POOL_SIZE + 1`, not `SESSION_POOL_SIZE`. A pool ring wrap mid-cycle forces two batches from one SubmissionConsumer guard. With capacity = `SESSION_POOL_SIZE`, the SubmissionConsumer blocks trying to push the second `BatchEntry` while its guard is still held. The CompletionConsumer, gated on SubmissionConsumer's cursor (which hasn't advanced because the guard is held), sees no events and spins. Deadlock. The `+1` absorbs exactly one wrap-induced extra batch per cycle, which is the maximum possible (pool ring can wrap at most once per SubmissionConsumer poll cycle given the pool and ring sizing constraints).

### Response write path: direct socket writes from CompletionConsumer

The Completion Consumer writes responses directly to client fds via its own io_uring ring. No IO thread coordination, no `InferenceResponse`, no result pool.

Wire format is encoded directly from the output tensor into pre-allocated fixed-size write buffers: `1 + MAX_VECTORS_PER_REQUEST × 4 = 257` bytes per slot, allocated statically at startup. Two sets of buffers (`2 × MAX_SESSION_BATCH_SIZE`) are required — the CQE drain confirming the previous guard's writes happens at the top of the next guard, so one set may still be in-flight while the next is being filled.

**Error handling:** fatal connection errors (`EPIPE`, `EBADF`, `ECONNRESET`) are swallowed and optionally counted via metrics — the IO thread discovers dead connections on the next OP_READ. EAGAIN and partial writes (which can occur on slow but live clients whose TCP receive window has shrunk to zero) trigger `shutdown(fd, SHUT_WR)` — the IO thread's next OP_READ returns 0 and triggers normal cleanup, and the client receives TCP EOF. Partial writes with bytes already delivered leave a corrupt partial frame; `shutdown` is the least-bad outcome. The IO thread owns all `close()` calls; the Completion Consumer never calls `close()`.

---

## What Exists (Done)

Additive changes to shared types:

| Item | Location | Notes |
|---|---|---|
| `fd: i32` in `InferenceEvent` | `ring_types.rs` | Fits in existing 4-byte padding; 64-byte assert preserved |
| `BufferPool::from_raw_ptr` | `buffer_pool.rs` | Unsafe constructor; `_backing: None` for external memory |
| `PoolSlice::is_contiguous` | `buffer_pool.rs` | Pointer comparison; detects pool ring wrap-around |
| `fd` parameter in `request_flow` | `request_flow.rs` | Propagates fd into every published slot |

No GPU-specific runtime files are intentionally kept in-tree at this stage. The plan remains the source of truth.

### Captured Implementation Notes

The following non-obvious behaviors were validated during the ONNX/CUDA server bring-up and should be preserved:

- **Ingress thread responsibilities:** accept/read/parse/publish only; no response poller, no eventfd, no write path. Completion consumer owns all response writes.
- **Ring-full retry behavior:** after each CQE batch, linearly rescan connections with `!read_inflight && read_len > 0` and re-run parse/publish. This prevents a connection from stalling when `RingBufferFull` occurred while its read buffer was full.
- **Read resubmission rule:** after parse/publish, submit a new OP_READ only when `read_len < READ_BUF_SIZE`.
- **Socket setup:** listener uses `SO_REUSEADDR`, `SO_REUSEPORT`, `NONBLOCK`, and `TCP_NODELAY`.
- **Factory pool prerequisite:** `set_factory_pool(BufferPool::new_boxed(1))` must run before disruptor factory pre-allocation (`InferenceEvent::factory` uses `PoolSlice::empty()`).
- **Provisional ring depth:** io_uring depth `4096` was sufficient for ingress-only accept/read scaffolding.
- **Allocator capability use:** ingress path should allocate via `PoolAllocator` (single-allocator invariant), not direct `BufferPool::alloc`.

---

## Status

The plan above is now implemented in the codebase as the default `disrust` server. The remaining value in this document is the design rationale, invariants, and operational notes for the ONNX/CUDA pipeline.

---

## Open Questions

- Whether `disable_device_sync` + manual stream sync is worth pursuing once basic pipeline is working
- Whether CUDA Graphs are viable given fixed-batch-size requirement and open deferred-sync bug (#20392)
- `request_seq` ordering: strict per-connection or best-effort (currently best-effort, no change planned)
- **GPU error handling mid-batch:** if `OrtRunHandle` resolves with an error, all connections in that batch receive no response, breaking the 1:1 invariant. Options: close all connections in the batch, send a synthetic error response (requires protocol change), or treat GPU failure as fatal and restart the process. Must be decided before implementing `CompletionConsumer`.
