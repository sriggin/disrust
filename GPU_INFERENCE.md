# GPU Inference Transition Plan

## Goal

Replace the placeholder inference in `batch_processor.rs` (currently: sum of feature vector) with real batched ML inference on a GPU, while preserving the low-latency properties of the existing pipeline.

## Target Architecture

Two sequenced disruptor consumers replace the current single batch processor thread:

```
IO Thread(s) → [request ring] → Submission Consumer → [batch queue] → Result Consumer
                                        ↓ (sequence barrier)
                                 (Result Consumer gated here)
```

**Submission Consumer**
- Accumulates `InferenceEvent`s up to a batch size limit or deadline
- Copies feature data into a contiguous GPU input tensor
- Releases each `PoolSlice` immediately after copy (returns memory to IO thread's buffer pool)
- Submits the batch to the GPU, obtaining a future/handle
- Pushes `(end_sequence, GpuHandle)` onto the batch queue (SPSC, one entry per batch)
- Advances its disruptor sequence by the batch size

**Result Consumer**
- Gated on Submission Consumer's sequence (cannot read slot N until Submission has passed it)
- Pops the next `(end_sequence, GpuHandle)` from the batch queue
- Awaits GPU completion for that handle
- Walks slots up to `end_sequence`, reading `conn_id`, `thread_id`, `request_seq`, `num_vectors` to build `InferenceResponse`s and route them to the correct IO thread response queue
- Signals affected IO threads via eventfd (same as today)

**Batch queue**
- SPSC between Submission and Result Consumer — same pattern as the existing per-IO-thread response queue
- Carries only `(end_sequence: u64, GpuHandle)` per batch, not per-slot data
- Provides the boundary information the Result Consumer needs to match GPU output indices back to ring slots

## What Stays the Same

- `InferenceEvent` and `InferenceResponse` structures
- IO thread, protocol, buffer pool, response queue, eventfd signaling
- Ring backpressure: if the Result Consumer stalls (GPU slow), sequence barrier backs up through the ring to the IO threads automatically

## Key Design Decisions To Resolve

**Batching policy**
- Maximum batch size (GPU memory / throughput tradeoff)
- Maximum wait deadline before submitting a partial batch (latency bound)
- Whether the Submission Consumer busy-spins or uses a timed wait between polls

**GPU runtime / async execution model**
- Runtime: ONNX Runtime via the `ort` crate (2.x) with CUDA Execution Provider
- `run_async()` is available and cancel-safe but only dispatches through ORT's intra-op thread pool — it does not itself give GPU-level overlap
- **Multiple in-flight batches are not worth pursuing here.** The CUDA EP disables the parallel executor, so concurrent `Run()` calls on a single session are serialized at the ORT level regardless of thread count. True pipelining requires multiple sessions, each with a dedicated `with_compute_stream` injection — but multiple CUDA EP sessions each hold their own copy of model weights in GPU memory with no sharing. For a single-model server this cost is prohibitive.
- **Conclusion: one session, one batch in flight at a time.** The Submission Consumer submits, the Result Consumer synchronously awaits. The GPU pipeline latency becomes the floor; throughput is maximized by keeping the batch full.
- `Session::run_with_options()` with `RunOptions::disable_device_sync()` can defer the `cudaStreamSynchronize` — useful if outputs are bound to device or pinned memory (see below), but adds the burden of manual stream sync before result reads
- Note: CUDA Graphs (`with_cuda_graph`) would reduce kernel launch overhead for fixed batch sizes but are incompatible with deferred sync (issue #20392, unfixed) and prohibit concurrent `Run()` calls entirely

**Disruptor crate support for dependent consumers**
- Verify whether the `disruptor` crate (3.7.1) supports chained/pipeline consumer topologies directly
- If not: Result Consumer polls Submission Consumer's published sequence manually before reading slots — straightforward to implement without crate support

**GPU tensor layout and IoBinding**
- Input shape: `[batch_vectors, FEATURE_DIM]` (f32); output shape: `[batch_vectors]` (f32)
- `IoBinding` works fully with the CUDA EP. Output can be pre-allocated as a device-resident tensor via `Allocator::new` with `AllocationDevice::CUDA` and bound once; ORT overwrites it in place each run. This avoids a device→host copy for the output until the Result Consumer explicitly reads it
- Input binding is more complex given variable batch sizes — device memory would need to be pre-allocated at max batch size and partially filled, or re-allocated per batch (negating the benefit). Pinned host memory for inputs is the more practical path (see below)
- `TensorRefMut::from_raw` (unsafe) wraps an existing CUDA device pointer directly, enabling zero-copy interop with buffers managed outside ORT (e.g. cudarc)

**Pinned memory and the buffer pool**
- `cuMemHostRegister` can register the buffer pool's backing allocation as pinned (page-locked) at startup — no copy required, no re-allocation. The CUDA DMA engine can then transfer from pool slices directly via `cudaMemcpyAsync`. As of CUDA 4.1+ there is no alignment or size requirement beyond what the OS page table can handle
- With the `CU_MEMHOSTREGISTER_DEVICEMAP` flag, a device-side pointer to the pinned pool memory is also available via `cuMemHostGetDevicePointer`. This pointer can be passed directly to `TensorRefMut::from_raw` with `AllocationDevice::CUDA_PINNED` memory info — the GPU reads from it without any explicit H2D copy
- Requires `CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED` to be true; guaranteed on standard x86 Linux + discrete NVIDIA GPU, not guaranteed on embedded platforms
- `ort` also exposes `AllocationDevice::CUDA_PINNED` through its own allocator for creating pinned tensors managed by ORT, but registering the existing pool avoids a second allocation entirely

**Thread model**
- Two new threads (one per consumer) replacing the current single batch processor thread
- Or: single thread with explicit interleaving of submission and result polling (simpler but serializes GPU pipeline)

## Open Questions

- GPU framework: ONNX Runtime (`ort` 2.x, CUDA EP) — resolved
- Multiple in-flight batches: not worth it — resolved (single session, one batch at a time)
- Whether to pursue `cuMemHostRegister` on the buffer pool backing allocation for zero-copy H2D, or accept a staging copy into a pinned ORT-managed tensor
- Whether `IoBinding` with a device-resident output tensor + `disable_device_sync` is worth the manual stream sync complexity vs. just letting ORT sync at run completion
- Whether CUDA Graphs are viable (requires fixed batch size or separate sessions per size; deferred sync bug open)
- `InferenceEvent` needs no changes — all needed metadata already present
- Handling GPU errors mid-batch (connections have in-flight requests; need a defined failure response or close)
- Whether `request_seq` should be used for strict per-connection response ordering or remains best-effort
