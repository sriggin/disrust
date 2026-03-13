//! Server sizing and operational configuration.
//!
//! Hardcoded values that are not necessarily shared protocol constants.
//! Protocol constants (e.g. `FEATURE_DIM`, `MAX_VECTORS_PER_REQUEST`) live in `constants`.

use crate::constants::{FEATURE_DIM, MAX_VECTORS_PER_REQUEST};
use std::mem::size_of;

/// io_thread_id is u8 in InferenceEvent; do not spawn more than this many IO threads.
pub const MAX_IO_THREADS: usize = 256;

/// Per-connection read buffer size (bytes).
pub const READ_BUF_SIZE: usize = 65536;

/// Max concurrent connections per IO thread. Must fit in u16 (conn_id).
pub const SLAB_CAPACITY: usize = 4096;

/// Size each buffer pool to handle all in-flight requests at max size.
/// CRITICAL: Pool must be >= request ring capacity * max request size to prevent
/// wraparound from overwriting unread data. Worst-case sizing (conservative).
/// See PERFORMANCE.md for right-sizing opportunities.
pub const BUFFER_POOL_CAPACITY: usize =
    GPU_REQUEST_RING_SIZE * MAX_VECTORS_PER_REQUEST * FEATURE_DIM;

// ---------------------------------------------------------------------------
// ONNX/CUDA server pipeline
// ---------------------------------------------------------------------------

/// Number of ORT sessions in the session pool (N=1 = one batch in-flight at a time).
pub const SESSION_POOL_SIZE: usize = 1;

/// ORT intra-op thread count per session. Must be >= 2 for RunAsync: ORT requires at
/// least one pool thread to dispatch async work (1 = caller only, no pool = RunAsync
/// aborts). For the CUDA EP the pool thread is otherwise idle since ops run on the GPU;
/// it is the minimum tax for using the async API.
pub const ORT_INTRA_THREADS: usize = 2;

/// Maximum disruptor ring slots accumulated into a single GPU batch.
pub const MAX_SESSION_BATCH_SIZE: usize = 256;

/// Default coalescing window, in microseconds, for a partial batch once a session is available.
pub const DEFAULT_BATCH_COALESCE_US: u64 = 500;

/// Request ring capacity for the ONNX pipeline.
///
/// This is intentionally independent from `MAX_SESSION_BATCH_SIZE`: the disruptor ring controls
/// how much ingress backlog the pipeline can absorb, while `MAX_SESSION_BATCH_SIZE` controls how
/// much work a single GPU submission may include.
pub const GPU_REQUEST_RING_SIZE: usize = 4096;

/// Request ring buffer size for the ONNX pipeline.
pub const GPU_DISRUPTOR_SIZE: usize = GPU_REQUEST_RING_SIZE;

/// Pinned host buffer pool capacity in f32 units (pass to `BufferPool::from_raw_ptr`).
/// Byte count for `cuMemAllocHost` = `GPU_BUFFER_POOL_CAPACITY * size_of::<f32>()`.
pub const GPU_BUFFER_POOL_CAPACITY: usize =
    GPU_REQUEST_RING_SIZE * MAX_VECTORS_PER_REQUEST * FEATURE_DIM;

/// Byte size of the pinned host buffer pool allocation.
pub const GPU_BUFFER_POOL_BYTES: usize = GPU_BUFFER_POOL_CAPACITY * size_of::<f32>();

/// Batch queue capacity: +1 absorbs one wrap-induced extra batch per cycle (prevents deadlock).
pub const BATCH_QUEUE_CAPACITY: usize = SESSION_POOL_SIZE + 1;

/// Total write buffer slots (double-buffered: 2 × MAX_SESSION_BATCH_SIZE).
pub const WRITE_BUF_SLOTS: usize = MAX_SESSION_BATCH_SIZE * 2;

/// Per-slot write buffer byte size: 1 byte num_vectors + up to MAX_VECTORS_PER_REQUEST f32 results.
pub const WRITE_BUF_SIZE: usize = 1 + MAX_VECTORS_PER_REQUEST * size_of::<f32>();

/// Maximum vectors in a single GPU batch (slots × max vectors per slot).
pub const MAX_BATCH_VECTORS: usize = MAX_SESSION_BATCH_SIZE * MAX_VECTORS_PER_REQUEST;

// ---------------------------------------------------------------------------
// Compile-time sanity checks
// ---------------------------------------------------------------------------

// Compile-time sanity checks
const _: () = assert!(
    SLAB_CAPACITY <= u16::MAX as usize,
    "SLAB_CAPACITY must fit in u16 (conn_id)"
);
const _: () = assert!(
    BUFFER_POOL_CAPACITY >= GPU_REQUEST_RING_SIZE * FEATURE_DIM,
    "buffer pool capacity is too small for disruptor size"
);
