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
/// CRITICAL: Pool must be >= disruptor capacity * max request size to prevent
/// wraparound from overwriting unread data. Worst-case sizing (conservative).
/// See PERFORMANCE.md for right-sizing opportunities.
pub const BUFFER_POOL_CAPACITY: usize =
    GPU_DISRUPTOR_SIZE * MAX_VECTORS_PER_REQUEST * FEATURE_DIM;

// ---------------------------------------------------------------------------
// ONNX/CUDA server pipeline
// ---------------------------------------------------------------------------

/// Number of ORT sessions in the session pool (N=1 = one batch in-flight at a time).
pub const SESSION_POOL_SIZE: usize = 1;

/// Maximum disruptor ring slots accumulated into a single GPU batch.
pub const MAX_SESSION_BATCH_SIZE: usize = 256;

/// Request ring buffer size for the ONNX pipeline.
///
/// With the current disruptor guard semantics, SubmissionConsumer must be able to
/// submit all slots visible in a single poll without waiting for CompletionConsumer.
/// That means the ring cannot expose more than one batch per in-flight session at once.
pub const GPU_DISRUPTOR_SIZE: usize = SESSION_POOL_SIZE * MAX_SESSION_BATCH_SIZE;

/// Pinned host buffer pool capacity in f32 units (pass to `BufferPool::from_raw_ptr`).
/// Byte count for `cuMemAllocHost` = `GPU_BUFFER_POOL_CAPACITY * size_of::<f32>()`.
pub const GPU_BUFFER_POOL_CAPACITY: usize =
    GPU_DISRUPTOR_SIZE * MAX_VECTORS_PER_REQUEST * FEATURE_DIM;

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
    BUFFER_POOL_CAPACITY >= GPU_DISRUPTOR_SIZE * FEATURE_DIM,
    "buffer pool capacity is too small for disruptor size"
);
