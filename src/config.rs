//! Server sizing and operational configuration.
//!
//! Hardcoded values that are not necessarily shared protocol constants.
//! Protocol constants (e.g. `FEATURE_DIM`, `MAX_VECTORS_PER_REQUEST`) live in `constants`.

use crate::constants::{FEATURE_DIM, MAX_VECTORS_PER_REQUEST};

/// io_thread_id is u8 in InferenceEvent; do not spawn more than this many IO threads.
pub const MAX_IO_THREADS: usize = 256;

/// Request ring buffer size (disruptor capacity).
pub const DISRUPTOR_SIZE: usize = 65536;

/// Per-thread response queue size. Must be >= DISRUPTOR_SIZE to avoid deadlock.
pub const RESPONSE_QUEUE_SIZE: usize = DISRUPTOR_SIZE;

/// Per-connection read buffer size (bytes).
pub const READ_BUF_SIZE: usize = 65536;

/// Max concurrent connections per IO thread. Must fit in u16 (conn_id).
pub const SLAB_CAPACITY: usize = 4096;

/// Size each buffer pool to handle all in-flight requests at max size.
/// CRITICAL: Pool must be >= disruptor capacity * max request size to prevent
/// wraparound from overwriting unread data. Worst-case sizing (conservative).
/// See PERFORMANCE.md for right-sizing opportunities.
pub const BUFFER_POOL_CAPACITY: usize = DISRUPTOR_SIZE * MAX_VECTORS_PER_REQUEST * FEATURE_DIM;

/// Result pool capacity (for responses > INLINE_RESULT_CAPACITY vectors).
/// Tunable based on expected workload. Most responses are inline.
/// Min: enough for a few large responses. Max: all response queue slots at max size.
pub const RESULT_POOL_CAPACITY: usize = RESPONSE_QUEUE_SIZE * 16;

// Compile-time sanity checks
const _: () = assert!(
    SLAB_CAPACITY <= u16::MAX as usize,
    "SLAB_CAPACITY must fit in u16 (conn_id)"
);
const _: () = assert!(
    BUFFER_POOL_CAPACITY >= DISRUPTOR_SIZE * FEATURE_DIM,
    "buffer pool capacity is too small for disruptor size"
);
const _: () = assert!(
    RESULT_POOL_CAPACITY >= MAX_VECTORS_PER_REQUEST * 4,
    "result pool capacity is too small"
);
const _: () = assert!(
    RESULT_POOL_CAPACITY <= RESPONSE_QUEUE_SIZE * MAX_VECTORS_PER_REQUEST,
    "result pool capacity exceeds maximum needed"
);
