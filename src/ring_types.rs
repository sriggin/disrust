use crate::buffer_pool::PoolSlice;
use crate::constants::FEATURE_DIM;
use std::mem::size_of;

/// Entry in the disruptor ring buffer. Pre-allocated per slot via factory.
/// IO threads fill these in the publish closure; the ONNX submission/completion
/// consumers read them.
///
/// Invariants:
/// - `io_thread_id`: ingress thread id (u8 → max 256 IO threads).
/// - `conn_id`: slab key from the IO thread's Slab<Connection> (u16 → slab capacity must be ≤ 65535).
/// - `num_vectors`: 1..=MAX_VECTORS_PER_REQUEST (u8).
#[repr(C, align(64))]
pub struct InferenceEvent {
    pub io_thread_id: u8,
    pub conn_id: u16,
    pub generation: u32,
    pub request_seq: u64,
    pub num_vectors: u8,
    pub published_at_ns: u64,
    pub features: PoolSlice,
}

impl InferenceEvent {
    /// Factory for disruptor - creates empty events that will be filled with real data.
    pub fn factory() -> Self {
        Self {
            io_thread_id: 0,
            conn_id: 0,
            generation: 0,
            request_seq: 0,
            num_vectors: 0,
            published_at_ns: 0,
            features: PoolSlice::empty(),
        }
    }

    /// Get the feature slice for vector `i`.
    pub fn vector(&self, i: usize) -> &[f32] {
        self.features.vector(i, FEATURE_DIM)
    }
}

const _: () = assert!(
    size_of::<InferenceEvent>() == 64,
    "InferenceEvent must be exactly 64 bytes"
);
