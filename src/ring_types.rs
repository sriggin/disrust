use crate::buffer_pool::PoolSlice;
use crate::connection_id::ConnectionRef;
use crate::constants::FEATURE_DIM;
use std::mem::size_of;

/// Entry in the disruptor ring buffer. Pre-allocated per slot via factory.
/// IO threads fill these in the publish closure; the ONNX submission/completion
/// consumers read them.
///
/// Invariants:
/// - `conn`: logical connection identity `(shard, conn_id, generation)` packed into 32 bits.
/// - `num_vectors`: 1..=MAX_VECTORS_PER_REQUEST (u8).
#[repr(C, align(64))]
pub struct InferenceEvent {
    pub conn: ConnectionRef,
    pub request_seq: u64,
    pub num_vectors: u8,
    pub published_at_ns: u64,
    pub features: PoolSlice,
}

impl InferenceEvent {
    /// Factory for disruptor - creates empty events that will be filled with real data.
    pub fn factory() -> Self {
        Self {
            conn: ConnectionRef::new(0, 0, 1),
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

    pub fn io_thread_id(&self) -> u8 {
        self.conn.shard_id()
    }

    pub fn conn_id(&self) -> u16 {
        self.conn.conn_id
    }

    pub fn generation(&self) -> u16 {
        self.conn.generation()
    }
}

const _: () = assert!(
    size_of::<InferenceEvent>() == 64,
    "InferenceEvent must be exactly 64 bytes"
);
