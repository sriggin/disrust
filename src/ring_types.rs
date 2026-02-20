use crate::buffer_pool::PoolSlice;

pub const FEATURE_DIM: usize = 128;
pub const MAX_VECTORS_PER_REQUEST: usize = 64;

/// Entry in the disruptor ring buffer. Pre-allocated per slot via factory.
/// IO threads fill these in the publish closure; batch processor reads them.
#[repr(C, align(64))]
pub struct InferenceEvent {
    pub io_thread_id: u16,
    pub conn_id: u32,
    pub request_seq: u64,
    pub num_vectors: u32,
    pub features: PoolSlice,
}

impl InferenceEvent {
    /// Factory for disruptor - creates empty events that will be filled with real data.
    pub fn factory() -> Self {
        Self {
            io_thread_id: 0,
            conn_id: 0,
            request_seq: 0,
            num_vectors: 0,
            features: PoolSlice::empty(),
        }
    }

    /// Get the feature slice for vector `i`.
    pub fn vector(&self, i: usize) -> &[f32] {
        self.features.vector(i, FEATURE_DIM)
    }
}

/// Response sent back from batch processor to IO threads.
pub struct InferenceResponse {
    pub conn_id: u32,
    pub request_seq: u64,
    pub num_vectors: u32,
    pub results: [f32; MAX_VECTORS_PER_REQUEST],
}

impl InferenceResponse {
    pub fn new() -> Self {
        Self {
            conn_id: 0,
            request_seq: 0,
            num_vectors: 0,
            results: [0.0f32; MAX_VECTORS_PER_REQUEST],
        }
    }

    pub fn results_slice(&self) -> &[f32] {
        &self.results[..self.num_vectors as usize]
    }
}
