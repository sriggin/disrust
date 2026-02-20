use crate::buffer_pool::{BufferPool, PoolSlice};
use crate::constants::FEATURE_DIM;
use std::mem::size_of;

// Calculate inline capacity using const math to fit exactly in 64-byte cache line
pub const INLINE_RESULT_CAPACITY: usize = {
    const METADATA_SIZE: usize = size_of::<u64>() +   // request_seq
        size_of::<u32>() +   // conn_id
        size_of::<u32>(); // num_vectors

    const TARGET_STRUCT_SIZE: usize = 64;
    const ENUM_BUDGET: usize = TARGET_STRUCT_SIZE - METADATA_SIZE; // 48 bytes

    // Enum is max(Inline size, Pooled size) + discriminant
    // Conservatively reserve 8 bytes for discriminant/padding
    const INLINE_BYTES: usize = ENUM_BUDGET - 8; // 40 bytes

    INLINE_BYTES / size_of::<f32>()
};

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

/// Storage for inference results - either inline or pooled depending on size.
pub enum ResultStorage {
    /// Small results (≤INLINE_RESULT_CAPACITY) stored inline in cache line.
    Inline([f32; INLINE_RESULT_CAPACITY]),

    /// Large results (>INLINE_RESULT_CAPACITY) stored in buffer pool.
    Pooled(PoolSlice),
}

/// Response sent back from batch processor to IO threads.
/// Sized to fit EXACTLY in a 64-byte cache line (no padding needed).
/// Optimized to inline small results (≤INLINE_RESULT_CAPACITY vectors).
#[repr(C, align(64))]
pub struct InferenceResponse {
    pub request_seq: u64,
    pub conn_id: u32,
    pub num_vectors: u32,
    pub results: ResultStorage,
}

// Static assertions - fail at compile time if size is wrong
const _: () = assert!(
    size_of::<InferenceResponse>() == 64,
    "InferenceResponse must be exactly 64 bytes"
);

const _: () = assert!(
    size_of::<InferenceEvent>() == 64,
    "InferenceEvent must be exactly 64 bytes"
);

const _: () = assert!(
    INLINE_RESULT_CAPACITY >= 8,
    "Inline capacity should handle typical workloads (1-8 vectors)"
);

impl InferenceResponse {
    /// Create a new response. Automatically chooses inline vs pooled storage.
    ///
    /// - Small results (≤INLINE_RESULT_CAPACITY): Uses inline storage (zero allocations)
    /// - Large results (>INLINE_RESULT_CAPACITY): Allocates from pool
    pub fn with_results(
        conn_id: u32,
        request_seq: u64,
        results: &[f32],
        pool: Option<&'static BufferPool>,
    ) -> Result<Self, &'static str> {
        let num_vectors = results.len() as u32;

        if results.len() <= INLINE_RESULT_CAPACITY {
            // Fast path: inline storage, no pool needed
            let mut inline_array = [0.0f32; INLINE_RESULT_CAPACITY];
            inline_array[..results.len()].copy_from_slice(results);

            Ok(Self {
                conn_id,
                request_seq,
                num_vectors,
                results: ResultStorage::Inline(inline_array),
            })
        } else {
            // Large results: allocate from pool
            let pool = pool.ok_or("pool required for large results")?;

            let mut pool_slice = pool
                .alloc(results.len())
                .map_err(|_| "pool allocation failed")?;
            pool_slice.as_mut_slice().copy_from_slice(results);

            Ok(Self {
                conn_id,
                request_seq,
                num_vectors,
                results: ResultStorage::Pooled(pool_slice.freeze()),
            })
        }
    }

    /// Get result slice for reading (works for both inline and pooled).
    pub fn results_slice(&self) -> &[f32] {
        match &self.results {
            ResultStorage::Inline(arr) => &arr[..self.num_vectors as usize],
            ResultStorage::Pooled(slice) => slice.as_slice(),
        }
    }

    /// Factory function for disruptor pre-allocation (creates empty inline response).
    pub fn new() -> Self {
        Self {
            conn_id: 0,
            request_seq: 0,
            num_vectors: 0,
            results: ResultStorage::Inline([0.0f32; INLINE_RESULT_CAPACITY]),
        }
    }
}
