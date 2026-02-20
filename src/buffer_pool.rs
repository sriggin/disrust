use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

/// Error returned when buffer pool allocation fails.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum AllocError {
    /// Allocation would exceed pool capacity (request too large).
    TooLarge { requested: usize, capacity: usize },
    /// Pool exhausted - producer outpacing consumer (apply backpressure).
    Exhausted { in_use: usize, capacity: usize },
}

/// Immutable slice backed by a buffer pool arena.
/// Automatically returns space to pool when dropped (if len > 0).
#[derive(Clone)]
pub struct PoolSlice {
    pool: Option<Arc<BufferPool>>,
    data: *const f32,
    len: usize,
}

impl PoolSlice {
    /// Create an empty PoolSlice (for initialization).
    pub fn empty() -> Self {
        Self {
            pool: None,
            data: std::ptr::null(),
            len: 0,
        }
    }
}

unsafe impl Send for PoolSlice {}
unsafe impl Sync for PoolSlice {}

impl Drop for PoolSlice {
    fn drop(&mut self) {
        if let Some(pool) = &self.pool
            && self.len > 0
        {
            pool.read_cursor.fetch_add(self.len, Ordering::Relaxed);
        }
    }
}

impl PoolSlice {
    /// Get a slice view of the pooled data.
    #[allow(dead_code)]
    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }

    /// Get the vector at index `i` (assumes FEATURE_DIM stride).
    pub fn vector(&self, i: usize, feature_dim: usize) -> &[f32] {
        let start = i * feature_dim;
        unsafe { std::slice::from_raw_parts(self.data.add(start), feature_dim) }
    }
}

/// Mutable slice backed by a buffer pool arena. Can be frozen to PoolSlice.
pub struct PoolSliceMut {
    pool: Option<Arc<BufferPool>>,
    data: *mut f32,
    len: usize,
}

impl PoolSliceMut {
    /// Get a mutable slice view for writing.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.data, self.len) }
    }

    /// Freeze to an immutable PoolSlice.
    pub fn freeze(self) -> PoolSlice {
        PoolSlice {
            pool: self.pool,
            data: self.data,
            len: self.len,
        }
    }
}

/// Ring buffer pool for variable-length feature vectors with explicit free tracking.
/// Producer allocates (advances write cursor), consumer frees (advances read cursor).
///
/// # Performance Characteristics
///
/// **Single-threaded access:** Despite using Arc and atomics, all pool operations happen
/// on a single IO thread. Allocations happen on the IO thread, and drops (via PoolSlice)
/// also happen on the IO thread when the disruptor wraps around. Uses Relaxed ordering
/// since no cross-thread synchronization is needed.
///
/// **Cache locality is critical:** Pool size dramatically impacts performance:
/// - Small pools (< 8MB): ~21 ns/op - fits in cache
/// - Large pools (> 128MB): ~82-485 ns/op - dominated by DRAM latency
///
/// See PERFORMANCE.md for detailed benchmarking results and optimization opportunities.
pub struct BufferPool {
    data: Box<[f32]>,
    capacity: usize,
    write_cursor: AtomicUsize,
    read_cursor: AtomicUsize,
}

impl BufferPool {
    /// Create a new buffer pool with capacity for `capacity` f32 values.
    pub fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            data: vec![0.0f32; capacity].into_boxed_slice(),
            capacity,
            write_cursor: AtomicUsize::new(0),
            read_cursor: AtomicUsize::new(0),
        })
    }

    /// Allocate space for `len` f32 values, returning a mutable slice.
    /// Wraps to offset 0 if allocation would straddle the end.
    ///
    /// # Errors
    /// - `AllocError::TooLarge` if `len` exceeds pool capacity
    /// - `AllocError::Exhausted` if pool is full (producer outpacing consumer)
    pub fn alloc(self: &Arc<Self>, len: usize) -> Result<PoolSliceMut, AllocError> {
        if len > self.capacity {
            return Err(AllocError::TooLarge {
                requested: len,
                capacity: self.capacity,
            });
        }

        // Check if we have space (write cursor can't lap read cursor by more than capacity)
        let write = self.write_cursor.load(Ordering::Relaxed);
        let read = self.read_cursor.load(Ordering::Relaxed);
        let in_use = write.wrapping_sub(read);

        if in_use + len > self.capacity {
            return Err(AllocError::Exhausted {
                in_use,
                capacity: self.capacity,
            });
        }

        // Allocate
        let offset = write % self.capacity;
        let actual_offset = if offset + len > self.capacity {
            // Would straddle the end, wrap to 0
            self.write_cursor
                .store(write + (self.capacity - offset) + len, Ordering::Relaxed);
            0
        } else {
            self.write_cursor.store(write + len, Ordering::Relaxed);
            offset
        };

        let ptr = self.data.as_ptr() as *mut f32;
        Ok(PoolSliceMut {
            pool: Some(Arc::clone(self)),
            data: unsafe { ptr.add(actual_offset) },
            len,
        })
    }

    /// Get current pool utilization for debugging.
    #[allow(dead_code)]
    pub fn utilization(&self) -> (usize, usize) {
        let write = self.write_cursor.load(Ordering::Relaxed);
        let read = self.read_cursor.load(Ordering::Relaxed);
        (write.wrapping_sub(read), self.capacity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_alloc() {
        let pool = BufferPool::new(1000);
        let mut m1 = pool.alloc(10).expect("alloc failed");
        m1.as_mut_slice().fill(1.0);
        let s1 = m1.freeze();

        let mut m2 = pool.alloc(20).expect("alloc failed");
        m2.as_mut_slice().fill(2.0);
        let s2 = m2.freeze();

        assert_eq!(s1.as_slice().len(), 10);
        assert_eq!(s2.as_slice().len(), 20);
        assert!(s1.as_slice().iter().all(|&x| x == 1.0));
        assert!(s2.as_slice().iter().all(|&x| x == 2.0));

        // Drop handles free automatically
        drop(s1);
        drop(s2);
    }

    #[test]
    fn wraparound() {
        let pool = BufferPool::new(100);
        let s1 = pool.alloc(95).expect("alloc failed").freeze();
        drop(s1); // Free to make space

        // Next allocation would straddle, should wrap to 0
        let mut m2 = pool.alloc(10).expect("alloc failed");
        m2.as_mut_slice().fill(42.0);
        let s2 = m2.freeze();

        assert_eq!(s2.as_slice().len(), 10);
        assert!(s2.as_slice().iter().all(|&x| x == 42.0));
    }

    #[test]
    fn read_write() {
        let pool = BufferPool::new(100);
        let mut mutable = pool.alloc(5).expect("alloc failed");
        mutable
            .as_mut_slice()
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let immutable = mutable.freeze();

        assert_eq!(immutable.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn alloc_too_large() {
        let pool = BufferPool::new(100);
        let result = pool.alloc(200);
        assert!(matches!(result, Err(AllocError::TooLarge { .. })));
    }

    #[test]
    fn exhaustion() {
        let pool = BufferPool::new(100);
        let _s1 = pool.alloc(90).expect("alloc failed");
        // Pool exhausted - can't allocate 20 more
        let result = pool.alloc(20);
        assert!(matches!(result, Err(AllocError::Exhausted { .. })));
    }
}
