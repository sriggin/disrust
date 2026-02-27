use std::cell::UnsafeCell;
use std::sync::{
    OnceLock,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use crate::metrics;
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
pub struct PoolSlice {
    pool: &'static BufferPool,
    data: *const f32,
    len: usize,
    freed: AtomicBool,
}

unsafe impl Send for PoolSlice {}
unsafe impl Sync for PoolSlice {}

impl Drop for PoolSlice {
    fn drop(&mut self) {
        if self.len > 0 && !self.freed.swap(true, Ordering::AcqRel) {
            self.pool.read_cursor.fetch_add(self.len, Ordering::Release);
        }
    }
}

impl PoolSlice {
    /// Create an empty PoolSlice (for initialization).
    pub fn empty() -> Self {
        Self {
            pool: factory_pool(),
            data: std::ptr::NonNull::dangling().as_ptr(),
            len: 0,
            freed: AtomicBool::new(false),
        }
    }

    /// Get a slice view of the pooled data.
    #[allow(dead_code)]
    pub fn as_slice(&self) -> &[f32] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }

    /// Get the vector at index `i` (assumes FEATURE_DIM stride).
    pub fn vector(&self, i: usize, feature_dim: usize) -> &[f32] {
        let start = i * feature_dim;
        let end = start.saturating_add(feature_dim);
        assert!(end <= self.len, "vector index out of bounds");
        unsafe { std::slice::from_raw_parts(self.data.add(start), feature_dim) }
    }

    /// Release this slice back to the pool early. Safe to call at most once.
    pub fn release(&self) {
        if self.len > 0 && !self.freed.swap(true, Ordering::AcqRel) {
            self.pool.read_cursor.fetch_add(self.len, Ordering::Release);
        }
    }
}

/// Mutable slice backed by a buffer pool arena. Can be frozen to PoolSlice.
pub struct PoolSliceMut {
    pool: &'static BufferPool,
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
            freed: AtomicBool::new(false),
        }
    }
}

/// Ring buffer pool for variable-length feature vectors with explicit free tracking.
/// Producer allocates (advances write cursor), consumer frees (advances read cursor).
///
/// # Performance Characteristics
///
/// **Single-threaded access:** Despite using atomics, all pool operations happen
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
    data: Box<[UnsafeCell<f32>]>,
    capacity: usize,
    write_cursor: AtomicUsize,
    read_cursor: AtomicUsize,
}

unsafe impl Send for BufferPool {}
unsafe impl Sync for BufferPool {}

impl BufferPool {
    /// Create a new buffer pool with capacity for `capacity` f32 values.
    /// Pre-touches all pages to fault them in upfront, avoiding page fault latency
    /// during operation. Should be called on the thread that will use the pool
    /// for correct NUMA placement.
    pub fn new_boxed(capacity: usize) -> Box<Self> {
        let data: Vec<UnsafeCell<f32>> = (0..capacity).map(|_| UnsafeCell::new(0.0f32)).collect();
        let data = data.into_boxed_slice();

        // Touch every page (4KB = 1024 f32s) to force physical allocation upfront.
        for i in (0..capacity).step_by(1024) {
            unsafe {
                *data[i].get() = 0.0f32;
            }
        }

        Box::new(Self {
            data,
            capacity,
            write_cursor: AtomicUsize::new(0),
            read_cursor: AtomicUsize::new(0),
        })
    }

    /// Create a new buffer pool and leak it to obtain a `'static` reference.
    /// Should be called on the thread that will use the pool for correct NUMA placement.
    pub fn leak_new(capacity: usize) -> &'static Self {
        Box::leak(Self::new_boxed(capacity))
    }

    /// Allocate space for `len` f32 values, returning a mutable slice.
    /// Wraps to offset 0 if allocation would straddle the end.
    ///
    /// # Errors
    /// - `AllocError::TooLarge` if `len` exceeds pool capacity
    /// - `AllocError::Exhausted` if pool is full (producer outpacing consumer)
    pub fn alloc(&'static self, len: usize) -> Result<PoolSliceMut, AllocError> {
        if len > self.capacity {
            metrics::inc_pool_too_large();
            return Err(AllocError::TooLarge {
                requested: len,
                capacity: self.capacity,
            });
        }

        // Check if we have space (write cursor can't lap read cursor by more than capacity)
        let write = self.write_cursor.load(Ordering::Acquire);
        let read = self.read_cursor.load(Ordering::Acquire);
        let in_use = write.wrapping_sub(read);

        if in_use + len > self.capacity {
            metrics::inc_pool_exhausted();
            return Err(AllocError::Exhausted {
                in_use,
                capacity: self.capacity,
            });
        }
        metrics::update_pool_in_use(in_use + len);

        // Allocate
        let offset = write % self.capacity;
        let actual_offset = if offset + len > self.capacity {
            // Would straddle the end, wrap to 0
            self.write_cursor
                .store(write + (self.capacity - offset) + len, Ordering::Release);
            0
        } else {
            self.write_cursor.store(write + len, Ordering::Release);
            offset
        };

        let base = self.data.as_ptr() as *mut f32;
        let ptr = unsafe { base.add(actual_offset) };
        Ok(PoolSliceMut {
            pool: self,
            data: ptr,
            len,
        })
    }

    /// Get current pool utilization for debugging.
    #[allow(dead_code)]
    pub fn utilization(&self) -> (usize, usize) {
        let write = self.write_cursor.load(Ordering::Acquire);
        let read = self.read_cursor.load(Ordering::Acquire);
        (write.wrapping_sub(read), self.capacity)
    }
}

static FACTORY_POOL: OnceLock<Box<BufferPool>> = OnceLock::new();

/// Set the pool used for factory-created empty slices.
pub fn set_factory_pool(pool: Box<BufferPool>) -> &'static BufferPool {
    let _ = FACTORY_POOL.set(pool);
    factory_pool()
}

fn factory_pool() -> &'static BufferPool {
    FACTORY_POOL
        .get()
        .expect("factory pool not initialized")
        .as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;

    static INIT_FACTORY_POOL: Once = Once::new();

    fn init_factory_pool() {
        INIT_FACTORY_POOL.call_once(|| {
            let pool = BufferPool::new_boxed(1);
            set_factory_pool(pool);
        });
    }

    fn with_pool<F>(capacity: usize, f: F)
    where
        F: FnOnce(&'static BufferPool),
    {
        let boxed = BufferPool::new_boxed(capacity);
        let ptr = Box::into_raw(boxed);
        struct PoolGuard {
            ptr: *mut BufferPool,
        }
        impl Drop for PoolGuard {
            fn drop(&mut self) {
                unsafe {
                    drop(Box::from_raw(self.ptr));
                }
            }
        }
        let _guard = PoolGuard { ptr };
        let pool = unsafe { &*ptr };
        f(pool);
    }

    #[test]
    fn basic_alloc() {
        with_pool(1000, |pool| {
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
        });
    }

    #[test]
    fn wraparound() {
        with_pool(100, |pool| {
            let s1 = pool.alloc(95).expect("alloc failed").freeze();
            drop(s1); // Free to make space

            // Next allocation would straddle, should wrap to 0
            let mut m2 = pool.alloc(10).expect("alloc failed");
            m2.as_mut_slice().fill(42.0);
            let s2 = m2.freeze();

            assert_eq!(s2.as_slice().len(), 10);
            assert!(s2.as_slice().iter().all(|&x| x == 42.0));
        });
    }

    #[test]
    fn read_write() {
        with_pool(100, |pool| {
            let mut mutable = pool.alloc(5).expect("alloc failed");
            mutable
                .as_mut_slice()
                .copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
            let immutable = mutable.freeze();

            assert_eq!(immutable.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
        });
    }

    #[test]
    fn alloc_too_large() {
        with_pool(100, |pool| {
            let result = pool.alloc(200);
            assert!(matches!(result, Err(AllocError::TooLarge { .. })));
        });
    }

    #[test]
    fn exhaustion() {
        with_pool(100, |pool| {
            let _s1 = pool.alloc(90).expect("alloc failed");
            // Pool exhausted - can't allocate 20 more
            let result = pool.alloc(20);
            assert!(matches!(result, Err(AllocError::Exhausted { .. })));
        });
    }

    #[test]
    fn alloc_zero_len_is_ok_and_noop() {
        with_pool(10, |pool| {
            let before = pool.utilization();
            let mut m = pool.alloc(0).expect("alloc failed");
            assert_eq!(m.as_mut_slice().len(), 0);
            drop(m.freeze());
            let after = pool.utilization();
            assert_eq!(before, after);
        });
    }

    #[test]
    fn alloc_exact_capacity_then_free_allows_reuse() {
        with_pool(8, |pool| {
            let s = pool.alloc(8).expect("alloc failed").freeze();
            assert!(matches!(pool.alloc(1), Err(AllocError::Exhausted { .. })));
            drop(s);
            assert!(pool.alloc(1).is_ok());
        });
    }

    #[test]
    fn exhaustion_then_reuse_after_drop() {
        with_pool(100, |pool| {
            let s1 = pool.alloc(95).expect("alloc failed").freeze();
            assert!(matches!(pool.alloc(10), Err(AllocError::Exhausted { .. })));
            drop(s1);
            assert!(pool.alloc(10).is_ok());
        });
    }

    #[test]
    fn utilization_tracks_simple_alloc_and_drop() {
        with_pool(50, |pool| {
            let s1 = pool.alloc(12).expect("alloc failed").freeze();
            assert_eq!(pool.utilization(), (12, 50));
            drop(s1);
            assert_eq!(pool.utilization(), (0, 50));
        });
    }

    #[test]
    fn release_frees_once() {
        with_pool(10, |pool| {
            let s = pool.alloc(10).expect("alloc failed").freeze();
            assert_eq!(pool.utilization(), (10, 10));
            s.release();
            assert_eq!(pool.utilization(), (0, 10));
            drop(s);
            assert_eq!(pool.utilization(), (0, 10));
        });
    }

    #[test]
    fn vector_access_returns_expected_slice() {
        with_pool(32, |pool| {
            let mut m = pool.alloc(8).expect("alloc failed");
            m.as_mut_slice()
                .copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);
            let s = m.freeze();

            assert_eq!(s.vector(0, 4), &[1.0, 2.0, 3.0, 4.0]);
            assert_eq!(s.vector(1, 4), &[10.0, 20.0, 30.0, 40.0]);
        });
    }

    #[test]
    #[should_panic]
    fn vector_access_out_of_bounds_panics() {
        with_pool(8, |pool| {
            let s = pool.alloc(4).expect("alloc failed").freeze();
            let _ = s.vector(1, 4);
        });
    }

    #[test]
    fn empty_slice_requires_factory_pool() {
        init_factory_pool();
        let s = PoolSlice::empty();
        drop(s);
    }

    #[test]
    fn wrap_like_behavior_still_allows_allocation() {
        with_pool(10, |pool| {
            let s1 = pool.alloc(7).expect("alloc failed").freeze();
            drop(s1);
            // This may require wrap internally, but from the API view it should succeed.
            let s2 = pool.alloc(7).expect("alloc failed").freeze();
            assert_eq!(s2.as_slice().len(), 7);
        });
    }

    #[test]
    fn alloc_too_large_error_includes_requested_and_capacity() {
        with_pool(5, |pool| match pool.alloc(6) {
            Err(AllocError::TooLarge {
                requested,
                capacity,
            }) => {
                assert_eq!(requested, 6);
                assert_eq!(capacity, 5);
            }
            Err(_) => panic!("unexpected error variant"),
            Ok(_) => panic!("expected error"),
        });
    }

    #[test]
    fn exhausted_error_includes_in_use_and_capacity() {
        with_pool(10, |pool| {
            let _s1 = pool.alloc(7).expect("alloc failed");
            let _s2 = pool.alloc(3).expect("alloc failed");
            match pool.alloc(1) {
                Err(AllocError::Exhausted { in_use, capacity }) => {
                    assert_eq!(capacity, 10);
                    assert_eq!(in_use, 10);
                }
                Err(_) => panic!("unexpected error variant"),
                Ok(_) => panic!("expected error"),
            }
        });
    }

    #[test]
    fn multiple_allocations_fill_capacity_then_fail() {
        with_pool(12, |pool| {
            let _a = pool.alloc(5).expect("alloc failed");
            let _b = pool.alloc(4).expect("alloc failed");
            let _c = pool.alloc(3).expect("alloc failed");
            assert!(matches!(pool.alloc(1), Err(AllocError::Exhausted { .. })));
        });
    }

    #[test]
    fn reuse_after_partial_drop_allows_additional_allocations() {
        with_pool(20, |pool| {
            let s1 = pool.alloc(8).expect("alloc failed").freeze();
            let s2 = pool.alloc(8).expect("alloc failed").freeze();
            assert!(matches!(pool.alloc(5), Err(AllocError::Exhausted { .. })));
            drop(s1);
            assert!(pool.alloc(5).is_ok());
            drop(s2);
        });
    }

    #[test]
    fn freeze_preserves_written_data_across_multiple_allocations() {
        with_pool(16, |pool| {
            let mut a = pool.alloc(4).expect("alloc failed");
            a.as_mut_slice().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
            let sa = a.freeze();

            let mut b = pool.alloc(4).expect("alloc failed");
            b.as_mut_slice().copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
            let sb = b.freeze();

            assert_eq!(sa.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
            assert_eq!(sb.as_slice(), &[5.0, 6.0, 7.0, 8.0]);
        });
    }
}
