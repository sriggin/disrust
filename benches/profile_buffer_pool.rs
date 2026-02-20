use disrust::buffer_pool::BufferPool;
use std::env;
use std::hint::black_box;

const FEATURE_DIM: usize = 128;
const POOL_CAPACITY: usize = 64 * 1024 * FEATURE_DIM; // 64MB pool
const DEFAULT_ITERATIONS: usize = 100_000_000; // 100M iterations ~= 7-10 seconds

fn main() {
    let iterations: usize = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_ITERATIONS);

    let pool = BufferPool::leak_new(POOL_CAPACITY);
    let alloc_size = FEATURE_DIM * 8; // 8 vectors

    // Warm up
    for _ in 0..10000 {
        let slice = pool.alloc(alloc_size).unwrap().freeze();
        black_box(&slice);
        drop(slice);
    }

    let ring_size = 1024;
    let mut ring: Vec<_> = (0..ring_size)
        .map(|_| pool.alloc(alloc_size).unwrap().freeze())
        .collect();

    eprintln!("Running {} iterations for perf profiling...", iterations);

    for i in 0..iterations {
        let new_slice = pool.alloc(alloc_size).unwrap();
        black_box(&new_slice);
        let frozen = new_slice.freeze();
        ring[i % ring_size] = frozen;
    }

    black_box(ring);
    eprintln!("Done.");
}
