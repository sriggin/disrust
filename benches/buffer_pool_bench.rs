use disrust::buffer_pool::BufferPool;
use std::env;
use std::hint::black_box;

const FEATURE_DIM: usize = 128;
const DEFAULT_POOL_CAPACITY: usize = 65536 * 64 * FEATURE_DIM; // Same as real config
const ITERATIONS: usize = 10_000_000;

fn bench_size(pool: &'static BufferPool, size: usize, label: &str) {
    // Warm up
    for _ in 0..10000 {
        let mut slice = pool.alloc(size).unwrap();
        slice.as_mut_slice()[0] = 1.0;
        black_box(&slice);
        drop(slice.freeze());
    }

    let start = std::time::Instant::now();

    // Simulate ring buffer wraparound pattern: keep last N allocations alive
    // Size ring to use ~50% of pool capacity to allow headroom
    let (_, pool_capacity) = pool.utilization();
    let max_possible = pool_capacity / size; // Max allocations that fit in pool
    let ring_size = (max_possible / 2).max(1).min(1024);
    let mut ring: Vec<_> = (0..ring_size)
        .map(|_| pool.alloc(size).unwrap().freeze())
        .collect();

    for i in 0..ITERATIONS {
        // Allocate new slice
        let mut new_slice = pool.alloc(size).unwrap();
        new_slice.as_mut_slice()[0] = i as f32;
        black_box(&new_slice);
        let frozen = new_slice.freeze();

        // Drop old slice (simulates disruptor wraparound)
        ring[i % ring_size] = frozen;
    }

    let elapsed = start.elapsed();
    let ops_per_sec = ITERATIONS as f64 / elapsed.as_secs_f64();
    let ns_per_op = elapsed.as_nanos() as f64 / ITERATIONS as f64;

    eprintln!(
        "{:25} {:8.2} ns/op  {:12.0} ops/sec",
        label, ns_per_op, ops_per_sec
    );

    // Keep ring alive
    black_box(ring);
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 && args[1] == "--pool-sizes" {
        // Benchmark different pool sizes with fixed allocation size
        let alloc_size = FEATURE_DIM * 8; // 8 vectors (common case)
        eprintln!(
            "Benchmarking different pool sizes (alloc size: {} f32s)\n",
            alloc_size
        );

        // Test from L1 cache (~32 KB) up to 1 GB
        for &multiplier in &[
            32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
            524288, 1048576, 2097152,
        ] {
            let capacity = multiplier * FEATURE_DIM;
            let pool = BufferPool::leak_new(capacity);
            let size_bytes = capacity * 4;
            let label = if size_bytes < 1024 * 1024 {
                format!("{} KB pool", size_bytes / 1024)
            } else {
                format!("{} MB pool", size_bytes / (1024 * 1024))
            };
            bench_size(&pool, alloc_size, &label);
        }
    } else if args.len() > 1 && args[1] == "--alloc-sizes" {
        // Benchmark different allocation sizes with fixed pool
        let pool = BufferPool::leak_new(DEFAULT_POOL_CAPACITY);
        eprintln!(
            "Pool capacity: {} f32s ({} MB)",
            DEFAULT_POOL_CAPACITY,
            DEFAULT_POOL_CAPACITY * 4 / 1_000_000
        );
        eprintln!("Benchmarking different allocation sizes\n");

        bench_size(&pool, FEATURE_DIM, "1 vector (128 f32)");
        bench_size(&pool, FEATURE_DIM * 2, "2 vectors (256 f32)");
        bench_size(&pool, FEATURE_DIM * 4, "4 vectors (512 f32)");
        bench_size(&pool, FEATURE_DIM * 8, "8 vectors (1024 f32)");
        bench_size(&pool, FEATURE_DIM * 16, "16 vectors (2048 f32)");
        bench_size(&pool, FEATURE_DIM * 32, "32 vectors (4096 f32)");
        bench_size(&pool, FEATURE_DIM * 64, "64 vectors (8192 f32)");
    } else {
        eprintln!("Usage:");
        eprintln!(
            "  cargo bench --bench buffer_pool_bench -- --alloc-sizes   # Test different allocation sizes"
        );
        eprintln!(
            "  cargo bench --bench buffer_pool_bench -- --pool-sizes    # Test different pool capacities"
        );
    }
}
