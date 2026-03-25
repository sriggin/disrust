use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use disrust::buffer_pool::BufferPool;
use disrust::constants::FEATURE_DIM;
use std::hint::black_box;
use std::time::{Duration, Instant};

const DEFAULT_POOL_CAPACITY: usize = 65536 * 64 * FEATURE_DIM;

fn ring_size(pool_capacity: usize, alloc_size: usize) -> usize {
    (pool_capacity / alloc_size / 2).clamp(1, 1024)
}

/// Run one Criterion sample using iter_custom.
///
/// Ring setup and pre-warming happen before the timer starts so that every
/// sample — regardless of the iters count Criterion requests — begins with
/// the full ring's working set already established in cache.
fn run_sample(
    alloc: &mut disrust::buffer_pool::PoolAllocator,
    alloc_size: usize,
    ring_sz: usize,
    iters: u64,
) -> Duration {
    // Untimed: initialize ring.
    let mut ring: Vec<_> = (0..ring_sz)
        .map(|_| alloc.alloc(alloc_size).unwrap().freeze())
        .collect();

    // Untimed: pre-warm — visit every slot once to establish steady-state
    // cache footprint before Criterion starts timing.
    for i in 0..ring_sz {
        let mut s = alloc.alloc(alloc_size).unwrap();
        s.as_mut_slice()[0] = i as f32;
        black_box(&s);
        ring[i % ring_sz] = s.freeze();
    }

    // Timed.
    let start = Instant::now();
    for i in 0..iters as usize {
        let mut s = alloc.alloc(alloc_size).unwrap();
        s.as_mut_slice()[0] = i as f32;
        black_box(&s);
        ring[i % ring_sz] = s.freeze();
    }
    let elapsed = start.elapsed();

    // Untimed: rotate so index 0 is the oldest allocation before dropping.
    // The pool's release() assumes FIFO order; without this, iters % ring_sz != 0
    // leaves the Vec in rotated order and out-of-order drops corrupt the read cursor.
    ring.rotate_left(iters as usize % ring_sz);
    drop(ring);
    elapsed
}

fn alloc_sizes(c: &mut Criterion) {
    let pool = BufferPool::leak_new(DEFAULT_POOL_CAPACITY);
    let mut alloc = pool.allocator();

    let cases: &[(&str, usize)] = &[
        ("1 vec (16 f32)", FEATURE_DIM),
        ("2 vec (32 f32)", FEATURE_DIM * 2),
        ("4 vec (64 f32)", FEATURE_DIM * 4),
        ("8 vec (128 f32)", FEATURE_DIM * 8),
        ("16 vec (256 f32)", FEATURE_DIM * 16),
        ("32 vec (512 f32)", FEATURE_DIM * 32),
        ("64 vec (1024 f32)", FEATURE_DIM * 64),
    ];

    let mut group = c.benchmark_group("buffer_pool/alloc_size");
    group.throughput(Throughput::Elements(1));

    for &(label, alloc_size) in cases {
        let ring_sz = ring_size(DEFAULT_POOL_CAPACITY, alloc_size);
        group.bench_function(BenchmarkId::new("wraparound", label), |b| {
            b.iter_custom(|iters| run_sample(&mut alloc, alloc_size, ring_sz, iters));
        });
    }

    group.finish();
}

fn pool_sizes(c: &mut Criterion) {
    let alloc_size = FEATURE_DIM * 8; // 8 vectors — common case

    // Multipliers span ~2 KB (L1) through ~1 GB (far DRAM).
    let multipliers: &[usize] = &[
        32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288,
        1048576, 2097152,
    ];

    // Allocate all pools upfront so pool creation doesn't touch bench timing.
    let pools: Vec<(&'static BufferPool, String)> = multipliers
        .iter()
        .map(|&m| {
            let capacity = m * FEATURE_DIM;
            let size_bytes = capacity * 4;
            let label = if size_bytes < 1024 * 1024 {
                format!("{} KB", size_bytes / 1024)
            } else {
                format!("{} MB", size_bytes / (1024 * 1024))
            };
            (BufferPool::leak_new(capacity), label)
        })
        .collect();

    let mut group = c.benchmark_group("buffer_pool/pool_size");
    group.throughput(Throughput::Elements(1));
    group.measurement_time(Duration::from_secs(10));

    for (pool, label) in &pools {
        let mut alloc = pool.allocator();
        let (_, pool_capacity) = pool.utilization();
        let ring_sz = ring_size(pool_capacity, alloc_size);
        group.bench_function(BenchmarkId::new("wraparound", label), |b| {
            b.iter_custom(|iters| run_sample(&mut alloc, alloc_size, ring_sz, iters));
        });
    }

    group.finish();
}

criterion_group!(benches, alloc_sizes, pool_sizes);
criterion_main!(benches);
