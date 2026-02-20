# Buffer Pool Performance Notes

## Key Design Decisions

### Single-Threaded Pool Access
The buffer pool appears to require Arc and atomic operations, but **all pool access is actually single-threaded**:
- IO thread allocates from pool
- IO thread publishes to disruptor (moves PoolSlice into event)
- When disruptor wraps, IO thread publishes to same slot → **drops old PoolSlice on IO thread**
- PoolSlice::drop() advances read_cursor

**Implication:** Arc is needed for *lifetime management* (PoolSlice outlives allocation call), not thread-safety. Now uses Relaxed ordering instead of Acquire/Release since no cross-thread synchronization is needed.

### Cache Locality is Critical

Benchmark results (with page pre-touching) show **pool size affects performance**:

| Pool Size | Performance | Slowdown | Notes |
|-----------|-------------|----------|-------|
| 32 KB - 512 KB | ~19-20 ns/op | 1.0x | Fits in L1/L2 cache - optimal |
| 2 MB - 8 MB | ~20 ns/op | 1.05x | Fits in L3 cache - still fast |
| 32 MB - 128 MB | ~30-32 ns/op | 1.6x | Partial cache thrashing |
| 512 MB | ~31 ns/op | 1.6x | DRAM latency, but cached working set |
| 1 GB | ~48 ns/op | **2.5x slower** | DRAM latency dominates |

**Note:** Without page pre-touching, large pools show 4-6x worse performance due to page fault latency.
Production systems should warm up pools on startup or use huge pages to minimize TLB overhead.

**Production config:** 2 GB pool per IO thread (DISRUPTOR_SIZE × MAX_VECTORS × FEATURE_DIM)
- Sized for worst-case: all in-flight requests at max size
- Expected performance: ~50-60 ns/op (2.5-3x slower than cache-fit pools)
- Real workloads are mostly 1-8 vectors, not 64
- **Opportunity:** Right-sizing to 8-32 MB based on typical workload would provide **2.5x speedup**
- **Important:** Warm up pools on startup (touch all pages) to avoid page fault latency

### Operation Cost Breakdown

Use `perf` to profile the buffer pool operations:

```bash
# Build and profile (runs 100M iterations, ~7-10 seconds)
cargo bench --no-run
perf record -g target/release/deps/profile_buffer_pool-*

# Or specify custom iteration count
perf record -g target/release/deps/profile_buffer_pool-* 1000000000  # 1B iterations

# View results
perf report
```

**Arc operations (clone + drop) likely dominate** the cost. This is acceptable given RAII benefits and the safety guarantees provided by automatic cleanup.

### Allocation Size Effects

Larger allocations perform better:
- 1-4 vectors: 365-1606 ns/op (slower)
- 8+ vectors: 47-137 ns/op (much faster)

Likely due to:
1. Amortization of Arc overhead over more data
2. Better memory access patterns for larger blocks
3. Benchmark ring size interactions

## Future Optimization Opportunities

1. **Right-size pools:** Use typical workload (1-8 vectors) instead of max (64) for capacity calculation
2. ~~**Relaxed atomics:** Change Acquire/Release to Relaxed since all access is single-threaded~~ **✓ Done**
3. **Pool warmup:** Pre-touch pages to avoid page faults during operation
4. **NUMA awareness:** Allocate pools on same NUMA node as IO thread

## Benchmark Commands

```bash
# Test different allocation sizes
cargo bench --bench buffer_pool_bench -- --alloc-sizes

# Test different pool sizes (shows cache effects)
cargo bench --bench buffer_pool_bench -- --pool-sizes

# Profile with perf (requires Linux perf tools, runs 100M iterations ~7-10s)
cargo bench --no-run
perf record -g target/release/deps/profile_buffer_pool-*
perf report

# Or use custom iteration count (1B iterations ~70-100s)
perf record -g target/release/deps/profile_buffer_pool-* 1000000000
```

## Related Files

- [buffer_pool.rs](src/buffer_pool.rs) - Implementation
- [benches/buffer_pool_bench.rs](benches/buffer_pool_bench.rs) - Allocation/pool size benchmarks
- [benches/profile_buffer_pool.rs](benches/profile_buffer_pool.rs) - Clean benchmark for perf profiling
