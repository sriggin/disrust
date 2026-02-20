# Buffer Pool Performance Notes

## Key Design Decisions

### Single-Threaded Pool Access
Despite using `Arc<PoolSliceInner>` and atomics, **request pool access is actually single-threaded**:
- IO thread allocates from pool
- IO thread publishes to disruptor (moves PoolSlice into event)
- When disruptor wraps, IO thread publishes to same slot → **drops old PoolSlice on IO thread**
- `PoolSliceInner::drop()` advances read_cursor

**Result pools** (for responses >INLINE_RESULT_CAPACITY vectors) are cross-threaded:
- Batch processor allocates from `result_pools[io_thread_id]`
- IO thread drops InferenceResponse → PoolSlice → read_cursor advanced on IO thread
- Uses Relaxed ordering (producer and consumer don't share data, only the cursor)

### Pages Are Pre-Touched on Construction

`BufferPool::new_boxed()` touches every page at creation time, paying the page fault cost
upfront rather than during operation. **Pool creation should happen on the thread that will
use the pool** for correct NUMA placement.

### Cache Locality is Critical

Benchmark results show **pool size affects performance**:

| Pool Size | Performance | Slowdown | Notes |
|-----------|-------------|----------|-------|
| 16 KB - 512 KB | ~11-14 ns/op | 1.0x | Fits in L1/L2 cache - optimal |
| 1 MB - 8 MB | ~13-16 ns/op | 1.2x | Fits in L3 cache - still fast |
| 16 MB - 128 MB | ~23-30 ns/op | 2.2x | Cache thrashing / DRAM |
| 256 MB - 1 GB | ~29-32 ns/op | 2.6x | DRAM latency dominates |

**Production config:** 2 GB pool per IO thread (DISRUPTOR_SIZE × MAX_VECTORS × FEATURE_DIM)
- Sized for worst-case: all in-flight requests at max size
- Expected performance: ~30-32 ns/op (DRAM latency dominates at this size)
- Real workloads are mostly 1-8 vectors, not 64
- **Opportunity:** Right-sizing to 8-32 MB based on typical workload would provide **~2x speedup**

### Allocation Size Effects

With a 2 GB pool (DRAM-latency dominated), allocation size has minimal impact:

| Allocation | Performance |
|------------|-------------|
| 1 vector (128 f32) | ~38 ns/op |
| 2-16 vectors | ~33-35 ns/op |
| 32-64 vectors | ~30-32 ns/op |

Numbers are flat because DRAM latency dominates at this pool size. With a cache-fit pool
(8-32 MB), allocation size effects would be negligible (~11-16 ns/op across all sizes).

## Future Optimization Opportunities

1. **Right-size pools:** Use typical workload (1-8 vectors) instead of max (64) for capacity calculation
2. ~~**Relaxed atomics:** Change Acquire/Release to Relaxed since all access is single-threaded~~ **✓ Done**
3. ~~**Pool warmup:** Pre-touch pages to avoid page faults during operation~~ **✓ Done (in constructor)**
4. **NUMA awareness:** Create pools on the thread that will use them (currently created on main thread)

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
