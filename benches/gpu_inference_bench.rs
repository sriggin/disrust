#[cfg(feature = "cuda")]
mod common;

#[cfg(feature = "cuda")]
use std::hint::black_box;
#[cfg(feature = "cuda")]
use std::time::{Duration, Instant};
#[cfg(feature = "cuda")]
use std::{ffi::c_void, ptr::NonNull};

#[cfg(feature = "cuda")]
use clap::{Parser, ValueEnum};
#[cfg(feature = "cuda")]
use ort::init_from;

#[cfg(feature = "cuda")]
use disrust::buffer_pool::{BufferPool, PoolSlice, set_factory_pool};
#[cfg(feature = "cuda")]
use disrust::constants::FEATURE_DIM;
#[cfg(feature = "cuda")]
use disrust::gpu::preflight::{verify_cuda_startup, verify_ort_dylib_present};
#[cfg(feature = "cuda")]
use disrust::gpu::session::GpuSession;

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Mode {
    Direct,
    Subsystem,
}

#[cfg(feature = "cuda")]
#[derive(Parser, Debug)]
#[command(about = "GPU inference subsystem benchmark")]
struct Args {
    /// Path to ONNX model file.
    #[arg(short, long, default_value = "tests/models/ort_sum_model.onnx")]
    model: String,

    /// Benchmark mode:
    /// direct = raw GpuSession RunAsync throughput by total batch vectors
    /// subsystem = synthetic pooled request slots + completion-style wait
    #[arg(long, value_enum, default_value_t = Mode::Direct)]
    mode: Mode,

    /// Comma-separated batch vector counts for direct mode.
    #[arg(long, default_value = "1,2,4,8,16,32,64,128")]
    batch_vectors: String,

    /// Comma-separated ring-slot counts for subsystem mode.
    #[arg(long, default_value = "1,2,4,8,16,32,64,128,256,512,1024")]
    slot_counts: String,

    /// Vectors per synthetic request slot in subsystem mode.
    #[arg(long, default_value_t = 1)]
    vectors_per_slot: usize,

    /// Warmup iterations before measurement.
    #[arg(long, default_value_t = 200)]
    warmup_iters: usize,

    /// Measurement iterations per size.
    #[arg(long, default_value_t = 2000)]
    iters: usize,
}

#[cfg(feature = "cuda")]
fn parse_csv_usize(s: &str) -> Vec<usize> {
    s.split(',')
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .unwrap_or_else(|e| panic!("invalid usize '{part}': {e}"))
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn main() {
    let filtered_args = std::env::args().filter(|arg| arg != "--bench");
    let args = Args::parse_from(filtered_args);
    if args.vectors_per_slot == 0 {
        panic!("--vectors-per-slot must be > 0");
    }
    let ort_dylib = verify_ort_dylib_present().expect("ORT dylib preflight failed");
    init_from(&ort_dylib)
        .expect("ort::init_from failed")
        .commit();
    verify_cuda_startup().expect("CUDA preflight failed");
    let _ = set_factory_pool(BufferPool::new_boxed(1));

    let model_bytes = std::fs::read(&args.model)
        .unwrap_or_else(|e| panic!("failed to read model '{}': {e}", args.model));

    eprintln!(
        "gpu inference bench: mode={:?} model={} warmup_iters={} iters={}",
        args.mode, args.model, args.warmup_iters, args.iters
    );
    eprintln!("size,batches_per_s,vectors_per_s,avg_us");

    match args.mode {
        Mode::Direct => {
            let batch_vectors = parse_csv_usize(&args.batch_vectors);
            for num_vectors in batch_vectors {
                let stats = bench_direct(&model_bytes, num_vectors, args.warmup_iters, args.iters);
                println!(
                    "{},{:.0},{:.0},{:.1}",
                    num_vectors, stats.batches_per_s, stats.vectors_per_s, stats.avg_us
                );
            }
        }
        Mode::Subsystem => {
            let slot_counts = parse_csv_usize(&args.slot_counts);
            for slot_count in slot_counts {
                let stats = bench_subsystem(
                    &model_bytes,
                    slot_count,
                    args.vectors_per_slot,
                    args.warmup_iters,
                    args.iters,
                );
                println!(
                    "{}x{},{:.0},{:.0},{:.1}",
                    slot_count,
                    args.vectors_per_slot,
                    stats.batches_per_s,
                    stats.vectors_per_s,
                    stats.avg_us
                );
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("gpu_inference_bench requires --features cuda");
}

#[cfg(feature = "cuda")]
struct Stats {
    batches_per_s: f64,
    vectors_per_s: f64,
    avg_us: f64,
}

#[cfg(feature = "cuda")]
struct PinnedHostBuffer {
    ptr: NonNull<f32>,
    len: usize,
}

#[cfg(feature = "cuda")]
impl PinnedHostBuffer {
    fn new(len: usize) -> Self {
        unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let status =
                cudarc::driver::sys::cuMemAllocHost_v2(&mut ptr, len * std::mem::size_of::<f32>());
            if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                panic!("cuMemAllocHost_v2 failed for benchmark input: {status:?}");
            }
            let ptr = NonNull::new(ptr as *mut f32).expect("cuMemAllocHost_v2 returned null");
            Self { ptr, len }
        }
    }

    fn as_ptr(&self) -> *const f32 {
        self.ptr.as_ptr()
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

#[cfg(feature = "cuda")]
impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        unsafe {
            let status = cudarc::driver::sys::cuMemFreeHost(self.ptr.as_ptr() as *mut c_void);
            if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                eprintln!("gpu_inference_bench: cuMemFreeHost failed: {status:?}");
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn bench_direct(
    model_bytes: &[u8],
    num_vectors: usize,
    warmup_iters: usize,
    iters: usize,
) -> Stats {
    let mut session = GpuSession::with_output_capacity(model_bytes, num_vectors);
    let pinned = allocate_pinned_input(num_vectors);
    let host_ptr = pinned.as_ptr();

    for _ in 0..warmup_iters {
        run_batch(&mut session, host_ptr, num_vectors, Vec::new());
    }

    let start = Instant::now();
    for _ in 0..iters {
        run_batch(&mut session, black_box(host_ptr), num_vectors, Vec::new());
    }
    let elapsed = start.elapsed();
    stats_from_elapsed(elapsed, iters, num_vectors)
}

#[cfg(feature = "cuda")]
fn bench_subsystem(
    model_bytes: &[u8],
    slot_count: usize,
    vectors_per_slot: usize,
    warmup_iters: usize,
    iters: usize,
) -> Stats {
    let total_vectors = slot_count * vectors_per_slot;
    let pool_capacity = total_vectors * FEATURE_DIM * 4;
    let mut session = GpuSession::with_output_capacity(model_bytes, total_vectors);
    let mut pinned_pool = PinnedHostBuffer::new(pool_capacity);
    let pool = Box::leak(unsafe {
        BufferPool::from_raw_ptr(pinned_pool.as_mut_slice().as_mut_ptr(), pool_capacity)
    });
    let mut alloc = pool.allocator();

    for _ in 0..warmup_iters {
        let slices = make_contiguous_slices(&mut alloc, slot_count, vectors_per_slot);
        let host_ptr = slices[0].as_slice().as_ptr();
        run_batch(&mut session, host_ptr, total_vectors, slices);
    }

    let start = Instant::now();
    for _ in 0..iters {
        let slices = make_contiguous_slices(&mut alloc, slot_count, vectors_per_slot);
        let host_ptr = slices[0].as_slice().as_ptr();
        run_batch(&mut session, black_box(host_ptr), total_vectors, slices);
    }
    let elapsed = start.elapsed();
    drop(pinned_pool);
    stats_from_elapsed(elapsed, iters, total_vectors)
}

#[cfg(feature = "cuda")]
fn run_batch(
    session: &mut GpuSession,
    host_ptr: *const f32,
    num_vectors: usize,
    input_slices: Vec<PoolSlice>,
) {
    assert!(
        session.try_acquire(),
        "GpuSession unexpectedly unavailable in benchmark"
    );
    let mut batch = session.submit_batch(host_ptr, num_vectors);
    batch.input_slices = input_slices;
    batch.completion.wait();
    black_box(unsafe { std::slice::from_raw_parts(batch.output_ptr, batch.output_len) });
    batch
        .session_available
        .store(true, std::sync::atomic::Ordering::Release);
    drop(batch);
}

#[cfg(feature = "cuda")]
fn stats_from_elapsed(elapsed: Duration, iters: usize, vectors_per_batch: usize) -> Stats {
    let seconds = elapsed.as_secs_f64();
    Stats {
        batches_per_s: iters as f64 / seconds,
        vectors_per_s: (iters * vectors_per_batch) as f64 / seconds,
        avg_us: elapsed.as_secs_f64() * 1_000_000.0 / iters as f64,
    }
}

#[cfg(feature = "cuda")]
fn allocate_pinned_input(num_vectors: usize) -> PinnedHostBuffer {
    let mut data = PinnedHostBuffer::new(num_vectors * FEATURE_DIM);
    fill_features(data.as_mut_slice());
    data
}

#[cfg(feature = "cuda")]
fn make_contiguous_slices(
    alloc: &mut disrust::buffer_pool::PoolAllocator,
    slot_count: usize,
    vectors_per_slot: usize,
) -> Vec<PoolSlice> {
    let mut slices = Vec::with_capacity(slot_count);
    let elems_per_slot = vectors_per_slot * FEATURE_DIM;
    for slot_idx in 0..slot_count {
        let mut slice = alloc.alloc(elems_per_slot).expect("pool alloc failed");
        fill_slot_features(slice.as_mut_slice(), slot_idx);
        slices.push(slice.freeze());
    }
    debug_assert!(slices.windows(2).all(|w| w[0].is_contiguous(&w[1])));
    slices
}

#[cfg(feature = "cuda")]
fn fill_features(data: &mut [f32]) {
    for (i, val) in data.iter_mut().enumerate() {
        *val = i as f32 * 0.01;
    }
}

#[cfg(feature = "cuda")]
fn fill_slot_features(data: &mut [f32], slot_idx: usize) {
    for (i, val) in data.iter_mut().enumerate() {
        *val = (slot_idx * FEATURE_DIM + i) as f32 * 0.01;
    }
}
