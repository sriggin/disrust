//! Two-thread inference pipeline benchmark.
//!
//! Measures end-to-end round-trip latency across the submission→BatchQueue→completion
//! thread boundary. Unlike gpu_inference_bench, RunAsync and completion.wait() happen
//! on separate threads, exposing SPSC queue latency and cross-core observation cost.
//!
//! Runs with or without --features cuda. CUDA mode uses pinned host memory; CPU mode
//! uses heap allocation. InferenceSession and make_pool select the right backing at
//! compile time.

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use clap::Parser;
use ort::init_from;

use disrust::pipeline::batch_queue::{BatchEntry, BatchQueue};
use disrust::pipeline::session::{BatchPoll, InferenceSession};
use disrust::pipeline::{make_pool, verify_ort_dylib_present};

#[cfg(feature = "cuda")]
use disrust::cuda::preflight::verify_cuda_startup;

#[derive(Parser, Debug)]
#[command(about = "Two-thread inference pipeline benchmark")]
struct Args {
    /// Path to ONNX model file.
    #[arg(short, long, default_value = "tests/models/ort_sum_model.onnx")]
    model: String,

    /// Comma-separated total vector counts per batch.
    #[arg(long, default_value = "1,4,16,64,128,256")]
    batch_vectors: String,

    /// Warmup iterations before measurement.
    #[arg(long, default_value_t = 200)]
    warmup_iters: usize,

    /// Measurement iterations per size.
    #[arg(long, default_value_t = 2000)]
    iters: usize,
}

fn main() {
    let filtered_args = std::env::args().filter(|arg| arg != "--bench");
    let args = Args::parse_from(filtered_args);

    let ort_dylib = verify_ort_dylib_present().expect("ORT dylib preflight failed");
    init_from(&ort_dylib)
        .expect("ort::init_from failed")
        .commit();

    #[cfg(feature = "cuda")]
    verify_cuda_startup().expect("CUDA preflight failed");

    let model_bytes = std::fs::read(&args.model)
        .unwrap_or_else(|e| panic!("failed to read model '{}': {e}", args.model));

    let batch_vectors: Vec<usize> = args
        .batch_vectors
        .split(',')
        .filter(|p| !p.is_empty())
        .map(|p| {
            p.trim()
                .parse::<usize>()
                .unwrap_or_else(|e| panic!("invalid usize '{p}': {e}"))
        })
        .collect();

    #[cfg(feature = "cuda")]
    let mode = "cuda";
    #[cfg(not(feature = "cuda"))]
    let mode = "cpu";

    eprintln!(
        "pipeline bench: mode={mode} model={} warmup_iters={} iters={}",
        args.model, args.warmup_iters, args.iters
    );
    eprintln!("num_vectors,batches_per_s,vectors_per_s,avg_us");

    // Pool is allocated once and shared across all bench sizes, matching server behaviour.
    let pool = make_pool();
    let mut alloc = pool.allocator();

    for num_vectors in batch_vectors {
        let stats = bench_pipeline(
            &model_bytes,
            num_vectors,
            &mut alloc,
            args.warmup_iters,
            args.iters,
        );
        println!(
            "{},{:.0},{:.0},{:.1}",
            num_vectors, stats.batches_per_s, stats.vectors_per_s, stats.avg_us
        );
    }
}

struct Stats {
    batches_per_s: f64,
    vectors_per_s: f64,
    avg_us: f64,
}

fn stats_from_elapsed(elapsed: Duration, iters: usize, vectors_per_batch: usize) -> Stats {
    let seconds = elapsed.as_secs_f64();
    Stats {
        batches_per_s: iters as f64 / seconds,
        vectors_per_s: (iters * vectors_per_batch) as f64 / seconds,
        avg_us: elapsed.as_secs_f64() * 1_000_000.0 / iters as f64,
    }
}

fn bench_pipeline(
    model_bytes: &[u8],
    num_vectors: usize,
    alloc: &mut disrust::buffer_pool::PoolAllocator,
    warmup_iters: usize,
    iters: usize,
) -> Stats {
    use disrust::constants::FEATURE_DIM;

    let mut session = InferenceSession::with_output_capacity(model_bytes, num_vectors);
    // capacity 2: one in flight + one slack to avoid push/pop deadlock at boundary
    let batch_queue = Arc::new(BatchQueue::new(2));
    let done = Arc::new(AtomicBool::new(false));

    let comp_handle = {
        let q = Arc::clone(&batch_queue);
        let d = Arc::clone(&done);
        std::thread::Builder::new()
            .name("bench-completion".into())
            .spawn(move || completion_thread(q, d))
            .expect("failed to spawn completion thread")
    };

    // Allocate input from the pool — same memory path as the real server's ingress thread.
    let input = alloc
        .alloc(num_vectors * FEATURE_DIM)
        .expect("pool alloc failed for bench input");
    let input = input.freeze();
    let host_ptr = input.as_slice().as_ptr();

    drive(
        &mut session,
        host_ptr,
        num_vectors,
        &batch_queue,
        warmup_iters,
    );

    let start = Instant::now();
    drive(&mut session, host_ptr, num_vectors, &batch_queue, iters);
    let elapsed = start.elapsed();

    done.store(true, Ordering::Release);
    comp_handle.join().expect("completion thread panicked");

    drop(input);
    stats_from_elapsed(elapsed, iters, num_vectors)
}

/// Submission side: acquire session, submit batch, push to queue. Repeats `iters` times,
/// then waits for the last batch to complete before returning.
fn drive(
    session: &mut InferenceSession,
    host_ptr: *const f32,
    num_vectors: usize,
    batch_queue: &BatchQueue,
    iters: usize,
) {
    for _ in 0..iters {
        while !session.try_acquire() {
            std::hint::spin_loop();
        }
        let batch = session.submit_batch(black_box(host_ptr), num_vectors);
        batch_queue.push(BatchEntry {
            slot_count: 1,
            #[cfg(feature = "metrics")]
            submitted_at: Instant::now(),
            batch,
        });
    }
    // wait for the last batch to be released by the completion thread
    while !session.is_available() {
        std::hint::spin_loop();
    }
}

/// Completion side: pop batches, spin-poll until ready, black-box the output, release session.
fn completion_thread(batch_queue: Arc<BatchQueue>, done: Arc<AtomicBool>) {
    loop {
        if let Some(entry) = batch_queue.pop() {
            loop {
                match entry.batch.completion.poll() {
                    BatchPoll::Pending => std::hint::spin_loop(),
                    BatchPoll::Ready => break,
                    BatchPoll::Failed => entry.batch.completion.wait(), // aborts internally
                }
            }
            black_box(unsafe {
                std::slice::from_raw_parts(entry.batch.output_ptr, entry.batch.output_len)
            });
            entry.batch.session_available.store(true, Ordering::Release);
            drop(entry);
        } else if done.load(Ordering::Acquire) {
            return;
        } else {
            std::hint::spin_loop();
        }
    }
}
