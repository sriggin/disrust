//! Benchmark: request path (process_requests_from_buffer) without io_uring.

use std::hint::black_box;

use disruptor::{BusySpin, build_single_producer};

use disrust::buffer_pool::{BufferPool, set_factory_pool};
use disrust::constants::{FEATURE_DIM, MAX_VECTORS_PER_REQUEST};
use disrust::request_flow;
use disrust::ring_types::InferenceEvent;

fn init_factory_pool() {
    let _ = set_factory_pool(BufferPool::new_boxed(1));
}

fn one_request_bytes(num_vectors: u32) -> Vec<u8> {
    let mut buf = num_vectors.to_le_bytes().to_vec();
    buf.resize(4 + num_vectors as usize * FEATURE_DIM * 4, 0u8);
    buf
}

fn main() {
    init_factory_pool();

    const RING_SIZE: usize = 65536;
    const REQUESTS_PER_BATCH: usize = 8;

    let builder = build_single_producer(RING_SIZE, InferenceEvent::factory, BusySpin);
    let (mut poller, builder) = builder.event_poller();
    let mut producer = builder.build();

    let pool_capacity = RING_SIZE * MAX_VECTORS_PER_REQUEST * FEATURE_DIM;
    let pool = BufferPool::leak_new(pool_capacity);

    let buf = one_request_bytes(REQUESTS_PER_BATCH as u32);
    let mut full_buf = buf.clone();
    for _ in 1..REQUESTS_PER_BATCH {
        full_buf.extend_from_slice(&buf);
    }

    let conn_id = 0u16;
    let thread_id = 0u8;
    let mut request_seq = 0u64;

    // Warm up
    for _ in 0..10_000 {
        let _ = request_flow::process_requests_from_buffer(
            &full_buf,
            &mut producer,
            pool,
            conn_id,
            thread_id,
            &mut request_seq,
        );
        while let Ok(mut guard) = poller.poll() {
            for _ in &mut guard {}
        }
    }

    request_seq = 0;
    let start = std::time::Instant::now();
    const TARGET_DURATION: std::time::Duration = std::time::Duration::from_secs(2);
    let mut iterations: u64 = 0;

    while start.elapsed() < TARGET_DURATION {
        let result = request_flow::process_requests_from_buffer(
            black_box(&full_buf),
            &mut producer,
            pool,
            conn_id,
            thread_id,
            &mut request_seq,
        );
        let _ = black_box(result);
        while let Ok(mut guard) = poller.poll() {
            for _ in &mut guard {}
        }
        iterations += 1;
    }

    let elapsed = start.elapsed();
    let total_requests = iterations * REQUESTS_PER_BATCH as u64;
    let total_bytes = iterations * full_buf.len() as u64;
    eprintln!(
        "request_flow: {} requests in {:?} (sustained)",
        total_requests, elapsed
    );
    eprintln!(
        "  {:.0} req/s  {:.0} MB/s (over {:.1}s)",
        total_requests as f64 / elapsed.as_secs_f64(),
        (total_bytes as f64 / 1_000_000.0) / elapsed.as_secs_f64(),
        elapsed.as_secs_f64()
    );
}
