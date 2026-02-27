//! End-to-end benchmark: request_flow -> batch processor -> response_flow (no io_uring).

use std::hint::black_box;
use std::os::unix::io::RawFd;

use disruptor::{BusySpin, build_single_producer};

use disrust::batch_processor::BatchProcessor;
use disrust::buffer_pool::{BufferPool, set_factory_pool};
use disrust::constants::{FEATURE_DIM, MAX_VECTORS_PER_REQUEST};
use disrust::request_flow;
use disrust::response_flow;
use disrust::ring_types::InferenceEvent;

fn init_factory_pool() {
    let _ = set_factory_pool(BufferPool::new_boxed(1));
}

fn create_eventfd() -> RawFd {
    unsafe { libc::eventfd(0, libc::EFD_NONBLOCK) }
}

fn one_request_bytes(num_vectors: u32) -> Vec<u8> {
    let mut buf = num_vectors.to_le_bytes().to_vec();
    buf.resize(4 + num_vectors as usize * FEATURE_DIM * 4, 0u8);
    buf
}

fn main() {
    init_factory_pool();

    const RING_SIZE: usize = 65536;
    const RESPONSE_QUEUE_SIZE: usize = 65536;
    const RESULT_POOL_CAPACITY: usize = RESPONSE_QUEUE_SIZE * 16;
    const REQUESTS_PER_BATCH: usize = 8;

    let builder = build_single_producer(RING_SIZE, InferenceEvent::factory, BusySpin);
    let (request_poller, builder) = builder.event_poller();
    let mut request_producer = builder.build();

    let request_pool = BufferPool::leak_new(RING_SIZE * MAX_VECTORS_PER_REQUEST * FEATURE_DIM);
    let efd = create_eventfd();
    assert!(efd >= 0);
    let (resp_producer, mut response_poller) =
        disrust::response_queue::build_response_channel(RESPONSE_QUEUE_SIZE, efd);
    let result_pool = BufferPool::leak_new(RESULT_POOL_CAPACITY);

    let mut batch = BatchProcessor {
        poller: request_poller,
        response_producers: vec![resp_producer],
        result_pools: vec![result_pool],
    };

    let buf = one_request_bytes(REQUESTS_PER_BATCH as u32);
    let mut full_buf = buf.clone();
    for _ in 1..REQUESTS_PER_BATCH {
        full_buf.extend_from_slice(&buf);
    }

    let conn_id = 0u16;
    let thread_id = 0u8;
    let mut request_seq = 0u64;

    for _ in 0..5_000 {
        let _ = request_flow::process_requests_from_buffer(
            &full_buf,
            &mut request_producer,
            request_pool,
            conn_id,
            thread_id,
            &mut request_seq,
        );
        let _ = batch.process_one_poll_cycle();
        if let Ok(mut guard) = response_poller.poll() {
            let _ = response_flow::guard_to_wire_per_conn(&mut guard);
        }
    }

    request_seq = 0;
    let start = std::time::Instant::now();
    const TARGET_DURATION: std::time::Duration = std::time::Duration::from_secs(2);
    let mut iterations: u64 = 0;

    while start.elapsed() < TARGET_DURATION {
        let _ = request_flow::process_requests_from_buffer(
            black_box(&full_buf),
            &mut request_producer,
            request_pool,
            conn_id,
            thread_id,
            &mut request_seq,
        );
        let _ = batch.process_one_poll_cycle();
        if let Ok(mut guard) = response_poller.poll() {
            let wire = response_flow::guard_to_wire_per_conn(&mut guard);
            black_box(wire);
        }
        iterations += 1;
    }

    let elapsed = start.elapsed();
    let total_requests = iterations * REQUESTS_PER_BATCH as u64;
    let total_bytes = iterations * full_buf.len() as u64;

    eprintln!("Pipeline (request_flow -> batch -> response_flow), no io_uring:");
    eprintln!("  {} requests in {:?} (sustained)", total_requests, elapsed);
    eprintln!(
        "  {:.0} req/s  {:.1} MB/s (over {:.1}s)",
        total_requests as f64 / elapsed.as_secs_f64(),
        (total_bytes as f64 / 1_000_000.0) / elapsed.as_secs_f64(),
        elapsed.as_secs_f64()
    );

    unsafe {
        libc::close(efd);
    }
}
