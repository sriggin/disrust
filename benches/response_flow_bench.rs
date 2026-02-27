//! Benchmark: response path (guard_to_wire_per_conn / guard_to_iovecs_per_conn) without io_uring.

use std::collections::HashMap;
use std::hint::black_box;
use std::os::unix::io::RawFd;

use disrust::response_flow;
use disrust::ring_types::InferenceResponse;

fn create_eventfd() -> RawFd {
    unsafe { libc::eventfd(0, libc::EFD_NONBLOCK) }
}

fn main() {
    const CAPACITY: usize = 8192;
    const RESPONSES_PER_BATCH: usize = 64;
    const TARGET_DURATION: std::time::Duration = std::time::Duration::from_secs(2);

    let efd = create_eventfd();
    assert!(efd >= 0);

    let (mut producer, mut poller) = disrust::response_queue::build_response_channel(CAPACITY, efd);

    // Warm up
    for _ in 0..1000 {
        for i in 0..RESPONSES_PER_BATCH {
            let resp =
                InferenceResponse::with_results((i % 4) as u16, 0, &[1.0f32; 8], None).unwrap();
            producer.send(resp);
        }
        producer.signal();
        if let Ok(mut guard) = poller.poll() {
            let _ = response_flow::guard_to_wire_per_conn(&mut guard);
        }
    }

    // Benchmark wire build: each iteration send batch, poll, build wire
    let start_wire = std::time::Instant::now();
    let mut iterations_wire: u64 = 0;
    while start_wire.elapsed() < TARGET_DURATION {
        for i in 0..RESPONSES_PER_BATCH {
            let resp =
                InferenceResponse::with_results((i % 4) as u16, 0, &[1.0f32; 8], None).unwrap();
            producer.send(resp);
        }
        producer.signal();
        if let Ok(mut guard) = poller.poll() {
            let wire = response_flow::guard_to_wire_per_conn(&mut guard);
            black_box(wire);
        }
        iterations_wire += 1;
    }
    let elapsed_wire = start_wire.elapsed();
    eprintln!(
        "guard_to_wire_per_conn: {} batches ({} resp/batch) in {:?} (sustained)",
        iterations_wire, RESPONSES_PER_BATCH, elapsed_wire
    );
    eprintln!(
        "  {:.0} batches/s  {:.0} resp/s (over {:.1}s)",
        iterations_wire as f64 / elapsed_wire.as_secs_f64(),
        (iterations_wire * RESPONSES_PER_BATCH as u64) as f64 / elapsed_wire.as_secs_f64(),
        elapsed_wire.as_secs_f64()
    );

    // Benchmark iovec build
    let mut out = HashMap::new();
    let start_iovec = std::time::Instant::now();
    let mut iterations_iovec: u64 = 0;
    while start_iovec.elapsed() < TARGET_DURATION {
        for i in 0..RESPONSES_PER_BATCH {
            let resp =
                InferenceResponse::with_results((i % 4) as u16, 0, &[1.0f32; 8], None).unwrap();
            producer.send(resp);
        }
        producer.signal();
        if let Ok(mut guard) = poller.poll() {
            response_flow::guard_to_iovecs_per_conn(&mut guard, &mut out);
            black_box(&out);
        }
        iterations_iovec += 1;
    }
    let elapsed_iovec = start_iovec.elapsed();
    eprintln!(
        "guard_to_iovecs_per_conn: {} batches in {:?} (sustained)",
        iterations_iovec, elapsed_iovec
    );
    eprintln!(
        "  {:.0} batches/s (over {:.1}s)",
        iterations_iovec as f64 / elapsed_iovec.as_secs_f64(),
        elapsed_iovec.as_secs_f64()
    );

    unsafe {
        libc::close(efd);
    }
}
