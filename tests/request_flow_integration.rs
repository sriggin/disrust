//! Integration test: request path (bytes → parse → alloc → publish) without io_uring.
//! Also includes full-pipeline test: request_flow → batch processor → response_flow.

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

/// Build a byte buffer for one request: [u32 num_vectors][f32 * num_vectors * FEATURE_DIM].
fn one_request_bytes(num_vectors: u32, feature_values: &[f32]) -> Vec<u8> {
    assert!(num_vectors as usize * FEATURE_DIM <= feature_values.len());
    let mut buf = num_vectors.to_le_bytes().to_vec();
    for val in feature_values
        .iter()
        .take(num_vectors as usize * FEATURE_DIM)
    {
        buf.extend_from_slice(&val.to_le_bytes());
    }
    buf
}

#[test]
fn request_flow_processes_one_request_and_consumer_sees_event() {
    init_factory_pool();

    const RING_SIZE: usize = 256;
    let builder = build_single_producer(RING_SIZE, InferenceEvent::factory, BusySpin);
    let (mut poller, builder) = builder.event_poller();
    let mut producer = builder.build();

    let pool_capacity = RING_SIZE * MAX_VECTORS_PER_REQUEST * FEATURE_DIM;
    let pool = BufferPool::leak_new(pool_capacity);

    let conn_id = 1u16;
    let thread_id = 0u8;
    let mut request_seq = 0u64;

    // One request: 2 vectors
    let num_vectors = 2u32;
    let features: Vec<f32> = (0..num_vectors as usize * FEATURE_DIM)
        .map(|i| i as f32 * 0.5)
        .collect();
    let buf = one_request_bytes(num_vectors, &features);

    let result = request_flow::process_requests_from_buffer(
        &buf,
        &mut producer,
        pool,
        conn_id,
        thread_id,
        &mut request_seq,
    );

    assert!(result.is_ok());
    let (consumed, num_published) = result.unwrap();
    assert_eq!(consumed, buf.len(), "consumed should match request length");
    assert_eq!(num_published, 1);

    // Consumer polls and sees one event
    match poller.poll() {
        Ok(mut guard) => {
            let events: Vec<_> = (&mut guard).collect();
            assert_eq!(events.len(), 1);
            let ev = &events[0];
            assert_eq!(ev.conn_id, conn_id);
            assert_eq!(ev.io_thread_id, thread_id);
            assert_eq!(ev.num_vectors, num_vectors as u8);
            assert_eq!(ev.request_seq, 0);
            for (v, expected_chunk) in features.chunks(FEATURE_DIM).enumerate() {
                let vec_slice = ev.vector(v);
                assert_eq!(vec_slice, expected_chunk);
            }
        }
        Err(_) => panic!("expected one event"),
    }
}

#[test]
fn request_flow_processes_multiple_requests_in_one_buffer() {
    init_factory_pool();

    const RING_SIZE: usize = 256;
    let builder = build_single_producer(RING_SIZE, InferenceEvent::factory, BusySpin);
    let (mut poller, builder) = builder.event_poller();
    let mut producer = builder.build();

    let pool_capacity = RING_SIZE * MAX_VECTORS_PER_REQUEST * FEATURE_DIM;
    let pool = BufferPool::leak_new(pool_capacity);

    let conn_id = 2u16;
    let thread_id = 0u8;
    let mut request_seq = 0u64;

    let r1 = one_request_bytes(1, &[1.0f32; FEATURE_DIM]);
    let r2 = one_request_bytes(1, &[2.0f32; FEATURE_DIM]);
    let mut buf = r1.clone();
    buf.extend_from_slice(&r2);
    let expected_consumed = buf.len();

    let result = request_flow::process_requests_from_buffer(
        &buf,
        &mut producer,
        pool,
        conn_id,
        thread_id,
        &mut request_seq,
    );

    assert!(result.is_ok());
    let (consumed, num_published) = result.unwrap();
    assert_eq!(consumed, expected_consumed);
    assert_eq!(num_published, 2);

    let mut seen = 0u64;
    while let Ok(mut guard) = poller.poll() {
        for ev in &mut guard {
            assert_eq!(ev.conn_id, conn_id);
            assert_eq!(ev.request_seq, seen);
            assert_eq!(ev.num_vectors, 1);
            let v = ev.vector(0);
            if seen == 0 {
                assert!(v.iter().all(|&x| x == 1.0));
            } else {
                assert!(v.iter().all(|&x| x == 2.0));
            }
            seen += 1;
        }
        if seen >= 2 {
            break;
        }
    }
    assert_eq!(seen, 2);
}

#[test]
fn request_flow_incomplete_returns_consumed_only() {
    init_factory_pool();

    let builder = build_single_producer(256, InferenceEvent::factory, BusySpin);
    let (_poller, builder) = builder.event_poller();
    let mut producer = builder.build();

    let pool = BufferPool::leak_new(256 * FEATURE_DIM);

    let mut request_seq = 0u64;
    // Only 4 bytes (num_vectors = 1) – no payload, so incomplete
    let buf = [1u8, 0, 0, 0];

    let result = request_flow::process_requests_from_buffer(
        &buf,
        &mut producer,
        pool,
        0,
        0,
        &mut request_seq,
    );

    assert!(result.is_ok());
    let (consumed, num_published) = result.unwrap();
    assert_eq!(consumed, 0);
    assert_eq!(num_published, 0);
}

#[test]
fn request_flow_parse_error_returns_err() {
    init_factory_pool();

    let builder = build_single_producer(256, InferenceEvent::factory, BusySpin);
    let (_poller, builder) = builder.event_poller();
    let mut producer = builder.build();

    let pool = BufferPool::leak_new(256 * FEATURE_DIM);

    let mut request_seq = 0u64;
    // num_vectors = 0 is invalid
    let buf = [0u8, 0, 0, 0];

    let result = request_flow::process_requests_from_buffer(
        &buf,
        &mut producer,
        pool,
        0,
        0,
        &mut request_seq,
    );

    assert!(result.is_err());
    if let Err(request_flow::ProcessRequestError::Parse(_)) = result {
    } else {
        panic!("expected Parse error");
    }
}

/// Full pipeline (request_flow → batch processor → response_flow) without io_uring.
#[test]
fn pipeline_request_to_response_end_to_end() {
    init_factory_pool();

    const RING_SIZE: usize = 256;
    const RESPONSE_QUEUE_SIZE: usize = 256;
    const RESULT_POOL_CAPACITY: usize = RESPONSE_QUEUE_SIZE * 16;

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

    let conn_id = 1u16;
    let thread_id = 0u8;
    let mut request_seq = 0u64;

    let num_vectors = 2u32;
    let features: Vec<f32> = (0..num_vectors as usize * FEATURE_DIM)
        .map(|i| (i / FEATURE_DIM + 1) as f32)
        .collect();
    let buf = one_request_bytes(num_vectors, &features);

    let result = request_flow::process_requests_from_buffer(
        &buf,
        &mut request_producer,
        request_pool,
        conn_id,
        thread_id,
        &mut request_seq,
    );
    assert!(result.is_ok());
    let (consumed, num_published) = result.unwrap();
    assert_eq!(consumed, buf.len());
    assert_eq!(num_published, 1);

    let cycle = batch.process_one_poll_cycle();
    assert!(cycle.is_ok());

    match response_poller.poll() {
        Ok(mut guard) => {
            let wire = response_flow::guard_to_wire_per_conn(&mut guard);
            assert_eq!(wire.len(), 1);
            let conn_buf = wire.get(&conn_id).expect("conn_id 1");
            assert_eq!(conn_buf.len(), 1 + 2 * 4);
            assert_eq!(conn_buf[0], 2);
            let r0 = f32::from_le_bytes(conn_buf[1..5].try_into().unwrap());
            let r1 = f32::from_le_bytes(conn_buf[5..9].try_into().unwrap());
            assert_eq!(r0, 16.0);
            assert_eq!(r1, 32.0);
        }
        Err(_) => panic!("expected one response batch"),
    }

    unsafe {
        libc::close(efd);
    }
}
