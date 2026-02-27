//! Integration test: request path (bytes → parse → alloc → publish) without io_uring.

mod common;

use disruptor::{BusySpin, build_single_producer};

use disrust::buffer_pool::BufferPool;
use disrust::constants::{FEATURE_DIM, MAX_VECTORS_PER_REQUEST};
use disrust::request_flow;
use disrust::ring_types::InferenceEvent;

#[test]
fn request_flow_processes_one_request_and_consumer_sees_event() {
    common::init_factory_pool();

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
    let buf = common::one_request_bytes(num_vectors, &features);

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
    common::init_factory_pool();

    const RING_SIZE: usize = 256;
    let builder = build_single_producer(RING_SIZE, InferenceEvent::factory, BusySpin);
    let (mut poller, builder) = builder.event_poller();
    let mut producer = builder.build();

    let pool_capacity = RING_SIZE * MAX_VECTORS_PER_REQUEST * FEATURE_DIM;
    let pool = BufferPool::leak_new(pool_capacity);

    let conn_id = 2u16;
    let thread_id = 0u8;
    let mut request_seq = 0u64;

    let r1 = common::one_request_bytes(1, &[1.0f32; FEATURE_DIM]);
    let r2 = common::one_request_bytes(1, &[2.0f32; FEATURE_DIM]);
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
    common::init_factory_pool();

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
    common::init_factory_pool();

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
