//! Full pipeline integration test: request_flow → batch processor → response_flow (no io_uring).

mod common;

use disruptor::{BusySpin, build_single_producer};

use disrust::batch_processor::BatchProcessor;
use disrust::buffer_pool::BufferPool;
use disrust::constants::{FEATURE_DIM, MAX_VECTORS_PER_REQUEST};
use disrust::request_flow;
use disrust::response_flow;
use disrust::ring_types::InferenceEvent;

#[test]
fn pipeline_request_to_response_end_to_end() {
    common::init_factory_pool();
    const RING_SIZE: usize = 256;
    const RESPONSE_QUEUE_SIZE: usize = 256;
    const RESULT_POOL_CAPACITY: usize = RESPONSE_QUEUE_SIZE * 16;

    let builder = build_single_producer(RING_SIZE, InferenceEvent::factory, BusySpin);
    let (request_poller, builder) = builder.event_poller();
    let mut request_producer = builder.build();
    let request_pool = BufferPool::leak_new(RING_SIZE * MAX_VECTORS_PER_REQUEST * FEATURE_DIM);
    let efd = common::create_eventfd();
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
    let buf = common::one_request_bytes(num_vectors, &features);

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

#[test]
fn pipeline_multiple_requests_same_conn() {
    common::init_factory_pool();
    const RING_SIZE: usize = 256;
    const RESPONSE_QUEUE_SIZE: usize = 256;
    const RESULT_POOL_CAPACITY: usize = RESPONSE_QUEUE_SIZE * 16;

    let builder = build_single_producer(RING_SIZE, InferenceEvent::factory, BusySpin);
    let (request_poller, builder) = builder.event_poller();
    let mut request_producer = builder.build();
    let request_pool = BufferPool::leak_new(RING_SIZE * MAX_VECTORS_PER_REQUEST * FEATURE_DIM);
    let efd = common::create_eventfd();
    assert!(efd >= 0);
    let (resp_producer, mut response_poller) =
        disrust::response_queue::build_response_channel(RESPONSE_QUEUE_SIZE, efd);
    let result_pool = BufferPool::leak_new(RESULT_POOL_CAPACITY);

    let mut batch = BatchProcessor {
        poller: request_poller,
        response_producers: vec![resp_producer],
        result_pools: vec![result_pool],
    };

    let conn_id = 2u16;
    let thread_id = 0u8;
    let mut request_seq = 0u64;
    let r1 = common::one_request_bytes(1, &[1.0f32; FEATURE_DIM]);
    let r2 = common::one_request_bytes(1, &[2.0f32; FEATURE_DIM]);
    let mut buf = r1;
    buf.extend_from_slice(&r2);

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
    assert_eq!(num_published, 2);

    let cycle = batch.process_one_poll_cycle();
    assert!(cycle.is_ok());

    match response_poller.poll() {
        Ok(mut guard) => {
            let wire = response_flow::guard_to_wire_per_conn(&mut guard);
            assert_eq!(wire.len(), 1);
            let conn_buf = wire.get(&conn_id).unwrap();
            assert_eq!(conn_buf.len(), 10);
            assert_eq!(conn_buf[0], 1);
            assert_eq!(f32::from_le_bytes(conn_buf[1..5].try_into().unwrap()), 16.0);
            assert_eq!(conn_buf[5], 1);
            assert_eq!(
                f32::from_le_bytes(conn_buf[6..10].try_into().unwrap()),
                32.0
            );
        }
        Err(_) => panic!("expected responses"),
    }
    unsafe {
        libc::close(efd);
    }
}
