mod common;

use std::collections::HashMap;
use std::os::fd::IntoRawFd;
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use disruptor::{BusySpin, Producer, build_multi_producer};

use disrust::buffer_pool::BufferPool;
use disrust::config::SLAB_CAPACITY;
use disrust::constants::FEATURE_DIM;
use disrust::metrics;
use disrust::pipeline::connection_registry::ConnectionRegistry;
use disrust::pipeline::inference::InferenceConsumer;
use disrust::pipeline::response_queue::ResponseQueue;
use disrust::pipeline::{InferenceBackend, OrtBackend};
use disrust::ring_types::InferenceEvent;

#[cfg(feature = "cuda")]
use disrust::cuda::memory::{alloc_pinned, free_pinned};
#[cfg(feature = "cuda")]
use disrust::cuda::preflight::verify_cuda_startup;

fn backend_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        #[cfg(feature = "cuda")]
        {
            if let Err(err) = verify_cuda_startup() {
                eprintln!("skipping CUDA-backed inference integration test: {err}");
                return false;
            }
            let ptr = match alloc_pinned(std::mem::size_of::<f32>()) {
                Ok(ptr) => ptr,
                Err(err) => {
                    eprintln!("skipping CUDA-backed inference integration test: {err}");
                    return false;
                }
            };
            unsafe {
                let _ = free_pinned(ptr);
            }
        }

        OrtBackend::init();
        true
    })
}

fn encode_expected_response(fill: f32) -> [u8; 5] {
    let mut bytes = [0u8; 5];
    bytes[0] = 1;
    let weighted_dot = fill * ((FEATURE_DIM * (FEATURE_DIM + 1)) / 2) as f32;
    bytes[1..].copy_from_slice(&weighted_dot.to_le_bytes());
    bytes
}

fn run_pipeline_order_test() {
    common::init_factory_pool();
    if !backend_available() {
        return;
    }

    const RING_SIZE: usize = 256;
    let builder = build_multi_producer(RING_SIZE, InferenceEvent::factory, BusySpin);
    let (submission_poller, builder) = builder.event_poller();
    let (completion_poller, builder) = builder.and_then().event_poller();
    let mut producer = builder.build();

    let model_bytes =
        std::fs::read("tests/models/ort_verify_model.onnx").expect("failed to read test model");
    let backend = OrtBackend::new(&model_bytes, 1);

    let response_queue = Arc::new(ResponseQueue::new(32));
    let registry = Arc::new(ConnectionRegistry::new(1, SLAB_CAPACITY));

    let inference = InferenceConsumer::new(
        submission_poller,
        completion_poller,
        backend,
        vec![Arc::clone(&response_queue)],
        Arc::clone(&registry),
        256,
        Duration::from_micros(500),
    );

    let _inference_handle = thread::Builder::new()
        .name("test-inference".into())
        .spawn(move || inference.run())
        .expect("failed to spawn inference thread");

    let pool = BufferPool::leak_new(RING_SIZE * FEATURE_DIM);
    let mut allocator = pool.allocator();

    let (conn0_sock, _peer0) = UnixStream::pair().expect("unix pair");
    let (conn1_sock, _peer1) = UnixStream::pair().expect("unix pair");
    let conn0 = registry.open(0, 0, conn0_sock.into_raw_fd());
    let conn1 = registry.open(0, 1, conn1_sock.into_raw_fd());

    let planned = [
        (conn0, 0u64, 1.0f32),
        (conn1, 0u64, 2.0f32),
        (conn0, 1u64, 3.0f32),
        (conn1, 1u64, 4.0f32),
        (conn0, 2u64, 5.0f32),
        (conn1, 2u64, 6.0f32),
    ];

    let mut published_at_ns = 1u64;
    for (conn, request_seq, fill) in planned {
        loop {
            match producer.try_publish(|slot| {
                let mut slice = allocator.alloc(FEATURE_DIM).expect("pool alloc");
                slice.as_mut_slice().fill(fill);
                slot.conn = conn;
                slot.request_seq = request_seq;
                slot.num_vectors = 1;
                slot.published_at_ns = published_at_ns;
                slot.features = slice.freeze();
            }) {
                Ok(_) => {
                    metrics::inc_requests_published();
                    metrics::inc_req_occ();
                    published_at_ns += 1;
                    break;
                }
                Err(_) => std::hint::spin_loop(),
            }
        }
    }

    let expected = HashMap::from([
        (
            conn0,
            vec![
                encode_expected_response(1.0),
                encode_expected_response(3.0),
                encode_expected_response(5.0),
            ],
        ),
        (
            conn1,
            vec![
                encode_expected_response(2.0),
                encode_expected_response(4.0),
                encode_expected_response(6.0),
            ],
        ),
    ]);

    let mut observed = HashMap::from([(conn0, Vec::new()), (conn1, Vec::new())]);
    let deadline = Instant::now() + Duration::from_secs(5);
    while observed.values().map(Vec::len).sum::<usize>() < planned.len() {
        if let Some(response) = response_queue.pop() {
            let entry = observed
                .get_mut(&response.conn)
                .expect("unexpected connection in observed map");
            let mut frame = [0u8; 5];
            frame.copy_from_slice(&response.data[..response.len]);
            entry.push(frame);
        } else {
            assert!(
                Instant::now() < deadline,
                "timed out waiting for inference pipeline to produce expected responses; observed={observed:?}",
            );
            thread::sleep(Duration::from_millis(1));
        }
    }

    assert_eq!(observed, expected);
}

fn run_sustained_response_queue_test() {
    common::init_factory_pool();
    if !backend_available() {
        return;
    }

    const RING_SIZE: usize = 512;
    const CONNECTIONS: usize = 4;
    const REQUESTS_PER_CONNECTION: usize = 512;

    let builder = build_multi_producer(RING_SIZE, InferenceEvent::factory, BusySpin);
    let (submission_poller, builder) = builder.event_poller();
    let (completion_poller, builder) = builder.and_then().event_poller();
    let mut producer = builder.build();

    let model_bytes =
        std::fs::read("tests/models/ort_verify_model.onnx").expect("failed to read test model");
    let backend = OrtBackend::new(&model_bytes, 1);

    let response_queue = Arc::new(ResponseQueue::new(CONNECTIONS * REQUESTS_PER_CONNECTION));
    let registry = Arc::new(ConnectionRegistry::new(1, SLAB_CAPACITY));
    let stop = Arc::new(AtomicBool::new(false));

    let inference = InferenceConsumer::new(
        submission_poller,
        completion_poller,
        backend,
        vec![Arc::clone(&response_queue)],
        Arc::clone(&registry),
        256,
        Duration::from_micros(500),
    );
    let handle = thread::Builder::new()
        .name("test-inference".into())
        .spawn({
            let stop = Arc::clone(&stop);
            move || inference.run_until(stop)
        })
        .expect("failed to spawn inference thread");

    let pool = BufferPool::leak_new(RING_SIZE * FEATURE_DIM);
    let mut allocator = pool.allocator();

    let mut conns = Vec::with_capacity(CONNECTIONS);
    let mut expected = HashMap::new();
    for conn_id in 0..CONNECTIONS {
        let (server_sock, _peer_sock) = UnixStream::pair().expect("unix pair");
        let conn = registry.open(0, conn_id as u16, server_sock.into_raw_fd());
        conns.push(conn);

        let expected_frames: Vec<[u8; 5]> = (0..REQUESTS_PER_CONNECTION)
            .map(|request_seq| encode_expected_response((conn_id * 1000 + request_seq) as f32))
            .collect();
        expected.insert(conn, expected_frames);
    }

    let publisher_deadline = Instant::now() + Duration::from_secs(10);
    let mut published_at_ns = 1u64;
    for request_seq in 0..REQUESTS_PER_CONNECTION {
        for (conn_id, conn) in conns.iter().copied().enumerate() {
            loop {
                match producer.try_publish(|slot| {
                    let fill = (conn_id * 1000 + request_seq) as f32;
                    let mut slice = allocator.alloc(FEATURE_DIM).expect("pool alloc");
                    slice.as_mut_slice().fill(fill);
                    slot.conn = conn;
                    slot.request_seq = request_seq as u64;
                    slot.num_vectors = 1;
                    slot.published_at_ns = published_at_ns;
                    slot.features = slice.freeze();
                }) {
                    Ok(_) => {
                        metrics::inc_requests_published();
                        metrics::inc_req_occ();
                        published_at_ns += 1;
                        break;
                    }
                    Err(_) => {
                        assert!(
                            Instant::now() < publisher_deadline,
                            "timed out publishing sustained requests"
                        );
                        std::hint::spin_loop();
                    }
                }
            }
        }
    }

    let receive_deadline = Instant::now() + Duration::from_secs(15);
    let mut observed = HashMap::from_iter(conns.iter().copied().map(|conn| (conn, Vec::new())));
    while observed.values().map(Vec::len).sum::<usize>() < CONNECTIONS * REQUESTS_PER_CONNECTION {
        if let Some(response) = response_queue.pop() {
            let frames = observed
                .get_mut(&response.conn)
                .expect("unexpected connection in observed map");
            let mut frame = [0u8; 5];
            frame.copy_from_slice(&response.data[..response.len]);
            frames.push(frame);
        } else {
            assert!(
                Instant::now() < receive_deadline,
                "timed out waiting for sustained response queue progress"
            );
            thread::sleep(Duration::from_millis(1));
        }
    }

    for conn in conns.iter().copied() {
        registry.mark_read_closed(conn, REQUESTS_PER_CONNECTION as u64);
    }
    stop.store(true, Ordering::Relaxed);
    handle.join().expect("pipeline thread panicked");

    assert_eq!(observed, expected);
}

#[test]
fn inference_consumer_preserves_per_connection_order() {
    run_pipeline_order_test();
}

#[test]
fn inference_and_response_queue_sustain_progress() {
    run_sustained_response_queue_test();
}
