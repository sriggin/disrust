mod common;

use std::collections::{HashMap, VecDeque};
use std::io::Read;
use std::os::fd::IntoRawFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use disruptor::{BusySpin, Producer, build_multi_producer};
use ort::init_from;
use std::os::unix::net::UnixStream;

use disrust::buffer_pool::BufferPool;
use disrust::config::SLAB_CAPACITY;
use disrust::constants::FEATURE_DIM;
use disrust::metrics;
use disrust::pipeline::connection_registry::ConnectionRegistry;
use disrust::pipeline::inference::InferenceConsumer;
use disrust::pipeline::ready_queue::{ConnectionRef, ReadyQueue};
use disrust::pipeline::session::InferenceSession;
use disrust::pipeline::verify_ort_dylib_present;
use disrust::pipeline::writer::WriterConsumer;
use disrust::ring_types::InferenceEvent;

#[cfg(feature = "cuda")]
use disrust::cuda::preflight::verify_cuda_startup;

fn ensure_ort_init() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        let ort_dylib = verify_ort_dylib_present().expect("ORT dylib preflight failed");
        init_from(&ort_dylib)
            .expect("ort::init_from failed")
            .commit();

        #[cfg(feature = "cuda")]
        verify_cuda_startup().expect("CUDA preflight failed");
    });
}

fn encode_expected_response(fill: f32) -> [u8; 5] {
    let mut bytes = [0u8; 5];
    bytes[0] = 1;
    let sum = fill * FEATURE_DIM as f32;
    bytes[1..].copy_from_slice(&sum.to_le_bytes());
    bytes
}

fn run_pipeline_order_test() {
    common::init_factory_pool();
    ensure_ort_init();

    const RING_SIZE: usize = 256;
    let builder = build_multi_producer(RING_SIZE, InferenceEvent::factory, BusySpin);
    let (submission_poller, builder) = builder.event_poller();
    let (completion_poller, builder) = builder.and_then().event_poller();
    let mut producer = builder.build();

    let model_bytes =
        std::fs::read("tests/models/ort_sum_model.onnx").expect("failed to read test model");
    let sessions = vec![InferenceSession::new(&model_bytes)];

    let ready_queue = Arc::new(ReadyQueue::new(32));
    let registry = Arc::new(ConnectionRegistry::new(1, SLAB_CAPACITY));

    let inference = InferenceConsumer::new(
        submission_poller,
        completion_poller,
        sessions,
        Arc::clone(&ready_queue),
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

    let mut observed: HashMap<ConnectionRef, Vec<[u8; 5]>> =
        HashMap::from([(conn0, Vec::new()), (conn1, Vec::new())]);
    let mut pending = VecDeque::new();
    let deadline = Instant::now() + Duration::from_secs(5);

    while observed.values().map(Vec::len).sum::<usize>() < planned.len() {
        while let Some(conn) = ready_queue.pop() {
            pending.push_back(conn);
        }

        if let Some(conn) = pending.pop_front() {
            let Some(flush) = registry.take_flush(conn) else {
                continue;
            };

            let iovecs =
                unsafe { std::slice::from_raw_parts(flush.iovecs, flush.iov_count as usize) };
            let mut written = 0usize;
            for iov in iovecs {
                let bytes =
                    unsafe { std::slice::from_raw_parts(iov.iov_base.cast::<u8>(), iov.iov_len) };
                written += iov.iov_len;
                assert_eq!(bytes.len(), 5, "expected one response frame per iovec");
                let mut frame = [0u8; 5];
                frame.copy_from_slice(bytes);
                observed
                    .get_mut(&conn)
                    .expect("unexpected connection in observed map")
                    .push(frame);
            }

            match registry.handle_write_result(conn, written as i32) {
                disrust::pipeline::connection_registry::WriteResult::NeedsResubmit
                | disrust::pipeline::connection_registry::WriteResult::ReadyAgain => {
                    pending.push_back(conn)
                }
                disrust::pipeline::connection_registry::WriteResult::Completed
                | disrust::pipeline::connection_registry::WriteResult::Idle
                | disrust::pipeline::connection_registry::WriteResult::Stale
                | disrust::pipeline::connection_registry::WriteResult::Error(_) => {}
            }
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

fn run_sustained_pipeline_writer_test() {
    common::init_factory_pool();
    ensure_ort_init();

    const RING_SIZE: usize = 512;
    const CONNECTIONS: usize = 4;
    const REQUESTS_PER_CONNECTION: usize = 512;

    let builder = build_multi_producer(RING_SIZE, InferenceEvent::factory, BusySpin);
    let (submission_poller, builder) = builder.event_poller();
    let (completion_poller, builder) = builder.and_then().event_poller();
    let mut producer = builder.build();

    let model_bytes =
        std::fs::read("tests/models/ort_sum_model.onnx").expect("failed to read test model");
    let sessions = vec![InferenceSession::new(&model_bytes)];

    let ready_queue = Arc::new(ReadyQueue::new(128));
    let registry = Arc::new(ConnectionRegistry::new(1, SLAB_CAPACITY));
    let stop = Arc::new(AtomicBool::new(false));

    let inference = InferenceConsumer::new(
        submission_poller,
        completion_poller,
        sessions,
        Arc::clone(&ready_queue),
        Arc::clone(&registry),
        256,
        Duration::from_micros(500),
    );
    let mut handles = vec![
        thread::Builder::new()
            .name("test-inference".into())
            .spawn({
                let stop = Arc::clone(&stop);
                move || inference.run_until(stop)
            })
            .expect("failed to spawn inference thread"),
    ];

    let writer = WriterConsumer::new(Arc::clone(&ready_queue), Arc::clone(&registry))
        .expect("failed to create writer consumer");
    let writer_stop = Arc::clone(&stop);
    handles.push(
        thread::Builder::new()
            .name("test-writer".into())
            .spawn(move || writer.run_until(writer_stop))
            .expect("failed to spawn writer thread"),
    );

    let pool = BufferPool::leak_new(RING_SIZE * FEATURE_DIM);
    let mut allocator = pool.allocator();

    let mut conns = Vec::with_capacity(CONNECTIONS);
    let mut expected = HashMap::new();
    let (result_tx, result_rx) = mpsc::channel();

    for conn_id in 0..CONNECTIONS {
        let (server_sock, mut peer_sock) = UnixStream::pair().expect("unix pair");
        peer_sock
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("set read timeout");
        let conn = registry.open(0, conn_id as u16, server_sock.into_raw_fd());
        conns.push(conn);

        let expected_frames: Vec<[u8; 5]> = (0..REQUESTS_PER_CONNECTION)
            .map(|request_seq| encode_expected_response((conn_id * 1000 + request_seq) as f32))
            .collect();
        expected.insert(conn, expected_frames.clone());

        let tx = result_tx.clone();
        thread::Builder::new()
            .name(format!("test-reader-{conn_id}"))
            .spawn(move || {
                let mut observed = Vec::with_capacity(REQUESTS_PER_CONNECTION);
                for _ in 0..REQUESTS_PER_CONNECTION {
                    let mut frame = [0u8; 5];
                    peer_sock
                        .read_exact(&mut frame)
                        .expect("reader failed to receive response frame");
                    observed.push(frame);
                }
                tx.send((conn, observed)).expect("send reader result");
            })
            .expect("failed to spawn reader thread");
    }
    drop(result_tx);

    let publisher_deadline = Instant::now() + Duration::from_secs(10);
    let mut published_at_ns = 1u64;
    let mut published_count = 0usize;
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
                        published_count += 1;
                        break;
                    }
                    Err(_) => {
                        assert!(
                            Instant::now() < publisher_deadline,
                            "timed out publishing requests after {} successes; {}",
                            published_count,
                            snapshot_summary()
                        );
                        std::hint::spin_loop();
                    }
                }
            }
        }
    }

    let receive_deadline = Instant::now() + Duration::from_secs(15);
    let mut observed = HashMap::new();
    while observed.len() < CONNECTIONS {
        let remaining = receive_deadline.saturating_duration_since(Instant::now());
        let (conn, frames) = result_rx.recv_timeout(remaining).expect("recv reader result");
        observed.insert(conn, frames);
    }

    for conn in conns.iter().copied() {
        registry.mark_read_closed(conn, REQUESTS_PER_CONNECTION as u64);
    }
    stop.store(true, Ordering::Relaxed);

    for handle in handles {
        handle.join().expect("pipeline thread panicked");
    }

    assert_eq!(observed, expected);
}

fn snapshot_summary() -> String {
    let snap = metrics::snapshot();
    format!(
        "req_pub={} batches_sub={} batches_cmp={} slots={} responses={} req_occ={} req_max={} session_waits={} cq_empty_waits={} poll_stalls={} writes_sqes={} writes_cqes={}",
        snap.requests_published,
        snap.batches_submitted,
        snap.batches_completed,
        snap.slots_submitted,
        snap.responses_written,
        snap.req_occ,
        snap.req_max_occ,
        snap.session_waits,
        snap.completion_queue_empty_waits,
        snap.completion_poll_stalls,
        snap.write_sqes,
        snap.write_cqes
    )
}

#[test]
#[cfg_attr(
    feature = "cuda",
    ignore = "pipeline integration test is validated on the no-cuda path"
)]
fn inference_consumer_preserves_per_connection_order() {
    run_pipeline_order_test();
}

#[test]
#[cfg_attr(
    feature = "cuda",
    ignore = "pipeline integration test is validated on the no-cuda path"
)]
fn inference_and_writer_sustain_progress() {
    run_sustained_pipeline_writer_test();
}
