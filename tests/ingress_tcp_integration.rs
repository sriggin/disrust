//! Integration test: TCP -> ingress io thread -> request ring.

mod common;

use std::io::Write;
use std::io::{ErrorKind, Read};
use std::net::{SocketAddr, TcpStream};
use std::os::fd::IntoRawFd;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use disruptor::{BusySpin, Polling, build_single_producer};
use socket2::{Domain, Protocol, Socket, Type};

use disrust::buffer_pool::BufferPool;
use disrust::config::{GPU_DISRUPTOR_SIZE, SLAB_CAPACITY};
use disrust::constants::{FEATURE_DIM, MAX_VECTORS_PER_REQUEST};
use disrust::pipeline::connection_registry::ConnectionRegistry;
use disrust::pipeline::response_queue::{ResponseQueue, ResponseReady};
use disrust::protocol;
use disrust::ring_types::InferenceEvent;
use disrust::server::IngressThread;

fn create_listener() -> (std::os::fd::RawFd, SocketAddr) {
    let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP))
        .expect("socket creation failed");
    socket.set_reuse_address(true).unwrap();
    socket.set_reuse_port(true).expect("SO_REUSEPORT failed");
    socket.set_nonblocking(true).unwrap();
    socket.set_nodelay(true).unwrap();
    let addr = SocketAddr::from(([127, 0, 0, 1], 0));
    socket.bind(&addr.into()).expect("bind failed");
    socket.listen(1024).expect("listen failed");
    let local = socket.local_addr().unwrap().as_socket().unwrap();
    (socket.into_raw_fd(), local)
}

fn collect_events(
    event_poller: &mut disruptor::EventPoller<InferenceEvent, disruptor::SingleProducerBarrier>,
    expected: usize,
) -> Vec<(disrust::connection_id::ConnectionRef, u8, u64, Vec<f32>)> {
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut out = Vec::new();
    while Instant::now() < deadline && out.len() < expected {
        match event_poller.poll() {
            Ok(mut guard) => {
                for ev in &mut guard {
                    out.push((
                        ev.conn,
                        ev.num_vectors,
                        ev.request_seq,
                        ev.features.as_slice().to_vec(),
                    ));
                }
            }
            Err(Polling::NoEvents) => thread::sleep(Duration::from_millis(10)),
            Err(Polling::Shutdown) => panic!("event poller shut down unexpectedly"),
        }
    }
    out
}

#[test]
fn ingress_thread_accepts_tcp_requests_and_publishes_ring_events() {
    common::init_factory_pool();

    let builder = build_single_producer(GPU_DISRUPTOR_SIZE, InferenceEvent::factory, BusySpin);
    let (mut event_poller, builder) = builder.event_poller();
    let producer = builder.build();

    let pool_capacity = GPU_DISRUPTOR_SIZE * MAX_VECTORS_PER_REQUEST * FEATURE_DIM;
    let pool = BufferPool::leak_new(pool_capacity);
    let allocator = pool.allocator();
    let response_queue = Arc::new(ResponseQueue::new(SLAB_CAPACITY * 2));
    let publish_gate = Arc::new(Mutex::new(()));
    let registry = Arc::new(ConnectionRegistry::new(4, SLAB_CAPACITY));
    let (listen_fd, addr) = create_listener();

    let ingress = IngressThread::new(
        3,
        listen_fd,
        producer,
        allocator,
        response_queue,
        publish_gate,
        registry,
    );
    thread::Builder::new()
        .name("ingress-test".into())
        .spawn(move || ingress.run())
        .expect("failed to spawn ingress thread");

    let req1_features: Vec<f32> = (0..FEATURE_DIM).map(|i| i as f32 + 1.0).collect();
    let req2_features: Vec<f32> = (0..FEATURE_DIM * 2).map(|i| i as f32 + 100.0).collect();
    let req1 = common::one_request_bytes(1, &req1_features);
    let req2 = common::one_request_bytes(2, &req2_features);

    let mut stream = TcpStream::connect(addr).expect("connect failed");
    stream.set_nodelay(true).unwrap();

    let split = 7usize.min(req1.len());
    stream
        .write_all(&req1[..split])
        .expect("write split req1 failed");
    stream
        .write_all(&req1[split..])
        .expect("write rest req1 failed");
    stream.write_all(&req2).expect("write req2 failed");
    drop(stream);

    let events = collect_events(&mut event_poller, 2);

    assert_eq!(events.len(), 2, "expected two published events");

    let (conn0, num_vectors0, seq0, features0) = &events[0];
    let (conn1, num_vectors1, seq1, features1) = &events[1];

    assert_eq!(conn0.shard_id(), 3);
    assert_eq!(conn1.shard_id(), 3);
    assert_eq!(
        conn0.conn_id, conn1.conn_id,
        "same TCP connection should keep same conn_id"
    );
    assert_eq!(
        conn0.generation(),
        conn1.generation(),
        "same TCP connection should keep same generation"
    );

    assert_eq!(*num_vectors0, 1);
    assert_eq!(*seq0, 0);
    assert_eq!(features0, &req1_features);

    assert_eq!(*num_vectors1, 2);
    assert_eq!(*seq1, 1);
    assert_eq!(features1, &req2_features);
}

#[test]
fn ingress_reuses_retired_slot_after_queued_parse_error() {
    common::init_factory_pool();

    let builder = build_single_producer(1, InferenceEvent::factory, BusySpin);
    let (mut event_poller, builder) = builder.event_poller();
    let producer = builder.build();

    let pool_capacity = GPU_DISRUPTOR_SIZE * MAX_VECTORS_PER_REQUEST * FEATURE_DIM;
    let pool = BufferPool::leak_new(pool_capacity);
    let allocator = pool.allocator();
    let response_queue = Arc::new(ResponseQueue::new(SLAB_CAPACITY * 2));
    let publish_gate = Arc::new(Mutex::new(()));
    let registry = Arc::new(ConnectionRegistry::new(1, SLAB_CAPACITY));
    let (listen_fd, addr) = create_listener();

    let ingress = IngressThread::new(
        0,
        listen_fd,
        producer,
        allocator,
        response_queue,
        publish_gate,
        Arc::clone(&registry),
    );
    thread::Builder::new()
        .name("ingress-retire-test".into())
        .spawn(move || ingress.run())
        .expect("failed to spawn ingress thread");

    let req_features: Vec<f32> = (0..FEATURE_DIM).map(|i| i as f32 + 1.0).collect();
    let req = common::one_request_bytes(1, &req_features);
    let mut malformed = Vec::with_capacity(req.len() * 2 + 4);
    malformed.extend_from_slice(&req);
    malformed.extend_from_slice(&req);
    malformed.extend_from_slice(&0u32.to_le_bytes());

    let mut stream_a = TcpStream::connect(addr).expect("connect failed");
    stream_a.set_nodelay(true).unwrap();
    stream_a
        .write_all(&malformed)
        .expect("write malformed pipeline failed");

    let events_a = collect_events(&mut event_poller, 2);
    assert_eq!(
        events_a.len(),
        2,
        "expected two published events before queued parse error"
    );

    let conn_a = events_a[0].0;
    assert_eq!(
        events_a[1].0, conn_a,
        "same TCP connection should keep same conn_id"
    );
    assert_eq!(events_a[0].2, 0);
    assert_eq!(events_a[1].2, 1);

    drop(stream_a);

    let mut stream_b = TcpStream::connect(addr).expect("second connect failed");
    stream_b.set_nodelay(true).unwrap();
    stream_b
        .write_all(&req)
        .expect("write second request failed");

    let events_b = collect_events(&mut event_poller, 1);
    assert_eq!(
        events_b.len(),
        1,
        "expected second connection request event"
    );

    let conn_b = events_b[0].0;
    assert_eq!(
        conn_b.conn_id, conn_a.conn_id,
        "retired ingress slot should be reclaimed before the next accept"
    );
    assert_ne!(
        conn_b.generation(),
        conn_a.generation(),
        "reused slab slot should advance generation"
    );
}

#[test]
fn ingress_submits_writes_while_queued_parse_work_remains() {
    common::init_factory_pool();

    let builder = build_single_producer(GPU_DISRUPTOR_SIZE, InferenceEvent::factory, BusySpin);
    let (mut event_poller, builder) = builder.event_poller();
    let producer = builder.build();

    let pool_capacity = GPU_DISRUPTOR_SIZE * MAX_VECTORS_PER_REQUEST * FEATURE_DIM;
    let pool = BufferPool::leak_new(pool_capacity);
    let allocator = pool.allocator();
    let response_queue = Arc::new(ResponseQueue::new(SLAB_CAPACITY * 2));
    let publish_gate = Arc::new(Mutex::new(()));
    let registry = Arc::new(ConnectionRegistry::new(1, SLAB_CAPACITY));
    let (listen_fd, addr) = create_listener();

    let ingress = IngressThread::new(
        0,
        listen_fd,
        producer,
        allocator,
        Arc::clone(&response_queue),
        publish_gate,
        registry,
    );
    thread::Builder::new()
        .name("ingress-write-progress-test".into())
        .spawn(move || ingress.run())
        .expect("failed to spawn ingress thread");

    let req1_features: Vec<f32> = (0..FEATURE_DIM).map(|i| i as f32 + 1.0).collect();
    let req2_features: Vec<f32> = (0..FEATURE_DIM).map(|i| i as f32 + 100.0).collect();
    let req1 = common::one_request_bytes(1, &req1_features);
    let req2 = common::one_request_bytes(1, &req2_features);

    let mut stream = TcpStream::connect(addr).expect("connect failed");
    stream.set_nodelay(true).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(1)))
        .expect("set read timeout");

    // Leave a partial second request buffered so ingress stays on the queued-parse path
    // after publishing the first event.
    let partial_req2 = 6usize.min(req2.len());
    stream.write_all(&req1).expect("write first request failed");
    stream
        .write_all(&req2[..partial_req2])
        .expect("write partial second request failed");

    let events = collect_events(&mut event_poller, 1);
    assert_eq!(events.len(), 1, "expected first published event");
    let conn = events[0].0;

    let expected_sum = req1_features.iter().copied().sum::<f32>();
    let mut expected = [0u8; 5];
    protocol::encode_response(&[expected_sum], &mut expected);
    response_queue.push(ResponseReady::encode(conn, 0, 1, &[expected_sum]));

    let mut response = [0u8; 5];
    match stream.read_exact(&mut response) {
        Ok(()) => {}
        Err(err) if err.kind() == ErrorKind::WouldBlock || err.kind() == ErrorKind::TimedOut => {
            panic!("timed out waiting for first response while queued parse work remained")
        }
        Err(err) => panic!("read first response failed: {err}"),
    }
    assert_eq!(response, expected, "unexpected first response bytes");
}

#[test]
fn ingress_accepts_many_simultaneous_connections() {
    // Verifies that accept resubmission (and later multishot accept) keeps
    // firing for all connections: N clients connect, each sends one request,
    // all N events must be published to the ring.
    const N: usize = 8;
    common::init_factory_pool();

    let builder = build_single_producer(GPU_DISRUPTOR_SIZE, InferenceEvent::factory, BusySpin);
    let (mut event_poller, builder) = builder.event_poller();
    let producer = builder.build();

    let pool_capacity = GPU_DISRUPTOR_SIZE * MAX_VECTORS_PER_REQUEST * FEATURE_DIM;
    let pool = BufferPool::leak_new(pool_capacity);
    let allocator = pool.allocator();
    let response_queue = Arc::new(ResponseQueue::new(SLAB_CAPACITY * 2));
    let publish_gate = Arc::new(Mutex::new(()));
    let registry = Arc::new(ConnectionRegistry::new(1, SLAB_CAPACITY));
    let (listen_fd, addr) = create_listener();

    let ingress = IngressThread::new(
        0,
        listen_fd,
        producer,
        allocator,
        response_queue,
        publish_gate,
        registry,
    );
    thread::Builder::new()
        .name("ingress-multi-conn-test".into())
        .spawn(move || ingress.run())
        .expect("failed to spawn ingress thread");

    let features: Vec<f32> = (0..FEATURE_DIM).map(|i| i as f32).collect();
    let req = common::one_request_bytes(1, &features);

    let _streams: Vec<_> = (0..N)
        .map(|_| {
            let mut stream = TcpStream::connect(addr).expect("connect failed");
            stream.set_nodelay(true).unwrap();
            stream.write_all(&req).expect("write failed");
            stream
        })
        .collect();

    let events = collect_events(&mut event_poller, N);

    assert_eq!(
        events.len(),
        N,
        "all {N} connections must produce a ring event"
    );

    // Each event must come from a distinct connection.
    let mut conn_ids: Vec<_> = events.iter().map(|(c, _, _, _)| c.conn_id).collect();
    conn_ids.sort_unstable();
    conn_ids.dedup();
    assert_eq!(
        conn_ids.len(),
        N,
        "each connection must have a distinct conn_id"
    );

    // Feature data must be correct for every event.
    for (_, num_vecs, _, feats) in &events {
        assert_eq!(*num_vecs, 1);
        assert_eq!(feats, &features);
    }
}
