//! Integration test: TCP -> ingress io thread -> request ring.

mod common;

use std::io::Write;
use std::net::{SocketAddr, TcpStream};
use std::os::fd::IntoRawFd;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use disruptor::{BusySpin, build_single_producer};
use socket2::{Domain, Protocol, Socket, Type};

use disrust::buffer_pool::BufferPool;
use disrust::config::{GPU_DISRUPTOR_SIZE, SLAB_CAPACITY};
use disrust::constants::{FEATURE_DIM, MAX_VECTORS_PER_REQUEST};
use disrust::pipeline::connection_registry::ConnectionRegistry;
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

#[test]
fn ingress_thread_accepts_tcp_requests_and_publishes_ring_events() {
    common::init_factory_pool();

    let builder = build_single_producer(GPU_DISRUPTOR_SIZE, InferenceEvent::factory, BusySpin);
    let (mut event_poller, builder) = builder.event_poller();
    let producer = builder.build();

    let pool_capacity = GPU_DISRUPTOR_SIZE * MAX_VECTORS_PER_REQUEST * FEATURE_DIM;
    let pool = BufferPool::leak_new(pool_capacity);
    let allocator = pool.allocator();
    let registry = Arc::new(ConnectionRegistry::new(SLAB_CAPACITY));
    let (listen_fd, addr) = create_listener();

    let ingress = IngressThread::new(3, listen_fd, producer, allocator, registry);
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

    let events = {
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut out = Vec::new();
        while Instant::now() < deadline && out.len() < 2 {
            match event_poller.poll() {
                Ok(mut guard) => {
                    for ev in &mut guard {
                        out.push((
                            ev.conn_id,
                            ev.generation,
                            ev.io_thread_id,
                            ev.num_vectors,
                            ev.request_seq,
                            ev.features.as_slice().to_vec(),
                        ));
                    }
                }
                Err(_) => thread::sleep(Duration::from_millis(10)),
            }
        }
        out
    };

    assert_eq!(events.len(), 2, "expected two published events");

    let (conn_id0, generation0, thread_id0, num_vectors0, seq0, features0) = &events[0];
    let (conn_id1, generation1, thread_id1, num_vectors1, seq1, features1) = &events[1];

    assert_eq!(*thread_id0, 3);
    assert_eq!(*thread_id1, 3);
    assert_eq!(
        *conn_id0, *conn_id1,
        "same TCP connection should keep same conn_id"
    );
    assert_eq!(
        *generation0, *generation1,
        "same TCP connection should keep same generation"
    );

    assert_eq!(*num_vectors0, 1);
    assert_eq!(*seq0, 0);
    assert_eq!(features0, &req1_features);

    assert_eq!(*num_vectors1, 2);
    assert_eq!(*seq1, 1);
    assert_eq!(features1, &req2_features);
}
