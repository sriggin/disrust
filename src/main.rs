mod batch_processor;
mod buffer_pool;
mod constants;
mod io_thread;
mod protocol;
mod response_queue;
mod ring_types;

use std::os::unix::io::{IntoRawFd, RawFd};
use std::thread;

use disruptor::{BusySpin, build_multi_producer};
use socket2::{Domain, Protocol, Socket, Type};

use batch_processor::BatchProcessor;
use buffer_pool::{BufferPool, set_factory_pool};
use constants::{FEATURE_DIM, MAX_VECTORS_PER_REQUEST};
use io_thread::IoThread;
use response_queue::build_response_channel;
use ring_types::InferenceEvent;

const DEFAULT_PORT: u16 = 9900;
const DISRUPTOR_SIZE: usize = 65536;
const RESPONSE_QUEUE_SIZE: usize = 8192;

// Size each buffer pool to handle all in-flight requests at max size.
// CRITICAL: Pool must be >= disruptor capacity * max request size to prevent
// wraparound from overwriting unread data. This is worst-case sizing (conservative).
//
// PERFORMANCE NOTE: This creates 2GB pools which will have poor cache behavior.
// Real workloads are mostly 1-8 vectors (not 64), so this is very conservative.
// See PERFORMANCE.md for right-sizing opportunities based on typical workload.
const BUFFER_POOL_CAPACITY: usize = DISRUPTOR_SIZE * MAX_VECTORS_PER_REQUEST * FEATURE_DIM;

// Result pool capacity (for responses >INLINE_RESULT_CAPACITY vectors).
// Tunable based on expected workload. Most responses are inline (≤10 vectors).
// Min: Enough for a few large responses. Max: All response queue slots at max size.
const RESULT_POOL_CAPACITY: usize = RESPONSE_QUEUE_SIZE * 16; // Conservative: ~16 vectors average

fn create_listener(port: u16) -> Socket {
    let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP))
        .expect("failed to create socket");
    socket.set_reuse_address(true).unwrap();

    // SO_REUSEPORT via raw setsockopt (not in socket2 API)
    unsafe {
        use std::os::unix::io::AsRawFd;
        let optval: libc::c_int = 1;
        libc::setsockopt(
            socket.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_REUSEPORT,
            &optval as *const _ as *const libc::c_void,
            std::mem::size_of::<libc::c_int>() as libc::socklen_t,
        );
    }

    socket.set_nonblocking(true).unwrap();
    socket.set_nodelay(true).unwrap();

    let addr = std::net::SocketAddrV4::new(std::net::Ipv4Addr::UNSPECIFIED, port);
    socket.bind(&addr.into()).expect("failed to bind");
    socket.listen(1024).expect("failed to listen");
    socket
}

fn create_eventfd() -> RawFd {
    unsafe { libc::eventfd(0, libc::EFD_NONBLOCK) }
}

fn main() {
    let num_threads: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            let n = thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            (n - 1).max(1)
        });

    let port: u16 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PORT);

    // Safety checks: ensure pool capacities are within valid bounds
    const MIN_BUFFER_POOL_CAPACITY: usize = DISRUPTOR_SIZE * FEATURE_DIM;
    const MIN_RESULT_POOL_CAPACITY: usize = MAX_VECTORS_PER_REQUEST * 4; // At least a few large responses
    const MAX_RESULT_POOL_CAPACITY: usize = RESPONSE_QUEUE_SIZE * MAX_VECTORS_PER_REQUEST;

    const {
        assert!(
            BUFFER_POOL_CAPACITY >= MIN_BUFFER_POOL_CAPACITY,
            "buffer pool capacity is too small for disruptor size"
        );
        assert!(
            RESULT_POOL_CAPACITY >= MIN_RESULT_POOL_CAPACITY,
            "result pool capacity is too small"
        );
        assert!(
            RESULT_POOL_CAPACITY <= MAX_RESULT_POOL_CAPACITY,
            "result pool capacity exceeds maximum needed"
        );
    }

    eprintln!("disrust: {} IO threads, port {}", num_threads, port);
    eprintln!(
        "  buffer pool: {} MB per thread ({} f32s)",
        BUFFER_POOL_CAPACITY * 4 / 1_000_000,
        BUFFER_POOL_CAPACITY
    );

    // Factory needs a pool for empty slices created during ring initialization.
    let factory_pool = BufferPool::leak_new(1);
    set_factory_pool(factory_pool);

    // Build the request disruptor (MPSC: IO threads → batch processor)
    let builder = build_multi_producer(DISRUPTOR_SIZE, InferenceEvent::factory, BusySpin);
    let (request_poller, builder) = builder.event_poller();
    let producer = builder.build();

    // Build per-IO-thread response channels
    let mut response_producers = Vec::with_capacity(num_threads);
    let mut response_pollers = Vec::with_capacity(num_threads);
    let mut eventfds = Vec::with_capacity(num_threads);

    for _ in 0..num_threads {
        let efd = create_eventfd();
        assert!(efd >= 0, "failed to create eventfd");
        let (resp_prod, resp_poll) = build_response_channel(RESPONSE_QUEUE_SIZE, efd);
        response_producers.push(resp_prod);
        response_pollers.push(resp_poll);
        eventfds.push(efd);
    }

    // Create result buffer pools - one per IO thread (for large responses >INLINE_RESULT_CAPACITY)
    let result_pools: Vec<&'static BufferPool> = (0..num_threads)
        .map(|_| BufferPool::leak_new(RESULT_POOL_CAPACITY))
        .collect();

    eprintln!(
        "  result pool: {} KB per thread ({} f32s)",
        RESULT_POOL_CAPACITY * 4 / 1_000,
        RESULT_POOL_CAPACITY
    );

    // Spawn batch processor thread
    let batch = BatchProcessor {
        poller: request_poller,
        response_producers,
        result_pools,
    };
    let batch_handle = thread::Builder::new()
        .name("batch-processor".into())
        .spawn(move || batch.run())
        .expect("failed to spawn batch processor");

    // Spawn IO threads
    let mut io_handles = Vec::with_capacity(num_threads);
    for (i, (response_poller, efd)) in response_pollers
        .into_iter()
        .zip(eventfds.into_iter())
        .enumerate()
    {
        let listen_socket = create_listener(port);
        let listen_fd = listen_socket.into_raw_fd();
        let buffer_pool = BufferPool::leak_new(BUFFER_POOL_CAPACITY);

        let io = IoThread {
            thread_id: i as u16,
            listen_fd,
            producer: producer.clone(),
            response_poller,
            eventfd: efd,
            buffer_pool,
        };

        let handle = thread::Builder::new()
            .name(format!("io-{}", i))
            .spawn(move || io.run())
            .expect("failed to spawn IO thread");
        io_handles.push(handle);
    }

    // Drop our copy of the producer so only IO threads hold refs
    drop(producer);

    eprintln!("disrust: ready");

    for h in io_handles {
        let _ = h.join();
    }
    let _ = batch_handle.join();
}
