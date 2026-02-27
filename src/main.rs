mod batch_processor;
mod buffer_pool;
mod config;
mod constants;
mod io_thread;
mod metrics;
mod protocol;
mod request_flow;
mod response_flow;
mod response_queue;
mod ring_types;

use std::os::unix::io::{IntoRawFd, RawFd};
use std::thread;

use clap::Parser;
use disruptor::{BusySpin, build_single_producer};
use socket2::{Domain, Protocol, Socket, Type};

#[derive(Parser)]
#[command(about = "High-performance io_uring inference server")]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value_t = 9900)]
    port: u16,
}

use batch_processor::BatchProcessor;
use buffer_pool::{BufferPool, set_factory_pool};
use config::{
    BUFFER_POOL_CAPACITY, DISRUPTOR_SIZE, MAX_IO_THREADS, RESPONSE_QUEUE_SIZE, RESULT_POOL_CAPACITY,
};
use io_thread::IoThread;
use response_queue::build_response_channel;
use ring_types::InferenceEvent;

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

fn create_eventfd() -> std::io::Result<RawFd> {
    let fd = unsafe { libc::eventfd(0, libc::EFD_NONBLOCK) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(fd)
}

fn main() {
    metrics::spawn_reporter();
    let args = Args::parse();
    let port = args.port;

    eprintln!("disrust: 1 IO thread, port {}", port);
    eprintln!(
        "  buffer pool: {} MB ({} f32s)",
        BUFFER_POOL_CAPACITY * 4 / 1_000_000,
        BUFFER_POOL_CAPACITY
    );

    // Factory needs a pool for empty slices created during ring initialization.
    let factory_pool = BufferPool::new_boxed(1);
    set_factory_pool(factory_pool);

    // Build the request disruptor (SPSC: one IO thread → batch processor).
    // To support multiple IO threads, switch to build_multi_producer, change
    // IoThread::producer to MultiProducer (Clone), and BatchProcessor::poller
    // to EventPoller<InferenceEvent, MultiProducerBarrier>.
    let builder = build_single_producer(DISRUPTOR_SIZE, InferenceEvent::factory, BusySpin);
    let (request_poller, builder) = builder.event_poller();
    let producer = builder.build();

    // Build response channel for the IO thread
    let efd = create_eventfd().expect("failed to create eventfd");
    let (resp_prod, resp_poll) = build_response_channel(RESPONSE_QUEUE_SIZE, efd);

    // Create result buffer pool (for large responses >INLINE_RESULT_CAPACITY vectors)
    let result_pool = BufferPool::leak_new(RESULT_POOL_CAPACITY);

    eprintln!(
        "  result pool: {} KB ({} f32s)",
        RESULT_POOL_CAPACITY * 4 / 1_000,
        RESULT_POOL_CAPACITY
    );

    // Spawn batch processor thread
    let batch = BatchProcessor {
        poller: request_poller,
        response_producers: vec![resp_prod],
        result_pools: vec![result_pool],
    };
    assert!(
        batch.response_producers.len() <= MAX_IO_THREADS,
        "io_thread_id is u8; max {} IO threads",
        MAX_IO_THREADS
    );
    let batch_handle = thread::Builder::new()
        .name("batch-processor".into())
        .spawn(move || batch.run())
        .expect("failed to spawn batch processor");

    // Spawn IO thread
    let listen_socket = create_listener(port);
    let io = IoThread {
        thread_id: 0,
        listen_fd: listen_socket.into_raw_fd(),
        producer,
        response_poller: resp_poll,
        eventfd: efd,
        buffer_pool: BufferPool::leak_new(BUFFER_POOL_CAPACITY),
    };
    let io_handle = thread::Builder::new()
        .name("io-0".into())
        .spawn(move || io.run())
        .expect("failed to spawn IO thread");

    eprintln!("disrust: ready");

    let _ = io_handle.join();
    let _ = batch_handle.join();
}
