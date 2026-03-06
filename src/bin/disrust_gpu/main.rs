//! GPU inference server binary.
//!
//! Pipeline topology:
//!
//! ```
//! IO Thread(s)
//!     │ publishes InferenceEvent (includes fd)
//!     ▼
//! [request ring]
//!     │
//!     ▼
//! SubmissionConsumer   (accumulates batch, submits to ORT/GPU)
//!     │ pushes (end_sequence, session_idx, OrtRunHandle) to batch queue
//!     ▼
//! CompletionConsumer   (awaits GPU, serializes results, OP_WRITE via own ring)
//!     │
//!     ▼
//! client fds
//! ```
//!
//! See REWORK.md for the full design.

mod io_thread_gpu;

use std::os::unix::io::IntoRawFd;
use std::thread;

use clap::Parser;
use disruptor::{BusySpin, build_single_producer};
use socket2::{Domain, Protocol, Socket, Type};

use disrust::buffer_pool::{BufferPool, set_factory_pool};
use disrust::config::{BUFFER_POOL_CAPACITY, DISRUPTOR_SIZE};
use disrust::metrics;
use disrust::ring_types::InferenceEvent;

use io_thread_gpu::IoThreadGpu;

#[derive(Parser)]
#[command(about = "GPU-backed io_uring inference server")]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value_t = 9900)]
    port: u16,
}

fn create_listener(port: u16) -> Socket {
    let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP))
        .expect("failed to create socket");
    socket.set_reuse_address(true).unwrap();
    socket.set_reuse_port(true).expect("SO_REUSEPORT failed");
    socket.set_nonblocking(true).unwrap();
    socket.set_nodelay(true).unwrap();

    let addr = std::net::SocketAddrV4::new(std::net::Ipv4Addr::UNSPECIFIED, port);
    socket.bind(&addr.into()).expect("failed to bind");
    socket.listen(1024).expect("failed to listen");
    socket
}

fn main() {
    metrics::spawn_reporter();
    let args = Args::parse();
    let port = args.port;

    eprintln!("disrust-gpu: 1 IO thread, port {}", port);

    // Factory pool for empty PoolSlice values during ring initialisation.
    let factory_pool = BufferPool::new_boxed(1);
    set_factory_pool(factory_pool);

    // TODO: allocate buffer pool backing memory with cudaMallocHost so the IO
    // thread's feature data is pinned and device-mappable for zero-copy H2D.
    // For now, use a standard heap-backed pool.
    let buffer_pool = BufferPool::leak_new(BUFFER_POOL_CAPACITY);

    // Request ring: IO thread -> SubmissionConsumer -> CompletionConsumer.
    let builder = build_single_producer(DISRUPTOR_SIZE, InferenceEvent::factory, BusySpin);
    let (_request_poller, builder) = builder.event_poller();
    let producer = builder.build();

    // TODO: wire SubmissionConsumer and CompletionConsumer.
    // The _request_poller above is a placeholder; the actual consumers will be
    // chained disruptor consumers gated on each other's sequences.

    // Spawn IO thread.
    let listen_socket = create_listener(port);
    let io = IoThreadGpu::new(0, listen_socket.into_raw_fd(), producer, buffer_pool);
    let io_handle = thread::Builder::new()
        .name("io-gpu-0".into())
        .spawn(move || io.run())
        .expect("failed to spawn IO thread");

    eprintln!("disrust-gpu: ready");

    let _ = io_handle.join();
}
