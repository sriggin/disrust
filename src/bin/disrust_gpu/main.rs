//! GPU inference server: accepts TCP connections, publishes to request ring,
//! runs ONNX inference on GPU via ORT CUDA EP, writes responses directly.

mod io_thread_gpu;

use std::os::unix::io::IntoRawFd;
use std::sync::Arc;
use std::thread;

use clap::Parser;
use disruptor::{BusySpin, build_single_producer};
use socket2::{Domain, Protocol, Socket, Type};

use disrust::batch_queue::BatchQueue;
use disrust::buffer_pool::{BufferPool, set_factory_pool};
use disrust::config::{
    BATCH_QUEUE_CAPACITY, GPU_BUFFER_POOL_BYTES, GPU_BUFFER_POOL_CAPACITY, GPU_DISRUPTOR_SIZE,
    MAX_SESSION_BATCH_SIZE, SESSION_POOL_SIZE,
};
use disrust::gpu::completion::CompletionConsumer;
use disrust::gpu::preflight::{verify_cuda_startup, verify_ort_dylib_present};
use disrust::gpu::session::GpuSession;
use disrust::gpu::submission::SubmissionConsumer;
use disrust::ring_types::InferenceEvent;

use io_thread_gpu::IoThreadGpu;

#[derive(Parser)]
#[command(about = "High-performance GPU io_uring inference server")]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value_t = 9900)]
    port: u16,

    /// Path to ONNX model file
    #[arg(short, long, default_value = "model.onnx")]
    model: String,

    /// Runtime cap on ring slots per GPU submission.
    #[arg(long, default_value_t = MAX_SESSION_BATCH_SIZE)]
    max_batch_slots: usize,
}

fn create_listener(port: u16) -> Socket {
    let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP))
        .expect("socket creation failed");
    socket.set_reuse_address(true).unwrap();
    socket.set_reuse_port(true).expect("SO_REUSEPORT failed");
    socket.set_nonblocking(true).unwrap();
    socket.set_nodelay(true).unwrap();
    let addr = std::net::SocketAddrV4::new(std::net::Ipv4Addr::UNSPECIFIED, port);
    socket.bind(&addr.into()).expect("bind failed");
    socket.listen(1024).expect("listen failed");
    socket
}

fn main() {
    let args = Args::parse();
    let port = args.port;
    let max_batch_slots = args.max_batch_slots;

    if max_batch_slots == 0 || max_batch_slots > MAX_SESSION_BATCH_SIZE {
        eprintln!(
            "disrust-gpu: --max-batch-slots must be in 1..={}",
            MAX_SESSION_BATCH_SIZE
        );
        std::process::exit(1);
    }

    eprintln!("disrust-gpu: starting on port {}", port);
    eprintln!(
        "disrust-gpu: max_batch_slots={} (compile-time max={})",
        max_batch_slots, MAX_SESSION_BATCH_SIZE
    );
    let ort_dylib = verify_ort_dylib_present().unwrap_or_else(|e| {
        eprintln!("disrust-gpu preflight failed: {e}");
        std::process::exit(1);
    });
    eprintln!("disrust-gpu: using ORT dylib {}", ort_dylib.display());
    verify_cuda_startup().unwrap_or_else(|e| {
        eprintln!("disrust-gpu preflight failed: {e}");
        std::process::exit(1);
    });
    eprintln!("disrust-gpu: CUDA driver preflight ok");

    // 1. Factory pool for empty PoolSlices during disruptor pre-allocation.
    set_factory_pool(BufferPool::new_boxed(1));

    // 2. Load ONNX model.
    let model_bytes = std::fs::read(&args.model).unwrap_or_else(|e| {
        eprintln!("Failed to read model '{}': {}", args.model, e);
        std::process::exit(1);
    });

    // 3. Construct session pool first so ORT/CUDA creates a context before we allocate
    // pinned host memory for the request BufferPool.
    let sessions: Vec<GpuSession> = (0..SESSION_POOL_SIZE)
        .map(|i| {
            eprintln!(
                "disrust-gpu: loading session {}/{}",
                i + 1,
                SESSION_POOL_SIZE
            );
            GpuSession::new(&model_bytes)
        })
        .collect();

    // 4. Allocate pinned host memory for the buffer pool.
    let host_ptr = unsafe {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let status = cudarc::driver::sys::cuMemAllocHost_v2(&mut ptr, GPU_BUFFER_POOL_BYTES);
        if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            eprintln!("cuMemAllocHost failed: {:?}", status);
            std::process::exit(1);
        }
        // Touch all pages to avoid page-fault latency during operation.
        std::ptr::write_bytes(ptr as *mut u8, 0u8, GPU_BUFFER_POOL_BYTES);

        ptr as *mut f32
    };

    // 5. Construct the BufferPool over the pinned memory (capacity in f32 units).
    let pool: &'static BufferPool =
        Box::leak(unsafe { BufferPool::from_raw_ptr(host_ptr, GPU_BUFFER_POOL_CAPACITY) });
    let allocator = pool.allocator();

    eprintln!(
        "disrust-gpu: pinned pool {} MB",
        GPU_BUFFER_POOL_BYTES / 1_000_000,
    );

    // 6. Build request disruptor: two-consumer chain.
    //    submission_poller: EventPoller<InferenceEvent, SingleProducerBarrier>
    //    completion_poller: EventPoller<InferenceEvent, SingleConsumerBarrier>
    //      (gated on submission_poller's cursor via .and_then())
    //    producer: SingleProducer<InferenceEvent, SingleConsumerBarrier>
    //      (gated on completion_poller — slowest consumer drives backpressure)
    let builder = build_single_producer(GPU_DISRUPTOR_SIZE, InferenceEvent::factory, BusySpin);
    let (submission_poller, builder) = builder.event_poller();
    let (completion_poller, builder) = builder.and_then().event_poller();
    let producer = builder.build();

    // 7. Shared batch queue.
    let batch_queue = Arc::new(BatchQueue::new(BATCH_QUEUE_CAPACITY));

    // 8. Spawn SubmissionConsumer (owns sessions).
    let sub_consumer = SubmissionConsumer::new(
        submission_poller,
        sessions,
        Arc::clone(&batch_queue),
        max_batch_slots,
    );
    let sub_handle = thread::Builder::new()
        .name("gpu-submission".into())
        .spawn(move || sub_consumer.run())
        .expect("failed to spawn SubmissionConsumer");

    // 9. Spawn CompletionConsumer (has output pointers only).
    let comp_consumer =
        CompletionConsumer::new(completion_poller, Arc::clone(&batch_queue), max_batch_slots)
            .expect("failed to create CompletionConsumer io_uring");
    let comp_handle = thread::Builder::new()
        .name("gpu-completion".into())
        .spawn(move || comp_consumer.run())
        .expect("failed to spawn CompletionConsumer");

    // 10. Spawn IO ingress thread.
    let listen_socket = create_listener(port);
    let io = IoThreadGpu::new(0, listen_socket.into_raw_fd(), producer, allocator);

    eprintln!("disrust-gpu: ready");

    let io_handle = thread::Builder::new()
        .name("io-gpu-0".into())
        .spawn(move || io.run())
        .expect("failed to spawn IO thread");

    let _ = io_handle.join();
    let _ = sub_handle.join();
    let _ = comp_handle.join();
}
