use std::os::unix::io::IntoRawFd;
use std::sync::Arc;
use std::thread;

use clap::Args;
use disruptor::{BusySpin, build_single_producer};
use ort::init_from;
use socket2::{Domain, Protocol, Socket, Type};

use crate::affinity;
use crate::buffer_pool::{BufferPool, set_factory_pool};
use crate::config::{
    BATCH_QUEUE_CAPACITY, DEFAULT_BATCH_COALESCE_US, GPU_BUFFER_POOL_BYTES, GPU_DISRUPTOR_SIZE,
    MAX_SESSION_BATCH_SIZE, SESSION_POOL_SIZE,
};
use crate::metrics;
use crate::pipeline::batch_queue::BatchQueue;
use crate::pipeline::completion::CompletionConsumer;
use crate::pipeline::connection_registry::ConnectionRegistry;
use crate::pipeline::ready_queue::ReadyQueue;
use crate::pipeline::session::InferenceSession;
use crate::pipeline::submission::SubmissionConsumer;
use crate::pipeline::writer::WriterConsumer;
use crate::pipeline::{make_pool, verify_ort_dylib_present};
use crate::ring_types::InferenceEvent;

mod ingress;

use ingress::IngressThread;

#[derive(Args, Clone)]
pub struct ServeArgs {
    /// Port to listen on
    #[arg(short, long, default_value_t = 9900)]
    pub port: u16,

    /// Path to ONNX model file
    #[arg(short, long)]
    pub model: String,

    /// Runtime cap on ring slots per GPU submission.
    #[arg(long, default_value_t = MAX_SESSION_BATCH_SIZE)]
    pub max_batch_slots: usize,

    /// Coalescing window for a partial batch once a session is available, in microseconds.
    #[arg(long, default_value_t = DEFAULT_BATCH_COALESCE_US)]
    pub batch_coalesce_us: u64,

    /// Metrics reporting interval in seconds.
    #[arg(long, default_value_t = 10)]
    pub metrics_interval_secs: u64,

    /// Pin the metrics reporter thread to a specific CPU id.
    #[arg(long)]
    pub metrics_cpu: Option<usize>,

    /// Pin the submission thread to a specific CPU id.
    #[arg(long)]
    pub submission_cpu: Option<usize>,

    /// Pin the completion thread to a specific CPU id.
    #[arg(long)]
    pub completion_cpu: Option<usize>,

    /// Pin the ingress io thread to a specific CPU id.
    #[arg(long)]
    pub io_cpu: Option<usize>,

    /// Pin the writer thread to a specific CPU id.
    #[arg(long)]
    pub writer_cpu: Option<usize>,
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

pub fn run(args: ServeArgs) {
    if args.metrics_interval_secs == 0 {
        eprintln!("disrust: --metrics-interval-secs must be > 0");
        std::process::exit(1);
    }

    metrics::spawn_reporter(args.metrics_interval_secs, args.metrics_cpu);
    let port = args.port;
    let max_batch_slots = args.max_batch_slots;
    let batch_coalesce = std::time::Duration::from_micros(args.batch_coalesce_us);

    if max_batch_slots == 0 || max_batch_slots > MAX_SESSION_BATCH_SIZE {
        eprintln!(
            "disrust: --max-batch-slots must be in 1..={}",
            MAX_SESSION_BATCH_SIZE
        );
        std::process::exit(1);
    }

    eprintln!("disrust: starting on port {}", port);
    eprintln!(
        "disrust: max_batch_slots={} (compile-time max={})",
        max_batch_slots, MAX_SESSION_BATCH_SIZE
    );
    eprintln!("disrust: batch_coalesce_us={}", args.batch_coalesce_us);
    if let Some(cpu) = args.metrics_cpu {
        eprintln!("disrust: metrics_cpu={cpu}");
    }
    if let Some(cpu) = args.submission_cpu {
        eprintln!("disrust: submission_cpu={cpu}");
    }
    if let Some(cpu) = args.completion_cpu {
        eprintln!("disrust: completion_cpu={cpu}");
    }
    if let Some(cpu) = args.io_cpu {
        eprintln!("disrust: io_cpu={cpu}");
    }
    if let Some(cpu) = args.writer_cpu {
        eprintln!("disrust: writer_cpu={cpu}");
    }
    let ort_dylib = verify_ort_dylib_present().unwrap_or_else(|e| {
        eprintln!("disrust preflight failed: {e}");
        std::process::exit(1);
    });
    eprintln!("disrust: using ORT dylib {}", ort_dylib.display());
    eprintln!("disrust: initializing ONNX Runtime");
    let committed = init_from(&ort_dylib)
        .unwrap_or_else(|e| {
            eprintln!("disrust preflight failed: ort::init_from failed: {e}");
            std::process::exit(1);
        })
        .commit();
    eprintln!("disrust: ONNX Runtime initialized (fresh={committed})");

    #[cfg(feature = "cuda")]
    {
        crate::cuda::preflight::verify_cuda_startup().unwrap_or_else(|e| {
            eprintln!("disrust preflight failed: {e}");
            std::process::exit(1);
        });
        eprintln!("disrust: CUDA driver preflight ok");
    }

    set_factory_pool(BufferPool::new_boxed(1));

    let model_bytes = std::fs::read(&args.model).unwrap_or_else(|e| {
        eprintln!("Failed to read model '{}': {}", args.model, e);
        std::process::exit(1);
    });

    let sessions: Vec<InferenceSession> = (0..SESSION_POOL_SIZE)
        .map(|i| {
            eprintln!("disrust: loading session {}/{}", i + 1, SESSION_POOL_SIZE);
            InferenceSession::new(&model_bytes)
        })
        .collect();

    let pool = make_pool();
    let allocator = pool.allocator();

    eprintln!(
        "disrust: buffer pool {} MB",
        GPU_BUFFER_POOL_BYTES / 1_000_000,
    );

    let builder = build_single_producer(GPU_DISRUPTOR_SIZE, InferenceEvent::factory, BusySpin);
    let (submission_poller, builder) = builder.event_poller();
    let (completion_poller, builder) = builder.and_then().event_poller();
    let producer = builder.build();

    let batch_queue = Arc::new(BatchQueue::new(BATCH_QUEUE_CAPACITY));
    let ready_queue = Arc::new(ReadyQueue::new(crate::config::SLAB_CAPACITY * 2));
    let registry = Arc::new(ConnectionRegistry::default());

    let sub_consumer = SubmissionConsumer::new(
        submission_poller,
        sessions,
        Arc::clone(&batch_queue),
        max_batch_slots,
        batch_coalesce,
    );
    let submission_cpu = args.submission_cpu;
    let sub_handle = thread::Builder::new()
        .name("submission".into())
        .spawn(move || {
            if let Some(cpu) = submission_cpu {
                affinity::pin_current_thread(cpu, "submission").unwrap_or_else(|e| panic!("{e}"));
            }
            sub_consumer.run()
        })
        .expect("failed to spawn SubmissionConsumer");

    let comp_consumer = CompletionConsumer::new(
        completion_poller,
        Arc::clone(&batch_queue),
        Arc::clone(&ready_queue),
        Arc::clone(&registry),
        max_batch_slots,
    );
    let completion_cpu = args.completion_cpu;
    let comp_handle = thread::Builder::new()
        .name("completion".into())
        .spawn(move || {
            if let Some(cpu) = completion_cpu {
                affinity::pin_current_thread(cpu, "completion").unwrap_or_else(|e| panic!("{e}"));
            }
            comp_consumer.run()
        })
        .expect("failed to spawn CompletionConsumer");

    let writer_consumer = WriterConsumer::new(Arc::clone(&ready_queue), Arc::clone(&registry))
        .expect("failed to create WriterConsumer io_uring");
    let writer_cpu = args.writer_cpu;
    let writer_handle = thread::Builder::new()
        .name("writer".into())
        .spawn(move || {
            if let Some(cpu) = writer_cpu {
                affinity::pin_current_thread(cpu, "writer").unwrap_or_else(|e| panic!("{e}"));
            }
            writer_consumer.run()
        })
        .expect("failed to spawn WriterConsumer");

    let listen_socket = create_listener(port);
    let ingress = IngressThread::new(
        0,
        listen_socket.into_raw_fd(),
        producer,
        allocator,
        Arc::clone(&registry),
    );

    eprintln!("disrust: ready");

    let io_cpu = args.io_cpu;
    let io_handle = thread::Builder::new()
        .name("io-0".into())
        .spawn(move || {
            if let Some(cpu) = io_cpu {
                affinity::pin_current_thread(cpu, "io-0").unwrap_or_else(|e| panic!("{e}"));
            }
            ingress.run()
        })
        .expect("failed to spawn IO thread");

    let _ = io_handle.join();
    let _ = sub_handle.join();
    let _ = comp_handle.join();
    let _ = writer_handle.join();
}
