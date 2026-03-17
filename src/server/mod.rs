use std::os::unix::io::IntoRawFd;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;

use clap::Args;
use disruptor::{BusySpin, build_multi_producer};
use ort::init_from;
use socket2::{Domain, Protocol, Socket, Type};

use crate::affinity;
use crate::buffer_pool::{BufferPool, set_factory_pool};
use crate::config::{
    DEFAULT_BATCH_COALESCE_US, GPU_BUFFER_POOL_BYTES, GPU_DISRUPTOR_SIZE, MAX_IO_THREADS,
    MAX_SESSION_BATCH_SIZE, SESSION_POOL_SIZE, SLAB_CAPACITY,
};
use crate::metrics;
use crate::pipeline::connection_registry::ConnectionRegistry;
use crate::pipeline::inference::InferenceConsumer;
use crate::pipeline::response_queue::ResponseQueue;
use crate::pipeline::session::InferenceSession;
use crate::pipeline::{make_pool, verify_ort_dylib_present};
use crate::ring_types::InferenceEvent;

mod ingress;

pub use ingress::IngressThread;

enum WorkerExit {
    Returned(&'static str),
    Panicked(&'static str, String),
}

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

    /// Number of ingress IO threads to run via SO_REUSEPORT sharding.
    #[arg(long, default_value_t = 1)]
    pub io_threads: u8,
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

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(msg) => *msg,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(msg) => (*msg).to_string(),
            Err(_) => "non-string panic payload".to_string(),
        },
    }
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
    let io_threads = args.io_threads as usize;

    if max_batch_slots == 0 || max_batch_slots > MAX_SESSION_BATCH_SIZE {
        eprintln!(
            "disrust: --max-batch-slots must be in 1..={}",
            MAX_SESSION_BATCH_SIZE
        );
        std::process::exit(1);
    }
    if io_threads == 0 || io_threads > MAX_IO_THREADS {
        eprintln!("disrust: --io-threads must be in 1..={MAX_IO_THREADS}");
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
        eprintln!("disrust: io_cpu_base={cpu}");
    }
    eprintln!("disrust: io_threads={io_threads}");
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

    let builder = build_multi_producer(GPU_DISRUPTOR_SIZE, InferenceEvent::factory, BusySpin);
    let (submission_poller, builder) = builder.event_poller();
    let (completion_poller, builder) = builder.and_then().event_poller();
    let producer = builder.build();

    let response_queues = (0..io_threads)
        .map(|_| Arc::new(ResponseQueue::new(SLAB_CAPACITY * 2)))
        .collect::<Vec<_>>();
    let publish_gate = Arc::new(std::sync::Mutex::new(()));
    let registry = Arc::new(ConnectionRegistry::new(io_threads, SLAB_CAPACITY));
    let (worker_exit_tx, worker_exit_rx) = mpsc::channel::<WorkerExit>();

    if let (Some(submission_cpu), Some(completion_cpu)) = (args.submission_cpu, args.completion_cpu)
        && submission_cpu != completion_cpu
    {
        eprintln!(
            "disrust: --submission-cpu and --completion-cpu must match when submission and completion share one inference thread"
        );
        std::process::exit(1);
    }

    let inference_consumer = InferenceConsumer::new(
        submission_poller,
        completion_poller,
        sessions,
        response_queues.clone(),
        Arc::clone(&registry),
        max_batch_slots,
        batch_coalesce,
    );
    let inference_cpu = args.submission_cpu.or(args.completion_cpu);
    thread::Builder::new()
        .name("inference".into())
        .spawn({
            let worker_exit_tx = worker_exit_tx.clone();
            move || {
                let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
                    if let Some(cpu) = inference_cpu {
                        affinity::pin_current_thread(cpu, "inference")
                            .unwrap_or_else(|e| panic!("{e}"));
                    }
                    inference_consumer.run()
                }));
                let _ = match outcome {
                    Ok(()) => worker_exit_tx.send(WorkerExit::Returned("inference")),
                    Err(payload) => worker_exit_tx
                        .send(WorkerExit::Panicked("inference", panic_message(payload))),
                };
            }
        })
        .expect("failed to spawn inference consumer");

    eprintln!("disrust: ready");

    for (thread_id, response_queue) in response_queues.iter().enumerate() {
        let listen_socket = create_listener(port);
        let ingress = IngressThread::new(
            thread_id as u8,
            listen_socket.into_raw_fd(),
            producer.clone(),
            allocator,
            Arc::clone(response_queue),
            Arc::clone(&publish_gate),
            Arc::clone(&registry),
        );
        let io_cpu = args.io_cpu.map(|base| base + thread_id);
        let thread_name = format!("io-{thread_id}");
        thread::Builder::new()
            .name(thread_name.clone())
            .spawn({
                let worker_exit_tx = worker_exit_tx.clone();
                move || {
                    let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
                        if let Some(cpu) = io_cpu {
                            affinity::pin_current_thread(cpu, &thread_name)
                                .unwrap_or_else(|e| panic!("{e}"));
                        }
                        ingress.run()
                    }));
                    let _ = match outcome {
                        Ok(()) => worker_exit_tx.send(WorkerExit::Returned("ingress")),
                        Err(payload) => worker_exit_tx
                            .send(WorkerExit::Panicked("ingress", panic_message(payload))),
                    };
                }
            })
            .expect("failed to spawn IO thread");
    }

    drop(worker_exit_tx);
    match worker_exit_rx
        .recv()
        .expect("worker exit channel closed unexpectedly")
    {
        WorkerExit::Returned(name) => {
            panic!("disrust: worker thread '{name}' exited unexpectedly");
        }
        WorkerExit::Panicked(name, message) => {
            panic!("disrust: worker thread '{name}' panicked: {message}");
        }
    }
}
