use std::collections::VecDeque;
use std::io;
use std::net::TcpStream;
use std::os::fd::{IntoRawFd, RawFd};
use std::sync::Arc;
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread;
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand};
use io_uring::{opcode, squeue::Entry, types::Fd};
use slab::Slab;

use disrust::constants::FEATURE_DIM;
use disrust::protocol::{RESPONSE_HEADER_BYTES, request_size, response_size};
use disrust::timer::{TimerMetric, TimerRecorder, TimerSnapshot};

const OP_READ: u64 = 1;
const OP_WRITE: u64 = 2;
const READ_BUF_SIZE: usize = 64 * 1024;

#[derive(Parser)]
#[command(about = "Test client for disrust inference server")]
struct Cli {
    /// Server port
    #[arg(short, long, default_value_t = 9900)]
    port: u16,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Send a few requests and verify results (default)
    Smoke,
    /// Send pipelined requests and verify all results
    Pipeline(PipelineArgs),
    /// Benchmark throughput with concurrent pipelined connections
    Bench(BenchArgs),
    /// Sustained load with per-request latency measurement
    Sustain(SustainArgs),
}

#[derive(Args, Clone)]
struct PipelineArgs {
    /// Number of concurrent connections
    #[arg(short, long, default_value_t = 1)]
    connections: usize,
    /// In-flight requests per connection
    #[arg(short, long, default_value_t = 1000)]
    window: usize,
    /// Vectors per request
    #[arg(short = 'v', long, default_value_t = 2)]
    vectors: u32,
    /// Requests per connection
    #[arg(short, long, default_value_t = 1000)]
    requests: usize,
}

#[derive(Args, Clone)]
struct BenchArgs {
    /// Number of concurrent connections
    #[arg(short, long, default_value_t = 4)]
    connections: usize,
    /// In-flight requests per connection
    #[arg(short, long, default_value_t = 256)]
    window: usize,
    /// Vectors per request
    #[arg(short = 'v', long, default_value_t = 1)]
    vectors: u32,
    /// Requests per connection
    #[arg(short, long, default_value_t = 100_000)]
    requests: usize,
}

#[derive(Args, Clone)]
struct SustainArgs {
    /// Number of concurrent connections
    #[arg(short, long, default_value_t = 4)]
    connections: usize,
    /// In-flight requests per connection
    #[arg(short, long, default_value_t = 64)]
    window: usize,
    /// Vectors per request
    #[arg(short = 'v', long, default_value_t = 1)]
    vectors: u32,
    /// Warmup duration in seconds (discarded from report)
    #[arg(short = 'W', long, default_value_t = 3)]
    warmup: u64,
    /// Measurement duration in seconds
    #[arg(short, long, default_value_t = 10)]
    duration: u64,
}

#[derive(Clone)]
struct RequestTemplate {
    num_vectors: u32,
    request_bytes: Arc<[u8]>,
    expected: Arc<[f32]>,
}

impl RequestTemplate {
    fn new(num_vectors: u32) -> Self {
        let mut buf = Vec::with_capacity(request_size(num_vectors as usize));
        buf.extend_from_slice(&num_vectors.to_le_bytes());

        let mut expected_sums = Vec::with_capacity(num_vectors as usize);
        for v in 0..num_vectors as usize {
            let mut sum = 0.0f32;
            for f in 0..FEATURE_DIM {
                let val = (v * FEATURE_DIM + f) as f32 * 0.01;
                buf.extend_from_slice(&val.to_le_bytes());
                sum += val;
            }
            expected_sums.push(sum);
        }

        Self {
            num_vectors,
            request_bytes: Arc::from(buf),
            expected: Arc::from(expected_sums),
        }
    }
}

enum StopMode {
    FixedCount { requests_per_connection: u64 },
    Duration { warmup: Duration, measure: Duration },
}

struct Scenario {
    name: &'static str,
    connections: usize,
    window: usize,
    templates: Arc<[RequestTemplate]>,
    verify: bool,
    collect_latency: bool,
    stop_mode: StopMode,
}

impl Scenario {
    fn pipeline(args: PipelineArgs) -> Self {
        Self {
            name: "pipeline",
            connections: args.connections,
            window: args.window,
            templates: Arc::from([RequestTemplate::new(args.vectors)]),
            verify: true,
            collect_latency: false,
            stop_mode: StopMode::FixedCount {
                requests_per_connection: args.requests as u64,
            },
        }
    }

    fn bench(args: BenchArgs) -> Self {
        Self {
            name: "bench",
            connections: args.connections,
            window: args.window,
            templates: Arc::from([RequestTemplate::new(args.vectors)]),
            verify: false,
            collect_latency: false,
            stop_mode: StopMode::FixedCount {
                requests_per_connection: args.requests as u64,
            },
        }
    }

    fn sustain(args: SustainArgs) -> Self {
        Self {
            name: "sustain",
            connections: args.connections,
            window: args.window,
            templates: Arc::from([RequestTemplate::new(args.vectors)]),
            verify: false,
            collect_latency: true,
            stop_mode: StopMode::Duration {
                warmup: Duration::from_secs(args.warmup),
                measure: Duration::from_secs(args.duration),
            },
        }
    }
}

#[derive(Clone, Copy)]
struct PendingRequest {
    template_idx: usize,
    submitted_at: Instant,
}

struct Connection {
    fd: RawFd,
    read_buf: Box<[u8; READ_BUF_SIZE]>,
    read_len: usize,
    read_inflight: bool,
    pending: VecDeque<PendingRequest>,
    submitted_total: u64,
    completed_total: u64,
}

impl Connection {
    fn new(fd: RawFd, window: usize) -> Self {
        Self {
            fd,
            read_buf: Box::new([0u8; READ_BUF_SIZE]),
            read_len: 0,
            read_inflight: false,
            pending: VecDeque::with_capacity(window.max(1)),
            submitted_total: 0,
            completed_total: 0,
        }
    }

    fn pending_count(&self) -> usize {
        self.pending.len()
    }

    fn can_issue_more(&self, scenario: &Scenario, now: Instant, run: &RunState) -> bool {
        if self.pending_count() >= scenario.window {
            return false;
        }

        match scenario.stop_mode {
            StopMode::FixedCount {
                requests_per_connection,
            } => self.submitted_total < requests_per_connection,
            StopMode::Duration { .. } => now < run.measure_end,
        }
    }

    fn is_finished(&self, scenario: &Scenario, now: Instant, run: &RunState) -> bool {
        match scenario.stop_mode {
            StopMode::FixedCount {
                requests_per_connection,
            } => self.completed_total >= requests_per_connection,
            StopMode::Duration { .. } => now >= run.measure_end && self.pending.is_empty(),
        }
    }

    fn read_tail(&mut self) -> (*mut u8, u32) {
        (
            unsafe { self.read_buf.as_mut_ptr().add(self.read_len) },
            (READ_BUF_SIZE - self.read_len) as u32,
        )
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.fd);
        }
    }
}

struct IoUring {
    inner: io_uring::IoUring,
    outstanding: usize,
}

impl IoUring {
    fn new(entries: u32) -> io::Result<Self> {
        Ok(Self {
            inner: io_uring::IoUring::new(entries)?,
            outstanding: 0,
        })
    }

    fn push(&mut self, sqe: &Entry) {
        loop {
            match unsafe { self.inner.submission().push(sqe) } {
                Ok(()) => {
                    self.outstanding += 1;
                    return;
                }
                Err(_) => {
                    self.inner.submit().expect("SQ flush failed");
                }
            }
        }
    }

    fn wait(&mut self, min_complete: usize) {
        self.inner
            .submit_and_wait(min_complete)
            .expect("submit_and_wait failed");
    }

    fn drain_cqes_into(&mut self, buf: &mut Vec<(u64, i32)>) {
        for cqe in self.inner.completion() {
            self.outstanding = self.outstanding.saturating_sub(1);
            buf.push((cqe.user_data(), cqe.result()));
        }
    }
}

fn encode_user_data(op: u64, key: u32) -> u64 {
    (op << 32) | key as u64
}

fn decode_user_data(user_data: u64) -> (u64, u32) {
    (user_data >> 32, user_data as u32)
}

struct RunState {
    start: Instant,
    warmup_end: Instant,
    measure_end: Instant,
    measurement_started: bool,
}

impl RunState {
    fn new(stop_mode: &StopMode) -> Self {
        let start = Instant::now();
        let (warmup_end, measure_end) = match stop_mode {
            StopMode::FixedCount { .. } => (start, start),
            StopMode::Duration { warmup, measure } => {
                let warmup_end = start + *warmup;
                (warmup_end, warmup_end + *measure)
            }
        };

        Self {
            start,
            warmup_end,
            measure_end,
            measurement_started: false,
        }
    }

    fn is_measuring(&self, scenario: &Scenario, now: Instant) -> bool {
        match scenario.stop_mode {
            StopMode::FixedCount { .. } => true,
            StopMode::Duration { .. } => now >= self.warmup_end && now < self.measure_end,
        }
    }
}

#[derive(Default)]
struct SummaryStats {
    measured_completions: u64,
}

impl SummaryStats {
    fn record_completion(&mut self, scenario: &Scenario, run: &RunState, now: Instant) {
        if !run.is_measuring(scenario, now) {
            return;
        }

        self.measured_completions += 1;
    }
}

enum ReportEvent {
    StartMeasurement(Instant),
    Finish {
        end: Instant,
        measured_completions: u64,
    },
}

struct Reporter {
    tx: Sender<ReportEvent>,
    handle: thread::JoinHandle<()>,
}

impl Reporter {
    fn spawn(interval_timer: Arc<TimerMetric>) -> Self {
        let (tx, rx) = mpsc::channel();
        let handle = thread::Builder::new()
            .name("client-reporter".into())
            .spawn(move || {
                let mut measurement_start: Option<Instant> = None;
                let mut last_interval_start: Option<Instant> = None;
                let mut interval_completions: u64 = 0;
                let mut total_hist = hdrhistogram::Histogram::new_with_bounds(1, 60_000_000_000, 3)
                    .expect("failed to create cumulative client histogram");

                loop {
                    let timeout = match last_interval_start {
                        Some(last) => {
                            let elapsed = last.elapsed();
                            if elapsed >= Duration::from_secs(1) {
                                Duration::ZERO
                            } else {
                                Duration::from_secs(1) - elapsed
                            }
                        }
                        None => Duration::from_secs(3600),
                    };

                    match rx.recv_timeout(timeout) {
                        Ok(ReportEvent::StartMeasurement(start)) => {
                            measurement_start = Some(start);
                            last_interval_start = Some(start);
                            let _ = interval_timer.snapshot_and_reset();
                            total_hist.reset();
                            interval_completions = 0;
                        }
                        Ok(ReportEvent::Finish {
                            end,
                            measured_completions,
                        }) => {
                            if let Some(last) = last_interval_start {
                                let snapshot = interval_timer.snapshot();
                                if let Some(ref snap) = snapshot {
                                    interval_completions = snap.count();
                                    snap.merge_into(&mut total_hist);
                                }
                                print_interval(snapshot, interval_completions, last, end);
                            }
                            print_summary(
                                if total_hist.is_empty() {
                                    None
                                } else {
                                    Some(TimerSnapshot::from_histogram(total_hist))
                                },
                                measurement_start,
                                end,
                                measured_completions,
                            );
                            break;
                        }
                        Err(RecvTimeoutError::Timeout) => {
                            if let Some(last) = last_interval_start {
                                let now = Instant::now();
                                let snapshot = interval_timer.snapshot_and_reset();
                                if let Some(ref snap) = snapshot {
                                    interval_completions = snap.count();
                                    snap.merge_into(&mut total_hist);
                                }
                                print_interval(snapshot, interval_completions, last, now);
                                interval_completions = 0;
                                last_interval_start = Some(now);
                            }
                        }
                        Err(RecvTimeoutError::Disconnected) => break,
                    }
                }
            })
            .expect("failed to spawn reporter thread");

        Self { tx, handle }
    }

    fn start_measurement(&self, now: Instant) {
        let _ = self.tx.send(ReportEvent::StartMeasurement(now));
    }

    fn finish(self, end: Instant, measured_completions: u64) {
        let _ = self.tx.send(ReportEvent::Finish {
            end,
            measured_completions,
        });
        self.handle.join().expect("reporter thread panicked");
    }
}

fn print_interval(snapshot: Option<TimerSnapshot>, completions: u64, start: Instant, end: Instant) {
    let Some(snapshot) = snapshot else {
        return;
    };
    let elapsed = end.duration_since(start);
    let qps = completions as f64 / elapsed.as_secs_f64();
    eprintln!(
        "{:>10.0}  {:>8.1}us  {:>8.1}us  {:>8.1}us  {:>8.1}us  {:>8}",
        qps,
        snapshot.p50_us(),
        snapshot.p95_us(),
        snapshot.p99_us(),
        snapshot.p999_us(),
        snapshot.count(),
    );
}

fn print_summary(
    snapshot: Option<TimerSnapshot>,
    measurement_start: Option<Instant>,
    end: Instant,
    measured_completions: u64,
) {
    let Some(start) = measurement_start else {
        eprintln!("no samples collected");
        return;
    };
    let Some(snapshot) = snapshot else {
        eprintln!("no samples collected");
        return;
    };

    let elapsed = end.duration_since(start);

    eprintln!();
    eprintln!(
        "summary ({:.1}s, {} requests)",
        elapsed.as_secs_f64(),
        measured_completions
    );
    eprintln!(
        "  qps     {:.0}",
        measured_completions as f64 / elapsed.as_secs_f64()
    );
    eprintln!("  p50     {:.1}us", snapshot.p50_us());
    eprintln!("  p95     {:.1}us", snapshot.p95_us());
    eprintln!("  p99     {:.1}us", snapshot.p99_us());
    eprintln!("  p99.9   {:.1}us", snapshot.p999_us());
    eprintln!("  p99.99  {:.1}us", snapshot.p9999_us());
    eprintln!("  max     {:.1}us", snapshot.max_us());
}

fn create_connection(addr: &str) -> io::Result<RawFd> {
    let stream = TcpStream::connect(addr)?;
    stream.set_nodelay(true)?;
    Ok(stream.into_raw_fd())
}

fn submit_read(ring: &mut IoUring, conn: &mut Connection, key: u32) {
    if conn.read_inflight || conn.read_len == READ_BUF_SIZE || conn.pending.is_empty() {
        return;
    }

    let (ptr, len) = conn.read_tail();
    let sqe = opcode::Read::new(Fd(conn.fd), ptr, len)
        .build()
        .user_data(encode_user_data(OP_READ, key));
    ring.push(&sqe);
    conn.read_inflight = true;
}

fn submit_writes(
    ring: &mut IoUring,
    conn: &mut Connection,
    key: u32,
    scenario: &Scenario,
    run: &RunState,
    now: Instant,
) {
    while conn.can_issue_more(scenario, now, run) {
        let template_idx = (conn.submitted_total as usize) % scenario.templates.len();
        let template = &scenario.templates[template_idx];
        let sqe = opcode::Write::new(
            Fd(conn.fd),
            template.request_bytes.as_ptr(),
            template.request_bytes.len() as u32,
        )
        .build()
        .user_data(encode_user_data(OP_WRITE, key));
        ring.push(&sqe);
        conn.pending.push_back(PendingRequest {
            template_idx,
            submitted_at: now,
        });
        conn.submitted_total += 1;
    }

    submit_read(ring, conn, key);
}

fn verify_response(frame: &[u8], template: &RequestTemplate) {
    let got_vectors = frame[0] as u32;
    assert_eq!(
        got_vectors, template.num_vectors,
        "response num_vectors={} does not match expected={} (protocol error or data corruption)",
        got_vectors, template.num_vectors
    );

    let body = &frame[RESPONSE_HEADER_BYTES..];
    assert_eq!(body.len(), template.expected.len() * 4);

    for (i, expected) in template.expected.iter().enumerate() {
        let offset = i * 4;
        let got = f32::from_le_bytes([
            body[offset],
            body[offset + 1],
            body[offset + 2],
            body[offset + 3],
        ]);
        let diff = (got - expected).abs();
        assert!(
            diff < 0.1,
            "vector {}: result {} != expected {}",
            i,
            got,
            expected
        );
    }
}

fn process_read_buffer(
    conn: &mut Connection,
    scenario: &Scenario,
    stats: &mut SummaryStats,
    run: &RunState,
    now: Instant,
    interval_latency_recorder: &mut Option<TimerRecorder>,
) {
    let mut consumed = 0usize;

    while let Some(pending) = conn.pending.front().copied() {
        let template = &scenario.templates[pending.template_idx];
        let expected_len = response_size(template.num_vectors as usize);
        if conn.read_len - consumed < expected_len {
            break;
        }

        let frame = &conn.read_buf[consumed..consumed + expected_len];
        if scenario.verify {
            verify_response(frame, template);
        }

        conn.pending.pop_front();
        conn.completed_total += 1;
        stats.record_completion(scenario, run, now);
        if scenario.collect_latency && run.is_measuring(scenario, now) {
            let latency = now.duration_since(pending.submitted_at);
            if let Some(recorder) = interval_latency_recorder.as_mut() {
                recorder.record_duration(latency);
            }
        }
        consumed += expected_len;
    }

    if consumed > 0 {
        conn.read_buf.copy_within(consumed..conn.read_len, 0);
        conn.read_len -= consumed;
    }
}

fn handle_write_cqe(result: i32, key: u32, scenario: &Scenario) {
    let template = &scenario.templates[0];
    assert!(
        result >= 0,
        "write failed for conn {}: {}",
        key,
        io::Error::from_raw_os_error(-result)
    );
    assert_eq!(
        result as usize,
        template.request_bytes.len(),
        "partial write for conn {}: wrote {} of {} bytes",
        key,
        result,
        template.request_bytes.len()
    );
}

#[allow(clippy::too_many_arguments)]
fn handle_read_cqe(
    result: i32,
    conn: &mut Connection,
    scenario: &Scenario,
    stats: &mut SummaryStats,
    run: &RunState,
    now: Instant,
    interval_latency_recorder: &mut Option<TimerRecorder>,
) {
    conn.read_inflight = false;
    if result == 0 {
        panic!(
            "connection closed with {} responses still pending",
            conn.pending_count()
        );
    }
    assert!(
        result > 0,
        "read failed: {}",
        io::Error::from_raw_os_error(-result)
    );

    conn.read_len += result as usize;
    process_read_buffer(
        conn,
        scenario,
        stats,
        run,
        now,
        interval_latency_recorder,
    );
}

fn run_scenario(addr: &str, scenario: Scenario) {
    assert!(scenario.connections > 0, "connections must be > 0");
    assert!(scenario.window > 0, "window must be > 0");

    let sq_entries = (scenario.connections * scenario.window * 2).clamp(256, 16384) as u32;
    let mut ring = IoUring::new(sq_entries).expect("io_uring creation failed");
    let mut conns: Slab<Connection> = Slab::with_capacity(scenario.connections);
    let mut cqe_buf: Vec<(u64, i32)> = Vec::with_capacity(sq_entries as usize);
    let mut stats = SummaryStats::default();
    let mut run = RunState::new(&scenario.stop_mode);
    let latency_interval_timer = scenario.collect_latency.then(|| Arc::new(TimerMetric::new()));
    let reporter = latency_interval_timer
        .as_ref()
        .map(|interval| Reporter::spawn(Arc::clone(interval)));
    let mut latency_interval_recorder = latency_interval_timer.as_ref().map(|timer| timer.recorder());

    for _ in 0..scenario.connections {
        let fd = create_connection(addr).expect("failed to connect");
        let entry = conns.vacant_entry();
        entry.insert(Connection::new(fd, scenario.window));
    }

    if matches!(scenario.stop_mode, StopMode::Duration { warmup, .. } if warmup > Duration::ZERO) {
        eprintln!(
            "{}: {} connections, window={}, {} vector(s)/req, warmup={}s, duration={}s -> {}",
            scenario.name,
            scenario.connections,
            scenario.window,
            scenario.templates[0].num_vectors,
            run.warmup_end.duration_since(run.start).as_secs(),
            run.measure_end.duration_since(run.warmup_end).as_secs(),
            addr
        );
        eprintln!(
            "{:>10}  {:>9}  {:>9}  {:>9}  {:>9}  {:>8}",
            "qps", "p50", "p95", "p99", "p99.9", "n"
        );
    } else {
        eprintln!(
            "{}: {} connections, window={}, {} vector(s)/req -> {}",
            scenario.name,
            scenario.connections,
            scenario.window,
            scenario.templates[0].num_vectors,
            addr
        );
    }

    if scenario.collect_latency
        && !matches!(scenario.stop_mode, StopMode::Duration { warmup, .. } if warmup > Duration::ZERO)
        && let Some(reporter) = reporter.as_ref()
    {
        reporter.start_measurement(run.start);
        run.measurement_started = true;
    }

    loop {
        let now = Instant::now();
        if scenario.collect_latency && !run.measurement_started && now >= run.warmup_end {
            if let Some(reporter) = reporter.as_ref() {
                reporter.start_measurement(now);
            }
            run.measurement_started = true;
        }
        for (key, conn) in &mut conns {
            submit_writes(&mut ring, conn, key as u32, &scenario, &run, now);
        }

        if conns
            .iter()
            .all(|(_, conn)| conn.is_finished(&scenario, now, &run))
            && ring.outstanding == 0
        {
            break;
        }

        if ring.outstanding == 0 {
            continue;
        }

        ring.wait(1);
        cqe_buf.clear();
        ring.drain_cqes_into(&mut cqe_buf);

        let now = Instant::now();
        for &(user_data, result) in &cqe_buf {
            let (op, key) = decode_user_data(user_data);
            let conn = conns
                .get_mut(key as usize)
                .unwrap_or_else(|| panic!("missing conn for key {}", key));
            match op {
                OP_WRITE => handle_write_cqe(result, key, &scenario),
                OP_READ => handle_read_cqe(
                    result,
                    conn,
                    &scenario,
                    &mut stats,
                    &run,
                    now,
                    &mut latency_interval_recorder,
                ),
                _ => panic!("unknown op {}", op),
            }
        }
    }

    let end = Instant::now();
    match scenario.stop_mode {
        StopMode::FixedCount {
            requests_per_connection,
        } => {
            let total = scenario.connections as u64 * requests_per_connection;
            let elapsed = end.duration_since(run.start);
            let qps = total as f64 / elapsed.as_secs_f64();
            eprintln!(
                "{}: {} requests in {:.2}s = {:.0} QPS",
                scenario.name,
                total,
                elapsed.as_secs_f64(),
                qps
            );
        }
        StopMode::Duration { .. } => {
            drop(latency_interval_recorder);
            if let Some(reporter) = reporter {
                reporter.finish(end, stats.measured_completions);
            }
        }
    }
}

fn smoke_test(addr: &str) {
    eprintln!("smoke test: connecting to {}", addr);

    run_scenario(
        addr,
        Scenario {
            name: "smoke-1",
            connections: 1,
            window: 1,
            templates: Arc::from([RequestTemplate::new(1)]),
            verify: true,
            collect_latency: false,
            stop_mode: StopMode::FixedCount {
                requests_per_connection: 1,
            },
        },
    );
    eprintln!("  1 vector: OK");

    run_scenario(
        addr,
        Scenario {
            name: "smoke-4",
            connections: 1,
            window: 1,
            templates: Arc::from([RequestTemplate::new(4)]),
            verify: true,
            collect_latency: false,
            stop_mode: StopMode::FixedCount {
                requests_per_connection: 1,
            },
        },
    );
    eprintln!("  4 vectors: OK");
    eprintln!("smoke test: PASSED");
}

fn main() {
    let cli = Cli::parse();
    let addr = format!("127.0.0.1:{}", cli.port);

    match cli.command.unwrap_or(Command::Smoke) {
        Command::Smoke => smoke_test(&addr),
        Command::Pipeline(args) => run_scenario(&addr, Scenario::pipeline(args)),
        Command::Bench(args) => run_scenario(&addr, Scenario::bench(args)),
        Command::Sustain(args) => run_scenario(&addr, Scenario::sustain(args)),
    }
}
