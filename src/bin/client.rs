use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};

use disrust::constants::FEATURE_DIM;

#[derive(Parser)]
#[command(about = "Test client for disrust inference server")]
struct Args {
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
    /// Send 1000 pipelined requests and verify all results
    Pipeline,
    /// Benchmark throughput with concurrent pipelined connections
    Bench {
        /// Number of concurrent connections
        #[arg(short, long, default_value_t = 4)]
        connections: usize,
        /// Requests per connection
        #[arg(short, long, default_value_t = 100_000)]
        requests: usize,
    },
    /// Sustained load with per-request latency measurement
    Sustain {
        /// Number of concurrent connections
        #[arg(short, long, default_value_t = 4)]
        connections: usize,
        /// In-flight requests per connection (pipeline window)
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
    },
}

fn build_request(num_vectors: u32) -> (Vec<u8>, Vec<f32>) {
    let mut buf = Vec::new();
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

    (buf, expected_sums)
}

fn read_response(stream: &mut TcpStream, expected: u32) -> Vec<f32> {
    let mut header = [0u8; 4];
    stream
        .read_exact(&mut header)
        .expect("failed to read response header");
    let num_vectors = u32::from_le_bytes(header);

    assert_eq!(
        num_vectors, expected,
        "response num_vectors={num_vectors} does not match expected={expected} (protocol error or data corruption)"
    );

    let mut result_bytes = vec![0u8; num_vectors as usize * 4];
    stream
        .read_exact(&mut result_bytes)
        .expect("failed to read response body");

    let mut results = Vec::with_capacity(num_vectors as usize);
    for i in 0..num_vectors as usize {
        let offset = i * 4;
        results.push(f32::from_le_bytes([
            result_bytes[offset],
            result_bytes[offset + 1],
            result_bytes[offset + 2],
            result_bytes[offset + 3],
        ]));
    }
    results
}

fn main() {
    let args = Args::parse();
    let addr = format!("127.0.0.1:{}", args.port);

    match args.command.unwrap_or(Command::Smoke) {
        Command::Smoke => smoke_test(&addr),
        Command::Pipeline => pipeline_test(&addr),
        Command::Bench {
            connections,
            requests,
        } => bench_test(&addr, connections, requests),
        Command::Sustain {
            connections,
            window,
            vectors,
            warmup,
            duration,
        } => sustain_test(&addr, connections, window, vectors, warmup, duration),
    }
}

fn smoke_test(addr: &str) {
    eprintln!("smoke test: connecting to {}", addr);
    let mut stream = TcpStream::connect(addr).expect("failed to connect");

    // Test with 1 vector
    let (req, expected) = build_request(1);
    stream.write_all(&req).expect("failed to write");
    let results = read_response(&mut stream, 1);
    assert_eq!(results.len(), 1);
    let diff = (results[0] - expected[0]).abs();
    assert!(
        diff < 0.1,
        "result {} != expected {} (diff={})",
        results[0],
        expected[0],
        diff
    );
    eprintln!(
        "  1 vector: OK (result={:.2}, expected={:.2})",
        results[0], expected[0]
    );

    // Test with 4 vectors
    let (req, expected) = build_request(4);
    stream.write_all(&req).expect("failed to write");
    let results = read_response(&mut stream, 4);
    assert_eq!(results.len(), 4);
    for (i, (got, exp)) in results.iter().zip(expected.iter()).enumerate() {
        let diff = (got - exp).abs();
        assert!(
            diff < 0.1,
            "vector {}: result {} != expected {}",
            i,
            got,
            exp
        );
    }
    eprintln!("  4 vectors: OK");

    eprintln!("smoke test: PASSED");
}

fn pipeline_test(addr: &str) {
    let num_requests = 1000;
    eprintln!(
        "pipeline test: sending {} pipelined requests to {}",
        num_requests, addr
    );
    let mut stream = TcpStream::connect(addr).expect("failed to connect");

    let mut all_expected = Vec::new();
    for _ in 0..num_requests {
        let (req, expected) = build_request(2);
        stream.write_all(&req).expect("failed to write");
        all_expected.push(expected);
    }

    for (i, expected) in all_expected.iter().enumerate() {
        let results = read_response(&mut stream, 2);
        assert_eq!(results.len(), 2, "request {}: wrong result count", i);
        for (j, (got, exp)) in results.iter().zip(expected.iter()).enumerate() {
            let diff = (got - exp).abs();
            assert!(diff < 0.1, "req {} vec {}: {} != {}", i, j, got, exp);
        }
    }

    eprintln!("pipeline test: PASSED ({} requests)", num_requests);
}

fn bench_test(addr: &str, num_connections: usize, requests_per_conn: usize) {
    eprintln!(
        "bench: {} connections x {} requests (pipelined) to {}",
        num_connections, requests_per_conn, addr
    );

    let (req, _) = build_request(1);
    let response_size: usize = 4 + 4;

    let start = Instant::now();

    let handles: Vec<_> = (0..num_connections)
        .map(|_| {
            let addr = addr.to_string();
            let req = req.clone();
            std::thread::spawn(move || {
                let stream = TcpStream::connect(&addr).expect("failed to connect");
                stream.set_nodelay(true).unwrap();

                let mut writer = stream.try_clone().expect("clone failed");
                let mut reader = stream;

                let write_handle = std::thread::spawn(move || {
                    for _ in 0..requests_per_conn {
                        writer.write_all(&req).expect("write failed");
                    }
                });

                let mut resp_buf = vec![0u8; response_size * 1024];
                let mut total_bytes_needed = requests_per_conn * response_size;
                while total_bytes_needed > 0 {
                    let to_read = total_bytes_needed.min(resp_buf.len());
                    let n = reader.read(&mut resp_buf[..to_read]).expect("read failed");
                    if n == 0 {
                        panic!(
                            "connection closed with {} bytes remaining",
                            total_bytes_needed
                        );
                    }
                    total_bytes_needed -= n;
                }

                write_handle.join().expect("writer panicked");
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }

    let elapsed = start.elapsed();
    let total = num_connections * requests_per_conn;
    let qps = total as f64 / elapsed.as_secs_f64();
    eprintln!(
        "bench: {} requests in {:.2}s = {:.0} QPS",
        total,
        elapsed.as_secs_f64(),
        qps
    );
}

fn percentile(sorted: &[u64], p: f64) -> f64 {
    let i = ((p / 100.0) * sorted.len() as f64) as usize;
    sorted[i.min(sorted.len() - 1)] as f64 / 1_000.0
}

fn print_interval(samples: &mut [u64], elapsed: Duration) {
    samples.sort_unstable();
    let n = samples.len();
    let qps = n as f64 / elapsed.as_secs_f64();
    eprintln!(
        "{:>10.0}  {:>8.1}µs  {:>8.1}µs  {:>8.1}µs  {:>8.1}µs  {:>8}",
        qps,
        percentile(samples, 50.0),
        percentile(samples, 95.0),
        percentile(samples, 99.0),
        percentile(samples, 99.9),
        n,
    );
}

fn sustain_test(
    addr: &str,
    num_connections: usize,
    window: usize,
    num_vectors: u32,
    warmup_secs: u64,
    duration_secs: u64,
) {
    eprintln!(
        "sustain: {} connections, window={}, {} vector(s)/req, warmup={}s, duration={}s → {}",
        num_connections, window, num_vectors, warmup_secs, duration_secs, addr
    );

    let (tx, rx) = mpsc::channel::<u64>();

    for _ in 0..num_connections {
        let addr = addr.to_string();
        let tx = tx.clone();
        let (req, _) = build_request(num_vectors);
        std::thread::spawn(move || {
            let mut stream = TcpStream::connect(&addr).expect("failed to connect");
            stream.set_nodelay(true).unwrap();
            let mut in_flight: VecDeque<Instant> = VecDeque::with_capacity(window);

            loop {
                while in_flight.len() < window {
                    stream.write_all(&req).expect("write failed");
                    in_flight.push_back(Instant::now());
                }
                read_response(&mut stream, num_vectors);
                let sent_at = in_flight.pop_front().unwrap();
                if tx.send(sent_at.elapsed().as_nanos() as u64).is_err() {
                    break;
                }
            }
        });
    }
    drop(tx);

    // Warmup: drain and discard until warmup period expires
    if warmup_secs > 0 {
        eprint!("warming up ({warmup_secs}s)");
        let warmup_end = Instant::now() + Duration::from_secs(warmup_secs);
        while Instant::now() < warmup_end {
            while rx.try_recv().is_ok() {}
            std::thread::sleep(Duration::from_millis(100));
            eprint!(".");
        }
        eprintln!(" ready");
    }

    // Measurement
    eprintln!(
        "{:>10}  {:>9}  {:>9}  {:>9}  {:>9}  {:>8}",
        "qps", "p50", "p95", "p99", "p99.9", "n"
    );

    let measure_start = Instant::now();
    let measure_end = measure_start + Duration::from_secs(duration_secs);
    let mut all_samples: Vec<u64> = Vec::new();
    let mut interval_samples: Vec<u64> = Vec::new();
    let mut last_print = Instant::now();

    loop {
        let now = Instant::now();
        if now >= measure_end {
            break;
        }
        let timeout = (measure_end - now).min(Duration::from_millis(100));
        match rx.recv_timeout(timeout) {
            Ok(ns) => {
                interval_samples.push(ns);
                all_samples.push(ns);
                while let Ok(ns) = rx.try_recv() {
                    interval_samples.push(ns);
                    all_samples.push(ns);
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                eprintln!("error: all worker connections died — is the server running?");
                break;
            }
            Err(RecvTimeoutError::Timeout) => {}
        }

        if last_print.elapsed() >= Duration::from_secs(1) && !interval_samples.is_empty() {
            print_interval(&mut interval_samples, last_print.elapsed());
            interval_samples.clear();
            last_print = Instant::now();
        }
    }

    // Flush partial final interval
    if !interval_samples.is_empty() {
        print_interval(&mut interval_samples, last_print.elapsed());
    }

    // Final report
    if all_samples.is_empty() {
        eprintln!("no samples collected");
        return;
    }
    all_samples.sort_unstable();
    let n = all_samples.len();
    let elapsed = measure_start.elapsed();
    eprintln!();
    eprintln!(
        "── summary ({:.1}s, {} requests) ──────────────────────────────────",
        elapsed.as_secs_f64(),
        n
    );
    eprintln!("  qps     {:.0}", n as f64 / elapsed.as_secs_f64());
    eprintln!("  p50     {:.1}µs", percentile(&all_samples, 50.0));
    eprintln!("  p95     {:.1}µs", percentile(&all_samples, 95.0));
    eprintln!("  p99     {:.1}µs", percentile(&all_samples, 99.0));
    eprintln!("  p99.9   {:.1}µs", percentile(&all_samples, 99.9));
    eprintln!("  p99.99  {:.1}µs", percentile(&all_samples, 99.99));
    eprintln!("  max     {:.1}µs", all_samples[n - 1] as f64 / 1_000.0);
}
