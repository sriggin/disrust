use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Instant;

use disrust::constants::FEATURE_DIM;

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

fn read_response(stream: &mut TcpStream) -> Vec<f32> {
    let mut header = [0u8; 4];
    stream
        .read_exact(&mut header)
        .expect("failed to read response header");
    let num_vectors = u32::from_le_bytes(header);

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
    let port: u16 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(9900);

    let mode = std::env::args().nth(2).unwrap_or_default();

    let addr = format!("127.0.0.1:{}", port);

    match mode.as_str() {
        "pipeline" => pipeline_test(&addr),
        "bench" => bench_test(&addr),
        _ => smoke_test(&addr),
    }
}

fn smoke_test(addr: &str) {
    eprintln!("smoke test: connecting to {}", addr);
    let mut stream = TcpStream::connect(addr).expect("failed to connect");

    // Test with 1 vector
    let (req, expected) = build_request(1);
    stream.write_all(&req).expect("failed to write");
    let results = read_response(&mut stream);
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
    let results = read_response(&mut stream);
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

    // Send all requests without reading
    let mut all_expected = Vec::new();
    for _ in 0..num_requests {
        let (req, expected) = build_request(2);
        stream.write_all(&req).expect("failed to write");
        all_expected.push(expected);
    }

    // Read all responses
    for (i, expected) in all_expected.iter().enumerate() {
        let results = read_response(&mut stream);
        assert_eq!(results.len(), 2, "request {}: wrong result count", i);
        for (j, (got, exp)) in results.iter().zip(expected.iter()).enumerate() {
            let diff = (got - exp).abs();
            assert!(diff < 0.1, "req {} vec {}: {} != {}", i, j, got, exp);
        }
    }

    eprintln!("pipeline test: PASSED ({} requests)", num_requests);
}

fn bench_test(addr: &str) {
    let num_connections: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let requests_per_conn: usize = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    eprintln!(
        "bench: {} connections x {} requests (pipelined) to {}",
        num_connections, requests_per_conn, addr
    );

    // Pre-build the request payload once
    let (req, _) = build_request(1);
    // Response size: 4-byte header + 1 f32 = 8 bytes
    let response_size: usize = 4 + 4;

    let start = Instant::now();

    let handles: Vec<_> = (0..num_connections)
        .map(|_| {
            let addr = addr.to_string();
            let req = req.clone();
            std::thread::spawn(move || {
                let stream = TcpStream::connect(&addr).expect("failed to connect");
                stream.set_nodelay(true).unwrap();

                // Split into writer and reader on separate threads
                let mut writer = stream.try_clone().expect("clone failed");
                let mut reader = stream;

                let write_handle = std::thread::spawn(move || {
                    for _ in 0..requests_per_conn {
                        writer.write_all(&req).expect("write failed");
                    }
                });

                // Reader: consume all responses
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
