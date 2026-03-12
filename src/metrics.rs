#[cfg(feature = "metrics")]
mod imp {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    // Stall / backpressure (cumulative counts)
    static REQ_RING_FULL: AtomicU64 = AtomicU64::new(0);
    static POOL_EXHAUSTED: AtomicU64 = AtomicU64::new(0);
    static POOL_TOO_LARGE: AtomicU64 = AtomicU64::new(0);
    // Throughput (cumulative)
    static REQUESTS_PUBLISHED: AtomicU64 = AtomicU64::new(0);
    // Gauges
    static POOL_MAX_IN_USE: AtomicUsize = AtomicUsize::new(0);
    static REQ_OCC: AtomicUsize = AtomicUsize::new(0);
    static REQ_MAX_OCC: AtomicUsize = AtomicUsize::new(0);

    #[derive(Clone, Copy)]
    pub struct MetricsSnapshot {
        pub req_ring_full: u64,
        pub pool_exhausted: u64,
        pub pool_too_large: u64,
        pub requests_published: u64,
        pub pool_max_in_use: usize,
        pub req_occ: usize,
        pub req_max_occ: usize,
    }

    pub fn inc_req_ring_full() {
        REQ_RING_FULL.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_pool_exhausted() {
        POOL_EXHAUSTED.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_pool_too_large() {
        POOL_TOO_LARGE.fetch_add(1, Ordering::Relaxed);
    }

    pub fn update_pool_in_use(value: usize) {
        update_max(&POOL_MAX_IN_USE, value);
    }

    fn update_max(target: &AtomicUsize, value: usize) {
        let mut prev = target.load(Ordering::Relaxed);
        while value > prev {
            match target.compare_exchange_weak(prev, value, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(next) => prev = next,
            }
        }
    }

    pub fn inc_req_occ() {
        let v = REQ_OCC.fetch_add(1, Ordering::Relaxed) + 1;
        update_max(&REQ_MAX_OCC, v);
    }

    pub fn dec_req_occ() {
        REQ_OCC.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn inc_requests_published() {
        REQUESTS_PUBLISHED.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot() -> MetricsSnapshot {
        MetricsSnapshot {
            req_ring_full: REQ_RING_FULL.load(Ordering::Relaxed),
            pool_exhausted: POOL_EXHAUSTED.load(Ordering::Relaxed),
            pool_too_large: POOL_TOO_LARGE.load(Ordering::Relaxed),
            requests_published: REQUESTS_PUBLISHED.load(Ordering::Relaxed),
            pool_max_in_use: POOL_MAX_IN_USE.load(Ordering::Relaxed),
            req_occ: REQ_OCC.load(Ordering::Relaxed),
            req_max_occ: REQ_MAX_OCC.load(Ordering::Relaxed),
        }
    }

    pub fn spawn_reporter() {
        const INTERVAL_SECS: u64 = 10;
        std::thread::spawn(|| {
            let mut last_snap = snapshot();
            loop {
                std::thread::sleep(Duration::from_secs(INTERVAL_SECS));
                let snap = snapshot();
                let req_full_d = snap.req_ring_full.saturating_sub(last_snap.req_ring_full);
                let pool_exh_d = snap.pool_exhausted.saturating_sub(last_snap.pool_exhausted);
                let pool_tl_d = snap.pool_too_large.saturating_sub(last_snap.pool_too_large);
                let req_pub_d = snap
                    .requests_published
                    .saturating_sub(last_snap.requests_published);
                println!(
                    "metrics delta {}s: published={} | stalls: req_ring_full={} pool_exh={} pool_too_large={} | gauges: req_occ={} req_max={} pool_max_in_use={}",
                    INTERVAL_SECS,
                    req_pub_d,
                    req_full_d,
                    pool_exh_d,
                    pool_tl_d,
                    snap.req_occ,
                    snap.req_max_occ,
                    snap.pool_max_in_use,
                );
                last_snap = snap;
            }
        });
    }
}

#[cfg(not(feature = "metrics"))]
#[allow(dead_code)]
mod imp {
    #[derive(Clone, Copy)]
    pub struct MetricsSnapshot {
        pub req_ring_full: u64,
        pub pool_exhausted: u64,
        pub pool_too_large: u64,
        pub requests_published: u64,
        pub pool_max_in_use: usize,
        pub req_occ: usize,
        pub req_max_occ: usize,
    }

    pub fn inc_req_ring_full() {}
    pub fn inc_pool_exhausted() {}
    pub fn inc_pool_too_large() {}
    pub fn update_pool_in_use(_: usize) {}
    pub fn inc_req_occ() {}
    pub fn dec_req_occ() {}
    pub fn inc_requests_published() {}
    pub fn snapshot() -> MetricsSnapshot {
        MetricsSnapshot {
            req_ring_full: 0,
            pool_exhausted: 0,
            pool_too_large: 0,
            requests_published: 0,
            pool_max_in_use: 0,
            req_occ: 0,
            req_max_occ: 0,
        }
    }
    pub fn spawn_reporter() {}
}

pub use imp::*;
