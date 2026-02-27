#[cfg(feature = "metrics")]
mod imp {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    // Stall / backpressure (cumulative counts)
    static REQ_RING_FULL: AtomicU64 = AtomicU64::new(0);
    static RESP_RING_FULL: AtomicU64 = AtomicU64::new(0);
    static POOL_EXHAUSTED: AtomicU64 = AtomicU64::new(0);
    static POOL_TOO_LARGE: AtomicU64 = AtomicU64::new(0);
    // Throughput (cumulative)
    static REQUESTS_PUBLISHED: AtomicU64 = AtomicU64::new(0);
    static RESPONSES_SENT: AtomicU64 = AtomicU64::new(0);
    // Batch processor: poll outcomes (stall = NoEvents)
    static POLL_EVENTS: AtomicU64 = AtomicU64::new(0);
    static POLL_NO_EVENTS: AtomicU64 = AtomicU64::new(0);
    // Gauges
    static POOL_MAX_IN_USE: AtomicUsize = AtomicUsize::new(0);
    static REQ_OCC: AtomicUsize = AtomicUsize::new(0);
    static RESP_OCC: AtomicUsize = AtomicUsize::new(0);
    static REQ_MAX_OCC: AtomicUsize = AtomicUsize::new(0);
    static RESP_MAX_OCC: AtomicUsize = AtomicUsize::new(0);

    #[derive(Clone, Copy)]
    pub struct MetricsSnapshot {
        pub req_ring_full: u64,
        pub resp_ring_full: u64,
        pub pool_exhausted: u64,
        pub pool_too_large: u64,
        pub requests_published: u64,
        pub responses_sent: u64,
        pub poll_events: u64,
        pub poll_no_events: u64,
        pub pool_max_in_use: usize,
        pub req_occ: usize,
        pub resp_occ: usize,
        pub req_max_occ: usize,
        pub resp_max_occ: usize,
    }

    pub fn inc_req_ring_full() {
        REQ_RING_FULL.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_resp_ring_full() {
        RESP_RING_FULL.fetch_add(1, Ordering::Relaxed);
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

    pub fn inc_resp_occ() {
        let v = RESP_OCC.fetch_add(1, Ordering::Relaxed) + 1;
        update_max(&RESP_MAX_OCC, v);
    }

    pub fn dec_resp_occ() {
        RESP_OCC.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn inc_requests_published() {
        REQUESTS_PUBLISHED.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_responses_sent() {
        RESPONSES_SENT.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_poll_events() {
        POLL_EVENTS.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_poll_no_events() {
        POLL_NO_EVENTS.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot() -> MetricsSnapshot {
        MetricsSnapshot {
            req_ring_full: REQ_RING_FULL.load(Ordering::Relaxed),
            resp_ring_full: RESP_RING_FULL.load(Ordering::Relaxed),
            pool_exhausted: POOL_EXHAUSTED.load(Ordering::Relaxed),
            pool_too_large: POOL_TOO_LARGE.load(Ordering::Relaxed),
            requests_published: REQUESTS_PUBLISHED.load(Ordering::Relaxed),
            responses_sent: RESPONSES_SENT.load(Ordering::Relaxed),
            poll_events: POLL_EVENTS.load(Ordering::Relaxed),
            poll_no_events: POLL_NO_EVENTS.load(Ordering::Relaxed),
            pool_max_in_use: POOL_MAX_IN_USE.load(Ordering::Relaxed),
            req_occ: REQ_OCC.load(Ordering::Relaxed),
            resp_occ: RESP_OCC.load(Ordering::Relaxed),
            req_max_occ: REQ_MAX_OCC.load(Ordering::Relaxed),
            resp_max_occ: RESP_MAX_OCC.load(Ordering::Relaxed),
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
                let resp_full_d = snap.resp_ring_full.saturating_sub(last_snap.resp_ring_full);
                let pool_exh_d = snap.pool_exhausted.saturating_sub(last_snap.pool_exhausted);
                let pool_tl_d = snap.pool_too_large.saturating_sub(last_snap.pool_too_large);
                let req_pub_d = snap
                    .requests_published
                    .saturating_sub(last_snap.requests_published);
                let resp_sent_d = snap.responses_sent.saturating_sub(last_snap.responses_sent);
                let poll_ev_d = snap.poll_events.saturating_sub(last_snap.poll_events);
                let poll_no_d = snap.poll_no_events.saturating_sub(last_snap.poll_no_events);
                let total_poll = poll_ev_d + poll_no_d;
                let stall_pct = if total_poll > 0 {
                    100.0 * (poll_no_d as f64 / total_poll as f64)
                } else {
                    0.0
                };
                println!(
                    "metrics delta {}s: published={} sent={} | stalls: req_ring_full={} resp_ring_full={} pool_exh={} pool_too_large={} | batch: poll_events={} poll_no_events={} stall_pct={:.1}% | gauges: req_occ={} resp_occ={} req_max={} resp_max={} pool_max_in_use={}",
                    INTERVAL_SECS,
                    req_pub_d,
                    resp_sent_d,
                    req_full_d,
                    resp_full_d,
                    pool_exh_d,
                    pool_tl_d,
                    poll_ev_d,
                    poll_no_d,
                    stall_pct,
                    snap.req_occ,
                    snap.resp_occ,
                    snap.req_max_occ,
                    snap.resp_max_occ,
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
        pub resp_ring_full: u64,
        pub pool_exhausted: u64,
        pub pool_too_large: u64,
        pub requests_published: u64,
        pub responses_sent: u64,
        pub poll_events: u64,
        pub poll_no_events: u64,
        pub pool_max_in_use: usize,
        pub req_occ: usize,
        pub resp_occ: usize,
        pub req_max_occ: usize,
        pub resp_max_occ: usize,
    }

    pub fn inc_req_ring_full() {}
    pub fn inc_resp_ring_full() {}
    pub fn inc_pool_exhausted() {}
    pub fn inc_pool_too_large() {}
    pub fn update_pool_in_use(_: usize) {}
    pub fn inc_req_occ() {}
    pub fn dec_req_occ() {}
    pub fn inc_resp_occ() {}
    pub fn dec_resp_occ() {}
    pub fn inc_requests_published() {}
    pub fn inc_responses_sent() {}
    pub fn inc_poll_events() {}
    pub fn inc_poll_no_events() {}
    pub fn snapshot() -> MetricsSnapshot {
        MetricsSnapshot {
            req_ring_full: 0,
            resp_ring_full: 0,
            pool_exhausted: 0,
            pool_too_large: 0,
            requests_published: 0,
            responses_sent: 0,
            poll_events: 0,
            poll_no_events: 0,
            pool_max_in_use: 0,
            req_occ: 0,
            resp_occ: 0,
            req_max_occ: 0,
            resp_max_occ: 0,
        }
    }
    pub fn spawn_reporter() {}
}

pub use imp::*;
