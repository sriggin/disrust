#[cfg(feature = "metrics")]
mod imp {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    static REQ_RING_FULL: AtomicU64 = AtomicU64::new(0);
    static RESP_RING_FULL: AtomicU64 = AtomicU64::new(0);
    static POOL_EXHAUSTED: AtomicU64 = AtomicU64::new(0);
    static POOL_TOO_LARGE: AtomicU64 = AtomicU64::new(0);
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
        let mut prev = POOL_MAX_IN_USE.load(Ordering::Relaxed);
        while value > prev {
            match POOL_MAX_IN_USE.compare_exchange_weak(
                prev,
                value,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(next) => prev = next,
            }
        }
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

    pub fn snapshot() -> MetricsSnapshot {
        MetricsSnapshot {
            req_ring_full: REQ_RING_FULL.load(Ordering::Relaxed),
            resp_ring_full: RESP_RING_FULL.load(Ordering::Relaxed),
            pool_exhausted: POOL_EXHAUSTED.load(Ordering::Relaxed),
            pool_too_large: POOL_TOO_LARGE.load(Ordering::Relaxed),
            pool_max_in_use: POOL_MAX_IN_USE.load(Ordering::Relaxed),
            req_occ: REQ_OCC.load(Ordering::Relaxed),
            resp_occ: RESP_OCC.load(Ordering::Relaxed),
            req_max_occ: REQ_MAX_OCC.load(Ordering::Relaxed),
            resp_max_occ: RESP_MAX_OCC.load(Ordering::Relaxed),
        }
    }

    pub fn spawn_reporter() {
        std::thread::spawn(|| {
            let mut last = Instant::now();
            loop {
                std::thread::sleep(Duration::from_secs(1));
                let snap = snapshot();
                let elapsed = last.elapsed().as_secs_f32();
                last = Instant::now();
                eprintln!(
                    "metrics: req_full={} resp_full={} req_occ={} resp_occ={} req_max={} resp_max={} pool_exh={} pool_too_large={} pool_max_in_use={} ({:.1}s)",
                    snap.req_ring_full,
                    snap.resp_ring_full,
                    snap.req_occ,
                    snap.resp_occ,
                    snap.req_max_occ,
                    snap.resp_max_occ,
                    snap.pool_exhausted,
                    snap.pool_too_large,
                    snap.pool_max_in_use,
                    elapsed,
                );
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
    pub fn snapshot() -> MetricsSnapshot {
        MetricsSnapshot {
            req_ring_full: 0,
            resp_ring_full: 0,
            pool_exhausted: 0,
            pool_too_large: 0,
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
