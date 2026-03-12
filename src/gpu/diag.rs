use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub static REQUESTS_PUBLISHED: AtomicU64 = AtomicU64::new(0);
pub static READ_SUBMITS: AtomicU64 = AtomicU64::new(0);
pub static READ_CQES: AtomicU64 = AtomicU64::new(0);
pub static READ_BYTES: AtomicU64 = AtomicU64::new(0);
pub static READ_NEGATIVE: AtomicU64 = AtomicU64::new(0);
pub static BYTES_CONSUMED: AtomicU64 = AtomicU64::new(0);
pub static RING_FULL_HITS: AtomicU64 = AtomicU64::new(0);
pub static BUFFERED_BYTES: AtomicU64 = AtomicU64::new(0);
pub static BATCHES_SUBMITTED: AtomicU64 = AtomicU64::new(0);
pub static VECTORS_SUBMITTED: AtomicU64 = AtomicU64::new(0);
pub static RESPONSES_WRITTEN: AtomicU64 = AtomicU64::new(0);
pub static BATCHES_COMPLETED: AtomicU64 = AtomicU64::new(0);
pub static WRITE_SQES: AtomicU64 = AtomicU64::new(0);
pub static WRITE_CQES: AtomicU64 = AtomicU64::new(0);
pub static WRITE_NEGATIVE: AtomicU64 = AtomicU64::new(0);

pub fn enabled() -> bool {
    std::env::var_os("DISRUST_GPU_DIAG").is_some()
}

pub fn bump(counter: &AtomicU64, delta: u64) {
    if !enabled() || delta == 0 {
        return;
    }
    counter.fetch_add(delta, Ordering::Relaxed);
}

pub struct Reporter {
    last: Instant,
}

impl Reporter {
    pub fn new() -> Self {
        Self {
            last: Instant::now(),
        }
    }

    pub fn maybe_report(&mut self) {
        if !enabled() || self.last.elapsed() < Duration::from_secs(1) {
            return;
        }
        self.last = Instant::now();
        eprintln!(
            "disrust-gpu diag: reads submit={} cqe={} bytes={} neg={} consumed={} buffered={} ring_full={} published={} batches_submitted={} vectors_submitted={} batches_completed={} responses_written={} write_sqes={} write_cqes={} write_neg={}",
            READ_SUBMITS.load(Ordering::Relaxed),
            READ_CQES.load(Ordering::Relaxed),
            READ_BYTES.load(Ordering::Relaxed),
            READ_NEGATIVE.load(Ordering::Relaxed),
            BYTES_CONSUMED.load(Ordering::Relaxed),
            BUFFERED_BYTES.load(Ordering::Relaxed),
            RING_FULL_HITS.load(Ordering::Relaxed),
            REQUESTS_PUBLISHED.load(Ordering::Relaxed),
            BATCHES_SUBMITTED.load(Ordering::Relaxed),
            VECTORS_SUBMITTED.load(Ordering::Relaxed),
            BATCHES_COMPLETED.load(Ordering::Relaxed),
            RESPONSES_WRITTEN.load(Ordering::Relaxed),
            WRITE_SQES.load(Ordering::Relaxed),
            WRITE_CQES.load(Ordering::Relaxed),
            WRITE_NEGATIVE.load(Ordering::Relaxed),
        );
    }
}

impl Default for Reporter {
    fn default() -> Self {
        Self::new()
    }
}
