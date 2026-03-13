use std::sync::OnceLock;
use std::time::{Duration, Instant};

fn start_instant() -> &'static Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now)
}

pub fn monotonic_now_ns() -> u64 {
    start_instant().elapsed().as_nanos().min(u64::MAX as u128) as u64
}

pub fn elapsed_since_ns(start_ns: u64) -> Duration {
    Duration::from_nanos(monotonic_now_ns().saturating_sub(start_ns))
}
