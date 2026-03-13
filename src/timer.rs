use std::sync::Mutex;
use std::time::Duration;

use hdrhistogram::Histogram;
use hdrhistogram::sync::{Recorder, SyncHistogram};

const REFRESH_TIMEOUT: Duration = Duration::from_millis(100);

pub struct TimerMetric {
    inner: Mutex<SyncHistogram<u64>>,
}

pub struct TimerRecorder {
    inner: Recorder<u64>,
}

#[derive(Clone)]
pub struct TimerSnapshot {
    hist: Histogram<u64>,
}

impl TimerMetric {
    pub fn new() -> Self {
        let hist = Histogram::new_with_bounds(1, 60_000_000_000, 3)
            .expect("failed to create hdrhistogram");
        Self {
            inner: Mutex::new(SyncHistogram::from(hist)),
        }
    }

    pub fn recorder(&self) -> TimerRecorder {
        let guard = self.inner.lock().expect("timer mutex poisoned");
        TimerRecorder {
            inner: guard.recorder(),
        }
    }

    pub fn snapshot(&self) -> Option<TimerSnapshot> {
        let mut guard = self.inner.lock().ok()?;
        guard.refresh_timeout(REFRESH_TIMEOUT);
        if guard.is_empty() {
            return None;
        }
        Some(TimerSnapshot {
            hist: (*guard).clone(),
        })
    }

    pub fn snapshot_and_reset(&self) -> Option<TimerSnapshot> {
        let mut guard = self.inner.lock().ok()?;
        guard.refresh_timeout(REFRESH_TIMEOUT);
        if guard.is_empty() {
            return None;
        }
        let snap = TimerSnapshot {
            hist: (*guard).clone(),
        };
        guard.reset();
        Some(snap)
    }
}

impl Default for TimerMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl TimerRecorder {
    pub fn record_duration(&mut self, duration: Duration) {
        self.record_nanos(duration.as_nanos() as u64);
    }

    pub fn record_nanos(&mut self, nanos: u64) {
        self.inner.saturating_record(nanos.max(1));
    }
}

impl TimerSnapshot {
    pub fn from_histogram(hist: Histogram<u64>) -> Self {
        Self { hist }
    }

    pub fn count(&self) -> u64 {
        self.hist.len()
    }

    pub fn p50_us(&self) -> f64 {
        self.value_us(50.0)
    }

    pub fn p95_us(&self) -> f64 {
        self.value_us(95.0)
    }

    pub fn p99_us(&self) -> f64 {
        self.value_us(99.0)
    }

    pub fn p999_us(&self) -> f64 {
        self.value_us(99.9)
    }

    pub fn p9999_us(&self) -> f64 {
        self.value_us(99.99)
    }

    pub fn max_us(&self) -> f64 {
        self.hist.max() as f64 / 1_000.0
    }

    pub fn merge_into(&self, target: &mut Histogram<u64>) {
        target
            .add(&self.hist)
            .expect("failed to merge timer snapshot histogram");
    }

    fn value_us(&self, percentile: f64) -> f64 {
        self.hist.value_at_quantile(percentile / 100.0) as f64 / 1_000.0
    }
}
