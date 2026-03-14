#[cfg(feature = "metrics")]
mod imp {
    use std::cell::RefCell;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::time::Duration;

    use crate::affinity;
    use crate::timer::{TimerMetric, TimerRecorder, TimerSnapshot};

    // Stall / backpressure (cumulative counts)
    static REQ_RING_FULL: AtomicU64 = AtomicU64::new(0);
    static POOL_EXHAUSTED: AtomicU64 = AtomicU64::new(0);
    static POOL_TOO_LARGE: AtomicU64 = AtomicU64::new(0);
    static SESSION_WAIT_LOOPS: AtomicU64 = AtomicU64::new(0);
    static COMPLETION_QUEUE_EMPTY_SPINS: AtomicU64 = AtomicU64::new(0);
    static COMPLETION_POLL_NO_EVENTS: AtomicU64 = AtomicU64::new(0);
    static WRITE_DRAIN_WAITS: AtomicU64 = AtomicU64::new(0);
    static WRITE_PARTIAL: AtomicU64 = AtomicU64::new(0);
    static WRITE_EAGAIN: AtomicU64 = AtomicU64::new(0);
    static WRITE_FATAL: AtomicU64 = AtomicU64::new(0);
    // Throughput (cumulative)
    static REQUESTS_PUBLISHED: AtomicU64 = AtomicU64::new(0);
    static BATCHES_SUBMITTED: AtomicU64 = AtomicU64::new(0);
    static VECTORS_SUBMITTED: AtomicU64 = AtomicU64::new(0);
    static BATCHES_COMPLETED: AtomicU64 = AtomicU64::new(0);
    static SLOTS_SUBMITTED: AtomicU64 = AtomicU64::new(0);
    static BACKLOG_SLOTS_AT_BUILD: AtomicU64 = AtomicU64::new(0);
    static BATCH_STOP_CAP: AtomicU64 = AtomicU64::new(0);
    static BATCH_STOP_BACKLOG_EMPTY: AtomicU64 = AtomicU64::new(0);
    static BATCH_STOP_NON_CONTIG: AtomicU64 = AtomicU64::new(0);
    static RESPONSES_WRITTEN: AtomicU64 = AtomicU64::new(0);
    static READ_SUBMITS: AtomicU64 = AtomicU64::new(0);
    static READ_CQES: AtomicU64 = AtomicU64::new(0);
    static READ_BYTES: AtomicU64 = AtomicU64::new(0);
    static READ_NEGATIVE: AtomicU64 = AtomicU64::new(0);
    static BYTES_CONSUMED: AtomicU64 = AtomicU64::new(0);
    static WRITE_SQES: AtomicU64 = AtomicU64::new(0);
    static WRITE_CQES: AtomicU64 = AtomicU64::new(0);
    static WRITE_NEGATIVE: AtomicU64 = AtomicU64::new(0);
    static BATCH_TOTAL_NS: OnceLock<TimerMetric> = OnceLock::new();
    static BATCH_WAIT_NS: OnceLock<TimerMetric> = OnceLock::new();
    static BACKLOG_AGE_NS: OnceLock<TimerMetric> = OnceLock::new();
    static PUBLISH_TO_SUBMIT_NS: OnceLock<TimerMetric> = OnceLock::new();
    static PUBLISH_TO_WRITE_SUBMIT_NS: OnceLock<TimerMetric> = OnceLock::new();
    static WRITE_DRAIN_NS: OnceLock<TimerMetric> = OnceLock::new();
    thread_local! {
        static BATCH_TOTAL_RECORDER: RefCell<Option<TimerRecorder>> = const { RefCell::new(None) };
        static BATCH_WAIT_RECORDER: RefCell<Option<TimerRecorder>> = const { RefCell::new(None) };
        static BACKLOG_AGE_RECORDER: RefCell<Option<TimerRecorder>> = const { RefCell::new(None) };
        static PUBLISH_TO_SUBMIT_RECORDER: RefCell<Option<TimerRecorder>> = const { RefCell::new(None) };
        static PUBLISH_TO_WRITE_SUBMIT_RECORDER: RefCell<Option<TimerRecorder>> = const { RefCell::new(None) };
        static WRITE_DRAIN_RECORDER: RefCell<Option<TimerRecorder>> = const { RefCell::new(None) };
    }
    // Gauges
    static POOL_MAX_IN_USE: AtomicUsize = AtomicUsize::new(0);
    static REQ_OCC: AtomicUsize = AtomicUsize::new(0);
    static REQ_MAX_OCC: AtomicUsize = AtomicUsize::new(0);
    static BUFFERED_BYTES: AtomicUsize = AtomicUsize::new(0);

    #[derive(Clone, Copy)]
    pub struct MetricsSnapshot {
        pub req_ring_full: u64,
        pub pool_exhausted: u64,
        pub pool_too_large: u64,
        pub requests_published: u64,
        pub batches_submitted: u64,
        pub vectors_submitted: u64,
        pub batches_completed: u64,
        pub slots_submitted: u64,
        pub backlog_slots_at_build: u64,
        pub batch_stop_cap: u64,
        pub batch_stop_backlog_empty: u64,
        pub batch_stop_non_contig: u64,
        pub responses_written: u64,
        pub read_submits: u64,
        pub read_cqes: u64,
        pub read_bytes: u64,
        pub read_negative: u64,
        pub bytes_consumed: u64,
        pub write_sqes: u64,
        pub write_cqes: u64,
        pub write_negative: u64,
        pub session_wait_loops: u64,
        pub completion_queue_empty_spins: u64,
        pub completion_poll_no_events: u64,
        pub write_drain_waits: u64,
        pub write_partial: u64,
        pub write_eagain: u64,
        pub write_fatal: u64,
        pub pool_max_in_use: usize,
        pub req_occ: usize,
        pub req_max_occ: usize,
        pub buffered_bytes: usize,
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

    pub fn inc_session_wait_loops() {
        SESSION_WAIT_LOOPS.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_completion_queue_empty_spins() {
        COMPLETION_QUEUE_EMPTY_SPINS.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_completion_poll_no_events() {
        COMPLETION_POLL_NO_EVENTS.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_write_drain_waits() {
        WRITE_DRAIN_WAITS.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_write_partial() {
        WRITE_PARTIAL.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_write_eagain() {
        WRITE_EAGAIN.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_write_fatal() {
        WRITE_FATAL.fetch_add(1, Ordering::Relaxed);
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
        let mut prev = REQ_OCC.load(Ordering::Relaxed);
        loop {
            let next = prev.saturating_add(1);
            match REQ_OCC.compare_exchange_weak(prev, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => {
                    update_max(&REQ_MAX_OCC, next);
                    return;
                }
                Err(actual) => prev = actual,
            }
        }
    }

    pub fn dec_req_occ() {
        let mut prev = REQ_OCC.load(Ordering::Relaxed);
        loop {
            let next = prev.saturating_sub(1);
            match REQ_OCC.compare_exchange_weak(prev, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return,
                Err(actual) => prev = actual,
            }
        }
    }

    pub fn inc_requests_published() {
        REQUESTS_PUBLISHED.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_batches_submitted() {
        BATCHES_SUBMITTED.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_vectors_submitted(count: u64) {
        VECTORS_SUBMITTED.fetch_add(count, Ordering::Relaxed);
    }

    pub fn inc_batches_completed() {
        BATCHES_COMPLETED.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_slots_submitted(count: u64) {
        SLOTS_SUBMITTED.fetch_add(count, Ordering::Relaxed);
    }

    pub fn add_backlog_slots_at_build(count: u64) {
        BACKLOG_SLOTS_AT_BUILD.fetch_add(count, Ordering::Relaxed);
    }

    pub fn inc_batch_stop_cap() {
        BATCH_STOP_CAP.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_batch_stop_backlog_empty() {
        BATCH_STOP_BACKLOG_EMPTY.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_batch_stop_non_contig() {
        BATCH_STOP_NON_CONTIG.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_responses_written() {
        RESPONSES_WRITTEN.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_read_submits() {
        READ_SUBMITS.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_read_cqes() {
        READ_CQES.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_read_bytes(count: u64) {
        READ_BYTES.fetch_add(count, Ordering::Relaxed);
    }

    pub fn inc_read_negative() {
        READ_NEGATIVE.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_bytes_consumed(count: u64) {
        BYTES_CONSUMED.fetch_add(count, Ordering::Relaxed);
    }

    pub fn inc_write_sqes() {
        WRITE_SQES.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_write_cqes() {
        WRITE_CQES.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_write_negative() {
        WRITE_NEGATIVE.fetch_add(1, Ordering::Relaxed);
    }

    fn batch_total_timer() -> &'static TimerMetric {
        BATCH_TOTAL_NS.get_or_init(TimerMetric::new)
    }

    fn batch_wait_timer() -> &'static TimerMetric {
        BATCH_WAIT_NS.get_or_init(TimerMetric::new)
    }

    fn backlog_age_timer() -> &'static TimerMetric {
        BACKLOG_AGE_NS.get_or_init(TimerMetric::new)
    }

    fn publish_to_submit_timer() -> &'static TimerMetric {
        PUBLISH_TO_SUBMIT_NS.get_or_init(TimerMetric::new)
    }

    fn publish_to_write_submit_timer() -> &'static TimerMetric {
        PUBLISH_TO_WRITE_SUBMIT_NS.get_or_init(TimerMetric::new)
    }

    fn write_drain_timer() -> &'static TimerMetric {
        WRITE_DRAIN_NS.get_or_init(TimerMetric::new)
    }

    pub fn record_batch_total(duration: Duration) {
        BATCH_TOTAL_RECORDER.with(|slot| {
            let mut slot = slot.borrow_mut();
            let recorder = slot.get_or_insert_with(|| batch_total_timer().recorder());
            recorder.record_duration(duration);
        });
    }

    pub fn record_batch_wait(duration: Duration) {
        BATCH_WAIT_RECORDER.with(|slot| {
            let mut slot = slot.borrow_mut();
            let recorder = slot.get_or_insert_with(|| batch_wait_timer().recorder());
            recorder.record_duration(duration);
        });
    }

    pub fn record_backlog_age(duration: Duration) {
        BACKLOG_AGE_RECORDER.with(|slot| {
            let mut slot = slot.borrow_mut();
            let recorder = slot.get_or_insert_with(|| backlog_age_timer().recorder());
            recorder.record_duration(duration);
        });
    }

    pub fn record_publish_to_submit(duration: Duration) {
        PUBLISH_TO_SUBMIT_RECORDER.with(|slot| {
            let mut slot = slot.borrow_mut();
            let recorder = slot.get_or_insert_with(|| publish_to_submit_timer().recorder());
            recorder.record_duration(duration);
        });
    }

    pub fn record_publish_to_write_submit(duration: Duration) {
        PUBLISH_TO_WRITE_SUBMIT_RECORDER.with(|slot| {
            let mut slot = slot.borrow_mut();
            let recorder = slot.get_or_insert_with(|| publish_to_write_submit_timer().recorder());
            recorder.record_duration(duration);
        });
    }

    pub fn record_write_drain(duration: Duration) {
        WRITE_DRAIN_RECORDER.with(|slot| {
            let mut slot = slot.borrow_mut();
            let recorder = slot.get_or_insert_with(|| write_drain_timer().recorder());
            recorder.record_duration(duration);
        });
    }

    pub fn idle_timers() {
        // No-op. Timer snapshots use bounded refresh timeouts instead of dropping recorders
        // on transient idle phases, which was perturbing the completion hot path.
    }

    pub fn set_buffered_bytes(value: usize) {
        BUFFERED_BYTES.store(value, Ordering::Relaxed);
    }

    pub fn snapshot() -> MetricsSnapshot {
        MetricsSnapshot {
            req_ring_full: REQ_RING_FULL.load(Ordering::Relaxed),
            pool_exhausted: POOL_EXHAUSTED.load(Ordering::Relaxed),
            pool_too_large: POOL_TOO_LARGE.load(Ordering::Relaxed),
            requests_published: REQUESTS_PUBLISHED.load(Ordering::Relaxed),
            batches_submitted: BATCHES_SUBMITTED.load(Ordering::Relaxed),
            vectors_submitted: VECTORS_SUBMITTED.load(Ordering::Relaxed),
            batches_completed: BATCHES_COMPLETED.load(Ordering::Relaxed),
            slots_submitted: SLOTS_SUBMITTED.load(Ordering::Relaxed),
            backlog_slots_at_build: BACKLOG_SLOTS_AT_BUILD.load(Ordering::Relaxed),
            batch_stop_cap: BATCH_STOP_CAP.load(Ordering::Relaxed),
            batch_stop_backlog_empty: BATCH_STOP_BACKLOG_EMPTY.load(Ordering::Relaxed),
            batch_stop_non_contig: BATCH_STOP_NON_CONTIG.load(Ordering::Relaxed),
            responses_written: RESPONSES_WRITTEN.load(Ordering::Relaxed),
            read_submits: READ_SUBMITS.load(Ordering::Relaxed),
            read_cqes: READ_CQES.load(Ordering::Relaxed),
            read_bytes: READ_BYTES.load(Ordering::Relaxed),
            read_negative: READ_NEGATIVE.load(Ordering::Relaxed),
            bytes_consumed: BYTES_CONSUMED.load(Ordering::Relaxed),
            write_sqes: WRITE_SQES.load(Ordering::Relaxed),
            write_cqes: WRITE_CQES.load(Ordering::Relaxed),
            write_negative: WRITE_NEGATIVE.load(Ordering::Relaxed),
            session_wait_loops: SESSION_WAIT_LOOPS.load(Ordering::Relaxed),
            completion_queue_empty_spins: COMPLETION_QUEUE_EMPTY_SPINS.load(Ordering::Relaxed),
            completion_poll_no_events: COMPLETION_POLL_NO_EVENTS.load(Ordering::Relaxed),
            write_drain_waits: WRITE_DRAIN_WAITS.load(Ordering::Relaxed),
            write_partial: WRITE_PARTIAL.load(Ordering::Relaxed),
            write_eagain: WRITE_EAGAIN.load(Ordering::Relaxed),
            write_fatal: WRITE_FATAL.load(Ordering::Relaxed),
            pool_max_in_use: POOL_MAX_IN_USE.load(Ordering::Relaxed),
            req_occ: REQ_OCC.load(Ordering::Relaxed),
            req_max_occ: REQ_MAX_OCC.load(Ordering::Relaxed),
            buffered_bytes: BUFFERED_BYTES.load(Ordering::Relaxed),
        }
    }

    pub fn spawn_reporter(interval_secs: u64, metrics_cpu: Option<usize>) {
        assert!(interval_secs > 0, "metrics interval must be > 0");
        std::thread::Builder::new()
            .name("metrics".into())
            .spawn(move || {
                if let Some(cpu) = metrics_cpu {
                    affinity::pin_current_thread(cpu, "metrics")
                        .unwrap_or_else(|e| panic!("{e}"));
                }
                let mut last_snap = snapshot();
                loop {
                    std::thread::sleep(Duration::from_secs(interval_secs));
                    let snap = snapshot();
                    let req_full_d = snap.req_ring_full.saturating_sub(last_snap.req_ring_full);
                    let pool_exh_d = snap.pool_exhausted.saturating_sub(last_snap.pool_exhausted);
                    let pool_tl_d = snap.pool_too_large.saturating_sub(last_snap.pool_too_large);
                    let req_pub_d = snap
                        .requests_published
                        .saturating_sub(last_snap.requests_published);
                    let batches_submitted_d = snap
                        .batches_submitted
                        .saturating_sub(last_snap.batches_submitted);
                    let vectors_submitted_d = snap
                        .vectors_submitted
                        .saturating_sub(last_snap.vectors_submitted);
                    let batches_completed_d = snap
                        .batches_completed
                        .saturating_sub(last_snap.batches_completed);
                    let slots_submitted_d = snap
                        .slots_submitted
                        .saturating_sub(last_snap.slots_submitted);
                    let backlog_slots_at_build_d = snap
                        .backlog_slots_at_build
                        .saturating_sub(last_snap.backlog_slots_at_build);
                    let batch_stop_cap_d =
                        snap.batch_stop_cap.saturating_sub(last_snap.batch_stop_cap);
                    let batch_stop_backlog_empty_d = snap
                        .batch_stop_backlog_empty
                        .saturating_sub(last_snap.batch_stop_backlog_empty);
                    let batch_stop_non_contig_d = snap
                        .batch_stop_non_contig
                        .saturating_sub(last_snap.batch_stop_non_contig);
                    let responses_written_d = snap
                        .responses_written
                        .saturating_sub(last_snap.responses_written);
                    let read_submits_d = snap.read_submits.saturating_sub(last_snap.read_submits);
                    let read_cqes_d = snap.read_cqes.saturating_sub(last_snap.read_cqes);
                    let read_bytes_d = snap.read_bytes.saturating_sub(last_snap.read_bytes);
                    let read_negative_d =
                        snap.read_negative.saturating_sub(last_snap.read_negative);
                    let bytes_consumed_d =
                        snap.bytes_consumed.saturating_sub(last_snap.bytes_consumed);
                    let write_sqes_d = snap.write_sqes.saturating_sub(last_snap.write_sqes);
                    let write_cqes_d = snap.write_cqes.saturating_sub(last_snap.write_cqes);
                    let write_negative_d =
                        snap.write_negative.saturating_sub(last_snap.write_negative);
                    let session_wait_loops_d = snap
                        .session_wait_loops
                        .saturating_sub(last_snap.session_wait_loops);
                    let completion_queue_empty_spins_d = snap
                        .completion_queue_empty_spins
                        .saturating_sub(last_snap.completion_queue_empty_spins);
                    let completion_poll_no_events_d = snap
                        .completion_poll_no_events
                        .saturating_sub(last_snap.completion_poll_no_events);
                    let write_drain_waits_d = snap
                        .write_drain_waits
                        .saturating_sub(last_snap.write_drain_waits);
                    let write_partial_d =
                        snap.write_partial.saturating_sub(last_snap.write_partial);
                    let write_eagain_d =
                        snap.write_eagain.saturating_sub(last_snap.write_eagain);
                    let write_fatal_d = snap.write_fatal.saturating_sub(last_snap.write_fatal);
                    let batch_total = batch_total_timer().snapshot_and_reset();
                    let batch_wait = batch_wait_timer().snapshot_and_reset();
                    let backlog_age = backlog_age_timer().snapshot_and_reset();
                    let publish_to_submit = publish_to_submit_timer().snapshot_and_reset();
                    let publish_to_write_submit =
                        publish_to_write_submit_timer().snapshot_and_reset();
                    let write_drain = write_drain_timer().snapshot_and_reset();
                    println!(
                        "metrics delta {}s: req_published={} batches_submitted={} batches_completed={} slots_submitted={} backlog_slots_at_build={} vectors_submitted={} responses_written={} | batch_build: stop_cap={} stop_empty={} stop_noncontig={} {} {} {} | reads: submits={} cqes={} bytes={} neg={} consumed={} | writes: sqes={} cqes={} neg={} partial={} eagain={} fatal={} drain_waits={} {} | stalls: req_ring_full={} pool_exh={} pool_too_large={} session_waits={} completion_queue_empty={} completion_poll_no_events={} | gauges: req_occ={} req_max={} buffered_bytes={} pool_max_in_use={} {} {}",
                        interval_secs,
                        req_pub_d,
                        batches_submitted_d,
                        batches_completed_d,
                        slots_submitted_d,
                        backlog_slots_at_build_d,
                        vectors_submitted_d,
                        responses_written_d,
                        batch_stop_cap_d,
                        batch_stop_backlog_empty_d,
                        batch_stop_non_contig_d,
                        format_timer("backlog_age_us", backlog_age.as_ref()),
                        format_timer("publish_to_submit_us", publish_to_submit.as_ref()),
                        format_timer(
                            "publish_to_write_submit_us",
                            publish_to_write_submit.as_ref()
                        ),
                        read_submits_d,
                        read_cqes_d,
                        read_bytes_d,
                        read_negative_d,
                        bytes_consumed_d,
                        write_sqes_d,
                        write_cqes_d,
                        write_negative_d,
                        write_partial_d,
                        write_eagain_d,
                        write_fatal_d,
                        write_drain_waits_d,
                        format_timer("write_drain_us", write_drain.as_ref()),
                        req_full_d,
                        pool_exh_d,
                        pool_tl_d,
                        session_wait_loops_d,
                        completion_queue_empty_spins_d,
                        completion_poll_no_events_d,
                        snap.req_occ,
                        snap.req_max_occ,
                        snap.buffered_bytes,
                        snap.pool_max_in_use,
                        format_timer("batch_total_us", batch_total.as_ref()),
                        format_timer("batch_wait_us", batch_wait.as_ref()),
                    );
                    last_snap = snap;
                }
            })
            .expect("failed to spawn metrics reporter");
    }

    fn format_timer(label: &str, snapshot: Option<&TimerSnapshot>) -> String {
        match snapshot {
            Some(snapshot) => format!(
                "{}[n={} p50={:.1} p95={:.1} p99={:.1} p99.9={:.1} max={:.1}]",
                label,
                snapshot.count(),
                snapshot.p50_us(),
                snapshot.p95_us(),
                snapshot.p99_us(),
                snapshot.p999_us(),
                snapshot.max_us(),
            ),
            None => format!("{}[n=0]", label),
        }
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
        pub batches_submitted: u64,
        pub vectors_submitted: u64,
        pub batches_completed: u64,
        pub slots_submitted: u64,
        pub backlog_slots_at_build: u64,
        pub batch_stop_cap: u64,
        pub batch_stop_backlog_empty: u64,
        pub batch_stop_non_contig: u64,
        pub responses_written: u64,
        pub read_submits: u64,
        pub read_cqes: u64,
        pub read_bytes: u64,
        pub read_negative: u64,
        pub bytes_consumed: u64,
        pub write_sqes: u64,
        pub write_cqes: u64,
        pub write_negative: u64,
        pub session_wait_loops: u64,
        pub completion_queue_empty_spins: u64,
        pub completion_poll_no_events: u64,
        pub write_drain_waits: u64,
        pub write_partial: u64,
        pub write_eagain: u64,
        pub write_fatal: u64,
        pub pool_max_in_use: usize,
        pub req_occ: usize,
        pub req_max_occ: usize,
        pub buffered_bytes: usize,
    }

    pub fn inc_req_ring_full() {}
    pub fn inc_pool_exhausted() {}
    pub fn inc_pool_too_large() {}
    pub fn inc_session_wait_loops() {}
    pub fn inc_completion_queue_empty_spins() {}
    pub fn inc_completion_poll_no_events() {}
    pub fn inc_write_drain_waits() {}
    pub fn inc_write_partial() {}
    pub fn inc_write_eagain() {}
    pub fn inc_write_fatal() {}
    pub fn update_pool_in_use(_: usize) {}
    pub fn inc_req_occ() {}
    pub fn dec_req_occ() {}
    pub fn inc_requests_published() {}
    pub fn inc_batches_submitted() {}
    pub fn add_vectors_submitted(_: u64) {}
    pub fn inc_batches_completed() {}
    pub fn add_slots_submitted(_: u64) {}
    pub fn add_backlog_slots_at_build(_: u64) {}
    pub fn inc_batch_stop_cap() {}
    pub fn inc_batch_stop_backlog_empty() {}
    pub fn inc_batch_stop_non_contig() {}
    pub fn inc_responses_written() {}
    pub fn inc_read_submits() {}
    pub fn inc_read_cqes() {}
    pub fn add_read_bytes(_: u64) {}
    pub fn inc_read_negative() {}
    pub fn add_bytes_consumed(_: u64) {}
    pub fn inc_write_sqes() {}
    pub fn inc_write_cqes() {}
    pub fn inc_write_negative() {}
    pub fn record_batch_total(_: std::time::Duration) {}
    pub fn record_batch_wait(_: std::time::Duration) {}
    pub fn record_backlog_age(_: std::time::Duration) {}
    pub fn record_publish_to_submit(_: std::time::Duration) {}
    pub fn record_publish_to_write_submit(_: std::time::Duration) {}
    pub fn record_write_drain(_: std::time::Duration) {}
    pub fn idle_timers() {}
    pub fn set_buffered_bytes(_: usize) {}
    pub fn snapshot() -> MetricsSnapshot {
        MetricsSnapshot {
            req_ring_full: 0,
            pool_exhausted: 0,
            pool_too_large: 0,
            requests_published: 0,
            batches_submitted: 0,
            vectors_submitted: 0,
            batches_completed: 0,
            slots_submitted: 0,
            backlog_slots_at_build: 0,
            batch_stop_cap: 0,
            batch_stop_backlog_empty: 0,
            batch_stop_non_contig: 0,
            responses_written: 0,
            read_submits: 0,
            read_cqes: 0,
            read_bytes: 0,
            read_negative: 0,
            bytes_consumed: 0,
            write_sqes: 0,
            write_cqes: 0,
            write_negative: 0,
            session_wait_loops: 0,
            completion_queue_empty_spins: 0,
            completion_poll_no_events: 0,
            write_drain_waits: 0,
            write_partial: 0,
            write_eagain: 0,
            write_fatal: 0,
            pool_max_in_use: 0,
            req_occ: 0,
            req_max_occ: 0,
            buffered_bytes: 0,
        }
    }
    pub fn spawn_reporter(_: u64, _: Option<usize>) {}
}

pub use imp::*;
