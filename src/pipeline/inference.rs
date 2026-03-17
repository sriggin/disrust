use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use disruptor::{EventGuard, EventPoller, MultiProducerBarrier, Polling, SingleConsumerBarrier};

use crate::buffer_pool::PoolSlice;
use crate::clock::elapsed_since_ns;
use crate::config::MAX_SESSION_BATCH_SIZE;
use crate::metrics;
use crate::pipeline::connection_registry::ConnectionRegistry;
use crate::pipeline::response_queue::{ResponseQueue, ResponseReady};
use crate::pipeline::session::{BatchPoll, InferenceBackend, InFlightBatch};
use crate::protocol;
use crate::ring_types::InferenceEvent;

const MAX_COMPLETIONS_PER_PASS: usize = 8;
const MAX_SUBMISSIONS_PER_PASS: usize = 8;

struct BatchEntry<R: Send> {
    slot_count: usize,
    #[cfg(feature = "metrics")]
    submitted_at: Instant,
    batch: InFlightBatch<R>,
}

struct PendingSlot {
    features: PoolSlice,
    num_vectors: usize,
    published_at_ns: u64,
}

enum BatchStopReason {
    Cap,
    BacklogEmpty,
    NonContiguous,
}

struct InflightBatchEntry<R: Send> {
    entry: BatchEntry<R>,
    #[cfg(feature = "metrics")]
    wait_started_at: Option<Instant>,
}

pub struct InferenceConsumer<B: InferenceBackend> {
    submission_poller: EventPoller<InferenceEvent, MultiProducerBarrier>,
    completion_poller: EventPoller<InferenceEvent, SingleConsumerBarrier>,
    backend: B,
    backlog: VecDeque<PendingSlot>,
    inflight: VecDeque<InflightBatchEntry<B::Resources>>,
    response_queues: Vec<Arc<ResponseQueue>>,
    registry: Arc<ConnectionRegistry>,
    max_batch_slots: usize,
    batch_coalesce_timeout: Duration,
    backlog_started_at: Option<Instant>,
    coalesce_check_spins: u32,
    timers_idle: bool,
}

unsafe impl<B: InferenceBackend> Send for InferenceConsumer<B> {}

impl<B: InferenceBackend> InferenceConsumer<B> {
    pub fn new(
        submission_poller: EventPoller<InferenceEvent, MultiProducerBarrier>,
        completion_poller: EventPoller<InferenceEvent, SingleConsumerBarrier>,
        backend: B,
        response_queues: Vec<Arc<ResponseQueue>>,
        registry: Arc<ConnectionRegistry>,
        max_batch_slots: usize,
        batch_coalesce_timeout: Duration,
    ) -> Self {
        Self {
            submission_poller,
            completion_poller,
            backend,
            backlog: VecDeque::new(),
            inflight: VecDeque::new(),
            response_queues,
            registry,
            max_batch_slots,
            batch_coalesce_timeout,
            backlog_started_at: None,
            coalesce_check_spins: 0,
            timers_idle: false,
        }
    }

    pub fn run(self) {
        self.run_inner(None);
    }

    pub fn run_until(self, stop: Arc<AtomicBool>) {
        self.run_inner(Some(stop));
    }

    fn run_inner(mut self, stop: Option<Arc<AtomicBool>>) {
        let mut idle_loops = 0u32;
        loop {
            if stop_requested(stop.as_ref()) {
                return;
            }
            let mut progressed = false;

            let backlog_was_empty = self.backlog.is_empty();
            match drain_visible_events(
                &mut self.submission_poller,
                &mut self.backlog,
                self.max_batch_slots,
            ) {
                Ok(()) => {}
                Err(Polling::Shutdown) => return,
                Err(Polling::NoEvents) => {
                    unreachable!("drain_visible_events does not return NoEvents")
                }
            }
            if backlog_was_empty && !self.backlog.is_empty() {
                self.backlog_started_at = Some(Instant::now());
                self.coalesce_check_spins = 0;
                progressed = true;
            } else if !self.backlog.is_empty() {
                progressed = true;
            }

            for _ in 0..MAX_COMPLETIONS_PER_PASS {
                match self.try_complete_front() {
                    Ok(true) => progressed = true,
                    Ok(false) => break,
                    Err(Polling::Shutdown) => return,
                    Err(Polling::NoEvents) => {
                        unreachable!("try_complete_front never returns NoEvents")
                    }
                }
            }

            for _ in 0..MAX_SUBMISSIONS_PER_PASS {
                if self.try_submit_next() {
                    progressed = true;
                } else {
                    break;
                }
            }

            if !progressed {
                if !self.timers_idle {
                    metrics::idle_timers();
                    self.timers_idle = true;
                }
                if self.inflight.is_empty() {
                    metrics::inc_completion_queue_empty_waits();
                }
                idle_wait(&mut idle_loops);
            } else {
                self.timers_idle = false;
                idle_loops = 0;
            }
        }
    }

    fn try_submit_next(&mut self) -> bool {
        if self.backlog.is_empty() {
            self.backlog_started_at = None;
            self.coalesce_check_spins = 0;
            return false;
        }

        if self.backlog.len() < self.max_batch_slots {
            if !self.backend.is_available() {
                return false;
            }
            if self.batch_coalesce_timeout > Duration::ZERO && self.coalesce_check_spins < 64 {
                self.coalesce_check_spins += 1;
                return false;
            }
            self.coalesce_check_spins = 0;
            if self
                .backlog_started_at
                .is_some_and(|started| started.elapsed() < self.batch_coalesce_timeout)
            {
                return false;
            }
        }

        if !self.backend.try_acquire() {
            if self.backlog.len() >= self.max_batch_slots {
                metrics::inc_session_waits();
            }
            return false;
        }

        if let Some(started) = self.backlog_started_at {
            metrics::record_backlog_age(started.elapsed());
        }
        let batch_entry = build_batch_entry(&mut self.backend, &mut self.backlog, self.max_batch_slots);
        metrics::inc_batches_submitted();
        metrics::add_vectors_submitted(batch_entry.batch.output_len as u64);
        if self.backlog.is_empty() {
            self.backlog_started_at = None;
        }
        self.coalesce_check_spins = 0;
        self.inflight.push_back(InflightBatchEntry {
            entry: batch_entry,
            #[cfg(feature = "metrics")]
            wait_started_at: None,
        });
        true
    }

    fn try_complete_front(&mut self) -> Result<bool, Polling> {
        let Some(front) = self.inflight.front_mut() else {
            return Ok(false);
        };

        match front.entry.batch.completion.poll() {
            BatchPoll::Pending => {
                #[cfg(feature = "metrics")]
                front.wait_started_at.get_or_insert_with(Instant::now);
                Ok(false)
            }
            BatchPoll::Ready => {
                let inflight = self.inflight.pop_front().expect("front just checked");
                #[cfg(feature = "metrics")]
                metrics::record_batch_wait(
                    inflight
                        .wait_started_at
                        .map(|started| started.elapsed())
                        .unwrap_or(Duration::ZERO),
                );
                #[cfg(not(feature = "metrics"))]
                metrics::record_batch_wait(Duration::ZERO);
                let response_queues = self.response_queues.clone();
                let registry = Arc::clone(&self.registry);
                let max_batch_slots = self.max_batch_slots;
                let mut guard = wait_for_completion_guard(
                    &mut self.completion_poller,
                    inflight.entry.slot_count,
                )?;
                process_batch(
                    &mut guard,
                    inflight.entry,
                    &response_queues,
                    &registry,
                    max_batch_slots,
                );
                Ok(true)
            }
            BatchPoll::Failed => {
                let inflight = self.inflight.pop_front().expect("front just checked");
                #[cfg(feature = "metrics")]
                metrics::record_batch_wait(
                    inflight
                        .wait_started_at
                        .map(|started| started.elapsed())
                        .unwrap_or(Duration::ZERO),
                );
                #[cfg(not(feature = "metrics"))]
                metrics::record_batch_wait(Duration::ZERO);
                inflight.entry.batch.completion.wait();
                unreachable!("BatchCompletion::wait aborts on failure")
            }
        }
    }
}

fn idle_wait(idle_loops: &mut u32) {
    *idle_loops = idle_loops.saturating_add(1);
    if *idle_loops < 64 {
        std::hint::spin_loop();
    } else if *idle_loops < 256 {
        thread::yield_now();
    } else {
        thread::sleep(Duration::from_micros(10));
    }
}

fn drain_visible_events(
    poller: &mut EventPoller<InferenceEvent, MultiProducerBarrier>,
    backlog: &mut VecDeque<PendingSlot>,
    max_batch_slots: usize,
) -> Result<(), Polling> {
    while backlog.len() < max_batch_slots {
        let remaining = (max_batch_slots - backlog.len()) as u64;
        match poller.poll_take(remaining) {
            Ok(mut guard) => drain_guard(&mut guard, backlog),
            Err(Polling::NoEvents) => return Ok(()),
            Err(Polling::Shutdown) => return Err(Polling::Shutdown),
        }
    }
    Ok(())
}

fn drain_guard(
    guard: &mut EventGuard<'_, InferenceEvent, MultiProducerBarrier>,
    backlog: &mut VecDeque<PendingSlot>,
) {
    for event in &mut *guard {
        backlog.push_back(PendingSlot {
            features: take_event_features(event),
            num_vectors: event.num_vectors as usize,
            published_at_ns: event.published_at_ns,
        });
    }
}

fn take_event_features(event: &InferenceEvent) -> PoolSlice {
    unsafe {
        let event_ptr = event as *const InferenceEvent as *mut InferenceEvent;
        std::ptr::replace(
            std::ptr::addr_of_mut!((*event_ptr).features),
            PoolSlice::empty(),
        )
    }
}

fn build_batch_entry<B: InferenceBackend>(
    backend: &mut B,
    backlog: &mut VecDeque<PendingSlot>,
    max_batch_slots: usize,
) -> BatchEntry<B::Resources> {
    debug_assert!(max_batch_slots > 0);
    debug_assert!(max_batch_slots <= MAX_SESSION_BATCH_SIZE);
    let first = backlog
        .front()
        .expect("build_batch_entry requires non-empty backlog");
    let host_ptr = first.features.as_slice().as_ptr();

    let mut slot_count = 0usize;
    let mut num_vectors = 0usize;
    let mut input_slices = Vec::new();
    let backlog_slots_at_build = backlog.len() as u64;
    let mut stop_reason = None;

    while let Some(next) = backlog.front() {
        let contiguous = input_slices
            .last()
            .is_none_or(|prev: &PoolSlice| prev.is_contiguous(&next.features));
        if !contiguous {
            stop_reason = Some(BatchStopReason::NonContiguous);
            break;
        }
        if slot_count >= max_batch_slots {
            stop_reason = Some(BatchStopReason::Cap);
            break;
        }

        let next = backlog.pop_front().expect("front just checked");
        metrics::record_publish_to_submit(elapsed_since_ns(next.published_at_ns));
        num_vectors += next.num_vectors;
        slot_count += 1;
        input_slices.push(next.features);
    }

    let stop_reason = if backlog.front().is_none() {
        BatchStopReason::BacklogEmpty
    } else {
        stop_reason.expect("non-empty backlog must have a stop reason")
    };

    metrics::add_backlog_slots_at_build(backlog_slots_at_build);
    metrics::add_slots_submitted(slot_count as u64);
    match stop_reason {
        BatchStopReason::Cap => metrics::inc_batch_stop_cap(),
        BatchStopReason::BacklogEmpty => metrics::inc_batch_stop_backlog_empty(),
        BatchStopReason::NonContiguous => metrics::inc_batch_stop_non_contig(),
    }

    let mut batch = backend.submit_batch(host_ptr, num_vectors);
    batch.input_slices = input_slices;
    BatchEntry {
        slot_count,
        #[cfg(feature = "metrics")]
        submitted_at: Instant::now(),
        batch,
    }
}

fn stop_requested(stop: Option<&Arc<AtomicBool>>) -> bool {
    stop.is_some_and(|flag| flag.load(Ordering::Relaxed))
}

fn wait_for_completion_guard<'a>(
    poller: &'a mut EventPoller<InferenceEvent, SingleConsumerBarrier>,
    slot_count: usize,
) -> Result<EventGuard<'a, InferenceEvent, SingleConsumerBarrier>, Polling> {
    let mut polled = false;
    let poller_ptr: *mut EventPoller<InferenceEvent, SingleConsumerBarrier> = poller;
    loop {
        match unsafe { (*poller_ptr).poll_take(slot_count as u64) } {
            Ok(guard) => return Ok(guard),
            Err(Polling::NoEvents) => {
                if !polled {
                    metrics::inc_completion_poll_stalls();
                    polled = true;
                }
                std::hint::spin_loop();
            }
            Err(Polling::Shutdown) => return Err(Polling::Shutdown),
        }
    }
}

fn process_batch<R: Send>(
    guard: &mut EventGuard<'_, InferenceEvent, SingleConsumerBarrier>,
    entry: BatchEntry<R>,
    response_queues: &[Arc<ResponseQueue>],
    registry: &Arc<ConnectionRegistry>,
    max_batch_slots: usize,
) {
    let mut guard_ref = &mut *guard;
    let output =
        unsafe { std::slice::from_raw_parts(entry.batch.output_ptr, entry.batch.output_len) };
    let mut output_offset = 0usize;

    debug_assert!(entry.slot_count <= max_batch_slots);
    for _ in 0..entry.slot_count {
        let event = guard_ref
            .next()
            .expect("guard exhausted before queued batch slot_count");
        let num_vecs = event.num_vectors as usize;

        let wire_len = protocol::response_size(num_vecs);
        let mut wire = vec![0u8; wire_len];
        let response = &output[output_offset..output_offset + num_vecs];
        protocol::encode_response(response, &mut wire);

        let conn = event.conn;
        if registry.is_open(conn) {
            response_queues[conn.shard_id() as usize].push(ResponseReady::new(
                conn,
                event.request_seq,
                event.published_at_ns,
                &wire,
            ));
        }

        output_offset += num_vecs;
        metrics::inc_responses_written();
        metrics::dec_req_occ();
    }

    debug_assert_eq!(
        output_offset, entry.batch.output_len,
        "slot_count/num_vectors mismatch between ring events and batch output"
    );

    let session_available = Arc::clone(&entry.batch.session_available);
    #[cfg(feature = "metrics")]
    metrics::record_batch_total(entry.submitted_at.elapsed());
    drop(entry);
    session_available.store(true, Ordering::Release);
    metrics::inc_batches_completed();
}
