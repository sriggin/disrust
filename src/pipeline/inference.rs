use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use disruptor::{EventGuard, EventPoller, MultiProducerBarrier, Polling, SingleConsumerBarrier};

use crate::metrics;
use crate::pipeline::batch_queue::BatchEntry;
use crate::pipeline::connection_registry::ConnectionRegistry;
use crate::pipeline::ready_queue::ReadyQueue;
use crate::pipeline::session::{BatchPoll, InferenceSession};
use crate::ring_types::InferenceEvent;

use super::completion::process_batch;
use super::submission::{
    PendingSlot, build_batch_entry, drain_visible_events, idle_wait, session_available,
    try_reserve_session,
};

const MAX_COMPLETIONS_PER_PASS: usize = 8;
const MAX_SUBMISSIONS_PER_PASS: usize = 8;

struct InflightBatchEntry {
    entry: BatchEntry,
    #[cfg(feature = "metrics")]
    wait_started_at: Option<Instant>,
}

pub struct InferenceConsumer {
    submission_poller: EventPoller<InferenceEvent, MultiProducerBarrier>,
    completion_poller: EventPoller<InferenceEvent, SingleConsumerBarrier>,
    sessions: Vec<InferenceSession>,
    session_cursor: usize,
    backlog: VecDeque<PendingSlot>,
    inflight: VecDeque<InflightBatchEntry>,
    ready_queue: Arc<ReadyQueue>,
    registry: Arc<ConnectionRegistry>,
    max_batch_slots: usize,
    batch_coalesce_timeout: Duration,
    backlog_started_at: Option<Instant>,
    coalesce_check_spins: u32,
    timers_idle: bool,
}

unsafe impl Send for InferenceConsumer {}

impl InferenceConsumer {
    pub fn new(
        submission_poller: EventPoller<InferenceEvent, MultiProducerBarrier>,
        completion_poller: EventPoller<InferenceEvent, SingleConsumerBarrier>,
        sessions: Vec<InferenceSession>,
        ready_queue: Arc<ReadyQueue>,
        registry: Arc<ConnectionRegistry>,
        max_batch_slots: usize,
        batch_coalesce_timeout: Duration,
    ) -> Self {
        Self {
            submission_poller,
            completion_poller,
            sessions,
            session_cursor: 0,
            backlog: VecDeque::new(),
            inflight: VecDeque::new(),
            ready_queue,
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
                    Err(Polling::NoEvents) => unreachable!("try_complete_front never returns NoEvents"),
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
            if !session_available(&self.sessions) {
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

        let Some(idx) = try_reserve_session(&self.sessions, &mut self.session_cursor) else {
            if self.backlog.len() >= self.max_batch_slots {
                metrics::inc_session_waits();
            }
            return false;
        };

        if let Some(started) = self.backlog_started_at {
            metrics::record_backlog_age(started.elapsed());
        }
        let batch_entry = build_batch_entry(
            &mut self.sessions[idx],
            &mut self.backlog,
            self.max_batch_slots,
        );
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
                let ready_queue = Arc::clone(&self.ready_queue);
                let registry = Arc::clone(&self.registry);
                let max_batch_slots = self.max_batch_slots;
                let mut guard = wait_for_completion_guard(
                    &mut self.completion_poller,
                    inflight.entry.slot_count,
                )?;
                process_batch(
                    &mut guard,
                    inflight.entry,
                    &ready_queue,
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
        // SAFETY: InferenceConsumer owns the completion poller on a single thread. This helper retries
        // until a guard is produced, then returns that guard directly.
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
