//! SubmissionConsumer: drains request-ring events into a local backlog, forms batches,
//! submits to ORT, and pushes BatchEntry values to the batch queue.

use std::collections::VecDeque;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use disruptor::{EventGuard, EventPoller, Polling, SingleProducerBarrier};

use crate::buffer_pool::PoolSlice;
use crate::clock::elapsed_since_ns;
use crate::config::MAX_SESSION_BATCH_SIZE;
use crate::metrics;
use crate::pipeline::batch_queue::{BatchEntry, BatchQueue};
use crate::ring_types::InferenceEvent;

use super::session::InferenceSession;

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

pub struct SubmissionConsumer {
    poller: EventPoller<InferenceEvent, SingleProducerBarrier>,
    sessions: Vec<InferenceSession>,
    /// Round-robin session index.
    session_cursor: usize,
    batch_queue: Arc<BatchQueue>,
    backlog: VecDeque<PendingSlot>,
    max_batch_slots: usize,
    batch_coalesce_timeout: Duration,
    backlog_started_at: Option<Instant>,
    coalesce_check_spins: u32,
}

// SAFETY: SubmissionConsumer runs on a single dedicated thread.
unsafe impl Send for SubmissionConsumer {}

impl SubmissionConsumer {
    pub fn new(
        poller: EventPoller<InferenceEvent, SingleProducerBarrier>,
        sessions: Vec<InferenceSession>,
        batch_queue: Arc<BatchQueue>,
        max_batch_slots: usize,
        batch_coalesce_timeout: Duration,
    ) -> Self {
        assert!(max_batch_slots > 0, "max_batch_slots must be > 0");
        assert!(
            max_batch_slots <= MAX_SESSION_BATCH_SIZE,
            "max_batch_slots ({max_batch_slots}) exceeds compile-time limit ({MAX_SESSION_BATCH_SIZE})"
        );
        Self {
            poller,
            sessions,
            session_cursor: 0,
            batch_queue,
            backlog: VecDeque::new(),
            max_batch_slots,
            batch_coalesce_timeout,
            backlog_started_at: None,
            coalesce_check_spins: 0,
        }
    }

    pub fn run(mut self) {
        let mut idle_loops = 0u32;
        loop {
            let backlog_was_empty = self.backlog.is_empty();
            match drain_visible_events(&mut self.poller, &mut self.backlog, self.max_batch_slots) {
                Ok(()) => {}
                Err(Polling::Shutdown) => return,
                Err(Polling::NoEvents) => {
                    unreachable!("drain_visible_events does not return NoEvents")
                }
            }
            if backlog_was_empty && !self.backlog.is_empty() {
                self.backlog_started_at = Some(Instant::now());
                self.coalesce_check_spins = 0;
            }
            if self.backlog.is_empty() {
                self.backlog_started_at = None;
                self.coalesce_check_spins = 0;
                idle_wait(&mut idle_loops);
                continue;
            }

            if self.backlog.len() < self.max_batch_slots {
                if !session_available(&self.sessions) {
                    idle_wait(&mut idle_loops);
                    continue;
                }
                if self.batch_coalesce_timeout > Duration::ZERO && self.coalesce_check_spins < 64 {
                    self.coalesce_check_spins += 1;
                    idle_wait(&mut idle_loops);
                    continue;
                }
                self.coalesce_check_spins = 0;
                if self
                    .backlog_started_at
                    .is_some_and(|started| started.elapsed() < self.batch_coalesce_timeout)
                {
                    idle_wait(&mut idle_loops);
                    continue;
                }
            }

            let idx = if self.backlog.len() >= self.max_batch_slots {
                reserve_session(&self.sessions, &mut self.session_cursor)
            } else if let Some(idx) = try_reserve_session(&self.sessions, &mut self.session_cursor)
            {
                idx
            } else {
                std::hint::spin_loop();
                continue;
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
            idle_loops = 0;
            self.batch_queue.push(batch_entry);
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
    poller: &mut EventPoller<InferenceEvent, SingleProducerBarrier>,
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
    guard: &mut EventGuard<'_, InferenceEvent, SingleProducerBarrier>,
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
    // SAFETY: while SubmissionConsumer holds the poll guard, no later consumer can
    // observe these slots yet and the producer cannot reuse them. That makes it
    // safe to move `features` out into the submission backlog before guard drop.
    unsafe {
        let event_ptr = event as *const InferenceEvent as *mut InferenceEvent;
        std::ptr::replace(
            std::ptr::addr_of_mut!((*event_ptr).features),
            PoolSlice::empty(),
        )
    }
}

fn build_batch_entry(
    session: &mut InferenceSession,
    backlog: &mut VecDeque<PendingSlot>,
    max_batch_slots: usize,
) -> BatchEntry {
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

    let mut batch = session.submit_batch(host_ptr, num_vectors);
    batch.input_slices = input_slices;

    BatchEntry {
        slot_count,
        submitted_at: Instant::now(),
        batch,
    }
}

fn reserve_session(sessions: &[InferenceSession], session_cursor: &mut usize) -> usize {
    loop {
        if let Some(idx) = try_reserve_session(sessions, session_cursor) {
            return idx;
        }
        metrics::inc_session_wait_loops();
        std::hint::spin_loop();
    }
}

fn try_reserve_session(sessions: &[InferenceSession], session_cursor: &mut usize) -> Option<usize> {
    for _ in 0..sessions.len() {
        let idx = *session_cursor;
        *session_cursor = (idx + 1) % sessions.len();
        if sessions[idx].try_acquire() {
            return Some(idx);
        }
    }
    None
}

fn session_available(sessions: &[InferenceSession]) -> bool {
    sessions.iter().any(InferenceSession::is_available)
}
