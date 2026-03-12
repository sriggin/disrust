//! SubmissionConsumer: drains request-ring events into a local backlog, forms GPU batches,
//! submits to ORT, and pushes BatchEntry values to the batch queue.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use disruptor::{EventGuard, EventPoller, Polling, SingleProducerBarrier};

use crate::buffer_pool::PoolSlice;
use crate::config::MAX_SESSION_BATCH_SIZE;
use crate::gpu::batch_queue::{BatchEntry, BatchQueue};
use crate::metrics;
use crate::ring_types::InferenceEvent;

use super::session::GpuSession;

struct PendingSlot {
    features: PoolSlice,
    num_vectors: usize,
}

pub struct SubmissionConsumer {
    poller: EventPoller<InferenceEvent, SingleProducerBarrier>,
    sessions: Vec<GpuSession>,
    /// Round-robin session index.
    session_cursor: usize,
    batch_queue: Arc<BatchQueue>,
    backlog: VecDeque<PendingSlot>,
    max_batch_slots: usize,
}

// SAFETY: SubmissionConsumer runs on a single dedicated thread.
unsafe impl Send for SubmissionConsumer {}

impl SubmissionConsumer {
    pub fn new(
        poller: EventPoller<InferenceEvent, SingleProducerBarrier>,
        sessions: Vec<GpuSession>,
        batch_queue: Arc<BatchQueue>,
        max_batch_slots: usize,
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
        }
    }

    pub fn run(mut self) {
        loop {
            match drain_visible_events(&mut self.poller, &mut self.backlog, self.max_batch_slots) {
                Ok(()) => {}
                Err(Polling::Shutdown) => return,
                Err(Polling::NoEvents) => {
                    unreachable!("drain_visible_events does not return NoEvents")
                }
            }
            if self.backlog.is_empty() {
                std::hint::spin_loop();
                continue;
            }

            let idx = reserve_session(&self.sessions, &mut self.session_cursor);
            let batch_entry = build_batch_entry(
                &mut self.sessions[idx],
                &mut self.backlog,
                self.max_batch_slots,
            );
            metrics::inc_batches_submitted();
            metrics::add_vectors_submitted(batch_entry.batch.output_len as u64);
            self.batch_queue.push(batch_entry);
        }
    }
}

fn drain_visible_events(
    poller: &mut EventPoller<InferenceEvent, SingleProducerBarrier>,
    backlog: &mut VecDeque<PendingSlot>,
    max_batch_slots: usize,
) -> Result<(), Polling> {
    loop {
        match poller.poll_take(max_batch_slots as u64) {
            Ok(mut guard) => drain_guard(&mut guard, backlog),
            Err(Polling::NoEvents) => return Ok(()),
            Err(Polling::Shutdown) => return Err(Polling::Shutdown),
        }
    }
}

fn drain_guard(
    guard: &mut EventGuard<'_, InferenceEvent, SingleProducerBarrier>,
    backlog: &mut VecDeque<PendingSlot>,
) {
    for event in &mut *guard {
        backlog.push_back(PendingSlot {
            features: take_event_features(event),
            num_vectors: event.num_vectors as usize,
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
    session: &mut GpuSession,
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

    while let Some(next) = backlog.front() {
        let contiguous = input_slices
            .last()
            .is_none_or(|prev: &PoolSlice| prev.is_contiguous(&next.features));
        if !contiguous || slot_count >= max_batch_slots {
            break;
        }

        let next = backlog.pop_front().expect("front just checked");
        num_vectors += next.num_vectors;
        slot_count += 1;
        input_slices.push(next.features);
    }

    let mut batch = session.submit_batch(host_ptr, num_vectors);
    batch.input_slices = input_slices;

    BatchEntry {
        slot_count,
        submitted_at: Instant::now(),
        batch,
    }
}

fn reserve_session(sessions: &[GpuSession], session_cursor: &mut usize) -> usize {
    loop {
        for _ in 0..sessions.len() {
            let idx = *session_cursor;
            *session_cursor = (idx + 1) % sessions.len();
            if sessions[idx].try_acquire() {
                return idx;
            }
        }
        metrics::inc_session_wait_loops();
        std::hint::spin_loop();
    }
}
