//! SubmissionConsumer: drains the request ring, accumulates GPU batches,
//! submits to ORT, and pushes BatchEntry values to the batch queue.

use std::sync::Arc;

use disruptor::{EventGuard, EventPoller, Polling, SingleProducerBarrier};

use crate::batch_queue::{BatchEntry, BatchQueue};
use crate::config::MAX_SESSION_BATCH_SIZE;
use crate::constants::FEATURE_DIM;
use crate::ring_types::InferenceEvent;

use super::session::GpuSession;

pub struct SubmissionConsumer {
    poller: EventPoller<InferenceEvent, SingleProducerBarrier>,
    sessions: Vec<GpuSession>,
    /// Round-robin session index.
    session_cursor: usize,
    batch_queue: Arc<BatchQueue>,
    /// Base address of the pinned host buffer (from `cuMemAllocHost`).
    host_base: *const u8,
    /// Device pointer returned by `cuMemHostGetDevicePointer` for `host_base`.
    device_base: u64,
    /// Tracks the absolute ring sequence of the last event processed.
    /// Starts at -1 (INITIAL_CURSOR_VALUE); cast to u64 gives BatchEntry.end_sequence.
    next_seq: i64,
}

// SAFETY: SubmissionConsumer runs on a single dedicated thread.
unsafe impl Send for SubmissionConsumer {}

impl SubmissionConsumer {
    pub fn new(
        poller: EventPoller<InferenceEvent, SingleProducerBarrier>,
        sessions: Vec<GpuSession>,
        batch_queue: Arc<BatchQueue>,
        host_base: *const u8,
        device_base: u64,
    ) -> Self {
        Self {
            poller,
            sessions,
            session_cursor: 0,
            batch_queue,
            host_base,
            device_base,
            next_seq: -1,
        }
    }

    pub fn run(mut self) {
        loop {
            match self.poller.poll() {
                Ok(mut guard) => {
                    // Extract fields needed inside guard processing to satisfy the borrow checker.
                    // The guard borrows `self.poller`; the extracted refs borrow disjoint fields.
                    process_guard(
                        &mut guard,
                        &mut self.sessions,
                        &mut self.session_cursor,
                        &self.batch_queue,
                        self.host_base,
                        self.device_base,
                        &mut self.next_seq,
                    );
                    // guard drop here advances SubmissionConsumer's cursor.
                }
                Err(Polling::NoEvents) => std::hint::spin_loop(),
                Err(Polling::Shutdown) => return,
            }
        }
    }
}

fn process_guard(
    guard: &mut EventGuard<'_, InferenceEvent, SingleProducerBarrier>,
    sessions: &mut Vec<GpuSession>,
    session_cursor: &mut usize,
    batch_queue: &Arc<BatchQueue>,
    host_base: *const u8,
    device_base: u64,
    next_seq: &mut i64,
) {
    // Accumulate events into GPU batches, splitting on:
    //  1. Pool ring wrap-around (non-contiguous PoolSlice addresses).
    //  2. MAX_SESSION_BATCH_SIZE slots per batch.
    //
    // All BatchEntry pushes must happen before the guard is dropped so the
    // CompletionConsumer's barrier can locate their entries in the queue.

    let mut batch_device_ptr: u64 = 0;
    let mut batch_num_vectors: usize = 0;
    let mut batch_slot_count: usize = 0;
    // Pointer to the end of the previous slot's feature data (for wrap detection).
    let mut prev_end: *const f32 = std::ptr::null();

    for event in &mut *guard {
        *next_seq += 1;

        let curr_start = event.features.as_slice().as_ptr();
        let num_vecs = event.num_vectors as usize;

        if batch_slot_count == 0 {
            // First slot of a new batch: record the device base pointer for this batch.
            let offset_bytes = (curr_start as usize).wrapping_sub(host_base as usize);
            batch_device_ptr = device_base + offset_bytes as u64;
            batch_num_vectors = num_vecs;
            batch_slot_count = 1;
        } else {
            let contiguous = prev_end == curr_start;
            let at_size_limit = batch_slot_count >= MAX_SESSION_BATCH_SIZE;
            if !contiguous || at_size_limit {
                // Flush the batch that ended at the previous slot.
                let end_seq = (*next_seq - 1) as u64;
                flush_batch(
                    sessions,
                    session_cursor,
                    batch_queue,
                    batch_device_ptr,
                    batch_num_vectors,
                    end_seq,
                );
                // Start a new batch at the current slot.
                let offset_bytes = (curr_start as usize).wrapping_sub(host_base as usize);
                batch_device_ptr = device_base + offset_bytes as u64;
                batch_num_vectors = num_vecs;
                batch_slot_count = 1;
            } else {
                batch_num_vectors += num_vecs;
                batch_slot_count += 1;
            }
        }

        prev_end = unsafe { curr_start.add(num_vecs * FEATURE_DIM) };
    }

    // Flush the final (or only) batch.
    if batch_slot_count > 0 {
        let end_seq = *next_seq as u64;
        flush_batch(
            sessions,
            session_cursor,
            batch_queue,
            batch_device_ptr,
            batch_num_vectors,
            end_seq,
        );
    }
    // Guard drops here, advancing SubmissionConsumer cursor to `end_seq`.
}

fn flush_batch(
    sessions: &mut Vec<GpuSession>,
    session_cursor: &mut usize,
    batch_queue: &Arc<BatchQueue>,
    device_ptr: u64,
    num_vectors: usize,
    end_sequence: u64,
) {
    let idx = *session_cursor;
    *session_cursor = (idx + 1) % sessions.len();

    let handle = sessions[idx].run_async(device_ptr, num_vectors);

    // Push to batch queue — spins when full (correct backpressure).
    batch_queue.push(BatchEntry {
        end_sequence,
        session_idx: idx,
        handle,
    });
}
