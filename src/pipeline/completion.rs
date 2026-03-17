//! CompletionConsumer: waits for async batch completion, encodes responses, and enqueues
//! them into the downstream writer stage.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Instant;

use disruptor::{EventGuard, EventPoller, Polling, SingleConsumerBarrier};

use crate::config::WRITE_BUF_SIZE;
use crate::metrics;
use crate::pipeline::batch_queue::{BatchEntry, BatchQueue};
use crate::pipeline::connection_registry::ConnectionRegistry;
use crate::pipeline::ready_queue::ReadyQueue;
use crate::pipeline::session::BatchPoll;
use crate::protocol;
use crate::ring_types::InferenceEvent;

pub struct CompletionConsumer {
    poller: EventPoller<InferenceEvent, SingleConsumerBarrier>,
    batch_queue: Arc<BatchQueue>,
    ready_queue: Arc<ReadyQueue>,
    registry: Arc<ConnectionRegistry>,
    max_batch_slots: usize,
    timers_idle: bool,
}

// SAFETY: CompletionConsumer runs on a single dedicated thread.
unsafe impl Send for CompletionConsumer {}

impl CompletionConsumer {
    pub fn new(
        poller: EventPoller<InferenceEvent, SingleConsumerBarrier>,
        batch_queue: Arc<BatchQueue>,
        ready_queue: Arc<ReadyQueue>,
        registry: Arc<ConnectionRegistry>,
        max_batch_slots: usize,
    ) -> Self {
        assert!(max_batch_slots > 0, "max_batch_slots must be > 0");
        Self {
            poller,
            batch_queue,
            ready_queue,
            registry,
            max_batch_slots,
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
            let entry = loop {
                if let Some(entry) = self.batch_queue.pop() {
                    self.timers_idle = false;
                    idle_loops = 0;
                    break entry;
                }
                if !self.timers_idle {
                    metrics::idle_timers();
                    self.timers_idle = true;
                }
                if idle_loops == 0 {
                    metrics::inc_completion_queue_empty_waits();
                }
                idle_loops = idle_loops.saturating_add(1);
                if idle_loops < 64 {
                    std::hint::spin_loop();
                } else {
                    thread::yield_now();
                }
                if stop_requested(stop.as_ref()) {
                    return;
                }
            };

            let batch_wait_start = Instant::now();
            loop {
                if stop_requested(stop.as_ref()) {
                    return;
                }
                match entry.batch.completion.poll() {
                    BatchPoll::Pending => std::hint::spin_loop(),
                    BatchPoll::Ready => break,
                    BatchPoll::Failed => entry.batch.completion.wait(),
                }
            }
            metrics::record_batch_wait(batch_wait_start.elapsed());

            let mut guard = {
                let mut polled = false;
                loop {
                    if stop_requested(stop.as_ref()) {
                        return;
                    }
                    match self.poller.poll_take(entry.slot_count as u64) {
                        Ok(guard) => break guard,
                        Err(Polling::NoEvents) => {
                            if !polled {
                                metrics::inc_completion_poll_stalls();
                                polled = true;
                            }
                            std::hint::spin_loop();
                        }
                        Err(Polling::Shutdown) => return,
                    }
                }
            };

            process_batch(
                &mut guard,
                entry,
                &self.ready_queue,
                &self.registry,
                self.max_batch_slots,
            );
        }
    }
}

fn stop_requested(stop: Option<&Arc<AtomicBool>>) -> bool {
    stop.is_some_and(|flag| flag.load(Ordering::Relaxed))
}

pub(crate) fn process_batch(
    guard: &mut EventGuard<'_, InferenceEvent, SingleConsumerBarrier>,
    entry: BatchEntry,
    ready_queue: &Arc<ReadyQueue>,
    registry: &Arc<ConnectionRegistry>,
    max_batch_slots: usize,
) {
    let mut guard_ref = &mut *guard;
    let output =
        unsafe { std::slice::from_raw_parts(entry.batch.output_ptr, entry.batch.output_len) };
    let mut output_offset: usize = 0;

    debug_assert!(entry.slot_count <= max_batch_slots);
    for _ in 0..entry.slot_count {
        let event = guard_ref
            .next()
            .expect("guard exhausted before queued batch slot_count");
        let num_vecs = event.num_vectors as usize;

        let mut wire = [0u8; WRITE_BUF_SIZE];
        let response = &output[output_offset..output_offset + num_vecs];
        protocol::encode_response(response, &mut wire[..protocol::response_size(num_vecs)]);

        let conn = event.conn;
        let wire_len = protocol::response_size(num_vecs);
        if registry.enqueue_response(
            conn,
            event.request_seq,
            event.published_at_ns,
            &wire[..wire_len],
        ) {
            ready_queue.push(conn);
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
    session_available.store(true, std::sync::atomic::Ordering::Release);
    metrics::inc_batches_completed();
}
