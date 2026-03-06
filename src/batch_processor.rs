use disruptor::{EventPoller, Polling, SingleProducerBarrier};

use crate::buffer_pool::PoolAllocator;
use crate::config::MAX_IO_THREADS;
use crate::constants::MAX_VECTORS_PER_REQUEST;
use crate::response_queue::ResponseProducer;
use crate::ring_types::{INLINE_RESULT_CAPACITY, InferenceEvent, InferenceResponse};

pub type PollCycleResult = Result<(), Polling>;

/// Batch processor: consumes from the request disruptor, runs inference,
/// pushes responses to per-IO-thread response channels.
///
/// Uses a single-producer disruptor (SPSC with one IO thread).
/// To support multiple IO threads: switch to build_multi_producer in main.rs,
/// use SingleProducer -> MultiProducer in IoThread, and MultiProducerBarrier here.
pub struct BatchProcessor {
    poller: EventPoller<InferenceEvent, SingleProducerBarrier>,
    response_producers: Vec<ResponseProducer>,
    result_allocators: Vec<PoolAllocator>,
    /// Reusable per-thread signal flags — avoids a heap allocation per poll cycle.
    signaled: Vec<bool>,
}

impl BatchProcessor {
    pub fn new(
        poller: EventPoller<InferenceEvent, SingleProducerBarrier>,
        response_producers: Vec<ResponseProducer>,
        result_allocators: Vec<PoolAllocator>,
    ) -> Self {
        let num_threads = response_producers.len();
        assert!(
            num_threads <= MAX_IO_THREADS,
            "io_thread_id is u8; max {} IO threads",
            MAX_IO_THREADS
        );
        assert_eq!(
            result_allocators.len(),
            num_threads,
            "result allocator count must match response producer count"
        );
        Self {
            poller,
            response_producers,
            result_allocators,
            signaled: vec![false; num_threads],
        }
    }

    pub fn process_one_poll_cycle(&mut self) -> PollCycleResult {
        let mut temp_results = [0.0f32; MAX_VECTORS_PER_REQUEST];

        match self.poller.poll() {
            Ok(mut guard) => {
                crate::metrics::inc_poll_events();
                self.signaled.iter_mut().for_each(|s| *s = false);

                for event in &mut guard {
                    let num_vecs = event.num_vectors as usize;

                    for (v, result) in temp_results.iter_mut().enumerate().take(num_vecs) {
                        let vector = event.vector(v);
                        *result = vector.iter().sum();
                    }
                    event.features.release();

                    let thread_id = event.io_thread_id as usize;
                    debug_assert!(
                        thread_id < self.response_producers.len(),
                        "io_thread_id {} out of range (max {})",
                        thread_id,
                        self.response_producers.len()
                    );

                    // Spin on recoverable failures (result pool exhausted until IO thread frees).
                    let response = loop {
                        let allocator_ref = if num_vecs > INLINE_RESULT_CAPACITY {
                            Some(&mut self.result_allocators[thread_id])
                        } else {
                            None
                        };
                        match InferenceResponse::with_results(
                            event.conn_id,
                            event.request_seq,
                            &temp_results[..num_vecs],
                            allocator_ref,
                        ) {
                            Ok(r) => break r,
                            Err(_) => std::hint::spin_loop(),
                        }
                    };

                    self.response_producers[thread_id].send(response);
                    self.signaled[thread_id] = true;
                    crate::metrics::dec_req_occ();
                }

                for (i, &had_responses) in self.signaled.iter().enumerate() {
                    if had_responses {
                        self.response_producers[i].signal();
                    }
                }
                Ok(())
            }
            Err(Polling::NoEvents) => {
                crate::metrics::inc_poll_no_events();
                Err(Polling::NoEvents)
            }
            Err(Polling::Shutdown) => Err(Polling::Shutdown),
        }
    }

    pub fn run(mut self) {
        loop {
            match self.process_one_poll_cycle() {
                Ok(()) => {}
                Err(Polling::NoEvents) => std::hint::spin_loop(),
                Err(Polling::Shutdown) => return,
            }
        }
    }
}
