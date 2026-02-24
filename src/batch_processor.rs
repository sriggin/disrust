use disruptor::{EventPoller, Polling, SingleProducerBarrier};

use crate::buffer_pool::BufferPool;
use crate::constants::MAX_VECTORS_PER_REQUEST;
use crate::response_queue::ResponseProducer;
use crate::ring_types::{INLINE_RESULT_CAPACITY, InferenceEvent, InferenceResponse};

/// Batch processor: consumes from the request disruptor, runs inference,
/// pushes responses to per-IO-thread response channels.
///
/// Uses a single-producer disruptor (SPSC with one IO thread).
/// To support multiple IO threads: switch to build_multi_producer in main.rs,
/// use SingleProducer -> MultiProducer in IoThread, and MultiProducerBarrier here.
pub struct BatchProcessor {
    pub poller: EventPoller<InferenceEvent, SingleProducerBarrier>,
    pub response_producers: Vec<ResponseProducer>,
    pub result_pools: Vec<&'static BufferPool>,
}

impl BatchProcessor {
    pub fn run(mut self) {
        let num_threads = self.response_producers.len();
        let mut signaled = vec![false; num_threads];
        let mut temp_results = [0.0f32; MAX_VECTORS_PER_REQUEST];

        loop {
            match self.poller.poll() {
                Ok(mut guard) => {
                    signaled.iter_mut().for_each(|s| *s = false);

                    for event in &mut guard {
                        let num_vecs = event.num_vectors as usize;

                        // POC inference: sum each feature vector into temp buffer
                        for (v, result) in temp_results.iter_mut().enumerate().take(num_vecs) {
                            let vector = event.vector(v);
                            *result = vector.iter().sum();
                        }
                        event.features.release();

                        // Create response - automatically chooses inline vs pooled
                        let pool_ref = if num_vecs > INLINE_RESULT_CAPACITY {
                            Some(self.result_pools[event.io_thread_id as usize])
                        } else {
                            None
                        };

                        let response = InferenceResponse::with_results(
                            event.conn_id,
                            event.request_seq,
                            &temp_results[..num_vecs],
                            pool_ref,
                        )
                        .expect("failed to create response");

                        let thread_id = event.io_thread_id as usize;
                        self.response_producers[thread_id].send(response);
                        signaled[thread_id] = true;
                        crate::metrics::dec_req_occ();
                    }

                    // Signal all IO threads that received responses
                    for (i, &had_responses) in signaled.iter().enumerate() {
                        if had_responses {
                            self.response_producers[i].signal();
                        }
                    }
                }
                Err(Polling::NoEvents) => {
                    std::hint::spin_loop();
                }
                Err(Polling::Shutdown) => {
                    return;
                }
            }
        }
    }
}
