use disruptor::{EventPoller, MultiProducerBarrier, Polling};

use crate::response_queue::ResponseProducer;
use crate::ring_types::{InferenceEvent, InferenceResponse};

// Concrete type for the request poller on the batch processor thread.
pub type ReqPoller = EventPoller<InferenceEvent, MultiProducerBarrier>;

/// Batch processor: consumes from the request disruptor, runs inference,
/// pushes responses to per-IO-thread response channels.
pub struct BatchProcessor {
    pub poller: ReqPoller,
    pub response_producers: Vec<ResponseProducer>,
}

impl BatchProcessor {
    pub fn run(mut self) {
        let num_threads = self.response_producers.len();
        let mut signaled = vec![false; num_threads];

        loop {
            match self.poller.poll() {
                Ok(mut guard) => {
                    signaled.iter_mut().for_each(|s| *s = false);

                    for event in &mut guard {
                        // POC inference: sum each feature vector
                        let mut response = InferenceResponse::new();
                        response.conn_id = event.conn_id;
                        response.request_seq = event.request_seq;
                        response.num_vectors = event.num_vectors;

                        for v in 0..event.num_vectors as usize {
                            let vector = event.vector(v);
                            response.results[v] = vector.iter().sum();
                        }

                        let thread_id = event.io_thread_id as usize;
                        self.response_producers[thread_id].send(&response);
                        signaled[thread_id] = true;
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
