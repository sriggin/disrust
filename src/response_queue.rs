use disruptor::{
    BusySpin, EventPoller, Producer, SingleConsumerBarrier, SingleProducer, SingleProducerBarrier,
    build_single_producer,
};

use crate::metrics;
use crate::ring_types::InferenceResponse;

// Concrete types for the response SPSC channel.
pub type RespProducer = SingleProducer<InferenceResponse, SingleConsumerBarrier>;
pub type RespPoller = EventPoller<InferenceResponse, SingleProducerBarrier>;

/// Producer half lives on the batch processor thread.
pub struct ResponseProducer {
    pub producer: RespProducer,
    pub eventfd: i32,
}

impl ResponseProducer {
    pub fn send(&mut self, response: InferenceResponse) {
        let InferenceResponse {
            conn_id,
            request_seq,
            num_vectors,
            results,
        } = response;
        let mut results = Some(results);
        loop {
            match self.producer.try_publish(|slot| {
                slot.conn_id = conn_id;
                slot.request_seq = request_seq;
                slot.num_vectors = num_vectors;
                slot.results = results.take().expect("results already moved into ring");
            }) {
                Ok(_) => {
                    metrics::inc_resp_occ();
                    break;
                }
                Err(_) => {
                    metrics::inc_resp_ring_full();
                    std::hint::spin_loop();
                }
            }
        }
    }

    /// Signal the IO thread's io_uring via eventfd. Call after sending a batch.
    /// Logs to stderr if the write fails (e.g. eventfd closed).
    pub fn signal(&self) {
        let val: u64 = 1;
        let ret =
            unsafe { libc::write(self.eventfd, &val as *const u64 as *const libc::c_void, 8) };
        if ret != 8 {
            eprintln!("eventfd write failed: {}", std::io::Error::last_os_error());
        }
    }
}

/// Build a matched producer/poller pair for one IO thread's response channel.
pub fn build_response_channel(capacity: usize, eventfd: i32) -> (ResponseProducer, RespPoller) {
    let builder = build_single_producer(capacity, InferenceResponse::new, BusySpin);
    let (poller, builder) = builder.event_poller();
    let producer = builder.build();

    (ResponseProducer { producer, eventfd }, poller)
}
