use disruptor::{
    BusySpin, EventPoller, Producer, SingleConsumerBarrier, SingleProducer, SingleProducerBarrier,
    build_single_producer,
};

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
        self.producer.publish(|slot| {
            slot.conn_id = conn_id;
            slot.request_seq = request_seq;
            slot.num_vectors = num_vectors;
            slot.results = results;
        });
    }

    /// Signal the IO thread's io_uring via eventfd. Call after sending a batch.
    pub fn signal(&self) {
        let val: u64 = 1;
        unsafe {
            libc::write(self.eventfd, &val as *const u64 as *const libc::c_void, 8);
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
