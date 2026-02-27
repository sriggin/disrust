//! Request path: bytes in → parse → alloc → publish to request ring.
//!
//! Extracted so integration tests and benchmarks can drive the flow without io_uring.

use disruptor::{Producer, RingBufferFull, SingleConsumerBarrier, SingleProducer};

use crate::buffer_pool::{AllocError, BufferPool};
use crate::constants::FEATURE_DIM;
use crate::protocol;
use crate::ring_types::InferenceEvent;

/// Error from processing request bytes (alloc or parse failure).
#[derive(Debug)]
#[allow(dead_code)]
pub enum ProcessRequestError {
    Alloc(AllocError),
    Parse(&'static str),
}

/// Process all complete requests in `buf`, publishing each to the request ring.
/// Returns (bytes consumed, number of requests published) on success.
///
/// On parse error or alloc failure, returns `Err` and no further bytes are consumed.
/// Caller should close the connection or back off.
pub fn process_requests_from_buffer(
    buf: &[u8],
    producer: &mut SingleProducer<InferenceEvent, SingleConsumerBarrier>,
    pool: &'static BufferPool,
    conn_id: u16,
    thread_id: u8,
    request_seq: &mut u64,
) -> Result<(usize, usize), ProcessRequestError> {
    let mut consumed = 0;
    let mut num_published = 0;
    while consumed < buf.len() {
        let slice = &buf[consumed..];
        match protocol::try_parse_request(slice) {
            protocol::ParseResult::Complete {
                num_vectors,
                bytes_consumed,
            } => {
                let feature_bytes = &slice[4..bytes_consumed];
                let seq = *request_seq;
                *request_seq += 1;

                let feature_count = num_vectors as usize * FEATURE_DIM;
                let mut pool_slice = pool
                    .alloc(feature_count)
                    .map_err(ProcessRequestError::Alloc)?;
                protocol::copy_features(feature_bytes, pool_slice.as_mut_slice(), num_vectors);
                let mut features = Some(pool_slice.freeze());

                loop {
                    match producer.try_publish(|slot| {
                        slot.io_thread_id = thread_id;
                        slot.conn_id = conn_id;
                        slot.request_seq = seq;
                        slot.num_vectors = num_vectors;
                        slot.features = features.take().expect("features already moved into ring");
                    }) {
                        Ok(_) => break,
                        Err(RingBufferFull) => std::hint::spin_loop(),
                    }
                }
                num_published += 1;
                consumed += bytes_consumed;
            }
            protocol::ParseResult::Incomplete(_) => break,
            protocol::ParseResult::Error(e) => return Err(ProcessRequestError::Parse(e)),
        }
    }
    Ok((consumed, num_published))
}
