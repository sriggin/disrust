//! Request path: bytes in -> parse -> publish to request ring.
//!
//! Extracted so integration tests and benchmarks can drive the flow without io_uring.

use disruptor::{Producer, RingBufferFull, SingleConsumerBarrier, SingleProducer};

use crate::buffer_pool::{AllocError, PoolAllocator};
use crate::clock::monotonic_now_ns;
use crate::constants::FEATURE_DIM;
use crate::protocol;
use crate::ring_types::InferenceEvent;

/// Error from processing request bytes.
#[derive(Debug)]
pub enum ProcessRequestError {
    Parse(#[allow(dead_code)] &'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessRequestOutcome {
    pub consumed: usize,
    pub num_published: usize,
    /// `true` if parsing stopped because more socket bytes are required to finish the
    /// next request, so the caller should re-arm a read when space is available.
    pub needs_read: bool,
}

/// Process all complete requests in `buf`, publishing each to the request ring.
/// Returns a [`ProcessRequestOutcome`] on success.
///
/// Pool allocation happens inside the `try_publish` closure, which only runs when
/// a ring slot is available. This means `RingBufferFull` never leaves a live
/// `PoolSlice` outside the ring, so FIFO pool-release order is always preserved.
///
/// Pool exhaustion spins inside the closure until the batch processor releases
/// slices on the other thread. `AllocError::TooLarge` cannot occur in practice
/// because `num_vectors * FEATURE_DIM` is bounded far below pool capacity.
///
/// Returns `Err` only on a parse error; caller should close the connection.
pub fn process_requests_from_buffer(
    buf: &[u8],
    producer: &mut SingleProducer<InferenceEvent, SingleConsumerBarrier>,
    allocator: &mut PoolAllocator,
    conn_id: u16,
    generation: u32,
    thread_id: u8,
    request_seq: &mut u64,
) -> Result<ProcessRequestOutcome, ProcessRequestError> {
    let mut consumed = 0;
    let mut num_published = 0;
    let mut needs_read = false;

    while consumed < buf.len() {
        let slice = &buf[consumed..];
        match protocol::try_parse_request(slice) {
            protocol::ParseResult::Complete {
                num_vectors,
                bytes_consumed,
            } => {
                let feature_bytes = &slice[protocol::REQUEST_HEADER_BYTES..bytes_consumed];
                let seq = *request_seq;
                let feature_count = num_vectors as usize * FEATURE_DIM;

                match producer.try_publish(|slot| {
                    // Alloc inside the closure: only runs when a ring slot is available,
                    // so RingBufferFull never leaves a live PoolSlice outside the ring.
                    // Spin on Exhausted — the batch processor releases on the other thread.
                    let mut pool_slice = loop {
                        match allocator.alloc(feature_count) {
                            Ok(s) => break s,
                            Err(AllocError::Exhausted { .. }) => std::hint::spin_loop(),
                            Err(AllocError::TooLarge { .. }) => unreachable!(
                                "feature_count {feature_count} cannot exceed pool capacity"
                            ),
                        }
                    };
                    protocol::copy_features(feature_bytes, pool_slice.as_mut_slice(), num_vectors);
                    slot.io_thread_id = thread_id;
                    slot.conn_id = conn_id;
                    slot.generation = generation;
                    slot.request_seq = seq;
                    slot.num_vectors = num_vectors;
                    slot.published_at_ns = monotonic_now_ns();
                    slot.features = pool_slice.freeze();
                }) {
                    Ok(_) => {}
                    Err(RingBufferFull) => {
                        crate::metrics::inc_req_ring_full();
                        break;
                    }
                }
                *request_seq += 1;
                num_published += 1;
                crate::metrics::inc_requests_published();
                crate::metrics::inc_req_occ();
                consumed += bytes_consumed;
            }
            protocol::ParseResult::Incomplete(_) => {
                needs_read = true;
                break;
            }
            protocol::ParseResult::Error(e) => return Err(ProcessRequestError::Parse(e)),
        }
    }
    Ok(ProcessRequestOutcome {
        consumed,
        num_published,
        needs_read,
    })
}
