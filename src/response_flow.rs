//! Response path: iterate responses (e.g. from a guard) → build wire bytes or iovecs per conn.
//!
//! Extracted so integration tests and benchmarks can drive the flow without io_uring.

#![allow(dead_code)] // guard_to_* used by integration tests and benchmarks

use std::collections::HashMap;

use libc::iovec;

use crate::ring_types::InferenceResponse;

/// Build wire bytes per connection from an iterator of responses.
/// Wire format per response: [u8 num_vectors][f32* num_vectors as LE bytes].
/// Multiple responses for the same conn_id are concatenated.
pub fn guard_to_wire_per_conn<'a, I>(guard: I) -> HashMap<u16, Vec<u8>>
where
    I: Iterator<Item = &'a InferenceResponse>,
{
    let mut map: HashMap<u16, Vec<u8>> = HashMap::new();
    for resp in guard {
        let buf = map.entry(resp.conn_id).or_default();
        buf.push(resp.num_vectors);
        let results = resp.results_slice();
        for &val in results {
            buf.extend_from_slice(&val.to_le_bytes());
        }
    }
    map
}

/// Build iovecs per connection from an iterator of responses (zero-copy).
/// Each response contributes two iovecs: header (1 byte num_vectors), payload (num_vectors*4 bytes).
/// Caller must ensure the guard/response memory outlives use of the iovecs.
pub fn guard_to_iovecs_per_conn<'a, I>(guard: I, out: &mut HashMap<u16, Vec<iovec>>)
where
    I: Iterator<Item = &'a InferenceResponse>,
{
    out.clear();
    for resp in guard {
        let list = out.entry(resp.conn_id).or_default();
        list.push(iovec {
            iov_base: (&resp.num_vectors as *const u8) as *mut libc::c_void,
            iov_len: 1,
        });
        let results = resp.results_slice();
        let payload_len = results.len() * 4;
        list.push(iovec {
            iov_base: (results.as_ptr() as *const u8) as *mut libc::c_void,
            iov_len: payload_len,
        });
    }
}
