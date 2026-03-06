use crate::constants::{FEATURE_DIM, MAX_VECTORS_PER_REQUEST};

/// Wire format sizes — single source of truth for both sides of the protocol.
///
/// Request:  `[u32 num_vectors LE][f32 × num_vectors × FEATURE_DIM LE]`
/// Response: `[u8 num_vectors][f32 × num_vectors LE]`
pub const REQUEST_HEADER_BYTES: usize = 4; // u32 num_vectors
pub const RESPONSE_HEADER_BYTES: usize = 1; // u8 num_vectors
pub const BYTES_PER_F32: usize = 4;

/// Total byte length of a request carrying `num_vectors` vectors.
pub const fn request_size(num_vectors: usize) -> usize {
    REQUEST_HEADER_BYTES + num_vectors * FEATURE_DIM * BYTES_PER_F32
}

/// Total byte length of a response carrying `num_vectors` results.
pub const fn response_size(num_vectors: usize) -> usize {
    RESPONSE_HEADER_BYTES + num_vectors * BYTES_PER_F32
}

/// Result of attempting to parse a request from a byte buffer.
#[allow(dead_code)]
pub enum ParseResult {
    /// Successfully parsed a request. Contains (num_vectors, total bytes consumed).
    Complete {
        num_vectors: u8,
        bytes_consumed: usize,
    },
    /// Need more data. Contains minimum bytes still needed.
    Incomplete(usize),
    /// Protocol error (e.g., num_vectors > MAX or == 0).
    Error(&'static str),
}

/// Try to parse a request from the buffer. Returns how many bytes were consumed
/// and the number of vectors. Feature data starts at offset 4 in the buffer.
pub fn try_parse_request(buf: &[u8]) -> ParseResult {
    if buf.len() < REQUEST_HEADER_BYTES {
        return ParseResult::Incomplete(REQUEST_HEADER_BYTES - buf.len());
    }

    let num_vectors_u32 = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);

    if num_vectors_u32 == 0 || num_vectors_u32 as usize > MAX_VECTORS_PER_REQUEST {
        return ParseResult::Error("num_vectors out of range");
    }

    let num_vectors = num_vectors_u32 as u8;
    let total_size = request_size(num_vectors as usize);

    if buf.len() < total_size {
        return ParseResult::Incomplete(total_size - buf.len());
    }

    ParseResult::Complete {
        num_vectors,
        bytes_consumed: total_size,
    }
}

/// Copy feature data from a raw byte buffer (starting after the 4-byte header)
/// into the pre-allocated f32 slice in the disruptor event.
///
/// # Safety
/// Caller must ensure `src` has at least `num_vectors * FEATURE_DIM * 4` bytes
/// and `dst` has at least `num_vectors * FEATURE_DIM` f32 slots.
pub fn copy_features(src: &[u8], dst: &mut [f32], num_vectors: u8) {
    let count = num_vectors as usize * FEATURE_DIM;
    // SAFETY: f32 and [u8; 4] have the same size, and we're just reinterpreting
    // little-endian bytes as f32. This is safe on little-endian platforms.
    // For portability, use from_le_bytes per element.
    for (i, dst_elem) in dst.iter_mut().enumerate().take(count) {
        let offset = i * 4;
        *dst_elem = f32::from_le_bytes([
            src[offset],
            src[offset + 1],
            src[offset + 2],
            src[offset + 3],
        ]);
    }
}
